from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import launch_em_real_kclass_halfmaps_slurm as launcher


def _row(tmp_path: Path, half: int) -> dict:
    root = tmp_path / f"half{half}"
    return {
        "half": half,
        "data_dir": str(root / "data"),
        "relion_dir": str(root / "relion"),
        "recovar_dir": str(root / "recovar"),
        "relion_command": ["mpirun", "-n", "3", "relion_refine_mpi", "--K", "4"],
        "recovar_command": ["python", "-m", "scripts.run_full_refinement", "--n_classes", "4"],
    }


def test_matched_commands_use_independent_all_data_k4_processes(tmp_path: Path) -> None:
    row = _row(tmp_path, 1)
    relion = launcher.build_relion_command(
        executable=tmp_path / "relion_refine_mpi",
        row=row,
        reference_star=tmp_path / "references.star",
        max_iter=8,
        seed=42001,
        particle_diameter=200.0,
        mpi_ranks=3,
        pool=3,
    )
    recovar = launcher.build_recovar_command(
        python=tmp_path / "python",
        row=row,
        max_iter=8,
        seed=42001,
        particle_diameter=200.0,
        image_batch_size=100,
        mpi_ranks=3,
    )

    assert relion[relion.index("--K") + 1] == "4"
    assert recovar[recovar.index("--n_classes") + 1] == "4"
    assert "--split_random_halves" not in relion
    assert "--relion_half_sets" not in recovar
    assert "--final-replay-relion-dir" not in recovar
    assert recovar[recovar.index("--initial-pose-source") + 1] == "none"
    assert recovar[recovar.index("--relion-scale-followers") + 1] == "2"


def test_rendered_job_is_nonexclusive_serial_one_gpu_and_audited(tmp_path: Path) -> None:
    rows = [_row(tmp_path, half) for half in (1, 2)]
    text = launcher.render_run_script(
        root=tmp_path,
        profile=launcher.PROFILES["shared200-128"],
        source={"commit": "a" * 40},
        relion_source=tmp_path / "relion_source",
        relion_module="relion/test",
        relion_refine_mpi=tmp_path / "relion_refine_mpi",
        cuda_module="cuda/test",
        halves=rows,
        max_iter=8,
        seed=42001,
        mpi_ranks=3,
        pool=3,
    )

    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --exclusive" not in text
    assert "for half in 1 2" in text
    assert text.index('run_relion_half "${half}"') < text.index('run_recovar_half "${half}"')
    assert "ReqTRES" in text and "AllocTRES" in text
    assert "input_sha256_check.txt" in text
    assert "audit_em_real_kclass_halfmaps" in text
    assert "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER" not in text


def test_shared_selection_preserves_source_row_order(tmp_path: Path, monkeypatch) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        '{"same_visited_particle_ids": true, "visited_particle_ids": '
        '["4@particles.256.mrcs", "2@particles.256.mrcs"]}'
    )
    monkeypatch.setattr(launcher, "SHARED200_SELECTION", selection)
    profile = launcher.Profile("tiny", 128, "shared200", 2, "00:10:00", "1G", 1)
    particles = pd.DataFrame(
        {
            "rlnImageName": [
                "1@particles.256.mrcs",
                "2@particles.256.mrcs",
                "3@particles.256.mrcs",
                "4@particles.256.mrcs",
            ],
            "rlnRandomSubset": [1, 2, 1, 2],
        }
    )

    selected = launcher._selected_particles(profile, particles)

    assert selected["rlnImageName"].tolist() == ["2@particles.256.mrcs", "4@particles.256.mrcs"]


def test_cli_is_dry_run_unless_submit_is_explicit(tmp_path: Path) -> None:
    args = launcher._parse_args(["--output-root", str(tmp_path / "run")])

    assert args.submit is False
    assert args.profile == "shared200-128"
    assert args.seed == 42001


def _scontrol_line(
    *,
    job_id: str = "101",
    state: str = "PENDING",
    requested: str = "cpu=4,mem=192G,node=1,billing=4,gres/gpu=1",
    allocated: str = "(null)",
    oversubscribe: str = "OK",
) -> str:
    return (
        f"JobId={job_id} JobState={state} "
        f"ReqTRES={requested} AllocTRES={allocated} "
        "Socks/Node=* NtasksPerN:B:S:C=0:0:*:* "
        f"OverSubscribe={oversubscribe}"
    )


@pytest.mark.parametrize(
    ("allocated", "expected_allocated_gpus"),
    [
        ("(null)", None),
        ("cpu=4,mem=192G,node=1,billing=4,gres/gpu=1", 1),
    ],
)
def test_post_submit_validation_accepts_pending_or_exact_allocation(
    tmp_path: Path,
    monkeypatch,
    allocated: str,
    expected_allocated_gpus: int | None,
) -> None:
    script = tmp_path / "job.sbatch"
    script.write_text("#!/usr/bin/env bash\n#SBATCH --gres=gpu:1\n")
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *_args, **_kwargs: _scontrol_line(allocated=allocated),
    )

    audit = launcher.validate_submitted_job("101", script)

    assert audit["requested_gpus"] == 1
    assert audit["allocated_gpus"] == expected_allocated_gpus
    assert audit["OverSubscribe"] == "OK"
    assert audit["valid_at_submission"] is True


@pytest.mark.parametrize(
    ("line", "script_text", "message"),
    [
        (
            _scontrol_line(requested="cpu=4,mem=192G,node=1,billing=4,gres/gpu=2"),
            "#!/usr/bin/env bash\n#SBATCH --gres=gpu:1\n",
            "did not request exactly one GPU",
        ),
        (
            _scontrol_line(oversubscribe="YES"),
            "#!/usr/bin/env bash\n#SBATCH --gres=gpu:1\n",
            "is exclusive",
        ),
        (
            _scontrol_line(
                state="RUNNING",
                allocated="cpu=8,mem=192G,node=1,billing=8,gres/gpu=1",
            ),
            "#!/usr/bin/env bash\n#SBATCH --gres=gpu:1\n",
            "ReqTRES != AllocTRES",
        ),
        (
            _scontrol_line(),
            "#!/usr/bin/env bash\n#SBATCH --gres=gpu:1\n#SBATCH --exclusive\n",
            "job script requests --exclusive",
        ),
    ],
)
def test_post_submit_validation_rejects_unsafe_resources(
    tmp_path: Path,
    monkeypatch,
    line: str,
    script_text: str,
    message: str,
) -> None:
    script = tmp_path / "job.sbatch"
    script.write_text(script_text)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *_args, **_kwargs: line)

    with pytest.raises(launcher.LaunchError, match=message):
        launcher.validate_submitted_job("101", script)


def test_submit_cancels_all_new_jobs_when_run_allocation_validation_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    setup_script = tmp_path / "setup.sbatch"
    run_script = tmp_path / "run.sbatch"
    setup_script.write_text("#!/usr/bin/env bash\n#SBATCH --gres=gpu:1\n")
    run_script.write_text("#!/usr/bin/env bash\n#SBATCH --gres=gpu:1\n")
    calls: list[list[str]] = []
    cancellations: list[list[str]] = []

    def fake_check_output(command, **_kwargs):
        calls.append(command)
        if command[:2] == ["sbatch", "--parsable"]:
            return "101\n" if str(setup_script) in command else "102\n"
        if command[-1] == "101":
            return _scontrol_line(job_id="101")
        if command[-1] == "102":
            return _scontrol_line(
                job_id="102",
                requested="cpu=4,mem=192G,node=1,billing=4,gres/gpu=2",
            )
        raise AssertionError(command)

    def fake_run(command, **_kwargs):
        cancellations.append(command)

    monkeypatch.setattr(launcher.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    with pytest.raises(launcher.LaunchError, match="cancelled newly submitted jobs: 101, 102"):
        launcher.submit_scripts(setup_script, run_script)

    assert [call for call in calls if call[0] == "scontrol"] == [
        ["scontrol", "show", "job", "-o", "101"],
        ["scontrol", "show", "job", "-o", "102"],
    ]
    assert cancellations == [["scancel", "102"], ["scancel", "101"]]
