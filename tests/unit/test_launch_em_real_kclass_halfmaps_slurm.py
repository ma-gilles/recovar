from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile

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


def test_verify_canonical_rehashes_large_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = (tmp_path / "large-stack.mrcs").resolve()
    artifact.write_bytes(b"sealed stack")
    expected = "a" * 64
    original_stat = Path.stat

    def report_large_size(path: Path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == artifact:
            fields = list(result)
            fields[6] = 1_000_000_001
            return os.stat_result(fields)
        return result

    calls: list[Path] = []

    def mismatched_hash(path: Path) -> str:
        calls.append(path)
        return "b" * 64

    monkeypatch.setattr(launcher, "CANONICAL_HASHES", {str(artifact): expected})
    monkeypatch.setattr(Path, "stat", report_large_size)
    monkeypatch.setattr(launcher, "sha256_file", mismatched_hash)

    with pytest.raises(launcher.LaunchError, match="canonical checksum changed"):
        launcher._verify_canonical(artifact)

    assert calls == [artifact]


def test_matched_commands_use_independent_all_data_k4_processes(tmp_path: Path) -> None:
    row = _row(tmp_path, 1)
    initial_class_volumes = [tmp_path / f"reference_init_class{class_id:03d}.mrc" for class_id in range(1, 5)]
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
        initial_class_volumes=initial_class_volumes,
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
    assert recovar[recovar.index("--init_class_volumes") + 1] == ",".join(
        str(path.resolve()) for path in initial_class_volumes
    )


def test_prepared_references_keep_recovar_and_relion_coordinate_frames_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for class_id in range(1, 5):
        (source_dir / f"run_it000_class{class_id:03d}.mrc").touch()

    volumes = {
        class_id: np.full((4, 4, 4), class_id, dtype=np.float32)
        for class_id in range(1, 5)
    }
    original_require = launcher._require
    original_load_relion = launcher.helpers.load_relion_volume

    def allow_tiny_source(condition: bool, message: str) -> None:
        if message.startswith("canonical class ") and message.endswith(" grid changed"):
            return
        original_require(condition, message)

    def load_relion(path: str, *, return_voxel_size: bool = False):
        class_id = int(Path(path).stem[-3:])
        result = volumes[class_id].copy()
        return (result, np.asarray([2.0], dtype=np.float32)) if return_voxel_size else result

    monkeypatch.setattr(launcher, "INITIAL_MAP_ROOT", source_dir)
    monkeypatch.setattr(launcher, "_verify_canonical", lambda _path: "unused")
    monkeypatch.setattr(launcher, "_require", allow_tiny_source)
    monkeypatch.setattr(launcher.helpers, "load_relion_volume", load_relion)

    profile = launcher.Profile("tiny", 4, "full10k", 4, "00:10:00", "1G", 1)
    recovar_paths, relion_paths, star_path = launcher._prepare_references(tmp_path / "run", profile)

    assert [path.name for path in recovar_paths] == [
        f"reference_init_class{class_id:03d}.mrc" for class_id in range(1, 5)
    ]
    assert [path.name for path in relion_paths] == [
        f"reference_init_class{class_id:03d}_relion.mrc" for class_id in range(1, 5)
    ]
    # Load each file through its intended consumer convention. Both must land
    # on the same internal array, while the old bug (native-load the RELION
    # file) is explicitly distinguishable.
    from recovar.utils import helpers as recovar_helpers

    for class_id, (recovar_path, relion_path) in enumerate(
        zip(recovar_paths, relion_paths, strict=True),
        start=1,
    ):
        native, native_voxel = recovar_helpers.load_mrc(recovar_path, return_voxel_size=True)
        relion, relion_voxel = original_load_relion(
            relion_path,
            return_voxel_size=True,
        )
        wrong_frame = recovar_helpers.load_mrc(relion_path)
        np.testing.assert_array_equal(native, volumes[class_id])
        np.testing.assert_array_equal(relion, volumes[class_id])
        np.testing.assert_array_equal(wrong_frame, -volumes[class_id])
        assert float(native_voxel.x) == pytest.approx(2.0)
        assert float(relion_voxel.x) == pytest.approx(2.0)

    classes = starfile.read(star_path, always_dict=True)["model_classes"]
    assert [Path(path).name for path in classes["rlnReferenceImage"]] == [
        path.name for path in relion_paths
    ]


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
        analysis_policy=launcher.expected_analysis_policy(128),
    )

    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --exclusive" not in text
    assert "for half in 1 2" in text
    assert text.index('run_relion_half "${half}"') < text.index('run_recovar_half "${half}"')
    assert "ReqTRES" in text and "AllocTRES" in text
    assert "input_sha256_check.txt" in text
    assert "audit_em_real_kclass_halfmaps" in text
    assert "run_particle_state_audit_half" in text
    assert "audit_em_particle_state_distribution" in text
    assert 'half${half}_particle_state.json' in text
    assert 'half${half}_particle_state_arrays.npz' in text
    assert 'half${half}_particle_state.sha256' in text
    assert 'iteration <= 8' in text
    assert text.index('run_recovar_half "${half}"') < text.index(
        'run_particle_state_audit_half "${half}"'
    )
    assert "--fit-max-shell 32" in text
    assert "--crossing-consecutive-shells 3" in text
    assert "--phase-randomization-corrected false" in text
    assert "--absolute-resolution-claim false" in text
    assert "--coarse-healpix-order 1" in text
    assert "--refine-healpix-order 2" in text
    assert "--interpolation-order 1" in text
    assert "--mask-threshold auto" in text
    assert "--mask-lowpass-sigma 2" in text
    assert "--mask-extend 4" in text
    assert "--mask-soft-edge 4" in text
    assert "--mask-cleanup true" in text
    assert "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER" not in text
    assert 'export TMPDIR="${RUNTIME_ROOT}/relion_half${half}"' in text
    assert '${ROOT}/runtime' not in text


def test_setup_job_records_and_validates_exact_allocation_before_build(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    record = tmp_path / "provenance" / "setup_slurm_allocation.json"

    script = launcher.write_setup_script(
        scratch_dir=tmp_path,
        jobs_dir=jobs,
        cuda_lib=tmp_path / "libcuda_backproject.so",
        account="gilles",
        partition="cryoem",
        constraint="h100",
        setup_gres="gpu:1",
        cuda_module="cudatoolkit/12.8",
        relion_src_dir=tmp_path / "relion_src",
        setup_allocation_record=record,
    )
    text = script.read_text()

    assert str(record) in text
    assert "row = _slurm_allocation()" in text
    assert 'row["requested_gpus"] = _gpu_count_from_tres(row["ReqTRES"])' in text
    assert 'row["allocated_gpus"] = _gpu_count_from_tres(row["AllocTRES"])' in text
    assert text.index("row = _slurm_allocation()") < text.index("flock")
    assert "#SBATCH --exclusive" not in text


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


def test_particle_input_generation_carries_immutable_source_indices(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    np.save(fixture / "source_indices.npy", np.arange(4, dtype=np.int64))
    stack = tmp_path / "particles.128.mrcs"
    stack.write_bytes(b"stack")
    optics = pd.DataFrame({"rlnImagePixelSize": [1.5], "rlnImageSize": [256]})
    particles = pd.DataFrame(
        {
            "rlnImageName": [f"{index}@particles.256.mrcs" for index in range(1, 5)],
            "rlnRandomSubset": [1, 2, 1, 2],
        }
    )
    monkeypatch.setattr(launcher, "SOURCE_FIXTURE", fixture)
    monkeypatch.setattr(launcher, "STACKS", {128: stack})
    monkeypatch.setattr(launcher, "_particle_tables", lambda: (optics.copy(), particles.copy()))
    profile = launcher.Profile("tiny", 128, "full10k", 4, "00:10:00", "1G", 1)

    halves, names, source_indices = launcher._write_particle_inputs(tmp_path / "run", profile)

    assert names == [f"{index}@{stack.resolve()}" for index in range(1, 5)]
    assert source_indices == [0, 1, 2, 3]
    assert [row["particle_count"] for row in halves] == [2, 2]
    assert all(
        Path(name.split("@", 1)[1]).is_absolute()
        for row in halves
        for name in starfile.read(row["particles_star"])["particles"]["rlnImageName"]
    )


def test_particle_input_generation_routes_dataset_specific_fixture_and_stack(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    np.save(fixture / "source_indices.npy", np.arange(4, dtype=np.int64))
    stack = tmp_path / "dataset-specific-particles.256.mrcs"
    stack.write_bytes(b"stack")
    optics = pd.DataFrame({"rlnImagePixelSize": [1.25], "rlnImageSize": [256]})
    particles = pd.DataFrame(
        {
            "rlnImageName": [f"{index}@original.mrcs" for index in range(1, 5)],
            "rlnRandomSubset": [1, 2, 1, 2],
        }
    )
    starfile.write({"optics": optics, "particles": particles}, fixture / "particles.star")
    dataset = launcher.DatasetSpec(
        key="test",
        label="TEST",
        source_fixture=fixture,
        shared200_selection=None,
        initial_map_root=tmp_path / "maps",
        stacks={256: stack},
        canonical_hashes={},
        supported_profiles=frozenset({"tiny"}),
    )
    profile = launcher.Profile("tiny", 256, "full10k", 4, "00:10:00", "1G", 1)

    halves, names, source_indices = launcher._write_particle_inputs(
        tmp_path / "run",
        profile,
        dataset,
    )

    assert names == [f"{index}@{stack.resolve()}" for index in range(1, 5)]
    assert source_indices == [0, 1, 2, 3]
    assert [row["particle_count"] for row in halves] == [2, 2]


def test_cli_is_dry_run_unless_submit_is_explicit(tmp_path: Path) -> None:
    args = launcher._parse_args(["--output-root", str(tmp_path / "run")])

    assert args.submit is False
    assert args.dataset == "10076"
    assert args.profile == "shared200-128"
    assert args.seed == 42001


def test_10345_dataset_contract_is_native_grid_only() -> None:
    dataset = launcher._dataset_spec("10345")

    assert dataset.label == "EMPIAR-10345"
    assert dataset.supported_profiles == frozenset({"native10k-256"})
    assert set(dataset.stacks) == {256}
    assert dataset.shared200_selection is None
    assert dataset.canonical_hashes[str(dataset.stacks[256])].startswith("7909a695")


def test_10345_rejects_unqualified_128_profile_before_creating_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "run"
    monkeypatch.setattr(launcher, "DEFAULT_RUN_ROOT", tmp_path)
    args = launcher._parse_args(
        [
            "--output-root",
            str(root),
            "--dataset",
            "10345",
            "--profile",
            "pilot10k-128",
        ]
    )

    with pytest.raises(launcher.LaunchError, match="does not have qualified inputs"):
        launcher.prepare(args)

    assert not root.exists()


def test_profiles_request_measured_host_memory_headroom() -> None:
    assert launcher.PROFILES["shared200-128"].memory == "32G"
    assert launcher.PROFILES["pilot10k-128"].memory == "64G"
    assert launcher.PROFILES["native10k-256"].memory == "256G"


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
