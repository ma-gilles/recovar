"""Focused guards for the matched EMPIAR-10202 set-6 launcher."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from recovar.data_io.starfile import write_star

SCRIPT = Path(__file__).parents[2] / "scripts" / "launch_empiar10202_set6_i1_matched.py"
SPEC = importlib.util.spec_from_file_location("launch_empiar10202_set6_i1_matched", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

pytestmark = pytest.mark.unit


def _source_star(path: Path) -> Path:
    halves = np.asarray(([1, 2] * 40) + ([1] * 4), dtype=np.int64)
    particles = pd.DataFrame(
        {
            "_rlnImageName": [f"{index + 1}@/sealed/particles.mrcs" for index in range(len(halves))],
            "_rlnRandomSubset": halves,
            "_rlnAngleRot": np.arange(len(halves), dtype=np.float64),
            "_rlnOpticsGroup": np.ones(len(halves), dtype=np.int64),
        }
    )
    optics = pd.DataFrame(
        {
            "_rlnOpticsGroup": [1],
            "_rlnOpticsGroupName": ["opticsGroup1"],
            "_rlnImagePixelSize": [0.788],
            "_rlnImageSize": [800],
        }
    )
    write_star(str(path), particles, optics)
    return path


def _prepared_inputs(tmp_path: Path) -> MODULE.PreparedInputs:
    stack = tmp_path / "stack.mrcs"
    stack.write_bytes(b"x" * (1024 * 1024))
    manifest = tmp_path / "preparation_manifest.json"
    manifest.write_text("{}\n")
    recovar_reference = tmp_path / "recovar.mrc"
    relion_reference = tmp_path / "relion.mrc"
    recovar_reference.write_bytes(b"recovar")
    relion_reference.write_bytes(b"relion")
    return MODULE.PreparedInputs(
        preparation_manifest=manifest,
        preparation_manifest_sha256=MODULE.sha256_file(manifest),
        particle_star=tmp_path / "particles.star",
        particle_star_sha256="1" * 64,
        particle_stack=stack,
        particle_stack_sha256="2" * 64,
        particle_stack_size_bytes=1024 * 1024,
        recovar_reference=recovar_reference,
        recovar_reference_sha256=MODULE.sha256_file(recovar_reference),
        relion_reference=relion_reference,
        relion_reference_sha256=MODULE.sha256_file(relion_reference),
        canonical_reference_sha256="3" * 64,
    )


def _subject(tmp_path: Path) -> dict[str, str]:
    repo = tmp_path / "repo"
    python = repo / ".pixi/envs/default/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("python\n")
    return {
        "repo": str(repo),
        "commit": "a" * 40,
        "tree": "b" * 40,
        "python": str(python),
        "driver": str(repo / "scripts/run_full_refinement.py"),
        "driver_sha256": "c" * 64,
        "cuda_source": str(repo / "recovar/cuda/cuda_backproject.cu"),
        "cuda_source_sha256": MODULE.CUDA_SOURCE_SHA256,
    }


def _option(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_balanced_smoke_star_is_stable_and_preserves_source_order(tmp_path: Path) -> None:
    source = _source_star(tmp_path / "source.star")
    first = tmp_path / "first" / "particles.star"
    second = tmp_path / "second" / "particles.star"

    first_manifest = MODULE.write_balanced_smoke_star(source, first)
    second_manifest = MODULE.write_balanced_smoke_star(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest["sha256"] == second_manifest["sha256"]
    assert first_manifest["half_counts"] == {"1": 32, "2": 32}
    assert first_manifest["particle_count"] == 64
    assert first_manifest["source_row_indices_zero_based"] == list(range(64))


def test_recovar_commands_are_autonomous_matched_i1() -> None:
    assert MODULE.os.environ["JAX_PLATFORMS"] == "cpu"
    repo = Path("/sealed/subject")
    smoke_data = Path("/sealed/smoke")
    full_data = Path("/sealed/full")
    smoke = MODULE._recovar_command(repo, smoke_data, Path("/out/smoke"), smoke=True)
    full = MODULE._recovar_command(repo, full_data, Path("/out/full"), smoke=False)

    for command, data in ((smoke, smoke_data), (full, full_data)):
        assert command[5:7] == ("-m", "scripts.run_full_refinement")
        assert not any(token.endswith("/scripts/run_full_refinement.py") for token in command)
        assert _option(command, "--sym") == "I1"
        assert _option(command, "--n_classes") == "1"
        assert _option(command, "--initial-pose-source") == "input-star"
        assert _option(command, "--data_dir") == str(data)
        assert _option(command, "--relion_half_sets") == str(data / "particles.star")
        assert _option(command, "--init_volume") == str(data / "reference_init.mrc")
        assert _option(command, "--seed") == "10202"
        assert _option(command, "--perturb_seed") == "10202"
        assert "--firstiter_cc" in command
        assert "--apply-initial-lowpass" in command
        assert not any("replay" in token or "local_only" in token or "fixed_pose" in token for token in command)
    assert _option(smoke, "--max_iter") == "1"
    assert _option(smoke, "--relion_current_sizes") == "800"
    assert "--skip_final_iteration" in smoke
    assert _option(full, "--max_iter") == "50"
    assert "--skip_final_iteration" not in full
    assert "--relion_current_sizes" not in full


def test_relion_commands_are_matched_i1_and_safe_for_preread() -> None:
    smoke_data = Path("/sealed/smoke")
    full_data = Path("/sealed/full")
    smoke = MODULE._relion_command(smoke_data, Path("/out/smoke"), smoke=True)
    full = MODULE._relion_command(full_data, Path("/out/full"), smoke=False)

    for command, data in ((smoke, smoke_data), (full, full_data)):
        assert _option(command, "--sym") == "I1"
        assert _option(command, "--K") == "1"
        assert _option(command, "--i") == str(data / "particles.star")
        assert _option(command, "--ref") == str(data / "reference_init_relion.mrc")
        assert _option(command, "--gpu") == "0:1"
        assert _option(command, "--offset") == "10"
        assert "--preread_images" in command
        assert "--no_parallel_disc_io" in command
        assert "--dont_combine_weights_via_disc" in command
    assert _option(smoke, "--auto_iter_max") == "1"
    assert _option(smoke, "--incr_size") == "800"
    assert _option(full, "--auto_iter_max") == "50"
    assert "--incr_size" not in full


@pytest.mark.parametrize(
    ("engine", "expected_gpus", "expected_tasks"),
    [("recovar", 1, 1), ("relion", 2, 3)],
)
def test_slurm_scripts_pin_exact_resources_and_environment(
    tmp_path: Path,
    engine: str,
    expected_gpus: int,
    expected_tasks: int,
) -> None:
    run_root = tmp_path / "run"
    specs = MODULE.build_run_specs(Path("/sealed/subject"), run_root)
    spec = next(item for item in specs if item.engine == engine and item.phase == "full")
    script = MODULE.render_sbatch(
        spec,
        run_root=run_root,
        runtime_root=tmp_path / "runtime",
        subject=_subject(tmp_path),
        inputs=_prepared_inputs(tmp_path),
        star_sha256="1" * 64,
        bind_library=tmp_path / "binding.so",
    )

    assert f"#SBATCH --gres=gpu:h100:{expected_gpus}" in script
    assert f"#SBATCH --ntasks={expected_tasks}" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=500G" in script
    assert "#SBATCH --nodes=1" in script
    assert "exclusive" not in script
    assert "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset" in script
    assert "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset" in script
    assert MODULE.CUDA_LIBRARY_SHA256 in script
    assert MODULE.RELION_BIND_SHA256 in script
    assert MODULE.RELION_MPI_SHA256 in script
    assert "scontrol show job -o" in script
    assert "Expected H100" in script


def test_full_jobs_depend_on_both_smokes() -> None:
    specs = {spec.key: spec for spec in MODULE.build_run_specs(Path("/repo"), Path("/run"))}

    assert specs["recovar_smoke"].depends_on == ()
    assert specs["relion_smoke"].depends_on == ()
    assert specs["recovar_full"].depends_on == ("recovar_smoke", "relion_smoke")
    assert specs["relion_full"].depends_on == ("recovar_smoke", "relion_smoke")


def test_subject_validation_requires_exact_clean_commit(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".pixi/envs/default/bin").mkdir(parents=True)
    (repo / ".pixi/envs/default/bin/python").write_text("python")
    (repo / "scripts").mkdir()
    (repo / "scripts/run_full_refinement.py").write_text("driver")
    (repo / "recovar/cuda").mkdir(parents=True)
    (repo / "recovar/cuda/cuda_backproject.cu").write_text("cuda")
    monkeypatch.setattr(MODULE, "require_sha256", lambda *_args, **_kwargs: "ok")
    monkeypatch.setattr(MODULE, "_git", lambda _repo, *_args: "b" * 40)

    with pytest.raises(ValueError, match="subject HEAD mismatch"):
        MODULE.validate_subject(repo, "a" * 40)


def test_preparation_manifest_rejects_bare_icosahedral_alias(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MODULE.PREPARATION_SCHEMA,
                "dataset": "EMPIAR-10202",
                "image_set": 6,
                "symmetry": {
                    "requested_label": "I",
                    "relion_label": "I2",
                    "recovar_label": "I2",
                    "operator_count": 60,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="explicit I1"):
        MODULE.validate_preparation_manifest(manifest)


def test_resource_audit_rejects_extra_gpu() -> None:
    spec = MODULE.build_run_specs(Path("/repo"), Path("/run"))[0]
    audit = "NumNodes=1 NumCPUs=4 ReqTRES=cpu=4,mem=500G,node=1,gres/gpu=2"

    with pytest.raises(ValueError, match="gres/gpu=1"):
        MODULE._validate_requested_resources(audit, spec)


def test_submit_uses_afterok_for_both_full_jobs(tmp_path: Path) -> None:
    specs = MODULE.build_run_specs(Path("/repo"), tmp_path)
    runs = {}
    for spec in specs:
        script = tmp_path / f"{spec.key}.sbatch"
        script.write_text("#!/bin/bash\n")
        runs[spec.key] = {"path": str(script)}
    manifest = tmp_path / "launch_manifest.json"
    manifest.write_text(json.dumps({"status": "planned", "submission": None, "runs": runs}))
    (tmp_path / "provenance").mkdir()
    calls: list[list[str]] = []
    next_job = iter(("101", "102", "103", "104"))

    def runner(command, **_kwargs):
        calls.append(list(command))
        if command[0] == "sbatch":
            return SimpleNamespace(stdout=next(next_job) + "\n")
        if command[0] == "scontrol":
            job_id = command[-1]
            spec = specs[int(job_id) - 101]
            cpus = spec.resources.ntasks * spec.resources.cpus_per_task
            return SimpleNamespace(
                stdout=(
                    f"JobId={job_id} NumNodes=1 NumCPUs={cpus} "
                    f"ReqTRES=cpu={cpus},mem=500G,node=1,gres/gpu={spec.resources.h100_gpus}\n"
                )
            )
        return SimpleNamespace(stdout="")

    submission = MODULE.submit_launch(manifest, specs, runner=runner)

    assert submission["job_ids"] == {
        "recovar_smoke": "101",
        "relion_smoke": "102",
        "recovar_full": "103",
        "relion_full": "104",
    }
    full_submissions = [command for command in calls if command[0] == "sbatch"][2:]
    assert all("--dependency=afterok:101:102" in command for command in full_submissions)
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "submitted"
    for key, job_id in submission["job_ids"].items():
        artifacts = submission["jobs"][key]["artifacts"]
        assert artifacts["stdout_log"] == str(tmp_path / "logs" / f"{key}-{job_id}.out")
        assert artifacts["stderr_log"] == str(tmp_path / "logs" / f"{key}-{job_id}.err")
        assert artifacts["executed_command"] == str(
            tmp_path / "provenance" / f"command_{job_id}.sh"
        )
        assert artifacts["scontrol_runtime"] == str(
            tmp_path / "provenance" / f"scontrol_{job_id}.txt"
        )
        assert len(submission["jobs"][key]["scontrol_submit_sha256"]) == 64


def test_resolved_job_artifacts_are_canonical(tmp_path: Path) -> None:
    specs = MODULE.build_run_specs(Path("/repo"), tmp_path)
    relion = next(spec for spec in specs if spec.key == "relion_full")
    artifacts = MODULE._resolved_job_artifacts(tmp_path, relion, "12345")

    assert artifacts["stdout_log"] == str(tmp_path / "logs/relion_full-12345.out")
    assert artifacts["stderr_log"] == str(tmp_path / "logs/relion_full-12345.err")
    assert artifacts["dynamic_libraries"] == str(
        tmp_path / "provenance/ldd_relion_12345.txt"
    )
    assert artifacts["completed_marker"] == str(relion.output_dir / "COMPLETED")


def test_native_pins_name_sm90_library() -> None:
    assert MODULE.CUDA_LIBRARY.name == "libcuda_backproject-2249bf352-r3-sealed-sm90.so"
    assert MODULE.CUDA_LIBRARY_SHA256 == "47a8a5c7878e7ea1f242942918a40d57da6b78ebd791339bd3ac70ec1ae395ac"
    assert MODULE.RELION_BIND_SHA256 == "82b0a8cf2c189463f9cf0181099f4e92ce4365cce3819463c27af44b3c1014a2"
