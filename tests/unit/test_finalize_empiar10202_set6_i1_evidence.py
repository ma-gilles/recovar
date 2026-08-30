"""Focused fail-closed guards for the set-6 evidence finalizer."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/finalize_empiar10202_set6_i1_evidence.py"
SPEC = importlib.util.spec_from_file_location("finalize_empiar10202_set6_i1_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

LAUNCHER_SCRIPT = Path(__file__).parents[2] / "scripts/launch_empiar10202_set6_i1_matched.py"
LAUNCHER_SPEC = importlib.util.spec_from_file_location("launch_empiar10202_set6_i1_matched_for_test", LAUNCHER_SCRIPT)
assert LAUNCHER_SPEC is not None and LAUNCHER_SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(LAUNCHER_SPEC)
sys.modules[LAUNCHER_SPEC.name] = LAUNCHER
LAUNCHER_SPEC.loader.exec_module(LAUNCHER)

pytestmark = pytest.mark.unit


def _write(path: Path, value: str = "sealed\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def _launch_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "run"
    runtime_root = tmp_path / "runtime/run"
    root.mkdir()
    runtime_root.mkdir(parents=True)
    (root / "SAFE_TO_DELETE").touch()
    (runtime_root / "SAFE_TO_DELETE").touch()
    for directory in (
        root / "logs",
        root / "evidence",
        root / "provenance",
        root / "scripts",
        root / "inputs/smoke",
        root / "inputs/full",
        root / "outputs",
        *((root / "outputs" / key) for key in MODULE.RUN_KEYS),
    ):
        directory.mkdir(parents=True, exist_ok=True)

    subject_repo = tmp_path / "subject"
    subject_python = _write(subject_repo / ".pixi/envs/default/bin/python", "#!/bin/sh\n")
    subject_driver = _write(subject_repo / "scripts/run_full_refinement.py", "# driver\n")
    subject_cuda = _write(subject_repo / "recovar/cuda/cuda_backproject.cu", "// cuda\n")
    monkeypatch.setattr(MODULE, "ALLOWED_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "SUBJECT_REPO", subject_repo)
    monkeypatch.setattr(MODULE, "SUBJECT_DRIVER_SHA256", MODULE.sha256_file(subject_driver))
    monkeypatch.setattr(MODULE, "SUBJECT_CUDA_SOURCE_SHA256", MODULE.sha256_file(subject_cuda))

    native_root = tmp_path / "native"
    relion_mpi = _write(native_root / "bin/relion_refine_mpi", "relion\n")
    relion_binding = _write(native_root / "relion_bind/_relion_bind_core.so", "binding\n")
    recovar_cuda = _write(native_root / "libcuda_backproject_sm90.so", "cuda\n")
    relion_source = native_root / "relion-source"
    relion_source.mkdir(parents=True)
    relion_cuda_lib_dir = native_root / "cuda/lib"
    relion_mpi_lib_dir = native_root / "mpi/lib"
    relion_cuda_lib_dir.mkdir(parents=True)
    relion_mpi_lib_dir.mkdir(parents=True)
    native_values = {
        "RELION_MPI": relion_mpi,
        "RELION_MPI_SHA256": MODULE.sha256_file(relion_mpi),
        "RELION_SOURCE": relion_source,
        "RELION_SOURCE_COMMIT": "d" * 40,
        "RELION_BIND_LIBRARY": relion_binding,
        "RELION_BIND_LIBRARY_SHA256": MODULE.sha256_file(relion_binding),
        "RELION_BIND_SOURCE_TREE": "e" * 40,
        "CUDA_LIBRARY": recovar_cuda,
        "CUDA_LIBRARY_SHA256": MODULE.sha256_file(recovar_cuda),
        "RELION_CUDA_LIB_DIR": relion_cuda_lib_dir,
        "RELION_MPI_LIB_DIR": relion_mpi_lib_dir,
    }
    for name, value in native_values.items():
        monkeypatch.setattr(MODULE, name, value)
    launcher_native_values = {
        "RELION_MPI": relion_mpi,
        "RELION_MPI_SHA256": MODULE.RELION_MPI_SHA256,
        "RELION_SOURCE": relion_source,
        "RELION_SOURCE_COMMIT": MODULE.RELION_SOURCE_COMMIT,
        "RELION_BIND_DIR": relion_binding.parent,
        "RELION_BIND_SHA256": MODULE.RELION_BIND_LIBRARY_SHA256,
        "CUDA_LIBRARY": recovar_cuda,
        "CUDA_LIBRARY_SHA256": MODULE.CUDA_LIBRARY_SHA256,
        "RELION_CUDA_LIB_DIR": relion_cuda_lib_dir,
        "RELION_MPI_LIB_DIR": relion_mpi_lib_dir,
    }
    for name, value in launcher_native_values.items():
        monkeypatch.setattr(LAUNCHER, name, value)

    def fixture_git_output(repo: Path, *arguments: str) -> str:
        if repo == subject_repo:
            if arguments == ("rev-parse", "HEAD"):
                return MODULE.SUBJECT_COMMIT
            if arguments == ("rev-parse", "HEAD^{tree}"):
                return MODULE.SUBJECT_TREE
            assert arguments == ("status", "--porcelain=v1", "--untracked-files=all")
            return ""
        assert repo == relion_source
        if arguments == ("rev-parse", "HEAD"):
            return MODULE.RELION_SOURCE_COMMIT
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return MODULE.RELION_BIND_SOURCE_TREE
        assert arguments == ("status", "--porcelain=v1", "--untracked-files=all")
        return ""

    monkeypatch.setattr(MODULE, "_git_output", fixture_git_output)

    prepared_root = tmp_path / "prepared"
    preparation_manifest = _write(prepared_root / "preparation_manifest.json", "{}\n")
    prepared_star = _write(prepared_root / "particles.star", "prepared-star\n")
    recovar_reference = _write(prepared_root / "reference_init.mrc", "recovar-reference\n")
    relion_reference = _write(prepared_root / "reference_init_relion.mrc", "relion-reference\n")
    particle_stack = prepared_root / "particles.mrcs"
    particle_stack.write_bytes(b"s" * (1024 * 1024))
    prepared_values = {
        "PREPARATION_MANIFEST": preparation_manifest,
        "PREPARATION_MANIFEST_SHA256": MODULE.sha256_file(preparation_manifest),
        "PREPARED_STAR": prepared_star,
        "PREPARED_STAR_SHA256": MODULE.sha256_file(prepared_star),
        "PARTICLE_STACK": particle_stack,
        "PARTICLE_STACK_SHA256": MODULE.sha256_file(particle_stack),
        "PARTICLE_STACK_SIZE_BYTES": particle_stack.stat().st_size,
        "RECOVAR_REFERENCE": recovar_reference,
        "RECOVAR_REFERENCE_SHA256": MODULE.sha256_file(recovar_reference),
        "RELION_REFERENCE": relion_reference,
        "RELION_REFERENCE_SHA256": MODULE.sha256_file(relion_reference),
        "CANONICAL_REFERENCE_SHA256": "c" * 64,
    }
    for name, value in prepared_values.items():
        monkeypatch.setattr(MODULE, name, value)
    smoke_star = _write(root / "inputs/smoke/particles.star", "smoke-star\n")
    (root / "inputs/full/particles.star").symlink_to(prepared_star)
    for phase in ("smoke", "full"):
        (root / f"inputs/{phase}/reference_init.mrc").symlink_to(recovar_reference)
        (root / f"inputs/{phase}/reference_init_relion.mrc").symlink_to(relion_reference)

    subject = {
        "repo": str(subject_repo),
        "commit": MODULE.SUBJECT_COMMIT,
        "tree": MODULE.SUBJECT_TREE,
        "tree_clean": True,
        "diff_sha256": MODULE.EMPTY_SHA256,
        "python": str(subject_python),
        "driver": str(subject_driver),
        "driver_sha256": MODULE.SUBJECT_DRIVER_SHA256,
        "cuda_source": str(subject_cuda),
        "cuda_source_sha256": MODULE.SUBJECT_CUDA_SOURCE_SHA256,
    }
    preparation = {
        "preparation_manifest": str(preparation_manifest),
        "preparation_manifest_sha256": MODULE.PREPARATION_MANIFEST_SHA256,
        "particle_star": str(prepared_star),
        "particle_star_sha256": MODULE.PREPARED_STAR_SHA256,
        "particle_stack": str(particle_stack),
        "particle_stack_sha256": MODULE.PARTICLE_STACK_SHA256,
        "particle_stack_size_bytes": MODULE.PARTICLE_STACK_SIZE_BYTES,
        "recovar_reference": str(recovar_reference),
        "recovar_reference_sha256": MODULE.RECOVAR_REFERENCE_SHA256,
        "relion_reference": str(relion_reference),
        "relion_reference_sha256": MODULE.RELION_REFERENCE_SHA256,
        "canonical_reference_sha256": MODULE.CANONICAL_REFERENCE_SHA256,
    }
    prepared_inputs = LAUNCHER.PreparedInputs(
        **{
            key: Path(value) if key in {
                "preparation_manifest",
                "particle_star",
                "particle_stack",
                "recovar_reference",
                "relion_reference",
            } else value
            for key, value in preparation.items()
        }
    )
    native = {
        "relion_refine_mpi": {"path": str(MODULE.RELION_MPI), "sha256": MODULE.RELION_MPI_SHA256},
        "relion_binding": {
            "path": str(MODULE.RELION_BIND_LIBRARY),
            "sha256": MODULE.RELION_BIND_LIBRARY_SHA256,
        },
        "relion_binding_source": {
            "path": str(MODULE.RELION_SOURCE),
            "commit": MODULE.RELION_SOURCE_COMMIT,
            "tree": MODULE.RELION_BIND_SOURCE_TREE,
        },
        "recovar_cuda_sm90": {
            "path": str(MODULE.CUDA_LIBRARY),
            "sha256": MODULE.CUDA_LIBRARY_SHA256,
        },
        "relion_runtime_library_dirs": [str(MODULE.RELION_CUDA_LIB_DIR), str(MODULE.RELION_MPI_LIB_DIR)],
    }
    symmetry = MODULE.SYMMETRY_CONTRACT
    monkeypatch.setattr(MODULE, "SMOKE_STAR_SHA256", MODULE.sha256_file(smoke_star))
    smoke = {
        "path": str(smoke_star),
        "sha256": MODULE.SMOKE_STAR_SHA256,
        "particle_count": 64,
        "half_counts": {"1": 32, "2": 32},
        "selection_policy": "first 32 source rows in each deposited half, restored to source-row order",
        "source_row_indices_zero_based": list(MODULE.SMOKE_SOURCE_ROWS),
        "source_image_names_sha256": MODULE.SMOKE_SOURCE_IMAGE_NAMES_SHA256,
        "half_assignment_sha256": MODULE.SMOKE_HALF_ASSIGNMENT_SHA256,
    }
    specs = LAUNCHER.build_run_specs(subject_repo, root)
    expected_specs = {spec.key: spec for spec in MODULE._expected_run_specs(root)}
    runs = {}
    jobs = {}
    job_ids = {}
    fixture_slurm_hashes = {}
    for index, spec in enumerate(specs, start=101):
        key = spec.key
        expected_spec = expected_specs[key]
        assert spec.command == expected_spec.command
        resources = vars(spec.resources)
        star_sha = smoke["sha256"] if spec.phase == "smoke" else MODULE.PREPARED_STAR_SHA256
        script = root / "scripts" / f"{key}.sbatch"
        script.write_text(
            LAUNCHER.render_sbatch(
                spec,
                run_root=root,
                runtime_root=runtime_root,
                subject=subject,
                inputs=prepared_inputs,
                star_sha256=star_sha,
                bind_library=MODULE.RELION_BIND_LIBRARY,
            )
        )
        fixture_slurm_hashes[key] = MODULE._normalized_slurm_sha256(script.read_text(), root, runtime_root)
        for expected in spec.expected_outputs:
            _write(expected)
        paths = MODULE._canonical_paths(root, key, str(index), spec.engine, spec.output_dir)
        _write(paths["stdout_log"], "completed and converged\n")
        _write(paths["stderr_log"], "")
        cpu = resources["ntasks"] * resources["cpus_per_task"]
        tres = f"cpu={cpu},mem=500G,node=1,gres/gpu={resources['h100_gpus']}"
        dependency = (
            "(null)"
            if not spec.depends_on
            else "afterok:101(unfulfilled),afterok:102(unfulfilled)"
        )
        common_scontrol = (
            f"JobId={index} JobName=10202-s6-{key.replace('_', '-')} "
            f"Dependency={dependency} NumCPUs={cpu} NumTasks={resources['ntasks']} "
            f"CPUs/Task={resources['cpus_per_task']} Features=h100 "
            f"Command={script} StdOut={paths['stdout_log']} StdErr={paths['stderr_log']} "
            f"TresPerNode=gres/gpu:h100:{resources['h100_gpus']}"
        )
        submit = _write(
            paths["scontrol_submit"],
            f"{common_scontrol} NumNodes=1-1 ReqTRES={tres} AllocTRES=(null)\n",
        )
        _write(
            paths["scontrol_runtime"],
            f"{common_scontrol} NumNodes=1 ReqTRES={tres} AllocTRES={tres}\n",
        )
        _write(paths["executed_command"], shlex.join(spec.command) + "\n")
        _write(paths["environment"])
        _write(paths["final_environment"], MODULE.FINAL_ENVIRONMENT_TEXT)
        _write(
            paths["gpu_inventory"],
            "\n".join("Product Name : NVIDIA H100 80GB HBM3" for _ in range(resources["h100_gpus"])) + "\n",
        )
        _write(paths["dynamic_libraries"], "all libraries resolved\n")
        _write(
            paths["walltime"],
            json.dumps({"job_id": str(index), "run_key": key, "wall_s": 1}),
        )
        _write(
            paths["gpu_identity"],
            "\n".join(
                f"GPU-00000000-0000-0000-0000-{gpu_index:012d}, NVIDIA H100 80GB HBM3"
                for gpu_index in range(resources["h100_gpus"])
            )
            + "\n",
        )
        _write(paths["completed_marker"], "")
        runs[key] = {
            "path": str(script),
            "sha256": MODULE.sha256_file(script),
            "engine": spec.engine,
            "phase": spec.phase,
            "resources": resources,
            "data_dir": str(spec.data_dir),
            "particle_star": str(spec.data_dir / "particles.star"),
            "particle_star_sha256": star_sha,
            "command": list(spec.command),
            "command_sha256": MODULE.sha256_json(list(spec.command)),
            "expected_outputs": [str(expected) for expected in spec.expected_outputs],
            "depends_on": list(spec.depends_on),
            "slurm_logs": {
                "stdout_template": str(root / "logs" / f"{key}-%j.out"),
                "stderr_template": str(root / "logs" / f"{key}-%j.err"),
            },
        }
        artifacts = {name: str(path) for name, path in paths.items()}
        sbatch_command = ["sbatch", "--parsable"]
        if spec.depends_on:
            sbatch_command.append("--dependency=afterok:101:102")
        sbatch_command.append(str(script))
        jobs[key] = {
            "job_id": str(index),
            "sbatch_command": sbatch_command,
            "artifacts": artifacts,
            "scontrol_submit_sha256": MODULE.sha256_file(submit),
        }
        job_ids[key] = str(index)
    monkeypatch.setattr(MODULE, "SLURM_TEMPLATE_SHA256", fixture_slurm_hashes)
    submission = {
        "job_ids": job_ids,
        "jobs": jobs,
        "dependency": {
            "recovar_full": ["101", "102"],
            "relion_full": ["101", "102"],
            "type": "afterok",
        },
        "submitted_utc": "2026-08-30T19:27:32+00:00",
    }
    _write(root / "submission.json", json.dumps(submission))
    payload = {
        "schema": MODULE.LAUNCH_SCHEMA,
        "status": "submitted",
        "created_utc": "2026-08-30T19:20:00+00:00",
        "run_root": str(root),
        "runtime_root": str(runtime_root),
        "safe_to_delete_markers": [
            str(root / "SAFE_TO_DELETE"),
            str(runtime_root / "SAFE_TO_DELETE"),
        ],
        "subject": subject,
        "preparation": preparation,
        "native_artifacts": native,
        "symmetry": symmetry,
        "scientific_contract": MODULE.EXPECTED_SCIENTIFIC_CONTRACT,
        "smoke_subset": smoke,
        "runs": runs,
        "submission": submission,
    }
    return _write(root / "launch_manifest.json", json.dumps(payload))


def _sacct_runner(command, **_kwargs):
    assert command[0] == "sacct"
    job_id = command[command.index("--jobs") + 1]
    return SimpleNamespace(stdout=f"{job_id}|COMPLETED|0:0\n")


def _store_payload(manifest: Path, payload: dict) -> None:
    manifest.write_text(json.dumps(payload))
    (manifest.parent / "submission.json").write_text(json.dumps(payload["submission"]))


def test_independent_run_specs_match_module_entry_launcher(tmp_path: Path) -> None:
    root = tmp_path / "run"
    expected = MODULE._expected_run_specs(root)
    launched = LAUNCHER.build_run_specs(MODULE.SUBJECT_REPO, root)

    assert tuple(spec.key for spec in expected) == MODULE.RUN_KEYS
    for exact, current in zip(expected, launched, strict=True):
        assert exact.key == current.key
        assert exact.engine == current.engine
        assert exact.phase == current.phase
        assert vars(exact.resources) == vars(current.resources)
        assert exact.command == current.command
        assert exact.expected_outputs == current.expected_outputs
        assert exact.depends_on == current.depends_on
    recovar_command = expected[0].command
    assert recovar_command[5:7] == ("-m", "scripts.run_full_refinement")


def test_audit_binds_commands_outputs_logs_and_exact_tres(monkeypatch, tmp_path: Path) -> None:
    manifest = _launch_fixture(tmp_path, monkeypatch)

    _, audit = MODULE.audit_launch(manifest, runner=_sacct_runner)

    assert set(audit["jobs"]) == set(MODULE.RUN_KEYS)
    for key, job in audit["jobs"].items():
        assert job["accounting"]["state"] == "COMPLETED"
        assert job["accounting"]["exit_code"] == "0:0"
        assert job["tres"]["requested"] == job["tres"]["allocated"]
        assert job["command_sha256"] == MODULE.sha256_json(job["command"])
        assert job["expected_outputs"][0]["sha256"]
        assert job["provenance"]["stdout_log"]["path"].endswith(
            f"{key}-{job['job_id']}.out"
        )


def test_audit_rejects_requested_allocated_tres_difference(monkeypatch, tmp_path: Path) -> None:
    manifest = _launch_fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    job = payload["submission"]["jobs"]["recovar_full"]
    scontrol = Path(job["artifacts"]["scontrol_runtime"])
    scontrol.write_text(scontrol.read_text().replace("AllocTRES=cpu=4", "AllocTRES=cpu=8"))

    with pytest.raises(ValueError, match="ReqTRES != AllocTRES"):
        MODULE.audit_launch(manifest, runner=_sacct_runner)


@pytest.mark.parametrize(
    "mutation",
    ("r1_direct_script", "engine", "phase", "resources", "science", "symmetry"),
)
def test_audit_rejects_self_consistent_manifest_mutations(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _launch_fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    run = payload["runs"]["recovar_smoke"]
    if mutation == "r1_direct_script":
        command = run["command"]
        run["command"] = command[:5] + [str(MODULE.SUBJECT_REPO / "scripts/run_full_refinement.py")] + command[7:]
        run["command_sha256"] = MODULE.sha256_json(run["command"])
        command_path = Path(payload["submission"]["jobs"]["recovar_smoke"]["artifacts"]["executed_command"])
        command_path.write_text(shlex.join(run["command"]) + "\n")
    elif mutation == "engine":
        run["engine"] = "relion"
    elif mutation == "phase":
        run["phase"] = "full"
    elif mutation == "resources":
        run["resources"]["h100_gpus"] = 2
    elif mutation == "science":
        payload["scientific_contract"]["local_only"] = True
    else:
        payload["symmetry"]["requested_label"] = "I2"
    _store_payload(manifest, payload)

    with pytest.raises(ValueError):
        MODULE.audit_launch(manifest, runner=_sacct_runner)


@pytest.mark.parametrize("mutation", ("full_ledger", "submit_scontrol", "smoke_dependency"))
def test_audit_rejects_dependency_mutations(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _launch_fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    if mutation == "full_ledger":
        payload["submission"]["jobs"]["recovar_full"]["sbatch_command"].pop(2)
    elif mutation == "smoke_dependency":
        command = payload["submission"]["jobs"]["recovar_smoke"]["sbatch_command"]
        command.insert(2, "--dependency=afterok:101:102")
    else:
        job = payload["submission"]["jobs"]["recovar_full"]
        scontrol = Path(job["artifacts"]["scontrol_submit"])
        scontrol.write_text(
            scontrol.read_text().replace(
                "afterok:101(unfulfilled),afterok:102(unfulfilled)",
                "afterok:101(unfulfilled)",
            )
        )
        job["scontrol_submit_sha256"] = MODULE.sha256_file(scontrol)
    _store_payload(manifest, payload)

    with pytest.raises(ValueError, match="sbatch command mismatch|Dependency mismatch"):
        MODULE.audit_launch(manifest, runner=_sacct_runner)


@pytest.mark.parametrize("mutation", ("wrong_type", "wrong_count", "duplicate_uuid", "scontrol_type"))
def test_audit_rejects_h100_identity_mutations(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _launch_fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    smoke_job = payload["submission"]["jobs"]["recovar_smoke"]
    if mutation == "wrong_type":
        Path(smoke_job["artifacts"]["gpu_identity"]).write_text(
            "GPU-00000000-0000-0000-0000-000000000000, NVIDIA A100\n"
        )
    elif mutation == "wrong_count":
        Path(smoke_job["artifacts"]["gpu_identity"]).write_text(
            "GPU-00000000-0000-0000-0000-000000000000, NVIDIA H100 80GB HBM3\n"
            "GPU-00000000-0000-0000-0000-000000000001, NVIDIA H100 80GB HBM3\n"
        )
    elif mutation == "scontrol_type":
        runtime = Path(smoke_job["artifacts"]["scontrol_runtime"])
        runtime.write_text(runtime.read_text().replace("Features=h100", "Features=a100"))
    else:
        relion_job = payload["submission"]["jobs"]["relion_smoke"]
        Path(relion_job["artifacts"]["gpu_identity"]).write_text(
            "GPU-00000000-0000-0000-0000-000000000000, NVIDIA H100 80GB HBM3\n" * 2
        )

    with pytest.raises(ValueError, match="H100|GPU|scontrol Features"):
        MODULE.audit_launch(manifest, runner=_sacct_runner)


@pytest.mark.parametrize("target", ("expected_output", "provenance", "evidence_directory"))
def test_audit_rejects_symlink_escape(
    target: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _launch_fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    if target == "evidence_directory":
        path = manifest.parent / "evidence"
        path.rmdir()
        outside = tmp_path / "outside-directory"
        outside.mkdir()
    elif target == "expected_output":
        path = Path(payload["runs"]["recovar_smoke"]["expected_outputs"][0])
        path.unlink()
        outside = _write(tmp_path / "outside-output")
    else:
        path = Path(payload["submission"]["jobs"]["recovar_smoke"]["artifacts"]["environment"])
        path.unlink()
        outside = _write(tmp_path / "outside-provenance")
    path.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|real directory"):
        MODULE.audit_launch(manifest, runner=_sacct_runner)


def test_audit_rejects_noncanonical_artifact_path(monkeypatch, tmp_path: Path) -> None:
    manifest = _launch_fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    artifacts = payload["submission"]["jobs"]["recovar_smoke"]["artifacts"]
    original = Path(artifacts["environment"])
    artifacts["environment"] = str(original.parent / ".." / "provenance" / original.name)
    _store_payload(manifest, payload)

    with pytest.raises(ValueError, match="noncanonical"):
        MODULE.audit_launch(manifest, runner=_sacct_runner)


@pytest.mark.parametrize("mutation", ("commit", "tree", "status"))
def test_subject_checkout_validation_rejects_post_launch_mutation(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subject_repo = tmp_path / "subject"
    python = _write(subject_repo / ".pixi/envs/default/bin/python", "#!/bin/sh\n")
    driver = _write(subject_repo / "scripts/run_full_refinement.py", "# driver\n")
    cuda_source = _write(subject_repo / "recovar/cuda/cuda_backproject.cu", "// cuda\n")
    values = {
        "SUBJECT_REPO": subject_repo,
        "SUBJECT_COMMIT": "a" * 40,
        "SUBJECT_TREE": "b" * 40,
        "SUBJECT_DRIVER_SHA256": MODULE.sha256_file(driver),
        "SUBJECT_CUDA_SOURCE_SHA256": MODULE.sha256_file(cuda_source),
    }
    for name, value in values.items():
        monkeypatch.setattr(MODULE, name, value)

    def git_output(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return "c" * 40 if mutation == "commit" else MODULE.SUBJECT_COMMIT
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "d" * 40 if mutation == "tree" else MODULE.SUBJECT_TREE
        assert arguments == ("status", "--porcelain=v1", "--untracked-files=all")
        return "?? changed" if mutation == "status" else ""

    monkeypatch.setattr(MODULE, "_git_output", git_output)
    payload = {
        "subject": {
            "repo": str(subject_repo),
            "commit": MODULE.SUBJECT_COMMIT,
            "tree": MODULE.SUBJECT_TREE,
            "tree_clean": True,
            "diff_sha256": MODULE.EMPTY_SHA256,
            "python": str(python),
            "driver": str(driver),
            "driver_sha256": MODULE.SUBJECT_DRIVER_SHA256,
            "cuda_source": str(cuda_source),
            "cuda_source_sha256": MODULE.SUBJECT_CUDA_SOURCE_SHA256,
        }
    }

    with pytest.raises(ValueError, match="subject repository"):
        MODULE._validate_subject(payload)


@pytest.mark.parametrize(
    "mutation",
    ("relion_mpi", "relion_binding", "recovar_cuda", "source_commit", "source_tree", "source_status"),
)
def test_native_artifact_validation_rejects_post_launch_mutation(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    relion_mpi = _write(tmp_path / "native/relion_refine_mpi", "relion\n")
    relion_binding = _write(tmp_path / "native/_relion_bind_core.so", "binding\n")
    recovar_cuda = _write(tmp_path / "native/libcuda_backproject_sm90.so", "cuda\n")
    relion_source = tmp_path / "relion-source"
    relion_source.mkdir()
    values = {
        "RELION_MPI": relion_mpi,
        "RELION_MPI_SHA256": MODULE.sha256_file(relion_mpi),
        "RELION_BIND_LIBRARY": relion_binding,
        "RELION_BIND_LIBRARY_SHA256": MODULE.sha256_file(relion_binding),
        "CUDA_LIBRARY": recovar_cuda,
        "CUDA_LIBRARY_SHA256": MODULE.sha256_file(recovar_cuda),
        "RELION_SOURCE": relion_source,
        "RELION_SOURCE_COMMIT": "a" * 40,
        "RELION_BIND_SOURCE_TREE": "b" * 40,
    }
    for name, value in values.items():
        monkeypatch.setattr(MODULE, name, value)

    def git_output(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return "c" * 40 if mutation == "source_commit" else MODULE.RELION_SOURCE_COMMIT
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "d" * 40 if mutation == "source_tree" else MODULE.RELION_BIND_SOURCE_TREE
        assert arguments == ("status", "--porcelain=v1", "--untracked-files=all")
        return "?? changed" if mutation == "source_status" else ""

    monkeypatch.setattr(MODULE, "_git_output", git_output)
    native = {
        "relion_refine_mpi": {"path": str(relion_mpi), "sha256": MODULE.RELION_MPI_SHA256},
        "relion_binding": {"path": str(relion_binding), "sha256": MODULE.RELION_BIND_LIBRARY_SHA256},
        "relion_binding_source": {
            "path": str(relion_source),
            "commit": MODULE.RELION_SOURCE_COMMIT,
            "tree": MODULE.RELION_BIND_SOURCE_TREE,
        },
        "recovar_cuda_sm90": {"path": str(recovar_cuda), "sha256": MODULE.CUDA_LIBRARY_SHA256},
        "relion_runtime_library_dirs": [str(MODULE.RELION_CUDA_LIB_DIR), str(MODULE.RELION_MPI_LIB_DIR)],
    }
    if mutation == "relion_mpi":
        relion_mpi.write_text("mutated\n")
    elif mutation == "relion_binding":
        relion_binding.write_text("mutated\n")
    elif mutation == "recovar_cuda":
        recovar_cuda.write_text("mutated\n")

    with pytest.raises(ValueError, match="changed|dirty"):
        MODULE._validate_native_artifacts({"native_artifacts": native})


@pytest.mark.parametrize(
    "mutation",
    ("submitted_utc", "runtime_marker", "smoke_hash", "particle_stack", "final_environment"),
)
def test_audit_rejects_remaining_envelope_mutations(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _launch_fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    if mutation == "submitted_utc":
        del payload["submission"]["submitted_utc"]
        _store_payload(manifest, payload)
    elif mutation == "runtime_marker":
        Path(payload["safe_to_delete_markers"][1]).unlink()
    elif mutation == "smoke_hash":
        payload["smoke_subset"]["source_image_names_sha256"] = "0" * 64
        _store_payload(manifest, payload)
    elif mutation == "particle_stack":
        MODULE.PARTICLE_STACK.write_bytes(b"mutated")
    else:
        final_env = payload["submission"]["jobs"]["recovar_smoke"]["artifacts"]["final_environment"]
        Path(final_env).write_text("RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=1\n")

    with pytest.raises(ValueError):
        MODULE.audit_launch(manifest, runner=_sacct_runner)


def test_analysis_uses_full_stdout_symlinks_and_pinned_commands(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "evidence").mkdir(parents=True)
    recovar_stdout = _write(root / "logs/recovar_full-103.out")
    relion_stdout = _write(root / "logs/relion_full-104.out")
    raw_collector = _write(tmp_path / "comparison/scripts/collect_metrics.py")
    producer = _write(tmp_path / "collect_em_k1_science_diagnostics.py")
    monkeypatch.setattr(MODULE, "RAW_COLLECTOR", raw_collector)
    monkeypatch.setattr(MODULE, "RAW_COLLECTOR_SHA256", MODULE.sha256_file(raw_collector))
    monkeypatch.setattr(MODULE, "DIAGNOSTIC_PRODUCER", producer)
    monkeypatch.setattr(MODULE, "DIAGNOSTIC_PRODUCER_SHA256", MODULE.sha256_file(producer))
    payload = {
        "run_root": str(root),
        "subject": {"python": sys.executable, "repo": str(tmp_path)},
        "runs": {
            "recovar_full": {
                "data_dir": str(root / "inputs/full"),
                "output_dir": str(root / "outputs/recovar_full"),
            },
            "relion_full": {"output_dir": str(root / "outputs/relion_full")},
        },
    }
    recovar_dir = Path(payload["runs"]["recovar_full"]["output_dir"])
    relion_dir = Path(payload["runs"]["relion_full"]["output_dir"])
    map_paths = {
        "recovar_final_sha256": _write(recovar_dir / "final_merged.mrc"),
        "recovar_half1_sha256": _write(recovar_dir / "final_half1_unfil.mrc"),
        "recovar_half2_sha256": _write(recovar_dir / "final_half2_unfil.mrc"),
        "relion_final_sha256": _write(relion_dir / "run_class001.mrc"),
        "relion_half1_sha256": _write(relion_dir / "run_half1_class001_unfil.mrc"),
        "relion_half2_sha256": _write(relion_dir / "run_half2_class001_unfil.mrc"),
    }
    audit = {
        "jobs": {
            "recovar_full": {"provenance": {"stdout_log": {"path": str(recovar_stdout)}}},
            "relion_full": {"provenance": {"stdout_log": {"path": str(relion_stdout)}}},
        }
    }
    calls = []

    def runner(command, **_kwargs):
        calls.append(list(command))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        if Path(command[1]) == raw_collector:
            _write(
                output / "metrics.json",
                json.dumps(
                    {
                        "schema": MODULE.COLLECTOR_SCHEMA,
                        "scientifically_valid": True,
                        "convergence_and_topology": {
                            "recovar_converged": True,
                            "relion_converged": True,
                        },
                        "artifacts": {
                            key: MODULE.sha256_file(path) for key, path in map_paths.items()
                        },
                    }
                ),
            )
            values = np.ones(8)
            np.savez(
                output / "fsc_curves.npz",
                **{
                    key: values
                    for key in (
                        "relion_final_half_fsc",
                        "recovar_final_half_fsc",
                        "final_cross_engine_raw",
                        "final_cross_engine_half1",
                        "final_cross_engine_half2",
                    )
                },
            )
        else:
            mask = _write(output / "common_soft_mask.mrc")
            fields = MODULE.ALIGNED_FIELDS | MODULE.MASKED_FIELDS
            np.savez(output / "science_diagnostic_curves.npz", **{key: np.ones(8) for key in fields})
            _write(
                output / "science_diagnostics.json",
                json.dumps(
                    {
                        "schema": MODULE.SCIENCE_SCHEMA,
                        "diagnostics": {
                            "proper_so3_alignment": {
                                "applied_unchanged_to": ["merged", "half1", "half2"],
                                "no_reflection": True,
                                "sign_fit": False,
                                "scale_fit": False,
                            },
                            "common_mask": {"path": str(mask)},
                        },
                    }
                ),
            )
        return SimpleNamespace(stdout="")

    collector, analysis = MODULE._run_analysis(payload, audit, runner=runner)

    logs = root / "evidence/collector_logs"
    assert (logs / "recovar_refine.log").resolve() == recovar_stdout.resolve()
    assert (logs / "relion_refine.log").resolve() == relion_stdout.resolve()
    assert collector["collector_sha256"] == MODULE.RAW_COLLECTOR_SHA256
    assert len(calls) == 2
    assert analysis["commands"]["proper_so3"]["sha256"] == MODULE.sha256_json(calls[1])
    assert set(analysis["analysis_artifacts"]["curve_archive"]["fields"]) == (
        MODULE.ALIGNED_FIELDS | MODULE.MASKED_FIELDS
    )
