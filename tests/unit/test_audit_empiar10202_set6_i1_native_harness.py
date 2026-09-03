from __future__ import annotations

import dataclasses
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/audit_empiar10202_set6_i1_native_harness.py"
SPEC = importlib.util.spec_from_file_location("audit_empiar10202_set6_i1_native_harness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

BUILDER_SCRIPT = Path(__file__).parents[2] / "scripts/build_empiar10202_set6_i1_standalone_replacement.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("standalone_replacement_builder_for_test", BUILDER_SCRIPT)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
sys.modules[BUILDER_SPEC.name] = BUILDER
BUILDER_SPEC.loader.exec_module(BUILDER)

FINALIZER_SCRIPT = Path(__file__).parents[2] / "scripts/finalize_empiar10202_set6_i1_evidence.py"
FINALIZER_SPEC = importlib.util.spec_from_file_location("native_evidence_finalizer_for_test", FINALIZER_SCRIPT)
assert FINALIZER_SPEC is not None and FINALIZER_SPEC.loader is not None
FINALIZER = importlib.util.module_from_spec(FINALIZER_SPEC)
sys.modules[FINALIZER_SPEC.name] = FINALIZER
FINALIZER_SPEC.loader.exec_module(FINALIZER)

SUMMARIZER_SCRIPT = Path(__file__).parents[2] / "scripts/summarize_em_k1_realdata_science_equivalence.py"
SUMMARIZER_SPEC = importlib.util.spec_from_file_location("native_evidence_summarizer_for_test", SUMMARIZER_SCRIPT)
assert SUMMARIZER_SPEC is not None and SUMMARIZER_SPEC.loader is not None
SUMMARIZER = importlib.util.module_from_spec(SUMMARIZER_SPEC)
sys.modules[SUMMARIZER_SPEC.name] = SUMMARIZER
SUMMARIZER_SPEC.loader.exec_module(SUMMARIZER)


def _write(path: Path, text: str = "sealed\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _scontrol(
    *,
    job_id: str,
    job_name: str,
    script: Path,
    root: Path,
    key: str,
    resources: dict,
    allocated: bool,
) -> str:
    cpu = resources["ntasks"] * resources["cpus_per_task"]
    tres = f"cpu={cpu},mem={resources['memory']},node=1,billing=40,gres/gpu={resources['h100_gpus']}"
    alloc = tres if allocated else "(null)"
    return (
        f"JobId={job_id} JobName={job_name} NumCPUs={cpu} NumTasks={resources['ntasks']} "
        f"CPUs/Task={resources['cpus_per_task']} Features=h100 Command={script} "
        f"StdOut={root}/logs/{key}-{job_id}.out StdErr={root}/logs/{key}-{job_id}.err "
        f"ReqTRES={tres} AllocTRES={alloc} TresPerNode=gres/gpu:h100:{resources['h100_gpus']}\n"
    )


def _gpu_inventory(count: int) -> str:
    rows = [f"Attached GPUs : {count}"]
    for index in range(count):
        rows.extend(
            [
                "    Product Name : NVIDIA H100 80GB HBM3",
                f"    GPU UUID : GPU-00000000-0000-0000-0000-{index:012d}",
            ]
        )
    return "\n".join(rows) + "\n"


def _gpu_identity(count: int) -> str:
    return "".join(f"GPU-00000000-0000-0000-0000-{index:012d}, NVIDIA H100 80GB HBM3\n" for index in range(count))


def _subject_values(repo: Path, cuda_lib: Path, cuda_sha: str) -> dict[str, str]:
    return {
        "SUBJECT_REPO": str(repo),
        "SUBJECT_COMMIT": "a" * 40,
        "SUBJECT_TREE": "b" * 40,
        "CUDA_LIB": str(cuda_lib),
        "CUDA_LIB_SHA256": cuda_sha,
    }


def _write_subject_file(path: Path, values: dict[str, str], *, lowercase: bool) -> Path:
    return _write(
        path,
        "".join(f"{name.lower() if lowercase else name}={shlex.quote(value)}\n" for name, value in values.items()),
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    completed_full: bool = False,
) -> tuple[Path, dict[str, str], dict[str, str], object]:
    root = tmp_path / "native"
    runtime = tmp_path / "runtime" / root.name
    for path in (
        root / "config",
        root / "scripts",
        root / "logs",
        root / "provenance",
        root / "inputs/smoke",
        root / "inputs/full",
        root / "outputs",
        runtime,
    ):
        path.mkdir(parents=True)
    _write(root / "SAFE_TO_DELETE", "")
    _write(runtime / "SAFE_TO_DELETE", "")
    monkeypatch.setattr(MODULE, "ALLOWED_ROOT", tmp_path)

    subject_repo = tmp_path / "subject"
    python = _write(subject_repo / ".pixi/envs/default/bin/python")
    python.chmod(0o755)
    _write(subject_repo / "scripts/run_full_refinement.py", "# driver\n")
    cuda_source = _write(subject_repo / "recovar/cuda/cuda_backproject.cu", "// cuda source\n")
    cuda_lib = _write(tmp_path / "native-lib/libcuda.so", "cuda library\n")
    cuda_sha = MODULE.sha256_file(cuda_lib)
    subject_values = _subject_values(subject_repo, cuda_lib, cuda_sha)
    _write_subject_file(root / "config/subject.env", subject_values, lowercase=False)
    monkeypatch.setattr(MODULE, "CUDA_SOURCE_SHA256", MODULE.sha256_file(cuda_source))

    preparation = _write(tmp_path / "prepared/preparation_manifest.json", "{}\n")
    full_star = _write(root / "inputs/full/particles.star", "full star\n")
    smoke_star = _write(root / "inputs/smoke/particles.star", "smoke star\n")
    rec_reference = _write(tmp_path / "prepared/reference_rec.mrc", "rec ref\n")
    rel_reference = _write(tmp_path / "prepared/reference_rel.mrc", "rel ref\n")
    particle_stack = _write(tmp_path / "particles.mrcs", "particles\n")
    for phase in ("smoke", "full"):
        _write(root / f"inputs/{phase}/reference_init.mrc", rec_reference.read_text())
        _write(root / f"inputs/{phase}/reference_init_relion.mrc", rel_reference.read_text())
    monkeypatch.setattr(MODULE, "PREPARATION_SHA256", MODULE.sha256_file(preparation))
    monkeypatch.setattr(MODULE, "FULL_STAR_SHA256", MODULE.sha256_file(full_star))
    monkeypatch.setattr(MODULE, "SMOKE_STAR_SHA256", MODULE.sha256_file(smoke_star))
    monkeypatch.setattr(MODULE, "RECOVAR_REFERENCE_SHA256", MODULE.sha256_file(rec_reference))
    monkeypatch.setattr(MODULE, "RELION_REFERENCE_SHA256", MODULE.sha256_file(rel_reference))
    monkeypatch.setattr(MODULE, "PARTICLE_STACK_SHA256", MODULE.sha256_file(particle_stack))
    monkeypatch.setattr(MODULE, "PARTICLE_STACK_SIZE_BYTES", particle_stack.stat().st_size)

    relion_mpi = _write(tmp_path / "native-lib/relion_refine_mpi", "relion\n")
    relion_bind = _write(tmp_path / "native-lib/relion_bind.so", "bind\n")
    relion_source = tmp_path / "relion-source"
    relion_source.mkdir()
    monkeypatch.setattr(MODULE, "RELION_MPI_SHA256", MODULE.sha256_file(relion_mpi))
    monkeypatch.setattr(MODULE, "RELION_BIND_SHA256", MODULE.sha256_file(relion_bind))
    monkeypatch.setattr(MODULE, "RELION_SOURCE_COMMIT", "c" * 40)
    monkeypatch.setattr(MODULE, "RELION_SOURCE_TREE", "d" * 40)

    def git_output(repo: Path, *arguments: str) -> str:
        if repo == subject_repo:
            if arguments == ("rev-parse", "HEAD"):
                return subject_values["SUBJECT_COMMIT"]
            if arguments == ("rev-parse", "HEAD^{tree}"):
                return subject_values["SUBJECT_TREE"]
        elif repo == relion_source:
            if arguments == ("rev-parse", "HEAD"):
                return MODULE.RELION_SOURCE_COMMIT
            if arguments == ("rev-parse", "HEAD^{tree}"):
                return MODULE.RELION_SOURCE_TREE
        assert arguments == ("status", "--porcelain=v1", "--untracked-files=all")
        return ""

    monkeypatch.setattr(MODULE, "_git_output", git_output)
    source_manifest = _write(tmp_path / "source-harness/launch_manifest.json", "source\n")
    common = _write(root / "scripts/common.sh")
    validate = _write(root / "scripts/validate_harness.sh")
    submit = _write(root / "submit_when_configured.sh")

    launcher = MODULE._load_launcher()
    specs = launcher.build_run_specs(subject_repo, root)
    job_ids = {key: str(101 + index) for index, key in enumerate(MODULE.RUN_KEYS)}
    states = {
        "recovar_smoke": "FAILED",
        "relion_smoke": "COMPLETED",
        "recovar_full": "COMPLETED" if completed_full else "PENDING",
        "relion_full": "COMPLETED" if completed_full else "PENDING",
    }
    science = {
        "recovar_smoke": {
            "particles": "smoke",
            "iterations": 1,
            "current_size": 800,
            "symmetry": "I1",
            "seed": 10202,
            "classes": 1,
        },
        "relion_smoke": {
            "particles": "smoke",
            "iterations": 1,
            "incr_size": 800,
            "symmetry": "I1",
            "seed": 10202,
            "classes": 1,
        },
        "recovar_full": {"particles": "full", "max_iterations": 50, "symmetry": "I1", "seed": 10202, "classes": 1},
        "relion_full": {"particles": "full", "max_iterations": 50, "symmetry": "I1", "seed": 10202, "classes": 1},
    }
    runs = {}
    for spec in specs:
        output_dir = root / "outputs" / spec.key
        output_dir.mkdir()
        script = _write(
            root / "scripts" / f"{spec.key}.sbatch",
            f"#!/bin/bash\n#SBATCH --job-name=10202-s6-{spec.key.replace('_', '-')}-fixture\n",
        )
        resources = dataclasses.asdict(spec.resources)
        row = {
            "script": str(script),
            "script_sha256": MODULE.sha256_file(script),
            "resources": resources,
            "science": science[spec.key],
        }
        if spec.depends_on:
            row["depends_on"] = list(spec.depends_on)
        runs[spec.key] = row
        job_id = job_ids[spec.key]
        submit_text = _scontrol(
            job_id=job_id,
            job_name=f"10202-s6-{spec.key.replace('_', '-')}-fixture",
            script=script,
            root=root,
            key=spec.key,
            resources=resources,
            allocated=False,
        )
        _write(root / f"provenance/scontrol_submit_{job_id}.txt", submit_text)
        if states[spec.key] != "PENDING":
            runtime_text = _scontrol(
                job_id=job_id,
                job_name=f"10202-s6-{spec.key.replace('_', '-')}-fixture",
                script=script,
                root=root,
                key=spec.key,
                resources=resources,
                allocated=True,
            )
            _write(root / f"provenance/scontrol_{job_id}.txt", runtime_text)
            command = (
                launcher._recovar_command(subject_repo, spec.data_dir, output_dir, smoke=spec.phase == "smoke")
                if spec.engine == "recovar"
                else launcher._relion_command(spec.data_dir, output_dir, smoke=spec.phase == "smoke")
            )
            _write(root / f"provenance/command_{job_id}.sh", shlex.join(command) + "\n")
            _write_subject_file(root / f"provenance/subject_{job_id}.txt", subject_values, lowercase=True)
            environment = "PYTHONNOUSERSITE=1\nXLA_PYTHON_CLIENT_PREALLOCATE=false\n"
            if spec.engine == "recovar":
                environment += f"RECOVAR_CUDA_LIB={cuda_lib}\nRECOVAR_EXPECTED_REPO_ROOT={subject_repo}\n"
            _write(root / f"provenance/environment_{job_id}.txt", environment)
            _write(
                root / f"provenance/final_env_{job_id}.txt",
                "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset\nRECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset\n",
            )
            _write(root / f"provenance/nvidia_smi_{job_id}.txt", _gpu_inventory(resources["h100_gpus"]))
            ldd = "ldd_relion_bind" if spec.engine == "recovar" else "ldd_relion"
            _write(root / f"provenance/{ldd}_{job_id}.txt", "all resolved\n")
            _write(root / f"logs/{spec.key}-{job_id}.out", "started\n")
            error = "CUDA: out of memory\n" if states[spec.key] == "FAILED" else ""
            _write(root / f"logs/{spec.key}-{job_id}.err", error)
        if states[spec.key] == "COMPLETED":
            for expected in spec.expected_outputs:
                _write(expected)
            _write(output_dir / "COMPLETED", "")
            _write(
                output_dir / "slurm_walltime.json",
                json.dumps({"job_id": job_id, "run_key": spec.key, "wall_s": 1}),
            )
            _write(output_dir / "gpu_identity.txt", _gpu_identity(resources["h100_gpus"]))

    payload = {
        "schema": MODULE.NATIVE_LAUNCH_SCHEMA,
        "created_utc": "2026-08-30T23:19:05Z",
        "status": "prepared_unsubmitted_subject_unconfigured",
        "run_root": str(root),
        "runtime_root": str(runtime),
        "safe_to_delete": {
            "run_marker": str(root / "SAFE_TO_DELETE"),
            "runtime_marker": str(runtime / "SAFE_TO_DELETE"),
        },
        "source_harness": {
            "run_root": str(source_manifest.parent),
            "launch_manifest": str(source_manifest),
            "launch_manifest_sha256": MODULE.sha256_file(source_manifest),
            "recovar_smoke_job_id": "1",
            "relion_smoke_job_id": "2",
            "recovar_full_job_id": "3",
            "relion_full_job_id": "4",
        },
        "subject": {
            "configuration": str(root / "config/subject.env"),
            "required_fields": list(MODULE.SUBJECT_FIELDS),
            "configured": False,
            "submission_gate": "sealed",
        },
        "harness_delta_from_r3": {
            "scientific_command_delta": "none after normalizing paths",
            "resource_delta": "none",
            "python_cache_fix": [],
            "command_equivalence_gate": {key: "passed" for key in MODULE.RUN_KEYS},
        },
        "inputs": {
            "preparation_manifest": {"path": str(preparation), "sha256": MODULE.PREPARATION_SHA256},
            "particle_stack": {
                "path": str(particle_stack),
                "size_bytes": particle_stack.stat().st_size,
                "first_mib_sha256": "unused",
                "full_sha256": MODULE.PARTICLE_STACK_SHA256,
            },
            "full_particles_star": {"path": str(full_star), "sha256": MODULE.FULL_STAR_SHA256},
            "smoke_particles_star": {"path": str(smoke_star), "sha256": MODULE.SMOKE_STAR_SHA256},
            "recovar_reference": {"canonical_path": str(rec_reference), "sha256": MODULE.RECOVAR_REFERENCE_SHA256},
            "relion_reference": {"canonical_path": str(rel_reference), "sha256": MODULE.RELION_REFERENCE_SHA256},
        },
        "immutable_native_artifacts": {
            "relion_refine_mpi": {"path": str(relion_mpi), "sha256": MODULE.RELION_MPI_SHA256},
            "relion_binding": {"path": str(relion_bind), "sha256": MODULE.RELION_BIND_SHA256},
            "relion_binding_source": {
                "path": str(relion_source),
                "commit": MODULE.RELION_SOURCE_COMMIT,
                "tree": MODULE.RELION_SOURCE_TREE,
            },
        },
        "runs": runs,
        "harness_scripts": {
            "common": {"path": str(common), "sha256": MODULE.sha256_file(common)},
            "validate": {"path": str(validate), "sha256": MODULE.sha256_file(validate)},
            "submit": {"path": str(submit), "sha256": MODULE.sha256_file(submit)},
        },
        "submission": None,
    }
    manifest = _write(root / "launch_manifest.json", json.dumps(payload))
    _write(
        root / "submission.json",
        json.dumps(
            {
                "schema": MODULE.NATIVE_SUBMISSION_SCHEMA,
                "dependency": f"afterok:{job_ids['recovar_smoke']}:{job_ids['relion_smoke']}",
                "job_ids": job_ids,
            }
        ),
    )

    def runner(command, **_kwargs):
        job_id = command[command.index("--jobs") + 1]
        key = next(name for name, value in job_ids.items() if value == job_id)
        state = states[key]
        if command[0] == "squeue":
            reason = "DependencyNeverSatisfied" if state == "PENDING" else "None"
            return SimpleNamespace(stdout=f"{job_id}|{state}|{reason}\n", returncode=0)
        resources = dataclasses.asdict(next(spec.resources for spec in specs if spec.key == key))
        cpu = resources["ntasks"] * resources["cpus_per_task"]
        tres = f"cpu={cpu},mem={resources['memory']},node=1,billing=40,gres/gpu={resources['h100_gpus']}"
        exit_code = "0:0" if state in {"COMPLETED", "PENDING"} else "1:0"
        allocated = "" if state == "PENDING" else tres
        return SimpleNamespace(
            stdout=f"{job_id}|{state}|{exit_code}|{tres}|{allocated}|1024K\n",
            returncode=0,
        )

    return manifest, job_ids, states, runner


def _standalone_scontrol(
    *,
    profile,
    script: Path,
    allocated: bool,
) -> str:
    tres = "cpu=4,mem=500G,node=1,billing=40,gres/gpu=1"
    alloc = tres if allocated else "(null)"
    nodes = "1" if allocated else "1-1"
    state = "RUNNING" if allocated else "PENDING"
    return (
        f"JobId={profile.job_id} JobName={profile.job_name} Account=gilles QOS=della-cryoem "
        f"JobState={state} Dependency=(null) Restarts=0 Partition=cryoem TimeLimit=5-00:00:00 "
        f"NumNodes={nodes} NumCPUs=4 NumTasks=1 CPUs/Task=4 Features=h100 OverSubscribe=OK "
        f"Command={script} StdOut={profile.run_root}/logs/recovar-full-{profile.job_id}.out "
        f"StdErr={profile.run_root}/logs/recovar-full-{profile.job_id}.err "
        f"ReqTRES={tres} AllocTRES={alloc} TresPerNode=gres/gpu:h100:1\n"
    )


def _standalone_npz(path: Path, profile, data_dir: Path, **overrides) -> None:
    convergence_iteration = 3
    values = {
        "n_iterations": np.int64(50),
        "convergence_iteration": np.int64(convergence_iteration),
        "convergence_has_converged": np.bool_(True),
        "final_all_data_ran": np.bool_(True),
        "final_all_data_grid_correct": np.bool_(False),
        "current_sizes": np.asarray([128, 256, 512]),
        "fsc_final_all_data": np.asarray([1.0, 0.8, 0.2]),
        "n_images": np.int64(profile.particle_count),
        "half1_indices": np.asarray([0, 2, 4]),
        "half2_indices": np.asarray([1, 3, 5]),
        "symmetry_label": np.asarray("I1"),
        "symmetry_family": np.asarray("icosahedral"),
        "symmetry_operator_count": np.int64(60),
        "symmetry_operator_sha256": np.asarray(
            "093a0876b93610ec141c87840ae3ff4dc4491b27dec87143558358ef556557b8"
        ),
        "firstiter_cc_effective": np.bool_(True),
        "tau2_fudge": np.float64(1.0),
        "tau2_fudge_source": np.asarray("explicit CLI"),
        "initial_pose_source_requested": np.asarray("input-star"),
        "initial_pose_source_resolved": np.asarray("input_star"),
        "initial_pose_source_path": np.asarray(str(data_dir / "particles.star")),
        "initial_pose_source_sha256": np.asarray(MODULE.FULL_STAR_SHA256),
        "git_commit": np.asarray(profile.source_commit),
        "git_branch": np.asarray("<detached>"),
        "git_dirty_count": np.int64(0),
        "git_diff_sha256": np.asarray(MODULE.EMPTY_SHA256),
        "git_status_porcelain": np.asarray(""),
        "perturb_replay_restart_state_iterations": np.asarray([], dtype=np.int64),
        "diagnostic_final_manifest_paths": np.asarray([], dtype=str),
        "diagnostic_final_manifest_sha256": np.asarray([], dtype=str),
        "state_swap_probe_applied_relion_iterations": np.asarray([], dtype=np.int64),
        "state_swap_probe_replay_override_keys": np.asarray([], dtype=str),
        "state_swap_probe_required_replay_override_keys": np.asarray([], dtype=str),
        "perturb_replay_restart_provenance_path": np.asarray(""),
        "perturb_replay_restart_provenance_sha256": np.asarray(""),
        "relion_projector_source_manifest_sha256": np.asarray(""),
        "relion_projector_capture_dir": np.asarray(""),
        "relion_projector_capture_manifest": np.asarray(""),
        "frozen_boundary_dir": np.asarray(""),
        "frozen_boundary_manifest_sha256": np.asarray(""),
        "frozen_boundary_sha256": np.asarray(""),
        "diagnostic_final_source_results_path": np.asarray(""),
        "diagnostic_final_source_results_sha256": np.asarray(""),
        "diagnostic_final_source_git_commit": np.asarray(""),
        "state_swap_probe_variant": np.asarray(""),
        "relion_projector_replay_slot": np.int64(-1),
        "frozen_boundary_completed_relion_iteration": np.int64(-1),
        "diagnostic_final_source_completed_relion_iteration": np.int64(-1),
        "state_swap_probe_target_relion_iteration": np.int64(-1),
        "state_swap_probe_loop_index": np.int64(-1),
    }
    values.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _standalone_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    manifest, job_ids, _, base_runner = _fixture(tmp_path, monkeypatch, completed_full=True)
    parent_payload = json.loads(manifest.read_text())
    run_root = tmp_path / "standalone"
    runtime_root = tmp_path / "runtime/standalone"
    source_repo = tmp_path / "standalone-source/checkout"
    python = _write(tmp_path / "toolchain/.pixi/envs/default/bin/python", "python\n")
    python.chmod(0o755)
    driver = _write(source_repo / "scripts/run_full_refinement.py", "# standalone driver\n")
    cuda_source = _write(source_repo / "recovar/cuda/cuda_backproject.cu", "// standalone cuda\n")
    cuda_library = _write(run_root / "sealed_input/libcuda.so", "postbuild cuda\n")
    for directory in (
        run_root / "jobs",
        run_root / "logs",
        run_root / "provenance",
        run_root / "outputs/recovar_full",
        run_root / "outputs/intermediates",
        runtime_root / "recovar_full_999",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _write(run_root / "SAFE_TO_DELETE", "")
    _write(runtime_root / "SAFE_TO_DELETE", "")
    _write(runtime_root / "recovar_full_999/SAFE_TO_DELETE", "")
    script = _write(
        run_root / "jobs/recovar_full.sbatch",
        "#!/usr/bin/env bash\n#SBATCH --job-name=10202-s6-nopad-fixture\n"
        "#SBATCH --gres=gpu:h100:1\n",
    )
    preflight_readme = _write(run_root / "README.md", "launch-time readme\n")
    preflight_readme_sha = MODULE.sha256_file(preflight_readme)
    preflight_cuda_sha = MODULE.hashlib.sha256(b"preflight cuda bytes").hexdigest()
    source_commit = "e" * 40
    source_tree = "f" * 40
    profile_template = next(iter(MODULE.STANDALONE_REPLACEMENT_PROFILES.values()))
    profile = dataclasses.replace(
        profile_template,
        name="fixture_standalone_999",
        job_id="999",
        job_name="10202-s6-nopad-fixture",
        run_root=run_root,
        runtime_root=runtime_root,
        script_sha256=MODULE.sha256_file(script),
        source_repo=source_repo,
        source_commit=source_commit,
        source_tree=source_tree,
        python=python,
        driver_sha256=MODULE.sha256_file(driver),
        cuda_source_sha256=MODULE.sha256_file(cuda_source),
        cuda_library_relative_path=Path("sealed_input/libcuda.so"),
        cuda_preflight_sha256=preflight_cuda_sha,
        cuda_postbuild_sha256=MODULE.sha256_file(cuda_library),
        cuda_postbuild_size_bytes=cuda_library.stat().st_size,
        preflight_readme_sha256=preflight_readme_sha,
        particle_count=6,
        half1_count=3,
        half2_count=3,
    )

    base_git_output = MODULE._git_output

    def git_output(repo: Path, *arguments: str) -> str:
        if repo != source_repo:
            return base_git_output(repo, *arguments)
        values = {
            ("rev-parse", "HEAD"): profile.source_commit,
            ("rev-parse", "HEAD^{tree}"): profile.source_tree,
            ("rev-parse", "--abbrev-ref", "HEAD"): "HEAD",
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
        }
        return values[arguments]

    monkeypatch.setattr(MODULE, "_git_output", git_output)

    data_dir = tmp_path / "standalone-inputs/full"
    data_dir.mkdir(parents=True)
    for name in ("particles.star", "reference_init.mrc", "reference_init_relion.mrc"):
        _write(data_dir / name, (manifest.parent / "inputs/full" / name).read_text())
    particle_stack = _write(tmp_path / "standalone-inputs/alternate-particles.mrcs", "particles\n")

    subject = {
        "source_repo": str(profile.source_repo),
        "source_commit": profile.source_commit,
        "source_tree": profile.source_tree,
        "allocator": "platform",
        "preallocate": "false",
        "big_jit_max_bucket_rotations": "256",
        "unused_native_projection_padding": "skipped_when_relion_projector_supplied",
        "symmetry": "I1",
        "particle_count": str(profile.particle_count),
        "half1_count": str(profile.half1_count),
        "half2_count": str(profile.half2_count),
    }
    _write(
        run_root / "provenance/subject-999.txt",
        "".join(f"{name}={value}\n" for name, value in subject.items()),
    )
    bind_dir = Path(parent_payload["immutable_native_artifacts"]["relion_binding"]["path"]).parent
    relion_source = Path(parent_payload["immutable_native_artifacts"]["relion_binding_source"]["path"])
    job_runtime = runtime_root / "recovar_full_999"
    environment = {
        "JAX_COMPILATION_CACHE_DIR": str(job_runtime / "jax_cache"),
        "RECOVAR_CUDA_LIB": str(cuda_library),
        "RECOVAR_EXACT_LOCAL_BIG_JIT_MAX_BUCKET_ROTATIONS": "256",
        "RECOVAR_RELION_BIND_BUILD_DIR": str(bind_dir),
        "RELION_SRC_DIR": str(relion_source / "src"),
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(job_runtime / "tmp"),
        "PIXI_HOME": str(job_runtime / "pixi_home"),
        "RATTLER_CACHE_DIR": str(job_runtime / "rattler_cache"),
    }
    _write(
        run_root / "provenance/environment-999.txt",
        "".join(f"{name}={value}\n" for name, value in sorted(environment.items())),
    )
    _write(
        run_root / "provenance/import-999.txt",
        f"recovar={source_repo}/recovar/__init__.py\n"
        f"jax={python.parent.parent}/lib/python3.11/site-packages/jax/__init__.py\n"
        "devices=[CudaDevice(id=0)]\n",
    )
    _write(run_root / "provenance/nvidia-smi-999.txt", _gpu_inventory(1))
    for name in ("ldd-cuda-pre-submit.txt", "ldd-cuda-999.txt", "ldd-relion-bind-999.txt"):
        _write(run_root / "provenance" / name, "all resolved\n")
    _write(run_root / "provenance/job_id.txt", "999\n")
    _write(
        run_root / "provenance/submitted_scontrol_retry.txt",
        _standalone_scontrol(profile=profile, script=script, allocated=False),
    )
    _write(
        run_root / "provenance/scontrol-999.txt",
        _standalone_scontrol(profile=profile, script=script, allocated=True),
    )
    launcher = MODULE._load_launcher()
    command = list(
        launcher._recovar_command(
            profile.source_repo,
            data_dir,
            run_root / "outputs/recovar_full",
            smoke=False,
        )
    )
    command[4] = str(profile.python)
    command.extend(
        [
            "--save_intermediates_dir",
            str(run_root / "outputs/intermediates"),
            "--save_intermediates_skip_unregularized",
        ]
    )
    _write(run_root / "provenance/command-999.sh", shlex.join(command) + "\n")

    preflight_ledger = _write(
        run_root / "provenance/pre_submission_retry_sha256.txt",
        f"{profile.script_sha256}  {script}\n"
        f"{profile.preflight_readme_sha256}  {preflight_readme}\n"
        f"{profile.cuda_preflight_sha256}  {cuda_library}\n",
    )
    profile = dataclasses.replace(profile, preflight_ledger_sha256=MODULE.sha256_file(preflight_ledger))
    postbuild = _write(
        run_root / "provenance/cuda-postbuild-999.txt",
        "job_id=999\n"
        f"path={cuda_library}\n"
        f"preflight_sha256={profile.cuda_preflight_sha256}\n"
        f"postbuild_sha256={profile.cuda_postbuild_sha256}\n"
        f"postbuild_size_bytes={profile.cuda_postbuild_size_bytes}\n"
        "postbuild_mtime=2026-09-03T01:54:48.530225457-04:00\n"
        f"source_cuda_sha256={profile.cuda_source_sha256}\n"
        f"reason={profile.postbuild_reason}\n",
    )
    profile = dataclasses.replace(profile, postbuild_record_sha256=MODULE.sha256_file(postbuild))
    monkeypatch.setattr(MODULE, "STANDALONE_REPLACEMENT_PROFILES", {profile.name: profile})
    # The launch-time README hash remains bound through the ledger even if the
    # explanatory README is amended after submission.
    preflight_readme.write_text("post-launch explanation\n")

    output_dir = run_root / "outputs/recovar_full"
    results = output_dir / "refinement_results.npz"
    _standalone_npz(results, profile, data_dir)
    for name in ("final_merged.mrc", "final_half1_unfil.mrc", "final_half2_unfil.mrc"):
        _write(output_dir / name, f"{name}\n")
    walltime = _write(
        output_dir / "slurm_walltime.json",
        json.dumps(
            {
                "schema": "recovar.em.walltime.v1",
                "job_id": "999",
                "start_epoch": 100,
                "end_epoch": 125,
                "wall_s": 25,
            }
        )
        + "\n",
    )
    output_paths = [
        results,
        output_dir / "final_merged.mrc",
        output_dir / "final_half1_unfil.mrc",
        output_dir / "final_half2_unfil.mrc",
        walltime,
    ]
    _write(
        output_dir / "output_sha256.txt",
        "".join(f"{MODULE.sha256_file(path)}  {path}\n" for path in output_paths),
    )
    _write(output_dir / "COMPLETED", "")
    uuid = "GPU-00000000-0000-0000-0000-000000000000"
    _write(
        run_root / "logs/recovar-full-999-hbm.csv",
        "timestamp,index,uuid,memory_used_mib,memory_free_mib,gpu_utilization_percent\n"
        f"2026-09-03T01:00:00-04:00,0,{uuid},10,81000,5\n"
        f"2026-09-03T01:00:05-04:00,0,{uuid},42,80968,80\n",
    )
    _write(output_dir / "hbm_summary.txt", "peak_hbm_mib=42\n")
    _write(run_root / "logs/recovar-full-999-time-v.txt", "\tExit status: 0\n")
    _write(
        run_root / "logs/recovar-full-999.out",
        f"nvcc -o {cuda_library} cuda_backproject.cu\n",
    )
    _write(
        run_root / "logs/recovar-full-999.err",
        f"{cuda_library} is older than its source\n"
        f"Building {cuda_library}\n"
        "CUDA backproject/project kernels enabled\n"
        "Convergence reached at iteration 3.\n"
        "=== RELION final all-data Nyquist iteration ===\n"
        "Final iter complete: current_size=800\n",
    )

    def record(path: Path, *, nonempty: bool = True) -> dict:
        return MODULE._file_record(path, nonempty=nonempty)

    artifact_paths = MODULE._standalone_artifact_paths(profile)
    empty_artifacts = {
        "run_safe_to_delete",
        "runtime_safe_to_delete",
        "job_runtime_safe_to_delete",
        "completed_marker",
    }
    artifacts = {
        name: record(path, nonempty=name not in empty_artifacts)
        for name, path in artifact_paths.items()
    }
    payload = {
        "schema": MODULE.STANDALONE_REPLACEMENT_SCHEMA,
        "profile": profile.name,
        "parent_launch_manifest": {"path": str(manifest), "sha256": MODULE.sha256_file(manifest)},
        "run_key": "recovar_full",
        "replaces_job_id": job_ids["recovar_full"],
        "job_id": profile.job_id,
        "run_root": str(run_root),
        "script": record(script),
        "source": {
            "repo": str(profile.source_repo),
            "commit": profile.source_commit,
            "tree": profile.source_tree,
            "python": str(profile.python),
            "driver_sha256": profile.driver_sha256,
            "cuda_source_sha256": profile.cuda_source_sha256,
        },
        "inputs": {
            "data_dir": str(data_dir),
            "particles_star": record(data_dir / "particles.star"),
            "recovar_reference": record(data_dir / "reference_init.mrc"),
            "relion_reference": record(data_dir / "reference_init_relion.mrc"),
            "particle_stack": record(particle_stack),
        },
        "cuda_rebuild": {
            "library": {
                "path": str(cuda_library),
                "preflight_sha256": profile.cuda_preflight_sha256,
                "postbuild_sha256": profile.cuda_postbuild_sha256,
                "postbuild_size_bytes": profile.cuda_postbuild_size_bytes,
            },
            "preflight_ledger_sha256": profile.preflight_ledger_sha256,
            "postbuild_record_sha256": profile.postbuild_record_sha256,
            "reason": profile.postbuild_reason,
        },
        "artifacts": artifacts,
        "reason": "replace the failed parent recovar_full with the corrected standalone run",
    }
    replacement = _write(tmp_path / "standalone-replacement.json", json.dumps(payload))

    def runner(command, **kwargs):
        job_id = command[command.index("--jobs") + 1]
        if job_id != profile.job_id:
            return base_runner(command, **kwargs)
        tres = "cpu=4,mem=500G,node=1,billing=40,gres/gpu=1"
        return SimpleNamespace(
            stdout=f"{profile.job_id}|COMPLETED|0:0|{tres}|{tres}|1024K\n",
            returncode=0,
        )

    return {
        "manifest": manifest,
        "replacement": replacement,
        "runner": runner,
        "profile": profile,
        "data_dir": data_dir,
        "run_root": run_root,
    }


def _reseal_standalone_artifact(fixture: dict, name: str) -> None:
    replacement = fixture["replacement"]
    payload = json.loads(replacement.read_text())
    path = Path(payload["artifacts"][name]["path"])
    payload["artifacts"][name] = MODULE._file_record(
        path,
        nonempty=name
        not in {
            "run_safe_to_delete",
            "runtime_safe_to_delete",
            "job_runtime_safe_to_delete",
            "completed_marker",
        },
    )
    replacement.write_text(json.dumps(payload))


def _reseal_standalone_outputs(fixture: dict) -> None:
    output_dir = fixture["run_root"] / "outputs/recovar_full"
    paths = [
        output_dir / "refinement_results.npz",
        output_dir / "final_merged.mrc",
        output_dir / "final_half1_unfil.mrc",
        output_dir / "final_half2_unfil.mrc",
        output_dir / "slurm_walltime.json",
    ]
    (output_dir / "output_sha256.txt").write_text(
        "".join(f"{MODULE.sha256_file(path)}  {path}\n" for path in paths)
    )
    _reseal_standalone_artifact(fixture, "refinement_results")
    _reseal_standalone_artifact(fixture, "walltime")
    _reseal_standalone_artifact(fixture, "output_sha256")


def test_standalone_recovar_full_replacement_is_content_bound_and_science_eligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)

    normalized, audit = MODULE.audit_native_launch(
        fixture["manifest"],
        replacement_records=(fixture["replacement"],),
        runner=fixture["runner"],
        require_science_ready=True,
    )

    selected = audit["jobs"]["recovar_full"]
    assert selected["source"] == "standalone_replacement"
    assert selected["replacement_profile"] == fixture["profile"].name
    assert selected["cuda_rebuild"]["preflight_sha256"] == fixture["profile"].cuda_preflight_sha256
    assert selected["cuda_rebuild"]["postbuild_sha256"] == fixture["profile"].cuda_postbuild_sha256
    assert selected["refinement"] == {
        "n_iterations_cap": 50,
        "numbered_iterations": 3,
        "converged": True,
        "final_all_data_ran": True,
        "final_all_data_grid_correct": False,
        "final_fsc_shells": 3,
    }
    assert normalized["runs"]["recovar_full"]["data_dir"] == str(fixture["data_dir"])
    assert normalized["runs"]["recovar_full"]["data_dir"] != str(
        fixture["manifest"].parent / "inputs/full"
    )
    profile_binding = audit["selected_standalone_replacement_profiles"]["recovar_full"]
    assert profile_binding["profile"] == fixture["profile"].name
    assert profile_binding["cuda_postbuild_sha256"] == fixture["profile"].cuda_postbuild_sha256


def test_standalone_replacement_builder_is_deterministic_and_refuses_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(BUILDER, "_load_adapter", lambda: MODULE)
    output = tmp_path / "built-replacement.json"
    arguments = [
        "--parent-launch-manifest",
        str(fixture["manifest"]),
        "--profile",
        fixture["profile"].name,
        "--data-dir",
        str(fixture["data_dir"]),
        "--particle-stack",
        str(tmp_path / "standalone-inputs/alternate-particles.mrcs"),
        "--reason",
        "replace the failed parent recovar_full with the corrected standalone run",
        "--output",
        str(output),
    ]

    assert BUILDER.main(arguments) == 0
    assert json.loads(output.read_text()) == json.loads(fixture["replacement"].read_text())
    with pytest.raises(FileExistsError):
        BUILDER.main(arguments)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("profile", "profile is not accepted"),
        ("parent", "parent binding changed"),
        ("job", "job ID changed"),
        ("extra", "record fields changed"),
    ),
)
def test_standalone_replacement_rejects_contract_mutation(
    mutation: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    payload = json.loads(fixture["replacement"].read_text())
    if mutation == "profile":
        payload["profile"] = "unaccepted-run"
    elif mutation == "parent":
        payload["parent_launch_manifest"]["sha256"] = "0" * 64
    elif mutation == "job":
        payload["job_id"] = "1000"
    else:
        payload["unexpected"] = True
    fixture["replacement"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


@pytest.mark.parametrize(
    ("artifact", "old", "new", "message"),
    (
        ("executed_command", "\n", " --skip_final_iteration\n", "executed command changed"),
        (
            "environment",
            "PYTHONNOUSERSITE=1\n",
            "PYTHONNOUSERSITE=1\nRECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=1\n",
            "sensitive environment changed",
        ),
        (
            "scontrol_runtime",
            "AllocTRES=cpu=4,mem=500G",
            "AllocTRES=cpu=4,mem=400G",
            "wrong standalone TRES",
        ),
    ),
)
def test_standalone_replacement_rejects_self_consistent_execution_tamper(
    artifact: str,
    old: str,
    new: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    payload = json.loads(fixture["replacement"].read_text())
    path = Path(payload["artifacts"][artifact]["path"])
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    _reseal_standalone_artifact(fixture, artifact)

    with pytest.raises(ValueError, match=message):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


@pytest.mark.parametrize(
    ("artifact", "old", "new", "message"),
    (
        ("preflight_ledger", "  /", "0  /", "preflight ledger changed"),
        ("cuda_postbuild", "reason=", "reason=tampered-", "CUDA postbuild record changed"),
    ),
)
def test_standalone_replacement_rejects_cuda_chain_tamper(
    artifact: str,
    old: str,
    new: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    payload = json.loads(fixture["replacement"].read_text())
    path = Path(payload["artifacts"][artifact]["path"])
    path.write_text(path.read_text().replace(old, new, 1))
    _reseal_standalone_artifact(fixture, artifact)

    with pytest.raises(ValueError, match=message):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


@pytest.mark.parametrize("artifact", ("preflight_ledger", "cuda_postbuild"))
def test_standalone_cuda_chain_semantics_survive_a_resealed_record_digest(
    artifact: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    payload = json.loads(fixture["replacement"].read_text())
    path = Path(payload["artifacts"][artifact]["path"])
    if artifact == "preflight_ledger":
        old = fixture["profile"].preflight_readme_sha256
        path.write_text(path.read_text().replace(old, "1" * 64))
        _reseal_standalone_artifact(fixture, artifact)
        payload = json.loads(fixture["replacement"].read_text())
        new_digest = payload["artifacts"][artifact]["sha256"]
        profile = dataclasses.replace(fixture["profile"], preflight_ledger_sha256=new_digest)
        payload["cuda_rebuild"]["preflight_ledger_sha256"] = new_digest
        message = "preflight digest chain changed"
    else:
        path.write_text(path.read_text().replace("reason=", "reason=tampered-", 1))
        _reseal_standalone_artifact(fixture, artifact)
        payload = json.loads(fixture["replacement"].read_text())
        new_digest = payload["artifacts"][artifact]["sha256"]
        profile = dataclasses.replace(fixture["profile"], postbuild_record_sha256=new_digest)
        payload["cuda_rebuild"]["postbuild_record_sha256"] = new_digest
        message = "CUDA postbuild reason changed"
    monkeypatch.setattr(MODULE, "STANDALONE_REPLACEMENT_PROFILES", {profile.name: profile})
    fixture["replacement"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"convergence_has_converged": np.bool_(False)}, "did not converge naturally"),
        ({"n_iterations": np.int64(3)}, "iteration cap changed"),
        ({"final_all_data_ran": np.bool_(False)}, "lacks final all-data"),
        ({"relion_projector_replay_slot": np.int64(2)}, "replay field relion_projector_replay_slot"),
        ({"half2_indices": np.asarray([0, 1, 3])}, "not a disjoint full partition"),
    ),
)
def test_standalone_replacement_rejects_refinement_semantic_tamper(
    overrides: dict,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    result_path = fixture["run_root"] / "outputs/recovar_full/refinement_results.npz"
    _standalone_npz(result_path, fixture["profile"], fixture["data_dir"], **overrides)
    _reseal_standalone_outputs(fixture)

    with pytest.raises(ValueError, match=message):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


def test_standalone_replacement_rejects_unsealed_or_missing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    merged = fixture["run_root"] / "outputs/recovar_full/final_merged.mrc"
    merged.write_text("changed after replacement record creation\n")

    with pytest.raises(ValueError, match="declared file record does not match artifact"):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


@pytest.mark.parametrize(
    ("artifact", "mutate", "message"),
    (
        ("hbm_summary", lambda text: text.replace("42", "41"), "HBM summary changed"),
        ("scontrol_runtime", lambda text: text.replace("OverSubscribe=OK", "OverSubscribe=EXCLUSIVE"), "OverSubscribe mismatch"),
    ),
)
def test_standalone_replacement_rejects_resource_telemetry_tamper(
    artifact: str,
    mutate,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    payload = json.loads(fixture["replacement"].read_text())
    path = Path(payload["artifacts"][artifact]["path"])
    path.write_text(mutate(path.read_text()))
    _reseal_standalone_artifact(fixture, artifact)

    with pytest.raises(ValueError, match=message):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


def test_standalone_replacement_rejects_walltime_arithmetic_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    walltime = fixture["run_root"] / "outputs/recovar_full/slurm_walltime.json"
    payload = json.loads(walltime.read_text())
    payload["wall_s"] -= 1
    walltime.write_text(json.dumps(payload) + "\n")
    _reseal_standalone_outputs(fixture)

    with pytest.raises(ValueError, match="walltime arithmetic changed"):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


def test_standalone_replacement_rejects_content_changed_at_alternate_input_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    star = fixture["data_dir"] / "particles.star"
    star.write_text("different input bytes\n")
    payload = json.loads(fixture["replacement"].read_text())
    payload["inputs"]["particles_star"] = {
        "path": str(star),
        "size_bytes": star.stat().st_size,
        "sha256": MODULE.sha256_file(star),
    }
    fixture["replacement"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="input SHA-256 changed"):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


def test_standalone_replacement_rejects_resealed_script_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    payload = json.loads(fixture["replacement"].read_text())
    script = Path(payload["script"]["path"])
    script.write_text(script.read_text() + "#SBATCH --exclusive\n")
    payload["script"] = MODULE._file_record(script)
    fixture["replacement"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Slurm script SHA-256 changed"):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


def test_standalone_replacement_rejects_non_detached_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _standalone_fixture(tmp_path, monkeypatch)
    base_git_output = MODULE._git_output

    def attached_git_output(repo: Path, *arguments: str) -> str:
        if repo == fixture["profile"].source_repo and arguments == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "dev"
        return base_git_output(repo, *arguments)

    monkeypatch.setattr(MODULE, "_git_output", attached_git_output)
    with pytest.raises(ValueError, match="not detached"):
        MODULE.audit_native_launch(
            fixture["manifest"],
            replacement_records=(fixture["replacement"],),
            runner=fixture["runner"],
        )


def test_native_audit_classifies_oom_and_dependencies_without_promoting_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, _, _, runner = _fixture(tmp_path, monkeypatch)

    _, audit = MODULE.audit_native_launch(manifest, runner=runner)

    assert audit["jobs"]["recovar_smoke"]["classification"] == "oom"
    assert audit["jobs"]["relion_smoke"]["classification"] == "completed"
    assert audit["jobs"]["recovar_full"]["classification"] == "dependency_never_satisfied"
    assert audit["jobs"]["relion_full"]["classification"] == "dependency_never_satisfied"
    assert audit["jobs"]["relion_smoke"]["science_role"] == "capability_only"
    assert audit["jobs"]["relion_smoke"]["smoke_can_promote"] is False
    assert audit["science_scoring"]["eligible"] is False
    assert audit["science_scoring"]["smoke_only_promotion_forbidden"] is True
    with pytest.raises(ValueError, match="not science-ready"):
        MODULE.audit_native_launch(manifest, runner=runner, require_science_ready=True)


def test_native_audit_accepts_only_completed_full_pair_for_science(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, _, _, runner = _fixture(tmp_path, monkeypatch, completed_full=True)

    normalized, audit = MODULE.audit_native_launch(
        manifest,
        runner=runner,
        require_science_ready=True,
    )

    assert audit["science_scoring"]["eligible"] is True
    assert normalized["subject"] == audit["jobs"]["recovar_full"]["subject"]
    assert audit["jobs"]["recovar_full"]["expected_outputs"]
    assert audit["jobs"]["relion_full"]["expected_outputs"]


@pytest.mark.parametrize("mutation", ("command", "tres", "identity"))
def test_native_audit_rejects_mutated_execution_provenance(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, job_ids, _, runner = _fixture(tmp_path, monkeypatch)
    root = manifest.parent
    job_id = job_ids["relion_smoke"]
    if mutation == "command":
        path = root / f"provenance/command_{job_id}.sh"
        path.write_text(path.read_text().rstrip() + " --local\n")
        message = "executed command mismatch"
    elif mutation == "tres":
        path = root / f"provenance/scontrol_{job_id}.txt"
        path.write_text(path.read_text().replace("AllocTRES=cpu=12", "AllocTRES=cpu=8"))
        message = "ReqTRES != AllocTRES"
    else:
        path = root / f"provenance/nvidia_smi_{job_id}.txt"
        path.write_text(path.read_text().replace("NVIDIA H100", "NVIDIA A100"))
        message = "non-H100 GPU"

    with pytest.raises(ValueError, match=message):
        MODULE.audit_native_launch(manifest, runner=runner)


def test_native_binding_rejects_subject_config_hash_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, _, _, runner = _fixture(tmp_path, monkeypatch)
    subject = manifest.parent / "config/subject.env"
    subject.write_text(subject.read_text().replace("SUBJECT_TREE=", "EXTRA=bad\nSUBJECT_TREE="))

    with pytest.raises(ValueError, match="unexpected key set"):
        MODULE.audit_native_launch(manifest, runner=runner)


def test_explicit_smoke_replacement_is_bound_as_oom_capability_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, job_ids, _, base_runner = _fixture(tmp_path, monkeypatch)
    parent_root = manifest.parent
    replacement_root = tmp_path / "replacement"
    for path in (
        replacement_root / "scripts",
        replacement_root / "logs",
        replacement_root / "provenance",
        replacement_root / "outputs/recovar_smoke",
    ):
        path.mkdir(parents=True)
    _write(replacement_root / "SAFE_TO_DELETE", "")
    subject_values = MODULE._parse_key_values(parent_root / "config/subject.env", MODULE.SUBJECT_FIELDS)
    subject_repo = Path(subject_values["SUBJECT_REPO"])
    replacement_id = "999"
    script = _write(
        replacement_root / "scripts/recovar_smoke.sbatch",
        "#!/bin/bash\n#SBATCH --job-name=10202-s6-recovar-replacement\n",
    )
    launcher = MODULE._load_launcher()
    resources = dataclasses.asdict(launcher.build_run_specs(subject_repo, parent_root)[0].resources)
    _write(
        replacement_root / f"provenance/scontrol_{replacement_id}.txt",
        _scontrol(
            job_id=replacement_id,
            job_name="10202-s6-recovar-replacement",
            script=script,
            root=replacement_root,
            key="recovar_smoke",
            resources=resources,
            allocated=True,
        ),
    )
    command = launcher._recovar_command(
        subject_repo,
        parent_root / "inputs/smoke",
        replacement_root / "outputs/recovar_smoke",
        smoke=True,
    )
    _write(replacement_root / f"provenance/command_{replacement_id}.sh", shlex.join(command) + "\n")
    _write_subject_file(
        replacement_root / f"provenance/subject_{replacement_id}.txt",
        subject_values,
        lowercase=True,
    )
    _write(
        replacement_root / f"provenance/environment_{replacement_id}.txt",
        "PYTHONNOUSERSITE=1\n"
        "XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
        f"RECOVAR_CUDA_LIB={subject_values['CUDA_LIB']}\n"
        f"RECOVAR_EXPECTED_REPO_ROOT={subject_values['SUBJECT_REPO']}\n",
    )
    _write(
        replacement_root / f"provenance/final_env_{replacement_id}.txt",
        "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset\nRECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset\n",
    )
    _write(replacement_root / f"provenance/nvidia_smi_{replacement_id}.txt", _gpu_inventory(1))
    _write(replacement_root / f"provenance/ldd_relion_bind_{replacement_id}.txt", "all resolved\n")
    _write(replacement_root / f"logs/recovar_smoke-{replacement_id}.out", "started\n")
    _write(replacement_root / f"logs/recovar_smoke-{replacement_id}.err", "CUDA: out of memory\n")
    replacement_payload = {
        "schema": MODULE.REPLACEMENT_SCHEMA,
        "parent_launch_manifest": {
            "path": str(manifest),
            "sha256": MODULE.sha256_file(manifest),
        },
        "run_key": "recovar_smoke",
        "replaces_job_id": job_ids["recovar_smoke"],
        "job_id": replacement_id,
        "run_root": str(replacement_root),
        "script_sha256": MODULE.sha256_file(script),
        "subject": subject_values,
        "reason": "capability-only retry",
    }
    replacement = _write(tmp_path / "replacement.json", json.dumps(replacement_payload))

    def runner(command, **kwargs):
        job_id = command[command.index("--jobs") + 1]
        if job_id != replacement_id:
            return base_runner(command, **kwargs)
        tres = "cpu=4,mem=500G,node=1,billing=40,gres/gpu=1"
        return SimpleNamespace(
            stdout=f"{replacement_id}|FAILED|1:0|{tres}|{tres}|1024K\n",
            returncode=0,
        )

    _, audit = MODULE.audit_native_launch(
        manifest,
        replacement_records=(replacement,),
        runner=runner,
    )

    selected = audit["jobs"]["recovar_smoke"]
    assert selected["job_id"] == replacement_id
    assert selected["source"] == "replacement"
    assert selected["classification"] == "oom"
    assert selected["science_role"] == "capability_only"
    assert selected["provenance_complete"] is False
    assert audit["science_scoring"]["eligible"] is False


def test_finalizer_dispatches_native_adapter_and_binds_replacement_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "native"
    root.mkdir()
    manifest = _write(
        root / "launch_manifest.json",
        json.dumps({"schema": FINALIZER.NATIVE_LAUNCH_SCHEMA}),
    )
    replacement = _write(tmp_path / "replacement.json", "{}\n")
    subject = {"commit": "a" * 40}
    audit = {
        "launch_manifest": {
            "path": str(manifest),
            "sha256": FINALIZER.sha256_file(manifest),
        },
        "jobs": {},
        "selected_standalone_replacement_profiles": {
            "recovar_full": {
                "profile": "fixture-profile",
                "job_id": "999",
                "cuda_preflight_sha256": "b" * 64,
                "cuda_postbuild_sha256": "c" * 64,
            }
        },
    }

    class Adapter:
        @staticmethod
        def audit_native_launch(path, *, replacement_records, runner, require_science_ready):
            assert path == manifest
            assert replacement_records == (replacement,)
            assert require_science_ready is True
            return {"run_root": str(root), "subject": subject}, audit

    monkeypatch.setattr(FINALIZER, "_load_native_adapter", lambda: Adapter)
    monkeypatch.setattr(
        FINALIZER,
        "_run_analysis",
        lambda *_args, **_kwargs: (
            {"schema": "collector"},
            {"commands": {}, "analysis_artifacts": {}},
        ),
    )
    monkeypatch.setattr(FINALIZER, "_input_contract", lambda _payload: {"k": 1})

    output = FINALIZER.finalize(manifest, replacement_records=(replacement,))
    evidence = json.loads(output.read_text())

    binding = evidence["execution_binding"]
    assert binding["launch_manifest_schema"] == FINALIZER.NATIVE_LAUNCH_SCHEMA
    assert binding["replacement_records"][0]["sha256"] == FINALIZER.sha256_file(replacement)
    assert binding["finalizer_argv"][-2:] == ["--replacement-record", str(replacement)]
    assert binding["selected_standalone_replacement_profiles"]["recovar_full"] == {
        "profile": "fixture-profile",
        "job_id": "999",
        "cuda_preflight_sha256": "b" * 64,
        "cuda_postbuild_sha256": "c" * 64,
    }


def _native_binding_fixture(tmp_path: Path) -> dict:
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest = _write(
        run_root / "launch_manifest.json",
        json.dumps(
            {
                "schema": SUMMARIZER.NATIVE_LAUNCH_MANIFEST_SCHEMA,
                "run_root": str(run_root),
            }
        ),
    )
    subject = {"commit": "a" * 40}
    jobs = {
        key: {
            "phase": "smoke" if key.endswith("smoke") else "full",
            "science_role": "capability_only" if key.endswith("smoke") else "science_candidate",
            "smoke_can_promote": False,
            "classification": "completed",
            "provenance_complete": True,
            "expected_outputs": [{"sha256": "b" * 64}],
        }
        for key in MODULE.RUN_KEYS
    }
    launch = {
        "schema": SUMMARIZER.NATIVE_EXECUTION_AUDIT_SCHEMA,
        "launch_manifest": {
            "path": str(manifest),
            "sha256": SUMMARIZER.sha256_file(manifest),
        },
        "replacement_records": [],
        "science_subject": subject,
        "science_scoring": {
            "eligible": True,
            "smoke_only_promotion_forbidden": True,
        },
        "jobs": jobs,
    }
    collector = {"schema": "collector"}
    commands: dict = {}
    artifacts: dict = {}
    envelope = _write(
        run_root / "execution_envelope.json",
        json.dumps(
            {
                "schema": SUMMARIZER.EXECUTION_BINDING_SCHEMA,
                "launch": launch,
                "collector": collector,
                "analysis_commands": commands,
                "analysis_artifacts": artifacts,
            }
        ),
    )
    finalizer = (Path(__file__).parents[2] / SUMMARIZER.FINALIZER_RELATIVE_PATH).resolve()
    argv = [sys.executable, str(finalizer), "--launch-manifest", str(manifest)]
    return {
        "subject": subject,
        "launch": launch,
        "collector": collector,
        "analysis_commands": commands,
        "analysis_artifacts": artifacts,
        "execution_binding": {
            "schema": SUMMARIZER.EXECUTION_BINDING_SCHEMA,
            "launch_manifest_path": str(manifest),
            "launch_manifest_sha256": SUMMARIZER.sha256_file(manifest),
            "launch_manifest_schema": SUMMARIZER.NATIVE_LAUNCH_MANIFEST_SCHEMA,
            "replacement_records": [],
            "finalizer_command_path": str(finalizer),
            "finalizer_command_sha256": SUMMARIZER.sha256_file(finalizer),
            "finalizer_argv": argv,
            "finalizer_argv_sha256": SUMMARIZER.sha256_json(argv),
            "evidence_envelope_path": str(envelope),
            "evidence_envelope_sha256": SUMMARIZER.sha256_file(envelope),
        },
    }


def test_scorecard_binding_accepts_native_full_pair_and_rejects_smoke_promotion(
    tmp_path: Path,
) -> None:
    evidence = _native_binding_fixture(tmp_path)

    assert SUMMARIZER._validate_execution_binding(evidence, required=True) == []

    evidence["launch"]["jobs"]["recovar_smoke"]["science_role"] = "science_candidate"
    failures = SUMMARIZER._validate_execution_binding(evidence, required=True)
    assert "execution_binding:native_recovar_smoke_capability_only" in failures


def test_native_subject_change_requires_deliberate_scorecard_contract_update(
    tmp_path: Path,
) -> None:
    scorecard = SUMMARIZER.load_and_validate_scorecard()
    target = next(case for case in scorecard["cases"] if case["id"] == SUMMARIZER.TARGET_CASE_ID)
    evidence = _native_binding_fixture(tmp_path)
    evidence.update(
        {
            "schema": SUMMARIZER.EVIDENCE_SCHEMA,
            "case_id": SUMMARIZER.TARGET_CASE_ID,
            "subject": {
                "commit": "a" * 40,
                "tree_clean": True,
                "diff_sha256": SUMMARIZER.EMPTY_SHA256,
            },
        }
    )

    failures = SUMMARIZER.validate_case_evidence_provenance(scorecard, target, evidence)

    assert "subject_commit" in failures
    assert "native_subject_requires_deliberate_scorecard_contract_update" in failures
