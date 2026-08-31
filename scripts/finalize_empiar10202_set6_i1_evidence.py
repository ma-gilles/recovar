#!/usr/bin/env python3
"""Finalize one matched EMPIAR-10202 set-6 run into PR158 evidence.

This command does not submit or modify Slurm jobs.  The original launch schema
still requires all four dependency-gated jobs to complete.  The native harness
adapter permits explicit retry records, but requires both full refinements to
complete with sealed provenance; smoke jobs remain capability evidence and can
never enter the science denominator.  Both paths require exact requested and
allocated TRES, declared outputs, commands, and hashes before invoking the
pinned FSC collectors and writing the PR158 scorecard envelope.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

LAUNCH_SCHEMA = "recovar.empiar10202_set6_i1_matched_launch.v1"
EVIDENCE_SCHEMA = "recovar.em_k1_realdata_science_equivalence_case_evidence.v1"
COLLECTOR_SCHEMA = "recovar-relion-realdata-metrics-v2"
SCIENCE_SCHEMA = "recovar.em_k1_science_diagnostics.v1"
EXECUTION_BINDING_SCHEMA = "recovar.em_k1_execution_binding.v1"
CASE_ID = "empiar-10202-set06-k1-I1"
RUN_KEYS = ("recovar_smoke", "relion_smoke", "recovar_full", "relion_full")
NATIVE_LAUNCH_SCHEMA = "recovar.em.matched_launch_harness.v1"
NATIVE_ADAPTER = Path(__file__).resolve().with_name("audit_empiar10202_set6_i1_native_harness.py")
ALLOWED_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/codex")
LAUNCHER = Path(__file__).resolve().with_name("launch_empiar10202_set6_i1_matched.py")
SUBJECT_REPO = Path("/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_box800_subject_2249bf352_20260830")
SUBJECT_COMMIT = "2249bf352377aae85ef9f3d378eda86dadef671f"
SUBJECT_TREE = "4573e7561741eb4120e585730fd73663da7c382c"
SUBJECT_DRIVER_SHA256 = "40affc02cb772fdee9d73777395d2321faa8d8677515a9fde6defc53b6795293"
SUBJECT_CUDA_SOURCE_SHA256 = "4238bc4eb344c0e4bb12989a0b41fdcf62c455164ab207e3add44612f377ee9a"
PREPARED_STAR_SHA256 = "d66afb3001e6e43463fb699804fe8b50f8cef1f2ccd9730f5275955b6be7b512"
PREPARATION_MANIFEST = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_20260830/preparation_manifest.json"
)
PREPARATION_MANIFEST_SHA256 = "05865373813f50b83060ef4c93993522472b5ca3189c4acd27647c18262e501d"
PREPARED_STAR = PREPARATION_MANIFEST.parent / "prepared/particles_set06_optics.star"
PARTICLE_STACK = Path("/projects/CRYOEM/singerlab/mg6942/10202/06_Final_Stack/2017-12-27_MagCorrect_Frames05-19.mrcs")
PARTICLE_STACK_SHA256 = "8eecf0fbf8e645ac51feff278a86e43e7e4be117921333dc6d3e22e52a628453"
PARTICLE_STACK_SIZE_BYTES = 78_118_401_024
RECOVAR_REFERENCE = PREPARATION_MANIFEST.parent / "prepared/initial_reference_recovar_I1_30A_box800.mrc"
RECOVAR_REFERENCE_SHA256 = "d77516a08e5e3ccdef07d9039e36d65b174afe5d56fe20d7775eef188d2e6cc6"
RELION_REFERENCE = PREPARATION_MANIFEST.parent / "prepared/initial_reference_relion_I1_30A_box800.mrc"
RELION_REFERENCE_SHA256 = "4f83710c999276d4f65cff586266ee121f271e8bed382dba329b384860af96bb"
CANONICAL_REFERENCE_SHA256 = "b617f90d55ef4b7a637bd495397f2c6f291371263f66b16a7fbc020bf6dbad09"
RELION_BIND_LIBRARY = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/"
    "pr158_symmetry_build_20260830/relion_bind/"
    "_relion_bind_core.cpython-311-x86_64-linux-gnu.so"
)
RELION_BIND_LIBRARY_SHA256 = "82b0a8cf2c189463f9cf0181099f4e92ce4365cce3819463c27af44b3c1014a2"
RELION_BIND_SOURCE_TREE = "1633d228e89d91ede8ad0996e727ec6ab1bc96ee"
RELION_MPI = Path("/projects/MOLBIO/local/relion-5.0.1-gcc-11.5.0-cuda-12.6-rhel9-arch80/bin/relion_refine_mpi")
RELION_MPI_SHA256 = "92cf3ba54038d5e162e238b952fe88f1414f440d4e6cba23bc4b097428087b4a"
RELION_SOURCE = Path("/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_d476e6f_clean_binding_20260713")
RELION_SOURCE_COMMIT = "d476e6f6a4f1f37627c06ace5227fc374c0c2b05"
CUDA_LIBRARY = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "pr158_box800_texture_memory_20260830/lib/"
    "libcuda_backproject-2249bf352-r3-sealed-sm90.so"
)
CUDA_LIBRARY_SHA256 = "47a8a5c7878e7ea1f242942918a40d57da6b78ebd791339bd3ac70ec1ae395ac"
RELION_CUDA_LIB_DIR = Path("/usr/local/cuda-12/targets/x86_64-linux/lib")
RELION_MPI_LIB_DIR = Path("/usr/local/openmpi/cuda-12.6/4.1.6/gcc/lib64")
SYMMETRY_CONTRACT = {
    "family": "icosahedral",
    "requested_label": "I1",
    "relion_label": "I1",
    "recovar_label": "I1",
    "operator_count": 60,
    "operator_order": "identity at index 0, then RELION SymList::get_matrices(isym) for isym=0..58",
    "left_operator_policy": "all left operators are identity and are not applied in BPref reconstruction",
    "hash_encoding": "float64 rounded to 12 decimals, stacked [left,right], little-endian C-order bytes",
    "operators_sha256": "093a0876b93610ec141c87840ae3ff4dc4491b27dec87143558358ef556557b8",
}
SMOKE_STAR_SHA256 = "500dc2b76fdd5554d1dba759168184109f16d91deed0fce1fc2d9f5daf164d7d"
SMOKE_HALF_ASSIGNMENT_SHA256 = "3a8f3eb11efce69ae772bed68639f05ac73b3e7162cc2316be987a28806227fc"
SMOKE_SOURCE_IMAGE_NAMES_SHA256 = "44d98ac85bc83243cd7cd38180fa480ab979a49aefc25c6f4e928ce11d5f1240"
SMOKE_SOURCE_ROWS = tuple(range(62)) + (63, 66)
FINAL_ENVIRONMENT_TEXT = "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset\nRECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset\n"
SLURM_TEMPLATE_SHA256 = {
    "recovar_smoke": "72dad390b3714eca443ee433629cacae5fd1dbe3c28055d02b910ffbd91f534f",
    "relion_smoke": "5934121c2bcddc29b3c776edd9138d862aa72c19fdb69bb1a73086f8ee1fe22e",
    "recovar_full": "61b9e97d0c500b542298743ab7cb08e7eb9f916aa10c7badf52f5b2c1bbfc08c",
    "relion_full": "31695769b117d62c72b8aa0c4a741a62d832b88f53d0c7b0951d30d2f4d49ed9",
}
EXPECTED_SCIENTIFIC_CONTRACT = {
    "dataset": "EMPIAR-10202",
    "image_set": 6,
    "k": 1,
    "autonomous": True,
    "fixed_poses": False,
    "local_only": False,
    "replay": False,
    "same_particle_star_per_phase": True,
    "same_deposited_half_assignments": True,
    "same_initial_reference_canonical_array": True,
    "final_all_data_grid_correct_environment": "unset",
    "final_all_data_after_max_iter_environment": "unset",
    "quality_metric": "signed canonical-frame shellwise FSC/FSC-AUC",
}
RAW_COLLECTOR = Path("/home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/scripts/collect_metrics.py")
RAW_COLLECTOR_SHA256 = "63d1a8f9f0a772ef0db45d7417902f36cbde884b309fa6e94d69bdfe2edf2d88"
DIAGNOSTIC_PRODUCER = Path(__file__).resolve().with_name("collect_em_k1_science_diagnostics.py")
DIAGNOSTIC_PRODUCER_SHA256 = "01b432c5c735a060f69d55569c897ac42aca8c2af6c3fa3a23b334fb2378ac9c"
SYMMETRY_OPERATORS_SHA256 = "093a0876b93610ec141c87840ae3ff4dc4491b27dec87143558358ef556557b8"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ALIGNED_FIELDS = {
    "final_cross_engine_proper_aligned",
    "final_cross_engine_half1_proper_aligned",
    "final_cross_engine_half2_proper_aligned",
}
MASKED_FIELDS = {
    "relion_final_half_fsc_common_masked",
    "recovar_final_half_fsc_common_masked",
    "final_cross_engine_proper_aligned_common_masked",
    "final_cross_engine_half1_proper_aligned_common_masked",
    "final_cross_engine_half2_proper_aligned_common_masked",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def _git_output(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_native_adapter() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_recovar_empiar10202_native_evidence_adapter",
        NATIVE_ADAPTER,
    )
    _require(spec is not None and spec.loader is not None, "cannot load native evidence adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _file_record(path: Path, *, nonempty: bool = True) -> dict[str, Any]:
    _require(path.is_absolute(), f"artifact path is not absolute: {path}")
    _require(not path.is_symlink(), f"artifact must not be a symlink: {path}")
    _require(path.is_file(), f"missing regular artifact: {path}")
    size = path.stat().st_size
    _require(not nonempty or size > 0, f"empty artifact: {path}")
    return {"path": str(path), "size_bytes": size, "sha256": sha256_file(path)}


@dataclasses.dataclass(frozen=True)
class ExpectedResources:
    nodes: int
    ntasks: int
    cpus_per_task: int
    memory: str
    h100_gpus: int
    walltime: str


@dataclasses.dataclass(frozen=True)
class ExpectedRunSpec:
    key: str
    engine: str
    phase: str
    resources: ExpectedResources
    data_dir: Path
    output_dir: Path
    command: tuple[str, ...]
    expected_outputs: tuple[Path, ...]
    depends_on: tuple[str, ...]


def _recovar_command(root: Path, key: str, *, smoke: bool) -> tuple[str, ...]:
    phase = "smoke" if smoke else "full"
    data_dir = root / "inputs" / phase
    output_dir = root / "outputs" / key
    command = [
        "srun",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--cpu-bind=none",
        str(SUBJECT_REPO / ".pixi/envs/default/bin/python"),
        "-m",
        "scripts.run_full_refinement",
        "--data_dir",
        str(data_dir),
        "--output",
        str(output_dir),
        "--init_volume",
        str(data_dir / "reference_init.mrc"),
        "--relion_half_sets",
        str(data_dir / "particles.star"),
    ]
    if smoke:
        command.extend(["--max_iter", "1", "--skip_final_iteration", "--relion_current_sizes", "800"])
    else:
        command.extend(["--max_iter", "50"])
    command.extend(
        [
            "--n_classes",
            "1",
            "--sym",
            "I1",
            "--initial-pose-source",
            "input-star",
            "--healpix_order",
            "3",
            "--max_healpix_order",
            "7",
            "--auto_local_healpix_order",
            "4",
            "--offset_range",
            "5",
            "--offset_step",
            "2",
            "--offset_sigma_angstrom",
            "10",
            "--adaptive_oversampling",
            "1",
            "--max_significants",
            "-1",
            "--tau2_fudge",
            "1",
            "--perturb_factor",
            ".5",
            "--seed",
            "10202",
            "--perturb_seed",
            "10202",
            "--init_resolution",
            "30",
            "--firstiter_cc",
            "--apply-initial-lowpass",
            "--particle_diameter_ang",
            "300",
            "--width_mask_edge_px",
            "5",
            "--image-fourier-backend",
            "relion_cuda",
            "--image_batch_size",
            "64",
            "--rotation_block_size",
            "8192",
        ]
    )
    return tuple(command)


def _relion_command(root: Path, key: str, *, smoke: bool) -> tuple[str, ...]:
    phase = "smoke" if smoke else "full"
    data_dir = root / "inputs" / phase
    output_dir = root / "outputs" / key
    command = [
        "srun",
        "--ntasks=3",
        "--ntasks-per-node=3",
        "--cpus-per-task=4",
        "--cpu-bind=none",
        str(RELION_MPI),
        "--i",
        str(data_dir / "particles.star"),
        "--ref",
        str(data_dir / "reference_init_relion.mrc"),
        "--o",
        str(output_dir / "run"),
        "--auto_refine",
        "--split_random_halves",
        "--K",
        "1",
        "--sym",
        "I1",
        "--particle_diameter",
        "300",
        "--maskedge",
        "5",
        "--ini_high",
        "30",
        "--firstiter_cc",
        "--healpix_order",
        "3",
        "--auto_local_healpix_order",
        "4",
        "--offset_range",
        "5",
        "--offset_step",
        "2",
        "--offset",
        "10",
        "--oversampling",
        "1",
        "--perturb",
        ".5",
        "--ctf",
        "--norm",
        "--scale",
        "--zero_mask",
        "--flatten_solvent",
        "--low_resol_join_halves",
        "40",
        "--tau2_fudge",
        "1",
        "--maxsig",
        "-1",
        "--pad",
        "2",
        "--random_seed",
        "10202",
    ]
    if smoke:
        command.extend(["--auto_iter_max", "1", "--incr_size", "800"])
    else:
        command.extend(["--auto_iter_max", "50"])
    command.extend(
        [
            "--pool",
            "3",
            "--j",
            "4",
            "--gpu",
            "0:1",
            "--free_gpu_memory",
            "2000",
            "--preread_images",
            "--no_parallel_disc_io",
            "--dont_combine_weights_via_disc",
        ]
    )
    return tuple(command)


def _expected_run_specs(root: Path) -> tuple[ExpectedRunSpec, ...]:
    smoke_data = root / "inputs/smoke"
    full_data = root / "inputs/full"
    recovar_smoke = root / "outputs/recovar_smoke"
    relion_smoke = root / "outputs/relion_smoke"
    recovar_full = root / "outputs/recovar_full"
    relion_full = root / "outputs/relion_full"
    return (
        ExpectedRunSpec(
            "recovar_smoke",
            "recovar",
            "smoke",
            ExpectedResources(1, 1, 4, "500G", 1, "12:00:00"),
            smoke_data,
            recovar_smoke,
            _recovar_command(root, "recovar_smoke", smoke=True),
            (recovar_smoke / "refinement_results.npz",),
            (),
        ),
        ExpectedRunSpec(
            "relion_smoke",
            "relion",
            "smoke",
            ExpectedResources(1, 3, 4, "500G", 2, "12:00:00"),
            smoke_data,
            relion_smoke,
            _relion_command(root, "relion_smoke", smoke=True),
            (
                relion_smoke / "run_it001_data.star",
                relion_smoke / "run_it001_half1_class001.mrc",
                relion_smoke / "run_it001_half2_class001.mrc",
            ),
            (),
        ),
        ExpectedRunSpec(
            "recovar_full",
            "recovar",
            "full",
            ExpectedResources(1, 1, 4, "500G", 1, "5-00:00:00"),
            full_data,
            recovar_full,
            _recovar_command(root, "recovar_full", smoke=False),
            (
                recovar_full / "refinement_results.npz",
                recovar_full / "final_merged.mrc",
                recovar_full / "final_half1_unfil.mrc",
                recovar_full / "final_half2_unfil.mrc",
            ),
            ("recovar_smoke", "relion_smoke"),
        ),
        ExpectedRunSpec(
            "relion_full",
            "relion",
            "full",
            ExpectedResources(1, 3, 4, "500G", 2, "5-00:00:00"),
            full_data,
            relion_full,
            _relion_command(root, "relion_full", smoke=False),
            (
                relion_full / "run_class001.mrc",
                relion_full / "run_half1_class001_unfil.mrc",
                relion_full / "run_half2_class001_unfil.mrc",
                relion_full / "run_data.star",
                relion_full / "run_model.star",
            ),
            ("recovar_smoke", "relion_smoke"),
        ),
    )


def _normalized_slurm_sha256(text: str, root: Path, runtime_root: Path) -> str:
    normalized = text.replace(str(runtime_root), "{RUNTIME_ROOT}").replace(str(root), "{RUN_ROOT}")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _require_real_directory(path: Path, label: str) -> None:
    _require(path.is_absolute(), f"{label} is not absolute: {path}")
    _require(not path.is_symlink() and path.is_dir(), f"{label} is not a real directory: {path}")


def _require_exact_path(value: Any, expected: Path, label: str) -> None:
    _require(expected.is_absolute(), f"internal expected {label} is not absolute: {expected}")
    _require(isinstance(value, str) and value == str(expected), f"noncanonical {label}: {value!r}")


def _require_utc_timestamp(value: Any, label: str) -> None:
    _require(isinstance(value, str), f"{label} is not a string")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    _require(parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0), f"{label} is not UTC")


def _field(text: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S+)", text)
    _require(match is not None, f"scontrol record lacks {name}")
    return match.group(1)


def _normalized_dependency(value: str) -> str:
    normalized = re.sub(r"\((?:unfulfilled|satisfied)\)", "", value)
    # Slurm expands ``afterok:A:B`` to ``afterok:A,afterok:B`` in scontrol.
    return normalized.replace(",afterok:", ":")


def _expected_dependency(key: str, job_ids: Mapping[str, str]) -> str:
    if key.endswith("_smoke"):
        return "(null)"
    return f"afterok:{job_ids['recovar_smoke']}:{job_ids['relion_smoke']}"


def _validate_scontrol_identity(
    text: str,
    *,
    key: str,
    job_id: str,
    resources: Mapping[str, Any],
    script: Path,
    stdout: Path,
    stderr: Path,
    dependency: str | None,
    submit_time: bool,
) -> None:
    expected_cpu = int(resources["ntasks"]) * int(resources["cpus_per_task"])
    exact_fields = {
        "JobId": job_id,
        "JobName": f"10202-s6-{key.replace('_', '-')}",
        "Command": str(script),
        "StdOut": str(stdout),
        "StdErr": str(stderr),
        "NumCPUs": str(expected_cpu),
        "NumTasks": str(resources["ntasks"]),
        "CPUs/Task": str(resources["cpus_per_task"]),
        "Features": "h100",
        "TresPerNode": f"gres/gpu:h100:{resources['h100_gpus']}",
    }
    for name, expected in exact_fields.items():
        _require(_field(text, name) == expected, f"scontrol {name} mismatch for {key}")
    expected_nodes = str(resources["nodes"])
    observed_nodes = _field(text, "NumNodes")
    allowed_nodes = {expected_nodes, f"{expected_nodes}-{expected_nodes}"} if submit_time else {expected_nodes}
    _require(observed_nodes in allowed_nodes, f"scontrol NumNodes mismatch for {key}")
    if dependency is not None:
        observed_dependency = _normalized_dependency(_field(text, "Dependency"))
        _require(observed_dependency == dependency, f"scontrol Dependency mismatch for {key}")


def _validate_requested_tres(text: str, resources: Mapping[str, Any], key: str) -> str:
    requested = _field(text, "ReqTRES")
    expected_cpu = int(resources["ntasks"]) * int(resources["cpus_per_task"])
    required = {
        f"cpu={expected_cpu}",
        f"mem={resources['memory']}",
        f"node={resources['nodes']}",
        f"gres/gpu={resources['h100_gpus']}",
    }
    _require(required.issubset(set(requested.split(","))), f"wrong requested TRES for {key}: {requested}")
    return requested


def _validate_tres(text: str, job_id: str, resources: Mapping[str, Any]) -> dict[str, str]:
    _require(_field(text, "JobId") == job_id, f"scontrol JobId mismatch for {job_id}")
    requested = _field(text, "ReqTRES")
    allocated = _field(text, "AllocTRES")
    _require(requested == allocated, f"ReqTRES != AllocTRES for job {job_id}")
    expected_cpu = int(resources["ntasks"]) * int(resources["cpus_per_task"])
    required = {
        f"cpu={expected_cpu}",
        f"mem={resources['memory']}",
        f"node={resources['nodes']}",
        f"gres/gpu={resources['h100_gpus']}",
    }
    observed = set(requested.split(","))
    _require(required.issubset(observed), f"wrong TRES for job {job_id}: {requested}")
    return {"requested": requested, "allocated": allocated}


def _validate_gpu_identity(path: Path, expected_count: int, key: str) -> None:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    _require(len(lines) == expected_count, f"wrong visible GPU count for {key}: {len(lines)} != {expected_count}")
    rows = [line.split(",", 1) for line in lines]
    _require(all(len(row) == 2 for row in rows), f"malformed GPU identity for {key}")
    uuids = [row[0].strip() for row in rows]
    names = [row[1].strip() for row in rows]
    _require(all(uuid.startswith("GPU-") and len(uuid) > 4 for uuid in uuids), f"malformed GPU UUID for {key}")
    _require(len(set(uuids)) == expected_count, f"duplicate GPU UUID for {key}")
    _require(all("H100" in name for name in names), f"non-H100 GPU identity for {key}")


def _validate_subject(payload: Mapping[str, Any]) -> None:
    subject = payload.get("subject", {})
    _require(subject.get("repo") == str(SUBJECT_REPO), "wrong subject repository")
    _require(subject.get("commit") == SUBJECT_COMMIT, "wrong subject commit")
    _require(subject.get("tree") == SUBJECT_TREE, "wrong subject tree")
    _require(subject.get("tree_clean") is True, "subject tree was not clean")
    _require(subject.get("diff_sha256") == EMPTY_SHA256, "subject diff was not empty")
    python = SUBJECT_REPO / ".pixi/envs/default/bin/python"
    driver = SUBJECT_REPO / "scripts/run_full_refinement.py"
    cuda_source = SUBJECT_REPO / "recovar/cuda/cuda_backproject.cu"
    _require_exact_path(subject.get("python"), python, "subject python")
    _require_exact_path(subject.get("driver"), driver, "subject driver")
    _require_exact_path(subject.get("cuda_source"), cuda_source, "subject CUDA source")
    _require(subject.get("driver_sha256") == SUBJECT_DRIVER_SHA256, "wrong subject driver digest")
    _require(subject.get("cuda_source_sha256") == SUBJECT_CUDA_SOURCE_SHA256, "wrong CUDA source digest")
    _require_real_directory(SUBJECT_REPO, "subject repository")
    _require(
        driver.is_file() and not driver.is_symlink() and sha256_file(driver) == SUBJECT_DRIVER_SHA256,
        "subject driver changed",
    )
    _require(
        cuda_source.is_file()
        and not cuda_source.is_symlink()
        and sha256_file(cuda_source) == SUBJECT_CUDA_SOURCE_SHA256,
        "CUDA source changed",
    )
    _require(python.is_file(), "subject Python is missing")
    live_commit = _git_output(SUBJECT_REPO, "rev-parse", "HEAD")
    live_tree = _git_output(SUBJECT_REPO, "rev-parse", "HEAD^{tree}")
    live_status = _git_output(SUBJECT_REPO, "status", "--porcelain=v1", "--untracked-files=all")
    _require(live_commit == SUBJECT_COMMIT, "subject repository commit changed")
    _require(live_tree == SUBJECT_TREE, "subject repository tree changed")
    _require(not live_status, "subject repository is dirty")


def _validate_scientific_contract(payload: Mapping[str, Any]) -> None:
    _require(payload.get("scientific_contract") == EXPECTED_SCIENTIFIC_CONTRACT, "scientific contract changed")
    _require(payload.get("symmetry") == SYMMETRY_CONTRACT, "symmetry contract changed")


def _completed_accounting(
    job_id: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    command = [
        "sacct",
        "-X",
        "--noheader",
        "--parsable2",
        "--jobs",
        job_id,
        "--format=JobIDRaw,State,ExitCode",
    ]
    result = runner(command, check=True, capture_output=True, text=True)
    rows = [line.split("|")[:3] for line in result.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) == 3 and row[0] == job_id]
    _require(len(matches) == 1, f"sacct did not return one top-level row for {job_id}")
    _, state, exit_code = matches[0]
    _require(state == "COMPLETED", f"job {job_id} state is {state}, not COMPLETED")
    _require(exit_code == "0:0", f"job {job_id} exit code is {exit_code}, not 0:0")
    return {"state": state, "exit_code": exit_code, "query": command}


def _canonical_paths(root: Path, key: str, job_id: str, engine: str, output_dir: Path) -> dict[str, Path]:
    ldd = "ldd_relion_bind" if engine == "recovar" else "ldd_relion"
    return {
        "stdout_log": root / "logs" / f"{key}-{job_id}.out",
        "stderr_log": root / "logs" / f"{key}-{job_id}.err",
        "scontrol_submit": root / "provenance" / f"scontrol_submit_{job_id}.txt",
        "scontrol_runtime": root / "provenance" / f"scontrol_{job_id}.txt",
        "executed_command": root / "provenance" / f"command_{job_id}.sh",
        "environment": root / "provenance" / f"environment_{job_id}.txt",
        "final_environment": root / "provenance" / f"final_env_{job_id}.txt",
        "gpu_inventory": root / "provenance" / f"nvidia_smi_{job_id}.txt",
        "dynamic_libraries": root / "provenance" / f"{ldd}_{job_id}.txt",
        "walltime": output_dir / "slurm_walltime.json",
        "gpu_identity": output_dir / "gpu_identity.txt",
        "completed_marker": output_dir / "COMPLETED",
    }


def _validate_preparation_and_inputs(
    payload: Mapping[str, Any],
    root: Path,
) -> None:
    expected_preparation = {
        "preparation_manifest": str(PREPARATION_MANIFEST),
        "preparation_manifest_sha256": PREPARATION_MANIFEST_SHA256,
        "particle_star": str(PREPARED_STAR),
        "particle_star_sha256": PREPARED_STAR_SHA256,
        "particle_stack": str(PARTICLE_STACK),
        "particle_stack_sha256": PARTICLE_STACK_SHA256,
        "particle_stack_size_bytes": PARTICLE_STACK_SIZE_BYTES,
        "recovar_reference": str(RECOVAR_REFERENCE),
        "recovar_reference_sha256": RECOVAR_REFERENCE_SHA256,
        "relion_reference": str(RELION_REFERENCE),
        "relion_reference_sha256": RELION_REFERENCE_SHA256,
        "canonical_reference_sha256": CANONICAL_REFERENCE_SHA256,
    }
    _require(payload.get("preparation") == expected_preparation, "preparation contract changed")
    _require(
        PREPARATION_MANIFEST.is_file()
        and not PREPARATION_MANIFEST.is_symlink()
        and sha256_file(PREPARATION_MANIFEST) == PREPARATION_MANIFEST_SHA256,
        "preparation manifest changed",
    )
    _require(PARTICLE_STACK.is_file() and not PARTICLE_STACK.is_symlink(), "particle stack is not a regular file")
    _require(PARTICLE_STACK.stat().st_size == PARTICLE_STACK_SIZE_BYTES, "particle stack size changed")
    _require(sha256_file(PARTICLE_STACK) == PARTICLE_STACK_SHA256, "particle stack digest changed")
    smoke_star = root / "inputs/smoke/particles.star"
    expected_smoke = {
        "path": str(smoke_star),
        "sha256": SMOKE_STAR_SHA256,
        "particle_count": 64,
        "half_counts": {"1": 32, "2": 32},
        "selection_policy": "first 32 source rows in each deposited half, restored to source-row order",
        "source_row_indices_zero_based": list(SMOKE_SOURCE_ROWS),
        "source_image_names_sha256": SMOKE_SOURCE_IMAGE_NAMES_SHA256,
        "half_assignment_sha256": SMOKE_HALF_ASSIGNMENT_SHA256,
    }
    _require(payload.get("smoke_subset") == expected_smoke, "smoke subset contract changed")
    _require(not smoke_star.is_symlink() and smoke_star.is_file(), "smoke STAR must be a regular file")
    _require(sha256_file(smoke_star) == SMOKE_STAR_SHA256, "smoke STAR digest changed")

    def require_input_link(link: Path, target: Path, digest: str, label: str) -> None:
        _require(target.is_file() and not target.is_symlink(), f"{label} target is not a regular file")
        _require(link.is_symlink(), f"{label} must be the launcher-created input symlink")
        _require(link.resolve(strict=True) == target.resolve(strict=True), f"{label} symlink target changed")
        _require(sha256_file(link) == digest, f"{label} digest changed")

    require_input_link(root / "inputs/full/particles.star", PREPARED_STAR, PREPARED_STAR_SHA256, "full STAR")
    for phase in ("smoke", "full"):
        require_input_link(
            root / f"inputs/{phase}/reference_init.mrc",
            RECOVAR_REFERENCE,
            RECOVAR_REFERENCE_SHA256,
            f"{phase} RECOVAR reference",
        )
        require_input_link(
            root / f"inputs/{phase}/reference_init_relion.mrc",
            RELION_REFERENCE,
            RELION_REFERENCE_SHA256,
            f"{phase} RELION reference",
        )

    allowed_symlinks = {
        root / "inputs/full/particles.star",
        root / "inputs/smoke/reference_init.mrc",
        root / "inputs/smoke/reference_init_relion.mrc",
        root / "inputs/full/reference_init.mrc",
        root / "inputs/full/reference_init_relion.mrc",
    }
    observed_symlinks = {path for path in root.rglob("*") if path.is_symlink()}
    _require(observed_symlinks == allowed_symlinks, "unexpected symlink in launch tree")


def _validate_native_artifacts(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "relion_refine_mpi": {"path": str(RELION_MPI), "sha256": RELION_MPI_SHA256},
        "relion_binding": {"path": str(RELION_BIND_LIBRARY), "sha256": RELION_BIND_LIBRARY_SHA256},
        "relion_binding_source": {
            "path": str(RELION_SOURCE),
            "commit": RELION_SOURCE_COMMIT,
            "tree": RELION_BIND_SOURCE_TREE,
        },
        "recovar_cuda_sm90": {"path": str(CUDA_LIBRARY), "sha256": CUDA_LIBRARY_SHA256},
        "relion_runtime_library_dirs": [str(RELION_CUDA_LIB_DIR), str(RELION_MPI_LIB_DIR)],
    }
    _require(payload.get("native_artifacts") == expected, "native-artifact contract changed")
    records = {
        "relion_refine_mpi": _file_record(RELION_MPI),
        "relion_binding": _file_record(RELION_BIND_LIBRARY),
        "recovar_cuda_sm90": _file_record(CUDA_LIBRARY),
    }
    _require(records["relion_refine_mpi"]["sha256"] == RELION_MPI_SHA256, "RELION MPI binary changed")
    _require(records["relion_binding"]["sha256"] == RELION_BIND_LIBRARY_SHA256, "RELION binding changed")
    _require(records["recovar_cuda_sm90"]["sha256"] == CUDA_LIBRARY_SHA256, "RECOVAR CUDA library changed")
    _require_real_directory(RELION_SOURCE, "RELION source checkout")
    source_commit = _git_output(RELION_SOURCE, "rev-parse", "HEAD")
    source_tree = _git_output(RELION_SOURCE, "rev-parse", "HEAD^{tree}")
    source_status = _git_output(RELION_SOURCE, "status", "--porcelain=v1", "--untracked-files=all")
    _require(source_commit == RELION_SOURCE_COMMIT, "RELION source commit changed")
    _require(source_tree == RELION_BIND_SOURCE_TREE, "RELION source tree changed")
    _require(not source_status, "RELION source checkout is dirty")
    records["relion_binding_source"] = {
        "path": str(RELION_SOURCE),
        "commit": source_commit,
        "tree": source_tree,
        "clean": True,
    }
    return records


def _validate_run_specs(
    payload: dict[str, Any],
    root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    specs = _expected_run_specs(root)
    _require(tuple(spec.key for spec in specs) == RUN_KEYS, "internal RunSpec order changed")
    expected_keys = {
        "path",
        "sha256",
        "engine",
        "phase",
        "resources",
        "data_dir",
        "particle_star",
        "particle_star_sha256",
        "command",
        "command_sha256",
        "expected_outputs",
        "depends_on",
        "slurm_logs",
    }
    for spec in specs:
        row = payload["runs"].get(spec.key, {})
        _require(set(row) == expected_keys, f"RunSpec manifest fields changed for {spec.key}")
        script = root / "scripts" / f"{spec.key}.sbatch"
        expected_star_sha = SMOKE_STAR_SHA256 if spec.phase == "smoke" else PREPARED_STAR_SHA256
        exact = {
            "engine": spec.engine,
            "phase": spec.phase,
            "resources": dataclasses.asdict(spec.resources),
            "data_dir": str(spec.data_dir),
            "particle_star": str(spec.data_dir / "particles.star"),
            "particle_star_sha256": expected_star_sha,
            "command": list(spec.command),
            "command_sha256": sha256_json(list(spec.command)),
            "expected_outputs": [str(path) for path in spec.expected_outputs],
            "depends_on": list(spec.depends_on),
            "slurm_logs": {
                "stdout_template": str(root / "logs" / f"{spec.key}-%j.out"),
                "stderr_template": str(root / "logs" / f"{spec.key}-%j.err"),
            },
        }
        _require_exact_path(row.get("path"), script, f"{spec.key} Slurm script")
        for field, expected in exact.items():
            _require(row.get(field) == expected, f"RunSpec {field} mismatch for {spec.key}")
        _require(not script.is_symlink() and script.is_file(), f"Slurm script is not regular for {spec.key}")
        script_text = script.read_text()
        _require(
            "{RUN_ROOT}" not in script_text and "{RUNTIME_ROOT}" not in script_text, "reserved Slurm token present"
        )
        _require(
            _normalized_slurm_sha256(script_text, root, runtime_root) == SLURM_TEMPLATE_SHA256[spec.key],
            f"Slurm script semantic template mismatch for {spec.key}",
        )
        _require(row.get("sha256") == sha256_file(script), f"Slurm script digest mismatch for {spec.key}")
        row["output_dir"] = str(spec.output_dir)
    return {spec.key: spec for spec in specs}


def audit_launch(
    manifest_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the launch payload and sealed per-job completion ledger."""

    _require(manifest_path.is_absolute(), "launch manifest path must be absolute")
    _require(str(manifest_path) == os.path.normpath(str(manifest_path)), "launch manifest path is not canonical")
    _require(manifest_path.name == "launch_manifest.json", "noncanonical launch manifest name")
    root = manifest_path.parent
    _require(root.parent == ALLOWED_ROOT, f"run root is outside {ALLOWED_ROOT}: {root}")
    runtime_parent = ALLOWED_ROOT / "runtime"
    runtime_root = runtime_parent / root.name
    for label, directory in (
        ("allowed run parent", ALLOWED_ROOT),
        ("runtime parent", runtime_parent),
        ("runtime root", runtime_root),
        ("run root", root),
        ("logs directory", root / "logs"),
        ("evidence directory", root / "evidence"),
        ("provenance directory", root / "provenance"),
        ("scripts directory", root / "scripts"),
        ("inputs directory", root / "inputs"),
        ("smoke input directory", root / "inputs/smoke"),
        ("full input directory", root / "inputs/full"),
        ("outputs directory", root / "outputs"),
        *((f"{key} output directory", root / "outputs" / key) for key in RUN_KEYS),
    ):
        _require_real_directory(directory, label)
    _file_record(manifest_path)
    _file_record(root / "SAFE_TO_DELETE", nonempty=False)
    _file_record(runtime_root / "SAFE_TO_DELETE", nonempty=False)

    payload = json.loads(manifest_path.read_text())
    expected_manifest_keys = {
        "created_utc",
        "native_artifacts",
        "preparation",
        "run_root",
        "runs",
        "runtime_root",
        "safe_to_delete_markers",
        "schema",
        "scientific_contract",
        "smoke_subset",
        "status",
        "subject",
        "submission",
        "symmetry",
    }
    _require(set(payload) == expected_manifest_keys, "launch manifest fields changed")
    _require(payload.get("schema") == LAUNCH_SCHEMA, "wrong launch schema")
    _require(payload.get("status") == "submitted", "launch was not submitted")
    _require_utc_timestamp(payload.get("created_utc"), "launch creation timestamp")
    _require_exact_path(payload.get("run_root"), root, "launch run root")
    _require_exact_path(payload.get("runtime_root"), runtime_root, "runtime root")
    expected_markers = [str(root / "SAFE_TO_DELETE"), str(runtime_root / "SAFE_TO_DELETE")]
    _require(payload.get("safe_to_delete_markers") == expected_markers, "safe-to-delete marker ledger changed")
    _require(set(payload.get("runs", {})) == set(RUN_KEYS), "launch run set changed")
    _validate_subject(payload)
    _validate_scientific_contract(payload)
    _validate_preparation_and_inputs(payload, root)
    native_artifacts = _validate_native_artifacts(payload)
    specs = _validate_run_specs(payload, root, runtime_root)

    submission = payload.get("submission") or {}
    _require(set(submission) == {"dependency", "job_ids", "jobs", "submitted_utc"}, "submission fields changed")
    _require_utc_timestamp(submission.get("submitted_utc"), "submission timestamp")
    _require(set(submission.get("job_ids", {})) == set(RUN_KEYS), "submission job set changed")
    _require(set(submission.get("jobs", {})) == set(RUN_KEYS), "submission path ledger missing")
    submission_file = root / "submission.json"
    _file_record(submission_file)
    _require(json.loads(submission_file.read_text()) == submission, "submission.json mismatch")
    job_ids = {key: str(submission["job_ids"][key]) for key in RUN_KEYS}
    _require(
        submission.get("dependency")
        == {
            "recovar_full": [job_ids["recovar_smoke"], job_ids["relion_smoke"]],
            "relion_full": [job_ids["recovar_smoke"], job_ids["relion_smoke"]],
            "type": "afterok",
        },
        "submission dependency ledger changed",
    )

    jobs: dict[str, Any] = {}
    seen: set[str] = set()
    for key in RUN_KEYS:
        run = payload["runs"][key]
        spec = specs[key]
        job_id = job_ids[key]
        _require(re.fullmatch(r"[0-9]+", job_id) is not None and job_id not in seen, "invalid job ID")
        seen.add(job_id)
        submitted = submission["jobs"][key]
        _require(
            set(submitted) == {"job_id", "sbatch_command", "artifacts", "scontrol_submit_sha256"},
            f"job ledger fields changed for {key}",
        )
        _require(str(submitted.get("job_id")) == job_id, f"job ledger mismatch for {key}")
        script = root / "scripts" / f"{key}.sbatch"
        expected_sbatch = ["sbatch", "--parsable"]
        dependency = _expected_dependency(key, job_ids)
        if spec.depends_on:
            expected_sbatch.append(f"--dependency={dependency}")
        expected_sbatch.append(str(script))
        _require(submitted.get("sbatch_command") == expected_sbatch, f"sbatch command mismatch for {key}")

        output_dir = spec.output_dir
        paths = _canonical_paths(root, key, job_id, run["engine"], output_dir)
        stored = submitted.get("artifacts", {})
        _require(set(stored) == set(paths), f"artifact path ledger changed for {key}")
        for name, path in paths.items():
            _require_exact_path(stored.get(name), path, f"{key} {name}")

        command = list(spec.command)
        executed = _file_record(paths["executed_command"])
        _require(shlex.split(Path(executed["path"]).read_text()) == command, f"executed command mismatch for {key}")

        submit_record = _file_record(paths["scontrol_submit"])
        _require(
            submit_record["sha256"] == submitted["scontrol_submit_sha256"],
            f"submission scontrol hash mismatch for {key}",
        )
        submit_scontrol = paths["scontrol_submit"].read_text()
        _validate_scontrol_identity(
            submit_scontrol,
            key=key,
            job_id=job_id,
            resources=run["resources"],
            script=script,
            stdout=paths["stdout_log"],
            stderr=paths["stderr_log"],
            dependency=dependency,
            submit_time=True,
        )
        submit_requested = _validate_requested_tres(submit_scontrol, run["resources"], key)

        _file_record(paths["scontrol_runtime"])
        runtime_scontrol = Path(paths["scontrol_runtime"]).read_text()
        _validate_scontrol_identity(
            runtime_scontrol,
            key=key,
            job_id=job_id,
            resources=run["resources"],
            script=script,
            stdout=paths["stdout_log"],
            stderr=paths["stderr_log"],
            dependency=None,
            submit_time=False,
        )
        tres = _validate_tres(runtime_scontrol, job_id, run["resources"])
        _require(tres["requested"] == submit_requested, f"requested TRES changed after submission for {key}")
        accounting = _completed_accounting(job_id, runner=runner)
        expected_outputs = [_file_record(path) for path in spec.expected_outputs]
        provenance = {
            name: _file_record(path, nonempty=name not in {"stderr_log", "completed_marker"})
            for name, path in paths.items()
        }
        walltime = json.loads(paths["walltime"].read_text())
        _require(str(walltime.get("job_id")) == job_id, f"walltime job mismatch for {key}")
        _require(walltime.get("run_key") == key, f"walltime run mismatch for {key}")
        _require("not found" not in paths["dynamic_libraries"].read_text(), f"unresolved library for {key}")
        _require(
            paths["final_environment"].read_text() == FINAL_ENVIRONMENT_TEXT,
            f"final all-data environment mismatch for {key}",
        )
        _validate_gpu_identity(paths["gpu_identity"], int(run["resources"]["h100_gpus"]), key)
        inventory = paths["gpu_inventory"].read_text()
        _require("H100" in inventory, f"GPU inventory does not prove H100 allocation for {key}")
        jobs[key] = {
            "job_id": job_id,
            "engine": run["engine"],
            "phase": run["phase"],
            "resources": run["resources"],
            "tres": tres,
            "accounting": accounting,
            "command": command,
            "command_sha256": run["command_sha256"],
            "slurm_script": _file_record(script),
            "provenance": provenance,
            "expected_outputs": expected_outputs,
        }
    return payload, {
        "launcher_at_finalization": _file_record(LAUNCHER),
        "launch_manifest": _file_record(manifest_path),
        "submission": _file_record(submission_file),
        "native_artifacts": native_artifacts,
        "jobs": jobs,
    }


def _symlink_exact(source: Path, destination: Path) -> None:
    _require(source.is_file(), f"missing log source: {source}")
    destination.symlink_to(source.resolve())


def _run_analysis(
    payload: dict[str, Any],
    audit: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(payload["run_root"])
    evidence_root = root / "evidence"
    logs = evidence_root / "collector_logs"
    raw_output = evidence_root / "raw_collector"
    diagnostics_output = evidence_root / "proper_so3"
    for path in (logs, raw_output, diagnostics_output):
        _require(not path.exists(), f"refusing to reuse analysis path: {path}")
    logs.mkdir()
    recovar_log = Path(audit["jobs"]["recovar_full"]["provenance"]["stdout_log"]["path"])
    relion_log = Path(audit["jobs"]["relion_full"]["provenance"]["stdout_log"]["path"])
    _symlink_exact(recovar_log, logs / "recovar_refine.log")
    _symlink_exact(relion_log, logs / "relion_refine.log")

    _require(sha256_file(RAW_COLLECTOR) == RAW_COLLECTOR_SHA256, "raw collector SHA-256 changed")
    _require(
        sha256_file(DIAGNOSTIC_PRODUCER) == DIAGNOSTIC_PRODUCER_SHA256,
        "proper-SO3 producer SHA-256 changed",
    )
    subject_python = payload["subject"]["python"]
    full_data = Path(payload["runs"]["recovar_full"]["data_dir"])
    recovar_dir = Path(payload["runs"]["recovar_full"]["output_dir"])
    relion_dir = Path(payload["runs"]["relion_full"]["output_dir"])
    raw_command = [
        subject_python,
        str(RAW_COLLECTOR),
        "--dataset",
        "10202",
        "--prepared-star",
        str(full_data / "particles.star"),
        "--relion-dir",
        str(relion_dir),
        "--recovar-dir",
        str(recovar_dir),
        "--relion-initial-reference",
        str(full_data / "reference_init_relion.mrc"),
        "--recovar-initial-reference",
        str(full_data / "reference_init.mrc"),
        "--log-dir",
        str(logs),
        "--output-dir",
        str(raw_output),
    ]
    diagnostic_command = [
        subject_python,
        str(DIAGNOSTIC_PRODUCER),
        "--case-id",
        CASE_ID,
        "--recovar-merged",
        str(recovar_dir / "final_merged.mrc"),
        "--recovar-half1",
        str(recovar_dir / "final_half1_unfil.mrc"),
        "--recovar-half2",
        str(recovar_dir / "final_half2_unfil.mrc"),
        "--relion-merged",
        str(relion_dir / "run_class001.mrc"),
        "--relion-half1",
        str(relion_dir / "run_half1_class001_unfil.mrc"),
        "--relion-half2",
        str(relion_dir / "run_half2_class001_unfil.mrc"),
        "--symmetry-label",
        "I1",
        "--symmetry-operators-sha256",
        SYMMETRY_OPERATORS_SHA256,
        "--output-dir",
        str(diagnostics_output),
    ]
    clean_env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"):
        clean_env.pop(name, None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    runner(raw_command, check=True, cwd=str(RAW_COLLECTOR.parents[1]), env=clean_env)
    runner(diagnostic_command, check=True, cwd=str(Path(payload["subject"]["repo"])), env=clean_env)

    metrics = raw_output / "metrics.json"
    curves = raw_output / "fsc_curves.npz"
    metrics_payload = json.loads(metrics.read_text())
    _require(metrics_payload.get("schema") == COLLECTOR_SCHEMA, "wrong raw collector schema")
    _require(metrics_payload.get("scientifically_valid") is True, "raw collector rejected the comparison")
    convergence = metrics_payload.get("convergence_and_topology", {})
    _require(convergence.get("recovar_converged") is True, "RECOVAR did not prove convergence")
    _require(convergence.get("relion_converged") is True, "RELION did not prove convergence")
    map_paths = {
        "recovar_final_sha256": recovar_dir / "final_merged.mrc",
        "recovar_half1_sha256": recovar_dir / "final_half1_unfil.mrc",
        "recovar_half2_sha256": recovar_dir / "final_half2_unfil.mrc",
        "relion_final_sha256": relion_dir / "run_class001.mrc",
        "relion_half1_sha256": relion_dir / "run_half1_class001_unfil.mrc",
        "relion_half2_sha256": relion_dir / "run_half2_class001_unfil.mrc",
    }
    metric_artifacts = metrics_payload.get("artifacts", {})
    for key, path in map_paths.items():
        _require(metric_artifacts.get(key) == sha256_file(path), f"raw collector map hash mismatch: {key}")
    with np.load(curves, allow_pickle=False) as archive:
        raw_fields = set(archive.files)
    _require(
        {
            "relion_final_half_fsc",
            "recovar_final_half_fsc",
            "final_cross_engine_raw",
            "final_cross_engine_half1",
            "final_cross_engine_half2",
        }.issubset(raw_fields),
        "primary FSC curve fields missing",
    )
    diagnostics = diagnostics_output / "science_diagnostics.json"
    diagnostic_curves = diagnostics_output / "science_diagnostic_curves.npz"
    mask = diagnostics_output / "common_soft_mask.mrc"
    diagnostic_payload = json.loads(diagnostics.read_text())
    _require(diagnostic_payload.get("schema") == SCIENCE_SCHEMA, "wrong diagnostic schema")
    with np.load(diagnostic_curves, allow_pickle=False) as archive:
        fields = set(archive.files)
    _require(ALIGNED_FIELDS.issubset(fields), "proper-SO3 curve fields missing")
    _require(MASKED_FIELDS.issubset(fields), "common-mask curve fields missing")
    alignment = diagnostic_payload.get("diagnostics", {}).get("proper_so3_alignment", {})
    _require(alignment.get("applied_unchanged_to") == ["merged", "half1", "half2"], "transform was not reused")
    _require(alignment.get("no_reflection") is True, "reflection is forbidden")
    _require(alignment.get("sign_fit") is False and alignment.get("scale_fit") is False, "sign/scale fit is forbidden")
    return {
        "schema": COLLECTOR_SCHEMA,
        "collector_path": str(RAW_COLLECTOR),
        "collector_sha256": RAW_COLLECTOR_SHA256,
        "metrics_json": str(metrics.resolve()),
        "metrics_sha256": sha256_file(metrics),
        "fsc_curves_npz": str(curves.resolve()),
        "curves_sha256": sha256_file(curves),
    }, {
        "commands": {
            "raw_collector": {"argv": raw_command, "sha256": sha256_json(raw_command)},
            "proper_so3": {"argv": diagnostic_command, "sha256": sha256_json(diagnostic_command)},
        },
        "analysis_artifacts": {
            "science_diagnostics": {
                "path": str(diagnostics.resolve()),
                "sha256": sha256_file(diagnostics),
                "schema": SCIENCE_SCHEMA,
            },
            "curve_archive": {
                "path": str(diagnostic_curves.resolve()),
                "sha256": sha256_file(diagnostic_curves),
                "fields": sorted(fields),
            },
            "common_mask": {"path": str(mask.resolve()), "sha256": sha256_file(mask)},
        },
    }


def _input_contract(payload: dict[str, Any]) -> dict[str, Any]:
    prep_path = Path(payload["preparation"]["preparation_manifest"])
    _require(
        sha256_file(prep_path) == payload["preparation"]["preparation_manifest_sha256"], "preparation manifest changed"
    )
    prep = json.loads(prep_path.read_text())
    source = prep["input_contract"]
    prepared = source["prepared_star"]
    normalized = source["normalized_legacy_star"]
    reference = prep["initial_reference"]
    symmetry = payload["symmetry"]
    return {
        "particle_count": prepared["particle_count"],
        "box_size": source["box_size"],
        "voxel_size_angstrom": source["voxel_size_angstrom"],
        "prepared_star_sha256": prepared["sha256"],
        "poses_pkl_sha256": source["poses_pkl"]["sha256"],
        "ctf_pkl_sha256": source["ctf_pkl"]["sha256"],
        "prepared_star_preserves_source_particle_order": prepared["preserves_legacy_particle_order"],
        "prepared_star_preserves_source_half_assignment": prepared["preserves_legacy_half_assignment"],
        "prepared_star_preserves_source_pose_shift_metadata": prepared["preserves_legacy_pose_metadata"],
        "particle_stack_size_bytes": source["particle_stack"]["size_bytes"],
        "particle_stack_sha256": source["particle_stack"]["sha256"],
        "half_assignment_sha256": prepared["half_assignment_sha256"],
        "half_assignment_encoding": normalized["half_assignment_encoding"],
        "half_assignment_bytes": normalized["half_assignment_bytes"],
        "half1_count": normalized["half1_count"],
        "half2_count": normalized["half2_count"],
        "initial_reference_canonical_exact": reference["canonical_exact"],
        "initial_reference_relion_file_sha256": reference["relion_file"]["sha256"],
        "initial_reference_recovar_frame_file_sha256": reference["recovar_frame_file"]["sha256"],
        "initial_reference_canonical_array_sha256": reference["canonical_array_sha256"],
        "k": 1,
        "autonomous_refinement": True,
        "fixed_poses": False,
        "local_only": False,
        "forced_final_after_nonconvergence": False,
        "symmetry": symmetry,
    }


def finalize(
    manifest_path: Path,
    *,
    replacement_records: Sequence[Path] = (),
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    manifest_schema = json.loads(manifest_path.read_text()).get("schema")
    native = manifest_schema == NATIVE_LAUNCH_SCHEMA
    if native:
        adapter = _load_native_adapter()
        payload, audit = adapter.audit_native_launch(
            manifest_path,
            replacement_records=replacement_records,
            runner=runner,
            require_science_ready=True,
        )
        evidence_root = Path(payload["run_root"]) / "evidence"
        _require(not evidence_root.exists(), f"refusing to reuse native evidence path: {evidence_root}")
        evidence_root.mkdir()
    else:
        _require(not replacement_records, "replacement records require the native launch schema")
        payload, audit = audit_launch(manifest_path, runner=runner)
    collector, analysis = _run_analysis(payload, audit, runner=runner)
    output = Path(payload["run_root"]) / "evidence" / "case_evidence.json"
    _require(not output.exists(), f"refusing to overwrite evidence: {output}")
    execution_envelope = output.with_name("execution_envelope.json")
    _require(not execution_envelope.exists(), f"refusing to overwrite envelope: {execution_envelope}")
    execution_payload = {
        "schema": EXECUTION_BINDING_SCHEMA,
        "launch": audit,
        "collector": collector,
        "analysis_commands": analysis["commands"],
        "analysis_artifacts": analysis["analysis_artifacts"],
    }
    execution_envelope.write_text(json.dumps(execution_payload, indent=2, sort_keys=True) + "\n")
    finalizer_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--launch-manifest",
        str(manifest_path.resolve()),
    ]
    for replacement in replacement_records:
        finalizer_argv.extend(["--replacement-record", str(replacement.resolve())])
    execution_binding = {
        "schema": EXECUTION_BINDING_SCHEMA,
        "launch_manifest_path": audit["launch_manifest"]["path"],
        "launch_manifest_sha256": audit["launch_manifest"]["sha256"],
        "finalizer_command_path": str(Path(__file__).resolve()),
        "finalizer_command_sha256": sha256_file(Path(__file__).resolve()),
        "finalizer_argv": finalizer_argv,
        "finalizer_argv_sha256": sha256_json(finalizer_argv),
        "evidence_envelope_path": str(execution_envelope.resolve()),
        "evidence_envelope_sha256": sha256_file(execution_envelope),
    }
    if native:
        execution_binding["launch_manifest_schema"] = NATIVE_LAUNCH_SCHEMA
        execution_binding["replacement_records"] = [_file_record(path.resolve()) for path in replacement_records]
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "case_id": CASE_ID,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "subject": payload["subject"],
        "input_contract": _input_contract(payload),
        "launch": audit,
        "collector": collector,
        "analysis_commands": analysis["commands"],
        "analysis_artifacts": analysis["analysis_artifacts"],
        "execution_binding": execution_binding,
    }
    partial = output.with_suffix(".json.partial")
    partial.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--replacement-record", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = finalize(
        args.launch_manifest.resolve(),
        replacement_records=tuple(path.resolve() for path in args.replacement_record),
    )
    print(json.dumps({"evidence": str(output), "sha256": sha256_file(output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
