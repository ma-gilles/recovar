#!/usr/bin/env python3
"""Audit native EMPIAR-10202 set-6 matched-harness execution evidence.

The original PR158 finalizer consumes the launcher's rich JSON schema.  The
box-800 runs were launched by a smaller, shell-native harness instead.  This
module validates that native harness without treating its manifest as a source
of truth for results: commands, subjects, native binaries, allocations, GPU
identities, terminal states, and outputs are all checked against independent
records.  It returns the normalized payload expected by the existing
finalizer, but marks it science-eligible only when both *full* engines have
completed successfully.  Smoke runs are capability evidence only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

NATIVE_LAUNCH_SCHEMA = "recovar.em.matched_launch_harness.v1"
NATIVE_SUBMISSION_SCHEMA = "recovar.em.matched_submission.v1"
REPLACEMENT_SCHEMA = "recovar.em.native_job_replacement.v1"
STANDALONE_REPLACEMENT_SCHEMA = "recovar.em.standalone_recovar_full_replacement.v1"
AUDIT_SCHEMA = "recovar.em.native_matched_execution_audit.v1"
RUN_KEYS = ("recovar_smoke", "relion_smoke", "recovar_full", "relion_full")
ALLOWED_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/codex")
LAUNCHER_PATH = Path(__file__).resolve().with_name("launch_empiar10202_set6_i1_matched.py")
LAUNCHER_SHA256 = "b5cecfcf8088149cfb28e1829cf1d6b65eed6f8914c450d8b7fb73e31353b5eb"
PREPARATION_SHA256 = "05865373813f50b83060ef4c93993522472b5ca3189c4acd27647c18262e501d"
FULL_STAR_SHA256 = "d66afb3001e6e43463fb699804fe8b50f8cef1f2ccd9730f5275955b6be7b512"
SMOKE_STAR_SHA256 = "500dc2b76fdd5554d1dba759168184109f16d91deed0fce1fc2d9f5daf164d7d"
PARTICLE_STACK_SHA256 = "8eecf0fbf8e645ac51feff278a86e43e7e4be117921333dc6d3e22e52a628453"
PARTICLE_STACK_SIZE_BYTES = 78_118_401_024
RECOVAR_REFERENCE_SHA256 = "d77516a08e5e3ccdef07d9039e36d65b174afe5d56fe20d7775eef188d2e6cc6"
RELION_REFERENCE_SHA256 = "4f83710c999276d4f65cff586266ee121f271e8bed382dba329b384860af96bb"
RELION_MPI_SHA256 = "92cf3ba54038d5e162e238b952fe88f1414f440d4e6cba23bc4b097428087b4a"
RELION_BIND_SHA256 = "82b0a8cf2c189463f9cf0181099f4e92ce4365cce3819463c27af44b3c1014a2"
RELION_SOURCE_COMMIT = "d476e6f6a4f1f37627c06ace5227fc374c0c2b05"
RELION_SOURCE_TREE = "1633d228e89d91ede8ad0996e727ec6ab1bc96ee"
CUDA_SOURCE_SHA256 = "4238bc4eb344c0e4bb12989a0b41fdcf62c455164ab207e3add44612f377ee9a"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SUBJECT_FIELDS = ("SUBJECT_REPO", "SUBJECT_COMMIT", "SUBJECT_TREE", "CUDA_LIB", "CUDA_LIB_SHA256")
OOM_PATTERNS = (
    re.compile(r"\bCUDA:\s*out of memory\b", re.IGNORECASE),
    re.compile(r"\bRESOURCE_EXHAUSTED:\s*Out of memory\b", re.IGNORECASE),
    re.compile(r"\bran out of memory\b", re.IGNORECASE),
    re.compile(r"\bCUDA_ERROR_OUT_OF_MEMORY\b", re.IGNORECASE),
)


@dataclasses.dataclass(frozen=True)
class StandaloneReplacementProfile:
    """Immutable acceptance contract for one independently launched full run."""

    name: str
    job_id: str
    job_name: str
    run_root: Path
    runtime_root: Path
    script_sha256: str
    source_repo: Path
    source_commit: str
    source_tree: str
    python: Path
    driver_sha256: str
    cuda_source_sha256: str
    cuda_library_relative_path: Path
    cuda_preflight_sha256: str
    cuda_postbuild_sha256: str
    cuda_postbuild_size_bytes: int
    preflight_ledger_sha256: str
    preflight_readme_sha256: str
    postbuild_record_sha256: str
    postbuild_reason: str
    particle_count: int
    half1_count: int
    half2_count: int
    artifact_relative_paths: tuple[tuple[str, str], ...]


_STANDALONE_13376414_PROFILE = StandaloneReplacementProfile(
    name="empiar10202_set6_i1_recovar_full_nopad_b31f7bb3a_job13376414",
    job_id="13376414",
    job_name="10202-s6-nopad",
    run_root=Path(
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "empiar10202_set6_i1_full_recovar_nopad_b31f7bb3a_20260903"
    ),
    runtime_root=Path(
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/"
        "empiar10202_set6_i1_full_recovar_nopad_b31f7bb3a_20260903"
    ),
    script_sha256="91ba2a34b603190b35c4d1c2961727f29d9839cd5c89b3726a38a3ad59ade5e1",
    source_repo=Path(
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "source_recovar_b31f7bb3a_20260903/checkout"
    ),
    source_commit="b31f7bb3a88b96885e568fa4a12d5ec265ab4aab",
    source_tree="e4a9c5e05700ea5a89177627fa48da48fae53b0a",
    python=Path(
        "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/"
        "recovar_pr158_k4_origin_docs_8cbebdecc_20260902/.pixi/envs/default/bin/python"
    ),
    driver_sha256="0d6a53fd4c499d58e4987f8d86667ada8c4426970b55f0e133662f0548b1e594",
    cuda_source_sha256="ce86b544f3d831d48c63f7b9dff5c39db724925a5fa78c82c159409e879417b7",
    cuda_library_relative_path=Path("sealed_input/libcuda_backproject_sm90_seed42002_141a8e2d.so"),
    cuda_preflight_sha256="141a8e2dee8adee4fb164d5d888452dc8b89bd0ad187fef34071eca65b532958",
    cuda_postbuild_sha256="8fe96cd3cba739617098b5573ccdd3deeb8678a67bb1aef28294f4802ade2aca",
    cuda_postbuild_size_bytes=47_142_296,
    preflight_ledger_sha256="b725f1a7e835a2b4abb94b96560f2e227d32b528b9d789338433ad2c52a1aac5",
    preflight_readme_sha256="56f10cec0d9ee53fda9d0176961f5962b8446da2e61da4842273f9706e99ec77",
    postbuild_record_sha256="06c1e2709d5419fbb55e365a16d1a55cbfc507ca7ea19ee1773ae24a332feef9",
    postbuild_reason=(
        "RECOVAR rebuilt the run-private binary because its copied mtime predated the sealed checkout "
        "source mtime; no shared binary was mutated."
    ),
    particle_count=30_515,
    half1_count=15_258,
    half2_count=15_257,
    artifact_relative_paths=(
        ("run_safe_to_delete", "SAFE_TO_DELETE"),
        ("job_id", "provenance/job_id.txt"),
        ("scontrol_submit", "provenance/submitted_scontrol_retry.txt"),
        ("scontrol_runtime", "provenance/scontrol-{job_id}.txt"),
        ("executed_command", "provenance/command-{job_id}.sh"),
        ("subject", "provenance/subject-{job_id}.txt"),
        ("environment", "provenance/environment-{job_id}.txt"),
        ("import", "provenance/import-{job_id}.txt"),
        ("gpu_inventory", "provenance/nvidia-smi-{job_id}.txt"),
        ("cuda_pre_submit_libraries", "provenance/ldd-cuda-pre-submit.txt"),
        ("cuda_runtime_libraries", "provenance/ldd-cuda-{job_id}.txt"),
        ("relion_bind_libraries", "provenance/ldd-relion-bind-{job_id}.txt"),
        ("preflight_ledger", "provenance/pre_submission_retry_sha256.txt"),
        ("cuda_postbuild", "provenance/cuda-postbuild-{job_id}.txt"),
        ("stdout_log", "logs/recovar-full-{job_id}.out"),
        ("stderr_log", "logs/recovar-full-{job_id}.err"),
        ("hbm_trace", "logs/recovar-full-{job_id}-hbm.csv"),
        ("time_verbose", "logs/recovar-full-{job_id}-time-v.txt"),
        ("completed_marker", "outputs/recovar_full/COMPLETED"),
        ("walltime", "outputs/recovar_full/slurm_walltime.json"),
        ("hbm_summary", "outputs/recovar_full/hbm_summary.txt"),
        ("output_sha256", "outputs/recovar_full/output_sha256.txt"),
        ("refinement_results", "outputs/recovar_full/refinement_results.npz"),
        ("final_merged", "outputs/recovar_full/final_merged.mrc"),
        ("final_half1_unfil", "outputs/recovar_full/final_half1_unfil.mrc"),
        ("final_half2_unfil", "outputs/recovar_full/final_half2_unfil.mrc"),
    ),
)

# Adding a future standalone run requires a new immutable profile, not a schema
# relaxation.  This adapter intentionally accepts only the corrected 13376414
# profile at present.
STANDALONE_REPLACEMENT_PROFILES: Mapping[str, StandaloneReplacementProfile] = {
    _STANDALONE_13376414_PROFILE.name: _STANDALONE_13376414_PROFILE,
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


def _file_record(path: Path, *, nonempty: bool = True) -> dict[str, Any]:
    _require(path.is_absolute(), f"artifact path is not absolute: {path}")
    _require(not path.is_symlink(), f"artifact must not be a symlink: {path}")
    _require(path.is_file(), f"missing regular artifact: {path}")
    size = path.stat().st_size
    _require(not nonempty or size > 0, f"empty artifact: {path}")
    return {"path": str(path), "size_bytes": size, "sha256": sha256_file(path)}


def _require_directory(path: Path, label: str) -> None:
    _require(path.is_absolute(), f"{label} is not absolute: {path}")
    _require(not path.is_symlink() and path.is_dir(), f"{label} is not a real directory: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    _file_record(path)
    payload = json.loads(path.read_text())
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_launcher() -> ModuleType:
    _require(sha256_file(LAUNCHER_PATH) == LAUNCHER_SHA256, "matched launcher SHA-256 changed")
    spec = importlib.util.spec_from_file_location("_recovar_empiar10202_matched_launcher", LAUNCHER_PATH)
    _require(spec is not None and spec.loader is not None, "cannot load matched launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_key_values(path: Path, expected: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = shlex.split(line, comments=True, posix=True)
        _require(len(tokens) == 1 and "=" in tokens[0], f"malformed key/value at {path}:{line_number}")
        name, value = tokens[0].split("=", 1)
        _require(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is not None,
            f"invalid key at {path}:{line_number}",
        )
        _require(name not in values, f"duplicate key {name} in {path}")
        values[name] = value
    _require(set(values) == set(expected), f"unexpected key set in {path}: {sorted(values)}")
    return values


def _field(text: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S+)", text)
    _require(match is not None, f"scontrol record lacks {name}")
    return match.group(1)


def _required_tres(resources: Mapping[str, Any]) -> set[str]:
    return {
        f"cpu={int(resources['ntasks']) * int(resources['cpus_per_task'])}",
        f"mem={resources['memory']}",
        f"node={resources['nodes']}",
        f"gres/gpu={resources['h100_gpus']}",
    }


def _validate_requested_tres(text: str, resources: Mapping[str, Any], label: str) -> str:
    requested = _field(text, "ReqTRES")
    _require(_required_tres(resources).issubset(set(requested.split(","))), f"wrong requested TRES for {label}")
    return requested


def _validate_runtime_tres(text: str, resources: Mapping[str, Any], label: str) -> dict[str, str]:
    requested = _validate_requested_tres(text, resources, label)
    allocated = _field(text, "AllocTRES")
    _require(requested == allocated, f"ReqTRES != AllocTRES for {label}")
    return {"requested": requested, "allocated": allocated}


def _script_job_name(script: Path) -> str:
    match = re.search(r"^#SBATCH\s+--job-name=(\S+)\s*$", script.read_text(), re.MULTILINE)
    _require(match is not None, f"Slurm script lacks --job-name: {script}")
    return match.group(1)


def _validate_scontrol_identity(
    text: str,
    *,
    job_id: str,
    script: Path,
    resources: Mapping[str, Any],
    run_root: Path,
) -> tuple[Path, Path]:
    exact = {
        "JobId": job_id,
        "JobName": _script_job_name(script),
        "Command": str(script),
        "NumCPUs": str(int(resources["ntasks"]) * int(resources["cpus_per_task"])),
        "NumTasks": str(resources["ntasks"]),
        "CPUs/Task": str(resources["cpus_per_task"]),
        "Features": "h100",
        "TresPerNode": f"gres/gpu:h100:{resources['h100_gpus']}",
    }
    for name, expected in exact.items():
        _require(_field(text, name) == expected, f"scontrol {name} mismatch for job {job_id}")
    stdout = Path(_field(text, "StdOut"))
    stderr = Path(_field(text, "StdErr"))
    for path, suffix in ((stdout, ".out"), (stderr, ".err")):
        _require(path.is_absolute(), f"non-absolute Slurm log for job {job_id}")
        _require(path.parent == run_root / "logs", f"Slurm log escaped run root for job {job_id}")
        _require(path.name.endswith(f"-{job_id}{suffix}"), f"Slurm log does not bind job {job_id}")
    return stdout, stderr


def _gpu_inventory(path: Path, expected_count: int, label: str) -> dict[str, Any]:
    record = _file_record(path)
    text = path.read_text()
    attached = re.findall(r"^Attached GPUs\s*:\s*(\d+)\s*$", text, re.MULTILINE)
    names = re.findall(r"^\s*Product Name\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    uuids = re.findall(r"^\s*GPU UUID\s*:\s*(GPU-\S+)\s*$", text, re.MULTILINE)
    _require(attached == [str(expected_count)], f"wrong attached GPU count for {label}")
    _require(len(names) == expected_count and all("H100" in name for name in names), f"non-H100 GPU for {label}")
    _require(len(uuids) == expected_count and len(set(uuids)) == expected_count, f"wrong GPU identities for {label}")
    return {**record, "count": expected_count, "uuids": uuids, "names": names}


def _validate_success_gpu_identity(path: Path, inventory: Mapping[str, Any], label: str) -> dict[str, Any]:
    record = _file_record(path)
    rows = [line.split(",", 1) for line in path.read_text().splitlines() if line.strip()]
    _require(all(len(row) == 2 for row in rows), f"malformed GPU identity for {label}")
    observed = {(row[0].strip(), row[1].strip()) for row in rows}
    expected = set(zip(inventory["uuids"], inventory["names"], strict=True))
    _require(observed == expected, f"GPU identity does not match allocation inventory for {label}")
    return record


def _subject_from_values(values: Mapping[str, str]) -> dict[str, Any]:
    repo = Path(values["SUBJECT_REPO"])
    cuda_lib = Path(values["CUDA_LIB"])
    _require(repo.is_absolute() and cuda_lib.is_absolute(), "subject paths must be absolute")
    _require(re.fullmatch(r"[0-9a-f]{40}", values["SUBJECT_COMMIT"]) is not None, "invalid subject commit")
    _require(re.fullmatch(r"[0-9a-f]{40}", values["SUBJECT_TREE"]) is not None, "invalid subject tree")
    _require(re.fullmatch(r"[0-9a-f]{64}", values["CUDA_LIB_SHA256"]) is not None, "invalid CUDA digest")
    _require_directory(repo, "subject repository")
    _require(_git_output(repo, "rev-parse", "HEAD") == values["SUBJECT_COMMIT"], "subject commit changed")
    _require(_git_output(repo, "rev-parse", "HEAD^{tree}") == values["SUBJECT_TREE"], "subject tree changed")
    _require(not _git_output(repo, "status", "--porcelain=v1", "--untracked-files=all"), "subject tree is dirty")
    _require(cuda_lib.is_file() and not cuda_lib.is_symlink(), "sealed CUDA library is missing or a symlink")
    _require(sha256_file(cuda_lib) == values["CUDA_LIB_SHA256"], "sealed CUDA library SHA-256 changed")
    python = repo / ".pixi/envs/default/bin/python"
    driver = repo / "scripts/run_full_refinement.py"
    cuda_source = repo / "recovar/cuda/cuda_backproject.cu"
    for path in (python, driver, cuda_source):
        _require(path.is_file(), f"subject artifact is missing: {path}")
    _require(sha256_file(cuda_source) == CUDA_SOURCE_SHA256, "subject CUDA source SHA-256 changed")
    return {
        "repo": str(repo),
        "commit": values["SUBJECT_COMMIT"],
        "tree": values["SUBJECT_TREE"],
        "tree_clean": True,
        "diff_sha256": EMPTY_SHA256,
        "python": str(python),
        "driver": str(driver),
        "driver_sha256": sha256_file(driver),
        "cuda_source": str(cuda_source),
        "cuda_source_sha256": CUDA_SOURCE_SHA256,
        "cuda_library": str(cuda_lib),
        "cuda_library_sha256": values["CUDA_LIB_SHA256"],
    }


def _subject_values(subject: Mapping[str, Any]) -> dict[str, str]:
    return {
        "SUBJECT_REPO": str(subject["repo"]),
        "SUBJECT_COMMIT": str(subject["commit"]),
        "SUBJECT_TREE": str(subject["tree"]),
        "CUDA_LIB": str(subject["cuda_library"]),
        "CUDA_LIB_SHA256": str(subject["cuda_library_sha256"]),
    }


def _validate_subject_record(path: Path, subject: Mapping[str, Any]) -> dict[str, Any]:
    record = _file_record(path)
    _require(
        _parse_key_values(path, tuple(name.lower() for name in SUBJECT_FIELDS))
        == {name.lower(): value for name, value in _subject_values(subject).items()},
        f"job subject record mismatch: {path}",
    )
    return record


def _validate_final_environment(path: Path) -> dict[str, Any]:
    record = _file_record(path)
    rows = _parse_key_values(
        path,
        tuple(
            line.split("=", 1)[0]
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
    )
    _require(rows.get("RECOVAR_FINAL_ALL_DATA_GRID_CORRECT") == "unset", "grid-correct environment changed")
    _require(rows.get("RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER") == "unset", "forced-final environment changed")
    return record


def _validate_environment(path: Path, subject: Mapping[str, Any], *, engine: str) -> dict[str, Any]:
    record = _file_record(path)
    lines = set(path.read_text(errors="replace").splitlines())
    _require("PYTHONNOUSERSITE=1" in lines, "job environment lacks PYTHONNOUSERSITE=1")
    _require(
        "XLA_PYTHON_CLIENT_PREALLOCATE=false" in lines,
        "job environment lacks XLA_PYTHON_CLIENT_PREALLOCATE=false",
    )
    _require(
        not any(
            line.startswith("RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=")
            or line.startswith("RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=")
            for line in lines
        ),
        "job environment unexpectedly sets final all-data overrides",
    )
    if engine == "recovar":
        _require(
            f"RECOVAR_CUDA_LIB={subject['cuda_library']}" in lines,
            "job environment names the wrong CUDA library",
        )
        _require(
            f"RECOVAR_EXPECTED_REPO_ROOT={subject['repo']}" in lines,
            "job environment names the wrong subject repository",
        )
    return record


def _accounting(
    job_id: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, str]:
    command = [
        "sacct",
        "-X",
        "--noheader",
        "--parsable2",
        "--jobs",
        job_id,
        "--format=JobIDRaw,State,ExitCode,ReqTRES,AllocTRES,MaxRSS",
    ]
    result = runner(command, check=True, capture_output=True, text=True)
    rows = [line.split("|")[:6] for line in result.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) == 6 and row[0] == job_id]
    _require(len(matches) == 1, f"sacct did not return one top-level row for {job_id}")
    _, state, exit_code, requested, allocated, max_rss = matches[0]
    return {
        "state": state.rstrip("+"),
        "exit_code": exit_code,
        "requested_tres": requested,
        "allocated_tres": allocated,
        "max_rss": max_rss,
        "query": shlex.join(command),
    }


def _pending_reason(job_id: str, *, runner: Callable[..., subprocess.CompletedProcess[str]]) -> str | None:
    command = ["squeue", "--noheader", "--jobs", job_id, "--format=%i|%T|%r"]
    result = runner(command, check=False, capture_output=True, text=True)
    rows = [line.split("|", 2) for line in result.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) == 3 and row[0] == job_id]
    return matches[0][2] if len(matches) == 1 else None


def _classify(accounting: Mapping[str, str], stderr_text: str, pending_reason: str | None) -> str:
    state = accounting["state"]
    exit_code = accounting["exit_code"]
    if state == "COMPLETED":
        _require(exit_code == "0:0", "COMPLETED job has nonzero exit code")
        return "completed"
    if state == "PENDING" and pending_reason == "DependencyNeverSatisfied":
        return "dependency_never_satisfied"
    if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
        return "active"
    if state == "OUT_OF_MEMORY" or any(pattern.search(stderr_text) for pattern in OOM_PATTERNS):
        _require(exit_code != "0:0", "OOM classification has zero exit code")
        return "oom"
    _require(exit_code != "0:0", f"failed terminal state {state} has zero exit code")
    return "failed"


@dataclasses.dataclass(frozen=True)
class JobSource:
    run_key: str
    job_id: str
    run_root: Path
    script: Path
    script_sha256: str
    subject: Mapping[str, Any]
    submit_record_required: bool
    replacement_record: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class StandaloneJobSource:
    run_key: str
    job_id: str
    run_root: Path
    script: Path
    script_sha256: str
    subject: Mapping[str, Any]
    data_dir: Path
    profile: StandaloneReplacementProfile
    artifacts: Mapping[str, Mapping[str, Any]]
    inputs: Mapping[str, Mapping[str, Any]]
    replacement_record: Mapping[str, Any]


def _validate_declared_file_record(
    declared: Any,
    expected_path: Path,
    *,
    nonempty: bool = True,
) -> dict[str, Any]:
    _require(isinstance(declared, dict), f"missing declared file record for {expected_path}")
    _require(set(declared) == {"path", "size_bytes", "sha256"}, f"file record fields changed: {expected_path}")
    observed = _file_record(expected_path, nonempty=nonempty)
    _require(declared == observed, f"declared file record does not match artifact: {expected_path}")
    return observed


def _input_file_record(
    declared: Any,
    expected_path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
    already_validated: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate content-addressed input while permitting alternate input paths."""

    _require(isinstance(declared, dict), f"missing declared input record for {expected_path}")
    _require(set(declared) == {"path", "size_bytes", "sha256"}, f"input record fields changed: {expected_path}")
    _require(expected_path.is_absolute(), f"input path is not absolute: {expected_path}")
    _require(expected_path.is_file(), f"missing input file: {expected_path}")
    size = expected_path.stat().st_size
    if expected_size_bytes is not None:
        _require(size == expected_size_bytes, f"input size changed: {expected_path}")
    digest: str | None = None
    for record in already_validated:
        candidate = Path(str(record.get("path", "")))
        try:
            same_file = candidate.is_file() and os.path.samefile(candidate, expected_path)
        except OSError:
            same_file = False
        if same_file and record.get("sha256") == expected_sha256:
            digest = expected_sha256
            break
    if digest is None:
        digest = sha256_file(expected_path)
    _require(digest == expected_sha256, f"input SHA-256 changed: {expected_path}")
    observed = {"path": str(expected_path), "size_bytes": size, "sha256": digest}
    _require(declared == observed, f"declared input record does not match content: {expected_path}")
    return observed


def _standalone_artifact_paths(profile: StandaloneReplacementProfile) -> dict[str, Path]:
    paths = {
        name: profile.run_root / relative.format(job_id=profile.job_id)
        for name, relative in profile.artifact_relative_paths
    }
    paths.update(
        {
            "runtime_safe_to_delete": profile.runtime_root / "SAFE_TO_DELETE",
            "job_runtime_safe_to_delete": profile.runtime_root
            / f"recovar_full_{profile.job_id}"
            / "SAFE_TO_DELETE",
        }
    )
    return paths


def _standalone_subject(profile: StandaloneReplacementProfile) -> dict[str, Any]:
    repo = profile.source_repo
    _require_directory(repo, "standalone subject repository")
    _require(repo.is_relative_to(ALLOWED_ROOT), "standalone subject repository is outside allowed root")
    _require(_git_output(repo, "rev-parse", "HEAD") == profile.source_commit, "standalone subject commit changed")
    _require(_git_output(repo, "rev-parse", "HEAD^{tree}") == profile.source_tree, "standalone subject tree changed")
    _require(_git_output(repo, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD", "standalone subject is not detached")
    _require(
        not _git_output(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "standalone subject tree is dirty",
    )
    driver = repo / "scripts/run_full_refinement.py"
    cuda_source = repo / "recovar/cuda/cuda_backproject.cu"
    for path in (profile.python, driver, cuda_source):
        _require(path.is_absolute() and path.is_file(), f"standalone subject artifact is missing: {path}")
    _require(sha256_file(driver) == profile.driver_sha256, "standalone driver SHA-256 changed")
    _require(sha256_file(cuda_source) == profile.cuda_source_sha256, "standalone CUDA source SHA-256 changed")
    cuda_library = profile.run_root / profile.cuda_library_relative_path
    _require(cuda_library.is_file() and not cuda_library.is_symlink(), "standalone CUDA library is missing or a symlink")
    _require(cuda_library.stat().st_size == profile.cuda_postbuild_size_bytes, "postbuild CUDA library size changed")
    _require(sha256_file(cuda_library) == profile.cuda_postbuild_sha256, "postbuild CUDA library SHA-256 changed")
    return {
        "repo": str(repo),
        "commit": profile.source_commit,
        "tree": profile.source_tree,
        "tree_clean": True,
        "diff_sha256": EMPTY_SHA256,
        "python": str(profile.python),
        "driver": str(driver),
        "driver_sha256": profile.driver_sha256,
        "cuda_source": str(cuda_source),
        "cuda_source_sha256": profile.cuda_source_sha256,
        "cuda_library": str(cuda_library),
        "cuda_library_sha256": profile.cuda_postbuild_sha256,
        "cuda_preflight_sha256": profile.cuda_preflight_sha256,
        "cuda_postbuild_sha256": profile.cuda_postbuild_sha256,
        "standalone_replacement_profile": profile.name,
    }


def _expected_command(launcher: ModuleType, spec: Any, source: JobSource) -> tuple[str, ...]:
    output_dir = source.run_root / "outputs" / source.run_key
    if spec.engine == "recovar":
        return launcher._recovar_command(
            Path(source.subject["repo"]),
            spec.data_dir,
            output_dir,
            smoke=spec.phase == "smoke",
        )
    return launcher._relion_command(spec.data_dir, output_dir, smoke=spec.phase == "smoke")


def _validate_job(
    launcher: ModuleType,
    spec: Any,
    source: JobSource,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    job_id = source.job_id
    resources = dataclasses.asdict(spec.resources)
    _require_directory(source.run_root, f"{source.run_key} run root")
    _file_record(source.run_root / "SAFE_TO_DELETE", nonempty=False)
    _require(source.script == source.run_root / "scripts" / f"{source.run_key}.sbatch", "noncanonical job script")
    script_record = _file_record(source.script)
    _require(script_record["sha256"] == source.script_sha256, f"Slurm script SHA-256 changed for {source.run_key}")
    accounting = _accounting(job_id, runner=runner)
    pending_reason = _pending_reason(job_id, runner=runner) if accounting["state"] == "PENDING" else None

    submit_path = source.run_root / "provenance" / f"scontrol_submit_{job_id}.txt"
    submit_record: dict[str, Any] | None = None
    if submit_path.is_file():
        submit_record = _file_record(submit_path)
        submit_text = submit_path.read_text()
        _validate_scontrol_identity(
            submit_text,
            job_id=job_id,
            script=source.script,
            resources=resources,
            run_root=source.run_root,
        )
        _validate_requested_tres(submit_text, resources, source.run_key)
    else:
        _require(not source.submit_record_required, f"missing submission scontrol for {source.run_key}")

    started = accounting["state"] not in {"PENDING"}
    runtime_path = source.run_root / "provenance" / f"scontrol_{job_id}.txt"
    stdout_path = source.run_root / "logs" / f"{source.run_key}-{job_id}.out"
    stderr_path = source.run_root / "logs" / f"{source.run_key}-{job_id}.err"
    runtime_record: dict[str, Any] | None = None
    tres: dict[str, str] | None = None
    provenance: dict[str, Any] = {}
    stderr_text = ""
    if started:
        runtime_record = _file_record(runtime_path)
        runtime_text = runtime_path.read_text()
        stdout_path, stderr_path = _validate_scontrol_identity(
            runtime_text,
            job_id=job_id,
            script=source.script,
            resources=resources,
            run_root=source.run_root,
        )
        tres = _validate_runtime_tres(runtime_text, resources, source.run_key)
        _require(
            accounting["requested_tres"] == accounting["allocated_tres"],
            f"sacct ReqTRES != AllocTRES for {source.run_key}",
        )
        _require(
            _required_tres(resources).issubset(set(accounting["requested_tres"].split(","))), "sacct TRES mismatch"
        )
        stdout_record = _file_record(stdout_path, nonempty=False)
        stderr_record = _file_record(stderr_path, nonempty=False)
        stderr_text = stderr_path.read_text(errors="replace")
        command_path = source.run_root / "provenance" / f"command_{job_id}.sh"
        command_record = _file_record(command_path)
        command = shlex.split(command_path.read_text())
        expected_command = list(_expected_command(launcher, spec, source))
        _require(command == expected_command, f"executed command mismatch for {source.run_key}")
        subject_record = _validate_subject_record(
            source.run_root / "provenance" / f"subject_{job_id}.txt",
            source.subject,
        )
        environment = _validate_environment(
            source.run_root / "provenance" / f"environment_{job_id}.txt",
            source.subject,
            engine=spec.engine,
        )
        final_environment = _validate_final_environment(source.run_root / "provenance" / f"final_env_{job_id}.txt")
        gpu_inventory = _gpu_inventory(
            source.run_root / "provenance" / f"nvidia_smi_{job_id}.txt",
            int(resources["h100_gpus"]),
            source.run_key,
        )
        ldd_stem = "ldd_relion_bind" if spec.engine == "recovar" else "ldd_relion"
        ldd_path = source.run_root / "provenance" / f"{ldd_stem}_{job_id}.txt"
        ldd_record = _file_record(ldd_path)
        _require("not found" not in ldd_path.read_text(), f"unresolved native library for {source.run_key}")
        provenance = {
            "stdout_log": stdout_record,
            "stderr_log": stderr_record,
            "executed_command": command_record,
            "subject": subject_record,
            "environment": environment,
            "final_environment": final_environment,
            "gpu_inventory": gpu_inventory,
            "dynamic_libraries": ldd_record,
        }
    classification = _classify(accounting, stderr_text, pending_reason)
    outputs: list[dict[str, Any]] = []
    success_records: dict[str, Any] = {}
    if classification == "completed":
        output_dir = source.run_root / "outputs" / source.run_key
        expected_outputs = tuple(output_dir / path.relative_to(spec.output_dir) for path in spec.expected_outputs)
        outputs = [_file_record(path) for path in expected_outputs]
        success_records = {
            "completed_marker": _file_record(output_dir / "COMPLETED", nonempty=False),
            "walltime": _file_record(output_dir / "slurm_walltime.json"),
            "gpu_identity": _validate_success_gpu_identity(
                output_dir / "gpu_identity.txt",
                provenance["gpu_inventory"],
                source.run_key,
            ),
        }
        walltime = json.loads(Path(success_records["walltime"]["path"]).read_text())
        _require(str(walltime.get("job_id")) == job_id, f"walltime job mismatch for {source.run_key}")
        _require(walltime.get("run_key") == source.run_key, f"walltime run mismatch for {source.run_key}")
    provenance_complete = bool(
        submit_record is not None and (not started or runtime_record is not None) and (not started or provenance)
    )
    return {
        "run_key": source.run_key,
        "job_id": job_id,
        "engine": spec.engine,
        "phase": spec.phase,
        "source": "replacement" if source.replacement_record is not None else "submission",
        "classification": classification,
        "science_role": "science_candidate" if spec.phase == "full" else "capability_only",
        "smoke_can_promote": False,
        "provenance_complete": provenance_complete,
        "pending_reason": pending_reason,
        "accounting": accounting,
        "resources": resources,
        "tres": tres,
        "subject": dict(source.subject),
        "slurm_script": script_record,
        "scontrol_submit": submit_record,
        "scontrol_runtime": runtime_record,
        "provenance": provenance,
        "expected_outputs": outputs,
        "success_records": success_records,
        "replacement_record": source.replacement_record,
    }


def _standalone_tres(text: str, label: str) -> str:
    tokens = text.split(",")
    parsed: dict[str, str] = {}
    for token in tokens:
        _require("=" in token, f"malformed TRES for {label}")
        name, value = token.split("=", 1)
        _require(name not in parsed, f"duplicate TRES for {label}: {name}")
        parsed[name] = value
    _require(
        {name: value for name, value in parsed.items() if name != "billing"}
        == {"cpu": "4", "mem": "500G", "node": "1", "gres/gpu": "1"},
        f"wrong standalone TRES for {label}",
    )
    _require(set(parsed).issubset({"cpu", "mem", "node", "billing", "gres/gpu"}), f"extra TRES for {label}")
    return text


def _validate_standalone_scontrol(
    text: str,
    *,
    source: StandaloneJobSource,
    resources: Mapping[str, Any],
    runtime: bool,
) -> tuple[Path, Path, dict[str, str] | None]:
    stdout, stderr = _validate_scontrol_identity(
        text,
        job_id=source.job_id,
        script=source.script,
        resources=resources,
        run_root=source.run_root,
    )
    exact = {
        "JobId": source.job_id,
        "JobName": source.profile.job_name,
        "Account": "gilles",
        "QOS": "della-cryoem",
        "Partition": "cryoem",
        "Dependency": "(null)",
        "Restarts": "0",
        "OverSubscribe": "OK",
        "TimeLimit": "5-00:00:00",
    }
    for name, expected in exact.items():
        _require(_field(text, name) == expected, f"standalone scontrol {name} mismatch")
    requested = _standalone_tres(_field(text, "ReqTRES"), "scontrol request")
    if runtime:
        _require(_field(text, "NumNodes") == "1", "standalone runtime did not allocate exactly one node")
        allocated = _standalone_tres(_field(text, "AllocTRES"), "scontrol allocation")
        _require(requested == allocated, "standalone scontrol ReqTRES != AllocTRES")
        return stdout, stderr, {"requested": requested, "allocated": allocated}
    _require(_field(text, "AllocTRES") == "(null)", "standalone submission record already allocated resources")
    return stdout, stderr, None


def _environment_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", raw)
        if match is None:
            continue
        name, value = match.groups()
        _require(name not in assignments, f"duplicate environment variable {name}")
        assignments[name] = value
    return assignments


def _validate_standalone_environment(
    path: Path,
    source: StandaloneJobSource,
    native_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    record = source.artifacts["environment"]
    assignments = _environment_assignments(path)
    job_runtime = source.profile.runtime_root / f"recovar_full_{source.job_id}"
    relion_bind_dir = Path(str(native_artifacts["relion_binding"]["path"])).parent
    relion_source = Path(str(native_artifacts["relion_binding_source"]["path"]))
    expected_sensitive = {
        "JAX_COMPILATION_CACHE_DIR": str(job_runtime / "jax_cache"),
        "RECOVAR_CUDA_LIB": str(source.subject["cuda_library"]),
        "RECOVAR_EXACT_LOCAL_BIG_JIT_MAX_BUCKET_ROTATIONS": "256",
        "RECOVAR_RELION_BIND_BUILD_DIR": str(relion_bind_dir),
        "RELION_SRC_DIR": str(relion_source / "src"),
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    observed_sensitive = {
        name: value
        for name, value in assignments.items()
        if name.startswith(("JAX_", "RECOVAR_", "RELION_", "XLA_"))
    }
    _require(observed_sensitive == expected_sensitive, "standalone sensitive environment changed")
    expected_runtime = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(job_runtime / "tmp"),
        "PIXI_HOME": str(job_runtime / "pixi_home"),
        "RATTLER_CACHE_DIR": str(job_runtime / "rattler_cache"),
    }
    for name, expected in expected_runtime.items():
        _require(assignments.get(name) == expected, f"standalone environment {name} changed")
    _require(
        not ({"PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"} & set(assignments)),
        "standalone environment retained an unsealed Python environment",
    )
    return dict(record)


def _validate_standalone_subject_record(path: Path, source: StandaloneJobSource) -> dict[str, Any]:
    expected = {
        "source_repo": str(source.profile.source_repo),
        "source_commit": source.profile.source_commit,
        "source_tree": source.profile.source_tree,
        "allocator": "platform",
        "preallocate": "false",
        "big_jit_max_bucket_rotations": "256",
        "unused_native_projection_padding": "skipped_when_relion_projector_supplied",
        "symmetry": "I1",
        "particle_count": str(source.profile.particle_count),
        "half1_count": str(source.profile.half1_count),
        "half2_count": str(source.profile.half2_count),
    }
    _require(_parse_key_values(path, tuple(expected)) == expected, "standalone subject record changed")
    return dict(source.artifacts["subject"])


def _validate_standalone_import(path: Path, source: StandaloneJobSource) -> dict[str, Any]:
    rows = path.read_text().splitlines()
    _require(len(rows) == 3, "standalone import provenance fields changed")
    values = dict(row.split("=", 1) for row in rows if "=" in row)
    _require(set(values) == {"recovar", "jax", "devices"}, "standalone import provenance fields changed")
    _require(
        Path(values["recovar"]) == source.profile.source_repo / "recovar/__init__.py",
        "standalone RECOVAR import escaped the sealed source",
    )
    python_root = source.profile.python.parent.parent
    jax_path = Path(values["jax"])
    _require(jax_path.is_relative_to(python_root), "standalone JAX import escaped the selected Python environment")
    _require(jax_path.as_posix().endswith("/site-packages/jax/__init__.py"), "standalone JAX import path changed")
    _require(values["devices"] == "[CudaDevice(id=0)]", "standalone import did not see exactly GPU 0")
    return dict(source.artifacts["import"])


def _plain_key_values(path: Path, expected: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        _require("=" in raw, f"malformed key/value at {path}:{line_number}")
        name, value = raw.split("=", 1)
        _require(name and name not in values, f"duplicate or empty key at {path}:{line_number}")
        values[name] = value
    _require(set(values) == set(expected), f"unexpected key set in {path}: {sorted(values)}")
    return values


def _validate_standalone_cuda_chain(source: StandaloneJobSource) -> dict[str, Any]:
    profile = source.profile
    library = Path(str(source.subject["cuda_library"]))
    ledger_path = Path(str(source.artifacts["preflight_ledger"]["path"]))
    ledger_rows: list[tuple[str, Path]] = []
    for raw in ledger_path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", raw)
        _require(match is not None, "malformed standalone preflight digest ledger")
        ledger_rows.append((match.group(1), Path(match.group(2))))
    _require(
        ledger_rows
        == [
            (profile.script_sha256, source.script),
            (profile.preflight_readme_sha256, profile.run_root / "README.md"),
            (profile.cuda_preflight_sha256, library),
        ],
        "standalone preflight digest chain changed",
    )
    postbuild_path = Path(str(source.artifacts["cuda_postbuild"]["path"]))
    postbuild = _plain_key_values(
        postbuild_path,
        (
            "job_id",
            "path",
            "preflight_sha256",
            "postbuild_sha256",
            "postbuild_size_bytes",
            "postbuild_mtime",
            "source_cuda_sha256",
            "reason",
        ),
    )
    exact = {
        "job_id": profile.job_id,
        "path": str(library),
        "preflight_sha256": profile.cuda_preflight_sha256,
        "postbuild_sha256": profile.cuda_postbuild_sha256,
        "postbuild_size_bytes": str(profile.cuda_postbuild_size_bytes),
        "source_cuda_sha256": profile.cuda_source_sha256,
        "reason": profile.postbuild_reason,
    }
    for name, expected in exact.items():
        _require(postbuild.get(name) == expected, f"standalone CUDA postbuild {name} changed")
    _require(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}", postbuild["postbuild_mtime"])
        is not None,
        "standalone CUDA postbuild mtime is invalid",
    )
    _require(library.stat().st_size == profile.cuda_postbuild_size_bytes, "standalone CUDA postbuild size changed")
    _require(sha256_file(library) == profile.cuda_postbuild_sha256, "standalone CUDA postbuild digest changed")
    return {
        "preflight_sha256": profile.cuda_preflight_sha256,
        "postbuild_sha256": profile.cuda_postbuild_sha256,
        "postbuild_size_bytes": profile.cuda_postbuild_size_bytes,
        "preflight_ledger": dict(source.artifacts["preflight_ledger"]),
        "postbuild_record": dict(source.artifacts["cuda_postbuild"]),
        "reason": profile.postbuild_reason,
    }


def _validate_standalone_hbm(
    path: Path,
    summary_path: Path,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = [
        "timestamp",
        "index",
        "uuid",
        "memory_used_mib",
        "memory_free_mib",
        "gpu_utilization_percent",
    ]
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == expected_fields, "standalone HBM trace header changed")
        rows = list(reader)
    _require(len(rows) >= 2, "standalone HBM trace is incomplete")
    used_values: list[int] = []
    for row in rows:
        _require(set(row) == set(expected_fields), "standalone HBM trace row changed")
        _require(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}", row["timestamp"] or "")
            is not None,
            "standalone HBM timestamp is invalid",
        )
        _require(row["index"] == "0" and row["uuid"] == inventory["uuids"][0], "standalone HBM GPU identity changed")
        try:
            used = int(row["memory_used_mib"] or "")
            free = int(row["memory_free_mib"] or "")
            utilization = int(row["gpu_utilization_percent"] or "")
        except ValueError as exc:
            raise ValueError("standalone HBM trace contains a non-integer sample") from exc
        _require(used >= 0 and free >= 0 and 0 <= utilization <= 100, "standalone HBM sample is out of range")
        used_values.append(used)
    peak = max(used_values)
    _require(peak > 0, "standalone HBM trace never recorded GPU memory use")
    _require(summary_path.read_text() == f"peak_hbm_mib={peak}\n", "standalone HBM summary changed")
    return {"samples": len(rows), "gpu_uuid": inventory["uuids"][0], "peak_hbm_mib": peak}


def _npz_scalar(archive: Any, name: str) -> Any:
    _require(name in archive.files, f"standalone refinement result lacks {name}")
    array = np.asarray(archive[name])
    _require(array.size == 1, f"standalone refinement result {name} is not scalar")
    return array.reshape(()).item()


def _validate_standalone_refinement(path: Path, source: StandaloneJobSource) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        n_iterations = int(_npz_scalar(archive, "n_iterations"))
        convergence_iteration = int(_npz_scalar(archive, "convergence_iteration"))
        converged = bool(_npz_scalar(archive, "convergence_has_converged"))
        final_all_data = bool(_npz_scalar(archive, "final_all_data_ran"))
        grid_correct = bool(_npz_scalar(archive, "final_all_data_grid_correct"))
        current_sizes = np.asarray(archive["current_sizes"], dtype=np.int64).reshape(-1)
        _require(n_iterations == 50, "standalone refinement iteration cap changed")
        _require(converged, "standalone refinement did not converge naturally")
        _require(1 <= convergence_iteration < n_iterations, "standalone refinement converged only at/after the cap")
        _require(
            current_sizes.size == convergence_iteration and current_sizes.size > 0,
            "standalone convergence iteration does not match its numbered trajectory",
        )
        _require(final_all_data, "standalone converged refinement lacks final all-data")
        _require(not grid_correct, "standalone final all-data enabled grid correction")
        final_fsc = np.asarray(archive["fsc_final_all_data"], dtype=np.float64).reshape(-1)
        _require(final_fsc.size > 1 and np.all(np.isfinite(final_fsc)), "standalone final FSC is empty or non-finite")

        _require(int(_npz_scalar(archive, "n_images")) == source.profile.particle_count, "standalone particle count changed")
        half1 = np.asarray(archive["half1_indices"], dtype=np.int64).reshape(-1)
        half2 = np.asarray(archive["half2_indices"], dtype=np.int64).reshape(-1)
        _require(half1.size == source.profile.half1_count, "standalone half-1 count changed")
        _require(half2.size == source.profile.half2_count, "standalone half-2 count changed")
        combined = np.concatenate((half1, half2))
        _require(
            np.array_equal(np.sort(combined), np.arange(source.profile.particle_count, dtype=np.int64)),
            "standalone half sets are not a disjoint full partition",
        )
        _require(_npz_scalar(archive, "symmetry_label") == "I1", "standalone symmetry label changed")
        _require(_npz_scalar(archive, "symmetry_family") == "icosahedral", "standalone symmetry family changed")
        _require(int(_npz_scalar(archive, "symmetry_operator_count")) == 60, "standalone symmetry operator count changed")
        _require(
            _npz_scalar(archive, "symmetry_operator_sha256")
            == "093a0876b93610ec141c87840ae3ff4dc4491b27dec87143558358ef556557b8",
            "standalone symmetry operator digest changed",
        )
        _require(bool(_npz_scalar(archive, "firstiter_cc_effective")), "standalone firstiter_cc was not effective")
        _require(float(_npz_scalar(archive, "tau2_fudge")) == 1.0, "standalone tau2 fudge changed")
        _require(_npz_scalar(archive, "tau2_fudge_source") == "explicit CLI", "standalone tau2 source changed")
        _require(_npz_scalar(archive, "initial_pose_source_requested") == "input-star", "standalone pose request changed")
        _require(_npz_scalar(archive, "initial_pose_source_resolved") == "input_star", "standalone pose source changed")
        _require(
            Path(str(_npz_scalar(archive, "initial_pose_source_path"))).resolve()
            == (source.data_dir / "particles.star").resolve(),
            "standalone pose source path changed",
        )
        _require(
            _npz_scalar(archive, "initial_pose_source_sha256") == FULL_STAR_SHA256,
            "standalone pose source digest changed",
        )
        _require(_npz_scalar(archive, "git_commit") == source.profile.source_commit, "standalone result commit changed")
        _require(_npz_scalar(archive, "git_branch") == "<detached>", "standalone result was not produced detached")
        _require(int(_npz_scalar(archive, "git_dirty_count")) == 0, "standalone result source was dirty")
        _require(_npz_scalar(archive, "git_diff_sha256") == EMPTY_SHA256, "standalone result diff digest changed")
        _require(_npz_scalar(archive, "git_status_porcelain") == "", "standalone result recorded a dirty status")

        empty_arrays = (
            "perturb_replay_restart_state_iterations",
            "diagnostic_final_manifest_paths",
            "diagnostic_final_manifest_sha256",
            "state_swap_probe_applied_relion_iterations",
            "state_swap_probe_replay_override_keys",
            "state_swap_probe_required_replay_override_keys",
        )
        for name in empty_arrays:
            _require(name in archive.files and np.asarray(archive[name]).size == 0, f"standalone replay field {name} is active")
        empty_scalars = (
            "perturb_replay_restart_provenance_path",
            "perturb_replay_restart_provenance_sha256",
            "relion_projector_source_manifest_sha256",
            "relion_projector_capture_dir",
            "relion_projector_capture_manifest",
            "frozen_boundary_dir",
            "frozen_boundary_manifest_sha256",
            "frozen_boundary_sha256",
            "diagnostic_final_source_results_path",
            "diagnostic_final_source_results_sha256",
            "diagnostic_final_source_git_commit",
            "state_swap_probe_variant",
        )
        for name in empty_scalars:
            _require(_npz_scalar(archive, name) == "", f"standalone replay field {name} is active")
        for name in (
            "relion_projector_replay_slot",
            "frozen_boundary_completed_relion_iteration",
            "diagnostic_final_source_completed_relion_iteration",
            "state_swap_probe_target_relion_iteration",
            "state_swap_probe_loop_index",
        ):
            _require(int(_npz_scalar(archive, name)) == -1, f"standalone replay field {name} is active")
    return {
        "n_iterations_cap": n_iterations,
        "numbered_iterations": convergence_iteration,
        "converged": converged,
        "final_all_data_ran": final_all_data,
        "final_all_data_grid_correct": grid_correct,
        "final_fsc_shells": int(final_fsc.size),
    }


def _validate_standalone_output_ledger(source: StandaloneJobSource) -> None:
    output_dir = source.run_root / "outputs/recovar_full"
    expected_paths = [
        output_dir / "refinement_results.npz",
        output_dir / "final_merged.mrc",
        output_dir / "final_half1_unfil.mrc",
        output_dir / "final_half2_unfil.mrc",
        output_dir / "slurm_walltime.json",
    ]
    rows: list[tuple[str, Path]] = []
    ledger = Path(str(source.artifacts["output_sha256"]["path"]))
    for raw in ledger.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", raw)
        _require(match is not None, "malformed standalone output digest ledger")
        rows.append((match.group(1), Path(match.group(2))))
    _require([path for _, path in rows] == expected_paths, "standalone output digest ledger path set changed")
    _require(all(digest == sha256_file(path) for digest, path in rows), "standalone output digest ledger changed")


def _validate_standalone_job(
    launcher: ModuleType,
    spec: Any,
    source: StandaloneJobSource,
    *,
    native_artifacts: Mapping[str, Mapping[str, Any]],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    _require(spec.engine == "recovar" and spec.phase == "full", "standalone profile bound to wrong run spec")
    resources = dataclasses.asdict(spec.resources)
    _require(resources == {"nodes": 1, "ntasks": 1, "cpus_per_task": 4, "memory": "500G", "h100_gpus": 1, "walltime": "5-00:00:00"}, "standalone resource contract changed")
    _require("--exclusive" not in source.script.read_text(), "standalone script requests an exclusive allocation")

    submit_text = Path(str(source.artifacts["scontrol_submit"]["path"])).read_text()
    submit_stdout, submit_stderr, _ = _validate_standalone_scontrol(
        submit_text,
        source=source,
        resources=resources,
        runtime=False,
    )
    runtime_text = Path(str(source.artifacts["scontrol_runtime"]["path"])).read_text()
    stdout_path, stderr_path, tres = _validate_standalone_scontrol(
        runtime_text,
        source=source,
        resources=resources,
        runtime=True,
    )
    _require(submit_stdout == stdout_path and submit_stderr == stderr_path, "standalone Slurm log paths changed after submit")

    accounting = _accounting(source.job_id, runner=runner)
    _require(accounting["state"] == "COMPLETED" and accounting["exit_code"] == "0:0", "standalone job did not complete cleanly")
    requested = _standalone_tres(accounting["requested_tres"], "sacct request")
    allocated = _standalone_tres(accounting["allocated_tres"], "sacct allocation")
    _require(requested == allocated == tres["requested"], "standalone sacct/scontrol TRES differ")

    command_path = Path(str(source.artifacts["executed_command"]["path"]))
    command = shlex.split(command_path.read_text())
    expected_command = list(
        launcher._recovar_command(
            source.profile.source_repo,
            source.data_dir,
            source.run_root / "outputs/recovar_full",
            smoke=False,
        )
    )
    expected_command[4] = str(source.profile.python)
    expected_command.extend(
        [
            "--save_intermediates_dir",
            str(source.run_root / "outputs/intermediates"),
            "--save_intermediates_skip_unregularized",
        ]
    )
    _require(command == expected_command, "standalone executed command changed")

    subject_record = _validate_standalone_subject_record(
        Path(str(source.artifacts["subject"]["path"])),
        source,
    )
    environment = _validate_standalone_environment(
        Path(str(source.artifacts["environment"]["path"])),
        source,
        native_artifacts,
    )
    import_record = _validate_standalone_import(Path(str(source.artifacts["import"]["path"])), source)
    inventory = _gpu_inventory(
        Path(str(source.artifacts["gpu_inventory"]["path"])),
        1,
        "standalone recovar_full",
    )
    for name in ("cuda_pre_submit_libraries", "cuda_runtime_libraries", "relion_bind_libraries"):
        path = Path(str(source.artifacts[name]["path"]))
        _require("not found" not in path.read_text(), f"unresolved standalone native library in {name}")
    cuda_rebuild = _validate_standalone_cuda_chain(source)

    hbm = _validate_standalone_hbm(
        Path(str(source.artifacts["hbm_trace"]["path"])),
        Path(str(source.artifacts["hbm_summary"]["path"])),
        inventory,
    )
    walltime_path = Path(str(source.artifacts["walltime"]["path"]))
    walltime = json.loads(walltime_path.read_text())
    _require(
        set(walltime) == {"schema", "job_id", "start_epoch", "end_epoch", "wall_s"}
        and walltime["schema"] == "recovar.em.walltime.v1"
        and str(walltime["job_id"]) == source.job_id,
        "standalone walltime identity changed",
    )
    for name in ("start_epoch", "end_epoch", "wall_s"):
        _require(type(walltime[name]) is int, f"standalone walltime {name} is not an integer")
    _require(
        walltime["end_epoch"] > walltime["start_epoch"]
        and walltime["wall_s"] == walltime["end_epoch"] - walltime["start_epoch"],
        "standalone walltime arithmetic changed",
    )
    _require(
        re.search(r"^\s*Exit status:\s*0\s*$", Path(str(source.artifacts["time_verbose"]["path"])).read_text(), re.MULTILINE)
        is not None,
        "standalone time -v record lacks exit status 0",
    )
    _validate_standalone_output_ledger(source)
    refinement = _validate_standalone_refinement(
        Path(str(source.artifacts["refinement_results"]["path"])),
        source,
    )

    stdout_text = stdout_path.read_text(errors="replace")
    stderr_text = stderr_path.read_text(errors="replace")
    combined_log = stdout_text + "\n" + stderr_text
    cuda_library = str(source.subject["cuda_library"])
    log_requirements = (
        f"{cuda_library} is older than its source",
        f"Building {cuda_library}",
        f"-o {cuda_library} cuda_backproject.cu",
        "CUDA backproject/project kernels enabled",
        f"Convergence reached at iteration {refinement['numbered_iterations']}.",
        "=== RELION final all-data Nyquist iteration ===",
        "Final iter complete:",
    )
    _require(all(text in combined_log for text in log_requirements), "standalone completion/CUDA log chain is incomplete")
    _require(not any(pattern.search(combined_log) for pattern in OOM_PATTERNS), "standalone completion log contains OOM evidence")
    _require(
        "force_max_iter_after_convergence=True" not in combined_log
        and "running RELION final all-data iteration after max_iter exhaustion" not in combined_log,
        "standalone final all-data was forced",
    )

    expected_outputs = [
        dict(source.artifacts[name])
        for name in ("refinement_results", "final_merged", "final_half1_unfil", "final_half2_unfil")
    ]
    provenance = {
        "stdout_log": dict(source.artifacts["stdout_log"]),
        "stderr_log": dict(source.artifacts["stderr_log"]),
        "executed_command": dict(source.artifacts["executed_command"]),
        "subject": subject_record,
        "environment": environment,
        "import": import_record,
        "gpu_inventory": inventory,
        "cuda_pre_submit_libraries": dict(source.artifacts["cuda_pre_submit_libraries"]),
        "cuda_runtime_libraries": dict(source.artifacts["cuda_runtime_libraries"]),
        "relion_bind_libraries": dict(source.artifacts["relion_bind_libraries"]),
        "hbm_trace": dict(source.artifacts["hbm_trace"]),
        "time_verbose": dict(source.artifacts["time_verbose"]),
    }
    return {
        "run_key": source.run_key,
        "job_id": source.job_id,
        "engine": "recovar",
        "phase": "full",
        "source": "standalone_replacement",
        "classification": "completed",
        "science_role": "science_candidate",
        "smoke_can_promote": False,
        "provenance_complete": True,
        "pending_reason": None,
        "accounting": accounting,
        "resources": resources,
        "tres": tres,
        "subject": dict(source.subject),
        "slurm_script": dict(source.replacement_record["payload"]["script"]),
        "scontrol_submit": dict(source.artifacts["scontrol_submit"]),
        "scontrol_runtime": dict(source.artifacts["scontrol_runtime"]),
        "provenance": provenance,
        "expected_outputs": expected_outputs,
        "success_records": {
            "completed_marker": dict(source.artifacts["completed_marker"]),
            "walltime": dict(source.artifacts["walltime"]),
            "hbm_summary": dict(source.artifacts["hbm_summary"]),
            "output_sha256": dict(source.artifacts["output_sha256"]),
        },
        "replacement_profile": source.profile.name,
        "cuda_rebuild": cuda_rebuild,
        "hbm": hbm,
        "refinement": refinement,
        "replacement_record": source.replacement_record,
    }


def _validate_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], Path, Path, dict[str, Any], dict[str, Any], ModuleType]:
    _require(manifest_path.is_absolute(), "launch manifest path must be absolute")
    _require(str(manifest_path) == os.path.normpath(str(manifest_path)), "launch manifest path is not canonical")
    _require(manifest_path.name == "launch_manifest.json", "noncanonical launch manifest name")
    root = manifest_path.parent
    _require(root.parent == ALLOWED_ROOT, f"run root is outside {ALLOWED_ROOT}: {root}")
    payload = _load_json(manifest_path)
    expected_keys = {
        "schema",
        "created_utc",
        "status",
        "run_root",
        "runtime_root",
        "safe_to_delete",
        "source_harness",
        "subject",
        "harness_delta_from_r3",
        "inputs",
        "immutable_native_artifacts",
        "runs",
        "harness_scripts",
        "submission",
    }
    _require(set(payload) == expected_keys, "native launch manifest fields changed")
    _require(payload.get("schema") == NATIVE_LAUNCH_SCHEMA, "wrong native launch schema")
    _require(
        isinstance(payload.get("created_utc"), str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", payload["created_utc"]) is not None,
        "invalid native launch timestamp",
    )
    _require(isinstance(payload.get("status"), str) and payload["status"], "missing native launch status")
    _require(payload.get("run_root") == str(root), "native launch run root changed")
    runtime_root = Path(payload.get("runtime_root", ""))
    _require(runtime_root == ALLOWED_ROOT / "runtime" / root.name, "native runtime root changed")
    for directory, label in (
        (root, "run root"),
        (runtime_root, "runtime root"),
        (root / "scripts", "scripts"),
        (root / "logs", "logs"),
        (root / "provenance", "provenance"),
        (root / "inputs", "inputs"),
        (root / "outputs", "outputs"),
    ):
        _require_directory(directory, label)
    markers = payload.get("safe_to_delete", {})
    _require(
        markers == {"run_marker": str(root / "SAFE_TO_DELETE"), "runtime_marker": str(runtime_root / "SAFE_TO_DELETE")},
        "SAFE_TO_DELETE ledger changed",
    )
    _file_record(root / "SAFE_TO_DELETE", nonempty=False)
    _file_record(runtime_root / "SAFE_TO_DELETE", nonempty=False)
    _require(payload.get("submission") is None, "native manifest submission field must remain immutable null")
    source_harness = payload.get("source_harness", {})
    _require(
        set(source_harness)
        == {
            "run_root",
            "launch_manifest",
            "launch_manifest_sha256",
            "recovar_smoke_job_id",
            "relion_smoke_job_id",
            "recovar_full_job_id",
            "relion_full_job_id",
        },
        "source harness fields changed",
    )
    source_root = Path(source_harness.get("run_root", ""))
    source_manifest = Path(source_harness.get("launch_manifest", ""))
    _require(source_manifest == source_root / "launch_manifest.json", "source harness manifest path changed")
    source_record = _file_record(source_manifest)
    _require(
        source_record["sha256"] == source_harness.get("launch_manifest_sha256"),
        "source harness manifest SHA-256 changed",
    )

    subject_spec = payload.get("subject", {})
    subject_path = root / "config/subject.env"
    _require(subject_spec.get("configuration") == str(subject_path), "subject configuration path changed")
    _require(subject_spec.get("required_fields") == list(SUBJECT_FIELDS), "subject required fields changed")
    subject_config_record = _file_record(subject_path)
    subject = _subject_from_values(_parse_key_values(subject_path, SUBJECT_FIELDS))

    inputs = payload.get("inputs", {})
    expected_input_hashes = {
        "preparation_manifest": PREPARATION_SHA256,
        "full_particles_star": FULL_STAR_SHA256,
        "smoke_particles_star": SMOKE_STAR_SHA256,
        "recovar_reference": RECOVAR_REFERENCE_SHA256,
        "relion_reference": RELION_REFERENCE_SHA256,
    }
    input_records: dict[str, Any] = {}
    for name, expected_hash in expected_input_hashes.items():
        row = inputs.get(name, {})
        path = Path(row.get("path", row.get("canonical_path", "")))
        _require(path.is_absolute() and path.is_file(), f"missing native input {name}")
        _require(row.get("sha256") == expected_hash, f"declared {name} SHA-256 changed")
        _require(sha256_file(path) == expected_hash, f"native input {name} SHA-256 changed")
        input_records[name] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": expected_hash}
    for phase, star_hash in (("smoke", SMOKE_STAR_SHA256), ("full", FULL_STAR_SHA256)):
        command_inputs = {
            root / "inputs" / phase / "particles.star": star_hash,
            root / "inputs" / phase / "reference_init.mrc": RECOVAR_REFERENCE_SHA256,
            root / "inputs" / phase / "reference_init_relion.mrc": RELION_REFERENCE_SHA256,
        }
        for path, expected_hash in command_inputs.items():
            _require(path.is_file(), f"missing command input: {path}")
            _require(sha256_file(path) == expected_hash, f"command input SHA-256 changed: {path}")
    stack = inputs.get("particle_stack", {})
    stack_path = Path(stack.get("path", ""))
    _require(stack.get("size_bytes") == PARTICLE_STACK_SIZE_BYTES, "particle stack declared size changed")
    _require(stack.get("full_sha256") == PARTICLE_STACK_SHA256, "particle stack declared SHA-256 changed")
    _require(
        stack_path.is_file() and stack_path.stat().st_size == PARTICLE_STACK_SIZE_BYTES, "particle stack size changed"
    )
    _require(sha256_file(stack_path) == PARTICLE_STACK_SHA256, "particle stack SHA-256 changed")
    input_records["particle_stack"] = {
        "path": str(stack_path),
        "size_bytes": stack_path.stat().st_size,
        "sha256": PARTICLE_STACK_SHA256,
    }

    native = payload.get("immutable_native_artifacts", {})
    expected_native = {
        "relion_refine_mpi": RELION_MPI_SHA256,
        "relion_binding": RELION_BIND_SHA256,
    }
    native_records: dict[str, Any] = {}
    for name, expected_hash in expected_native.items():
        row = native.get(name, {})
        path = Path(row.get("path", ""))
        _require(row.get("sha256") == expected_hash, f"declared {name} SHA-256 changed")
        record = _file_record(path)
        _require(record["sha256"] == expected_hash, f"{name} SHA-256 changed")
        native_records[name] = record
    source = native.get("relion_binding_source", {})
    source_path = Path(source.get("path", ""))
    _require(
        source.get("commit") == RELION_SOURCE_COMMIT and source.get("tree") == RELION_SOURCE_TREE,
        "RELION source identity changed",
    )
    _require(_git_output(source_path, "rev-parse", "HEAD") == RELION_SOURCE_COMMIT, "RELION source commit changed")
    _require(_git_output(source_path, "rev-parse", "HEAD^{tree}") == RELION_SOURCE_TREE, "RELION source tree changed")
    _require(
        not _git_output(source_path, "status", "--porcelain=v1", "--untracked-files=all"), "RELION source is dirty"
    )
    native_records["relion_binding_source"] = dict(source, clean=True)

    for name, row in payload.get("harness_scripts", {}).items():
        path = Path(row.get("path", ""))
        _require(path.parent == (root / "scripts" if name != "submit" else root), f"noncanonical harness script {name}")
        _require(sha256_file(path) == row.get("sha256"), f"harness script SHA-256 changed: {name}")
    delta = payload.get("harness_delta_from_r3", {})
    _require(delta.get("scientific_command_delta", "").startswith("none"), "native harness changed science commands")
    _require(delta.get("resource_delta") == "none", "native harness changed resources")
    _require(
        set(delta.get("command_equivalence_gate", {}).values()) == {"passed"}, "native command-equivalence gate failed"
    )

    submission_path = root / "submission.json"
    submission = _load_json(submission_path)
    _require(set(submission) == {"schema", "dependency", "job_ids"}, "native submission fields changed")
    _require(submission.get("schema") == NATIVE_SUBMISSION_SCHEMA, "wrong native submission schema")
    _require(set(submission.get("job_ids", {})) == set(RUN_KEYS), "native submission job set changed")
    job_ids = {key: str(submission["job_ids"][key]) for key in RUN_KEYS}
    _require(all(re.fullmatch(r"[0-9]+", value) for value in job_ids.values()), "invalid native job ID")
    _require(len(set(job_ids.values())) == len(RUN_KEYS), "duplicate native job ID")
    expected_dependency = f"afterok:{job_ids['recovar_smoke']}:{job_ids['relion_smoke']}"
    _require(submission.get("dependency") == expected_dependency, "native dependency ledger changed")

    launcher = _load_launcher()
    specs = {spec.key: spec for spec in launcher.build_run_specs(Path(subject["repo"]), root)}
    _require(tuple(specs) == RUN_KEYS, "matched launcher run order changed")
    expected_science = {
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
    _require(set(payload.get("runs", {})) == set(RUN_KEYS), "native run set changed")
    for key, spec in specs.items():
        row = payload["runs"][key]
        expected_keys = {"script", "script_sha256", "resources", "science"}
        if spec.depends_on:
            expected_keys.add("depends_on")
        _require(set(row) == expected_keys, f"native run fields changed for {key}")
        script = root / "scripts" / f"{key}.sbatch"
        _require(row["script"] == str(script), f"native script path changed for {key}")
        _require(sha256_file(script) == row["script_sha256"], f"native script digest changed for {key}")
        _require(row["resources"] == dataclasses.asdict(spec.resources), f"native resources changed for {key}")
        _require(row["science"] == expected_science[key], f"native science contract changed for {key}")
        _require(row.get("depends_on", []) == list(spec.depends_on), f"native dependencies changed for {key}")
    metadata = {
        "subject_config": subject_config_record,
        "source_harness_manifest": source_record,
        "inputs": input_records,
        "native_artifacts": native_records,
    }
    return payload, root, runtime_root, submission, {"subject": subject, "metadata": metadata, "specs": specs}, launcher


def _standalone_replacement_source(
    path: Path,
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    submission: Mapping[str, Any],
    parent_inputs: Mapping[str, Mapping[str, Any]],
) -> StandaloneJobSource:
    expected_fields = {
        "schema",
        "profile",
        "parent_launch_manifest",
        "run_key",
        "replaces_job_id",
        "job_id",
        "run_root",
        "script",
        "source",
        "inputs",
        "cuda_rebuild",
        "artifacts",
        "reason",
    }
    _require(set(payload) == expected_fields, f"standalone replacement record fields changed: {path}")
    _require(payload.get("schema") == STANDALONE_REPLACEMENT_SCHEMA, "wrong standalone replacement schema")
    profile_name = payload.get("profile")
    _require(isinstance(profile_name, str), "standalone replacement profile is missing")
    profile = STANDALONE_REPLACEMENT_PROFILES.get(profile_name)
    _require(profile is not None and profile.name == profile_name, "standalone replacement profile is not accepted")
    _require(
        payload.get("parent_launch_manifest")
        == {"path": str(manifest_path), "sha256": manifest_sha256},
        "standalone replacement parent binding changed",
    )
    _require(payload.get("run_key") == "recovar_full", "standalone replacement may replace only recovar_full")
    original_id = str(submission["job_ids"]["recovar_full"])
    _require(str(payload.get("replaces_job_id")) == original_id, "standalone replacement names wrong original job")
    _require(str(payload.get("job_id")) == profile.job_id, "standalone replacement job ID changed")
    _require(payload.get("run_root") == str(profile.run_root), "standalone replacement run root changed")
    _require(
        isinstance(payload.get("reason"), str) and bool(str(payload["reason"]).strip()),
        "standalone replacement reason is missing",
    )
    _require_directory(profile.run_root, "standalone replacement run root")
    _require(profile.run_root.parent == ALLOWED_ROOT, "standalone replacement run root is outside allowed root")
    _require_directory(profile.runtime_root, "standalone replacement runtime root")
    _require(profile.runtime_root.parent == ALLOWED_ROOT / "runtime", "standalone runtime root is outside allowed root")

    script = profile.run_root / "jobs/recovar_full.sbatch"
    script_record = _validate_declared_file_record(payload.get("script"), script)
    _require(script_record["sha256"] == profile.script_sha256, "standalone Slurm script SHA-256 changed")

    declared_source = payload.get("source")
    expected_source = {
        "repo": str(profile.source_repo),
        "commit": profile.source_commit,
        "tree": profile.source_tree,
        "python": str(profile.python),
        "driver_sha256": profile.driver_sha256,
        "cuda_source_sha256": profile.cuda_source_sha256,
    }
    _require(declared_source == expected_source, "standalone source declaration changed")
    subject = _standalone_subject(profile)

    declared_inputs = payload.get("inputs")
    _require(isinstance(declared_inputs, dict), "standalone input declaration is missing")
    _require(
        set(declared_inputs)
        == {"data_dir", "particles_star", "recovar_reference", "relion_reference", "particle_stack"},
        "standalone input fields changed",
    )
    data_dir = Path(str(declared_inputs["data_dir"]))
    _require(data_dir.is_absolute() and str(data_dir) == os.path.normpath(str(data_dir)), "standalone data dir is not canonical")
    _require(data_dir.is_dir(), f"standalone data dir is missing: {data_dir}")
    parent_rows = tuple(parent_inputs.values())
    input_records = {
        "particles_star": _input_file_record(
            declared_inputs["particles_star"],
            data_dir / "particles.star",
            expected_sha256=FULL_STAR_SHA256,
            already_validated=parent_rows,
        ),
        "recovar_reference": _input_file_record(
            declared_inputs["recovar_reference"],
            data_dir / "reference_init.mrc",
            expected_sha256=RECOVAR_REFERENCE_SHA256,
            already_validated=parent_rows,
        ),
        "relion_reference": _input_file_record(
            declared_inputs["relion_reference"],
            data_dir / "reference_init_relion.mrc",
            expected_sha256=RELION_REFERENCE_SHA256,
            already_validated=parent_rows,
        ),
    }
    stack_path = Path(str(declared_inputs.get("particle_stack", {}).get("path", "")))
    input_records["particle_stack"] = _input_file_record(
        declared_inputs["particle_stack"],
        stack_path,
        expected_sha256=PARTICLE_STACK_SHA256,
        expected_size_bytes=PARTICLE_STACK_SIZE_BYTES,
        already_validated=parent_rows,
    )

    cuda_library = profile.run_root / profile.cuda_library_relative_path
    expected_cuda_rebuild = {
        "library": {
            "path": str(cuda_library),
            "preflight_sha256": profile.cuda_preflight_sha256,
            "postbuild_sha256": profile.cuda_postbuild_sha256,
            "postbuild_size_bytes": profile.cuda_postbuild_size_bytes,
        },
        "preflight_ledger_sha256": profile.preflight_ledger_sha256,
        "postbuild_record_sha256": profile.postbuild_record_sha256,
        "reason": profile.postbuild_reason,
    }
    _require(payload.get("cuda_rebuild") == expected_cuda_rebuild, "standalone CUDA rebuild declaration changed")

    declared_artifacts = payload.get("artifacts")
    _require(isinstance(declared_artifacts, dict), "standalone artifact declaration is missing")
    artifact_paths = _standalone_artifact_paths(profile)
    _require(set(declared_artifacts) == set(artifact_paths), "standalone artifact set changed")
    empty_artifacts = {
        "run_safe_to_delete",
        "runtime_safe_to_delete",
        "job_runtime_safe_to_delete",
        "completed_marker",
    }
    artifacts = {
        name: _validate_declared_file_record(record, artifact_paths[name], nonempty=name not in empty_artifacts)
        for name, record in declared_artifacts.items()
    }
    _require(artifacts["preflight_ledger"]["sha256"] == profile.preflight_ledger_sha256, "preflight ledger changed")
    _require(artifacts["cuda_postbuild"]["sha256"] == profile.postbuild_record_sha256, "CUDA postbuild record changed")
    _require(Path(artifacts["job_id"]["path"]).read_text().strip() == profile.job_id, "standalone job ledger changed")

    replacement_record = {"artifact": _file_record(path), "payload": dict(payload)}
    return StandaloneJobSource(
        run_key="recovar_full",
        job_id=profile.job_id,
        run_root=profile.run_root,
        script=script,
        script_sha256=profile.script_sha256,
        subject=subject,
        data_dir=data_dir,
        profile=profile,
        artifacts=artifacts,
        inputs=input_records,
        replacement_record=replacement_record,
    )


def build_standalone_replacement_payload(
    manifest_path: Path,
    *,
    profile_name: str,
    data_dir: Path,
    particle_stack: Path,
    reason: str,
) -> dict[str, Any]:
    """Build a deterministic post-completion record for a sealed profile.

    This builder performs no Slurm action and writes nothing.  In particular,
    its all-artifact ledger cannot be constructed while a profile is still
    running because completion, timing, HBM, and final-map artifacts are
    mandatory.  The caller is responsible for an exclusive-create write of
    the returned JSON payload.
    """

    _require(profile_name in STANDALONE_REPLACEMENT_PROFILES, "standalone replacement profile is not accepted")
    profile = STANDALONE_REPLACEMENT_PROFILES[profile_name]
    _require(isinstance(reason, str) and bool(reason.strip()), "standalone replacement reason is missing")
    manifest_path = manifest_path.resolve()
    _, _, _, submission, context, _ = _validate_manifest(manifest_path)
    _standalone_subject(profile)
    data_dir = Path(data_dir)
    particle_stack = Path(particle_stack)
    _require(data_dir.is_absolute() and str(data_dir) == os.path.normpath(str(data_dir)), "standalone data dir is not canonical")
    _require(particle_stack.is_absolute(), "standalone particle stack path is not absolute")
    parent_rows = tuple(context["metadata"]["inputs"].values())

    def input_record(path: Path, expected_sha256: str, expected_size_bytes: int | None = None) -> dict[str, Any]:
        declared = {
            "path": str(path),
            "size_bytes": path.stat().st_size if path.is_file() else -1,
            "sha256": expected_sha256,
        }
        return _input_file_record(
            declared,
            path,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            already_validated=parent_rows,
        )

    inputs = {
        "data_dir": str(data_dir),
        "particles_star": input_record(data_dir / "particles.star", FULL_STAR_SHA256),
        "recovar_reference": input_record(data_dir / "reference_init.mrc", RECOVAR_REFERENCE_SHA256),
        "relion_reference": input_record(data_dir / "reference_init_relion.mrc", RELION_REFERENCE_SHA256),
        "particle_stack": input_record(
            particle_stack,
            PARTICLE_STACK_SHA256,
            PARTICLE_STACK_SIZE_BYTES,
        ),
    }
    script = profile.run_root / "jobs/recovar_full.sbatch"
    script_record = _file_record(script)
    _require(script_record["sha256"] == profile.script_sha256, "standalone Slurm script SHA-256 changed")
    artifact_paths = _standalone_artifact_paths(profile)
    empty_artifacts = {
        "run_safe_to_delete",
        "runtime_safe_to_delete",
        "job_runtime_safe_to_delete",
        "completed_marker",
    }
    artifacts = {
        name: _file_record(path, nonempty=name not in empty_artifacts)
        for name, path in artifact_paths.items()
    }
    cuda_library = profile.run_root / profile.cuda_library_relative_path
    return {
        "schema": STANDALONE_REPLACEMENT_SCHEMA,
        "profile": profile.name,
        "parent_launch_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "run_key": "recovar_full",
        "replaces_job_id": str(submission["job_ids"]["recovar_full"]),
        "job_id": profile.job_id,
        "run_root": str(profile.run_root),
        "script": script_record,
        "source": {
            "repo": str(profile.source_repo),
            "commit": profile.source_commit,
            "tree": profile.source_tree,
            "python": str(profile.python),
            "driver_sha256": profile.driver_sha256,
            "cuda_source_sha256": profile.cuda_source_sha256,
        },
        "inputs": inputs,
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
        "reason": reason,
    }


def _replacement_sources(
    paths: Sequence[Path],
    *,
    manifest_path: Path,
    submission: Mapping[str, Any],
    parent_inputs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, JobSource | StandaloneJobSource], list[dict[str, Any]]]:
    replacements: dict[str, JobSource | StandaloneJobSource] = {}
    records: list[dict[str, Any]] = []
    manifest_sha = sha256_file(manifest_path)
    for path in paths:
        _require(path.is_absolute(), "replacement record path must be absolute")
        payload = _load_json(path)
        if payload.get("schema") == STANDALONE_REPLACEMENT_SCHEMA:
            source = _standalone_replacement_source(
                path,
                payload,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha,
                submission=submission,
                parent_inputs=parent_inputs,
            )
            _require(source.run_key not in replacements, "invalid or duplicate replacement run key")
            replacements[source.run_key] = source
            records.append(source.replacement_record)
            continue
        _require(
            set(payload)
            == {
                "schema",
                "parent_launch_manifest",
                "run_key",
                "replaces_job_id",
                "job_id",
                "run_root",
                "script_sha256",
                "subject",
                "reason",
            },
            f"replacement record fields changed: {path}",
        )
        _require(payload.get("schema") == REPLACEMENT_SCHEMA, "wrong replacement schema")
        parent = payload.get("parent_launch_manifest", {})
        _require(parent == {"path": str(manifest_path), "sha256": manifest_sha}, "replacement parent binding changed")
        key = payload.get("run_key")
        _require(key in RUN_KEYS and key not in replacements, "invalid or duplicate replacement run key")
        original_id = str(submission["job_ids"][key])
        _require(str(payload.get("replaces_job_id")) == original_id, "replacement names wrong original job")
        job_id = str(payload.get("job_id"))
        _require(re.fullmatch(r"[0-9]+", job_id) is not None and job_id != original_id, "invalid replacement job ID")
        run_root = Path(payload.get("run_root", ""))
        _require(run_root.parent == ALLOWED_ROOT, "replacement run root is outside allowed root")
        values = payload.get("subject", {})
        _require(set(values) == set(SUBJECT_FIELDS), "replacement subject fields changed")
        subject = _subject_from_values(values)
        script = run_root / "scripts" / f"{key}.sbatch"
        source = JobSource(
            run_key=key,
            job_id=job_id,
            run_root=run_root,
            script=script,
            script_sha256=str(payload.get("script_sha256")),
            subject=subject,
            submit_record_required=False,
            replacement_record={"artifact": _file_record(path), "payload": payload},
        )
        replacements[key] = source
        records.append(source.replacement_record)
    return replacements, records


def audit_native_launch(
    manifest_path: Path,
    *,
    replacement_records: Sequence[Path] = (),
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    require_science_ready: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate native records and return finalizer-compatible normalized data."""

    payload, root, runtime_root, submission, context, launcher = _validate_manifest(manifest_path)
    base_subject = context["subject"]
    specs = context["specs"]
    replacements, replacement_artifacts = _replacement_sources(
        replacement_records,
        manifest_path=manifest_path,
        submission=submission,
        parent_inputs=context["metadata"]["inputs"],
    )
    jobs: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    selected_sources: dict[str, JobSource | StandaloneJobSource] = {}
    for key in RUN_KEYS:
        base = JobSource(
            run_key=key,
            job_id=str(submission["job_ids"][key]),
            run_root=root,
            script=root / "scripts" / f"{key}.sbatch",
            script_sha256=payload["runs"][key]["script_sha256"],
            subject=base_subject,
            submit_record_required=True,
        )
        jobs[f"{key}:submission"] = _validate_job(launcher, specs[key], base, runner=runner)
        selected_source = replacements.get(key, base)
        if selected_source is not base:
            if isinstance(selected_source, StandaloneJobSource):
                jobs[f"{key}:replacement"] = _validate_standalone_job(
                    launcher,
                    specs[key],
                    selected_source,
                    native_artifacts=context["metadata"]["native_artifacts"],
                    runner=runner,
                )
            else:
                jobs[f"{key}:replacement"] = _validate_job(
                    launcher,
                    specs[key],
                    selected_source,
                    runner=runner,
                )
        selected_sources[key] = selected_source
        selected[key] = jobs[f"{key}:replacement" if selected_source is not base else f"{key}:submission"]

    reasons: list[str] = []
    for key in ("recovar_full", "relion_full"):
        job = selected[key]
        if job["classification"] != "completed":
            reasons.append(f"{key}:{job['classification']}")
        if not job["provenance_complete"]:
            reasons.append(f"{key}:provenance_incomplete")
        if not job["expected_outputs"]:
            reasons.append(f"{key}:full_outputs_missing")
    science_eligible = not reasons
    science = {
        "eligible": science_eligible,
        "reason_codes": reasons,
        "required_run_keys": ["recovar_full", "relion_full"],
        "smoke_only_promotion_forbidden": True,
        "smoke_runs_are_capability_only": True,
    }
    if require_science_ready:
        _require(science_eligible, "native harness is not science-ready: " + ", ".join(reasons))

    recovar_subject = selected["recovar_full"]["subject"]
    normalized_runs = {}
    for key, spec in specs.items():
        source = selected_sources[key]
        normalized_runs[key] = {
            "engine": spec.engine,
            "phase": spec.phase,
            "data_dir": str(source.data_dir if isinstance(source, StandaloneJobSource) else spec.data_dir),
            "output_dir": str(source.run_root / "outputs" / key),
        }
    preparation_path = Path(payload["inputs"]["preparation_manifest"]["path"])
    normalized = {
        "run_root": str(root),
        "runtime_root": str(runtime_root),
        "subject": recovar_subject,
        "runs": normalized_runs,
        "preparation": {
            "preparation_manifest": str(preparation_path),
            "preparation_manifest_sha256": PREPARATION_SHA256,
        },
        "symmetry": {
            "family": "icosahedral",
            "requested_label": "I1",
            "relion_label": "I1",
            "recovar_label": "I1",
            "operator_count": 60,
            "operator_order": "identity at index 0, then RELION SymList::get_matrices(isym) for isym=0..58",
            "left_operator_policy": "all left operators are identity and are not applied in BPref reconstruction",
            "hash_encoding": "float64 rounded to 12 decimals, stacked [left,right], little-endian C-order bytes",
            "operators_sha256": "093a0876b93610ec141c87840ae3ff4dc4491b27dec87143558358ef556557b8",
        },
    }
    audit = {
        "schema": AUDIT_SCHEMA,
        "adapter": _file_record(Path(__file__).resolve()),
        "launcher": _file_record(LAUNCHER_PATH),
        "launch_manifest": _file_record(manifest_path),
        "submission": _file_record(root / "submission.json"),
        "replacement_records": replacement_artifacts,
        "manifest_metadata": context["metadata"],
        "jobs": selected,
        "job_history": jobs,
        "selected_jobs": {key: value["job_id"] for key, value in selected.items()},
        "science_subject": recovar_subject,
        "science_scoring": science,
    }
    standalone_profiles = {
        key: {
            "profile": source.profile.name,
            "job_id": source.job_id,
            "cuda_preflight_sha256": source.profile.cuda_preflight_sha256,
            "cuda_postbuild_sha256": source.profile.cuda_postbuild_sha256,
        }
        for key, source in selected_sources.items()
        if isinstance(source, StandaloneJobSource)
    }
    if standalone_profiles:
        audit["selected_standalone_replacement_profiles"] = standalone_profiles
    return normalized, audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--replacement-record", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _require(args.output.is_absolute(), "output path must be absolute")
    _require(not args.output.exists(), f"refusing to overwrite audit: {args.output}")
    _, audit = audit_native_launch(
        args.launch_manifest.resolve(),
        replacement_records=tuple(path.resolve() for path in args.replacement_record),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": str(args.output), "sha256": sha256_file(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
