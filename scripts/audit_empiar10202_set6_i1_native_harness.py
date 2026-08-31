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

NATIVE_LAUNCH_SCHEMA = "recovar.em.matched_launch_harness.v1"
NATIVE_SUBMISSION_SCHEMA = "recovar.em.matched_submission.v1"
REPLACEMENT_SCHEMA = "recovar.em.native_job_replacement.v1"
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


def _replacement_sources(
    paths: Sequence[Path],
    *,
    manifest_path: Path,
    submission: Mapping[str, Any],
) -> tuple[dict[str, JobSource], list[dict[str, Any]]]:
    replacements: dict[str, JobSource] = {}
    records: list[dict[str, Any]] = []
    manifest_sha = sha256_file(manifest_path)
    for path in paths:
        _require(path.is_absolute(), "replacement record path must be absolute")
        payload = _load_json(path)
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
    )
    jobs: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    selected_sources: dict[str, JobSource] = {}
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
            "data_dir": str(spec.data_dir),
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
