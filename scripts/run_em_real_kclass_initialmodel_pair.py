#!/usr/bin/env python3
"""Run a matched real-data K-class InitialModel pair on one Slurm GPU.

Fresh mode runs RELION and RECOVAR sequentially on the same physical GPU and
is the only mode that produces a runtime ratio.  Frozen mode validates and
reuses a sealed RELION pair report, then runs only the current RECOVAR source.
Both modes emit the same permutation-invariant trajectory audit.

This is an InitialModel diagnostic.  It does not produce independent
gold-standard half maps and therefore cannot establish final map resolution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

import recovar
from recovar.utils.parity_provenance import git_worktree_provenance

if __package__:
    from scripts.audit_em_real_kclass_initialmodel import audit_trajectory
    from scripts.run_k1_parity_smoke import referenced_particle_stacks
else:
    from audit_em_real_kclass_initialmodel import audit_trajectory
    from run_k1_parity_smoke import referenced_particle_stacks


DEFAULT_RELION = Path(
    "/scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/build_clean_pinned/bin/relion_refine"
)
PAIR_SCHEMAS = {
    "recovar.vdam_kclass_pair.v1",
    "recovar.em_real_kclass_initialmodel_pair.v2",
}
RESOURCE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORMS",
    "OMP_NUM_THREADS",
    "RECOVAR_CUDA_LIB",
    "RECOVAR_RELION_BIND_BUILD_DIR",
    "RELION_SRC_DIR",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_GPUS",
    "TMPDIR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


class PairRunError(RuntimeError):
    """Raised when paired execution or provenance validation fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PairRunError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PairRunError(f"{label} must contain a JSON object: {path}")
    return value


def _fixture_source_name(path: Path, fixture_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(fixture_dir.resolve()))
    except ValueError:
        return str(resolved)


def _required_fixture_paths(fixture_dir: Path) -> list[Path]:
    data_star = fixture_dir / "particles.star"
    if not data_star.is_file():
        return [data_star]
    particle_stacks = referenced_particle_stacks(data_star, fixture_dir)
    if not particle_stacks:
        raise PairRunError(f"no particle stacks are referenced by {data_star}")
    return [data_star, *particle_stacks]


def _fixture_lineage_paths(fixture_dir: Path) -> list[Path]:
    return [
        path
        for path in (fixture_dir / "fixture_manifest.json", fixture_dir / "source_indices.npy")
        if path.is_file()
    ]


def _input_record(path: Path, fixture_dir: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "fixture_name": _fixture_source_name(path, fixture_dir),
        "fixture_path": str(path.absolute()),
        "resolved_path": str(resolved),
        "is_symlink": path.is_symlink(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _selected_environment(env: dict[str, str]) -> dict[str, str]:
    keys = set(RESOURCE_ENV_KEYS)
    keys.update(key for key in env if key.startswith("RECOVAR_"))
    return {key: env[key] for key in sorted(keys) if key in env}


def _git_branch(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if result.returncode == 1:
        return "<detached>"
    raise PairRunError(f"could not determine git branch: {result.stderr.strip()}")


def _source_provenance(repo_root: Path, *, allow_dirty: bool) -> dict[str, Any]:
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if tracked_status and not allow_dirty:
        raise PairRunError(
            "paired qualification requires a clean tracked worktree; commit or pass --allow-dirty for diagnostics"
        )
    recovar_file = Path(recovar.__file__).resolve()
    jax_file = Path(jax.__file__).resolve()
    if not recovar_file.is_relative_to(repo_root):
        raise PairRunError(f"recovar import is outside source checkout: {recovar_file}")
    return {
        "repo_root": str(repo_root),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
        "git_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root, text=True
        ).strip(),
        "git_branch": _git_branch(repo_root),
        "tracked_dirty": bool(tracked_status),
        "worktree": git_worktree_provenance(),
        "python_executable": str(Path(sys.executable).resolve()),
        "recovar_file": str(recovar_file),
        "jax_file": str(jax_file),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _gpu_inventory() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row for row in csv.reader(result.stdout.splitlines()) if row]
    if len(rows) != 1 or len(rows[0]) != 5 or not rows[0][2].strip().startswith("GPU-"):
        raise PairRunError(f"expected exactly one visible physical GPU, found {rows}")
    row = [value.strip() for value in rows[0]]
    return {
        "visible_index": int(row[0]),
        "name": row[1],
        "uuid": row[2],
        "memory_total_mib": float(row[3]),
        "driver_version": row[4],
    }


def _parse_scontrol_fields(text: str) -> dict[str, str]:
    # Slurm field names are not restricted to alphanumerics.  In particular,
    # the one-line ``scontrol show job`` output contains keys such as
    # ``Socks/Node`` and ``NtasksPerN:B:S:C``.  If those delimiters are not
    # recognised, their text is accidentally appended to the preceding value
    # (most dangerously ``AllocTRES``), making an exact allocation look like a
    # resource mismatch.
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:.-]*)=", text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[match.group(1)] = text[start:end].strip()
    return fields


def _gpu_count_from_tres(value: str) -> int | None:
    generic = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", value)
    if generic is not None:
        return int(generic.group(1))
    typed = re.findall(r"(?:^|,)gres/gpu:[^=,]+=(\d+)(?:,|$)", value)
    return sum(int(item) for item in typed) if typed else None


def _slurm_allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return {"under_slurm": False, "job_id": None}
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = _parse_scontrol_fields(result.stdout.strip())
    requested = fields.get("ReqTRES", "")
    allocated = fields.get("AllocTRES", "")
    if requested != allocated:
        raise PairRunError(f"Slurm ReqTRES != AllocTRES: {requested!r} != {allocated!r}")
    if _gpu_count_from_tres(requested) != 1:
        raise PairRunError(f"paired run requires exactly one requested and allocated GPU: {requested}")
    if fields.get("OverSubscribe") != "OK":
        raise PairRunError(
            f"paired run must be nonexclusive (expected OverSubscribe=OK): {fields.get('OverSubscribe')}"
        )
    return {
        "under_slurm": True,
        "job_id": job_id,
        "ReqTRES": requested,
        "AllocTRES": allocated,
        "TresPerNode": fields.get("TresPerNode"),
        "OverSubscribe": fields.get("OverSubscribe"),
        "NumNodes": fields.get("NumNodes"),
        "NodeList": fields.get("NodeList"),
        "raw_scontrol": result.stdout.strip(),
    }


def _relion_source_provenance(source_dir: Path | None) -> dict[str, Any] | None:
    if source_dir is None:
        return None
    source_dir = source_dir.resolve()
    repo = source_dir.parent if (source_dir.parent / ".git").exists() else source_dir
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    except subprocess.CalledProcessError as exc:
        raise PairRunError(f"cannot record RELION source provenance from {source_dir}") from exc
    return {
        "source_dir": str(source_dir),
        "git_head": head,
        "git_tree": tree,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_dirty": bool(diff),
    }


def build_relion_command(args: argparse.Namespace, output_prefix: Path) -> list[str]:
    return [
        str(args.relion_refine),
        "--o",
        str(output_prefix),
        "--iter",
        str(args.nr_iter),
        "--grad",
        "--denovo_3dref",
        "--grad_write_iter",
        "1",
        "--i",
        "particles.star",
        "--ctf",
        "--K",
        str(args.K),
        "--sym",
        args.symmetry,
        "--flatten_solvent",
        "--zero_mask",
        "--dont_combine_weights_via_disc",
        "--pool",
        "3",
        "--pad",
        str(args.padding_factor),
        "--particle_diameter",
        str(args.particle_diameter),
        "--oversampling",
        str(args.oversampling),
        "--healpix_order",
        str(args.healpix_order),
        "--offset_range",
        str(args.offset_range),
        "--offset_step",
        str(args.offset_step),
        "--auto_sampling",
        "--tau2_fudge",
        str(args.tau2_fudge),
        "--random_seed",
        str(args.random_seed),
        "--j",
        str(args.threads),
        "--gpu",
        "0",
    ]


def build_recovar_command(args: argparse.Namespace, output_prefix: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.run_ab_initio",
        "--i",
        str(args.fixture_dir / "particles.star"),
        "--datadir",
        str(args.fixture_dir),
        "--o",
        str(output_prefix),
        "--nr_iter",
        str(args.nr_iter),
        "--K",
        str(args.K),
        "--tau2_fudge",
        str(args.tau2_fudge),
        "--sym",
        args.symmetry,
        "--do_run_C1",
        "0",
        "--particle_diameter",
        str(args.particle_diameter),
        "--random_seed",
        str(args.random_seed),
        "--healpix_order",
        str(args.healpix_order),
        "--oversampling",
        str(args.oversampling),
        "--offset_range",
        str(args.offset_range),
        "--offset_step",
        str(args.offset_step),
        "--padding_factor",
        str(args.padding_factor),
        "--image_batch_size",
        str(args.image_batch_size),
        "--image_fourier_backend",
        args.image_fourier_backend,
        "--rotation_block_size",
        str(args.rotation_block_size),
        "--j",
        str(args.threads),
        "--gpu",
        "0",
    ]


def _parse_gpu_monitor(path: Path) -> dict[str, Any]:
    samples: list[tuple[float, float]] = []
    if path.is_file():
        with path.open(newline="") as stream:
            for row in csv.reader(stream):
                if len(row) < 6:
                    continue
                try:
                    used = float(row[3].strip())
                    total = float(row[4].strip())
                except ValueError:
                    continue
                samples.append((used, total))
    return {
        "path": str(path),
        "sample_count": len(samples),
        "peak_hbm_mib": max((used for used, _ in samples), default=None),
        "gpu_memory_total_mib": max((total for _, total in samples), default=None),
    }


def _parse_max_rss(path: Path) -> int | None:
    if not path.is_file():
        return None
    match = re.search(r"RECOVAR_EM_MAX_RSS_KIB=(\d+)", path.read_text())
    return int(match.group(1)) if match else None


def _run_measured(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    time_executable = Path("/usr/bin/time")
    if not time_executable.is_file():
        raise PairRunError("GNU /usr/bin/time is required for per-engine MaxRSS capture")
    log_path = output_dir / "run.log"
    resource_path = output_dir / "process_resources.txt"
    monitor_path = output_dir / "gpu_monitor.csv"
    monitor_command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-l",
        "1",
    ]
    started_epoch = time.time()
    started = time.perf_counter()
    with monitor_path.open("w") as monitor_stream, log_path.open("w") as log_stream:
        monitor = subprocess.Popen(
            monitor_command,
            stdout=monitor_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            result = subprocess.run(
                [
                    str(time_executable),
                    "-f",
                    "RECOVAR_EM_MAX_RSS_KIB=%M",
                    "-o",
                    str(resource_path),
                    "--",
                    *command,
                ],
                cwd=cwd,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=10)
    timing = {
        "wall_s": time.perf_counter() - started,
        "start_epoch": started_epoch,
        "end_epoch": time.time(),
        "exit_code": result.returncode,
        "max_rss_kib": _parse_max_rss(resource_path),
        "max_rss_source": "GNU time %M for the engine process tree",
        "gpu": _parse_gpu_monitor(monitor_path),
        "log_path": str(log_path),
        "resource_path": str(resource_path),
    }
    if result.returncode:
        raise PairRunError(f"command failed with exit {result.returncode}; see {log_path}")
    return timing


def _validated_frozen_reference(
    args: argparse.Namespace,
    fixture_sha256: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    report_path = args.reference_pair_report.resolve()
    report = _load_json_object(report_path, label="reference pair report")
    if report.get("schema") not in PAIR_SCHEMAS:
        raise PairRunError(f"unsupported frozen K-class pair schema: {report.get('schema')}")
    if bool(report.get("git_dirty") or report.get("source", {}).get("tracked_dirty")):
        raise PairRunError("frozen reference was generated from a dirty source tree")
    audit = report.get("audit")
    if not isinstance(audit, dict):
        raise PairRunError("frozen reference pair report has no audit object")
    thresholds = audit.get("thresholds") or {}
    expected_contract = {
        "K": int(args.K),
        "checkpoints": [int(value) for value in args.checkpoint],
        "minimum_per_class_fsc_auc": float(args.minimum_fsc_auc),
        "minimum_class_assignment_accuracy": float(args.minimum_assignment_accuracy),
    }
    if report.get("schema") == "recovar.em_real_kclass_initialmodel_pair.v2":
        expected_contract["minimum_class_fraction"] = float(args.minimum_class_fraction)
    actual_contract = {
        "K": int(audit.get("K", 0)),
        "checkpoints": [int(value) for value in audit.get("checkpoints", ())],
        "minimum_per_class_fsc_auc": float(thresholds.get("minimum_per_class_fsc_auc", np.nan)),
        "minimum_class_assignment_accuracy": float(
            thresholds.get("minimum_class_assignment_accuracy", np.nan)
        ),
    }
    if report.get("schema") == "recovar.em_real_kclass_initialmodel_pair.v2":
        actual_contract["minimum_class_fraction"] = float(
            thresholds.get("minimum_class_fraction", np.nan)
        )
    if actual_contract != expected_contract:
        raise PairRunError(
            f"frozen reference audit contract differs: {actual_contract} != {expected_contract}"
        )
    if report.get("fixture_sha256") != fixture_sha256:
        raise PairRunError("frozen reference fixture hashes differ from the requested fixture")

    reference_dir = report_path.parent / "relion"
    command_path = reference_dir / "command.json"
    try:
        recorded_command = json.loads(command_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PairRunError(f"cannot read frozen RELION command {command_path}: {exc}") from exc
    if recorded_command != build_relion_command(args, reference_dir / "run"):
        raise PairRunError("frozen RELION command differs from the requested scientific contract")

    executable = Path(str(report.get("relion_executable", "")))
    executable_sha256 = str(report.get("relion_sha256", ""))
    if not executable.is_file() or len(executable_sha256) != 64 or _sha256(executable) != executable_sha256:
        raise PairRunError("frozen reference RELION executable is missing or has changed")
    timing = report.get("relion_timing") or (report.get("engines") or {}).get("relion")
    if not isinstance(timing, dict) or int(timing.get("exit_code", -1)) != 0:
        raise PairRunError("frozen reference has no successful RELION timing record")
    return report, reference_dir


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _build_artifacts(env: dict[str, str]) -> dict[str, Any]:
    cuda_value = env.get("RECOVAR_CUDA_LIB")
    cuda_path = Path(cuda_value).resolve() if cuda_value else None
    binding_value = env.get("RECOVAR_RELION_BIND_BUILD_DIR")
    binding_dir = Path(binding_value).resolve() if binding_value else None
    binding_files = sorted(binding_dir.glob("_relion_bind_core*.so")) if binding_dir else []
    return {
        "cuda_library": (
            _artifact_record(cuda_path) if cuda_path is not None and cuda_path.is_file() else None
        ),
        "cuda_library_missing_reason": (
            None
            if cuda_path is not None and cuda_path.is_file()
            else "RECOVAR_CUDA_LIB is unset or does not name a built library"
        ),
        "relion_binding": [_artifact_record(path) for path in binding_files],
        "relion_binding_missing_reason": (
            None if binding_files else "RECOVAR_RELION_BIND_BUILD_DIR has no built binding"
        ),
    }


def _frozen_relion_resources(report: dict[str, Any]) -> dict[str, Any]:
    stored = (report.get("engines") or {}).get("relion") or report.get("relion_timing")
    resources = dict(stored)
    resources.setdefault("max_rss_kib", None)
    resources.setdefault(
        "max_rss_missing_reason",
        "frozen RELION reference predates per-engine RSS capture",
    )
    resources.setdefault(
        "gpu",
        {
            "sample_count": 0,
            "peak_hbm_mib": None,
            "gpu_memory_total_mib": None,
            "missing_reason": "frozen RELION reference predates per-engine HBM capture",
        },
    )
    return resources


def _relion_oracle_source(
    frozen_report: dict[str, Any] | None,
    binding_source: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if frozen_report is None:
        return binding_source
    frozen_source = frozen_report.get("relion_source")
    if isinstance(frozen_source, dict):
        return frozen_source
    return {
        "available": False,
        "missing_reason": (
            "legacy frozen pair report did not record the RELION source tree; "
            "relion_binding_source records only the source used to build the current "
            "RECOVAR binding"
        ),
    }


def run_pair(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    source = _source_provenance(repo_root, allow_dirty=args.allow_dirty)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise PairRunError(f"refusing to reuse non-empty output root: {args.output_root}")
    candidate_dir = args.output_root / "recovar"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (args.output_root / "SAFE_TO_DELETE").touch()

    required_fixture = _required_fixture_paths(args.fixture_dir)
    missing = [str(path) for path in required_fixture if not path.is_file()]
    if missing:
        raise PairRunError(f"missing K-class fixture paths: {missing}")
    input_records = [_input_record(path, args.fixture_dir) for path in required_fixture]
    fixture_sha256 = {record["fixture_name"]: record["sha256"] for record in input_records}
    lineage_records = [_input_record(path, args.fixture_dir) for path in _fixture_lineage_paths(args.fixture_dir)]

    frozen_report = None
    if args.reference_pair_report is None:
        reference_mode = "fresh_same_gpu"
        reference_dir = args.output_root / "relion"
        reference_dir.mkdir(parents=True, exist_ok=True)
        if not args.relion_refine.is_file() or not os.access(args.relion_refine, os.X_OK):
            raise PairRunError(f"RELION executable is unavailable: {args.relion_refine}")
        relion_command = build_relion_command(args, reference_dir / "run")
        (reference_dir / "command.json").write_text(json.dumps(relion_command, indent=2) + "\n")
    else:
        reference_mode = "frozen_pair_report"
        frozen_report, reference_dir = _validated_frozen_reference(args, fixture_sha256)
        (args.output_root / "relion").symlink_to(reference_dir, target_is_directory=True)
        relion_command = None
    recovar_command = build_recovar_command(args, candidate_dir / "run")
    (candidate_dir / "command.json").write_text(json.dumps(recovar_command, indent=2) + "\n")

    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV", "JAX_PLATFORM_NAME"):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        CUDA_LAUNCH_BLOCKING="0",
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    gpu_before = _gpu_inventory()
    slurm = _slurm_allocation()
    if relion_command is None:
        relion_resources = _frozen_relion_resources(frozen_report)
        gpu_between = gpu_before
    else:
        relion_resources = _run_measured(
            relion_command,
            cwd=args.fixture_dir,
            env=env,
            output_dir=reference_dir,
        )
        gpu_between = _gpu_inventory()
    recovar_resources = _run_measured(
        recovar_command,
        cwd=repo_root,
        env=env,
        output_dir=candidate_dir,
    )
    gpu_after = _gpu_inventory()
    if len({gpu_before["uuid"], gpu_between["uuid"], gpu_after["uuid"]}) != 1:
        raise PairRunError("physical GPU identity changed during paired execution")

    audit, shellwise = audit_trajectory(
        candidate_dir=candidate_dir,
        reference_dir=reference_dir,
        K=args.K,
        checkpoints=tuple(args.checkpoint),
        minimum_fsc_auc=args.minimum_fsc_auc,
        minimum_assignment_accuracy=args.minimum_assignment_accuracy,
        minimum_class_fraction=args.minimum_class_fraction,
    )
    audit_path = args.output_root / "trajectory_audit.json"
    shellwise_path = args.output_root / "trajectory_shellwise_fsc.npz"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(shellwise_path, **shellwise)

    runtime_comparable = reference_mode == "fresh_same_gpu"
    relion_binding_source = _relion_source_provenance(args.relion_source_dir)
    relion_oracle_source = _relion_oracle_source(frozen_report, relion_binding_source)
    relion_executable = (
        str(args.relion_refine.resolve())
        if frozen_report is None
        else str(frozen_report["relion_executable"])
    )
    relion_sha256 = (
        _sha256(args.relion_refine)
        if frozen_report is None
        else str(frozen_report["relion_sha256"])
    )
    relion_wall = float(relion_resources["wall_s"])
    recovar_wall = float(recovar_resources["wall_s"])
    recorded_relion_command = (
        relion_command
        if relion_command is not None
        else json.loads((reference_dir / "command.json").read_text())
    )
    evidence_paths = [
        candidate_dir / "command.json",
        candidate_dir / "run.log",
        candidate_dir / "gpu_monitor.csv",
        candidate_dir / "process_resources.txt",
        audit_path,
        shellwise_path,
    ]
    if relion_command is not None:
        evidence_paths.extend(
            [
                reference_dir / "command.json",
                reference_dir / "run.log",
                reference_dir / "gpu_monitor.csv",
                reference_dir / "process_resources.txt",
            ]
        )
    report = {
        "schema": "recovar.em_real_kclass_initialmodel_pair.v2",
        "dataset": args.dataset,
        "scientific_scope": "real-particle K-class InitialModel cross-engine diagnostic",
        "scientific_limitations": {
            "gold_standard_halfmaps_available": False,
            "halfmap_fsc_available": False,
            "reason": (
                "InitialModel emits one map per class; this run measures cross-engine trajectory parity "
                "and class stability, not independent-half-map resolution or biological class validity"
            ),
        },
        "source": source,
        "git_head": source["git_head"],
        "git_dirty": source["tracked_dirty"],
        "reference_mode": reference_mode,
        "reference_pair_report": (
            None if args.reference_pair_report is None else str(args.reference_pair_report.resolve())
        ),
        "reference_pair_report_sha256": (
            None if args.reference_pair_report is None else _sha256(args.reference_pair_report.resolve())
        ),
        "runtime_comparable": runtime_comparable,
        "fixture_dir": str(args.fixture_dir),
        "fixture_sha256": fixture_sha256,
        "input_records": input_records,
        "fixture_lineage_records": lineage_records,
        "commands": {"relion": recorded_relion_command, "recovar": recovar_command},
        "environment": _selected_environment(env),
        "build_artifacts": _build_artifacts(env),
        "slurm": slurm,
        "physical_gpu": gpu_before,
        "physical_gpu_uuid": gpu_before["uuid"],
        "relion_executable": relion_executable,
        "relion_sha256": relion_sha256,
        "relion_source": relion_oracle_source,
        "relion_binding_source": relion_binding_source,
        "relion_executable_source_binding": {
            "cryptographically_attested": False,
            "reason": (
                "the executable SHA-256 and source Git tree are sealed independently; no build "
                "attestation binds that binary hash to that source tree"
            ),
        },
        "engines": {"relion": relion_resources, "recovar": recovar_resources},
        "relion_timing": relion_resources,
        "recovar_timing": recovar_resources,
        "runtime_ratio_recovar_over_relion": recovar_wall / relion_wall if runtime_comparable else None,
        "audit": audit,
        "evidence_artifacts": [_artifact_record(path) for path in evidence_paths],
        "safe_to_delete_marker": str(args.output_root / "SAFE_TO_DELETE"),
    }
    report_path = args.output_root / "pair_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--relion-refine", type=Path, default=DEFAULT_RELION)
    parser.add_argument("--relion-source-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-pair-report",
        type=Path,
        default=None,
        help="Reuse immutable RELION artifacts; correctness only, with no runtime ratio.",
    )
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--nr-iter", type=int, default=8)
    parser.add_argument("--checkpoint", type=int, action="append", default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--tau2-fudge", type=float, default=4.0)
    parser.add_argument("--symmetry", default="C1")
    parser.add_argument("--particle-diameter", type=float, default=200.0)
    parser.add_argument("--healpix-order", type=int, default=1)
    parser.add_argument("--oversampling", type=int, default=1)
    parser.add_argument("--offset-range", type=float, default=6.0)
    parser.add_argument("--offset-step", type=float, default=2.0)
    parser.add_argument("--padding-factor", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=500)
    parser.add_argument(
        "--image-fourier-backend",
        choices=("host_numpy", "jax_gpu", "relion_cuda"),
        default="relion_cuda",
        help="Pin the image Fourier preprocessing implementation used by RECOVAR.",
    )
    parser.add_argument("--rotation-block-size", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--minimum-fsc-auc", type=float, default=0.999)
    parser.add_argument("--minimum-assignment-accuracy", type=float, default=0.995)
    parser.add_argument("--minimum-class-fraction", type=float, default=0.01)
    parser.add_argument("--allow-dirty", action="store_true", help="Permit non-qualification diagnostics")
    args = parser.parse_args(argv)
    args.fixture_dir = args.fixture_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.relion_refine = args.relion_refine.expanduser().resolve()
    if args.relion_source_dir is not None:
        args.relion_source_dir = args.relion_source_dir.expanduser().resolve()
    if args.reference_pair_report is not None:
        args.reference_pair_report = args.reference_pair_report.expanduser().resolve()
    if args.K < 2:
        parser.error("--K must be at least 2")
    if args.nr_iter < 1:
        parser.error("--nr-iter must be positive")
    if not 0.0 <= args.minimum_class_fraction < 1.0 / args.K:
        parser.error("--minimum-class-fraction must be in [0, 1/K)")
    if args.checkpoint is None:
        # Native InitialModel emits artifacts only after iterations 1..N.
        # Keep zero valid for explicit legacy/frozen-oracle replays, but do
        # not request an artifact that a fresh current run cannot produce.
        args.checkpoint = list(range(1, args.nr_iter + 1))
    if sorted(set(args.checkpoint)) != args.checkpoint or not all(
        0 <= value <= args.nr_iter for value in args.checkpoint
    ):
        parser.error("--checkpoint values must be sorted, unique, and within 0..nr-iter")
    return args


def main(argv: list[str] | None = None) -> int:
    report = run_pair(_parse_args(argv))
    return 0 if report["audit"]["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
