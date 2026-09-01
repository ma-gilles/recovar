#!/usr/bin/env python3
"""Analyze the sealed VDAM raw-image-cache OFF/AUTO/AUTO/OFF experiment.

The four arms continue the same GF46 iteration-180 checkpoint through exactly
iteration 181 with the H100 native-atomic eight-stream selector fixed.  The
analyzer separates evidence-integrity failures (which are fatal) from a valid
diagnostic result that does not clear the decision gate (``NO_GO``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from scripts import summarize_vdam_nsys_sqlite as nsys_sqlite
from scripts.analyze_vdam_coarse_multistream_late_pair import (
    LatePairSetupError,
    _coarse_translation_count,
    _git_rev_parse,
    _integer,
    _load_json,
    _load_map,
    _load_star,
    _map_delta,
    _numeric_scalars,
    _parse_sha_line,
    _read_single_line,
    _require,
    _resolved_inside,
    _sha256,
    _star_equal,
    _validate_manifest,
    _validate_selector_audit,
    _values_equal,
)

SCHEMA = "recovar.vdam_raw_cache_abba_analysis.v1"
RUN_SCHEMA = "recovar.vdam_raw_cache_abba.v1"
PROFILE_SCHEMA = "recovar.vdam_late_iteration_profile.v1"
NSIGHT_SCHEMA = "recovar.vdam_nsys_sqlite_summary.v1"
PROFILED_ITERATION = 181
EXPECTED_CACHE_BYTES = 196_608_000
EXPECTED_CACHE_IMAGES = 3_000
EXPECTED_IMAGE_SIZE = 128
EXPECTED_CACHE_DTYPE = "<f4"
EXPECTED_CACHE_MAX_GB = 16.0
EXPECTED_SCHEDULE = {
    "current_size": 128,
    "healpix_order": 3,
    "n_rotations": 294_912,
    "n_translations": 116,
    "subset_size": None,
    "random_perturbation": 0.4751259684562683,
}
EXPECTED_COARSE_LAUNCHES = 1_000
HWM_SLACK_BYTES = 128 * 1024**2
MIN_PERCENT_WIN = 5.0
MIN_EXPECTATION_WIN_S = 0.30
MIN_NO_KERNEL_WIN_S = 0.30
COARSE_KERNEL_NAME = "relion_coarse_diff2_projector_f32_kernel"
GETITEM_RANGE = "ParticleImageDataset.__getitem__"
LOADER_RANGE = "MRCLoader._load"
PASS1_RANGE = "kclass.adaptive.pass1_significance"
PASS2_RANGE = "local.run_local_em_exact"
STAGE_RANGES = (PASS1_RANGE, PASS2_RANGE)
SCIENCE_BASE_HEAD = "77e09c292e438a265a6d157414b2a0fe525710e6"
SCIENCE_BASE_TREE = "384448a3ce6639e571cad53d2b82c1b18df1979a"
PARTICLE_STACK_SHA256 = "804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9"
QUALIFIED_GATE_SHA256SUMS_SHA256 = "9ab443af3a90f63bc0fd6eeac90f9c15f84d7f667c6c19171682a11f0c168cc8"
NSYS_SHA256 = "9b32b4e9beee469bc8c26640db228b04583234505c315e23f7e622efd61a68ab"
CUSPARSE_SHA256 = "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf"

ARM_SPECS = (
    ("cache_off_1", "off", 1),
    ("cache_auto_1", "auto", 1),
    ("cache_auto_2", "auto", 2),
    ("cache_off_2", "off", 2),
)
ARM_LABELS = tuple(spec[0] for spec in ARM_SPECS)
MODES = ("off", "auto")
DISCRETE_META_KEYS = (
    "selected_particle_ids",
    "best_pose_rotation_ids",
    "best_pose_rotations",
    "best_pose_translations",
    "class_assignments",
    "max_posterior_per_image",
    "pose_assignments",
    "halfset_0_class_assignments",
)
PERFORMANCE_METRICS = (
    "warm_wall_s",
    "warm_expectation_s",
    "pass1_s",
    "pass2_s",
    "stage_no_kernel_s",
    "getitem_union_s",
    "loader_union_s",
    "gpu_kernel_union_s",
    "gpu_kernel_sum_s",
    "coarse_kernel_sum_s",
    "coarse_kernel_union_s",
    "coarse_kernel_launch_count",
    "process_user_cpu_s",
    "process_system_cpu_s",
    "process_total_cpu_s",
    "process_read_bytes",
    "process_write_bytes",
    "process_hwm_increment_bytes",
    "process_current_rss_after_bytes",
    "process_high_water_rss_after_bytes",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

# Public alias with a task-specific name while retaining the shared primitive's
# exception type (and therefore its fail-closed behavior).
RawCacheSetupError = LatePairSetupError


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_)),
        f"{label} must be numeric",
    )
    parsed = float(value)
    _require(math.isfinite(parsed), f"{label} must be finite")
    _require(parsed > 0.0 if positive else parsed >= 0.0, f"{label} is out of range")
    return parsed


def _exact_option(tokens: list[str], option: str, expected: str, label: str) -> None:
    positions = [index for index, token in enumerate(tokens) if token == option]
    _require(len(positions) == 1, f"{label} command must contain {option} exactly once")
    index = positions[0]
    _require(index + 1 < len(tokens), f"{label} command has no value for {option}")
    _require(tokens[index + 1] == expected, f"{label} command has the wrong {option}")


def _validate_external_sha_line(path: Path, expected: str, label: str) -> dict[str, str]:
    line = _read_single_line(path, label)
    try:
        digest, raw_path = line.split(maxsplit=1)
    except ValueError as exc:
        raise RawCacheSetupError(f"{label} has malformed sha256sum output") from exc
    _require(digest == expected and _SHA256_RE.fullmatch(digest), f"{label} digest differs")
    artifact = Path(raw_path).resolve()
    _require(artifact.is_file(), f"{label} target is missing: {artifact}")
    _require(_sha256(artifact) == digest, f"{label} target digest differs")
    return {"path": str(artifact), "sha256": digest}


def _validate_command(
    path: Path,
    *,
    root: Path,
    label: str,
    mode: str,
) -> dict[str, str]:
    _require(path.is_file(), f"missing command ledger for {label}: {path}")
    try:
        tokens = shlex.split(path.read_text())
    except (OSError, ValueError) as exc:
        raise RawCacheSetupError(f"cannot parse command ledger for {label}: {exc}") from exc
    assignments = {
        "RECOVAR_EM_RAW_IMAGE_CACHE": mode,
        "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB": str(int(EXPECTED_CACHE_MAX_GB)),
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "8",
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "1",
    }
    for name, expected in assignments.items():
        matches = [token for token in tokens if token.startswith(f"{name}=")]
        # The max-GiB spelling may be 16 or 16.0; normalize only that value.
        if name == "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB" and len(matches) == 1:
            observed = matches[0].split("=", 1)[1]
            _require(float(observed) == EXPECTED_CACHE_MAX_GB, f"{label} command has the wrong {name}")
        else:
            _require(matches == [f"{name}={expected}"], f"{label} command has the wrong {name}")
    _require(
        tokens.count("scripts.run_vdam_late_iteration_profile") == 1,
        f"{label} command does not invoke the late profiler exactly once",
    )
    for flag in ("--audit-raw-image-cache", "--cuda-profiler-range"):
        _require(tokens.count(flag) == 1, f"{label} command must contain {flag} exactly once")
    _exact_option(tokens, "--checkpoint-iteration", "180", label)
    _exact_option(tokens, "--nr-iter", "200", label)
    _exact_option(tokens, "--image-batch-size", "500", label)
    _exact_option(tokens, "--exact-local-bucket-radix", "4", label)
    _exact_option(tokens, "--exact-local-physical-order-chunk-size", "0", label)
    cache_matches = [token for token in tokens if token.startswith("JAX_COMPILATION_CACHE_DIR=")]
    _require(len(cache_matches) == 1, f"{label} command has no unique JAX cache")
    cache_path = _resolved_inside(
        Path(cache_matches[0].split("=", 1)[1]), root, f"{label} JAX cache"
    )
    expected_cache = (root / "runs" / label / "jax_cache").resolve()
    _require(cache_path == expected_cache, f"{label} JAX cache path differs")
    return {"sha256": _sha256(path), "jax_cache": str(cache_path)}


def _validate_execution_order(path: Path, root: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing execution order: {path}")
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except OSError as exc:
        raise RawCacheSetupError(f"cannot read execution order: {exc}") from exc
    _require(len(rows) == 4, "execution order must contain exactly four arms")
    expected_columns = (
        "order",
        "label",
        "raw_image_cache_mode",
        "raw_image_cache_max_gb",
        "workers",
        "single_lane_canonical",
        "native_atomic_reduction",
        "nsys_base",
    )
    required = set(expected_columns)
    _require(required.issubset(rows[0].keys()) if rows else False, "execution-order columns differ")
    _require(tuple(rows[0].keys()) == expected_columns, "execution-order columns differ")
    normalized = []
    for order, (row, spec) in enumerate(zip(rows, ARM_SPECS, strict=True), start=1):
        label, mode, _repeat = spec
        expected = {
            "order": str(order),
            "label": label,
            "raw_image_cache_mode": mode,
            "raw_image_cache_max_gb": "16",
        }
        mismatches = {key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value}
        _require(not mismatches, f"execution-order arm {order} differs: {mismatches}")
        for column, expected_value in (
            ("workers", "8"),
            ("single_lane_canonical", "0"),
            ("native_atomic_reduction", "1"),
        ):
            if column in row:
                _require(row[column] == expected_value, f"execution-order {label} {column} differs")
        nsys_base = _resolved_inside(Path(row["nsys_base"]), root, f"{label} Nsight base")
        _require(
            nsys_base == (root / "nsight" / f"{label}_it181_warm").resolve(),
            f"{label} Nsight base differs",
        )
        normalized.append({**expected, "nsys_base": str(nsys_base)})
    return normalized


def _validate_jax_cache(root: Path, label: str, recorded_path: str) -> dict[str, Any]:
    run_root = root / "runs" / label
    cache = Path(recorded_path)
    _require(cache == (run_root / "jax_cache").resolve(), f"{label} JAX cache is not private")
    _require(cache.is_dir(), f"{label} JAX cache directory is missing")
    inventory_path = run_root / "jax_cache_files.txt"
    count_path = run_root / "jax_cache_file_count.txt"
    _require(inventory_path.is_file() and count_path.is_file(), f"{label} JAX cache inventory is missing")
    recorded = inventory_path.read_text().splitlines()
    _require(recorded == sorted(recorded) and len(recorded) == len(set(recorded)), f"{label} JAX inventory differs")
    observed = sorted(path.name for path in cache.iterdir() if path.is_file())
    _require(recorded == observed, f"{label} JAX cache inventory does not match the directory")
    count = _integer(int(_read_single_line(count_path, f"{label} JAX cache count")), f"{label} cache count")
    _require(count == len(recorded), f"{label} JAX cache count differs")
    non_marker = [name for name in recorded if name != "SAFE_TO_DELETE"]
    _require(non_marker, f"{label} JAX cache contains no compiled entries")
    return {
        "path": str(cache),
        "inventory_sha256": _sha256(inventory_path),
        "file_count": count,
        "compiled_file_count": len(non_marker),
        "files": recorded,
    }


def _validate_provenance(root: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(root.is_dir(), f"ABBA root does not exist: {root}")
    _require(
        (root / "ARMS_COMPLETED").is_file() or (root / "COMPLETED").is_file(),
        "ABBA arms are incomplete",
    )
    provenance = root / "provenance"
    _require(not (provenance / "failure.txt").exists(), "ABBA root records a failure")
    run_path = provenance / "run.json"
    run = _load_json(run_path, "raw-cache ABBA run provenance")
    expected = {
        "schema": RUN_SCHEMA,
        "classification": "diagnostic_performance_only",
        "execution_order": list(ARM_LABELS),
        "raw_image_cache_modes": [spec[1] for spec in ARM_SPECS],
        "raw_image_cache_max_gb": EXPECTED_CACHE_MAX_GB,
        "raw_image_cache_expected_bytes": EXPECTED_CACHE_BYTES,
        "raw_image_cache_force_used": False,
        "coarse_multistream_workers": 8,
        "single_lane_canonical": False,
        "native_atomic_reduction": True,
        "checkpoint_iteration": 180,
        "profiled_iteration": 181,
        "nr_iter_schedule": 200,
        "random_seed": 29,
        "image_batch_size": 500,
        "exact_local_bucket_radix": 4,
        "exact_local_physical_order_chunk_size": 0,
        "source_manifest_scope": "selected_high_risk_files",
        "science_promotion_allowed": False,
        "science_base_head": SCIENCE_BASE_HEAD,
        "science_base_tree": SCIENCE_BASE_TREE,
    }
    mismatches = {
        key: {"expected": value, "observed": run.get(key)}
        for key, value in expected.items()
        if run.get(key) != value
    }
    _require(not mismatches, f"run provenance differs: {mismatches}")
    expected_run_fields = {
        "schema", "classification", "job_id", "git_head", "git_tree",
        "science_base_head", "science_base_tree", "source_manifest_sha256",
        "source_manifest_scope", "input_manifest_sha256", "particle_stack_sha256",
        "gpu_uuid", "gpu_name", "node", "qualified_gpu_gate_root",
        "qualified_gate_sha256sums_sha256", "cuda_sha256", "relion_bind_sha256",
        "interpreter_sha256", "nsys_sha256", "cusparse_sha256", "execution_order",
        "raw_image_cache_modes", "raw_image_cache_max_gb", "raw_image_cache_expected_bytes",
        "raw_image_cache_force_used", "checkpoint_iteration", "profiled_iteration",
        "nr_iter_schedule", "random_seed", "image_batch_size", "coarse_multistream_workers",
        "single_lane_canonical", "native_atomic_reduction", "exact_local_bucket_radix",
        "exact_local_physical_order_chunk_size", "science_promotion_allowed",
    }
    _require(set(run) == expected_run_fields, f"run provenance fields differ: {sorted(set(run) ^ expected_run_fields)}")
    _require(isinstance(run.get("qualified_gpu_gate_root"), str) and run["qualified_gpu_gate_root"], "qualified gate root is invalid")
    for name in ("git_head", "git_tree"):
        _require(isinstance(run.get(name), str) and _GIT_OBJECT_RE.fullmatch(run[name]), f"run {name} is invalid")
    resolved_head = _git_rev_parse(repo, f"{run['git_head']}^{{commit}}", "run git_head")
    resolved_tree = _git_rev_parse(repo, f"{run['git_head']}^{{tree}}", "run git tree")
    _require(resolved_head == run["git_head"], "run git_head differs")
    _require(resolved_tree == run["git_tree"], "run git tree differs")
    digest_fields = (
        "cuda_sha256",
        "relion_bind_sha256",
        "interpreter_sha256",
        "source_manifest_sha256",
        "input_manifest_sha256",
        "particle_stack_sha256",
        "qualified_gate_sha256sums_sha256",
        "nsys_sha256",
        "cusparse_sha256",
    )
    for name in digest_fields:
        _require(isinstance(run.get(name), str) and _SHA256_RE.fullmatch(run[name]), f"run {name} is invalid")
    _require(isinstance(run.get("job_id"), str) and run["job_id"], "run job ID is invalid")
    _require(isinstance(run.get("gpu_uuid"), str) and run["gpu_uuid"].startswith("GPU-"), "run GPU UUID is invalid")
    _require("H100" in str(run.get("gpu_name", "")), "run did not use an H100")
    _require(isinstance(run.get("node"), str) and run["node"], "run node is invalid")
    recorded = {
        "git_head": _read_single_line(provenance / "repo_head.txt", "recorded repo HEAD"),
        "git_tree": _read_single_line(provenance / "repo_tree.txt", "recorded repo tree"),
        "job_id": _read_single_line(provenance / "slurm_job_id.txt", "recorded Slurm job"),
        "gpu_uuid": _read_single_line(provenance / "selected_gpu_uuid.txt", "selected GPU UUID"),
        "node": _read_single_line(provenance / "node.txt", "recorded node"),
        "gpu_name": _read_single_line(provenance / "gpu_name.txt", "recorded GPU name"),
    }
    _require(all(recorded[key] == run[key] for key in recorded), "scalar provenance differs from run.json")
    _require(
        _read_single_line(provenance / "science_base_head.txt", "science base HEAD") == run["science_base_head"],
        "science base HEAD ledger differs",
    )
    _require(
        _read_single_line(provenance / "science_base_tree.txt", "science base tree") == run["science_base_tree"],
        "science base tree ledger differs",
    )
    _require((provenance / "repo_status.txt").read_text() == "", "run repository was dirty")
    repo_diff = _read_single_line(provenance / "repo_diff.sha256", "repo diff digest").split()[0]
    _require(repo_diff == _EMPTY_SHA256, "run repository diff was nonempty")
    for name in ("allocated_gpu_uuids.csv", "visible_gpu_uuids.csv"):
        values = _read_single_line(provenance / name, name).split(",")
        _require(values == [run["gpu_uuid"]], f"{name} does not prove single-GPU ownership")
    source_path = provenance / "source_manifest.sha256"
    source = _validate_manifest(
        source_path,
        expected_digest=run["source_manifest_sha256"],
        relative_base=repo,
        label="source manifest",
    )
    final_source = provenance / "source_manifest.final.sha256"
    # The runner calls the analyzer after ARMS_COMPLETED and writes this final
    # ledger immediately after the analyzer.  A post-hoc re-analysis must
    # verify it; the in-run analysis records that the final seal is pending.
    if (root / "COMPLETED").is_file():
        _require(final_source.is_file(), "completed result lacks its final source manifest")
    if final_source.exists():
        _require(final_source.is_file(), "final source manifest is invalid")
        _require(final_source.read_bytes() == source_path.read_bytes(), "source changed during the ABBA run")
    inputs = _validate_manifest(
        provenance / "input_manifest.sha256",
        expected_digest=run["input_manifest_sha256"],
        relative_base=None,
        label="input manifest",
    )
    _require(
        sum(entry["sha256"] == run["particle_stack_sha256"] for entry in inputs["entries"]) == 1,
        "input manifest does not uniquely seal the GF46 particle stack",
    )
    gate_ledger = provenance / "qualified_gate.SHA256SUMS"
    gate_ledger_digest = provenance / "qualified_gate.SHA256SUMS.sha256"
    _require(gate_ledger.is_file() and gate_ledger_digest.is_file(), "qualified gate ledgers are missing")
    _require(_sha256(gate_ledger) == run["qualified_gate_sha256sums_sha256"], "qualified gate ledger digest differs")
    _require(
        _read_single_line(gate_ledger_digest, "qualified gate ledger digest").split()[0]
        == run["qualified_gate_sha256sums_sha256"],
        "qualified gate digest ledger differs",
    )
    cuda_sha, cuda_path = _parse_sha_line(provenance / "qualified_cuda.sha256", "CUDA binary", root=root)
    bind_sha, bind_path = _parse_sha_line(provenance / "relion_bind.sha256", "RELION binding", root=root)
    interpreter = _validate_external_sha_line(
        provenance / "interpreter.sha256",
        run["interpreter_sha256"],
        "interpreter",
    )
    _require(cuda_sha == run["cuda_sha256"], "CUDA digest differs from run.json")
    _require(bind_sha == run["relion_bind_sha256"], "RELION binding digest differs from run.json")
    nsys_binary = _validate_external_sha_line(provenance / "nsys.sha256", run["nsys_sha256"], "Nsight binary")
    cusparse = _validate_external_sha_line(
        provenance / "cusparse.sha256", run["cusparse_sha256"], "cuSPARSE library"
    )
    execution = _validate_execution_order(provenance / "execution_order.tsv", root)
    run_dirs = {path.name for path in (root / "runs").iterdir() if path.is_dir()}
    _require(run_dirs == set(ARM_LABELS), f"run-directory topology differs: {sorted(run_dirs)}")
    commands = {}
    caches = {}
    for label, mode, _repeat in ARM_SPECS:
        command = _validate_command(provenance / f"{label}_command.sh", root=root, label=label, mode=mode)
        commands[label] = command
        caches[label] = _validate_jax_cache(root, label, command["jax_cache"])
    _require(len({value["path"] for value in caches.values()}) == 4, "JAX caches are not private per arm")
    return run, {
        "run_json_sha256": _sha256(run_path),
        "source_manifest": source,
        "input_manifest": inputs,
        "cuda_binary": {"path": str(cuda_path), "sha256": cuda_sha},
        "relion_binding": {"path": str(bind_path), "sha256": bind_sha},
        "interpreter": interpreter,
        "nsys_binary": nsys_binary,
        "cusparse_library": cusparse,
        "git_repository": {"path": str(repo), "resolved_head": resolved_head, "resolved_tree": resolved_tree},
        "execution_order": execution,
        "commands": commands,
        "jax_caches": caches,
        "final_source_manifest_state": "verified" if final_source.is_file() else "pending_runner_seal",
    }


def _interval_union(intervals: list[tuple[int, int]]) -> int:
    return int(nsys_sqlite._union_ns(intervals))


def _clipped_union(intervals: list[tuple[int, int]], bounds: tuple[int, int]) -> int:
    start_bound, end_bound = bounds
    clipped = [
        (max(start, start_bound), min(end, end_bound))
        for start, end in intervals
        if min(end, end_bound) > max(start, start_bound)
    ]
    return _interval_union(clipped)


def _inside(interval: tuple[int, int], bounds: tuple[int, int]) -> bool:
    return bounds[0] <= interval[0] and interval[1] <= bounds[1]


def _load_nsight(root: Path, label: str, mode: str) -> dict[str, Any]:
    nsight = root / "nsight"
    sqlite_path = nsight / f"{label}.sqlite"
    summary_path = nsight / f"{label}_summary.json"
    report_path = nsight / f"{label}_it181_warm.nsys-rep"
    _require(sqlite_path.is_file(), f"missing {label} Nsight SQLite")
    _require(summary_path.is_file(), f"missing {label} Nsight summary")
    _require(report_path.is_file(), f"missing {label} Nsight report")
    summary = _load_json(summary_path, f"{label} Nsight summary")
    _require(summary.get("schema") == NSIGHT_SCHEMA, f"{label} Nsight schema differs")
    summary_sqlite = _resolved_inside(Path(str(summary.get("sqlite", ""))), root, f"{label} summary SQLite")
    _require(summary_sqlite == sqlite_path.resolve(), f"{label} Nsight SQLite path differs")
    sqlite_sha = _sha256(sqlite_path)
    try:
        connection = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            tables = nsys_sqlite._tables(connection)
            _require("CUPTI_ACTIVITY_KIND_KERNEL" in tables, f"{label} has no kernel table")
            _require("NVTX_EVENTS" in tables, f"{label} has no NVTX table")
            strings = nsys_sqlite._string_ids(connection, tables)
            kernel_rows = list(connection.execute('SELECT * FROM "CUPTI_ACTIVITY_KIND_KERNEL"'))
            nvtx_rows = list(connection.execute('SELECT * FROM "NVTX_EVENTS"'))
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise RawCacheSetupError(f"cannot read {label} Nsight SQLite: {exc}") from exc
    _require(_sha256(sqlite_path) == sqlite_sha, f"{label} Nsight SQLite changed during analysis")
    _require(kernel_rows, f"{label} has no CUDA kernels")
    kernels: list[tuple[int, int]] = []
    coarse: list[tuple[int, int]] = []
    devices: set[int] = set()
    signature_counts: dict[str, int] = {}
    shape_columns = tuple(
        column
        for column in ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ")
        if column in kernel_rows[0].keys()
    )
    for row in kernel_rows:
        start, end = int(row["start"]), int(row["end"])
        _require(end >= start, f"{label} has a negative kernel interval")
        interval = (start, end)
        kernels.append(interval)
        devices.add(int(row["deviceId"]) if "deviceId" in row.keys() else 0)
        name = nsys_sqlite._name(
            row,
            candidates=("shortName", "demangledName", "mangledName", "name"),
            strings=strings,
        )
        signature = json.dumps(
            {"name": name, "device": int(row["deviceId"]) if "deviceId" in row.keys() else 0,
             **{column: int(row[column]) for column in shape_columns}},
            sort_keys=True,
        )
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        if name == COARSE_KERNEL_NAME:
            coarse.append(interval)
    _require(len(devices) == 1, f"{label} Nsight GPU topology differs")
    _require(coarse, f"{label} has no coarse kernels")
    ranges: dict[str, list[tuple[int, int]]] = {
        name: [] for name in (GETITEM_RANGE, LOADER_RANGE, PASS1_RANGE, PASS2_RANGE)
    }
    for row in nvtx_rows:
        if "end" not in row.keys() or row["end"] is None:
            continue
        start, end = int(row["start"]), int(row["end"])
        _require(end >= start, f"{label} has a negative NVTX interval")
        text_value = row["text"] if "text" in row.keys() else None
        if text_value is None and "textId" in row.keys() and row["textId"] is not None:
            text_value = strings.get(int(row["textId"]))
        name = str(text_value) if text_value is not None else ""
        if name in ranges:
            ranges[name].append((start, end))
    _require(len(ranges[PASS1_RANGE]) == 1, f"{label} pass-1 NVTX topology differs")
    _require(len(ranges[PASS2_RANGE]) == 1, f"{label} pass-2 NVTX topology differs")
    _require(len(ranges[GETITEM_RANGE]) == 2_000, f"{label} getitem NVTX count differs")
    expected_loader_count = 2_000 if mode == "off" else 1
    _require(len(ranges[LOADER_RANGE]) == expected_loader_count, f"{label} loader NVTX count differs")
    pass_bounds = {name: ranges[name][0] for name in STAGE_RANGES}
    loader_inside = {
        name: sum(_inside(interval, pass_bounds[name]) for interval in ranges[LOADER_RANGE])
        for name in STAGE_RANGES
    }
    getitem_inside = {
        name: sum(_inside(interval, pass_bounds[name]) for interval in ranges[GETITEM_RANGE])
        for name in STAGE_RANGES
    }
    if mode == "off":
        _require(loader_inside == {PASS1_RANGE: 1_000, PASS2_RANGE: 1_000}, f"{label} loader pass split differs")
        _require(getitem_inside == {PASS1_RANGE: 1_000, PASS2_RANGE: 1_000}, f"{label} getitem pass split differs")
    else:
        _require(loader_inside == {PASS1_RANGE: 0, PASS2_RANGE: 0}, f"{label} cached loader entered a pass")
        _require(ranges[LOADER_RANGE][0][1] <= pass_bounds[PASS1_RANGE][0], f"{label} preload did not finish before pass 1")
        _require(getitem_inside == {PASS1_RANGE: 1_000, PASS2_RANGE: 1_000}, f"{label} getitem pass split differs")
    gpu_union_ns = _interval_union(kernels)
    gpu_sum_ns = sum(end - start for start, end in kernels)
    coarse_sum_ns = sum(end - start for start, end in coarse)
    coarse_union_ns = _interval_union(coarse)
    stage_rows = {}
    for name, bounds in pass_bounds.items():
        duration_ns = bounds[1] - bounds[0]
        kernel_union_ns = _clipped_union(kernels, bounds)
        _require(kernel_union_ns <= duration_ns, f"{label} {name} clipped GPU union exceeds the stage")
        stage_rows[name] = {
            "duration_s": duration_ns / 1e9,
            "gpu_kernel_union_s": kernel_union_ns / 1e9,
            "no_kernel_s": (duration_ns - kernel_union_ns) / 1e9,
            "loader_count": loader_inside[name],
            "getitem_count": getitem_inside[name],
            "loader_union_s": _clipped_union(ranges[LOADER_RANGE], bounds) / 1e9,
            "getitem_union_s": _clipped_union(ranges[GETITEM_RANGE], bounds) / 1e9,
        }
    devices_summary = summary.get("devices")
    device_id = str(next(iter(devices)))
    _require(isinstance(devices_summary, dict) and set(devices_summary) == {device_id}, f"{label} summary GPU topology differs")
    _require(devices_summary[device_id].get("kernel_count") == len(kernels), f"{label} summary kernel count differs")
    _require(devices_summary[device_id].get("gpu_busy_ns") == gpu_union_ns, f"{label} summary GPU union differs")
    coarse_rows = [
        row for row in summary.get("kernels", []) if isinstance(row, dict) and row.get("name") == COARSE_KERNEL_NAME
    ]
    _require(len(coarse_rows) == 1, f"{label} summary coarse row differs")
    _require(coarse_rows[0].get("count") == len(coarse), f"{label} summary coarse count differs")
    _require(coarse_rows[0].get("total_ns") == coarse_sum_ns, f"{label} summary coarse sum differs")
    return {
        "sqlite_path": str(sqlite_path.resolve()),
        "sqlite_sha256": sqlite_sha,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "nsys_report_path": str(report_path.resolve()),
        "nsys_report_sha256": _sha256(report_path),
        "getitem_count": len(ranges[GETITEM_RANGE]),
        "loader_count": len(ranges[LOADER_RANGE]),
        "getitem_union_s": _interval_union(ranges[GETITEM_RANGE]) / 1e9,
        "loader_union_s": _interval_union(ranges[LOADER_RANGE]) / 1e9,
        "gpu_kernel_union_s": gpu_union_ns / 1e9,
        "gpu_kernel_sum_s": gpu_sum_ns / 1e9,
        "coarse_kernel_sum_s": coarse_sum_ns / 1e9,
        "coarse_kernel_union_s": coarse_union_ns / 1e9,
        "coarse_kernel_launch_count": len(coarse),
        "stages": stage_rows,
        "stage_no_kernel_s": sum(row["no_kernel_s"] for row in stage_rows.values()),
        "kernel_signature_counts": signature_counts,
        "summary_crosscheck_exact": True,
    }


def _validate_cache_event(event: Any, label: str) -> dict[str, Any]:
    _require(isinstance(event, dict), f"{label} cache event is invalid")
    expected = {
        "num_images": EXPECTED_CACHE_IMAGES,
        "image_size": EXPECTED_IMAGE_SIZE,
        "dtype": EXPECTED_CACHE_DTYPE,
        "estimated_bytes": EXPECTED_CACHE_BYTES,
        "cached_before": False,
        "cached_after": True,
        "cached_nbytes": EXPECTED_CACHE_BYTES,
    }
    mismatches = {key: (event.get(key), value) for key, value in expected.items() if event.get(key) != value}
    _require(not mismatches, f"{label} cache admission differs: {mismatches}")
    _require(isinstance(event.get("loader_type"), str) and event["loader_type"], f"{label} loader type is invalid")
    elapsed = _finite_number(event.get("elapsed_s"), f"{label} cache preload time", positive=True)
    return {**event, "elapsed_s": elapsed}


def _validate_cache_audit(phase: dict[str, Any], *, label: str, mode: str, phase_name: str) -> list[dict[str, Any]]:
    audit = phase.get("raw_image_cache_audit")
    _require(isinstance(audit, dict), f"{label} {phase_name} cache audit is missing")
    _require(audit.get("mode") == mode, f"{label} {phase_name} cache mode differs")
    _require(float(audit.get("max_gb", -1)) == EXPECTED_CACHE_MAX_GB, f"{label} {phase_name} cache cap differs")
    events = audit.get("load_all_events")
    _require(isinstance(events, list), f"{label} {phase_name} cache events are invalid")
    expected_count = 0 if mode == "off" else 1
    _require(len(events) == expected_count, f"{label} {phase_name} load_all event count differs")
    return [_validate_cache_event(event, f"{label} {phase_name}") for event in events]


def _process_metrics(value: Any, label: str) -> dict[str, float]:
    _require(isinstance(value, dict), f"{label} process resources are missing")
    before, after, delta = value.get("before"), value.get("after"), value.get("delta")
    _require(all(isinstance(item, dict) for item in (before, after, delta)), f"{label} process resource topology differs")
    assert isinstance(before, dict) and isinstance(after, dict) and isinstance(delta, dict)
    user = _finite_number(delta.get("user_cpu_s"), f"{label} user CPU")
    system = _finite_number(delta.get("system_cpu_s"), f"{label} system CPU")
    before_hwm = _finite_number(before.get("high_water_rss_kb"), f"{label} before HWM")
    after_hwm = _finite_number(after.get("high_water_rss_kb"), f"{label} after HWM")
    _require(after_hwm >= before_hwm, f"{label} HWM decreased")
    current_after = _finite_number(after.get("current_rss_kb"), f"{label} current RSS")
    _require(current_after <= after_hwm, f"{label} current RSS exceeds HWM")
    proc_io = delta.get("proc_io")
    _require(isinstance(proc_io, dict), f"{label} process I/O delta is missing")
    io_values = {str(key): _finite_number(item, f"{label} I/O {key}") for key, item in proc_io.items()}
    _require("read_bytes" in io_values and "write_bytes" in io_values, f"{label} process I/O fields differ")
    return {
        "process_user_cpu_s": user,
        "process_system_cpu_s": system,
        "process_total_cpu_s": user + system,
        "process_read_bytes": io_values["read_bytes"],
        "process_write_bytes": io_values["write_bytes"],
        "process_hwm_increment_bytes": (after_hwm - before_hwm) * 1024.0,
        "process_current_rss_after_bytes": current_after * 1024.0,
        "process_high_water_rss_after_bytes": after_hwm * 1024.0,
        "process_io": io_values,
    }


def _validate_schedule(schedule: Any, label: str) -> dict[str, Any]:
    _require(isinstance(schedule, dict), f"{label} schedule is missing")
    expected_keys = {
        "current_size",
        "healpix_order",
        "n_rotations",
        "n_translations",
        "subset_size",
        "random_perturbation",
    }
    _require(set(schedule) == expected_keys, f"{label} schedule fields differ")
    for key in expected_keys - {"random_perturbation", "subset_size"}:
        _require(_integer(schedule[key], f"{label} {key}") > 0, f"{label} {key} must be positive")
    _require(schedule["current_size"] == EXPECTED_IMAGE_SIZE, f"{label} current size differs from GF46")
    _require(schedule["subset_size"] is None, f"{label} GF46 subset size must be null (full dataset)")
    perturbation = schedule["random_perturbation"]
    _require(
        isinstance(perturbation, (int, float))
        and not isinstance(perturbation, bool)
        and math.isfinite(float(perturbation)),
        f"{label} random perturbation must be finite",
    )
    _require(schedule == EXPECTED_SCHEDULE, f"{label} GF46 iteration-181 schedule differs")
    return dict(schedule)


def _load_phase_outputs(root: Path, label: str, phase_name: str, phase: dict[str, Any]) -> dict[str, Any]:
    phase_root = root / "runs" / label / "profile" / phase_name
    meta_path = phase_root / "run_it181_recovar_meta.json"
    recorded_meta = _resolved_inside(Path(str(phase.get("meta_path", ""))), root, f"{label} {phase_name} metadata")
    _require(recorded_meta == meta_path.resolve(), f"{label} {phase_name} metadata path differs")
    _require(_sha256(meta_path) == phase.get("meta_sha256"), f"{label} {phase_name} metadata digest differs")
    metadata = _load_json(meta_path, f"{label} {phase_name} metadata")
    schedule = _validate_schedule(phase.get("schedule"), f"{label} {phase_name}")
    for key, value in schedule.items():
        _require(key in metadata, f"{label} {phase_name} metadata lacks schedule field {key}")
        _require(_values_equal(metadata.get(key), value), f"{label} {phase_name} metadata schedule differs for {key}")
    _require(metadata.get("joint_halfset_particle_stream") is True, f"{label} {phase_name} is not joint-halfset")
    _require(_values_equal(metadata.get("halfset_ids"), [0, 1]), f"{label} {phase_name} halfset IDs differ")
    profile_keys = sorted(key for key in metadata if re.fullmatch(r"halfset_\d+_profile_summary", str(key)))
    _require(profile_keys == ["halfset_0_profile_summary"], f"{label} {phase_name} halfset profile topology differs")
    translations = _coarse_translation_count(metadata, f"{label} {phase_name}")
    half_profile = metadata["halfset_0_profile_summary"]
    _require(isinstance(half_profile, dict), f"{label} {phase_name} halfset profile is invalid")
    selector = _validate_selector_audit(
        half_profile.get("coarse_selector_audit"),
        workers=8,
        atomic=True,
        translations=translations,
        label=f"{label} {phase_name}",
    )
    meta_files = sorted(phase_root.glob("run_it*_recovar_meta.json"))
    _require(meta_files == [meta_path], f"{label} {phase_name} iteration metadata topology differs")
    star_path = phase_root / "run_it181_data.star"
    map_path = phase_root / "run_it181_class001.mrc"
    star = _load_star(star_path, f"{label} {phase_name} particle STAR")
    _require(len(star["particles"]) == EXPECTED_CACHE_IMAGES, f"{label} {phase_name} particle count differs")
    return {
        "metadata": metadata,
        "metadata_sha256": _sha256(meta_path),
        "schedule": schedule,
        "selector_audit": selector,
        "star": star,
        "star_sha256": _sha256(star_path),
        "map": _load_map(map_path, f"{label} {phase_name} map"),
        "map_sha256": _sha256(map_path),
    }


def _load_arm(root: Path, spec: tuple[str, str, int]) -> dict[str, Any]:
    label, mode, repeat = spec
    run_root = root / "runs" / label
    for artifact_name in ("process.time", "runner.stdout"):
        artifact = run_root / artifact_name
        _require(artifact.is_file() and artifact.stat().st_size > 0, f"{label} {artifact_name} is missing or empty")
    _require((run_root / "runner.stderr").is_file(), f"{label} runner.stderr is missing")
    profile_root = run_root / "profile"
    summary_path = profile_root / "profile_summary.json"
    summary = _load_json(summary_path, f"{label} profile summary")
    expected = {
        "schema": PROFILE_SCHEMA,
        "classification": "diagnostic_performance_only",
        "checkpoint_iteration": 180,
        "profiled_iteration": 181,
        "nr_iter_schedule": 200,
        "cuda_profiler_range": True,
        "raw_image_cache_audit_enabled": True,
        "exact_local_bucket_radix": 4,
        "exact_local_physical_order_chunk_size": 0,
    }
    mismatches = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    _require(not mismatches, f"{label} profile summary differs: {mismatches}")
    _require(isinstance(summary.get("cold"), dict) and isinstance(summary.get("warm"), dict), f"{label} phases differ")
    cold, warm = summary["cold"], summary["warm"]
    cold_events = _validate_cache_audit(cold, label=label, mode=mode, phase_name="cold")
    warm_events = _validate_cache_audit(warm, label=label, mode=mode, phase_name="warm")
    cold_outputs = _load_phase_outputs(root, label, "cold", cold)
    warm_outputs = _load_phase_outputs(root, label, "warm", warm)
    _require(cold_outputs["schedule"] == warm_outputs["schedule"], f"{label} cold/warm schedule differs")
    wall = _finite_number(warm.get("wall_s"), f"{label} warm wall", positive=True)
    iteration = _numeric_scalars(warm.get("iteration_profile"))
    _require("expectation_time_s" in iteration, f"{label} warm expectation timing is missing")
    sparse = _numeric_scalars(warm_outputs["metadata"].get("sparse_pass2_profile_summary"))
    _require("pass1_time_s" in sparse and "pass2_time_s" in sparse, f"{label} pass timings are missing")
    process = _process_metrics(warm.get("process_resources"), f"{label} warm")
    cold_process = _process_metrics(cold.get("process_resources"), f"{label} cold")
    nsight = _load_nsight(root, label, mode)
    performance = {
        "warm_wall_s": wall,
        "warm_expectation_s": iteration["expectation_time_s"],
        "pass1_s": sparse["pass1_time_s"],
        "pass2_s": sparse["pass2_time_s"],
        "stage_no_kernel_s": nsight["stage_no_kernel_s"],
        "approx_expectation_no_kernel_s": max(iteration["expectation_time_s"] - nsight["gpu_kernel_union_s"], 0.0),
        "getitem_union_s": nsight["getitem_union_s"],
        "loader_union_s": nsight["loader_union_s"],
        "gpu_kernel_union_s": nsight["gpu_kernel_union_s"],
        "gpu_kernel_sum_s": nsight["gpu_kernel_sum_s"],
        "coarse_kernel_sum_s": nsight["coarse_kernel_sum_s"],
        "coarse_kernel_union_s": nsight["coarse_kernel_union_s"],
        "coarse_kernel_launch_count": nsight["coarse_kernel_launch_count"],
        **process,
    }
    admission_path = root / "runs" / label / "cache_admission.json"
    _require(admission_path.is_file(), f"{label} cache-admission ledger is missing")
    admission = _load_json(admission_path, f"{label} cache admission")
    expected_admission = {
        "schema": "recovar.vdam_raw_cache_admission.v1",
        "label": label,
        "mode": mode,
        "max_gb": EXPECTED_CACHE_MAX_GB,
        "expected_bytes": EXPECTED_CACHE_BYTES,
    }
    admission_mismatches = {
        key: (admission.get(key), value)
        for key, value in expected_admission.items()
        if admission.get(key) != value
    }
    _require(not admission_mismatches, f"{label} cache-admission ledger differs: {admission_mismatches}")
    phases = admission.get("phases")
    _require(isinstance(phases, dict) and set(phases) == {"cold", "warm"}, f"{label} cache-admission phases differ")
    for phase_name, events in (("cold", cold_events), ("warm", warm_events)):
        phase_admission = phases[phase_name]
        _require(isinstance(phase_admission, dict), f"{label} {phase_name} admission is invalid")
        expected_phase = {
            "load_all_count": len(events),
            "load_all_events": events,
            "admitted": mode == "auto",
        }
        _require(phase_admission == expected_phase, f"{label} {phase_name} cache-admission ledger differs")
    return {
        "label": label,
        "mode": mode,
        "repeat": repeat,
        "profile_summary_sha256": _sha256(summary_path),
        "cache_admission_sha256": _sha256(admission_path),
        "cache_events": {"cold": cold_events, "warm": warm_events},
        "cold": cold_outputs,
        "warm": warm_outputs,
        "cold_process": cold_process,
        "nsight": nsight,
        "performance": performance,
    }


def _within_envelope(value: float, envelope: float) -> bool:
    return bool(value <= np.nextafter(envelope, math.inf) if envelope > 0.0 else value == 0.0)


def _science(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # Cold and warm must themselves make the same discrete decision for every
    # arm.  This is a decision result (NO_GO), not malformed evidence.
    phase_checks = {}
    all_exact = True
    for label in ARM_LABELS:
        arm = arms[label]
        meta = {
            key: key in arm["cold"]["metadata"]
            and key in arm["warm"]["metadata"]
            and _values_equal(arm["cold"]["metadata"][key], arm["warm"]["metadata"][key])
            for key in DISCRETE_META_KEYS
        }
        star = _star_equal(arm["cold"]["star"], arm["warm"]["star"])
        exact = all(meta.values()) and star
        all_exact &= exact
        phase_checks[label] = {"metadata_exact": meta, "particle_star_exact": star, "all_exact": exact}
    warm_panel_checks = {}
    reference = arms[ARM_LABELS[0]]["warm"]
    for label in ARM_LABELS[1:]:
        candidate = arms[label]["warm"]
        meta = {
            key: key in reference["metadata"]
            and key in candidate["metadata"]
            and _values_equal(reference["metadata"][key], candidate["metadata"][key])
            for key in DISCRETE_META_KEYS
        }
        star = _star_equal(reference["star"], candidate["star"])
        exact = all(meta.values()) and star
        all_exact &= exact
        warm_panel_checks[f"{ARM_LABELS[0]}__{label}"] = {
            "metadata_exact": meta,
            "particle_star_exact": star,
            "all_exact": exact,
        }
    repeat_deltas = {
        mode: _map_delta(arms[f"cache_{mode}_1"]["warm"]["map"], arms[f"cache_{mode}_2"]["warm"]["map"])
        for mode in MODES
    }
    repeat_envelopes = {
        "relative_l2": max(row["relative_l2"] for row in repeat_deltas.values()),
        "max_abs": max(row["max_abs"] for row in repeat_deltas.values()),
        "abs_relative_scale_drift": max(abs(row["relative_scale_drift"]) for row in repeat_deltas.values()),
    }
    repeat_envelope = repeat_envelopes["relative_l2"]
    signed_repeat_envelope = max(abs(row["signed_mean_over_delta_rms"]) for row in repeat_deltas.values())
    cross = {}
    maps_bounded = True
    nondirectional = True
    signed_cross_drifts: list[float] = []
    for repeat in (1, 2):
        off = arms[f"cache_off_{repeat}"]
        auto = arms[f"cache_auto_{repeat}"]
        meta = {
            key: key in off["warm"]["metadata"]
            and key in auto["warm"]["metadata"]
            and _values_equal(off["warm"]["metadata"][key], auto["warm"]["metadata"][key])
            for key in DISCRETE_META_KEYS
        }
        star = _star_equal(off["warm"]["star"], auto["warm"]["star"])
        exact = all(meta.values()) and star
        all_exact &= exact
        delta = _map_delta(off["warm"]["map"], auto["warm"]["map"])
        bounded_metrics = {
            "relative_l2": _within_envelope(delta["relative_l2"], repeat_envelopes["relative_l2"]),
            "max_abs": _within_envelope(delta["max_abs"], repeat_envelopes["max_abs"]),
            "abs_relative_scale_drift": _within_envelope(
                abs(delta["relative_scale_drift"]), repeat_envelopes["abs_relative_scale_drift"]
            ),
        }
        bounded = all(bounded_metrics.values())
        signed_ok = _within_envelope(abs(delta["signed_mean_over_delta_rms"]), signed_repeat_envelope)
        signed_cross_drifts.append(delta["signed_mean"])
        maps_bounded &= bounded
        nondirectional &= signed_ok
        delta.update(
            {
                "repeat_envelope_relative_l2": repeat_envelope,
                "within_repeat_envelope": bounded,
                "bounded_metrics": bounded_metrics,
                "repeat_envelopes": repeat_envelopes,
                "repeat_envelope_abs_signed_mean_over_delta_rms": signed_repeat_envelope,
                "nondirectional_signed_drift": signed_ok,
            }
        )
        cross[f"cache_off_{repeat}__cache_auto_{repeat}"] = {
            "metadata_exact": meta,
            "particle_star_exact": star,
            "all_discrete_exact": exact,
            "map": delta,
        }
    cross_pair_nondirectional = signed_cross_drifts[0] * signed_cross_drifts[1] <= 0.0
    nondirectional &= cross_pair_nondirectional
    return {
        "discrete_meta_keys": list(DISCRETE_META_KEYS),
        "cold_warm_checks": phase_checks,
        "warm_all_arm_checks": warm_panel_checks,
        "repeat_map_deltas": repeat_deltas,
        "repeat_relative_l2_envelope": repeat_envelope,
        "repeat_map_envelopes": repeat_envelopes,
        "repeat_abs_signed_drift_envelope": signed_repeat_envelope,
        "cross_pair_signed_means": signed_cross_drifts,
        "cross_pair_signed_drift_opposes_or_is_zero": cross_pair_nondirectional,
        "cross_mode_comparisons": cross,
        "all_particle_star_and_discrete_metadata_exact": all_exact,
        "all_cross_mode_maps_within_repeat_envelope": maps_bounded,
        "all_cross_mode_signed_drift_nondirectional": nondirectional,
        "pass": all_exact and maps_bounded and nondirectional,
    }


def _percent_improvement(off: float, auto: float) -> float:
    _require(off > 0.0 and auto >= 0.0, "performance denominator is invalid")
    return 100.0 * (off - auto) / off


def _performance(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    modes = {}
    for mode in MODES:
        rows = [arms[f"cache_{mode}_{repeat}"]["performance"] for repeat in (1, 2)]
        raw = {f"R{repeat}": {name: rows[repeat - 1][name] for name in PERFORMANCE_METRICS} for repeat in (1, 2)}
        medians = {name: float(median([float(row[name]) for row in rows])) for name in PERFORMANCE_METRICS}
        spans = {name: abs(float(rows[1][name]) - float(rows[0][name])) for name in PERFORMANCE_METRICS}
        modes[mode] = {"raw": raw, "median": medians, "repeat_span": spans}
    off, auto = modes["off"]["median"], modes["auto"]["median"]
    improvement = {name: float(off[name] - auto[name]) for name in PERFORMANCE_METRICS}
    improvement_percent = {
        name: _percent_improvement(off[name], auto[name]) if off[name] > 0 else 0.0 for name in PERFORMANCE_METRICS
    }
    adjacent = {}
    for repeat in (1, 2):
        off_row = arms[f"cache_off_{repeat}"]["performance"]
        auto_row = arms[f"cache_auto_{repeat}"]["performance"]
        adjacent[f"R{repeat}"] = {
            "warm_wall_s": {"off": off_row["warm_wall_s"], "auto": auto_row["warm_wall_s"], "auto_faster": auto_row["warm_wall_s"] < off_row["warm_wall_s"]},
            "warm_expectation_s": {"off": off_row["warm_expectation_s"], "auto": auto_row["warm_expectation_s"], "auto_faster": auto_row["warm_expectation_s"] < off_row["warm_expectation_s"]},
        }
    max_span = {
        name: max(modes["off"]["repeat_span"][name], modes["auto"]["repeat_span"][name])
        for name in PERFORMANCE_METRICS
    }
    off_hwm_median = off["process_high_water_rss_after_bytes"]
    auto_hwm_median = auto["process_high_water_rss_after_bytes"]
    hwm_overhead = auto_hwm_median - off_hwm_median
    hwm_limit = EXPECTED_CACHE_BYTES + HWM_SLACK_BYTES
    hwm_ok = hwm_overhead <= hwm_limit
    launch_repeat_envelope = max_span["coarse_kernel_launch_count"]
    launch_ok = all(
        arms[label]["performance"]["coarse_kernel_launch_count"] == EXPECTED_COARSE_LAUNCHES
        for label in ARM_LABELS
    )
    kernel_signatures_exact = all(
        arms[label]["nsight"]["kernel_signature_counts"] == arms[ARM_LABELS[0]]["nsight"]["kernel_signature_counts"]
        for label in ARM_LABELS[1:]
    )
    summed_work_bounded = {}
    for metric in ("coarse_kernel_sum_s", "gpu_kernel_sum_s"):
        envelope = max_span[metric]
        pairs = {
            f"R{repeat}": abs(
                arms[f"cache_off_{repeat}"]["performance"][metric]
                - arms[f"cache_auto_{repeat}"]["performance"][metric]
            )
            for repeat in (1, 2)
        }
        summed_work_bounded[metric] = {
            "repeat_envelope_s": envelope,
            "cross_mode_absolute_deltas_s": pairs,
            "within_repeat_envelope": all(_within_envelope(value, envelope) for value in pairs.values()),
        }
    preload_raw = {
        label: {
            phase: (events[0]["elapsed_s"] if events else 0.0)
            for phase, events in arms[label]["cache_events"].items()
        }
        for label in ARM_LABELS
    }
    warm_preloads = [preload_raw[f"cache_auto_{repeat}"]["warm"] for repeat in (1, 2)]
    median_preload = float(median(warm_preloads))
    post_preload_auto_wall = auto["warm_wall_s"] - median_preload
    gross_post_preload_saving = off["warm_wall_s"] - post_preload_auto_wall
    break_even = median_preload / gross_post_preload_saving if gross_post_preload_saving > 0.0 else math.inf
    gates = {
        "both_adjacent_warm_wall_faster": all(row["warm_wall_s"]["auto_faster"] for row in adjacent.values()),
        "both_adjacent_expectation_faster": all(row["warm_expectation_s"]["auto_faster"] for row in adjacent.values()),
        "median_wall_at_least_5_percent_faster": improvement_percent["warm_wall_s"] >= MIN_PERCENT_WIN,
        "median_expectation_at_least_5_percent_faster": improvement_percent["warm_expectation_s"] >= MIN_PERCENT_WIN,
        "median_expectation_at_least_0_30s_faster": improvement["warm_expectation_s"] >= MIN_EXPECTATION_WIN_S,
        "wall_improvement_exceeds_repeat_span": improvement["warm_wall_s"] > max_span["warm_wall_s"],
        "expectation_improvement_exceeds_repeat_span": improvement["warm_expectation_s"] > max_span["warm_expectation_s"],
        "stage_no_kernel_at_least_0_30s_lower": improvement["stage_no_kernel_s"] >= MIN_NO_KERNEL_WIN_S,
        "coarse_launches_1000_within_repeat_envelope": launch_ok,
        "cuda_kernel_signature_count_topology_exact": kernel_signatures_exact,
        "coarse_summed_work_within_repeat_envelope": summed_work_bounded["coarse_kernel_sum_s"]["within_repeat_envelope"],
        "total_gpu_summed_work_within_repeat_envelope": summed_work_bounded["gpu_kernel_sum_s"]["within_repeat_envelope"],
        "hwm_auto_minus_off_overhead_within_cache_plus_128mib": hwm_ok,
    }
    return {
        "metric_order": list(PERFORMANCE_METRICS),
        "arms": {label: arms[label]["performance"] for label in ARM_LABELS},
        "modes": modes,
        "adjacent_pairs": adjacent,
        "median_off_minus_auto": improvement,
        "median_percent_improvement": improvement_percent,
        "max_repeat_span": max_span,
        "preload": {
            "raw_s": preload_raw,
            "warm_auto_median_s": median_preload,
            "post_preload_auto_wall_median_s": post_preload_auto_wall,
            "gross_post_preload_wall_saving_s": gross_post_preload_saving,
            "break_even_iterations": break_even,
            "break_even_iterations_ceiling": math.ceil(break_even) if math.isfinite(break_even) else None,
        },
        "hwm": {
            "off_median_high_water_rss_bytes": off_hwm_median,
            "auto_median_high_water_rss_bytes": auto_hwm_median,
            "auto_minus_off_median_hwm_bytes": hwm_overhead,
            "limit_bytes": hwm_limit,
        },
        "coarse_launches": {"expected_per_gpu": EXPECTED_COARSE_LAUNCHES, "repeat_envelope": launch_repeat_envelope},
        "kernel_signature_counts": {label: arms[label]["nsight"]["kernel_signature_counts"] for label in ARM_LABELS},
        "summed_gpu_work": summed_work_bounded,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _markdown(report: dict[str, Any]) -> str:
    status = report["decision"]["status"]
    science = report["science"]
    performance = report["performance"]
    lines = [
        f"# {status} — VDAM raw-image-cache ABBA gate",
        "",
        "Scope: diagnostic iteration 180 → 181 only; this report does not promote a science result.",
        "",
        "## Decision gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for name, value in report["decision"]["gates"].items():
        lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    lines.extend(("", "## Crossed performance", "", "| Metric | OFF R1 | OFF R2 | OFF median | OFF span | AUTO R1 | AUTO R2 | AUTO median | AUTO span | OFF−AUTO | Improvement |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for metric in PERFORMANCE_METRICS:
        off = performance["modes"]["off"]
        auto = performance["modes"]["auto"]
        lines.append(
            f"| {metric} | {off['raw']['R1'][metric]:.6g} | {off['raw']['R2'][metric]:.6g} | "
            f"{off['median'][metric]:.6g} | {off['repeat_span'][metric]:.6g} | "
            f"{auto['raw']['R1'][metric]:.6g} | {auto['raw']['R2'][metric]:.6g} | "
            f"{auto['median'][metric]:.6g} | {auto['repeat_span'][metric]:.6g} | "
            f"{performance['median_off_minus_auto'][metric]:.6g} | "
            f"{performance['median_percent_improvement'][metric]:+.2f}% |"
        )
    preload = performance["preload"]
    lines.extend(
        (
            "",
            "## Preload and break-even",
            "",
            f"- Warm AUTO preload median: `{preload['warm_auto_median_s']:.6f} s`.",
            f"- Gross post-preload wall saving: `{preload['gross_post_preload_wall_saving_s']:.6f} s/iteration`.",
            f"- Break-even: `{preload['break_even_iterations']:.3f}` iterations "
            f"(ceiling `{preload['break_even_iterations_ceiling']}`).",
            "",
            "## Correctness",
            "",
            f"- Particle STAR and discrete metadata exact: `{science['all_particle_star_and_discrete_metadata_exact']}`.",
            f"- Cross-mode maps within repeat envelope: `{science['all_cross_mode_maps_within_repeat_envelope']}`.",
            f"- Signed map drift nondirectional: `{science['all_cross_mode_signed_drift_nondirectional']}`.",
            f"- Repeat relative-L2 envelope: `{science['repeat_relative_l2_envelope']:.6e}`.",
            "",
            "## Compact provenance",
            "",
            f"- Root: `{report['provenance']['root']}`",
            f"- Slurm job: `{report['provenance']['job_id']}`",
            f"- Commit/tree: `{report['provenance']['git_head']}` / `{report['provenance']['git_tree']}`",
            f"- GPU: `{report['provenance']['gpu_name']}` (`{report['provenance']['gpu_uuid']}`)",
            f"- Analyzer SHA-256: `{report['provenance']['analyzer_source_sha256']}`",
            "",
        )
    )
    return "\n".join(lines)


def analyze(root: Path, *, repo: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    repo = (repo or Path(__file__).resolve().parents[1]).resolve()
    run, provenance = _validate_provenance(root, repo)
    expected_nsight = {
        *(f"{label}.sqlite" for label in ARM_LABELS),
        *(f"{label}_summary.json" for label in ARM_LABELS),
        *(f"{label}_it181_warm.nsys-rep" for label in ARM_LABELS),
    }
    observed_nsight = {path.name for path in (root / "nsight").iterdir() if path.is_file()}
    _require(observed_nsight == expected_nsight, f"Nsight artifact topology differs: {sorted(observed_nsight)}")
    arms = {spec[0]: _load_arm(root, spec) for spec in ARM_SPECS}
    schedules = [arms[label][phase]["schedule"] for label in ARM_LABELS for phase in ("cold", "warm")]
    _require(all(schedule == schedules[0] for schedule in schedules[1:]), "GF46 schedule differs across arms")
    science = _science(arms)
    performance = _performance(arms)
    gates = {
        "provenance_topology_cache_selector_schedule": True,
        "particle_star_and_discrete_metadata_exact": science["all_particle_star_and_discrete_metadata_exact"],
        "map_deltas_within_repeat_envelope": science["all_cross_mode_maps_within_repeat_envelope"],
        "map_signed_drift_nondirectional": science["all_cross_mode_signed_drift_nondirectional"],
        **performance["gates"],
    }
    passed = all(gates.values())
    report = {
        "schema": SCHEMA,
        "provenance": {
            "root": str(root),
            "job_id": run["job_id"],
            "git_head": run["git_head"],
            "git_tree": run["git_tree"],
            "node": run["node"],
            "gpu_name": run["gpu_name"],
            "gpu_uuid": run["gpu_uuid"],
            "cuda_sha256": run["cuda_sha256"],
            "source_manifest_sha256": run["source_manifest_sha256"],
            "input_manifest_sha256": run["input_manifest_sha256"],
            "analyzer_source_sha256": _sha256(Path(__file__).resolve()),
            **provenance,
        },
        "schedule": schedules[0],
        "cache_audit": {label: arms[label]["cache_events"] for label in ARM_LABELS},
        "selector_audits": {
            label: {phase: arms[label][phase]["selector_audit"] for phase in ("cold", "warm")}
            for label in ARM_LABELS
        },
        "nsight": {label: arms[label]["nsight"] for label in ARM_LABELS},
        "science": science,
        "performance": performance,
        "decision": {
            "status": "GO" if passed else "NO_GO",
            "gates": gates,
            "pass": passed,
            "classification": "diagnostic_performance_only",
            "profiled_transition": "iteration_180_to_181",
        },
        "acceptance": {"pass": passed},
    }
    report["markdown"] = _markdown(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(args.root, repo=args.repo)
    except RawCacheSetupError as exc:
        parser.error(str(exc))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(report)
    markdown = serializable.pop("markdown")
    args.output_json.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(markdown)
    print(markdown, end="")
    return 0 if report["decision"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
