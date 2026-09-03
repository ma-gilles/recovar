#!/usr/bin/env python3
"""Audit a box-800 K=1 exact-local padding optimization against a baseline.

The gate compares semantic ledger content and ordered execution telemetry while
allowing paths, commits, and timings to differ.  It also requires complete,
fatal-free logs and reports wall-time and sampled HBM reductions.  Missing or
malformed evidence fails closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "recovar.em_k1_box800_local_padding_gate.v1"

# These top-level ledger fields describe provenance, output placement, or
# performance.  Everything else is execution semantics and must match exactly.
LEDGER_VOLATILE_FIELDS = frozenset(
    {
        "git_commit",
        "global_profile_rows",
        "local_profile_rows",
        "output_dir",
        "setup_phase_seconds",
        "timing_dir",
        "timing_rows",
        "timing_summary",
        "total_time_s",
        "wall_times_trajectory",
    }
)

FATAL_LOG_PATTERNS = (
    ("python_traceback", re.compile(r"Traceback \(most recent call last\):")),
    ("resource_exhausted", re.compile(r"\bRESOURCE_EXHAUSTED\b", re.IGNORECASE)),
    ("cuda_out_of_memory", re.compile(r"\bCUDA_ERROR_OUT_OF_MEMORY\b", re.IGNORECASE)),
    ("out_of_memory", re.compile(r"\bout of memory\b", re.IGNORECASE)),
    ("oom", re.compile(r"\bOOM\b")),
    ("killed", re.compile(r"\bKilled\b")),
    ("segmentation_fault", re.compile(r"\bSegmentation fault\b", re.IGNORECASE)),
    ("fatal_python", re.compile(r"Fatal Python error", re.IGNORECASE)),
    ("slurmstepd_error", re.compile(r"slurmstepd:\s+error:", re.IGNORECASE)),
    ("guard_refused", re.compile(r"BOX800_PADDING_GUARD_REFUSED")),
)

_RELION_BATCH_RE = re.compile(
    r"requested image_batch_size=(?P<requested_image_batch_size>\d+)\s+"
    r"rotation_block_size=(?P<requested_rotation_block_size>\d+);\s+"
    r"using image_batch_size=(?P<image_batch_size>\d+)\s+"
    r"rotation_block_size=(?P<rotation_block_size>\d+)"
)
_LOCAL_CALLER_BATCH_RE = re.compile(
    r"requested image_batch_size=(?P<requested_image_batch_size>\d+)\s+"
    r"rotation_block_size=(?P<requested_rotation_block_size>\d+);\s+"
    r"using image_batch_size=(?P<image_batch_size>\d+)\s+"
    r"rotation_block_size=(?P<rotation_block_size>\d+)\s+"
    r"\(local_rot_max=(?P<local_rot_max>\d+)\s+n_trans=(?P<n_trans>\d+)\)"
)
_ADAPTIVE_MASK_RE = re.compile(
    r"RELION local adaptive pass 2 mask:\s+"
    r"parent significant samples median=(?P<parent_median>\d+)\s+max=(?P<parent_max>\d+);\s+"
    r"fine valid candidates median=(?P<fine_median>\d+)\s+max=(?P<fine_max>\d+)"
)
_BUCKET_DONE_RE = re.compile(
    r"Exact local bucket loop done:\s+"
    r"chunks=(?P<chunks_done>\d+)/(?P<chunks_total>\d+)\s+"
    r"images=(?P<images_done>\d+)/(?P<images_total>\d+)\s+"
    r"wall=(?P<wall_time_s>[-+]?\d+(?:\.\d+)?)s\s+"
    r"images/s=(?P<images_per_s>[-+]?\d+(?:\.\d+)?)"
)
_REFINEMENT_COMPLETE_RE = re.compile(
    r"Refinement complete in (?P<wall_time_s>[-+]?\d+(?:\.\d+)?)s\s+"
    r"\((?P<iterations>\d+) iterations\)"
)
_KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,;\)\s]+)")


class AuditError(RuntimeError):
    """Raised when required audit evidence is missing or malformed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"missing regular {label}: {path}")
    return resolved


def _artifact_record(path: Path, label: str) -> dict[str, Any]:
    resolved = _require_regular_file(path, label)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    resolved = _require_regular_file(path, label)
    try:
        value = json.loads(resolved.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"failed to read {label} {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"{label} must contain a JSON object: {resolved}")
    _validate_json_finite(value, label)
    return value


def _validate_json_finite(value: Any, label: str, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AuditError(f"{label} contains non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_json_finite(item, label, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_json_finite(item, label, f"{path}[{index}]")


def _ledger_semantics(ledger: Mapping[str, Any]) -> dict[str, Any]:
    semantics = {key: value for key, value in ledger.items() if key not in LEDGER_VOLATILE_FIELDS}
    if not semantics:
        raise AuditError("ledger has no semantic fields after removing volatile telemetry")
    return semantics


def _json_differences(left: Any, right: Any, path: str = "$", limit: int = 32) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def visit(left_value: Any, right_value: Any, current_path: str) -> None:
        if len(differences) >= limit:
            return
        if type(left_value) is not type(right_value):
            differences.append(
                {
                    "path": current_path,
                    "baseline": left_value,
                    "candidate": right_value,
                    "reason": "type_mismatch",
                }
            )
            return
        if isinstance(left_value, Mapping):
            left_keys = set(left_value)
            right_keys = set(right_value)
            for key in sorted(left_keys | right_keys):
                child_path = f"{current_path}.{key}"
                if key not in left_value:
                    differences.append(
                        {
                            "path": child_path,
                            "baseline": None,
                            "candidate": right_value[key],
                            "reason": "missing_baseline",
                        }
                    )
                elif key not in right_value:
                    differences.append(
                        {
                            "path": child_path,
                            "baseline": left_value[key],
                            "candidate": None,
                            "reason": "missing_candidate",
                        }
                    )
                else:
                    visit(left_value[key], right_value[key], child_path)
                if len(differences) >= limit:
                    return
            return
        if isinstance(left_value, Sequence) and not isinstance(left_value, (str, bytes)):
            if len(left_value) != len(right_value):
                differences.append(
                    {
                        "path": current_path,
                        "baseline": len(left_value),
                        "candidate": len(right_value),
                        "reason": "length_mismatch",
                    }
                )
                return
            for index, (left_item, right_item) in enumerate(zip(left_value, right_value, strict=True)):
                visit(left_item, right_item, f"{current_path}[{index}]")
                if len(differences) >= limit:
                    return
            return
        if left_value != right_value:
            differences.append(
                {
                    "path": current_path,
                    "baseline": left_value,
                    "candidate": right_value,
                    "reason": "value_mismatch",
                }
            )

    visit(left, right, path)
    return differences


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise AuditError(f"non-finite numeric log token: {raw}")
        return parsed
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _key_value_fields(line: str) -> dict[str, Any]:
    return {match.group(1): _parse_value(match.group(2)) for match in _KEY_VALUE_RE.finditer(line)}


def _require_fields(fields: Mapping[str, Any], names: Sequence[str], context: str) -> dict[str, Any]:
    missing = [name for name in names if name not in fields]
    if missing:
        raise AuditError(f"malformed {context}; missing fields: {', '.join(missing)}")
    return {name: fields[name] for name in names}


def _parse_batch_plan(line: str) -> dict[str, Any] | None:
    if "RELION EM batch sizing:" in line:
        match = _RELION_BATCH_RE.search(line)
        if match is None:
            raise AuditError(f"malformed RELION EM batch sizing line: {line}")
        plan: dict[str, Any] = {"kind": "relion_em"}
        plan.update({key: int(value) for key, value in match.groupdict().items()})
        fields = _key_value_fields(line)
        plan.update(
            _require_fields(fields, ("n_rot", "n_trans", "K", "score_budget", "score_pixels", "mode"), "batch plan")
        )
        plan["n_classes"] = plan.pop("K")
        return plan

    if "Local search caller-qualified batch sizing:" in line:
        match = _LOCAL_CALLER_BATCH_RE.search(line)
        if match is None:
            raise AuditError(f"malformed local caller-qualified batch sizing line: {line}")
        return {
            "kind": "local_caller_qualified",
            **{key: int(value) for key, value in match.groupdict().items()},
        }

    if "Local search batch sizing:" in line:
        fields = _key_value_fields(line)
        plan = {"kind": "local_search"}
        plan.update(
            _require_fields(
                fields,
                (
                    "cone_radius",
                    "est_cone_rots",
                    "eff_n_rot",
                    "n_trans",
                    "image_batch_size",
                    "rotation_block_size",
                ),
                "local-search batch plan",
            )
        )
        return plan
    return None


def _parse_significant_support(line: str) -> dict[str, Any] | None:
    if "Exact local significant-support summary:" not in line:
        return None
    fields = _key_value_fields(line)
    return _require_fields(
        fields,
        (
            "chunks",
            "big_jit_buckets",
            "sparse_big_jit_buckets",
            "reconstruction_rows",
            "padded_rows",
            "significant_samples",
            "mean_reconstruction_rows_per_image",
            "mean_significant_samples_per_image",
        ),
        "exact-local significant-support summary",
    )


def _parse_adaptive_mask(line: str) -> dict[str, int] | None:
    if "RELION local adaptive pass 2 mask:" not in line:
        return None
    match = _ADAPTIVE_MASK_RE.search(line)
    if match is None:
        raise AuditError(f"malformed adaptive-mask line: {line}")
    return {key: int(value) for key, value in match.groupdict().items()}


def _parse_bucket_timing(line: str) -> dict[str, Any] | None:
    if "Exact local bucket loop done:" not in line:
        return None
    match = _BUCKET_DONE_RE.search(line)
    if match is None:
        raise AuditError(f"malformed exact-local bucket completion line: {line}")
    values = match.groupdict()
    chunks_done = int(values["chunks_done"])
    chunks_total = int(values["chunks_total"])
    images_done = int(values["images_done"])
    images_total = int(values["images_total"])
    if chunks_done != chunks_total or images_done != images_total:
        raise AuditError(
            "incomplete exact-local bucket loop: "
            f"chunks={chunks_done}/{chunks_total} images={images_done}/{images_total}"
        )
    wall_time_s = float(values["wall_time_s"])
    images_per_s = float(values["images_per_s"])
    if not math.isfinite(wall_time_s) or wall_time_s < 0.0 or not math.isfinite(images_per_s):
        raise AuditError(f"invalid exact-local bucket timing: {line}")
    return {
        "chunks": chunks_total,
        "images": images_total,
        "wall_time_s": wall_time_s,
        "images_per_s": images_per_s,
    }


def _fatal_log_matches(lines: Sequence[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        for label, pattern in FATAL_LOG_PATTERNS:
            if pattern.search(line):
                matches.append({"kind": label, "line_number": line_number, "line": line[:1000]})
                break
    return matches


def _parse_log(path: Path, label: str) -> dict[str, Any]:
    resolved = _require_regular_file(path, label)
    try:
        lines = resolved.read_text(errors="replace").splitlines()
    except OSError as error:
        raise AuditError(f"failed to read {label} {resolved}: {error}") from error
    if not lines:
        raise AuditError(f"empty {label}: {resolved}")

    batch_plans: list[dict[str, Any]] = []
    significant_support: list[dict[str, Any]] = []
    adaptive_masks: list[dict[str, Any]] = []
    bucket_timings: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    stopped_after_score_only = False
    for line in lines:
        if (plan := _parse_batch_plan(line)) is not None:
            batch_plans.append(plan)
        if (support := _parse_significant_support(line)) is not None:
            significant_support.append(support)
        if (mask := _parse_adaptive_mask(line)) is not None:
            adaptive_masks.append(mask)
        if (timing := _parse_bucket_timing(line)) is not None:
            bucket_timings.append(timing)
        if match := _REFINEMENT_COMPLETE_RE.search(line):
            completion_rows.append(
                {
                    "wall_time_s": float(match.group("wall_time_s")),
                    "iterations": int(match.group("iterations")),
                }
            )
        stopped_after_score_only |= "Stopping after local-search diagnostic" in line and "score_only=True" in line

    required_plan_kinds = {"relion_em", "local_search", "local_caller_qualified"}
    observed_plan_kinds = {plan["kind"] for plan in batch_plans}
    if not required_plan_kinds.issubset(observed_plan_kinds):
        raise AuditError(
            f"{label} lacks required batch plan kinds: {sorted(required_plan_kinds - observed_plan_kinds)}"
        )
    if not significant_support:
        raise AuditError(f"{label} has no exact-local significant-support summaries")
    if not adaptive_masks:
        raise AuditError(f"{label} has no adaptive-mask counts")
    if not bucket_timings:
        raise AuditError(f"{label} has no exact-local bucket completion timings")
    if len(bucket_timings) != len(significant_support):
        raise AuditError(
            f"{label} bucket/support event count mismatch: {len(bucket_timings)} versus {len(significant_support)}"
        )
    if len(completion_rows) != 1:
        raise AuditError(f"{label} must contain exactly one refinement completion row, got {len(completion_rows)}")

    return {
        "batch_plans": batch_plans,
        "significant_support": significant_support,
        "adaptive_masks": adaptive_masks,
        "bucket_timings": bucket_timings,
        "completion": completion_rows[0],
        "stopped_after_score_only": stopped_after_score_only,
        "fatal_matches": _fatal_log_matches(lines),
    }


def _parse_hbm_csv(path: Path, label: str) -> dict[str, Any]:
    resolved = _require_regular_file(path, label)
    try:
        with resolved.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"timestamp", "uuid", "memory_used_mib"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise AuditError(f"{label} lacks HBM columns {sorted(required)}: {resolved}")
            memory_values: list[float] = []
            gpu_uuids: set[str] = set()
            first_timestamp = None
            last_timestamp = None
            for row_number, row in enumerate(reader, start=2):
                try:
                    memory_used_mib = float(row["memory_used_mib"])
                except (TypeError, ValueError) as error:
                    raise AuditError(f"invalid HBM value at {resolved}:{row_number}") from error
                if not math.isfinite(memory_used_mib) or memory_used_mib < 0.0:
                    raise AuditError(f"invalid HBM value at {resolved}:{row_number}: {memory_used_mib}")
                timestamp = (row.get("timestamp") or "").strip()
                gpu_uuid = (row.get("uuid") or "").strip()
                if not timestamp or not gpu_uuid:
                    raise AuditError(f"missing HBM timestamp/GPU UUID at {resolved}:{row_number}")
                first_timestamp = first_timestamp or timestamp
                last_timestamp = timestamp
                gpu_uuids.add(gpu_uuid)
                memory_values.append(memory_used_mib)
    except OSError as error:
        raise AuditError(f"failed to read {label} {resolved}: {error}") from error
    if not memory_values:
        raise AuditError(f"empty {label}: {resolved}")
    if len(gpu_uuids) != 1:
        raise AuditError(f"{label} must describe exactly one GPU, got {sorted(gpu_uuids)}")
    if max(memory_values) <= 0.0:
        raise AuditError(f"{label} has no positive HBM sample: {resolved}")
    return {
        "sample_count": len(memory_values),
        "gpu_uuid": next(iter(gpu_uuids)),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "peak_hbm_mib": max(memory_values),
    }


def _ledger_wall_timing(ledger: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        total_time_s = float(ledger["total_time_s"])
        trajectory = [float(value) for value in ledger["wall_times_trajectory"]]
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError(f"{label} lacks valid total_time_s/wall_times_trajectory") from error
    if not math.isfinite(total_time_s) or total_time_s <= 0.0:
        raise AuditError(f"{label} total_time_s must be finite and positive")
    if not trajectory or any(not math.isfinite(value) or value < 0.0 for value in trajectory):
        raise AuditError(f"{label} wall_times_trajectory must contain finite nonnegative values")
    return {
        "total_time_s": total_time_s,
        "trajectory_count": len(trajectory),
        "trajectory_total_s": sum(trajectory),
        "trajectory_last_s": trajectory[-1],
    }


def _comparison_record(name: str, baseline: Any, candidate: Any, failures: list[str]) -> dict[str, Any]:
    differences = _json_differences(baseline, candidate)
    exact = not differences
    if not exact:
        failures.append(f"semantic drift in {name}: {len(differences)} recorded difference(s)")
    return {
        "exact": exact,
        "baseline": baseline,
        "candidate": candidate,
        "differences": differences,
    }


def _validate_nonnegative_threshold(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise AuditError(f"{label} must be finite and nonnegative")
    return value


def run_audit(
    *,
    baseline_ledger: Path,
    candidate_ledger: Path,
    baseline_log: Path,
    candidate_log: Path,
    baseline_hbm: Path,
    candidate_hbm: Path,
    baseline_completed: Path,
    candidate_completed: Path,
    minimum_hbm_reduction_mib: float = 0.0,
    minimum_wall_reduction_s: float = 0.0,
) -> dict[str, Any]:
    minimum_hbm_reduction_mib = _validate_nonnegative_threshold(minimum_hbm_reduction_mib, "minimum HBM reduction")
    minimum_wall_reduction_s = _validate_nonnegative_threshold(minimum_wall_reduction_s, "minimum wall reduction")

    paths = {
        "baseline": {
            "ledger": baseline_ledger,
            "log": baseline_log,
            "hbm": baseline_hbm,
            "completed": baseline_completed,
        },
        "candidate": {
            "ledger": candidate_ledger,
            "log": candidate_log,
            "hbm": candidate_hbm,
            "completed": candidate_completed,
        },
    }
    artifacts = {
        side: {name: _artifact_record(path, f"{side} {name}") for name, path in side_paths.items()}
        for side, side_paths in paths.items()
    }

    baseline_ledger_data = _load_json_object(baseline_ledger, "baseline ledger")
    candidate_ledger_data = _load_json_object(candidate_ledger, "candidate ledger")
    baseline_log_data = _parse_log(baseline_log, "baseline log")
    candidate_log_data = _parse_log(candidate_log, "candidate log")
    baseline_hbm_data = _parse_hbm_csv(baseline_hbm, "baseline HBM trace")
    candidate_hbm_data = _parse_hbm_csv(candidate_hbm, "candidate HBM trace")
    baseline_wall = _ledger_wall_timing(baseline_ledger_data, "baseline ledger")
    candidate_wall = _ledger_wall_timing(candidate_ledger_data, "candidate ledger")

    failures: list[str] = []
    comparisons = {
        "ledger_semantics": _comparison_record(
            "ledger semantics",
            _ledger_semantics(baseline_ledger_data),
            _ledger_semantics(candidate_ledger_data),
            failures,
        ),
        "selected_batch_plans": _comparison_record(
            "selected batch plans",
            baseline_log_data["batch_plans"],
            candidate_log_data["batch_plans"],
            failures,
        ),
        "significant_support": _comparison_record(
            "significant-support summaries",
            baseline_log_data["significant_support"],
            candidate_log_data["significant_support"],
            failures,
        ),
        "adaptive_masks": _comparison_record(
            "adaptive-mask counts",
            baseline_log_data["adaptive_masks"],
            candidate_log_data["adaptive_masks"],
            failures,
        ),
        "bucket_work": _comparison_record(
            "exact-local bucket work",
            [{"chunks": event["chunks"], "images": event["images"]} for event in baseline_log_data["bucket_timings"]],
            [{"chunks": event["chunks"], "images": event["images"]} for event in candidate_log_data["bucket_timings"]],
            failures,
        ),
        "run_shape": _comparison_record(
            "run shape",
            {
                "completed_iterations": baseline_log_data["completion"]["iterations"],
                "wall_trajectory_count": baseline_wall["trajectory_count"],
            },
            {
                "completed_iterations": candidate_log_data["completion"]["iterations"],
                "wall_trajectory_count": candidate_wall["trajectory_count"],
            },
            failures,
        ),
    }

    baseline_complete = baseline_log_data["stopped_after_score_only"] and not baseline_log_data["fatal_matches"]
    candidate_complete = candidate_log_data["stopped_after_score_only"] and not candidate_log_data["fatal_matches"]
    if not baseline_complete:
        failures.append("baseline log is not a clean completed score-only diagnostic")
    if not candidate_complete:
        failures.append("candidate log is not a clean completed score-only diagnostic")

    hbm_reduction_mib = baseline_hbm_data["peak_hbm_mib"] - candidate_hbm_data["peak_hbm_mib"]
    wall_reduction_s = baseline_wall["total_time_s"] - candidate_wall["total_time_s"]
    hbm_accepted = hbm_reduction_mib >= minimum_hbm_reduction_mib
    wall_accepted = wall_reduction_s >= minimum_wall_reduction_s
    if not hbm_accepted:
        failures.append(
            f"HBM reduction {hbm_reduction_mib:.3f} MiB is below required {minimum_hbm_reduction_mib:.3f} MiB"
        )
    if not wall_accepted:
        failures.append(f"wall reduction {wall_reduction_s:.3f}s is below required {minimum_wall_reduction_s:.3f}s")

    bucket_timing_comparison = []
    for index, (baseline_event, candidate_event) in enumerate(
        zip(baseline_log_data["bucket_timings"], candidate_log_data["bucket_timings"])
    ):
        delta_s = candidate_event["wall_time_s"] - baseline_event["wall_time_s"]
        bucket_timing_comparison.append(
            {
                "event_index": index,
                "baseline_wall_time_s": baseline_event["wall_time_s"],
                "candidate_wall_time_s": candidate_event["wall_time_s"],
                "candidate_minus_baseline_s": delta_s,
            }
        )

    semantic_accepted = all(record["exact"] for record in comparisons.values())
    accepted = not failures
    return {
        "schema": SCHEMA,
        "thresholds": {
            "minimum_hbm_reduction_mib": minimum_hbm_reduction_mib,
            "minimum_wall_reduction_s": minimum_wall_reduction_s,
        },
        "artifacts": artifacts,
        "comparisons": comparisons,
        "completion": {
            "baseline": {
                "accepted": baseline_complete,
                "refinement": baseline_log_data["completion"],
                "stopped_after_score_only": baseline_log_data["stopped_after_score_only"],
                "fatal_matches": baseline_log_data["fatal_matches"],
            },
            "candidate": {
                "accepted": candidate_complete,
                "refinement": candidate_log_data["completion"],
                "stopped_after_score_only": candidate_log_data["stopped_after_score_only"],
                "fatal_matches": candidate_log_data["fatal_matches"],
            },
        },
        "performance": {
            "hbm": {
                "baseline": baseline_hbm_data,
                "candidate": candidate_hbm_data,
                "reduction_mib": hbm_reduction_mib,
                "reduction_percent_of_baseline": 100.0 * hbm_reduction_mib / baseline_hbm_data["peak_hbm_mib"],
                "accepted": hbm_accepted,
            },
            "wall": {
                "baseline": baseline_wall,
                "candidate": candidate_wall,
                "reduction_s": wall_reduction_s,
                "reduction_percent_of_baseline": 100.0 * wall_reduction_s / baseline_wall["total_time_s"],
                "accepted": wall_accepted,
            },
            "exact_local_bucket_timings": bucket_timing_comparison,
        },
        "failures": failures,
        "summary": {
            "semantic_equivalence_accepted": semantic_accepted,
            "candidate_completion_accepted": candidate_complete,
            "hbm_gate_accepted": hbm_accepted,
            "wall_gate_accepted": wall_accepted,
            "accepted": accepted,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ledger", type=Path, required=True)
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--baseline-hbm", type=Path, required=True)
    parser.add_argument("--candidate-hbm", type=Path, required=True)
    parser.add_argument("--baseline-completed", type=Path, required=True)
    parser.add_argument("--candidate-completed", type=Path, required=True)
    parser.add_argument("--minimum-hbm-reduction-mib", type=float, default=0.0)
    parser.add_argument("--minimum-wall-reduction-s", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run_audit(
            baseline_ledger=args.baseline_ledger,
            candidate_ledger=args.candidate_ledger,
            baseline_log=args.baseline_log,
            candidate_log=args.candidate_log,
            baseline_hbm=args.baseline_hbm,
            candidate_hbm=args.candidate_hbm,
            baseline_completed=args.baseline_completed,
            candidate_completed=args.candidate_completed,
            minimum_hbm_reduction_mib=args.minimum_hbm_reduction_mib,
            minimum_wall_reduction_s=args.minimum_wall_reduction_s,
        )
    except AuditError as error:
        report = {
            "schema": SCHEMA,
            "failures": [str(error)],
            "summary": {"accepted": False},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output.resolve()}")
    return 0 if report["summary"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
