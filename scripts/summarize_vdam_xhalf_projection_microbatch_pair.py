#!/usr/bin/env python3
"""Summarize one warmed VDAM x-half row-pixel cap pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from recovar.em.diagnostics.coarse_score_diagnostics import _validate_coarse_selector_audit
from recovar.em.local.local_batch_planning import _exact_local_max_hypotheses_per_microbatch
from scripts.summarize_em_completion_bench import (
    _load_relion_volume,
    normalized_fsc_auc,
    shell_fsc,
)

SCHEMA = "recovar.vdam_xhalf_projection_microbatch_pair.v2"
SCORED_LABELS = (
    "scored-control-1",
    "scored-control-2",
    "scored-candidate-1",
    "scored-candidate-2",
)
PAIR_LABELS = {
    "cross-1": ("scored-candidate-1", "scored-control-1"),
    "cross-2": ("scored-candidate-2", "scored-control-2"),
    "control-repeat": ("scored-control-1", "scored-control-2"),
    "candidate-repeat": ("scored-candidate-1", "scored-candidate-2"),
}
ARTIFACT_SUFFIXES = ("data.star", "model.star", "class001.mrc")
_ROUTE_ONLY_PROFILE_CONTAINER = "halfset_0_profile_summary"
_ROUTE_ONLY_PROFILE_SUMMARY_KEYS = frozenset({"coarse_selector_audit"})
_COARSE_SELECTOR_AUDIT_KEYS = frozenset(
    {
        "score_mode",
        "translation_count",
        "requested_fused",
        "effective_fused",
        "requested_workers",
        "effective_workers",
        "requested_atomic",
        "effective_atomic",
        "wrapper",
        "target",
        "counts",
        "requested_prehalf",
        "effective_prehalf",
    }
)
_COARSE_SELECTOR_COUNT_KEYS = frozenset(
    {
        "fused_calls",
        "actual_rows",
        "multistream_calls",
        "native_atomic_selected_calls",
        "prehalf_selected_calls",
    }
)
_TARGET_SCHEDULE_FIELDS = (
    "current_size",
    "healpix_order",
    "n_rotations",
    "n_translations",
    "subset_size",
    "oversampling",
    "max_significants",
    "effective_image_batch_size",
    "pass2_engine",
)


class XHalfPairError(RuntimeError):
    """Raised when a cap pair is incomplete or mixed."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise XHalfPairError(f"cannot read JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise XHalfPairError(f"expected a JSON object at {path}")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _map_metric(lhs_path: Path, rhs_path: Path, *, with_fsc: bool) -> dict[str, Any]:
    lhs = _load_relion_volume(lhs_path)
    rhs = _load_relion_volume(rhs_path)
    if lhs.shape != rhs.shape:
        raise XHalfPairError(
            f"map shapes differ: {lhs_path} {lhs.shape} vs {rhs_path} {rhs.shape}"
        )
    delta = np.asarray(lhs - rhs, dtype=np.float64)
    denominator = float(np.linalg.norm(np.asarray(rhs, dtype=np.float64).reshape(-1)))
    result: dict[str, Any] = {
        "bytewise_array_equal": bool(np.array_equal(lhs, rhs)),
        "differing_voxel_count": int(np.count_nonzero(lhs != rhs)),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "relative_l2_error": float(
            np.linalg.norm(delta.reshape(-1)) / max(denominator, 1e-300)
        ),
    }
    if with_fsc:
        result["fsc_auc"] = float(normalized_fsc_auc(shell_fsc(lhs, rhs)))
    return result


def _arms(root: Path) -> dict[str, int]:
    rows = {}
    for raw in (root / "provenance" / "arms.tsv").read_text().splitlines():
        arm, value = raw.split("\t")
        rows[arm] = int(value)
    if set(rows) != {"control", "candidate"} or rows["control"] == rows["candidate"]:
        raise XHalfPairError(f"invalid arm definitions: {rows}")
    return rows


def _execution_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in (root / "provenance" / "execution_order.tsv").read_text().splitlines():
        label, arm, row_pixels, elapsed_ns, gpu_uuid, cache_dir, run_status = raw.split("\t")
        rows.append(
            {
                "label": label,
                "arm": arm,
                "row_pixels": int(row_pixels),
                "external_wall_seconds": int(elapsed_ns) / 1.0e9,
                "gpu_uuid": gpu_uuid,
                "jax_cache_dir": cache_dir,
                "run_status": int(run_status),
            }
        )
    expected = [
        ("warmup-control", "control"),
        ("warmup-candidate", "candidate"),
        ("scored-control-1", "control"),
        ("scored-candidate-1", "candidate"),
        ("scored-candidate-2", "candidate"),
        ("scored-control-2", "control"),
    ]
    observed = [(row["label"], row["arm"]) for row in rows]
    if observed != expected:
        raise XHalfPairError(f"execution order differs: {observed!r}")
    if any(row["run_status"] != 0 for row in rows):
        raise XHalfPairError("one or more arms failed")
    if len({row["gpu_uuid"] for row in rows}) != 1:
        raise XHalfPairError("pair spans multiple physical GPUs")
    caches = {
        arm: {row["jax_cache_dir"] for row in rows if row["arm"] == arm}
        for arm in ("control", "candidate")
    }
    if any(len(values) != 1 for values in caches.values()) or caches["control"] == caches["candidate"]:
        raise XHalfPairError(f"arms do not have separate stable caches: {caches}")
    for row in rows:
        run = _load_json(root / row["label"] / "provenance" / "run.json")
        row["runner_wall_seconds"] = float(run["recovar_wall_s"])
        cap_path = root / row["label"] / "provenance" / "xhalf_target_row_pixels.txt"
        try:
            row["provenance_row_pixels"] = int(cap_path.read_text().strip())
        except (OSError, ValueError) as exc:
            raise XHalfPairError(
                f"cannot read per-run x-half cap at {cap_path}: {exc}"
            ) from exc
    return rows


def _profile_metadata_contract(
    meta: dict[str, Any],
    iteration: int,
    *,
    require_route_audit: bool = True,
) -> dict[str, Any]:
    """Accept only the route audit that is emitted with stage profiling off."""

    profile_keys = sorted(key for key in meta if "profile" in key.lower())
    violations = [
        f"unexpected profile metadata key: {key}"
        for key in profile_keys
        if key != _ROUTE_ONLY_PROFILE_CONTAINER
    ]
    if require_route_audit and _ROUTE_ONLY_PROFILE_CONTAINER not in meta:
        violations.append(
            f"missing required route-only profile container: {_ROUTE_ONLY_PROFILE_CONTAINER}"
        )

    if _ROUTE_ONLY_PROFILE_CONTAINER in meta:
        summary = meta[_ROUTE_ONLY_PROFILE_CONTAINER]
        if not isinstance(summary, dict):
            violations.append(
                f"{_ROUTE_ONLY_PROFILE_CONTAINER} must be a JSON object"
            )
        elif set(summary) != _ROUTE_ONLY_PROFILE_SUMMARY_KEYS:
            violations.append(
                f"{_ROUTE_ONLY_PROFILE_CONTAINER} keys differ from the route-only contract: "
                f"{sorted(summary)}"
            )
        else:
            audit = summary["coarse_selector_audit"]
            if not isinstance(audit, dict):
                violations.append("coarse_selector_audit must be a JSON object")
            else:
                audit_keys = frozenset(audit)
                if audit_keys != _COARSE_SELECTOR_AUDIT_KEYS:
                    violations.append(
                        "coarse_selector_audit keys differ from the route-only contract: "
                        f"{sorted(audit)}"
                    )
                counts = audit.get("counts")
                if isinstance(counts, dict):
                    if frozenset(counts) != _COARSE_SELECTOR_COUNT_KEYS:
                        violations.append(
                            "coarse_selector_audit counts keys differ from the route-only "
                            f"contract: {sorted(counts)}"
                        )
                if audit_keys == _COARSE_SELECTOR_AUDIT_KEYS and not any(
                    violation.startswith("coarse_selector_audit counts keys")
                    for violation in violations
                ):
                    try:
                        _validate_coarse_selector_audit(audit)
                    except (TypeError, ValueError) as exc:
                        violations.append(f"invalid coarse_selector_audit: {exc}")

    return {
        "iteration": int(iteration),
        "profile_keys": profile_keys,
        "route_only_profile_container_present": _ROUTE_ONLY_PROFILE_CONTAINER in meta,
        "route_only_profile_container_required": bool(require_route_audit),
        "profile_contract_violations": violations,
        "profile_unset_contract_pass": not violations,
    }


def _profile_absence(root: Path, target_iteration: int) -> dict[str, Any]:
    result = {}
    for label in ("warmup-control", "warmup-candidate", *SCORED_LABELS):
        rows = []
        for iteration in range(1, target_iteration + 1):
            meta = _load_json(
                root / label / "recovar_prefix" / f"run_it{iteration:03d}_recovar_meta.json"
            )
            rows.append(
                _profile_metadata_contract(
                    meta,
                    iteration,
                    require_route_audit=True,
                )
            )
        result[label] = {
            "iteration_count": len(rows),
            "iterations_with_profile_metadata": sum(bool(row["profile_keys"]) for row in rows),
            "iterations_with_disallowed_profile_metadata": sum(
                not row["profile_unset_contract_pass"] for row in rows
            ),
            "profile_unset_contract_pass": all(
                row["profile_unset_contract_pass"] for row in rows
            ),
            "rows": rows,
        }
    return result


def _benchmark_topology_evidence(
    arms: dict[str, int],
    execution: list[dict[str, Any]],
    topology: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate the declared and observed cap pair before scoring its speed."""

    declared_caps_positive = all(int(arms[arm]) > 0 for arm in ("control", "candidate"))
    candidate_cap_gt_control = int(arms["candidate"]) > int(arms["control"])
    execution_caps_match_declaration = all(
        int(row["row_pixels"]) == int(arms[row["arm"]]) for row in execution
    )
    per_run_provenance_caps_match_declaration = all(
        int(row["provenance_row_pixels"]) == int(arms[row["arm"]])
        for row in execution
    )
    candidate_bucket_count_lt_control = int(
        topology["candidate"]["predicted_big_jit_bucket_count"]
    ) < int(topology["control"]["predicted_big_jit_bucket_count"])
    predicates = {
        "declared_caps_positive": declared_caps_positive,
        "candidate_cap_gt_control": candidate_cap_gt_control,
        "execution_caps_match_declaration": execution_caps_match_declaration,
        "per_run_provenance_caps_match_declaration": (
            per_run_provenance_caps_match_declaration
        ),
        "candidate_bucket_count_lt_control": candidate_bucket_count_lt_control,
    }
    return {**predicates, "pass": all(predicates.values())}


def _target_schedule_evidence(
    root: Path,
    execution: list[dict[str, Any]],
    target_iteration: int,
) -> dict[str, Any]:
    """Compare every workload schedule row across every timed run."""

    per_iteration = {}
    for iteration in range(1, target_iteration + 1):
        per_run = {}
        for row in execution:
            meta = _load_json(
                root
                / row["label"]
                / "recovar_prefix"
                / f"run_it{iteration:03d}_recovar_meta.json"
            )
            missing = [field for field in _TARGET_SCHEDULE_FIELDS if field not in meta]
            if missing:
                raise XHalfPairError(
                    f"{row['label']} iteration {iteration} schedule is missing fields: {missing}"
                )
            per_run[row["label"]] = {
                field: meta[field] for field in _TARGET_SCHEDULE_FIELDS
            }
        per_iteration[iteration] = per_run
    target_per_run = per_iteration[target_iteration]
    mismatched_iterations = [
        iteration
        for iteration, per_run in per_iteration.items()
        if not _target_schedules_match(per_run)
    ]
    return {
        "fields": list(_TARGET_SCHEDULE_FIELDS),
        "per_run": target_per_run,
        "matched": _target_schedules_match(target_per_run),
        "all_iterations_matched": not mismatched_iterations,
        "mismatched_iterations": mismatched_iterations,
    }


def _target_schedules_match(per_run: dict[str, dict[str, Any]]) -> bool:
    """Return whether all timed runs expose one exact target schedule."""

    if not per_run:
        return False
    signatures = {
        tuple(schedule[field] for field in _TARGET_SCHEDULE_FIELDS)
        for schedule in per_run.values()
    }
    return len(signatures) == 1


def _performance_rung_evidence_pass(
    *,
    benchmark_topology_valid: bool,
    profile_unset: bool,
    repeatable_gain: bool,
) -> bool:
    """Separate performance evidence from the later science-promotion gate."""

    return bool(benchmark_topology_valid and profile_unset and repeatable_gain)


def _causal_speedup_magnitude_valid(
    *,
    performance_rung_evidence_pass: bool,
    all_iteration_schedules_matched: bool,
    cross_arm_state_discrete_equal: bool,
) -> bool:
    """Require equal work and trajectory state before attributing wall magnitude."""

    return bool(
        performance_rung_evidence_pass
        and all_iteration_schedules_matched
        and cross_arm_state_discrete_equal
    )


def _memory(root: Path, execution: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for row in execution:
        path = root / row["label"] / "telemetry" / "gpu_memory.tsv"
        samples = []
        for raw in path.read_text().splitlines():
            timestamp_ns, used_mib, total_mib = raw.split("\t")
            samples.append(
                {
                    "timestamp_ns": int(timestamp_ns),
                    "used_mib": int(used_mib),
                    "total_mib": int(total_mib),
                }
            )
        if not samples:
            raise XHalfPairError(f"no GPU-memory samples for {row['label']}")
        result[row["label"]] = {
            "sample_count": len(samples),
            "peak_used_mib": max(sample["used_mib"] for sample in samples),
            "total_mib": max(sample["total_mib"] for sample in samples),
        }
    return result


def derive_it80_topology(layout_meta: dict[str, Any], row_pixels: int) -> dict[str, Any]:
    """Derive cap buckets from frozen layout evidence without enabling PROFILE."""

    profile = layout_meta.get("halfset_0_profile_summary")
    if not isinstance(profile, dict):
        raise XHalfPairError("layout evidence lacks halfset_0_profile_summary")
    sizes = [int(value) for value in profile["chunk_sizes"]]
    padded = [int(value) for value in profile["chunk_padded_rotations"]]
    if len(sizes) != len(padded) or not sizes or any(size <= 0 for size in sizes):
        raise XHalfPairError("layout evidence has invalid chunks")
    widths = [pad // size for pad, size in zip(padded, sizes, strict=True)]
    if any(pad % size for pad, size in zip(padded, sizes, strict=True)) or len(set(widths)) != 1:
        raise XHalfPairError("layout evidence does not have one reconstructible bucket width")
    width = widths[0]
    n_projection = int(profile["n_projection_windowed"])
    n_translations = int(layout_meta["n_translations"])
    upstream_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        n_projection,
        n_trans=n_translations,
        n_recon_windowed=n_projection,
        allow_high_memory_default=False,
    )
    projection_cap = max(width, int(row_pixels) // n_projection)
    effective_cap = min(int(upstream_cap), int(projection_cap))
    images_per_chunk = max(1, effective_cap // width)
    remaining = sum(sizes)
    predicted_sizes = []
    while remaining:
        take = min(remaining, images_per_chunk)
        predicted_sizes.append(take)
        remaining -= take
    return {
        "source_profile_chunk_sizes": sizes,
        "source_profile_chunk_padded_rotations": padded,
        "bucket_rotation_width": width,
        "particle_count": sum(sizes),
        "n_projection_windowed": n_projection,
        "n_translations": n_translations,
        "upstream_exact_local_cap": int(upstream_cap),
        "requested_projection_row_pixel_cap": int(row_pixels),
        "projection_hypothesis_cap": int(projection_cap),
        "effective_hypothesis_cap": int(effective_cap),
        "images_per_chunk": int(images_per_chunk),
        "predicted_chunk_sizes": predicted_sizes,
        "predicted_chunk_padded_rotations": [size * width for size in predicted_sizes],
        "predicted_big_jit_bucket_count": len(predicted_sizes),
    }


def _artifact_exactness(root: Path, target_iteration: int) -> dict[str, Any]:
    digests: dict[Path, str] = {}

    def digest(path: Path) -> str:
        if path not in digests:
            digests[path] = _sha256(path)
        return digests[path]

    result = {}
    for pair, (lhs, rhs) in PAIR_LABELS.items():
        rows = []
        for iteration in range(1, target_iteration + 1):
            for suffix in ARTIFACT_SUFFIXES:
                lhs_path = root / lhs / "recovar_prefix" / f"run_it{iteration:03d}_{suffix}"
                rhs_path = root / rhs / "recovar_prefix" / f"run_it{iteration:03d}_{suffix}"
                lhs_sha = digest(lhs_path)
                rhs_sha = digest(rhs_path)
                rows.append(
                    {
                        "iteration": iteration,
                        "suffix": suffix,
                        "lhs_sha256": lhs_sha,
                        "rhs_sha256": rhs_sha,
                        "byte_exact": lhs_sha == rhs_sha,
                    }
                )
        result[pair] = {
            "comparison_count": len(rows),
            "byte_mismatch_count": sum(not row["byte_exact"] for row in rows),
            "rows": rows,
        }
    return result


def _map_metrics(root: Path, relion_dir: Path, target_iteration: int) -> dict[str, Any]:
    checkpoint_candidates = (1, 2, 3, 4, 8, 20, 40, 60, target_iteration)
    checkpoints = tuple(sorted({value for value in checkpoint_candidates if value <= target_iteration}))
    pairwise = {}
    for pair, (lhs, rhs) in PAIR_LABELS.items():
        pairwise[pair] = [
            {
                "iteration": iteration,
                **_map_metric(
                    root / lhs / "recovar_prefix" / f"run_it{iteration:03d}_class001.mrc",
                    root / rhs / "recovar_prefix" / f"run_it{iteration:03d}_class001.mrc",
                    with_fsc=iteration == target_iteration,
                ),
            }
            for iteration in checkpoints
        ]
    versus_relion = {}
    for label in SCORED_LABELS:
        versus_relion[label] = [
            {
                "iteration": iteration,
                **_map_metric(
                    root / label / "recovar_prefix" / f"run_it{iteration:03d}_class001.mrc",
                    relion_dir / f"run_it{iteration:03d}_class001.mrc",
                    with_fsc=iteration == target_iteration,
                ),
            }
            for iteration in checkpoints
        ]
    repeat_final = [pairwise[name][-1] for name in ("control-repeat", "candidate-repeat")]
    cross_final = [pairwise[name][-1] for name in ("cross-1", "cross-2")]
    repeat_rel_l2_max = max(row["relative_l2_error"] for row in repeat_final)
    cross_rel_l2_max = max(row["relative_l2_error"] for row in cross_final)
    repeat_fsc_min = min(row["fsc_auc"] for row in repeat_final)
    cross_fsc_min = min(row["fsc_auc"] for row in cross_final)
    return {
        "checkpoints": checkpoints,
        "pairwise": pairwise,
        "versus_relion": versus_relion,
        "terminal_repeat_envelope": {
            "repeat_relative_l2_max": repeat_rel_l2_max,
            "cross_relative_l2_max": cross_rel_l2_max,
            "repeat_fsc_auc_min": repeat_fsc_min,
            "cross_fsc_auc_min": cross_fsc_min,
            "cross_within_2x_repeat_relative_l2": cross_rel_l2_max
            <= 2.0 * repeat_rel_l2_max + 1e-15,
            "cross_at_least_repeat_fsc_floor": cross_fsc_min >= repeat_fsc_min - 1e-12,
        },
    }


def _state_audits(root: Path) -> dict[str, Any]:
    reports = {
        pair: _load_json(root / "audits" / f"{pair}_state.json") for pair in PAIR_LABELS
    }
    reports.update(
        {
            f"{label}-vs-relion": _load_json(
                root / "audits" / f"{label}_vs_relion_state.json"
            )
            for label in SCORED_LABELS
        }
    )
    summaries = {
        name: {
            "first_divergent_iteration": report["first_divergent_iteration"],
            "maximum_divergent_particle_count": max(
                int(row["divergent_particle_count"]) for row in report["iterations"]
            ),
            "maximum_pmax_absolute_error": max(
                float(row["pmax_absolute_error"]["max"]) for row in report["iterations"]
            ),
        }
        for name, report in reports.items()
    }
    return {"summaries": summaries, "reports": reports}


def summarize(
    root: Path,
    relion_dir: Path,
    layout_evidence_meta: Path,
    target_iteration: int,
) -> dict[str, Any]:
    root = root.resolve()
    relion_dir = relion_dir.resolve()
    arms = _arms(root)
    execution = _execution_rows(root)
    profile_absence = _profile_absence(root, target_iteration)
    memory = _memory(root, execution)
    external = {
        arm: [
            row["external_wall_seconds"]
            for row in execution
            if row["label"].startswith("scored-") and row["arm"] == arm
        ]
        for arm in ("control", "candidate")
    }
    medians = {arm: float(statistics.median(values)) for arm, values in external.items()}
    runner = {
        arm: [
            row["runner_wall_seconds"]
            for row in execution
            if row["label"].startswith("scored-") and row["arm"] == arm
        ]
        for arm in ("control", "candidate")
    }
    runner_medians = {arm: float(statistics.median(values)) for arm, values in runner.items()}
    scored_memory = {
        arm: [
            memory[row["label"]]["peak_used_mib"]
            for row in execution
            if row["label"].startswith("scored-") and row["arm"] == arm
        ]
        for arm in ("control", "candidate")
    }
    topology_evidence = _load_json(layout_evidence_meta)
    topology = {
        arm: derive_it80_topology(topology_evidence, row_pixels)
        for arm, row_pixels in arms.items()
    }
    topology_validation = _benchmark_topology_evidence(arms, execution, topology)
    target_schedule = _target_schedule_evidence(root, execution, target_iteration)
    artifacts = _artifact_exactness(root, target_iteration)
    maps = _map_metrics(root, relion_dir, target_iteration)
    states = _state_audits(root)
    cross_states_exact = all(
        states["summaries"][name]["first_divergent_iteration"] is None
        for name in ("cross-1", "cross-2")
    )
    map_envelope_pass = bool(
        maps["terminal_repeat_envelope"]["cross_within_2x_repeat_relative_l2"]
        and maps["terminal_repeat_envelope"]["cross_at_least_repeat_fsc_floor"]
    )
    profile_unset = all(row["profile_unset_contract_pass"] for row in profile_absence.values())
    speedup = medians["control"] / medians["candidate"]
    pairwise_speedups = [
        external["control"][0] / external["candidate"][0],
        external["control"][1] / external["candidate"][1],
    ]
    repeatable_gain = speedup >= 1.02 and min(pairwise_speedups) >= 1.01
    science_pass = bool(cross_states_exact and map_envelope_pass)
    benchmark_topology_valid = bool(topology_validation["pass"])
    performance_rung_evidence_pass = _performance_rung_evidence_pass(
        benchmark_topology_valid=benchmark_topology_valid,
        profile_unset=profile_unset,
        repeatable_gain=repeatable_gain,
    )
    causal_speedup_magnitude_valid = _causal_speedup_magnitude_valid(
        performance_rung_evidence_pass=performance_rung_evidence_pass,
        all_iteration_schedules_matched=bool(
            target_schedule["all_iterations_matched"]
        ),
        cross_arm_state_discrete_equal=cross_states_exact,
    )
    return {
        "schema": SCHEMA,
        "root": str(root),
        "relion_reference_dir": str(relion_dir),
        "layout_evidence_meta": str(layout_evidence_meta.resolve()),
        "layout_evidence_sha256": _sha256(layout_evidence_meta),
        "target_iteration": int(target_iteration),
        "arms": arms,
        "execution": execution,
        "runtime": {
            "scored_external_wall_seconds": external,
            "scored_external_median_wall_seconds": medians,
            "candidate_speedup": speedup,
            "pairwise_speedups": pairwise_speedups,
            "scored_runner_wall_seconds": runner,
            "scored_runner_median_wall_seconds": runner_medians,
        },
        "gpu_memory": {"per_run": memory, "scored_peak_used_mib": scored_memory},
        "profile_metadata": profile_absence,
        "derived_it80_topology": topology,
        "benchmark_topology_evidence": topology_validation,
        "target_schedule_evidence": target_schedule,
        "artifact_exactness": artifacts,
        "map_metrics": maps,
        "particle_state": states,
        "classification": {
            "benchmark_topology_valid": benchmark_topology_valid,
            "profile_unset_contract_pass": profile_unset,
            "performance_rung_evidence_pass": performance_rung_evidence_pass,
            "target_schedule_matched": bool(target_schedule["matched"]),
            "all_iteration_schedules_matched": bool(
                target_schedule["all_iterations_matched"]
            ),
            "causal_speedup_magnitude_valid": causal_speedup_magnitude_valid,
            "cross_particle_pose_translation_exact": cross_states_exact,
            "cross_terminal_map_within_repeat_envelope": map_envelope_pass,
            "science_within_repeat_envelope": science_pass,
            "repeatable_warmed_gain_at_least_2pct": repeatable_gain,
            "promote_candidate_factor": bool(science_pass and repeatable_gain),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--relion-dir", type=Path, required=True)
    parser.add_argument("--layout-evidence-meta", type=Path, required=True)
    parser.add_argument("--target-iteration", type=int, default=80)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = summarize(
        args.root,
        args.relion_dir,
        args.layout_evidence_meta,
        args.target_iteration,
    )
    output = args.root / "pair_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.check and not report["classification"]["profile_unset_contract_pass"]:
        raise SystemExit("PROFILE unexpectedly enabled in x-half cap benchmark")


if __name__ == "__main__":
    main()
