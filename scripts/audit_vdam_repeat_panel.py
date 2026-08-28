#!/usr/bin/env python3
"""Audit a same-GPU panel against stock RELION's native repeat envelope."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc
else:
    from summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc


SCHEMA = "recovar.vdam_relion_repeat_panel.v2"
TRAJECTORY_SCHEMA = "recovar.vdam_relion_fsc_trajectory_audit.v1"


class RepeatPanelError(RuntimeError):
    """Raised when repeat-panel evidence is incomplete or mixed."""


def classify_native_radius_support(
    *,
    relion_self_fsc_auc: list[float],
    cross_engine_fsc_auc: list[list[float]],
) -> dict[str, Any]:
    """Test bidirectional support using each native's nearest native peer."""

    cross = np.asarray(cross_engine_fsc_auc, dtype=np.float64)
    if cross.ndim != 2 or min(cross.shape) < 2 or not np.all(np.isfinite(cross)):
        raise RepeatPanelError(
            "native-radius support requires a finite candidate-by-native panel of at least 2x2"
        )
    native_count = int(cross.shape[1])
    expected_pairs = native_count * (native_count - 1) // 2
    native_pairs = np.asarray(relion_self_fsc_auc, dtype=np.float64)
    if (
        native_pairs.shape != (expected_pairs,)
        or not np.all(np.isfinite(native_pairs))
    ):
        raise RepeatPanelError(
            f"native-radius support expected {expected_pairs} finite native pairs"
        )

    native_matrix = np.full((native_count, native_count), -np.inf, dtype=np.float64)
    np.fill_diagonal(native_matrix, 1.0)
    for value, (lhs, rhs) in zip(
        native_pairs,
        itertools.combinations(range(native_count), 2),
        strict=True,
    ):
        native_matrix[lhs, rhs] = value
        native_matrix[rhs, lhs] = value
    np.fill_diagonal(native_matrix, -np.inf)
    native_nearest = np.max(native_matrix, axis=1)
    matches = cross >= native_nearest[np.newaxis, :]
    candidate_matches = [
        (np.flatnonzero(row) + 1).astype(int).tolist() for row in matches
    ]
    native_matches = [
        (np.flatnonzero(matches[:, index]) + 1).astype(int).tolist()
        for index in range(native_count)
    ]
    candidate_best = np.max(cross, axis=1)
    native_best = np.max(cross, axis=0)
    candidate_validity = all(candidate_matches)
    reverse_coverage = all(native_matches)
    return {
        "pass": candidate_validity and reverse_coverage,
        "candidate_validity_pass": candidate_validity,
        "reverse_native_coverage_pass": reverse_coverage,
        "native_nearest_peer_fsc_auc": native_nearest.tolist(),
        "candidate_matching_native_repeat_indices": candidate_matches,
        "native_matching_candidate_repeat_indices": native_matches,
        "candidate_best_native_fsc_auc": candidate_best.tolist(),
        "native_best_candidate_fsc_auc": native_best.tolist(),
        "minimum_candidate_radius_margin_fsc_auc": float(
            np.min(
                [
                    np.max(cross[index] - native_nearest)
                    for index in range(cross.shape[0])
                ]
            )
        ),
        "minimum_native_radius_margin_fsc_auc": float(
            np.min(native_best - native_nearest)
        ),
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RepeatPanelError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RepeatPanelError(f"{label} must contain a JSON object: {path}")
    return value


def classify_checkpoint(
    *,
    relion_self_fsc_auc: list[float],
    recovar_self_fsc_auc: list[float],
    cross_engine_fsc_auc: list[list[float]],
    gt_deltas: list[float],
    cross_engine_min: float,
    gt_delta_min: float,
) -> dict[str, Any]:
    """Classify parity without widening the frozen point thresholds."""

    if not relion_self_fsc_auc or not recovar_self_fsc_auc or not cross_engine_fsc_auc:
        raise RepeatPanelError("repeat classification requires native, candidate, and cross-engine comparisons")
    if not gt_deltas:
        raise RepeatPanelError("repeat classification requires per-run GT nondegradation evidence")
    native_floor = float(min(relion_self_fsc_auc))
    candidate_floor = float(min(recovar_self_fsc_auc))
    cross_matrix = np.asarray(cross_engine_fsc_auc, dtype=np.float64)
    if cross_matrix.ndim != 2 or 0 in cross_matrix.shape or not np.all(np.isfinite(cross_matrix)):
        raise RepeatPanelError("cross-engine comparisons must form a finite candidate-by-native matrix")
    candidate_best_native = np.max(cross_matrix, axis=1)
    native_best_candidate = np.max(cross_matrix, axis=0)
    cross_floor = float(np.min(cross_matrix))
    cross_ceiling = float(np.max(cross_matrix))
    gt_floor = float(min(gt_deltas))
    checks = {
        "candidate_repeat_within_native_envelope": (
            candidate_floor >= float(cross_engine_min) or candidate_floor >= native_floor
        ),
        "every_candidate_run_matches_native_mode_at_frozen_gate": bool(
            np.all(candidate_best_native >= float(cross_engine_min))
        ),
        "every_native_run_has_candidate_mode_at_frozen_gate": bool(
            np.all(native_best_candidate >= float(cross_engine_min))
        ),
        "all_runs_meet_frozen_gt_nondegradation_gate": gt_floor >= float(gt_delta_min),
    }
    native_radius_support = classify_native_radius_support(
        relion_self_fsc_auc=relion_self_fsc_auc,
        cross_engine_fsc_auc=cross_engine_fsc_auc,
    )
    native_radius_checks = {
        "candidate_repeat_within_native_envelope": checks[
            "candidate_repeat_within_native_envelope"
        ],
        "every_candidate_run_inside_native_nearest_peer_radius": (
            native_radius_support["candidate_validity_pass"]
        ),
        "every_native_radius_covered_by_candidate_panel": (
            native_radius_support["reverse_native_coverage_pass"]
        ),
        "all_runs_meet_frozen_gt_nondegradation_gate": checks[
            "all_runs_meet_frozen_gt_nondegradation_gate"
        ],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "native_relion_repeat_floor_fsc_auc": native_floor,
        "recovar_repeat_floor_fsc_auc": candidate_floor,
        "cross_engine_floor_fsc_auc": cross_floor,
        "cross_engine_ceiling_fsc_auc": cross_ceiling,
        "minimum_candidate_best_native_fsc_auc": float(np.min(candidate_best_native)),
        "minimum_native_best_candidate_fsc_auc": float(np.min(native_best_candidate)),
        "minimum_recovar_minus_relion_gt_fsc_auc": gt_floor,
        "native_radius_distribution": {
            **native_radius_support,
            "pass": all(native_radius_checks.values()),
            "checks": native_radius_checks,
        },
    }


def _metric(lhs: np.ndarray, rhs: np.ndarray, *, key: str, shellwise: dict[str, np.ndarray]) -> float:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    if curve.size <= 1 or not np.any(np.isfinite(curve[1:])):
        raise RepeatPanelError(f"{key} produced no finite non-DC FSC shells")
    shellwise[key] = curve
    value = float(normalized_fsc_auc(curve))
    if not np.isfinite(value):
        raise RepeatPanelError(f"{key} produced a non-finite FSC-AUC")
    return value


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise RepeatPanelError("runtime distribution values must be finite and nonempty")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _runtime_summary(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for repeat in repeats:
        root = Path(repeat["root"])
        relion = _load_json(
            root / "relion" / "relion.timing.json",
            label=f"repeat {repeat['index']} RELION timing",
        )
        recovar = _load_json(
            root / "recovar" / "recovar.timing.json",
            label=f"repeat {repeat['index']} RECOVAR timing",
        )
        if int(relion.get("exit_status", -1)) != 0 or int(recovar.get("exit_status", -1)) != 0:
            raise RepeatPanelError(f"repeat {repeat['index']}: engine timing reports nonzero exit")
        relion_wall = float(relion.get("external_wall_s", float("nan")))
        recovar_wall = float(recovar.get("external_wall_s", float("nan")))
        if not np.isfinite(relion_wall) or not np.isfinite(recovar_wall):
            raise RepeatPanelError(f"repeat {repeat['index']}: engine timing is non-finite")
        if relion_wall <= 0.0 or recovar_wall <= 0.0:
            raise RepeatPanelError(f"repeat {repeat['index']}: engine timing must be positive")
        rows.append(
            {
                "repeat_index": int(repeat["index"]),
                "relion_wall_s": relion_wall,
                "recovar_wall_s": recovar_wall,
                "recovar_over_relion": recovar_wall / relion_wall,
            }
        )
    return {
        "scoring": False,
        "reason": "the frozen scorecard defines no runtime acceptance threshold",
        "repeats": rows,
        "relion_wall_s": _distribution([row["relion_wall_s"] for row in rows]),
        "recovar_wall_s": _distribution([row["recovar_wall_s"] for row in rows]),
        "recovar_over_relion": _distribution([row["recovar_over_relion"] for row in rows]),
    }


def _map_path(repeat_root: Path, engine: str, iteration: int) -> Path:
    return repeat_root / engine / f"run_it{iteration:03d}_class001.mrc"


def audit_repeat_panel(
    *,
    scorecard_path: Path,
    case_id: str,
    panel_root: Path,
    repeat_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if repeat_count < 2:
        raise RepeatPanelError("repeat_count must be at least two")
    scorecard = _load_json(scorecard_path, label="scorecard")
    matches = [case for case in scorecard.get("cases", []) if case.get("id") == case_id]
    if len(matches) != 1:
        raise RepeatPanelError(f"expected one scorecard row for {case_id}, found {len(matches)}")
    case = matches[0]
    acceptance = scorecard["acceptance_contract"]
    checkpoints = tuple(int(value) for value in acceptance["required_checkpoints"])
    cross_min = float(acceptance["cross_engine_fsc_auc_min"])
    gt_min = float(acceptance["recovar_minus_relion_gt_fsc_auc_min"])

    repeats: list[dict[str, Any]] = []
    source_heads: set[str] = set()
    gpu_uuids: set[str] = set()
    relion_hashes: set[str] = set()
    for repeat_index in range(1, repeat_count + 1):
        root = panel_root / f"repeat-{repeat_index:02d}" / case_id
        trajectory = _load_json(root / "trajectory_audit.json", label="trajectory audit")
        provenance = _load_json(root / "run_provenance.json", label="run provenance")
        gpu = _load_json(root / "paired_gpu_uuid.json", label="paired GPU report")
        if trajectory.get("schema") != TRAJECTORY_SCHEMA or trajectory.get("case_id") != case_id:
            raise RepeatPanelError(f"repeat {repeat_index}: trajectory schema or case identity differs")
        if trajectory.get("suite_id") != scorecard.get("suite_id"):
            raise RepeatPanelError(f"repeat {repeat_index}: suite identity differs")
        if not bool(trajectory.get("artifact_topology_exact")):
            raise RepeatPanelError(f"repeat {repeat_index}: artifact topology differs")
        observed_checkpoints = tuple(int(row["iteration"]) for row in trajectory.get("checkpoints", ()))
        if observed_checkpoints != checkpoints:
            raise RepeatPanelError(f"repeat {repeat_index}: checkpoint topology differs")
        physical = str(gpu.get("physical_gpu_uuid", ""))
        if {
            physical,
            str(gpu.get("relion_gpu_uuid", "")),
            str(gpu.get("recovar_gpu_uuid", "")),
        } != {physical} or not physical.startswith("GPU-"):
            raise RepeatPanelError(f"repeat {repeat_index}: paired physical GPU identity differs")
        source_heads.add(str(provenance.get("git_head", "")))
        gpu_uuids.add(physical)
        relion_hashes.add(str(provenance.get("relion_reference", {}).get("executable_sha256", "")))
        repeats.append({"index": repeat_index, "root": root, "trajectory": trajectory})
    if len(source_heads) != 1 or any(len(value) != 40 for value in source_heads):
        raise RepeatPanelError(f"repeat panel contains mixed or invalid source heads: {sorted(source_heads)}")
    if len(gpu_uuids) != 1:
        raise RepeatPanelError(f"repeat panel did not stay on one physical GPU: {sorted(gpu_uuids)}")
    if len(relion_hashes) != 1 or any(len(value) != 64 for value in relion_hashes):
        raise RepeatPanelError("repeat panel contains mixed or invalid RELION executable hashes")

    shellwise: dict[str, np.ndarray] = {}
    checkpoint_rows = []
    for iteration in checkpoints:
        relion_volumes = [
            _load_relion_volume(_map_path(repeat["root"], "relion", iteration)) for repeat in repeats
        ]
        recovar_volumes = [
            _load_relion_volume(_map_path(repeat["root"], "recovar", iteration)) for repeat in repeats
        ]
        relion_self = []
        recovar_self = []
        for lhs, rhs in itertools.combinations(range(repeat_count), 2):
            relion_self.append(
                _metric(
                    relion_volumes[lhs],
                    relion_volumes[rhs],
                    key=f"it{iteration:03d}_relion_r{lhs + 1:02d}_r{rhs + 1:02d}",
                    shellwise=shellwise,
                )
            )
            recovar_self.append(
                _metric(
                    recovar_volumes[lhs],
                    recovar_volumes[rhs],
                    key=f"it{iteration:03d}_recovar_r{lhs + 1:02d}_r{rhs + 1:02d}",
                    shellwise=shellwise,
                )
            )
        cross = np.empty((repeat_count, repeat_count), dtype=np.float64)
        for rec_index, recovar_volume in enumerate(recovar_volumes):
            for rel_index, relion_volume in enumerate(relion_volumes):
                cross[rec_index, rel_index] = _metric(
                    recovar_volume,
                    relion_volume,
                    key=f"it{iteration:03d}_recovar_r{rec_index + 1:02d}_relion_r{rel_index + 1:02d}",
                    shellwise=shellwise,
                )
        gt_deltas = []
        for repeat in repeats:
            row = next(
                item for item in repeat["trajectory"]["checkpoints"] if int(item["iteration"]) == iteration
            )
            gt_deltas.append(float(row["recovar_minus_relion_gt_fsc_auc"]))
        classification = classify_checkpoint(
            relion_self_fsc_auc=relion_self,
            recovar_self_fsc_auc=recovar_self,
            cross_engine_fsc_auc=cross.tolist(),
            gt_deltas=gt_deltas,
            cross_engine_min=cross_min,
            gt_delta_min=gt_min,
        )
        checkpoint_rows.append({"iteration": iteration, **classification})

    result = "pass" if all(row["pass"] for row in checkpoint_rows) else "fail"
    report = {
        "schema": SCHEMA,
        "suite_id": scorecard["suite_id"],
        "case_id": case_id,
        "case_name": case.get("name"),
        "result": result,
        "repeat_count": repeat_count,
        "source_head": next(iter(source_heads)),
        "physical_gpu_uuid": next(iter(gpu_uuids)),
        "relion_executable_sha256": next(iter(relion_hashes)),
        "thresholds": {
            "cross_engine_fsc_auc_min": cross_min,
            "recovar_minus_relion_gt_fsc_auc_min": gt_min,
        },
        "metric_policy": (
            "signed shellwise FSC and normalized non-DC FSC-AUC only; the frozen point gates are retained, "
            "candidate repeat variability must stay inside the same-GPU native RELION envelope, and every "
            "candidate and native repeat must have a cross-engine mode match at the frozen point gate; a "
            "separate non-scoring diagnostic uses each native repeat's nearest-native FSC-AUC radius"
        ),
        "correlation_used": False,
        "runtime": _runtime_summary(repeats),
        "individual_results": [repeat["trajectory"]["result"] for repeat in repeats],
        "checkpoints": checkpoint_rows,
    }
    return report, shellwise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shells-npz", type=Path, required=True)
    args = parser.parse_args(argv)
    report, shellwise = audit_repeat_panel(
        scorecard_path=args.scorecard.resolve(),
        case_id=args.case_id,
        panel_root=args.panel_root.resolve(),
        repeat_count=int(args.repeat_count),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_shells_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
