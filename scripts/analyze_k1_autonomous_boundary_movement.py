#!/usr/bin/env python3
"""Measure whether a K=1 autonomous boundary candidate moves toward RELION."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "recovar.em.k1_autonomous_boundary_movement.v1"
POSE_THRESHOLD_DEG = 0.01
TRANSLATION_THRESHOLD_ANGSTROM = 0.01


def _relative_l2(delta: np.ndarray, reference: np.ndarray) -> float:
    delta64 = np.asarray(delta, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    denominator = float(np.linalg.norm(reference64.reshape(-1)))
    if denominator == 0.0:
        raise ValueError("relative-L2 reference norm is zero")
    return float(np.linalg.norm(delta64.reshape(-1)) / denominator)


def _load_particle_arrays(path: Path, iteration: int) -> dict[str, Any]:
    prefix = f"it{iteration:03d}"
    with np.load(path, allow_pickle=False) as payload:
        return {
            "identity_sha256": str(np.asarray(payload["identity_sha256"]).item()),
            "rows": np.asarray(payload["identity_row_index"], dtype=np.int64),
            "pmax_recovar": np.asarray(payload[f"{prefix}_pmax_recovar"], dtype=np.float64),
            "pmax_relion": np.asarray(payload[f"{prefix}_pmax_relion"], dtype=np.float64),
            "support_recovar": np.asarray(payload[f"{prefix}_support_recovar"], dtype=np.int64),
            "support_relion": np.asarray(payload[f"{prefix}_support_relion"], dtype=np.int64),
            "rotation_error_deg": np.asarray(
                payload[f"{prefix}_rotation_geodesic_deg"], dtype=np.float64
            ),
            "translation_error_angstrom": np.asarray(
                payload[f"{prefix}_translation_l2"], dtype=np.float64
            ),
        }


def _load_signed_fsc_auc(path: Path, relion_iteration: int) -> float:
    report = json.loads(path.read_text())
    matches = [
        item
        for item in report["numbered_iterations"]
        if int(item["relion_iteration"]) == relion_iteration
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one RELION iteration {relion_iteration} in {path}, got {len(matches)}"
        )
    return float(matches[0]["cross_engine"]["merged"]["signed_fsc_auc"])


def _arm_metrics(state: dict[str, Any], signed_fsc_auc: float) -> dict[str, Any]:
    pmax_delta = state["pmax_recovar"] - state["pmax_relion"]
    support_bad = state["support_recovar"] != state["support_relion"]
    abs_delta = np.abs(pmax_delta)
    return {
        "pmax": {
            "relative_l2": _relative_l2(pmax_delta, state["pmax_relion"]),
            "rmse": float(np.sqrt(np.mean(np.square(pmax_delta)))),
            "mean_abs": float(np.mean(abs_delta)),
            "p95_abs": float(np.quantile(abs_delta, 0.95)),
            "p99_abs": float(np.quantile(abs_delta, 0.99)),
            "max_abs": float(np.max(abs_delta)),
            "mean_signed": float(np.mean(pmax_delta)),
        },
        "support": {
            "mismatch_count": int(np.count_nonzero(support_bad)),
            "mismatch_source_rows": state["rows"][support_bad].astype(int).tolist(),
        },
        "pose_error_gt_0p01_deg_count": int(
            np.count_nonzero(state["rotation_error_deg"] > POSE_THRESHOLD_DEG)
        ),
        "translation_error_gt_0p01_angstrom_count": int(
            np.count_nonzero(
                state["translation_error_angstrom"] > TRANSLATION_THRESHOLD_ANGSTROM
            )
        ),
        "merged_signed_fsc_auc": signed_fsc_auc,
    }


def analyze(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_fsc_auc: float,
    candidate_fsc_auc: float,
) -> dict[str, Any]:
    if baseline["identity_sha256"] != candidate["identity_sha256"]:
        raise ValueError("particle identity hashes differ between arms")
    for key in ("rows", "pmax_relion", "support_relion"):
        if not np.array_equal(baseline[key], candidate[key]):
            raise ValueError(f"RELION reference field differs between arms: {key}")

    baseline_metrics = _arm_metrics(baseline, baseline_fsc_auc)
    candidate_metrics = _arm_metrics(candidate, candidate_fsc_auc)
    baseline_abs = np.abs(baseline["pmax_recovar"] - baseline["pmax_relion"])
    candidate_abs = np.abs(candidate["pmax_recovar"] - candidate["pmax_relion"])
    baseline_support_bad = baseline["support_recovar"] != baseline["support_relion"]
    candidate_support_bad = candidate["support_recovar"] != candidate["support_relion"]

    baseline_deficit = 1.0 - baseline_fsc_auc
    candidate_deficit = 1.0 - candidate_fsc_auc
    deficit_ratio = None if baseline_deficit == 0.0 else candidate_deficit / baseline_deficit
    movement = {
        "pmax_abs_error": {
            "improved_count": int(np.count_nonzero(candidate_abs < baseline_abs)),
            "equal_count": int(np.count_nonzero(candidate_abs == baseline_abs)),
            "worsened_count": int(np.count_nonzero(candidate_abs > baseline_abs)),
            "relative_l2_improvement_factor": (
                baseline_metrics["pmax"]["relative_l2"]
                / candidate_metrics["pmax"]["relative_l2"]
                if candidate_metrics["pmax"]["relative_l2"] != 0.0
                else None
            ),
        },
        "support": {
            "fixed_source_rows": baseline["rows"][
                baseline_support_bad & ~candidate_support_bad
            ].astype(int).tolist(),
            "new_source_rows": baseline["rows"][
                ~baseline_support_bad & candidate_support_bad
            ].astype(int).tolist(),
            "retained_source_rows": baseline["rows"][
                baseline_support_bad & candidate_support_bad
            ].astype(int).tolist(),
        },
        "merged_fsc_deficit_ratio_candidate_over_baseline": deficit_ratio,
    }
    gates = {
        "pmax_relative_l2_not_worse": bool(
            candidate_metrics["pmax"]["relative_l2"]
            <= baseline_metrics["pmax"]["relative_l2"]
        ),
        "support_mismatch_count_not_worse": bool(
            candidate_metrics["support"]["mismatch_count"]
            <= baseline_metrics["support"]["mismatch_count"]
        ),
        "no_new_support_mismatch_rows": not movement["support"]["new_source_rows"],
        "pose_outliers_not_worse": bool(
            candidate_metrics["pose_error_gt_0p01_deg_count"]
            <= baseline_metrics["pose_error_gt_0p01_deg_count"]
        ),
        "translation_outliers_not_worse": bool(
            candidate_metrics["translation_error_gt_0p01_angstrom_count"]
            <= baseline_metrics["translation_error_gt_0p01_angstrom_count"]
        ),
        "merged_signed_fsc_auc_not_worse": bool(candidate_fsc_auc >= baseline_fsc_auc),
    }
    return {
        "schema": SCHEMA,
        "metric_policy": "source-row state metrics and signed non-DC FSC-AUC; no correlation",
        "identity_sha256": baseline["identity_sha256"],
        "arms": {"baseline": baseline_metrics, "candidate": candidate_metrics},
        "movement": movement,
        "gates": gates,
        "classification": (
            "moves_toward_relion_without_measured_regression"
            if all(gates.values())
            else "mixed_or_regressive_boundary_result"
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# K=1 autonomous boundary movement",
        "",
        "| Arm | Pmax relative L2 | Pmax RMSE | Support mismatches | Pose outliers | Translation outliers | Merged signed FSC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "candidate"):
        arm = report["arms"][name]
        lines.append(
            f"| {name} | {arm['pmax']['relative_l2']:.12g} | "
            f"{arm['pmax']['rmse']:.12g} | {arm['support']['mismatch_count']} | "
            f"{arm['pose_error_gt_0p01_deg_count']} | "
            f"{arm['translation_error_gt_0p01_angstrom_count']} | "
            f"{arm['merged_signed_fsc_auc']:.13f} |"
        )
    lines.extend(["", f"Classification: **{report['classification']}**.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-particles", type=Path, required=True)
    parser.add_argument("--candidate-particles", type=Path, required=True)
    baseline_fsc = parser.add_mutually_exclusive_group(required=True)
    baseline_fsc.add_argument("--baseline-fsc", type=Path)
    baseline_fsc.add_argument("--baseline-fsc-auc", type=float)
    candidate_fsc = parser.add_mutually_exclusive_group(required=True)
    candidate_fsc.add_argument("--candidate-fsc", type=Path)
    candidate_fsc.add_argument("--candidate-fsc-auc", type=float)
    parser.add_argument("--particle-iteration", type=int, default=2)
    parser.add_argument("--relion-iteration", type=int, default=2)
    parser.add_argument("--scope", default="K=1 autonomous boundary A/B")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    baseline_fsc_auc = (
        float(args.baseline_fsc_auc)
        if args.baseline_fsc_auc is not None
        else _load_signed_fsc_auc(args.baseline_fsc, args.relion_iteration)
    )
    candidate_fsc_auc = (
        float(args.candidate_fsc_auc)
        if args.candidate_fsc_auc is not None
        else _load_signed_fsc_auc(args.candidate_fsc, args.relion_iteration)
    )
    report = analyze(
        _load_particle_arrays(args.baseline_particles, args.particle_iteration),
        _load_particle_arrays(args.candidate_particles, args.particle_iteration),
        baseline_fsc_auc,
        candidate_fsc_auc,
    )
    report["scope"] = args.scope
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
