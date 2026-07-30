#!/usr/bin/env python3
"""Localize a K=1 final-all-data FSC defect relative to the last numbered radius.

The input trajectory already contains the accepted shellwise FSC curves and
normalized non-DC FSC-AUC values.  This analyzer partitions each FSC-AUC
defect into trapezoid segments at or below the last numbered reconstruction
radius and segments introduced beyond that radius by the final full-grid
expectation.  Correlation is not computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "em-k1-final-grid-fsc-deficit-v2"
TRAJECTORY_SCHEMA = "em_k1_fsc_trajectory_audit_v2"
OUTSIDE_DEFICIT_FRACTION_GATE = 0.95
DEFICIT_AMPLIFICATION_GATE = 250.0
ACTIVE_RADIUS_FSC_AUC_MIN = 0.995
OUTSIDE_RADIUS_FSC_AUC_MAX = 0.995
CLASSIFICATION = (
    "final_full_grid_fsc_deficit_is_over_95pct_outside_last_numbered_radius"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_non_dc_fsc_auc(curve: np.ndarray) -> float:
    """Match the trajectory auditor's normalized non-DC trapezoid metric."""

    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    _require(values.size >= 3, "FSC curve needs DC and at least two non-DC shells")
    values = values[1:]
    _require(np.all(np.isfinite(values)), "FSC curve contains non-finite non-DC values")
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(values) / (values.size - 1))


def partition_fsc_deficit(
    curve: np.ndarray,
    *,
    last_numbered_radius: int,
) -> dict[str, float | int]:
    """Partition normalized FSC-AUC defect over non-DC shell segments."""

    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    auc = normalized_non_dc_fsc_auc(values)
    non_dc = values[1:]
    maximum_shell = non_dc.size
    _require(
        1 < last_numbered_radius < maximum_shell,
        "last numbered radius must be inside the final shell range",
    )
    deficit = 1.0 - non_dc
    segment_deficit = 0.5 * (deficit[:-1] + deficit[1:]) / (non_dc.size - 1)
    # Segment index 0 connects shells 1 and 2.  Segments through index
    # radius-2 have both endpoints at or below the numbered radius.
    inside = float(np.sum(segment_deficit[: last_numbered_radius - 1]))
    outside = float(np.sum(segment_deficit[last_numbered_radius - 1 :]))
    total = float(np.sum(segment_deficit))
    active_values = non_dc[:last_numbered_radius]
    outside_values = non_dc[last_numbered_radius - 1 :]
    integrate = getattr(np, "trapezoid", np.trapz)
    active_radius_auc = float(integrate(active_values) / (active_values.size - 1))
    outside_radius_auc = float(integrate(outside_values) / (outside_values.size - 1))
    _require(total > 0.0, "FSC-AUC defect must be positive")
    _require(np.isclose(total, 1.0 - auc, rtol=0.0, atol=2.0e-15), "deficit partition does not close")
    return {
        "normalized_non_dc_fsc_auc": auc,
        "fsc_auc_deficit": total,
        "inside_or_at_radius_deficit": inside,
        "outside_radius_deficit": outside,
        "inside_or_at_radius_fraction": inside / total,
        "outside_radius_fraction": outside / total,
        "active_radius_normalized_fsc_auc": active_radius_auc,
        "outside_radius_normalized_fsc_auc": outside_radius_auc,
        "last_numbered_radius": int(last_numbered_radius),
        "maximum_shell": int(maximum_shell),
    }


def build_report(
    *,
    trajectory_json: Path,
    shellwise_npz: Path,
    relion_iteration: int,
    last_numbered_current_size: int,
) -> dict[str, Any]:
    trajectory_json = trajectory_json.resolve()
    shellwise_npz = shellwise_npz.resolve()
    _require(trajectory_json.is_file(), f"missing trajectory JSON: {trajectory_json}")
    _require(shellwise_npz.is_file(), f"missing shellwise NPZ: {shellwise_npz}")
    trajectory = json.loads(trajectory_json.read_text())
    _require(trajectory.get("schema") == TRAJECTORY_SCHEMA, "wrong trajectory schema")
    _require(last_numbered_current_size > 0 and last_numbered_current_size % 2 == 0, "current size must be positive and even")
    last_numbered_radius = last_numbered_current_size // 2

    matches = [
        row
        for row in trajectory["numbered_iterations"]
        if int(row["relion_iteration"]) == relion_iteration
    ]
    _require(len(matches) == 1, "requested numbered iteration is absent or duplicated")
    numbered = matches[0]
    final = trajectory["final"]
    _require(final.get("status", "measured") != "not_available", "final all-data FSC is unavailable")

    products: dict[str, Any] = {}
    with np.load(shellwise_npz, allow_pickle=False) as archive:
        for product in ("half1", "half2", "merged"):
            numbered_metric = numbered["cross_engine"][product]
            final_metric = final["cross_engine"][product]
            numbered_key = numbered_metric["shellwise_key"]
            final_key = final_metric["shellwise_key"]
            _require(numbered_key in archive.files, f"missing numbered curve {numbered_key}")
            _require(final_key in archive.files, f"missing final curve {final_key}")
            numbered_partition = partition_fsc_deficit(
                archive[numbered_key],
                last_numbered_radius=last_numbered_radius,
            )
            final_partition = partition_fsc_deficit(
                archive[final_key],
                last_numbered_radius=last_numbered_radius,
            )
            _require(
                np.isclose(
                    numbered_partition["normalized_non_dc_fsc_auc"],
                    float(numbered_metric["fsc_auc"]),
                    rtol=0.0,
                    atol=2.0e-15,
                ),
                f"{product} numbered FSC-AUC does not replay",
            )
            _require(
                np.isclose(
                    final_partition["normalized_non_dc_fsc_auc"],
                    float(final_metric["fsc_auc"]),
                    rtol=0.0,
                    atol=2.0e-15,
                ),
                f"{product} final FSC-AUC does not replay",
            )
            amplification = (
                float(final_partition["fsc_auc_deficit"])
                / float(numbered_partition["fsc_auc_deficit"])
            )
            products[product] = {
                "numbered": numbered_partition,
                "final": final_partition,
                "final_over_numbered_deficit_amplification": amplification,
                "outside_fraction_gate_passed": bool(
                    float(final_partition["outside_radius_fraction"])
                    > OUTSIDE_DEFICIT_FRACTION_GATE
                ),
                "amplification_gate_passed": bool(
                    amplification > DEFICIT_AMPLIFICATION_GATE
                ),
                "active_radius_fsc_auc_gate_passed": bool(
                    float(final_partition["active_radius_normalized_fsc_auc"])
                    >= ACTIVE_RADIUS_FSC_AUC_MIN
                ),
                "outside_radius_fsc_auc_gate_failed": bool(
                    float(final_partition["outside_radius_normalized_fsc_auc"])
                    < OUTSIDE_RADIUS_FSC_AUC_MAX
                ),
            }

    _require(
        all(row["outside_fraction_gate_passed"] for row in products.values()),
        "one or more final products fail the outside-radius deficit gate",
    )
    _require(
        all(row["amplification_gate_passed"] for row in products.values()),
        "one or more final products fail the deficit-amplification gate",
    )
    _require(
        all(row["active_radius_fsc_auc_gate_passed"] for row in products.values()),
        "one or more final products fail FSC-AUC inside the numbered radius",
    )
    _require(
        all(row["outside_radius_fsc_auc_gate_failed"] for row in products.values()),
        "one or more final products do not fail FSC-AUC outside the numbered radius",
    )
    return {
        "schema": SCHEMA,
        "status": "pass",
        "classification": CLASSIFICATION,
        "metric_policy": (
            "signed shellwise FSC and normalized non-DC FSC-AUC; exact trapezoid-defect "
            "partition; no correlation; diagnostic only and not a fixed-scorecard promotion"
        ),
        "identity": {
            "relion_iteration": int(relion_iteration),
            "last_numbered_current_size": int(last_numbered_current_size),
            "last_numbered_radius": int(last_numbered_radius),
        },
        "fixed_gates": {
            "final_outside_radius_deficit_fraction_strictly_greater_than": OUTSIDE_DEFICIT_FRACTION_GATE,
            "final_over_numbered_deficit_amplification_strictly_greater_than": DEFICIT_AMPLIFICATION_GATE,
            "active_radius_fsc_auc_min": ACTIVE_RADIUS_FSC_AUC_MIN,
            "outside_radius_fsc_auc_strictly_below": OUTSIDE_RADIUS_FSC_AUC_MAX,
            "required_products": ["half1", "half2", "merged"],
        },
        "products": products,
        "input_artifacts": {
            "trajectory_json": {
                "path": str(trajectory_json),
                "sha256": _sha256(trajectory_json),
            },
            "shellwise_npz": {
                "path": str(shellwise_npz),
                "sha256": _sha256(shellwise_npz),
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-json", required=True, type=Path)
    parser.add_argument("--shellwise-npz", required=True, type=Path)
    parser.add_argument("--relion-iteration", required=True, type=int)
    parser.add_argument("--last-numbered-current-size", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_report(
        trajectory_json=args.trajectory_json,
        shellwise_npz=args.shellwise_npz,
        relion_iteration=args.relion_iteration,
        last_numbered_current_size=args.last_numbered_current_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
