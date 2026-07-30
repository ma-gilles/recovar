from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_em_k1_final_grid_fsc_deficit import (
    CLASSIFICATION,
    build_report,
    normalized_non_dc_fsc_auc,
    partition_fsc_deficit,
)


def test_partition_fsc_deficit_closes_normalized_auc() -> None:
    curve = np.asarray([1.0, 0.99, 0.98, 0.9, 0.8, 0.7])
    report = partition_fsc_deficit(curve, last_numbered_radius=2)

    assert report["fsc_auc_deficit"] == pytest.approx(
        1.0 - normalized_non_dc_fsc_auc(curve)
    )
    assert report["inside_or_at_radius_deficit"] + report["outside_radius_deficit"] == pytest.approx(
        report["fsc_auc_deficit"]
    )
    assert report["outside_radius_fraction"] > report["inside_or_at_radius_fraction"]
    assert report["active_radius_normalized_fsc_auc"] == pytest.approx(
        normalized_non_dc_fsc_auc(curve[:3])
    )


def _inputs(tmp_path: Path, *, outside_dominated: bool = True) -> tuple[Path, Path]:
    numbered_curve = np.ones(21)
    numbered_curve[1:] -= np.linspace(1.0e-7, 1.0e-5, 20)
    final_curve = np.ones(21)
    if outside_dominated:
        final_curve[1:5] -= 1.0e-4
        final_curve[5:] -= 0.02
    else:
        final_curve[1:5] -= 0.02
        final_curve[5:] -= 1.0e-4
    arrays = {}
    numbered_row = {"relion_iteration": 7, "cross_engine": {}}
    final_row = {"cross_engine": {}}
    for product in ("half1", "half2", "merged"):
        numbered_key = f"it007_cross_{product}"
        final_key = f"final_cross_{product}"
        arrays[numbered_key] = numbered_curve
        arrays[final_key] = final_curve
        numbered_row["cross_engine"][product] = {
            "fsc_auc": normalized_non_dc_fsc_auc(numbered_curve),
            "shellwise_key": numbered_key,
        }
        final_row["cross_engine"][product] = {
            "fsc_auc": normalized_non_dc_fsc_auc(final_curve),
            "shellwise_key": final_key,
        }
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema": "em_k1_fsc_trajectory_audit_v2",
                "numbered_iterations": [numbered_row],
                "final": final_row,
            }
        )
    )
    shellwise_path = tmp_path / "shellwise.npz"
    np.savez(shellwise_path, **arrays)
    return trajectory_path, shellwise_path


def test_build_report_accepts_outside_radius_amplification(tmp_path: Path) -> None:
    trajectory, shellwise = _inputs(tmp_path)
    report = build_report(
        trajectory_json=trajectory,
        shellwise_npz=shellwise,
        relion_iteration=7,
        last_numbered_current_size=8,
    )

    assert report["status"] == "pass"
    assert report["classification"] == CLASSIFICATION
    assert report["products"]["merged"]["final"]["outside_radius_fraction"] > 0.95
    assert report["products"]["merged"]["final_over_numbered_deficit_amplification"] > 250.0
    assert report["products"]["merged"]["active_radius_fsc_auc_gate_passed"]
    assert report["products"]["merged"]["outside_radius_fsc_auc_gate_failed"]


def test_build_report_fails_closed_for_inside_radius_deficit(tmp_path: Path) -> None:
    trajectory, shellwise = _inputs(tmp_path, outside_dominated=False)
    with pytest.raises(ValueError, match="outside-radius deficit gate"):
        build_report(
            trajectory_json=trajectory,
            shellwise_npz=shellwise,
            relion_iteration=7,
            last_numbered_current_size=8,
        )
