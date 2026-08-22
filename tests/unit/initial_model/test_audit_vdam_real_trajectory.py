from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_vdam_real_trajectory import (
    _compare_active_particle_state,
    _particle_state_gate,
)


def _acceptance() -> dict[str, float]:
    return {
        "pose_tolerance_deg": 1e-3,
        "translation_tolerance_angst": 1e-4,
        "pmax_absolute_error_p95_max": 5e-3,
        "pmax_absolute_error_max": 1e-2,
    }


def _state(*, divergent: int = 0, p95: float = 1e-3, maximum: float = 2e-3):
    return {
        "identity_alignment_exact": True,
        "visited_topology_exact": True,
        "divergent_particle_count": divergent,
        "pmax_absolute_error": {"p95": p95, "max": maximum},
    }


def test_particle_state_gate_accepts_exact_topology_and_bounded_pmax():
    report = _particle_state_gate(_state(), _acceptance())

    assert report["pass"] is True
    assert all(report["checks"].values())


def test_particle_state_gate_rejects_topology_or_pmax_drift():
    report = _particle_state_gate(
        _state(divergent=1, p95=6e-3, maximum=2e-2),
        _acceptance(),
    )

    assert report["pass"] is False
    assert report["checks"] == {
        "identity_alignment_exact": True,
        "visited_topology_exact": True,
        "zero_pose_or_translation_mismatches": False,
        "pmax_p95_within_tolerance": False,
        "pmax_max_within_tolerance": False,
    }


def test_active_particle_comparison_uses_selected_subset_and_checks_topology():
    columns = (
        "_rlnImageName",
        "_rlnClassNumber",
        "_rlnAngleRot",
        "_rlnAngleTilt",
        "_rlnAnglePsi",
        "_rlnOriginXAngst",
        "_rlnOriginYAngst",
        "_rlnMaxValueProbDistribution",
    )
    recovar = pd.DataFrame(
        [
            ("1@stack.mrcs", 0, 0, 0, 0, 0, 0, 0),
            ("2@stack.mrcs", 1, 0, 0, 0, 0, 0, 0.4),
        ],
        columns=columns,
    )
    relion = pd.DataFrame(
        [
            ("2@stack.mrcs", 1, 0, 0, 0, 0, 0, 0.41),
            ("1@stack.mrcs", 0, 90, 90, 90, 10, 10, 1),
        ],
        columns=columns,
    )

    report = _compare_active_particle_state(
        recovar,
        relion,
        iteration=1,
        pose_tolerance_deg=1e-3,
        translation_tolerance_angst=1e-4,
    )

    assert report["full_particle_count"] == 2
    assert report["evaluated_particle_count"] == 1
    assert report["visited_topology_exact"] is True
    assert report["pmax_absolute_error"]["max"] == pytest.approx(0.01)
