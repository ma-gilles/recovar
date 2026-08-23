from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_vdam_real_trajectory import (
    _compare_active_particle_state,
    _particle_state_gate,
)

SBATCH_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_vdam_relion_real_data_case.sbatch"


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


def test_real_data_sbatch_sets_paired_launch_mode_before_gpu_gate():
    text = SBATCH_PATH.read_text()

    launch_mode = text.index("export CUDA_LAUNCH_BLOCKING=1")
    gpu_gate = text.index('"${PIXI_PY}" - <<\'PY\'')
    assert launch_mode < gpu_gate
    assert "RECOVAR_CUDA_MODE_ARGS+=(--deterministic_cuda)" in text
    assert '"cuda_launch_blocking_value"' in text


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


def test_active_particle_comparison_ignores_stale_relion_classes_outside_selected_prefix():
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
            ("1@stack.mrcs", 0, 90, 90, 90, 10, 10, 1),
            ("2@stack.mrcs", 1, 0, 0, 0, 0, 0, 0.4),
        ],
        columns=columns,
    )
    relion = pd.DataFrame(
        [
            ("2@stack.mrcs", 1, 0, 0, 0, 0, 0, 0.41),
            ("1@stack.mrcs", 1, -90, -90, -90, -10, -10, 0),
        ],
        columns=columns,
    )

    report = _compare_active_particle_state(
        recovar,
        relion,
        iteration=1,
        pose_tolerance_deg=1e-3,
        translation_tolerance_angst=1e-4,
        active_image_ids={"2@stack.mrcs"},
    )

    assert report["evaluated_particle_count"] == 1
    assert report["relion_visited_particle_count"] == 1
    assert report["visited_topology_exact"] is True
