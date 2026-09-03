from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.analyze_vdam_stable_shape_trajectory import (
    ARM_ORDER,
    GateSetupError,
    _parse_args,
    normalized_l2,
    summarize_pair_distances,
    summarize_runtime_panel,
    summarize_size_schedule,
    validate_stable_flat_capacity_execution,
)


def test_analyzer_cli_can_isolate_flat_row_abi_from_fourier_shapes():
    args = _parse_args(
        [
            "--root",
            "/tmp/root",
            "--acceptance",
            "/tmp/acceptance.json",
            "--output",
            "/tmp/report.json",
            "--markdown-output",
            "/tmp/report.md",
            "--no-stable-fourier-window-shapes",
            "--stable-flat-row-capacity",
        ]
    )

    assert args.stable_fourier_window_shapes is False
    assert args.stable_flat_row_capacity is True


def test_normalized_l2_is_symmetric_and_scale_normalized():
    left = np.asarray([1.0, 2.0, 3.0])
    right = np.asarray([1.0, 2.0, 4.0])
    assert normalized_l2(left, right) == normalized_l2(right, left)
    assert normalized_l2(7.0 * left, 7.0 * right) == pytest.approx(normalized_l2(left, right))
    assert normalized_l2(np.zeros(3), np.zeros(3)) == 0.0
    assert normalized_l2(np.zeros(3), np.ones(3)) == pytest.approx(math.sqrt(2.0))


def test_pair_panel_keeps_repeat_and_cross_mode_distances_separate():
    values = {
        "stable_off_1": np.asarray([0.0, 1.0]),
        "stable_on_1": np.asarray([0.0, 1.2]),
        "stable_on_2": np.asarray([0.0, 1.3]),
        "stable_off_2": np.asarray([0.0, 1.1]),
    }
    report = summarize_pair_distances(values)
    assert report["control_repeat"]["left"] == "stable_off_1"
    assert report["candidate_repeat"]["right"] == "stable_on_2"
    assert len(report["cross_mode"]) == 4
    assert report["maximum_any_pair_distance"] == max(
        report["maximum_repeat_distance"], report["maximum_cross_mode_distance"]
    )


def test_pair_panel_rejects_missing_arm():
    values = {name: np.ones(2) for name in ARM_ORDER[:-1]}
    with pytest.raises(GateSetupError, match="wrong arms"):
        summarize_pair_distances(values)


def test_runtime_panel_applies_speed_and_memory_gates_to_medians():
    rows = {
        "stable_off_1": {
            "end_to_end_wall_s": 100.0,
            "expectation_stage_s": 80.0,
            "peak_gpu_memory_mib": 1000.0,
        },
        "stable_on_1": {
            "end_to_end_wall_s": 89.0,
            "expectation_stage_s": 69.0,
            "peak_gpu_memory_mib": 1030.0,
        },
        "stable_on_2": {
            "end_to_end_wall_s": 91.0,
            "expectation_stage_s": 71.0,
            "peak_gpu_memory_mib": 1010.0,
        },
        "stable_off_2": {
            "end_to_end_wall_s": 102.0,
            "expectation_stage_s": 82.0,
            "peak_gpu_memory_mib": 1000.0,
        },
    }
    report = summarize_runtime_panel(
        rows,
        minimum_speedup_percent=5.0,
        maximum_memory_increase_percent=5.0,
    )
    assert report["median_control_end_to_end_wall_s"] == 101.0
    assert report["median_candidate_end_to_end_wall_s"] == 90.0
    assert report["median_end_to_end_percent_change"] == pytest.approx(-10.8910891089)
    assert report["pass"] is True


def test_runtime_panel_fails_when_only_wall_time_improves():
    rows = {
        name: {
            "end_to_end_wall_s": 90.0 if "on" in name else 100.0,
            "expectation_stage_s": 100.0,
            "peak_gpu_memory_mib": 1000.0,
        }
        for name in ARM_ORDER
    }
    report = summarize_runtime_panel(
        rows,
        minimum_speedup_percent=5.0,
        maximum_memory_increase_percent=5.0,
    )
    assert report["end_to_end_pass"] is True
    assert report["expectation_stage_pass"] is False
    assert report["pass"] is False


def _stable_flat_meta(*, enabled: bool, flat_rows: list[int], padded_rows: list[int]):
    return {
        "requested_stable_flat_row_capacity": enabled,
        "effective_stable_flat_row_capacity": enabled,
        "halfset_0_profile_summary": {
            "flat_local_rows_enabled": True,
            "stable_flat_row_capacity_enabled": enabled,
            "chunk_flat_score_rows": flat_rows,
            "chunk_padded_rotations": padded_rows,
        },
    }


def test_stable_flat_execution_accepts_fixed_capacity_and_counts_ordinary_reductions():
    ordinary = validate_stable_flat_capacity_execution(
        _stable_flat_meta(enabled=False, flat_rows=[7, 11], padded_rows=[12, 12]),
        expected=False,
        arm="stable_off_1",
        iteration=9,
    )
    stable = validate_stable_flat_capacity_execution(
        _stable_flat_meta(enabled=True, flat_rows=[12, 12], padded_rows=[12, 12]),
        expected=True,
        arm="stable_on_1",
        iteration=9,
    )

    assert ordinary["strict_reduction_count"] == 2
    assert ordinary["flat_row_sum"] == 18
    assert stable["strict_reduction_count"] == 0
    assert stable["flat_row_sum"] == stable["padded_row_sum"] == 24


def test_stable_flat_execution_rejects_candidate_with_data_dependent_rows():
    meta = _stable_flat_meta(enabled=True, flat_rows=[11], padded_rows=[12])
    with pytest.raises(GateSetupError, match=r"did not use fixed B \* R rows"):
        validate_stable_flat_capacity_execution(
            meta,
            expected=True,
            arm="stable_on_1",
            iteration=9,
        )


def test_schedule_divergence_is_a_science_failure_not_a_setup_error():
    """A basin split must still produce a complete diagnostic report."""

    metas = {
        arm: {"current_size": 84 if arm == "stable_on_1" else 78}
        for arm in ARM_ORDER
    }
    sizes, failure = summarize_size_schedule(metas, 46)

    assert sizes["stable_on_1"] == 84
    assert sizes["stable_off_1"] == 78
    assert failure == {
        "iteration": 46,
        "feature": "current_size_schedule",
        "values": sizes,
    }
