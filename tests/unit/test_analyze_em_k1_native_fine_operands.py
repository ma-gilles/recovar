"""Unit tests for the focused native K=1 fine-score operand analyzer."""

import numpy as np
import pytest

from scripts.analyze_em_k1_native_fine_operands import (
    _full_to_compact,
    _infer_float32_common_addend,
    _score_unit_factors,
    _tree_sum,
    _validate_alternate_reference_layout,
    _winner_boundary,
)

pytestmark = pytest.mark.unit


def test_alternate_reference_contract_does_not_require_same_candidate_mask():
    rotations = np.eye(3, dtype=np.float32)[None]
    window = np.asarray([3, 5], dtype=np.int32)
    primary = {
        "rotations": rotations,
        "window_indices": window,
        "candidate_mask": np.asarray([[True, False]]),
    }
    alternate = {
        "rotations": rotations.copy(),
        "window_indices": window.copy(),
        "candidate_mask": np.asarray([[False, True]]),
    }

    _validate_alternate_reference_layout(primary, alternate)

    alternate["window_indices"] = np.asarray([3, 6], dtype=np.int32)
    with pytest.raises(ValueError, match="different window_indices"):
        _validate_alternate_reference_layout(primary, alternate)


def test_tree_sum_matches_relion_lane_order():
    values = np.arange(600, dtype=np.float32).reshape(2, 300) / np.float32(17.0)
    lanes = np.zeros((2, 256), dtype=np.float32)
    lanes[:, :] += values[:, :256]
    lanes[:, :44] += values[:, 256:]
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes = lanes[:, :width] + lanes[:, width : 2 * width]

    np.testing.assert_array_equal(_tree_sum(values), lanes[:, 0])


def test_full_to_compact_maps_centered_rows_to_relion_fftw_rows():
    # Full-size centered rows 2, 3, 4 correspond to ky -2, -1, 0. In a
    # current-size-4 packed FFT they become RELION rows 2, 3, 0.
    window = np.array([2 * 5, 3 * 5 + 1, 4 * 5 + 2], dtype=np.int32)

    lookup = _full_to_compact(window, full_size=8, current_size=4)

    expected = np.full(12, -1, dtype=np.int32)
    expected[[2 * 3, 3 * 3 + 1, 2]] = np.arange(3, dtype=np.int32)
    np.testing.assert_array_equal(lookup, expected)


def test_infer_float32_common_addend_replays_large_costs():
    base = np.linspace(250.0, 280.0, 1000, dtype=np.float32)
    expected_addend = np.float32(0.029396063)
    target = np.asarray(base + expected_addend, dtype=np.float32)

    inferred, exact = _infer_float32_common_addend(base, target)

    np.testing.assert_array_equal(base + inferred, target)
    assert exact == base.size


def test_score_unit_factors_distinguish_native_and_normalized_frames():
    assert _score_unit_factors("native", 384) == (np.float32(1.0), np.float32(1.0))
    assert _score_unit_factors("normalized", 384) == (
        np.float32(384**2),
        np.float32(384**4),
    )
    with pytest.raises(ValueError, match="unsupported score units"):
        _score_unit_factors("unknown", 384)


def test_winner_boundary_reports_opposite_native_and_recovar_preferences():
    report = _winner_boundary(
        native_raw_cost=np.asarray([10.0, 9.5]),
        native_log_prior=np.asarray([0.0, 0.0]),
        native_probability=np.asarray([0.4, 0.6]),
        native_significant=np.asarray([False, True]),
        recovar_preprior_score=np.asarray([-9.4, -9.5]),
        recovar_log_prior=np.asarray([0.0, 0.0]),
        recovar_total_score=np.asarray([-9.4, -9.5]),
        recovar_probability=np.asarray([0.525, 0.475]),
        recovar_significant=np.asarray([True, False]),
        recovar_rotation_row=np.asarray([3, 7]),
        translation_row=np.asarray([107, 110]),
    )

    assert report["same_winner"] is False
    assert report["native_winner"]["translation_row"] == 110
    assert report["recovar_winner"]["translation_row"] == 107
    assert report["native_top_preprior_score_margin"] == pytest.approx(0.5)
    assert report["native_top_log_prior_margin"] == pytest.approx(0.0)
    assert report["recovar_top_preprior_score_margin"] == pytest.approx(0.1)
    assert report["recovar_top_log_prior_margin"] == pytest.approx(0.0)
    assert report["native_preference_native_minus_recovar_winner"] == pytest.approx(0.5)
    assert report["recovar_preference_recovar_minus_native_winner"] == pytest.approx(0.1)
