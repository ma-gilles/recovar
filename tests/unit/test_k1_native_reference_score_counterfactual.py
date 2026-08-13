import numpy as np
import pytest

from scripts.analyze_k1_native_reference_score_counterfactual import (
    _apply_high_shell_retained_fraction,
    _apply_shell_factors,
    _apply_uniform_pixel_weight_factor,
    _pixel_weight_shell_stats,
    _raw_parent_margin,
)


def test_uniform_pixel_weight_factor_preserves_zero_support():
    actual = _apply_uniform_pixel_weight_factor(
        np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        0.5,
    )
    np.testing.assert_array_equal(actual, np.asarray([0.0, 0.5, 1.0], dtype=np.float32))


def test_shell_factors_change_only_selected_active_shells():
    weights = np.asarray([0.0, 2.0, 3.0, 4.0], dtype=np.float32)
    score_indices = np.asarray([20, 21, 22, 23], dtype=np.int32)
    actual, count = _apply_shell_factors(weights, score_indices, (8, 8), {1: 0.5, 3: 2.0})
    np.testing.assert_array_equal(actual, np.asarray([0.0, 1.0, 3.0, 8.0], dtype=np.float32))
    assert count == 2


@pytest.mark.unit
def test_raw_parent_margin_uses_rotation_identity_and_translation_rows():
    diff2 = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
        ],
        dtype=np.float32,
    )

    margin = _raw_parent_margin(
        diff2,
        np.asarray([15173, 29447]),
        native_parent=(29447, 1),
        recovar_parent=(15173, 2),
    )

    # Raw log score is -diff2, so native minus RECOVAR is 12 - 21.
    assert margin == -9.0


@pytest.mark.unit
def test_pixel_weight_shell_stats_use_centered_row_half_layout():
    # Packed centered-row half layout for a 4x4 image has width 3. These are
    # (x, y) = (1, -1), (1, 0), and (0, 1), all in rounded shell 1.
    indices = np.asarray([4, 7, 9], dtype=np.int32)
    reference = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    candidate = np.asarray([1.5, 3.0, 6.0], dtype=np.float32)

    report = _pixel_weight_shell_stats(candidate, reference, indices, (4, 4))

    assert report["active_count"] == 3
    assert report["support_mismatch_count"] == 0
    assert report["bitwise_mismatch_count"] == 3
    assert report["shell_count"] == 1
    assert report["shells"] == [
        {
            "shell": 1,
            "count": 3,
            "candidate_over_reference_min": 1.5,
            "candidate_over_reference_median": 1.5,
            "candidate_over_reference_max": 1.5,
            "candidate_over_reference_std": 0.0,
        }
    ]
    assert report["first_bitwise_mismatch"]["centered_y"] == -1
    assert report["first_bitwise_mismatch"]["x"] == 1


@pytest.mark.unit
def test_apply_high_shell_retained_fraction_changes_only_shells_above_cutoff():
    weights = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    # Centred packed-half rows for an 8x8 image: shells 0, 1, 2, and 3.
    score_indices = np.asarray([20, 21, 22, 23], dtype=np.int32)

    corrected, count = _apply_high_shell_retained_fraction(
        weights,
        score_indices,
        (8, 8),
        shell_cutoff=1,
        retained_fraction=0.999,
    )

    np.testing.assert_array_equal(corrected[:2], weights[:2])
    np.testing.assert_array_equal(corrected[2:], (weights[2:] * np.float32(0.999)).astype(np.float32))
    assert count == 2
