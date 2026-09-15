import numpy as np

from scripts.analyze_vdam_posterior_support_boundary import (
    _partial_rotation_match,
    summarize_support_boundary,
)


def _rotation(value: float) -> np.ndarray:
    result = np.eye(3, dtype=np.float32)
    result[0, 1] = np.float32(value)
    return result


def test_partial_rotation_match_reports_both_unmatched_sides():
    native = np.stack([_rotation(0.0), _rotation(0.1), _rotation(0.2)])
    recovar = np.stack([_rotation(0.0), _rotation(0.2), _rotation(0.3)])
    mapping, recovar_only, distances = _partial_rotation_match(native, recovar, 1.0e-6)
    np.testing.assert_array_equal(mapping, np.asarray([0, -1, 1]))
    np.testing.assert_array_equal(recovar_only, np.asarray([2]))
    assert distances[1] > 1.0e-6


def test_support_summary_attributes_missing_mass_and_winner():
    native_rotations = np.stack([_rotation(0.0), _rotation(0.1), _rotation(0.2)])
    recovar_rotations = np.stack([_rotation(0.0), _rotation(0.2)])
    native_probs = np.asarray([[0.10, 0.05], [0.50, 0.05], [0.20, 0.10]], dtype=np.float32)
    recovar_probs = np.asarray([[0.30, 0.10], [0.40, 0.20]], dtype=np.float32)

    report = summarize_support_boundary(
        native_rotations,
        native_probs,
        recovar_rotations,
        recovar_probs,
        rotation_tolerance=1.0e-6,
    )

    support = report["rotation_support"]
    assert support["matched_count"] == 2
    assert support["native_only_rows"] == [1]
    np.testing.assert_allclose(support["native_only_retained_mass"], 0.55)
    assert report["winner"]["native_rotation_recovar_row"] == -1
    assert report["winner"]["same_hypothesis"] is False
