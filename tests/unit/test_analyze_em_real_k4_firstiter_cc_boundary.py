"""Unit tests for the real K-class firstiter-CC boundary analyzer."""

import numpy as np
import pytest

from scripts.analyze_em_real_k4_firstiter_cc_boundary import (
    AnalysisError,
    relion_round,
    relion_to_recovar_rotation_indices,
    score_metrics,
)

pytestmark = pytest.mark.unit


def test_relion_to_recovar_rotation_indices_transposes_direction_psi_order():
    mapping = relion_to_recovar_rotation_indices(3, 2)
    np.testing.assert_array_equal(mapping, np.asarray([0, 3, 1, 4, 2, 5]))
    np.testing.assert_array_equal(np.sort(mapping), np.arange(6))


def test_relion_round_uses_half_away_from_zero():
    values = np.asarray([-2.5, -2.49, -0.5, 0.0, 0.5, 2.49, 2.5])
    np.testing.assert_array_equal(relion_round(values), [-3, -2, -1, 0, 1, 2, 3])


def test_score_metrics_distinguishes_additive_offset_from_shape_error():
    reference = np.asarray([1.0, 2.0, 4.0, 8.0])
    shifted = reference + 3.25
    metrics = score_metrics(reference, shifted)
    assert metrics["correlation"] == pytest.approx(1.0)
    assert metrics["centered_relative_l2"] == pytest.approx(0.0)
    assert metrics["mean_offset"] == pytest.approx(3.25)
    assert metrics["max_abs"] == pytest.approx(3.25)


def test_score_metrics_rejects_shape_mismatch_and_constant_reference():
    with pytest.raises(AnalysisError, match="different shapes"):
        score_metrics(np.ones(3), np.ones(4))
    with pytest.raises(AnalysisError, match="constant"):
        score_metrics(np.ones(3), np.arange(3))
