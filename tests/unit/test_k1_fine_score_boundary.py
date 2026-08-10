import numpy as np
import pytest

from scripts.analyze_k1_fine_score_boundary import (
    _classify_particle,
    _metric,
    _stable_top_n_mask,
)


@pytest.mark.unit
def test_stable_top_n_mask_preserves_tie_order():
    weights = np.asarray([0.5, 0.5, 0.25, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(
        _stable_top_n_mask(weights, 2),
        np.asarray([True, True, False, False]),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ("active_tuple_subset", "active_candidate_tuple_mismatch"),
        ("raw_diff2_close", "raw_fine_diff2_mismatch"),
        ("priors_close", "fine_prior_mismatch"),
        ("centered_log_weight_close", "fine_log_weight_arithmetic_mismatch"),
        ("posterior_close", "fine_normalized_posterior_mismatch"),
        ("support_exact", "fine_significant_support_mismatch"),
    ],
)
def test_classification_reports_first_unequal_boundary(changed, expected):
    values = {
        "active_tuple_subset": True,
        "raw_diff2_close": True,
        "priors_close": True,
        "centered_log_weight_close": True,
        "posterior_close": True,
        "support_exact": True,
    }
    values[changed] = False
    assert _classify_particle(**values) == expected


@pytest.mark.unit
def test_metric_uses_exact_and_relative_l2_without_correlation():
    reference = np.asarray([1.0, 2.0], dtype=np.float32)
    candidate = np.asarray([1.0, 2.001], dtype=np.float32)
    metric = _metric(reference, candidate)
    assert metric["exact_equal"] is False
    assert metric["relative_l2_over_reference"] > 0.0
    assert "correlation" not in metric
