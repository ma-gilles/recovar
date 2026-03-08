import pytest

from helpers.metrics_regression import compare_metric, metric_direction

pytestmark = pytest.mark.unit


def test_metric_direction_classification():
    assert metric_direction("noise_mean_relative_error") == "lower"
    assert metric_direction("state_0_ninety_pc_locres") == "lower"
    assert metric_direction("mean_fsc") == "higher"
    assert metric_direction("noise_correlation") == "higher"
    assert metric_direction("pcs_relative_variance_4") == "higher"
    assert metric_direction("pcs_relative_variance_10") == "higher"
    assert metric_direction("some_unknown_metric") == "ignore"


def test_compare_metric_lower_and_higher():
    ok, _ = compare_metric(current=1.05, baseline=1.0, direction="lower", tol_frac=0.1)
    assert ok
    ok, _ = compare_metric(current=1.20, baseline=1.0, direction="lower", tol_frac=0.1)
    assert not ok

    ok, _ = compare_metric(current=0.95, baseline=1.0, direction="higher", tol_frac=0.1)
    assert ok
    ok, _ = compare_metric(current=0.70, baseline=1.0, direction="higher", tol_frac=0.1)
    assert not ok


def test_compare_metric_non_finite_fails():
    ok, msg = compare_metric(current=float("nan"), baseline=1.0, direction="lower", tol_frac=0.1)
    assert not ok
    assert "non-finite" in msg


def test_metric_direction_for_canonical_key_names():
    """Verify that renamed canonical keys are classified correctly."""
    assert metric_direction("svd_relative_variance_4") == "higher"
    assert metric_direction("svd_relative_variance_10") == "higher"
    assert metric_direction("contrast_abs_error_4") == "lower"
    assert metric_direction("contrast_abs_error_4_noreg") == "lower"
    assert metric_direction("state_0_locres_90pct") == "lower"
    assert metric_direction("state_1_locres_median") == "lower"
