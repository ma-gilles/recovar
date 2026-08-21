import numpy as np

from scripts.analyze_k1_direction_prior_serialization import compare_direction_priors


def test_direction_prior_serialization_reports_probability_and_log_loss():
    live = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=np.float32)
    serialized = np.asarray([0.0, 0.333333, 0.666667], dtype=np.float32)

    report = compare_direction_priors(live, serialized)

    assert report["zero_mask_exact"]
    assert report["probability"]["mismatch_count"] == 2
    assert report["finite_log_probability"]["mismatch_count"] == 2
    assert report["maximum_log_delta"]["direction_index"] == 1
    assert report["finite_log_probability"]["max_abs"] > 0.0
