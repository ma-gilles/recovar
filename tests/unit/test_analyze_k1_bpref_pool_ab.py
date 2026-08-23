import numpy as np

from scripts.analyze_k1_bpref_pool_ab import _metric, _movement, _spherical_mask


def test_spherical_mask_pins_native_case4_support():
    mask = _spherical_mask((115, 115, 115), radius=56)

    assert mask.shape == (115**3,)
    assert np.count_nonzero(mask) == 735317


def test_metric_and_movement_report_improvement():
    target = np.asarray([1.0, 2.0], dtype=np.float64)
    control = _metric(np.asarray([1.0, 2.5]), target)
    candidate = _metric(np.asarray([1.0, 2.25]), target)

    movement = _movement(control, candidate)

    assert movement["classification"] == "improved"
    assert movement["candidate_over_control_relative_l2"] == 0.5
    assert candidate["value_equal_fraction"] == 0.5
