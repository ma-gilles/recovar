import numpy as np
import pytest

from scripts.analyze_vdam_direct_repeat_map_spread import (
    DirectRepeatMapSpreadError,
    summarize_repeat_map_spread,
)


def _scalar_map(value: float) -> np.ndarray:
    return np.asarray([[[value]]], dtype=np.float64)


@pytest.mark.unit
def test_repeat_map_spread_reports_candidate_outlier_against_native_diameter():
    report = summarize_repeat_map_spread(
        [_scalar_map(1.0), _scalar_map(5.0)],
        [_scalar_map(1.0), _scalar_map(2.0)],
    )

    assert not report["pass"]
    assert report["native_diameter_relative_l2"] == 0.5
    assert report["candidate_diameter_relative_l2"] == 0.8
    assert report["candidate_nearest_native_relative_l2"] == [0.0, 0.6]
    assert report["candidate_nearest_over_native_diameter"] == [0.0, 1.2]
    assert report["candidate_within_native_diameter"] == [True, False]
    assert report["candidate_indices_outside_native_diameter"] == [2]


@pytest.mark.unit
def test_repeat_map_spread_accepts_each_candidate_inside_native_panel():
    report = summarize_repeat_map_spread(
        [_scalar_map(1.0), _scalar_map(3.0)],
        [_scalar_map(1.0), _scalar_map(2.0)],
    )

    assert report["pass"]
    assert report["candidate_nearest_native_index"] == [1, 2]
    assert report["candidate_within_native_diameter"] == [True, True]
    assert report["candidate_indices_outside_native_diameter"] == []


@pytest.mark.unit
def test_repeat_map_spread_rejects_mixed_shapes():
    with pytest.raises(DirectRepeatMapSpreadError, match="candidate/native map shapes differ"):
        summarize_repeat_map_spread(
            [np.zeros((1, 1, 1)), np.ones((1, 1, 1))],
            [np.zeros((2, 2, 2)), np.ones((2, 2, 2))],
        )
