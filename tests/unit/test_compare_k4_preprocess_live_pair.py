import numpy as np

from scripts.compare_k4_preprocess_live_pair import _center, _residual_summary


def test_live_preprocess_center_removes_common_score_offset():
    np.testing.assert_array_equal(
        _center(np.asarray([101, 103, 108], dtype=np.float32)),
        np.asarray([-3, -1, 4], dtype=np.float64),
    )


def test_live_preprocess_residual_summary_tracks_centered_energy():
    report = _residual_summary(
        np.asarray([10, 20, 30], dtype=np.float32),
        np.asarray([111, 118, 131], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        report["delta_recovar_minus_relion"],
        np.asarray([1, -2, 1], dtype=np.float64),
    )
    assert report["residual_energy"] == 6.0
    assert report["residual_l2"] == np.sqrt(6.0)
    assert report["residual_max_abs"] == 2.0
