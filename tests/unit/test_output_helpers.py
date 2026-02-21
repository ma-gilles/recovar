import numpy as np
import pytest

pytest.importorskip("jax")
from recovar import output

pytestmark = pytest.mark.unit


def test_sum_over_other_matches_manual_reduction():
    x = np.arange(24).reshape(2, 3, 4)
    out = output.sum_over_other(x, use_axis=[1])
    expected = np.sum(x, axis=(0, 2))
    np.testing.assert_array_equal(out, expected)


def test_half_slice_other_kept_axes():
    density = np.arange(27).reshape(3, 3, 3)
    out = output.half_slice_other(density, axes=[0, 2])
    np.testing.assert_array_equal(out, density[:, 1, :])


def test_slice_at_point_kept_axes():
    density = np.arange(27).reshape(3, 3, 3)
    point = np.array([2, 0, 1])
    out = output.slice_at_point(density, axes=[0, 2], point=point)
    np.testing.assert_array_equal(out, density[:, 0, :])


def test_resample_trajectory_uses_interpolated_indices(monkeypatch):
    monkeypatch.setattr(output, "get_resampled_distances", lambda _: np.array([0.0, 2.0, 4.0]))
    gt_vols = np.zeros((3, 8))
    indices = output.resample_trajectory(gt_vols, n_vols_along_path=5)
    np.testing.assert_array_equal(indices, np.array([0, 0, 1, 2, 2]))
