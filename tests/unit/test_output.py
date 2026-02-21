import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import output

pytestmark = pytest.mark.unit


def test_get_resampled_distances_and_resample_trajectory():
    vols = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    d = output.get_resampled_distances(vols)
    idx = output.resample_trajectory(vols, n_vols_along_path=3)
    assert np.all(d[1:] >= d[:-1])
    assert idx.shape == (3,)
    assert idx[0] == 0
    assert idx[-1] == 4


def test_sum_over_other_and_slice_helpers():
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    summed = output.sum_over_other(x, use_axis=[0, 2])
    assert summed.shape == (2, 4)
    assert np.allclose(summed, x.sum(axis=1))

    density = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    half = output.half_slice_other(density, axes=[0, 2])
    assert half.shape == (5, 7)
    assert np.allclose(half, density[:, density.shape[1] // 2, :])

    pt = [1, 2, 3]
    sl = output.slice_at_point(density, axes=[0, 2], point=pt)
    assert sl.shape == (5, 7)
    assert np.allclose(sl, density[:, pt[1], :])
