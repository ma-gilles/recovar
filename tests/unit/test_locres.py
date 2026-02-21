import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.locres as locres

pytestmark = pytest.mark.unit


def test_find_first_zero_in_bool_handles_mixed_and_all_ones():
    arr = np.array([True, True, False, False])
    out = int(np.asarray(locres.find_first_zero_in_bool(arr)))
    assert out == 2

    all_ones = np.array([True, True, True])
    out_all = int(np.asarray(locres.find_first_zero_in_bool(all_ones)))
    assert out_all == 2


def test_integral_fsc_uses_prefix_until_first_negative():
    fsc = np.array([0.9, 0.5, -0.1, 0.4], dtype=np.float32)
    out = float(np.asarray(locres.integral_fsc(fsc, fourier_pixel_size=2.0)))
    # Only first two entries should contribute.
    assert out == pytest.approx((0.9 + 0.5) * 2.0)


def test_local_error_subvolume_size_uses_multiplier():
    s1 = locres.get_local_error_subvolume_size(locres_maskrad=6.0, voxel_size=1.0, multiplier=1)
    s3 = locres.get_local_error_subvolume_size(locres_maskrad=6.0, voxel_size=1.0, multiplier=3)
    assert s3 == 3 * s1


def test_sampling_points_and_volume_shapes():
    sampling_points = np.asarray(locres.get_sampling_points(grid_size=16, locres_sampling=4, locres_maskrad=4, voxel_size=1.0))
    assert sampling_points.ndim == 2
    assert sampling_points.shape[1] == 3
    assert sampling_points.shape[0] > 0

    vol = locres.make_sampling_volume(grid_size=16, locres_sampling=4, voxel_size=1.0, locres_maskrad=4)
    assert vol.shape == (16, 16, 16)
    assert np.isfinite(vol).all()


def test_sampling_helpers_accept_none_maskrad_defaults():
    sp = np.asarray(locres.get_sampling_points(grid_size=8, locres_sampling=2, locres_maskrad=None, voxel_size=1.5))
    assert sp.ndim == 2 and sp.shape[1] == 3
    assert sp.shape[0] > 0

    vol = np.asarray(locres.make_sampling_volume(grid_size=8, locres_sampling=2, voxel_size=1.5, locres_maskrad=None))
    assert vol.shape == (8, 8, 8)
    assert np.isfinite(vol).all()


def test_expensive_local_error_with_cov_accepts_none_defaults():
    rng = np.random.default_rng(0)
    map1 = rng.normal(size=(8, 8, 8)).astype(np.float32)
    map2 = rng.normal(size=(8, 8, 8)).astype(np.float32)
    noise_variance = np.ones((8, 8, 8), dtype=np.float32)

    out = locres.expensive_local_error_with_cov(
        map1=map1,
        map2=map2,
        voxel_size=1.5,
        noise_variance=noise_variance,
        locres_sampling=2,
        locres_maskrad=None,
        locres_edgwidth=None,
        use_v2=True,
        split_shell=False,
    )
    out = np.asarray(out)
    assert out.shape == map1.shape
    assert np.isfinite(out).all()
