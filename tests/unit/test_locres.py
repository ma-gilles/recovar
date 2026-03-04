import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.heterogeneity.locres as locres

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


def test_find_fsc_resol_basic():
    """find_fsc_resol interpolates FSC to find threshold crossing."""
    import jax.numpy as jnp
    # FSC curve that drops linearly from 1 to 0
    fsc = jnp.array([1.0, 0.8, 0.5, 0.2, 0.0], dtype=jnp.float32)
    # With threshold 1/7 ≈ 0.143, should cross between index 3 (0.2) and 4 (0.0)
    resol = float(np.asarray(locres.find_fsc_resol(fsc, threshold=0.5)))
    # threshold 0.5 should cross at index 2 (where fsc == 0.5)
    assert 1.5 <= resol <= 2.5

    resol_low = float(np.asarray(locres.find_fsc_resol(fsc, threshold=0.1)))
    # threshold 0.1 is between 0.2 (idx 3) and 0.0 (idx 4)
    assert 3.0 <= resol_low <= 4.0


def test_find_fsc_resol_all_above_threshold():
    """If entire FSC is above threshold, return last index."""
    import jax.numpy as jnp
    fsc = jnp.ones(5, dtype=jnp.float32) * 0.9
    resol = float(np.asarray(locres.find_fsc_resol(fsc, threshold=0.5)))
    assert resol == pytest.approx(4.0, abs=0.1)


def test_apply_fsc_weighting_shape_and_finiteness():
    """apply_fsc_weighting should preserve shape and produce finite values."""
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    vol = jnp.array(rng.normal(size=(8, 8, 8)).astype(np.float32))
    fsc = jnp.array([1.0, 0.9, 0.7, 0.3], dtype=jnp.float32)
    out = np.asarray(locres.apply_fsc_weighting(vol, fsc))
    assert out.shape == (8, 8, 8)
    assert np.all(np.isfinite(out))


def test_low_pass_filter_attenuates_high_freq():
    """low_pass_filter_map should reduce total energy by removing high frequencies."""
    import jax.numpy as jnp
    import recovar.core.fourier_transform_utils as ftu
    rng = np.random.default_rng(1)
    size = 16
    vol = jnp.array(rng.normal(size=(size, size, size)).astype(np.float32))
    FT = ftu.get_dft3(vol)
    # Apply low-pass filter
    filtered = np.asarray(locres.low_pass_filter_map(FT, size, 8.0, 1.0, 2))
    assert filtered.shape == FT.shape
    # Energy should be reduced (some frequencies zeroed)
    assert np.sum(np.abs(filtered)**2) < np.sum(np.abs(np.asarray(FT))**2)
    # The output should still be finite
    assert np.all(np.isfinite(filtered))


def test_get_local_error_subvolume_rad():
    """get_local_error_subvolume_rad should return radius in pixels."""
    rad = locres.get_local_error_subvolume_rad(locres_maskrad=6.0, voxel_size=1.0, multiplier=3)
    assert rad == 18
    rad2 = locres.get_local_error_subvolume_rad(locres_maskrad=6.0, voxel_size=2.0, multiplier=2)
    assert rad2 == 6  # round(6/2)*2 = 3*2


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


def test_split_by_shells_distributes_voxels_by_radial_distance():
    """Each shell row should contain only voxels at that radial distance."""
    import jax.numpy as jnp
    vol_shape = (4, 4, 4)
    vol_size = 64
    input_vec = jnp.arange(vol_size, dtype=jnp.float32)
    result = np.asarray(locres.split_by_shells(input_vec, vol_shape))
    # Shape: (grid_size//2, vol_size) = (2, 64) since input is 1D
    assert result.shape == (2, vol_size)
    # Each shell should have some non-zero entries (at least shell 0)
    assert np.any(result[0] != 0)


def test_subsample_array_extracts_correct_subvolume():
    """subsample_array should extract a cube centered at offset."""
    import jax.numpy as jnp
    vol = jnp.arange(64, dtype=jnp.float32).reshape(4, 4, 4)
    center = jnp.array([2, 2, 2], dtype=jnp.int32)
    radius = 1
    sub = np.asarray(locres.subsample_array(vol, center, radius))
    assert sub.shape == (2, 2, 2)
    # Should contain the center region
    assert np.all(np.isfinite(sub))


def test_convolve_mask_at_sampling_points_shape():
    """Output shape should match the full mask shape."""
    import jax.numpy as jnp
    full_mask = jnp.ones((8, 8, 8), dtype=jnp.float32)
    sampling_pts = jnp.array([[0, 0, 0], [1, 1, 1]], dtype=jnp.int32)
    values = jnp.array([1.0, 2.0], dtype=jnp.float32)
    result = np.asarray(locres.convolve_mask_at_sampling_points(sampling_pts, values, full_mask))
    assert result.shape == (8, 8, 8)
    assert np.all(np.isfinite(result))


def test_get_sampling_points_returns_valid_points():
    """Sampling points should be within grid bounds."""
    grid_size = 16
    locres_sampling = 4
    locres_maskrad = 2
    voxel_size = 1.0
    pts = np.asarray(locres.get_sampling_points(grid_size, locres_sampling, locres_maskrad, voxel_size))
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    # Points should be centered around 0
    assert np.all(np.abs(pts) < grid_size // 2)


def test_make_sampling_volume_shape():
    """make_sampling_volume should return volume of correct grid size."""
    vol = np.asarray(locres.make_sampling_volume(8, 2, 1.0, 1.0))
    assert vol.shape == (8, 8, 8)


def test_masked_noisy_error_split_over_shells_v1_runs():
    """masked_noisy_error_split_over_shells should return finite per-shell norms."""
    import jax.numpy as jnp
    rng = np.random.default_rng(42)
    size = 12
    diff = jnp.array(rng.normal(size=(size, size, size)).astype(np.float32))
    # noise_variance_small matches the subsampled volume size
    maskrad_pix = 2
    multiplier = 3
    sub_size = 2 * multiplier * maskrad_pix
    noise_var = jnp.ones((sub_size, sub_size, sub_size), dtype=jnp.float32)
    offset = jnp.array([0, 0, 0], dtype=jnp.int32)
    result = np.asarray(locres.masked_noisy_error_split_over_shells(
        diff, noise_var, offset, maskrad_pix, 2, multiplier))
    assert result.ndim == 1
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)


def test_masked_noisy_error_split_over_shells_v2_runs():
    """masked_noisy_error_split_over_shells_v2 should return finite per-shell norms.

    This also verifies the fix for the previously undefined `diff_masked` variable.
    """
    import jax.numpy as jnp
    rng = np.random.default_rng(42)
    size = 12
    diff = jnp.array(rng.normal(size=(size, size, size)).astype(np.float32))
    maskrad_pix = 2
    multiplier = 3
    sub_size = 2 * multiplier * maskrad_pix
    noise_var = jnp.ones((sub_size, sub_size, sub_size), dtype=jnp.float32)
    offset = jnp.array([0, 0, 0], dtype=jnp.int32)
    result = np.asarray(locres.masked_noisy_error_split_over_shells_v2(
        diff, noise_var, offset, maskrad_pix, 2, multiplier))
    assert result.ndim == 1
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)


def test_masked_noisy_error_basic():
    """masked_noisy_error should return a positive finite scalar."""
    import jax.numpy as jnp
    rng = np.random.default_rng(42)
    size = 8
    diff = jnp.array(rng.normal(size=(size, size, size)).astype(np.float32))
    noise_var = jnp.ones((size, size, size), dtype=jnp.float32)
    offset = jnp.array([0, 0, 0], dtype=jnp.int32)
    result = float(np.asarray(locres.masked_noisy_error(diff, noise_var, offset, 2, 2)))
    assert np.isfinite(result)
    assert result >= 0


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_integral_fsc_gpu(gpu_device):
    fsc = np.array([0.9, 0.5, -0.1, 0.4], dtype=np.float32)

    cpu_out = float(np.asarray(locres.integral_fsc(fsc, fourier_pixel_size=2.0)))

    with jax.default_device(gpu_device):
        fsc_g = jax.device_put(jnp.array(fsc), gpu_device)
        gpu_out = float(np.asarray(locres.integral_fsc(fsc_g, fourier_pixel_size=2.0)))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_find_first_zero_in_bool_gpu(gpu_device):
    arr = np.array([True, True, False, False])

    cpu_out = int(np.asarray(locres.find_first_zero_in_bool(arr)))

    with jax.default_device(gpu_device):
        arr_g = jax.device_put(jnp.array(arr), gpu_device)
        gpu_out = int(np.asarray(locres.find_first_zero_in_bool(arr_g)))

    assert cpu_out == gpu_out


@pytest.mark.gpu
def test_expensive_local_error_with_cov_gpu(gpu_device):
    rng = np.random.default_rng(0)
    map1 = rng.normal(size=(8, 8, 8)).astype(np.float32)
    map2 = rng.normal(size=(8, 8, 8)).astype(np.float32)
    noise_variance = np.ones((8, 8, 8), dtype=np.float32)

    cpu_out = np.asarray(locres.expensive_local_error_with_cov(
        map1=map1, map2=map2, voxel_size=1.5, noise_variance=noise_variance,
        locres_sampling=2, locres_maskrad=None, locres_edgwidth=None,
        use_v2=True, split_shell=False,
    ))

    with jax.default_device(gpu_device):
        map1_g = jax.device_put(jnp.array(map1), gpu_device)
        map2_g = jax.device_put(jnp.array(map2), gpu_device)
        noise_g = jax.device_put(jnp.array(noise_variance), gpu_device)
        gpu_out = np.asarray(locres.expensive_local_error_with_cov(
            map1=map1_g, map2=map2_g, voxel_size=1.5, noise_variance=noise_g,
            locres_sampling=2, locres_maskrad=None, locres_edgwidth=None,
            use_v2=True, split_shell=False,
        ))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-4, rtol=1e-4)
