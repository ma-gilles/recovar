"""Unit tests for recovar.core.mask."""

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, distance_transform_edt

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.mask as mask
import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.utils as utils

pytestmark = pytest.mark.unit


def _reference_threshold_map(arr, prob=0.99, dthresh=None):
    if dthresh is None:
        x2 = np.sort(arr.flatten())
        f2 = np.arange(len(x2)) / float(len(x2) - 1)
        loc = np.where(f2 >= prob)
        thresh = x2[loc[0][0]]
    else:
        thresh = dthresh
    return arr * (arr > thresh)


def _reference_soften_volume_mask(binary_volume_mask, kern_rad=3):
    distance_to_mask = distance_transform_edt(binary_volume_mask < 0.9)
    softened = np.zeros_like(binary_volume_mask)
    softened = np.where(
        (distance_to_mask >= 0) & (distance_to_mask < kern_rad),
        0.5 + 0.5 * np.cos(np.pi * distance_to_mask / kern_rad),
        softened,
    )
    return np.asarray(softened.astype(np.float32))


def _reference_make_soft_edged_kernel(r1, shape):
    if r1 < 3:
        boxsize = 5
    else:
        boxsize = 2 * r1 + 1

    volume_coords = (
        np.asarray(fourier_transform_utils.get_k_coordinate_of_each_pixel(shape, voxel_size=1, scaled=False)).reshape(
            list(shape) + [len(list(shape))]
        )
        + 1
    )
    distances = np.linalg.norm(volume_coords, axis=-1)
    half_boxsize = boxsize // 2
    r1 = half_boxsize
    r0 = r1 - 2

    kern = np.where(distances < r0, 1.0, 0.0)
    kern = np.where(
        (distances <= r1) & (distances >= r0),
        (1 + np.cos(np.pi * (distances - r0) / (r1 - r0))) / 2.0,
        kern,
    )
    return kern / np.sum(kern)


def _reference_local_correlation_3d(half1, half2, kern):
    import scipy.signal

    loc3_A = scipy.signal.fftconvolve(half1, kern, "same")
    loc3_A2 = scipy.signal.fftconvolve(half1 * half1, kern, "same")
    loc3_B = scipy.signal.fftconvolve(half2, kern, "same")
    loc3_B2 = scipy.signal.fftconvolve(half2 * half2, kern, "same")
    loc3_AB = scipy.signal.fftconvolve(half1 * half2, kern, "same")
    cov3_AB = loc3_AB - loc3_A * loc3_B
    var3_A = loc3_A2 - loc3_A**2
    var3_B = loc3_B2 - loc3_B**2
    reg_a = np.max(var3_A) / 1000
    reg_b = np.max(var3_B) / 1000
    var3_A = np.where(var3_A < reg_a, reg_a, var3_A)
    var3_B = np.where(var3_B < reg_b, reg_b, var3_B)
    return cov3_AB / np.sqrt(var3_A * var3_B)


def _reference_make_mask_from_half_maps(halfmap1, halfmap2, smax=3):
    kern = _reference_make_soft_edged_kernel(smax, halfmap1.shape)
    h1 = _reference_threshold_map(halfmap1)
    h2 = _reference_threshold_map(halfmap2)
    halfcc3d = _reference_local_correlation_3d(h1, h2, kern)
    halfcc3d *= np.asarray(mask.get_radial_mask(halfmap1.shape))
    ccmap_binary = (halfcc3d >= 1e-3).astype(int)
    dilated = binary_dilation(ccmap_binary, iterations=int(6 * halfmap1.shape[0] // 128))
    softened = _reference_soften_volume_mask(dilated, kern_rad=2)
    return softened * (softened >= 1e-3)


def _reference_make_mask_from_gt(gt_map, smax=3, iter=10, from_ft=True):
    del smax
    vol_shape = utils.guess_vol_shape_from_vol_size(gt_map.size) if from_ft else gt_map.shape
    if from_ft:
        vol_real = fourier_transform_utils.get_idft3(gt_map.reshape(vol_shape)).real
    else:
        vol_real = gt_map.reshape(vol_shape)
    thresholded = _reference_threshold_map(vol_real) > 0
    dilated = binary_dilation(thresholded, iterations=iter)
    return _reference_soften_volume_mask(dilated, kern_rad=2)


def _reference_make_moving_gt_mask(gt_volumes_real, volume_shape, smax=3, iter=1, dilation_iters=None, kern_rad=3):
    if dilation_iters is None:
        dilation_iters = int(np.ceil(6 * volume_shape[0] / 128))

    if isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 2:
        gt_volumes_real = [gt_volumes_real[i].reshape(volume_shape) for i in range(gt_volumes_real.shape[0])]
    elif isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 4:
        gt_volumes_real = [gt_volumes_real[i] for i in range(gt_volumes_real.shape[0])]
    elif isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 3:
        gt_volumes_real = [gt_volumes_real]

    volumes = np.asarray([np.asarray(vol).reshape(volume_shape) for vol in gt_volumes_real], dtype=np.float32)
    mean_volume = np.mean(volumes, axis=0)
    moving_signal = np.sqrt(np.mean((volumes - mean_volume[None]) ** 2, axis=0))
    moving_mask = _reference_make_mask_from_gt(moving_signal, smax=smax, iter=iter, from_ft=False) > 0.5
    if dilation_iters > 0 and np.any(moving_mask):
        moving_mask = binary_dilation(moving_mask, iterations=dilation_iters)

    binary_mask = np.asarray(moving_mask, dtype=bool)
    if np.any(binary_mask):
        soft_mask = _reference_soften_volume_mask(binary_mask, kern_rad=kern_rad)
    else:
        soft_mask = np.zeros(volume_shape, dtype=np.float32)
    return np.asarray(soft_mask, dtype=np.float32), binary_mask


# ---------------------------------------------------------------------------
# threshold_map
# ---------------------------------------------------------------------------


class TestThresholdMap:
    def test_basic_threshold(self):
        arr = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = mask.threshold_map(arr, dthresh=1.0)
        expected = np.array([0.0, 0.0, 0.0, 1.5, 2.0])
        np.testing.assert_allclose(result, expected)

    def test_prob_threshold(self):
        arr = np.arange(100, dtype=np.float32)
        result = mask.threshold_map(arr, prob=0.95)
        # 95th percentile of 0..99 is 95
        assert np.sum(result > 0) <= 6  # top ~5% are nonzero
        assert result[-1] == arr[-1]  # largest value kept

    def test_all_zero_below_thresh(self):
        arr = np.ones(10) * 0.5
        result = mask.threshold_map(arr, dthresh=1.0)
        np.testing.assert_allclose(result, 0.0)

    def test_matches_reference_threshold_map(self):
        arr = np.linspace(-2.0, 3.0, 57, dtype=np.float32).reshape(3, 19)
        np.testing.assert_allclose(
            mask.threshold_map(arr, prob=0.93),
            _reference_threshold_map(arr, prob=0.93),
        )


# ---------------------------------------------------------------------------
# smooth_circular_mask
# ---------------------------------------------------------------------------


class TestSmoothCircularMask:
    def test_shape(self):
        m = mask.smooth_circular_mask(32, radius=10, thickness=5)
        assert m.shape == (32, 32)

    def test_center_is_one(self):
        m = mask.smooth_circular_mask(32, radius=10, thickness=5)
        assert m[16, 16] == 1.0

    def test_far_corner_is_zero(self):
        m = mask.smooth_circular_mask(32, radius=8, thickness=3)
        assert m[0, 0] == 0.0

    def test_values_between_0_and_1(self):
        m = mask.smooth_circular_mask(64, radius=20, thickness=10)
        assert np.all(m >= 0.0)
        assert np.all(m <= 1.0)

    def test_radial_monotone(self):
        """Mask decreases from center outward."""
        m = mask.smooth_circular_mask(32, radius=10, thickness=5)
        center = m[16, 16]
        edge = m[0, 16]
        assert center >= edge


class TestRelionSoftImageMask:
    def test_matches_smooth_circular_mask_parameterization(self):
        image_size = 128
        pixel_size = 4.25
        particle_diameter_ang = 200.0
        width_mask_edge_px = 5.0

        result = mask.relion_soft_image_mask(
            image_size=image_size,
            pixel_size=pixel_size,
            particle_diameter_ang=particle_diameter_ang,
            width_mask_edge_px=width_mask_edge_px,
        )
        expected = mask.smooth_circular_mask(
            image_size=image_size,
            radius=particle_diameter_ang / (2.0 * pixel_size),
            thickness=width_mask_edge_px,
        )
        np.testing.assert_allclose(result, expected)

    def test_radius_tracks_relion_particle_diameter(self):
        image_size = 128
        pixel_size = 4.25
        particle_diameter_ang = 200.0
        width_mask_edge_px = 5.0

        result = mask.relion_soft_image_mask(
            image_size=image_size,
            pixel_size=pixel_size,
            particle_diameter_ang=particle_diameter_ang,
            width_mask_edge_px=width_mask_edge_px,
        )

        half = image_size // 2
        radius_px = particle_diameter_ang / (2.0 * pixel_size)
        inside = int(np.floor(radius_px)) - 1
        outside = int(np.ceil(radius_px + width_mask_edge_px)) + 1

        assert result[half, half + inside] == pytest.approx(1.0, abs=1e-6)
        assert result[half, half + outside] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# window_mask
# ---------------------------------------------------------------------------


class TestWindowMask:
    def test_shape(self):
        m = mask.window_mask(32, in_rad=0.5, out_rad=0.8)
        assert m.shape == (32, 32)

    def test_center_is_one(self):
        m = mask.window_mask(32, in_rad=0.8, out_rad=1.0)
        # Center pixel should be inside in_rad
        assert m[16, 16] == pytest.approx(1.0, abs=0.01)

    def test_values_clamped(self):
        m = mask.window_mask(64, in_rad=0.3, out_rad=0.6)
        assert np.all(m >= 0.0)
        assert np.all(m <= 1.0)


# ---------------------------------------------------------------------------
# get_radial_mask
# ---------------------------------------------------------------------------


class TestGetRadialMask:
    def test_3d_shape(self):
        m = mask.get_radial_mask((8, 8, 8))
        assert m.shape == (8, 8, 8)

    def test_center_is_true(self):
        m = mask.get_radial_mask((8, 8, 8))
        # Center voxel should be inside the mask
        assert bool(m[4, 4, 4])

    def test_custom_radius(self):
        m_small = mask.get_radial_mask((16, 16, 16), radius=3)
        m_large = mask.get_radial_mask((16, 16, 16), radius=6)
        assert float(jnp.sum(m_small)) < float(jnp.sum(m_large))

    def test_2d_shape(self):
        m = mask.get_radial_mask((10, 10))
        assert m.shape == (10, 10)


# ---------------------------------------------------------------------------
# make_soft_edged_kernel
# ---------------------------------------------------------------------------


class TestCreateSoftEdgedKernel:
    def test_3d_kernel_shape(self):
        k = mask.make_soft_edged_kernel(3, (8, 8, 8))
        assert k.shape == (8, 8, 8)

    def test_sums_to_one(self):
        k = mask.make_soft_edged_kernel(3, (8, 8, 8))
        np.testing.assert_allclose(float(jnp.sum(k)), 1.0, atol=1e-5)

    def test_non_negative(self):
        k = mask.make_soft_edged_kernel(5, (12, 12, 12))
        assert float(jnp.min(k)) >= 0.0

    def test_small_radius(self):
        k = mask.make_soft_edged_kernel(2, (6, 6, 6))
        np.testing.assert_allclose(float(jnp.sum(k)), 1.0, atol=1e-5)

    def test_matches_reference_kernel(self):
        got = np.asarray(mask.make_soft_edged_kernel(4, (10, 10, 10)))
        expected = _reference_make_soft_edged_kernel(4, (10, 10, 10))
        np.testing.assert_allclose(got, expected, atol=2e-9, rtol=3e-6)


# ---------------------------------------------------------------------------
# soften_volume_mask
# ---------------------------------------------------------------------------


class TestSoftenVolumeMask:
    def test_output_range(self):
        binary = np.zeros((16, 16, 16), dtype=np.float32)
        binary[4:12, 4:12, 4:12] = 1.0
        result = mask.soften_volume_mask(binary, kern_rad=3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-5

    def test_interior_preserved(self):
        binary = np.zeros((20, 20, 20), dtype=np.float32)
        binary[5:15, 5:15, 5:15] = 1.0
        result = mask.soften_volume_mask(binary, kern_rad=2)
        # Deep interior should still be ~1
        np.testing.assert_allclose(result[9, 9, 9], 1.0, atol=0.05)

    def test_matches_reference_softening(self):
        binary = np.zeros((12, 12, 12), dtype=np.float32)
        binary[3:9, 4:8, 2:10] = 1.0
        np.testing.assert_allclose(
            mask.soften_volume_mask(binary, kern_rad=3),
            _reference_soften_volume_mask(binary, kern_rad=3),
        )


# ---------------------------------------------------------------------------
# raised_cosine_mask
# ---------------------------------------------------------------------------


class TestRaisedCosineMask:
    def test_shape(self):
        vol_shape = (8, 8, 8)
        m = mask.raised_cosine_mask(vol_shape, radius=2, radius_p=4, offset=np.zeros(3))
        assert m.shape == vol_shape

    def test_center_is_one(self):
        vol_shape = (16, 16, 16)
        m = mask.raised_cosine_mask(vol_shape, radius=5, radius_p=8, offset=np.zeros(3))
        # Center should be inside the inner radius
        np.testing.assert_allclose(float(m[8, 8, 8]), 1.0, atol=1e-5)

    def test_values_clamped(self):
        vol_shape = (12, 12, 12)
        m = mask.raised_cosine_mask(vol_shape, radius=3, radius_p=5, offset=np.zeros(3))
        assert float(jnp.min(m)) >= -1e-6
        assert float(jnp.max(m)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# soft_mask_outside_map
# ---------------------------------------------------------------------------


class TestSoftMaskOutsideMap:
    def test_shape_preserved(self):
        vol = jnp.ones((8, 8, 8))
        result, m = mask.soft_mask_outside_map(vol, radius=3, cosine_width=2)
        assert result.shape == (8, 8, 8)
        assert m.shape == (8, 8, 8)

    def test_mask_values(self):
        vol = jnp.ones((12, 12, 12))
        result, m = mask.soft_mask_outside_map(vol, radius=4, cosine_width=2)
        assert float(jnp.min(m)) >= 0.0
        assert float(jnp.max(m)) <= 1.0 + 1e-5

    def test_matches_relion_background_weighting(self):
        shape = (9, 9, 9)
        radius = 2.5
        cosine_width = 2.0
        vol = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

        coords = np.asarray(
            fourier_transform_utils.get_k_coordinate_of_each_pixel(shape, voxel_size=1, scaled=False)
        ).reshape(shape + (3,))
        r = np.linalg.norm(coords, axis=-1)
        radius_p = radius + cosine_width
        raised_cos = 0.5 + 0.5 * np.cos(np.pi * (radius_p - r) / cosine_width)
        protein_weight = np.zeros(shape, dtype=np.float32)
        protein_weight = np.where(r < radius, 1.0, protein_weight)
        protein_weight = np.where((r >= radius) & (r <= radius_p), 1.0 - raised_cos, protein_weight)
        background_weight = np.zeros(shape, dtype=np.float32)
        background_weight = np.where(r > radius_p, 1.0, background_weight)
        background_weight = np.where((r >= radius) & (r <= radius_p), raised_cos, background_weight)
        avg_bg = np.sum(vol * background_weight) / np.sum(background_weight)
        expected = protein_weight * vol + background_weight * avg_bg

        result, returned_mask = mask.soft_mask_outside_map(jnp.asarray(vol), radius=radius, cosine_width=cosine_width)

        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(np.asarray(returned_mask), protein_weight, rtol=1e-6, atol=1e-6)

    def test_raised_cosine_mask_matches_relion_solvent_flatten_mask(self):
        shape = (9, 9, 9)
        radius = 2.5
        width = 2.0
        coords = np.asarray(
            fourier_transform_utils.get_k_coordinate_of_each_pixel_3d(shape, voxel_size=1, scaled=False)
        ).reshape(shape + (3,))
        r = np.linalg.norm(coords, axis=-1)
        expected = np.zeros(shape, dtype=np.float32)
        expected = np.where(r < radius, 1.0, expected)
        expected = np.where(
            (r >= radius) & (r <= radius + width),
            0.5 - 0.5 * np.cos(np.pi * (radius + width - r) / width),
            expected,
        )

        result = mask.raised_cosine_mask(shape, radius=radius, radius_p=radius + width, offset=jnp.zeros(3))

        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Reference equivalence
# ---------------------------------------------------------------------------


class TestMakeUnionGtMask:
    """Tests for make_union_gt_mask."""

    def test_single_volume_matches_per_volume(self):
        """Union mask of a single volume should match the per-volume mask."""
        rng = np.random.RandomState(42)
        vol = rng.randn(16, 16, 16).astype(np.float32)
        vol[4:12, 4:12, 4:12] += 5.0  # strong signal region

        per_vol = mask.make_mask_from_gt(vol, smax=3, iter=1, from_ft=False)
        per_vol_binary = per_vol > 0.5

        soft_union, binary_union = mask.make_union_gt_mask(
            [vol], volume_shape=(16, 16, 16), smax=3, iter=1, dilation_iters=1, kern_rad=3
        )

        # The union of a single volume should match
        per_vol_dilated = binary_dilation(per_vol_binary, iterations=1)
        np.testing.assert_array_equal(binary_union, per_vol_dilated)

    def test_union_of_two_blobs_is_superset(self):
        """Union of two non-overlapping blobs covers both regions."""
        from scipy.ndimage import gaussian_filter

        vol_shape = (32, 32, 32)

        # Create Gaussian blobs (realistic signal profile, passes threshold_map)
        vol1 = np.zeros(vol_shape, dtype=np.float32)
        vol1[8, 8, 8] = 100.0
        vol1 = gaussian_filter(vol1, sigma=2.0)

        vol2 = np.zeros(vol_shape, dtype=np.float32)
        vol2[24, 24, 24] = 100.0
        vol2 = gaussian_filter(vol2, sigma=2.0)

        soft_union, binary_union = mask.make_union_gt_mask(
            [vol1, vol2],
            volume_shape=vol_shape,
            smax=3,
            iter=1,
            dilation_iters=1,
            kern_rad=3,
        )

        # Each blob's core should be inside the union
        assert binary_union[8, 8, 8], "Blob 1 core must be in union"
        assert binary_union[24, 24, 24], "Blob 2 core must be in union"

        # Union should be larger than either individual mask
        _, mask1_only = mask.make_union_gt_mask([vol1], vol_shape, smax=3, iter=1, dilation_iters=1, kern_rad=3)
        _, mask2_only = mask.make_union_gt_mask([vol2], vol_shape, smax=3, iter=1, dilation_iters=1, kern_rad=3)
        assert binary_union.sum() >= mask1_only.sum()
        assert binary_union.sum() >= mask2_only.sum()

    def test_accepts_flattened_input(self):
        """Should accept (n_vols, n_voxels) 2-D input."""
        vol_shape = (8, 8, 8)
        n_voxels = 8 * 8 * 8
        rng = np.random.RandomState(0)
        vols_flat = rng.randn(3, n_voxels).astype(np.float32)
        vols_flat[:, : n_voxels // 2] += 5.0  # signal in first half

        soft, binary = mask.make_union_gt_mask(
            vols_flat,
            volume_shape=vol_shape,
            smax=3,
            iter=1,
        )
        assert soft.shape == vol_shape
        assert binary.shape == vol_shape
        assert soft.dtype == np.float32

    def test_soft_mask_range(self):
        """Soft mask values should be in [0, 1]."""
        rng = np.random.RandomState(7)
        vol = rng.randn(12, 12, 12).astype(np.float32)
        vol[3:9, 3:9, 3:9] += 5.0
        soft, _ = mask.make_union_gt_mask([vol], (12, 12, 12))
        assert soft.min() >= 0.0
        assert soft.max() <= 1.0 + 1e-5

    def test_default_dilation_iters(self):
        """Default dilation_iters should be ceil(6 * N / 128)."""
        vol_shape = (128, 128, 128)
        expected = int(np.ceil(6 * 128 / 128))  # = 6
        # Just verify the function runs with defaults
        vol = np.zeros(vol_shape, dtype=np.float32)
        vol[50:78, 50:78, 50:78] = 10.0
        soft, binary = mask.make_union_gt_mask([vol], vol_shape)
        assert soft.shape == vol_shape


class TestReferenceEquivalence:
    def test_make_mask_from_gt_matches_reference_impl(self):
        rng = np.random.RandomState(42)
        vol = rng.randn(16, 16, 16).astype(np.float32)
        got = mask.make_mask_from_gt(vol, smax=3, iter=2, from_ft=False)
        expected = _reference_make_mask_from_gt(vol, smax=3, iter=2, from_ft=False)
        np.testing.assert_allclose(got, expected)

    def test_make_mask_from_half_maps_matches_reference_impl(self):
        rng = np.random.RandomState(42)
        h1 = rng.randn(16, 16, 16).astype(np.float32)
        h2 = h1 + rng.randn(16, 16, 16).astype(np.float32) * 0.1
        got = mask.make_mask_from_half_maps(h1, h2, smax=3)
        expected = _reference_make_mask_from_half_maps(h1, h2, smax=3)
        np.testing.assert_allclose(got, expected)

    def test_make_moving_gt_mask_matches_reference_impl(self):
        rng = np.random.default_rng(0)
        vols = rng.standard_normal((3, 10, 10, 10), dtype=np.float32)
        vols[:, 3:7, 3:7, 3:7] += np.array([0.0, 1.0, 2.0], dtype=np.float32)[:, None, None, None]

        got_soft, got_binary = mask.make_moving_gt_mask(
            vols,
            volume_shape=(10, 10, 10),
            smax=3,
            iter=2,
            dilation_iters=1,
            kern_rad=3,
        )
        exp_soft, exp_binary = _reference_make_moving_gt_mask(
            vols,
            volume_shape=(10, 10, 10),
            smax=3,
            iter=2,
            dilation_iters=1,
            kern_rad=3,
        )
        np.testing.assert_array_equal(got_binary, exp_binary)
        np.testing.assert_allclose(got_soft, exp_soft)
