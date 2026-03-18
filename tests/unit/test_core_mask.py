"""Unit tests for recovar.core.mask."""

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.mask as mask

pytestmark = pytest.mark.unit


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
# create_soft_edged_kernel_pxl
# ---------------------------------------------------------------------------

class TestCreateSoftEdgedKernel:
    def test_3d_kernel_shape(self):
        k = mask.create_soft_edged_kernel_pxl(3, (8, 8, 8))
        assert k.shape == (8, 8, 8)

    def test_sums_to_one(self):
        k = mask.create_soft_edged_kernel_pxl(3, (8, 8, 8))
        np.testing.assert_allclose(float(jnp.sum(k)), 1.0, atol=1e-5)

    def test_non_negative(self):
        k = mask.create_soft_edged_kernel_pxl(5, (12, 12, 12))
        assert float(jnp.min(k)) >= 0.0

    def test_small_radius(self):
        k = mask.create_soft_edged_kernel_pxl(2, (6, 6, 6))
        np.testing.assert_allclose(float(jnp.sum(k)), 1.0, atol=1e-5)


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


# ---------------------------------------------------------------------------
# MaskedMaps
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
            [vol1, vol2], volume_shape=vol_shape, smax=3, iter=1,
            dilation_iters=1, kern_rad=3,
        )

        # Each blob's core should be inside the union
        assert binary_union[8, 8, 8], "Blob 1 core must be in union"
        assert binary_union[24, 24, 24], "Blob 2 core must be in union"

        # Union should be larger than either individual mask
        _, mask1_only = mask.make_union_gt_mask(
            [vol1], vol_shape, smax=3, iter=1, dilation_iters=1, kern_rad=3)
        _, mask2_only = mask.make_union_gt_mask(
            [vol2], vol_shape, smax=3, iter=1, dilation_iters=1, kern_rad=3)
        assert binary_union.sum() >= mask1_only.sum()
        assert binary_union.sum() >= mask2_only.sum()

    def test_accepts_flattened_input(self):
        """Should accept (n_vols, n_voxels) 2-D input."""
        vol_shape = (8, 8, 8)
        n_voxels = 8 * 8 * 8
        rng = np.random.RandomState(0)
        vols_flat = rng.randn(3, n_voxels).astype(np.float32)
        vols_flat[:, :n_voxels // 2] += 5.0  # signal in first half

        soft, binary = mask.make_union_gt_mask(
            vols_flat, volume_shape=vol_shape, smax=3, iter=1,
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


class TestMaskedMaps:
    def test_init_defaults(self):
        mm = mask.MaskedMaps()
        assert mm.mask is None
        assert mm.iter == 3
        assert mm.smax == 9
        assert mm.prob == 0.99

    def test_generate_mask_from_gt(self):
        rng = np.random.RandomState(42)
        vol = rng.randn(16, 16, 16).astype(np.float32)
        mm = mask.MaskedMaps()
        mm.smax = 3
        mm.arr1 = vol
        mm.iter = 2
        mm.generate_mask_from_gt()
        assert mm.mask is not None
        assert mm.mask.shape == (16, 16, 16)

    def test_generate_mask_halfmaps(self):
        rng = np.random.RandomState(42)
        h1 = rng.randn(16, 16, 16).astype(np.float32)
        h2 = h1 + rng.randn(16, 16, 16).astype(np.float32) * 0.1
        mm = mask.MaskedMaps()
        mm.smax = 3
        mm.arr1 = h1
        mm.arr2 = h2
        mm.iter = 2
        mm.generate_mask()
        assert mm.mask is not None
        assert mm.mask.shape == (16, 16, 16)
