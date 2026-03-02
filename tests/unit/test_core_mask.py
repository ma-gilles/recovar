"""Unit tests for recovar.core.mask."""

import numpy as np
import pytest

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
