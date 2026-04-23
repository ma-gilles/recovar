"""P2: softMaskOutsideMap parity — RELION C++ vs recovar Python.

Tests that recovar's relion_soft_image_mask + smooth_circular_mask
produce the same result as RELION's softMaskOutsideMap (mask.cpp:43).

Note: softMaskOutsideMap replaces exterior pixels with the average
background value (avg_bg), while recovar's relion_soft_image_mask
returns a multiplicative mask. The test compares the MASK VALUES
directly, then verifies the full masked-image workflow matches.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import soft_mask_outside_map_2d

from recovar.core.mask import apply_relion_soft_image_mask, relion_soft_image_mask, smooth_circular_mask


@pytest.fixture(params=[32, 64, 128])
def box_size(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestMaskShape:
    """Verify recovar's mask shape matches RELION's raised-cosine."""

    @pytest.mark.parametrize("radius_frac", [0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("cosine_width", [3, 5, 10])
    def test_mask_values_match_relion_formula(self, box_size, radius_frac, cosine_width):
        """Compare recovar smooth_circular_mask against RELION's formula.

        RELION mask.cpp:90-92:
          raisedcos = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width)
          result = (1 - raisedcos) * vol + raisedcos * add
        where add = avg background.

        The mask itself is: 1 - raisedcos = 0.5 + 0.5 * cos(PI*(r-radius)/width)
        which is what smooth_circular_mask computes.
        """
        D = box_size
        radius = radius_frac * D / 2
        mask = smooth_circular_mask(D, radius, cosine_width)

        half = D // 2
        coords = np.arange(-half, D - half, dtype=float)
        gy, gx = np.meshgrid(coords, coords, indexing="ij")
        r = np.sqrt(gx**2 + gy**2)

        expected = np.zeros((D, D))
        expected[r < radius] = 1.0
        band = (r >= radius) & (r <= radius + cosine_width)
        expected[band] = 0.5 + 0.5 * np.cos(np.pi * (r[band] - radius) / cosine_width)

        np.testing.assert_allclose(mask, expected, atol=1e-14)


class TestRelionBinding:
    """Compare RELION's C++ softMaskOutsideMap against recovar."""

    @pytest.mark.parametrize("radius_frac", [0.3, 0.5, 0.7])
    @pytest.mark.parametrize("cosine_width", [3, 5])
    def test_masked_image_matches(self, box_size, radius_frac, cosine_width, rng):
        """Apply both masks to the same image, compare results.

        RELION replaces exterior with avg_bg, while recovar multiplies
        by mask and adds (1-mask)*avg_bg separately. The results should
        match exactly since both use the same formula.
        """
        D = box_size
        radius = radius_frac * D / 2
        image = rng.standard_normal((D, D))

        relion_result = soft_mask_outside_map_2d(image, radius, cosine_width)

        recovar_mask = smooth_circular_mask(D, radius, cosine_width)
        half = D // 2
        coords = np.arange(-half, D - half, dtype=float)
        gy, gx = np.meshgrid(coords, coords, indexing="ij")
        r = np.sqrt(gx**2 + gy**2)
        radius_p = radius + cosine_width

        raisedcos = np.zeros((D, D))
        band = (r >= radius) & (r <= radius_p)
        raisedcos[band] = 0.5 + 0.5 * np.cos(np.pi * (radius_p - r[band]) / cosine_width)
        raisedcos[r > radius_p] = 1.0

        bg_sum = np.sum(raisedcos * image)
        bg_weight = np.sum(raisedcos)
        avg_bg = bg_sum / bg_weight if bg_weight > 0 else 0.0

        recovar_result = recovar_mask * image + (1.0 - recovar_mask) * avg_bg

        np.testing.assert_allclose(
            recovar_result,
            relion_result,
            atol=1e-12,
            err_msg=f"D={D}, radius={radius:.1f}, width={cosine_width}",
        )

    @pytest.mark.parametrize("pixel_size", [1.0, 1.35, 2.0])
    @pytest.mark.parametrize("particle_diameter", [100.0, 200.0])
    def test_relion_soft_image_mask_shape(self, box_size, pixel_size, particle_diameter):
        """Verify relion_soft_image_mask mask VALUES match RELION's formula.

        relion_soft_image_mask returns float32 for production use. We compare
        the float64 smooth_circular_mask (which it wraps) against the RELION
        formula to confirm exact parity of the mask shape itself.
        """
        D = box_size
        width_mask_edge = 5
        radius = min(particle_diameter / (2.0 * pixel_size), D / 2.0)

        recovar_mask_f64 = smooth_circular_mask(D, radius, float(width_mask_edge))
        recovar_mask_f32 = relion_soft_image_mask(D, pixel_size, particle_diameter, width_mask_edge)

        assert recovar_mask_f32.dtype == np.float32
        np.testing.assert_allclose(
            recovar_mask_f32.astype(np.float64),
            recovar_mask_f64,
            atol=1e-7,
            err_msg="float32 truncation exceeds expected precision",
        )

    @pytest.mark.parametrize("pixel_size", [1.0, 1.35, 2.0])
    @pytest.mark.parametrize("particle_diameter", [100.0, 200.0])
    def test_relion_soft_image_mask_matches_binding(self, box_size, pixel_size, particle_diameter, rng):
        """Verify masked image from relion_soft_image_mask matches RELION C++.

        Uses float64 mask (smooth_circular_mask) for the comparison to avoid
        float32 truncation noise. Production uses float32 masks, giving ~1e-7
        precision — acceptable since images are also float32.
        """
        D = box_size
        width_mask_edge = 5
        radius = min(particle_diameter / (2.0 * pixel_size), D / 2.0)
        image = rng.standard_normal((D, D))

        relion_result = soft_mask_outside_map_2d(image, radius, float(width_mask_edge))
        recovar_mask = smooth_circular_mask(D, radius, float(width_mask_edge))

        half = D // 2
        coords = np.arange(-half, D - half, dtype=float)
        gy, gx = np.meshgrid(coords, coords, indexing="ij")
        r = np.sqrt(gx**2 + gy**2)
        radius_p = radius + width_mask_edge

        raisedcos = np.zeros((D, D))
        band = (r >= radius) & (r <= radius_p)
        raisedcos[band] = 0.5 + 0.5 * np.cos(np.pi * (radius_p - r[band]) / width_mask_edge)
        raisedcos[r > radius_p] = 1.0

        bg_sum = np.sum(raisedcos * image)
        bg_weight = np.sum(raisedcos)
        avg_bg = bg_sum / bg_weight if bg_weight > 0 else 0.0

        recovar_result = recovar_mask * image + (1.0 - recovar_mask) * avg_bg

        np.testing.assert_allclose(
            recovar_result,
            relion_result,
            atol=1e-12,
            err_msg=f"D={D}, pixel_size={pixel_size}, particle_diameter={particle_diameter}",
        )

    @pytest.mark.parametrize("pixel_size", [1.0, 1.35, 2.0])
    @pytest.mark.parametrize("particle_diameter", [100.0, 200.0])
    def test_apply_relion_soft_image_mask_matches_binding(self, box_size, pixel_size, particle_diameter, rng):
        D = box_size
        width_mask_edge = 5
        radius = min(particle_diameter / (2.0 * pixel_size), D / 2.0)
        image = rng.standard_normal((D, D)).astype(np.float32)

        relion_result = soft_mask_outside_map_2d(image.astype(np.float64), radius, float(width_mask_edge))
        recovar_mask = relion_soft_image_mask(D, pixel_size, particle_diameter, width_mask_edge)
        recovar_result = apply_relion_soft_image_mask(image, recovar_mask)

        np.testing.assert_allclose(
            recovar_result.astype(np.float64),
            relion_result,
            atol=5e-6,
            err_msg=f"D={D}, pixel_size={pixel_size}, particle_diameter={particle_diameter}",
        )


class TestMaskEdgeCases:
    """Edge cases for mask computation."""

    def test_radius_larger_than_box(self):
        """When particle_diameter exceeds box, radius is clamped to D/2."""
        D = 32
        mask = relion_soft_image_mask(D, 1.0, 1000.0, 5)
        assert mask.shape == (D, D)
        assert mask[D // 2, D // 2] == 1.0

    def test_zero_image_gives_zero_bg(self):
        """softMaskOutsideMap on a zero image should return all zeros."""
        D = 32
        image = np.zeros((D, D))
        result = soft_mask_outside_map_2d(image, 10.0, 5.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_constant_image_preserved(self):
        """softMaskOutsideMap on a constant image returns the same constant."""
        D = 32
        image = np.full((D, D), 7.5)
        result = soft_mask_outside_map_2d(image, 10.0, 5.0)
        np.testing.assert_allclose(result, 7.5, atol=1e-12)
