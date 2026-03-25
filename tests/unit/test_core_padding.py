"""Unit tests for recovar.core.padding."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.padding as padding

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# pad / unpad images in spatial domain
# ---------------------------------------------------------------------------


class TestPadUnpadImagesSpatial:
    def test_roundtrip(self):
        rng = np.random.RandomState(42)
        imgs = jnp.array(rng.randn(3, 8, 8).astype(np.float32))
        pad = 4
        padded = padding.pad_images_spatial_domain(imgs, pad)
        assert padded.shape == (3, 12, 12)
        unpadded = padding.unpad_images_spatial_domain(padded, pad)
        np.testing.assert_allclose(unpadded, imgs, atol=1e-6)

    def test_padded_shape(self):
        imgs = jnp.zeros((2, 6, 6))
        padded = padding.pad_images_spatial_domain(imgs, 2)
        assert padded.shape == (2, 8, 8)

    def test_zero_padding(self):
        imgs = jnp.ones((1, 4, 4))
        padded = padding.pad_images_spatial_domain(imgs, 0)
        np.testing.assert_allclose(padded, imgs, atol=1e-7)

    def test_border_is_zero(self):
        imgs = jnp.ones((1, 4, 4))
        padded = padding.pad_images_spatial_domain(imgs, 4)
        # Top rows should be zero
        np.testing.assert_allclose(padded[0, :2, :], 0.0)
        # Bottom rows should be zero
        np.testing.assert_allclose(padded[0, -2:, :], 0.0)
        # Left cols
        np.testing.assert_allclose(padded[0, :, :2], 0.0)
        # Right cols
        np.testing.assert_allclose(padded[0, :, -2:], 0.0)
        # Center should be ones
        np.testing.assert_allclose(padded[0, 2:6, 2:6], 1.0)


# ---------------------------------------------------------------------------
# pad / unpad volume in spatial domain
# ---------------------------------------------------------------------------


class TestPadUnpadVolumeSpatial:
    def test_roundtrip(self):
        rng = np.random.RandomState(42)
        vol = jnp.array(rng.randn(8, 8, 8).astype(np.float32))
        pad = 4
        padded = padding.pad_volume_spatial_domain(vol, pad)
        assert padded.shape == (12, 12, 12)
        unpadded = padding.unpad_volume_spatial_domain(padded, pad)
        np.testing.assert_allclose(unpadded, vol, atol=1e-6)

    def test_padded_shape(self):
        vol = jnp.zeros((6, 6, 6))
        padded = padding.pad_volume_spatial_domain(vol, 2)
        assert padded.shape == (8, 8, 8)

    def test_zero_padding_is_identity(self):
        vol = jnp.ones((4, 4, 4))
        padded = padding.pad_volume_spatial_domain(vol, 0)
        np.testing.assert_allclose(padded, vol, atol=1e-7)

    def test_border_is_zero(self):
        vol = jnp.ones((4, 4, 4))
        padded = padding.pad_volume_spatial_domain(vol, 2)
        # First slice along dim 0 should be zero
        np.testing.assert_allclose(padded[0, :, :], 0.0)
        np.testing.assert_allclose(padded[-1, :, :], 0.0)
        # Center should be ones
        np.testing.assert_allclose(padded[1:5, 1:5, 1:5], 1.0)


# ---------------------------------------------------------------------------
# pad / unpad images in Fourier domain
# ---------------------------------------------------------------------------


class TestPadUnpadImagesFourier:
    def test_roundtrip(self):
        rng = np.random.RandomState(42)
        imgs_spatial = rng.randn(2, 8, 8).astype(np.float32)
        # Create Fourier-domain images
        imgs_ft = jnp.array(np.fft.fft2(imgs_spatial, axes=(-2, -1)).reshape(2, -1))
        image_shape = (8, 8)
        pad = 4

        padded = padding.pad_images_fourier_domain(imgs_ft, image_shape, pad)
        assert padded.shape == (2, 12 * 12)

        padded_shape = (12, 12)
        unpadded = padding.unpad_images_fourier_domain(padded, padded_shape, pad)
        np.testing.assert_allclose(np.abs(unpadded), np.abs(imgs_ft), atol=1e-4)


# ---------------------------------------------------------------------------
# pad / unpad volume in Fourier domain
# ---------------------------------------------------------------------------


class TestPadUnpadVolumeFourier:
    def test_roundtrip_spatial(self):
        """Pad then unpad via Fourier domain recovers original spatial volume."""
        rng = np.random.RandomState(42)
        vol_spatial = rng.randn(6, 6, 6).astype(np.float32)
        pad = 2

        # Pad in spatial domain, convert to Fourier, then unpad in Fourier domain
        padded = padding.pad_volume_spatial_domain(jnp.array(vol_spatial), pad)
        assert padded.shape == (8, 8, 8)

        # unpad_volume_fourier_domain expects Fourier input from the library's DFT
        import recovar.core.fourier_transform_utils as ftu

        padded_ft = ftu.get_dft3(padded).ravel()
        unpadded_ft = padding.unpad_volume_fourier_domain(padded_ft, (8, 8, 8), pad)
        # Convert back to spatial to compare
        unpadded_spatial = np.array(ftu.get_idft3(unpadded_ft.reshape(6, 6, 6)).real)
        np.testing.assert_allclose(unpadded_spatial, vol_spatial, atol=1e-4)


# ---------------------------------------------------------------------------
# padded_dft
# ---------------------------------------------------------------------------


class TestPaddedDft:
    def test_output_shape(self):
        imgs = jnp.zeros((2, 8, 8))
        result = padding.padded_dft(imgs, 64, 4)
        assert result.shape == (2, 12 * 12)

    def test_zero_image_gives_zero_spectrum(self):
        imgs = jnp.zeros((1, 6, 6))
        result = padding.padded_dft(imgs, 36, 2)
        np.testing.assert_allclose(jnp.abs(result), 0.0, atol=1e-7)

    def test_nonzero_image(self):
        imgs = jnp.ones((1, 4, 4))
        result = padding.padded_dft(imgs, 16, 2)
        # The DC component should capture the sum of the original image
        assert jnp.abs(result).max() > 0
