import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.padding as padding

pytestmark = pytest.mark.unit


def test_pad_unpad_images_spatial_roundtrip():
    images = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    padded = np.asarray(padding.pad_images_spatial_domain(images, padding=2))
    assert padded.shape == (2, 6, 6)
    unpadded = np.asarray(padding.unpad_images_spatial_domain(padded, padding=2))
    np.testing.assert_allclose(unpadded, images)


def test_pad_unpad_volume_spatial_roundtrip():
    vol = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    padded = np.asarray(padding.pad_volume_spatial_domain(vol, padding=2))
    assert padded.shape == (6, 6, 6)
    unpadded = np.asarray(padding.unpad_volume_spatial_domain(padded, padding=2))
    np.testing.assert_allclose(unpadded, vol)


def test_pad_unpad_images_fourier_roundtrip():
    rng = np.random.default_rng(0)
    images = rng.normal(size=(3, 4, 4)).astype(np.float32)
    ft_padded = np.asarray(padding.pad_images_fourier_domain(images, image_shape=(4, 4), padding=2))
    assert ft_padded.shape == (3, 36)
    ft_unpadded = np.asarray(padding.unpad_images_fourier_domain(ft_padded, padded_image_shape=(6, 6), padding=2))
    assert ft_unpadded.shape == (3, 16)


def test_unpad_volume_fourier_domain_shape():
    rng = np.random.default_rng(1)
    vol = rng.normal(size=(6, 6, 6)).astype(np.float32)
    out = np.asarray(padding.unpad_volume_fourier_domain(vol.reshape(-1), padded_image_shape=(6, 6, 6), padding=2))
    assert out.shape == (4 * 4 * 4,)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_pad_unpad_images_spatial_gpu(gpu_device):
    images = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)

    cpu_padded = np.asarray(padding.pad_images_spatial_domain(images, padding=2))
    cpu_unpadded = np.asarray(padding.unpad_images_spatial_domain(cpu_padded, padding=2))

    with jax.default_device(gpu_device):
        images_g = jax.device_put(jnp.array(images), gpu_device)
        gpu_padded = np.asarray(padding.pad_images_spatial_domain(images_g, padding=2))
        gpu_unpadded = np.asarray(padding.unpad_images_spatial_domain(gpu_padded, padding=2))

    np.testing.assert_allclose(cpu_padded, gpu_padded, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_unpadded, gpu_unpadded, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_pad_unpad_volume_spatial_gpu(gpu_device):
    vol = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)

    cpu_padded = np.asarray(padding.pad_volume_spatial_domain(vol, padding=2))
    cpu_unpadded = np.asarray(padding.unpad_volume_spatial_domain(cpu_padded, padding=2))

    with jax.default_device(gpu_device):
        vol_g = jax.device_put(jnp.array(vol), gpu_device)
        gpu_padded = np.asarray(padding.pad_volume_spatial_domain(vol_g, padding=2))
        gpu_unpadded = np.asarray(padding.unpad_volume_spatial_domain(gpu_padded, padding=2))

    np.testing.assert_allclose(cpu_padded, gpu_padded, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_unpadded, gpu_unpadded, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_pad_images_fourier_domain_gpu(gpu_device):
    rng = np.random.default_rng(0)
    images = rng.normal(size=(3, 4, 4)).astype(np.float32)

    cpu_ft_padded = np.asarray(padding.pad_images_fourier_domain(images, image_shape=(4, 4), padding=2))

    with jax.default_device(gpu_device):
        images_g = jax.device_put(jnp.array(images), gpu_device)
        gpu_ft_padded = np.asarray(padding.pad_images_fourier_domain(images_g, image_shape=(4, 4), padding=2))

    np.testing.assert_allclose(cpu_ft_padded, gpu_ft_padded, atol=1e-4, rtol=1e-4)
