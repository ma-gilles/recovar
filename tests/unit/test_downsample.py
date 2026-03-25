"""Tests for recovar.downsample — 2D Fourier-crop downsampling."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from recovar.data_io.downsample import downsample_images

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_identity_returns_copy():
    """Downsampling D→D returns an identical (but copied) array."""
    images = np.random.randn(5, 32, 32).astype(np.float32)
    result = downsample_images(images, 32)
    assert_allclose(result, images, atol=1e-5)
    assert result is not images  # must be a copy


def test_output_shape_batch():
    """Downsampled batch has correct shape."""
    images = np.random.randn(10, 64, 64).astype(np.float32)
    result = downsample_images(images, 32)
    assert result.shape == (10, 32, 32)


def test_output_shape_single():
    """Downsampling a single 2D image works."""
    image = np.random.randn(64, 64).astype(np.float32)
    result = downsample_images(image, 32)
    assert result.shape == (32, 32)


def test_gaussian_blob():
    """Downsampled Gaussian should match a directly-generated smaller Gaussian."""
    D = 64
    target = 32
    sigma = 8.0

    # Create Gaussian at D=64
    x = np.arange(D) - D / 2
    xx, yy = np.meshgrid(x, x)
    gauss_64 = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float64)

    # Downsample to D=32
    result = downsample_images(gauss_64, target)

    # Create Gaussian directly at D=32
    x32 = np.arange(target) - target / 2
    xx32, yy32 = np.meshgrid(x32, x32)
    gauss_32 = np.exp(-(xx32**2 + yy32**2) / (2 * (sigma / 2) ** 2))

    # Should be roughly similar (Fourier cropping of a Gaussian is a Gaussian)
    # Normalize both for comparison
    result_norm = result / result.max()
    gauss_32_norm = gauss_32 / gauss_32.max()
    assert_allclose(result_norm, gauss_32_norm, atol=0.15)


def test_preserves_dc_component():
    """DC component (mean) should be preserved after downsampling."""
    images = np.random.randn(5, 64, 64).astype(np.float64) + 10.0
    result = downsample_images(images, 32)
    # Mean should be approximately preserved
    for i in range(5):
        assert_allclose(result[i].mean(), images[i].mean(), rtol=0.01)


def test_dtype_preserved():
    """Output dtype matches input dtype."""
    images = np.random.randn(3, 32, 32).astype(np.float32)
    result = downsample_images(images, 16)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def test_batch_matches_individual():
    """Batched downsampling matches per-image downsampling."""
    images = np.random.randn(10, 64, 64).astype(np.float64)

    batch_result = downsample_images(images, 32)

    individual = np.stack([downsample_images(images[i], 32) for i in range(10)])

    assert_allclose(batch_result, individual, atol=1e-10)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_odd_target_raises():
    images = np.random.randn(3, 32, 32)
    with pytest.raises(ValueError, match="even"):
        downsample_images(images, 15)


def test_target_larger_than_source_raises():
    images = np.random.randn(3, 32, 32)
    with pytest.raises(ValueError, match="<="):
        downsample_images(images, 64)


def test_non_square_raises():
    images = np.random.randn(3, 32, 16)
    with pytest.raises(ValueError, match="square"):
        downsample_images(images, 16)


def test_1d_raises():
    images = np.random.randn(32)
    with pytest.raises(ValueError, match="2D or 3D"):
        downsample_images(images, 16)


def test_4d_raises():
    images = np.random.randn(2, 3, 32, 32)
    with pytest.raises(ValueError, match="2D or 3D"):
        downsample_images(images, 16)
