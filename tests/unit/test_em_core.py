"""
Unit tests for recovar.em.core.

Covers:
  crosscorr_from_ft  – output shape and self-correlation peak at (0,0)
  norm_squared_residuals_from_ft_one_image  – shape passthrough with 1-D and 2-D leading dims
  norm_squared_residuals_from_ft  – vmapped batch shapes
  IMAGE_AXIS / VOL_AXIS / ROT_AXIS / TRANS_AXIS constants
"""
import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp
from recovar.em import core as em_core

pytestmark = pytest.mark.unit

# Tiny image shape used across all tests
IMAGE_SHAPE = (4, 4)
IMAGE_SIZE = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]  # 16


def _ft_images(n: int, seed: int = 0) -> jnp.ndarray:
    """Return complex64 FT images with shape (n, IMAGE_SIZE)."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n, IMAGE_SIZE)).astype(np.float32)
    imag = rng.standard_normal((n, IMAGE_SIZE)).astype(np.float32)
    return jnp.array(real + 1j * imag)


# ---------------------------------------------------------------------------
# axis constants
# ---------------------------------------------------------------------------

def test_axis_constants_are_distinct_integers():
    axes = {em_core.IMAGE_AXIS, em_core.VOL_AXIS, em_core.ROT_AXIS, em_core.TRANS_AXIS}
    assert len(axes) == 4
    for ax in axes:
        assert isinstance(ax, int)


# ---------------------------------------------------------------------------
# crosscorr_from_ft
# ---------------------------------------------------------------------------

def test_crosscorr_from_ft_output_shape():
    """crosscorr_from_ft must return shape (n_imgs, H, W)."""
    n = 3
    many = _ft_images(n)
    one = _ft_images(1)[0]
    result = em_core.crosscorr_from_ft(many, one, IMAGE_SHAPE)
    assert result.shape == (n, *IMAGE_SHAPE)


def test_crosscorr_from_ft_single_image():
    """Works with n=1 images."""
    many = _ft_images(1)
    one = _ft_images(1)[0]
    result = em_core.crosscorr_from_ft(many, one, IMAGE_SHAPE)
    assert result.shape == (1, *IMAGE_SHAPE)


def test_crosscorr_from_ft_self_correlation_peaks_at_center():
    """Self cross-correlation must peak at the zero-lag position.

    get_idft2 applies ifftshift so zero-lag sits at (H//2, W//2) = (2, 2)
    for a 4x4 image, not at flat index 0.
    """
    img = _ft_images(1)           # shape (1, 16)
    one = img[0]                  # shape (16,)
    result = em_core.crosscorr_from_ft(img, one, IMAGE_SHAPE)  # (1, 4, 4)
    corr = np.asarray(result)[0].real  # (4, 4)
    peak_flat = int(np.argmax(np.abs(corr.ravel())))
    peak_hw = np.unravel_index(peak_flat, IMAGE_SHAPE)
    expected = (IMAGE_SHAPE[0] // 2, IMAGE_SHAPE[1] // 2)
    assert peak_hw == expected


# ---------------------------------------------------------------------------
# norm_squared_residuals_from_ft_one_image
# ---------------------------------------------------------------------------

def test_norm_squared_residuals_shape_2d_input():
    """With (n_poses, image_size) input, output shape matches input."""
    n_poses = 5
    many = _ft_images(n_poses)    # (5, 16)
    one = _ft_images(1)[0]        # (16,)
    result = em_core.norm_squared_residuals_from_ft_one_image(many, one, IMAGE_SHAPE)
    assert result.shape == many.shape


def test_norm_squared_residuals_shape_3d_input():
    """With (A, B, image_size) input the shape is preserved end-to-end."""
    many = _ft_images(6).reshape(2, 3, IMAGE_SIZE)
    one = _ft_images(1)[0]
    result = em_core.norm_squared_residuals_from_ft_one_image(many, one, IMAGE_SHAPE)
    assert result.shape == (2, 3, IMAGE_SIZE)


# ---------------------------------------------------------------------------
# norm_squared_residuals_from_ft  (vmapped over first axis)
# ---------------------------------------------------------------------------

def test_norm_squared_residuals_from_ft_batch_shape():
    """vmapped version maps independently over images: (n_images, n_poses, image_size)."""
    n_images = 4
    n_poses = 3
    many = jnp.stack([_ft_images(n_poses, seed=i) for i in range(n_images)])  # (4, 3, 16)
    ones = _ft_images(n_images, seed=99)                                        # (4, 16)
    result = em_core.norm_squared_residuals_from_ft(many, ones, IMAGE_SHAPE)
    assert result.shape == (n_images, n_poses, IMAGE_SIZE)


def test_norm_squared_residuals_from_ft_result_is_finite():
    """All values in the vmapped output must be finite (no NaN/Inf)."""
    n_images = 2
    n_poses = 2
    many = jnp.stack([_ft_images(n_poses, seed=i) for i in range(n_images)])
    ones = _ft_images(n_images, seed=5)
    result = em_core.norm_squared_residuals_from_ft(many, ones, IMAGE_SHAPE)
    assert np.all(np.isfinite(np.asarray(result)))
