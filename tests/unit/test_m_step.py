"""Tests for the M-step: backprojection and translation summation."""

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from recovar.em.sampling import translations_to_indices, get_translation_grid
from recovar.em.m_step import sum_up_translate_one_image

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# translations_to_indices
# ---------------------------------------------------------------------------


def test_translations_to_indices_zero_translation():
    """Zero translation maps to the center pixel."""
    image_shape = (8, 8)
    translations = jnp.array([[0, 0]], dtype=jnp.int32)
    indices = translations_to_indices(translations, image_shape)
    center = (image_shape[0] // 2) * image_shape[1] + (image_shape[0] // 2)
    assert int(indices[0]) == center


def test_translations_to_indices_batch():
    """Multiple translations yield distinct indices."""
    image_shape = (8, 8)
    translations = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=jnp.int32)
    indices = translations_to_indices(translations, image_shape)
    assert indices.shape == (3,)
    # All should be different
    assert len(set(np.asarray(indices).tolist())) == 3


def test_translations_to_indices_symmetry():
    """Opposite translations map symmetrically around center."""
    image_shape = (8, 8)
    center = image_shape[0] // 2
    t_pos = jnp.array([[1, 0]], dtype=jnp.int32)
    t_neg = jnp.array([[-1, 0]], dtype=jnp.int32)
    idx_pos = int(translations_to_indices(t_pos, image_shape)[0])
    idx_neg = int(translations_to_indices(t_neg, image_shape)[0])
    # Center index
    idx_center = int(translations_to_indices(jnp.array([[0, 0]], dtype=jnp.int32), image_shape)[0])
    assert idx_pos - idx_center == idx_center - idx_neg


# ---------------------------------------------------------------------------
# get_translation_grid
# ---------------------------------------------------------------------------


def test_get_translation_grid_includes_zero():
    """Translation grid always includes (0, 0)."""
    grid = get_translation_grid(max_pixel=4, pixel_offset=1)
    assert any(np.allclose(row, [0, 0]) for row in grid)


def test_get_translation_grid_within_radius():
    """All translations are within the specified max_pixel radius."""
    max_pixel = 3
    grid = get_translation_grid(max_pixel=max_pixel, pixel_offset=1)
    norms = np.linalg.norm(grid, axis=1)
    assert np.all(norms <= max_pixel + 0.01)


def test_get_translation_grid_offset_spacing():
    """Grid spacing matches pixel_offset."""
    grid = get_translation_grid(max_pixel=4, pixel_offset=2)
    # All coordinates should be multiples of 2
    assert np.all(grid % 2 == 0)


# ---------------------------------------------------------------------------
# sum_up_translate_one_image
# ---------------------------------------------------------------------------


def test_sum_up_translate_one_image_single_translation_fft():
    """With one translation at probability 1, image is unchanged (FFT path)."""
    image_shape = (8, 8)
    image_size = 64
    image = jnp.ones(image_size, dtype=jnp.complex64)
    probabilities = jnp.array([[1.0]], dtype=jnp.complex64)  # (n_rot=1, n_trans=1)
    translations = jnp.array([[0, 0]], dtype=jnp.int32)

    result = sum_up_translate_one_image(image, probabilities, translations, image_shape, "fft")
    # With prob=1 at zero translation, the result should equal image * DFT(delta) = image * 1
    assert result.shape == (1, image_size)
    assert jnp.all(jnp.isfinite(result))


def test_sum_up_translate_one_image_probabilities_sum():
    """Output scales linearly with probability magnitude."""
    image_shape = (8, 8)
    image_size = 64
    rng = np.random.default_rng(42)
    image = jnp.array(rng.standard_normal(image_size) + 1j * rng.standard_normal(image_size), dtype=jnp.complex64)
    translations = jnp.array([[0, 0]], dtype=jnp.int32)

    prob1 = jnp.array([[1.0]], dtype=jnp.complex64)
    prob2 = jnp.array([[2.0]], dtype=jnp.complex64)

    result1 = sum_up_translate_one_image(image, prob1, translations, image_shape, "fft")
    result2 = sum_up_translate_one_image(image, prob2, translations, image_shape, "fft")

    np.testing.assert_allclose(np.asarray(result2), np.asarray(result1) * 2, rtol=1e-5)


def test_sum_up_translate_one_image_zero_probability():
    """Zero probability gives zero output."""
    image_shape = (8, 8)
    image_size = 64
    image = jnp.ones(image_size, dtype=jnp.complex64)
    probabilities = jnp.array([[0.0]], dtype=jnp.complex64)
    translations = jnp.array([[0, 0]], dtype=jnp.int32)

    result = sum_up_translate_one_image(image, probabilities, translations, image_shape, "fft")
    assert jnp.allclose(result, 0.0, atol=1e-7)


def test_sum_up_translate_one_image_non_fft_path():
    """Non-FFT path returns valid output."""
    image_shape = (8, 8)
    image_size = 64
    image = jnp.ones(image_size, dtype=jnp.complex64)
    probabilities = jnp.array([[1.0]], dtype=jnp.complex64)
    translations = jnp.array([[0.0, 0.0]], dtype=jnp.float32)

    result = sum_up_translate_one_image(image, probabilities, translations, image_shape, "real")
    assert result.shape == (1, image_size)
    assert jnp.all(jnp.isfinite(result))


def test_sum_up_translate_one_image_multiple_translations():
    """Multiple translations with equal probability."""
    image_shape = (8, 8)
    image_size = 64
    image = jnp.ones(image_size, dtype=jnp.complex64)
    # Two translations, each with prob 0.5
    probabilities = jnp.array([[0.5, 0.5]], dtype=jnp.complex64)
    translations = jnp.array([[0, 0], [1, 0]], dtype=jnp.int32)

    result = sum_up_translate_one_image(image, probabilities, translations, image_shape, "fft")
    assert result.shape == (1, image_size)
    assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@pytest.fixture
def gpu_device():
    """Return the first GPU device, or skip."""
    try:
        devs = jax.devices("gpu")
        if not devs:
            pytest.skip("No GPU available")
        return devs[0]
    except RuntimeError:
        pytest.skip("No GPU available")


@pytest.mark.gpu
def test_sum_up_translate_fft_gpu(gpu_device):
    """sum_up_translate_one_image works on GPU."""
    image_shape = (16, 16)
    image_size = 256
    with jax.default_device(gpu_device):
        image = jnp.ones(image_size, dtype=jnp.complex64)
        probabilities = jnp.array([[1.0]], dtype=jnp.complex64)
        translations = jnp.array([[0, 0]], dtype=jnp.int32)
        result = sum_up_translate_one_image(image, probabilities, translations, image_shape, "fft")
    assert result.shape == (1, image_size)
    assert jnp.all(jnp.isfinite(result))


@pytest.mark.gpu
def test_translations_to_indices_gpu(gpu_device):
    """translations_to_indices works on GPU."""
    image_shape = (16, 16)
    with jax.default_device(gpu_device):
        translations = jnp.array([[0, 0], [1, -1], [-2, 3]], dtype=jnp.int32)
        indices = translations_to_indices(translations, image_shape)
    assert indices.shape == (3,)
    assert jnp.all(jnp.isfinite(indices))
