import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.em_engine import normalized_cc_score_inverse_power
from recovar.em.dense_single_volume.helpers.preprocessing import (
    preprocess_batch,
    preprocess_batch_firstiter_cc,
)


class _Float64CtfConfig:
    image_shape = (8, 8)

    def compute_ctf_half(self, ctf_params):
        return jnp.ones((ctf_params.shape[0], 40), dtype=jnp.float64)


class _Complex64HalfDataset:
    def process_images_half(self, batch, apply_image_mask=False):
        return jnp.ones((batch.shape[0], 40), dtype=jnp.complex64)


def test_dense_preprocessing_tiles_score_dtype_before_translation_expansion():
    batch = jnp.zeros((2, 64), dtype=jnp.float32)
    ctf_params = jnp.ones((2, 9), dtype=jnp.float64)
    noise = np.ones(40, dtype=np.float64)
    translations = np.zeros((3, 2), dtype=np.float64)

    shifted, batch_norm, ctf2_over_noise = preprocess_batch(
        _Complex64HalfDataset(),
        batch,
        ctf_params,
        noise,
        translations,
        _Float64CtfConfig(),
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        norm_real_dtype=jnp.float64,
    )

    assert shifted.shape == (6, 40)
    assert shifted.dtype == jnp.complex64
    assert batch_norm.dtype == jnp.float64
    assert ctf2_over_noise.dtype == jnp.float32


def test_firstiter_cc_preprocessing_tiles_score_dtype_before_translation_expansion():
    batch = jnp.zeros((2, 64), dtype=jnp.float32)
    ctf_params = jnp.ones((2, 9), dtype=jnp.float64)
    noise = np.ones(40, dtype=np.float64)
    translations = np.zeros((3, 2), dtype=np.float64)

    shifted, image_power, ctf2, ctf2_over_noise = preprocess_batch_firstiter_cc(
        _Complex64HalfDataset(),
        batch,
        ctf_params,
        noise,
        translations,
        _Float64CtfConfig(),
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        norm_real_dtype=jnp.float64,
    )

    assert shifted.shape == (6, 40)
    assert shifted.dtype == jnp.complex64
    assert image_power.dtype == jnp.float64
    assert ctf2.dtype == jnp.float32
    assert ctf2_over_noise.dtype == jnp.float32


def test_firstiter_cc_inverse_power_stays_in_score_dtype_before_tile_multiply():
    batch_norm = jnp.ones((2, 1), dtype=jnp.float64)

    inv_xi2 = normalized_cc_score_inverse_power(batch_norm, score_real_dtype=jnp.float32)
    shifted = jnp.ones((6, 40), dtype=jnp.complex64)
    scaled = shifted * jnp.repeat(inv_xi2, 3, axis=0)

    assert inv_xi2.dtype == jnp.float32
    assert scaled.dtype == jnp.complex64
