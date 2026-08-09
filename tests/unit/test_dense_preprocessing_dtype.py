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


def test_dense_preprocessing_can_return_unshifted_weighted_score_operand():
    batch = jnp.zeros((2, 64), dtype=jnp.float32)
    ctf_params = jnp.ones((2, 9), dtype=jnp.float64)
    noise = np.ones(40, dtype=np.float64)
    translations = np.zeros((3, 2), dtype=np.float64)

    shifted, _, _, unshifted_weighted = preprocess_batch(
        _Complex64HalfDataset(),
        batch,
        ctf_params,
        noise,
        translations,
        _Float64CtfConfig(),
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        norm_real_dtype=jnp.float64,
        return_unshifted_score_weighted=True,
    )

    assert unshifted_weighted.shape == (2, 40)
    assert unshifted_weighted.dtype == jnp.complex64
    np.testing.assert_array_equal(
        np.asarray(shifted).reshape(2, 3, 40),
        np.repeat(np.asarray(unshifted_weighted)[:, None, :], 3, axis=1),
    )


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


def test_dense_preprocessing_forwards_relion_cuda_operands():
    captured = {}

    class _StrictHalfDataset:
        def process_images_half(self, batch, apply_image_mask=False, **kwargs):
            captured.update(apply_image_mask=apply_image_mask, **kwargs)
            return jnp.ones((batch.shape[0], 40), dtype=jnp.complex64)

    factors = jnp.asarray([0.9, 1.1], dtype=jnp.float32)
    shifts = jnp.asarray([[1, -1], [0, 2]], dtype=jnp.int32)
    preprocess_batch(
        _StrictHalfDataset(),
        jnp.zeros((2, 64), dtype=jnp.float32),
        jnp.ones((2, 9), dtype=jnp.float64),
        np.ones(40, dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        _Float64CtfConfig(),
        score_with_masked_images=True,
        relion_preprocess_kwargs={
            "relion_normalization_factors": factors,
            "relion_integer_shifts": shifts,
        },
    )

    assert captured["apply_image_mask"] is True
    np.testing.assert_array_equal(np.asarray(captured["relion_normalization_factors"]), np.asarray(factors))
    np.testing.assert_array_equal(np.asarray(captured["relion_integer_shifts"]), np.asarray(shifts))
