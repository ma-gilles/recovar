import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_mstep_sums,
    compute_local_noise_scalar_terms,
    compute_local_weighted_sums,
    compute_relion_f32_sequential_mstep_sums,
)


def _numpy_relion_f32_loop(probs, shifted, ctf2_over_nv):
    probs = np.asarray(probs, dtype=np.float32)
    shifted = np.asarray(shifted, dtype=np.complex64)
    ctf2_over_nv = np.asarray(ctf2_over_nv, dtype=np.float32)
    numerator = np.zeros((probs.shape[0], probs.shape[1], shifted.shape[-1]), dtype=np.complex64)
    denominator = np.zeros(numerator.shape, dtype=np.float32)
    for trans_idx in range(probs.shape[-1]):
        weight = probs[:, :, trans_idx, None]
        numerator += weight * shifted[:, None, trans_idx, :]
        denominator += weight * ctf2_over_nv[:, None, :]
    return numerator, denominator


def test_relion_f32_sequential_mstep_sums_match_numpy_translation_loop_exactly():
    probs = np.array([[[1.0, 1.0, 1.0], [0.5, 0.25, 0.125]]], dtype=np.float64)
    shifted = np.array(
        [[[1.0e8 + 2.0j, -8.0j], [1.0 + 4.0j, 2.0j], [-1.0e8 - 2.0j, 4.0j]]],
        dtype=np.complex64,
    )
    ctf2_over_nv = np.array([[1.0e8, 8.0]], dtype=np.float64)

    expected_y, expected_ctf = _numpy_relion_f32_loop(probs, shifted, ctf2_over_nv)
    actual_y, actual_ctf = compute_relion_f32_sequential_mstep_sums(probs, shifted, ctf2_over_nv)

    assert actual_y.dtype == np.dtype(np.complex64)
    assert actual_ctf.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(np.asarray(actual_y), expected_y)
    np.testing.assert_array_equal(np.asarray(actual_ctf), expected_ctf)
    # This cancellation pattern distinguishes left-to-right float32 carrying
    # from a higher-precision or reassociated reduction.
    assert np.asarray(actual_y)[0, 0, 0] == np.complex64(0.0 + 4.0j)


def test_local_weighted_sums_requests_highest_dot_precision():
    probs = jnp.ones((1, 2, 3), dtype=jnp.float32)
    shifted = jnp.ones((1, 3, 4), dtype=jnp.complex64)

    jaxpr = str(jax.make_jaxpr(compute_local_weighted_sums)(probs, shifted))

    assert "Precision.HIGHEST" in jaxpr


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU HLO regression")
def test_local_weighted_sums_requests_highest_precision_in_complex64_gpu_hlo():
    probs = jnp.ones((1, 2, 3), dtype=jnp.float32)
    shifted = jnp.ones((1, 3, 4), dtype=jnp.complex64)

    assert compute_local_weighted_sums(probs, shifted).dtype == jnp.complex64
    hlo = compute_local_weighted_sums.lower(probs, shifted).compiler_ir("hlo").as_hlo_text()

    assert "operand_precision={highest,highest}" in hlo


def test_local_weighted_sums_match_explicit_highest_precision_matmul():
    probs = jnp.asarray(
        [[[1.0, 1.0, 1.0], [0.5, 0.25, 0.125]]],
        dtype=jnp.float32,
    )
    shifted = jnp.asarray(
        [[[1.0e8 + 2.0j], [1.0 + 4.0j], [-1.0e8 - 2.0j]]],
        dtype=jnp.complex64,
    )

    actual = compute_local_weighted_sums(probs, shifted)
    expected = jnp.matmul(probs, shifted, precision=jax.lax.Precision.HIGHEST)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_local_noise_scalar_terms_match_dense_relion_order_exactly():
    probs = jnp.asarray(
        [
            [
                [0.50000006, 0.125, 0.0],
                [0.0, 0.06250001, 0.0],
                [0.03125, 0.0, 0.015625],
                [0.0, 0.0, 0.0078125],
                [0.00390625, 0.001953125, 0.0],
            ],
            [
                [0.25, 0.125, 0.0625],
                [0.03125, 0.015625, 0.0078125],
                [0.00390625, 0.001953125, 0.0009765625],
                [0.00048828125, 0.000244140625, 0.0001220703125],
                [0.00006103515625, 0.0, 0.0],
            ],
            [
                [0.875, 0.0625, 0.03125],
                [0.015625, 0.0078125, 0.00390625],
                [0.001953125, 0.0009765625, 0.00048828125],
                [0.000244140625, 0.0001220703125, 0.00006103515625],
                [0.000030517578125, 0.0, 0.0],
            ],
        ],
        dtype=jnp.float32,
    )
    translation_sqdist = jnp.asarray(
        [[1.25, 3.5, 8.75], [0.5, 2.0, 7.0], [4.0, 5.0, 6.0]],
        dtype=jnp.float32,
    )
    valid_image_mask = jnp.asarray([True, True, False])

    support_mass = jnp.sum(probs.reshape(probs.shape[0], -1), axis=1).astype(
        jnp.float32
    )
    support_mass = jnp.where(valid_image_mask, support_mass, 0.0)
    translation_posterior = jnp.sum(probs, axis=1).astype(jnp.float32)
    noise_sumw_offset = jnp.sum(translation_posterior * translation_sqdist)
    retained_mass = jnp.sum(support_mass)

    actual = compute_local_noise_scalar_terms(
        probs,
        translation_sqdist,
        valid_image_mask,
    )

    for actual_value, expected_value in zip(
        actual,
        (support_mass, translation_posterior, noise_sumw_offset, retained_mass),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(actual_value),
            np.asarray(expected_value),
        )
    assert np.asarray(actual[0]).dtype == np.dtype(np.float32)
    assert np.asarray(actual[1]).dtype == np.dtype(np.float32)
    assert np.asarray(actual[0])[-1] == np.float32(0.0)


def test_local_noise_scalar_terms_inline_the_mature_big_jit_primitives():
    probs = jnp.ones((3, 5, 7), dtype=jnp.float32)
    translation_sqdist = jnp.ones((3, 7), dtype=jnp.float32)
    valid_image_mask = jnp.asarray([True, True, False])

    def inline_oracle(probs_arg, translation_sqdist_arg, valid_mask_arg):
        batch_size = probs_arg.shape[0]
        support_mass = jnp.sum(
            probs_arg.reshape(batch_size, -1),
            axis=1,
        ).astype(jnp.float32)
        support_mass = jnp.where(valid_mask_arg, support_mass, 0.0)
        translation_posterior = jnp.sum(probs_arg, axis=1).astype(jnp.float32)
        noise_sumw_offset = jnp.sum(
            translation_posterior
            * jnp.asarray(translation_sqdist_arg, dtype=jnp.float32)
        )
        retained_mass = jnp.sum(support_mass)
        return (
            support_mass,
            translation_posterior,
            noise_sumw_offset,
            retained_mass,
        )

    helper_jaxpr = jax.make_jaxpr(compute_local_noise_scalar_terms)(
        probs,
        translation_sqdist,
        valid_image_mask,
    )
    oracle_jaxpr = jax.make_jaxpr(inline_oracle)(
        probs,
        translation_sqdist,
        valid_image_mask,
    )

    assert str(helper_jaxpr) == str(oracle_jaxpr)
    assert "name=compute_local_noise_scalar_terms" not in str(helper_jaxpr)


def test_local_mstep_sums_env_gate_only_changes_relion_x_half(monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION", "1")
    probs = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float64)
    shifted = np.array([[[1.0e8 + 0j], [1.0 + 0j], [-1.0e8 + 0j]]], dtype=np.complex64)
    ctf2_over_nv = np.array([[2.0]], dtype=np.float64)

    xhalf_y, xhalf_ctf = compute_local_mstep_sums(probs, shifted, ctf2_over_nv, relion_x_half=True)
    normal_y, normal_ctf = compute_local_mstep_sums(probs, shifted, ctf2_over_nv, relion_x_half=False)

    assert xhalf_y.dtype == np.dtype(np.complex64)
    assert xhalf_ctf.dtype == np.dtype(np.float32)
    assert normal_y.dtype == np.dtype(np.complex128)
    assert normal_ctf.dtype == np.dtype(np.float64)
    assert np.asarray(xhalf_y)[0, 0, 0] == 0.0
    assert np.asarray(normal_y)[0, 0, 0] == 1.0
    np.testing.assert_array_equal(np.asarray(normal_ctf), np.array([[[6.0]]], dtype=np.float64))
