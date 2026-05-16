"""Nonfinite-score guardrails for exact-local EM's big-JIT normalizer."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.local_big_jit import _score_normalize_mstep
from recovar.em.dense_single_volume.local_backprojection import compute_local_ctf_sums
from recovar.em.dense_single_volume.local_score_pass import fused_score_normalize_mstep_abs2_on_demand
from recovar.em.dense_single_volume.helpers.projection import compute_noise_block

pytestmark = pytest.mark.unit


def _base_inputs():
    batch_size, n_rot, n_trans, n_half = 2, 1, 2, 3
    shifted = jnp.asarray(
        [
            [[0.2 + 0.1j, -0.1 + 0.3j, 0.4 - 0.2j], [0.1 - 0.2j, 0.3 + 0.2j, -0.2 + 0.1j]],
            [[np.inf + 0.0j, np.inf + 0.0j, np.inf + 0.0j], [np.inf + 0.0j, np.inf + 0.0j, np.inf + 0.0j]],
        ],
        dtype=jnp.complex64,
    )
    return dict(
        shifted_score_split=shifted,
        ctf2_over_nv_score=jnp.ones((batch_size, n_half), dtype=jnp.float32),
        proj_weighted=jnp.ones((batch_size, n_rot, n_half), dtype=jnp.complex64),
        half_weights=jnp.ones((n_half,), dtype=jnp.float32),
        rotation_log_prior=jnp.zeros((batch_size, n_rot), dtype=jnp.float32),
        translation_log_prior=jnp.zeros((batch_size, n_trans), dtype=jnp.float32),
        rotation_mask=jnp.ones((batch_size, n_rot), dtype=bool),
        sample_mask=jnp.ones((batch_size, n_rot, n_trans), dtype=bool),
        valid_image_mask=jnp.ones((batch_size,), dtype=bool),
        normalization_log_z=jnp.zeros((batch_size,), dtype=jnp.float32),
        shifted_recon_split=jnp.ones((batch_size, n_trans, n_half), dtype=jnp.complex64),
        ctf2_over_nv_recon=jnp.ones((batch_size, n_half), dtype=jnp.float32),
    )


def test_score_normalize_mstep_zeroes_all_nonfinite_score_rows():
    """A bad image row must not create NaN probabilities or noise weights."""

    inputs = _base_inputs()
    inputs["ctf2_over_nv_recon"] = inputs["ctf2_over_nv_recon"].at[1].set(
        jnp.asarray([jnp.inf, jnp.inf, jnp.inf], dtype=jnp.float32)
    )

    result = _score_normalize_mstep(
        **inputs,
        has_normalization_log_z=False,
        half_spectrum_scoring=True,
        use_float64_normalization=True,
        reconstruct_significant_only=False,
        adaptive_fraction=0.999,
        max_significants=-1,
    )
    (
        log_z,
        best_log_score,
        _best_argmax,
        max_posterior,
        reconstruction_rotation_mask,
        n_significant_samples,
        reconstruction_probs,
        probs_sum_t,
        reconstruction_probs_sum_t,
        summed,
        ctf_probs,
    ) = [np.asarray(x) for x in result]

    assert np.all(np.isfinite(log_z))
    assert np.isneginf(best_log_score[1])
    assert max_posterior[1] == 0.0
    assert n_significant_samples[1] == 0
    assert not reconstruction_rotation_mask[1, 0]
    np.testing.assert_array_equal(reconstruction_probs[1], np.zeros_like(reconstruction_probs[1]))
    np.testing.assert_array_equal(probs_sum_t[1], np.zeros_like(probs_sum_t[1]))
    np.testing.assert_array_equal(reconstruction_probs_sum_t[1], np.zeros_like(reconstruction_probs_sum_t[1]))
    np.testing.assert_array_equal(summed[1], np.zeros_like(summed[1]))
    np.testing.assert_array_equal(ctf_probs[1], np.zeros_like(ctf_probs[1]))


def test_score_normalize_mstep_zeroes_nonfinite_external_logz_rows():
    """A nonfinite provided log-Z must not leak NaNs into M-step tensors."""

    inputs = _base_inputs()
    finite_shifted = inputs["shifted_score_split"].at[1].set(inputs["shifted_score_split"][0])
    inputs["shifted_score_split"] = finite_shifted
    inputs["normalization_log_z"] = jnp.asarray([0.0, jnp.nan], dtype=jnp.float32)

    result = _score_normalize_mstep(
        **inputs,
        has_normalization_log_z=True,
        half_spectrum_scoring=True,
        use_float64_normalization=True,
        reconstruct_significant_only=False,
        adaptive_fraction=0.999,
        max_significants=-1,
    )
    log_z, best_log_score, _best_argmax, max_posterior, _, _, reconstruction_probs, probs_sum_t, _, summed, ctf_probs = [
        np.asarray(x) for x in result
    ]

    assert np.all(np.isfinite(log_z))
    assert np.isneginf(best_log_score[1])
    assert max_posterior[1] == 0.0
    np.testing.assert_array_equal(reconstruction_probs[1], np.zeros_like(reconstruction_probs[1]))
    np.testing.assert_array_equal(probs_sum_t[1], np.zeros_like(probs_sum_t[1]))
    np.testing.assert_array_equal(summed[1], np.zeros_like(summed[1]))
    np.testing.assert_array_equal(ctf_probs[1], np.zeros_like(ctf_probs[1]))


def test_compute_noise_block_zero_weight_nonfinite_projection_is_zero():
    """Zero posterior support must not become NaN via 0 * inf products."""

    proj_half = jnp.asarray(
        [
            [complex(np.inf, 1.0), complex(1.0, np.inf), complex(np.inf, np.inf)],
            [complex(np.inf, 2.0), complex(2.0, np.inf), complex(np.inf, np.inf)],
        ],
        dtype=jnp.complex64,
    )
    proj_abs2_half = jnp.abs(proj_half) ** 2
    summed_masked = jnp.zeros_like(proj_half)
    ctf_probs = jnp.zeros((2, 3), dtype=jnp.float32)
    noise_variance_half = jnp.asarray([1.0, np.inf, np.inf], dtype=jnp.float32)
    shell_indices = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    noise_shells, a2_shells, xa_shells = compute_noise_block(
        proj_half,
        proj_abs2_half,
        summed_masked,
        ctf_probs,
        noise_variance_half,
        shell_indices,
        3,
        True,
    )

    np.testing.assert_array_equal(np.asarray(noise_shells), np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(a2_shells), np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(xa_shells), np.zeros(3, dtype=np.float32))


def test_compute_local_ctf_sums_zero_mass_nonfinite_ctf_is_zero():
    """Zero posterior CTF rows must not become NaN when CTF/noise is nonfinite."""

    probs = jnp.asarray(
        [
            [[0.25, 0.75]],
            [[0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    ctf2_over_nv = jnp.asarray([[1.0, 2.0, 3.0], [jnp.inf, jnp.inf, jnp.inf]], dtype=jnp.float32)

    ctf_sums = np.asarray(compute_local_ctf_sums(probs, ctf2_over_nv))

    assert np.all(np.isfinite(ctf_sums))
    np.testing.assert_array_equal(ctf_sums[1], np.zeros_like(ctf_sums[1]))


def test_fused_local_score_zero_mass_nonfinite_ctf_is_zero():
    """The non-big-JIT fused local path must share the zero-mass CTF guard."""

    inputs = _base_inputs()
    inputs["ctf2_over_nv_recon"] = inputs["ctf2_over_nv_recon"].at[1].set(
        jnp.asarray([jnp.inf, jnp.inf, jnp.inf], dtype=jnp.float32)
    )
    result = fused_score_normalize_mstep_abs2_on_demand(
        inputs["shifted_score_split"],
        inputs["ctf2_over_nv_score"],
        inputs["proj_weighted"],
        inputs["half_weights"],
        inputs["rotation_log_prior"],
        inputs["translation_log_prior"],
        inputs["rotation_mask"],
        inputs["sample_mask"],
        inputs["shifted_recon_split"],
        inputs["ctf2_over_nv_recon"],
        half_spectrum_scoring=True,
        use_float64_normalization=True,
        reconstruct_significant_only=True,
        adaptive_fraction=0.0,
        max_significants=0,
    )
    ctf_probs = np.asarray(result[-1])

    assert np.all(np.isfinite(ctf_probs))
    assert np.any(ctf_probs[0] != 0.0)
    np.testing.assert_array_equal(ctf_probs[1], np.zeros_like(ctf_probs[1]))
