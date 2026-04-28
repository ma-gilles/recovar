"""Equivalence tests for the dense/global bucket big-JIT boundary."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.dense_big_jit import run_dense_bucket_big_jit
from recovar.em.dense_single_volume.em_engine import (
    _adjoint_slice_volume_half,
    _compute_projections_block,
    _e_step_block_scores,
    _m_step_block_compute,
    _update_logsumexp,
)

pytestmark = pytest.mark.unit


IMAGE_SHAPE = (4, 4)
VOLUME_SHAPE = (4, 4, 4)
N_IMAGES = 2
N_TRANS = 2
N_ROT = 3
N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
VOLUME_SIZE = np.prod(VOLUME_SHAPE)


def _rot_z(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _complex_grid(rows, cols, *, scale, offset=0.0):
    values = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    real = offset + scale * values
    imag = 0.5 * scale * (values + 1.0)
    return jnp.asarray(real + 1j * imag, dtype=jnp.complex64)


def _real_grid(rows, cols, *, scale, offset=1.0):
    values = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    return jnp.asarray(offset + scale * values, dtype=jnp.float32)


def _inputs():
    rng = np.random.default_rng(7)
    real_volume = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
    mean_for_proj = jnp.asarray(np.fft.fftshift(np.fft.fftn(real_volume)).ravel(), dtype=jnp.complex64)
    rotations = jnp.asarray(
        np.stack(
            [
                np.eye(3, dtype=np.float32),
                _rot_z(np.pi / 2.0),
                _rot_z(np.pi),
            ],
            axis=0,
        ),
        dtype=jnp.float32,
    )

    shifted_score_half = _complex_grid(N_IMAGES * N_TRANS, N_HALF, scale=0.015, offset=0.2)
    shifted_recon_half = _complex_grid(N_IMAGES * N_TRANS, N_HALF, scale=0.011, offset=-0.1)
    score_weight_half = _real_grid(N_IMAGES, N_HALF, scale=0.02, offset=0.7)
    ctf2_over_nv_recon_half = _real_grid(N_IMAGES, N_HALF, scale=0.013, offset=0.9)

    return {
        "shifted_score_half": shifted_score_half,
        "batch_norm": jnp.zeros((N_IMAGES, 1), dtype=jnp.float32),
        "score_weight_half": score_weight_half,
        "shifted_recon_half": shifted_recon_half,
        "ctf2_over_nv_recon_half": ctf2_over_nv_recon_half,
        "mean_for_proj": mean_for_proj,
        "rotations": rotations,
        "half_weights": _real_grid(1, N_HALF, scale=0.05, offset=0.6).reshape(-1),
        "rotation_log_prior": jnp.asarray(
            [[0.0, -0.07, -0.2], [-0.03, 0.04, -0.5]],
            dtype=jnp.float32,
        ),
        "translation_log_prior": jnp.asarray(
            [[0.02, -0.04], [-0.01, 0.03]],
            dtype=jnp.float32,
        ),
        "candidate_mask": jnp.asarray(
            [[True, True], [False, True], [True, True]],
            dtype=bool,
        ),
        "valid_rotation_mask": jnp.asarray([True, True, False], dtype=bool),
    }


def _reference_scores(s):
    proj_half, proj_abs2_half = _compute_projections_block(
        s["mean_for_proj"],
        s["rotations"],
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
    )
    scores = _e_step_block_scores(
        s["shifted_score_half"],
        s["batch_norm"],
        s["score_weight_half"],
        proj_half.astype(jnp.complex64) * s["half_weights"],
        proj_abs2_half.astype(jnp.float32) * s["half_weights"],
        s["half_weights"],
        N_IMAGES,
        N_TRANS,
        IMAGE_SHAPE,
        VOLUME_SHAPE,
    )
    scores = scores + s["rotation_log_prior"][:, :, None]
    scores = scores + s["translation_log_prior"][:, None, :]
    scores = jnp.where(s["candidate_mask"][None, :, :], scores, -jnp.inf)
    scores = jnp.where(s["valid_rotation_mask"][None, :, None], scores, -jnp.inf)
    return scores


def _run_big_jit(s, *, run_mstep, log_z=None):
    return run_dense_bucket_big_jit(
        s["shifted_score_half"],
        s["batch_norm"],
        s["score_weight_half"],
        s["shifted_recon_half"],
        s["ctf2_over_nv_recon_half"],
        s["mean_for_proj"],
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        s["rotations"],
        s["half_weights"],
        s["rotation_log_prior"],
        s["translation_log_prior"],
        s["candidate_mask"],
        s["valid_rotation_mask"],
        jnp.zeros(N_IMAGES, dtype=jnp.float32) if log_z is None else log_z,
        jnp.zeros(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        s["shifted_recon_half"],
        jnp.ones(N_HALF, dtype=jnp.float32),
        jnp.zeros(N_HALF, dtype=jnp.int32),
        jnp.zeros((N_IMAGES, N_TRANS), dtype=jnp.float32),
        jnp.arange(N_HALF, dtype=jnp.int32),
        jnp.arange(N_HALF, dtype=jnp.int32),
        score_mode="gaussian",
        zero_dc_for_scoring=False,
        use_window=False,
        use_float64_scoring=False,
        use_float64_normalization=True,
        run_mstep=run_mstep,
        accumulate_noise=False,
        return_noise_split=False,
        has_translation_sqdist=False,
        image_shape=IMAGE_SHAPE,
        proj_volume_shape=VOLUME_SHAPE,
        recon_volume_shape=VOLUME_SHAPE,
        disc_type="linear_interp",
        projection_half_volume=False,
        projection_max_r="auto",
        mstep_half_volume=False,
        backprojection_max_r="auto",
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        n_shells=1,
    )


def test_dense_big_jit_pass1_matches_dense_primitives():
    s = _inputs()
    scores = _reference_scores(s)
    ref_max, ref_sum_exp = _update_logsumexp(
        jnp.full((N_IMAGES,), -jnp.inf),
        jnp.zeros((N_IMAGES,), dtype=jnp.float64),
        scores,
    )

    result = _run_big_jit(s, run_mstep=False)

    np.testing.assert_allclose(np.asarray(result.block_max), np.asarray(ref_max), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(result.block_sum_exp),
        np.asarray(ref_sum_exp),
        rtol=1e-6,
        atol=1e-6,
    )


def test_dense_big_jit_mstep_matches_dense_primitives_and_adjoint():
    s = _inputs()
    scores = _reference_scores(s)
    max_s, sum_exp = _update_logsumexp(
        jnp.full((N_IMAGES,), -jnp.inf),
        jnp.zeros((N_IMAGES,), dtype=jnp.float64),
        scores,
    )
    log_z = max_s + jnp.log(sum_exp)

    result = _run_big_jit(s, run_mstep=True, log_z=log_z)

    ref_Ft_y0 = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)
    ref_Ft_ctf0 = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)
    _, _, probs, ref_best, ref_argmax, summed_half, ctf_probs_half = _m_step_block_compute(
        s["shifted_recon_half"],
        scores,
        log_z,
        s["rotations"],
        s["ctf2_over_nv_recon_half"],
        ref_Ft_y0,
        ref_Ft_ctf0,
        N_IMAGES,
        N_TRANS,
        IMAGE_SHAPE,
        VOLUME_SHAPE,
    )
    ref_Ft_y = _adjoint_slice_volume_half(
        summed_half,
        s["rotations"],
        ref_Ft_y0,
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        True,
    )
    ref_Ft_ctf = _adjoint_slice_volume_half(
        ctf_probs_half,
        s["rotations"],
        ref_Ft_ctf0,
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        True,
    )

    np.testing.assert_allclose(np.asarray(result.block_best), np.asarray(ref_best), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(np.asarray(result.block_argmax), np.asarray(ref_argmax))
    np.testing.assert_allclose(
        np.asarray(result.probs_sum_t),
        np.asarray(jnp.sum(probs, axis=-1)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(result.Ft_y), np.asarray(ref_Ft_y), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.Ft_ctf), np.asarray(ref_Ft_ctf), rtol=1e-6, atol=1e-6)
