"""Equivalence tests for the dense/global bucket big-JIT boundary."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.dense_big_jit import run_dense_bucket_big_jit
from recovar.em.dense_single_volume.em_engine import (
    _dense_big_jit_disabled_reason,
    _pad_dense_big_jit_image_axis,
)
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_half as _adjoint_slice_volume_half,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_projections_block as _compute_projections_block,
)
from recovar.em.dense_single_volume.helpers.scoring import (
    _e_step_block_scores,
    _e_step_block_scores_normalized_cc,
    _e_step_block_scores_windowed,
    _e_step_block_scores_windowed_normalized_cc,
    _m_step_block_compute,
    _update_logsumexp,
)
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_indices_np

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


def _window_indices(current_size):
    score_idx, _ = make_fourier_window_indices_np(
        IMAGE_SHAPE,
        current_size,
        square=False,
        include_dc=False,
    )
    recon_idx, _ = make_fourier_window_indices_np(
        IMAGE_SHAPE,
        current_size,
        square=False,
        include_dc=True,
        exact_radius=True,
    )
    return jnp.asarray(score_idx, dtype=jnp.int32), jnp.asarray(recon_idx, dtype=jnp.int32)


def _reference_scores(s, *, score_mode="gaussian", use_window=False, current_size=None):
    projection_kwargs = {}
    if use_window:
        if current_size is None:
            raise ValueError("current_size is required for windowed reference scores")
        projection_kwargs["max_r"] = float(current_size // 2)
    proj_half, proj_abs2_half = _compute_projections_block(
        s["mean_for_proj"],
        s["rotations"],
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        **projection_kwargs,
    )
    if use_window:
        window_indices, _ = _window_indices(current_size)
        half_weights_windowed = s["half_weights"][window_indices]
        shifted_score = s["shifted_score_half"][:, window_indices]
        score_weight = s["score_weight_half"][:, window_indices]
        proj_score = proj_half[:, window_indices].astype(jnp.complex64)
        proj_abs2_score = proj_abs2_half[:, window_indices].astype(jnp.float32)
        if score_mode == "normalized_cc":
            scores = _e_step_block_scores_windowed_normalized_cc(
                shifted_score,
                s["batch_norm"],
                score_weight,
                proj_score * half_weights_windowed,
                proj_abs2_score * half_weights_windowed,
                N_IMAGES,
                N_TRANS,
                int(window_indices.shape[0]),
                IMAGE_SHAPE,
                VOLUME_SHAPE,
            )
        else:
            scores = _e_step_block_scores_windowed(
                shifted_score,
                s["batch_norm"],
                score_weight,
                proj_score * half_weights_windowed,
                proj_abs2_score * half_weights_windowed,
                half_weights_windowed,
                N_IMAGES,
                N_TRANS,
                int(window_indices.shape[0]),
                IMAGE_SHAPE,
                VOLUME_SHAPE,
            )
    else:
        proj_score = proj_half.astype(jnp.complex64)
        proj_abs2_score = proj_abs2_half.astype(jnp.float32)
        if score_mode == "normalized_cc":
            scores = _e_step_block_scores_normalized_cc(
                s["shifted_score_half"],
                s["batch_norm"],
                s["score_weight_half"],
                proj_score * s["half_weights"],
                proj_abs2_score * s["half_weights"],
                N_IMAGES,
                N_TRANS,
                IMAGE_SHAPE,
                VOLUME_SHAPE,
            )
        else:
            scores = _e_step_block_scores(
                s["shifted_score_half"],
                s["batch_norm"],
                s["score_weight_half"],
                proj_score * s["half_weights"],
                proj_abs2_score * s["half_weights"],
                s["half_weights"],
                N_IMAGES,
                N_TRANS,
                IMAGE_SHAPE,
                VOLUME_SHAPE,
            )
    if score_mode == "gaussian":
        scores = scores + s["rotation_log_prior"][:, :, None]
        scores = scores + s["translation_log_prior"][:, None, :]
    scores = jnp.where(s["candidate_mask"][None, :, :], scores, -jnp.inf)
    scores = jnp.where(s["valid_rotation_mask"][None, :, None], scores, -jnp.inf)
    return scores


def _run_big_jit(
    s,
    *,
    run_mstep,
    log_z=None,
    score_mode="gaussian",
    use_window=False,
    current_size=None,
    valid_image_mask=None,
):
    if use_window:
        if current_size is None:
            raise ValueError("current_size is required for windowed big-JIT scores")
        window_indices, recon_window_indices = _window_indices(current_size)
        projection_max_r = float(current_size // 2)
    else:
        window_indices = jnp.arange(N_HALF, dtype=jnp.int32)
        recon_window_indices = jnp.arange(N_HALF, dtype=jnp.int32)
        projection_max_r = "auto"
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
        jnp.ones(N_IMAGES, dtype=bool) if valid_image_mask is None else valid_image_mask,
        jnp.zeros(N_IMAGES, dtype=jnp.float32) if log_z is None else log_z,
        jnp.zeros(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        s["shifted_recon_half"],
        jnp.ones(N_HALF, dtype=jnp.float32),
        jnp.zeros(N_HALF, dtype=jnp.int32),
        jnp.zeros((N_IMAGES, N_TRANS), dtype=jnp.float32),
        window_indices,
        recon_window_indices,
        score_mode=score_mode,
        zero_dc_for_scoring=False,
        use_window=use_window,
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
        projection_max_r=projection_max_r,
        mstep_half_volume=False,
        backprojection_max_r=projection_max_r,
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


def test_dense_big_jit_allows_sparse_pass2_skip_path():
    assert (
        _dense_big_jit_disabled_reason(
            relion_firstiter_winner_take_all=False,
            accumulate_noise=False,
            dense_noise_component_dump_enabled=False,
            per_pose_debug_dump_enabled=False,
        )
        is None
    )


def test_pad_dense_big_jit_image_axis_preserves_ctf_rows():
    batch = np.ones((1, 4, 4), dtype=np.float32)
    ctf_params = np.arange(9, dtype=np.float32).reshape(1, 9)

    padded_batch, padded_ctf, valid_mask, actual_batch_size = _pad_dense_big_jit_image_axis(
        batch,
        ctf_params,
        3,
    )

    assert actual_batch_size == 1
    assert padded_batch.shape == (3, 4, 4)
    assert padded_ctf.shape == (3, 9)
    np.testing.assert_array_equal(valid_mask, np.array([True, False, False]))
    np.testing.assert_allclose(padded_batch[1:], 0.0)
    np.testing.assert_allclose(padded_ctf[1:], np.broadcast_to(ctf_params[0], (2, 9)))


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"relion_firstiter_winner_take_all": True}, "winner_take_all"),
        ({"accumulate_noise": True}, "noise_accumulation"),
        ({"dense_noise_component_dump_enabled": True}, "dense_noise_component_dump"),
        ({"per_pose_debug_dump_enabled": True}, "per_pose_debug_dump"),
    ],
)
def test_dense_big_jit_disabled_reasons(kwargs, reason):
    base = {
        "relion_firstiter_winner_take_all": False,
        "accumulate_noise": False,
        "dense_noise_component_dump_enabled": False,
        "per_pose_debug_dump_enabled": False,
    }
    base.update(kwargs)
    assert _dense_big_jit_disabled_reason(**base) == reason


@pytest.mark.parametrize(
    ("score_mode", "use_window", "current_size"),
    [
        ("normalized_cc", False, None),
        ("gaussian", True, 4),
        ("normalized_cc", True, 4),
    ],
)
def test_dense_big_jit_pass1_matches_dense_primitives_for_modes(score_mode, use_window, current_size):
    s = _inputs()
    scores = _reference_scores(s, score_mode=score_mode, use_window=use_window, current_size=current_size)
    ref_max, ref_sum_exp = _update_logsumexp(
        jnp.full((N_IMAGES,), -jnp.inf),
        jnp.zeros((N_IMAGES,), dtype=jnp.float64),
        scores,
    )

    result = _run_big_jit(
        s,
        run_mstep=False,
        score_mode=score_mode,
        use_window=use_window,
        current_size=current_size,
    )

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
    probs, ref_best, ref_argmax, summed_half, ctf_probs_half = _m_step_block_compute(
        s["shifted_recon_half"],
        scores,
        log_z,
        s["rotations"],
        s["ctf2_over_nv_recon_half"],
        N_IMAGES,
        N_TRANS,
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


def test_dense_big_jit_masks_padded_image_rows():
    s = _inputs()
    scores = _reference_scores(s)
    ref_max, ref_sum_exp = _update_logsumexp(
        jnp.full((N_IMAGES,), -jnp.inf),
        jnp.zeros((N_IMAGES,), dtype=jnp.float64),
        scores,
    )
    valid_image_mask = jnp.asarray([True, False], dtype=bool)

    pass1 = _run_big_jit(
        s,
        run_mstep=False,
        valid_image_mask=valid_image_mask,
    )
    np.testing.assert_allclose(np.asarray(pass1.block_max[0]), np.asarray(ref_max[0]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(pass1.block_sum_exp[0]), np.asarray(ref_sum_exp[0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(pass1.block_max[1]), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(pass1.block_sum_exp[1]), N_ROT * N_TRANS, rtol=0.0, atol=0.0)

    log_z = ref_max + jnp.log(ref_sum_exp)
    log_z = log_z.at[1].set(0.0)
    mstep = _run_big_jit(
        s,
        run_mstep=True,
        log_z=log_z,
        valid_image_mask=valid_image_mask,
    )
    assert np.isneginf(np.asarray(mstep.block_best[1]))
    assert int(np.asarray(mstep.block_argmax[1])) == 0
    np.testing.assert_allclose(np.asarray(mstep.max_posterior[1]), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(mstep.probs_sum_t[1]), 0.0, rtol=0.0, atol=0.0)
