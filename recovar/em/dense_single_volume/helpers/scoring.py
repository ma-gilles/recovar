"""JAX scoring and M-step kernels shared by dense single-volume EM helpers."""

from functools import partial

import jax
import jax.numpy as jnp

from .dtype_policy import DensePrecisionPolicy


def _score_rotation_block(
    window_spec,
    *,
    shifted_score,
    batch_norm,
    score_weight,
    proj_half,
    proj_abs2_half,
    half_weights,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
    score_mode: str,
    precision_policy: DensePrecisionPolicy,
):
    """Score one rotation block against the active Fourier-window spec."""

    proj_score = window_spec.score_values(proj_half)
    proj_abs2_score = window_spec.score_values(proj_abs2_half)
    proj_score, proj_abs2_score = precision_policy.cast_projection_scores(
        proj_score,
        proj_abs2_score,
    )
    weights = window_spec.score_values(half_weights)
    proj_weighted = proj_score * weights
    proj_abs2_weighted = proj_abs2_score * weights
    n_score = window_spec.n_score
    if score_mode == "normalized_cc":
        return _e_step_block_scores_windowed_normalized_cc(
            shifted_score,
            batch_norm,
            score_weight,
            proj_weighted,
            proj_abs2_weighted,
            n_images,
            n_trans,
            n_score,
            image_shape,
            volume_shape,
        )
    return _e_step_block_scores_windowed(
        shifted_score,
        batch_norm,
        score_weight,
        proj_weighted,
        proj_abs2_weighted,
        weights,
        n_images,
        n_trans,
        n_score,
        image_shape,
        volume_shape,
    )


@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def _e_step_block_scores(
    shifted_half,
    batch_norm,
    ctf2_over_nv_half,
    proj_half_weighted,
    proj_abs2_half,
    half_weights,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
):
    """E-step for one rotation block using half-spectrum GEMMs.

    The cross-term GEMM uses weighted projections (half_weights absorbed into
    projections, precomputed once per rotation block) to recover the full inner
    product from half-spectrum data:

        cross[i,r] = -2 Re(conj(shifted_half) @ proj_half_weighted.T)

    The norm-term similarly uses half-weighted |proj|^2.
    """
    rot_block_size = proj_half_weighted.shape[0]
    cross = (
        -2.0
        * jnp.matmul(
            jnp.conj(shifted_half),
            proj_half_weighted.T,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv_half,
        proj_abs2_half.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    residuals = cross + norms[..., None]
    return -0.5 * residuals


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10))
def _e_step_block_scores_windowed(
    shifted_windowed,
    batch_norm,
    ctf2_over_nv_windowed,
    proj_windowed_weighted,
    proj_abs2_windowed,
    half_weights_windowed,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """E-step for one rotation block using windowed half-spectrum GEMMs."""
    rot_block_size = proj_windowed_weighted.shape[0]
    cross = (
        -2.0
        * jnp.matmul(
            jnp.conj(shifted_windowed),
            proj_windowed_weighted.T,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv_windowed,
        proj_abs2_windowed.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    residuals = cross + norms[..., None]
    return -0.5 * residuals


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _e_step_block_scores_normalized_cc(
    shifted_half,
    batch_norm,
    ctf2_over_nv_half,
    proj_half_weighted,
    proj_abs2_half,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
):
    """RELION iter-1 normalized cross-correlation score."""
    del batch_norm, image_shape, volume_shape
    rot_block_size = proj_half_weighted.shape[0]
    cross = (
        -2.0
        * jnp.matmul(
            jnp.conj(shifted_half),
            proj_half_weighted.T,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv_half,
        proj_abs2_half.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    return (-0.5 * cross) / denom[..., None]


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def _e_step_block_scores_windowed_normalized_cc(
    shifted_windowed,
    batch_norm,
    ctf2_over_nv_windowed,
    proj_windowed_weighted,
    proj_abs2_windowed,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """Windowed RELION iter-1 normalized cross-correlation score."""
    del batch_norm, n_windowed, image_shape, volume_shape
    rot_block_size = proj_windowed_weighted.shape[0]
    cross = (
        -2.0
        * jnp.matmul(
            jnp.conj(shifted_windowed),
            proj_windowed_weighted.T,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv_windowed,
        proj_abs2_windowed.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    return (-0.5 * cross) / denom[..., None]


def _winner_take_all_probs_for_block(best_argmax, r0, actual_rot, rotation_block_size, n_trans, dtype):
    """Return one-hot pose probabilities for one rotation block."""
    best_argmax = jnp.asarray(best_argmax, dtype=jnp.int32)
    winning_rot = best_argmax // n_trans
    winning_trans = best_argmax % n_trans
    in_block = (winning_rot >= r0) & (winning_rot < (r0 + actual_rot))
    safe_actual_rot = max(int(actual_rot), 1)
    local_rot = jnp.clip(winning_rot - r0, 0, safe_actual_rot - 1)
    flat_local = local_rot * n_trans + winning_trans
    probs = jax.nn.one_hot(
        flat_local,
        rotation_block_size * n_trans,
        dtype=dtype,
    ).reshape(best_argmax.shape[0], rotation_block_size, n_trans)
    return probs * in_block[:, None, None]


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11))
def _m_step_block_windowed(
    shifted_windowed,
    scores_block,
    log_Z,
    rotations_block,
    ctf2_over_nv_windowed,
    Ft_y,
    Ft_ctf,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """Normalize scores to probabilities and compute one windowed M-step block.

    Uses an isfinite-guarded ``exp(scores - log_Z)`` so K-class adaptive 2-pass
    poses where ``scores = -inf`` and ``log_Z = -inf`` give ``probs = 0`` rather
    than NaN.
    """
    rot_block_size = rotations_block.shape[0]
    diff = scores_block - log_Z[:, None, None]
    probs = jnp.where(jnp.isfinite(diff), jnp.exp(diff), 0.0)
    P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
    summed_windowed = P @ shifted_windowed
    probs_sum_t = jnp.sum(probs, axis=-1)
    ctf_probs_windowed = probs_sum_t.T @ ctf2_over_nv_windowed
    block_best = jnp.max(scores_block.reshape(n_images, -1), axis=1)
    block_argmax = jnp.argmax(scores_block.reshape(n_images, -1), axis=1)
    return Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_windowed, ctf_probs_windowed


@partial(jax.jit, static_argnums=())
def _update_logsumexp(max_s, sum_exp, scores_block):
    """Streaming logsumexp update from one score block.

    Robust to all-(-inf) score blocks (K-class adaptive 2-pass with an
    empty significance mask): a finite ``safe_new_max`` is used inside the
    exp so we never form -inf - (-inf) = NaN; ``new_max`` is still returned
    as -inf so the streaming logsumexp is exactly -inf for empty inputs.
    """

    accumulator_dtype = sum_exp.dtype
    scores_flat = scores_block.reshape(scores_block.shape[0], -1)
    block_max = jnp.max(scores_flat, axis=1)
    new_max = jnp.maximum(max_s, block_max)
    safe_new_max = jnp.where(jnp.isfinite(new_max), new_max, jnp.zeros_like(new_max))
    exp_terms = jnp.sum(
        jnp.exp((scores_flat - safe_new_max[:, None]).astype(accumulator_dtype)),
        axis=1,
    )
    old_term = jnp.where(
        jnp.isfinite(max_s),
        sum_exp * jnp.exp((max_s - safe_new_max).astype(accumulator_dtype)),
        jnp.zeros_like(sum_exp),
    )
    sum_exp = old_term + exp_terms
    return new_max, sum_exp


@jax.jit
def _merge_block_logsumexp(max_s, sum_exp, block_max, block_sum_exp):
    """Merge one pre-reduced block logsumexp into streaming batch stats.

    See ``_update_logsumexp`` for the all-(-inf) handling. The ``safe_*``
    shifts here mirror the same pattern: when both ``max_s`` and
    ``block_max`` are -inf the merge degenerates to 0 + 0 = 0, giving a
    final ``log_Z = -inf`` that the K-class aggregator treats as "no
    contribution" rather than NaN.
    """

    accumulator_dtype = sum_exp.dtype
    new_max = jnp.maximum(max_s, block_max)
    safe_new_max = jnp.where(jnp.isfinite(new_max), new_max, jnp.zeros_like(new_max))
    old_term = jnp.where(
        jnp.isfinite(max_s),
        sum_exp * jnp.exp((max_s - safe_new_max).astype(accumulator_dtype)),
        jnp.zeros_like(sum_exp),
    )
    block_term = jnp.where(
        jnp.isfinite(block_max),
        block_sum_exp.astype(accumulator_dtype) * jnp.exp((block_max - safe_new_max).astype(accumulator_dtype)),
        jnp.zeros_like(block_sum_exp.astype(accumulator_dtype)),
    )
    return new_max, old_term + block_term


@partial(jax.jit, static_argnums=(5, 6))
def _m_step_block_compute(
    shifted_half,
    scores_block,
    log_Z,
    rotations_block,
    ctf2_over_nv_half,
    n_images,
    n_trans,
):
    """Normalize scores to probabilities and compute one non-windowed M-step block.

    Uses an isfinite-guarded ``exp(scores - log_Z)`` so K-class adaptive 2-pass
    poses where ``scores = -inf`` and ``log_Z = -inf`` give ``probs = 0`` rather
    than NaN.
    """
    rot_block_size = rotations_block.shape[0]
    diff = scores_block - log_Z[:, None, None]
    probs = jnp.where(jnp.isfinite(diff), jnp.exp(diff), 0.0)
    P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
    summed_half = P @ shifted_half
    probs_sum_t = jnp.sum(probs, axis=-1)
    ctf_probs_half = probs_sum_t.T @ ctf2_over_nv_half
    block_best = jnp.max(scores_block.reshape(n_images, -1), axis=1)
    block_argmax = jnp.argmax(scores_block.reshape(n_images, -1), axis=1)
    return probs, block_best, block_argmax, summed_half, ctf_probs_half
