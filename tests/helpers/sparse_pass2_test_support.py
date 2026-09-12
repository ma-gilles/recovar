"""Test-only sparse scoring variants and materialized comparison helpers.

These former production helpers have no runtime callers. Keep them here to
compare cached/packed layouts and chunked noise paths without shipping unused
entry points. They share the named production primitives below, so they are not
independent numerical oracles for those primitives. The tests retain their
separate NumPy arithmetic references.
"""

from functools import partial

import jax
import jax.numpy as jnp

from recovar.em.sparse_pass2.sparse_pass2_scoring import (
    _RELION_CUDA_FINE_REF3D_BLOCK_SIZE,
    _relion_cuda_fine_diff2_to_scores,
    _relion_cuda_fine_normalized_cc_score,
    _relion_cuda_fine_reduce_lanes,
    _score_pass2_bucket_relion_gpu_diff2_single_cached_raw,
)


def _relion_cuda_fine_tree_sum(values):
    """Reproduce RELION CUDA's 256-lane pixel-pass accumulation and tree."""

    block_size = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
    n_values = int(values.shape[-1])
    if n_values == 0:
        return jnp.zeros(values.shape[:-1], dtype=jnp.float32)
    n_passes = (n_values + block_size - 1) // block_size
    padded_size = n_passes * block_size
    values = jnp.asarray(values, dtype=jnp.float32)
    values = jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, padded_size - n_values)])
    pass_values = values.reshape(values.shape[:-1] + (n_passes, block_size))
    lanes = jnp.zeros(values.shape[:-1] + (block_size,), dtype=jnp.float32)
    for pass_index in range(n_passes):
        lanes = lanes + pass_values[..., pass_index, :]
    return _relion_cuda_fine_reduce_lanes(lanes)


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_bucket_relion_gpu_diff2_single_cached(
    shifted_corrected,  # (T, N) complex
    corr_img_score,  # (N,) real
    proj_half,  # (R, N) complex
    half_weights,  # (N,) real
    rotation_log_prior,  # (R,) real
    translation_log_prior,  # (T,) real
    candidate_mask,  # (R, T) bool
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    min_diff2=None,  # scalar or (1,) optional external common minimum
    highres_xi2_half=None,  # scalar float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """Single-image cached-projection variant that avoids a ``(1, R, N)`` copy."""

    score_dtype = jnp.float64 if jnp.asarray(corr_img_score).dtype == jnp.float64 else jnp.float32
    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=score_dtype)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=score_dtype)
    diff2 = _score_pass2_bucket_relion_gpu_diff2_single_cached_raw(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        relion_full_to_compact,
        highres_xi2_half,
        use_fused_ffi=use_fused_ffi,
    )
    return _relion_cuda_fine_diff2_to_scores(
        diff2[jnp.newaxis, :, :],
        rotation_log_prior[jnp.newaxis, :, None],
        translation_log_prior[jnp.newaxis, None, :],
        candidate_mask[jnp.newaxis, :, :],
        min_diff2=min_diff2,
    )[0]


@jax.jit
def _score_pass2_bucket_relion_gpu_normalized_cc_single_cached(
    shifted_score,  # (T, N) complex, RELION-corrected image after shift
    score_weight,  # (N,) real
    proj_half,  # (R, N) complex
    half_weights,  # (N,) real
    candidate_mask,  # (R, T) bool
    relion_full_to_compact=None,  # packed current-size FFTW order -> compact row
):
    """Single-image normalized-CC scorer for cached ``(R, N)`` projections."""

    scores = _relion_cuda_fine_normalized_cc_score(
        proj_half[:, None, :],
        shifted_score[None, :, :],
        score_weight[None, None, :],
        half_weights,
        relion_full_to_compact,
    )
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _score_pass2_pairs_relion_gpu_normalized_cc(
    shifted_score,  # (B, T, N) complex, RELION-corrected image after shift
    score_weight,  # (B, N) real, CTF^2 / Xi2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    local_rotation_row,  # (B, P) int
    translation_idx,  # (B, P) int
    pair_mask,  # (B, P) bool
    relion_full_to_compact=None,  # packed current-size FFTW order -> compact row
):
    """RELION iter-1 normalized-CC scoring for compact pass-2 pairs."""

    batch = shifted_score.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)

    shifted_pair = shifted_score[row, safe_translation_idx, :]
    proj_pair = proj_half[row, safe_rotation_row, :]
    scores = _relion_cuda_fine_normalized_cc_score(
        proj_pair,
        shifted_pair,
        score_weight[:, None, :],
        half_weights,
        relion_full_to_compact,
    )
    scores = jnp.where(pair_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _normalize_pass2_pairs_score_only(pair_scores, pair_mask):
    """Compute compact pair score stats without materializing dense posteriors."""
    pair_scores = jnp.where(pair_mask & jnp.isfinite(pair_scores), pair_scores, -jnp.inf)
    best_log_score = jnp.max(pair_scores, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    shifted = jnp.where(has_finite_score[:, None], pair_scores - safe_best_log_score[:, None], -jnp.inf)
    exp_terms = jnp.exp(shifted.astype(jnp.float64))
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms, axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    safe_sum_exp = jnp.where(has_mass, sum_exp, 1.0)
    log_Z = jnp.where(has_mass, safe_best_log_score + jnp.log(safe_sum_exp), 0.0)
    best_argmax = jnp.where(has_mass, jnp.argmax(pair_scores, axis=1), 0)
    max_posterior = jnp.exp(best_log_score - log_Z)
    max_posterior = jnp.where(has_mass & jnp.isfinite(max_posterior), max_posterior, 0.0)
    best_log_score = jnp.where(has_mass, best_log_score, -jnp.inf)
    return log_Z, best_log_score, best_argmax, max_posterior


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _select_active_noise_rows(
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs_for_noise,
    active_indices,
    active_mask,
    *,
    n_rotation_rows: int,
):
    """Gather compact active noise rows and their image ids in one launch."""

    active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
    active_mask = jnp.asarray(active_mask)
    row_mask = active_mask.astype(jnp.asarray(summed_masked_noise).real.dtype)
    active_image_indices = jnp.where(active_mask.astype(jnp.int32) != 0, active_indices // int(n_rotation_rows), 0)

    def gather(values):
        flat_values = values.reshape((values.shape[0] * values.shape[1], values.shape[-1]))
        gathered = flat_values[active_indices]
        return gathered * row_mask[:, None]

    return (
        gather(proj_for_noise),
        gather(proj_abs2_for_noise),
        gather(summed_masked_noise),
        gather(ctf_probs_for_noise),
        active_image_indices,
    )
