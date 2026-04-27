"""Exact local score and normalization helpers."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.oversampling import find_significant_mask


@jax.jit
def score_local_bucket(
    shifted,
    ctf2_over_nv,
    proj_weighted,
    proj_abs2_weighted,
    rotation_log_prior,
    translation_log_prior,
    rotation_mask,
    sample_mask=None,
):
    """Compute exact local scores on a padded per-image hypothesis bucket."""

    # shifted: (B, T, N), proj_weighted: (B, R, N)
    cross = -2.0 * jnp.einsum(
        "btn,brn->btr",
        jnp.conj(shifted),
        proj_weighted,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    cross = cross.swapaxes(1, 2)  # (B, R, T)
    norms = jnp.einsum(
        "bn,brn->br",
        ctf2_over_nv,
        proj_abs2_weighted,
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = -0.5 * (cross + norms[..., None])
    scores = scores + rotation_log_prior[:, :, None]
    scores = scores + translation_log_prior[:, None, :]
    valid_mask = rotation_mask[:, :, None]
    if sample_mask is not None:
        valid_mask = valid_mask & sample_mask
    return jnp.where(valid_mask, scores, -jnp.inf)


@jax.jit
def normalize_local_scores(scores):
    """Return exact per-image log normalizer, posterior, and argmax."""

    flat_scores = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    log_shift = best_log_score[:, None, None]
    probs = jnp.exp((scores - log_shift).astype(jnp.float64))
    sum_exp = jnp.sum(probs.reshape(scores.shape[0], -1), axis=1)
    log_Z = best_log_score + jnp.log(sum_exp)
    probs = jnp.exp(scores - log_Z[:, None, None])
    best_argmax = jnp.argmax(flat_scores, axis=1)
    max_posterior = jnp.exp(best_log_score - log_Z)
    return log_Z, probs, best_log_score, best_argmax, max_posterior


def compute_reconstruction_support(probs, adaptive_fraction=0.999, max_significants=-1):
    """Return RELION-style significant reconstruction support.

    RELION computes the full posterior over local (rotation, translation)
    hypotheses, then reconstructs only those samples whose weight is at least
    the per-image ``significant_weight`` threshold implied by
    ``adaptive_fraction`` and ``maximum_significants``.
    """

    flat_probs = probs.reshape(probs.shape[0], -1)
    significant_flat, n_significant_samples = find_significant_mask(
        flat_probs,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
    )
    significant_samples = significant_flat.reshape(probs.shape)
    significant_rotations = jnp.any(significant_samples, axis=-1)
    return significant_samples, significant_rotations, n_significant_samples


@partial(jax.jit, static_argnums=(2,))
def decode_local_argmax(best_argmax, bucket_rotation_count, n_trans):
    """Decode flattened local argmax indices into rotation and translation ids."""

    local_rot_idx = best_argmax // n_trans
    trans_idx = best_argmax % n_trans
    return local_rot_idx, trans_idx
