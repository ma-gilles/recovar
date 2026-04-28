"""Exact local score and normalization helpers."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.oversampling import (
    _find_significant_mask_full_sort,
    find_significant_mask,
)


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
def score_local_bucket_abs2_on_demand(
    shifted,
    ctf2_over_nv,
    proj_weighted,
    rotation_log_prior,
    translation_log_prior,
    rotation_mask,
    sample_mask=None,
):
    """Compute local scores without storing a separate |projection|^2 tensor."""

    cross = -2.0 * jnp.einsum(
        "btn,brn->btr",
        jnp.conj(shifted),
        proj_weighted,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    cross = cross.swapaxes(1, 2)
    norms = jnp.einsum(
        "bn,brn->br",
        ctf2_over_nv,
        jnp.abs(proj_weighted) ** 2,
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
def score_local_bucket_abs2_weighted_on_demand(
    shifted,
    ctf2_over_nv,
    proj_weighted,
    half_weights,
    rotation_log_prior,
    translation_log_prior,
    rotation_mask,
    sample_mask=None,
):
    """Compute local scores without materializing ``|projection|^2 * weights``.

    ``proj_weighted`` is already multiplied by the Hermitian half-spectrum
    weights for the cross term.  The projection norm needs exactly one copy of
    those weights, so recover ``|proj|^2 * weight`` as
    ``|proj * weight|^2 / weight``.
    """

    cross = -2.0 * jnp.einsum(
        "btn,brn->btr",
        jnp.conj(shifted),
        proj_weighted,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    cross = cross.swapaxes(1, 2)
    weighted_abs2 = (jnp.abs(proj_weighted) ** 2) / half_weights[None, None, :]
    norms = jnp.einsum(
        "bn,brn->br",
        ctf2_over_nv,
        weighted_abs2,
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


@jax.jit
def normalize_local_scores_with_log_z(scores, log_z):
    """Normalize local scores with an externally computed full-grid log-Z."""

    flat_scores = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    probs = jnp.exp(scores - log_z[:, None, None])
    best_argmax = jnp.argmax(flat_scores, axis=1)
    max_posterior = jnp.exp(best_log_score - log_z)
    return log_z, probs, best_log_score, best_argmax, max_posterior


@jax.jit
def normalize_local_scores_float32(scores):
    """Return local posterior statistics without upcasting to float64."""

    flat_scores = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    log_shift = best_log_score[:, None, None]
    probs = jnp.exp(scores - log_shift)
    sum_exp = jnp.sum(probs.reshape(scores.shape[0], -1), axis=1)
    log_Z = best_log_score + jnp.log(sum_exp)
    probs = jnp.exp(scores - log_Z[:, None, None])
    best_argmax = jnp.argmax(flat_scores, axis=1)
    max_posterior = jnp.exp(best_log_score - log_Z)
    return log_Z, probs, best_log_score, best_argmax, max_posterior


@jax.jit
def normalize_local_scores_with_log_z_float32(scores, log_z):
    """Normalize local scores with an externally computed full-grid log-Z without x64."""

    flat_scores = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    probs = jnp.exp(scores - log_z[:, None, None])
    best_argmax = jnp.argmax(flat_scores, axis=1)
    max_posterior = jnp.exp(best_log_score - log_z)
    return log_z, probs, best_log_score, best_argmax, max_posterior


def normalize_local_scores_auto(scores, log_z=None, *, use_float64_normalization: bool):
    """Select the local posterior normalizer from explicit precision inputs."""

    if log_z is None:
        if use_float64_normalization:
            return normalize_local_scores(scores)
        return normalize_local_scores_float32(scores)
    if use_float64_normalization:
        return normalize_local_scores_with_log_z(scores, log_z)
    return normalize_local_scores_with_log_z_float32(scores, log_z)


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


def _compute_reconstruction_support_full_sort_jit(probs, adaptive_fraction=0.999, max_significants=-1):
    flat_probs = probs.reshape(probs.shape[0], -1)
    significant_flat, n_significant_samples = _find_significant_mask_full_sort(
        flat_probs,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
    )
    significant_samples = significant_flat.reshape(probs.shape)
    significant_rotations = jnp.any(significant_samples, axis=-1)
    return significant_samples, significant_rotations, n_significant_samples


@partial(
    jax.jit,
    static_argnames=(
        "half_spectrum_scoring",
        "use_float64_normalization",
        "reconstruct_significant_only",
        "adaptive_fraction",
        "max_significants",
    ),
)
def fused_score_normalize_mstep_abs2_on_demand(
    shifted_score_split,
    ctf2_over_nv_score,
    proj_weighted,
    half_weights,
    rotation_log_prior,
    translation_log_prior,
    rotation_mask,
    sample_mask,
    shifted_recon_split,
    ctf2_over_nv_recon,
    *,
    half_spectrum_scoring: bool,
    use_float64_normalization: bool,
    reconstruct_significant_only: bool,
    adaptive_fraction: float,
    max_significants: int,
):
    """Fuse exact-local score, posterior, support, and M-step reductions.

    This intentionally covers the common on-demand ``|projection|^2`` path.
    Keeping projection and backprojection outside the fused block preserves the
    existing custom-kernel boundaries while removing intermediate score/posterior
    launch and materialization overhead.
    """

    cross = -2.0 * jnp.einsum(
        "btn,brn->btr",
        jnp.conj(shifted_score_split),
        proj_weighted,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    cross = cross.swapaxes(1, 2)
    if half_spectrum_scoring:
        weighted_abs2 = jnp.abs(proj_weighted) ** 2
    else:
        weighted_abs2 = (jnp.abs(proj_weighted) ** 2) / half_weights[None, None, :]
    norms = jnp.einsum(
        "bn,brn->br",
        ctf2_over_nv_score,
        weighted_abs2,
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = -0.5 * (cross + norms[..., None])
    scores = scores + rotation_log_prior[:, :, None]
    scores = scores + translation_log_prior[:, None, :]
    valid_mask = rotation_mask[:, :, None]
    if sample_mask is not None:
        valid_mask = valid_mask & sample_mask
    scores = jnp.where(valid_mask, scores, -jnp.inf)

    flat_scores = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    if use_float64_normalization:
        log_shift = best_log_score[:, None, None]
        shifted_exp = jnp.exp((scores - log_shift).astype(jnp.float64))
        sum_exp = jnp.sum(shifted_exp.reshape(scores.shape[0], -1), axis=1)
        log_Z = best_log_score + jnp.log(sum_exp)
        probs = jnp.exp(scores - log_Z[:, None, None])
    else:
        log_shift = best_log_score[:, None, None]
        shifted_exp = jnp.exp(scores - log_shift)
        sum_exp = jnp.sum(shifted_exp.reshape(scores.shape[0], -1), axis=1)
        log_Z = best_log_score + jnp.log(sum_exp)
        probs = jnp.exp(scores - log_Z[:, None, None])
    best_argmax = jnp.argmax(flat_scores, axis=1)
    max_posterior = jnp.exp(best_log_score - log_Z)

    if reconstruct_significant_only:
        (
            reconstruction_sample_mask,
            reconstruction_rotation_mask,
            n_significant_samples,
        ) = _compute_reconstruction_support_full_sort_jit(
            probs,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )
        reconstruction_probs = jnp.where(reconstruction_sample_mask, probs, 0.0)
    else:
        reconstruction_rotation_mask = rotation_mask
        reconstruction_sample_mask = jnp.broadcast_to(
            reconstruction_rotation_mask[:, :, None],
            probs.shape,
        )
        n_significant_samples = jnp.sum(reconstruction_rotation_mask, axis=1).astype(jnp.int32) * probs.shape[-1]
        reconstruction_probs = probs

    probs_sum_t = jnp.sum(probs, axis=-1)
    reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
    summed = jnp.matmul(reconstruction_probs, shifted_recon_split)
    ctf_probs = reconstruction_probs_sum_t[..., None] * ctf2_over_nv_recon[:, None, :]
    return (
        log_Z,
        probs,
        best_log_score,
        best_argmax,
        max_posterior,
        reconstruction_sample_mask,
        reconstruction_rotation_mask,
        n_significant_samples,
        reconstruction_probs,
        probs_sum_t,
        reconstruction_probs_sum_t,
        summed,
        ctf_probs,
    )
