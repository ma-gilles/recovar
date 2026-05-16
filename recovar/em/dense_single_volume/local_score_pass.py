"""Exact local score and normalization helpers."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.oversampling import (
    _find_significant_mask_full_sort,
    find_significant_mask,
)


def _local_scores_from_weighted_abs2(
    shifted,
    ctf2_over_nv,
    proj_weighted,
    weighted_abs2,
    rotation_log_prior,
    translation_log_prior,
    rotation_mask,
    sample_mask,
):
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


def _weighted_abs2_from_weighted_projection(proj_weighted, half_weights, *, weights_absorbed_once: bool):
    if weights_absorbed_once:
        return (jnp.abs(proj_weighted) ** 2) / half_weights[None, None, :]
    return jnp.abs(proj_weighted) ** 2


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

    return _local_scores_from_weighted_abs2(
        shifted,
        ctf2_over_nv,
        proj_weighted,
        proj_abs2_weighted,
        rotation_log_prior,
        translation_log_prior,
        rotation_mask,
        sample_mask,
    )


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

    return _local_scores_from_weighted_abs2(
        shifted,
        ctf2_over_nv,
        proj_weighted,
        _weighted_abs2_from_weighted_projection(
            proj_weighted,
            None,
            weights_absorbed_once=False,
        ),
        rotation_log_prior,
        translation_log_prior,
        rotation_mask,
        sample_mask,
    )


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

    return _local_scores_from_weighted_abs2(
        shifted,
        ctf2_over_nv,
        proj_weighted,
        _weighted_abs2_from_weighted_projection(
            proj_weighted,
            half_weights,
            weights_absorbed_once=True,
        ),
        rotation_log_prior,
        translation_log_prior,
        rotation_mask,
        sample_mask,
    )


@jax.jit
def normalize_local_scores(scores):
    """Return exact per-image log normalizer, posterior, and argmax."""

    return _normalize_scores(scores, use_float64=True)


def _normalize_scores(scores, *, use_float64: bool):
    flat_scores = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    has_valid_sample = jnp.isfinite(best_log_score)
    safe_best = jnp.where(has_valid_sample, best_log_score, 0.0)
    exp_dtype = jnp.float64 if use_float64 else scores.dtype
    shifted_exp = jnp.exp((scores - safe_best[:, None, None]).astype(exp_dtype))
    shifted_exp = jnp.where(has_valid_sample[:, None, None], shifted_exp, 0.0)
    sum_exp = jnp.sum(shifted_exp.reshape(scores.shape[0], -1), axis=1)
    log_Z = best_log_score + jnp.log(sum_exp)
    log_Z = jnp.where(has_valid_sample, log_Z, -jnp.inf)
    safe_log_Z = jnp.where(has_valid_sample, log_Z, 0.0)
    probs = jnp.exp((scores - safe_log_Z[:, None, None]).astype(exp_dtype))
    probs = jnp.where(has_valid_sample[:, None, None], probs, 0.0)
    best_argmax = jnp.argmax(flat_scores, axis=1)
    best_argmax = jnp.where(has_valid_sample, best_argmax, 0)
    max_posterior = jnp.exp((best_log_score - safe_log_Z).astype(exp_dtype))
    max_posterior = jnp.where(has_valid_sample, max_posterior, 0.0)
    return log_Z, probs, best_log_score, best_argmax, max_posterior


@jax.jit
def normalize_local_scores_with_log_z(scores, log_z):
    """Normalize local scores with an externally computed full-grid log-Z."""

    return _normalize_scores_with_log_z(scores, log_z)


def _normalize_scores_with_log_z(scores, log_z):
    flat_scores = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    has_valid_sample = jnp.isfinite(best_log_score) & jnp.isfinite(log_z)
    safe_log_z = jnp.where(has_valid_sample, log_z, 0.0)
    probs = jnp.exp(scores - safe_log_z[:, None, None])
    probs = jnp.where(has_valid_sample[:, None, None], probs, 0.0)
    best_argmax = jnp.argmax(flat_scores, axis=1)
    best_argmax = jnp.where(has_valid_sample, best_argmax, 0)
    max_posterior = jnp.exp(best_log_score - safe_log_z)
    max_posterior = jnp.where(has_valid_sample, max_posterior, 0.0)
    return log_z, probs, best_log_score, best_argmax, max_posterior


@jax.jit
def normalize_local_scores_float32(scores):
    """Return local posterior statistics without upcasting to float64."""

    return _normalize_scores(scores, use_float64=False)


@jax.jit
def normalize_local_scores_with_log_z_float32(scores, log_z):
    """Normalize local scores with an externally computed full-grid log-Z without x64."""

    return _normalize_scores_with_log_z(scores, log_z)


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
    has_posterior_mass = jnp.sum(flat_probs, axis=1) > 0.0
    significant_flat = jnp.where(has_posterior_mass[:, None], significant_flat, False)
    n_significant_samples = jnp.where(has_posterior_mass, n_significant_samples, 0)
    significant_samples = significant_flat.reshape(probs.shape)
    significant_rotations = jnp.any(significant_samples, axis=-1)
    return significant_samples, significant_rotations, n_significant_samples


def compute_reconstruction_support_from_threshold(probs, threshold):
    """Return RELION-style support from a per-image global probability cutoff."""

    flat_probs = probs.reshape(probs.shape[0], -1)
    threshold = jnp.asarray(threshold, dtype=probs.dtype).reshape((probs.shape[0], 1, 1))
    has_posterior_mass = jnp.sum(flat_probs, axis=1) > 0.0
    significant_samples = (probs > 0.0) & (probs >= threshold)
    significant_samples = jnp.where(has_posterior_mass[:, None, None], significant_samples, False)
    n_significant_samples = jnp.sum(significant_samples.reshape(probs.shape[0], -1), axis=1).astype(jnp.int32)
    significant_rotations = jnp.any(significant_samples, axis=-1)
    return significant_samples, significant_rotations, n_significant_samples


def _compute_reconstruction_support_full_sort_jit(probs, adaptive_fraction=0.999, max_significants=-1):
    flat_probs = probs.reshape(probs.shape[0], -1)
    significant_flat, n_significant_samples = _find_significant_mask_full_sort(
        flat_probs,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
    )
    has_posterior_mass = jnp.sum(flat_probs, axis=1) > 0.0
    significant_flat = jnp.where(has_posterior_mass[:, None], significant_flat, False)
    n_significant_samples = jnp.where(has_posterior_mass, n_significant_samples, 0)
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
    reconstruction_probability_threshold=None,
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

    scores = _local_scores_from_weighted_abs2(
        shifted_score_split,
        ctf2_over_nv_score,
        proj_weighted,
        _weighted_abs2_from_weighted_projection(
            proj_weighted,
            half_weights,
            weights_absorbed_once=not half_spectrum_scoring,
        ),
        rotation_log_prior,
        translation_log_prior,
        rotation_mask,
        sample_mask,
    )

    log_Z, probs, best_log_score, best_argmax, max_posterior = _normalize_scores(
        scores,
        use_float64=use_float64_normalization,
    )

    if reconstruct_significant_only:
        if reconstruction_probability_threshold is None:
            (
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
            ) = _compute_reconstruction_support_full_sort_jit(
                probs,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
        else:
            (
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
            ) = compute_reconstruction_support_from_threshold(
                probs,
                reconstruction_probability_threshold,
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
    ctf_probs = jnp.where(
        reconstruction_probs_sum_t[..., None] != 0.0,
        reconstruction_probs_sum_t[..., None] * ctf2_over_nv_recon[:, None, :],
        0.0,
    )
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
