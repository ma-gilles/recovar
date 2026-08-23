"""Two-pass adaptive oversampling for dense single-volume EM.

Implements Phase 5 of the RELION-parity plan: significance pruning after
a coarse E-step pass, then oversampled evaluation of only significant
orientations x translations.

RELION's approach:
- Pass 1 (coarse): evaluate ALL rotations at base angular sampling using
  a smaller Fourier window.  Compute posterior weights.  Identify significant
  (rotation, translation) pairs per image.
- Pass 2 (fine): for each image, evaluate ONLY its significant coarse
  rotations' children at oversampled angles using a larger Fourier window.

The significance criterion matches RELION's ``adaptive_fraction``: keep
the smallest set of (rotation, translation) samples whose cumulative weight is
strictly greater than ``adaptive_fraction`` of the total posterior weight.
Cap at max_significants
to bound memory/compute (RELION's --maxsig semantics, counting SAMPLES
not just orientations -- see C5 in plan_relion_parity.md).

See docs/math/plan_relion_parity.md, Phase 5.
"""

import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .types import make_noise_stats, make_relion_stats

logger = logging.getLogger(__name__)
_FAST_SIGNIFICANCE_TOPK = 64


def _relion_cuda_f32_tail_target(sum_weight, adaptive_fraction: float):
    """Match RELION's parsed adaptive-fraction arithmetic at the CUDA cutoff.

    RELION initializes ``adaptive_fraction`` with ``textToFloat`` even when
    ``RFLOAT`` is double.  The resulting float32 value is widened for the host
    product with ``op.sum_weight`` and finally narrowed to the CUDA ``XFLOAT``
    threshold argument.  Starting from Python's float64 value can move a fine
    significance cutoff across one or more nearly tied candidates.
    """

    parsed_fraction = jnp.asarray(adaptive_fraction, dtype=jnp.float32)
    return jnp.asarray(
        (jnp.float64(1.0) - parsed_fraction.astype(jnp.float64))
        * jnp.asarray(sum_weight, dtype=jnp.float32).astype(jnp.float64),
        dtype=jnp.float32,
    )


@partial(jax.jit, static_argnames=("adaptive_fraction", "max_significants"))
def relion_cuda_f32_coarse_posterior(
    scores_flat,
    *,
    adaptive_fraction=0.999,
    max_significants=500,
):
    """Reproduce RELION CUDA coarse-weight and significance arithmetic.

    RELION's accelerated coarse pass stores log weights in ``XFLOAT``
    (float32 in the deployed build), shifts their maximum to 50, applies
    ``expf``, radix-sorts positive weights in ascending order, and uses an
    inclusive float32 scan to select the lower-tail cutoff.  The surviving
    weights remain normalized by the full, pre-pruning sum.

    ``cutoff_count`` is the pre-tie rank serialized by RELION.  ``mask`` and
    ``n_significant`` include every positive weight tied at the cutoff.
    """

    scores_f32 = jnp.asarray(scores_flat, dtype=jnp.float32)
    finite = jnp.isfinite(scores_f32)
    best = jnp.max(jnp.where(finite, scores_f32, -jnp.inf), axis=1)
    has_finite = jnp.isfinite(best)
    safe_best = jnp.where(has_finite, best, jnp.float32(0.0))
    shifted = jnp.where(
        finite,
        scores_f32 - safe_best[:, None] + jnp.float32(50.0),
        -jnp.inf,
    )
    raw_weights = jnp.where(
        shifted < jnp.float32(-88.0),
        jnp.float32(0.0),
        jnp.exp(shifted),
    )
    raw_weights = jnp.where(
        finite & jnp.isfinite(raw_weights),
        raw_weights,
        jnp.float32(0.0),
    )

    if jax.default_backend() == "gpu":
        from recovar.cuda_backproject import relion_cub_sort_scan_f32

        sorted_weights, cumulative = jax.vmap(relion_cub_sort_scan_f32)(raw_weights)
    else:
        # Keep a CPU reference path for isolated unit tests. The live opt-in
        # route is CUDA-only and uses RELION's exact CUB primitives above.
        sorted_weights = jnp.sort(raw_weights, axis=1)
        cumulative = jnp.cumsum(sorted_weights, axis=1, dtype=jnp.float32)
    sum_weight = cumulative[:, -1]
    has_mass = has_finite & jnp.isfinite(sum_weight) & (sum_weight > jnp.float32(0.0))
    tail_target = _relion_cuda_f32_tail_target(sum_weight, adaptive_fraction)
    threshold_idx = jax.vmap(
        lambda row, target: jnp.searchsorted(row, target, side="right"),
    )(cumulative, tail_target)

    n_samples = scores_f32.shape[1]
    positive_count = jnp.sum(raw_weights > jnp.float32(0.0), axis=1).astype(jnp.int32)
    first_positive = jnp.asarray(n_samples, dtype=jnp.int32) - positive_count
    threshold_idx = jnp.maximum(threshold_idx.astype(jnp.int32), first_positive)
    threshold_idx = jnp.minimum(threshold_idx, jnp.asarray(n_samples - 1, dtype=jnp.int32))
    if max_significants is not None and int(max_significants) > 0:
        threshold_idx = jnp.maximum(
            threshold_idx,
            jnp.asarray(n_samples - int(max_significants), dtype=jnp.int32),
        )

    threshold = sorted_weights[jnp.arange(scores_f32.shape[0]), threshold_idx]
    mask = has_mass[:, None] & (raw_weights > jnp.float32(0.0)) & (
        raw_weights >= threshold[:, None]
    )
    safe_sum_weight = jnp.where(has_mass, sum_weight, jnp.float32(1.0))
    probabilities = jnp.where(
        has_mass[:, None],
        raw_weights / safe_sum_weight[:, None],
        jnp.float32(0.0),
    )
    n_significant = jnp.sum(mask, axis=1).astype(jnp.int32)
    cutoff_count = jnp.where(
        has_mass,
        jnp.asarray(n_samples, dtype=jnp.int32) - threshold_idx,
        jnp.int32(0),
    )
    return probabilities, mask, n_significant, cutoff_count, sum_weight, threshold


def map_translation_log_prior_to_fine_grid(
    translation_log_prior,
    fine_translation_parent,
):
    """Map coarse-grid translation priors onto oversampled translation children."""
    if translation_log_prior is None:
        return None
    translation_log_prior = np.asarray(translation_log_prior, dtype=np.float32)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int64)
    if translation_log_prior.ndim == 1:
        return translation_log_prior[fine_translation_parent]
    if translation_log_prior.ndim == 2:
        return translation_log_prior[:, fine_translation_parent]
    raise ValueError(
        f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions",
    )


# ---------------------------------------------------------------------------
# Significance pruning
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(1, 2, 3))
def _find_significant_mask_full_sort(
    weights_flat,
    adaptive_fraction=0.999,
    max_significants=500,
    return_cutoff_count=False,
):
    """Find significant orientation x translation pairs per image.

    For each image, identifies the smallest set of (rotation, translation)
    samples whose cumulative posterior weight is strictly greater than
    ``adaptive_fraction`` of total, matching RELION's
    ``frac_weight > adaptive_fraction * exp_sum_weight`` check.
    Caps at max_significants per image.

    Parameters
    ----------
    weights_flat : jnp.ndarray, shape (n_images, n_rot * n_trans)
        Posterior weights (probabilities) for each image, flattened over
        the rotation x translation grid.  Must sum to ~1.0 per image.
    adaptive_fraction : float
        Fraction of total weight to keep (default 0.999 = 99.9%).
    max_significants : int
        Maximum number of significant samples per image. Values ``<= 0``
        disable the cap, matching RELION's ``_rlnMaximumSignificantPoses=-1``.

    Returns
    -------
    mask : jnp.ndarray, shape (n_images, n_rot * n_trans), dtype bool
        True for significant samples.
    n_significant : jnp.ndarray, shape (n_images,), dtype int32
        Number of significant samples per image after expanding cutoff ties.
    cutoff_count : jnp.ndarray, shape (n_images,), dtype int32, optional
        Pre-tie cutoff rank, returned only when ``return_cutoff_count=True``.
        This is the count RELION serializes as ``rlnNrOfSignificantSamples``;
        the threshold-expanded mask remains the support used by pass 2.
    """
    n_images, _ = weights_flat.shape

    # Sort descending per image. RELION only adds strictly positive weights to
    # its sorted significant-pose list; zero-probability samples must not become
    # significant if the threshold falls through to the tail.
    sorted_w = jnp.sort(weights_flat, axis=-1)[:, ::-1]
    cumsum = jnp.cumsum(sorted_w, axis=-1)
    total = weights_flat.sum(axis=-1, keepdims=True)
    positive_counts = jnp.sum(weights_flat > 0.0, axis=-1)
    last_positive_idx = jnp.maximum(positive_counts - 1, 0)

    # Fraction of total weight accumulated so far
    frac = cumsum / jnp.maximum(total, 1e-30)

    # Find the index where we first strictly exceed adaptive_fraction. RELION
    # uses `>` rather than `>=`; if no value strictly crosses the target, the
    # loop finishes at the smallest nonzero weight.
    crosses = frac > adaptive_fraction
    threshold_idx = jnp.where(
        jnp.any(crosses, axis=-1),
        jnp.argmax(crosses, axis=-1),
        last_positive_idx,
    )
    threshold_idx = jnp.minimum(threshold_idx, last_positive_idx)

    # RELION treats maximum_significants <= 0 as "no cap".
    if max_significants is not None and int(max_significants) > 0:
        threshold_idx = jnp.minimum(threshold_idx, int(max_significants) - 1)

    # Get the threshold value: the weight at the threshold index
    threshold_val = sorted_w[jnp.arange(n_images), threshold_idx]

    # Mask: keep all positive samples with weight >= threshold. The positive
    # guard matches RELION's nonzero sorted list while preserving threshold ties.
    mask = (weights_flat > 0.0) & (weights_flat >= threshold_val[:, None])

    # Count significant samples per image
    n_significant = jnp.sum(mask, axis=-1).astype(jnp.int32)
    cutoff_count = jnp.minimum(threshold_idx + 1, positive_counts).astype(jnp.int32)

    if return_cutoff_count:
        return mask, n_significant, cutoff_count
    return mask, n_significant


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _find_significant_mask_topk(
    weights_flat,
    adaptive_fraction=0.999,
    max_significants=500,
    topk=64,
    return_cutoff_count=False,
):
    """Fast exact significance thresholding when the cutoff lives in the top-k.

    This mirrors RELION semantics exactly when the significant-weight threshold
    is determined by one of the top-k samples. If not, the caller must fall
    back to the full-sort implementation.
    """

    n_images, _ = weights_flat.shape
    top_weights, _ = jax.lax.top_k(weights_flat, topk)
    cumsum = jnp.cumsum(top_weights, axis=-1)
    total = weights_flat.sum(axis=-1, keepdims=True)
    positive_counts = jnp.sum(weights_flat > 0.0, axis=-1)
    positive_counts_in_top = jnp.sum(top_weights > 0.0, axis=-1)
    last_top_positive_idx = jnp.maximum(positive_counts_in_top - 1, 0)
    frac = cumsum / jnp.maximum(total, 1e-30)
    crosses = frac > adaptive_fraction
    threshold_idx = jnp.where(
        jnp.any(crosses, axis=-1),
        jnp.argmax(crosses, axis=-1),
        last_top_positive_idx,
    )
    threshold_idx = jnp.minimum(threshold_idx, last_top_positive_idx)

    if max_significants is not None and int(max_significants) > 0:
        threshold_idx = jnp.minimum(threshold_idx, int(max_significants) - 1)
        topk_covers_threshold = (
            jnp.any(crosses, axis=-1)
            | jnp.full((n_images,), int(max_significants) <= int(topk), dtype=bool)
            | (positive_counts <= int(topk))
        )
    else:
        topk_covers_threshold = (
            jnp.any(crosses, axis=-1)
            | (positive_counts <= int(topk))
            | jnp.full(
                (n_images,),
                int(topk) == int(weights_flat.shape[-1]),
                dtype=bool,
            )
        )

    threshold_val = top_weights[jnp.arange(n_images), threshold_idx]
    mask = (weights_flat > 0.0) & (weights_flat >= threshold_val[:, None])
    n_significant = jnp.sum(mask, axis=-1).astype(jnp.int32)
    cutoff_count = jnp.minimum(threshold_idx + 1, positive_counts).astype(jnp.int32)
    if return_cutoff_count:
        return mask, n_significant, topk_covers_threshold, cutoff_count
    return mask, n_significant, topk_covers_threshold


def find_significant_mask(
    weights_flat,
    adaptive_fraction=0.999,
    max_significants=500,
    *,
    return_cutoff_count=False,
):
    """Find significant orientation x translation pairs per image.

    Uses an exact top-k threshold when possible and falls back to the original
    full-sort path if the significance cutoff lies beyond the fast top-k band.
    """

    n_samples = int(weights_flat.shape[-1])
    topk = min(_FAST_SIGNIFICANCE_TOPK, n_samples)
    if topk <= 0:
        return _find_significant_mask_full_sort(
            weights_flat,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            return_cutoff_count=return_cutoff_count,
        )

    fast_result = _find_significant_mask_topk(
        weights_flat,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        topk=topk,
        return_cutoff_count=return_cutoff_count,
    )
    if return_cutoff_count:
        fast_mask, fast_n_significant, topk_covers_threshold, fast_cutoff_count = fast_result
    else:
        fast_mask, fast_n_significant, topk_covers_threshold = fast_result
    if bool(np.all(np.asarray(topk_covers_threshold))):
        if return_cutoff_count:
            return fast_mask, fast_n_significant, fast_cutoff_count
        return fast_mask, fast_n_significant
    return _find_significant_mask_full_sort(
        weights_flat,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        return_cutoff_count=return_cutoff_count,
    )


def find_significant_rotations(
    weights_flat,
    n_rot,
    n_trans,
    adaptive_fraction=0.999,
    max_significants=500,
    *,
    return_cutoff_count=False,
):
    """Find significant coarse rotations per image from (rot x trans) weights.

    This extracts the unique rotation indices that have at least one
    significant (rotation, translation) pair, which is what we need
    for generating oversampled children in pass 2.

    Parameters
    ----------
    weights_flat : jnp.ndarray, shape (n_images, n_rot * n_trans)
        Posterior weights.
    n_rot : int
        Number of rotations in the coarse grid.
    n_trans : int
        Number of translations.
    adaptive_fraction : float
        Fraction of total weight to keep.
    max_significants : int
        Maximum significant samples (rot x trans).

    Returns
    -------
    sig_mask : jnp.ndarray, shape (n_images, n_rot * n_trans), dtype bool
        Significance mask over the full rot x trans grid.
    sig_rot_mask : jnp.ndarray, shape (n_images, n_rot), dtype bool
        True for rotations that have at least one significant translation.
    n_significant : jnp.ndarray, shape (n_images,), dtype int32
        Total significant (rot x trans) samples per image.
    cutoff_count : jnp.ndarray, shape (n_images,), dtype int32, optional
        Pre-tie cutoff rank, returned only when ``return_cutoff_count=True``.
    """
    significance_result = find_significant_mask(
        weights_flat,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        return_cutoff_count=return_cutoff_count,
    )
    if return_cutoff_count:
        sig_mask, n_significant, cutoff_count = significance_result
    else:
        sig_mask, n_significant = significance_result

    # Reshape to (n_images, n_rot, n_trans) and check if any translation
    # is significant for each rotation
    sig_2d = sig_mask.reshape(-1, n_rot, n_trans)
    sig_rot_mask = jnp.any(sig_2d, axis=-1)  # (n_images, n_rot)

    if return_cutoff_count:
        return sig_mask, sig_rot_mask, n_significant, cutoff_count
    return sig_mask, sig_rot_mask, n_significant


# ---------------------------------------------------------------------------
# Pass 2: sparse oversampled evaluation
# ---------------------------------------------------------------------------


def compute_pass2_stats(
    experiment_dataset,
    volume,
    mean_variance,
    noise_variance,
    coarse_rotations,
    translations,
    sig_rot_mask,
    nside_level,
    disc_type,
    oversampling_order=1,
    current_size=None,
    image_batch_size=500,
    max_union_pixels=5000,
    translation_step=None,
    *,
    rotation_log_prior=None,
    score_with_masked_images=False,
    return_stats=False,
    return_rotation_indices=False,
    translation_log_prior=None,
    accumulate_noise=False,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    image_corrections=None,
    scale_corrections=None,
    group_ids=None,
    image_pre_shifts=None,
    use_float64_scoring=False,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation=0.0,
    translation_prior_centers=None,
):
    """Pass 2: evaluate oversampled children of significant coarse rotations.

    For each image, generates child rotations (4x per parent at next healpix
    level) for its significant coarse rotations, then evaluates the E-step
    and accumulates M-step statistics.

    This is a simplified implementation that unions all significant rotations
    across the batch and evaluates them densely.  This is correct but may
    evaluate extra rotations for some images.  A per-image sparse approach
    would be more efficient but is deferred to a later optimization pass.

    Parameters
    ----------
    experiment_dataset : dataset object
        One half-set dataset.
    volume : jnp.ndarray, shape (volume_size,)
        Current volume estimate.
    mean_variance : jnp.ndarray
        Signal prior (tau^2).
    noise_variance : jnp.ndarray
        Per-pixel noise variance.
    coarse_rotations : np.ndarray, shape (n_coarse_rot, 3, 3)
        Coarse rotation grid (base healpix level).
    translations : jnp.ndarray, shape (n_trans, 2)
        Translation grid.
    sig_rot_mask : jnp.ndarray, shape (n_images, n_coarse_rot), dtype bool
        Per-image significant rotation mask from pass 1.
    nside_level : int
        HEALPix level of the coarse rotation grid.
    disc_type : str
        Discretization type.
    oversampling_order : int
        Number of healpix subdivision levels (1 = 4x children).
    current_size : int or None
        Fourier window size for pass 2 (can be larger than pass 1).
    image_batch_size : int
        Images per GPU batch.
    max_union_pixels : int
        Maximum number of unique parent HEALPix pixels allowed in the
        union of significant rotations.  If the union exceeds this cap,
        pass 2 is skipped and ``None`` is returned to signal the caller
        to fall back to pass-1-only mode.  Default 200, which yields at
        most 200 * 4^oversampling_order oversampled children (800 for
        oversampling_order=1).  This prevents pass 2 from becoming more
        expensive than pass 1 when the posterior is nearly flat.
    translation_step : float or None
        Coarse translation-grid step in pixels. When provided, pass 2
        oversamples translations by subdividing each coarse translation cell.
        If None, infer the step from ``translations``.
    return_stats : bool
        When True, also return the per-image E-step statistics from the
        oversampled dense pass.

    Returns
    -------
    Ft_y : jnp.ndarray, shape (volume_size,), or None
        Accumulated weighted image sums.  ``None`` if pass 2 was skipped.
    Ft_ctf : jnp.ndarray, shape (volume_size,), or None
        Accumulated CTF^2 weights.  ``None`` if pass 2 was skipped.
    hard_assignments : np.ndarray, shape (n_images,), or None
        Best (rotation_idx * n_trans + trans_idx) indices into the
        OVERSAMPLED grid.  ``None`` if pass 2 was skipped.
    oversampled_rotations : np.ndarray, shape (n_oversampled, 3, 3), or None
        The oversampled rotation matrices used.  ``None`` if pass 2 was
        skipped.
    relion_stats : RelionStats or None
        Per-image E-step statistics from the oversampled dense pass.
        Only returned when ``return_stats=True``.
    oversampled_rotation_indices : np.ndarray or None
        Nearest full-grid indices of the oversampled orientations on the
        fine grid. RELION's oversampled psi children are midpoint samples,
        so these indices are only an approximation for downstream tracking.
        Only returned when ``return_rotation_indices=True``.
    """
    from recovar.em.sampling import (
        get_oversampled_rotation_grid_from_samples,
        get_oversampled_translation_grid,
    )

    from ..em_engine import run_em

    n_images = experiment_dataset.n_units
    n_coarse_rot = coarse_rotations.shape[0]

    # Union of all significant rotation indices across all images
    # This is conservative but ensures we evaluate all needed rotations
    # Support both per-image mask (n_images, n_rot) and global mask (n_rot,)
    sig_mask_np = np.asarray(sig_rot_mask)
    if sig_mask_np.ndim == 2:
        sig_rot_any = np.any(sig_mask_np, axis=0)  # (n_coarse_rot,)
    else:
        sig_rot_any = sig_mask_np.astype(bool)  # already (n_coarse_rot,)
    sig_rot_indices = np.where(sig_rot_any)[0]

    if len(sig_rot_indices) == 0:
        logger.warning("No significant rotations found; skipping pass 2")
        recon_vol_size = experiment_dataset.volume_size * reconstruction_padding_factor**3
        Ft_y = jnp.zeros(recon_vol_size, dtype=experiment_dataset.dtype)
        Ft_ctf = jnp.zeros(recon_vol_size, dtype=experiment_dataset.dtype)
        ha = np.zeros(n_images, dtype=np.int32)
        empty_indices = np.empty((0,), dtype=np.int64)
        if return_stats:
            zero_stats = make_relion_stats(
                log_evidence_per_image=jnp.full(n_images, -jnp.inf, dtype=jnp.float32),
                best_log_score_per_image=jnp.full(n_images, -jnp.inf, dtype=jnp.float32),
                max_posterior_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(n_coarse_rot, dtype=jnp.float32),
            )
            if return_rotation_indices:
                return Ft_y, Ft_ctf, ha, coarse_rotations[:0], empty_indices, zero_stats
            return Ft_y, Ft_ctf, ha, coarse_rotations[:0], zero_stats
        if return_rotation_indices:
            return Ft_y, Ft_ctf, ha, coarse_rotations[:0], empty_indices
        return Ft_y, Ft_ctf, ha, coarse_rotations[:0]

    angle_res = 360 / (6 * 2**nside_level)
    n_in_planes = int(np.round(360 / angle_res))

    # Preserve the full coarse orientation sample identity (direction + psi),
    # not just the HEALPix pixel, so pass 2 generates the exact RELION-style
    # child orientations for each significant coarse sample.
    max_union_rotations = max_union_pixels * n_in_planes
    if len(sig_rot_indices) > max_union_rotations:
        n_oversampled_would_be = len(sig_rot_indices) * (8**oversampling_order)
        logger.warning(
            "Pass 2: union has %d significant coarse rotations (> cap %d), "
            "which would produce %d oversampled rotations. "
            "Falling back to pass-1-only mode.",
            len(sig_rot_indices),
            max_union_rotations,
            n_oversampled_would_be,
        )
        if return_stats:
            if return_rotation_indices:
                return None, None, None, None, None, None
            return None, None, None, None, None
        if return_rotation_indices:
            return None, None, None, None, None
        return None, None, None, None

    logger.info(
        "Pass 2: %d significant coarse orientations -> generating %d oversampled child orientations (order=%d)",
        len(sig_rot_indices),
        len(sig_rot_indices) * (8**oversampling_order),
        oversampling_order,
    )

    # Generate oversampled children from the exact coarse orientation samples.
    oversampled_outputs = get_oversampled_rotation_grid_from_samples(
        sig_rot_indices,
        nside_level,
        oversampling_order=oversampling_order,
        random_perturbation=random_perturbation,
        return_rotation_indices=return_rotation_indices,
    )
    if return_rotation_indices:
        oversampled_rots, parent_map, oversampled_rot_indices = oversampled_outputs
    else:
        oversampled_rots, parent_map = oversampled_outputs
    oversampled_rots = np.asarray(oversampled_rots, dtype=np.float32)
    parent_map = np.asarray(parent_map, dtype=np.int32)
    oversampled_rotation_log_prior = None
    if rotation_log_prior is not None:
        rotation_log_prior = np.asarray(rotation_log_prior, dtype=np.float32)
        oversampled_rotation_log_prior = rotation_log_prior[sig_rot_indices][parent_map]

    logger.info(
        "Pass 2: %d oversampled rotations (from %d parent coarse orientations)",
        len(oversampled_rots),
        len(sig_rot_indices),
    )

    translations_np = np.asarray(translations, dtype=np.float32)
    if translation_step is None:
        unique_vals = np.unique(translations_np)
        diffs = np.diff(np.sort(unique_vals))
        diffs = diffs[diffs > 1e-6]
        translation_step = float(diffs.min()) if diffs.size else 1.0
    oversampled_translations, oversampled_translation_parent = get_oversampled_translation_grid(
        translations_np,
        translation_step,
        oversampling_order=oversampling_order,
    )
    oversampled_translations = np.asarray(oversampled_translations, dtype=np.float32)
    oversampled_translation_prior = map_translation_log_prior_to_fine_grid(
        translation_log_prior,
        oversampled_translation_parent,
    )
    logger.info(
        "Pass 2: %d oversampled translations (from %d coarse translations)",
        len(oversampled_translations),
        len(translations_np),
    )

    # Run a full dense E+M at the oversampled grid
    # This is correct: we evaluate all oversampled rotations for all images.
    # The significance pruning's benefit is that len(oversampled_rots) <<
    # len(coarse_rotations) * 4^oversampling_order.
    run_em_outputs = run_em(
        experiment_dataset,
        volume,
        mean_variance,
        noise_variance,
        oversampled_rots,
        oversampled_translations,
        disc_type,
        image_batch_size=image_batch_size,
        rotation_block_size=min(5000, len(oversampled_rots)),
        current_size=current_size,
        rotation_log_prior=oversampled_rotation_log_prior,
        translation_log_prior=oversampled_translation_prior,
        score_with_masked_images=score_with_masked_images,
        return_stats=return_stats,
        accumulate_noise=accumulate_noise,
        half_spectrum_scoring=half_spectrum_scoring,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        use_float64_scoring=use_float64_scoring,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
    )

    # Unpack: run_em returns (mean, ha, Ft_y, Ft_ctf, [relion_stats], [noise_stats])
    # depending on return_stats and accumulate_noise flags.
    noise_stats = None
    if return_stats and accumulate_noise:
        _, ha, Ft_y, Ft_ctf, relion_stats, noise_stats = run_em_outputs
    elif return_stats:
        _, ha, Ft_y, Ft_ctf, relion_stats = run_em_outputs
    elif accumulate_noise:
        _, ha, Ft_y, Ft_ctf, noise_stats = run_em_outputs
    else:
        _, ha, Ft_y, Ft_ctf = run_em_outputs

    if return_stats:
        coarse_rotation_sums = np.zeros(n_coarse_rot, dtype=np.float64)
        np.add.at(
            coarse_rotation_sums,
            sig_rot_indices[parent_map],
            np.asarray(relion_stats.rotation_posterior_sums, dtype=np.float64),
        )
        relion_stats = make_relion_stats(
            log_evidence_per_image=relion_stats.log_evidence_per_image,
            best_log_score_per_image=relion_stats.best_log_score_per_image,
            max_posterior_per_image=relion_stats.max_posterior_per_image,
            rotation_posterior_sums=coarse_rotation_sums,
        )
        result = [Ft_y, Ft_ctf, ha, oversampled_rots]
        if return_rotation_indices:
            result.append(oversampled_rot_indices)
        result.append(relion_stats)
        if accumulate_noise:
            result.append(noise_stats)
        return tuple(result)

    result = [Ft_y, Ft_ctf, ha, oversampled_rots]
    if return_rotation_indices:
        result.append(oversampled_rot_indices)
    if accumulate_noise:
        result.append(noise_stats)
    return tuple(result)


def compute_pass2_stats_sparse(
    experiment_dataset,
    volume,
    mean_variance,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    oversampling_order=1,
    current_size=None,
    reconstruction_current_size=None,
    translation_step=None,
    *,
    rotation_log_prior=None,
    score_with_masked_images=False,
    return_stats=False,
    translation_log_prior=None,
    accumulate_noise=False,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    image_corrections=None,
    scale_corrections=None,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    image_pre_shifts=None,
    use_float64_scoring=False,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation=0.0,
    translation_prior_centers=None,
    normalization_log_z=None,
    normalization_other_score_log_z=None,
    normalization_score_mode=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_x_half_mstep=False,
    relion_fine_mstep_prune=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_fine_diff2_fused_ffi=False,
    relion_f32_fine_posterior=False,
    relion_exact_fine_normalized_cc=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    use_perimage_reference=False,
    bpref_device_signature_active: bool = False,
    bpref_class_index: int = 0,
    include_unweighted_norm_high_shell: bool = True,
    preserve_bpref_particle_order: bool = False,
    source_faithful_spectrum_norm: bool = False,
):
    """Exact sparse pass 2 over per-image significant coarse samples.

    This matches RELION's pass-2 structure more closely than the dense-union
    approximation: each image only evaluates oversampled children of the
    coarse (rotation, translation) pairs that survived pass 1.

    Implementation note
    -------------------
    By default the bucketed batched implementation in
    :mod:`recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed`
    is used: images are grouped by their oversampled rotation count
    (quantized) and evaluated as one GPU call per bucket, which keeps
    the number of distinct XLA shapes bounded across iterations.

    The original per-image Python loop is preserved as
    :func:`_compute_pass2_stats_sparse_perimage_reference` for testing
    parity; it can be selected by setting ``use_perimage_reference=True``.
    The two paths must produce identical outputs (modulo float rounding).

    ``relion_exact_fine_gaussian`` selects the float32 scorer that follows
    RELION's fine-search diff2/minimum ordering. Float64 diagnostics retain
    the historical algebraic scorer so they do not silently downcast.
    """
    has_external_score_normalization = (
        normalization_log_z is not None or normalization_other_score_log_z is not None
    )
    if has_external_score_normalization and normalization_score_mode is None:
        raise ValueError(
            "external sparse pass-2 score normalization requires normalization_score_mode; "
            "Gaussian logZ is absolute while normalized-CC logZ is centered"
        )
    if normalization_score_mode is not None and not has_external_score_normalization:
        raise ValueError("normalization_score_mode requires an external score normalization")
    if (
        normalization_score_mode is not None
        and normalization_score_mode != relion_firstiter_score_mode
    ):
        raise ValueError(
            "external score normalization mode does not match this pass: "
            f"external={normalization_score_mode!r}, pass={relion_firstiter_score_mode!r}"
        )
    if use_perimage_reference and (return_score_log_z or return_score_log_z_only):
        raise NotImplementedError("score-logZ returns are only implemented for the bucketed sparse pass-2 path")
    if (
        use_perimage_reference
        and relion_exact_fine_gaussian
        and relion_firstiter_score_mode == "gaussian"
        and not use_float64_scoring
    ):
        raise NotImplementedError(
            "exact RELION fine Gaussian scoring requires the bucketed sparse pass-2 path"
        )
    if use_perimage_reference and group_ids is not None:
        logger.warning(
            "Sparse per-image reference pass-2 does not accumulate native group-scale correction stats; "
            "use the bucketed sparse pass-2 path for native scale updates."
        )
    if use_perimage_reference and fine_mstep_rotations_override is not None:
        raise NotImplementedError(
            "fine_mstep_rotations_override is only implemented for the bucketed sparse pass-2 path",
        )
    full_grid_reference = (
        all(samples is None for samples in significant_sample_indices)
        and not return_score_log_z
        and not return_score_log_z_only
        and normalization_log_z is None
        and normalization_other_score_log_z is None
        and normalization_score_mode is None
        and group_ids is None
        and scale_correction_group_count is None
        and scale_correction_data_vs_prior is None
        and fine_rotations_override is None
        and fine_mstep_rotations_override is None
        and fine_rotation_parent_override is None
        and fine_translations_override is None
        and fine_translation_parent_override is None
        and not relion_x_half_mstep
        and not relion_fine_mstep_prune
        and relion_firstiter_score_mode == "gaussian"
        and not relion_firstiter_winner_take_all
        and include_unweighted_norm_high_shell
        and not preserve_bpref_particle_order
        and reconstruction_current_size is None
        and not (
            relion_exact_fine_gaussian
            and not use_float64_scoring
        )
    )
    if not use_perimage_reference and not full_grid_reference:
        from .sparse_pass2_bucketed import compute_pass2_stats_sparse_bucketed

        return compute_pass2_stats_sparse_bucketed(
            experiment_dataset,
            volume,
            mean_variance,
            noise_variance,
            translations,
            significant_sample_indices,
            nside_level,
            disc_type,
            oversampling_order=oversampling_order,
            current_size=current_size,
            reconstruction_current_size=reconstruction_current_size,
            translation_step=translation_step,
            rotation_log_prior=rotation_log_prior,
            score_with_masked_images=score_with_masked_images,
            return_stats=return_stats,
            translation_log_prior=translation_log_prior,
            accumulate_noise=accumulate_noise,
            half_spectrum_scoring=half_spectrum_scoring,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            group_ids=group_ids,
            scale_correction_group_count=scale_correction_group_count,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior,
            image_pre_shifts=image_pre_shifts,
            use_float64_scoring=use_float64_scoring,
            translation_prior_centers=translation_prior_centers,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            random_perturbation=random_perturbation,
            normalization_log_z=normalization_log_z,
            normalization_other_score_log_z=normalization_other_score_log_z,
            normalization_score_mode=normalization_score_mode,
            return_score_log_z=return_score_log_z,
            return_score_log_z_only=return_score_log_z_only,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            fine_rotations_override=fine_rotations_override,
            fine_mstep_rotations_override=fine_mstep_rotations_override,
            fine_rotation_parent_override=fine_rotation_parent_override,
            fine_translations_override=fine_translations_override,
            fine_translation_parent_override=fine_translation_parent_override,
            relion_half_volume_mstep=relion_half_volume_mstep,
            relion_x_half_mstep=relion_x_half_mstep,
            relion_fine_mstep_prune=relion_fine_mstep_prune,
            relion_firstiter_score_mode=relion_firstiter_score_mode,
            relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
            relion_exact_fine_gaussian=relion_exact_fine_gaussian,
            relion_fine_diff2_fused_ffi=relion_fine_diff2_fused_ffi,
            relion_f32_fine_posterior=relion_f32_fine_posterior,
            relion_exact_fine_normalized_cc=relion_exact_fine_normalized_cc,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            adaptive_fraction=adaptive_fraction,
            bpref_device_signature_active=bpref_device_signature_active,
            bpref_class_index=bpref_class_index,
            include_unweighted_norm_high_shell=include_unweighted_norm_high_shell,
            preserve_bpref_particle_order=preserve_bpref_particle_order,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        )

    if relion_projector_half is not None:
        raise NotImplementedError("RELION projector sparse pass-2 requires the bucketed implementation")
    if reconstruction_current_size is not None:
        raise NotImplementedError(
            "separate score/reconstruction current sizes require the bucketed sparse pass-2 path",
        )

    return _compute_pass2_stats_sparse_perimage_reference(
        experiment_dataset,
        volume,
        mean_variance,
        noise_variance,
        translations,
        significant_sample_indices,
        nside_level,
        disc_type,
        oversampling_order=oversampling_order,
        current_size=current_size,
        translation_step=translation_step,
        rotation_log_prior=rotation_log_prior,
        score_with_masked_images=score_with_masked_images,
        return_stats=return_stats,
        translation_log_prior=translation_log_prior,
        accumulate_noise=accumulate_noise,
        half_spectrum_scoring=half_spectrum_scoring,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        group_ids=group_ids,
        image_pre_shifts=image_pre_shifts,
        translation_prior_centers=translation_prior_centers,
        use_float64_scoring=use_float64_scoring,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
        random_perturbation=random_perturbation,
        normalization_log_z=normalization_log_z,
        normalization_other_score_log_z=normalization_other_score_log_z,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        relion_half_volume_mstep=relion_half_volume_mstep,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
    )


def _compute_pass2_stats_sparse_perimage_reference(
    experiment_dataset,
    volume,
    mean_variance,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    oversampling_order=1,
    current_size=None,
    translation_step=None,
    *,
    rotation_log_prior=None,
    score_with_masked_images=False,
    return_stats=False,
    translation_log_prior=None,
    accumulate_noise=False,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    image_corrections=None,
    scale_corrections=None,
    group_ids=None,
    image_pre_shifts=None,
    translation_prior_centers=None,
    use_float64_scoring=False,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation=0.0,
    normalization_log_z=None,
    normalization_other_score_log_z=None,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    relion_half_volume_mstep=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
):
    """Per-image reference implementation for sparse pass-2.

    Runs ``run_em`` once per image with ``image_batch_size=1`` and the
    image's oversampled rotation/translation grid.  This is correct but
    causes a JIT recompile per image when the rotation count varies, so
    it is only kept as a numerical reference for testing — production
    paths should use :func:`compute_pass2_stats_sparse` with the default
    bucketed implementation.
    """
    from recovar.em.sampling import (
        get_oversampled_rotation_grid_from_samples,
        get_oversampled_translation_grid,
        rotation_grid_size,
    )

    from ..em_engine import run_em
    if normalization_log_z is not None:
        raise NotImplementedError(
            "normalization_log_z is only implemented for the bucketed sparse pass-2 path",
        )
    if normalization_other_score_log_z is not None:
        raise NotImplementedError(
            "normalization_other_score_log_z is only implemented for the bucketed sparse pass-2 path",
        )

    n_images = experiment_dataset.n_units
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = rotation_grid_size(nside_level)
    recon_vol_size = experiment_dataset.volume_size * reconstruction_padding_factor**3
    Ft_y_total = jnp.zeros(recon_vol_size, dtype=experiment_dataset.dtype)
    Ft_ctf_total = jnp.zeros(recon_vol_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    best_rotations = np.empty((n_images, 3, 3), dtype=np.float32)
    best_rotation_indices = np.empty(n_images, dtype=np.int64)

    log_evidence = None
    best_log_score = None
    max_posterior = None
    rotation_posterior_sums = None
    if return_stats:
        # K-class assignment depends on small inter-class score deltas after
        # adding a large image-power offset. Keep these in float64 like dense
        # run_em.
        log_evidence = np.empty(n_images, dtype=np.float64)
        best_log_score = np.empty(n_images, dtype=np.float64)
        max_posterior = np.empty(n_images, dtype=np.float32)
        rotation_posterior_sums = np.zeros(n_coarse_rot, dtype=np.float64)

    # Noise accumulators (additive across per-image calls)
    noise_wsum_total = None
    noise_img_power_total = None
    noise_sumw_total = 0.0
    noise_sigma2_offset_total = 0.0
    if accumulate_noise:
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        noise_wsum_total = np.zeros(n_shells, dtype=np.float64)
        noise_img_power_total = np.zeros(n_shells, dtype=np.float64)

    translations_np = np.asarray(translations, dtype=np.float32)
    if translation_step is None:
        unique_vals = np.unique(translations_np)
        diffs = np.diff(np.sort(unique_vals))
        diffs = diffs[diffs > 1e-6]
        translation_step = float(diffs.min()) if diffs.size else 1.0
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        translations_np,
        translation_step,
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=np.float32)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_trans = fine_translations.shape[0]
    fine_translation_prior = map_translation_log_prior_to_fine_grid(
        translation_log_prior,
        fine_translation_parent,
    )
    local_rot_counts = []
    valid_candidate_counts = []

    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if sig_samples is None:
            coarse_rot = np.arange(n_coarse_rot, dtype=np.int32)
            coarse_trans = None
            unique_rot = coarse_rot
            use_full_candidate_mask = True
        else:
            sig_samples = np.asarray(sig_samples, dtype=np.int32).reshape(-1)
            if sig_samples.size == 0:
                raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")
            coarse_rot = sig_samples // n_coarse_trans
            coarse_trans = sig_samples % n_coarse_trans
            unique_rot = np.unique(coarse_rot)
            use_full_candidate_mask = False

        if unique_rot.size == 0:
            raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")

        oversampled_rots, parent_map, oversampled_rot_indices = get_oversampled_rotation_grid_from_samples(
            unique_rot,
            nside_level,
            oversampling_order=oversampling_order,
            random_perturbation=random_perturbation,
            return_rotation_indices=True,
        )
        oversampled_rots = np.asarray(oversampled_rots, dtype=np.float32)
        parent_map = np.asarray(parent_map, dtype=np.int32)
        oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int64)
        local_rotation_log_prior = None
        if rotation_log_prior is not None:
            rotation_log_prior = np.asarray(rotation_log_prior, dtype=np.float32)
            local_rotation_log_prior = rotation_log_prior[unique_rot][parent_map]

        if use_full_candidate_mask:
            candidate_mask = np.ones(
                (oversampled_rots.shape[0], n_fine_trans),
                dtype=bool,
            )
        else:
            sig_trans_by_rot = {
                int(rot_idx): set(coarse_trans[coarse_rot == rot_idx].tolist()) for rot_idx in unique_rot
            }
            candidate_mask = np.zeros(
                (oversampled_rots.shape[0], n_fine_trans),
                dtype=bool,
            )
            for parent_local_idx, coarse_rot_idx in enumerate(unique_rot):
                row_mask = parent_map == parent_local_idx
                valid_coarse_trans = sig_trans_by_rot[int(coarse_rot_idx)]
                col_mask = np.isin(fine_translation_parent, list(valid_coarse_trans))
                candidate_mask[row_mask, :] = col_mask[None, :]

        if not np.any(candidate_mask):
            raise ValueError(f"Image {image_idx} has no valid sparse pass-2 candidates after oversampling")

        local_rot_counts.append(int(oversampled_rots.shape[0]))
        valid_candidate_counts.append(int(candidate_mask.sum()))

        run_em_outputs = run_em(
            experiment_dataset,
            volume,
            mean_variance,
            noise_variance,
            oversampled_rots,
            fine_translations,
            disc_type,
            image_batch_size=1,
            rotation_block_size=min(5000, max(1, oversampled_rots.shape[0])),
            current_size=current_size,
            rotation_log_prior=local_rotation_log_prior,
            translation_log_prior=(
                None
                if fine_translation_prior is None
                else np.asarray(fine_translation_prior[image_idx : image_idx + 1], dtype=np.float32)
                if np.asarray(fine_translation_prior).ndim == 2
                else fine_translation_prior
            ),
            image_indices=np.array([image_idx], dtype=np.int32),
            rotation_translation_mask=candidate_mask,
            score_with_masked_images=score_with_masked_images,
            return_stats=return_stats,
            accumulate_noise=accumulate_noise,
            half_spectrum_scoring=half_spectrum_scoring,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            translation_prior_centers=(
                None
                if translation_prior_centers is None
                else np.asarray(translation_prior_centers[image_idx : image_idx + 1], dtype=np.float32)
                if np.asarray(translation_prior_centers).ndim == 2
                else np.asarray(translation_prior_centers, dtype=np.float32)
            ),
            use_float64_scoring=use_float64_scoring,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            relion_half_volume_mstep=relion_half_volume_mstep,
            relion_firstiter_score_mode=relion_firstiter_score_mode,
            relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
        )

        # Unpack return based on flags
        noise_stats_i = None
        if return_stats and accumulate_noise:
            _, ha_i, Ft_y_i, Ft_ctf_i, stats_i, noise_stats_i = run_em_outputs
        elif return_stats:
            _, ha_i, Ft_y_i, Ft_ctf_i, stats_i = run_em_outputs
        elif accumulate_noise:
            _, ha_i, Ft_y_i, Ft_ctf_i, noise_stats_i = run_em_outputs
        else:
            _, ha_i, Ft_y_i, Ft_ctf_i = run_em_outputs

        if return_stats:
            log_evidence[image_idx] = float(np.asarray(stats_i.log_evidence_per_image)[0])
            best_log_score[image_idx] = float(np.asarray(stats_i.best_log_score_per_image)[0])
            max_posterior[image_idx] = float(np.asarray(stats_i.max_posterior_per_image)[0])
            np.add.at(
                rotation_posterior_sums,
                unique_rot[parent_map],
                np.asarray(stats_i.rotation_posterior_sums, dtype=np.float64),
            )

        if accumulate_noise and noise_stats_i is not None:
            noise_wsum_total += np.asarray(noise_stats_i.wsum_sigma2_noise, dtype=np.float64)
            noise_img_power_total += np.asarray(noise_stats_i.wsum_img_power, dtype=np.float64)
            noise_sigma2_offset_total += float(getattr(noise_stats_i, "wsum_sigma2_offset", 0.0))
            noise_sumw_total += noise_stats_i.sumw

        Ft_y_total = Ft_y_total + Ft_y_i
        Ft_ctf_total = Ft_ctf_total + Ft_ctf_i

        best_idx = int(np.asarray(ha_i)[0])
        rot_idx = best_idx // n_fine_trans
        trans_idx = best_idx % n_fine_trans
        hard_assignment[image_idx] = best_idx
        best_rotations[image_idx] = oversampled_rots[rot_idx]
        best_rotation_indices[image_idx] = oversampled_rot_indices[rot_idx]

    logger.info(
        "Sparse pass 2: median local rotations=%d, mean local rotations=%.1f, median valid candidates/image=%d",
        int(np.median(local_rot_counts)) if local_rot_counts else 0,
        float(np.mean(local_rot_counts)) if local_rot_counts else 0.0,
        int(np.median(valid_candidate_counts)) if valid_candidate_counts else 0,
    )

    best_translations = fine_translations[hard_assignment % n_fine_trans]

    merged_noise_stats = None
    if accumulate_noise:
        merged_noise_stats = make_noise_stats(
            wsum_sigma2_noise=noise_wsum_total,
            wsum_img_power=noise_img_power_total,
            wsum_sigma2_offset=noise_sigma2_offset_total,
            sumw=noise_sumw_total,
        )

    if return_stats:
        relion_stats = make_relion_stats(
            log_evidence_per_image=log_evidence,
            best_log_score_per_image=best_log_score,
            max_posterior_per_image=max_posterior,
            rotation_posterior_sums=rotation_posterior_sums,
        )
        result = (
            Ft_y_total,
            Ft_ctf_total,
            hard_assignment,
            best_rotations,
            best_translations,
            best_rotation_indices,
            relion_stats,
        )
        if accumulate_noise:
            result = result + (merged_noise_stats,)
        return result

    result = (
        Ft_y_total,
        Ft_ctf_total,
        hard_assignment,
        best_rotations,
        best_translations,
        best_rotation_indices,
    )
    if accumulate_noise:
        result = result + (merged_noise_stats,)
    return result
