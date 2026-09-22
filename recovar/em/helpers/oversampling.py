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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

_FAST_SIGNIFICANCE_TOPK = 64


def build_adaptive_pass2_grids(
    coarse_rotations,
    coarse_translations,
    base_translations,
    coarse_healpix_order: int,
    adaptive_oversampling: int,
    translation_step_px: float,
    random_perturbation: float,
    *,
    return_mstep_rotations: bool = False,
    coarse_rotation_ids=None,
    symmetry: str = "C1",
):
    """Build coarse/fine pose grids for ordinary and first-CC adaptive scoring.

    Return coarse rotations/translations, fine rotations/translations, and the
    two fine-to-coarse parent maps. ``return_mstep_rotations`` appends a seventh
    array for reconstruction. Nonpositive oversampling keeps the coarse rotation
    grid and identity parent maps; fine translations still receive perturbation.

    Preserve host-double base translations through subdivision and perturbation.
    Scoring translations and the separate translation-phase source are distinct
    caller inputs. ``coarse_rotation_ids`` identifies a supplied rotation subset
    in the full HEALPix grid when oversampling is enabled.
    """
    from recovar.em.sampling import (
        apply_relion_translation_perturbation,
        get_oversampled_rotation_grid_from_samples,
        get_oversampled_translation_grid,
    )
    coarse_rot_np = np.asarray(coarse_rotations)
    coarse_trans_np = np.asarray(coarse_translations)
    base_translations_f64 = np.asarray(base_translations, dtype=np.float64)
    if int(adaptive_oversampling) <= 0:
        n_rot = int(coarse_rot_np.shape[0])
        n_trans = int(coarse_trans_np.shape[0])
        rot_parent_map = np.arange(n_rot, dtype=np.int64)
        trans_parent_map = np.arange(n_trans, dtype=np.int64)
        fine_translations = apply_relion_translation_perturbation(
            base_translations_f64,
            float(random_perturbation),
            float(translation_step_px),
        )
        outputs = (
            coarse_rot_np,
            coarse_trans_np,
            coarse_rot_np,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
        )
        if return_mstep_rotations:
            return (*outputs, coarse_rot_np.copy())
        return outputs

    adaptive_os = int(adaptive_oversampling)
    all_coarse_rot_indices = (
        np.arange(int(coarse_rot_np.shape[0]), dtype=np.int64)
        if coarse_rotation_ids is None
        else np.asarray(coarse_rotation_ids, dtype=np.int64)
    )
    if all_coarse_rot_indices.shape != (int(coarse_rot_np.shape[0]),):
        raise ValueError(
            "coarse_rotation_ids must identify every supplied coarse rotation exactly once"
        )
    fine_rotation_outputs = get_oversampled_rotation_grid_from_samples(
        all_coarse_rot_indices,
        parent_nside_level=int(coarse_healpix_order),
        oversampling_order=adaptive_os,
        random_perturbation=float(random_perturbation),
        return_mstep_rotations=return_mstep_rotations,
        dtype=coarse_rotations.dtype,
        **({} if symmetry == "C1" else {"symmetry": symmetry}),
    )
    fine_rotations, rot_parent_map = fine_rotation_outputs[:2]
    fine_mstep_rotations = fine_rotation_outputs[2] if return_mstep_rotations else None
    rot_parent_map = np.asarray(rot_parent_map, dtype=np.int64)

    fine_base_translations, trans_parent_map = get_oversampled_translation_grid(
        base_translations_f64,
        float(translation_step_px),
        oversampling_order=adaptive_os,
    )
    fine_translations = apply_relion_translation_perturbation(
        fine_base_translations,
        float(random_perturbation),
        float(translation_step_px),
    )
    trans_parent_map = np.asarray(trans_parent_map, dtype=np.int64)

    outputs = (
        coarse_rot_np,
        coarse_trans_np,
        fine_rotations,
        fine_translations,
        rot_parent_map,
        trans_parent_map,
    )
    if return_mstep_rotations:
        return (*outputs, fine_mstep_rotations)
    return outputs



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


def relion_cuda_f32_coarse_log_weights(
    raw_scores,
    rotation_log_prior,
    translation_log_prior,
):
    """Apply RELION's coarse CUDA prior/min-diff arithmetic in float32.

    ``raw_scores`` is RECOVAR's negative-diff2 table up to an arbitrary
    per-image additive constant.  RELION's CUDA kernel evaluates

    ``orientation_prior + offset_prior + min_diff2 - diff2``

    in that exact left-associative order.  Adding priors to the large absolute
    raw scores and centering afterwards is algebraically equivalent, but can
    erase one-ULP pose margins at the significance cutoff.
    """

    raw = jnp.asarray(raw_scores, dtype=jnp.float32)
    if raw.ndim != 3:
        raise ValueError(f"raw_scores must have shape (batch, rotations, translations), got {raw.shape}")
    rotation_prior = jnp.asarray(rotation_log_prior, dtype=jnp.float32).reshape(-1)
    translation_prior = jnp.asarray(translation_log_prior, dtype=jnp.float32)
    if translation_prior.ndim == 1:
        translation_prior = jnp.broadcast_to(
            translation_prior[None, :],
            (raw.shape[0], translation_prior.shape[0]),
        )
    if translation_prior.ndim != 2:
        raise ValueError("translation_log_prior must be one- or two-dimensional")
    if raw.shape[1:] != (rotation_prior.shape[0], translation_prior.shape[1]):
        raise ValueError(
            "raw-score and prior topology mismatch: "
            f"raw={raw.shape}, rotation={rotation_prior.shape}, translation={translation_prior.shape}",
        )

    finite = jnp.isfinite(raw) & jnp.isfinite(rotation_prior)[None, :, None]
    finite &= jnp.isfinite(translation_prior)[:, None, :]
    raw_best = jnp.max(jnp.where(finite, raw, -jnp.inf), axis=(1, 2))
    min_diff2 = -raw_best
    diff2 = -raw
    prior_sum = rotation_prior[None, :, None] + translation_prior[:, None, :]
    log_weights = (prior_sum + min_diff2[:, None, None]) - diff2
    return jnp.where(finite, log_weights, -jnp.inf)


@partial(
    jax.jit,
    static_argnames=(
        "adaptive_fraction",
        "max_significants",
        "tie_score_ulps",
        "filter_positive_before_sort",
    ),
)
def relion_cuda_f32_coarse_posterior(
    scores_flat,
    *,
    adaptive_fraction=0.999,
    max_significants=500,
    tie_score_ulps=0,
    min_diff2_offsets=None,
    filter_positive_before_sort=False,
):
    """Reproduce RELION CUDA coarse-weight and significance arithmetic.

    RELION's accelerated coarse pass stores log weights in ``XFLOAT``
    (float32 in the deployed build), shifts their maximum to 50, applies
    ``expf``, radix-sorts positive weights in ascending order, and uses an
    inclusive float32 scan to select the lower-tail cutoff.  The surviving
    weights remain normalized by the full, pre-pruning sum.

    RELION constructs each coarse log weight as ``prior + min_diff2 - diff2``
    before finding the maximum and adding the exponentiation offset. Although
    ``min_diff2`` is common to every pose, omitting it can change float32
    cancellation in ``score + (50 - maximum)`` and collapse adjacent cutoff
    scores into a false tie. ``min_diff2_offsets`` restores that absolute
    score frame without changing normalized probabilities mathematically.

    ``cutoff_count`` is the pre-tie rank serialized by RELION. ``mask`` and
    ``n_significant`` include every positive weight tied at the cutoff.
    ``tie_score_ulps`` optionally absorbs a small score-level atomic-rounding
    envelope below that exact cutoff. It is an explicit diagnostic control;
    production InitialModel keeps the exact threshold comparison.

    ``filter_positive_before_sort`` explicitly selects RELION's native
    positive-only CUB primitive. Its fixed-size sort/scan outputs are
    right-aligned so all downstream indexing remains unchanged. It defaults
    off until a separate GPU parity/performance gate accepts it.
    """

    tie_score_ulps = int(tie_score_ulps)
    if tie_score_ulps < 0:
        raise ValueError("tie_score_ulps must be non-negative")

    scores_f32 = jnp.asarray(scores_flat, dtype=jnp.float32)
    if min_diff2_offsets is not None:
        offsets_f32 = jnp.asarray(min_diff2_offsets, dtype=jnp.float32)
        if offsets_f32.ndim != 1 or offsets_f32.shape[0] != scores_f32.shape[0]:
            raise ValueError(
                "min_diff2_offsets must have shape (n_images,), got "
                f"{offsets_f32.shape} for scores {scores_f32.shape}",
            )
        scores_f32 = scores_f32 + offsets_f32[:, None]
    finite = jnp.isfinite(scores_f32)
    best = jnp.max(jnp.where(finite, scores_f32, -jnp.inf), axis=1)
    has_finite = jnp.isfinite(best)
    safe_best = jnp.where(has_finite, best, jnp.float32(0.0))
    exponent_add = jnp.float32(50.0) - safe_best
    use_native_cuda = False
    if jax.default_backend() == "gpu":
        from recovar import cuda_backproject

        use_native_cuda = cuda_backproject.custom_cuda_requested()
    if use_native_cuda:
        finite_scores = jnp.where(finite, scores_f32, -jnp.inf)
        batched_primitives = (
            cuda_backproject.relion_batched_posterior_primitives_requested()
        )
        if batched_primitives:
            raw_weights = cuda_backproject.relion_exponentiate_batched_f32(
                finite_scores,
                exponent_add,
            )
        else:
            raw_weights = jax.vmap(cuda_backproject.relion_exponentiate_f32)(
                finite_scores,
                exponent_add,
            )
        sort_scan = (
            cuda_backproject.relion_cub_positive_sort_scan_f32
            if filter_positive_before_sort
            else cuda_backproject.relion_cub_sort_scan_f32
        )
        if batched_primitives and not filter_positive_before_sort:
            sorted_weights, cumulative = (
                cuda_backproject.relion_cub_sort_scan_batched_f32(raw_weights)
            )
        else:
            sorted_weights, cumulative = jax.vmap(sort_scan)(raw_weights)
    else:
        shifted = jnp.where(
            finite,
            scores_f32 + exponent_add[:, None],
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
    if tie_score_ulps > 0:
        cutoff_score = jnp.min(
            jnp.where(mask, scores_f32, jnp.float32(jnp.inf)),
            axis=1,
        )
        expanded_cutoff_score = cutoff_score
        for _ in range(tie_score_ulps):
            expanded_cutoff_score = jnp.nextafter(
                expanded_cutoff_score,
                jnp.full_like(expanded_cutoff_score, -jnp.inf),
            )
        mask = mask | (
            has_mass[:, None]
            & finite
            & (raw_weights > jnp.float32(0.0))
            & (scores_f32 >= expanded_cutoff_score[:, None])
        )
    safe_sum_weight = jnp.where(has_mass, sum_weight, jnp.float32(1.0))
    if use_native_cuda:
        if batched_primitives:
            probabilities = cuda_backproject.relion_divide_batched_f32(
                raw_weights,
                safe_sum_weight,
            )
        else:
            probabilities = jax.vmap(cuda_backproject.relion_divide_f32)(
                raw_weights,
                safe_sum_weight,
            )
        probabilities = jnp.where(
            has_mass[:, None],
            probabilities,
            jnp.float32(0.0),
        )
    else:
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


def _map_translation_log_prior_to_fine_grid(
    translation_log_prior,
    fine_translation_parent,
):
    """Map coarse-grid translation priors onto oversampled translation children."""
    if translation_log_prior is None:
        return None
    translation_log_prior = np.asarray(translation_log_prior)
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
