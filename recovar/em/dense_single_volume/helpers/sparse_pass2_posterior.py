"""Posterior normalization of the sparse bucketed pass 2.

RELION's ``convertAllSquaredDifferencesToWeights`` counterpart for pass-2
buckets and compact pairs: log-sum-exp normalization, the score-only and
winner-take-all variants, the float32 fine posterior of the RELION x-half
reconstruction path, the reconstruction probabilities handed to the M-step
and the joint class masks. ``sparse_pass2_bucketed`` calls them per bucket.
"""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag
from recovar.em.dense_single_volume.helpers.oversampling import (
    _find_significant_mask_full_sort,
    _relion_cuda_f32_tail_target,
)

_RELION_FINE_ROTATION_EXECUTION_ORDER_ENV = (
    "RECOVAR_RELION_FINE_ROTATION_EXECUTION_ORDER"
)


_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV = "RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE"


_RELION_X_HALF_F32_FINE_POSTERIOR_ENV = "RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR"


_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_JOINT_MODES = {"joint", "global", "class_pose", "class-pose"}


@jax.jit
def _normalize_pass2_bucket(scores):
    """Compute per-image normalization stats from (B, R, T) scores."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    log_shift = safe_best_log_score[:, None, None]
    shifted = jnp.where(has_finite_score[:, None, None], scores - log_shift, -jnp.inf)
    probs = jnp.exp(shifted.astype(jnp.float64))
    probs = jnp.where(jnp.isfinite(probs), probs, 0.0)
    sum_exp = jnp.sum(probs.reshape(scores.shape[0], -1), axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    safe_sum_exp = jnp.where(has_mass, sum_exp, 1.0)
    log_Z = jnp.where(has_mass, safe_best_log_score + jnp.log(safe_sum_exp), 0.0)
    probs = probs / safe_sum_exp[:, None, None]
    probs = jnp.where(has_mass[:, None, None], probs, 0.0)
    best_argmax = jnp.where(has_mass, jnp.argmax(flat, axis=1), 0)
    max_posterior = jnp.where(has_mass, jnp.max(probs.reshape(scores.shape[0], -1), axis=1), 0.0)
    best_log_score = jnp.where(has_mass, best_log_score, -jnp.inf)
    return log_Z, probs, best_log_score, best_argmax, max_posterior


@jax.jit
def _normalize_pass2_bucket_score_only(scores):
    """Compute sparse pass-2 score stats without materializing posteriors."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    shifted = jnp.where(has_finite_score[:, None, None], scores - safe_best_log_score[:, None, None], -jnp.inf)
    exp_terms = jnp.exp(shifted.astype(jnp.float64))
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms.reshape(scores.shape[0], -1), axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    safe_sum_exp = jnp.where(has_mass, sum_exp, 1.0)
    log_Z = jnp.where(has_mass, safe_best_log_score + jnp.log(safe_sum_exp), 0.0)
    best_argmax = jnp.where(has_mass, jnp.argmax(flat, axis=1), 0)
    max_posterior = jnp.exp(best_log_score - log_Z)
    max_posterior = jnp.where(has_mass & jnp.isfinite(max_posterior), max_posterior, 0.0)
    best_log_score = jnp.where(has_mass, best_log_score, -jnp.inf)
    return log_Z, best_log_score, best_argmax, max_posterior


@jax.jit
def _normalize_pass2_pairs_with_log_z(pair_scores, pair_mask, global_log_z):
    """Normalize compact pair scores against a precomputed joint log-Z.

    ``global_log_z`` may include other classes, so the returned pair
    probabilities need not sum to one within this class. Padded pairs and
    non-finite scores always contribute zero probability.
    """

    pair_scores = jnp.where(pair_mask & jnp.isfinite(pair_scores), pair_scores, -jnp.inf)
    best_log_score = jnp.max(pair_scores, axis=1)
    has_finite_score = jnp.isfinite(best_log_score) & jnp.isfinite(global_log_z)
    safe_log_z = jnp.where(has_finite_score, global_log_z, 0.0)
    pair_probs = jnp.exp(pair_scores - safe_log_z[:, None])
    pair_probs = jnp.where(has_finite_score[:, None] & jnp.isfinite(pair_probs), pair_probs, 0.0)
    best_pair_argmax = jnp.where(has_finite_score, jnp.argmax(pair_scores, axis=1), 0)
    max_posterior = jnp.exp(best_log_score - safe_log_z)
    max_posterior = jnp.where(has_finite_score & jnp.isfinite(max_posterior), max_posterior, 0.0)
    best_log_score = jnp.where(has_finite_score, best_log_score, -jnp.inf)
    return safe_log_z, pair_probs, best_log_score, best_pair_argmax, max_posterior


@jax.jit
def _logsumexp_pass2_bucket_score_only(scores):
    """Compute per-image sparse pass-2 logZ only."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    shifted = jnp.where(has_finite_score[:, None, None], scores - safe_best_log_score[:, None, None], -jnp.inf)
    exp_terms = jnp.exp(shifted.astype(jnp.float64))
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms.reshape(scores.shape[0], -1), axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    safe_sum_exp = jnp.where(has_mass, sum_exp, 1.0)
    return jnp.where(has_mass, safe_best_log_score + jnp.log(safe_sum_exp), -jnp.inf)


@jax.jit
def _logsumexp_pass2_pairs_score_only(pair_scores, pair_mask):
    """Compute per-image compact pass-2 logZ over valid pairs only."""
    pair_scores = jnp.where(pair_mask & jnp.isfinite(pair_scores), pair_scores, -jnp.inf)
    best_log_score = jnp.max(pair_scores, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    shifted = jnp.where(has_finite_score[:, None], pair_scores - safe_best_log_score[:, None], -jnp.inf)
    exp_terms = jnp.exp(shifted.astype(jnp.float64))
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms, axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    return jnp.where(has_mass, safe_best_log_score + jnp.log(sum_exp), -jnp.inf)


@jax.jit
def _logsumexp_class_log_z(class_log_z):
    """Stable logsumexp over class-local sparse score normalizers."""

    finite = jnp.isfinite(class_log_z)
    max_value = jnp.max(jnp.where(finite, class_log_z, -jnp.inf), axis=0)
    has_finite = jnp.isfinite(max_value)
    shifted = jnp.where(finite & has_finite[None, :], class_log_z - max_value[None, :], -jnp.inf)
    exp_terms = jnp.exp(shifted)
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms, axis=0)
    return jnp.where(has_finite & (sum_exp > 0.0), max_value + jnp.log(sum_exp), -jnp.inf)


@jax.jit
def _winner_take_all_bucket_probs(scores, best_argmax, best_log_score):
    """One-hot sparse bucket probabilities for RELION firstiter_cc."""

    flat_size = scores.shape[1] * scores.shape[2]
    valid = jnp.isfinite(best_log_score)
    probs = jax.nn.one_hot(best_argmax, flat_size, dtype=scores.real.dtype).reshape(scores.shape)
    return probs * valid[:, None, None].astype(probs.dtype)


@jax.jit
def _winner_take_all_bucket_probs_from_global_argmax(scores, global_argmax, chunk_rotation_start, best_log_score):
    """One-hot probabilities for a rotation chunk using full-bucket argmax."""

    flat_size = scores.shape[1] * scores.shape[2]
    local_argmax = global_argmax - chunk_rotation_start * scores.shape[2]
    valid = (local_argmax >= 0) & (local_argmax < flat_size) & jnp.isfinite(best_log_score)
    safe_argmax = jnp.where(valid, local_argmax, 0)
    probs = jax.nn.one_hot(safe_argmax, flat_size, dtype=scores.real.dtype).reshape(scores.shape)
    return probs * valid[:, None, None].astype(probs.dtype)


@jax.jit
def _winner_take_all_pair_probs(pair_scores, best_pair_argmax, best_log_score):
    """One-hot compact pair probabilities for RELION firstiter_cc."""

    valid = jnp.isfinite(best_log_score)
    probs = jax.nn.one_hot(best_pair_argmax, pair_scores.shape[1], dtype=pair_scores.real.dtype)
    return probs * valid[:, None].astype(probs.dtype)


@jax.jit
def _normalize_pass2_bucket_with_log_z(scores, log_z):
    """Normalize sparse candidate scores with a precomputed full-grid log-Z."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score) & jnp.isfinite(log_z)
    safe_log_z = jnp.where(has_finite_score, log_z, 0.0)
    probs = jnp.exp(scores - safe_log_z[:, None, None])
    probs = jnp.where(has_finite_score[:, None, None] & jnp.isfinite(probs), probs, 0.0)
    best_argmax = jnp.where(has_finite_score, jnp.argmax(flat, axis=1), 0)
    max_posterior = jnp.exp(best_log_score - safe_log_z)
    max_posterior = jnp.where(has_finite_score & jnp.isfinite(max_posterior), max_posterior, 0.0)
    best_log_score = jnp.where(has_finite_score, best_log_score, -jnp.inf)
    return safe_log_z, probs, best_log_score, best_argmax, max_posterior


@jax.jit
def _diagnostics_from_normalized_pass2_probs(scores, probs, log_z):
    """Return normalization diagnostics without recomputing probabilities."""

    flat_scores = jnp.asarray(scores).reshape(scores.shape[0], -1)
    flat_probs = jnp.asarray(probs).reshape(probs.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    has_finite = jnp.isfinite(best_log_score) & jnp.isfinite(log_z)
    best_argmax = jnp.where(has_finite, jnp.argmax(flat_scores, axis=1), 0)
    max_posterior = jnp.where(
        has_finite,
        jnp.max(flat_probs, axis=1),
        jnp.asarray(0.0, dtype=flat_probs.dtype),
    )
    return (
        jnp.where(has_finite, log_z, 0.0),
        jnp.where(has_finite, best_log_score, -jnp.inf),
        best_argmax,
        max_posterior,
    )


def _relion_pass2_reconstruction_probs(probs, *, adaptive_fraction: float):
    """Apply RELION's fine-pass significant threshold before M-step sums."""

    flat_probs = probs.reshape(probs.shape[0], -1)
    mask_flat, n_significant = _find_significant_mask_full_sort(
        flat_probs,
        float(adaptive_fraction),
        -1,
    )
    mask = mask_flat.reshape(probs.shape)
    return jnp.where(mask, probs, 0.0), mask, n_significant


@partial(jax.jit, static_argnames=("adaptive_fraction", "keep_all"))
def _relion_f32_fine_posterior(
    scores,
    *,
    adaptive_fraction: float,
    normalization_sum_weight=None,
    keep_all: bool = False,
):
    """Build full and pruned fine probabilities with RELION GPU arithmetic.

    The reference GPU path shifts its float32 log weights so the maximum is
    50, applies ``expf``, sorts the raw weights in ascending order, and obtains
    both ``sum_weight`` and the lower-tail significance cutoff from a float32
    cumulative scan.  Surviving weights are divided by the full pre-pruning
    ``sum_weight``; they are intentionally not renormalized afterward.
    """

    scores_f32 = jnp.asarray(scores, dtype=jnp.float32)
    flat_scores = scores_f32.reshape(scores_f32.shape[0], -1)
    finite = jnp.isfinite(flat_scores)
    best = jnp.max(jnp.where(finite, flat_scores, -jnp.inf), axis=1)
    has_finite = jnp.isfinite(best)
    safe_best = jnp.where(has_finite, best, jnp.float32(0.0))
    exponent_add = jnp.float32(50.0) - safe_best
    use_native_cuda = False
    if jax.default_backend() == "gpu":
        from recovar import cuda_backproject

        use_native_cuda = cuda_backproject.custom_cuda_requested()
    if use_native_cuda:
        # RELION computes ``50 - weights_max`` once in XFLOAT, then its CUDA
        # kernel evaluates ``expf(score + add)``.  Reassociating this as
        # ``exp(score - best + 50)`` changes hundreds of thousands of raw
        # weights at the iteration-2 case-22 boundary.  Its deployed sm_80
        # scan policy also remains observable when the same binary is JITed
        # on Hopper, so the CUDA primitive pins that policy explicitly.
        finite_scores = jnp.where(finite, flat_scores, -jnp.inf)
        batched_primitives = (
            cuda_backproject.relion_batched_posterior_primitives_requested()
        )
        if batched_primitives:
            raw_weights = cuda_backproject.relion_exponentiate_batched_f32(
                finite_scores,
                exponent_add,
            )
            sorted_weights, cumulative = (
                cuda_backproject.relion_cub_sort_scan_batched_f32(raw_weights)
            )
        else:
            raw_weights = jax.vmap(cuda_backproject.relion_exponentiate_f32)(
                finite_scores,
                exponent_add,
            )
            sorted_weights, cumulative = jax.vmap(
                cuda_backproject.relion_cub_sort_scan_f32,
            )(raw_weights)
    else:
        shifted = jnp.where(
            finite,
            flat_scores + exponent_add[:, None],
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
        sorted_weights = jnp.sort(raw_weights, axis=1)
        cumulative = jnp.cumsum(sorted_weights, axis=1, dtype=jnp.float32)
    fine_sum_weight = cumulative[:, -1]
    if normalization_sum_weight is None:
        sum_weight = fine_sum_weight
    else:
        # Gradient InitialModel with adaptive_oversampling==0 does not update
        # op.sum_weight in the symbolic fine pass.  The CUDA fine weights are
        # shifted by their own maximum, then divided directly by the numeric
        # float32 denominator retained from the coarse pass.
        sum_weight = jnp.asarray(normalization_sum_weight, dtype=jnp.float32)
    has_mass = has_finite & jnp.isfinite(sum_weight) & (sum_weight > jnp.float32(0.0))
    if keep_all:
        threshold = jnp.zeros_like(sum_weight)
        mask_flat = has_mass[:, None] & finite & (raw_weights > jnp.float32(0.0))
    else:
        tail_target = _relion_cuda_f32_tail_target(fine_sum_weight, adaptive_fraction)
        threshold_idx = jax.vmap(
            lambda row, target: jnp.searchsorted(row, target, side="right")
        )(
            cumulative,
            tail_target,
        )
        threshold_idx = jnp.minimum(threshold_idx, cumulative.shape[1] - 1)
        threshold = sorted_weights[jnp.arange(flat_scores.shape[0]), threshold_idx]
        mask_flat = has_mass[:, None] & finite & (raw_weights >= threshold[:, None])
    safe_sum_weight = jnp.where(has_mass, sum_weight, jnp.float32(1.0))
    if use_native_cuda:
        if batched_primitives:
            normalized_weights = cuda_backproject.relion_divide_batched_f32(
                raw_weights,
                safe_sum_weight,
            )
        else:
            normalized_weights = jax.vmap(cuda_backproject.relion_divide_f32)(
                raw_weights,
                safe_sum_weight,
            )
    else:
        normalized_weights = raw_weights / safe_sum_weight[:, None]
    reconstruction_probs_flat = jnp.where(
        mask_flat,
        normalized_weights,
        jnp.float32(0.0),
    )
    n_significant = jnp.sum(mask_flat, axis=1).astype(jnp.int32)
    output_shape = scores_f32.shape
    return (
        normalized_weights.reshape(output_shape),
        reconstruction_probs_flat.reshape(output_shape),
        mask_flat.reshape(output_shape),
        n_significant,
        sum_weight,
        threshold,
    )


def _relion_f32_fine_reconstruction_probs(scores, *, adaptive_fraction: float):
    """Return the legacy pruned view of :func:`_relion_f32_fine_posterior`."""

    full = _relion_f32_fine_posterior(
        scores,
        adaptive_fraction=adaptive_fraction,
    )
    return full[1:]


def relion_x_half_f32_fine_posterior_enabled(*, default: bool = False) -> bool:
    """Return whether the RELION float32 fine posterior is enabled.

    Preserve the PR179 production default; enable this separately qualified
    arithmetic path explicitly for boundary experiments.
    """

    return parse_env_flag(_RELION_X_HALF_F32_FINE_POSTERIOR_ENV, default=default)


def _relion_fine_parent_execution_order_enabled(
    *,
    use_relion_f32_fine_posterior: bool,
) -> bool:
    """Keep exact fine-posterior diagnostics in RELION's candidate order."""

    return bool(use_relion_f32_fine_posterior) or parse_env_flag(
        _RELION_FINE_ROTATION_EXECUTION_ORDER_ENV,
        default=False,
    )


def _relion_pass2_reconstruction_probs_for_mstep(
    scores,
    probs,
    *,
    adaptive_fraction: float,
    use_relion_x_half_mstep: bool,
    use_relion_f32_fine_posterior: bool = False,
    winner_take_all: bool = False,
    return_diagnostics: bool = False,
):
    """Select the default or diagnostic fine-posterior reconstruction path."""

    if (
        use_relion_x_half_mstep
        and not winner_take_all
        and (
            bool(use_relion_f32_fine_posterior)
            or relion_x_half_f32_fine_posterior_enabled()
        )
    ):
        reconstruction_probs, mask, n_significant, sum_weight, threshold = (
            _relion_f32_fine_reconstruction_probs(
                scores,
                adaptive_fraction=float(adaptive_fraction),
            )
        )
        if return_diagnostics:
            return reconstruction_probs, mask, n_significant, sum_weight, threshold
        return reconstruction_probs, mask, n_significant
    reconstruction_probs, mask, n_significant = _relion_pass2_reconstruction_probs(
        probs,
        adaptive_fraction=float(adaptive_fraction),
    )
    if not return_diagnostics:
        return reconstruction_probs, mask, n_significant
    flat_probs = jnp.asarray(probs, dtype=jnp.float64).reshape(probs.shape[0], -1)
    sum_weight = jnp.sum(flat_probs, axis=1, dtype=jnp.float64)
    threshold = jnp.min(
        jnp.where(mask.reshape(mask.shape[0], -1), flat_probs, jnp.inf),
        axis=1,
    )
    threshold = jnp.where(jnp.isfinite(threshold), threshold, 0.0)
    return reconstruction_probs, mask, n_significant, sum_weight, threshold


def _relion_pass2_reconstruction_pair_probs(pair_probs, pair_mask, *, adaptive_fraction: float):
    """Apply RELION's fine-pass significant threshold to compact pair probs."""

    pair_probs = jnp.where(pair_mask, pair_probs, 0.0)
    mask, n_significant = _find_significant_mask_full_sort(
        pair_probs,
        float(adaptive_fraction),
        -1,
    )
    mask = mask & pair_mask
    return jnp.where(mask, pair_probs, 0.0), mask, n_significant


def _relion_fine_mstep_prune_mode(*, use_relion_x_half_mstep: bool, mode_override: str | None = None) -> str:
    """Return the diagnostic fine-pass M-step pruning mode.

    ``per_class`` preserves the original opt-in diagnostic. ``joint`` matches
    RELION Class3D storeWeightedSums: threshold one flattened class x pose
    posterior list per image before accumulating M-step sums.
    ``joint_keep_all`` preserves every admitted coarse candidate while still
    using the joint RELION float32 posterior arithmetic.
    """

    value = mode_override
    if value is None:
        value = os.environ.get(_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV)
    if value is None or not value.strip():
        return "per_class" if use_relion_x_half_mstep else "none"
    mode = value.strip().lower()
    if mode in {"joint_keep_all", "joint-keep-all"}:
        return "joint_keep_all"
    if mode in {"all", "keep_all", "keep-all", "no_prune", "no-prune"}:
        return "none"
    if mode in _SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_JOINT_MODES:
        return "joint"
    if mode in {"1", "true", "yes", "on", "class", "per_class", "per-class"}:
        return "per_class"
    if mode in {"0", "false", "no", "off", "none"}:
        return "per_class" if use_relion_x_half_mstep else "none"
    raise ValueError(
        f"{_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV} must be one of "
        "0/1/per_class/joint/joint_keep_all"
    )


def _relion_pass2_reconstruction_joint_masks(flat_probs_by_class, *, adaptive_fraction: float):
    """Threshold a per-image flattened class x pose posterior list."""

    if not flat_probs_by_class:
        return []
    flat_sizes = [int(probs.shape[1]) for probs in flat_probs_by_class]
    joint_probs = jnp.concatenate(flat_probs_by_class, axis=1)
    joint_mask, _n_significant = _find_significant_mask_full_sort(
        joint_probs,
        float(adaptive_fraction),
        -1,
    )
    split_points = np.cumsum(flat_sizes[:-1], dtype=np.int64).tolist()
    return list(jnp.split(joint_mask, split_points, axis=1))


def _relion_joint_winner_take_all_masks(scores_by_class, masks_by_class=None):
    """Return one global class x pose winner mask per image.

    ``scores_by_class`` may hold rectangular ``(B, R, T)`` or compact ``(B, P)``
    scores; ``masks_by_class`` optionally masks compact pairs to ``-inf`` first.
    Returns one flat ``(B, R * T)`` / ``(B, P)`` mask per class.
    """

    if not scores_by_class:
        return []
    if masks_by_class is None:
        masks_by_class = [None] * len(scores_by_class)
    return _relion_joint_winner_take_all_masks_jit(tuple(scores_by_class), tuple(masks_by_class))


def _flatten_joint_class_scores(scores_by_class, masks_by_class):
    flat_scores = []
    for scores, mask in zip(scores_by_class, masks_by_class, strict=True):
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
        flat_scores.append(scores.reshape(scores.shape[0], -1))
    return flat_scores


@jax.jit
def _relion_joint_winner_take_all_masks_jit(scores_by_class, masks_by_class):
    flat_scores_by_class = _flatten_joint_class_scores(scores_by_class, masks_by_class)
    flat_sizes = [int(scores.shape[1]) for scores in flat_scores_by_class]
    joint_scores = jnp.concatenate(flat_scores_by_class, axis=1)
    finite = jnp.isfinite(joint_scores)
    safe_scores = jnp.where(finite, joint_scores, -jnp.inf)
    best_idx = jnp.argmax(safe_scores, axis=1)
    valid = jnp.any(finite, axis=1)
    joint_mask = (jnp.arange(joint_scores.shape[1])[None, :] == best_idx[:, None]) & valid[:, None]
    split_points = np.cumsum(flat_sizes[:-1], dtype=np.int64).tolist()
    return list(jnp.split(joint_mask, split_points, axis=1))


@partial(jax.jit, static_argnames=("adaptive_fraction", "keep_all"))
def _relion_f32_fine_posterior_by_class(
    scores_by_class,
    masks_by_class,
    *,
    adaptive_fraction: float,
    normalization_sum_weight=None,
    keep_all: bool = False,
):
    """Run the joint RELION float32 fine posterior over one bucket's classes.

    Concatenates the per-class scores (compact pairs masked to ``-inf``), runs
    :func:`_relion_f32_fine_posterior`, and splits the mask, full and
    reconstruction probabilities back into each class's score shape inside one
    compiled program instead of one XLA program per concatenate/split/reshape.
    """

    flat_scores_by_class = _flatten_joint_class_scores(scores_by_class, masks_by_class)
    flat_sizes = [int(scores.shape[1]) for scores in flat_scores_by_class]
    joint_scores = jnp.concatenate(flat_scores_by_class, axis=1)
    joint_full_probs, joint_reconstruction_probs, joint_mask, *_diagnostics = _relion_f32_fine_posterior(
        joint_scores,
        adaptive_fraction=adaptive_fraction,
        normalization_sum_weight=normalization_sum_weight,
        keep_all=keep_all,
    )
    split_points = np.cumsum(flat_sizes[:-1], dtype=np.int64).tolist()
    shapes = [scores.shape for scores in scores_by_class]

    def _split(joint):
        return [
            part.reshape(shape)
            for part, shape in zip(jnp.split(joint, split_points, axis=1), shapes, strict=True)
        ]

    return _split(joint_mask), _split(joint_full_probs), _split(joint_reconstruction_probs)
