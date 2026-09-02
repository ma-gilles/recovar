"""Certified block selection building blocks for a K=1 coarse-score hybrid.

This module deliberately has no production caller.  It provides the small,
default-off pieces needed to turn streamed candidate score intervals into a
fail-closed set of aligned 16-rotation blocks.  Exact rescoring and posterior
arithmetic remain owned by the existing dense E-step.
"""

from __future__ import annotations

import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

SOURCE_ROTATION_BLOCK_SIZE = 16
DEFAULT_ROTATION_BLOCK_CAPACITY = 64
RELION_COARSE_NONZERO_SCORE_SPAN = 138.0


class CoarseGemmHybridIntervalState(NamedTuple):
    """Compact complete-coverage reductions for raw and post-prior intervals."""

    raw_block_lower_max: jax.Array
    raw_block_upper_max: jax.Array
    posterior_block_lower_max: jax.Array
    posterior_block_upper_max: jax.Array
    rotation_visit_count: jax.Array
    candidate_count: jax.Array
    invalid_candidate_count: jax.Array


class CoarseGemmHybridBlockSelection(NamedTuple):
    """Fixed-capacity block selection, or a reason to use full direct scoring."""

    eligible: bool
    fallback_reason: str | None
    block_ids: np.ndarray
    block_count: np.ndarray
    posterior_block_count: np.ndarray
    raw_max_block_count: np.ndarray


def certified_f32_dot_product_gamma(
    traversed_full_position_bound: int,
    n_translations: int,
) -> float:
    """Return an upward-rounded ``gamma_k`` for the direct CUDA topology.

    ``traversed_full_position_bound`` is the upper bound on full image
    positions traversed by the direct kernel, including positions whose
    compacted scientific weight may be zero.  It is not the number of retained
    or non-skipped compact terms.
    """

    try:
        n_positions = operator.index(traversed_full_position_bound)
        n_translations = operator.index(n_translations)
    except TypeError as error:
        raise ValueError("kernel dimensions must be integers") from error
    if n_positions <= 0 or not 1 <= n_translations <= 128:
        raise ValueError(
            "traversed_full_position_bound must be positive and n_translations must be in [1, 128]",
        )
    translations_per_lane = 128 // n_translations
    lane_term_bound = ((n_positions + 31) // 32) * ((32 + translations_per_lane - 1) // translations_per_lane)
    operation_count = lane_term_bound + translations_per_lane + 5
    unit_roundoff = 2.0**-24
    scaled_roundoff = operation_count * unit_roundoff
    denominator = 1.0 - scaled_roundoff
    if denominator <= 0.0:
        raise ValueError("dot-product error-bound denominator is non-positive")
    gamma = scaled_roundoff / denominator
    return float(np.nextafter(np.float64(gamma), np.float64(np.inf)))


def coarse_gemm_hybrid_interval_state_bytes(
    batch_size: int,
    n_rotations: int,
) -> int:
    """Return persistent bytes for the four interval tables and coverage state."""

    batch_size = operator.index(batch_size)
    n_rotations = operator.index(n_rotations)
    if batch_size <= 0 or n_rotations <= 0:
        raise ValueError("batch_size and n_rotations must be positive")
    n_blocks = (n_rotations + SOURCE_ROTATION_BLOCK_SIZE - 1) // SOURCE_ROTATION_BLOCK_SIZE
    return int(
        4 * batch_size * n_blocks * np.dtype(np.float64).itemsize
        + n_rotations * np.dtype(np.int32).itemsize
        + 2 * batch_size * np.dtype(np.int32).itemsize
    )


@jax.jit
def coarse_gemm_direct_score_intervals(
    expanded_score_f64,
    eta_f64,
    gamma_f64,
):
    """Construct outward-rounded direct-score intervals around live ``g64``.

    ``eta_f64`` certifies ``abs(g64 - S)`` and may be scalar or broadcastable
    to ``g64``.  Invalid/nonfinite inputs become NaN endpoints so the streamed
    state and host selector fail closed without inspecting a full score cube.
    """

    with jax.enable_x64(True):
        score = jnp.asarray(expanded_score_f64, dtype=jnp.float64)
        eta = jnp.broadcast_to(jnp.asarray(eta_f64, dtype=jnp.float64), score.shape)
        gamma = jnp.asarray(gamma_f64, dtype=jnp.float64)
        if gamma.shape != ():
            raise ValueError("gamma_f64 must be scalar")
        positive_inf = jnp.full_like(score, jnp.inf)
        negative_inf = jnp.full_like(score, -jnp.inf)
        q_upper = jnp.nextafter(
            jnp.maximum(jnp.float64(0.0), -score + eta),
            positive_inf,
        )
        gamma_term = jnp.nextafter(gamma * q_upper, positive_inf)
        error_upper = jnp.nextafter(eta + gamma_term, positive_inf)
        lower = jnp.nextafter(score - error_upper, negative_inf)
        upper = jnp.nextafter(score + error_upper, positive_inf)
        valid = (
            jnp.isfinite(score)
            & jnp.isfinite(eta)
            & (eta >= 0.0)
            # A squared-difference score is non-positive.  Given |g-S|<=eta,
            # g<=eta is the subtraction-free form of the required g-eta<=0
            # certificate check and avoids another rounded endpoint operation.
            & (score <= eta)
            & jnp.isfinite(gamma)
            & (gamma >= 0.0)
            & jnp.isfinite(error_upper)
            & jnp.isfinite(lower)
            & jnp.isfinite(upper)
            & (lower <= upper)
        )
        nan = jnp.full_like(score, jnp.nan)
        return jnp.where(valid, lower, nan), jnp.where(valid, upper, nan)


def _enclosing_f32_add(lower, upper, prior):
    """Enclose one production-order FP32 addition, entirely on device."""

    prior_f32 = jnp.asarray(prior, dtype=jnp.float32)
    prior_f64 = prior_f32.astype(jnp.float64)
    negative_inf64 = jnp.full_like(lower, -jnp.inf)
    positive_inf64 = jnp.full_like(upper, jnp.inf)
    lower_sum = jnp.nextafter(lower + prior_f64, negative_inf64)
    upper_sum = jnp.nextafter(upper + prior_f64, positive_inf64)
    lower_f32 = lower_sum.astype(jnp.float32)
    upper_f32 = upper_sum.astype(jnp.float32)
    lower_f32 = jnp.where(
        lower_f32.astype(jnp.float64) > lower_sum,
        jnp.nextafter(lower_f32, jnp.full_like(lower_f32, -jnp.inf)),
        lower_f32,
    )
    upper_f32 = jnp.where(
        upper_f32.astype(jnp.float64) < upper_sum,
        jnp.nextafter(upper_f32, jnp.full_like(upper_f32, jnp.inf)),
        upper_f32,
    )
    result_lower = lower_f32.astype(jnp.float64)
    result_upper = upper_f32.astype(jnp.float64)
    valid = (
        jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & (lower <= upper)
        & jnp.isfinite(prior_f32)
        & jnp.isfinite(result_lower)
        & jnp.isfinite(result_upper)
        & (result_lower <= result_upper)
    )
    nan = jnp.full_like(result_lower, jnp.nan)
    return jnp.where(valid, result_lower, nan), jnp.where(valid, result_upper, nan)


@jax.jit
def propagate_coarse_gemm_intervals_through_f32_priors(
    raw_lower,
    raw_upper,
    *,
    class_log_prior,
    rotation_log_prior=None,
    translation_log_prior=None,
):
    """Apply class, rotation, then translation priors as dense ``_add_priors``.

    The class addition is mandatory even for K=1.  Endpoint work stays in JAX;
    malformed shapes fail while tracing and invalid values become NaN bounds.
    """

    with jax.enable_x64(True):
        lower = jnp.asarray(raw_lower, dtype=jnp.float64)
        upper = jnp.asarray(raw_upper, dtype=jnp.float64)
        if lower.ndim != 3 or lower.shape != upper.shape:
            raise ValueError("raw endpoints must be equal [image, rotation, translation] arrays")
        class_prior = jnp.asarray(class_log_prior, dtype=jnp.float32)
        if class_prior.shape != ():
            raise ValueError("class_log_prior must be scalar")
        lower, upper = _enclosing_f32_add(lower, upper, class_prior)
        if rotation_log_prior is not None:
            rotation_prior = jnp.asarray(rotation_log_prior, dtype=jnp.float32)
            if rotation_prior.shape != (lower.shape[1],):
                raise ValueError("rotation_log_prior must have one value per rotation")
            lower, upper = _enclosing_f32_add(
                lower,
                upper,
                rotation_prior[None, :, None],
            )
        if translation_log_prior is not None:
            translation_prior = jnp.asarray(translation_log_prior, dtype=jnp.float32)
            if translation_prior.shape == (lower.shape[2],):
                translation_prior = translation_prior[None, None, :]
            elif translation_prior.shape == (lower.shape[0], lower.shape[2]):
                translation_prior = translation_prior[:, None, :]
            else:
                raise ValueError(
                    "translation_log_prior must be [translation] or [image, translation]",
                )
            lower, upper = _enclosing_f32_add(lower, upper, translation_prior)
        return lower, upper


def initialize_coarse_gemm_hybrid_interval_state(
    batch_size: int,
    n_rotations: int,
) -> CoarseGemmHybridIntervalState:
    """Create an empty compact state; rotations need not yet be block aligned."""

    batch_size = operator.index(batch_size)
    n_rotations = operator.index(n_rotations)
    if batch_size <= 0 or n_rotations <= 0:
        raise ValueError("batch_size and n_rotations must be positive")
    n_blocks = (n_rotations + SOURCE_ROTATION_BLOCK_SIZE - 1) // SOURCE_ROTATION_BLOCK_SIZE
    with jax.enable_x64(True):
        empty = jnp.full((batch_size, n_blocks), -jnp.inf, dtype=jnp.float64)
        return CoarseGemmHybridIntervalState(
            raw_block_lower_max=empty,
            raw_block_upper_max=empty,
            posterior_block_lower_max=empty,
            posterior_block_upper_max=empty,
            rotation_visit_count=jnp.zeros((n_rotations,), dtype=jnp.int32),
            candidate_count=jnp.zeros((batch_size,), dtype=jnp.int32),
            invalid_candidate_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        )


@jax.jit
def _update_coarse_gemm_hybrid_interval_state(
    state,
    raw_lower,
    raw_upper,
    posterior_lower,
    posterior_upper,
    rotation_ids,
    actual_image_count,
):
    batch_size, rotation_count, translation_count = raw_lower.shape
    active_rows = jnp.arange(batch_size, dtype=jnp.int32) < actual_image_count
    valid = (
        jnp.isfinite(raw_lower)
        & jnp.isfinite(raw_upper)
        & (raw_lower <= raw_upper)
        & jnp.isfinite(posterior_lower)
        & jnp.isfinite(posterior_upper)
        & (posterior_lower <= posterior_upper)
    )

    def update_block_maximum(current, values):
        rotation_maximum = jnp.max(
            jnp.where(active_rows[:, None, None], values, -jnp.inf),
            axis=2,
        )
        block_ids = rotation_ids // SOURCE_ROTATION_BLOCK_SIZE
        return current.at[:, block_ids].max(rotation_maximum)

    visits = state.rotation_visit_count.at[rotation_ids].add(
        jnp.ones((rotation_count,), dtype=jnp.int32),
    )
    candidate_increment = jnp.int32(rotation_count * translation_count)
    invalid_increment = jnp.sum(~valid, axis=(1, 2), dtype=jnp.int32)
    return CoarseGemmHybridIntervalState(
        raw_block_lower_max=update_block_maximum(state.raw_block_lower_max, raw_lower),
        raw_block_upper_max=update_block_maximum(state.raw_block_upper_max, raw_upper),
        posterior_block_lower_max=update_block_maximum(
            state.posterior_block_lower_max,
            posterior_lower,
        ),
        posterior_block_upper_max=update_block_maximum(
            state.posterior_block_upper_max,
            posterior_upper,
        ),
        rotation_visit_count=visits,
        candidate_count=state.candidate_count + jnp.where(active_rows, candidate_increment, jnp.int32(0)),
        invalid_candidate_count=state.invalid_candidate_count + jnp.where(active_rows, invalid_increment, jnp.int32(0)),
    )


def update_coarse_gemm_hybrid_interval_state(
    state: CoarseGemmHybridIntervalState,
    raw_lower,
    raw_upper,
    posterior_lower,
    posterior_upper,
    *,
    rotation_offset: int,
    valid_rotation_count: int,
    actual_image_count: int,
) -> CoarseGemmHybridIntervalState:
    """Stream one contiguous rotation chunk without inspecting its score cubes."""

    arrays = tuple(
        jnp.asarray(value)
        for value in (
            raw_lower,
            raw_upper,
            posterior_lower,
            posterior_upper,
        )
    )
    if arrays[0].ndim != 3 or any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("all endpoints must be equal [image, rotation, translation] arrays")
    batch_size, supplied_rotations, _ = arrays[0].shape
    rotation_offset = operator.index(rotation_offset)
    valid_rotation_count = operator.index(valid_rotation_count)
    actual_image_count = operator.index(actual_image_count)
    n_rotations = state.rotation_visit_count.shape[0]
    if (
        batch_size != state.raw_block_lower_max.shape[0]
        or rotation_offset < 0
        or valid_rotation_count <= 0
        or valid_rotation_count > supplied_rotations
        or rotation_offset + valid_rotation_count > n_rotations
        or actual_image_count <= 0
        or actual_image_count > batch_size
    ):
        raise ValueError("invalid hybrid interval update bounds")
    rotation_ids = jnp.arange(
        rotation_offset,
        rotation_offset + valid_rotation_count,
        dtype=jnp.int32,
    )
    sliced = tuple(value[:, :valid_rotation_count, :] for value in arrays)
    return _update_coarse_gemm_hybrid_interval_state(
        state,
        *sliced,
        rotation_ids,
        jnp.int32(actual_image_count),
    )


def _failed_selection(batch_size: int, capacity: int, reason: str):
    width = max(1, capacity)
    return CoarseGemmHybridBlockSelection(
        eligible=False,
        fallback_reason=reason,
        block_ids=np.full((batch_size, width), -1, dtype=np.int32),
        block_count=np.zeros((batch_size,), dtype=np.int32),
        posterior_block_count=np.zeros((batch_size,), dtype=np.int32),
        raw_max_block_count=np.zeros((batch_size,), dtype=np.int32),
    )


def select_coarse_gemm_hybrid_rotation_blocks(
    state: CoarseGemmHybridIntervalState,
    *,
    actual_image_count: int,
    n_rotations: int,
    n_translations: int,
    certificate_valid: bool | None = None,
    block_capacity: int = DEFAULT_ROTATION_BLOCK_CAPACITY,
    nonzero_score_span: float = RELION_COARSE_NONZERO_SCORE_SPAN,
) -> CoarseGemmHybridBlockSelection:
    """Transfer compact state and select a conservative source-block union."""

    batch_size = state.raw_block_lower_max.shape[0]
    try:
        actual_image_count = operator.index(actual_image_count)
        n_rotations = operator.index(n_rotations)
        n_translations = operator.index(n_translations)
        block_capacity = operator.index(block_capacity)
        nonzero_score_span = float(nonzero_score_span)
    except (TypeError, ValueError, OverflowError):
        return _failed_selection(batch_size, 1, "invalid_selection_configuration")
    if not isinstance(certificate_valid, (bool, np.bool_)) or not certificate_valid:
        return _failed_selection(
            batch_size,
            block_capacity,
            "missing_or_invalid_certificate_contract",
        )
    if n_rotations <= 0 or n_rotations % SOURCE_ROTATION_BLOCK_SIZE:
        return _failed_selection(batch_size, block_capacity, "rotation_tail_not_supported")
    if (
        actual_image_count <= 0
        or actual_image_count > batch_size
        or n_translations <= 0
        or block_capacity <= 0
        or not np.isfinite(nonzero_score_span)
        or nonzero_score_span < 0.0
    ):
        return _failed_selection(batch_size, block_capacity, "invalid_selection_configuration")
    n_blocks = n_rotations // SOURCE_ROTATION_BLOCK_SIZE
    block_fields = (
        state.raw_block_lower_max,
        state.raw_block_upper_max,
        state.posterior_block_lower_max,
        state.posterior_block_upper_max,
    )
    if (
        state.rotation_visit_count.shape != (n_rotations,)
        or state.candidate_count.shape != (batch_size,)
        or state.invalid_candidate_count.shape != (batch_size,)
        or any(field.shape != (batch_size, n_blocks) for field in block_fields)
    ):
        return _failed_selection(batch_size, block_capacity, "incomplete_block_table")
    visits = np.asarray(state.rotation_visit_count, dtype=np.int32)
    if not np.all(visits == 1):
        return _failed_selection(
            batch_size,
            block_capacity,
            "incomplete_or_duplicate_rotation_coverage",
        )
    expected_candidates = n_rotations * n_translations
    counts = np.asarray(state.candidate_count, dtype=np.int64)[:actual_image_count]
    if not np.all(counts == expected_candidates):
        return _failed_selection(batch_size, block_capacity, "incomplete_candidate_coverage")
    invalid = np.asarray(state.invalid_candidate_count, dtype=np.int64)[:actual_image_count]
    if np.any(invalid):
        return _failed_selection(
            batch_size,
            block_capacity,
            "invalid_or_nonfinite_candidate_interval",
        )
    raw_lower, raw_upper, posterior_lower, posterior_upper = (
        np.asarray(field, dtype=np.float64)[:actual_image_count] for field in block_fields
    )
    if not (
        np.all(np.isfinite(raw_lower))
        and np.all(np.isfinite(raw_upper))
        and np.all(np.isfinite(posterior_lower))
        and np.all(np.isfinite(posterior_upper))
        and np.all(raw_lower <= raw_upper)
        and np.all(posterior_lower <= posterior_upper)
    ):
        return _failed_selection(
            batch_size,
            block_capacity,
            "invalid_or_nonfinite_block_interval",
        )

    block_ids = np.full((batch_size, block_capacity), -1, dtype=np.int32)
    block_count = np.zeros((batch_size,), dtype=np.int32)
    posterior_count = np.zeros((batch_size,), dtype=np.int32)
    raw_count = np.zeros((batch_size,), dtype=np.int32)
    for row in range(actual_image_count):
        raw_ids = np.flatnonzero(raw_upper[row] >= np.max(raw_lower[row]))
        posterior_ids = np.flatnonzero(
            posterior_upper[row] >= np.max(posterior_lower[row]) - nonzero_score_span,
        )
        selected = np.union1d(raw_ids, posterior_ids).astype(np.int32, copy=False)
        if selected.size == 0:
            return _failed_selection(batch_size, block_capacity, "empty_block_selection")
        if selected.size > block_capacity:
            return _failed_selection(batch_size, block_capacity, "block_capacity_overflow")
        block_ids[row, : selected.size] = selected
        block_count[row] = selected.size
        posterior_count[row] = posterior_ids.size
        raw_count[row] = raw_ids.size
    return CoarseGemmHybridBlockSelection(
        eligible=True,
        fallback_reason=None,
        block_ids=block_ids,
        block_count=block_count,
        posterior_block_count=posterior_count,
        raw_max_block_count=raw_count,
    )
