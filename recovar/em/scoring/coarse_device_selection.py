"""Device counterpart of the certified host coarse source-block selector.

This is an opt-in primitive, not a production route. Configuration is static
except the optional device image-prefix count. Compose this function inside an
outer JIT. No interval values or result values are transferred to the host.
"""

import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.scoring.coarse_gemm_hybrid import (
    DEFAULT_ROTATION_BLOCK_CAPACITY,
    RELION_COARSE_NONZERO_SCORE_SPAN,
    SOURCE_ROTATION_BLOCK_SIZE,
    CoarseGemmHybridBlockSelection,
    CoarseGemmHybridIntervalState,
)

# Zero means success. The remaining names are the unchanged host fallback
# reasons, in validation order (configuration conversion happens first).
SELECTION_REASONS = (
    None,
    "invalid_selection_configuration",
    "missing_or_invalid_certificate_contract",
    "rotation_tail_not_supported",
    "incomplete_block_table",
    "incomplete_or_duplicate_rotation_coverage",
    "incomplete_candidate_coverage",
    "invalid_or_nonfinite_candidate_interval",
    "invalid_or_nonfinite_block_interval",
    "empty_block_selection",
    "block_capacity_overflow",
)


class DeviceCoarseBlockSelection(NamedTuple):
    """Device scalars and bounded IDs/counts; failed batches are entirely clear."""

    eligible: jax.Array
    reason_code: jax.Array
    block_ids: jax.Array
    block_count: jax.Array
    posterior_block_count: jax.Array
    raw_max_block_count: jax.Array


def _failed(batch_size, capacity, reason):
    return DeviceCoarseBlockSelection(
        jnp.bool_(False),
        jnp.int32(SELECTION_REASONS.index(reason)),
        jnp.full((batch_size, max(1, capacity)), -1, dtype=jnp.int32),
        jnp.zeros(batch_size, dtype=jnp.int32),
        jnp.zeros(batch_size, dtype=jnp.int32),
        jnp.zeros(batch_size, dtype=jnp.int32),
    )


def select_coarse_rotation_blocks_device(
    state: CoarseGemmHybridIntervalState,
    *,
    actual_image_count: int,
    n_rotations: int,
    n_translations: int,
    certificate_valid: bool | None = None,
    block_capacity: int = DEFAULT_ROTATION_BLOCK_CAPACITY,
    nonzero_score_span: float = RELION_COARSE_NONZERO_SCORE_SPAN,
) -> DeviceCoarseBlockSelection:
    """Select exactly the host selector's sorted union using FP64 thresholds.

    The valid image prefix may be a host integer or a strong int32 device scalar;
    other keyword arguments must be static host configuration. Admission returns
    the same failed buffers/reason as the host selector. Arrays retain the host's
    casts (visits int32, counts int64,
    intervals float64). Inactive image rows are ignored, including nonfinite data.

    Unlike the host API, results remain on device and JAX x64 must be enabled;
    disabling it raises rather than silently rounding the FP64 selection boundary.
    Input fields must be JAX-compatible numeric arrays with shape metadata. Invalid
    array types are outside this primitive's contract, as are other dynamic
    configuration scalars. Storage is O(B * number_of_source_blocks +
    B * block_capacity), without a rotation/translation score surface or
    Cartesian mask expansion.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Device coarse selection requires JAX x64 for host-exact thresholds")
    batch_size = state.raw_block_lower_max.shape[0]
    dynamic_count = hasattr(actual_image_count, "shape") and hasattr(actual_image_count, "dtype")
    try:
        if dynamic_count:
            if (
                actual_image_count.shape != ()
                or actual_image_count.dtype != np.dtype(np.int32)
                or getattr(actual_image_count, "weak_type", False)
            ):
                return _failed(batch_size, 1, "invalid_selection_configuration")
        else:
            actual_image_count = operator.index(actual_image_count)
        n_rotations = operator.index(n_rotations)
        n_translations = operator.index(n_translations)
        block_capacity = operator.index(block_capacity)
        nonzero_score_span = float(nonzero_score_span)
    except (TypeError, ValueError, OverflowError):
        return _failed(batch_size, 1, "invalid_selection_configuration")
    if not isinstance(certificate_valid, (bool, np.bool_)) or not certificate_valid:
        return _failed(batch_size, block_capacity, "missing_or_invalid_certificate_contract")
    if n_rotations <= 0 or n_rotations % SOURCE_ROTATION_BLOCK_SIZE:
        return _failed(batch_size, block_capacity, "rotation_tail_not_supported")
    if (
        batch_size <= 0
        or (not dynamic_count and (actual_image_count <= 0 or actual_image_count > batch_size))
        or n_translations <= 0
        or block_capacity <= 0
        or not np.isfinite(nonzero_score_span)
        or nonzero_score_span < 0.0
    ):
        return _failed(batch_size, block_capacity, "invalid_selection_configuration")
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
        result = _failed(batch_size, block_capacity, "incomplete_block_table")
        # Invalid image configuration precedes malformed table shape in the
        # reference, even when image count is supplied as a runtime scalar.
        if dynamic_count:
            return result._replace(
                reason_code=jnp.where(
                    (actual_image_count > 0) & (actual_image_count <= batch_size), result.reason_code, jnp.int32(1)
                )
            )
        return result

    visits = jnp.asarray(state.rotation_visit_count, dtype=jnp.int32)
    active = jnp.arange(batch_size) < actual_image_count
    counts = jnp.asarray(state.candidate_count, dtype=jnp.int64)
    invalid = jnp.asarray(state.invalid_candidate_count, dtype=jnp.int64)
    raw_lower, raw_upper, posterior_lower, posterior_upper = (
        jnp.asarray(field, dtype=jnp.float64) for field in block_fields
    )
    intervals_valid = jnp.all(
        ~active[:, None]
        | (
            jnp.isfinite(raw_lower)
            & jnp.isfinite(raw_upper)
            & jnp.isfinite(posterior_lower)
            & jnp.isfinite(posterior_upper)
            & (raw_lower <= raw_upper)
            & (posterior_lower <= posterior_upper)
        )
    )
    raw_mask = active[:, None] & (raw_upper >= jnp.max(raw_lower, axis=1, keepdims=True))
    posterior_mask = active[:, None] & (
        posterior_upper >= (jnp.max(posterior_lower, axis=1, keepdims=True) - jnp.float64(nonzero_score_span))
    )
    selected = raw_mask | posterior_mask
    count = jnp.sum(selected, axis=1, dtype=jnp.int32)
    raw_count = jnp.sum(raw_mask, axis=1, dtype=jnp.int32)
    posterior_count = jnp.sum(posterior_mask, axis=1, dtype=jnp.int32)
    # nonzero preserves ascending source IDs. Check the untruncated count;
    # overflow is never accepted even though the output buffer is bounded.
    ids = jax.vmap(lambda mask: jnp.nonzero(mask, size=block_capacity, fill_value=-1)[0])(selected)
    ids = ids.astype(jnp.int32)
    row_reason = jnp.where(active, jnp.where(count == 0, 9, jnp.where(count > block_capacity, 10, 0)), 0)
    # The host returns on the first failed active row, after global validation.
    first_bad = jnp.argmax(row_reason != 0)
    reason = jnp.where(jnp.any(row_reason != 0), row_reason[first_bad], 0)
    expected_candidates = n_rotations * n_translations
    coverage_valid = (
        jnp.all(~active | (counts == expected_candidates))
        if expected_candidates <= np.iinfo(np.int64).max
        else jnp.bool_(False)
    )
    for valid, failure in (
        (intervals_valid, 8),
        (jnp.all(~active | (invalid == 0)), 7),
        (coverage_valid, 6),
        (jnp.all(visits == 1), 5),
        ((actual_image_count > 0) & (actual_image_count <= batch_size), 1),
    ):
        reason = jnp.where(valid, reason, failure)
    eligible = reason == 0
    return DeviceCoarseBlockSelection(
        eligible,
        reason.astype(jnp.int32),
        jnp.where(eligible, ids, -1),
        jnp.where(eligible, count, 0),
        jnp.where(eligible, posterior_count, 0),
        jnp.where(eligible, raw_count, 0),
    )


def decode_device_coarse_selection(result: DeviceCoarseBlockSelection) -> CoarseGemmHybridBlockSelection:
    """Explicit host transfer for diagnostics only; never call inside a JIT."""
    eligible, code, ids, counts, posterior, raw = jax.device_get(result)
    return CoarseGemmHybridBlockSelection(bool(eligible), SELECTION_REASONS[int(code)], ids, counts, posterior, raw)
