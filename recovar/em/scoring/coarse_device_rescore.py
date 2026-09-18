"""Opt-in composed certificate, source16 CUDA rescore and compact assembly.

Used only by explicitly opted-in routes. Only the mature CUDA scorer produces
published scores; the certificate is used solely to select ordered source blocks.
"""

import operator
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.scoring.coarse_device_certificate import (
    _certify_coarse_rotation_blocks_jit,
    _prepare_coarse_certificate_inputs,
)
from recovar.em.scoring.coarse_device_selection import SELECTION_REASONS, DeviceCoarseBlockSelection
from recovar.em.scoring.coarse_gemm_hybrid import (
    DEFAULT_ROTATION_BLOCK_CAPACITY,
    SOURCE_ROTATION_BLOCK_SIZE,
    CoarseGemmHybridCompactScores,
    _assemble_coarse_gemm_hybrid_compact_scores_f32_jit,
)

RESCORE_REASONS = SELECTION_REASONS + (
    "invalid_selected_exact_output",
    "invalid_runtime_prefix_contract",
)


STATUS_VALID = 0
STATUS_REASON = 1
STATUS_USED_SELECTED_RESCORE = 2
STATUS_BLOCK_SUM = 3
STATUS_BLOCK_MAX = 4


class DeviceCoarseRescore(NamedTuple):
    selection: DeviceCoarseBlockSelection
    compact_scores: CoarseGemmHybridCompactScores
    status: jax.Array  # int64[5], one 40-byte device-to-host route/telemetry read
    selected_diff2: jax.Array | None  # optional diagnostic, never approximate scores


def _empty_compact(batch, capacity, translations):
    return CoarseGemmHybridCompactScores(
        posterior_scores_flat=jnp.full(
            (batch, capacity * SOURCE_ROTATION_BLOCK_SIZE * translations), -jnp.inf, jnp.float32
        ),
        source_block_ids=jnp.full((batch, capacity), -1, jnp.int32),
        block_count=jnp.zeros(batch, jnp.int32),
        raw_score_max=jnp.full(batch, -jnp.inf, jnp.float32),
        min_diff2_offsets=jnp.zeros(batch, jnp.float32),
        best_score=jnp.full(batch, -jnp.inf, jnp.float32),
        # Existing assembly uses the x64 argmax index dtype for best_pose.
        best_pose=jnp.zeros(batch, jnp.int64),
        selected_output_valid=jnp.zeros(batch, jnp.bool_),
    )


@partial(jax.jit, static_argnames=("chunk_rows", "block_capacity", "capture_selected_diff2"))
def _rescore_coarse_rotation_blocks_jit(
    operands, mapping, logical_count, *, chunk_rows, block_capacity, capture_selected_diff2=False
):
    reference, shifted, weight, initial, actual, class_prior, rotation_prior, translation_prior = operands[:8]
    selection = _certify_coarse_rotation_blocks_jit(*operands, chunk_rows=chunk_rows, block_capacity=block_capacity)
    batch, translations, pixels = shifted.shape
    prefix_valid = jnp.bool_(True)
    if logical_count is not None:
        # The runtime scorer packs rows at stride L. Its first L lookup entries
        # must address values within that stride. The sealed topology already
        # guarantees unique nonnegative IDs; -1 entries are legal holes.
        positions = jnp.arange(mapping.size) < logical_count
        lookup_valid = jnp.all(~positions | (mapping < logical_count))
        visited_ids = jnp.where(positions & (mapping >= 0), mapping, jnp.int32(pixels))
        visited = jnp.zeros(pixels, jnp.bool_).at[visited_ids].set(True, mode="drop")
        active_images = jnp.arange(batch) < actual
        zero_omitted_weight = jnp.all(~(active_images[:, None] & ~visited[None, :]) | (weight == 0))
        prefix_valid = (
            (logical_count > 0)
            & (logical_count <= pixels)
            & (logical_count <= mapping.size)
            & lookup_valid
            & zero_omitted_weight
        )
    use_selected = selection.eligible & prefix_valid
    empty = _empty_compact(batch, block_capacity, translations)
    empty_diff2 = (
        jnp.full((batch, block_capacity, SOURCE_ROTATION_BLOCK_SIZE, translations), jnp.inf, jnp.float32)
        if capture_selected_diff2
        else None
    )

    def rescore(_):
        args = (reference, shifted, weight, initial, selection.block_ids, mapping)
        if logical_count is None:
            diff2 = cuda_backproject.relion_coarse_diff2_rotation_blocks_f32(*args)
        else:
            diff2 = cuda_backproject.relion_coarse_diff2_rotation_blocks_runtime_f32(*args, logical_count)
        rotations = reference.shape[0]
        rp = jnp.zeros(rotations, jnp.float32) if rotation_prior is None else rotation_prior
        if translation_prior is None:
            tp = jnp.zeros((batch, translations), jnp.float32)
        else:
            tp = jnp.broadcast_to(translation_prior, (batch, translations))
        compact = _assemble_coarse_gemm_hybrid_compact_scores_f32_jit(
            diff2,
            selection.block_ids,
            selection.block_count,
            actual,
            class_prior,
            rp,
            tp,
            jnp.bool_(rotation_prior is not None),
            jnp.bool_(translation_prior is not None),
        )
        return compact, diff2 if capture_selected_diff2 else None

    compact, selected_diff2 = jax.lax.cond(use_selected, rescore, lambda _: (empty, empty_diff2), operand=None)
    valid = use_selected & jnp.all(compact.selected_output_valid)
    reason = jnp.where(selection.eligible, jnp.where(prefix_valid, jnp.where(valid, 0, 11), 12), selection.reason_code)
    # A failed row invalidates the complete selected result. Keep the selection
    # separately for diagnostics, but never leave partially usable score rows.
    compact = jax.tree.map(lambda value, blank: jnp.where(valid, value, blank), compact, empty)
    status = jnp.stack(
        tuple(
            jnp.asarray(value, dtype=jnp.int64)
            for value in (
                valid,
                reason,
                use_selected,
                jnp.sum(selection.block_count, dtype=jnp.int32),
                jnp.max(selection.block_count),
            )
        )
    )
    return DeviceCoarseRescore(selection, compact, status, selected_diff2)


def rescore_coarse_rotation_blocks(
    projections,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    topology,
    class_log_prior,
    rotation_log_prior=None,
    translation_log_prior=None,
    chunk_rows=256,
    block_capacity=DEFAULT_ROTATION_BLOCK_CAPACITY,
    logical_full_pixel_count=None,
    capture_selected_diff2=False,
) -> DeviceCoarseRescore:
    """Compose selected exact scoring with a small whole-batch route result.

    Admission reads only immutable host topology and operand metadata. Active
    image count and optional logical pixel count are device S32 scalars; all
    other configuration is static. The topology object never enters a JIT key.

    Runtime-prefix mode additionally requires nonnegative lookup[0:L] IDs to
    lie within [0,L), and active-image weights on every unvisited compact column
    to be zero. The lookup may have -1 holes. These device checks bind the
    full-capacity certificate to the truncated scorer, and protect packed row
    strides. The ordinary stable square layout satisfies this contract. Invalid
    prefix contracts return reason12 without executing CUDA. Static mode retains
    the mature general compact lookup, including -1 holes.

    Status is one int64[5] buffer indexed by STATUS_* constants. A false
    status[STATUS_VALID] requires the caller's existing lazy whole-batch direct
    fallback. There is no full-dense buffer, posterior calculation or fallback
    implementation here. Failure scores are -inf with zero counts/-1 IDs. This
    primitive still requires CUDA availability when tracing either cond branch;
    skipped execution does not make this a CPU scoring implementation.

    Optional static capture_selected_diff2 retains the same CUDA output without
    rescoring. It is None when disabled, an all-+inf diagnostic buffer when the
    selected branch was skipped, or the actual output (including invalid values)
    when executed. STATUS_USED_SELECTED_RESCORE distinguishes these cases.
    """
    operands = _prepare_coarse_certificate_inputs(
        projections,
        shifted_corrected,
        pixel_weight,
        initial_diff2,
        actual_image_count,
        topology=topology,
        class_log_prior=class_log_prior,
        rotation_log_prior=rotation_log_prior,
        translation_log_prior=translation_log_prior,
        chunk_rows=chunk_rows,
    )
    capacity = operator.index(block_capacity)
    if not isinstance(capture_selected_diff2, (bool, np.bool_)):
        raise TypeError("capture_selected_diff2 must be a static boolean")
    batch, translations, _pixels = operands[1].shape
    if (
        capacity <= 0
        or translations > 128
        or batch * capacity * SOURCE_ROTATION_BLOCK_SIZE * translations > np.iinfo(np.int32).max
    ):
        raise ValueError("invalid selected-rescore capacity, translation count or output size")
    actual = operands[4]
    if getattr(actual, "weak_type", False):
        raise TypeError("actual_image_count must be a strong scalar int32")
    logical = None
    if logical_full_pixel_count is not None:
        if isinstance(logical_full_pixel_count, (int, np.integer)):
            logical_full_pixel_count = operator.index(logical_full_pixel_count)
            if not 0 < logical_full_pixel_count <= min(topology.full_position_count, operands[0].shape[1]):
                raise ValueError("logical_full_pixel_count is outside stored capacity")
            logical_full_pixel_count = np.int32(logical_full_pixel_count)
        logical = jnp.asarray(logical_full_pixel_count)
        if logical.shape != () or logical.dtype != jnp.int32 or getattr(logical, "weak_type", False):
            raise TypeError("logical_full_pixel_count must be a strong scalar int32")
    with jax.enable_x64(True):
        return _rescore_coarse_rotation_blocks_jit(
            operands,
            jnp.asarray(topology.full_to_compact, dtype=jnp.int32),
            logical,
            chunk_rows=operator.index(chunk_rows),
            block_capacity=capacity,
            capture_selected_diff2=bool(capture_selected_diff2),
        )
