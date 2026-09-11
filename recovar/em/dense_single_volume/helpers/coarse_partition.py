"""Unused coarse-row partition experiment using the existing exact primitives.

Certification still covers every image/rotation/translation. Only capacity
overflow may split a batch; all other certificate failures retain full scoring.
No engine selects this path. Callers must include certificate, grouping and
posterior/publication work when measuring its cost.
"""

import operator
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    CoarseGemmHybridBlockSelection,
    CoarseGemmHybridIntervalState,
    assemble_coarse_gemm_hybrid_compact_scores_f32,
    select_coarse_gemm_hybrid_rotation_blocks,
    validate_coarse_gemm_certificate_topology,
    validate_coarse_gemm_hybrid_block_selection_for_rescore,
)


class CoarseRowGroup(NamedTuple):
    image_indices: np.ndarray
    physical_count: int
    selection: CoarseGemmHybridBlockSelection | None


class CoarseRowPlan(NamedTuple):
    groups: tuple[CoarseRowGroup, ...]
    fallback_reason: str | None
    partitioned: bool


class CoarseRowResult(NamedTuple):
    image_indices: np.ndarray
    result: object
    translation_log_prior: jax.Array | None


def plan_coarse_rows(state, *, actual_image_count, n_rotations, n_translations, block_capacity=64, row_quantum=32):
    """Keep source image order within two groups; preserve the unsplit fast path."""
    actual = operator.index(actual_image_count)
    capacity = operator.index(block_capacity)
    quantum = operator.index(row_quantum)
    physical = state.raw_block_lower_max.shape[0]
    if not 0 < actual <= physical or capacity <= 0 or quantum <= 0:
        raise ValueError("Invalid coarse row counts or capacity")
    # Publish each small certificate field once, even if capacity overflow
    # requires a second selection pass. Never publish a dense score cube here.
    host_state = CoarseGemmHybridIntervalState(*(np.asarray(x) for x in state))
    config = dict(
        actual_image_count=actual, n_rotations=n_rotations, n_translations=n_translations, certificate_valid=True
    )
    selected = select_coarse_gemm_hybrid_rotation_blocks(host_state, **config, block_capacity=capacity)
    indices = np.arange(actual, dtype=np.int32)
    if selected.eligible:
        return CoarseRowPlan((CoarseRowGroup(indices, physical, selected),), None, False)
    reason = selected.fallback_reason
    if reason != "block_capacity_overflow":
        return CoarseRowPlan((CoarseRowGroup(indices, physical, None),), reason, False)
    complete = select_coarse_gemm_hybrid_rotation_blocks(host_state, **config, block_capacity=n_rotations // 16)
    if not complete.eligible:
        return CoarseRowPlan((CoarseRowGroup(indices, physical, None),), complete.fallback_reason, False)
    safe = np.flatnonzero(complete.block_count[:actual] <= capacity).astype(np.int32)
    full = np.flatnonzero(complete.block_count[:actual] > capacity).astype(np.int32)
    if not safe.size:
        return CoarseRowPlan((CoarseRowGroup(indices, physical, None),), reason, False)
    groups = []
    for rows, is_selected in ((safe, True), (full, False)):
        size = ((len(rows) + quantum - 1) // quantum) * quantum
        subset = None
        if is_selected:
            ids = np.full((size, capacity), -1, np.int32)
            ids[: len(rows)] = complete.block_ids[rows, :capacity]
            count_fields = []
            for field in (complete.block_count, complete.posterior_block_count, complete.raw_max_block_count):
                values = np.zeros(size, np.int32)
                values[: len(rows)] = field[rows]
                count_fields.append(values)
            subset = CoarseGemmHybridBlockSelection(True, None, ids, *count_fields)
            validate_coarse_gemm_hybrid_block_selection_for_rescore(
                subset, actual_image_count=len(rows), n_rotations=n_rotations
            )
        groups.append(CoarseRowGroup(rows, size, subset))
    return CoarseRowPlan(tuple(groups), reason, True)


@partial(jax.jit, static_argnames=("physical_count",))
def gather_coarse_rows(shifted, weight, initial, translation_prior, indices, *, physical_count):
    """One device gather/pad for image operands; padded rows contain finite zeros."""
    actual = indices.shape[0]
    padded = jnp.pad(indices, ((0, physical_count - actual),))
    valid = jnp.arange(physical_count, dtype=jnp.int32) < actual

    def gather(value):
        mask = valid.reshape((physical_count,) + (1,) * (value.ndim - 1))
        return jnp.where(mask, value[padded], jnp.zeros((), value.dtype))

    return (
        gather(shifted),
        gather(weight),
        gather(initial),
        None if translation_prior is None else gather(translation_prior),
    )


def compute_partitioned_coarse_batch(
    projection_cache,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    *,
    topology,
    actual_image_count,
    class_log_prior=0.0,
    rotation_log_prior=None,
    translation_log_prior=None,
    certificate_chunk_rows=4608,
    block_capacity=64,
    logical_full_pixel_count=None,
    real_cross=False,
    row_quantum=32,
    capture_selected_diff2=False,
):
    """Certify once, then score at most two substantial row groups.

    Results stay grouped because dense and compact candidate tables have different
    source-pose mappings. Posterior processing must use each result's original
    mapping before restoring the image order. No full score expansion is needed.
    """
    from recovar.em.dense_single_volume.helpers import significance as s

    validate_coarse_gemm_certificate_topology(topology)
    cache, shifted, weight, initial = map(
        jnp.asarray, (projection_cache, shifted_corrected, pixel_weight, initial_diff2)
    )
    if cache.ndim != 3 or cache.shape[0] != 1 or shifted.ndim != 3:
        raise ValueError("Expected [1,R,P] cache and [B,T,P] images")
    batch, translations, pixels = shifted.shape
    rotations = cache.shape[1]
    actual = operator.index(actual_image_count)
    chunk = operator.index(certificate_chunk_rows)
    if (
        cache.dtype != jnp.complex64
        or shifted.dtype != jnp.complex64
        or weight.dtype != jnp.float32
        or initial.dtype != jnp.float32
        or weight.shape != (batch, pixels)
        or initial.shape != (batch,)
        or cache.shape[2] != pixels
        or rotations <= 0
        or rotations % 16
        or topology.compact_pixel_count != pixels
        or topology.translation_count != translations
        or not 0 < actual <= batch
        or chunk <= 0
        or chunk % 16
    ):
        raise ValueError("Inconsistent coarse operands, topology or counts")
    if not isinstance(real_cross, (bool, np.bool_)):
        raise TypeError("real_cross must be boolean")
    rp = None if rotation_log_prior is None else jnp.asarray(rotation_log_prior, jnp.float32)
    tp = None if translation_log_prior is None else jnp.asarray(translation_log_prior, jnp.float32)
    if rp is not None and rp.shape != (rotations,):
        raise ValueError("Rotation priors must match source rotations")
    if tp is not None and tp.shape != (batch, translations):
        raise ValueError("Translation priors must match image/translation dimensions")
    images = s._prepare_relion_coarse_gaussian_gemm_f64_image_batch(shifted, weight, initial, actual)
    state = s.initialize_coarse_gemm_hybrid_interval_state(batch, rotations)
    for start in range(0, rotations, chunk):
        stop = min(start + chunk, rotations)
        state = s._relion_coarse_gaussian_gemm_update_certificate_state(
            state,
            cache[0, start:stop],
            images,
            topology=topology,
            rotation_offset=start,
            class_log_prior=class_log_prior,
            rotation_log_prior=None if rp is None else rp[start:stop],
            translation_log_prior=tp,
            real_cross=real_cross,
        )
    plan = plan_coarse_rows(
        state,
        actual_image_count=actual,
        n_rotations=rotations,
        n_translations=translations,
        block_capacity=block_capacity,
        row_quantum=row_quantum,
    )
    results = []
    for group in plan.groups:
        if plan.partitioned:
            operands = gather_coarse_rows(
                shifted, weight, initial, tp, jnp.asarray(group.image_indices), physical_count=group.physical_count
            )
        else:
            operands = shifted, weight, initial, tp
        x, w, ini, prior = operands
        result = None
        reason = plan.fallback_reason
        if group.selection is not None:
            kwargs = dict(topology=topology)
            if logical_full_pixel_count is not None:
                kwargs["logical_full_pixel_count"] = logical_full_pixel_count
            diff2 = s._relion_coarse_diff2_rotation_blocks_from_topology_f32(
                cache[0], x, w, ini, jnp.asarray(group.selection.block_ids), **kwargs
            )
            compact = assemble_coarse_gemm_hybrid_compact_scores_f32(
                diff2,
                group.selection,
                actual_image_count=len(group.image_indices),
                n_rotations=rotations,
                class_log_prior=class_log_prior,
                rotation_log_prior=rp,
                translation_log_prior=prior,
            )
            if np.all(np.asarray(compact.selected_output_valid, dtype=bool)):
                result = s.CoarseGaussianGemmHybridBatchResult(
                    scores=None,
                    raw_score_max=jnp.where(
                        jnp.arange(group.physical_count, dtype=jnp.int32) < len(group.image_indices),
                        compact.raw_score_max,
                        jnp.float32(0),
                    ),
                    scores_include_priors=True,
                    used_selected_rescore=True,
                    fallback_reason=None,
                    selection=group.selection,
                    compact_scores=compact,
                    score_representation="compact_selected_exact",
                    diagnostic_selected_diff2=diff2 if capture_selected_diff2 else None,
                )
            else:
                reason = "invalid_selected_exact_output"
        if result is None:
            result = s._compute_coarse_gaussian_gemm_hybrid_batch(
                cache,
                x,
                w,
                ini,
                topology=topology,
                actual_image_count=len(group.image_indices),
                class_log_prior=class_log_prior,
                rotation_log_prior=rp,
                translation_log_prior=prior,
                certificate_chunk_rows=chunk,
                block_capacity=block_capacity,
                compact_posterior=True,
                force_static_dense_after_overflow=True,
                logical_full_pixel_count=logical_full_pixel_count,
                real_cross=real_cross,
            )._replace(fallback_reason=reason)
        results.append(CoarseRowResult(group.image_indices, result, prior))
    return plan, tuple(results)
