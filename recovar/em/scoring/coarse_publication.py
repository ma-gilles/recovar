"""Shared posterior and source-order publication for grouped K=1 coarse work."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.oversampling import relion_cuda_f32_coarse_posterior
from recovar.em.scoring.coarse_gemm_hybrid import map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids


def _packed_support_prefix(support, count):
    """Copy a bounded prefix; bucket lengths avoid one slice executable per count.

    The count is already on the host. The transfer includes at most twice the
    live support, capped by the full output capacity. It never truncates ties.
    Count synchronization and any slice compilation belong in publication timing.
    """
    if not 0 <= count <= support.size:
        raise ValueError("Packed coarse support count exceeds capacity")
    if count == 0:
        return np.empty(0, np.int32)
    capacity = min(support.size, 1 << (count - 1).bit_length())
    return np.asarray(jax.device_get(support[:capacity]))[:count]


def _cuda_posterior_host(values, raw_max, compact, actual, *, adaptive_fraction, max_significants):
    """Publish the explicit CUDA primitive without a full posterior mask transfer."""
    from recovar import cuda_backproject

    statistics, indices, support, count = cuda_backproject.relion_coarse_posterior_transaction_f32(
        values,
        raw_max,
        jnp.asarray(actual, jnp.int32),
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
    )
    blocks = None if compact is None else compact.source_block_ids
    block_count = None if compact is None else compact.block_count
    statistics, indices, count, blocks, block_count = jax.device_get((statistics, indices, count, blocks, block_count))
    physical, width = values.shape
    if (
        statistics.shape != (physical, 4)
        or statistics.dtype != np.float32
        or indices.shape != (physical, 4)
        or indices.dtype != np.int32
        or np.shape(count) != ()
        or np.asarray(count).dtype != np.int32
        or support.shape != (physical * width,)
        or support.dtype != np.int32
    ):
        raise ValueError("Invalid packed coarse posterior output contract")
    count = int(count)
    counts = indices[:, 2]
    if (
        np.any(counts < 0)
        or np.any(counts > width)
        or np.any(counts[actual:] != 0)
        or int(counts.sum(dtype=np.int64)) != count
    ):
        raise ValueError("Packed coarse support count disagrees with row counts")
    positions = _packed_support_prefix(support, count).astype(np.int64)
    if (
        np.any(positions < 0)
        or np.any(positions >= actual * width)
        or np.any(np.diff(positions) <= 0)
        or not np.array_equal(np.bincount(positions // width, minlength=physical), counts)
    ):
        raise ValueError("Packed coarse support must be ordered and match actual rows")
    if blocks is not None:
        if (
            blocks.ndim != 2
            or blocks.shape[0] != physical
            or blocks.dtype != np.int32
            or blocks.shape[1] <= 0
            or width % (16 * blocks.shape[1])
            or block_count.shape != (physical,)
            or block_count.dtype != np.int32
        ):
            raise ValueError("Invalid compact coarse source block geometry")
        for row, n in enumerate(block_count):
            if (
                not 0 <= n <= blocks.shape[1]
                or np.any(blocks[row, :n] < 0)
                or np.any(np.diff(blocks[row, :n]) <= 0)
                or np.any(blocks[row, n:] != -1)
            ):
                raise ValueError("Compact coarse source blocks must be canonical ordered prefixes")

    def global_pose(rows, local):
        if np.any(local < 0) or np.any(local >= width):
            raise ValueError("Packed coarse pose index is outside the score table")
        if blocks is None:
            return local
        per_slot = width // blocks.shape[1]
        slots = local // per_slot
        if np.any(slots >= block_count[rows]):
            raise ValueError("Packed coarse pose selects an inactive compact slot")
        return blocks[rows, slots].astype(np.int64) * per_slot + local % per_slot

    rows = np.arange(actual)
    index_dtype = np.int64 if jax.config.x64_enabled else np.int32
    host = {key: statistics[:, col] for col, key in enumerate(("best_score", "pmax", "sum_weight", "threshold"))}
    for col, key in enumerate(("best_pose", "winner")):
        host[key] = global_pose(rows, indices[:actual, col].astype(np.int64)).astype(index_dtype)
    host["n_significant"] = counts
    host["cutoff_count"] = indices[:, 3]
    ids = global_pose(positions // width, positions % width)
    poses = np.split(ids, np.cumsum(counts[:actual], dtype=np.int64)[:-1])
    return host, poses


@jax.jit
def _dense_prior_scores(raw, class_prior, rotation_prior, translation_prior, actual):
    scores = raw + jnp.asarray(class_prior, jnp.float32)
    if rotation_prior is not None:
        scores = scores + jnp.asarray(rotation_prior, jnp.float32)[None, :, None]
    if translation_prior is not None:
        scores = scores + jnp.asarray(translation_prior, jnp.float32)[:, None, :]
    scores = jnp.where(jnp.arange(scores.shape[0])[:, None, None] < actual, scores, -jnp.inf)
    return scores.reshape(scores.shape[0], -1)


@partial(jax.jit, static_argnames=("adaptive_fraction", "max_significants", "tie_score_ulps"))
def _posterior_statistics(values, raw_max, source_blocks, *, adaptive_fraction, max_significants, tie_score_ulps):
    probabilities, mask, count, cutoff, total, threshold = relion_cuda_f32_coarse_posterior(
        values,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        tie_score_ulps=tie_score_ulps,
        min_diff2_offsets=-raw_max,
        filter_positive_before_sort=False,
    )
    winner = jnp.argmax(probabilities, axis=1)
    best = jnp.argmax(values, axis=1)
    if source_blocks is not None:
        per_slot = values.shape[1] // source_blocks.shape[1]

        def global_pose(local):
            block = jnp.take_along_axis(source_blocks, (local // per_slot)[:, None], axis=1)[:, 0]
            return block * per_slot + local % per_slot

        winner = global_pose(winner)
        best = global_pose(best)
    return dict(
        mask=mask,
        winner=winner,
        best_pose=best,
        best_score=jnp.max(values, axis=1),
        pmax=jnp.max(probabilities, axis=1),
        n_significant=count,
        cutoff_count=cutoff,
        sum_weight=total,
        threshold=threshold,
    )


def publish_coarse_rows(
    groups,
    *,
    actual_image_count,
    n_rotations,
    n_translations,
    class_log_prior,
    rotation_log_prior,
    rotation_chunk_rows,
    adaptive_fraction,
    max_significants,
    tie_score_ulps,
    posterior_backend="jax",
):
    """Publish complete coarse statistics without expanding compact score tables.

    Groups carry original image indices and source-block mappings. Dense groups
    retain the existing rotation-chunk logsumexp order; compact groups use the
    existing single-table reduction. Only final statistics/support cross to the
    host, matching the significance engine's public output boundary.
    """
    from recovar.em.scoring.significance import _update_logsumexp

    if posterior_backend not in ("jax", "cuda"):
        raise ValueError("Unknown coarse posterior backend")
    if posterior_backend == "cuda" and tie_score_ulps != 0:
        raise ValueError("CUDA coarse posterior requires exact ties (tie_score_ulps=0)")
    if actual_image_count <= 0 or n_rotations <= 0 or n_translations <= 0 or rotation_chunk_rows <= 0:
        raise ValueError("Positive image, candidate and chunk counts are required")
    indices = [np.asarray(group.image_indices) for group in groups]
    if (
        not indices
        or any(rows.ndim != 1 or rows.dtype.kind not in "iu" or not rows.size for rows in indices)
        or not np.array_equal(np.sort(np.concatenate(indices)), np.arange(actual_image_count))
    ):
        raise ValueError("Coarse groups must cover each original image exactly once")
    arrays = {}
    support = [None] * actual_image_count
    for group in groups:
        result = group.result
        compact = result.compact_scores
        actual = len(group.image_indices)
        if compact is not None:
            values = compact.posterior_scores_flat
            blocks = compact.source_block_ids
        else:
            if result.scores_include_priors:
                raise ValueError("Dense grouped scores must precede prior addition")
            if result.scores.shape[1:] != (n_rotations, n_translations):
                raise ValueError("Dense grouped scores have inconsistent candidate geometry")
            values = _dense_prior_scores(
                result.scores, class_log_prior, rotation_log_prior, group.translation_log_prior, actual
            )
            blocks = None
        physical = values.shape[0]
        if actual > physical or result.raw_score_max.shape != (physical,):
            raise ValueError("Coarse score rows and maxima must cover the actual group")
        maximum = jnp.full(physical, -jnp.inf)
        total = jnp.zeros(physical, jnp.float64)
        width = values.shape[1] if compact is not None else rotation_chunk_rows * n_translations
        for start in range(0, values.shape[1], width):
            maximum, total = _update_logsumexp(maximum, total, values[:, start : start + width])
        if posterior_backend == "cuda":
            host, poses = _cuda_posterior_host(
                values,
                result.raw_score_max,
                compact,
                actual,
                adaptive_fraction=float(adaptive_fraction),
                max_significants=max_significants,
            )
            host.update(jax.device_get(dict(global_log_z=maximum + jnp.log(total), raw_max=result.raw_score_max)))
        else:
            device = _posterior_statistics(
                values,
                result.raw_score_max,
                blocks,
                adaptive_fraction=float(adaptive_fraction),
                max_significants=max_significants,
                tie_score_ulps=int(tie_score_ulps),
            )
            device["global_log_z"] = maximum + jnp.log(total)
            device["raw_max"] = result.raw_score_max
            host = jax.device_get(device)
            mask = np.asarray(host.pop("mask"), dtype=bool)
            if compact is not None:
                poses = map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(compact, mask)
            else:
                poses = [np.flatnonzero(row).astype(np.int64) for row in mask[:actual]]
        for key, value in host.items():
            if not np.isfinite(value[:actual]).all():
                raise ValueError(f"Nonfinite actual coarse statistic: {key}")
            if key not in arrays:
                arrays[key] = np.empty(actual_image_count, dtype=value.dtype)
            arrays[key][group.image_indices] = value[:actual]
        for local, image in enumerate(group.image_indices):
            ids = np.asarray(poses[local], np.int64)
            if (
                len(ids) != int(host["n_significant"][local])
                or not np.all((ids >= 0) & (ids < n_rotations * n_translations))
                or not np.all(np.diff(ids) > 0)
            ):
                raise ValueError("Coarse support must contain the counted, ordered source poses")
            support[int(image)] = ids
    lengths = np.asarray([len(ids) for ids in support], np.int64)
    arrays["support_offsets"] = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths)))
    arrays["support_ids"] = np.concatenate(support)
    return arrays
