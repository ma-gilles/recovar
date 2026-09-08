"""Shared posterior and source-order publication for grouped K=1 coarse work."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids,
)
from recovar.em.dense_single_volume.helpers.oversampling import relion_cuda_f32_coarse_posterior


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
):
    """Publish complete coarse statistics without expanding compact score tables.

    Groups carry original image indices and source-block mappings. Dense groups
    retain the existing rotation-chunk logsumexp order; compact groups use the
    existing single-table reduction. Only final statistics/support cross to the
    host, matching the significance engine's public output boundary.
    """
    from recovar.em.dense_single_volume.helpers.significance import _update_logsumexp

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
