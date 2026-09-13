"""Host planning and array assembly for rectangular and compact sparse buckets.

Callers select execution policies and budgets. These helpers group supplied
support into bounded buckets, expand parent support and gather/pad rows. They
preserve particle order, precision, scoring/M-step aliases and inert padding;
they neither execute scoring nor choose scientific or device policies.
"""

from __future__ import annotations

import os

import numpy as np

from recovar.em.dense_single_volume.batch_planning import _plan_consecutive_padded_batches
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_binary_flag
from recovar.em.dense_single_volume.helpers.compact_candidates import (
    SparseCandidateMask,
    _candidate_mask_to_dense,
    build_compact_pair_index_arrays,
    compact_candidate_indices_in_source_order,
    compact_pair_index_arrays_device,
)
from recovar.em.dense_single_volume.helpers.significant_samples import ComplementSignificantSampleIndices
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size

_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH = 1_000_000
_DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION = 2.0
_DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE = 4096


LADDER_CHUNKS_ENV = "RECOVAR_SPARSE_PASS2_LADDER_CHUNKS"
LADDER_CHUNK_FLOOR = 16

IMAGE_CAPACITY_ENV = "RECOVAR_SPARSE_PASS2_IMAGE_CAPACITY"
IMAGE_CAPACITY_FLOOR = 16
IMAGE_CAPACITY_MAX_GROWTH_ENV = "RECOVAR_SPARSE_PASS2_IMAGE_CAPACITY_MAX_GROWTH"
DEFAULT_IMAGE_CAPACITY_MAX_GROWTH = 2.0


def image_capacity_max_growth() -> float:
    """Return the largest factor by which the image axis may be padded."""

    raw = os.environ.get(IMAGE_CAPACITY_MAX_GROWTH_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_IMAGE_CAPACITY_MAX_GROWTH
    value = float(raw)
    if not value >= 1.0:
        raise ValueError(f"{IMAGE_CAPACITY_MAX_GROWTH_ENV} must be at least 1.0, got {raw!r}")
    return value


def image_capacity_enabled() -> bool:
    """Return whether bucket image axes are padded to a repeating capacity."""

    return parse_env_binary_flag(IMAGE_CAPACITY_ENV)


def quantized_image_capacity(
    n_images: int,
    *,
    max_images: int | None = None,
    floor: int = IMAGE_CAPACITY_FLOOR,
    max_growth: float | None = None,
) -> int:
    """Round a bucket's image count up to a power of two so shapes repeat.

    Fused K-class pass 2 keys one XLA program per (helper, shape), and the image
    axis is part of every shape. Measured 2026-09-12 on a 20-iteration exactly-K4
    run: 128 buckets produced 88 distinct (pair width, rotation rows, images)
    shapes across 62 distinct image counts, and JAX traced and lowered a program
    for 5939 distinct keys, executing each exactly once. Quantizing this axis
    collapses those 88 shapes to 38. Padded rows carry ``pair_mask``/
    ``candidate_mask`` false, and
    :func:`_normalize_pass2_pairs_with_log_z` maps an all-masked row to exactly
    zero probability, so they contribute nothing to any accumulator.

    ``max_images`` is the caller's byte budget for gather, preparation and the
    dense M step. The capacity never exceeds it: padding past that budget would
    trade a compile saving for an out-of-memory failure. When no power of two
    fits, the exact count is returned and that bucket keeps its own shape.
    """

    n_images = int(n_images)
    if n_images <= 0:
        return 0
    floor = max(1, int(floor))
    if max_growth is None:
        max_growth = image_capacity_max_growth()
    candidate = max(floor, 1 << (n_images - 1).bit_length())
    if candidate <= n_images:
        return n_images
    # Growth bound. Rounding up to a power of two never more than doubles, so this
    # only bites for a bucket smaller than the floor, and there the absolute row
    # count stays at the floor -- a handful of rows, whose memory cost is governed
    # by ``max_images`` whenever a byte budget is known.
    growth_limit = max(floor, int(float(max_growth) * n_images))
    if candidate > growth_limit:
        return n_images
    if max_images is not None and candidate > max(1, int(max_images)):
        return n_images
    return candidate



def ladder_chunks_enabled() -> bool:
    """Return whether bucket image lists are split into power-of-two chunks.

    Fused K-class pass 2 compiles one XLA program per helper and bucket shape.
    Without the ladder each rotation/pair bucket holds however many images
    fall into it (a different count every iteration), so every bucket of every
    iteration is a new shape. With the ladder the image axis takes only
    power-of-two sizes at or above :data:`LADDER_CHUNK_FLOOR` plus one
    remainder below the floor, so bucket shapes repeat across iterations.
    Per-image results are unchanged; only the grouping of per-bucket
    reductions differs.
    """

    return parse_env_binary_flag(LADDER_CHUNKS_ENV)


def bucket_chunk_bounds(n_images: int, max_per_chunk: int, *, ladder: bool | None = None):
    """Return ``(start, stop)`` chunk bounds for one bucket's image list.

    Default: consecutive chunks of ``max_per_chunk``. Ladder: greedy powers of
    two no larger than ``max_per_chunk`` while at least
    :data:`LADDER_CHUNK_FLOOR` images remain, then one remainder chunk.
    """

    n_images = int(n_images)
    max_per_chunk = max(1, int(max_per_chunk))
    if ladder is None:
        ladder = ladder_chunks_enabled()
    if not ladder:
        return [(start, min(start + max_per_chunk, n_images)) for start in range(0, n_images, max_per_chunk)]
    largest_power = 1 << (max_per_chunk.bit_length() - 1)
    bounds = []
    start = 0
    while n_images - start >= LADDER_CHUNK_FLOOR:
        size = min(largest_power, 1 << ((n_images - start).bit_length() - 1))
        bounds.append((start, start + size))
        start += size
    # The remainder must still respect the caller's cap: it is derived from
    # gather/prepare/dense-M-step byte budgets, so emitting the tail as one
    # chunk would overshoot them (a cap of 1 image would yield a chunk of 15).
    while start < n_images:
        stop = min(start + max_per_chunk, n_images)
        bounds.append((start, stop))
        start = stop
    return bounds


def _bucket_pass2_inputs(
    per_image_inputs,
    n_fine_trans,
    rotation_block_size_for_quantization=5000,
    max_hypotheses_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    small_bucket_coalesce_size=None,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
    processing_order_override=None,
    processing_order_chunk_size=1,
    processing_order_group_by_bucket_size=False,
    processing_order_batch_consecutive_bucket_sizes=False,
):
    """Group images into buckets that share a padded rotation count.

    Return bucket specifications: a padded rotation count and the image indices
    assigned to that bucket. Array builders materialize the selected rows later.

    To avoid OOM when one bucket is very large
    (``bucket_size * n_images_in_bucket * n_fine_trans`` is the (B, R, T)
    score tensor footprint), we split each per-quantization-size group
    into chunks of at most ``max_hypotheses_per_microbatch /
    (bucket_size * n_fine_trans)`` images.
    """
    n_images = len(per_image_inputs["oversampled_rots"])
    rotation_counts = np.array(
        [rots.shape[0] for rots in per_image_inputs["oversampled_rots"]],
        dtype=np.int64,
    )
    if n_images == 0:
        return []

    bucket_sizes = np.array(
        [_exact_bucket_rotation_size(int(count), rotation_block_size_for_quantization) for count in rotation_counts],
        dtype=np.int64,
    )
    if small_bucket_coalesce_size is not None:
        bucket_sizes = _coalesce_small_bucket_sizes(bucket_sizes, small_bucket_coalesce_size)
    bucket_sizes = _coalesce_tail_bucket_sizes(
        bucket_sizes,
        max_images=tail_bucket_coalesce_max_images,
        max_inflation=tail_bucket_coalesce_max_inflation,
        min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        n_fine_trans=n_fine_trans,
        n_classes=1,
    )

    if processing_order_override is not None:
        processing_order = np.asarray(processing_order_override, dtype=np.int64).reshape(-1)
        if processing_order.shape != (n_images,):
            raise ValueError(
                "processing_order_override must have shape "
                f"({n_images},), got {processing_order.shape}",
            )
        if not np.array_equal(np.sort(processing_order), np.arange(n_images, dtype=np.int64)):
            raise ValueError("processing_order_override must be a permutation of image indices")
        if not processing_order_group_by_bucket_size:
            if processing_order_batch_consecutive_bucket_sizes:
                buckets = []
                run_start = 0
                while run_start < n_images:
                    run_bucket_size = int(bucket_sizes[processing_order[run_start]])
                    run_end = run_start + 1
                    while (
                        run_end < n_images
                        and int(bucket_sizes[processing_order[run_end]]) == run_bucket_size
                    ):
                        run_end += 1
                    cap_by_hypotheses = max(
                        1,
                        int(max_hypotheses_per_microbatch)
                        // max(1, run_bucket_size * int(n_fine_trans)),
                    )
                    max_per_chunk = max(
                        1,
                        min(int(max_images_per_microbatch), cap_by_hypotheses),
                    )
                    for start in range(run_start, run_end, max_per_chunk):
                        chunk = processing_order[start : min(start + max_per_chunk, run_end)]
                        buckets.append(
                            {
                                "bucket_size": run_bucket_size,
                                "image_indices": np.asarray(chunk, dtype=np.int64),
                            }
                        )
                    run_start = run_end
                return buckets
            plans = _plan_consecutive_padded_batches(
                bucket_sizes,
                processing_order=processing_order,
                target_items_per_batch=int(processing_order_chunk_size),
                max_items_per_batch=int(max_images_per_microbatch),
                max_padded_values_per_batch=int(max_hypotheses_per_microbatch),
                values_per_padded_size=int(n_fine_trans),
            )
            return [
                {
                    "bucket_size": int(plan.padded_size),
                    "image_indices": np.asarray(plan.item_indices, dtype=np.int64),
                }
                for plan in plans
            ]
    else:
        # Group by bucket size, smaller buckets first. The secondary rotation
        # count key is historical RECOVAR behavior; an explicit order keeps
        # RELION order stable within each equal padded-size bucket.
        processing_order = np.lexsort((rotation_counts, bucket_sizes)).astype(np.int64)

    unique_bucket_sizes = np.unique(bucket_sizes[processing_order])

    buckets = []
    for bucket_size in unique_bucket_sizes:
        bucket_size = int(bucket_size)
        bucket_image_indices = processing_order[bucket_sizes[processing_order] == bucket_size]
        # Chunk by max_hypotheses_per_microbatch and max_images_per_microbatch
        cap_by_hypotheses = max(
            1,
            int(max_hypotheses_per_microbatch) // max(1, bucket_size * int(n_fine_trans)),
        )
        max_per_chunk = max(1, min(int(max_images_per_microbatch), cap_by_hypotheses))
        for start, stop in bucket_chunk_bounds(bucket_image_indices.shape[0], max_per_chunk):
            chunk = bucket_image_indices[start:stop]
            buckets.append(
                {
                    "bucket_size": bucket_size,
                    "image_indices": np.asarray(chunk, dtype=np.int64),
                }
            )
    return buckets


def _bucket_sparse_k_class_pass2_inputs(
    per_image_inputs_by_class,
    n_fine_trans,
    *,
    rotation_block_size_for_quantization=5000,
    max_hypotheses_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    small_bucket_threshold=None,
    small_bucket_max_images_per_microbatch=None,
    small_bucket_coalesce_size=None,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
):
    """Group images by the largest padded class support in a fused K-class pass."""

    n_classes = len(per_image_inputs_by_class)
    if n_classes == 0:
        return []
    n_images = len(per_image_inputs_by_class[0]["oversampled_rots"])
    if n_images == 0:
        return []
    bucket_sizes_by_class = []
    for per_image_inputs in per_image_inputs_by_class:
        if len(per_image_inputs["oversampled_rots"]) != n_images:
            raise ValueError("All classes must have the same image count for fused sparse pass-2")
        counts = np.asarray(
            [rots.shape[0] for rots in per_image_inputs["oversampled_rots"]],
            dtype=np.int64,
        )
        bucket_sizes_by_class.append(
            np.asarray(
                [
                    _exact_bucket_rotation_size(int(count), rotation_block_size_for_quantization)
                    for count in counts
                ],
                dtype=np.int64,
            )
        )
    fused_bucket_sizes = np.max(np.stack(bucket_sizes_by_class, axis=0), axis=0)
    if small_bucket_coalesce_size is not None:
        fused_bucket_sizes = _coalesce_small_bucket_sizes(fused_bucket_sizes, small_bucket_coalesce_size)
    fused_bucket_sizes = _coalesce_tail_bucket_sizes(
        fused_bucket_sizes,
        max_images=tail_bucket_coalesce_max_images,
        max_inflation=tail_bucket_coalesce_max_inflation,
        min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        n_fine_trans=n_fine_trans,
        n_classes=n_classes,
    )
    processing_order = np.argsort(fused_bucket_sizes, kind="stable").astype(np.int64)
    unique_bucket_sizes = np.unique(fused_bucket_sizes[processing_order])

    buckets = []
    for bucket_size in unique_bucket_sizes:
        bucket_size = int(bucket_size)
        bucket_image_indices = processing_order[fused_bucket_sizes[processing_order] == bucket_size]
        cap_by_hypotheses = max(
            1,
            int(max_hypotheses_per_microbatch)
            // max(1, int(n_classes) * bucket_size * int(n_fine_trans)),
        )
        image_cap = int(max_images_per_microbatch)
        if (
            small_bucket_threshold is not None
            and small_bucket_max_images_per_microbatch is not None
            and bucket_size <= int(small_bucket_threshold)
        ):
            image_cap = max(image_cap, int(small_bucket_max_images_per_microbatch))
        max_per_chunk = max(
            1,
            min(
                image_cap,
                cap_by_hypotheses,
            ),
        )
        for start, stop in bucket_chunk_bounds(bucket_image_indices.shape[0], max_per_chunk):
            buckets.append(
                {
                    "bucket_size": bucket_size,
                    "image_indices": np.asarray(
                        bucket_image_indices[start:stop],
                        dtype=np.int64,
                    ),
                }
            )
    return buckets


def _coalesce_small_bucket_sizes(bucket_sizes, small_bucket_coalesce_size):
    bucket_sizes = np.asarray(bucket_sizes, dtype=np.int64)
    coalesce_size = int(small_bucket_coalesce_size)
    if coalesce_size <= 1:
        return bucket_sizes
    small_or_target_mask = bucket_sizes <= coalesce_size
    if np.unique(bucket_sizes[small_or_target_mask]).size <= 1:
        return bucket_sizes
    return np.where(bucket_sizes < coalesce_size, coalesce_size, bucket_sizes)


def _coalesce_tail_bucket_sizes(
    bucket_sizes,
    *,
    max_images,
    max_inflation,
    min_bucket_size,
    max_hypotheses_per_microbatch,
    max_images_per_microbatch,
    n_fine_trans,
    n_classes,
):
    """Merge only tiny adjacent high-bucket groups under strict padding caps."""

    bucket_sizes = np.asarray(bucket_sizes, dtype=np.int64)
    if bucket_sizes.size == 0 or max_images is None:
        return bucket_sizes
    max_images = int(max_images)
    if max_images <= 1:
        return bucket_sizes
    max_inflation = (
        _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION
        if max_inflation is None
        else float(max_inflation)
    )
    min_bucket_size = (
        _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
        if min_bucket_size is None
        else int(min_bucket_size)
    )
    if max_inflation < 1.0:
        return bucket_sizes

    unique_sizes, inverse, counts = np.unique(
        bucket_sizes,
        return_inverse=True,
        return_counts=True,
    )
    if unique_sizes.size <= 1:
        return bucket_sizes

    assigned_sizes = unique_sizes.copy()
    group_count = unique_sizes.size
    i = 0
    while i < group_count:
        size_i = int(unique_sizes[i])
        count_i = int(counts[i])
        if size_i < min_bucket_size or count_i > max_images:
            i += 1
            continue

        best_j = i
        total_images = 0
        total_rows = 0
        for j in range(i, group_count):
            size_j = int(unique_sizes[j])
            count_j = int(counts[j])
            if size_j < min_bucket_size or count_j > max_images:
                break
            total_images += count_j
            total_rows += size_j * count_j
            if total_images > max_images or total_images > int(max_images_per_microbatch):
                break
            target_size = size_j
            padded_rows = target_size * total_images
            if total_rows <= 0:
                continue
            inflation = float(padded_rows) / float(total_rows)
            # The bucket builder chunks each coalesced size by
            # max_hypotheses_per_microbatch before execution.  Applying the
            # same cap here prevents adjacent tiny high-tail groups from
            # sharing one padded size even when every eventual chunk remains
            # within the score-tensor budget.
            if inflation <= max_inflation:
                best_j = j

        if best_j > i:
            assigned_sizes[i : best_j + 1] = unique_sizes[best_j]
            i = best_j + 1
        else:
            i += 1

    if np.array_equal(assigned_sizes, unique_sizes):
        return bucket_sizes
    return assigned_sizes[inverse]


VECTORIZED_HYPOTHESIS_PREP_ENV = "RECOVAR_SPARSE_PASS2_VECTORIZED_HYPOTHESIS_PREP"


def vectorized_hypothesis_prep_enabled() -> bool:
    """Prepare the per-image pass-2 hypotheses for all images at once.

    ``_prepare_per_image_pass2_inputs`` loops over every image in Python
    (np.unique, boolean child masks, searchsorted, per-image gathers); at
    100k/256 K=4 that is 34 s of host time per iteration (job 13832892,
    "hypothesis_prep=34.06s").  The vectorized path builds the same per-image
    arrays from flat concatenations and hands out views, for the paired
    RELION fine-grid override with a parent-major fine grid.  Images that need
    another branch (no samples, complement masks, empty sets) keep the loop.
    """

    return parse_env_binary_flag(VECTORIZED_HYPOTHESIS_PREP_ENV)


def _fine_children_ranges(fine_parent_np, n_coarse_rot):
    """``(child_start, child_count)`` per coarse rotation, or ``None`` if not parent-major."""

    fine_parent_np = np.asarray(fine_parent_np, dtype=np.int64)
    if fine_parent_np.ndim != 1 or (fine_parent_np.size > 1 and np.any(np.diff(fine_parent_np) < 0)):
        return None
    child_count = np.bincount(fine_parent_np, minlength=int(n_coarse_rot)).astype(np.int64)
    child_start = np.concatenate(([0], np.cumsum(child_count)[:-1])).astype(np.int64)
    return child_start, child_count


def _prepare_coarse_images_vectorized(
    sample_lists,
    *,
    n_coarse_rot,
    n_coarse_trans,
    n_fine_trans,
    fine_translation_parent,
    rotation_log_prior_np,
    fine_rotations_np,
    fine_mstep_rotations_np,
    fine_source_eulers,
    child_start,
    child_count,
    dtype,
):
    """Vectorized equivalent of the per-image ``coarse`` branch of the loop.

    Every returned per-image array is a view into one flat array and equals
    the loop's result element for element (values, dtypes and order); the
    unique coarse rotations are ascending, the fine rows are the ascending
    children of those rotations (``np.flatnonzero`` order for a parent-major
    grid) and ``parent_map`` is the rank of each row's parent.
    """

    n_images = len(sample_lists)
    lengths = np.fromiter((int(np.asarray(s).size) for s in sample_lists), dtype=np.int64, count=n_images)
    flat_sig = (
        np.concatenate([np.asarray(s, dtype=np.int32).reshape(-1) for s in sample_lists]).astype(np.int64)
        if n_images
        else np.zeros(0, dtype=np.int64)
    )
    sig_img = np.repeat(np.arange(n_images, dtype=np.int64), lengths)
    # The loop marks a boolean (coarse rotation, coarse translation) table, so a
    # significant index listed twice for one image counts once; dedupe here so
    # the per-pair sample sums below give the loop's ``count`` (lead review
    # em_clean_vectorized_hypothesis_duplicate_20260913).
    total_cells = int(n_coarse_rot) * int(n_coarse_trans)
    unique_cells = np.unique(sig_img * total_cells + flat_sig)
    sig_img = unique_cells // total_cells
    flat_sig = unique_cells % total_cells
    coarse_rot = flat_sig // int(n_coarse_trans)
    coarse_trans = flat_sig % int(n_coarse_trans)
    pair_key, sig_pair = np.unique(sig_img * int(n_coarse_rot) + coarse_rot, return_inverse=True)
    sig_pair = np.asarray(sig_pair).reshape(-1)
    pair_img = pair_key // int(n_coarse_rot)
    pair_urot = pair_key % int(n_coarse_rot)
    n_pairs = int(pair_key.shape[0])
    pairs_per_image = np.bincount(pair_img, minlength=n_images).astype(np.int64)
    pair_starts = np.concatenate(([0], np.cumsum(pairs_per_image)[:-1])).astype(np.int64)
    pair_rank = np.arange(n_pairs, dtype=np.int64) - np.repeat(pair_starts, pairs_per_image)

    rows_per_pair = child_count[pair_urot]
    n_rows = int(rows_per_pair.sum())
    row_pair = np.repeat(np.arange(n_pairs, dtype=np.int64), rows_per_pair)
    row_starts = np.concatenate(([0], np.cumsum(rows_per_pair)[:-1])).astype(np.int64)
    row_offset = np.arange(n_rows, dtype=np.int64) - np.repeat(row_starts, rows_per_pair)
    flat_rot_indices = child_start[pair_urot][row_pair] + row_offset
    flat_parent_map = pair_rank[row_pair].astype(np.int32)
    rows_per_image = np.bincount(pair_img, weights=rows_per_pair, minlength=n_images).astype(np.int64)

    # np.take on a contiguous first axis is the fastest host gather for these
    # (rows, 3, 3) / (rows, 3) tables; the per-image loop gathers the same rows.
    flat_rots = np.take(np.asarray(fine_rotations_np, dtype=dtype), flat_rot_indices, axis=0)
    flat_mstep_rots = (
        None
        if fine_mstep_rotations_np is None
        else np.take(np.asarray(fine_mstep_rotations_np, dtype=dtype), flat_rot_indices, axis=0)
    )
    flat_eulers = None if fine_source_eulers is None else np.take(np.asarray(fine_source_eulers), flat_rot_indices, axis=0)
    if rotation_log_prior_np is not None:
        flat_log_prior = np.take(np.asarray(rotation_log_prior_np, dtype=dtype), pair_urot[row_pair])
    else:
        flat_log_prior = np.zeros(n_rows, dtype=dtype)

    coarse_valid_flat = np.zeros((n_pairs, int(n_coarse_trans)), dtype=bool)
    coarse_valid_flat[sig_pair, coarse_trans] = True
    ftp = np.asarray(fine_translation_parent, dtype=np.int64).reshape(-1)
    fine_children_per_coarse_trans = np.bincount(ftp, minlength=int(n_coarse_trans)).astype(np.int64)
    # Each significant sample is one distinct (coarse rotation, coarse translation)
    # cell of its image, so the fine-translation children of a pair's valid cells
    # sum over its samples; exact in float64 for these small integers.
    valid_fine_per_pair = np.rint(
        np.bincount(sig_pair, weights=fine_children_per_coarse_trans[coarse_trans], minlength=n_pairs)
    ).astype(np.int64)
    counts_per_image = np.rint(
        np.bincount(pair_img, weights=rows_per_pair * valid_fine_per_pair, minlength=n_images)
    ).astype(np.int64)

    row_bounds = np.concatenate(([0], np.cumsum(rows_per_image))).astype(np.int64)
    pair_bounds = np.concatenate(([0], np.cumsum(pairs_per_image))).astype(np.int64)
    unique_rot_flat = pair_urot.astype(np.int32)
    out = {
        "source_eulers": [],
        "oversampled_rots": [],
        "oversampled_mstep_rots": [],
        "parent_map": [],
        "oversampled_rot_indices": [],
        "unique_rot": [],
        "log_prior": [],
        "candidate_mask": [],
    }
    for i in range(n_images):
        r0, r1 = int(row_bounds[i]), int(row_bounds[i + 1])
        p0, p1 = int(pair_bounds[i]), int(pair_bounds[i + 1])
        rots = flat_rots[r0:r1]
        parent_map = flat_parent_map[r0:r1]
        out["source_eulers"].append(None if flat_eulers is None else flat_eulers[r0:r1])
        out["oversampled_rots"].append(rots)
        out["oversampled_mstep_rots"].append(rots if flat_mstep_rots is None else flat_mstep_rots[r0:r1])
        out["parent_map"].append(parent_map)
        out["oversampled_rot_indices"].append(flat_rot_indices[r0:r1])
        out["unique_rot"].append(unique_rot_flat[p0:p1])
        out["log_prior"].append(flat_log_prior[r0:r1])
        out["candidate_mask"].append(
            SparseCandidateMask(
                mode="coarse",
                n_rows=r1 - r0,
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_valid=coarse_valid_flat[p0:p1],
                fine_translation_parent=fine_translation_parent,
                count=int(counts_per_image[i]),
            )
        )
    return out


def _prepare_per_image_pass2_inputs(
    significant_sample_indices,
    n_coarse_rot,
    n_coarse_trans,
    nside_level,
    oversampling_order,
    n_fine_trans,
    fine_translation_parent,
    rotation_log_prior,
    random_perturbation,
    fine_source_eulers_override=None,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    relion_parent_execution_order=False,
    dtype: np.dtype = np.float32,
):
    """Compute per-image oversampled rotations / parent maps / candidate masks.

    Mirrors the per-image branch in the reference implementation in
    :func:`compute_pass2_stats_sparse_perimage_reference` exactly so the
    batched path is a strict per-image equivalent.

    ``dtype`` controls the precision of every fine/oversampled rotation
    matrix this function builds or accepts: the RELION-supplied fine
    rotation override (``fine_rotations_override`` /
    ``fine_mstep_rotations_override``) *and* the standard
    ``get_oversampled_rotation_grid_from_samples`` grid built when no
    override is supplied. RELION's own fine-search rotation matrices stay
    ``RFLOAT`` (double) end to end in a double-precision build; pass
    ``precision_policy.score_real_dtype`` from the caller so this matches
    ``use_float64_scoring`` instead of always narrowing to float32.
    """
    from recovar.em.sampling import get_oversampled_rotation_grid_from_samples

    n_images = len(significant_sample_indices)
    per_image_source_eulers = []
    per_image_oversampled_rots = []
    per_image_oversampled_mstep_rots = []
    per_image_parent_map = []
    per_image_oversampled_rot_indices = []
    per_image_unique_rot = []
    per_image_log_prior = []
    per_image_candidate_mask = []
    per_image_lists = {
        "source_eulers": per_image_source_eulers,
        "oversampled_rots": per_image_oversampled_rots,
        "oversampled_mstep_rots": per_image_oversampled_mstep_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
    }
    full_unique_rot = np.arange(n_coarse_rot, dtype=np.int32)
    full_support_rotation_cache = None
    full_support_log_prior_cache = None
    full_support_candidate_mask_cache = None

    if rotation_log_prior is not None:
        rotation_log_prior_np = np.asarray(rotation_log_prior, dtype=dtype)
    else:
        rotation_log_prior_np = None

    fine_rotations_np = None
    fine_mstep_rotations_np = None
    fine_parent_np = None
    if fine_rotations_override is None and fine_rotation_parent_override is None:
        pass
    elif fine_rotations_override is not None and fine_rotation_parent_override is not None:
        fine_rotations_np = np.asarray(fine_rotations_override, dtype=dtype)
        fine_parent_np = np.asarray(fine_rotation_parent_override, dtype=np.int64)
        if fine_parent_np.ndim != 1:
            raise ValueError("fine_rotation_parent_override must be a 1D array")
        if fine_rotations_np.shape[0] != fine_parent_np.shape[0]:
            raise ValueError(
                "fine_rotations_override and fine_rotation_parent_override disagree on rotation count: "
                f"{fine_rotations_np.shape[0]} vs {fine_parent_np.shape[0]}",
            )
        if int(fine_parent_np.min(initial=0)) < 0 or int(fine_parent_np.max(initial=-1)) >= int(n_coarse_rot):
            raise ValueError("fine_rotation_parent_override values must be in [0, n_coarse_rot)")
    else:
        raise ValueError("fine_rotations_override and fine_rotation_parent_override must be provided together")

    if fine_mstep_rotations_override is not None:
        if fine_rotations_np is None:
            raise ValueError("fine_mstep_rotations_override requires fine_rotations_override")
        fine_mstep_rotations_np = np.asarray(fine_mstep_rotations_override, dtype=dtype)
        if fine_mstep_rotations_np.shape != fine_rotations_np.shape:
            raise ValueError(
                "fine_mstep_rotations_override must match fine_rotations_override shape: "
                f"{fine_mstep_rotations_np.shape} vs {fine_rotations_np.shape}",
            )

    fine_source_eulers = None
    if fine_source_eulers_override is not None:
        fine_source_eulers = np.asarray(fine_source_eulers_override)
        if (
            fine_rotations_np is None
            or fine_source_eulers.dtype != np.float64
            or fine_source_eulers.shape != (len(fine_rotations_np), 3)
            or not np.all(np.isfinite(fine_source_eulers))
        ):
            raise ValueError(
                "fine source Euler metadata must match the supplied rotation rows as finite float64 triples"
            )

    def _reorder_children(rotations, parent_map, rotation_indices, source_eulers, parent_ids):
        if not relion_parent_execution_order:
            return rotations, parent_map, rotation_indices, source_eulers
        parent_ids = np.asarray(parent_ids, dtype=np.int64).reshape(-1)
        n_pixels = 12 * (2 ** int(nside_level)) ** 2
        n_psi = 6 * 2 ** int(nside_level)
        if parent_ids.shape != np.asarray(parent_map).shape:
            raise ValueError("RELION parent execution keys must match fine rotations")
        if parent_ids.size and (
            int(parent_ids.min(initial=0)) < 0 or int(parent_ids.max(initial=-1)) >= int(n_pixels * n_psi)
        ):
            raise ValueError("RELION parent execution key is outside the coarse grid")
        relion_parent_key = (parent_ids % n_pixels) * n_psi + parent_ids // n_pixels
        order = np.argsort(relion_parent_key, kind="stable")
        return (
            np.asarray(rotations)[order],
            np.asarray(parent_map)[order],
            np.asarray(rotation_indices)[order],
            None if source_eulers is None else source_eulers[order],
        )

    vectorized_indices = []
    if (
        fine_rotations_np is not None
        and not relion_parent_execution_order
        and vectorized_hypothesis_prep_enabled()
    ):
        children = _fine_children_ranges(fine_parent_np, n_coarse_rot)
        if children is not None:
            vectorized_indices = [
                image_idx
                for image_idx, sig_samples in enumerate(significant_sample_indices)
                if sig_samples is not None
                and not isinstance(sig_samples, ComplementSignificantSampleIndices)
                and np.asarray(sig_samples).size > 0
            ]
    vectorized = None
    if vectorized_indices:
        vectorized = _prepare_coarse_images_vectorized(
            [significant_sample_indices[image_idx] for image_idx in vectorized_indices],
            n_coarse_rot=n_coarse_rot,
            n_coarse_trans=n_coarse_trans,
            n_fine_trans=n_fine_trans,
            fine_translation_parent=fine_translation_parent,
            rotation_log_prior_np=rotation_log_prior_np,
            fine_rotations_np=fine_rotations_np,
            fine_mstep_rotations_np=fine_mstep_rotations_np,
            fine_source_eulers=fine_source_eulers,
            child_start=children[0],
            child_count=children[1],
            dtype=dtype,
        )
    vectorized_set = set(vectorized_indices)
    vectorized_position = {image_idx: position for position, image_idx in enumerate(vectorized_indices)}

    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if image_idx in vectorized_set:
            position = vectorized_position[image_idx]
            for key, values in vectorized.items():
                per_image_lists[key].append(values[position])
            continue
        coarse_excluded = None
        if sig_samples is None:
            unique_rot = full_unique_rot
            use_full_candidate_mask = True
            use_full_rotation_support = True
            coarse_rot = unique_rot
            coarse_trans = None
        elif isinstance(sig_samples, ComplementSignificantSampleIndices):
            if int(sig_samples.total_size) != int(n_coarse_rot * n_coarse_trans):
                raise ValueError(
                    "Complement significant sample mask total_size does not match coarse pose grid: "
                    f"{int(sig_samples.total_size)} vs {int(n_coarse_rot * n_coarse_trans)}",
                )
            unique_rot = full_unique_rot
            use_full_candidate_mask = False
            use_full_rotation_support = True
            coarse_rot = unique_rot
            coarse_trans = None
            coarse_excluded = np.asarray(sig_samples.excluded_indices, dtype=np.int32).reshape(-1)
        else:
            use_full_candidate_mask = False
            use_full_rotation_support = False
            sig_samples = np.asarray(sig_samples, dtype=np.int32).reshape(-1)
            if sig_samples.size == 0:
                coarse_rot = np.empty(0, dtype=np.int32)
                coarse_trans = np.empty(0, dtype=np.int32)
                unique_rot = np.array([0], dtype=np.int32)
            else:
                coarse_rot = sig_samples // n_coarse_trans
                coarse_trans = sig_samples % n_coarse_trans
                unique_rot = np.unique(coarse_rot)

        if unique_rot.size == 0:
            raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")

        if use_full_rotation_support:
            if full_support_rotation_cache is None:
                if fine_rotations_override is None and fine_rotation_parent_override is None:
                    full_rots, full_parent_map, full_rot_indices, full_eulers = (
                        get_oversampled_rotation_grid_from_samples(
                            full_unique_rot,
                            nside_level,
                            oversampling_order=oversampling_order,
                            random_perturbation=random_perturbation,
                            return_rotation_indices=True,
                            return_source_eulers=True,
                            dtype=dtype,
                        )
                    )
                    full_support_rotation_cache = (
                        np.asarray(full_rots, dtype=dtype),
                        np.asarray(full_parent_map, dtype=np.int32),
                        np.asarray(full_rot_indices, dtype=np.int64),
                        full_eulers,
                    )
                else:  # Paired overrides were validated before entering the image loop.
                    full_support_rotation_cache = (
                        fine_rotations_np,
                        fine_parent_np.astype(np.int32, copy=False),
                        np.arange(fine_rotations_np.shape[0], dtype=np.int64),
                        fine_source_eulers,
                    )
                full_support_rotation_cache = _reorder_children(
                    *full_support_rotation_cache,
                    parent_ids=full_unique_rot[full_support_rotation_cache[1]],
                )
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = full_support_rotation_cache
        elif fine_rotations_override is None and fine_rotation_parent_override is None:
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = (
                get_oversampled_rotation_grid_from_samples(
                    unique_rot,
                    nside_level,
                    oversampling_order=oversampling_order,
                    random_perturbation=random_perturbation,
                    return_rotation_indices=True,
                    return_source_eulers=True,
                    dtype=dtype,
                )
            )
            oversampled_rots = np.asarray(oversampled_rots, dtype=dtype)
            parent_map = np.asarray(parent_map, dtype=np.int32)
            oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int64)
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = _reorder_children(
                oversampled_rots,
                parent_map,
                oversampled_rot_indices,
                source_eulers,
                unique_rot[parent_map],
            )
        else:  # Paired overrides were validated before entering the image loop.
            selected_parent = np.zeros(n_coarse_rot, dtype=bool)
            selected_parent[unique_rot] = True
            child_mask = selected_parent[fine_parent_np]
            oversampled_rot_indices = np.flatnonzero(child_mask).astype(np.int64)
            oversampled_rots = fine_rotations_np[oversampled_rot_indices]
            parent_map = np.searchsorted(unique_rot, fine_parent_np[oversampled_rot_indices]).astype(np.int32)
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = _reorder_children(
                oversampled_rots,
                parent_map,
                oversampled_rot_indices,
                None if fine_source_eulers is None else fine_source_eulers[oversampled_rot_indices],
                fine_parent_np[oversampled_rot_indices],
            )

        oversampled_mstep_rots = (
            oversampled_rots if fine_mstep_rotations_np is None else fine_mstep_rotations_np[oversampled_rot_indices]
        )

        if use_full_rotation_support and full_support_log_prior_cache is not None:
            local_rotation_log_prior = full_support_log_prior_cache
        elif rotation_log_prior_np is not None:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map]
            if use_full_rotation_support:
                local_rotation_log_prior = local_rotation_log_prior.astype(dtype, copy=False)
        else:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=dtype)
        if use_full_rotation_support:
            full_support_log_prior_cache = local_rotation_log_prior

        if use_full_candidate_mask:
            if full_support_candidate_mask_cache is None:
                full_support_candidate_mask_cache = SparseCandidateMask(
                    mode="full",
                    n_rows=oversampled_rots.shape[0],
                    n_fine_trans=n_fine_trans,
                    count=int(oversampled_rots.shape[0]) * int(n_fine_trans),
                )
            candidate_mask = full_support_candidate_mask_cache
        elif coarse_excluded is not None:
            excluded = np.unique(coarse_excluded.astype(np.int64, copy=False))
            if excluded.size and (
                int(excluded.min(initial=0)) < 0 or int(excluded.max(initial=-1)) >= int(n_coarse_rot * n_coarse_trans)
            ):
                raise ValueError("Complement significant sample exclusions must index the coarse pose grid")
            excluded_rot = excluded // int(n_coarse_trans)
            excluded_trans = excluded % int(n_coarse_trans)
            fine_rot_children = np.bincount(np.asarray(parent_map, dtype=np.int64), minlength=int(n_coarse_rot))
            fine_trans_children = np.bincount(
                np.asarray(fine_translation_parent, dtype=np.int64),
                minlength=int(n_coarse_trans),
            )
            excluded_fine_count = int(
                np.sum(fine_rot_children[excluded_rot] * fine_trans_children[excluded_trans], dtype=np.int64),
            )
            candidate_mask = SparseCandidateMask(
                mode="coarse_exclude",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_excluded=excluded.astype(np.int32, copy=False),
                fine_translation_parent=fine_translation_parent,
                count=int(oversampled_rots.shape[0]) * int(n_fine_trans) - excluded_fine_count,
            )
        elif coarse_trans.size == 0:
            candidate_mask = SparseCandidateMask(
                mode="empty",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                count=0,
            )
        else:
            coarse_valid = np.zeros((unique_rot.size, n_coarse_trans), dtype=bool)
            coarse_valid[np.searchsorted(unique_rot, coarse_rot), coarse_trans] = True
            # count = sum over fine rows of the parent coarse row's fine-translation
            # count; O(cR*T + R) instead of expanding the (R, T) table per image.
            fine_per_coarse_row = coarse_valid[:, fine_translation_parent].sum(axis=1, dtype=np.int64)
            candidate_mask = SparseCandidateMask(
                mode="coarse",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_valid=coarse_valid,
                fine_translation_parent=fine_translation_parent,
                count=int(fine_per_coarse_row[parent_map].sum()),
            )

        per_image_source_eulers.append(source_eulers)
        per_image_oversampled_rots.append(oversampled_rots)
        per_image_oversampled_mstep_rots.append(oversampled_mstep_rots)
        per_image_parent_map.append(parent_map)
        per_image_oversampled_rot_indices.append(oversampled_rot_indices)
        per_image_unique_rot.append(unique_rot)
        per_image_log_prior.append(local_rotation_log_prior.astype(dtype, copy=False))
        per_image_candidate_mask.append(candidate_mask)

    assert len(per_image_oversampled_rots) == n_images
    return {
        "source_eulers": per_image_source_eulers,
        "oversampled_rots": per_image_oversampled_rots,
        "oversampled_mstep_rots": per_image_oversampled_mstep_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
    }


def _prepare_per_image_compact_candidate_pairs(per_image_inputs, *, image_mask=None):
    """Flatten per-image sparse pass-2 masks into valid candidate pairs.

    Materialize source-ordered pairs for diagnostic planning and score checks.
    Compact execution instead builds selected buckets on demand with
    ``_build_compact_pair_bucket_arrays_from_per_image_inputs``.
    """

    n_images = len(per_image_inputs["candidate_mask"])
    if image_mask is not None:
        image_mask = np.asarray(image_mask, dtype=bool)
        if image_mask.shape != (n_images,):
            raise ValueError(f"compact pair image mask shape mismatch: {image_mask.shape} vs {(n_images,)}")
    compact_local_rotation_row = []
    compact_translation_idx = []
    compact_rotation_index = []
    compact_log_prior = []
    compact_pair_mask = []
    pair_counts = np.zeros(n_images, dtype=np.int32)
    log_prior_dtype = (
        np.result_type(*(np.asarray(prior).dtype for prior in per_image_inputs["log_prior"]))
        if n_images
        else np.dtype(np.float32)
    )

    for image_idx in range(n_images):
        if image_mask is not None and not bool(image_mask[image_idx]):
            compact_local_rotation_row.append(np.zeros(0, dtype=np.int32))
            compact_translation_idx.append(np.zeros(0, dtype=np.int32))
            compact_rotation_index.append(np.zeros(0, dtype=np.int64))
            compact_log_prior.append(np.zeros(0, dtype=log_prior_dtype))
            compact_pair_mask.append(np.zeros(0, dtype=bool))
            continue
        local_rot_rows, translation_idx = compact_candidate_indices_in_source_order(
            per_image_inputs["candidate_mask"][image_idx]
        )
        local_rot_rows = local_rot_rows.astype(np.int32, copy=False)
        translation_idx = translation_idx.astype(np.int32, copy=False)
        rotation_indices = np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64)
        rotation_log_prior = np.asarray(per_image_inputs["log_prior"][image_idx], dtype=log_prior_dtype)

        compact_local_rotation_row.append(local_rot_rows)
        compact_translation_idx.append(translation_idx)
        compact_rotation_index.append(rotation_indices[local_rot_rows].astype(np.int64, copy=False))
        compact_log_prior.append(rotation_log_prior[local_rot_rows].astype(log_prior_dtype, copy=False))
        compact_pair_mask.append(np.ones(local_rot_rows.shape[0], dtype=bool))
        pair_counts[image_idx] = int(local_rot_rows.shape[0])

    return {
        "local_rotation_row": compact_local_rotation_row,
        "translation_idx": compact_translation_idx,
        "rotation_index": compact_rotation_index,
        "log_prior": compact_log_prior,
        "pair_mask": compact_pair_mask,
        "pair_counts": pair_counts,
    }


def _build_compact_pair_bucket_arrays(bucket, compact_inputs):
    """Stack/pad compact candidate pairs for one class and bucket."""

    pair_bucket_size = int(bucket["pair_bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    batch = int(image_indices.shape[0])

    padded_local_rotation_row = np.full((batch, pair_bucket_size), -1, dtype=np.int32)
    padded_translation_idx = np.full((batch, pair_bucket_size), -1, dtype=np.int32)
    padded_rotation_index = np.zeros((batch, pair_bucket_size), dtype=np.int64)
    log_prior_dtype = np.result_type(
        *(np.asarray(compact_inputs["log_prior"][int(image_idx)]).dtype for image_idx in image_indices)
    )
    padded_log_prior = np.full((batch, pair_bucket_size), -1e30, dtype=log_prior_dtype)
    padded_pair_mask = np.zeros((batch, pair_bucket_size), dtype=bool)
    pair_counts = np.zeros(batch, dtype=np.int32)

    for row, image_idx in enumerate(image_indices.tolist()):
        count = int(compact_inputs["pair_counts"][image_idx])
        pair_counts[row] = count
        if count == 0:
            continue
        padded_local_rotation_row[row, :count] = compact_inputs["local_rotation_row"][image_idx]
        padded_translation_idx[row, :count] = compact_inputs["translation_idx"][image_idx]
        padded_rotation_index[row, :count] = compact_inputs["rotation_index"][image_idx]
        padded_log_prior[row, :count] = compact_inputs["log_prior"][image_idx]
        padded_pair_mask[row, :count] = compact_inputs["pair_mask"][image_idx]

    return {
        "image_indices": image_indices,
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": pair_counts,
        "local_rotation_row": padded_local_rotation_row,
        "translation_idx": padded_translation_idx,
        "rotation_index": padded_rotation_index,
        "log_prior": padded_log_prior,
        "pair_mask": padded_pair_mask,
    }


def _pad_rows(values, capacity, fill):
    """Extend ``values`` along axis 0 to ``capacity`` rows filled with ``fill``."""

    if values is None:
        return None
    pad = int(capacity) - int(values.shape[0])
    if pad <= 0:
        return values  # device arrays stay on the device when nothing is padded
    values = np.asarray(values)
    tail = np.full((pad,) + values.shape[1:], fill, dtype=values.dtype)
    return np.concatenate([values, tail], axis=0)


def pad_bucket_arrays_to_image_capacity(arrays, capacity):
    """Pad one class's dense bucket arrays out to a quantized image capacity.

    The padded rows carry ``actual_counts`` zero and ``candidate_mask`` false, which
    is the same contract the rotation axis already uses for its padding, so they
    contribute exactly zero to every accumulator. Identity rotations are harmless
    for the same reason.
    """

    capacity = int(capacity)
    if capacity <= int(np.asarray(arrays["image_indices"]).shape[0]):
        return arrays
    shared_mstep = arrays["mstep_rotations"] is arrays["rotations"]
    rotations = _pad_rows(arrays["rotations"], capacity, 0)
    rotations[int(np.asarray(arrays["rotations"]).shape[0]):] = np.eye(3, dtype=rotations.dtype)
    padded = dict(arrays)
    padded["rotations"] = rotations
    padded["mstep_rotations"] = rotations if shared_mstep else _pad_rows(arrays["mstep_rotations"], capacity, 0)
    if not shared_mstep:
        start = int(np.asarray(arrays["mstep_rotations"]).shape[0])
        padded["mstep_rotations"][start:] = np.eye(3, dtype=padded["mstep_rotations"].dtype)
    padded["rotation_indices"] = _pad_rows(arrays["rotation_indices"], capacity, 0)
    padded["actual_counts"] = _pad_rows(arrays["actual_counts"], capacity, 0)
    padded["log_prior"] = _pad_rows(arrays["log_prior"], capacity, -1e30)
    padded["row_log_prior"] = _pad_rows(arrays.get("row_log_prior"), capacity, -1e30)
    padded["candidate_mask"] = _pad_rows(arrays["candidate_mask"], capacity, False)
    padded["parent_map"] = _pad_rows(arrays["parent_map"], capacity, -1)
    return padded


def pad_compact_pair_arrays_to_image_capacity(pair_arrays, capacity):
    """Pad one class's compact-pair arrays out to a quantized image capacity.

    Padded rows carry ``pair_counts`` zero and ``pair_mask`` false.
    ``_normalize_pass2_pairs_with_log_z`` maps an all-masked row to exactly zero
    probability, so a padded image contributes nothing to the M step, the noise
    accumulators or the posterior sums.
    """

    capacity = int(capacity)
    if capacity <= int(np.asarray(pair_arrays["image_indices"]).shape[0]):
        return pair_arrays
    padded = dict(pair_arrays)
    padded["pair_counts"] = _pad_rows(pair_arrays["pair_counts"], capacity, 0)
    padded["local_rotation_row"] = _pad_rows(pair_arrays["local_rotation_row"], capacity, 0)
    padded["translation_idx"] = _pad_rows(pair_arrays["translation_idx"], capacity, 0)
    padded["rotation_index"] = _pad_rows(pair_arrays["rotation_index"], capacity, 0)
    padded["log_prior"] = _pad_rows(pair_arrays["log_prior"], capacity, -1e30)
    padded["pair_mask"] = _pad_rows(pair_arrays["pair_mask"], capacity, False)
    return padded


COMPACT_PAIR_LAZY_TABLES_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_LAZY_TABLES"
COMPACT_PAIR_DEVICE_INDEX_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DEVICE_INDEX"


def compact_pair_device_index_enabled() -> bool:
    """Build the compact pair index arrays on the device from the coarse tables.

    Requires the lazy pair tables (the per-pair host tables need host indices).
    The host otherwise allocates, fills and uploads ``(images, pairs)`` int32/bool
    arrays for every chunk: ~215 s of "build" in the 788 s iteration 2 at 100k/256
    (job 13812775). See ``compact_pair_index_arrays_device``.
    """
    return parse_env_binary_flag(COMPACT_PAIR_DEVICE_INDEX_ENV)


BUCKET_ROTATIONS_DEVICE_ENV = "RECOVAR_SPARSE_KCLASS_BUCKET_ROTATIONS_DEVICE"


def bucket_rotations_device_enabled() -> bool:
    """Assemble the padded ``(images, rows, 3, 3)`` bucket rotations on the device.

    The host otherwise allocates the identity-filled array at the class bucket
    size and copies every image's rows into it (twice when M-step rotations
    differ), then uploads it; only the real rows travel with this flag and the
    device pads them. Values are copied, not recomputed, so the arrays are
    bit-identical to the host build.
    """
    return parse_env_binary_flag(BUCKET_ROTATIONS_DEVICE_ENV)


def _flat_rows_quantum(n_rows: int) -> int:
    n_rows = max(1, int(n_rows))
    if n_rows <= 4096:
        return 1 << (n_rows - 1).bit_length()
    return ((n_rows + 4095) // 4096) * 4096


def _padded_rows_from_flat_impl(flat, starts, counts, fill, *, rows):
    """``out[b, r] = flat[starts[b] + r]`` for ``r < counts[b]``, else ``fill``."""
    import jax.numpy as jnp

    slots = jnp.arange(int(rows), dtype=jnp.int32)[None, :]
    src = jnp.clip(starts[:, None] + slots, 0, int(flat.shape[0]) - 1)
    valid = slots < counts[:, None]
    gathered = jnp.take(flat, src, axis=0)
    return jnp.where(valid.reshape(valid.shape + (1,) * (flat.ndim - 1)), gathered, fill)


def padded_rows_from_flat_device(per_row_values, counts, rows, fill):
    """Stack per-image row blocks into a device ``(images, rows, ...)`` array.

    ``per_row_values`` is a sequence of ``(count_i, ...)`` host arrays (one per
    real image), ``counts`` the ``(images,)`` int vector including capacity rows
    (zero), ``fill`` the value for padding slots (identity for rotations). One
    upload of the concatenated real rows, padded to a size ladder so the jitted
    padder compiles once per ``(images, rows, ladder step)``.
    """
    import jax
    import jax.numpy as jnp

    counts = np.asarray(counts, dtype=np.int32)
    real = [np.asarray(v) for v in per_row_values]
    if real:
        flat = np.concatenate(real, axis=0)
    else:
        flat = np.zeros((0,) + np.asarray(fill).shape, dtype=np.asarray(fill).dtype)
    n_flat = int(flat.shape[0])
    n_pad = _flat_rows_quantum(n_flat)
    if n_pad > n_flat:
        tail = np.empty((n_pad - n_flat,) + flat.shape[1:], dtype=flat.dtype)
        tail[...] = fill
        flat = np.concatenate([flat, tail], axis=0)
    starts = (np.cumsum(counts, dtype=np.int64) - counts).astype(np.int32)
    fn = padded_rows_from_flat_device.__dict__.get("_compiled")
    if fn is None:
        fn = jax.jit(_padded_rows_from_flat_impl, static_argnames=("rows",))
        padded_rows_from_flat_device.__dict__["_compiled"] = fn
    return fn(
        jnp.asarray(flat),
        jnp.asarray(starts),
        jnp.asarray(counts),
        jnp.asarray(np.asarray(fill, dtype=flat.dtype)),
        rows=int(rows),
    )


def compact_pair_lazy_tables_enabled() -> bool:
    """Skip the per-pair ``rotation_index``/``log_prior`` host tables.

    Both are pure gathers of per-row tables (``rotation_indices`` and
    ``row_log_prior`` of the class bucket) along ``local_rotation_row``; the
    consumers gather them on device or per argmax row instead. At 100k/256 the
    wide-pair class holds tens of thousands of pairs per image and building the
    int64/float64 tables was 167 s of iteration 2 on the host (job 13805735).
    """
    return parse_env_binary_flag(COMPACT_PAIR_LAZY_TABLES_ENV)


def _rows_at_capacity(values, capacity_rows, fill):
    """Return ``values`` extended along axis 0 to ``capacity_rows`` rows of ``fill`` (or as is)."""
    values = np.asarray(values)
    if capacity_rows is None or int(capacity_rows) <= int(values.shape[0]):
        return values
    out = np.full((int(capacity_rows),) + values.shape[1:], fill, dtype=values.dtype)
    out[: values.shape[0]] = values
    return out


def _build_compact_pair_bucket_arrays_from_per_image_inputs(
    bucket, per_image_inputs, *, lazy_pair_tables=None, capacity_rows=None, rows_capacity=None, device_index=None
):
    """Stack/pad compact candidate pairs for one class and bucket on demand.

    ``capacity_rows`` allocates the image axis at the quantized capacity directly
    (padded rows: counts 0, mask False, indices 0, prior -1e30, image index =
    last real image), which is exactly what ``pad_compact_pair_arrays_to_image_capacity``
    produced afterwards by copying every array (12 % of host time at 100k/256,
    job 13808609).
    """

    pair_bucket_size = int(bucket["pair_bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    batch = int(image_indices.shape[0])
    if lazy_pair_tables is None:
        lazy_pair_tables = compact_pair_lazy_tables_enabled()
    if device_index is None:
        device_index = compact_pair_device_index_enabled()
    n_alloc = batch if capacity_rows is None else max(batch, int(capacity_rows))
    padded_image_indices = image_indices
    if n_alloc > batch:
        padded_image_indices = np.concatenate([image_indices, np.repeat(image_indices[-1:], n_alloc - batch)])
    if lazy_pair_tables and device_index:
        device_arrays = compact_pair_index_arrays_device(
            [per_image_inputs["candidate_mask"][int(image_idx)] for image_idx in image_indices],
            pair_bucket_size=pair_bucket_size,
            n_alloc=n_alloc,
            rows_capacity=rows_capacity,
        )
        if device_arrays is not None:
            return {
                "image_indices": padded_image_indices,
                "pair_bucket_size": pair_bucket_size,
                "pair_counts": device_arrays["pair_counts"],
                "local_rotation_row": device_arrays["local_rotation_row"],
                "translation_idx": device_arrays["translation_idx"],
                "rotation_index": None,
                "log_prior": None,
                "pair_mask": device_arrays["pair_mask"],
            }
    index_arrays = build_compact_pair_index_arrays(
        (per_image_inputs["candidate_mask"][int(image_idx)] for image_idx in image_indices),
        pair_bucket_size=pair_bucket_size,
    )
    if lazy_pair_tables:
        return {
            "image_indices": padded_image_indices,
            "pair_bucket_size": pair_bucket_size,
            "pair_counts": _rows_at_capacity(index_arrays["pair_counts"], n_alloc, 0),
            "local_rotation_row": _rows_at_capacity(index_arrays["local_rotation_row"], n_alloc, 0),
            "translation_idx": _rows_at_capacity(index_arrays["translation_idx"], n_alloc, 0),
            "rotation_index": None,
            "log_prior": None,
            "pair_mask": _rows_at_capacity(index_arrays["pair_mask"], n_alloc, False),
        }
    padded_rotation_index = np.zeros((n_alloc, pair_bucket_size), dtype=np.int64)
    log_prior_dtype = np.result_type(
        *(np.asarray(per_image_inputs["log_prior"][int(image_idx)]).dtype for image_idx in image_indices)
    )
    padded_log_prior = np.full((n_alloc, pair_bucket_size), -1e30, dtype=log_prior_dtype)

    # One flat gather per field instead of a Python loop of per-image gathers and
    # slice assignments. Measured 2026-09-12 on the 100k/256 K=4 fixture: the loop
    # form made this "build" stage 17.6% of the pass-2 bucket loop, against 1.0% on
    # the 5k/128 fixture, because its cost scales with the particle count. The values
    # written and their positions are identical; only the marshalling changes.
    pair_counts_np = np.asarray(index_arrays["pair_counts"], dtype=np.int64)
    total_pairs = int(pair_counts_np.sum())
    if total_pairs:
        image_list = image_indices.tolist()
        rotation_rows = [
            np.asarray(per_image_inputs["oversampled_rot_indices"][i], dtype=np.int64)
            for i in image_list
        ]
        prior_rows = [
            np.asarray(per_image_inputs["log_prior"][i], dtype=log_prior_dtype)
            for i in image_list
        ]
        source_sizes = np.fromiter((a.shape[0] for a in rotation_rows), dtype=np.int64, count=batch)
        source_starts = np.concatenate(([0], np.cumsum(source_sizes)[:-1])) if batch else np.zeros(0, np.int64)
        dest_starts = np.concatenate(([0], np.cumsum(pair_counts_np)[:-1])) if batch else np.zeros(0, np.int64)
        dest_rows = np.repeat(np.arange(batch, dtype=np.int64), pair_counts_np)
        dest_cols = np.arange(total_pairs, dtype=np.int64) - np.repeat(dest_starts, pair_counts_np)
        flat_local_rows = np.asarray(index_arrays["local_rotation_row"])[dest_rows, dest_cols].astype(
            np.int64, copy=False
        )
        gather = flat_local_rows + np.repeat(source_starts, pair_counts_np)
        padded_rotation_index[dest_rows, dest_cols] = np.concatenate(rotation_rows)[gather]
        padded_log_prior[dest_rows, dest_cols] = np.concatenate(prior_rows)[gather]

    return {
        "image_indices": padded_image_indices,
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": _rows_at_capacity(index_arrays["pair_counts"], n_alloc, 0),
        "local_rotation_row": _rows_at_capacity(index_arrays["local_rotation_row"], n_alloc, 0),
        "translation_idx": _rows_at_capacity(index_arrays["translation_idx"], n_alloc, 0),
        "rotation_index": padded_rotation_index,
        "log_prior": padded_log_prior,
        "pair_mask": _rows_at_capacity(index_arrays["pair_mask"], n_alloc, False),
    }


def _build_bucket_arrays(
    bucket,
    per_image_inputs,
    n_fine_trans,
    *,
    include_dense_score_fields: bool = True,
    capacity_rows=None,
    device_rotations: bool = False,
):
    """Stack/pad per-image arrays into batched bucket tensors.

    ``capacity_rows`` allocates the image axis at the quantized capacity up front
    (identity rotations, zero counts/indices, -1e30 priors, false masks, -1
    parents, image index = last real image), matching what
    ``pad_bucket_arrays_to_image_capacity`` produced afterwards by copying.
    """
    bucket_size = int(bucket["bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    n_real = int(image_indices.shape[0])
    batch = n_real if capacity_rows is None else max(n_real, int(capacity_rows))
    if batch > n_real:
        image_indices = np.concatenate([image_indices, np.repeat(image_indices[-1:], batch - n_real)])

    # padded_rotations: identity-fill — projection of identity is harmless
    # because we mask via candidate_mask=False everywhere for padded rows.
    rotation_dtype = np.result_type(
        *(np.asarray(per_image_inputs["oversampled_rots"][int(image_idx)]).dtype for image_idx in image_indices)
    )
    padded_rotations = (
        None
        if device_rotations
        else np.broadcast_to(
            np.eye(3, dtype=rotation_dtype),
            (batch, bucket_size, 3, 3),
        ).copy()
    )
    separate_mstep_rotations = any(
        per_image_inputs["oversampled_mstep_rots"][int(image_idx)]
        is not per_image_inputs["oversampled_rots"][int(image_idx)]
        for image_idx in image_indices.tolist()
    )
    mstep_rotation_dtype = np.result_type(
        *(np.asarray(per_image_inputs["oversampled_mstep_rots"][int(image_idx)]).dtype for image_idx in image_indices)
    )
    padded_mstep_rotations = (
        (
            None
            if device_rotations
            else np.broadcast_to(
                np.eye(3, dtype=mstep_rotation_dtype),
                (batch, bucket_size, 3, 3),
            ).copy()
        )
        if separate_mstep_rotations
        else padded_rotations
    )
    padded_log_prior = (
        np.full(
            (batch, bucket_size),
            -1e30,
            dtype=np.result_type(
                *(np.asarray(per_image_inputs["log_prior"][int(image_idx)]).dtype for image_idx in image_indices)
            ),
        )
        if include_dense_score_fields
        else None
    )
    padded_candidate_mask = (
        np.zeros((batch, bucket_size, n_fine_trans), dtype=bool) if include_dense_score_fields else None
    )
    padded_parent_map = np.full((batch, bucket_size), -1, dtype=np.int32) if include_dense_score_fields else None
    # Per-row rotation prior, always present: the compact-pair path gathers its
    # per-pair prior from this table instead of materializing one per pair.
    padded_row_log_prior = np.full(
        (batch, bucket_size),
        -1e30,
        dtype=np.result_type(
            *(np.asarray(per_image_inputs["log_prior"][int(image_idx)]).dtype for image_idx in image_indices)
        ),
    )
    padded_rotation_indices = np.zeros((batch, bucket_size), dtype=np.int64)
    actual_counts = np.zeros(batch, dtype=np.int32)
    for row, image_idx in enumerate(image_indices[:n_real].tolist()):
        rots = per_image_inputs["oversampled_rots"][image_idx]
        cnt = int(rots.shape[0])
        actual_counts[row] = cnt
        if not device_rotations:
            padded_rotations[row, :cnt] = rots
            if separate_mstep_rotations:
                padded_mstep_rotations[row, :cnt] = per_image_inputs["oversampled_mstep_rots"][image_idx]
        if include_dense_score_fields:
            padded_log_prior[row, :cnt] = per_image_inputs["log_prior"][image_idx]
            padded_candidate_mask[row, :cnt, :] = _candidate_mask_to_dense(
                per_image_inputs["candidate_mask"][image_idx]
            )
            padded_parent_map[row, :cnt] = per_image_inputs["parent_map"][image_idx]
        padded_rotation_indices[row, :cnt] = per_image_inputs["oversampled_rot_indices"][image_idx]
        padded_row_log_prior[row, :cnt] = per_image_inputs["log_prior"][image_idx]

    if device_rotations:
        real_images = image_indices[:n_real].tolist()
        padded_rotations = padded_rows_from_flat_device(
            [np.asarray(per_image_inputs["oversampled_rots"][i], dtype=rotation_dtype) for i in real_images],
            actual_counts,
            bucket_size,
            np.eye(3, dtype=rotation_dtype),
        )
        padded_mstep_rotations = (
            padded_rows_from_flat_device(
                [np.asarray(per_image_inputs["oversampled_mstep_rots"][i], dtype=mstep_rotation_dtype) for i in real_images],
                actual_counts,
                bucket_size,
                np.eye(3, dtype=mstep_rotation_dtype),
            )
            if separate_mstep_rotations
            else padded_rotations
        )

    return {
        "image_indices": image_indices,
        "bucket_size": bucket_size,
        "actual_counts": actual_counts,
        "rotations": padded_rotations,
        "mstep_rotations": padded_mstep_rotations,
        "rotation_indices": padded_rotation_indices,
        "row_log_prior": padded_row_log_prior,
        "log_prior": padded_log_prior,
        "candidate_mask": padded_candidate_mask,
        "parent_map": padded_parent_map,
    }


def _compact_bucket_size_for_class(
    bucket,
    per_image_inputs,
    rotation_block_size_for_quantization,
) -> int:
    """Return the padded class-local bucket size for a fused K-class chunk."""

    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    if image_indices.size == 0:
        return 1
    max_count = max(
        int(per_image_inputs["oversampled_rots"][int(image_idx)].shape[0]) for image_idx in image_indices.tolist()
    )
    return _exact_bucket_rotation_size(int(max_count), rotation_block_size_for_quantization)


def _build_k_class_bucket_arrays(
    bucket,
    per_image_inputs_by_class,
    n_fine_trans,
    *,
    compact_buckets: bool = False,
    include_dense_score_fields: bool = True,
    rotation_block_size_for_quantization=5000,
    capacity_rows=None,
):
    """Build per-class padded arrays for fused sparse K-class pass 2.

    Default fused scoring keeps one rectangular bucket size shared by every
    class. The opt-in compact path keeps the same image chunk and joint
    class x pose normalization, but pads each class only to its class-local
    maximum rotation support inside that chunk.
    """

    class_arrays = []
    device_rotations = bucket_rotations_device_enabled()
    forced_class_bucket_sizes = bucket.get("class_bucket_sizes") if compact_buckets else None
    for class_index, per_image_inputs in enumerate(per_image_inputs_by_class):
        class_bucket = bucket
        if compact_buckets:
            class_bucket = dict(bucket)
            class_bucket["bucket_size"] = (
                int(forced_class_bucket_sizes[class_index])
                if forced_class_bucket_sizes is not None
                else _compact_bucket_size_for_class(
                    bucket,
                    per_image_inputs,
                    rotation_block_size_for_quantization,
                )
            )
        class_arrays.append(
            _build_bucket_arrays(
                class_bucket,
                per_image_inputs,
                n_fine_trans,
                include_dense_score_fields=include_dense_score_fields,
                capacity_rows=capacity_rows,
                device_rotations=device_rotations,
            )
        )
    return class_arrays
