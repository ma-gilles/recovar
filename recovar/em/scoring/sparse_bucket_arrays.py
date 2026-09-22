"""Host planning and array assembly for rectangular and compact sparse buckets.

Callers select execution policies and budgets. These helpers group supplied
support into bounded buckets, expand parent support and gather/pad rows. They
preserve particle order, precision, scoring/M-step aliases and inert padding;
they neither execute scoring nor choose scientific or device policies.
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np

from recovar.em.helpers.batch_planning import _plan_consecutive_padded_batches
from recovar.em.helpers.env_flags import parse_env_binary_flag, parse_env_flag
from recovar.em.helpers.shape_buckets import power_of_two_bucket
from recovar.em.local.local_layout import _exact_bucket_rotation_size

_LARGE_BUCKET_POW2_ENV = "RECOVAR_SPARSE_PASS2_LARGE_BUCKET_POW2"
_LARGE_BUCKET_POW2_THRESHOLD = 1024
def _pass2_bucket_rotation_size(count: int, rotation_block_size_for_quantization: int) -> int:
    """Padded rotation rows for one pass-2 image.

    The shared quantiser pads small supports to powers of two and large ones to
    multiples of a ~4096 quantum, which at HEALPix order 3 produced 17-31
    distinct large sizes per half (12288, 20480, 24576, 28672, ..., 217088);
    each distinct size compiles its own set of bucket programs.  With
    ``RECOVAR_SPARSE_PASS2_LARGE_BUCKET_POW2=1`` (default off, measurement
    knob) sizes above the engine cap are rounded up to a power of two instead,
    bounding the large sizes to about eight and costing at most 2x padding on
    those rows.  Padded rows carry zero posterior mass, so this changes shape
    reuse and atomic interleaving, not the candidate set.
    """

    size = int(_exact_bucket_rotation_size(int(count), rotation_block_size_for_quantization))
    if size > _LARGE_BUCKET_POW2_THRESHOLD and parse_env_flag(_LARGE_BUCKET_POW2_ENV, default=False):
        return int(power_of_two_bucket(size))
    return size
from recovar.em.scoring.compact_candidates import (
    SparseCandidateMask,
    _candidate_mask_to_dense,
    build_compact_pair_index_arrays,
    compact_candidate_indices_in_source_order,
    compact_pair_index_arrays_device,
)
from recovar.em.scoring.significant_samples import ComplementSignificantSampleIndices

_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH = 1_000_000
_DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION = 2.0
_DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE = 4096


def _split_run_into_chunks(run, max_per_chunk):
    """Split one support-size run into consecutive bounded views."""
    cap = max(1, int(max_per_chunk))
    return [run[start:stop] for start, stop in bucket_chunk_bounds(len(run), cap)]


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
        [_pass2_bucket_rotation_size(int(count), rotation_block_size_for_quantization) for count in rotation_counts],
        dtype=np.int64,
    )
    if small_bucket_coalesce_size is not None:
        bucket_sizes = _coalesce_small_bucket_sizes(bucket_sizes, small_bucket_coalesce_size)
    bucket_sizes = _coalesce_tail_bucket_sizes(
        bucket_sizes,
        max_images=tail_bucket_coalesce_max_images,
        max_inflation=tail_bucket_coalesce_max_inflation,
        min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        max_images_per_microbatch=max_images_per_microbatch,
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
                    for chunk in _split_run_into_chunks(
                        processing_order[run_start:run_end],
                        max_per_chunk,
                    ):
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
        for chunk in _split_run_into_chunks(
            bucket_image_indices,
            max_per_chunk,
        ):
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
        max_images_per_microbatch=max_images_per_microbatch,
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
    max_images_per_microbatch,
):
    """Merge tiny adjacent groups within image-count and padding-inflation caps.

    Bucket builders enforce class/translation hypothesis budgets when chunking
    the resulting groups. Coalescing does not change those execution budgets.
    """

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


_SPARSE_KCLASS_PAIR_BUCKET_QUANTUM_ENV = "RECOVAR_SPARSE_KCLASS_PAIR_BUCKET_QUANTUM"
_AUTO_COMPACT_PAIR_QUANTUM_MIN = 4096
_AUTO_COMPACT_PAIR_QUANTUM_MAX = 32768
_AUTO_COMPACT_PAIR_QUANTUM_MEAN_MULTIPLE = 2.0
logger = logging.getLogger(__name__)


def _compact_pair_bucket_quantum(pair_counts_by_class=None) -> int | None:
    """Optional coarser quantum for compact pair widths above the engine cap.

    The default ladder steps pair widths by 4096 above the cap, which gave 55
    distinct widths and 112 distinct bucket shapes in one 100k/256 iteration
    (job 13807792), each compiling the whole per-class stage chain. With masked
    pairs skipped by the fused score kernel and the pair-sparse sums, pair
    padding is nearly free, so ``RECOVAR_SPARSE_KCLASS_PAIR_BUCKET_QUANTUM``
    (for example 32768) trades a little padding for far fewer programs. Rows are
    not affected. ``None`` keeps the default ladder.

    ``auto`` picks the quantum per call from the valid-pair counts themselves:
    the smallest power of two at or above twice the mean valid pairs per
    image-class, clamped to [4096, 32768]. The two costs a fixed quantum trades
    move in opposite directions over a run. Early iterations have wide, flat
    posteriors (mean valid pairs 16446 at iteration 2 of the 100k/256 K=4
    fixture, job 13905556) and a fine quantum there creates many group shapes
    to compile: quantum 4096 cost +57 percent at iteration 2 and +27 percent at
    iteration 3 against 16384 (job 13912594). Late iterations are sparse (mean
    1371-1654 from iteration 6 on) and a coarse quantum there is mostly
    padding: 16384 padded 7x the valid pairs at iteration 12 and 4096 was 17
    percent faster at iterations 12-13. The crossover in those runs sits near a
    mean of 3000, which twice-the-mean rounded up reproduces: 32768 at
    iteration 2, 16384 at 3-4, 8192 at 5, 4096 from 6 on.
    """
    raw = os.environ.get(_SPARSE_KCLASS_PAIR_BUCKET_QUANTUM_ENV, "").strip()
    if raw.lower() != "auto":
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{_SPARSE_KCLASS_PAIR_BUCKET_QUANTUM_ENV} must be a positive integer, got {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{_SPARSE_KCLASS_PAIR_BUCKET_QUANTUM_ENV} must be a positive integer, got {raw!r}")
        return value
    if not pair_counts_by_class:
        return None
    valid_counts = np.concatenate([np.asarray(c, dtype=np.int64).reshape(-1) for c in pair_counts_by_class])
    if valid_counts.size == 0:
        return None
    mean_valid = float(np.mean(valid_counts))
    target = max(1.0, _AUTO_COMPACT_PAIR_QUANTUM_MEAN_MULTIPLE * mean_valid)
    quantum = 1 << int(math.ceil(math.log2(target)))
    quantum = int(min(max(quantum, _AUTO_COMPACT_PAIR_QUANTUM_MIN), _AUTO_COMPACT_PAIR_QUANTUM_MAX))
    logger.info(
        "sparse K-class compact pairs: auto pair bucket quantum=%d from mean valid pairs/image-class=%.1f",
        quantum,
        mean_valid,
    )
    return quantum


def _compact_pair_fused_bucket_sizes(pair_counts_by_class, *, pair_block_size_for_quantization=5000):
    if not pair_counts_by_class:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    fused_pair_counts = np.max(np.stack(pair_counts_by_class, axis=0), axis=0)
    # Resolve the quantum once per plan, not once per image: the accessor reads the
    # environment and, in auto mode, the whole pair-count distribution.
    large_bucket_quantum = _compact_pair_bucket_quantum(pair_counts_by_class)
    pair_bucket_sizes = np.asarray(
        [
            _exact_bucket_rotation_size(
                int(count),
                pair_block_size_for_quantization,
                large_bucket_quantum=large_bucket_quantum,
            )
            for count in fused_pair_counts
        ],
        dtype=np.int64,
    )
    return fused_pair_counts, pair_bucket_sizes

def _compact_pair_image_mask_for_threshold(
    pair_counts_by_class,
    min_pair_bucket_size: int | None,
    *,
    pair_block_size_for_quantization=5000,
):
    if min_pair_bucket_size is None:
        return None
    _, pair_bucket_sizes = _compact_pair_fused_bucket_sizes(
        pair_counts_by_class,
        pair_block_size_for_quantization=pair_block_size_for_quantization,
    )
    return pair_bucket_sizes >= int(min_pair_bucket_size)


def _bucket_sparse_k_class_compact_pair_counts(
    pair_counts_by_class,
    *,
    pair_block_size_for_quantization=5000,
    max_pair_candidates_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    image_mask=None,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
):
    """Group images by padded valid-pair count for a compact K-class pass-2."""

    n_classes = len(pair_counts_by_class)
    if n_classes == 0:
        return []
    n_images = int(np.asarray(pair_counts_by_class[0]).shape[0])
    if n_images == 0:
        return []
    if image_mask is not None:
        image_mask = np.asarray(image_mask, dtype=bool)
        if image_mask.shape != (n_images,):
            raise ValueError(f"compact pair image mask shape mismatch: {image_mask.shape} vs {(n_images,)}")
    normalized_counts_by_class = []
    for pair_counts in pair_counts_by_class:
        pair_counts = np.asarray(pair_counts, dtype=np.int64)
        if pair_counts.shape[0] != n_images:
            raise ValueError("All classes must have the same image count for compact sparse pass-2")
        normalized_counts_by_class.append(pair_counts)

    _, pair_bucket_sizes = _compact_pair_fused_bucket_sizes(
        normalized_counts_by_class,
        pair_block_size_for_quantization=pair_block_size_for_quantization,
    )
    if image_mask is not None:
        masked_indices = np.nonzero(image_mask)[0]
        if masked_indices.size:
            pair_bucket_sizes = pair_bucket_sizes.copy()
            pair_bucket_sizes[masked_indices] = _coalesce_tail_bucket_sizes(
                pair_bucket_sizes[masked_indices],
                max_images=tail_bucket_coalesce_max_images,
                max_inflation=tail_bucket_coalesce_max_inflation,
                min_bucket_size=tail_bucket_coalesce_min_bucket_size,
                max_images_per_microbatch=max_images_per_microbatch,
            )
    else:
        pair_bucket_sizes = _coalesce_tail_bucket_sizes(
            pair_bucket_sizes,
            max_images=tail_bucket_coalesce_max_images,
            max_inflation=tail_bucket_coalesce_max_inflation,
            min_bucket_size=tail_bucket_coalesce_min_bucket_size,
            max_images_per_microbatch=max_images_per_microbatch,
        )
    processing_order = np.argsort(pair_bucket_sizes, kind="stable").astype(np.int64)
    if image_mask is not None:
        processing_order = processing_order[image_mask[processing_order]]
    unique_bucket_sizes = np.unique(pair_bucket_sizes[processing_order])

    buckets = []
    for pair_bucket_size in unique_bucket_sizes:
        pair_bucket_size = int(pair_bucket_size)
        bucket_image_indices = processing_order[pair_bucket_sizes[processing_order] == pair_bucket_size]
        cap_by_pairs = max(
            1,
            int(max_pair_candidates_per_microbatch) // max(1, int(n_classes) * pair_bucket_size),
        )
        max_per_chunk = max(1, min(int(max_images_per_microbatch), cap_by_pairs))
        for start, stop in bucket_chunk_bounds(bucket_image_indices.shape[0], max_per_chunk):
            buckets.append(
                {
                    "pair_bucket_size": pair_bucket_size,
                    "image_indices": np.asarray(
                        bucket_image_indices[start:stop],
                        dtype=np.int64,
                    ),
                }
            )
    return buckets



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


def _pow2_at_least(value: int, floor: int) -> int:
    value = max(int(value), int(floor))
    return 1 << (value - 1).bit_length()


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
    cell_keys = sig_img * total_cells + flat_sig
    if cell_keys.size > 1 and not bool(np.all(np.diff(cell_keys) > 0)):
        # Only sort when some image lists an index twice or out of order; the
        # significance pass emits sorted unique indices, so this is the rare path.
        cell_keys = np.unique(cell_keys)
        sig_img = cell_keys // total_cells
        flat_sig = cell_keys % total_cells
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
    # Flat coarse tables and row parents for the device index builder
    # (RECOVAR_SPARSE_KCLASS_RESIDENT_HYPOTHESIS_TABLES): one upload per class
    # and iteration instead of a host (images, cR, cT)/(images, rows) build and
    # upload per class-chunk. ``token`` identifies this content for the cache.
    import uuid

    out["_resident"] = {
        "coarse_valid_flat": coarse_valid_flat,
        "parent_map_flat": flat_parent_map,
        "rot_indices_flat": flat_rot_indices.astype(np.int32, copy=False),
        "pair_bounds": pair_bounds,
        "row_bounds": row_bounds,
        "n_coarse_trans": int(n_coarse_trans),
        "token": uuid.uuid4().hex,
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


def _rotation_table_key(table) -> tuple:
    """Content key for the device table cache: shape, dtype and a SHA-1 of the bytes."""

    import hashlib

    table = np.ascontiguousarray(table)
    return (table.shape, str(table.dtype), hashlib.sha1(table.view(np.uint8)).hexdigest())


def relion_parent_execution_key(parent_ids, *, n_coarse_rot: int, nside_level: int) -> np.ndarray:
    """RELION's execution key for coarse rotation ids.

    RECOVAR's coarse grid is psi-slow and direction-fast
    (``id = psi * n_directions + direction``); RELION executes parents
    direction-major (``direction * n_psi + psi``). The direction count comes
    from the grid itself, ``n_coarse_rot / n_psi``, so a symmetry-reduced grid
    keeps its own direction count. The host pass-2 preparation and the resident
    CSR tables both order rows with this key.
    """

    from recovar.em.sampling import rotation_grid_n_in_planes

    n_psi = int(rotation_grid_n_in_planes(nside_level))
    n_coarse_rot = int(n_coarse_rot)
    if n_coarse_rot <= 0 or n_coarse_rot % n_psi:
        raise ValueError(
            f"RELION parent execution order needs whole psi rows: {n_coarse_rot} coarse "
            f"rotations with {n_psi} psi angles at healpix level {int(nside_level)}"
        )
    n_directions = n_coarse_rot // n_psi
    parent_ids = np.asarray(parent_ids, dtype=np.int64).reshape(-1)
    if parent_ids.size and (int(parent_ids.min()) < 0 or int(parent_ids.max()) >= n_coarse_rot):
        raise ValueError("RELION parent execution key is outside the coarse grid")
    return (parent_ids % n_directions) * n_psi + parent_ids // n_directions


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
    symmetry_label: str = "C1",
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
    from recovar.em.sampling import (
        get_oversampled_rotation_grid_from_samples,
        rotation_grid_size,
    )
    from recovar.em.symmetry import canonicalize_rotational_symmetry

    symmetry_label = canonicalize_rotational_symmetry(symmetry_label)
    if symmetry_label != "C1":
        expected_coarse_rot = rotation_grid_size(nside_level, symmetry_label)
        if int(n_coarse_rot) != int(expected_coarse_rot):
            raise ValueError(
                f"{symmetry_label} sparse pass-2 coarse rotation count mismatch: "
                f"{n_coarse_rot} != {expected_coarse_rot}"
            )

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
        if parent_ids.shape != np.asarray(parent_map).shape:
            raise ValueError("RELION parent execution keys must match fine rotations")
        relion_parent_key = relion_parent_execution_key(
            parent_ids, n_coarse_rot=n_coarse_rot, nside_level=nside_level
        )
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
    resident_hypothesis = None
    if vectorized is not None:
        positions = np.full(n_images, -1, dtype=np.int64)
        positions[np.asarray(vectorized_indices, dtype=np.int64)] = np.arange(len(vectorized_indices), dtype=np.int64)
        resident_hypothesis = dict(vectorized.pop("_resident"), positions=positions)

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
                            **({} if symmetry_label == "C1" else {"symmetry": symmetry_label}),
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
                    **({} if symmetry_label == "C1" else {"symmetry": symmetry_label}),
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
    # Global fine-grid tables (paired override only): every image's rows are
    # gathers of these by ``oversampled_rot_indices``, so the device can gather
    # them from one resident copy instead of receiving the rows per chunk.
    rotation_table = None if fine_rotations_np is None else np.asarray(fine_rotations_np, dtype=dtype)
    mstep_rotation_table = (
        None if fine_mstep_rotations_np is None else np.asarray(fine_mstep_rotations_np, dtype=dtype)
    )
    return {
        "source_eulers": per_image_source_eulers,
        "oversampled_rots": per_image_oversampled_rots,
        "oversampled_mstep_rots": per_image_oversampled_mstep_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
        "rotation_table": rotation_table,
        "rotation_table_key": None if rotation_table is None else _rotation_table_key(rotation_table),
        "mstep_rotation_table": mstep_rotation_table,
        "mstep_rotation_table_key": None if mstep_rotation_table is None else _rotation_table_key(mstep_rotation_table),
        "resident_hypothesis": resident_hypothesis,
        # One coarse-row capacity for the whole pass: the device index builder
        # otherwise compiled a family per chunk-wise coarse-row count (census job
        # 13837258: 28 compiles, 18.7 s). Padded coarse rows are inert.
        "coarse_rows_capacity": _pow2_at_least(
            max((int(np.asarray(u).shape[0]) for u in per_image_unique_rot), default=1), 64
        ),
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
            compact_log_prior.append(np.zeros(0, dtype=log_prior_dtype))
            compact_pair_mask.append(np.zeros(0, dtype=bool))
            continue
        local_rot_rows, translation_idx = compact_candidate_indices_in_source_order(
            per_image_inputs["candidate_mask"][image_idx]
        )
        local_rot_rows = local_rot_rows.astype(np.int32, copy=False)
        translation_idx = translation_idx.astype(np.int32, copy=False)
        rotation_log_prior = np.asarray(per_image_inputs["log_prior"][image_idx], dtype=log_prior_dtype)

        compact_local_rotation_row.append(local_rot_rows)
        compact_translation_idx.append(translation_idx)
        compact_log_prior.append(rotation_log_prior[local_rot_rows].astype(log_prior_dtype, copy=False))
        compact_pair_mask.append(np.ones(local_rot_rows.shape[0], dtype=bool))
        pair_counts[image_idx] = int(local_rot_rows.shape[0])

    return {
        "local_rotation_row": compact_local_rotation_row,
        "translation_idx": compact_translation_idx,
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
        padded_log_prior[row, :count] = compact_inputs["log_prior"][image_idx]
        padded_pair_mask[row, :count] = compact_inputs["pair_mask"][image_idx]

    return {
        "image_indices": image_indices,
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": pair_counts,
        "local_rotation_row": padded_local_rotation_row,
        "translation_idx": padded_translation_idx,
        "log_prior": padded_log_prior,
        "pair_mask": padded_pair_mask,
    }


def _build_compact_pair_bucket_arrays_from_per_image_inputs(
    bucket, per_image_inputs, *, device_index=False, rows_capacity=None, capacity_rows=None
):
    """Stack/pad compact candidate pairs for one class and bucket on demand."""

    pair_bucket_size = int(bucket["pair_bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    n_real = len(image_indices)
    n_alloc = n_real if capacity_rows is None else max(n_real, int(capacity_rows))
    padded_image_indices = image_indices
    if n_alloc > n_real:
        padded_image_indices = np.concatenate([image_indices, np.repeat(image_indices[-1:], n_alloc - n_real)])
    # This engine already gathers pair priors from row tables on device. The
    # device builder therefore needs no separate lazy-table policy or host
    # materialization of per-pair priors/rotation IDs.
    masks = [per_image_inputs["candidate_mask"][int(image_idx)] for image_idx in image_indices]
    index_arrays = None
    if device_index:
        resident = None
        resident_positions = None
        if resident_hypothesis_tables_enabled() and isinstance(per_image_inputs, dict):
            candidate = per_image_inputs.get("resident_hypothesis")
            if candidate is not None:
                positions = np.asarray(candidate["positions"], dtype=np.int64)[image_indices]
                if np.all(positions >= 0):
                    resident, resident_positions = candidate, positions
        index_arrays = compact_pair_index_arrays_device(
            masks,
            pair_bucket_size=pair_bucket_size,
            rows_capacity=rows_capacity,
            n_alloc=n_alloc,
            resident=resident,
            resident_positions=resident_positions,
            coarse_rows_capacity=per_image_inputs.get("coarse_rows_capacity"),
        )
    if index_arrays is None:
        index_arrays = build_compact_pair_index_arrays(
            masks, pair_bucket_size=pair_bucket_size,
        )
        index_arrays = {
            key: _rows_at_capacity(value, n_alloc, False if key == "pair_mask" else 0)
            for key, value in index_arrays.items()
            if key != "pair_bucket_size"
        }
    return {
        "image_indices": padded_image_indices,
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": index_arrays["pair_counts"],
        "local_rotation_row": index_arrays["local_rotation_row"],
        "translation_idx": index_arrays["translation_idx"],
        "pair_mask": index_arrays["pair_mask"],
    }


BUCKET_ROTATIONS_DEVICE_ENV = "RECOVAR_SPARSE_KCLASS_BUCKET_ROTATIONS_DEVICE"
ROTATIONS_BY_INDEX_ENV = "RECOVAR_SPARSE_KCLASS_ROTATIONS_BY_INDEX"
RESIDENT_HYPOTHESIS_TABLES_ENV = "RECOVAR_SPARSE_KCLASS_RESIDENT_HYPOTHESIS_TABLES"
_ROTATION_TABLE_DEVICE_CACHE: dict = {}
_RESIDENT_ROW_INDEX_CACHE: dict = {}


def bucket_rotations_device_enabled() -> bool:
    """Assemble the padded ``(images, rows, 3, 3)`` bucket rotations on the device.

    The host otherwise allocates the identity-filled array at the class bucket
    size and copies every image's rows into it (twice when M-step rotations
    differ), then uploads it; only the real rows travel with this flag and the
    device pads them. Values are copied, not recomputed, so the arrays are
    bit-identical to the host build.
    """
    return parse_env_binary_flag(BUCKET_ROTATIONS_DEVICE_ENV)


def resident_hypothesis_tables_enabled() -> bool:
    """Feed the device index builder from device-resident flat hypothesis tables.

    With the vectorized hypothesis preparation the coarse validity table and
    the row parent map of every image already exist as one flat array per
    class; the device index builder otherwise rebuilt a padded
    ``(images, cR, cT)``/``(images, rows)`` pair on the host and uploaded it for
    every class-chunk (7.7 s of device_put plus the host fill in the 100k/256
    K=4 iteration 2, job 13834297).  Chunks whose images all took the
    vectorized path gather those tables on the device instead; values are
    identical, so the pair index arrays are bit-identical.
    """

    return parse_env_binary_flag(RESIDENT_HYPOTHESIS_TABLES_ENV)


def rotations_by_index_enabled() -> bool:
    """Gather the padded bucket rotations from a device-resident fine-grid table.

    With ``RECOVAR_SPARSE_KCLASS_BUCKET_ROTATIONS_DEVICE`` the host still
    concatenates every image's (rows, 3, 3) float32 rows per class-chunk and
    uploads them (13.6 s of device_put plus ~7 s of host concatenation in the
    100k/256 K=4 iteration 2, job 13834297).  When the per-image inputs carry
    the fine-grid table (paired RELION override), the rows are gathers of that
    table by ``oversampled_rot_indices``; this flag uploads the table once per
    distinct content and gathers on the device from the padded rotation index
    array the builder already forms.  Values are copied, so bit-identical.
    """

    return parse_env_binary_flag(ROTATIONS_BY_INDEX_ENV)


def _rotation_table_device(table, key):
    """Device copy of a fine-grid rotation table, keyed by content (bounded cache)."""

    import jax.numpy as jnp

    cached = _ROTATION_TABLE_DEVICE_CACHE.get(key)
    if cached is None:
        if len(_ROTATION_TABLE_DEVICE_CACHE) >= 8:
            _ROTATION_TABLE_DEVICE_CACHE.clear()
        cached = jnp.asarray(np.ascontiguousarray(table))
        _ROTATION_TABLE_DEVICE_CACHE[key] = cached
    return cached


def _padded_rotations_from_table_impl(table, rotation_indices, counts, fill, *, rows):
    import jax.numpy as jnp

    gathered = jnp.take(table, jnp.clip(rotation_indices, 0, table.shape[0] - 1), axis=0)
    valid = jnp.arange(rows, dtype=jnp.int32)[None, :] < counts[:, None]
    return jnp.where(valid[:, :, None, None], gathered, fill[None, None])


def _resident_row_indices_device(resident):
    """Device copy of one class's flat per-row global rotation indices, keyed by prep token."""

    import jax.numpy as jnp

    key = resident["token"]
    cached = _RESIDENT_ROW_INDEX_CACHE.get(key)
    if cached is None:
        if len(_RESIDENT_ROW_INDEX_CACHE) >= 8:
            _RESIDENT_ROW_INDEX_CACHE.clear()
        cached = (
            jnp.asarray(np.asarray(resident["rot_indices_flat"], dtype=np.int32)),
            jnp.asarray(np.asarray(resident["row_bounds"], dtype=np.int32)),
        )
        _RESIDENT_ROW_INDEX_CACHE[key] = cached
    return cached


def _padded_rotations_from_resident_impl(table, rot_flat, row_bounds, positions, counts, fill, *, rows):
    import jax.numpy as jnp

    safe = jnp.clip(positions, 0, row_bounds.shape[0] - 2)
    start = row_bounds[safe]
    offsets = jnp.arange(rows, dtype=jnp.int32)[None, :]
    valid = offsets < counts[:, None]
    source = jnp.clip(start[:, None] + offsets, 0, rot_flat.shape[0] - 1)
    gathered = jnp.take(table, jnp.where(valid, jnp.take(rot_flat, source), 0), axis=0)
    return jnp.where(valid[:, :, None, None], gathered, fill[None, None])


def padded_rotations_from_resident_device(table, table_key, resident, positions, counts, rows, fill):
    """``(images, rows, 3, 3)`` rotations gathered entirely on the device.

    The per-row global rotation indices live in one resident int32 table per class and
    iteration, so a chunk uploads only its per-image positions and counts instead of an
    ``(images, rows)`` index array. At 100k/256 K=4 that index upload was ~17 s of the
    iteration (job 13838987 stack samples: padded_rotations_from_table_device). Requires
    every image of the chunk to have come from the vectorized preparation; the values are
    the same table entries, so the result is bit-identical to the index-array gather.
    """
    import jax
    import jax.numpy as jnp

    fn = padded_rotations_from_resident_device.__dict__.get("_compiled")
    if fn is None:
        fn = jax.jit(_padded_rotations_from_resident_impl, static_argnames=("rows",))
        padded_rotations_from_resident_device.__dict__["_compiled"] = fn
    rot_flat, row_bounds = _resident_row_indices_device(resident)
    return fn(
        _rotation_table_device(table, table_key),
        rot_flat,
        row_bounds,
        jnp.asarray(np.asarray(positions, dtype=np.int32)),
        jnp.asarray(np.asarray(counts, dtype=np.int32)),
        jnp.asarray(np.asarray(fill, dtype=table.dtype)),
        rows=int(rows),
    )


def padded_rotations_from_table_device(table, table_key, rotation_indices, counts, rows, fill):
    """``(images, rows, 3, 3)`` device rotations gathered from a resident table.

    ``rotation_indices`` is the host ``(images, rows)`` int64 padded index array
    (zeros beyond ``counts``); padded slots receive ``fill`` (identity).  Equals
    :func:`padded_rows_from_flat_device` on the concatenated per-image rows.
    """
    import jax
    import jax.numpy as jnp

    fn = padded_rotations_from_table_device.__dict__.get("_compiled")
    if fn is None:
        fn = jax.jit(_padded_rotations_from_table_impl, static_argnames=("rows",))
        padded_rotations_from_table_device.__dict__["_compiled"] = fn
    table_dev = _rotation_table_device(table, table_key)
    return fn(
        table_dev,
        jnp.asarray(np.asarray(rotation_indices, dtype=np.int32)),
        jnp.asarray(np.asarray(counts, dtype=np.int32)),
        jnp.asarray(np.asarray(fill, dtype=table.dtype)),
        rows=int(rows),
    )


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


def _build_bucket_arrays(
    bucket,
    per_image_inputs,
    n_fine_trans,
    *,
    include_dense_score_fields: bool = True,
    capacity_rows=None,
    device_rotations: bool = False,
):
    """Stack/pad per-image arrays into batched bucket tensors."""
    bucket_size = int(bucket["bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    n_real = int(image_indices.shape[0])
    batch = n_real if capacity_rows is None else max(n_real, int(capacity_rows))
    padded_image_indices = image_indices
    if batch > n_real:
        padded_image_indices = np.concatenate([image_indices, np.repeat(image_indices[-1:], batch - n_real)])

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
    padded_log_prior = np.full(
        (batch, bucket_size),
        -1e30,
        dtype=np.result_type(
            *(np.asarray(per_image_inputs["log_prior"][int(image_idx)]).dtype for image_idx in image_indices)
        ),
    )
    padded_candidate_mask = (
        np.zeros((batch, bucket_size, n_fine_trans), dtype=bool) if include_dense_score_fields else None
    )
    padded_parent_map = np.full((batch, bucket_size), -1, dtype=np.int32) if include_dense_score_fields else None
    padded_rotation_indices = np.zeros((batch, bucket_size), dtype=np.int64)
    actual_counts = np.zeros(batch, dtype=np.int32)
    for row, image_idx in enumerate(image_indices.tolist()):
        rots = per_image_inputs["oversampled_rots"][image_idx]
        cnt = int(rots.shape[0])
        actual_counts[row] = cnt
        if not device_rotations:
            padded_rotations[row, :cnt] = rots
            if separate_mstep_rotations:
                padded_mstep_rotations[row, :cnt] = per_image_inputs["oversampled_mstep_rots"][image_idx]
        padded_log_prior[row, :cnt] = per_image_inputs["log_prior"][image_idx]
        if include_dense_score_fields:
            padded_candidate_mask[row, :cnt, :] = _candidate_mask_to_dense(
                per_image_inputs["candidate_mask"][image_idx]
            )
            padded_parent_map[row, :cnt] = per_image_inputs["parent_map"][image_idx]
        padded_rotation_indices[row, :cnt] = per_image_inputs["oversampled_rot_indices"][image_idx]

    rotation_table = per_image_inputs.get("rotation_table") if isinstance(per_image_inputs, dict) else None
    resident_rows = None
    if isinstance(per_image_inputs, dict) and resident_hypothesis_tables_enabled():
        candidate = per_image_inputs.get("resident_hypothesis")
        if candidate is not None and candidate.get("rot_indices_flat") is not None:
            row_positions = np.asarray(candidate["positions"], dtype=np.int64)[padded_image_indices[:n_real]]
            if np.all(row_positions >= 0):
                resident_rows = (candidate, np.concatenate(
                    [row_positions, np.full(batch - n_real, -1, dtype=np.int64)]
                ) if batch > n_real else row_positions)
    if (
        device_rotations
        and rotation_table is not None
        and np.dtype(rotation_table.dtype) == np.dtype(rotation_dtype)
        and rotations_by_index_enabled()
    ):
        if resident_rows is not None:
            padded_rotations = padded_rotations_from_resident_device(
                rotation_table,
                per_image_inputs["rotation_table_key"],
                resident_rows[0],
                resident_rows[1],
                actual_counts,
                bucket_size,
                np.eye(3, dtype=rotation_dtype),
            )
        else:
            padded_rotations = padded_rotations_from_table_device(
                rotation_table,
                per_image_inputs["rotation_table_key"],
                padded_rotation_indices,
                actual_counts,
                bucket_size,
                np.eye(3, dtype=rotation_dtype),
            )
        mstep_table = per_image_inputs.get("mstep_rotation_table")
        if not separate_mstep_rotations:
            padded_mstep_rotations = padded_rotations
        elif mstep_table is not None and np.dtype(mstep_table.dtype) == np.dtype(mstep_rotation_dtype):
            padded_mstep_rotations = padded_rotations_from_table_device(
                mstep_table,
                per_image_inputs["mstep_rotation_table_key"],
                padded_rotation_indices,
                actual_counts,
                bucket_size,
                np.eye(3, dtype=mstep_rotation_dtype),
            )
        else:
            padded_mstep_rotations = padded_rows_from_flat_device(
                [np.asarray(per_image_inputs["oversampled_mstep_rots"][i], dtype=mstep_rotation_dtype) for i in padded_image_indices[:n_real].tolist()],
                actual_counts,
                bucket_size,
                np.eye(3, dtype=mstep_rotation_dtype),
            )
    elif device_rotations:
        real_images = padded_image_indices[:n_real].tolist()
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
        "image_indices": padded_image_indices,
        "bucket_size": bucket_size,
        "actual_counts": actual_counts,
        "rotations": padded_rotations,
        "mstep_rotations": padded_mstep_rotations,
        "rotation_indices": padded_rotation_indices,
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
    capacity_rows=None,
    device_rotations: bool = False,
    rotation_block_size_for_quantization=5000,
):
    """Build per-class padded arrays for fused sparse K-class pass 2.

    Default fused scoring keeps one rectangular bucket size shared by every
    class. The opt-in compact path keeps the same image chunk and joint
    class x pose normalization, but pads each class only to its class-local
    maximum rotation support inside that chunk.
    """

    class_arrays = []
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


def coarse_winner_local_pose_ids(per_image_inputs, coarse_pose_ids, fine_translation_parent, n_coarse_trans):
    """Locate retained coarse winners without reconstructing canonical Euler angles.

    Only the one-child-per-parent, zero-oversampling layout is supported.
    The returned local IDs index the existing per-image source-Euler/pose rows.
    """
    poses = np.asarray(coarse_pose_ids)
    if poses.shape != (len(per_image_inputs["unique_rot"]),) or not np.all(np.isfinite(poses)):
        raise ValueError("coarse winners must be one finite pose ID per image")
    if np.any(poses < 0) or not np.array_equal(poses, poses.astype(np.int64)):
        raise ValueError("coarse winners must be nonnegative integer pose IDs")
    trans_parents = np.asarray(fine_translation_parent, dtype=np.int64)
    output = np.empty(poses.size, dtype=np.int64)
    for image_index, pose_id in enumerate(poses.astype(np.int64)):
        coarse_rot, coarse_trans = divmod(int(pose_id), int(n_coarse_trans))
        rotation_parents = np.asarray(per_image_inputs["unique_rot"][image_index])[
            np.asarray(per_image_inputs["parent_map"][image_index])
        ]
        rotation_rows = np.flatnonzero(rotation_parents == coarse_rot)
        translation_rows = np.flatnonzero(trans_parents == coarse_trans)
        if rotation_rows.size != 1 or translation_rows.size != 1:
            raise ValueError("zero-oversampling coarse winner must have exactly one selected fine child")
        r, t = int(rotation_rows[0]), int(translation_rows[0])
        if not _candidate_mask_to_dense(per_image_inputs["candidate_mask"][image_index])[r, t]:
            raise ValueError("coarse winner is missing from selected fine support")
        output[image_index] = r * trans_parents.size + t
    return output


def _rows_at_capacity(values, capacity_rows, fill):
    """Return ``values`` extended along axis 0 to ``capacity_rows`` rows of ``fill`` (or as is)."""
    values = np.asarray(values)
    if capacity_rows is None or int(capacity_rows) <= int(values.shape[0]):
        return values
    out = np.full((int(capacity_rows),) + values.shape[1:], fill, dtype=values.dtype)
    out[: values.shape[0]] = values
    return out



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


def _pad_rows(values, capacity, fill):
    """Extend ``values`` along axis 0 to ``capacity`` rows filled with ``fill``."""

    if values is None:
        return None
    pad = int(capacity) - int(values.shape[0])
    if pad <= 0:
        return values  # device arrays stay on the device when nothing is padded
    if not isinstance(values, np.ndarray):
        import jax.numpy as jnp
        tail = jnp.full((pad,) + values.shape[1:], fill, dtype=values.dtype)
        return jnp.concatenate([values, tail], axis=0)
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
    padded["rotation_index"] = _pad_rows(pair_arrays.get("rotation_index"), capacity, 0)
    padded["log_prior"] = _pad_rows(pair_arrays.get("log_prior"), capacity, -1e30)
    padded["pair_mask"] = _pad_rows(pair_arrays["pair_mask"], capacity, False)
    return padded


def _padded_active_row_count(total: int, n_slots: int, pad_multiple: int) -> int:
    """The row count :func:`_real_flat_row_indices_from_actual_counts` would pad to."""

    total = int(total)
    if total <= 0:
        return 0
    pad_multiple = max(1, int(pad_multiple))
    padded_count = ((total + pad_multiple - 1) // pad_multiple) * pad_multiple
    if pad_multiple > 1:
        pow2_count = max(pad_multiple, 1 << (padded_count - 1).bit_length())
        if pow2_count <= padded_count + padded_count // 8:
            padded_count = pow2_count
    return min(int(n_slots), padded_count)


def _attach_group_static_active_row_targets(
    execution_buckets,
    per_image_inputs_by_class,
    *,
    pad_multiple: int,
    rotation_block_size_for_quantization,
) -> dict:
    """Store per-class active-row padding targets on every compact-pair bucket.

    Returns planner statistics: groups, chunks, real and padded row totals.
    """

    row_counts_by_class = [
        {int(i): int(np.asarray(r).shape[0]) for i, r in per_image_inputs["oversampled_rots"].items()}
        if isinstance(per_image_inputs["oversampled_rots"], dict)
        else [int(np.asarray(r).shape[0]) for r in per_image_inputs["oversampled_rots"]]
        for per_image_inputs in per_image_inputs_by_class
    ]
    capacity = image_capacity_enabled()
    per_bucket = []
    for bucket in execution_buckets:
        if str(bucket.get("_execution_mode")) != "compact_pair":
            per_bucket.append(None)
            continue
        image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
        forced = bucket.get("class_bucket_sizes")
        class_sizes = tuple(
            int(forced[c]) if forced is not None
            else int(_compact_bucket_size_for_class(bucket, per_image_inputs_by_class[c], rotation_block_size_for_quantization))
            for c in range(len(per_image_inputs_by_class))
        )
        n_images = int(image_indices.size)
        images_axis = (
            quantized_image_capacity(n_images, max_images=bucket.get("image_capacity_budget")) if capacity else n_images
        )
        totals = tuple(
            int(sum(min(row_counts_by_class[c][int(i)], class_sizes[c]) for i in image_indices))
            for c in range(len(class_sizes))
        )
        padded = tuple(
            _padded_active_row_count(totals[c], images_axis * class_sizes[c], pad_multiple) for c in range(len(class_sizes))
        )
        key = (int(bucket["_execution_bucket_size"]), class_sizes, int(images_axis))
        per_bucket.append((key, totals, padded))
    targets: dict = {}
    for entry in per_bucket:
        if entry is None:
            continue
        key, _totals, padded = entry
        current = targets.get(key)
        targets[key] = padded if current is None else tuple(max(a, b) for a, b in zip(current, padded))
    real_rows = padded_rows = target_rows = 0
    chunks = 0
    for bucket, entry in zip(execution_buckets, per_bucket):
        if entry is None:
            continue
        key, totals, padded = entry
        bucket["_active_row_pad_targets"] = targets[key]
        chunks += 1
        real_rows += sum(totals)
        padded_rows += sum(padded)
        target_rows += sum(targets[key])
    stats = {
        "groups": len(targets),
        "chunks": chunks,
        "real_rows": real_rows,
        "multiple_padded_rows": padded_rows,
        "group_static_rows": target_rows,
    }
    logger.info(
        "sparse K-class group-static active rows: %d groups, %d chunks, real rows %d, "
        "multiple-padded rows %d (+%.1f%%), group-static rows %d (+%.1f%%)",
        stats["groups"],
        stats["chunks"],
        real_rows,
        padded_rows,
        100.0 * (padded_rows - real_rows) / max(real_rows, 1),
        target_rows,
        100.0 * (target_rows - real_rows) / max(real_rows, 1),
    )
    return stats


def _group_static_pad_to(bucket_meta, class_index: int) -> int | None:
    targets = bucket_meta.get("_active_row_pad_targets")
    if targets is None:
        return None
    return int(targets[int(class_index)])


LADDER_CHUNKS_ENV = "RECOVAR_SPARSE_PASS2_LADDER_CHUNKS"
LADDER_CHUNK_FLOOR = 16

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
