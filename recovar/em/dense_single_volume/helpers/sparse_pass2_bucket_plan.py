"""Bucket planning of the sparse bucketed pass 2.

The hypothesis-bucket summaries and execution tags, the compact-pair plan
statistics and execution masks, the hybrid K-class bucket partition with its
validation and the threshold reports that explain a partition choice.
``sparse_pass2_bucketed`` plans every pass through these owners.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np

from recovar.em.dense_single_volume.helpers.compact_candidates import SparseCandidateMask, _candidate_mask_is_full
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_int_set
from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import (
    _DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    _bucket_sparse_k_class_compact_pair_counts,
    _compact_pair_counts_from_inputs,
    _prepare_per_image_compact_candidate_pairs,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_policy import _compact_pair_max_images_per_microbatch_for_pass

_COMPACT_KCLASS_PAIR_STATS_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS"


_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT_ENV = (
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT"
)


_DEFAULT_COMPACT_PAIR_THRESHOLD_REPORT = (8192, 16384, 32768, 65536, 131072)


class SparseKClassCompactPairPlanStats(NamedTuple):
    """Host-side accounting for the compact K-class pass-2 planner."""

    buckets: tuple[dict, ...]
    valid_pair_candidates: int
    padded_pair_candidates: int
    rectangular_candidates: int
    reduction_factor: float
    padded_reduction_factor: float
    median_valid_pairs_per_image: int
    mean_valid_pairs_per_image: float
    max_valid_pairs_per_image: int
    max_images_per_microbatch: int


def _candidate_mask_prefers_rectangular_execution(candidate_mask) -> bool:
    if isinstance(candidate_mask, SparseCandidateMask):
        return candidate_mask.mode in {"full", "coarse_exclude"}
    return _candidate_mask_is_full(candidate_mask)


def _compact_pair_execution_mask_excluding_full_support(per_image_inputs_by_class, image_mask):
    """Filter compact-pair execution away from masks with no sparse reduction.

    Compact pairs are a memory win only when they represent a strict subset of
    the rectangular rotation x translation tile.  A full-support mask would
    materialize the same candidate set in larger host-side arrays, which is
    pathological for the first global RELION-style iteration.
    """

    if not per_image_inputs_by_class:
        return image_mask, 0
    n_images = len(per_image_inputs_by_class[0]["candidate_mask"])
    for per_image_inputs in per_image_inputs_by_class[1:]:
        if len(per_image_inputs["candidate_mask"]) != n_images:
            raise ValueError("All classes must have the same image count for compact sparse pass-2")
    if image_mask is None:
        filtered = np.ones(n_images, dtype=bool)
        had_input_mask = False
    else:
        filtered = np.asarray(image_mask, dtype=bool).copy()
        if filtered.shape != (n_images,):
            raise ValueError(f"compact pair image mask shape mismatch: {filtered.shape} vs {(n_images,)}")
        had_input_mask = True

    excluded = 0
    for image_idx in np.flatnonzero(filtered):
        if any(
            _candidate_mask_prefers_rectangular_execution(per_image_inputs["candidate_mask"][int(image_idx)])
            for per_image_inputs in per_image_inputs_by_class
        ):
            filtered[int(image_idx)] = False
            excluded += 1

    if excluded == 0 and not had_input_mask:
        return None, 0
    return filtered, excluded


def _compact_k_class_pair_plan_stats(
    per_image_inputs_by_class,
    dense_buckets,
    n_fine_trans,
    *,
    pair_block_size_for_quantization=5000,
    max_pair_candidates_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
    image_mask=None,
) -> SparseKClassCompactPairPlanStats:
    """Compute compact-pair work counters without changing pass-2 execution."""

    compact_inputs_by_class = tuple(
        _prepare_per_image_compact_candidate_pairs(per_image_inputs)
        for per_image_inputs in per_image_inputs_by_class
    )
    return _compact_k_class_pair_plan_stats_from_counts(
        _compact_pair_counts_from_inputs(compact_inputs_by_class),
        dense_buckets,
        n_fine_trans,
        pair_block_size_for_quantization=pair_block_size_for_quantization,
        max_pair_candidates_per_microbatch=max_pair_candidates_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        tail_bucket_coalesce_max_images=tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation=tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        image_mask=image_mask,
    )


def _compact_k_class_pair_plan_stats_from_counts(
    pair_counts_by_class,
    dense_buckets,
    n_fine_trans,
    *,
    pair_block_size_for_quantization=5000,
    max_pair_candidates_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
    image_mask=None,
) -> SparseKClassCompactPairPlanStats:
    """Compute compact-pair work counters from per-image valid-pair counts."""

    if image_mask is not None:
        image_mask = np.asarray(image_mask, dtype=bool)
        n_images = int(np.asarray(pair_counts_by_class[0]).shape[0]) if pair_counts_by_class else 0
        if image_mask.shape != (n_images,):
            raise ValueError(f"compact pair image mask shape mismatch: {image_mask.shape} vs {(n_images,)}")

    compact_buckets = _bucket_sparse_k_class_compact_pair_counts(
        pair_counts_by_class,
        pair_block_size_for_quantization=pair_block_size_for_quantization,
        max_pair_candidates_per_microbatch=max_pair_candidates_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        image_mask=image_mask,
        tail_bucket_coalesce_max_images=tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation=tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size=tail_bucket_coalesce_min_bucket_size,
    )

    n_classes = len(pair_counts_by_class)
    valid_count_arrays = []
    for pair_counts in pair_counts_by_class:
        pair_counts = np.asarray(pair_counts, dtype=np.int64)
        if image_mask is not None:
            pair_counts = pair_counts[image_mask]
        valid_count_arrays.append(pair_counts)
    valid_counts = np.concatenate(valid_count_arrays) if valid_count_arrays else np.zeros(0, dtype=np.int64)
    valid_pair_candidates = int(valid_counts.sum(dtype=np.int64))
    padded_pair_candidates = int(
        sum(
            n_classes * len(bucket["image_indices"]) * int(bucket["pair_bucket_size"])
            for bucket in compact_buckets
        )
    )
    rectangular_candidates = int(
        sum(
            n_classes
            * (
                int(np.count_nonzero(image_mask[np.asarray(bucket["image_indices"], dtype=np.int64)]))
                if image_mask is not None
                else len(bucket["image_indices"])
            )
            * int(bucket["bucket_size"])
            * int(n_fine_trans)
            for bucket in dense_buckets
        )
    )
    reduction_factor = (
        float(rectangular_candidates) / float(valid_pair_candidates)
        if valid_pair_candidates > 0
        else float("inf")
    )
    padded_reduction_factor = (
        float(rectangular_candidates) / float(padded_pair_candidates)
        if padded_pair_candidates > 0
        else float("inf")
    )

    return SparseKClassCompactPairPlanStats(
        buckets=tuple(compact_buckets),
        valid_pair_candidates=valid_pair_candidates,
        padded_pair_candidates=padded_pair_candidates,
        rectangular_candidates=rectangular_candidates,
        reduction_factor=reduction_factor,
        padded_reduction_factor=padded_reduction_factor,
        median_valid_pairs_per_image=int(np.median(valid_counts)) if valid_counts.size else 0,
        mean_valid_pairs_per_image=float(np.mean(valid_counts)) if valid_counts.size else 0.0,
        max_valid_pairs_per_image=int(valid_counts.max(initial=0)) if valid_counts.size else 0,
        max_images_per_microbatch=max(1, int(max_images_per_microbatch)),
    )


def _maybe_prepare_sparse_k_class_compact_pair_plan(
    per_image_inputs_by_class,
    dense_buckets,
    n_fine_trans,
    *,
    max_pair_candidates_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
) -> SparseKClassCompactPairPlanStats | None:
    """Return compact-pair planner stats only when explicitly enabled.

    The planner is diagnostic only: dense rectangular scoring/M-step remains
    authoritative.
    """

    stats_flag = os.environ.get(_COMPACT_KCLASS_PAIR_STATS_ENV)
    if stats_flag is None or stats_flag.strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    compact_pair_max_images_per_microbatch = _compact_pair_max_images_per_microbatch_for_pass(
        max_images_per_microbatch,
    )
    return _compact_k_class_pair_plan_stats(
        per_image_inputs_by_class,
        dense_buckets,
        n_fine_trans,
        max_pair_candidates_per_microbatch=max_pair_candidates_per_microbatch,
        max_images_per_microbatch=compact_pair_max_images_per_microbatch,
        tail_bucket_coalesce_max_images=tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation=tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        image_mask=None,
    )


def _bucket_summary(buckets, size_key: str = "bucket_size") -> str:
    if not buckets:
        return "empty"
    sizes = np.asarray([int(bucket[size_key]) for bucket in buckets], dtype=np.int64)
    image_counts = np.asarray([len(bucket["image_indices"]) for bucket in buckets], dtype=np.int64)
    unique, counts = np.unique(sizes, return_counts=True)
    top = sorted(zip(unique.tolist(), counts.tolist(), strict=True), key=lambda item: item[1], reverse=True)[:8]
    return (
        f"bucket_size min/med/mean/max={int(sizes.min())}/{int(np.median(sizes))}/"
        f"{float(np.mean(sizes)):.1f}/{int(sizes.max())}, "
        f"images_per_bucket med/max={int(np.median(image_counts))}/{int(image_counts.max())}, "
        f"top_bucket_counts={top}"
    )


def _bucket_group_stats(buckets, size_key: str = "bucket_size") -> dict[int, tuple[int, int]]:
    stats: dict[int, list[int]] = {}
    for bucket in buckets:
        bucket_size = int(bucket[size_key])
        entry = stats.setdefault(bucket_size, [0, 0])
        entry[0] += 1
        entry[1] += len(bucket["image_indices"])
    return {bucket_size: (counts[0], counts[1]) for bucket_size, counts in stats.items()}


def _tag_k_class_execution_bucket(bucket, *, mode: str):
    tagged = dict(bucket)
    tagged["_execution_mode"] = mode
    if mode == "compact_pair":
        tagged["_execution_size_key"] = "pair_bucket_size"
    elif mode == "rectangular":
        tagged["_execution_size_key"] = "bucket_size"
    else:
        raise ValueError(f"Unknown sparse K-class execution mode {mode!r}")
    tagged["_execution_bucket_size"] = int(tagged[tagged["_execution_size_key"]])
    return tagged


def _k_class_execution_bucket_group_stats(buckets):
    stats: dict[tuple[str, str, int], list[int]] = {}
    for bucket in buckets:
        mode = str(bucket.get("_execution_mode", "rectangular"))
        size_key = str(bucket.get("_execution_size_key", "bucket_size"))
        bucket_size = int(bucket.get("_execution_bucket_size", bucket[size_key]))
        key = (mode, size_key, bucket_size)
        entry = stats.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += len(bucket["image_indices"])
    return {key: (counts[0], counts[1]) for key, counts in stats.items()}


def _hybrid_k_class_compact_pair_execution_buckets(
    dense_buckets,
    compact_pair_buckets,
    *,
    min_pair_bucket_size: int,
):
    """Route low pair-count images through rectangular buckets and high tails through compact pairs."""

    min_pair_bucket_size = int(min_pair_bucket_size)
    if min_pair_bucket_size <= 0:
        raise ValueError("min_pair_bucket_size must be positive")

    compact_execution_buckets = []
    compact_image_indices = set()
    for bucket in compact_pair_buckets:
        if int(bucket["pair_bucket_size"]) < min_pair_bucket_size:
            continue
        tagged = _tag_k_class_execution_bucket(bucket, mode="compact_pair")
        compact_execution_buckets.append(tagged)
        compact_image_indices.update(int(idx) for idx in np.asarray(bucket["image_indices"], dtype=np.int64))

    rectangular_execution_buckets = []
    for bucket in dense_buckets:
        image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
        if compact_image_indices:
            keep_mask = np.asarray([int(idx) not in compact_image_indices for idx in image_indices], dtype=bool)
            image_indices = image_indices[keep_mask]
        if image_indices.size == 0:
            continue
        rectangular_bucket = dict(bucket)
        rectangular_bucket["image_indices"] = np.asarray(image_indices, dtype=np.int64)
        rectangular_execution_buckets.append(
            _tag_k_class_execution_bucket(rectangular_bucket, mode="rectangular"),
        )

    return rectangular_execution_buckets + compact_execution_buckets


def _compact_pair_buckets_for_execution_threshold(compact_pair_buckets, min_pair_bucket_size: int | None):
    """Return compact buckets eligible for execution under the hybrid threshold."""

    if min_pair_bucket_size is None:
        return list(compact_pair_buckets)
    min_pair_bucket_size = int(min_pair_bucket_size)
    if min_pair_bucket_size <= 0:
        raise ValueError("min_pair_bucket_size must be positive")
    return [
        bucket
        for bucket in compact_pair_buckets
        if int(bucket["pair_bucket_size"]) >= min_pair_bucket_size
    ]


def _validate_k_class_execution_bucket_partition(execution_buckets, *, n_images: int) -> None:
    """Validate that execution buckets cover each image exactly once."""

    n_images = int(n_images)
    if n_images < 0:
        raise ValueError("n_images must be non-negative")
    if n_images == 0:
        if execution_buckets:
            raise ValueError("Execution buckets must be empty when n_images=0")
        return
    if not execution_buckets:
        raise ValueError(f"Execution buckets are empty for {n_images} images")

    image_indices = np.concatenate(
        [
            np.asarray(bucket["image_indices"], dtype=np.int64).reshape(-1)
            for bucket in execution_buckets
        ],
    )
    if image_indices.size != n_images:
        raise ValueError(
            "Execution bucket image coverage count mismatch: "
            f"got {image_indices.size}, expected {n_images}",
        )
    if int(image_indices.min(initial=0)) < 0 or int(image_indices.max(initial=-1)) >= n_images:
        raise ValueError(
            "Execution bucket image indices out of range for "
            f"{n_images} images",
        )

    counts = np.bincount(image_indices, minlength=n_images)
    missing = np.flatnonzero(counts == 0)
    duplicated = np.flatnonzero(counts > 1)
    if missing.size or duplicated.size:
        raise ValueError(
            "Execution buckets must partition images exactly once "
            f"(missing={missing[:8].tolist()}, duplicated={duplicated[:8].tolist()})",
        )


def _compact_pair_threshold_report_thresholds() -> tuple[int, ...]:
    raw_thresholds = parse_env_int_set(_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT_ENV)
    if raw_thresholds is None:
        return _DEFAULT_COMPACT_PAIR_THRESHOLD_REPORT
    return tuple(sorted(int(value) for value in raw_thresholds if int(value) > 0))


def _compact_pair_hybrid_threshold_reports(
    dense_buckets,
    compact_pair_buckets,
    *,
    thresholds: tuple[int, ...],
    n_classes: int,
    n_fine_trans: int,
):
    """Estimate hybrid compact-pair routing cost for candidate thresholds."""

    baseline_rectangular_candidates = int(
        sum(
            int(n_classes) * len(bucket["image_indices"]) * int(bucket["bucket_size"]) * int(n_fine_trans)
            for bucket in dense_buckets
        )
    )
    reports = []
    for threshold in thresholds:
        threshold = int(threshold)
        compact_buckets_for_threshold = [
            bucket
            for bucket in compact_pair_buckets
            if int(bucket["pair_bucket_size"]) >= threshold
        ]
        compact_image_indices = {
            int(idx)
            for bucket in compact_buckets_for_threshold
            for idx in np.asarray(bucket["image_indices"], dtype=np.int64)
        }
        compact_candidates = int(
            sum(
                int(n_classes) * len(bucket["image_indices"]) * int(bucket["pair_bucket_size"])
                for bucket in compact_buckets_for_threshold
            )
        )

        rectangular_buckets = 0
        rectangular_images = 0
        rectangular_candidates = 0
        for bucket in dense_buckets:
            image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
            if compact_image_indices:
                keep_mask = np.asarray(
                    [int(idx) not in compact_image_indices for idx in image_indices],
                    dtype=bool,
                )
                image_indices = image_indices[keep_mask]
            if image_indices.size == 0:
                continue
            rectangular_buckets += 1
            rectangular_images += int(image_indices.size)
            rectangular_candidates += (
                int(n_classes)
                * int(image_indices.size)
                * int(bucket["bucket_size"])
                * int(n_fine_trans)
            )

        total_candidates = int(rectangular_candidates + compact_candidates)
        reports.append(
            {
                "threshold": threshold,
                "compact_buckets": len(compact_buckets_for_threshold),
                "compact_images": len(compact_image_indices),
                "rectangular_buckets": rectangular_buckets,
                "rectangular_images": rectangular_images,
                "rectangular_candidate_slots": rectangular_candidates,
                "compact_candidate_slots": compact_candidates,
                "total_candidate_slots": total_candidates,
                "slot_reduction": (
                    float(baseline_rectangular_candidates) / float(total_candidates)
                    if total_candidates > 0
                    else float("inf")
                ),
            }
        )
    return reports
