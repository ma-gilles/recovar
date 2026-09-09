"""Host array assembly for rectangular and compact sparse pass-2 buckets.

Candidate selection and bucket scheduling remain with their callers. These
builders only gather/pad the supplied rows, preserving source image order,
precision, scoring/M-step rotation aliases and inert padding conventions.
"""

from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume.helpers.compact_candidates import (
    _candidate_mask_to_dense,
    build_compact_pair_index_arrays,
)
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size


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


def _build_compact_pair_bucket_arrays_from_per_image_inputs(bucket, per_image_inputs):
    """Stack/pad compact candidate pairs for one class and bucket on demand."""

    pair_bucket_size = int(bucket["pair_bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    index_arrays = build_compact_pair_index_arrays(
        (per_image_inputs["candidate_mask"][int(image_idx)] for image_idx in image_indices),
        pair_bucket_size=pair_bucket_size,
    )
    batch = int(image_indices.shape[0])
    padded_rotation_index = np.zeros((batch, pair_bucket_size), dtype=np.int64)
    log_prior_dtype = np.result_type(
        *(np.asarray(per_image_inputs["log_prior"][int(image_idx)]).dtype for image_idx in image_indices)
    )
    padded_log_prior = np.full((batch, pair_bucket_size), -1e30, dtype=log_prior_dtype)

    for row, image_idx in enumerate(image_indices.tolist()):
        count = int(index_arrays["pair_counts"][row])
        if count == 0:
            continue

        local_rot_rows = index_arrays["local_rotation_row"][row, :count]
        rotation_indices = np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64)
        rotation_log_prior = np.asarray(per_image_inputs["log_prior"][image_idx], dtype=log_prior_dtype)

        padded_rotation_index[row, :count] = rotation_indices[local_rot_rows]
        padded_log_prior[row, :count] = rotation_log_prior[local_rot_rows]

    return {
        "image_indices": image_indices,
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": index_arrays["pair_counts"],
        "local_rotation_row": index_arrays["local_rotation_row"],
        "translation_idx": index_arrays["translation_idx"],
        "rotation_index": padded_rotation_index,
        "log_prior": padded_log_prior,
        "pair_mask": index_arrays["pair_mask"],
    }


def _build_bucket_arrays(
    bucket,
    per_image_inputs,
    n_fine_trans,
    *,
    include_dense_score_fields: bool = True,
):
    """Stack/pad per-image arrays into batched bucket tensors."""
    bucket_size = int(bucket["bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    batch = int(image_indices.shape[0])

    # padded_rotations: identity-fill — projection of identity is harmless
    # because we mask via candidate_mask=False everywhere for padded rows.
    rotation_dtype = np.result_type(
        *(np.asarray(per_image_inputs["oversampled_rots"][int(image_idx)]).dtype for image_idx in image_indices)
    )
    padded_rotations = np.broadcast_to(
        np.eye(3, dtype=rotation_dtype),
        (batch, bucket_size, 3, 3),
    ).copy()
    separate_mstep_rotations = any(
        per_image_inputs["oversampled_mstep_rots"][int(image_idx)]
        is not per_image_inputs["oversampled_rots"][int(image_idx)]
        for image_idx in image_indices.tolist()
    )
    mstep_rotation_dtype = np.result_type(
        *(np.asarray(per_image_inputs["oversampled_mstep_rots"][int(image_idx)]).dtype for image_idx in image_indices)
    )
    padded_mstep_rotations = (
        np.broadcast_to(
            np.eye(3, dtype=mstep_rotation_dtype),
            (batch, bucket_size, 3, 3),
        ).copy()
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
    padded_rotation_indices = np.zeros((batch, bucket_size), dtype=np.int64)
    actual_counts = np.zeros(batch, dtype=np.int32)
    for row, image_idx in enumerate(image_indices.tolist()):
        rots = per_image_inputs["oversampled_rots"][image_idx]
        cnt = int(rots.shape[0])
        actual_counts[row] = cnt
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

    return {
        "image_indices": image_indices,
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
    rotation_block_size_for_quantization=5000,
):
    """Build per-class padded arrays for fused sparse K-class pass 2.

    Default fused scoring keeps one rectangular bucket size shared by every
    class. The opt-in compact path keeps the same image chunk and joint
    class x pose normalization, but pads each class only to its class-local
    maximum rotation support inside that chunk.
    """

    if not compact_buckets:
        return [
            _build_bucket_arrays(
                bucket,
                per_image_inputs,
                n_fine_trans,
                include_dense_score_fields=include_dense_score_fields,
            )
            for per_image_inputs in per_image_inputs_by_class
        ]

    class_arrays = []
    forced_class_bucket_sizes = bucket.get("class_bucket_sizes")
    for class_index, per_image_inputs in enumerate(per_image_inputs_by_class):
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
            )
        )
    return class_arrays
