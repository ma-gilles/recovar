"""Host candidate masks and source-ordered compact pair/fine-job layouts.

Numerical execution stays in the local and sparse engines. This module owns
mask materialization, index order, valid prefixes and static padding only.
"""

from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size


class SparseCandidateMask:
    """Compact host representation of one image's pass-2 candidate mask."""

    __slots__ = (
        "mode",
        "n_rows",
        "n_fine_trans",
        "parent_map",
        "coarse_valid",
        "coarse_excluded",
        "fine_translation_parent",
        "count",
    )

    def __init__(
        self,
        *,
        mode: str,
        n_rows: int,
        n_fine_trans: int,
        parent_map=None,
        coarse_valid=None,
        coarse_excluded=None,
        fine_translation_parent=None,
        count: int | None = None,
    ):
        self.mode = str(mode)
        self.n_rows = int(n_rows)
        self.n_fine_trans = int(n_fine_trans)
        self.parent_map = None if parent_map is None else np.asarray(parent_map, dtype=np.int32)
        self.coarse_valid = None if coarse_valid is None else np.asarray(coarse_valid, dtype=bool)
        self.coarse_excluded = None if coarse_excluded is None else np.asarray(coarse_excluded, dtype=np.int32)
        self.fine_translation_parent = (
            None if fine_translation_parent is None else np.asarray(fine_translation_parent, dtype=np.int32)
        )
        self.count = int(_dense_candidate_mask_from_spec(self).sum()) if count is None else int(count)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_fine_trans)

    def __array__(self, dtype=None, copy=None):
        dense = _dense_candidate_mask_from_spec(self)
        if dtype is not None:
            return dense.astype(dtype, copy=False if copy is None else bool(copy))
        if copy:
            return dense.copy()
        return dense


def _dense_candidate_mask_from_spec(mask: SparseCandidateMask) -> np.ndarray:
    if mask.mode == "full":
        return np.ones(mask.shape, dtype=bool)
    if mask.mode == "empty":
        return np.zeros(mask.shape, dtype=bool)
    if mask.mode == "coarse":
        if mask.coarse_valid is None or mask.parent_map is None or mask.fine_translation_parent is None:
            raise ValueError("coarse candidate mask spec is missing parent/coarse arrays")
        return mask.coarse_valid[:, mask.fine_translation_parent][mask.parent_map]
    if mask.mode == "coarse_exclude":
        if mask.coarse_excluded is None or mask.parent_map is None or mask.fine_translation_parent is None:
            raise ValueError("coarse_exclude candidate mask spec is missing excluded/parent arrays")
        dense = np.ones(mask.shape, dtype=bool)
        excluded = np.asarray(mask.coarse_excluded, dtype=np.int64).reshape(-1)
        if excluded.size:
            n_coarse_trans = int(mask.fine_translation_parent.max(initial=-1) + 1)
            if n_coarse_trans <= 0:
                raise ValueError("coarse_exclude candidate mask has empty translation parent map")
            excluded_rot = excluded // n_coarse_trans
            excluded_trans = excluded % n_coarse_trans
            for coarse_rot, coarse_trans in zip(excluded_rot.tolist(), excluded_trans.tolist(), strict=False):
                rows = np.flatnonzero(mask.parent_map == int(coarse_rot))
                cols = np.flatnonzero(mask.fine_translation_parent == int(coarse_trans))
                if rows.size and cols.size:
                    dense[np.ix_(rows, cols)] = False
        return dense
    raise ValueError(f"Unknown sparse candidate mask mode {mask.mode!r}")


def _candidate_mask_to_dense(candidate_mask) -> np.ndarray:
    if isinstance(candidate_mask, SparseCandidateMask):
        return _dense_candidate_mask_from_spec(candidate_mask)
    return np.asarray(candidate_mask, dtype=bool)


def _candidate_mask_count(candidate_mask) -> int:
    if isinstance(candidate_mask, SparseCandidateMask):
        return int(candidate_mask.count)
    return int(np.asarray(candidate_mask, dtype=bool).sum())


def _candidate_mask_is_full(candidate_mask) -> bool:
    if isinstance(candidate_mask, SparseCandidateMask):
        total = int(candidate_mask.n_rows) * int(candidate_mask.n_fine_trans)
        return total > 0 and int(candidate_mask.count) >= total
    dense = np.asarray(candidate_mask, dtype=bool)
    return dense.size > 0 and bool(np.all(dense))


def compact_candidate_indices_in_source_order(candidate_mask):
    """Return compact ``(rotation, translation)`` ids in dense source order.

    Compact pass 2 and the exact-local scorer must agree on one ordering:
    rotations are the major axis and translations are the minor axis, exactly
    as if the dense ``(R, T)`` mask had been flattened in C order.  Keep this
    encoder shared so selected-pair CUDA paths cannot silently invent a
    different posterior order.
    """

    if isinstance(candidate_mask, SparseCandidateMask):
        if candidate_mask.mode == "empty":
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        if candidate_mask.mode == "full":
            rows = np.repeat(np.arange(candidate_mask.n_rows, dtype=np.int64), candidate_mask.n_fine_trans)
            trans = np.tile(np.arange(candidate_mask.n_fine_trans, dtype=np.int64), candidate_mask.n_rows)
            return rows, trans
        if candidate_mask.mode == "coarse_exclude":
            dense = _dense_candidate_mask_from_spec(candidate_mask)
            return np.nonzero(dense)
    return np.nonzero(_candidate_mask_to_dense(candidate_mask))


def build_compact_pair_index_arrays(
    candidate_masks,
    *,
    pair_bucket_size: int | None = None,
    pair_block_size_for_quantization: int = 5000,
):
    """Pack candidate masks with the mature compact-pass-2 pair ABI.

    Valid pairs occupy a source-ordered prefix of each image row. Padding uses
    ``-1`` indices plus a false ``pair_mask``.  When no explicit capacity is
    supplied, use the same compile-friendly bucket quantization as compact
    pass 2 instead of creating exact-count shape families.
    """

    candidate_masks = tuple(candidate_masks)
    compact_indices = tuple(
        compact_candidate_indices_in_source_order(candidate_mask) for candidate_mask in candidate_masks
    )
    pair_counts = np.asarray(
        [rotation_rows.shape[0] for rotation_rows, _ in compact_indices],
        dtype=np.int32,
    )
    required_capacity = int(pair_counts.max(initial=0))
    if pair_bucket_size is None:
        pair_bucket_size = _exact_bucket_rotation_size(
            required_capacity,
            pair_block_size_for_quantization,
        )
    pair_bucket_size = int(pair_bucket_size)
    if pair_bucket_size <= 0:
        raise ValueError("compact pair bucket size must be positive")
    if required_capacity > pair_bucket_size:
        raise ValueError(
            "compact pair bucket is smaller than the source-ordered candidate "
            f"prefix: required={required_capacity}, capacity={pair_bucket_size}"
        )

    batch_size = len(candidate_masks)
    local_rotation_row = np.full(
        (batch_size, pair_bucket_size),
        -1,
        dtype=np.int32,
    )
    translation_idx = np.full_like(local_rotation_row, -1)
    pair_mask = np.zeros((batch_size, pair_bucket_size), dtype=bool)
    for image_row, (rotation_rows, translation_ids) in enumerate(compact_indices):
        count = int(rotation_rows.shape[0])
        if count == 0:
            continue
        local_rotation_row[image_row, :count] = rotation_rows.astype(
            np.int32,
            copy=False,
        )
        translation_idx[image_row, :count] = translation_ids.astype(
            np.int32,
            copy=False,
        )
        pair_mask[image_row, :count] = True
    return {
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": pair_counts,
        "local_rotation_row": local_rotation_row,
        "translation_idx": translation_idx,
        "pair_mask": pair_mask,
    }


def build_compact_fine_job_plan(
    candidate_masks,
    reference_row_lookup,
    *,
    job_bucket_size: int | None = None,
    job_block_size_for_quantization: int = 5000,
):
    """Pack all selected fine hypotheses into one global source-order plan.

    The mature pair encoder quantizes a separate capacity for every image and
    therefore executes ``B * max(P_i)`` slots.  This companion ABI preserves
    the identical image-major, rotation-major, translation-major order while
    quantizing only the total selected count.  Rows encode ``(image,
    projected-reference row, dense rotation row, translation)``; an all-``-1``
    tail is inert static padding for JAX compilation reuse.
    """

    candidate_masks = tuple(np.asarray(mask, dtype=bool) for mask in candidate_masks)
    if not candidate_masks:
        raise ValueError("compact fine jobs require at least one image mask")
    first_shape = candidate_masks[0].shape
    if len(first_shape) != 2 or first_shape[0] <= 0 or first_shape[1] <= 0:
        raise ValueError("compact fine-job masks must have nonempty (rotation, translation) shape")
    if any(mask.shape != first_shape for mask in candidate_masks):
        raise ValueError("compact fine-job masks must share one dense shape")

    batch_size = len(candidate_masks)
    rotation_count, _ = first_shape
    reference_row_lookup = np.asarray(reference_row_lookup)
    if reference_row_lookup.dtype != np.int32 or reference_row_lookup.shape != (batch_size, rotation_count):
        raise ValueError(
            "compact fine-job reference lookup must be int32 with shape "
            f"{(batch_size, rotation_count)}, got "
            f"{reference_row_lookup.shape} {reference_row_lookup.dtype}"
        )

    pair_arrays = build_compact_pair_index_arrays(candidate_masks)
    return build_compact_fine_job_plan_from_pair_arrays(
        pair_arrays,
        reference_row_lookup,
        job_bucket_size=job_bucket_size,
        job_block_size_for_quantization=job_block_size_for_quantization,
    )


def build_compact_fine_job_plan_from_pair_arrays(
    pair_arrays,
    reference_row_lookup,
    *,
    job_bucket_size: int | None = None,
    job_block_size_for_quantization: int = 5000,
):
    """Collapse the mature per-image pair ABI into one global job prefix."""

    local_rotation_row = np.asarray(pair_arrays["local_rotation_row"])
    translation_idx = np.asarray(pair_arrays["translation_idx"])
    pair_mask = np.asarray(pair_arrays["pair_mask"])
    job_counts = np.asarray(pair_arrays["pair_counts"])
    if (
        local_rotation_row.dtype != np.int32
        or translation_idx.dtype != np.int32
        or pair_mask.dtype != np.bool_
        or job_counts.dtype != np.int32
        or local_rotation_row.ndim != 2
        or translation_idx.shape != local_rotation_row.shape
        or pair_mask.shape != local_rotation_row.shape
        or job_counts.shape != (local_rotation_row.shape[0],)
    ):
        raise ValueError("compact pair arrays are not aligned with the mature ABI")
    expected_mask = np.arange(local_rotation_row.shape[1])[None, :] < job_counts[:, None]
    if not np.array_equal(pair_mask, expected_mask):
        raise ValueError("compact pair validity must be a source-ordered prefix")

    batch_size = int(local_rotation_row.shape[0])
    reference_row_lookup = np.asarray(reference_row_lookup)
    if reference_row_lookup.dtype != np.int32 or reference_row_lookup.ndim != 2:
        raise ValueError("compact fine-job reference lookup must be a 2-D int32 array")
    if reference_row_lookup.shape[0] != batch_size:
        raise ValueError("compact fine-job reference lookup batch axis is misaligned")
    valid_job_count = int(np.sum(job_counts, dtype=np.int64))
    if job_bucket_size is None:
        job_bucket_size = _exact_bucket_rotation_size(
            valid_job_count,
            job_block_size_for_quantization,
        )
    job_bucket_size = int(job_bucket_size)
    if job_bucket_size <= 0:
        raise ValueError("compact fine-job bucket size must be positive")
    if valid_job_count > job_bucket_size:
        raise ValueError(
            "compact fine-job bucket is smaller than the source-ordered prefix: "
            f"required={valid_job_count}, capacity={job_bucket_size}"
        )

    job_plan = np.full((job_bucket_size, 4), -1, dtype=np.int32)
    cursor = 0
    for image_row, count_value in enumerate(job_counts):
        count = int(count_value)
        if count == 0:
            continue
        rotation_rows = local_rotation_row[image_row, :count]
        translation_ids = translation_idx[image_row, :count]
        if np.any(rotation_rows < 0) or np.any(rotation_rows >= reference_row_lookup.shape[1]):
            raise ValueError("a selected fine job has an invalid dense rotation row")
        if np.any(translation_ids < 0):
            raise ValueError("a selected fine job has an invalid translation id")
        reference_rows = reference_row_lookup[image_row, rotation_rows]
        if np.any(reference_rows < 0):
            raise ValueError("a selected fine job has no projected-reference row")
        next_cursor = cursor + count
        job_plan[cursor:next_cursor, 0] = image_row
        job_plan[cursor:next_cursor, 1] = reference_rows
        job_plan[cursor:next_cursor, 2] = rotation_rows
        job_plan[cursor:next_cursor, 3] = translation_ids
        cursor = next_cursor

    return {
        "job_bucket_size": job_bucket_size,
        "job_counts": job_counts,
        "valid_job_count": valid_job_count,
        "job_plan": job_plan,
    }


# Preserve the stored class identity; the sparse owner imports this same class.
SparseCandidateMask.__module__ = "recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed"
