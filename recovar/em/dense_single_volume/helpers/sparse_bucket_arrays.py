"""Host array assembly for rectangular and compact sparse pass-2 buckets.

Coarse selection and bucket scheduling remain with their callers. These
builders expand supplied parent support and gather/pad rows, preserving source order,
precision, scoring/M-step rotation aliases and inert padding conventions.
"""

from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume.helpers.compact_candidates import (
    SparseCandidateMask,
    _candidate_mask_to_dense,
    build_compact_pair_index_arrays,
)
from recovar.em.dense_single_volume.helpers.significant_samples import ComplementSignificantSampleIndices
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size


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

    for image_idx, sig_samples in enumerate(significant_sample_indices):
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
            translated_valid = coarse_valid[:, fine_translation_parent]
            candidate_mask = SparseCandidateMask(
                mode="coarse",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_valid=coarse_valid,
                fine_translation_parent=fine_translation_parent,
                count=int(translated_valid[parent_map].sum()),
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
            )
        )
    return class_arrays
