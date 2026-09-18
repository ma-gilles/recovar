"""High-resolution PPCA pose refinement over the K-class pose hierarchy."""

from __future__ import annotations

import numpy as np

from recovar.em.local.local_layout import (
    LocalHypothesisLayout,
    build_local_hypothesis_layout,
)
from recovar.em.sampling import build_local_search_grid_metadata


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    max_value = np.max(values, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(max_value), max_value, 0.0)
    summed = np.sum(np.exp(np.where(np.isfinite(values), values - safe_max, -np.inf)), axis=axis)
    with np.errstate(divide="ignore"):
        return np.squeeze(safe_max, axis=axis) + np.log(summed)


def build_top_p_local_hypothesis_layout(
    top_rotation_ids,
    top_translation_idx,
    *,
    center_rotation_grid,
    top_rotation_matrices=None,
    center_translation_grid=None,
    target_rotation_grid,
    healpix_order: int,
    translations,
    sigma_rot: float,
    sigma_psi: float,
    sigma_offset_angstrom: float,
    voxel_size: float,
    grid_metadata=None,
) -> LocalHypothesisLayout:
    """Build exact-local support as the union of top-p pose neighborhoods."""

    top_rotation_ids = np.asarray(top_rotation_ids, dtype=np.int64)
    top_translation_idx = np.asarray(top_translation_idx, dtype=np.int64)
    if top_rotation_ids.shape != top_translation_idx.shape:
        raise ValueError(f"top pose shapes differ: {top_rotation_ids.shape} vs {top_translation_idx.shape}")
    if top_rotation_ids.ndim == 1:
        top_rotation_ids = top_rotation_ids[:, None]
        top_translation_idx = top_translation_idx[:, None]
    top_mats = None
    if top_rotation_matrices is not None:
        top_mats = np.asarray(top_rotation_matrices, dtype=np.float32)
        if top_mats.ndim == 3:
            top_mats = top_mats[:, None, :, :]
        if top_mats.shape[:2] != top_rotation_ids.shape or top_mats.shape[-2:] != (3, 3):
            raise ValueError(
                "top_rotation_matrices must have shape "
                f"{top_rotation_ids.shape} + (3, 3), got {top_mats.shape}"
            )
    translations_np = np.asarray(translations, dtype=np.float32)
    center_translations_np = (
        translations_np
        if center_translation_grid is None
        else np.asarray(center_translation_grid, dtype=np.float32).reshape(-1, translations_np.shape[1])
    )
    center_grid = np.asarray(center_rotation_grid, dtype=np.float32).reshape(-1, 3, 3)
    target_grid = np.asarray(target_rotation_grid, dtype=np.float32).reshape(-1, 3, 3)
    metadata = build_local_search_grid_metadata(int(healpix_order)) if grid_metadata is None else grid_metadata

    offsets = np.zeros(top_rotation_ids.shape[0] + 1, dtype=np.int64)
    counts = np.zeros(top_rotation_ids.shape[0], dtype=np.int32)
    rotation_ids_parts: list[np.ndarray] = []
    rotations_parts: list[np.ndarray] = []
    rotation_prior_parts: list[np.ndarray] = []
    translation_prior_rows: list[np.ndarray] = []

    for image_idx in range(top_rotation_ids.shape[0]):
        valid = (top_rotation_ids[image_idx] >= 0) & (top_translation_idx[image_idx] >= 0)
        if not np.any(valid):
            valid = np.zeros_like(top_rotation_ids[image_idx], dtype=bool)
            valid[0] = True
            top_rotation_ids[image_idx, 0] = 0
            top_translation_idx[image_idx, 0] = 0
        center_ids = top_rotation_ids[image_idx, valid]
        center_trans = top_translation_idx[image_idx, valid]
        if top_mats is None:
            if int(np.max(center_ids, initial=0)) >= center_grid.shape[0]:
                raise ValueError("top_rotation_ids exceed center_rotation_grid size")
            center_mats = center_grid[center_ids]
        else:
            center_mats = top_mats[image_idx, valid]
        if int(np.max(center_trans, initial=0)) >= center_translations_np.shape[0]:
            raise ValueError("top_translation_idx exceed center_translation_grid size")
        center_translations = center_translations_np[center_trans]
        local = build_local_hypothesis_layout(
            center_mats,
            target_grid,
            float(sigma_rot),
            float(sigma_psi),
            int(healpix_order),
            translations_np,
            center_translations,
            float(sigma_offset_angstrom),
            None,
            float(voxel_size),
            grid_metadata=metadata,
        )
        unique_ids, first_idx, inverse = np.unique(
            np.asarray(local.rotation_ids_flat, dtype=np.int32),
            return_index=True,
            return_inverse=True,
        )
        priors = np.full(unique_ids.shape[0], -np.inf, dtype=np.float64)
        for local_idx, inverse_idx in enumerate(inverse):
            priors[inverse_idx] = np.logaddexp(priors[inverse_idx], float(local.rotation_log_priors_flat[local_idx]))
        priors = (priors - np.log(max(1, center_ids.size))).astype(np.float32)
        rotations_selected = np.asarray(local.rotations_flat, dtype=np.float32)[first_idx]
        trans_prior = _logsumexp_np(np.asarray(local.translation_log_priors, dtype=np.float32), axis=0)
        trans_prior = (trans_prior - np.log(max(1, center_ids.size))).astype(np.float32)

        counts[image_idx] = int(unique_ids.shape[0])
        offsets[image_idx + 1] = offsets[image_idx] + int(unique_ids.shape[0])
        rotation_ids_parts.append(unique_ids.astype(np.int32, copy=False))
        rotations_parts.append(rotations_selected.astype(np.float32, copy=False))
        rotation_prior_parts.append(priors)
        translation_prior_rows.append(trans_prior)

    return LocalHypothesisLayout(
        n_global_rotations=int(target_grid.shape[0]),
        n_pixels=int(metadata["n_pixels"]),
        n_psi=int(metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=np.concatenate(rotation_ids_parts) if rotation_ids_parts else np.zeros(0, dtype=np.int32),
        rotations_flat=(
            np.concatenate(rotations_parts, axis=0) if rotations_parts else np.zeros((0, 3, 3), dtype=np.float32)
        ),
        rotation_log_priors_flat=(
            np.concatenate(rotation_prior_parts) if rotation_prior_parts else np.zeros(0, dtype=np.float32)
        ),
        rotation_counts=counts,
        translation_grid=translations_np,
        translation_log_priors=(
            np.stack(translation_prior_rows, axis=0)
            if translation_prior_rows
            else np.zeros((0, translations_np.shape[0]), dtype=np.float32)
        ),
    )
