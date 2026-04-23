"""Per-image local hypothesis layout and bucketization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recovar import utils
from recovar.em.dense_single_volume.helpers.local_search import _local_search_engine_rotation_block_size
from recovar.em.dense_single_volume.helpers.orientation_priors import make_relion_translation_log_prior
from recovar.em.sampling import (
    _normalized_log_weights,
    _wrapped_abs_diff_deg,
    get_local_rotation_grid_fast,
    rotation_indices_to_relion_eulers,
)


def _exact_bucket_rotation_size(local_rotation_count: int, rotation_block_size: int) -> int:
    """Return a compile-friendly padded size for one exact local neighborhood.

    Unlike the grouped-union path, the exact local engine cannot safely cap the
    bucket size below the true per-image neighborhood cardinality. Use the same
    power-of-two style padding for smaller neighborhoods. For larger exact
    neighborhoods, round up to a coarse fixed quantum so nearby local-support
    sizes reuse the same compiled shapes instead of each exact count generating
    its own XLA program.
    """
    local_rotation_count = int(local_rotation_count)
    if local_rotation_count <= 0:
        return 1
    engine_cap = int(_local_search_engine_rotation_block_size(rotation_block_size))
    if local_rotation_count <= engine_cap:
        bucket = 1 << max(int(local_rotation_count - 1).bit_length(), 4)
        return int(max(local_rotation_count, min(bucket, engine_cap)))
    large_bucket_quantum = max(64, engine_cap // 8)
    return int(((local_rotation_count + large_bucket_quantum - 1) // large_bucket_quantum) * large_bucket_quantum)


@dataclass(frozen=True)
class LocalHypothesisLayout:
    """Flat per-image local hypothesis storage."""

    n_global_rotations: int
    n_pixels: int
    n_psi: int
    rotation_offsets: np.ndarray
    rotation_ids_flat: np.ndarray
    rotations_flat: np.ndarray
    rotation_log_priors_flat: np.ndarray
    rotation_counts: np.ndarray
    translation_grid: np.ndarray
    translation_log_priors: np.ndarray

    @property
    def n_images(self) -> int:
        return int(self.rotation_counts.shape[0])

    @property
    def total_local_rotations(self) -> int:
        return int(self.rotation_ids_flat.shape[0])


@dataclass(frozen=True)
class LocalBucketSpec:
    """Static-shape padded execution batch for the exact local engine."""

    image_indices: np.ndarray
    bucket_rotation_count: int
    actual_rotation_counts: np.ndarray
    local_rotation_ids: np.ndarray
    local_rotations: np.ndarray
    local_rotation_log_prior: np.ndarray
    local_rotation_mask: np.ndarray
    translation_log_prior: np.ndarray


def _resolve_prior_rotations(prior_rotations: np.ndarray, healpix_order: int, grid_metadata):
    """Return RELION eulers and rotation matrices for local-support construction."""

    prior_rotations = np.asarray(prior_rotations)
    if prior_rotations.ndim == 0:
        prior_rotations = prior_rotations.reshape(1)

    if prior_rotations.ndim == 1:
        if "eulers_full" in grid_metadata:
            prior_eulers = np.asarray(grid_metadata["eulers_full"], dtype=np.float32)[prior_rotations.astype(np.int64)]
        else:
            prior_eulers = rotation_indices_to_relion_eulers(prior_rotations.astype(np.int64), healpix_order)
        prior_rotation_mats = utils.R_from_relion(prior_eulers, degrees=True)
        return np.asarray(prior_eulers, dtype=np.float32), np.asarray(prior_rotation_mats, dtype=np.float64)
    if prior_rotations.ndim == 2 and prior_rotations.shape[-1] == 3:
        prior_eulers = np.asarray(prior_rotations, dtype=np.float32).reshape(-1, 3)
        prior_rotation_mats = utils.R_from_relion(prior_eulers, degrees=True)
        return prior_eulers, np.asarray(prior_rotation_mats, dtype=np.float64)
    prior_rotation_mats = np.asarray(prior_rotations, dtype=np.float64).reshape(-1, 3, 3)
    prior_eulers = utils.R_to_relion(prior_rotation_mats, degrees=True).astype(np.float32)
    return prior_eulers, prior_rotation_mats


def _build_factorized_local_entries(
    prior_rotations: np.ndarray,
    healpix_order: int,
    sigma_rot: float,
    sigma_psi: float,
    grid_metadata,
):
    """Build exact per-image local supports for factorized HEALPix x psi grids."""

    prior_eulers, prior_rotation_mats = _resolve_prior_rotations(prior_rotations, healpix_order, grid_metadata)
    dir_vecs = np.asarray(grid_metadata["dir_vecs"], dtype=np.float64)
    psi_deg_grid = np.asarray(grid_metadata["psi_deg"], dtype=np.float64)
    n_pixels = int(grid_metadata["n_pixels"])

    prior_dir_vecs = np.asarray(prior_rotation_mats[:, 2, :], dtype=np.float64)
    prior_dir_norm = np.linalg.norm(prior_dir_vecs, axis=1, keepdims=True)
    prior_dir_norm = np.where(prior_dir_norm > 0.0, prior_dir_norm, 1.0)
    prior_dir_vecs = prior_dir_vecs / prior_dir_norm
    prior_psi_deg = np.mod(np.asarray(prior_eulers[:, 2], dtype=np.float64), 360.0)

    sigma_rot_deg = float(np.rad2deg(sigma_rot))
    sigma_psi_deg = float(np.rad2deg(sigma_psi))
    biggest_sigma_deg = float(max(sigma_rot_deg, sigma_psi_deg))
    cutoff_dir_deg = 3.0 * biggest_sigma_deg
    cutoff_psi_deg = 3.0 * sigma_psi_deg

    if sigma_rot_deg > 0.0:
        dots = np.clip(prior_dir_vecs @ dir_vecs.T, -1.0, 1.0)
        diffang = np.rad2deg(np.arccos(dots))
    else:
        diffang = None

    if sigma_psi_deg > 0.0:
        diffpsi = _wrapped_abs_diff_deg(psi_deg_grid[None, :], prior_psi_deg[:, None])
    else:
        diffpsi = None

    rotation_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    counts = np.zeros(prior_eulers.shape[0], dtype=np.int32)
    offsets = np.zeros(prior_eulers.shape[0] + 1, dtype=np.int64)

    for image_idx in range(prior_eulers.shape[0]):
        if sigma_rot_deg > 0.0:
            dir_mask = diffang[image_idx] < cutoff_dir_deg
            dir_indices = np.flatnonzero(dir_mask).astype(np.int64)
            if dir_indices.size == 0:
                dir_indices = np.array([int(np.argmin(diffang[image_idx]))], dtype=np.int64)
                dir_log_prior = np.zeros(1, dtype=np.float32)
            else:
                dir_log_prior = _normalized_log_weights(diffang[image_idx, dir_indices], biggest_sigma_deg)
        else:
            dir_indices = np.arange(n_pixels, dtype=np.int64)
            dir_log_prior = np.full(n_pixels, -np.log(max(n_pixels, 1)), dtype=np.float32)

        if sigma_psi_deg > 0.0:
            psi_mask = diffpsi[image_idx] < cutoff_psi_deg
            psi_indices = np.flatnonzero(psi_mask).astype(np.int64)
            if psi_indices.size == 0:
                psi_indices = np.array([int(np.argmin(diffpsi[image_idx]))], dtype=np.int64)
                psi_log_prior = np.zeros(1, dtype=np.float32)
            else:
                psi_log_prior = _normalized_log_weights(diffpsi[image_idx, psi_indices], sigma_psi_deg)
        else:
            psi_indices = np.arange(int(grid_metadata["n_psi"]), dtype=np.int64)
            psi_log_prior = np.full(psi_indices.shape[0], -np.log(max(psi_indices.shape[0], 1)), dtype=np.float32)

        local_ids = (psi_indices[:, None] * n_pixels + dir_indices[None, :]).reshape(-1).astype(np.int32)
        local_log_prior = (psi_log_prior[:, None] + dir_log_prior[None, :]).reshape(-1).astype(np.float32)
        counts[image_idx] = int(local_ids.shape[0])
        offsets[image_idx + 1] = offsets[image_idx] + local_ids.shape[0]
        rotation_ids_parts.append(local_ids)
        log_prior_parts.append(local_log_prior)

    rotation_ids_flat = np.concatenate(rotation_ids_parts, axis=0) if rotation_ids_parts else np.zeros(0, dtype=np.int32)
    rotation_log_priors_flat = np.concatenate(log_prior_parts, axis=0) if log_prior_parts else np.zeros(0, dtype=np.float32)
    return offsets, counts, rotation_ids_flat, rotation_log_priors_flat


def build_local_hypothesis_layout(
    prior_rotations: np.ndarray,
    rotation_grid_rotations: np.ndarray,
    sigma_rot: float,
    sigma_psi: float,
    healpix_order: int,
    translations: np.ndarray,
    prior_translations: np.ndarray,
    sigma_offset_angstrom: float,
    offset_range_pixels: float | None,
    voxel_size: float,
    *,
    grid_metadata,
    translation_prior_reference_translations: np.ndarray | None = None,
) -> LocalHypothesisLayout:
    """Build exact per-image local neighborhoods and translation priors."""

    prior_rotations = np.asarray(prior_rotations, dtype=np.float32)
    rotation_grid_rotations = np.asarray(rotation_grid_rotations, dtype=np.float32).reshape(-1, 3, 3)
    translations = np.asarray(translations, dtype=np.float32)
    prior_translations = np.asarray(prior_translations, dtype=np.float32).reshape(-1, translations.shape[1])

    if str(grid_metadata["mode"]) == "factorized":
        offsets, counts, rotation_ids_flat, rotation_log_priors_flat = _build_factorized_local_entries(
            prior_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            grid_metadata,
        )
    else:
        n_images = int(prior_rotations.shape[0])
        offsets = np.zeros(n_images + 1, dtype=np.int64)
        counts = np.zeros(n_images, dtype=np.int32)
        rotation_ids_parts: list[np.ndarray] = []
        log_prior_parts: list[np.ndarray] = []

        for image_idx in range(n_images):
            local_ids, local_log_prior = get_local_rotation_grid_fast(
                prior_rotations[image_idx : image_idx + 1],
                sigma_rot,
                sigma_psi,
                healpix_order,
                sigma_cutoff=3.0,
                per_image=True,
                grid_metadata=grid_metadata,
            )
            local_ids = np.asarray(local_ids, dtype=np.int32).reshape(-1)
            local_log_prior = np.asarray(local_log_prior[0], dtype=np.float32).reshape(-1)
            counts[image_idx] = int(local_ids.shape[0])
            offsets[image_idx + 1] = offsets[image_idx] + local_ids.shape[0]
            rotation_ids_parts.append(local_ids)
            log_prior_parts.append(local_log_prior)

        rotation_ids_flat = np.concatenate(rotation_ids_parts, axis=0) if rotation_ids_parts else np.zeros(0, dtype=np.int32)
        rotation_log_priors_flat = np.concatenate(log_prior_parts, axis=0) if log_prior_parts else np.zeros(0, dtype=np.float32)

    rotations_flat = rotation_grid_rotations[rotation_ids_flat] if rotation_ids_flat.size else np.zeros((0, 3, 3), dtype=np.float32)

    reference_translations = (
        np.asarray(translation_prior_reference_translations, dtype=np.float32)
        if translation_prior_reference_translations is not None
        else translations
    )

    translation_log_priors = make_relion_translation_log_prior(
        reference_translations,
        voxel_size,
        sigma_offset_angstrom,
        prior_translations,
        offset_range_pixels=offset_range_pixels,
    ).astype(np.float32, copy=False)

    return LocalHypothesisLayout(
        n_global_rotations=int(rotation_grid_rotations.shape[0]),
        n_pixels=int(grid_metadata["n_pixels"]),
        n_psi=int(grid_metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=translations,
        translation_log_priors=np.asarray(translation_log_priors, dtype=np.float32),
    )


def bucket_local_hypothesis_layout(
    layout: LocalHypothesisLayout,
    image_batch_size: int,
    rotation_block_size: int,
    *,
    max_hypotheses_per_microbatch: int = 32768,
) -> list[LocalBucketSpec]:
    """Bucket images by exact local-rotation count for static-shape execution."""

    image_batch_size = int(max(1, image_batch_size))
    max_hypotheses_per_microbatch = int(max(1, max_hypotheses_per_microbatch))
    bucket_sizes = np.asarray(
        [_exact_bucket_rotation_size(int(count), rotation_block_size) for count in layout.rotation_counts],
        dtype=np.int32,
    )
    processing_order = np.lexsort((layout.rotation_counts, bucket_sizes)).astype(np.int32)
    bucket_specs: list[LocalBucketSpec] = []

    if processing_order.size == 0:
        return bucket_specs

    unique_bucket_sizes = np.unique(bucket_sizes[processing_order])
    for bucket_size in unique_bucket_sizes:
        bucket_images = processing_order[bucket_sizes[processing_order] == bucket_size]
        max_images = max(1, min(image_batch_size, max_hypotheses_per_microbatch // int(bucket_size)))
        for start in range(0, bucket_images.shape[0], max_images):
            image_indices = np.asarray(bucket_images[start : start + max_images], dtype=np.int32)
            actual_counts = layout.rotation_counts[image_indices].astype(np.int32, copy=False)
            batch_size = int(image_indices.shape[0])
            padded_rotations = np.broadcast_to(
                np.eye(3, dtype=np.float32),
                (batch_size, int(bucket_size), 3, 3),
            ).copy()
            padded_rotation_ids = np.full((batch_size, int(bucket_size)), -1, dtype=np.int32)
            padded_log_prior = np.full((batch_size, int(bucket_size)), -1e30, dtype=np.float32)
            padded_mask = np.zeros((batch_size, int(bucket_size)), dtype=bool)

            for row, image_idx in enumerate(image_indices.tolist()):
                start_off = int(layout.rotation_offsets[image_idx])
                end_off = int(layout.rotation_offsets[image_idx + 1])
                count = end_off - start_off
                padded_rotations[row, :count] = layout.rotations_flat[start_off:end_off]
                padded_rotation_ids[row, :count] = layout.rotation_ids_flat[start_off:end_off]
                padded_log_prior[row, :count] = layout.rotation_log_priors_flat[start_off:end_off]
                padded_mask[row, :count] = True

            bucket_specs.append(
                LocalBucketSpec(
                    image_indices=image_indices,
                    bucket_rotation_count=int(bucket_size),
                    actual_rotation_counts=actual_counts,
                    local_rotation_ids=padded_rotation_ids,
                    local_rotations=padded_rotations,
                    local_rotation_log_prior=padded_log_prior,
                    local_rotation_mask=padded_mask,
                    translation_log_prior=np.asarray(layout.translation_log_priors[image_indices], dtype=np.float32),
                )
            )

    return bucket_specs
