"""Per-image local hypothesis layout and bucketization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recovar.em.dense_single_volume.helpers.local_search import _local_search_engine_rotation_block_size
from recovar.em.dense_single_volume.helpers.orientation_priors import make_relion_translation_log_prior
from recovar.em.sampling import get_local_rotation_grid_fast


def _exact_bucket_rotation_size(local_rotation_count: int, rotation_block_size: int) -> int:
    """Return a compile-friendly padded size for one exact local neighborhood.

    Unlike the grouped-union path, the exact local engine cannot safely cap the
    bucket size below the true per-image neighborhood cardinality. Use the same
    power-of-two style padding for smaller neighborhoods, but keep larger exact
    neighborhoods intact.
    """
    local_rotation_count = int(local_rotation_count)
    if local_rotation_count <= 0:
        return 1
    engine_cap = int(_local_search_engine_rotation_block_size(rotation_block_size))
    if local_rotation_count <= engine_cap:
        bucket = 1 << max(int(local_rotation_count - 1).bit_length(), 4)
        return int(max(local_rotation_count, min(bucket, engine_cap)))
    return int(local_rotation_count)


@dataclass(frozen=True)
class LocalHypothesisLayout:
    """Flat per-image local hypothesis storage."""

    n_global_rotations: int
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
) -> LocalHypothesisLayout:
    """Build exact per-image local neighborhoods and translation priors."""

    prior_rotations = np.asarray(prior_rotations, dtype=np.float32)
    rotation_grid_rotations = np.asarray(rotation_grid_rotations, dtype=np.float32).reshape(-1, 3, 3)
    translations = np.asarray(translations, dtype=np.float32)
    prior_translations = np.asarray(prior_translations, dtype=np.float32).reshape(-1, translations.shape[1])

    n_images = int(prior_rotations.shape[0])
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    counts = np.zeros(n_images, dtype=np.int32)
    rotation_ids_parts: list[np.ndarray] = []
    rotations_parts: list[np.ndarray] = []
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
        rotations_parts.append(rotation_grid_rotations[local_ids])
        log_prior_parts.append(local_log_prior)

    translation_log_priors = make_relion_translation_log_prior(
        translations,
        voxel_size,
        sigma_offset_angstrom,
        prior_translations,
        offset_range_pixels=offset_range_pixels,
    ).astype(np.float32, copy=False)

    return LocalHypothesisLayout(
        n_global_rotations=int(rotation_grid_rotations.shape[0]),
        rotation_offsets=offsets,
        rotation_ids_flat=np.concatenate(rotation_ids_parts, axis=0) if rotation_ids_parts else np.zeros(0, dtype=np.int32),
        rotations_flat=np.concatenate(rotations_parts, axis=0) if rotations_parts else np.zeros((0, 3, 3), dtype=np.float32),
        rotation_log_priors_flat=np.concatenate(log_prior_parts, axis=0) if log_prior_parts else np.zeros(0, dtype=np.float32),
        rotation_counts=counts,
        translation_grid=translations,
        translation_log_priors=np.asarray(translation_log_priors, dtype=np.float32),
    )


def bucket_local_hypothesis_layout(
    layout: LocalHypothesisLayout,
    image_batch_size: int,
    rotation_block_size: int,
    *,
    max_hypotheses_per_microbatch: int = 16384,
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
