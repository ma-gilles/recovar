"""Bounded RELION projection caches for consecutive local-search buckets.

The engine decides when to plan/build/release a cache. This module owns its
budget, stable grouping, rotation-ID mapping and projection materialization.
Only IDs selected by a bucket mask refer to initialized cache rows; unused
capacity retains the existing uninitialized padding contract.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.jax_runtime import block_until_ready as _block_until_ready
from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
)
from recovar.em.dense_single_volume.local_layout import LocalBucketSpec

# Keep the established log category for existing run collectors.
logger = logging.getLogger("recovar.em.dense_single_volume.local_em_engine")

EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB = 0.0
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB"
EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS = 64_000_000
EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV = (
    "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS"
)
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS = 64
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV = "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS"


@dataclass
class LocalRelionProjectionCache:
    projections: jnp.ndarray
    id_map: jnp.ndarray
    enabled: bool
    row_count: int = 0
    id_map_row_count: int = 0
    n_projection_pixels: int = 0
    estimated_gb: float = 0.0
    build_s: float = 0.0


@dataclass(frozen=True)
class ProjectionCachePlan:
    """Bucket order and capacity decisions, separate from live cache buffers."""

    buckets: list[LocalBucketSpec]
    groups: list[tuple[int, int, int]] = field(default_factory=list)
    capacity_rows: int = 0
    budget_gb: float = 0.0
    projection_pixels: int = 0
    requested_rows: int = 0

    def log_enabled(self, id_map_rows: int) -> None:
        logger.info(
            "Exact local RELION projection cache groups enabled: groups=%d capacity_rows=%d "
            "requested_rows=%d cap=%.2f GB projection_pixels=%d id_map_rows=%d",
            len(self.groups),
            self.capacity_rows,
            int(self.requested_rows),
            float(self.budget_gb),
            self.projection_pixels,
            id_map_rows,
        )


def plan_cache(
    bucket_specs: list[LocalBucketSpec],
    *,
    n_projection_pixels: int,
) -> ProjectionCachePlan:
    """Plan eligible local buckets, preserving sort order even if caching is rejected.

    The caller decides whether this policy applies. Planning never materializes
    projections or the layout ID map; their allocation stays with execution.
    """
    requested_rows, budget_gb = cache_capacity_rows(n_projection_pixels)
    if requested_rows > 0:
        bucket_specs = sort_buckets(bucket_specs)
    groups = plan_cache_groups(bucket_specs, cache_row_capacity=int(requested_rows))
    max_groups = max_cache_groups()
    if len(groups) > max_groups:
        logger.info(
            "Exact local RELION projection cache disabled: planned groups=%d exceeds max_groups=%d "
            "(capacity_rows=%d cap=%.2f GB projection_pixels=%d)",
            len(groups),
            max_groups,
            int(requested_rows),
            float(budget_gb),
            n_projection_pixels,
        )
        groups = []
    capacity_rows = 0
    projection_pixels = 0
    if groups:
        capacity_rows = int(max(row_count for _, _, row_count in groups))
        projection_pixels = n_projection_pixels
    return ProjectionCachePlan(
        buckets=bucket_specs,
        groups=groups,
        capacity_rows=capacity_rows,
        budget_gb=budget_gb,
        projection_pixels=projection_pixels,
        requested_rows=requested_rows,
    )


def read_nonnegative_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative float, got {raw!r}") from exc
    if value < 0.0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a non-negative finite float, got {raw!r}")
    return value


def _projection_chunk_rows(n_projection_pixels: int) -> int:
    target = int(
        os.environ.get(
            EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV,
            EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS,
        )
    )
    if target <= 0:
        raise ValueError(f"{EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV} must be positive")
    return max(1, int(target) // max(1, int(n_projection_pixels)))


def disabled_cache() -> LocalRelionProjectionCache:
    return LocalRelionProjectionCache(
        projections=jnp.zeros((1, 1), dtype=jnp.complex64),
        id_map=jnp.zeros((1,), dtype=jnp.int32),
        enabled=False,
    )


def cache_capacity_rows(n_projection_pixels: int) -> tuple[int, float]:
    max_gb = read_nonnegative_float_env(
        EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV,
        EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB,
    )
    if max_gb <= 0.0 or int(n_projection_pixels) <= 0:
        return 0, float(max_gb)
    bytes_per_row = int(n_projection_pixels) * np.dtype(np.complex64).itemsize
    return int((max_gb * 1e9) // max(1, bytes_per_row)), float(max_gb)


def _bucket_valid_rotation_ids(bucket: LocalBucketSpec) -> np.ndarray:
    ids = np.asarray(bucket.local_rotation_ids, dtype=np.int64)
    mask = np.asarray(bucket.local_rotation_mask, dtype=bool) & (ids >= 0)
    if not np.any(mask):
        return np.zeros(0, dtype=np.int64)
    return np.unique(ids[mask])


def _bucket_rotation_id_center(bucket: LocalBucketSpec) -> int:
    ids = _bucket_valid_rotation_ids(bucket)
    if ids.size == 0:
        return -1
    return int(np.median(ids))


def sort_buckets(bucket_specs: list[LocalBucketSpec]) -> list[LocalBucketSpec]:
    return sorted(
        bucket_specs,
        key=lambda bucket: (
            int(bucket.bucket_rotation_count),
            _bucket_rotation_id_center(bucket),
            int(bucket.image_indices[0]) if int(bucket.image_indices.shape[0]) else -1,
        ),
    )


def max_cache_groups() -> int:
    raw = os.environ.get(
        EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV,
        str(EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS),
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV} must be positive") from exc
    if value <= 0:
        raise ValueError(f"{EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV} must be positive")
    return value


def plan_cache_groups(
    bucket_specs: list[LocalBucketSpec],
    *,
    cache_row_capacity: int,
) -> list[tuple[int, int, int]]:
    """Greedily group consecutive buckets under a compact projection-cache row cap."""

    if cache_row_capacity <= 0 or not bucket_specs:
        return []

    groups: list[tuple[int, int, int]] = []
    start = 0
    active_ids: set[int] = set()
    for bucket_index, bucket in enumerate(bucket_specs):
        bucket_ids = set(int(x) for x in _bucket_valid_rotation_ids(bucket).tolist())
        if len(bucket_ids) > cache_row_capacity:
            logger.info(
                "Exact local RELION projection cache disabled: one bucket needs %d rows, cap is %d",
                len(bucket_ids),
                cache_row_capacity,
            )
            return []
        if active_ids and len(active_ids | bucket_ids) > cache_row_capacity:
            groups.append((start, bucket_index, len(active_ids)))
            start = bucket_index
            active_ids = set(bucket_ids)
        else:
            active_ids |= bucket_ids
    if active_ids or start < len(bucket_specs):
        groups.append((start, len(bucket_specs), len(active_ids)))
    return groups


def build_cache(
    bucket_specs: list[LocalBucketSpec],
    relion_projector_half,
    *,
    image_shape,
    n_projection_pixels: int,
    relion_projector_r_max: int,
    projection_padding_factor: int,
    projection_relion_texture_interp: bool | None,
    projection_pixel_indices,
    projection_relion_acc_double_floorf_quirk: bool = False,
    projector_output_size: int,
    cache_row_capacity: int,
    max_global_rotation_id: int,
    group_index: int,
    n_groups: int,
    projection_mask_current_image_disk: bool = True,
) -> LocalRelionProjectionCache:
    """Precompute compact RELION projections for one bounded bucket group."""

    if cache_row_capacity <= 0 or not bucket_specs:
        return disabled_cache()

    ids_parts = []
    rotation_parts = []
    for bucket in bucket_specs:
        ids = np.asarray(bucket.local_rotation_ids, dtype=np.int64)
        mask = np.asarray(bucket.local_rotation_mask, dtype=bool) & (ids >= 0)
        if not np.any(mask):
            continue
        ids_parts.append(ids[mask])
        rotation_parts.append(np.asarray(bucket.local_rotations, dtype=np.float32)[mask])
    if not ids_parts:
        return disabled_cache()

    valid_ids = np.concatenate(ids_parts, axis=0)
    valid_rotations = np.concatenate(rotation_parts, axis=0)
    unique_ids, first_positions = np.unique(valid_ids, return_index=True)
    row_count = int(unique_ids.size)
    if row_count > int(cache_row_capacity):
        raise RuntimeError(
            "internal projection-cache planner error: group has "
            f"{row_count} rows but capacity is {int(cache_row_capacity)}"
        )
    id_map_row_count = int(max(max_global_rotation_id + 1, int(np.max(valid_ids)) + 1))
    estimated_gb = float(cache_row_capacity * n_projection_pixels * np.dtype(np.complex64).itemsize / 1e9)

    cache_t0 = time.time()
    cache_rotations = valid_rotations[first_positions]
    id_map = np.zeros(id_map_row_count, dtype=np.int32)
    id_map[unique_ids] = np.arange(row_count, dtype=np.int32)

    chunk_rows = _projection_chunk_rows(n_projection_pixels)
    host_cache = np.empty((cache_row_capacity, n_projection_pixels), dtype=np.complex64)
    logger.info(
        "Exact local RELION projection cache group %d/%d build: rows=%d capacity=%d "
        "id_map_rows=%d projection_pixels=%d estimated=%.2f GB chunk_rows=%d buckets=%d",
        int(group_index) + 1,
        int(n_groups),
        row_count,
        int(cache_row_capacity),
        id_map_row_count,
        n_projection_pixels,
        estimated_gb,
        int(chunk_rows),
        len(bucket_specs),
    )
    for start in range(0, row_count, chunk_rows):
        stop = min(row_count, start + chunk_rows)
        disk_kwargs = {}
        if not projection_mask_current_image_disk:
            disk_kwargs["mask_current_image_disk"] = False
        proj_chunk, _ = _compute_relion_projector_projections_block(
            relion_projector_half,
            jnp.asarray(cache_rotations[start:stop], dtype=jnp.float32),
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            relion_texture_interp=projection_relion_texture_interp,
            relion_acc_double_floorf_quirk=projection_relion_acc_double_floorf_quirk,
            projector_output_size=int(projector_output_size) if int(projector_output_size) > 0 else None,
            pixel_indices=projection_pixel_indices,
            **disk_kwargs,
        )
        _block_until_ready(proj_chunk)
        host_cache[start:stop] = np.asarray(proj_chunk, dtype=np.complex64)
        del proj_chunk

    projections = jnp.asarray(host_cache)
    id_map_jnp = jnp.asarray(id_map, dtype=jnp.int32)
    _block_until_ready(projections, id_map_jnp)
    build_s = time.time() - cache_t0
    logger.info(
        "Exact local RELION projection cache group %d/%d ready: rows=%d capacity=%d "
        "id_map_rows=%d projection_pixels=%d estimated=%.2f GB build=%.1fs",
        int(group_index) + 1,
        int(n_groups),
        row_count,
        int(cache_row_capacity),
        id_map_row_count,
        n_projection_pixels,
        estimated_gb,
        build_s,
    )
    return LocalRelionProjectionCache(
        projections=projections,
        id_map=id_map_jnp,
        enabled=True,
        row_count=row_count,
        id_map_row_count=id_map_row_count,
        n_projection_pixels=n_projection_pixels,
        estimated_gb=estimated_gb,
        build_s=build_s,
    )
