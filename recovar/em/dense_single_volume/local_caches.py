"""Host caches for the exact local EM engine.

The local engine visits every image exactly once but bucket sorting turns
that into many small indexed dataset reads. These helpers precompute raw
image / CTF batches and per-image half spectra once so the per-bucket loop
only resorts cached arrays. Extracted from ``local_em_engine.py``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

import numpy as np

from recovar.em.dense_single_volume.batch_planning import (
    _fixed_capacity_plan_descriptor_fingerprint,
    _FixedCapacityLocalGenerationToken,
    _FixedCapacityPhysicalOrder,
    _FixedCapacityWholeLocalPlan,
)
from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.dense_single_volume.helpers.image_shifts import apply_relion_integer_pre_shifts
from recovar.em.dense_single_volume.helpers.preprocessing import process_half_image

# Cache and sparse-M-step allocation limits are owned here. Engine callers
# and tests import this module rather than maintaining copies of the caps.
EXACT_LOCAL_RAW_CACHE_MAX_GB = 16.0
EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_RAW_CACHE_MAX_GB"

EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB = 0.0
EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB"

EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB = 12.0
EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB"
EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_DEVICE_FRACTION = 0.15


@lru_cache(maxsize=1)
def _default_sparse_big_jit_mstep_max_gb() -> float:
    """Scale the packed-M-step cap to the smallest visible GPU.

    The tensor estimator covers returned M-step arrays, not the scorer's
    resident inputs or XLA's output-allocation margin.  A fixed 12 GB cap is
    appropriate on 80 GB accelerators but can request a 16+ GB allocation
    after the rest of a 40 GB device is resident.  Keep the historical 12 GB
    ceiling while limiting these outputs to 15% of device memory.
    """

    try:
        import jax

        limits = []
        for device in jax.local_devices():
            if device.platform not in {"gpu", "cuda"}:
                continue
            stats = device.memory_stats() or {}
            if int(stats.get("bytes_limit", 0)) > 0:
                limits.append(int(stats["bytes_limit"]))
        if limits:
            device_cap_gb = min(limits) * EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_DEVICE_FRACTION / 1e9
            return min(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB, device_cap_gb)
    except Exception:
        # CPU-only imports/tests and older JAX backends have no memory stats.
        pass
    return EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB


@dataclass(frozen=True)
class _LocalProcessedHalfCache:
    ctf_params: np.ndarray
    score_half: np.ndarray
    recon_half: np.ndarray | None
    integer_pre_shifts_applied: bool


@dataclass(frozen=True)
class _FixedCapacityLocalOperands:
    """Cache-once host operands in one sealed fixed-capacity physical order."""

    physical_image_capacity: int
    valid_image_count: int
    image_indices: np.ndarray
    valid_image_mask: np.ndarray
    raw_images: np.ndarray
    ctf_params: np.ndarray
    metadata_by_name: Mapping[str, np.ndarray]
    physical_position_by_image_id: Mapping[int, int]
    plan_fingerprint: str
    plan_generation_token: _FixedCapacityLocalGenerationToken


def _local_raw_cache_enabled(n_images: int, image_shape, dtype) -> bool:
    bytes_per_pixel = np.dtype(dtype).itemsize if dtype is not None else np.dtype(np.float32).itemsize
    estimated_gb = int(n_images) * int(np.prod(image_shape)) * bytes_per_pixel / 1e9
    max_gb = float(os.environ.get(EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV, EXACT_LOCAL_RAW_CACHE_MAX_GB))
    return estimated_gb <= max_gb


def _local_processed_half_cache_enabled(n_images: int, n_half: int, dtype, *, store_recon_half: bool) -> bool:
    bytes_per_value = np.dtype(dtype).itemsize
    n_arrays = 2 if store_recon_half else 1
    estimated_gb = int(n_images) * int(n_half) * bytes_per_value * n_arrays / 1e9
    max_gb = float(
        os.environ.get(
            EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV,
            EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB,
        )
    )
    return estimated_gb <= max_gb


def _sparse_big_jit_mstep_tensors_memory_gb(
    *,
    image_count: int,
    rotation_count: int,
    n_recon_windowed: int,
    use_float64_scoring: bool,
) -> tuple[float, float]:
    summed_bytes = 16 if use_float64_scoring else 8
    ctf_bytes = 8 if use_float64_scoring else 4
    # Keep margin for XLA output buffers and the following packed tensors.
    estimated_gb = int(image_count) * int(rotation_count) * int(n_recon_windowed) * (summed_bytes + ctf_bytes) / 1e9
    configured_max_gb = os.environ.get(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV)
    max_gb = (
        float(configured_max_gb)
        if configured_max_gb is not None
        else _default_sparse_big_jit_mstep_max_gb()
    )
    return estimated_gb, max_gb


def _validate_native_half_batch(batch, image_shape):
    batch_np = np.asarray(batch)
    if batch_np.ndim != 3 or tuple(batch_np.shape[-2:]) != tuple(image_shape):
        raise ValueError(
            "Exact local big-JIT requires raw real-space image batches with shape "
            f"(B, {int(image_shape[0])}, {int(image_shape[1])}); got {batch_np.shape}",
        )
    if np.iscomplexobj(batch_np):
        raise ValueError("Exact local big-JIT does not support pre-Fourier complex image batches")


def _fetch_local_raw_rows_once(experiment_dataset, image_indices):
    batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, image_indices)
    return np.asarray(batch_data), np.asarray(ctf_params), np.asarray(fetched_indices)


def _build_local_raw_cache(experiment_dataset, n_images: int):
    """Fetch all local images/CTF rows once for exact local search.

    The local engine visits every image exactly once, but bucket sorting turns
    that into many small indexed dataset reads. Caching raw images preserves the
    per-bucket preprocessing behavior while avoiding repeated source lookups.
    """

    indices = np.arange(int(n_images), dtype=np.int32)
    batch_np, ctf_np, fetched_indices = _fetch_local_raw_rows_once(experiment_dataset, indices)
    fetched_indices = np.asarray(fetched_indices, dtype=np.int32)
    if np.array_equal(fetched_indices, indices):
        return batch_np, ctf_np

    batch_cache = np.empty((int(n_images),) + tuple(batch_np.shape[1:]), dtype=batch_np.dtype)
    ctf_cache = np.empty((int(n_images),) + tuple(ctf_np.shape[1:]), dtype=ctf_np.dtype)
    batch_cache[fetched_indices] = batch_np
    ctf_cache[fetched_indices] = ctf_np
    return batch_cache, ctf_cache


def _assemble_fixed_capacity_local_operands_once(
    experiment_dataset,
    plan: _FixedCapacityWholeLocalPlan,
    expected_order: _FixedCapacityPhysicalOrder,
    *,
    metadata_by_image: Mapping[str, np.ndarray] | None = None,
    tail_fill_value=0,
    enabled: bool = False,
) -> _FixedCapacityLocalOperands | None:
    """Fetch and snapshot raw/CTF/metadata rows in a sealed physical order."""

    if not enabled:
        return None
    if not isinstance(expected_order, _FixedCapacityPhysicalOrder):
        raise ValueError("fixed-capacity operand assembly requires an independently sealed physical order")
    current_fingerprint = _fixed_capacity_plan_descriptor_fingerprint(plan)
    if current_fingerprint != plan.descriptor_fingerprint:
        raise ValueError("fixed-capacity operand plan descriptors changed after sealing")

    requested_indices = expected_order.image_indices
    valid_image_count = int(plan.valid_image_count)
    physical_image_capacity = int(plan.physical_image_capacity)
    if valid_image_count <= 0 or valid_image_count > physical_image_capacity:
        raise ValueError("fixed-capacity operand plan has an invalid real image count")
    if valid_image_count != requested_indices.size:
        raise ValueError(
            "fixed-capacity operand plan image count does not match the sealed physical order",
        )
    if plan.image_indices.shape != (physical_image_capacity,):
        raise ValueError("fixed-capacity operand plan image axis has an invalid shape")
    if not np.array_equal(plan.image_indices[:valid_image_count], requested_indices):
        raise ValueError("fixed-capacity operand plan chronology does not match the sealed physical order")
    if np.any(plan.image_indices[valid_image_count:] != -1):
        raise ValueError("fixed-capacity operand plan inactive image IDs must be -1")

    raw_rows, ctf_rows, fetched_indices = _fetch_local_raw_rows_once(
        experiment_dataset,
        requested_indices,
    )
    if fetched_indices.ndim != 1 or not np.issubdtype(fetched_indices.dtype, np.integer):
        raise ValueError("fixed-capacity operand fetch returned invalid image IDs")
    fetched_indices = fetched_indices.astype(np.int64, copy=False)
    if np.unique(fetched_indices).size != fetched_indices.size:
        raise ValueError("fixed-capacity operand fetch returned duplicate image IDs")
    if fetched_indices.shape != requested_indices.shape or not np.array_equal(
        np.sort(fetched_indices),
        np.sort(requested_indices),
    ):
        raise ValueError("fixed-capacity operand fetch has missing or unexpected image IDs")
    if not np.array_equal(fetched_indices, requested_indices):
        raise ValueError("fixed-capacity operand fetch returned image IDs out of order")
    if raw_rows.ndim < 1 or raw_rows.shape[0] != valid_image_count:
        raise ValueError("fixed-capacity raw image rows do not match the sealed physical order")
    if ctf_rows.ndim < 1 or ctf_rows.shape[0] != valid_image_count:
        raise ValueError("fixed-capacity CTF rows do not match the sealed physical order")

    raw_images = np.full(
        (physical_image_capacity,) + raw_rows.shape[1:],
        tail_fill_value,
        dtype=raw_rows.dtype,
    )
    ctf_params = np.full(
        (physical_image_capacity,) + ctf_rows.shape[1:],
        tail_fill_value,
        dtype=ctf_rows.dtype,
    )
    raw_images[:valid_image_count] = raw_rows
    ctf_params[:valid_image_count] = ctf_rows

    if metadata_by_image is None:
        metadata_items = ()
    elif isinstance(metadata_by_image, Mapping):
        metadata_items = metadata_by_image.items()
    else:
        raise ValueError("fixed-capacity metadata must be a mapping from names to image-indexed arrays")
    packed_metadata = {}
    max_image_id = int(np.max(requested_indices))
    for name, values in metadata_items:
        if not isinstance(name, str) or not name:
            raise ValueError("fixed-capacity metadata names must be non-empty strings")
        source = np.asarray(values)
        if source.ndim < 1 or source.shape[0] <= max_image_id:
            raise ValueError(f"fixed-capacity metadata {name!r} does not cover every sealed image ID")
        selected = source[requested_indices]
        packed = np.full(
            (physical_image_capacity,) + selected.shape[1:],
            tail_fill_value,
            dtype=selected.dtype,
        )
        packed[:valid_image_count] = selected
        packed.setflags(write=False)
        packed_metadata[name] = packed

    image_indices = np.asarray(plan.image_indices, dtype=np.int64).copy()
    valid_image_mask = np.arange(physical_image_capacity, dtype=np.int64) < valid_image_count
    for values in (image_indices, valid_image_mask, raw_images, ctf_params):
        values.setflags(write=False)
    physical_positions = MappingProxyType(
        {int(image_id): position for position, image_id in enumerate(requested_indices.tolist())}
    )
    return _FixedCapacityLocalOperands(
        physical_image_capacity=physical_image_capacity,
        valid_image_count=valid_image_count,
        image_indices=image_indices,
        valid_image_mask=valid_image_mask,
        raw_images=raw_images,
        ctf_params=ctf_params,
        metadata_by_name=MappingProxyType(packed_metadata),
        physical_position_by_image_id=physical_positions,
        plan_fingerprint=current_fingerprint,
        plan_generation_token=plan.generation_token,
    )


def _all_integer_pre_shifts_or_none(image_pre_shifts, n_images: int):
    if image_pre_shifts is None:
        return None
    shifts = np.asarray(image_pre_shifts).reshape(int(n_images), 2)
    rounded = np.rint(shifts)
    if not np.allclose(shifts, rounded, rtol=0.0, atol=1e-6):
        return None
    return rounded.astype(np.int32)


def _build_local_processed_half_cache(
    experiment_dataset,
    n_images: int,
    *,
    score_with_masked_images: bool,
    image_pre_shifts=None,
    batch_size: int = 1024,
) -> _LocalProcessedHalfCache:
    """Precompute per-image half spectra for explicit exact-local buckets.

    This preserves the existing image preprocessing function and only changes
    scheduling: one large pass over images instead of thousands of one-image
    FFT/mask calls from local-search bucketization.
    """

    integer_pre_shifts = _all_integer_pre_shifts_or_none(image_pre_shifts, n_images)
    apply_integer_pre_shifts_once = image_pre_shifts is not None and integer_pre_shifts is not None
    if image_pre_shifts is not None and integer_pre_shifts is None:
        raise ValueError("processed half-image cache requires all pre-shifts to be integral")
    score_parts = []
    recon_parts = [] if score_with_masked_images else None
    ctf_parts = []
    indices = np.arange(int(n_images), dtype=np.int32)
    for start in range(0, int(n_images), int(batch_size)):
        chunk_indices = indices[start : start + int(batch_size)]
        batch, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, chunk_indices)
        fetched_indices = np.asarray(fetched_indices, dtype=np.int32)
        if not np.array_equal(fetched_indices, chunk_indices):
            raise RuntimeError("processed half-image cache requires dataset fetches in requested order")
        batch_np = np.asarray(batch)
        ctf_parts.append(np.asarray(ctf_params))
        if apply_integer_pre_shifts_once:
            batch_np = apply_relion_integer_pre_shifts(
                batch_np,
                integer_pre_shifts[chunk_indices],
            )
        score_parts.append(
            np.asarray(
                process_half_image(
                    experiment_dataset,
                    batch_np,
                    score_with_masked_images,
                )
            )
        )
        if recon_parts is not None:
            recon_parts.append(
                np.asarray(
                    process_half_image(
                        experiment_dataset,
                        batch_np,
                        False,
                    )
                )
            )
    score_half = np.concatenate(score_parts, axis=0)
    recon_half = None if recon_parts is None else np.concatenate(recon_parts, axis=0)
    ctf_cache = np.concatenate(ctf_parts, axis=0)
    return _LocalProcessedHalfCache(
        ctf_params=ctf_cache,
        score_half=score_half,
        recon_half=recon_half,
        integer_pre_shifts_applied=apply_integer_pre_shifts_once,
    )
