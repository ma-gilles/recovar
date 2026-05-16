"""Host caches for the exact local EM engine.

The local engine visits every image exactly once but bucket sorting turns
that into many small indexed dataset reads. These helpers precompute raw
image / CTF batches and per-image half spectra once so the per-bucket loop
only resorts cached arrays. Extracted from ``local_em_engine.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.dense_single_volume.helpers.image_shifts import apply_relion_integer_pre_shifts
from recovar.em.dense_single_volume.helpers.preprocessing import process_half_image

# Mirror local_em_engine's environment-tunable caps. Re-exporting the
# constants from local_em_engine keeps test imports stable.
EXACT_LOCAL_RAW_CACHE_MAX_GB = 16.0
EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_RAW_CACHE_MAX_GB"

EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB = 0.0
EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB"

EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB = 12.0
EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB"


@dataclass(frozen=True)
class _LocalProcessedHalfCache:
    ctf_params: np.ndarray
    score_half: np.ndarray
    recon_half: np.ndarray | None
    integer_pre_shifts_applied: bool


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


def _sparse_big_jit_mstep_tensors_within_memory(
    *,
    image_count: int,
    rotation_count: int,
    n_recon_windowed: int,
    use_float64_scoring: bool,
) -> bool:
    summed_bytes = 16 if use_float64_scoring else 8
    ctf_bytes = 8 if use_float64_scoring else 4
    # Keep margin for XLA output buffers and the following packed tensors.
    estimated_gb = int(image_count) * int(rotation_count) * int(n_recon_windowed) * (summed_bytes + ctf_bytes) / 1e9
    max_gb = float(
        os.environ.get(
            EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV,
            EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB,
        )
    )
    return max_gb > 0.0 and estimated_gb <= max_gb


def _validate_native_half_batch(batch, image_shape):
    batch_np = np.asarray(batch)
    if batch_np.ndim != 3 or tuple(batch_np.shape[-2:]) != tuple(image_shape):
        raise ValueError(
            "Exact local big-JIT requires raw real-space image batches with shape "
            f"(B, {int(image_shape[0])}, {int(image_shape[1])}); got {batch_np.shape}",
        )
    if np.iscomplexobj(batch_np):
        raise ValueError("Exact local big-JIT does not support pre-Fourier complex image batches")


def _build_local_raw_cache(experiment_dataset, n_images: int):
    """Fetch all local images/CTF rows once for exact local search.

    The local engine visits every image exactly once, but bucket sorting turns
    that into many small indexed dataset reads. Caching raw images preserves the
    per-bucket preprocessing behavior while avoiding repeated source lookups.
    """

    indices = np.arange(int(n_images), dtype=np.int32)
    batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, indices)
    fetched_indices = np.asarray(fetched_indices, dtype=np.int32)
    batch_np = np.asarray(batch_data)
    ctf_np = np.asarray(ctf_params)
    if np.array_equal(fetched_indices, indices):
        return batch_np, ctf_np

    batch_cache = np.empty((int(n_images),) + tuple(batch_np.shape[1:]), dtype=batch_np.dtype)
    ctf_cache = np.empty((int(n_images),) + tuple(ctf_np.shape[1:]), dtype=ctf_np.dtype)
    batch_cache[fetched_indices] = batch_np
    ctf_cache[fetched_indices] = ctf_np
    return batch_cache, ctf_cache


def _all_integer_pre_shifts_or_none(image_pre_shifts, n_images: int):
    if image_pre_shifts is None:
        return None
    shifts = np.asarray(image_pre_shifts, dtype=np.float32).reshape(int(n_images), 2)
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
