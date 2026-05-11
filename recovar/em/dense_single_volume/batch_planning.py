"""GPU-aware batch sizing + raw-image host cache for the EM iteration loop.

``_estimate_relion_em_batch_sizes`` chooses microbatch sizes from pose-grid,
image, class, and GPU size so the dense RELION loop's transient memory
drivers (score tensor + projection tile + translation-expanded half-images)
stay within available memory. ``_maybe_cache_raw_image_loaders`` keeps
file-backed raw particles in host memory across passes.

Extracted from ``iteration_loop.py`` so the master loop stays focused on
EM dispatch.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# Constants mirror those in iteration_loop.py so monkeypatches at either
# module level continue to bind correctly.
RELION_SCORE_TENSOR_FLOAT_BUDGET = 200_000_000
_RELION_EM_BATCH_DEFAULT_GPU_GB = 80.0
_RELION_EM_BATCH_USABLE_FRACTION = 0.65
_RELION_EM_BATCH_PROJECTION_FRACTION = 0.20
_RELION_EM_BATCH_SCORE_FRACTION = 0.20
_RELION_EM_BATCH_MAX_PROJECTION_GB = 10.0
_RELION_EM_BATCH_MIN_PROJECTION_GB = 0.5
_RELION_EM_BATCH_PROJECTION_LIVE_FACTOR = 1.5
_RELION_EM_BATCH_TRANSLATION_TILE_FRACTION = 0.35
_RELION_EM_BATCH_RUNTIME_TRANSLATION_TILE_FRACTION = 0.17
_RELION_EM_BATCH_MAX_TRANSLATION_TILE_GB = 14.0
_RELION_EM_BATCH_MIN_TRANSLATION_TILE_GB = 0.5
_RELION_EM_BATCH_RUNTIME_FREE_FRACTION = 0.80

_EM_RAW_IMAGE_CACHE_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE"
_EM_RAW_IMAGE_CACHE_MAX_GB_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB"
_EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB = 16.0


@dataclass(frozen=True)
class _RelionEMBatchPlan:
    image_batch_size: int
    rotation_block_size: int
    score_float_budget: int
    projection_budget_gb: float
    translation_tile_budget_gb: float
    persistent_estimate_gb: float
    usable_estimate_gb: float
    gpu_used_estimate_gb: float
    runtime_free_estimate_gb: float
    projection_block_gb: float
    translation_tile_gb: float


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _estimate_relion_em_batch_sizes(
    *,
    requested_image_batch_size: int,
    requested_rotation_block_size: int,
    n_rot: int,
    n_trans: int,
    image_shape,
    volume_shape,
    padding_factor: int,
    n_classes: int = 1,
    gpu_memory_gb: float | None = None,
) -> _RelionEMBatchPlan:
    """Choose EM microbatch sizes from pose-grid, image, class, and GPU size."""
    # Indirection through iteration_loop module so test monkeypatches on
    # ``iteration_loop.utils.get_gpu_memory_total`` / ``get_gpu_memory_used``
    # win at this call site too.
    from recovar.em.dense_single_volume import iteration_loop as _il

    requested_image_batch_size = max(1, _safe_int(requested_image_batch_size, 1))
    requested_rotation_block_size = max(1, _safe_int(requested_rotation_block_size, 1))
    n_rot = max(1, _safe_int(n_rot, 1))
    n_trans = max(1, _safe_int(n_trans, 1))
    n_classes = max(1, _safe_int(n_classes, 1))
    padding_factor = max(1, _safe_int(padding_factor, 1))
    image_shape = tuple(int(s) for s in image_shape)
    volume_shape = tuple(int(s) for s in volume_shape)

    gpu_used_gb = 0.0
    if gpu_memory_gb is None:
        try:
            gpu_memory_gb = float(_il.utils.get_gpu_memory_total())
        except Exception:
            gpu_memory_gb = _RELION_EM_BATCH_DEFAULT_GPU_GB
        try:
            gpu_used_gb = float(_il.utils.get_gpu_memory_used())
        except Exception:
            gpu_used_gb = 0.0
    if not np.isfinite(gpu_memory_gb) or gpu_memory_gb <= 0:
        gpu_memory_gb = _RELION_EM_BATCH_DEFAULT_GPU_GB
    if not np.isfinite(gpu_used_gb) or gpu_used_gb < 0:
        gpu_used_gb = 0.0
    gpu_used_gb = min(gpu_used_gb, max(0.0, gpu_memory_gb - 1.0))

    padded_volume_voxels = float(np.prod([d * padding_factor for d in volume_shape]))
    native_volume_voxels = float(np.prod(volume_shape))
    persistent_bytes = (
        2.0 * padded_volume_voxels * np.dtype(np.complex64).itemsize * n_classes
        + 4.0 * native_volume_voxels * np.dtype(np.complex64).itemsize * n_classes
    )
    persistent_gb = persistent_bytes / 1e9
    runtime_free_gb = max(1.0, gpu_memory_gb - gpu_used_gb)
    usable_from_total_gb = max(1.0, gpu_memory_gb * _RELION_EM_BATCH_USABLE_FRACTION - persistent_gb)
    usable_from_runtime_gb = max(1.0, runtime_free_gb * _RELION_EM_BATCH_RUNTIME_FREE_FRACTION)
    usable_gb = min(usable_from_total_gb, usable_from_runtime_gb)

    score_float_budget = int(
        max(
            1_000_000,
            min(
                RELION_SCORE_TENSOR_FLOAT_BUDGET,
                usable_gb * _RELION_EM_BATCH_SCORE_FRACTION * 1e9 / np.dtype(np.float32).itemsize,
            ),
        )
    )
    projection_budget_gb = max(
        _RELION_EM_BATCH_MIN_PROJECTION_GB,
        min(_RELION_EM_BATCH_MAX_PROJECTION_GB, usable_gb * _RELION_EM_BATCH_PROJECTION_FRACTION),
    )
    translation_tile_budget_gb = max(
        _RELION_EM_BATCH_MIN_TRANSLATION_TILE_GB,
        min(
            _RELION_EM_BATCH_MAX_TRANSLATION_TILE_GB,
            usable_gb * _RELION_EM_BATCH_TRANSLATION_TILE_FRACTION,
            usable_from_runtime_gb * _RELION_EM_BATCH_RUNTIME_TRANSLATION_TILE_FRACTION,
        ),
    )

    score_image_cap = max(1, score_float_budget // max(n_rot * n_trans * n_classes, 1))
    full_half_pixels = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    translation_bytes_per_image = max(
        1,
        2 * n_trans * full_half_pixels * np.dtype(np.complex64).itemsize * n_classes,
    )
    translation_image_cap = max(1, int(translation_tile_budget_gb * 1e9 // translation_bytes_per_image))
    image_batch = min(requested_image_batch_size, score_image_cap, translation_image_cap)

    score_rotation_cap = max(1, score_float_budget // max(image_batch * n_trans * n_classes, 1))
    projection_bytes_per_rotation = max(
        1,
        int(
            np.ceil(
                full_half_pixels
                * np.dtype(np.complex64).itemsize
                * n_classes
                * _RELION_EM_BATCH_PROJECTION_LIVE_FACTOR,
            )
        ),
    )
    projection_rotation_cap = max(1, int(projection_budget_gb * 1e9 // projection_bytes_per_rotation))
    rotation_cap = min(score_rotation_cap, projection_rotation_cap)

    if requested_rotation_block_size >= 64:
        rotation_block = max(64, min(requested_rotation_block_size, rotation_cap))
    else:
        rotation_block = min(requested_rotation_block_size, rotation_cap)
    rotation_block = max(1, min(rotation_block, n_rot))

    projection_block_gb = rotation_block * projection_bytes_per_rotation / 1e9
    translation_tile_gb = image_batch * translation_bytes_per_image / 1e9
    return _RelionEMBatchPlan(
        image_batch_size=int(image_batch),
        rotation_block_size=int(rotation_block),
        score_float_budget=int(score_float_budget),
        projection_budget_gb=float(projection_budget_gb),
        translation_tile_budget_gb=float(translation_tile_budget_gb),
        persistent_estimate_gb=float(persistent_gb),
        usable_estimate_gb=float(usable_gb),
        gpu_used_estimate_gb=float(gpu_used_gb),
        runtime_free_estimate_gb=float(runtime_free_gb),
        projection_block_gb=float(projection_block_gb),
        translation_tile_gb=float(translation_tile_gb),
    )


def _image_backend(ds):
    return getattr(getattr(ds, "image_source", None), "backend", None)


def _dataset_raw_image_loader(ds):
    backend = _image_backend(ds)
    loader = getattr(backend, "source", None)
    if loader is None or not hasattr(loader, "load_all"):
        return None
    return loader


def _estimate_raw_image_cache_bytes(loader) -> int:
    n_images = int(getattr(loader, "num_images", getattr(loader, "n", 0)))
    image_size = int(getattr(loader, "image_size", getattr(loader, "D", 0)))
    dtype = np.dtype(getattr(loader, "_dtype", np.float32))
    return int(n_images * image_size * image_size * dtype.itemsize)


def _em_raw_image_cache_mode() -> str:
    return os.environ.get(_EM_RAW_IMAGE_CACHE_ENV, "auto").strip().lower()


def _maybe_cache_raw_image_loaders(experiment_datasets) -> None:
    """Keep file-backed raw particles in host memory across RELION EM passes."""
    mode = _em_raw_image_cache_mode()
    if mode in {"0", "false", "no", "off", "disable", "disabled"}:
        logger.info("RELION mode raw image cache disabled by %s=%s", _EM_RAW_IMAGE_CACHE_ENV, mode)
        return
    force = mode in {"1", "true", "yes", "on", "force", "always"}

    planned = []
    seen = set()
    total_bytes = 0
    for ds in experiment_datasets:
        loader = _dataset_raw_image_loader(ds)
        if loader is None:
            continue
        loader_id = id(loader)
        if loader_id in seen:
            continue
        seen.add(loader_id)
        if getattr(loader, "_cached", None) is not None:
            continue
        estimated_bytes = _estimate_raw_image_cache_bytes(loader)
        if estimated_bytes <= 0:
            continue
        planned.append((loader, estimated_bytes))
        total_bytes += estimated_bytes

    if not planned:
        return

    max_gb = float(os.environ.get(_EM_RAW_IMAGE_CACHE_MAX_GB_ENV, _EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB))
    max_bytes = int(max_gb * (1024**3))
    if not force and total_bytes > max_bytes:
        logger.info(
            "RELION mode raw image cache skipped: estimated %.2f GiB exceeds %.2f GiB; "
            "set %s=force or increase %s to override",
            total_bytes / (1024**3),
            max_gb,
            _EM_RAW_IMAGE_CACHE_ENV,
            _EM_RAW_IMAGE_CACHE_MAX_GB_ENV,
        )
        return

    cache_t0 = time.time()
    for loader, estimated_bytes in planned:
        loader_t0 = time.time()
        loader.load_all()
        logger.info(
            "RELION mode raw image cache loaded %.2f GiB for %s in %.1fs",
            estimated_bytes / (1024**3),
            type(loader).__name__,
            time.time() - loader_t0,
        )
    logger.info(
        "RELION mode raw image cache ready: %.2f GiB across %d loader(s) in %.1fs",
        total_bytes / (1024**3),
        len(planned),
        time.time() - cache_t0,
    )
