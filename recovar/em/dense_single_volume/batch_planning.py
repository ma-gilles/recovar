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


# Memory budgets and cache defaults used by this planner.
RELION_SCORE_TENSOR_FLOAT_BUDGET = 200_000_000
_RELION_EM_BATCH_DEFAULT_GPU_GB = 80.0
_RELION_EM_BATCH_USABLE_FRACTION = 0.65
_RELION_EM_BATCH_PROJECTION_FRACTION = 0.20
_RELION_EM_BATCH_SCORE_FRACTION = 0.20
_RELION_EM_BATCH_MAX_PROJECTION_GB = 10.0
_RELION_EM_BATCH_MIN_PROJECTION_GB = 0.5
_RELION_EM_BATCH_PROJECTION_LIVE_FACTOR = 1.5
_RELION_EM_BATCH_SCORE_MATMUL_LIVE_FACTOR = 7.0
_RELION_EM_BATCH_POSE_PIXEL_LIVE_FACTOR = 1.25
_RELION_EM_BATCH_POSE_PIXEL_FRACTION = 0.035
_RELION_EM_BATCH_POSE_PIXEL_WINDOW_FRACTION = 0.30
_RELION_EM_BATCH_ACTIVE_SCORE_TILE_LIVE_FACTOR = 4.0
_RELION_EM_BATCH_ACTIVE_SCORE_TILE_FRACTION = 0.10
_RELION_EM_BATCH_TRANSLATION_TILE_FRACTION = 0.35
_RELION_EM_BATCH_RUNTIME_TRANSLATION_TILE_FRACTION = 0.17
_RELION_EM_BATCH_MAX_TRANSLATION_TILE_GB = 14.0
_RELION_EM_BATCH_MIN_TRANSLATION_TILE_GB = 0.5
_RELION_EM_BATCH_RUNTIME_FREE_FRACTION = 0.80
_RELION_EM_BATCH_PROJECTION_FRACTION_ENV = "RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION"

_EM_RAW_IMAGE_CACHE_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE"
_EM_RAW_IMAGE_CACHE_MAX_GB_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB"
_EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB = 16.0


# Dense reconstruction-tile and class-hypothesis limits.
RELION_FIRSTITER_RECON_COMPLEX_BUDGET = 268_435_456
RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV = "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET"
RELION_DENSE_K_CLASS_HYPOTHESES_BUDGET = 2_000_000


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
    active_score_tile_budget_gb: float
    active_score_tile_gb: float
    pose_pixel_tile_gb: float
    translation_tile_gb: float
    score_pixel_count: int


def _firstiter_cc_recon_complex_budget() -> int:
    raw = os.environ.get(RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV)
    if raw is None or raw.strip() == "":
        return int(RELION_FIRSTITER_RECON_COMPLEX_BUDGET)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{RELION_FIRSTITER_RECON_COMPLEX_BUDGET_ENV} must be a positive integer")
    return value


def _safe_firstiter_cc_image_batch_size(n_trans, image_shape):
    """Cap dense K-class reconstruction batches by the temporary footprint.

    ``prepare_reconstruction_batch`` materializes a
    ``batch_size × n_trans × n_half`` complex tensor before any class or
    pose masking can trim anything.  The generic score-tensor budget does
    not account for that temporary, so dense K-class runs that keep
    ``score_with_masked_images=True`` need a separate clamp.  The
    first-iteration winner-take-all route is the most obvious case, but
    the same bound also protects later dense K-class iterations that reuse
    the same reconstruction path.
    """

    n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    return max(1, _firstiter_cc_recon_complex_budget() // max(int(n_trans) * n_half, 1))


def _safe_dense_k_class_rotation_block_size(n_trans, image_batch_size):
    """Cap dense K-class rotation buckets by a microbatch hypothesis budget.

    The dense K-class adaptive probe still has to evaluate a dense (batch,
    rotation, translation) tensor in its big-JIT path.  RELION's own
    bucketed pass2 code keeps similar hypothesis tensors under a ~2e6
    per-microbatch ceiling, so mirror that bound here to keep the probe
    memory-safe without changing the score math.
    """

    return max(
        64,
        RELION_DENSE_K_CLASS_HYPOTHESES_BUDGET // max(int(image_batch_size) * max(int(n_trans), 1), 1),
    )


@dataclass(frozen=True)
class _AdaptiveDenseBatchSizes:
    """Separate dense batch plans for adaptive pass 1 and pass 2."""

    pass2_image_batch_size: int
    pass2_rotation_block_size: int
    significance_image_batch_size: int
    significance_rotation_block_size: int


def _plan_adaptive_dense_batch_sizes(
    *,
    n_rot: int,
    n_trans: int,
    n_classes: int,
    image_shape,
    cs_for_engine,
    coarse_cs,
    k_class_enabled: bool,
    safe_batch_sizes,
) -> _AdaptiveDenseBatchSizes:
    """Plan adaptive dense microbatches from each pass' Fourier window."""

    pass2_image_batch_size, pass2_rotation_block_size = safe_batch_sizes(
        n_rot,
        n_trans,
        classes=n_classes,
        image_shape_for_batch=image_shape,
        current_size_for_batch=cs_for_engine,
    )
    if k_class_enabled:
        pass2_image_batch_size = min(
            pass2_image_batch_size,
            _safe_firstiter_cc_image_batch_size(
                n_trans,
                image_shape,
            ),
        )
        pass2_rotation_block_size = min(
            pass2_rotation_block_size,
            _safe_dense_k_class_rotation_block_size(
                n_trans,
                pass2_image_batch_size,
            ),
        )

    significance_image_batch_size, significance_rotation_block_size = safe_batch_sizes(
        n_rot,
        n_trans,
        classes=n_classes,
        image_shape_for_batch=image_shape,
        current_size_for_batch=coarse_cs,
    )
    return _AdaptiveDenseBatchSizes(
        pass2_image_batch_size=int(pass2_image_batch_size),
        pass2_rotation_block_size=int(pass2_rotation_block_size),
        significance_image_batch_size=int(significance_image_batch_size),
        significance_rotation_block_size=int(significance_rotation_block_size),
    )


def _plan_kclass_adaptive_grid_batch_sizes(
    *,
    coarse_rotations,
    coarse_translations,
    fine_rotations,
    fine_translations,
    n_classes: int,
    image_shape,
    coarse_current_size,
    fine_current_size,
    safe_batch_sizes,
) -> _AdaptiveDenseBatchSizes:
    """Plan K-class adaptive pass-1/pass-2 batches from the actual grids."""

    pass2_image_batch_size, pass2_rotation_block_size = safe_batch_sizes(
        int(np.asarray(fine_rotations).shape[0]),
        int(np.asarray(fine_translations).shape[0]),
        classes=n_classes,
        image_shape_for_batch=image_shape,
        current_size_for_batch=fine_current_size,
    )
    pass2_image_batch_size = min(
        pass2_image_batch_size,
        _safe_firstiter_cc_image_batch_size(
            int(np.asarray(fine_translations).shape[0]),
            image_shape,
        ),
    )
    if int(n_classes) > 1:
        pass2_rotation_block_size = min(
            pass2_rotation_block_size,
            _safe_dense_k_class_rotation_block_size(
                int(np.asarray(fine_translations).shape[0]),
                pass2_image_batch_size,
            ),
        )

    significance_image_batch_size, significance_rotation_block_size = safe_batch_sizes(
        int(np.asarray(coarse_rotations).shape[0]),
        int(np.asarray(coarse_translations).shape[0]),
        classes=n_classes,
        image_shape_for_batch=image_shape,
        current_size_for_batch=coarse_current_size,
    )
    significance_image_batch_size = min(
        significance_image_batch_size,
        _safe_firstiter_cc_image_batch_size(
            int(np.asarray(coarse_translations).shape[0]),
            image_shape,
        ),
    )
    if int(n_classes) > 1:
        significance_rotation_block_size = min(
            significance_rotation_block_size,
            _safe_dense_k_class_rotation_block_size(
                int(np.asarray(coarse_translations).shape[0]),
                significance_image_batch_size,
            ),
        )

    return _AdaptiveDenseBatchSizes(
        pass2_image_batch_size=int(pass2_image_batch_size),
        pass2_rotation_block_size=int(pass2_rotation_block_size),
        significance_image_batch_size=int(significance_image_batch_size),
        significance_rotation_block_size=int(significance_rotation_block_size),
    )


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite float, got {raw!r}") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite float, got {raw!r}")
    return float(value)


def _active_half_spectrum_pixels(image_shape, current_size: int | None) -> int:
    full_half_pixels = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    if current_size is None:
        return full_half_pixels
    radius = max(0, int(current_size) // 2)
    if radius <= 0:
        return 1
    row_freq = np.minimum(np.arange(int(image_shape[0])), int(image_shape[0]) - np.arange(int(image_shape[0])))
    col_freq = np.arange(int(image_shape[1]) // 2 + 1)
    keep = row_freq[:, None] * row_freq[:, None] + col_freq[None, :] * col_freq[None, :] <= radius * radius
    return max(1, min(full_half_pixels, int(np.count_nonzero(keep))))


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
    current_size: int | None = None,
) -> _RelionEMBatchPlan:
    """Choose EM microbatch sizes from pose-grid, image, class, and GPU size."""
    from recovar import utils

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
            gpu_memory_gb = float(utils.get_gpu_memory_total())
        except Exception:
            gpu_memory_gb = _RELION_EM_BATCH_DEFAULT_GPU_GB
        try:
            gpu_used_gb = float(utils.get_gpu_memory_used())
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
    projection_fraction = _positive_env_float(
        _RELION_EM_BATCH_PROJECTION_FRACTION_ENV,
        _RELION_EM_BATCH_PROJECTION_FRACTION,
    )
    projection_budget_gb = max(
        _RELION_EM_BATCH_MIN_PROJECTION_GB,
        min(_RELION_EM_BATCH_MAX_PROJECTION_GB, usable_gb * projection_fraction),
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
    score_half_pixels = _active_half_spectrum_pixels(image_shape, current_size)
    active_score_tile_budget_gb = max(
        _RELION_EM_BATCH_MIN_PROJECTION_GB,
        min(
            projection_budget_gb,
            gpu_memory_gb * _RELION_EM_BATCH_ACTIVE_SCORE_TILE_FRACTION,
            usable_from_runtime_gb * _RELION_EM_BATCH_ACTIVE_SCORE_TILE_FRACTION,
        ),
    )
    active_score_bytes_per_image = max(
        1,
        int(
            np.ceil(
                n_trans
                * score_half_pixels
                * np.dtype(np.complex128).itemsize
                * n_classes
                * _RELION_EM_BATCH_ACTIVE_SCORE_TILE_LIVE_FACTOR,
            )
        ),
    )
    active_score_image_cap = max(1, int(active_score_tile_budget_gb * 1e9 // active_score_bytes_per_image))
    translation_bytes_per_image = max(
        1,
        2 * n_trans * full_half_pixels * np.dtype(np.complex64).itemsize * n_classes,
    )
    translation_image_cap = max(1, int(translation_tile_budget_gb * 1e9 // translation_bytes_per_image))
    image_batch = min(requested_image_batch_size, score_image_cap, translation_image_cap, active_score_image_cap)

    score_rotation_cap = max(1, score_float_budget // max(image_batch * n_trans * n_classes, 1))
    projection_half_pixels = full_half_pixels if current_size is None else score_half_pixels
    projection_bytes_per_rotation = max(
        1,
        int(
            np.ceil(
                projection_half_pixels
                * np.dtype(np.complex64).itemsize
                * n_classes
                * _RELION_EM_BATCH_PROJECTION_LIVE_FACTOR,
            ),
        ),
    )
    projection_rotation_cap = max(1, int(projection_budget_gb * 1e9 // projection_bytes_per_rotation))
    score_matmul_bytes_per_rotation = max(
        1,
        int(
            np.ceil(
                score_half_pixels
                * np.dtype(np.complex64).itemsize
                * n_classes
                * _RELION_EM_BATCH_SCORE_MATMUL_LIVE_FACTOR,
            )
        ),
    )
    score_matmul_rotation_cap = max(1, int(projection_budget_gb * 1e9 // score_matmul_bytes_per_rotation))
    active_window_fraction = score_half_pixels / max(1, full_half_pixels)
    if active_window_fraction > _RELION_EM_BATCH_POSE_PIXEL_WINDOW_FRACTION:
        pose_pixel_budget_gb = max(
            _RELION_EM_BATCH_MIN_PROJECTION_GB,
            min(
                projection_budget_gb,
                gpu_memory_gb * _RELION_EM_BATCH_POSE_PIXEL_FRACTION,
                usable_from_runtime_gb * _RELION_EM_BATCH_RUNTIME_FREE_FRACTION,
            ),
        )
    else:
        pose_pixel_budget_gb = projection_budget_gb
    # Dense scoring can materialize a rotation x translation x active-pixel
    # complex tile. JAX runs with x64 enabled, so budget this as complex128.
    # This is load-bearing for 100k/256 K=1 global searches: otherwise the
    # planner allows the full 36,864-rotation block and XLA tries to allocate
    # a 22 GiB pose-pixel tile in one shot.
    pose_pixel_bytes_per_rotation = max(
        1,
        int(
            np.ceil(
                n_trans
                * score_half_pixels
                * np.dtype(np.complex128).itemsize
                * n_classes
                * _RELION_EM_BATCH_POSE_PIXEL_LIVE_FACTOR,
            )
        ),
    )
    pose_pixel_rotation_cap = max(1, int(pose_pixel_budget_gb * 1e9 // pose_pixel_bytes_per_rotation))
    rotation_cap = min(
        score_rotation_cap,
        projection_rotation_cap,
        score_matmul_rotation_cap,
        pose_pixel_rotation_cap,
    )

    if requested_rotation_block_size >= 64:
        rotation_block = min(requested_rotation_block_size, rotation_cap)
        if rotation_cap >= 64:
            rotation_block = max(64, rotation_block)
    else:
        rotation_block = min(requested_rotation_block_size, rotation_cap)
    rotation_block = max(1, min(rotation_block, n_rot))

    projection_block_gb = rotation_block * projection_bytes_per_rotation / 1e9
    active_score_tile_gb = image_batch * active_score_bytes_per_image / 1e9
    pose_pixel_tile_gb = rotation_block * pose_pixel_bytes_per_rotation / 1e9
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
        active_score_tile_budget_gb=float(active_score_tile_budget_gb),
        active_score_tile_gb=float(active_score_tile_gb),
        pose_pixel_tile_gb=float(pose_pixel_tile_gb),
        translation_tile_gb=float(translation_tile_gb),
        score_pixel_count=int(score_half_pixels),
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
