"""Exact-local batch limits and runtime memory probes.

The engine applies these policies at dispatch; reporting uses the same limits
without importing execution kernels. Probe results are cached here. Importing
this module retains its dependencies' normal import behavior; memory probes
execute when the planning functions request them.
"""

from __future__ import annotations

import logging
import os
import subprocess

import jax
import numpy as np

from .helpers.deterministic_reduce import deterministic_reductions_enabled
from .local_layout import (
    LocalHypothesisLayout,
    _exact_bucket_rotation_size,
    _exact_local_large_bucket_quantum,
)

logger = logging.getLogger(__name__)

# Keeps common 256^2 local-search buckets at two images without entering the
# three-image working set that previously exceeded memory.
EXACT_LOCAL_TARGET_ROW_PIXELS = 190_000_000
EXACT_LOCAL_TARGET_ROW_PIXELS_ENV = "RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS"
EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB = 4.0
EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB"
EXACT_LOCAL_HIGH_MEMORY_GPU_BYTES = 70 * 1024**3
EXACT_LOCAL_HIGH_MEMORY_TARGET_ROW_PIXELS = 256_000_000
EXACT_LOCAL_HIGH_MEMORY_BIG_JIT_MATMUL_MAX_GB = 8.0
EXACT_LOCAL_AUTO_MICROBATCH_BOOST = 2.0
EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV = "RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST"
EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST = 1.0
EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST_ENV = "RECOVAR_EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST"
# The fused RELION-projector M-step has a projection/interpolation temporary
# whose peak follows padded rotation rows times projected pixels.  A 384-box
# H100 run completed 37x128x8258 row-pixels but OOMed when the next static
# bucket doubled to 37x256x8258 and requested a 10.12-GiB allocation.  Keep
# automatic x-half buckets on the proven side of that boundary.  The cap never
# splits one particle's exact rotation neighborhood.
EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS = 40_000_000
EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV = (
    "RECOVAR_EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS"
)
# Score-only big-JIT lowers the score residual to a dense
# (image, rotation, translation, pixel) float32 tile.  Limit that one tile to
# a conservative share of memory that is still free at local-search entry;
# the remaining memory is needed by projections, inputs, outputs, and XLA.
EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION = 0.20
EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR = 1.25

_VISIBLE_GPU_MEMORY_BYTES_CACHE: int | None = None


def _visible_gpu_memory_bytes() -> int | None:
    """Return visible GPU memory in bytes when nvidia-smi is available."""

    global _VISIBLE_GPU_MEMORY_BYTES_CACHE
    if _VISIBLE_GPU_MEMORY_BYTES_CACHE is not None:
        return _VISIBLE_GPU_MEMORY_BYTES_CACHE
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        _VISIBLE_GPU_MEMORY_BYTES_CACHE = 0
        return None
    if proc.returncode != 0:
        _VISIBLE_GPU_MEMORY_BYTES_CACHE = 0
        return None
    values = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(int(float(stripped.split()[0])))
        except ValueError:
            continue
    if not values:
        _VISIBLE_GPU_MEMORY_BYTES_CACHE = 0
        return None
    _VISIBLE_GPU_MEMORY_BYTES_CACHE = int(max(values)) * 1024**2
    return _VISIBLE_GPU_MEMORY_BYTES_CACHE


def _exact_local_runtime_free_memory_bytes() -> int | None:
    """Return allocator bytes not currently live on the first local GPU."""

    try:
        devices = jax.local_devices()
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    bytes_limit = stats.get("bytes_limit")
    bytes_in_use = stats.get("bytes_in_use")
    if bytes_limit is None or bytes_in_use is None:
        return None
    free_bytes = int(bytes_limit) - int(bytes_in_use)
    return free_bytes if free_bytes > 0 else None


def _exact_local_default_target_row_pixels(*, allow_high_memory_default: bool = True) -> int:
    raw = os.environ.get(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "").strip()
    if raw:
        return int(raw)
    memory_bytes = _visible_gpu_memory_bytes()
    if (
        bool(allow_high_memory_default)
        and memory_bytes is not None
        and int(memory_bytes) >= EXACT_LOCAL_HIGH_MEMORY_GPU_BYTES
    ):
        return int(EXACT_LOCAL_HIGH_MEMORY_TARGET_ROW_PIXELS)
    return int(EXACT_LOCAL_TARGET_ROW_PIXELS)


def _exact_local_default_big_jit_matmul_max_gb(*, allow_high_memory_default: bool = True) -> float:
    raw = os.environ.get(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, "").strip()
    if raw:
        return float(raw)
    memory_bytes = _visible_gpu_memory_bytes()
    if (
        bool(allow_high_memory_default)
        and memory_bytes is not None
        and int(memory_bytes) >= EXACT_LOCAL_HIGH_MEMORY_GPU_BYTES
    ):
        return float(EXACT_LOCAL_HIGH_MEMORY_BIG_JIT_MATMUL_MAX_GB)
    return float(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB)


def _exact_local_max_hypotheses_per_microbatch(
    default: int | None,
    n_windowed: int,
    *,
    n_trans: int = 1,
    n_recon_windowed: int | None = None,
    allow_high_memory_default: bool = True,
) -> int:
    """Return exact-local microbatch cap.

    The automatic default targets the proven 5k/128 local-search working set
    while scaling down for larger Fourier windows.
    """
    if default is not None:
        value = int(default)
        if value <= 0:
            raise ValueError("max_hypotheses_per_microbatch must be positive")
        return value
    target_row_pixels = _exact_local_default_target_row_pixels(
        allow_high_memory_default=allow_high_memory_default
    )
    if target_row_pixels <= 0:
        raise ValueError(f"{EXACT_LOCAL_TARGET_ROW_PIXELS_ENV} must be positive")
    value = target_row_pixels // max(1, int(n_windowed))
    max_gb = _exact_local_default_big_jit_matmul_max_gb(
        allow_high_memory_default=allow_high_memory_default
    )
    if max_gb > 0.0:
        # The fused local M-step lowers to a matmul whose large outputs are
        # per-rotation image sums, not a literal (rotation, translation, pixel)
        # tensor. Cap the row count by those output rows; multiplying by
        # ``n_trans * n_recon`` here serializes broad local pass-2 supports into
        # one-image buckets without reflecting the actual compiled working set.
        n_recon = int(n_windowed if n_recon_windowed is None else n_recon_windowed)
        if int(n_trans) <= 1:
            matmul_row_bytes = 4 * max(1, n_recon)
        else:
            matmul_row_bytes = (
                # score projection row, normally complex64
                8 * max(1, int(n_windowed))
                # posterior-weighted image row; keep complex128 headroom because
                # RELION-mode normalization may keep probabilities in float64.
                + 16 * max(1, n_recon)
                # CTF/probability row and score/probability vectors.
                + 4 * max(1, n_recon)
                + 16 * max(1, int(n_trans))
            )
        matmul_cap = int((max_gb * 1e9) // max(1, matmul_row_bytes))
        value = min(value, matmul_cap)
    return int(max(512, min(65536, value)))


def _exact_local_microbatch_env_overridden() -> bool:
    return bool(
        os.environ.get(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "").strip()
        or os.environ.get(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, "").strip()
    )


def _exact_local_auto_microbatch_boost() -> float:
    raw = os.environ.get(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using default %.1f",
                EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV,
                raw,
                EXACT_LOCAL_AUTO_MICROBATCH_BOOST,
            )
            return float(EXACT_LOCAL_AUTO_MICROBATCH_BOOST)
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"{EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV} must be positive and finite")
        return value
    return float(EXACT_LOCAL_AUTO_MICROBATCH_BOOST)


def _exact_local_xhalf_auto_microbatch_boost() -> float:
    raw = os.environ.get(EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using default %.2f",
                EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST_ENV,
                raw,
                EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST,
            )
            return float(EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST)
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"{EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST_ENV} must be positive and finite")
        return value
    return float(EXACT_LOCAL_XHALF_AUTO_MICROBATCH_BOOST)


def _exact_local_planned_hypotheses_floor(
    local_layout: LocalHypothesisLayout,
    *,
    image_batch_size: int,
    rotation_block_size: int,
    exact_local_bucket_radix: int | None = None,
) -> int:
    """Minimum row cap needed to honor the memory planner's image batch."""

    image_batch_size = max(1, int(image_batch_size))
    rotation_counts = np.asarray(local_layout.rotation_counts, dtype=np.int64)
    if rotation_counts.size == 0:
        return image_batch_size
    max_rotation_count = int(np.max(rotation_counts, initial=1))
    large_bucket_quantum = _exact_local_large_bucket_quantum(rotation_block_size)
    bucket_rotation_count = _exact_bucket_rotation_size(
        max_rotation_count,
        rotation_block_size,
        large_bucket_quantum=large_bucket_quantum,
        exact_local_bucket_radix=exact_local_bucket_radix,
    )
    return int(image_batch_size * max(1, bucket_rotation_count))


def _exact_local_effective_max_hypotheses_per_microbatch(
    default: int | None,
    n_windowed: int,
    *,
    n_trans: int = 1,
    n_recon_windowed: int | None = None,
    local_layout: LocalHypothesisLayout,
    image_batch_size: int,
    rotation_block_size: int,
    exact_local_bucket_radix: int | None = None,
    allow_auto_boost: bool = True,
    auto_boost_factor: float | None = None,
    allow_high_memory_default: bool = True,
    score_only: bool = False,
    runtime_free_memory_bytes: int | None = None,
) -> int:
    cap = _exact_local_max_hypotheses_per_microbatch(
        default,
        n_windowed,
        n_trans=n_trans,
        n_recon_windowed=n_recon_windowed,
        allow_high_memory_default=allow_high_memory_default,
    )
    if default is not None or _exact_local_microbatch_env_overridden():
        return cap
    if not bool(allow_auto_boost):
        return cap
    planned_floor = _exact_local_planned_hypotheses_floor(
        local_layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        exact_local_bucket_radix=exact_local_bucket_radix,
    )
    boost_factor = _exact_local_auto_microbatch_boost() if auto_boost_factor is None else float(auto_boost_factor)
    if boost_factor <= 0.0 or not np.isfinite(boost_factor):
        raise ValueError("auto_boost_factor must be positive and finite")
    boost_cap = int(np.floor(cap * boost_factor))
    effective_cap = int(max(cap, min(65536, planned_floor, boost_cap)))
    if not score_only:
        return effective_cap

    if deterministic_reductions_enabled():
        # The live-memory probe below makes the microbatch cap depend on the
        # allocator state of this process, which changes the accumulation
        # grouping between otherwise identical runs.  Under the opt-in keep
        # the profiled base cap, which is memory-safe and process independent.
        return cap
    if runtime_free_memory_bytes is None:
        runtime_free_memory_bytes = _exact_local_runtime_free_memory_bytes()
    if runtime_free_memory_bytes is None:
        # Without a runtime-free probe, retain the profiled base cap rather
        # than applying an unbounded image-batch boost.
        return cap
    score_tile_bytes_per_hypothesis = (
        max(1, int(n_trans))
        * max(1, int(n_windowed))
        * np.dtype(np.float32).itemsize
        * EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR
    )
    score_tile_cap = int(
        int(runtime_free_memory_bytes)
        * EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION
        // score_tile_bytes_per_hypothesis
    )
    return int(max(1, min(effective_cap, score_tile_cap)))


def _exact_local_xhalf_tail_microbatch_cap(
    cap: int,
    local_layout: LocalHypothesisLayout,
    *,
    image_batch_size: int,
    rotation_block_size: int,
) -> int:
    """Respect the outer memory plan for oversized x-half neighborhoods.

    The outer planner sizes a local M-step tile as
    ``image_batch_size * rotation_block_size``. Exact neighborhoods may be
    wider than ``rotation_block_size`` and therefore cannot be split along the
    rotation axis, but they must reduce the number of images in the bucket.
    Otherwise a tail bucket can keep the ordinary image count and exceed the
    planned row tile by several times.
    """

    cap = max(1, int(cap))
    rotation_block_size = max(1, int(rotation_block_size))
    rotation_counts = np.asarray(local_layout.rotation_counts, dtype=np.int64)
    max_rotation_count = int(np.max(rotation_counts, initial=0))
    if max_rotation_count <= rotation_block_size:
        return cap
    planned_row_cap = max(1, int(image_batch_size)) * rotation_block_size
    return min(cap, planned_row_cap)


def _exact_local_xhalf_projection_target_row_pixels() -> int:
    """Resolve the x-half projection row-pixel budget."""

    target_row_pixels = int(EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS)
    raw = os.environ.get(EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV, "").strip()
    if raw:
        try:
            target_row_pixels = max(1, int(raw))
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using default %d row-pixels",
                EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV,
                raw,
                EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS,
            )
            target_row_pixels = int(EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS)
    return target_row_pixels


def _exact_local_xhalf_projection_microbatch_cap(
    cap: int,
    local_layout: LocalHypothesisLayout,
    *,
    n_projection_pixels: int,
    rotation_block_size: int,
    exact_local_bucket_radix: int | None = None,
) -> int:
    """Bound fused x-half projection rows without truncating neighborhoods."""

    cap = max(1, int(cap))
    n_projection_pixels = max(1, int(n_projection_pixels))
    rotation_block_size = max(1, int(rotation_block_size))
    target_row_pixels = _exact_local_xhalf_projection_target_row_pixels()

    rotation_counts = np.asarray(local_layout.rotation_counts, dtype=np.int64)
    if rotation_counts.size == 0:
        return cap
    large_bucket_quantum = _exact_local_large_bucket_quantum(rotation_block_size)
    max_bucket_rotation_count = max(
        _exact_bucket_rotation_size(
            int(count),
            rotation_block_size,
            large_bucket_quantum=large_bucket_quantum,
            exact_local_bucket_radix=exact_local_bucket_radix,
        )
        for count in rotation_counts
    )
    projection_row_cap = max(1, target_row_pixels // n_projection_pixels)
    # Exact neighborhoods are indivisible; permit at least one image from the
    # largest padded bucket even when that exceeds the configured row target.
    safe_cap = max(int(max_bucket_rotation_count), int(projection_row_cap))
    return min(cap, safe_cap)
