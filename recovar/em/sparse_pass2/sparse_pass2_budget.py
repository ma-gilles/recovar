"""Device-memory budgets of the sparse bucketed pass 2.

Device memory limits and free-memory probes (JAX allocator, nvidia-smi) and
the per-pass byte budgets derived from them: hypotheses per microbatch,
translation tiles, projection gathers and caches, noise and adjoint blocks.
The positive-int/float environment parsers live here because the budgets
and the policy switches both read them.
"""

from __future__ import annotations

import logging
import os
import subprocess

import jax
import numpy as np

from recovar.em.helpers.env_flags import parse_env_nonnegative_int
from recovar.em.scoring.sparse_bucket_arrays import _DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH

logger = logging.getLogger(__name__)


_DEFAULT_SCORE_ONLY_MAX_HYPOTHESES_PER_MICROBATCH = 1_250_000


_DEFAULT_MAX_TRANSLATION_TILE_BYTES = 384 * 1024**2


_AUTO_SCORE_ONLY_HYPOTHESIS_DEVICE_FRACTION = 0.640


_AUTO_FULL_HYPOTHESIS_DEVICE_FRACTION = 0.305


_AUTO_FUSED_KCLASS_SCORE_GATHER_DEVICE_FRACTION = 0.100


_AUTO_FUSED_KCLASS_LIVE_COMPLEX_GATHERS = 2


_AUTO_TRANSLATION_TILE_DEVICE_FRACTION = 0.020


_AUTO_EXTERNAL_NORMALIZATION_TRANSLATION_TILE_DEVICE_FRACTION = 0.014


_AUTO_FUSED_KCLASS_TRANSLATION_TILE_DEVICE_FRACTION = 0.007


# Fine-projection cache cap as a fraction of device memory.  The K=1 sparse
# pass-2 recomputes every image chunk's fine projections when the cache is
# skipped; at HEALPix order 3 (294912 fine rotations, current_size 92, 256^2)
# the score+recon+abs2 cache estimate is 18.4 GiB, which the former 10% cap
# (7.96 GiB on an 80 GB device) rejected in every hp3 iteration of the 10k
# EMPIAR-10097 convergence run while per-chunk recompute cost 1200-3900 s per
# iteration.  25% admits that cache on 80 GB devices and still rejects it on
# 40 GB devices.  Measured evidence: docs handoff em_soft_posterior_block_bpref
# prototype 2026-09-17, jobs 14045912 / 14046044.
_AUTO_PROJECTION_CACHE_DEVICE_FRACTION = 0.250


_AUTO_PROJECTED_ROTATIONS_DEVICE_FRACTION = 0.040


_AUTO_PROJECTION_GATHER_DEVICE_FRACTION = 0.020


_AUTO_NOISE_BLOCK_DEVICE_FRACTION = 0.0125


_AUTO_ADJOINT_BLOCK_DEVICE_FRACTION = 0.006


_DEFAULT_PROJECTION_GATHER_MAX_BYTES = 1024 * 1024**2


_DEFAULT_NOISE_BLOCK_MAX_BYTES = 512 * 1024**2


_DEFAULT_ADJOINT_BLOCK_MAX_BYTES = 512 * 1024**2


_EXACT_RAW_DIFF2_CACHE_MAX_BYTES = 512 * 1024**2


_EXACT_RAW_DIFF2_CACHE_DEVICE_FRACTION = 0.01


_EXACT_RAW_DIFF2_CACHE_FREE_FRACTION = 0.25


_MAX_HYPOTHESES_ENV = "RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES"


_SCORE_ONLY_MAX_HYPOTHESES_ENV = "RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES"


_MAX_TRANSLATION_TILE_BYTES_ENV = "RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES"


_MAX_PROJECTION_GATHER_BYTES_ENV = "RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES"


_MAX_NOISE_BLOCK_BYTES_ENV = "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES"


_MAX_ADJOINT_BLOCK_BYTES_ENV = "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES"


_COMPACT_PAIR_DENSE_MSTEP_MAX_BYTES_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DENSE_MSTEP_MAX_BYTES"


_MAX_PROJECTED_ROTATIONS_ENV = "RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS"


_PROJECTION_CACHE_MAX_BYTES_ENV = "RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES"


_DEFAULT_PROJECTION_CACHE_MAX_BYTES = 3 * 1024**3


def _optional_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _optional_positive_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive float, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive float, got {raw!r}")
    return value


def _parse_nvidia_smi_memory_rows(output: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index, uuid, memory_mib = parts[:3]
        try:
            memory_bytes = int(memory_mib.split()[0]) * 1024**2
        except (ValueError, IndexError):
            continue
        if memory_bytes <= 0:
            continue
        rows[index] = memory_bytes
        rows[uuid] = memory_bytes
        if uuid.startswith("GPU-"):
            rows[uuid[4:]] = memory_bytes
    return rows


def _nvidia_smi_visible_device_memory_bytes(output: str, visible_devices: str | None) -> int | None:
    rows = _parse_nvidia_smi_memory_rows(output)
    if not rows:
        return None
    if visible_devices:
        tokens = [
            part.strip()
            for part in visible_devices.split(",")
            if part.strip() and part.strip() not in {"-1", "none", "NoDevFiles"}
        ]
        if not tokens:
            return None
        for token in tokens:
            if token in rows:
                return rows[token]
        return None
    return next(iter(rows.values()))


_CONCURRENT_DEVICE_SHARES = 1


def _share(total_bytes: int | None) -> int | None:
    """This worker's slice of a device total, given the declared share count."""

    if total_bytes is None:
        return None
    shares = _CONCURRENT_DEVICE_SHARES
    if shares <= 1:
        return int(total_bytes)
    return max(1, int(total_bytes) // shares)


def set_concurrent_device_shares(shares: int) -> int:
    """Declare how many workers are sharing this device, and return the previous count.

    Every budget in this module is a fraction of the device's memory and is
    written for one worker at a time. When two half-sets run concurrently they
    size their caches against the same device, so each must be told it owns
    only its share; otherwise both admit a plan that fits alone and neither
    fits together. That is not hypothetical: overlapping the two halves at
    HEALPix order 3 failed with RESOURCE_EXHAUSTED building the second half's
    projection cache, because each half had budgeted the whole 80 GiB device.

    **This declaration covers device memory only.** Host-side caches are not
    fractions of a device and do not pass through this module, so they neither
    shrink nor are checked when the share count rises, while a second
    concurrent worker doubles them just the same. Two at the time of writing:
    the exact-CTF operand memo in ``recovar/em/relion/relion_ctf.py``, about
    2.4-2.6 GB of host RAM for two half operands when enabled, and the image
    loader's prefetch slots at roughly 65 MB each. Budget those on the host
    side; nothing here will.
    """

    global _CONCURRENT_DEVICE_SHARES
    shares = int(shares)
    if shares < 1:
        raise ValueError(f"concurrent device shares must be at least 1, got {shares}")
    previous, _CONCURRENT_DEVICE_SHARES = _CONCURRENT_DEVICE_SHARES, shares
    return previous


def concurrent_device_shares() -> int:
    """How many workers are currently declared to share this device."""

    return _CONCURRENT_DEVICE_SHARES


def _device_memory_limit_bytes() -> int | None:
    """Return this worker's share of the selected accelerator's memory.

    The share is the whole device unless several workers have been declared
    through :func:`set_concurrent_device_shares`, in which case every fraction
    computed downstream is a fraction of one worker's share.
    """

    # ``RECOVAR_SPARSE_PASS2_DEVICE_MEMORY_GB`` overrides the nvidia-smi probe.
    # Keep this as a manual escape hatch for reserving headroom on shared GPUs
    # or working around inaccurate allocator/device probes.
    _override = os.environ.get("RECOVAR_SPARSE_PASS2_DEVICE_MEMORY_GB")
    if _override is not None:
        try:
            override_gb = float(_override.strip())
            if override_gb > 0:
                return _share(int(override_gb * (1024 ** 3)))
        except ValueError:
            pass

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if query.returncode == 0:
            memory_bytes = _nvidia_smi_visible_device_memory_bytes(
                query.stdout,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
            if memory_bytes is not None:
                return _share(memory_bytes)
    except Exception:
        pass
    try:
        devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    for key in ("bytes_limit", "bytesLimit", "memory_limit", "total_memory"):
        value = stats.get(key)
        if value is not None and int(value) > 0:
            return _share(int(value))
    return None


def _device_free_memory_bytes() -> int | None:
    """Return current free memory for the selected physical GPU, if known."""

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if query.returncode == 0:
            return _nvidia_smi_visible_device_memory_bytes(
                query.stdout,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
    except Exception:
        pass
    return None


def _jax_allocator_free_memory_bytes() -> int | None:
    """Return unused bytes in the active JAX GPU allocator, if reported."""

    try:
        devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None

    limit = next(
        (
            int(stats[key])
            for key in ("bytes_limit", "bytesLimit", "memory_limit", "total_memory")
            if stats.get(key) is not None and int(stats[key]) > 0
        ),
        None,
    )
    bytes_in_use = next(
        (
            int(stats[key])
            for key in ("bytes_in_use", "bytesInUse", "memory_in_use")
            if stats.get(key) is not None and int(stats[key]) >= 0
        ),
        None,
    )
    if limit is None or bytes_in_use is None:
        return None
    return max(0, limit - bytes_in_use)


def _exact_raw_diff2_cache_limit_bytes(
    device_memory_bytes: int | None,
    free_device_memory_bytes: int | None,
    allocator_free_memory_bytes: int | None,
    *,
    max_cache_bytes: int = _EXACT_RAW_DIFF2_CACHE_MAX_BYTES,
) -> int:
    """Return the strict per-bucket cap for exact fine-score reuse."""

    if (
        device_memory_bytes is None
        or free_device_memory_bytes is None
        or allocator_free_memory_bytes is None
        or int(device_memory_bytes) <= 0
        or int(free_device_memory_bytes) <= 0
        or int(allocator_free_memory_bytes) <= 0
        or int(max_cache_bytes) <= 0
    ):
        return 0
    return min(
        int(max_cache_bytes),
        int(int(device_memory_bytes) * _EXACT_RAW_DIFF2_CACHE_DEVICE_FRACTION),
        int(int(free_device_memory_bytes) * _EXACT_RAW_DIFF2_CACHE_FREE_FRACTION),
        int(int(allocator_free_memory_bytes) * _EXACT_RAW_DIFF2_CACHE_FREE_FRACTION),
    )


def _exact_raw_diff2_cache_estimated_bytes(
    batch_size: int,
    bucket_size: int,
    n_fine_translations: int,
    dtype=np.float32,
) -> int:
    return (
        int(batch_size)
        * int(bucket_size)
        * int(n_fine_translations)
        * np.dtype(dtype).itemsize
    )


def _exact_raw_diff2_cache_fits_budget(estimated_bytes: int, cache_limit_bytes: int) -> bool:
    return int(estimated_bytes) > 0 and int(estimated_bytes) <= int(cache_limit_bytes)


def _dtype_itemsize(dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def _complex_counterpart_real_dtype(complex_dtype):
    complex_dtype = np.dtype(complex_dtype)
    if complex_dtype.itemsize <= np.dtype(np.complex64).itemsize:
        return np.float32
    return np.float64


def _auto_hypotheses_per_microbatch(
    *,
    score_only: bool,
    fused_k_class: bool = False,
    fused_k_class_count: int | None = None,
    n_score_pixels: int | None,
    device_memory_bytes: int | None,
    score_complex_dtype=np.complex64,
) -> int | None:
    if device_memory_bytes is None or n_score_pixels is None or int(n_score_pixels) <= 0:
        return None
    if score_only:
        fraction = _AUTO_SCORE_ONLY_HYPOTHESIS_DEVICE_FRACTION
    elif fused_k_class:
        if fused_k_class_count is None or int(fused_k_class_count) <= 0:
            raise ValueError("fused_k_class_count must be positive for fused K-class planning")
        bytes_per_score_pixel = _dtype_itemsize(score_complex_dtype)
        return max(
            1,
            int(
                float(device_memory_bytes)
                * _AUTO_FUSED_KCLASS_SCORE_GATHER_DEVICE_FRACTION
                * int(fused_k_class_count)
                / (
                    int(n_score_pixels)
                    * bytes_per_score_pixel
                    * _AUTO_FUSED_KCLASS_LIVE_COMPLEX_GATHERS
                )
            ),
        )
    else:
        fraction = _AUTO_FULL_HYPOTHESIS_DEVICE_FRACTION
    # The score kernel's dominant live block scales with candidate count times
    # active Fourier pixels. This keeps larger windows and smaller GPUs from
    # inheriting the same candidate cap as low-resolution H100 runs.
    bytes_per_score_pixel = _dtype_itemsize(score_complex_dtype)
    return max(1, int(float(device_memory_bytes) * fraction / (int(n_score_pixels) * bytes_per_score_pixel)))


def _max_hypotheses_per_microbatch_for_pass(
    *,
    score_only: bool,
    use_window: bool,
    has_external_normalization: bool,
    conservative_dump_execution: bool,
    fused_k_class: bool = False,
    fused_k_class_count: int | None = None,
    n_score_pixels: int | None = None,
    device_memory_bytes: int | None = None,
    score_complex_dtype=np.complex64,
) -> int:
    if score_only and use_window and not has_external_normalization and not conservative_dump_execution:
        override = _optional_positive_int_env(_SCORE_ONLY_MAX_HYPOTHESES_ENV)
        auto = _auto_hypotheses_per_microbatch(
            score_only=True,
            fused_k_class=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory_bytes,
            score_complex_dtype=score_complex_dtype,
        )
        if override is not None:
            if auto is not None and int(override) < int(auto):
                logger.warning(
                    "%s=%d is below the auto sparse pass-2 score-only cap %d; "
                    "this can fragment buckets and slow pass-2.",
                    _SCORE_ONLY_MAX_HYPOTHESES_ENV,
                    int(override),
                    int(auto),
                )
            return override
        return int(auto) if auto is not None else _DEFAULT_SCORE_ONLY_MAX_HYPOTHESES_PER_MICROBATCH
    override = _optional_positive_int_env(_MAX_HYPOTHESES_ENV)
    auto = _auto_hypotheses_per_microbatch(
        score_only=False,
        fused_k_class=fused_k_class,
        fused_k_class_count=fused_k_class_count,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory_bytes,
        score_complex_dtype=score_complex_dtype,
    )
    if override is not None:
        if auto is not None and int(override) < int(auto):
            logger.warning(
                "%s=%d is below the auto sparse pass-2 cap %d; "
                "this can fragment buckets and slow pass-2.",
                _MAX_HYPOTHESES_ENV,
                int(override),
                int(auto),
            )
        return override
    return int(auto) if auto is not None else _DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH


def _max_translation_tile_bytes_for_pass(
    device_memory_bytes: int | None = None,
    *,
    has_external_normalization: bool = False,
    fused_k_class: bool = False,
) -> int:
    override = _optional_positive_int_env(_MAX_TRANSLATION_TILE_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_MAX_TRANSLATION_TILE_BYTES
    if fused_k_class:
        fraction = _AUTO_FUSED_KCLASS_TRANSLATION_TILE_DEVICE_FRACTION
    elif has_external_normalization:
        fraction = _AUTO_EXTERNAL_NORMALIZATION_TRANSLATION_TILE_DEVICE_FRACTION
    else:
        fraction = _AUTO_TRANSLATION_TILE_DEVICE_FRACTION
    return max(1, int(float(device_memory_bytes) * fraction))


def _max_projection_gather_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_MAX_PROJECTION_GATHER_BYTES_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_GATHER_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTION_GATHER_DEVICE_FRACTION))


def _max_noise_block_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_MAX_NOISE_BLOCK_BYTES_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None:
        return _DEFAULT_NOISE_BLOCK_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_NOISE_BLOCK_DEVICE_FRACTION))


def _max_adjoint_block_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_MAX_ADJOINT_BLOCK_BYTES_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None:
        return _DEFAULT_ADJOINT_BLOCK_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_ADJOINT_BLOCK_DEVICE_FRACTION))


def _compact_pair_dense_mstep_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_COMPACT_PAIR_DENSE_MSTEP_MAX_BYTES_ENV)
    if override is not None:
        return int(override)
    return _max_adjoint_block_bytes_for_pass(device_memory_bytes)


def _projection_cache_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = parse_env_nonnegative_int(_PROJECTION_CACHE_MAX_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_CACHE_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTION_CACHE_DEVICE_FRACTION))


def _projection_call_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = parse_env_nonnegative_int(_PROJECTION_CACHE_MAX_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_CACHE_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTED_ROTATIONS_DEVICE_FRACTION))


def _max_images_for_translation_tile(
    image_shape,
    n_fine_trans,
    *,
    max_tile_bytes=384 * 1024**2,
    complex_dtype=np.complex64,
    n_half_pixels: int | None = None,
):
    """Limit one translated-image tile allocation to a bounded size."""
    half_image_size = (
        max(1, int(n_half_pixels))
        if n_half_pixels is not None
        else int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    )
    bytes_per_complex_value = _dtype_itemsize(complex_dtype)
    bytes_per_image = int(n_fine_trans) * half_image_size * bytes_per_complex_value
    return max(1, int(max_tile_bytes) // max(1, bytes_per_image))


def _projection_cache_transient_bytes(
    n_rotations: int,
    n_half_pixels: int,
    *,
    projection_complex_dtype=np.complex64,
    include_abs2: bool,
) -> int:
    complex_bytes = _dtype_itemsize(projection_complex_dtype)
    total = int(n_rotations) * int(n_half_pixels) * complex_bytes
    if include_abs2:
        real_dtype = _complex_counterpart_real_dtype(projection_complex_dtype)
        total += int(n_rotations) * int(n_half_pixels) * _dtype_itemsize(real_dtype)
    return int(total)


def _projection_cache_budget_complex_dtype(
    projection_source_dtype,
    score_complex_dtype,
    *,
    use_relion_projector: bool = False,
):
    dtype = np.promote_types(np.dtype(projection_source_dtype), np.dtype(score_complex_dtype))
    if use_relion_projector:
        # RELION Projector parity uses float64 interpolation weights, which
        # promotes complex64 projector data to complex128 before the caller's
        # output cast. Budget the transient allocation, not the retained cache.
        dtype = np.promote_types(dtype, np.dtype(np.complex128))
    return dtype


def _projection_cache_fits_budget(transient_bytes: int, max_bytes: int, *, n_classes: int = 1) -> bool:
    return int(transient_bytes) * max(1, int(n_classes)) <= int(max_bytes)


def _max_projected_rotations_per_call_for_pass(
    *,
    device_memory_bytes: int | None,
    n_projection_pixels: int,
    projection_complex_dtype,
    include_abs2: bool,
) -> int | None:
    override = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None or int(n_projection_pixels) <= 0:
        return None
    max_bytes = _projection_call_max_bytes_for_pass(device_memory_bytes)
    bytes_per_rotation = _projection_cache_transient_bytes(
        1,
        int(n_projection_pixels),
        projection_complex_dtype=projection_complex_dtype,
        include_abs2=bool(include_abs2),
    )
    if max_bytes <= 0 or bytes_per_rotation <= 0:
        return None
    return max(1, int(max_bytes) // int(bytes_per_rotation))


_PROJECTION_CACHE_BUILD_ROTATION_MULTIPLIER = 4


def _projection_cache_build_max_rotations_per_call(
    per_call_max_rotations: int | None,
    n_fine_rotations: int,
) -> int | None:
    """Rotations per projection call while building the fine-projection cache.

    The per-chunk budget ``per_call_max_rotations`` is sized for the scoring
    loop, where per-image translation tiles, score blocks and Wavg operands are
    live next to the projection intermediates.  The cache build runs before
    any of those exist, so it may project several times more rotations per
    call: at HEALPix order 3 (294912 fine rotations, current_size 92) the
    build took 19.9 s per half at 809 rotations per call and 3.6 s at 4096
    (jobs 14051847 / 14052595), while the scoring-loop budget kept the peak
    memory profile unchanged.  An explicit
    ``RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS`` still applies verbatim.
    """

    if per_call_max_rotations is None:
        return None
    override = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)
    if override is not None:
        return int(override)
    scaled = int(per_call_max_rotations) * _PROJECTION_CACHE_BUILD_ROTATION_MULTIPLIER
    return max(1, min(int(n_fine_rotations), scaled))


def _projection_budget_pixels_for_pass(
    n_half_pixels: int,
    *,
    use_window: bool,
    use_relion_projector: bool,
) -> int:
    """Effective projection pixels for the sparse pass-2 projection cap.

    Windowed sparse pass-2 only keeps score/reconstruction rows after
    projection, but RELION's centered Projector handoff currently materializes
    full-half intermediates before gathering the requested windows. Budget that
    path with extra headroom for the centered-row scatter, dense scaling, and
    other live pass-2 buffers so huge one-image compact-pair buckets still split
    before the projection helper allocates.
    """

    pixels = int(n_half_pixels)
    if bool(use_window) and bool(use_relion_projector):
        return max(1, 8 * pixels)
    return max(1, pixels)


def _kclass_raw_diff2_bytes(class_bucket_arrays, compact_pair_arrays, *, n_fine_trans, dtype):
    """Bytes of all raw class scores, using the actual padded bucket shapes."""
    if compact_pair_arrays is not None:
        elements = sum(int(arrays["pair_mask"].size) for arrays in compact_pair_arrays)
    else:
        elements = sum(
            int(arrays["rotations"].shape[0]) * int(arrays["rotations"].shape[1]) * int(n_fine_trans)
            for arrays in class_bucket_arrays
        )
    return elements * np.dtype(dtype).itemsize
