"""Exact per-image local EM engine for RELION-mode local search."""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as _sparse_pass2_diagnostics
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_maybe_windowed as _adjoint_slice_volume_maybe_windowed,
)
from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.fourier_window import (
    centered_half_indices_to_fftw_half_indices,
    make_fourier_window_spec,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    bin_shell_values_jax,
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
    make_scoring_half_image_weights,
    make_shell_indices_half,
    mask_relion_noise_shell_indices_to_current_window,
)
from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    enforce_half_volume_x0,
    half_volume_accumulator_shape,
    half_volume_accumulators_to_full,
    relion_backprojector_volume_shape,
    relion_x_half_mstep_accumulator_dtypes,
    relion_x_half_accumulators_to_public_layout,
)
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    integer_pre_shifts_or_none,
    tiled_half_image_phase_factors,
)
from recovar.em.dense_single_volume.helpers.jax_runtime import block_until_ready as _block_until_ready
from recovar.em.dense_single_volume.helpers.preprocessing import (
    _cast_shift_inputs,
    apply_half_translation_phases as _apply_half_translation_phases,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    half_translation_phase_table as _half_translation_phase_table,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    _norm_inputs,
    process_half_image,
    resolve_image_mask_for_half_preprocess,
)
from recovar.em.dense_single_volume.local_timing import (  # noqa: F401
    _LOCAL_ACCOUNTED_TIMING_FIELDS,
    _LOCAL_ACCOUNTED_TIMING_SETUP_FIELDS,
    _LOCAL_PREPROCESS_TIMER_KEYS,
    _LOCAL_TIMING_PROFILE_FIELDS,
    _LOCAL_TRANSFER_TIMER_KEYS,
    _LocalTiming,
    _local_timing_profile,
    _new_local_preprocess_timer,
    _new_local_transfer_timer,
    _new_zero_timer,
    _prefixed_timer_profile,
)
from recovar.em.dense_single_volume.local_caches import (  # noqa: F401
    EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB,
    EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV,
    EXACT_LOCAL_RAW_CACHE_MAX_GB,
    EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV,
    EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB,
    EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV,
    _LocalProcessedHalfCache,
    _all_integer_pre_shifts_or_none,
    _build_local_processed_half_cache,
    _build_local_raw_cache,
    _local_processed_half_cache_enabled,
    _local_raw_cache_enabled,
    _sparse_big_jit_mstep_tensors_memory_gb,
    _validate_native_half_batch,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_noise_block as _compute_noise_block,
    compute_norm_residual_per_image as _compute_norm_residual_per_image,
    compute_scale_correction_terms_per_image as _compute_scale_correction_terms_per_image,
    relion_scale_correction_pixel_mask as _relion_scale_correction_pixel_mask,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_projections_block as _compute_projections_block,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
)
from recovar.em.dense_single_volume.helpers.projection import (
    indexed_projection_available as _indexed_projection_available,
)
from recovar.em.dense_single_volume.helpers.projection import (
    project_indexed_half_spectrum as _project_indexed_half_spectrum,
)
from recovar.em.dense_single_volume.helpers.timing import TimingAccumulator
from recovar.em.dense_single_volume.helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from recovar.em.dense_single_volume.helpers.types import make_noise_stats, make_relion_stats
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.dense_single_volume.local_big_jit import (
    _norm_correction_image_power_per_image,
    _preprocess_half as _big_jit_preprocess_half,
    run_local_bucket_big_jit,
)
from recovar.em.dense_single_volume.local_debug import (
    current_size_matches_request,
    iteration_matches_request,
    maybe_write_debug_fused_posterior_dump,
    maybe_write_debug_noise_component_dump,
    maybe_write_debug_score_dump,
    noise_split_diagnostics_requested,
    parse_debug_fused_posterior_dump_request,
    parse_debug_noise_component_dump_request,
    parse_debug_score_dump_request,
)
from recovar.em.dense_single_volume.local_layout import (
    LocalBucketSpec,
    LocalHypothesisLayout,
    _exact_bucket_rotation_size,
    _exact_local_large_bucket_quantum,
    bucket_local_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_score_pass import (
    compute_reconstruction_support,
    compute_reconstruction_support_from_threshold,
    fused_score_normalize_mstep_abs2_on_demand,
    fused_score_normalize_mstep_abs2_with_log_z_on_demand,
    fused_score_normalize_support_abs2_on_demand,
    fused_score_normalize_support_probs_abs2_with_log_z_on_demand,
    fused_score_normalize_support_probs_abs2_on_demand,
    normalize_local_scores,
    normalize_local_scores_float32,
    normalize_local_scores_with_log_z,
    normalize_local_scores_with_log_z_float32,
    score_local_bucket_abs2_on_demand,
    score_local_bucket_abs2_weighted_on_demand,
)
from recovar.em.dense_single_volume.shape_buckets import pad_axis, pad_batch_data_ctf_and_valid_mask
from recovar.reconstruction import noise as noise_utils
from recovar.utils.nvtx_shim import nvtx

logger = logging.getLogger(__name__)


def _maybe_dump_exact_local_bpref_contribution_rows(**kwargs) -> None:
    """Write exact-local pre-scatter rows without claiming device geometry.

    The reusable contribution schema already describes the posterior-reduced
    BPref operands needed for canonical replay.  Exact-local search reaches the
    same boundary through a different engine, so forward its materialized
    bucket there when explicitly requested.  Device-produced neighbor
    signatures remain unsupported on this route and must continue to fail
    before execution rather than silently emitting an incomplete capture.
    """

    if not os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", "").strip():
        return
    if os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip():
        raise RuntimeError(
            "Exact-local BPref contribution capture does not yet support device signatures"
        )
    _sparse_pass2_diagnostics._maybe_dump_bpref_contribution_rows(**kwargs)


def _exact_local_bpref_contribution_capture_active(
    *, current_size: int | None, debug_iteration: int | None
) -> bool:
    """Return whether this exact-local half is the explicitly targeted boundary."""

    if not os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", "").strip():
        return False
    context = _sparse_pass2_diagnostics._bpref_contribution_context
    context_iteration = int(context["iteration"])
    context_half = int(context["half"])
    target_iteration = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION", "").strip()
    target_half = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF", "").strip()
    target_current_size = os.environ.get(
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE", ""
    ).strip()
    if not (target_iteration and target_half and target_current_size):
        return False
    if context_iteration != int(target_iteration):
        return False
    if context_half != int(target_half):
        return False
    if current_size is None or int(current_size) != int(target_current_size):
        return False
    if debug_iteration is not None and context_iteration != int(debug_iteration):
        return False
    return context_iteration > 0 and context_half in {1, 2}


def _exact_local_bpref_contribution_capture_for_call(
    *,
    current_size: int | None,
    debug_iteration: int | None,
    score_only: bool,
    mstep_relion_x_half: bool,
) -> bool:
    """Activate capture only at a compatible fine-pass M-step boundary."""

    requested = _exact_local_bpref_contribution_capture_active(
        current_size=current_size,
        debug_iteration=debug_iteration,
    )
    if not requested or score_only:
        return False
    if not mstep_relion_x_half:
        raise RuntimeError(
            "Exact-local BPref contribution capture requires RELION x-half M-step geometry"
        )
    return True


NVTX_DOMAIN_EM = "recovar_em"

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
# Keep the deferred exact-local noise projection chunks small enough for
# low-image-count/high-candidate outlier cases that otherwise fragment H100
# memory. This cap is total projected row-pixels across the active image batch,
# i.e. about a 512 MB complex64 projection temporary before JAX overhead.
EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS = 64_000_000
EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS_ENV = "RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS"
EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM = 512
EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM_ENV = "RECOVAR_EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM"
EXACT_LOCAL_DEFER_PACKED_MSTEP_ENV = "RECOVAR_EXACT_LOCAL_DEFER_PACKED_MSTEP"
EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP_ENV = "RECOVAR_EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP"
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB = 0.0
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB"
EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS = 64_000_000
EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV = (
    "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS"
)
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS = 64
EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV = "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS"
LOCAL_SCORE_DUMP_FORCE_SPLIT_ENV = "RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT"
LOCAL_SCORE_DUMP_OPERANDS_ENV = "RECOVAR_LOCAL_SCORE_DUMP_OPERANDS"
LOCAL_SCORE_DUMP_TARGET_ONLY_ENV = "RECOVAR_LOCAL_SCORE_DUMP_TARGET_ONLY"
EXACT_LOCAL_SPARSE_ADJOINT_TARGET_ROWS_ENV = "RECOVAR_EXACT_LOCAL_SPARSE_ADJOINT_TARGET_ROWS"
EXACT_LOCAL_PROGRESS_CHUNKS_ENV = "RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS"
EXACT_LOCAL_PROGRESS_SECONDS_ENV = "RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS"
DEFAULT_EXACT_LOCAL_PROGRESS_CHUNKS = 1000
DEFAULT_EXACT_LOCAL_PROGRESS_SECONDS = 300
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_VISIBLE_GPU_MEMORY_BYTES_CACHE: int | None = None
# Disabled by default: on the 50k/256 local-search target this cache made the
# iteration slower by precomputing more spectra than the bucket schedule reuses.
# Upper bound for the extra M-step tensors materialized by the sparse big-JIT
# hybrid path. This path still packs rows before backprojection; the cap only
# guards the temporary fused summed/ctf tensor outputs.
EXACT_LOCAL_BIG_JIT_MIN_SIGNIFICANT_ROW_FRACTION = 0.25


def _local_mstep_rotations(bucket: LocalBucketSpec) -> np.ndarray:
    """Return the adjoint-only rotations, falling back to scoring rotations."""

    rotations = bucket.local_mstep_rotations
    if rotations is None:
        rotations = bucket.local_rotations
    return np.asarray(rotations, dtype=np.float32)


def _bucket_contains_debug_target(experiment_dataset, image_indices, pending_targets: set[int] | None) -> bool:
    if not pending_targets:
        return False
    original_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(image_indices),
        dtype=np.int64,
    )
    return any(int(original_idx) in pending_targets for original_idx in original_indices.tolist())


def _filter_buckets_to_debug_targets(
    experiment_dataset,
    bucket_specs: list[LocalBucketSpec],
    pending_targets: set[int],
) -> list[LocalBucketSpec]:
    if not pending_targets:
        return bucket_specs
    return [
        bucket
        for bucket in bucket_specs
        if _bucket_contains_debug_target(
            experiment_dataset,
            bucket.image_indices,
            pending_targets,
        )
    ]


@dataclass
class _LocalPostprocessBuffers:
    hard_assignment: np.ndarray
    log_evidence_per_image: np.ndarray
    best_log_score_per_image: np.ndarray
    max_posterior_per_image: np.ndarray
    rotation_posterior_sums: np.ndarray
    transfer_profile: dict[str, float]
    chunk_nonzero_posterior_rows: list[int]
    chunk_significant_samples: list[int]
    chunk_reconstruction_rows: list[int]
    seen_global_rotations: np.ndarray
    seen_nonzero_global_rotations: np.ndarray
    seen_reconstruction_global_rotations: np.ndarray
    significant_counts: np.ndarray | None = None
    best_pose_rotations: np.ndarray | None = None
    best_pose_translations: np.ndarray | None = None
    best_pose_rotation_ids: np.ndarray | None = None
    reconstruction_sample_indices_by_image: list[np.ndarray] | None = None


@dataclass
class _LocalProjectionBlock:
    proj_weighted: jnp.ndarray
    proj_for_noise: jnp.ndarray | None


@dataclass
class _LocalRelionProjectionCache:
    projections: jnp.ndarray
    id_map: jnp.ndarray
    enabled: bool
    row_count: int = 0
    id_map_row_count: int = 0
    n_projection_pixels: int = 0
    estimated_gb: float = 0.0
    build_s: float = 0.0


def _local_mstep_adjoint_window(
    image_shape,
    n_half: int,
    current_size: int | None,
    *,
    use_window: bool,
    recon_window_indices,
    mstep_relion_x_half: bool,
):
    """Return coordinate indices/max radius for exact-local M-step adjoints."""

    mstep_recon_window_indices = recon_window_indices
    if mstep_relion_x_half:
        if mstep_recon_window_indices is None:
            mstep_recon_window_indices = jnp.arange(int(n_half), dtype=jnp.int32)
        mstep_recon_window_indices = centered_half_indices_to_fftw_half_indices(
            image_shape,
            mstep_recon_window_indices,
        )
    mstep_adjoint_max_r = None
    if use_window or mstep_relion_x_half:
        mstep_current_size = int(current_size) if current_size is not None else int(image_shape[0])
        mstep_adjoint_max_r = float(mstep_current_size // 2)
    return mstep_recon_window_indices, mstep_adjoint_max_r


def _packed_noise_projection_chunk_rows(n_recon_pixels: int, *, batch_size: int = 1) -> int:
    """Return packed local noise-projection rows per chunk."""

    target = int(EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS)
    raw = os.environ.get(EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS_ENV, "").strip()
    if raw:
        try:
            target = max(1, int(raw))
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using default %d row-pixels",
                EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS_ENV,
                raw,
                EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS,
            )
            target = int(EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS)
    row_pixels = max(1, int(n_recon_pixels)) * max(1, int(batch_size))
    return max(1, int(target) // row_pixels)


def _reconstruction_pack_large_bucket_quantum() -> int:
    raw = os.environ.get(EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM_ENV, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using default %d rows",
                EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM_ENV,
                raw,
                EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM,
            )
    return int(EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM)


def _optional_nonnegative_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _optional_nonnegative_float_env(name: str, default: float) -> float:
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


def _exact_local_relion_projection_cache_chunk_rows(n_projection_pixels: int) -> int:
    target = int(os.environ.get(
        EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV,
        EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS,
    ))
    if target <= 0:
        raise ValueError(f"{EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV} must be positive")
    return max(1, int(target) // max(1, int(n_projection_pixels)))


def _disabled_relion_projection_cache() -> _LocalRelionProjectionCache:
    return _LocalRelionProjectionCache(
        projections=jnp.zeros((1, 1), dtype=jnp.complex64),
        id_map=jnp.zeros((1,), dtype=jnp.int32),
        enabled=False,
    )


def _exact_local_relion_projection_cache_capacity_rows(n_projection_pixels: int) -> tuple[int, float]:
    max_gb = _optional_nonnegative_float_env(
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


def _sort_buckets_for_relion_projection_cache(bucket_specs: list[LocalBucketSpec]) -> list[LocalBucketSpec]:
    return sorted(
        bucket_specs,
        key=lambda bucket: (
            int(bucket.bucket_rotation_count),
            _bucket_rotation_id_center(bucket),
            int(bucket.image_indices[0]) if int(bucket.image_indices.shape[0]) else -1,
        ),
    )


def _exact_local_relion_projection_cache_max_groups() -> int:
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


def _plan_exact_local_relion_projection_cache_groups(
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


def _build_exact_local_relion_projection_cache_for_buckets(
    bucket_specs: list[LocalBucketSpec],
    relion_projector_half,
    *,
    image_shape,
    n_projection_pixels: int,
    relion_projector_r_max: int,
    projection_padding_factor: int,
    projection_relion_texture_interp: bool | None,
    projection_pixel_indices,
    projector_output_size: int,
    cache_row_capacity: int,
    max_global_rotation_id: int,
    group_index: int,
    n_groups: int,
    projection_mask_current_image_disk: bool = True,
) -> _LocalRelionProjectionCache:
    """Precompute compact RELION projections for one bounded bucket group."""

    if cache_row_capacity <= 0 or not bucket_specs:
        return _disabled_relion_projection_cache()

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
        return _disabled_relion_projection_cache()

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

    chunk_rows = _exact_local_relion_projection_cache_chunk_rows(n_projection_pixels)
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
    return _LocalRelionProjectionCache(
        projections=projections,
        id_map=id_map_jnp,
        enabled=True,
        row_count=row_count,
        id_map_row_count=id_map_row_count,
        n_projection_pixels=n_projection_pixels,
        estimated_gb=estimated_gb,
        build_s=build_s,
    )


def _adjoint_slice_volume_maybe_windowed_row_chunks(
    half_rows,
    window_indices,
    rotations,
    volume,
    image_shape,
    volume_shape,
    disc_type,
    *,
    use_window: bool,
    max_r,
    relion_x_half: bool,
    target_rows: int,
):
    """Apply an exact-local sparse adjoint in row chunks when requested."""

    n_rows = int(half_rows.shape[0])
    target_rows = int(target_rows)
    if target_rows <= 0 or n_rows <= target_rows:
        return (
            _adjoint_slice_volume_maybe_windowed(
                half_rows,
                window_indices,
                rotations,
                volume,
                image_shape,
                volume_shape,
                disc_type,
                True,
                True,
                use_window=use_window,
                max_r=max_r,
                relion_x_half=relion_x_half,
            ),
            1,
        )

    updated = volume
    n_chunks = 0
    for start in range(0, n_rows, target_rows):
        stop = min(n_rows, start + target_rows)
        updated = _adjoint_slice_volume_maybe_windowed(
            half_rows[start:stop],
            window_indices,
            rotations[start:stop],
            updated,
            image_shape,
            volume_shape,
            disc_type,
            True,
            True,
            use_window=use_window,
            max_r=max_r,
            relion_x_half=relion_x_half,
        )
        n_chunks += 1
    return updated, n_chunks


def _local_em_return_tuple(
    Ft_y,
    Ft_ctf,
    hard_assignment,
    relion_stats,
    *,
    accumulate_noise: bool,
    return_profile: bool,
    return_best_pose_details: bool,
    return_significant_counts: bool = False,
    best_pose_rotations=None,
    best_pose_translations=None,
    best_pose_rotation_ids=None,
    noise_stats=None,
    profile_summary=None,
    significant_counts=None,
):
    result = [Ft_y, Ft_ctf, hard_assignment]
    if return_best_pose_details:
        result.extend(
            [
                best_pose_rotations,
                best_pose_translations,
                best_pose_rotation_ids,
            ]
        )
    result.append(relion_stats)
    if accumulate_noise:
        result.append(noise_stats)
    if return_profile:
        result.append(profile_summary)
    if return_significant_counts:
        result.append(significant_counts)
    return tuple(result)


def _project_local_bucket(
    *,
    mean_for_proj,
    bucket: LocalBucketSpec,
    image_shape,
    proj_volume_shape,
    disc_type: str,
    projection_kwargs: dict,
    window_spec,
    n_half: int,
    half_weights,
    precision_policy: DensePrecisionPolicy,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    materialize_recon_projection: bool = True,
) -> _LocalProjectionBlock:
    """Project a local bucket and return score/noise projection views."""

    batch_size = int(bucket.image_indices.shape[0])
    bucket_rotation_count = int(bucket.bucket_rotation_count)
    # Do not retry per-bucket projection dedupe here unless the real 5k
    # duplicate factor changes materially. It regressed a measured local run
    # from ~76.7s to ~126.9s when duplicate factor was only ~1.004-1.005.
    flat_rotations = flatten_bucket_rotations(jnp.asarray(bucket.local_rotations))
    if relion_projector_half is not None:
        if relion_projector_r_max is None:
            raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
        relion_projector_half = jnp.asarray(relion_projector_half)
        if relion_projector_half.ndim == 4:
            if int(relion_projector_half.shape[0]) != 1:
                raise ValueError(
                    "local RELION projector path expected a single-class projector slab, "
                    f"got {relion_projector_half.shape}",
                )
            relion_projector_half = relion_projector_half[0]
        if relion_projector_half.ndim != 3:
            raise ValueError(
                "local RELION projector path expected Projector::data shape (z, y, x_half), "
                f"got {relion_projector_half.shape}",
            )
        relion_texture_interp = projection_kwargs.get("relion_texture_interp")
        mask_current_image_disk = bool(projection_kwargs.get("mask_current_image_disk", True))
        projector_kwargs = {}
        if window_spec.use_window and window_spec.max_r is not None:
            projector_kwargs["projector_output_size"] = int(2 * window_spec.max_r)
        projection_indices = None
        if window_spec.use_window:
            projection_indices = (
                window_spec.projection_indices if materialize_recon_projection else window_spec.score_indices
            )
        if projection_indices is not None:
            projector_kwargs["pixel_indices"] = projection_indices
        if not mask_current_image_disk:
            projector_kwargs["mask_current_image_disk"] = False
        proj_relion_flat, _ = _compute_relion_projector_projections_block(
            relion_projector_half,
            flat_rotations,
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            relion_texture_interp=relion_texture_interp,
            **projector_kwargs,
        )
        if window_spec.use_window:
            if materialize_recon_projection:
                proj_half = proj_relion_flat[..., window_spec.score_projection_take].reshape(
                    batch_size,
                    bucket_rotation_count,
                    window_spec.n_score,
                )
                proj_for_noise = proj_relion_flat[..., window_spec.recon_projection_take].reshape(
                    batch_size,
                    bucket_rotation_count,
                    window_spec.n_recon,
                )
            else:
                proj_half = proj_relion_flat.reshape(batch_size, bucket_rotation_count, window_spec.n_score)
                proj_for_noise = None
            score_half_weights = window_spec.score_values(half_weights)
            proj_weighted = proj_half * score_half_weights[None, None, :]
            proj_weighted, proj_for_noise, _, _ = precision_policy.cast_local_projection_scores(
                proj_weighted,
                proj_for_noise,
                None,
                None,
            )
            return _LocalProjectionBlock(proj_weighted=proj_weighted, proj_for_noise=proj_for_noise)
        proj_half_flat = proj_relion_flat
    elif (
        window_spec.use_window
        and not bool(projection_kwargs.get("relion_texture_interp", False))
        and not bool(projection_kwargs.get("force_jax", False))
        and _indexed_projection_available()
    ):
        projection_indices = (
            window_spec.projection_indices if materialize_recon_projection else window_spec.score_indices
        )
        proj_window_flat = _project_indexed_half_spectrum(
            mean_for_proj,
            projection_indices,
            flat_rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            max_r=projection_kwargs.get("max_r"),
        )
        if materialize_recon_projection:
            proj_half = proj_window_flat[..., window_spec.score_projection_take].reshape(
                batch_size,
                bucket_rotation_count,
                window_spec.n_score,
            )
            proj_for_noise = proj_window_flat[..., window_spec.recon_projection_take].reshape(
                batch_size,
                bucket_rotation_count,
                window_spec.n_recon,
            )
        else:
            proj_half = proj_window_flat.reshape(batch_size, bucket_rotation_count, window_spec.n_score)
            proj_for_noise = None
        score_half_weights = window_spec.score_values(half_weights)
        proj_weighted = proj_half * score_half_weights[None, None, :]
        proj_weighted, proj_for_noise, _, _ = precision_policy.cast_local_projection_scores(
            proj_weighted,
            proj_for_noise,
            None,
            None,
        )
        return _LocalProjectionBlock(proj_weighted=proj_weighted, proj_for_noise=proj_for_noise)
    else:
        ordinary_projection_kwargs = dict(projection_kwargs)
        ordinary_projection_kwargs.pop("mask_current_image_disk", None)
        proj_half_flat, _ = _compute_projections_block(
            mean_for_proj,
            flat_rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            return_abs2=False,
            **ordinary_projection_kwargs,
        )

    if window_spec.use_window:
        proj_half = window_spec.score_values(proj_half_flat).reshape(
            batch_size,
            bucket_rotation_count,
            window_spec.n_score,
        )
        proj_for_noise = (
            window_spec.recon_values(proj_half_flat).reshape(
                batch_size,
                bucket_rotation_count,
                window_spec.n_recon,
            )
            if materialize_recon_projection
            else None
        )
        score_half_weights = window_spec.score_values(half_weights)
    else:
        proj_half = proj_half_flat.reshape(batch_size, bucket_rotation_count, n_half)
        proj_for_noise = proj_half if materialize_recon_projection else None
        score_half_weights = half_weights

    proj_weighted = proj_half * score_half_weights[None, None, :]
    proj_weighted, proj_for_noise, _, _ = precision_policy.cast_local_projection_scores(
        proj_weighted,
        proj_for_noise,
        None,
        None,
    )
    return _LocalProjectionBlock(proj_weighted=proj_weighted, proj_for_noise=proj_for_noise)


def _project_packed_noise_rows(
    *,
    mean_for_proj,
    packed_flat_rotations,
    packed_rotation_count: int,
    batch_size: int,
    image_shape,
    proj_volume_shape,
    disc_type: str,
    projection_kwargs: dict,
    window_spec,
    n_half: int,
    precision_policy: DensePrecisionPolicy,
    reconstruction_pack_mask_jnp,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
) -> jnp.ndarray:
    """Project only packed reconstruction rows for local noise accumulation."""

    if relion_projector_half is not None:
        if relion_projector_r_max is None:
            raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
        relion_projector_half = jnp.asarray(relion_projector_half)
        if relion_projector_half.ndim == 4:
            if int(relion_projector_half.shape[0]) != 1:
                raise ValueError(
                    "local RELION projector path expected a single-class projector slab, "
                    f"got {relion_projector_half.shape}",
                )
            relion_projector_half = relion_projector_half[0]
        if relion_projector_half.ndim != 3:
            raise ValueError(
                "local RELION projector path expected Projector::data shape (z, y, x_half), "
                f"got {relion_projector_half.shape}",
            )
        relion_texture_interp = projection_kwargs.get("relion_texture_interp")
        mask_current_image_disk = bool(projection_kwargs.get("mask_current_image_disk", True))
        projector_kwargs = {}
        if window_spec.use_window and window_spec.max_r is not None:
            projector_kwargs["projector_output_size"] = int(2 * window_spec.max_r)
        projection_indices = None
        if window_spec.use_window:
            projection_indices = (
                window_spec.recon_indices if window_spec.recon_indices is not None else window_spec.score_indices
            )
        if projection_indices is not None:
            projector_kwargs["pixel_indices"] = projection_indices
        if not mask_current_image_disk:
            projector_kwargs["mask_current_image_disk"] = False
        proj_relion_flat, _ = _compute_relion_projector_projections_block(
            relion_projector_half,
            packed_flat_rotations,
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            relion_texture_interp=relion_texture_interp,
            **projector_kwargs,
        )
        flat_proj_for_noise = proj_relion_flat
    elif (
        window_spec.use_window
        and not bool(projection_kwargs.get("relion_texture_interp", False))
        and not bool(projection_kwargs.get("force_jax", False))
        and _indexed_projection_available()
    ):
        flat_proj_for_noise = _project_indexed_half_spectrum(
            mean_for_proj,
            window_spec.recon_or_full_indices(n_half),
            packed_flat_rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            max_r=projection_kwargs.get("max_r"),
        )
    else:
        ordinary_projection_kwargs = dict(projection_kwargs)
        ordinary_projection_kwargs.pop("mask_current_image_disk", None)
        proj_half_flat, _ = _compute_projections_block(
            mean_for_proj,
            packed_flat_rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            return_abs2=False,
            **ordinary_projection_kwargs,
        )
        flat_proj_for_noise = window_spec.recon_values(proj_half_flat) if window_spec.use_window else proj_half_flat

    packed_proj_for_noise = flat_proj_for_noise.reshape(
        int(batch_size),
        int(packed_rotation_count),
        window_spec.n_recon if window_spec.use_window else int(n_half),
    )
    packed_proj_for_noise = jnp.where(
        reconstruction_pack_mask_jnp[:, :, None],
        packed_proj_for_noise,
        0.0,
    )
    packed_proj_for_noise, _ = precision_policy.cast_local_noise_projection_scores(
        packed_proj_for_noise,
        None,
    )
    return packed_proj_for_noise


def _local_projection_mode(window_spec, projection_kwargs: dict, relion_projector_half=None) -> str:
    if relion_projector_half is not None:
        return "relion_projector"
    if not window_spec.use_window:
        return "full"
    if bool(projection_kwargs.get("force_jax", False)):
        return "windowed_full_jax"
    if bool(projection_kwargs.get("relion_texture_interp", False)):
        return "windowed_full_texture"
    if not _indexed_projection_available():
        return "windowed_full_cuda_unavailable"
    return "windowed_indexed_cuda"


def _postprocess_local_bucket(
    *,
    image_indices,
    local_rotation_ids,
    local_rotation_mask,
    local_rotations,
    local_rotation_posterior_ids,
    translation_grid,
    n_trans,
    best_argmax,
    batch_norm,
    log_Z,
    best_log_score,
    max_posterior,
    probs_sum_t,
    n_significant_samples,
    reconstruction_sample_mask,
    collect_profile_stats: bool,
    reconstruction_row_count: int,
    reconstruction_take_indices,
    reconstruction_pack_mask,
    buffers: _LocalPostprocessBuffers,
):
    """Scatter one local bucket's host-side pose, posterior, and profile stats."""

    image_indices_np = np.asarray(image_indices, dtype=np.int32)
    local_rotation_ids_np = np.asarray(local_rotation_ids, dtype=np.int32)
    local_mask_np = np.asarray(local_rotation_mask, dtype=bool)

    transfer_t0 = time.time()
    best_rot_idx = np.asarray(best_argmax // n_trans, dtype=np.int32)
    best_trans_idx = np.asarray(best_argmax % n_trans, dtype=np.int32)
    buffers.transfer_profile["postprocess_argmax_to_host_s"] += time.time() - transfer_t0

    best_rotation_ids = np.take_along_axis(
        local_rotation_ids_np,
        best_rot_idx[:, None],
        axis=1,
    ).reshape(-1)
    if np.any(best_rotation_ids < 0):
        raise RuntimeError("exact local engine selected padded local rotation")
    buffers.hard_assignment[image_indices_np] = (best_rotation_ids * n_trans + best_trans_idx).astype(np.int32)

    transfer_t0 = time.time()
    log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
    log_z_np = np.asarray(log_Z, dtype=np.float32)
    best_log_score_np = np.asarray(best_log_score, dtype=np.float32)
    max_posterior_np = np.asarray(max_posterior, dtype=np.float32)
    buffers.transfer_profile["postprocess_scores_to_host_s"] += time.time() - transfer_t0
    buffers.log_evidence_per_image[image_indices_np] = log_z_np + log_score_offset.astype(np.float32)
    buffers.best_log_score_per_image[image_indices_np] = best_log_score_np + log_score_offset.astype(np.float32)
    buffers.max_posterior_per_image[image_indices_np] = max_posterior_np

    transfer_t0 = time.time()
    probs_sum_t_np = np.asarray(probs_sum_t, dtype=np.float64)
    collect_significant_counts = collect_profile_stats or buffers.significant_counts is not None
    n_significant_samples_np = (
        np.asarray(n_significant_samples, dtype=np.int32) if collect_significant_counts else None
    )
    buffers.transfer_profile["postprocess_posterior_to_host_s"] += time.time() - transfer_t0

    if buffers.significant_counts is not None:
        buffers.significant_counts[image_indices_np] = n_significant_samples_np

    posterior_ids_np = (
        local_rotation_ids_np
        if local_rotation_posterior_ids is None
        else np.asarray(local_rotation_posterior_ids, dtype=np.int32)
    )
    np.add.at(buffers.rotation_posterior_sums, posterior_ids_np[local_mask_np], probs_sum_t_np[local_mask_np])

    significant_sample_count = 0
    if collect_profile_stats:
        nonzero_mask = (probs_sum_t_np > 0.0) & local_mask_np
        significant_sample_count = int(np.sum(n_significant_samples_np, dtype=np.int64))
        buffers.chunk_nonzero_posterior_rows.append(int(np.count_nonzero(nonzero_mask)))
        buffers.chunk_significant_samples.append(significant_sample_count)
        buffers.chunk_reconstruction_rows.append(int(reconstruction_row_count))

    if buffers.seen_global_rotations.size:
        nonzero_mask = (probs_sum_t_np > 0.0) & local_mask_np
        buffers.seen_global_rotations[posterior_ids_np[local_mask_np]] = True
        buffers.seen_nonzero_global_rotations[posterior_ids_np[nonzero_mask]] = True
        packed_posterior_ids_np = np.take_along_axis(posterior_ids_np, reconstruction_take_indices, axis=1)
        buffers.seen_reconstruction_global_rotations[packed_posterior_ids_np[reconstruction_pack_mask]] = True

    if buffers.reconstruction_sample_indices_by_image is not None:
        if reconstruction_sample_mask is None:
            raise RuntimeError("reconstruction_sample_mask is required when collecting local significant samples")
        sample_mask_np = np.asarray(reconstruction_sample_mask, dtype=bool)
        for row, image_idx in enumerate(image_indices_np):
            valid_sample_mask = sample_mask_np[row] & local_mask_np[row, :, None]
            rot_rows, trans_cols = np.nonzero(valid_sample_mask)
            sample_ids = posterior_ids_np[row, rot_rows].astype(np.int64) * int(n_trans) + trans_cols.astype(np.int64)
            if np.any(sample_ids < 0):
                raise RuntimeError("local significant-sample collection encountered padded rotation ids")
            buffers.reconstruction_sample_indices_by_image[int(image_idx)] = sample_ids.astype(np.int64, copy=False)

    if buffers.best_pose_rotations is not None:
        buffers.best_pose_rotations[image_indices_np] = np.take_along_axis(
            np.asarray(local_rotations, dtype=np.float32),
            best_rot_idx[:, None, None, None],
            axis=1,
        ).reshape(-1, 3, 3)
        buffers.best_pose_translations[image_indices_np] = np.asarray(translation_grid, dtype=np.float32)[
            best_trans_idx
        ]
        buffers.best_pose_rotation_ids[image_indices_np] = best_rotation_ids.astype(np.int32, copy=False)

    return significant_sample_count, int(reconstruction_row_count)


def _pad_local_big_jit_image_axis(bucket: LocalBucketSpec, batch_data, ctf_params):
    """Pad a local big-JIT bucket to its planned image shape class."""

    actual_batch_size = int(bucket.image_indices.shape[0])
    padded_batch_size = int(max(actual_batch_size, getattr(bucket, "bucket_image_count", actual_batch_size)))
    if actual_batch_size == padded_batch_size:
        return bucket, batch_data, ctf_params, np.ones(actual_batch_size, dtype=bool), actual_batch_size

    padded_rotations = pad_axis(bucket.local_rotations, 0, padded_batch_size, value=0).astype(np.float32)
    padded_rotations[actual_batch_size:] = np.eye(3, dtype=np.float32)
    padded_mstep_rotations = pad_axis(
        _local_mstep_rotations(bucket),
        0,
        padded_batch_size,
        value=0,
    ).astype(np.float32)
    padded_mstep_rotations[actual_batch_size:] = np.eye(3, dtype=np.float32)
    padded_bucket = LocalBucketSpec(
        image_indices=np.asarray(bucket.image_indices, dtype=np.int32),
        bucket_image_count=padded_batch_size,
        bucket_rotation_count=int(bucket.bucket_rotation_count),
        actual_rotation_counts=pad_axis(bucket.actual_rotation_counts, 0, padded_batch_size, value=0).astype(np.int32),
        local_rotation_ids=pad_axis(bucket.local_rotation_ids, 0, padded_batch_size, value=-1).astype(np.int32),
        local_rotations=padded_rotations,
        local_mstep_rotations=padded_mstep_rotations,
        local_rotation_log_prior=pad_axis(
            bucket.local_rotation_log_prior,
            0,
            padded_batch_size,
            value=-1e30,
        ).astype(np.float32),
        local_rotation_mask=pad_axis(bucket.local_rotation_mask, 0, padded_batch_size, value=False).astype(bool),
        translation_log_prior=pad_axis(bucket.translation_log_prior, 0, padded_batch_size, value=0).astype(np.float32),
        local_rotation_posterior_ids=(
            None
            if bucket.local_rotation_posterior_ids is None
            else pad_axis(bucket.local_rotation_posterior_ids, 0, padded_batch_size, value=-1).astype(np.int32)
        ),
        local_sample_mask=(
            None
            if bucket.local_sample_mask is None
            else pad_axis(bucket.local_sample_mask, 0, padded_batch_size, value=False).astype(bool)
        ),
    )
    padded_batch_data, padded_ctf_params, valid_image_mask, _, _ = pad_batch_data_ctf_and_valid_mask(
        batch_data,
        ctf_params,
        padded_batch_size,
    )
    return padded_bucket, padded_batch_data, padded_ctf_params, valid_image_mask, padded_batch_size


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
    )
    boost_factor = _exact_local_auto_microbatch_boost() if auto_boost_factor is None else float(auto_boost_factor)
    if boost_factor <= 0.0 or not np.isfinite(boost_factor):
        raise ValueError("auto_boost_factor must be positive and finite")
    boost_cap = int(np.floor(cap * boost_factor))
    effective_cap = int(max(cap, min(65536, planned_floor, boost_cap)))
    if not score_only:
        return effective_cap

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
        )
        for count in rotation_counts
    )
    projection_row_cap = max(1, target_row_pixels // n_projection_pixels)
    # Exact neighborhoods are indivisible; permit at least one image from the
    # largest padded bucket even when that exceeds the configured row target.
    safe_cap = max(int(max_bucket_rotation_count), int(projection_row_cap))
    return min(cap, safe_cap)


def _reorder_bucket_to_indices(bucket: LocalBucketSpec, returned_indices: np.ndarray) -> LocalBucketSpec:
    if np.array_equal(returned_indices, bucket.image_indices):
        return bucket
    position = {int(idx): pos for pos, idx in enumerate(np.asarray(bucket.image_indices).tolist())}
    order = np.asarray([position[int(idx)] for idx in np.asarray(returned_indices).tolist()], dtype=np.int32)
    return LocalBucketSpec(
        image_indices=np.asarray(returned_indices, dtype=np.int32),
        bucket_image_count=int(bucket.bucket_image_count),
        bucket_rotation_count=int(bucket.bucket_rotation_count),
        actual_rotation_counts=np.asarray(bucket.actual_rotation_counts[order], dtype=np.int32),
        local_rotation_ids=np.asarray(bucket.local_rotation_ids[order], dtype=np.int32),
        local_rotations=np.asarray(bucket.local_rotations[order], dtype=np.float32),
        local_mstep_rotations=np.asarray(_local_mstep_rotations(bucket)[order], dtype=np.float32),
        local_rotation_log_prior=np.asarray(bucket.local_rotation_log_prior[order], dtype=np.float32),
        local_rotation_mask=np.asarray(bucket.local_rotation_mask[order], dtype=bool),
        translation_log_prior=np.asarray(bucket.translation_log_prior[order], dtype=np.float32),
        local_rotation_posterior_ids=(
            None
            if bucket.local_rotation_posterior_ids is None
            else np.asarray(bucket.local_rotation_posterior_ids[order], dtype=np.int32)
        ),
        local_sample_mask=(
            None if bucket.local_sample_mask is None else np.asarray(bucket.local_sample_mask[order], dtype=bool)
        ),
    )


def _prepare_local_exact_bucket(
    experiment_dataset,
    batch,
    ctf_params,
    image_indices,
    noise_variance_half,
    translation_phases_half,
    config,
    norm_half_weights,
    score_with_masked_images: bool,
    relion_score_translation_angles=None,
    image_pre_shifts=None,
    processed_half_cache: _LocalProcessedHalfCache | None = None,
    timer: dict[str, float] | None = None,
    synchronize_profile: bool = False,
    score_complex_dtype=None,
    score_real_dtype=None,
    norm_real_dtype=None,
    relion_exact_bpref_operands: bool = False,
):
    """Prepare score, reconstruction, and noise inputs for one local bucket.

    This keeps the exact-local path separate from the dense engine and avoids
    recomputing CTF / translation tiling scaffolding across masked, unmasked,
    and noise-specific preprocessing.
    """

    integer_t0 = time.time()
    real_space_pre_shift_applied = False
    if processed_half_cache is None or not processed_half_cache.integer_pre_shifts_applied:
        integer_pre_shifts = integer_pre_shifts_or_none(image_pre_shifts, image_indices, batch=batch)
        if integer_pre_shifts is not None:
            batch = apply_relion_integer_pre_shifts(batch, integer_pre_shifts)
            real_space_pre_shift_applied = True
    else:
        integer_pre_shifts = None
        real_space_pre_shift_applied = True
    if timer is not None:
        timer["integer_shift_s"] += time.time() - integer_t0

    phase_t0 = time.time()
    translation_phases_half = jnp.asarray(translation_phases_half)
    raw_translations = translation_phases_half.shape[-1] == len(config.image_shape)
    if raw_translations:
        # Backward compatibility for tests and direct callers that pass raw
        # translations instead of the precomputed phase table used by the hot path.
        translation_phases_half = _half_translation_phase_table(
            translation_phases_half,
            config.image_shape,
        )
    if raw_translations and synchronize_profile:
        _block_until_ready(translation_phases_half)
    if raw_translations and timer is not None:
        timer["translation_phase_s"] += time.time() - phase_t0

    exact_image_mask = None
    exact_image_mask_mode = None
    if relion_exact_bpref_operands:
        exact_image_mask, exact_image_mask_mode = resolve_image_mask_for_half_preprocess(
            experiment_dataset,
            config.image_shape,
            require_mask=score_with_masked_images,
        )

    def _process_half(apply_image_mask: bool):
        if relion_exact_bpref_operands:
            # Contribution capture forces the split scorer, but its operands
            # must still be those of the production big-JIT path.  In
            # particular, do not fall back through the image-source backend:
            # a relion_cuda source requires separate normalization/shift
            # operands and would introduce a different preprocessing graph.
            return _big_jit_preprocess_half(
                jnp.asarray(batch),
                jnp.asarray(exact_image_mask),
                config,
                apply_image_mask=apply_image_mask,
                mask_mode=exact_image_mask_mode,
            )
        return process_half_image(
            experiment_dataset,
            batch,
            apply_image_mask,
        )

    ctf_t0 = time.time()
    if relion_exact_bpref_operands:
        ctf_rfloat = np.asarray(
            _sparse_pass2_diagnostics._relion_exact_ctf_half_from_source_star(
                experiment_dataset,
                image_indices,
                config.image_shape,
            ),
            dtype=np.float64,
        )
        inverse_noise_rfloat_cast = np.reciprocal(
            np.asarray(noise_variance_half, dtype=np.float64)
        ).astype(np.float32)
        ctf_half = jnp.asarray(ctf_rfloat, dtype=jnp.float64).astype(jnp.float32)
        inverse_noise_half = jnp.asarray(inverse_noise_rfloat_cast, dtype=jnp.float32)
        weighted_ctf_half = ctf_half * inverse_noise_half[None, :]
        ctf2_over_nv_recon_half = weighted_ctf_half * ctf_half
        ctf2_over_nv_score_half = jnp.asarray(
            (
                inverse_noise_rfloat_cast[None, :].astype(np.float64)
                * ctf_rfloat
                * ctf_rfloat
            ).astype(np.float32),
            dtype=jnp.float32,
        )
    else:
        ctf_half = config.compute_ctf_half(ctf_params)
        ctf2_over_nv_ctf = ctf_half.astype(score_real_dtype) if score_real_dtype is not None else ctf_half
        ctf2_over_nv_noise = (
            noise_variance_half.astype(score_real_dtype) if score_real_dtype is not None else noise_variance_half
        )
        ctf2_over_nv_recon_half = ctf2_over_nv_ctf**2 / ctf2_over_nv_noise
        ctf2_over_nv_score_half = ctf2_over_nv_recon_half
        weighted_ctf_half = None
    if synchronize_profile:
        _block_until_ready(ctf2_over_nv_score_half, ctf2_over_nv_recon_half)
    if timer is not None:
        timer["ctf_s"] += time.time() - ctf_t0

    if processed_half_cache is None:
        score_process_t0 = time.time()
        processed_score_half = _process_half(score_with_masked_images)
        if synchronize_profile:
            _block_until_ready(processed_score_half)
        if timer is not None:
            timer["score_process_s"] += time.time() - score_process_t0
    else:
        cache_fetch_t0 = time.time()
        processed_score_half = jnp.asarray(processed_half_cache.score_half[np.asarray(image_indices, dtype=np.int32)])
        if timer is not None:
            timer["cache_fetch_s"] += time.time() - cache_fetch_t0

    shift_score_t0 = time.time()
    shift_processed_score_half, shift_ctf_half, shift_noise_half, shift_phases_half = _cast_shift_inputs(
        processed_score_half,
        ctf_half,
        noise_variance_half,
        translation_phases_half,
        score_complex_dtype=score_complex_dtype,
        score_real_dtype=score_real_dtype,
    )
    score_weighted_half = (
        shift_processed_score_half * weighted_ctf_half
        if relion_exact_bpref_operands
        else shift_processed_score_half * shift_ctf_half / shift_noise_half
    )
    if relion_score_translation_angles is not None:
        from recovar import cuda_backproject

        shifted_score_half = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(score_weighted_half, dtype=jnp.complex64),
            jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
            jnp.arange(score_weighted_half.shape[1], dtype=jnp.int32),
            config.image_shape,
        )
    else:
        shifted_score_half = _apply_half_translation_phases(score_weighted_half, shift_phases_half)
    if synchronize_profile:
        _block_until_ready(shifted_score_half)
    if timer is not None:
        timer["tile_shift_score_s"] += time.time() - shift_score_t0

    norm_t0 = time.time()
    norm_processed_score_half, norm_noise_half, norm_weights = _norm_inputs(
        processed_score_half,
        noise_variance_half,
        norm_half_weights,
        norm_real_dtype=norm_real_dtype,
    )
    norm_power_over_noise = (
        jnp.abs(norm_processed_score_half) ** 2
        * inverse_noise_half.astype(norm_processed_score_half.real.dtype)[None, :]
        if relion_exact_bpref_operands
        else jnp.abs(norm_processed_score_half) ** 2 / norm_noise_half
    )
    batch_norm = jnp.sum(
        norm_power_over_noise * norm_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real
    if synchronize_profile:
        _block_until_ready(batch_norm)
    if timer is not None:
        timer["norm_s"] += time.time() - norm_t0

    if score_with_masked_images:
        if processed_half_cache is None:
            recon_process_t0 = time.time()
            processed_recon_half = _process_half(False)
            if synchronize_profile:
                _block_until_ready(processed_recon_half)
            if timer is not None:
                timer["recon_process_s"] += time.time() - recon_process_t0
        else:
            cache_fetch_t0 = time.time()
            if processed_half_cache.recon_half is None:
                raise RuntimeError("processed half-image cache is missing unmasked reconstruction images")
            processed_recon_half = jnp.asarray(
                processed_half_cache.recon_half[np.asarray(image_indices, dtype=np.int32)]
            )
            if timer is not None:
                timer["cache_fetch_s"] += time.time() - cache_fetch_t0

        shift_recon_t0 = time.time()
        shift_processed_recon_half, shift_ctf_half, shift_noise_half, shift_phases_half = _cast_shift_inputs(
            processed_recon_half,
            ctf_half,
            noise_variance_half,
            translation_phases_half,
            score_complex_dtype=score_complex_dtype,
            score_real_dtype=score_real_dtype,
        )
        recon_weighted_half = (
            shift_processed_recon_half * weighted_ctf_half
            if relion_exact_bpref_operands
            else shift_processed_recon_half * shift_ctf_half / shift_noise_half
        )
        shifted_recon_half = _apply_half_translation_phases(recon_weighted_half, shift_phases_half)
        if synchronize_profile:
            _block_until_ready(shifted_recon_half)
        if timer is not None:
            timer["tile_shift_recon_s"] += time.time() - shift_recon_t0
    else:
        if relion_score_translation_angles is None:
            shifted_recon_half = shifted_score_half
        else:
            # Exact RELION arithmetic applies only to score translation.
            # Reconstruction retains the ordinary JAX phase table.
            shifted_recon_half = _apply_half_translation_phases(
                score_weighted_half,
                shift_phases_half,
            )
    return (
        shifted_score_half,
        shifted_recon_half,
        batch_norm,
        ctf2_over_nv_score_half,
        ctf2_over_nv_recon_half,
        processed_score_half,
        real_space_pre_shift_applied,
    )


def _build_reconstruction_pack_indices(
    significant_rotation_mask: np.ndarray,
    local_rotation_mask: np.ndarray,
    rotation_block_size: int,
):
    """Pack RELION-style reconstruction rows into a smaller padded bucket."""

    significant_rotation_mask = np.asarray(significant_rotation_mask, dtype=bool)
    local_rotation_mask = np.asarray(local_rotation_mask, dtype=bool)
    pack_mask = significant_rotation_mask & local_rotation_mask
    actual_counts = np.sum(pack_mask, axis=1, dtype=np.int32)
    max_count = int(np.max(actual_counts, initial=0))
    if max_count <= 0:
        max_count = 1
    packed_rotation_count = _exact_bucket_rotation_size(
        max_count,
        rotation_block_size,
        large_bucket_quantum=_reconstruction_pack_large_bucket_quantum(),
    )
    batch_size = int(pack_mask.shape[0])
    take_indices = np.zeros((batch_size, packed_rotation_count), dtype=np.int32)
    padded_pack_mask = np.zeros((batch_size, packed_rotation_count), dtype=bool)
    for row in range(batch_size):
        selected = np.flatnonzero(pack_mask[row])
        count = int(selected.shape[0])
        if count:
            take_indices[row, :count] = selected
            padded_pack_mask[row, :count] = True
    return take_indices, padded_pack_mask, actual_counts, int(np.sum(actual_counts, dtype=np.int64))


def _build_nonzero_reconstruction_pack_indices(
    significant_rotation_mask: np.ndarray,
    local_rotation_mask: np.ndarray,
    probs_sum_t_np: np.ndarray,
    rotation_block_size: int,
):
    """Pack rows that can make a nonzero M-step contribution.

    RELION os0 reconstruction semantics keep all local candidates, but rows
    whose summed posterior over translations is exactly zero contribute zeros to
    Ft_y, Ft_ctf, and noise. Dropping only those rows keeps the math unchanged
    while avoiding millions of no-op backprojection/noise rows.
    """

    nonzero_rotation_mask = np.asarray(probs_sum_t_np) > 0.0
    return _build_reconstruction_pack_indices(
        np.asarray(significant_rotation_mask, dtype=bool) & nonzero_rotation_mask,
        local_rotation_mask,
        rotation_block_size,
    )


@nvtx.annotate("local.run_local_em_exact", color="purple", domain=NVTX_DOMAIN_EM)
def run_local_em_exact(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    disc_type: str,
    *,
    image_batch_size: int,
    rotation_block_size: int,
    current_size: int | None,
    reconstruction_current_size: int | None = None,
    accumulate_noise: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
    score_with_masked_images: bool = True,
    half_spectrum_scoring: bool = False,
    relion_exact_score_translation: bool = False,
    use_float64_scoring: bool = False,
    use_float64_normalization: bool = True,
    use_float64_projections: bool = False,
    projection_relion_texture_interp: bool = False,
    projection_force_jax: bool = False,
    projection_mask_current_image_disk: bool = True,
    relion_exact_bpref_operands: bool = False,
    relion_exact_fine_diff2: bool = False,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    image_corrections: np.ndarray | None = None,
    scale_corrections: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    scale_correction_group_count: int | None = None,
    scale_correction_data_vs_prior: np.ndarray | None = None,
    image_pre_shifts: np.ndarray | None = None,
    mstep_subtract_ctf_projection: bool = False,
    mstep_relion_x_half: bool = False,
    host_accumulator_finalize: bool = False,
    return_half_volume_accumulators: bool = False,
    return_profile: bool = False,
    disable_adjoint_y: bool = False,
    disable_adjoint_ctf: bool = False,
    max_hypotheses_per_microbatch: int | None = None,
    reconstruct_significant_only: bool = False,
    adaptive_fraction: float = 0.999,
    max_significants: int = -1,
    debug_iteration: int | None = None,
    return_best_pose_details: bool = False,
    normalization_log_z: np.ndarray | None = None,
    class_log_prior: float = 0.0,
    normalization_log_evidence: np.ndarray | None = None,
    translation_prior_centers: np.ndarray | None = None,
    unify_local_bucket_sizes: bool | None = None,
    stats_use_reconstruction_probs: bool = False,
    relion_f32_fine_posterior: bool = False,
    include_unweighted_norm_high_shell: bool = True,
    reconstruction_probability_threshold: np.ndarray | None = None,
    return_reconstruction_probability_values: bool = False,
    return_reconstruction_sample_indices: bool = False,
    return_significant_counts: bool = False,
    score_only: bool = False,
):
    """Run exact local EM over per-image local hypothesis sets."""

    score_only = bool(score_only)
    use_relion_f32_fine_posterior = bool(
        relion_f32_fine_posterior
        and mstep_relion_x_half
        and reconstruct_significant_only
        and not score_only
    )
    if relion_f32_fine_posterior and not use_relion_f32_fine_posterior:
        raise ValueError(
            "RELION float32 fine posterior requires a significant-only "
            "RELION x-half M-step"
        )
    include_unweighted_norm_high_shell = bool(include_unweighted_norm_high_shell)
    relion_exact_score_translation = bool(relion_exact_score_translation)
    relion_exact_bpref_operands = bool(relion_exact_bpref_operands)
    relion_exact_fine_diff2 = bool(relion_exact_fine_diff2)
    if relion_exact_score_translation and not half_spectrum_scoring:
        raise ValueError("exact RELION score translation requires half_spectrum_scoring=True")
    if relion_exact_score_translation and use_float64_scoring:
        raise ValueError("exact RELION score translation is a float32 scoring path")
    if relion_exact_fine_diff2 and not relion_exact_bpref_operands:
        raise ValueError("exact RELION fine diff2 requires exact BPref operands")
    if score_only:
        if not (disable_adjoint_y and disable_adjoint_ctf):
            raise ValueError("score_only exact-local EM requires both adjoints disabled")
        if accumulate_noise:
            raise ValueError("score_only exact-local EM does not support noise accumulation")
        if mstep_subtract_ctf_projection:
            raise ValueError("score_only exact-local EM does not support residual M-step subtraction")
        if return_half_volume_accumulators:
            raise ValueError("score_only exact-local EM does not return half-volume accumulators")

    return_profile = bool(
        return_profile or return_reconstruction_probability_values or return_reconstruction_sample_indices
    )
    overall_t0 = time.time()
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    mstep_current_size = (
        current_size
        if reconstruction_current_size is None
        else int(reconstruction_current_size)
    )
    n_half = H * (W // 2 + 1)
    n_trans = int(local_layout.translation_grid.shape[0])
    n_images = int(local_layout.n_images)
    class_log_prior = float(class_log_prior)
    group_ids_np = None
    n_scale_groups = 0
    explicit_scale_group_count = 0
    if scale_correction_group_count is not None:
        explicit_scale_group_count = int(scale_correction_group_count)
        if (
            explicit_scale_group_count < 0
            or not np.isfinite(float(scale_correction_group_count))
            or float(scale_correction_group_count) != float(explicit_scale_group_count)
        ):
            raise ValueError(
                "scale_correction_group_count must be a non-negative integer, "
                f"got {scale_correction_group_count!r}"
            )
    if group_ids is not None:
        group_ids_np = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if group_ids_np.shape != (n_images,):
            raise ValueError(f"group_ids must have shape ({n_images},), got {group_ids_np.shape}")
        if group_ids_np.size and int(np.min(group_ids_np)) < 0:
            raise ValueError("group_ids must be non-negative")
        inferred_scale_group_count = int(np.max(group_ids_np)) + 1 if group_ids_np.size else 1
        n_scale_groups = max(explicit_scale_group_count, inferred_scale_group_count)
    normalization_log_z_np = None
    if normalization_log_z is not None:
        normalization_log_z_np = np.asarray(normalization_log_z, dtype=np.float64)
        if normalization_log_z_np.shape != (n_images,):
            raise ValueError(
                f"normalization_log_z must have shape ({n_images},), got {normalization_log_z_np.shape}",
            )
    normalization_log_evidence_np = None
    if normalization_log_evidence is not None:
        normalization_log_evidence_np = np.asarray(normalization_log_evidence, dtype=np.float64)
        if normalization_log_evidence_np.shape != (n_images,):
            raise ValueError(
                f"normalization_log_evidence must have shape ({n_images},), got {normalization_log_evidence_np.shape}",
            )
    if normalization_log_z_np is not None and normalization_log_evidence_np is not None:
        raise ValueError("Provide only one of normalization_log_z or normalization_log_evidence")
    reconstruction_probability_threshold_np = None
    if reconstruction_probability_threshold is not None:
        reconstruction_probability_threshold_np = np.asarray(reconstruction_probability_threshold, dtype=np.float64)
        if reconstruction_probability_threshold_np.shape != (n_images,):
            raise ValueError(
                "reconstruction_probability_threshold must have shape "
                f"({n_images},), got {reconstruction_probability_threshold_np.shape}",
            )
        if not np.all(np.isfinite(reconstruction_probability_threshold_np)):
            raise ValueError("reconstruction_probability_threshold must be finite")
        if np.any(reconstruction_probability_threshold_np < 0.0):
            raise ValueError("reconstruction_probability_threshold must be non-negative")
    translation_prior_centers_np = validate_translation_prior_centers(
        translation_prior_centers,
        n_images=n_images,
        n_dims=local_layout.translation_grid.shape[1],
    )
    (
        debug_score_dump_dir,
        debug_score_dump_targets,
        debug_score_dump_current_sizes,
        debug_score_dump_iterations,
    ) = parse_debug_score_dump_request()
    (
        debug_fused_posterior_dump_dir,
        debug_fused_posterior_dump_targets,
        debug_fused_posterior_dump_current_sizes,
        debug_fused_posterior_dump_iterations,
    ) = parse_debug_fused_posterior_dump_request()
    (
        debug_noise_dump_dir,
        debug_noise_dump_targets,
        debug_noise_dump_current_sizes,
        debug_noise_dump_iterations,
    ) = parse_debug_noise_component_dump_request()
    debug_score_dump_filter_matches = (
        debug_score_dump_dir is not None
        and current_size_matches_request(debug_score_dump_current_sizes, current_size)
        and iteration_matches_request(debug_score_dump_iterations, debug_iteration)
    )
    debug_fused_posterior_dump_filter_matches = (
        debug_fused_posterior_dump_dir is not None
        and current_size_matches_request(debug_fused_posterior_dump_current_sizes, current_size)
        and iteration_matches_request(debug_fused_posterior_dump_iterations, debug_iteration)
    )
    debug_noise_dump_filter_matches = (
        debug_noise_dump_dir is not None
        and current_size_matches_request(debug_noise_dump_current_sizes, current_size)
        and iteration_matches_request(debug_noise_dump_iterations, debug_iteration)
    )
    bpref_contribution_capture_active = _exact_local_bpref_contribution_capture_for_call(
        current_size=current_size,
        debug_iteration=debug_iteration,
        score_only=score_only,
        mstep_relion_x_half=mstep_relion_x_half,
    )
    debug_score_dump_operands = bool(
        debug_score_dump_filter_matches
        and _env_flag(LOCAL_SCORE_DUMP_OPERANDS_ENV)
    )
    debug_score_dump_force_split = bool(
        debug_score_dump_filter_matches
        and _env_flag(LOCAL_SCORE_DUMP_FORCE_SPLIT_ENV)
    )
    debug_score_dump_big_jit = bool(debug_score_dump_filter_matches and not debug_score_dump_force_split)
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    if return_half_volume_accumulators and mstep_relion_x_half:
        raise ValueError("return_half_volume_accumulators only supports native half-volume accumulators")

    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            mean,
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=mstep_current_size,
        )
    else:
        mean_for_proj = mean
        proj_volume_shape = volume_shape

    precision_policy = DensePrecisionPolicy(
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
        use_float64_normalization=use_float64_normalization,
    )
    mean_for_proj = precision_policy.cast_projection_volume(mean_for_proj)

    if mstep_relion_x_half:
        # RELION BPref::initZeros(current_size) sizes the accumulator from the
        # iteration r_max.  The reconstruction boundary then crops the output
        # back to ``volume_shape``.
        recon_volume_shape = relion_backprojector_volume_shape(
            volume_shape,
            reconstruction_padding_factor,
            current_size=mstep_current_size,
        )
    elif reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
    else:
        recon_volume_shape = volume_shape
    if score_only:
        logger.info("Exact local score-only: M-step accumulators disabled")
    elif mstep_relion_x_half:
        logger.info(
            "Exact local M-step: using RELION x-half BPref-layout backprojection shape=%s",
            recon_volume_shape,
        )
    else:
        logger.info("Exact local M-step: using native half-volume backprojection")
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape)
    recon_volume_size = int(np.prod(recon_accum_shape))
    score_only_accumulator_size = 1 if score_only else recon_volume_size

    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        reconstruction_current_size=mstep_current_size,
        square=square_window,
        include_recon_window=True,
    )
    use_window = window_spec.use_window
    window_indices = window_spec.score_indices
    if relion_exact_fine_diff2:
        if not use_window:
            raise ValueError("exact RELION fine diff2 currently requires a current-size score window")
        relion_fine_full_to_compact = (
            _sparse_pass2_diagnostics._relion_cuda_fine_full_to_compact_lookup(
                image_shape,
                int(current_size),
                window_spec.score_indices_np,
            )
        )
    else:
        relion_fine_full_to_compact = np.zeros(1, dtype=np.int32)
    recon_window_indices = window_spec.recon_indices
    mstep_recon_window_indices, mstep_adjoint_max_r = _local_mstep_adjoint_window(
        image_shape,
        n_half,
        mstep_current_size,
        use_window=use_window,
        recon_window_indices=recon_window_indices,
        mstep_relion_x_half=bool(mstep_relion_x_half),
    )
    n_windowed = window_spec.n_score
    projection_kwargs = window_spec.projection_kwargs()
    projection_kwargs["relion_texture_interp"] = projection_relion_texture_interp
    projection_kwargs["force_jax"] = bool(projection_force_jax)
    projection_kwargs["mask_current_image_disk"] = bool(projection_mask_current_image_disk)
    projection_mode = _local_projection_mode(window_spec, projection_kwargs, relion_projector_half)

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )
    norm_half_weights = make_half_image_weights(image_shape)
    half_weights_windowed = window_spec.score_values(half_weights)
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()

    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        experiment_dataset.dtype,
        use_relion_x_half_mstep=bool(mstep_relion_x_half),
    )
    Ft_y = jnp.zeros(score_only_accumulator_size, dtype=recon_y_accum_dtype)
    Ft_ctf = jnp.zeros(score_only_accumulator_size, dtype=recon_ctf_accum_dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    log_evidence_per_image = np.empty(n_images, dtype=np.float32)
    best_log_score_per_image = np.empty(n_images, dtype=np.float32)
    max_posterior_per_image = np.empty(n_images, dtype=np.float32)
    significant_counts = np.empty(n_images, dtype=np.int32) if return_significant_counts else None
    rotation_posterior_sums = np.zeros(int(local_layout.n_global_rotations), dtype=np.float64)
    best_pose_rotations = np.empty((n_images, 3, 3), dtype=np.float32) if return_best_pose_details else None
    best_pose_translations = (
        np.empty((n_images, local_layout.translation_grid.shape[1]), dtype=np.float32)
        if return_best_pose_details
        else None
    )
    best_pose_rotation_ids = np.empty(n_images, dtype=np.int32) if return_best_pose_details else None

    noise_wsum = None
    noise_img_power = None
    noise_norm_correction = None
    noise_a2 = None
    noise_xa = None
    noise_scale_xa = None
    noise_scale_aa = None
    noise_sigma2_offset = jnp.asarray(0.0, dtype=jnp.float32)
    noise_sumw = jnp.asarray(0.0, dtype=jnp.float32)
    return_noise_split = noise_split_diagnostics_requested()
    require_materialized_recon_projection = bool(
        mstep_subtract_ctf_projection
        or debug_noise_dump_filter_matches
        or return_noise_split
    )
    can_defer_local_noise_projection = (
        relion_projector_half is None
        and not bool(projection_kwargs.get("relion_texture_interp", False))
        and not bool(projection_kwargs.get("force_jax", False))
        and _indexed_projection_available()
    )
    defer_local_noise_projection = (
        not require_materialized_recon_projection
        and window_spec.use_window
        and can_defer_local_noise_projection
    )
    need_local_recon_projection = require_materialized_recon_projection or (
        accumulate_noise and not defer_local_noise_projection
    )
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        if use_window:
            shell_indices_half = mask_relion_noise_shell_indices_to_current_window(
                shell_indices_half,
                image_shape,
                current_size,
                window_indices,
            )
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        norm_unweighted_shell_cutoff = image_shape[0] // 2 if current_size is None else int(current_size // 2)
        noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
        scale_correction_pixel_mask = _relion_scale_correction_pixel_mask(
            scale_correction_data_vs_prior,
            shell_indices_noise,
            n_shells=n_shells,
        )
        noise_wsum = jnp.zeros(n_shells, dtype=jnp.float32)
        noise_img_power = jnp.zeros(n_shells, dtype=jnp.float32)
        noise_norm_correction = jnp.zeros(n_images, dtype=jnp.float32)
        noise_a2 = jnp.zeros(n_shells, dtype=jnp.float32)
        noise_xa = jnp.zeros(n_shells, dtype=jnp.float32)
        if group_ids_np is not None:
            noise_scale_xa = jnp.zeros(n_scale_groups, dtype=jnp.float32)
            noise_scale_aa = jnp.zeros(n_scale_groups, dtype=jnp.float32)

    default_fused_score_mstep = max_significants is None or int(max_significants) <= 0
    fused_score_mstep_enabled = default_fused_score_mstep
    timing = _LocalTiming()
    raw_cache_enabled = False
    processed_half_cache_enabled = False
    preprocess_profile = _new_local_preprocess_timer()
    transfer_profile = _new_local_transfer_timer()
    big_jit_bucket_count = 0
    sparse_big_jit_bucket_count = 0
    big_jit_debug_bucket_count = 0
    sparse_adjoint_chunk_count = 0
    sparse_adjoint_target_rows = _optional_nonnegative_int_env(EXACT_LOCAL_SPARSE_ADJOINT_TARGET_ROWS_ENV) or 0
    total_local_rotations = int(local_layout.total_local_rotations)
    logged_deferred_mstep_chunking = False
    logged_deferred_noise_projection_chunking = False
    logged_cached_noise_projection_chunking = False
    logged_sparse_big_jit_deferred_fallback = False
    collect_profile_stats = bool(return_profile or reconstruct_significant_only)
    seen_global_rotations = (
        np.zeros(rotation_posterior_sums.shape[0], dtype=bool)
        if return_profile and rotation_posterior_sums.size
        else np.zeros(0, dtype=bool)
    )
    seen_nonzero_global_rotations = np.zeros_like(seen_global_rotations)
    seen_reconstruction_global_rotations = np.zeros_like(seen_global_rotations)
    total_padded_rotations = 0
    chunk_sizes = []
    chunk_local_rotations = []
    chunk_padded_rotations = []
    chunk_unique_rotations = []
    chunk_nonzero_posterior_rows = []
    chunk_reconstruction_rows = []
    chunk_significant_samples = []
    n_chunks = 0
    local_total_hypotheses = 0
    total_significant_samples = 0
    total_reconstruction_rows = 0
    reconstruction_sample_indices_by_image = (
        [np.zeros(0, dtype=np.int64) for _ in range(n_images)] if return_reconstruction_sample_indices else None
    )
    postprocess_buffers = _LocalPostprocessBuffers(
        hard_assignment=hard_assignment,
        log_evidence_per_image=log_evidence_per_image,
        best_log_score_per_image=best_log_score_per_image,
        max_posterior_per_image=max_posterior_per_image,
        rotation_posterior_sums=rotation_posterior_sums,
        transfer_profile=transfer_profile,
        chunk_nonzero_posterior_rows=chunk_nonzero_posterior_rows,
        chunk_significant_samples=chunk_significant_samples,
        chunk_reconstruction_rows=chunk_reconstruction_rows,
        seen_global_rotations=seen_global_rotations,
        seen_nonzero_global_rotations=seen_nonzero_global_rotations,
        seen_reconstruction_global_rotations=seen_reconstruction_global_rotations,
        significant_counts=significant_counts,
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
        reconstruction_sample_indices_by_image=reconstruction_sample_indices_by_image,
    )
    reconstruction_probability_values_by_image = (
        [[] for _ in range(n_images)] if return_reconstruction_probability_values else None
    )

    def _collect_reconstruction_probability_values(image_indices, posterior_probs):
        """Collect unpruned positive posterior values for global support thresholding."""

        if reconstruction_probability_values_by_image is None:
            return
        image_indices_np = np.asarray(image_indices, dtype=np.int32)
        probs_np = np.asarray(posterior_probs, dtype=np.float32).reshape(len(image_indices_np), -1)
        for row, image_index in enumerate(image_indices_np):
            values = probs_np[row]
            values = values[values > 0.0]
            if values.size:
                reconstruction_probability_values_by_image[int(image_index)].append(values.copy())

    # The cap model already accounts for the active score/reconstruction
    # windows and the x-half M-step row footprint, but RELION projector x-half
    # buckets at 256 OOMed at both 2x and 1.25x in c180 probes. Keep the default
    # conservative and allow explicit experiments through the x-half env knob.
    allow_microbatch_auto_boost = True
    xhalf_bpref_mstep = bool(relion_projector_half is not None and mstep_relion_x_half and not score_only)
    xhalf_auto_microbatch_boost = _exact_local_xhalf_auto_microbatch_boost() if xhalf_bpref_mstep else None
    xhalf_full_bpref_mstep = bool(
        xhalf_bpref_mstep and int(recon_volume_shape[0]) >= (2 * int(image_shape[0]) + 1)
    )
    if xhalf_bpref_mstep and max_hypotheses_per_microbatch is None and not _exact_local_microbatch_env_overridden():
        bpreftype = "full-BPref" if xhalf_full_bpref_mstep else "current-size BPref"
        logger.info(
            "Exact local RELION x-half %s M-step: using conservative microbatch cap "
            "(image_shape=%s, recon_volume_shape=%s)",
            bpreftype,
            tuple(int(x) for x in image_shape),
            tuple(int(x) for x in recon_volume_shape),
        )
    max_hypotheses_per_microbatch = _exact_local_effective_max_hypotheses_per_microbatch(
        max_hypotheses_per_microbatch,
        n_windowed,
        n_trans=n_trans,
        n_recon_windowed=window_spec.n_recon,
        local_layout=local_layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        allow_auto_boost=allow_microbatch_auto_boost,
        auto_boost_factor=xhalf_auto_microbatch_boost,
        allow_high_memory_default=not xhalf_bpref_mstep,
        score_only=score_only,
    )
    if xhalf_bpref_mstep:
        uncapped_hypotheses_per_microbatch = int(max_hypotheses_per_microbatch)
        max_hypotheses_per_microbatch = _exact_local_xhalf_tail_microbatch_cap(
            uncapped_hypotheses_per_microbatch,
            local_layout,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
        )
        if max_hypotheses_per_microbatch < uncapped_hypotheses_per_microbatch:
            logger.info(
                "Exact local RELION x-half tail microbatch cap: %d -> %d "
                "(max_local_rotations=%d, planned_image_batch=%d, planned_rotation_block=%d)",
                uncapped_hypotheses_per_microbatch,
                int(max_hypotheses_per_microbatch),
                int(np.max(np.asarray(local_layout.rotation_counts), initial=0)),
                int(image_batch_size),
                int(rotation_block_size),
            )
        tail_capped_hypotheses_per_microbatch = int(max_hypotheses_per_microbatch)
        max_hypotheses_per_microbatch = _exact_local_xhalf_projection_microbatch_cap(
            tail_capped_hypotheses_per_microbatch,
            local_layout,
            n_projection_pixels=int(window_spec.n_projection),
            rotation_block_size=rotation_block_size,
        )
        if max_hypotheses_per_microbatch < tail_capped_hypotheses_per_microbatch:
            logger.info(
                "Exact local RELION x-half projection microbatch cap: %d -> %d "
                "(projection_pixels=%d target_row_pixels=%d)",
                tail_capped_hypotheses_per_microbatch,
                int(max_hypotheses_per_microbatch),
                int(window_spec.n_projection),
                int(_exact_local_xhalf_projection_target_row_pixels()),
            )
    bucket_build_t0 = time.time()
    bucket_specs = bucket_local_hypothesis_layout(
        local_layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        unify_bucket_sizes=unify_local_bucket_sizes,
    )
    timing.bucket_build_s += time.time() - bucket_build_t0
    debug_target_only_targets: set[int] = set()
    if debug_score_dump_filter_matches:
        debug_target_only_targets.update(debug_score_dump_targets)
    if debug_fused_posterior_dump_filter_matches:
        debug_target_only_targets.update(debug_fused_posterior_dump_targets)
    debug_score_dump_target_only = bool(
        score_only
        and debug_target_only_targets
        # ``score_only`` also implements the science-critical local parent
        # pass that supplies pass-2 support.  Filtering it merely because a
        # dump target is configured makes the diagnostic change refinement
        # results.  Keep target-only execution as an explicit opt-in for
        # standalone diagnostics.
        and _env_flag(LOCAL_SCORE_DUMP_TARGET_ONLY_ENV)
    )
    debug_target_only_original_bucket_count = len(bucket_specs)
    debug_target_only_original_image_count = int(
        sum(int(bucket.image_indices.shape[0]) for bucket in bucket_specs)
    )
    debug_target_only_original_rotations = int(total_local_rotations)
    if debug_score_dump_target_only:
        filter_t0 = time.time()
        bucket_specs = _filter_buckets_to_debug_targets(
            experiment_dataset,
            bucket_specs,
            debug_target_only_targets,
        )
        timing.bucket_build_s += time.time() - filter_t0
        total_local_rotations = int(
            sum(int(np.sum(bucket.actual_rotation_counts, dtype=np.int64)) for bucket in bucket_specs)
        )
        target_only_images = int(sum(int(bucket.image_indices.shape[0]) for bucket in bucket_specs))
        logger.info(
            "Exact local debug target-only: keeping %d/%d buckets and %d/%d images "
            "for requested original ids %s; unset %s to retain the full score-only computation",
            len(bucket_specs),
            debug_target_only_original_bucket_count,
            target_only_images,
            debug_target_only_original_image_count,
            sorted(int(target) for target in debug_target_only_targets),
            LOCAL_SCORE_DUMP_TARGET_ONLY_ENV,
        )
    if bucket_specs:
        bucket_rotation_counts = np.asarray(
            [int(bucket.bucket_rotation_count) for bucket in bucket_specs],
            dtype=np.int64,
        )
        bucket_image_counts = np.asarray(
            [int(bucket.image_indices.shape[0]) for bucket in bucket_specs],
            dtype=np.int64,
        )
        unique_bucket_counts, unique_bucket_freq = np.unique(bucket_rotation_counts, return_counts=True)
        top_bucket_counts = sorted(
            (
                (int(bucket_count), int(freq))
                for bucket_count, freq in zip(unique_bucket_counts, unique_bucket_freq)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:6]
        logger.info(
            "Exact local bucketing: %d images -> %d buckets "
            "(bucket_size min/med/mean/max=%d/%d/%.1f/%d, images_per_bucket med/max=%d/%d, "
            "top_bucket_counts=%s; max_hypotheses_per_microbatch=%d, n_score_pixels=%d, "
            "n_recon_pixels=%d, n_trans=%d, score_only=%s, relion_x_half_mstep=%s)",
            n_images,
            len(bucket_specs),
            int(np.min(bucket_rotation_counts)),
            int(np.median(bucket_rotation_counts)),
            float(np.mean(bucket_rotation_counts)),
            int(np.max(bucket_rotation_counts)),
            int(np.median(bucket_image_counts)),
            int(np.max(bucket_image_counts)),
            top_bucket_counts,
            int(max_hypotheses_per_microbatch),
            int(n_windowed),
            int(window_spec.n_recon),
            int(n_trans),
            bool(score_only),
            bool(mstep_relion_x_half),
        )
    progress_chunks_override = _optional_nonnegative_int_env(EXACT_LOCAL_PROGRESS_CHUNKS_ENV)
    progress_seconds_override = _optional_nonnegative_int_env(EXACT_LOCAL_PROGRESS_SECONDS_ENV)
    exact_local_progress_chunks = (
        DEFAULT_EXACT_LOCAL_PROGRESS_CHUNKS
        if progress_chunks_override is None
        else int(progress_chunks_override)
    )
    exact_local_progress_seconds = (
        DEFAULT_EXACT_LOCAL_PROGRESS_SECONDS
        if progress_seconds_override is None
        else int(progress_seconds_override)
    )
    progress_total_chunks = len(bucket_specs)
    progress_total_images = int(sum(int(bucket.image_indices.shape[0]) for bucket in bucket_specs))
    progress_completed_chunks = 0
    progress_completed_images = 0
    progress_t0 = time.time()
    progress_last_log_t = progress_t0
    if progress_total_chunks:
        logger.info(
            "Exact local bucket loop start: chunks=%d images=%d total_local_rot=%d n_trans=%d "
            "progress_chunks=%d progress_seconds=%d",
            progress_total_chunks,
            progress_total_images,
            total_local_rotations,
            n_trans,
            exact_local_progress_chunks,
            exact_local_progress_seconds,
        )

    def _log_exact_local_progress(*, force: bool = False, done: bool = False) -> None:
        nonlocal progress_last_log_t
        if not progress_total_chunks:
            return
        now = time.time()
        chunk_due = (
            exact_local_progress_chunks > 0
            and progress_completed_chunks > 0
            and progress_completed_chunks % exact_local_progress_chunks == 0
        )
        time_due = (
            exact_local_progress_seconds > 0
            and progress_last_log_t is not None
            and now - progress_last_log_t >= float(exact_local_progress_seconds)
        )
        if not (force or chunk_due or time_due):
            return
        elapsed = max(0.0, now - progress_t0)
        images_per_second = float(progress_completed_images) / elapsed if elapsed > 0.0 else 0.0
        label = "done" if done else "progress"
        logger.info(
            "Exact local bucket loop %s: chunks=%d/%d images=%d/%d wall=%.1fs images/s=%.1f",
            label,
            progress_completed_chunks,
            progress_total_chunks,
            progress_completed_images,
            progress_total_images,
            elapsed,
            images_per_second,
        )
        progress_last_log_t = now

    def _mark_exact_local_bucket_done(bucket: LocalBucketSpec) -> None:
        nonlocal progress_completed_chunks, progress_completed_images
        progress_completed_chunks += 1
        progress_completed_images += int(bucket.image_indices.shape[0])
        _log_exact_local_progress()

    raw_batch_cache = None
    ctf_param_cache = None
    processed_half_cache = None

    phase_t0 = time.time()
    translation_phases_half = _half_translation_phase_table(
        local_layout.translation_grid,
        image_shape,
    )
    relion_score_translation_angles = (
        _sparse_pass2_diagnostics._relion_cuda_score_translation_angles_if_available(
            local_layout.translation_grid,
            image_shape,
            enabled=relion_exact_score_translation,
        )
    )
    if return_profile:
        _block_until_ready(translation_phases_half)
    translation_phase_time = time.time() - phase_t0
    timing.preprocess_s += translation_phase_time
    preprocess_profile["translation_phase_s"] += translation_phase_time

    big_jit_image_mask_arg, big_jit_mask_mode = resolve_image_mask_for_half_preprocess(
        experiment_dataset,
        image_shape,
        require_mask=score_with_masked_images,
    )
    big_jit_image_mask_arg = jnp.asarray(big_jit_image_mask_arg)
    relion_cuda_preprocess_radius = 0.0
    relion_cuda_preprocess_cosine_width = 0.0
    if relion_exact_bpref_operands:
        image_source = getattr(experiment_dataset, "image_source", None)
        while hasattr(image_source, "parent"):
            image_source = image_source.parent
        preprocess_backend = getattr(image_source, "backend", image_source)
        if getattr(preprocess_backend, "relion_fourier_backend", None) != "relion_cuda":
            raise ValueError("exact RELION BPref operands require RELION CUDA image preprocessing")
        preprocess_params = getattr(preprocess_backend, "_relion_image_mask_params", None)
        if preprocess_params is None:
            raise ValueError("RELION CUDA image preprocessing requires explicit image-mask parameters")
        pixel_size, particle_diameter_ang, cosine_width = preprocess_params
        relion_cuda_preprocess_radius = float(particle_diameter_ang) / (2.0 * float(pixel_size))
        relion_cuda_preprocess_cosine_width = float(cosine_width)

    big_jit_window_indices_arg = window_spec.score_or_full_indices(n_half)
    big_jit_recon_window_indices_arg = window_spec.recon_or_full_indices(n_half)
    big_jit_mstep_recon_window_indices_arg = (
        mstep_recon_window_indices if mstep_relion_x_half else big_jit_recon_window_indices_arg
    )
    disabled_noise_wsum = jnp.zeros(1, dtype=jnp.float32)
    disabled_noise_img_power = jnp.zeros(1, dtype=jnp.float32)
    disabled_noise_a2 = jnp.zeros(1, dtype=jnp.float32)
    disabled_noise_xa = jnp.zeros(1, dtype=jnp.float32)
    disabled_noise_scale = jnp.zeros(1, dtype=jnp.float32)
    disabled_group_ids = jnp.zeros(1, dtype=jnp.int32)
    disabled_noise_shell_indices = jnp.zeros(n_half, dtype=jnp.int32)

    local_support_rows = int(np.sum(local_layout.rotation_counts))
    significant_backprojection_candidate = (
        reconstruct_significant_only
        and n_images > 0
        and local_support_rows >= int(np.ceil(max(n_images, 1) / EXACT_LOCAL_BIG_JIT_MIN_SIGNIFICANT_ROW_FRACTION))
    )
    use_relion_projector = relion_projector_half is not None
    compact_relion_projector_big_jit = bool(use_relion_projector and window_spec.use_window)
    relion_projector_big_jit_supported = bool(use_relion_projector and (not use_window or compact_relion_projector_big_jit))
    disable_big_jit_buckets = os.environ.get("RECOVAR_DISABLE_LOCAL_BIG_JIT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    processed_half_cache_preferred = (
        image_pre_shifts is None or _all_integer_pre_shifts_or_none(image_pre_shifts, n_images) is not None
    ) and _local_processed_half_cache_enabled(
        n_images,
        n_half,
        np.complex64,
        store_recon_half=bool(score_with_masked_images),
    )
    use_big_jit_buckets = (
        ((not use_relion_projector) or relion_projector_big_jit_supported)
        and not disable_big_jit_buckets
        and not bpref_contribution_capture_active
        and not return_reconstruction_probability_values
        and not (accumulate_noise and debug_noise_dump_dir is not None)
        and not processed_half_cache_preferred
    )
    if relion_exact_fine_diff2 and not use_big_jit_buckets:
        raise ValueError("exact RELION fine diff2 requires the local big-JIT bucket path")
    mean_for_proj_big_jit = mean_for_proj
    projection_half_volume_big_jit = False
    relion_projector_half_big_jit = jnp.zeros((1, 1, 1), dtype=jnp.complex64)
    relion_projector_r_max_big_jit = 0
    big_jit_projection_pixel_indices_arg = jnp.zeros((1,), dtype=jnp.int32)
    big_jit_projection_score_take_arg = jnp.zeros((1,), dtype=jnp.int32)
    big_jit_projection_recon_take_arg = jnp.zeros((1,), dtype=jnp.int32)
    big_jit_relion_projector_output_size = 0
    relion_projection_cache = _disabled_relion_projection_cache()
    relion_projection_cache_groups: list[tuple[int, int, int]] = []
    relion_projection_cache_group_cursor = 0
    relion_projection_cache_capacity_rows = 0
    relion_projection_cache_cap_gb = 0.0
    relion_projection_cache_groups_built = 0
    relion_projection_cache_total_build_s = 0.0
    relion_projection_cache_max_rows = 0
    relion_projection_cache_max_estimated_gb = 0.0
    relion_projection_cache_n_projection_pixels = 0
    relion_projection_cache_id_map_rows = 0
    if use_relion_projector:
        if relion_projector_r_max is None:
            raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
        relion_projector_half_big_jit = jnp.asarray(relion_projector_half)
        if relion_projector_half_big_jit.ndim == 4:
            if int(relion_projector_half_big_jit.shape[0]) != 1:
                raise ValueError(
                    "local RELION projector big-JIT path expected a single-class projector slab, "
                    f"got {relion_projector_half_big_jit.shape}",
                )
            relion_projector_half_big_jit = relion_projector_half_big_jit[0]
        if relion_projector_half_big_jit.ndim != 3:
            raise ValueError(
                "local RELION projector big-JIT path expected Projector::data shape (z, y, x_half), "
                f"got {relion_projector_half_big_jit.shape}",
            )
        relion_projector_r_max_big_jit = int(relion_projector_r_max)
        if compact_relion_projector_big_jit:
            big_jit_relion_projector_output_size = (
                int(2 * window_spec.max_r) if window_spec.max_r is not None else 0
            )
            if score_only:
                big_jit_projection_pixel_indices_arg = jnp.asarray(window_spec.score_indices, dtype=jnp.int32)
                big_jit_projection_score_take_arg = jnp.arange(window_spec.n_score, dtype=jnp.int32)
                big_jit_projection_recon_take_arg = jnp.zeros((1,), dtype=jnp.int32)
            else:
                big_jit_projection_pixel_indices_arg = jnp.asarray(window_spec.projection_indices, dtype=jnp.int32)
                big_jit_projection_score_take_arg = jnp.asarray(window_spec.score_projection_take, dtype=jnp.int32)
                big_jit_projection_recon_take_arg = jnp.asarray(window_spec.recon_projection_take, dtype=jnp.int32)
    if (
        use_big_jit_buckets
        and compact_relion_projector_big_jit
        and not score_only
    ):
        requested_cache_rows, relion_projection_cache_cap_gb = _exact_local_relion_projection_cache_capacity_rows(
            int(window_spec.n_projection)
        )
        if requested_cache_rows > 0:
            bucket_specs = _sort_buckets_for_relion_projection_cache(bucket_specs)
        relion_projection_cache_groups = _plan_exact_local_relion_projection_cache_groups(
            bucket_specs,
            cache_row_capacity=int(requested_cache_rows),
        )
        max_cache_groups = _exact_local_relion_projection_cache_max_groups()
        if len(relion_projection_cache_groups) > max_cache_groups:
            logger.info(
                "Exact local RELION projection cache disabled: planned groups=%d exceeds max_groups=%d "
                "(capacity_rows=%d cap=%.2f GB projection_pixels=%d)",
                len(relion_projection_cache_groups),
                max_cache_groups,
                int(requested_cache_rows),
                float(relion_projection_cache_cap_gb),
                int(window_spec.n_projection),
            )
            relion_projection_cache_groups = []
        if relion_projection_cache_groups:
            relion_projection_cache_capacity_rows = int(
                max(row_count for _, _, row_count in relion_projection_cache_groups)
            )
            relion_projection_cache_n_projection_pixels = int(window_spec.n_projection)
            valid_layout_ids = np.asarray(local_layout.rotation_ids_flat, dtype=np.int64)
            valid_layout_ids = valid_layout_ids[valid_layout_ids >= 0]
            relion_projection_cache_id_map_rows = (
                int(np.max(valid_layout_ids)) + 1 if valid_layout_ids.size else 1
            )
            logger.info(
                "Exact local RELION projection cache groups enabled: groups=%d capacity_rows=%d "
                "requested_rows=%d cap=%.2f GB projection_pixels=%d id_map_rows=%d",
                len(relion_projection_cache_groups),
                relion_projection_cache_capacity_rows,
                int(requested_cache_rows),
                float(relion_projection_cache_cap_gb),
                relion_projection_cache_n_projection_pixels,
                relion_projection_cache_id_map_rows,
            )
    if use_big_jit_buckets and not use_relion_projector and not projection_relion_texture_interp:
        mean_for_proj_big_jit = fourier_transform_utils.full_volume_to_half_volume(
            mean_for_proj,
            proj_volume_shape,
        ).reshape(-1)
        projection_half_volume_big_jit = True

    can_use_processed_half_cache = not use_big_jit_buckets and processed_half_cache_preferred
    if can_use_processed_half_cache:
        processed_half_cache_t0 = time.time()
        processed_half_cache = _build_local_processed_half_cache(
            experiment_dataset,
            n_images,
            score_with_masked_images=bool(score_with_masked_images),
            image_pre_shifts=image_pre_shifts,
        )
        cache_elapsed = time.time() - processed_half_cache_t0
        timing.preprocess_s += cache_elapsed
        preprocess_profile["cache_build_s"] += cache_elapsed
        processed_half_cache_enabled = True
    else:
        raw_cache_enabled = _local_raw_cache_enabled(
            n_images,
            image_shape,
            getattr(experiment_dataset, "dtype", np.float32),
        )
        if raw_cache_enabled:
            raw_cache_t0 = time.time()
            raw_batch_cache, ctf_param_cache = _build_local_raw_cache(experiment_dataset, n_images)
            timing.raw_cache_build_s = time.time() - raw_cache_t0

    for bucket_index, bucket in enumerate(bucket_specs):
        if (
            relion_projection_cache_groups
            and relion_projection_cache_group_cursor < len(relion_projection_cache_groups)
            and bucket_index == relion_projection_cache_groups[relion_projection_cache_group_cursor][0]
        ):
            group_start, group_stop, _ = relion_projection_cache_groups[relion_projection_cache_group_cursor]
            relion_projection_cache = _build_exact_local_relion_projection_cache_for_buckets(
                bucket_specs[group_start:group_stop],
                relion_projector_half_big_jit,
                image_shape=image_shape,
                n_projection_pixels=int(window_spec.n_projection),
                relion_projector_r_max=int(relion_projector_r_max_big_jit),
                projection_padding_factor=int(projection_padding_factor),
                projection_relion_texture_interp=projection_relion_texture_interp,
                projection_mask_current_image_disk=bool(projection_mask_current_image_disk),
                projection_pixel_indices=jnp.asarray(window_spec.projection_indices, dtype=jnp.int32),
                projector_output_size=int(big_jit_relion_projector_output_size),
                cache_row_capacity=int(relion_projection_cache_capacity_rows),
                max_global_rotation_id=max(relion_projection_cache_id_map_rows - 1, 0),
                group_index=relion_projection_cache_group_cursor,
                n_groups=len(relion_projection_cache_groups),
            )
            relion_projection_cache_group_cursor += 1
            relion_projection_cache_groups_built += int(relion_projection_cache.enabled)
            relion_projection_cache_total_build_s += float(relion_projection_cache.build_s)
            relion_projection_cache_max_rows = max(relion_projection_cache_max_rows, int(relion_projection_cache.row_count))
            relion_projection_cache_max_estimated_gb = max(
                relion_projection_cache_max_estimated_gb,
                float(relion_projection_cache.estimated_gb),
            )
        n_chunks += 1
        if collect_profile_stats:
            chunk_sizes.append(int(bucket.image_indices.shape[0]))
            chunk_local_rotations.append(int(np.sum(bucket.actual_rotation_counts)))
            chunk_padded_rotations.append(int(bucket.image_indices.shape[0] * bucket.bucket_rotation_count))
            bucket_valid_rotation_ids = np.asarray(bucket.local_rotation_ids, dtype=np.int64)[
                np.asarray(bucket.local_rotation_mask, dtype=bool)
            ]
            chunk_unique_rotations.append(int(np.unique(bucket_valid_rotation_ids).shape[0]))
            total_padded_rotations += int(bucket.image_indices.shape[0] * bucket.bucket_rotation_count)
            local_total_hypotheses += int(np.sum(bucket.actual_rotation_counts) * n_trans)
        fetch_t0 = time.time()
        if raw_batch_cache is None:
            bucket_image_indices = np.asarray(bucket.image_indices, dtype=np.int32)
            if processed_half_cache is None:
                batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, bucket.image_indices)
            else:
                batch_data = None
                ctf_params = processed_half_cache.ctf_params[bucket_image_indices]
                fetched_indices = bucket_image_indices
        else:
            bucket_image_indices = np.asarray(bucket.image_indices, dtype=np.int32)
            batch_data = raw_batch_cache[bucket_image_indices]
            ctf_params = ctf_param_cache[bucket_image_indices]
            fetched_indices = bucket_image_indices
        timing.batch_fetch_s += time.time() - fetch_t0
        bucket = _reorder_bucket_to_indices(bucket, fetched_indices)
        batch_size = int(bucket.image_indices.shape[0])
        debug_fused_posterior_bucket_matches = (
            debug_fused_posterior_dump_filter_matches
            and _bucket_contains_debug_target(
                experiment_dataset,
                bucket.image_indices,
                debug_fused_posterior_dump_targets,
            )
        )
        debug_score_dump_bucket_matches = (
            debug_score_dump_filter_matches
            and _bucket_contains_debug_target(
                experiment_dataset,
                bucket.image_indices,
                debug_score_dump_targets,
            )
        )
        use_big_jit_buckets_for_bucket = bool(
            use_big_jit_buckets
            and not (debug_score_dump_force_split and debug_score_dump_bucket_matches)
        )
        need_local_recon_projection_for_bucket = bool(
            need_local_recon_projection
            or (
                debug_score_dump_bucket_matches
                and debug_score_dump_operands
            )
        )
        translation_sqdist_ang = None
        if translation_prior_centers_np is not None:
            centers = translation_prior_centers_for_images(
                translation_prior_centers_np,
                bucket.image_indices,
                batch_size=batch_size,
            )
            translation_sqdist_ang = translation_sqdist_angstrom(
                local_layout.translation_grid,
                centers,
                experiment_dataset.voxel_size,
            )

        sparse_big_jit_backprojection = False
        sparse_big_jit_mstep_estimated_gb = 0.0
        sparse_big_jit_mstep_cap_gb = 0.0
        if use_big_jit_buckets_for_bucket and significant_backprojection_candidate:
            sparse_big_jit_mstep_estimated_gb, sparse_big_jit_mstep_cap_gb = (
                _sparse_big_jit_mstep_tensors_memory_gb(
                    image_count=max(batch_size, int(getattr(bucket, "bucket_image_count", batch_size))),
                    rotation_count=int(bucket.bucket_rotation_count),
                    n_recon_windowed=window_spec.n_recon,
                    use_float64_scoring=use_float64_scoring,
                )
            )
            sparse_big_jit_backprojection = (
                sparse_big_jit_mstep_cap_gb > 0.0
                and sparse_big_jit_mstep_estimated_gb <= sparse_big_jit_mstep_cap_gb
            )
        can_defer_big_jit_backprojection = (
            use_big_jit_buckets_for_bucket
            and significant_backprojection_candidate
            and not score_only
            and not mstep_subtract_ctf_projection
            and (not disable_adjoint_y or not disable_adjoint_ctf or accumulate_noise)
        )
        force_deferred_big_jit_backprojection = _env_flag(EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP_ENV)
        deferred_big_jit_backprojection = (
            can_defer_big_jit_backprojection
            and (
                force_deferred_big_jit_backprojection
                or not sparse_big_jit_backprojection
            )
        )
        if (
            deferred_big_jit_backprojection
            and not sparse_big_jit_backprojection
            and not logged_sparse_big_jit_deferred_fallback
        ):
            logger.info(
                "Exact local big-JIT using deferred packed M-step because sparse M-step tensors "
                "estimate %.3f GB exceeds cap %.3f GB "
                "(images=%d, bucket_rot=%d, n_recon_pixels=%d, forced=%s)",
                float(sparse_big_jit_mstep_estimated_gb),
                float(sparse_big_jit_mstep_cap_gb),
                int(max(batch_size, int(getattr(bucket, "bucket_image_count", batch_size)))),
                int(bucket.bucket_rotation_count),
                int(window_spec.n_recon),
                bool(force_deferred_big_jit_backprojection),
            )
            logged_sparse_big_jit_deferred_fallback = True

        if use_big_jit_buckets_for_bucket and (
            not significant_backprojection_candidate
            or sparse_big_jit_backprojection
            or deferred_big_jit_backprojection
            or score_only
        ):
            if batch_data is None:
                raise RuntimeError("exact local big-JIT requires fetched native image batches")
            big_jit_t0 = time.time()
            unpadded_bucket = bucket
            unpadded_batch_size = batch_size
            _validate_native_half_batch(batch_data, image_shape)
            integer_pre_shifts = integer_pre_shifts_or_none(
                image_pre_shifts,
                np.asarray(unpadded_bucket.image_indices, dtype=np.int32),
                batch=batch_data,
            )
            bucket, batch_data, ctf_params, valid_image_mask, batch_size = _pad_local_big_jit_image_axis(
                bucket,
                batch_data,
                ctf_params,
            )
            bucket_image_indices = np.asarray(unpadded_bucket.image_indices, dtype=np.int32)
            if relion_exact_bpref_operands:
                ctf_rfloat_unpadded = np.asarray(
                    _sparse_pass2_diagnostics._relion_exact_ctf_half_from_source_star(
                        experiment_dataset,
                        bucket_image_indices,
                        image_shape,
                    ),
                    dtype=np.float64,
                )
                ctf_rfloat_half_arg = jnp.asarray(
                    pad_axis(
                        ctf_rfloat_unpadded,
                        0,
                        batch_size,
                        value=0,
                    ),
                    dtype=jnp.float64,
                )
                inverse_noise_rfloat_cast_np = np.reciprocal(
                    np.asarray(noise_variance_half, dtype=np.float64)
                ).astype(np.float32)
                ctf_squared_rfloat = ctf_rfloat_unpadded * ctf_rfloat_unpadded
                corr_img_rfloat_square_unpadded = (
                    inverse_noise_rfloat_cast_np[None, :].astype(np.float64)
                    * ctf_squared_rfloat
                ).astype(np.float32)
                inverse_noise_rfloat_cast_arg = jnp.asarray(
                    inverse_noise_rfloat_cast_np,
                    dtype=jnp.float32,
                )
                corr_img_rfloat_square_arg = jnp.asarray(
                    pad_axis(
                        corr_img_rfloat_square_unpadded,
                        0,
                        batch_size,
                        value=0,
                    ),
                    dtype=jnp.float32,
                )
            else:
                ctf_rfloat_half_arg = jnp.zeros(
                    (batch_size, n_half),
                    dtype=jnp.float64,
                )
                inverse_noise_rfloat_cast_arg = jnp.zeros(
                    (n_half,),
                    dtype=jnp.float32,
                )
                corr_img_rfloat_square_arg = jnp.zeros(
                    (batch_size, n_half),
                    dtype=jnp.float32,
                )
            apply_integer_pre_shift = integer_pre_shifts is not None
            if apply_integer_pre_shift:
                integer_pre_shifts_arg = jnp.asarray(
                    pad_axis(integer_pre_shifts, 0, batch_size, value=0),
                    dtype=jnp.int32,
                )
                fourier_pre_shifts_arg = jnp.zeros((batch_size, 2), dtype=jnp.float32)
                apply_fourier_pre_shift = False
            elif image_pre_shifts is not None:
                integer_pre_shifts_arg = jnp.zeros((batch_size, 2), dtype=jnp.int32)
                fourier_pre_shifts_arg = jnp.asarray(
                    pad_axis(
                        np.asarray(image_pre_shifts, dtype=np.float32)[bucket_image_indices],
                        0,
                        batch_size,
                        value=0,
                    ),
                    dtype=jnp.float32,
                )
                apply_fourier_pre_shift = True
            else:
                integer_pre_shifts_arg = jnp.zeros((batch_size, 2), dtype=jnp.int32)
                fourier_pre_shifts_arg = jnp.zeros((batch_size, 2), dtype=jnp.float32)
                apply_fourier_pre_shift = False

            image_corrections_arg = (
                jnp.asarray(
                    pad_axis(
                        np.asarray(image_corrections, dtype=np.float32)[bucket_image_indices],
                        0,
                        batch_size,
                        value=1,
                    ),
                )
                if image_corrections is not None
                else jnp.ones(batch_size, dtype=jnp.float32)
            )
            scale_corrections_arg = (
                jnp.asarray(
                    pad_axis(
                        np.asarray(scale_corrections, dtype=np.float32)[bucket_image_indices],
                        0,
                        batch_size,
                        value=1,
                    ),
                )
                if scale_corrections is not None
                else jnp.ones(batch_size, dtype=jnp.float32)
            )
            image_only_corrections_arg = (
                image_corrections_arg / scale_corrections_arg
                if image_corrections is not None
                else jnp.ones(batch_size, dtype=jnp.float32)
            )
            translation_sqdist_arg = (
                jnp.asarray(
                    pad_axis(translation_sqdist_ang, 0, batch_size, value=0),
                    dtype=jnp.float32,
                )
                if translation_sqdist_ang is not None
                else jnp.zeros((batch_size, n_trans), dtype=jnp.float32)
            )
            sample_mask_arg = (
                None
                if bucket.local_sample_mask is None
                else jnp.asarray(bucket.local_sample_mask)
            )
            normalization_log_z_arg = (
                jnp.asarray(
                    pad_axis(normalization_log_z_np[bucket_image_indices], 0, batch_size, value=0),
                    dtype=(jnp.float64 if use_float64_normalization else jnp.float32),
                )
                if normalization_log_z_np is not None
                else jnp.zeros(batch_size, dtype=jnp.float32)
            )
            normalization_log_evidence_arg = (
                jnp.asarray(
                    pad_axis(normalization_log_evidence_np[bucket_image_indices], 0, batch_size, value=0),
                    dtype=(jnp.float64 if use_float64_normalization else jnp.float32),
                )
                if normalization_log_evidence_np is not None
                else jnp.zeros(batch_size, dtype=jnp.float32)
            )
            local_rotation_log_prior_arg = jnp.asarray(bucket.local_rotation_log_prior)
            if class_log_prior != 0.0:
                local_rotation_log_prior_arg = local_rotation_log_prior_arg + jnp.asarray(
                    class_log_prior,
                    dtype=local_rotation_log_prior_arg.dtype,
                )
            if accumulate_noise:
                noise_wsum_arg = noise_wsum
                noise_img_power_arg = noise_img_power
                noise_a2_arg = noise_a2
                noise_xa_arg = noise_xa
                noise_scale_xa_arg = noise_scale_xa if noise_scale_xa is not None else disabled_noise_scale
                noise_scale_aa_arg = noise_scale_aa if noise_scale_aa is not None else disabled_noise_scale
                group_ids_arg = (
                    jnp.asarray(
                        pad_axis(group_ids_np[bucket_image_indices], 0, batch_size, value=0),
                        dtype=jnp.int32,
                    )
                    if group_ids_np is not None
                    else jnp.zeros(batch_size, dtype=jnp.int32)
                )
                shell_indices_half_arg = shell_indices_half
                shell_indices_noise_arg = shell_indices_noise
                noise_variance_for_noise_arg = noise_variance_for_noise
                scale_correction_pixel_mask_arg = scale_correction_pixel_mask
                n_shells_arg = n_shells
            else:
                noise_wsum_arg = disabled_noise_wsum
                noise_img_power_arg = disabled_noise_img_power
                noise_a2_arg = disabled_noise_a2
                noise_xa_arg = disabled_noise_xa
                noise_scale_xa_arg = disabled_noise_scale
                noise_scale_aa_arg = disabled_noise_scale
                group_ids_arg = disabled_group_ids
                shell_indices_half_arg = disabled_noise_shell_indices
                shell_indices_noise_arg = disabled_noise_shell_indices
                noise_variance_for_noise_arg = noise_variance_half
                scale_correction_pixel_mask_arg = jnp.zeros(n_half, dtype=bool)
                n_shells_arg = 1
            if reconstruction_probability_threshold_np is None:
                reconstruction_probability_threshold_arg = jnp.zeros((batch_size,), dtype=jnp.float32)
                has_reconstruction_probability_threshold = False
            else:
                threshold_values = reconstruction_probability_threshold_np[bucket_image_indices]
                threshold_values = pad_axis(threshold_values, 0, batch_size, value=np.inf)
                reconstruction_probability_threshold_arg = jnp.asarray(threshold_values, dtype=jnp.float64)
                has_reconstruction_probability_threshold = True

            projection_max_r_big_jit = window_spec.dense_big_jit_max_r()
            return_big_jit_mstep_tensors = sparse_big_jit_backprojection and (
                not disable_adjoint_y or not disable_adjoint_ctf
            )
            return_big_jit_deferred_mstep_inputs = (
                deferred_big_jit_backprojection and not return_big_jit_mstep_tensors
            )
            big_jit_disable_adjoint_y = (
                disable_adjoint_y or return_big_jit_mstep_tensors or return_big_jit_deferred_mstep_inputs
            )
            big_jit_disable_adjoint_ctf = (
                disable_adjoint_ctf or return_big_jit_mstep_tensors or return_big_jit_deferred_mstep_inputs
            )
            fused_debug_bucket_matches = debug_fused_posterior_bucket_matches
            score_debug_bucket_matches = bool(debug_score_dump_big_jit and debug_score_dump_bucket_matches)
            return_big_jit_debug_arrays = bool(fused_debug_bucket_matches or score_debug_bucket_matches)
            return_big_jit_debug_scores = bool(score_debug_bucket_matches)
            return_big_jit_debug_operands = bool(debug_score_dump_operands and score_debug_bucket_matches)
            if return_big_jit_debug_arrays:
                big_jit_debug_bucket_count += 1
            big_jit_result = run_local_bucket_big_jit(
                jnp.asarray(batch_data),
                jnp.asarray(ctf_params),
                ctf_rfloat_half_arg,
                inverse_noise_rfloat_cast_arg,
                corr_img_rfloat_square_arg,
                mean_for_proj_big_jit,
                relion_projector_half_big_jit,
                Ft_y,
                Ft_ctf,
                noise_wsum_arg,
                noise_img_power_arg,
                noise_a2_arg,
                noise_xa_arg,
                noise_scale_xa_arg,
                noise_scale_aa_arg,
                noise_sigma2_offset,
                noise_sumw,
                big_jit_image_mask_arg,
                integer_pre_shifts_arg,
                fourier_pre_shifts_arg,
                image_corrections_arg,
                image_only_corrections_arg,
                scale_corrections_arg,
                translation_sqdist_arg,
                noise_variance_half,
                translation_phases_half,
                relion_score_translation_angles,
                half_weights,
                norm_half_weights,
                big_jit_window_indices_arg,
                jnp.asarray(relion_fine_full_to_compact, dtype=jnp.int32),
                big_jit_recon_window_indices_arg,
                big_jit_mstep_recon_window_indices_arg,
                shell_indices_half_arg,
                shell_indices_noise_arg,
                noise_variance_for_noise_arg,
                scale_correction_pixel_mask_arg,
                big_jit_projection_pixel_indices_arg,
                big_jit_projection_score_take_arg,
                big_jit_projection_recon_take_arg,
                relion_projection_cache.projections,
                relion_projection_cache.id_map,
                jnp.asarray(bucket.local_rotation_ids, dtype=jnp.int32),
                jnp.asarray(bucket.local_rotations),
                jnp.asarray(_local_mstep_rotations(bucket)),
                local_rotation_log_prior_arg,
                jnp.asarray(bucket.translation_log_prior),
                jnp.asarray(bucket.local_rotation_mask),
                sample_mask_arg,
                jnp.asarray(valid_image_mask),
                group_ids_arg,
                normalization_log_z_arg,
                normalization_log_evidence_arg,
                reconstruction_probability_threshold_arg,
                config,
                mask_mode=big_jit_mask_mode,
                score_with_masked_images=score_with_masked_images,
                apply_integer_pre_shift=apply_integer_pre_shift,
                apply_fourier_pre_shift=apply_fourier_pre_shift,
                half_spectrum_scoring=half_spectrum_scoring,
                use_float64_scoring=use_float64_scoring,
                use_float64_normalization=use_float64_normalization,
                use_window=use_window,
                reconstruct_significant_only=reconstruct_significant_only,
                use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                image_shape=image_shape,
                proj_volume_shape=proj_volume_shape,
                recon_volume_shape=recon_volume_shape,
                disc_type=disc_type,
                projection_half_volume=projection_half_volume_big_jit,
                projection_max_r=projection_max_r_big_jit,
                mstep_max_r=mstep_adjoint_max_r,
                use_compact_relion_projector_projection=bool(compact_relion_projector_big_jit),
                use_relion_projection_cache=bool(relion_projection_cache.enabled),
                relion_projector_output_size=int(big_jit_relion_projector_output_size),
                projection_relion_texture_interp=bool(projection_relion_texture_interp),
                projection_force_jax=bool(projection_force_jax),
                projection_mask_current_image_disk=bool(projection_mask_current_image_disk),
                relion_exact_bpref_operands=relion_exact_bpref_operands,
                relion_exact_fine_diff2=relion_exact_fine_diff2,
                relion_cuda_preprocess_radius=relion_cuda_preprocess_radius,
                relion_cuda_preprocess_cosine_width=relion_cuda_preprocess_cosine_width,
                mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
                mstep_relion_x_half=bool(mstep_relion_x_half),
                disable_adjoint_y=big_jit_disable_adjoint_y,
                disable_adjoint_ctf=big_jit_disable_adjoint_ctf,
                accumulate_noise=accumulate_noise and not return_big_jit_deferred_mstep_inputs,
                accumulate_scale_correction=group_ids_np is not None,
                return_noise_split=return_noise_split,
                return_mstep_tensors=return_big_jit_mstep_tensors,
                return_deferred_mstep_inputs=return_big_jit_deferred_mstep_inputs,
                return_deferred_noise_inputs=bool(return_big_jit_deferred_mstep_inputs and accumulate_noise),
                n_shells=n_shells_arg,
                norm_current_size=current_size,
                include_unweighted_norm_high_shell=include_unweighted_norm_high_shell,
                has_normalization_log_z=normalization_log_z_np is not None,
                has_normalization_log_evidence=normalization_log_evidence_np is not None,
                has_reconstruction_probability_threshold=has_reconstruction_probability_threshold,
                score_only=score_only,
                use_relion_projector=bool(use_relion_projector),
                relion_projector_r_max=relion_projector_r_max_big_jit,
                projection_padding_factor=int(projection_padding_factor),
                return_debug_arrays=return_big_jit_debug_arrays,
                return_debug_scores=return_big_jit_debug_scores,
                return_debug_operands=return_big_jit_debug_operands,
            )
            debug_scores = None
            debug_probs = None
            debug_shifted_score_split = None
            debug_shifted_recon_split = None
            debug_ctf2_over_nv_score = None
            debug_ctf2_over_nv_recon = None
            debug_proj_weighted = None
            debug_proj_for_noise = None
            if return_big_jit_debug_arrays:
                if return_big_jit_debug_operands:
                    (
                        *big_jit_result,
                        debug_scores,
                        debug_probs,
                        debug_shifted_score_split,
                        debug_shifted_recon_split,
                        debug_ctf2_over_nv_score,
                        debug_ctf2_over_nv_recon,
                        debug_proj_weighted,
                        debug_proj_for_noise,
                    ) = big_jit_result
                    if score_only:
                        debug_shifted_recon_split = None
                        debug_ctf2_over_nv_recon = None
                        debug_proj_for_noise = None
                else:
                    *big_jit_result, debug_scores, debug_probs = big_jit_result
            if return_big_jit_deferred_mstep_inputs:
                (
                    Ft_y,
                    Ft_ctf,
                    noise_wsum,
                    noise_img_power,
                    noise_a2,
                    noise_xa,
                    noise_scale_xa,
                    noise_scale_aa,
                    bucket_norm_correction,
                    noise_sigma2_offset,
                    noise_sumw,
                    batch_norm,
                    log_Z,
                    best_log_score,
                    best_argmax,
                    max_posterior,
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    n_significant_samples,
                    reconstruction_sample_mask,
                    reconstruction_rotation_mask,
                    reconstruction_row_count_jax,
                    reconstruction_probs,
                    shifted_recon_split,
                    ctf2_over_nv_recon,
                    shifted_noise_split,
                    processed_score_half,
                ) = big_jit_result
                summed = None
                ctf_probs = None
            elif return_big_jit_mstep_tensors:
                (
                    Ft_y,
                    Ft_ctf,
                    noise_wsum,
                    noise_img_power,
                    noise_a2,
                    noise_xa,
                    noise_scale_xa,
                    noise_scale_aa,
                    bucket_norm_correction,
                    noise_sigma2_offset,
                    noise_sumw,
                    batch_norm,
                    log_Z,
                    best_log_score,
                    best_argmax,
                    max_posterior,
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    n_significant_samples,
                    reconstruction_sample_mask,
                    reconstruction_rotation_mask,
                    reconstruction_row_count_jax,
                    summed,
                    ctf_probs,
                ) = big_jit_result
            else:
                (
                    Ft_y,
                    Ft_ctf,
                    noise_wsum,
                    noise_img_power,
                    noise_a2,
                    noise_xa,
                    noise_scale_xa,
                    noise_scale_aa,
                    bucket_norm_correction,
                    noise_sigma2_offset,
                    noise_sumw,
                    batch_norm,
                    log_Z,
                    best_log_score,
                    best_argmax,
                    max_posterior,
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    n_significant_samples,
                    reconstruction_sample_mask,
                    reconstruction_rotation_mask,
                    reconstruction_row_count_jax,
                ) = big_jit_result
                summed = None
                ctf_probs = None
            if group_ids_np is None:
                noise_scale_xa = None
                noise_scale_aa = None
            if return_profile:
                _block_until_ready(
                    Ft_y,
                    Ft_ctf,
                    batch_norm,
                    log_Z,
                    best_log_score,
                    best_argmax,
                    max_posterior,
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    n_significant_samples,
                    reconstruction_sample_mask,
                    reconstruction_rotation_mask,
                    reconstruction_row_count_jax,
                    noise_wsum,
                    noise_img_power,
                    bucket_norm_correction,
                    *(() if summed is None else (summed, ctf_probs)),
                    *(
                        ()
                        if not return_big_jit_deferred_mstep_inputs
                        else (
                            reconstruction_probs,
                            shifted_recon_split,
                            ctf2_over_nv_recon,
                            shifted_noise_split,
                            processed_score_half,
                        )
                    ),
                )
            if accumulate_noise:
                noise_norm_correction = noise_norm_correction.at[jnp.asarray(bucket_image_indices, dtype=jnp.int32)].add(
                    bucket_norm_correction[:unpadded_batch_size],
                )
            timing.big_jit_bucket_s += time.time() - big_jit_t0
            big_jit_bucket_count += 1
            if sparse_big_jit_backprojection or return_big_jit_deferred_mstep_inputs:
                sparse_big_jit_bucket_count += 1

            if return_big_jit_debug_arrays:
                debug_probs_unpadded = debug_probs[:unpadded_batch_size]
                debug_scores_unpadded = (
                    debug_scores[:unpadded_batch_size]
                    if return_big_jit_debug_scores
                    else None
                )
                log_Z_unpadded = log_Z[:unpadded_batch_size]
                best_log_score_unpadded = best_log_score[:unpadded_batch_size]
                best_argmax_unpadded = best_argmax[:unpadded_batch_size]
                max_posterior_unpadded = max_posterior[:unpadded_batch_size]
                reconstruction_sample_mask_unpadded = reconstruction_sample_mask[:unpadded_batch_size]
                reconstruction_rotation_mask_unpadded = reconstruction_rotation_mask[:unpadded_batch_size]
                n_significant_samples_unpadded = n_significant_samples[:unpadded_batch_size]
                if fused_debug_bucket_matches and debug_fused_posterior_dump_targets:
                    debug_fused_posterior_dump_targets = maybe_write_debug_fused_posterior_dump(
                        experiment_dataset=experiment_dataset,
                        local_layout=local_layout,
                        bucket=unpadded_bucket,
                        image_pre_shifts=image_pre_shifts,
                        probs=debug_probs_unpadded,
                        log_Z=log_Z_unpadded,
                        best_log_score=best_log_score_unpadded,
                        best_argmax=best_argmax_unpadded,
                        max_posterior=max_posterior_unpadded,
                        reconstruction_sample_mask=reconstruction_sample_mask_unpadded,
                        reconstruction_rotation_mask=reconstruction_rotation_mask_unpadded,
                        n_significant_samples=n_significant_samples_unpadded,
                        current_size=current_size,
                        debug_iteration=debug_iteration,
                        dump_dir=debug_fused_posterior_dump_dir,
                        pending_targets=debug_fused_posterior_dump_targets,
                        requested_current_sizes=debug_fused_posterior_dump_current_sizes,
                        requested_iterations=debug_fused_posterior_dump_iterations,
                    )
                if score_debug_bucket_matches and debug_score_dump_targets:
                    debug_score_dump_targets = maybe_write_debug_score_dump(
                        experiment_dataset=experiment_dataset,
                        local_layout=local_layout,
                        bucket=unpadded_bucket,
                        image_pre_shifts=image_pre_shifts,
                        scores=debug_scores_unpadded,
                        probs=debug_probs_unpadded,
                        log_Z=log_Z_unpadded,
                        best_log_score=best_log_score_unpadded,
                        max_posterior=max_posterior_unpadded,
                        reconstruction_sample_mask=reconstruction_sample_mask_unpadded,
                        reconstruction_rotation_mask=reconstruction_rotation_mask_unpadded,
                        n_significant_samples=n_significant_samples_unpadded,
                        current_size=current_size,
                        debug_iteration=debug_iteration,
                        shifted_score_split=debug_shifted_score_split,
                        shifted_recon_split=debug_shifted_recon_split,
                        ctf2_over_nv_score=debug_ctf2_over_nv_score,
                        ctf2_over_nv_recon=debug_ctf2_over_nv_recon,
                        proj_weighted=debug_proj_weighted,
                        proj_for_noise=debug_proj_for_noise,
                        proj_abs2_weighted=None,
                        dump_dir=debug_score_dump_dir,
                        pending_targets=debug_score_dump_targets,
                        requested_current_sizes=debug_score_dump_current_sizes,
                        requested_iterations=debug_score_dump_iterations,
                    )

            pack_t0 = time.time()
            reconstruction_rotation_mask_np = np.asarray(reconstruction_rotation_mask, dtype=bool)[:unpadded_batch_size]
            local_mask_np = np.asarray(bucket.local_rotation_mask, dtype=bool)[:unpadded_batch_size]
            # Latent bug fix 2026-05-08: the pack branch below uses ``summed``
            # and ``ctf_probs`` which the upstream big_jit only returns when
            # ``return_big_jit_mstep_tensors`` is True. That gate is set to
            # ``sparse_big_jit_backprojection AND (not disable_adjoint_y or
            # not disable_adjoint_ctf)`` (this file, line 1532). The probe
            # phase of run_local_k_class_em (k_class.py:874) sets both adjoint
            # disables, so summed/ctf_probs are None and the previous
            # ``if sparse_big_jit_backprojection:`` gate would crash with
            # NoneType subscription when the memory heuristic enabled
            # sparse_big_jit. Mirror the upstream gate so we only enter the
            # pack branch when summed/ctf_probs are actually returned (and
            # the downstream ``if sparse_big_jit_backprojection and not
            # disable_adjoint_y/ctf`` consumers at lines 1716/1736 can use
            # them).
            packed_reconstruction_probs = None
            packed_reconstruction_probs_sum_t = None
            if return_big_jit_deferred_mstep_inputs:
                probs_sum_t_np = np.asarray(probs_sum_t[:unpadded_batch_size], dtype=np.float64)
                (
                    reconstruction_take_indices,
                    reconstruction_pack_mask_np,
                    _,
                    reconstruction_row_count,
                ) = _build_nonzero_reconstruction_pack_indices(
                    reconstruction_rotation_mask_np,
                    local_mask_np,
                    probs_sum_t_np,
                    rotation_block_size,
                )
                reconstruction_take_indices_jnp = jnp.asarray(reconstruction_take_indices, dtype=jnp.int32)
                reconstruction_pack_mask_jnp = jnp.asarray(reconstruction_pack_mask_np)
                packed_rotations_np = np.take_along_axis(
                    np.asarray(bucket.local_rotations[:unpadded_batch_size], dtype=np.float32),
                    reconstruction_take_indices[:, :, None, None],
                    axis=1,
                )
                packed_mstep_rotations_np = np.take_along_axis(
                    _local_mstep_rotations(bucket)[:unpadded_batch_size],
                    reconstruction_take_indices[:, :, None, None],
                    axis=1,
                )
                packed_reconstruction_probs = jnp.take_along_axis(
                    reconstruction_probs[:unpadded_batch_size],
                    reconstruction_take_indices_jnp[:, :, None],
                    axis=1,
                )
                packed_reconstruction_probs = jnp.where(
                    reconstruction_pack_mask_jnp[:, :, None],
                    packed_reconstruction_probs,
                    0.0,
                )
                packed_reconstruction_probs_sum_t = jnp.take_along_axis(
                    reconstruction_probs_sum_t[:unpadded_batch_size],
                    reconstruction_take_indices_jnp,
                    axis=1,
                )
                packed_reconstruction_probs_sum_t = jnp.where(
                    reconstruction_pack_mask_jnp,
                    packed_reconstruction_probs_sum_t,
                    0.0,
                )
                packed_summed = None
                packed_ctf_probs = None
                packed_flat_rotations = None
            elif sparse_big_jit_backprojection and (not disable_adjoint_y or not disable_adjoint_ctf):
                probs_sum_t_np = np.asarray(probs_sum_t[:unpadded_batch_size], dtype=np.float64)
                (
                    reconstruction_take_indices,
                    reconstruction_pack_mask_np,
                    _,
                    reconstruction_row_count,
                ) = _build_nonzero_reconstruction_pack_indices(
                    reconstruction_rotation_mask_np,
                    local_mask_np,
                    probs_sum_t_np,
                    rotation_block_size,
                )
                reconstruction_take_indices_jnp = jnp.asarray(reconstruction_take_indices, dtype=jnp.int32)
                reconstruction_pack_mask_jnp = jnp.asarray(reconstruction_pack_mask_np)
                packed_rotations_np = np.take_along_axis(
                    np.asarray(bucket.local_rotations[:unpadded_batch_size], dtype=np.float32),
                    reconstruction_take_indices[:, :, None, None],
                    axis=1,
                )
                packed_mstep_rotations_np = np.take_along_axis(
                    _local_mstep_rotations(bucket)[:unpadded_batch_size],
                    reconstruction_take_indices[:, :, None, None],
                    axis=1,
                )
                packed_summed = jnp.take_along_axis(
                    summed[:unpadded_batch_size],
                    reconstruction_take_indices_jnp[:, :, None],
                    axis=1,
                )
                packed_summed = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_summed, 0.0)
                packed_ctf_probs = jnp.take_along_axis(
                    ctf_probs[:unpadded_batch_size],
                    reconstruction_take_indices_jnp[:, :, None],
                    axis=1,
                )
                packed_ctf_probs = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_ctf_probs, 0.0)
                packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_mstep_rotations_np))
            else:
                probs_sum_t_np = None
                reconstruction_take_indices = np.broadcast_to(
                    np.arange(int(bucket.bucket_rotation_count), dtype=np.int32)[None, :],
                    (unpadded_batch_size, int(bucket.bucket_rotation_count)),
                )
                reconstruction_pack_mask_np = reconstruction_rotation_mask_np & local_mask_np
                reconstruction_row_count = int(np.asarray(reconstruction_row_count_jax, dtype=np.int32))
            timing.pack_s += time.time() - pack_t0

            flat_packed_summed = None
            flat_packed_ctf_probs = None
            if sparse_big_jit_backprojection and (not disable_adjoint_y or not disable_adjoint_ctf):
                flat_packed_summed = flatten_bucket_rows(packed_summed)
                flat_packed_ctf_probs = flatten_bucket_rows(packed_ctf_probs)

            if return_big_jit_deferred_mstep_inputs and (not disable_adjoint_y or not disable_adjoint_ctf):
                if packed_reconstruction_probs is None:
                    raise RuntimeError("deferred big-JIT local M-step requires packed posterior rows")
                if packed_reconstruction_probs_sum_t is None:
                    raise RuntimeError("deferred big-JIT local M-step requires packed posterior sums")
                packed_rotation_count = int(packed_rotations_np.shape[1])
                n_recon_pixels = window_spec.n_recon if window_spec.use_window else int(n_half)
                chunk_rows = min(
                    packed_rotation_count,
                    _packed_noise_projection_chunk_rows(n_recon_pixels, batch_size=unpadded_batch_size),
                )
                if chunk_rows < packed_rotation_count and not logged_deferred_mstep_chunking:
                    logger.info(
                        "Exact local big-JIT deferred packed M-step chunking: "
                        "packed_rows=%d chunk_rows=%d n_recon_pixels=%d batch_size=%d",
                        packed_rotation_count,
                        chunk_rows,
                        n_recon_pixels,
                        unpadded_batch_size,
                    )
                    logged_deferred_mstep_chunking = True
                shifted_recon_split_unpadded = shifted_recon_split[:unpadded_batch_size]
                ctf2_over_nv_recon_unpadded = ctf2_over_nv_recon[:unpadded_batch_size]
                for chunk_start in range(0, packed_rotation_count, chunk_rows):
                    chunk_stop = min(packed_rotation_count, chunk_start + chunk_rows)
                    chunk_probs = packed_reconstruction_probs[:, chunk_start:chunk_stop]
                    chunk_scoring_rotations = packed_rotations_np[:, chunk_start:chunk_stop]
                    chunk_rotations = packed_mstep_rotations_np[:, chunk_start:chunk_stop]
                    chunk_flat_rotations = flatten_bucket_rotations(jnp.asarray(chunk_rotations))
                    if not disable_adjoint_y:
                        mstep_t0 = time.time()
                        chunk_summed = compute_local_weighted_sums(chunk_probs, shifted_recon_split_unpadded)
                        if mstep_subtract_ctf_projection:
                            # The memory-deferred big-JIT path returns posterior rows instead
                            # of the already reduced residual-image rows. Recreate RELION's
                            # Fimg_store = Fimg - Frefctf operand before backprojection, just
                            # as the ordinary deferred path does below.
                            chunk_proj_for_residual = _project_packed_noise_rows(
                                mean_for_proj=mean_for_proj,
                                packed_flat_rotations=flatten_bucket_rotations(
                                    jnp.asarray(chunk_scoring_rotations)
                                ),
                                packed_rotation_count=chunk_stop - chunk_start,
                                batch_size=unpadded_batch_size,
                                image_shape=image_shape,
                                proj_volume_shape=proj_volume_shape,
                                disc_type=disc_type,
                                projection_kwargs=projection_kwargs,
                                window_spec=window_spec,
                                n_half=n_half,
                                precision_policy=precision_policy,
                                reconstruction_pack_mask_jnp=reconstruction_pack_mask_jnp[
                                    :, chunk_start:chunk_stop
                                ],
                                relion_projector_half=relion_projector_half,
                                relion_projector_r_max=relion_projector_r_max,
                                projection_padding_factor=projection_padding_factor,
                            )
                            chunk_probs_sum_t = packed_reconstruction_probs_sum_t[
                                :, chunk_start:chunk_stop
                            ]
                            frefctf_weighted = (
                                chunk_proj_for_residual * ctf2_over_nv_recon_unpadded[:, None, :]
                            )
                            chunk_summed = chunk_summed - chunk_probs_sum_t[..., None] * frefctf_weighted
                        if return_profile:
                            _block_until_ready(chunk_summed)
                        timing.mstep_s += time.time() - mstep_t0

                        adjoint_y_t0 = time.time()
                        Ft_y, n_adjoint_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
                            flatten_bucket_rows(chunk_summed),
                            mstep_recon_window_indices,
                            chunk_flat_rotations,
                            Ft_y,
                            image_shape,
                            recon_volume_shape,
                            "linear_interp",
                            use_window=use_window,
                            max_r=mstep_adjoint_max_r,
                            relion_x_half=bool(mstep_relion_x_half),
                            target_rows=sparse_adjoint_target_rows,
                        )
                        if return_profile:
                            _block_until_ready(Ft_y)
                        sparse_adjoint_chunk_count += int(n_adjoint_chunks)
                        timing.adjoint_y_s += time.time() - adjoint_y_t0

                    if not disable_adjoint_ctf:
                        mstep_t0 = time.time()
                        chunk_probs_sum_t = packed_reconstruction_probs_sum_t[:, chunk_start:chunk_stop]
                        chunk_ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
                            chunk_probs_sum_t,
                            ctf2_over_nv_recon_unpadded,
                        )
                        if return_profile:
                            _block_until_ready(chunk_ctf_probs)
                        timing.mstep_s += time.time() - mstep_t0

                        adjoint_ctf_t0 = time.time()
                        Ft_ctf, n_adjoint_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
                            flatten_bucket_rows(chunk_ctf_probs),
                            mstep_recon_window_indices,
                            chunk_flat_rotations,
                            Ft_ctf,
                            image_shape,
                            recon_volume_shape,
                            "linear_interp",
                            use_window=use_window,
                            max_r=mstep_adjoint_max_r,
                            relion_x_half=bool(mstep_relion_x_half),
                            target_rows=sparse_adjoint_target_rows,
                        )
                        if return_profile:
                            _block_until_ready(Ft_ctf)
                        sparse_adjoint_chunk_count += int(n_adjoint_chunks)
                        timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

            if sparse_big_jit_backprojection and (not disable_adjoint_y or not disable_adjoint_ctf):
                if not disable_adjoint_y:
                    adjoint_y_t0 = time.time()
                    Ft_y, n_adjoint_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
                        flat_packed_summed,
                        mstep_recon_window_indices,
                        packed_flat_rotations,
                        Ft_y,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        use_window=use_window,
                        max_r=mstep_adjoint_max_r,
                        relion_x_half=bool(mstep_relion_x_half),
                        target_rows=sparse_adjoint_target_rows,
                    )
                    if return_profile:
                        _block_until_ready(Ft_y)
                    sparse_adjoint_chunk_count += int(n_adjoint_chunks)
                    timing.adjoint_y_s += time.time() - adjoint_y_t0

                if not disable_adjoint_ctf:
                    adjoint_ctf_t0 = time.time()
                    Ft_ctf, n_adjoint_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
                        flat_packed_ctf_probs,
                        mstep_recon_window_indices,
                        packed_flat_rotations,
                        Ft_ctf,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        use_window=use_window,
                        max_r=mstep_adjoint_max_r,
                        relion_x_half=bool(mstep_relion_x_half),
                        target_rows=sparse_adjoint_target_rows,
                    )
                    if return_profile:
                        _block_until_ready(Ft_ctf)
                    sparse_adjoint_chunk_count += int(n_adjoint_chunks)
                    timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

            if return_big_jit_deferred_mstep_inputs and accumulate_noise:
                noise_t0 = time.time()
                reconstruction_probs_unpadded = reconstruction_probs[:unpadded_batch_size]
                support_mass = jnp.sum(reconstruction_probs_unpadded.reshape(unpadded_batch_size, -1), axis=1).astype(
                    jnp.float32
                )
                translation_posterior = jnp.sum(reconstruction_probs_unpadded, axis=1).astype(jnp.float32)
                noise_sumw_offset = jnp.sum(
                    translation_posterior * translation_sqdist_arg[:unpadded_batch_size].astype(jnp.float32),
                )
                processed_noise_power_half = processed_score_half[:unpadded_batch_size]
                processed_noise_power_half = (
                    processed_noise_power_half * image_only_corrections_arg[:unpadded_batch_size, None]
                )
                batch_img_power = jnp.sum(
                    (jnp.abs(processed_noise_power_half) ** 2) * support_mass[:, None],
                    axis=0,
                ).astype(jnp.float32)
                batch_img_power_per_image = _norm_correction_image_power_per_image(
                    processed_noise_power_half,
                    support_mass,
                    shell_indices_half,
                    jnp.ones_like(support_mass, dtype=bool),
                    norm_unweighted_shell_cutoff,
                    shell_count=n_shells,
                    image_shape=image_shape,
                    current_size=current_size,
                    include_unweighted_high_shell=include_unweighted_norm_high_shell,
                )
                batch_img_power_shells = bin_shell_values_jax(batch_img_power, shell_indices_half, n_shells)
                noise_img_power = noise_img_power + batch_img_power_shells
                noise_sumw = noise_sumw + jnp.sum(support_mass)

                shifted_noise_split_unpadded = shifted_noise_split[:unpadded_batch_size]
                packed_rotation_count = int(packed_rotations_np.shape[1])
                n_recon_pixels = window_spec.n_recon if window_spec.use_window else int(n_half)
                noise_projection_pixels = int(n_half) if relion_projector_half is not None else int(n_recon_pixels)
                chunk_rows = min(
                    packed_rotation_count,
                    _packed_noise_projection_chunk_rows(noise_projection_pixels, batch_size=unpadded_batch_size),
                )
                if chunk_rows < packed_rotation_count and not logged_deferred_noise_projection_chunking:
                    logger.info(
                        "Exact local big-JIT deferred noise projection chunking: "
                        "packed_rows=%d chunk_rows=%d n_recon_pixels=%d projection_pixels=%d batch_size=%d",
                        packed_rotation_count,
                        chunk_rows,
                        n_recon_pixels,
                        noise_projection_pixels,
                        unpadded_batch_size,
                    )
                    logged_deferred_noise_projection_chunking = True
                block_noise_shells = jnp.zeros(n_shells, dtype=jnp.float32)
                block_a2_shells = jnp.zeros(n_shells, dtype=jnp.float32)
                block_xa_shells = jnp.zeros(n_shells, dtype=jnp.float32)
                block_norm_residual = jnp.zeros(unpadded_batch_size, dtype=jnp.float32)
                bucket_group_ids = (
                    group_ids_arg[:unpadded_batch_size] if group_ids_np is not None else None
                )
                batch_scale_unpadded = scale_corrections_arg[:unpadded_batch_size]
                for chunk_start in range(0, packed_rotation_count, chunk_rows):
                    chunk_stop = min(packed_rotation_count, chunk_start + chunk_rows)
                    chunk_rotations = packed_rotations_np[:, chunk_start:chunk_stop]
                    chunk_proj_for_noise = _project_packed_noise_rows(
                        mean_for_proj=mean_for_proj,
                        packed_flat_rotations=flatten_bucket_rotations(jnp.asarray(chunk_rotations)),
                        packed_rotation_count=chunk_stop - chunk_start,
                        batch_size=unpadded_batch_size,
                        image_shape=image_shape,
                        proj_volume_shape=proj_volume_shape,
                        disc_type=disc_type,
                        projection_kwargs=projection_kwargs,
                        window_spec=window_spec,
                        n_half=n_half,
                        precision_policy=precision_policy,
                        reconstruction_pack_mask_jnp=reconstruction_pack_mask_jnp[:, chunk_start:chunk_stop],
                        relion_projector_half=relion_projector_half,
                        relion_projector_r_max=relion_projector_r_max,
                        projection_padding_factor=projection_padding_factor,
                    )
                    chunk_probs = packed_reconstruction_probs[:, chunk_start:chunk_stop]
                    chunk_probs_sum_t = packed_reconstruction_probs_sum_t[:, chunk_start:chunk_stop]
                    chunk_summed_masked_noise = compute_local_weighted_sums(chunk_probs, shifted_noise_split_unpadded)
                    chunk_ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
                        chunk_probs_sum_t,
                        ctf2_over_nv_recon_unpadded,
                    )
                    flat_proj_for_noise = flatten_bucket_rows(chunk_proj_for_noise)
                    flat_proj_abs2_for_noise = jnp.abs(flat_proj_for_noise) ** 2
                    chunk_noise_shells, chunk_a2_shells, chunk_xa_shells = _compute_noise_block(
                        flat_proj_for_noise,
                        flat_proj_abs2_for_noise,
                        flatten_bucket_rows(chunk_summed_masked_noise),
                        flatten_bucket_rows(chunk_ctf_probs),
                        noise_variance_for_noise,
                        shell_indices_noise,
                        n_shells,
                        return_noise_split,
                    )
                    block_noise_shells = block_noise_shells + chunk_noise_shells
                    block_a2_shells = block_a2_shells + chunk_a2_shells
                    block_xa_shells = block_xa_shells + chunk_xa_shells
                    chunk_proj_abs2_for_norm = flat_proj_abs2_for_noise.reshape(chunk_proj_for_noise.shape)
                    block_norm_residual = block_norm_residual + _compute_norm_residual_per_image(
                        chunk_proj_for_noise,
                        chunk_proj_abs2_for_norm,
                        chunk_summed_masked_noise,
                        chunk_ctf_probs,
                        noise_variance_for_noise,
                    )
                    if noise_scale_xa is not None:
                        scale_xa_per_image, scale_aa_per_image = _compute_scale_correction_terms_per_image(
                            chunk_proj_for_noise,
                            chunk_proj_abs2_for_norm,
                            chunk_summed_masked_noise,
                            chunk_ctf_probs,
                            noise_variance_for_noise,
                            batch_scale_unpadded,
                            scale_correction_pixel_mask,
                        )
                        noise_scale_xa = noise_scale_xa.at[bucket_group_ids].add(scale_xa_per_image)
                        noise_scale_aa = noise_scale_aa.at[bucket_group_ids].add(scale_aa_per_image)
                if return_profile:
                    _block_until_ready(block_noise_shells, block_norm_residual)
                noise_wsum = noise_wsum + block_noise_shells
                if return_noise_split:
                    noise_a2 = noise_a2 + block_a2_shells
                    noise_xa = noise_xa + block_xa_shells
                noise_norm_correction = noise_norm_correction.at[jnp.asarray(bucket_image_indices, dtype=jnp.int32)].add(
                    batch_img_power_per_image + block_norm_residual,
                )
                noise_sigma2_offset = noise_sigma2_offset + noise_sumw_offset
                timing.noise_s += time.time() - noise_t0

            postprocess_t0 = time.time()
            stats_probs_sum_t = reconstruction_probs_sum_t if stats_use_reconstruction_probs else probs_sum_t
            stats_probs_sum_t_np = (
                None
                if probs_sum_t_np is None
                else np.asarray(stats_probs_sum_t[:unpadded_batch_size], dtype=np.float64)
            )
            significant_sample_count, reconstruction_row_count = _postprocess_local_bucket(
                image_indices=unpadded_bucket.image_indices,
                local_rotation_ids=bucket.local_rotation_ids[:unpadded_batch_size],
                local_rotation_mask=bucket.local_rotation_mask[:unpadded_batch_size],
                local_rotations=bucket.local_rotations[:unpadded_batch_size],
                local_rotation_posterior_ids=(
                    None
                    if bucket.local_rotation_posterior_ids is None
                    else bucket.local_rotation_posterior_ids[:unpadded_batch_size]
                ),
                translation_grid=local_layout.translation_grid,
                n_trans=n_trans,
                best_argmax=best_argmax[:unpadded_batch_size],
                batch_norm=batch_norm[:unpadded_batch_size],
                log_Z=log_Z[:unpadded_batch_size],
                best_log_score=best_log_score[:unpadded_batch_size],
                max_posterior=max_posterior[:unpadded_batch_size],
                probs_sum_t=(
                    stats_probs_sum_t[:unpadded_batch_size]
                    if stats_probs_sum_t_np is None
                    else stats_probs_sum_t_np
                ),
                n_significant_samples=n_significant_samples[:unpadded_batch_size],
                reconstruction_sample_mask=reconstruction_sample_mask[:unpadded_batch_size],
                collect_profile_stats=collect_profile_stats,
                reconstruction_row_count=reconstruction_row_count,
                reconstruction_take_indices=reconstruction_take_indices,
                reconstruction_pack_mask=reconstruction_pack_mask_np,
                buffers=postprocess_buffers,
            )
            if collect_profile_stats:
                total_significant_samples += significant_sample_count
                total_reconstruction_rows += int(reconstruction_row_count)
            timing.postprocess_s += time.time() - postprocess_t0

            host_stats_t0 = time.time()
            logger.debug(
                "Exact local big-JIT bucket: %d images, bucket_rot=%d, total_local_rot=%d",
                unpadded_batch_size,
                int(bucket.bucket_rotation_count),
                int(np.sum(unpadded_bucket.actual_rotation_counts)),
            )
            timing.host_stats_s += time.time() - host_stats_t0
            _mark_exact_local_bucket_done(bucket)
            continue

        preprocess_t0 = time.time()
        (
            shifted_half,
            shifted_recon_half,
            batch_norm,
            ctf2_over_nv_score_half,
            ctf2_over_nv_recon_half,
            processed_score_half,
            real_space_pre_shift_applied,
        ) = _prepare_local_exact_bucket(
            experiment_dataset,
            batch_data,
            ctf_params,
            bucket.image_indices,
            noise_variance_half,
            translation_phases_half,
            config,
            norm_half_weights,
            score_with_masked_images,
            relion_score_translation_angles=relion_score_translation_angles,
            image_pre_shifts=image_pre_shifts,
            processed_half_cache=processed_half_cache,
            timer=preprocess_profile if return_profile else None,
            synchronize_profile=return_profile,
            score_complex_dtype=precision_policy.score_complex_dtype,
            score_real_dtype=precision_policy.score_real_dtype,
            norm_real_dtype=precision_policy.normalization_real_dtype,
            relion_exact_bpref_operands=relion_exact_bpref_operands,
        )
        if scale_corrections is not None:
            batch_scale = jnp.asarray(scale_corrections[np.asarray(bucket.image_indices)])
        else:
            batch_scale = jnp.ones(batch_size, dtype=batch_norm.dtype)
        bucket_group_ids = (
            jnp.asarray(group_ids_np[np.asarray(bucket.image_indices)], dtype=jnp.int32)
            if group_ids_np is not None
            else None
        )

        if image_corrections is not None:
            batch_corr = jnp.asarray(image_corrections[np.asarray(bucket.image_indices)])
            image_only_corr = batch_corr / batch_scale
            corr_expanded = jnp.repeat(batch_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            shifted_recon_half = shifted_recon_half * corr_expanded[:, None]
            batch_norm = batch_norm * (image_only_corr**2)[:, None]
        else:
            batch_corr = None
            image_only_corr = None

        if scale_corrections is not None:
            ctf2_over_nv_score_half = ctf2_over_nv_score_half * (batch_scale**2)[:, None]
            ctf2_over_nv_recon_half = ctf2_over_nv_recon_half * (batch_scale**2)[:, None]

        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(bucket.image_indices)])
            phase_expanded = tiled_half_image_phase_factors(image_shape, batch_shifts, n_trans)
            shifted_half = shifted_half * phase_expanded
            shifted_recon_half = shifted_recon_half * phase_expanded
        shifted_half_with_dc = shifted_half
        ctf2_over_nv_recon_half_with_dc = ctf2_over_nv_recon_half

        if half_spectrum_scoring:
            dc_mask = make_shell_indices_half(image_shape) == 0
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            ctf2_over_nv_score_half = jnp.where(
                dc_mask[None, :], 0.0, ctf2_over_nv_score_half
            )

        if use_window:
            shifted_score = shifted_half[:, window_indices]
            shifted_recon = shifted_recon_half[:, recon_window_indices]
            ctf2_over_nv_score = ctf2_over_nv_score_half[:, window_indices]
            ctf2_over_nv_recon = ctf2_over_nv_recon_half_with_dc[:, recon_window_indices]
            shifted_noise = shifted_half_with_dc[:, recon_window_indices]
        else:
            shifted_score = shifted_half
            shifted_recon = shifted_recon_half
            ctf2_over_nv_score = ctf2_over_nv_score_half
            ctf2_over_nv_recon = ctf2_over_nv_recon_half_with_dc
            shifted_noise = shifted_half_with_dc

        (
            shifted_score,
            shifted_recon,
            shifted_noise,
            ctf2_over_nv_score,
            ctf2_over_nv_recon,
        ) = precision_policy.cast_local_preprocessed_inputs(
            shifted_score,
            shifted_recon,
            shifted_noise,
            ctf2_over_nv_score,
            ctf2_over_nv_recon,
        )
        timing.preprocess_s += time.time() - preprocess_t0

        projection_t0 = time.time()
        projection_block = _project_local_bucket(
            mean_for_proj=mean_for_proj,
            bucket=bucket,
            image_shape=image_shape,
            proj_volume_shape=proj_volume_shape,
            disc_type=disc_type,
            projection_kwargs=projection_kwargs,
            window_spec=window_spec,
            n_half=n_half,
            half_weights=half_weights,
            precision_policy=precision_policy,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            materialize_recon_projection=need_local_recon_projection_for_bucket,
        )
        proj_weighted = projection_block.proj_weighted
        proj_for_noise = projection_block.proj_for_noise
        if return_profile:
            _block_until_ready(proj_weighted)
        timing.projection_s += time.time() - projection_t0

        shifted_score_split = shifted_score.reshape(batch_size, n_trans, -1)
        shifted_recon_split = shifted_recon.reshape(batch_size, n_trans, -1)
        local_rotation_log_prior = jnp.asarray(bucket.local_rotation_log_prior)
        if class_log_prior != 0.0:
            local_rotation_log_prior = local_rotation_log_prior + jnp.asarray(
                class_log_prior,
                dtype=local_rotation_log_prior.dtype,
            )
        defer_packed_mstep_reduction = False
        has_external_normalization = normalization_log_z_np is not None or normalization_log_evidence_np is not None
        can_use_fused_score_mstep = (
            fused_score_mstep_enabled
            and reconstruction_probability_threshold_np is None
            and not debug_score_dump_bucket_matches
            and not bpref_contribution_capture_active
        )
        defer_packed_mstep_requested = _env_flag(EXACT_LOCAL_DEFER_PACKED_MSTEP_ENV)
        threshold_for_bucket = (
            None
            if reconstruction_probability_threshold_np is None
            else jnp.asarray(reconstruction_probability_threshold_np[np.asarray(bucket.image_indices)], dtype=jnp.float64)
        )
        if can_use_fused_score_mstep and score_only and not has_external_normalization:
            fused_t0 = time.time()
            (
                log_Z,
                probs,
                best_log_score,
                best_argmax,
                max_posterior,
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
                probs_sum_t,
                reconstruction_probs_sum_t,
            ) = fused_score_normalize_support_abs2_on_demand(
                shifted_score_split,
                ctf2_over_nv_score,
                proj_weighted,
                half_weights_windowed if use_window else half_weights,
                local_rotation_log_prior,
                jnp.asarray(bucket.translation_log_prior),
                jnp.asarray(bucket.local_rotation_mask),
                None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask),
                None,
                half_spectrum_scoring=half_spectrum_scoring,
                use_float64_normalization=use_float64_normalization,
                reconstruct_significant_only=reconstruct_significant_only,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            reconstruction_probs = None
            summed = None
            ctf_probs = None
            if return_profile:
                _block_until_ready(
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    reconstruction_rotation_mask,
                    n_significant_samples,
                    best_argmax,
                    log_Z,
                    best_log_score,
                    max_posterior,
                )
            debug_fused_posterior_dump_targets = maybe_write_debug_fused_posterior_dump(
                experiment_dataset=experiment_dataset,
                local_layout=local_layout,
                bucket=bucket,
                image_pre_shifts=image_pre_shifts,
                probs=probs,
                log_Z=log_Z,
                best_log_score=best_log_score,
                best_argmax=best_argmax,
                max_posterior=max_posterior,
                reconstruction_sample_mask=reconstruction_sample_mask,
                reconstruction_rotation_mask=reconstruction_rotation_mask,
                n_significant_samples=n_significant_samples,
                current_size=current_size,
                debug_iteration=debug_iteration,
                dump_dir=debug_fused_posterior_dump_dir,
                pending_targets=debug_fused_posterior_dump_targets,
                requested_current_sizes=debug_fused_posterior_dump_current_sizes,
                requested_iterations=debug_fused_posterior_dump_iterations,
            )
            fused_elapsed = time.time() - fused_t0
            timing.fused_score_mstep_s += fused_elapsed
        elif can_use_fused_score_mstep and not has_external_normalization and defer_packed_mstep_requested:
            fused_t0 = time.time()
            (
                log_Z,
                probs,
                best_log_score,
                best_argmax,
                max_posterior,
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
                reconstruction_probs,
                probs_sum_t,
                reconstruction_probs_sum_t,
            ) = fused_score_normalize_support_probs_abs2_on_demand(
                shifted_score_split,
                ctf2_over_nv_score,
                proj_weighted,
                half_weights_windowed if use_window else half_weights,
                local_rotation_log_prior,
                jnp.asarray(bucket.translation_log_prior),
                jnp.asarray(bucket.local_rotation_mask),
                None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask),
                None,
                half_spectrum_scoring=half_spectrum_scoring,
                use_float64_normalization=use_float64_normalization,
                reconstruct_significant_only=reconstruct_significant_only,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            summed = None
            ctf_probs = None
            defer_packed_mstep_reduction = True
            if return_profile:
                _block_until_ready(
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    reconstruction_probs,
                    reconstruction_rotation_mask,
                    n_significant_samples,
                    best_argmax,
                    log_Z,
                    best_log_score,
                    max_posterior,
                )
            debug_fused_posterior_dump_targets = maybe_write_debug_fused_posterior_dump(
                experiment_dataset=experiment_dataset,
                local_layout=local_layout,
                bucket=bucket,
                image_pre_shifts=image_pre_shifts,
                probs=probs,
                log_Z=log_Z,
                best_log_score=best_log_score,
                best_argmax=best_argmax,
                max_posterior=max_posterior,
                reconstruction_sample_mask=reconstruction_sample_mask,
                reconstruction_rotation_mask=reconstruction_rotation_mask,
                n_significant_samples=n_significant_samples,
                current_size=current_size,
                debug_iteration=debug_iteration,
                dump_dir=debug_fused_posterior_dump_dir,
                pending_targets=debug_fused_posterior_dump_targets,
                requested_current_sizes=debug_fused_posterior_dump_current_sizes,
                requested_iterations=debug_fused_posterior_dump_iterations,
            )
            fused_elapsed = time.time() - fused_t0
            timing.fused_score_mstep_s += fused_elapsed
        elif can_use_fused_score_mstep and not has_external_normalization:
            fused_t0 = time.time()
            (
                log_Z,
                probs,
                best_log_score,
                best_argmax,
                max_posterior,
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
                reconstruction_probs,
                probs_sum_t,
                reconstruction_probs_sum_t,
                summed,
                ctf_probs,
            ) = fused_score_normalize_mstep_abs2_on_demand(
                shifted_score_split,
                ctf2_over_nv_score,
                proj_weighted,
                half_weights_windowed if use_window else half_weights,
                local_rotation_log_prior,
                jnp.asarray(bucket.translation_log_prior),
                jnp.asarray(bucket.local_rotation_mask),
                None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask),
                shifted_recon_split,
                ctf2_over_nv_recon,
                None,
                half_spectrum_scoring=half_spectrum_scoring,
                use_float64_normalization=use_float64_normalization,
                reconstruct_significant_only=reconstruct_significant_only,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            if mstep_subtract_ctf_projection:
                # RELION's VDAM/--grad path backprojects the residual image,
                # not the raw unmasked image: Fimg_store = Fimg - Frefctf.
                if proj_for_noise is None:
                    raise RuntimeError("Residual local M-step requires materialized recon projections")
                frefctf_weighted = proj_for_noise * ctf2_over_nv_recon[:, None, :]
                summed = summed - reconstruction_probs_sum_t[..., None] * frefctf_weighted
            if return_profile:
                _block_until_ready(
                    summed,
                    ctf_probs,
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    reconstruction_probs,
                    reconstruction_rotation_mask,
                    n_significant_samples,
                    best_argmax,
                    log_Z,
                    best_log_score,
                    max_posterior,
                )
            debug_fused_posterior_dump_targets = maybe_write_debug_fused_posterior_dump(
                experiment_dataset=experiment_dataset,
                local_layout=local_layout,
                bucket=bucket,
                image_pre_shifts=image_pre_shifts,
                probs=probs,
                log_Z=log_Z,
                best_log_score=best_log_score,
                best_argmax=best_argmax,
                max_posterior=max_posterior,
                reconstruction_sample_mask=reconstruction_sample_mask,
                reconstruction_rotation_mask=reconstruction_rotation_mask,
                n_significant_samples=n_significant_samples,
                current_size=current_size,
                debug_iteration=debug_iteration,
                dump_dir=debug_fused_posterior_dump_dir,
                pending_targets=debug_fused_posterior_dump_targets,
                requested_current_sizes=debug_fused_posterior_dump_current_sizes,
                requested_iterations=debug_fused_posterior_dump_iterations,
            )
            fused_elapsed = time.time() - fused_t0
            timing.fused_score_mstep_s += fused_elapsed
        elif can_use_fused_score_mstep and not score_only:
            fused_t0 = time.time()
            if normalization_log_evidence_np is None:
                bucket_log_z = jnp.asarray(
                    normalization_log_z_np[np.asarray(bucket.image_indices)],
                    dtype=precision_policy.normalization_real_dtype,
                )
            else:
                normalization_dtype = precision_policy.normalization_real_dtype
                log_score_offset = (-0.5 * jnp.squeeze(batch_norm, axis=1)).astype(normalization_dtype)
                bucket_log_z = (
                    jnp.asarray(
                        normalization_log_evidence_np[np.asarray(bucket.image_indices)],
                        dtype=normalization_dtype,
                    )
                    - log_score_offset
                )
            (
                log_Z,
                probs,
                best_log_score,
                best_argmax,
                max_posterior,
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
                reconstruction_probs,
                probs_sum_t,
                reconstruction_probs_sum_t,
            ) = fused_score_normalize_support_probs_abs2_with_log_z_on_demand(
                shifted_score_split,
                ctf2_over_nv_score,
                proj_weighted,
                half_weights_windowed if use_window else half_weights,
                local_rotation_log_prior,
                jnp.asarray(bucket.translation_log_prior),
                jnp.asarray(bucket.local_rotation_mask),
                None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask),
                bucket_log_z,
                half_spectrum_scoring=half_spectrum_scoring,
                reconstruct_significant_only=reconstruct_significant_only,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            summed = None
            ctf_probs = None
            defer_packed_mstep_reduction = True
            if return_profile:
                _block_until_ready(
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    reconstruction_probs,
                    reconstruction_rotation_mask,
                    n_significant_samples,
                    best_argmax,
                    log_Z,
                    best_log_score,
                    max_posterior,
                )
            debug_fused_posterior_dump_targets = maybe_write_debug_fused_posterior_dump(
                experiment_dataset=experiment_dataset,
                local_layout=local_layout,
                bucket=bucket,
                image_pre_shifts=image_pre_shifts,
                probs=probs,
                log_Z=log_Z,
                best_log_score=best_log_score,
                best_argmax=best_argmax,
                max_posterior=max_posterior,
                reconstruction_sample_mask=reconstruction_sample_mask,
                reconstruction_rotation_mask=reconstruction_rotation_mask,
                n_significant_samples=n_significant_samples,
                current_size=current_size,
                debug_iteration=debug_iteration,
                dump_dir=debug_fused_posterior_dump_dir,
                pending_targets=debug_fused_posterior_dump_targets,
                requested_current_sizes=debug_fused_posterior_dump_current_sizes,
                requested_iterations=debug_fused_posterior_dump_iterations,
            )
            fused_elapsed = time.time() - fused_t0
            timing.fused_score_mstep_s += fused_elapsed
        else:
            score_t0 = time.time()
            if half_spectrum_scoring:
                scores = score_local_bucket_abs2_on_demand(
                    shifted_score_split,
                    ctf2_over_nv_score,
                    proj_weighted,
                    local_rotation_log_prior,
                    jnp.asarray(bucket.translation_log_prior),
                    jnp.asarray(bucket.local_rotation_mask),
                    None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask),
                )
            else:
                score_half_weights = half_weights_windowed if use_window else half_weights
                scores = score_local_bucket_abs2_weighted_on_demand(
                    shifted_score_split,
                    ctf2_over_nv_score,
                    proj_weighted,
                    score_half_weights,
                    local_rotation_log_prior,
                    jnp.asarray(bucket.translation_log_prior),
                    jnp.asarray(bucket.local_rotation_mask),
                    None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask),
                )
            if return_profile:
                _block_until_ready(scores)
            timing.score_s += time.time() - score_t0

            normalize_t0 = time.time()
            if normalization_log_z_np is None and normalization_log_evidence_np is None:
                if use_float64_normalization:
                    log_Z, probs, best_log_score, best_argmax, max_posterior = normalize_local_scores(scores)
                else:
                    log_Z, probs, best_log_score, best_argmax, max_posterior = normalize_local_scores_float32(scores)
            else:
                if normalization_log_evidence_np is None:
                    bucket_log_z = jnp.asarray(
                        normalization_log_z_np[np.asarray(bucket.image_indices)],
                        dtype=scores.real.dtype,
                    )
                else:
                    normalization_dtype = precision_policy.normalization_real_dtype
                    log_score_offset = (-0.5 * jnp.squeeze(batch_norm, axis=1)).astype(normalization_dtype)
                    bucket_log_z = (
                        jnp.asarray(
                            normalization_log_evidence_np[np.asarray(bucket.image_indices)],
                            dtype=normalization_dtype,
                        )
                        - log_score_offset
                    )
                if use_float64_normalization:
                    log_Z, probs, best_log_score, best_argmax, max_posterior = normalize_local_scores_with_log_z(
                        scores,
                        bucket_log_z,
                    )
                else:
                    log_Z, probs, best_log_score, best_argmax, max_posterior = (
                        normalize_local_scores_with_log_z_float32(
                            scores,
                            bucket_log_z,
                        )
                    )
            if return_profile:
                _block_until_ready(log_Z, probs, best_log_score, best_argmax, max_posterior)
            timing.normalize_s += time.time() - normalize_t0

            significance_t0 = time.time()
            if reconstruct_significant_only and use_relion_f32_fine_posterior:
                if threshold_for_bucket is not None:
                    raise ValueError(
                        "RELION float32 fine posterior does not accept an external "
                        "reconstruction threshold"
                    )
                (
                    reconstruction_probs,
                    reconstruction_sample_mask,
                    n_significant_samples,
                    _relion_sum_weight,
                    _relion_significant_weight,
                ) = _sparse_pass2_diagnostics._relion_f32_fine_reconstruction_probs(
                    scores,
                    adaptive_fraction=adaptive_fraction,
                )
                reconstruction_rotation_mask = jnp.any(
                    reconstruction_sample_mask,
                    axis=-1,
                )
                max_posterior = jnp.max(
                    reconstruction_probs.reshape(reconstruction_probs.shape[0], -1),
                    axis=1,
                )
            elif reconstruct_significant_only:
                if threshold_for_bucket is None:
                    reconstruction_sample_mask, reconstruction_rotation_mask, n_significant_samples = (
                        compute_reconstruction_support(
                            probs,
                            adaptive_fraction=adaptive_fraction,
                            max_significants=max_significants,
                        )
                    )
                else:
                    reconstruction_sample_mask, reconstruction_rotation_mask, n_significant_samples = (
                        compute_reconstruction_support_from_threshold(
                            probs,
                            threshold_for_bucket,
                        )
                    )
                reconstruction_probs = jnp.where(reconstruction_sample_mask, probs, 0.0)
            else:
                reconstruction_rotation_mask = jnp.asarray(bucket.local_rotation_mask)
                reconstruction_sample_mask = jnp.broadcast_to(
                    reconstruction_rotation_mask[:, :, None],
                    probs.shape,
                )
                n_significant_samples = jnp.sum(reconstruction_rotation_mask, axis=1).astype(jnp.int32) * n_trans
                reconstruction_probs = probs
            if return_profile:
                _block_until_ready(reconstruction_probs, reconstruction_rotation_mask, n_significant_samples)
            timing.significance_s += time.time() - significance_t0

            debug_score_dump_targets = maybe_write_debug_score_dump(
                experiment_dataset=experiment_dataset,
                local_layout=local_layout,
                bucket=bucket,
                image_pre_shifts=image_pre_shifts,
                scores=scores,
                probs=probs,
                log_Z=log_Z,
                best_log_score=best_log_score,
                max_posterior=max_posterior,
                reconstruction_sample_mask=reconstruction_sample_mask,
                reconstruction_rotation_mask=reconstruction_rotation_mask,
                n_significant_samples=n_significant_samples,
                current_size=current_size,
                debug_iteration=debug_iteration,
                shifted_score_split=shifted_score.reshape(batch_size, n_trans, -1),
                shifted_recon_split=shifted_recon_split,
                ctf2_over_nv_score=ctf2_over_nv_score,
                ctf2_over_nv_recon=ctf2_over_nv_recon,
                proj_weighted=proj_weighted,
                proj_for_noise=proj_for_noise,
                proj_abs2_weighted=None,
                dump_dir=debug_score_dump_dir,
                pending_targets=debug_score_dump_targets,
                requested_current_sizes=debug_score_dump_current_sizes,
                requested_iterations=debug_score_dump_iterations,
            )

            mstep_t0 = time.time()
            probs_sum_t = jnp.sum(probs, axis=-1)
            reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
            if score_only:
                summed = None
                ctf_probs = None
                if return_profile:
                    _block_until_ready(probs_sum_t, reconstruction_probs_sum_t)
            else:
                summed = compute_local_weighted_sums(reconstruction_probs, shifted_recon_split)
                ctf_probs = compute_local_ctf_sums(reconstruction_probs, ctf2_over_nv_recon)
                if mstep_subtract_ctf_projection:
                    # RELION's VDAM/--grad path backprojects the residual image,
                    # not the raw unmasked image: Fimg_store = Fimg - Frefctf.
                    if proj_for_noise is None:
                        raise RuntimeError("Residual local M-step requires materialized recon projections")
                    frefctf_weighted = proj_for_noise * ctf2_over_nv_recon[:, None, :]
                    summed = summed - reconstruction_probs_sum_t[..., None] * frefctf_weighted
                if return_profile:
                    _block_until_ready(summed, ctf_probs, probs_sum_t, reconstruction_probs_sum_t)
            timing.mstep_s += time.time() - mstep_t0

            if bpref_contribution_capture_active and not score_only:
                candidate_mask = jnp.broadcast_to(
                    jnp.asarray(bucket.local_rotation_mask)[:, :, None],
                    probs.shape,
                )
                if bucket.local_sample_mask is not None:
                    candidate_mask = candidate_mask & jnp.asarray(bucket.local_sample_mask)
                rotation_prior = local_rotation_log_prior
                translation_prior = jnp.asarray(bucket.translation_log_prior)
                preprior_scores = scores - rotation_prior[:, :, None] - translation_prior[:, None, :]
                preprior_scores = jnp.where(
                    candidate_mask & jnp.isfinite(preprior_scores),
                    preprior_scores,
                    -jnp.inf,
                )
                reconstruction_threshold_for_dump = (
                    jnp.zeros((batch_size,), dtype=jnp.float64)
                    if threshold_for_bucket is None
                    else threshold_for_bucket
                )
                _maybe_dump_exact_local_bpref_contribution_rows(
                    experiment_dataset=experiment_dataset,
                    image_indices=bucket.image_indices,
                    current_size=current_size,
                    summed=summed,
                    ctf_probs=ctf_probs,
                    rotations=_local_mstep_rotations(bucket),
                    actual_counts=bucket.actual_rotation_counts,
                    rotation_indices=bucket.local_rotation_ids,
                    fine_translations=local_layout.translation_grid,
                    scores=scores,
                    preprior_scores=preprior_scores,
                    probs=probs,
                    rotation_log_prior=rotation_prior,
                    translation_log_prior=translation_prior,
                    log_z=log_Z,
                    best_log_score=best_log_score,
                    reconstruction_probs=reconstruction_probs,
                    reconstruction_mask=reconstruction_sample_mask,
                    reconstruction_sum_weight=jnp.sum(reconstruction_probs, axis=(1, 2)),
                    reconstruction_threshold=reconstruction_threshold_for_dump,
                    candidate_mask=candidate_mask,
                    high_precision_operand_bundle=False,
                    raw_batch_data=None,
                    ctf_params=None,
                    noise_variance_half=None,
                    integer_pre_shifts=None,
                    batch_image_corrections=None,
                    batch_scale_corrections=None,
                    relion_preprocess_normalization_factors=None,
                    relion_cuda_preprocess=False,
                    score_with_masked_images=score_with_masked_images,
                    image_mask=None,
                    image_mask_mode="not-captured",
                    voxel_size=experiment_dataset.voxel_size,
                    ctf_mode="not-captured",
                    ctf_dose_per_tilt=0.0,
                    ctf_angle_per_tilt=0.0,
                    disc_type=disc_type,
                    projection_padding_factor=projection_padding_factor,
                    reconstruction_padding_factor=reconstruction_padding_factor,
                    use_relion_x_half_mstep=mstep_relion_x_half,
                    winner_take_all=False,
                    max_r=mstep_adjoint_max_r,
                    window_indices=mstep_recon_window_indices,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    shadow_only_mode=False,
                    shadow_score_bitwise_equal=True,
                    shadow_reduction_agreement=None,
                )
            scores = None

        _collect_reconstruction_probability_values(bucket.image_indices, probs)

        pack_t0 = time.time()
        probs_sum_t_np = None
        if reconstruct_significant_only:
            reconstruction_rotation_mask_np = np.asarray(reconstruction_rotation_mask, dtype=bool)
            transfer_profile["reconstruction_mask_to_host_s"] += time.time() - pack_t0
        else:
            reconstruction_rotation_mask_np = np.asarray(bucket.local_rotation_mask, dtype=bool)
        transfer_t0 = time.time()
        probs_sum_t_np = np.asarray(probs_sum_t, dtype=np.float64)
        stats_probs_sum_t_np = (
            np.asarray(reconstruction_probs_sum_t, dtype=np.float64)
            if stats_use_reconstruction_probs
            else probs_sum_t_np
        )
        transfer_profile["mstep_posterior_sum_to_host_s"] += time.time() - transfer_t0
        (
            reconstruction_take_indices,
            reconstruction_pack_mask_np,
            reconstruction_counts_np,
            reconstruction_row_count,
        ) = _build_nonzero_reconstruction_pack_indices(
            reconstruction_rotation_mask_np,
            np.asarray(bucket.local_rotation_mask, dtype=bool),
            probs_sum_t_np,
            rotation_block_size,
        )
        reconstruction_take_indices_jnp = jnp.asarray(reconstruction_take_indices, dtype=jnp.int32)
        reconstruction_pack_mask_jnp = jnp.asarray(reconstruction_pack_mask_np)
        packed_rotations_np = np.take_along_axis(
            np.asarray(bucket.local_rotations, dtype=np.float32),
            reconstruction_take_indices[:, :, None, None],
            axis=1,
        )
        packed_mstep_rotations_np = np.take_along_axis(
            _local_mstep_rotations(bucket),
            reconstruction_take_indices[:, :, None, None],
            axis=1,
        )
        packed_reconstruction_probs = None
        if score_only:
            packed_summed = None
            packed_ctf_probs = None
        elif defer_packed_mstep_reduction:
            packed_reconstruction_probs = jnp.take_along_axis(
                reconstruction_probs,
                reconstruction_take_indices_jnp[:, :, None],
                axis=1,
            )
            packed_reconstruction_probs = jnp.where(
                reconstruction_pack_mask_jnp[:, :, None],
                packed_reconstruction_probs,
                0.0,
            )
            if mstep_subtract_ctf_projection:
                # RELION's VDAM/--grad path backprojects the residual image,
                # not the raw unmasked image: Fimg_store = Fimg - Frefctf.
                if proj_for_noise is None:
                    raise RuntimeError("Residual local M-step requires materialized recon projections")
            packed_summed = None
            packed_ctf_probs = None
        else:
            packed_summed = jnp.take_along_axis(summed, reconstruction_take_indices_jnp[:, :, None], axis=1)
            packed_summed = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_summed, 0.0)
            packed_ctf_probs = jnp.take_along_axis(ctf_probs, reconstruction_take_indices_jnp[:, :, None], axis=1)
            packed_ctf_probs = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_ctf_probs, 0.0)
        packed_flat_rotations = None
        if (not defer_packed_mstep_reduction) and (
            not disable_adjoint_y or not disable_adjoint_ctf or (accumulate_noise and proj_for_noise is None)
        ):
            packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_mstep_rotations_np))
        timing.pack_s += time.time() - pack_t0

        if defer_packed_mstep_reduction and not score_only and (not disable_adjoint_y or not disable_adjoint_ctf):
            if packed_reconstruction_probs is None:
                raise RuntimeError("packed posterior rows are required for deferred local M-step")
            packed_rotation_count = int(packed_rotations_np.shape[1])
            n_recon_pixels = window_spec.n_recon if window_spec.use_window else int(n_half)
            chunk_rows = min(
                packed_rotation_count,
                _packed_noise_projection_chunk_rows(n_recon_pixels, batch_size=batch_size),
            )
            if chunk_rows < packed_rotation_count and not logged_deferred_mstep_chunking:
                logger.info(
                    "Exact local packed M-step chunking: packed_rows=%d chunk_rows=%d n_recon_pixels=%d batch_size=%d",
                    packed_rotation_count,
                    chunk_rows,
                    n_recon_pixels,
                    batch_size,
                )
                logged_deferred_mstep_chunking = True
            for chunk_start in range(0, packed_rotation_count, chunk_rows):
                chunk_stop = min(packed_rotation_count, chunk_start + chunk_rows)
                chunk_probs = packed_reconstruction_probs[:, chunk_start:chunk_stop]
                chunk_rotations = packed_mstep_rotations_np[:, chunk_start:chunk_stop]
                chunk_flat_rotations = flatten_bucket_rotations(jnp.asarray(chunk_rotations))
                chunk_summed = None
                chunk_ctf_probs = None
                if not disable_adjoint_y:
                    mstep_t0 = time.time()
                    chunk_summed = compute_local_weighted_sums(chunk_probs, shifted_recon_split)
                    if mstep_subtract_ctf_projection:
                        chunk_take_indices = reconstruction_take_indices_jnp[:, chunk_start:chunk_stop]
                        chunk_pack_mask = reconstruction_pack_mask_jnp[:, chunk_start:chunk_stop]
                        chunk_proj_for_residual = jnp.take_along_axis(
                            proj_for_noise,
                            chunk_take_indices[:, :, None],
                            axis=1,
                        )
                        chunk_proj_for_residual = jnp.where(
                            chunk_pack_mask[:, :, None],
                            chunk_proj_for_residual,
                            0.0,
                        )
                        chunk_probs_sum_t = jnp.sum(chunk_probs, axis=-1)
                        frefctf_weighted = chunk_proj_for_residual * ctf2_over_nv_recon[:, None, :]
                        chunk_summed = chunk_summed - chunk_probs_sum_t[..., None] * frefctf_weighted
                    if return_profile:
                        _block_until_ready(chunk_summed)
                    timing.mstep_s += time.time() - mstep_t0

                    adjoint_y_t0 = time.time()
                    Ft_y = _adjoint_slice_volume_maybe_windowed(
                        flatten_bucket_rows(chunk_summed),
                        mstep_recon_window_indices,
                        chunk_flat_rotations,
                        Ft_y,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                        True,
                        use_window=use_window,
                        max_r=mstep_adjoint_max_r,
                        relion_x_half=bool(mstep_relion_x_half),
                    )
                    if return_profile:
                        _block_until_ready(Ft_y)
                    timing.adjoint_y_s += time.time() - adjoint_y_t0

                if not disable_adjoint_ctf:
                    mstep_t0 = time.time()
                    chunk_ctf_probs = compute_local_ctf_sums(chunk_probs, ctf2_over_nv_recon)
                    if return_profile:
                        _block_until_ready(chunk_ctf_probs)
                    timing.mstep_s += time.time() - mstep_t0

                    adjoint_ctf_t0 = time.time()
                    Ft_ctf = _adjoint_slice_volume_maybe_windowed(
                        flatten_bucket_rows(chunk_ctf_probs),
                        mstep_recon_window_indices,
                        chunk_flat_rotations,
                        Ft_ctf,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                        True,
                        use_window=use_window,
                        max_r=mstep_adjoint_max_r,
                        relion_x_half=bool(mstep_relion_x_half),
                    )
                    if return_profile:
                        _block_until_ready(Ft_ctf)
                    timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

        elif not disable_adjoint_y:
            adjoint_y_t0 = time.time()
            Ft_y = _adjoint_slice_volume_maybe_windowed(
                flatten_bucket_rows(packed_summed),
                mstep_recon_window_indices,
                packed_flat_rotations,
                Ft_y,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                True,
                use_window=use_window,
                max_r=mstep_adjoint_max_r,
                relion_x_half=bool(mstep_relion_x_half),
            )
            if return_profile:
                _block_until_ready(Ft_y)
            timing.adjoint_y_s += time.time() - adjoint_y_t0

        if (not defer_packed_mstep_reduction) and not disable_adjoint_ctf:
            adjoint_ctf_t0 = time.time()
            Ft_ctf = _adjoint_slice_volume_maybe_windowed(
                flatten_bucket_rows(packed_ctf_probs),
                mstep_recon_window_indices,
                packed_flat_rotations,
                Ft_ctf,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                True,
                use_window=use_window,
                max_r=mstep_adjoint_max_r,
                relion_x_half=bool(mstep_relion_x_half),
            )
            if return_profile:
                _block_until_ready(Ft_ctf)
            timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

        if accumulate_noise:
            noise_t0 = time.time()
            support_mass = jnp.sum(reconstruction_probs.reshape(batch_size, -1), axis=1).astype(jnp.float32)
            if translation_sqdist_ang is not None:
                translation_posterior = jnp.sum(reconstruction_probs, axis=1).astype(jnp.float32)
                noise_sumw_offset = jnp.sum(
                    translation_posterior * jnp.asarray(translation_sqdist_ang, dtype=jnp.float32),
                )
            else:
                noise_sumw_offset = jnp.asarray(0.0, dtype=jnp.float32)
            processed_noise_power_half = processed_score_half
            if image_only_corr is not None:
                processed_noise_power_half = processed_noise_power_half * image_only_corr[:, None]
            batch_img_power = jnp.sum(
                (jnp.abs(processed_noise_power_half) ** 2) * support_mass[:, None],
                axis=0,
            ).astype(jnp.float32)
            batch_img_power_per_image = _norm_correction_image_power_per_image(
                processed_noise_power_half,
                support_mass,
                shell_indices_half,
                jnp.ones_like(support_mass, dtype=bool),
                norm_unweighted_shell_cutoff,
                shell_count=n_shells,
                image_shape=image_shape,
                current_size=current_size,
                include_unweighted_high_shell=include_unweighted_norm_high_shell,
            )
            batch_img_power_shells = bin_shell_values_jax(batch_img_power, shell_indices_half, n_shells)
            noise_img_power = noise_img_power + batch_img_power_shells
            noise_sumw = noise_sumw + jnp.sum(support_mass)

            shifted_noise_split = shifted_noise.reshape(batch_size, n_trans, -1)
            packed_summed_masked_noise = None
            if defer_packed_mstep_reduction:
                if packed_reconstruction_probs is None:
                    raise RuntimeError("packed posterior rows are required for deferred local noise accumulation")
                if debug_noise_dump_dir is not None and debug_noise_dump_targets:
                    summed_masked_noise = compute_local_weighted_sums(reconstruction_probs, shifted_noise_split)
                    ctf_probs_for_debug = (
                        ctf_probs
                        if ctf_probs is not None
                        else compute_local_ctf_sums(reconstruction_probs, ctf2_over_nv_recon)
                    )
                    debug_noise_dump_targets = maybe_write_debug_noise_component_dump(
                        experiment_dataset=experiment_dataset,
                        bucket=bucket,
                        support_mass=support_mass,
                        processed_noise_power_half=processed_noise_power_half,
                        proj_for_noise=proj_for_noise,
                        proj_abs2_for_noise=None,
                        summed_masked_noise=summed_masked_noise,
                        ctf_probs=ctf_probs_for_debug,
                        noise_variance_for_noise=noise_variance_for_noise,
                        shell_indices_half=shell_indices_half,
                        shell_indices_noise=shell_indices_noise,
                        n_shells=n_shells,
                        current_size=current_size,
                        debug_iteration=debug_iteration,
                        reconstruction_sample_mask=reconstruction_sample_mask,
                        n_significant_samples=n_significant_samples,
                        dump_dir=debug_noise_dump_dir,
                        pending_targets=debug_noise_dump_targets,
                        requested_current_sizes=debug_noise_dump_current_sizes,
                        requested_iterations=debug_noise_dump_iterations,
                    )
            else:
                summed_masked_noise = compute_local_weighted_sums(reconstruction_probs, shifted_noise_split)
                debug_noise_dump_targets = maybe_write_debug_noise_component_dump(
                    experiment_dataset=experiment_dataset,
                    bucket=bucket,
                    support_mass=support_mass,
                    processed_noise_power_half=processed_noise_power_half,
                    proj_for_noise=proj_for_noise,
                    proj_abs2_for_noise=None,
                    summed_masked_noise=summed_masked_noise,
                    ctf_probs=ctf_probs,
                    noise_variance_for_noise=noise_variance_for_noise,
                    shell_indices_half=shell_indices_half,
                    shell_indices_noise=shell_indices_noise,
                    n_shells=n_shells,
                    current_size=current_size,
                    debug_iteration=debug_iteration,
                    reconstruction_sample_mask=reconstruction_sample_mask,
                    n_significant_samples=n_significant_samples,
                    dump_dir=debug_noise_dump_dir,
                    pending_targets=debug_noise_dump_targets,
                    requested_current_sizes=debug_noise_dump_current_sizes,
                    requested_iterations=debug_noise_dump_iterations,
                )
                packed_summed_masked_noise = jnp.take_along_axis(
                    summed_masked_noise,
                    reconstruction_take_indices_jnp[:, :, None],
                    axis=1,
                )
                packed_summed_masked_noise = jnp.where(
                    reconstruction_pack_mask_jnp[:, :, None],
                    packed_summed_masked_noise,
                    0.0,
                )
            block_noise_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            block_a2_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            block_xa_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            block_norm_residual = jnp.zeros(batch_size, dtype=jnp.float32)
            if proj_for_noise is None:
                packed_rotation_count = int(packed_rotations_np.shape[1])
                n_recon_pixels = window_spec.n_recon if window_spec.use_window else int(n_half)
                noise_projection_pixels = int(n_half) if relion_projector_half is not None else int(n_recon_pixels)
                chunk_rows = min(
                    packed_rotation_count,
                    _packed_noise_projection_chunk_rows(noise_projection_pixels, batch_size=batch_size),
                )
                if chunk_rows < packed_rotation_count and not logged_deferred_noise_projection_chunking:
                    logger.info(
                        "Exact local noise projection chunking: packed_rows=%d chunk_rows=%d n_recon_pixels=%d projection_pixels=%d batch_size=%d",
                        packed_rotation_count,
                        chunk_rows,
                        n_recon_pixels,
                        noise_projection_pixels,
                        batch_size,
                    )
                    logged_deferred_noise_projection_chunking = True
                for chunk_start in range(0, packed_rotation_count, chunk_rows):
                    chunk_stop = min(packed_rotation_count, chunk_start + chunk_rows)
                    chunk_rotations = packed_rotations_np[:, chunk_start:chunk_stop]
                    chunk_proj_for_noise = _project_packed_noise_rows(
                        mean_for_proj=mean_for_proj,
                        packed_flat_rotations=flatten_bucket_rotations(jnp.asarray(chunk_rotations)),
                        packed_rotation_count=chunk_stop - chunk_start,
                        batch_size=batch_size,
                        image_shape=image_shape,
                        proj_volume_shape=proj_volume_shape,
                        disc_type=disc_type,
                        projection_kwargs=projection_kwargs,
                        window_spec=window_spec,
                        n_half=n_half,
                        precision_policy=precision_policy,
                        reconstruction_pack_mask_jnp=reconstruction_pack_mask_jnp[:, chunk_start:chunk_stop],
                        relion_projector_half=relion_projector_half,
                        relion_projector_r_max=relion_projector_r_max,
                        projection_padding_factor=projection_padding_factor,
                    )
                    if defer_packed_mstep_reduction:
                        chunk_probs = packed_reconstruction_probs[:, chunk_start:chunk_stop]
                        chunk_summed_masked_noise = compute_local_weighted_sums(chunk_probs, shifted_noise_split)
                        chunk_ctf_probs = compute_local_ctf_sums(chunk_probs, ctf2_over_nv_recon)
                    else:
                        chunk_summed_masked_noise = packed_summed_masked_noise[:, chunk_start:chunk_stop]
                        chunk_ctf_probs = packed_ctf_probs[:, chunk_start:chunk_stop]
                    flat_proj_for_noise = flatten_bucket_rows(chunk_proj_for_noise)
                    flat_proj_abs2_for_noise = jnp.abs(flat_proj_for_noise) ** 2
                    chunk_noise_shells, chunk_a2_shells, chunk_xa_shells = _compute_noise_block(
                        flat_proj_for_noise,
                        flat_proj_abs2_for_noise,
                        flatten_bucket_rows(chunk_summed_masked_noise),
                        flatten_bucket_rows(chunk_ctf_probs),
                        noise_variance_for_noise,
                        shell_indices_noise,
                        n_shells,
                        return_noise_split,
                    )
                    block_noise_shells = block_noise_shells + chunk_noise_shells
                    block_a2_shells = block_a2_shells + chunk_a2_shells
                    block_xa_shells = block_xa_shells + chunk_xa_shells
                    chunk_proj_abs2_for_norm = flat_proj_abs2_for_noise.reshape(chunk_proj_for_noise.shape)
                    block_norm_residual = block_norm_residual + _compute_norm_residual_per_image(
                        chunk_proj_for_noise,
                        chunk_proj_abs2_for_norm,
                        chunk_summed_masked_noise,
                        chunk_ctf_probs,
                        noise_variance_for_noise,
                    )
                    if noise_scale_xa is not None:
                        scale_xa_per_image, scale_aa_per_image = _compute_scale_correction_terms_per_image(
                            chunk_proj_for_noise,
                            chunk_proj_abs2_for_norm,
                            chunk_summed_masked_noise,
                            chunk_ctf_probs,
                            noise_variance_for_noise,
                            batch_scale,
                            scale_correction_pixel_mask,
                        )
                        noise_scale_xa = noise_scale_xa.at[bucket_group_ids].add(scale_xa_per_image)
                        noise_scale_aa = noise_scale_aa.at[bucket_group_ids].add(scale_aa_per_image)
            else:
                packed_rotation_count = int(reconstruction_take_indices_jnp.shape[1])
                noise_projection_pixels = int(proj_for_noise.shape[-1])
                chunk_rows = min(
                    packed_rotation_count,
                    _packed_noise_projection_chunk_rows(noise_projection_pixels, batch_size=batch_size),
                )
                if chunk_rows < packed_rotation_count and not logged_cached_noise_projection_chunking:
                    logger.info(
                        "Exact local cached noise projection chunking: packed_rows=%d chunk_rows=%d projection_pixels=%d batch_size=%d",
                        packed_rotation_count,
                        chunk_rows,
                        noise_projection_pixels,
                        batch_size,
                    )
                    logged_cached_noise_projection_chunking = True
                for chunk_start in range(0, packed_rotation_count, chunk_rows):
                    chunk_stop = min(packed_rotation_count, chunk_start + chunk_rows)
                    chunk_take_indices = reconstruction_take_indices_jnp[:, chunk_start:chunk_stop]
                    chunk_pack_mask = reconstruction_pack_mask_jnp[:, chunk_start:chunk_stop]
                    chunk_proj_for_noise = jnp.take_along_axis(
                        proj_for_noise,
                        chunk_take_indices[:, :, None],
                        axis=1,
                    )
                    chunk_proj_for_noise = jnp.where(
                        chunk_pack_mask[:, :, None],
                        chunk_proj_for_noise,
                        0.0,
                    )
                    if defer_packed_mstep_reduction:
                        chunk_probs = packed_reconstruction_probs[:, chunk_start:chunk_stop]
                        chunk_summed_masked_noise = compute_local_weighted_sums(chunk_probs, shifted_noise_split)
                        chunk_ctf_probs = compute_local_ctf_sums(chunk_probs, ctf2_over_nv_recon)
                    else:
                        chunk_summed_masked_noise = packed_summed_masked_noise[:, chunk_start:chunk_stop]
                        chunk_ctf_probs = packed_ctf_probs[:, chunk_start:chunk_stop]
                    flat_proj_for_noise = flatten_bucket_rows(chunk_proj_for_noise)
                    flat_proj_abs2_for_noise = jnp.abs(flat_proj_for_noise) ** 2
                    chunk_noise_shells, chunk_a2_shells, chunk_xa_shells = _compute_noise_block(
                        flat_proj_for_noise,
                        flat_proj_abs2_for_noise,
                        flatten_bucket_rows(chunk_summed_masked_noise),
                        flatten_bucket_rows(chunk_ctf_probs),
                        noise_variance_for_noise,
                        shell_indices_noise,
                        n_shells,
                        return_noise_split,
                    )
                    block_noise_shells = block_noise_shells + chunk_noise_shells
                    block_a2_shells = block_a2_shells + chunk_a2_shells
                    block_xa_shells = block_xa_shells + chunk_xa_shells
                    chunk_proj_abs2_for_norm = flat_proj_abs2_for_noise.reshape(chunk_proj_for_noise.shape)
                    block_norm_residual = block_norm_residual + _compute_norm_residual_per_image(
                        chunk_proj_for_noise,
                        chunk_proj_abs2_for_norm,
                        chunk_summed_masked_noise,
                        chunk_ctf_probs,
                        noise_variance_for_noise,
                    )
                    if noise_scale_xa is not None:
                        scale_xa_per_image, scale_aa_per_image = _compute_scale_correction_terms_per_image(
                            chunk_proj_for_noise,
                            chunk_proj_abs2_for_norm,
                            chunk_summed_masked_noise,
                            chunk_ctf_probs,
                            noise_variance_for_noise,
                            batch_scale,
                            scale_correction_pixel_mask,
                        )
                        noise_scale_xa = noise_scale_xa.at[bucket_group_ids].add(scale_xa_per_image)
                        noise_scale_aa = noise_scale_aa.at[bucket_group_ids].add(scale_aa_per_image)
            if return_profile:
                _block_until_ready(block_noise_shells, block_norm_residual)
            noise_wsum = noise_wsum + block_noise_shells
            if return_noise_split:
                noise_a2 = noise_a2 + block_a2_shells
                noise_xa = noise_xa + block_xa_shells
            noise_norm_correction = noise_norm_correction.at[jnp.asarray(bucket.image_indices, dtype=jnp.int32)].add(
                batch_img_power_per_image + block_norm_residual,
            )
            noise_sigma2_offset = noise_sigma2_offset + noise_sumw_offset
            timing.noise_s += time.time() - noise_t0

        postprocess_t0 = time.time()
        significant_sample_count, reconstruction_row_count = _postprocess_local_bucket(
            image_indices=bucket.image_indices,
            local_rotation_ids=bucket.local_rotation_ids,
            local_rotation_mask=bucket.local_rotation_mask,
            local_rotations=bucket.local_rotations,
            local_rotation_posterior_ids=bucket.local_rotation_posterior_ids,
            translation_grid=local_layout.translation_grid,
            n_trans=n_trans,
            best_argmax=best_argmax,
            batch_norm=batch_norm,
            log_Z=log_Z,
            best_log_score=best_log_score,
            max_posterior=max_posterior,
            probs_sum_t=stats_probs_sum_t_np,
            n_significant_samples=n_significant_samples,
            reconstruction_sample_mask=reconstruction_sample_mask,
            collect_profile_stats=collect_profile_stats,
            reconstruction_row_count=reconstruction_row_count,
            reconstruction_take_indices=reconstruction_take_indices,
            reconstruction_pack_mask=reconstruction_pack_mask_np,
            buffers=postprocess_buffers,
        )
        if collect_profile_stats:
            total_significant_samples += significant_sample_count
            total_reconstruction_rows += int(reconstruction_row_count)
        timing.postprocess_s += time.time() - postprocess_t0

        host_stats_t0 = time.time()
        logger.debug(
            "Exact local bucket: %d images, bucket_rot=%d, total_local_rot=%d",
            batch_size,
            int(bucket.bucket_rotation_count),
            int(np.sum(bucket.actual_rotation_counts)),
        )
        timing.host_stats_s += time.time() - host_stats_t0
        _mark_exact_local_bucket_done(bucket)
        if debug_score_dump_force_split and debug_score_dump_bucket_matches:
            cleanup_t0 = time.time()
            shifted_half = None
            shifted_recon_half = None
            shifted_score = None
            shifted_recon = None
            shifted_noise = None
            ctf2_over_nv_score = None
            ctf2_over_nv_recon = None
            projection_block = None
            proj_weighted = None
            proj_for_noise = None
            shifted_score_split = None
            shifted_recon_split = None
            scores = None
            probs = None
            reconstruction_probs = None
            reconstruction_sample_mask = None
            reconstruction_rotation_mask = None
            gc.collect()
            jax.clear_caches()
            timing.host_stats_s += time.time() - cleanup_t0

    _log_exact_local_progress(force=True, done=True)
    final_accumulator_t0 = time.time()
    if not score_only:
        Ft_y, Ft_ctf = enforce_half_volume_x0(
            Ft_y,
            Ft_ctf,
            recon_volume_shape,
            logger=logger,
            label="Exact local",
            force_host=host_accumulator_finalize,
        )
        if return_half_volume_accumulators:
            logger.info("Exact local M-step: keeping native half-volume accumulators for downstream reconstruction")
        elif mstep_relion_x_half:
            Ft_y, Ft_ctf = relion_x_half_accumulators_to_public_layout(
                Ft_y,
                Ft_ctf,
                recon_volume_shape,
                force_host=host_accumulator_finalize,
            )
        else:
            Ft_y, Ft_ctf = half_volume_accumulators_to_full(Ft_y, Ft_ctf, recon_volume_shape)

        if return_profile:
            _block_until_ready(Ft_y, Ft_ctf)
    timing.final_accumulator_s += time.time() - final_accumulator_t0

    stats_finalize_t0 = time.time()
    relion_stats = make_relion_stats(
        log_evidence_per_image=log_evidence_per_image,
        best_log_score_per_image=best_log_score_per_image,
        max_posterior_per_image=max_posterior_per_image,
        rotation_posterior_sums=rotation_posterior_sums,
    )
    noise_stats = None
    if accumulate_noise:
        transfer_t0 = time.time()
        noise_sigma2_offset_value = float(np.asarray(noise_sigma2_offset, dtype=np.float64))
        noise_sumw_value = float(np.asarray(noise_sumw, dtype=np.float64))
        transfer_profile["final_noise_to_host_s"] += time.time() - transfer_t0
        noise_stats = make_noise_stats(
            wsum_sigma2_noise=noise_wsum,
            wsum_img_power=noise_img_power,
            wsum_sigma2_offset=noise_sigma2_offset_value,
            sumw=noise_sumw_value,
            wsum_noise_a2=(noise_a2 if return_noise_split else None),
            wsum_noise_xa=(noise_xa if return_noise_split else None),
            wsum_norm_correction=noise_norm_correction,
            wsum_scale_correction_xa=noise_scale_xa,
            wsum_scale_correction_aa=noise_scale_aa,
        )
    timing.stats_finalize_s += time.time() - stats_finalize_t0

    if debug_score_dump_filter_matches and debug_score_dump_targets and debug_score_dump_iterations is None:
        logger.warning(
            "Requested local score dump indices were not observed in this dataset view: %s",
            sorted(debug_score_dump_targets),
        )
    if (
        debug_fused_posterior_dump_filter_matches
        and debug_fused_posterior_dump_targets
        and debug_fused_posterior_dump_iterations is None
    ):
        logger.warning(
            "Requested fused posterior dump indices were not observed in this dataset view: %s",
            sorted(debug_fused_posterior_dump_targets),
        )

    if reconstruct_significant_only:
        logger.info(
            "Exact local significant-support summary: chunks=%d big_jit_buckets=%d "
            "sparse_big_jit_buckets=%d reconstruction_rows=%d padded_rows=%d "
            "significant_samples=%d mean_reconstruction_rows_per_image=%.2f "
            "mean_significant_samples_per_image=%.2f",
            n_chunks,
            big_jit_bucket_count,
            sparse_big_jit_bucket_count,
            total_reconstruction_rows,
            total_padded_rotations,
            total_significant_samples,
            0.0 if n_images == 0 else total_reconstruction_rows / n_images,
            0.0 if n_images == 0 else total_significant_samples / n_images,
        )

    if not return_profile:
        return _local_em_return_tuple(
            Ft_y,
            Ft_ctf,
            hard_assignment,
            relion_stats,
            accumulate_noise=accumulate_noise,
            return_profile=False,
            return_best_pose_details=return_best_pose_details,
            return_significant_counts=return_significant_counts,
            best_pose_rotations=best_pose_rotations,
            best_pose_translations=best_pose_translations,
            best_pose_rotation_ids=best_pose_rotation_ids,
            noise_stats=noise_stats,
            significant_counts=significant_counts,
        )

    _block_until_ready(Ft_y, Ft_ctf)
    total_wall_time = time.time() - overall_t0
    profile_summary = {
        "big_jit_bucket_count": np.int32(big_jit_bucket_count),
        "sparse_big_jit_bucket_count": np.int32(sparse_big_jit_bucket_count),
        "big_jit_debug_bucket_count": np.int32(big_jit_debug_bucket_count),
        "score_only": np.asarray(score_only),
        "fused_score_mstep_enabled": np.asarray(fused_score_mstep_enabled),
        "defer_local_noise_projection": np.asarray(defer_local_noise_projection),
        "bucket_build_time_s": np.float64(timing.bucket_build_s),
        "raw_cache_build_time_s": np.float64(timing.raw_cache_build_s),
        "raw_cache_enabled": np.asarray(raw_cache_enabled),
        "processed_half_cache_enabled": np.asarray(processed_half_cache_enabled),
        "relion_projection_cache_enabled": np.asarray(relion_projection_cache_groups_built > 0),
        "relion_projection_cache_groups": np.int64(len(relion_projection_cache_groups)),
        "relion_projection_cache_groups_built": np.int64(relion_projection_cache_groups_built),
        "relion_projection_cache_rows": np.int64(relion_projection_cache_max_rows),
        "relion_projection_cache_capacity_rows": np.int64(relion_projection_cache_capacity_rows),
        "relion_projection_cache_id_map_rows": np.int64(relion_projection_cache_id_map_rows),
        "relion_projection_cache_pixels": np.int64(relion_projection_cache_n_projection_pixels),
        "relion_projection_cache_estimated_gb": np.float64(relion_projection_cache_max_estimated_gb),
        "relion_projection_cache_build_s": np.float64(relion_projection_cache_total_build_s),
        "relion_projection_cache_cap_gb": np.float64(relion_projection_cache_cap_gb),
        "batch_fetch_time_s": np.float64(timing.batch_fetch_s),
        "preprocess_time_s": np.float64(timing.preprocess_s),
        **_prefixed_timer_profile("preprocess_", preprocess_profile),
        **_prefixed_timer_profile("transfer_", transfer_profile),
        "transfer_total_to_host_s": np.float64(sum(transfer_profile.values())),
        **_local_timing_profile(timing),
        "em_time_s": np.float64(total_wall_time),
        "accounted_em_time_s": np.float64(timing.accounted_s()),
        "unattributed_em_time_s": np.float64(max(total_wall_time - timing.accounted_s(), 0.0)),
        "n_chunks": np.int32(n_chunks),
        "projection_mode": np.asarray(projection_mode),
        "n_projection_windowed": np.int32(window_spec.n_projection),
        "chunk_sizes": np.asarray(chunk_sizes, dtype=np.int32),
        "chunk_local_rotations": np.asarray(chunk_local_rotations, dtype=np.int32),
        "chunk_padded_rotations": np.asarray(chunk_padded_rotations, dtype=np.int32),
        "chunk_unique_rotations": np.asarray(chunk_unique_rotations, dtype=np.int32),
        "chunk_nonzero_posterior_rows": np.asarray(chunk_nonzero_posterior_rows, dtype=np.int32),
        "chunk_reconstruction_rows": np.asarray(chunk_reconstruction_rows, dtype=np.int32),
        "chunk_significant_samples": np.asarray(chunk_significant_samples, dtype=np.int32),
        "sum_union_rows": np.int64(total_local_rotations),
        "sum_padded_rows": np.int64(total_padded_rotations),
        "sum_nonzero_posterior_rows": np.int64(np.sum(chunk_nonzero_posterior_rows)),
        "sum_reconstruction_rows": np.int64(total_reconstruction_rows),
        "sum_significant_samples": np.int64(total_significant_samples),
        "unique_global_rotations": np.int64(np.count_nonzero(seen_global_rotations)),
        "unique_nonzero_global_rotations": np.int64(np.count_nonzero(seen_nonzero_global_rotations)),
        "unique_reconstruction_global_rotations": np.int64(np.count_nonzero(seen_reconstruction_global_rotations)),
        "duplicate_rotation_factor": np.float64(
            0.0
            if not np.any(seen_global_rotations)
            else total_local_rotations / np.count_nonzero(seen_global_rotations)
        ),
        "reconstruction_duplicate_rotation_factor": np.float64(
            0.0
            if not np.any(seen_reconstruction_global_rotations)
            else total_reconstruction_rows / np.count_nonzero(seen_reconstruction_global_rotations)
        ),
        "local_total_hypotheses": np.int64(local_total_hypotheses),
        "local_mean_rotations_per_image": np.float64(0.0 if n_images == 0 else total_local_rotations / n_images),
        "local_mean_reconstruction_rows_per_image": np.float64(
            0.0 if n_images == 0 else total_reconstruction_rows / n_images
        ),
        "local_mean_significant_samples_per_image": np.float64(
            0.0 if n_images == 0 else total_significant_samples / n_images
        ),
        "local_num_buckets": np.int32(n_chunks),
        "max_hypotheses_per_microbatch": np.int64(max_hypotheses_per_microbatch),
        "sparse_adjoint_target_rows": np.int64(sparse_adjoint_target_rows),
        "sparse_adjoint_chunk_count": np.int64(sparse_adjoint_chunk_count),
        "local_pad_fraction": np.float64(
            0.0 if total_padded_rotations == 0 else 1.0 - total_local_rotations / total_padded_rotations
        ),
        "n_windowed": np.int32(n_windowed),
    }
    if reconstruction_probability_values_by_image is not None:
        profile_summary["reconstruction_probability_values_by_image"] = tuple(
            np.concatenate(values).astype(np.float32, copy=False) if values else np.zeros(0, dtype=np.float32)
            for values in reconstruction_probability_values_by_image
        )
    if reconstruction_sample_indices_by_image is not None:
        profile_summary["reconstruction_sample_indices_by_image"] = tuple(reconstruction_sample_indices_by_image)
    return _local_em_return_tuple(
        Ft_y,
        Ft_ctf,
        hard_assignment,
        relion_stats,
        accumulate_noise=accumulate_noise,
        return_profile=True,
        return_best_pose_details=return_best_pose_details,
        return_significant_counts=return_significant_counts,
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
        noise_stats=noise_stats,
        profile_summary=profile_summary,
        significant_counts=significant_counts,
    )
