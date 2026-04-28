"""Exact per-image local EM engine for RELION-mode local search."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.configs import ForwardModelConfig
from recovar.reconstruction import noise as noise_utils
from recovar.em.dense_single_volume.em_primitives import (
    _adjoint_slice_volume_half,
    _adjoint_slice_volume_windowed,
    _batch_adjoint_slice_volume_half,
    _batch_adjoint_slice_volume_windowed,
    _block_until_ready,
    _compute_noise_block,
    _compute_projections_block,
)
from recovar.em.dense_single_volume.helpers.env_flags import (
    parse_env_auto_bool,
    parse_env_bool,
)
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
    make_shell_indices_half,
)
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    integer_pre_shifts_or_none,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    apply_half_translation_phases as _apply_half_translation_phases,
    half_translation_phase_table as _half_translation_phase_table,
    image_preprocess_backend as _image_preprocess_backend,
    process_half_image,
    try_process_masked_and_unmasked_half_together as _try_process_masked_and_unmasked_half_together,
)
from recovar.em.dense_single_volume.helpers.types import NoiseStats, RelionStats
from recovar.em.dense_single_volume.local_debug import (
    maybe_write_debug_noise_component_dump,
    maybe_write_debug_score_dump,
    noise_split_diagnostics_requested,
    parse_debug_noise_component_dump_request,
    parse_debug_score_dump_request,
)
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_weighted_sums,
    enforce_relion_half_volume_x0_hermitian,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.dense_single_volume.local_big_jit import run_local_bucket_big_jit
from recovar.em.dense_single_volume.local_layout import (
    LocalBucketSpec,
    LocalHypothesisLayout,
    _exact_bucket_rotation_size,
    bucket_local_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_score_pass import (
    compute_reconstruction_support,
    fused_score_normalize_mstep_abs2_on_demand,
    normalize_local_scores_auto,
    score_local_bucket_abs2_on_demand,
    score_local_bucket_abs2_weighted_on_demand,
    score_local_bucket,
)
from recovar.em.dense_single_volume.helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from recovar.em.dense_single_volume.shape_buckets import pad_axis, pad_batch_data_ctf_and_valid_mask

logger = logging.getLogger(__name__)

_LOCAL_PREPROCESS_TIMER_KEYS = (
    "integer_shift_s",
    "translation_phase_s",
    "processed_cache_gather_s",
    "combined_process_s",
    "score_process_s",
    "recon_process_s",
    "ctf_s",
    "tile_shift_score_s",
    "tile_shift_recon_s",
    "norm_s",
)

_LOCAL_TRANSFER_TIMER_KEYS = (
    "reconstruction_mask_to_host_s",
    "mstep_posterior_sum_to_host_s",
    "postprocess_argmax_to_host_s",
    "postprocess_scores_to_host_s",
    "postprocess_posterior_to_host_s",
    "final_noise_to_host_s",
)

_LOCAL_TIMING_PROFILE_FIELDS = (
    ("projection_time_s", "projection_s"),
    ("big_jit_bucket_s", "big_jit_bucket_s"),
    ("fused_score_mstep_s", "fused_score_mstep_s"),
    ("local_score_s", "score_s"),
    ("local_normalize_s", "normalize_s"),
    ("local_significance_s", "significance_s"),
    ("local_mstep_s", "mstep_s"),
    ("local_pack_s", "pack_s"),
    ("local_backproject_y_s", "adjoint_y_s"),
    ("local_backproject_ctf_s", "adjoint_ctf_s"),
    ("local_noise_s", "noise_s"),
    ("local_postprocess_s", "postprocess_s"),
    ("local_host_stats_s", "host_stats_s"),
    ("local_final_accumulator_s", "final_accumulator_s"),
    ("local_stats_finalize_s", "stats_finalize_s"),
)

_LOCAL_ACCOUNTED_TIMING_SETUP_FIELDS = (
    "bucket_build_s",
    "raw_cache_build_s",
    "processed_cache_build_s",
    "batch_fetch_s",
    "preprocess_s",
)

_LOCAL_ACCOUNTED_TIMING_FIELDS = _LOCAL_ACCOUNTED_TIMING_SETUP_FIELDS + tuple(
    timing_attr for _, timing_attr in _LOCAL_TIMING_PROFILE_FIELDS
)


def _new_zero_timer(keys):
    return {key: 0.0 for key in keys}


@dataclass
class _LocalTiming:
    """Mutable host-side timers for one exact-local EM call."""

    bucket_build_s: float = 0.0
    batch_fetch_s: float = 0.0
    raw_cache_build_s: float = 0.0
    processed_cache_build_s: float = 0.0
    preprocess_s: float = 0.0
    projection_s: float = 0.0
    fused_score_mstep_s: float = 0.0
    big_jit_bucket_s: float = 0.0
    score_s: float = 0.0
    normalize_s: float = 0.0
    significance_s: float = 0.0
    postprocess_s: float = 0.0
    mstep_s: float = 0.0
    pack_s: float = 0.0
    adjoint_y_s: float = 0.0
    adjoint_ctf_s: float = 0.0
    noise_s: float = 0.0
    host_stats_s: float = 0.0
    final_accumulator_s: float = 0.0
    stats_finalize_s: float = 0.0

    def accounted_s(self) -> float:
        return sum(
            getattr(self, timing_attr)
            for timing_attr in _LOCAL_ACCOUNTED_TIMING_FIELDS
        )


@dataclass(frozen=True)
class _LocalSpectrumSetup:
    half_weights: jax.Array
    norm_half_weights: jax.Array
    half_weights_windowed: jax.Array
    noise_variance_half: jax.Array
    materialize_projection_abs2: bool


@dataclass(frozen=True)
class _BigJitMaskArgs:
    mask_mode: str
    image_mask: jax.Array


@dataclass(frozen=True)
class _BigJitNoiseArgs:
    noise_wsum: jax.Array
    noise_img_power: jax.Array
    noise_a2: jax.Array
    noise_xa: jax.Array
    shell_indices_half: jax.Array
    shell_indices_noise: jax.Array
    noise_variance_for_noise: jax.Array
    n_shells: int


@dataclass(frozen=True)
class _LocalReconstructionPack:
    local_mask: np.ndarray
    take_indices: np.ndarray
    pack_mask: np.ndarray
    row_count: int


@dataclass(frozen=True)
class _LocalPackSelection:
    rotation_mask: np.ndarray
    take_indices: np.ndarray
    pack_mask: np.ndarray
    row_count: int
    probs_sum_t: np.ndarray


def _pad_local_big_jit_image_axis(bucket: LocalBucketSpec, batch_data, ctf_params):
    """Pad a local big-JIT bucket to its planned image shape class."""

    actual_batch_size = int(bucket.image_indices.shape[0])
    padded_batch_size = int(max(actual_batch_size, getattr(bucket, "bucket_image_count", actual_batch_size)))
    if actual_batch_size == padded_batch_size:
        return bucket, batch_data, ctf_params, np.ones(actual_batch_size, dtype=bool), actual_batch_size

    padded_rotations = pad_axis(bucket.local_rotations, 0, padded_batch_size, value=0).astype(np.float32)
    padded_rotations[actual_batch_size:] = np.eye(3, dtype=np.float32)
    padded_bucket = LocalBucketSpec(
        image_indices=np.asarray(bucket.image_indices, dtype=np.int32),
        bucket_image_count=padded_batch_size,
        bucket_rotation_count=int(bucket.bucket_rotation_count),
        actual_rotation_counts=pad_axis(bucket.actual_rotation_counts, 0, padded_batch_size, value=0).astype(np.int32),
        local_rotation_ids=pad_axis(bucket.local_rotation_ids, 0, padded_batch_size, value=-1).astype(np.int32),
        local_rotations=padded_rotations,
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


def _exact_local_max_hypotheses_per_microbatch(default: int | None, n_windowed: int) -> int:
    """Return exact-local microbatch cap, optionally overridden for profiling.

    The automatic default targets the proven 5k/128 local-search working set
    while scaling down for larger Fourier windows.
    """
    raw = os.environ.get("RECOVAR_RELION_EXACT_LOCAL_MAX_HYPOTHESES_PER_MICROBATCH")
    if raw is None or raw.strip() == "":
        if default is not None:
            return int(default)
        target_row_pixels = int(
            os.environ.get("RECOVAR_RELION_EXACT_LOCAL_TARGET_ROW_PIXELS", "180000000"),
        )
        value = target_row_pixels // max(1, int(n_windowed))
        return int(max(8192, min(65536, value)))
    value = int(raw)
    if value <= 0:
        raise ValueError(
            "RECOVAR_RELION_EXACT_LOCAL_MAX_HYPOTHESES_PER_MICROBATCH must be positive",
        )
    return value


def _make_local_spectrum_setup(
    image_shape,
    n_half: int,
    noise_variance,
    window_spec,
    *,
    half_spectrum_scoring: bool,
) -> _LocalSpectrumSetup:
    if half_spectrum_scoring:
        half_weights = jnp.ones(n_half, dtype=jnp.float32)
    else:
        half_weights = make_half_image_weights(image_shape)
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    return _LocalSpectrumSetup(
        half_weights=half_weights,
        norm_half_weights=make_half_image_weights(image_shape),
        half_weights_windowed=window_spec.score_values(half_weights),
        noise_variance_half=noise_variance_half,
        materialize_projection_abs2=_local_materialize_projection_abs2_enabled(False),
    )


def _local_big_jit_mask_args(experiment_dataset, image_shape) -> _BigJitMaskArgs:
    backend = _image_preprocess_backend(experiment_dataset)
    mask_mode = getattr(backend, "image_mask_mode", "multiply")
    if mask_mode not in {"relion_background_fill", "multiply"}:
        mask_mode = "none"

    image_mask = getattr(backend, "image_mask", None)
    if image_mask is None:
        image_mask = getattr(backend, "mask", None)
    if image_mask is None:
        image_mask = getattr(experiment_dataset, "image_mask", None)
    if image_mask is None:
        image_mask = np.ones(image_shape, dtype=np.float32)
        mask_mode = "none"

    return _BigJitMaskArgs(mask_mode=mask_mode, image_mask=jnp.asarray(image_mask))


def _local_big_jit_noise_args(
    *,
    accumulate_noise: bool,
    noise_wsum,
    noise_img_power,
    noise_a2,
    noise_xa,
    shell_indices_half,
    shell_indices_noise,
    noise_variance_for_noise,
    n_shells: int,
    noise_variance_half,
) -> _BigJitNoiseArgs:
    """Return concrete noise arrays for the big-JIT call boundary."""

    if accumulate_noise:
        return _BigJitNoiseArgs(
            noise_wsum=noise_wsum,
            noise_img_power=noise_img_power,
            noise_a2=noise_a2,
            noise_xa=noise_xa,
            shell_indices_half=shell_indices_half,
            shell_indices_noise=shell_indices_noise,
            noise_variance_for_noise=noise_variance_for_noise,
            n_shells=int(n_shells),
        )

    disabled = jnp.zeros(1, dtype=jnp.float32)
    disabled_shell_indices = jnp.zeros(noise_variance_half.shape[0], dtype=jnp.int32)
    return _BigJitNoiseArgs(
        noise_wsum=disabled,
        noise_img_power=disabled,
        noise_a2=disabled,
        noise_xa=disabled,
        shell_indices_half=disabled_shell_indices,
        shell_indices_noise=disabled_shell_indices,
        noise_variance_for_noise=noise_variance_half,
        n_shells=1,
    )


def _can_use_local_big_jit_bucket(
    *,
    big_jit_enabled: bool,
    processed_cache_enabled: bool,
    batch_data,
    materialize_projection_abs2: bool,
    debug_score_dump_filter_matches: bool,
    debug_noise_dump_dir,
) -> bool:
    return (
        big_jit_enabled
        and not processed_cache_enabled
        and batch_data is not None
        and not materialize_projection_abs2
        and not debug_score_dump_filter_matches
        and debug_noise_dump_dir is None
    )


def _local_big_jit_reconstruction_pack(
    bucket: LocalBucketSpec,
    reconstruction_rotation_mask,
    reconstruction_row_count,
    unpadded_batch_size: int,
) -> _LocalReconstructionPack:
    local_mask = np.asarray(bucket.local_rotation_mask, dtype=bool)[:unpadded_batch_size]
    reconstruction_mask = np.asarray(reconstruction_rotation_mask, dtype=bool)[:unpadded_batch_size]
    take_indices = np.broadcast_to(
        np.arange(int(bucket.bucket_rotation_count), dtype=np.int32)[None, :],
        (unpadded_batch_size, int(bucket.bucket_rotation_count)),
    )
    return _LocalReconstructionPack(
        local_mask=local_mask,
        take_indices=take_indices,
        pack_mask=reconstruction_mask & local_mask,
        row_count=int(np.asarray(reconstruction_row_count, dtype=np.int32)),
    )


def _local_batch_adjoint_pair(
    summed_rows,
    ctf_rows,
    rotations,
    Ft_y,
    Ft_ctf,
    recon_window_indices,
    image_shape,
    recon_volume_shape,
    *,
    use_window: bool,
    use_native_half_volume_mstep: bool,
    current_size,
):
    rows = jnp.stack([summed_rows, ctf_rows], axis=0)
    volumes = jnp.stack([Ft_y, Ft_ctf], axis=0)
    if use_window:
        updated = _batch_adjoint_slice_volume_windowed(
            rows,
            recon_window_indices,
            rotations,
            volumes,
            image_shape,
            recon_volume_shape,
            "linear_interp",
            True,
            use_native_half_volume_mstep,
            float(current_size // 2),
        )
    else:
        updated = _batch_adjoint_slice_volume_half(
            rows,
            rotations,
            volumes,
            image_shape,
            recon_volume_shape,
            "linear_interp",
            True,
            use_native_half_volume_mstep,
        )
    return updated[0], updated[1]


def _local_adjoint_one(
    rows,
    rotations,
    volume,
    recon_window_indices,
    image_shape,
    recon_volume_shape,
    *,
    use_window: bool,
    use_native_half_volume_mstep: bool,
    current_size,
):
    if use_window:
        return _adjoint_slice_volume_windowed(
            rows,
            recon_window_indices,
            rotations,
            volume,
            image_shape,
            recon_volume_shape,
            "linear_interp",
            True,
            use_native_half_volume_mstep,
            float(current_size // 2),
        )
    return _adjoint_slice_volume_half(
        rows,
        rotations,
        volume,
        image_shape,
        recon_volume_shape,
        "linear_interp",
        True,
        use_native_half_volume_mstep,
    )


def _accumulate_local_adjoint_rows(
    *,
    packed_summed_rows,
    packed_ctf_rows,
    packed_flat_rotations,
    Ft_y,
    Ft_ctf,
    recon_window_indices,
    image_shape,
    recon_volume_shape,
    use_window: bool,
    use_native_half_volume_mstep: bool,
    current_size,
    batch_backproject_enabled: bool,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    return_profile: bool,
    timing: _LocalTiming,
):
    if batch_backproject_enabled and not disable_adjoint_y and not disable_adjoint_ctf:
        adjoint_y_t0 = time.time()
        Ft_y, Ft_ctf = _local_batch_adjoint_pair(
            packed_summed_rows,
            packed_ctf_rows,
            packed_flat_rotations,
            Ft_y,
            Ft_ctf,
            recon_window_indices,
            image_shape,
            recon_volume_shape,
            use_window=use_window,
            use_native_half_volume_mstep=use_native_half_volume_mstep,
            current_size=current_size,
        )
        if return_profile:
            _block_until_ready(Ft_y, Ft_ctf)
        timing.adjoint_y_s += time.time() - adjoint_y_t0
        return Ft_y, Ft_ctf

    if not disable_adjoint_y:
        adjoint_y_t0 = time.time()
        Ft_y = _local_adjoint_one(
            packed_summed_rows,
            packed_flat_rotations,
            Ft_y,
            recon_window_indices,
            image_shape,
            recon_volume_shape,
            use_window=use_window,
            use_native_half_volume_mstep=use_native_half_volume_mstep,
            current_size=current_size,
        )
        if return_profile:
            _block_until_ready(Ft_y)
        timing.adjoint_y_s += time.time() - adjoint_y_t0

    if not disable_adjoint_ctf:
        adjoint_ctf_t0 = time.time()
        Ft_ctf = _local_adjoint_one(
            packed_ctf_rows,
            packed_flat_rotations,
            Ft_ctf,
            recon_window_indices,
            image_shape,
            recon_volume_shape,
            use_window=use_window,
            use_native_half_volume_mstep=use_native_half_volume_mstep,
            current_size=current_size,
        )
        if return_profile:
            _block_until_ready(Ft_ctf)
        timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

    return Ft_y, Ft_ctf


def _fetch_indexed_batch(experiment_dataset, image_indices):
    batch_iter = experiment_dataset.iter_batches(
        len(image_indices),
        indices=np.asarray(image_indices),
        by_image=False,
    )
    batch_data, _, _, ctf_params, _, _, indices = next(batch_iter)
    return batch_data, ctf_params, np.asarray(indices)

def _local_raw_cache_enabled(n_images: int, image_shape, dtype) -> bool:
    parsed = parse_env_auto_bool("RECOVAR_RELION_EXACT_LOCAL_RAW_CACHE", default="auto")
    if parsed is not None:
        return parsed
    max_gb = float(os.environ.get("RECOVAR_RELION_EXACT_LOCAL_RAW_CACHE_MAX_GB", "2.0"))
    bytes_per_pixel = np.dtype(dtype).itemsize if dtype is not None else np.dtype(np.float32).itemsize
    estimated_gb = int(n_images) * int(np.prod(image_shape)) * bytes_per_pixel / 1e9
    return estimated_gb <= max_gb


def _local_processed_cache_enabled(n_images: int, image_shape, score_with_masked_images: bool) -> bool:
    parsed = parse_env_auto_bool("RECOVAR_RELION_EXACT_LOCAL_PROCESSED_CACHE", default="0")
    if parsed is not None:
        return parsed
    max_gb = float(os.environ.get("RECOVAR_RELION_EXACT_LOCAL_PROCESSED_CACHE_MAX_GB", "2.0"))
    n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    n_copies = 2 if score_with_masked_images else 1
    estimated_gb = int(n_images) * n_half * np.dtype(np.complex64).itemsize * n_copies / 1e9
    return estimated_gb <= max_gb


def _local_batch_backproject_enabled() -> bool:
    return parse_env_bool("RECOVAR_RELION_EXACT_LOCAL_BATCH_BACKPROJECT", default=False)


def _local_big_jit_enabled() -> bool:
    """Return whether exact-local should try the large fused JIT path.

    Default is on; explicit false-like values keep the older local branches
    available for parity bisects and fallback debugging.
    """

    return parse_env_bool("RECOVAR_RELION_EXACT_LOCAL_BIG_JIT", default=True)


def _local_compact_zero_posterior_rows_enabled() -> bool:
    return parse_env_bool("RECOVAR_RELION_EXACT_LOCAL_COMPACT_ZERO_POSTERIOR_ROWS", default=True)


def _local_combined_masked_preprocess_enabled() -> bool:
    return parse_env_bool("RECOVAR_RELION_EXACT_LOCAL_COMBINED_MASKED_PREPROCESS", default=False)


def _local_materialize_projection_abs2_enabled(default: bool) -> bool:
    parsed = parse_env_auto_bool("RECOVAR_RELION_EXACT_LOCAL_MATERIALIZE_PROJECTION_ABS2", default="auto")
    return bool(default) if parsed is None else parsed


def _local_keep_half_volume_accumulators_enabled() -> bool:
    return parse_env_bool("RECOVAR_RELION_EXACT_LOCAL_KEEP_HALF_VOLUME_ACCUMULATORS", default=False)


def _local_fused_score_mstep_enabled(default: bool = False) -> bool:
    parsed = parse_env_auto_bool("RECOVAR_RELION_EXACT_LOCAL_FUSED_SCORE_MSTEP", default="auto")
    return bool(default) if parsed is None else parsed


def _build_local_raw_cache(experiment_dataset, n_images: int):
    """Fetch all local images/CTF rows once for exact local search.

    The local engine visits every image exactly once, but bucket sorting turns
    that into many small indexed dataset reads. Caching raw images preserves the
    per-bucket preprocessing behavior while avoiding repeated source lookups.
    """

    indices = np.arange(int(n_images), dtype=np.int32)
    batch_data, ctf_params, fetched_indices = _fetch_indexed_batch(experiment_dataset, indices)
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


def _processed_cache_shifted_raw(raw_cache, image_pre_shifts, n_images: int):
    if image_pre_shifts is None:
        return raw_cache, False, True
    if getattr(raw_cache, "ndim", np.asarray(raw_cache).ndim) != 3:
        return raw_cache, False, False
    shifts = np.asarray(image_pre_shifts, dtype=np.float32)[np.arange(int(n_images), dtype=np.int32)]
    if shifts.size == 0:
        return raw_cache, True, True
    rounded = np.rint(shifts)
    row_integral = np.all(np.isclose(shifts, rounded, rtol=0.0, atol=1e-6), axis=1)
    if np.all(row_integral):
        return apply_relion_integer_pre_shifts(raw_cache, rounded.astype(np.int32)), True, True
    if not np.any(row_integral):
        return raw_cache, False, True
    return raw_cache, False, False


def _build_processed_half_cache(
    experiment_dataset,
    raw_cache,
    image_pre_shifts,
    n_images: int,
    score_with_masked_images: bool,
):
    raw_for_processing, real_space_pre_shift_applied, can_cache = _processed_cache_shifted_raw(
        raw_cache,
        image_pre_shifts,
        n_images,
    )
    if not can_cache:
        return None, None, False

    score_half = process_half_image(
        experiment_dataset,
        raw_for_processing,
        score_with_masked_images,
    )
    if score_with_masked_images:
        recon_half = process_half_image(
            experiment_dataset,
            raw_for_processing,
            False,
        )
    else:
        recon_half = score_half

    return jnp.asarray(score_half), jnp.asarray(recon_half), real_space_pre_shift_applied


def _new_local_preprocess_timer():
    return _new_zero_timer(_LOCAL_PREPROCESS_TIMER_KEYS)


def _new_local_transfer_timer():
    return _new_zero_timer(_LOCAL_TRANSFER_TIMER_KEYS)


def _prefixed_timer_profile(prefix: str, timer: dict[str, float]) -> dict[str, np.float64]:
    return {f"{prefix}{key}": np.float64(value) for key, value in timer.items()}


def _local_timing_profile(timing: _LocalTiming) -> dict[str, np.float64]:
    return {
        output_key: np.float64(getattr(timing, timing_attr))
        for output_key, timing_attr in _LOCAL_TIMING_PROFILE_FIELDS
    }


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
    batch_size: int,
    n_trans: int,
    score_with_masked_images: bool,
    image_pre_shifts=None,
    processed_score_half=None,
    processed_recon_half=None,
    real_space_pre_shift_applied_cache: bool = False,
    timer: dict[str, float] | None = None,
    synchronize_profile: bool = False,
    combined_masked_preprocess: bool = False,
):
    """Prepare score, reconstruction, and noise inputs for one local bucket.

    This keeps the exact-local path separate from the dense engine and avoids
    recomputing CTF / translation tiling scaffolding across masked, unmasked,
    and noise-specific preprocessing.
    """

    using_processed_cache = processed_score_half is not None
    if using_processed_cache:
        integer_pre_shifts = None
        real_space_pre_shift_applied = bool(real_space_pre_shift_applied_cache)
    else:
        integer_t0 = time.time()
        integer_pre_shifts = integer_pre_shifts_or_none(image_pre_shifts, image_indices, batch=batch)
        if integer_pre_shifts is not None:
            batch = apply_relion_integer_pre_shifts(batch, integer_pre_shifts)
        if timer is not None:
            timer["integer_shift_s"] += time.time() - integer_t0
        real_space_pre_shift_applied = integer_pre_shifts is not None

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

    def _process_half(apply_image_mask: bool):
        return process_half_image(
            experiment_dataset,
            batch,
            apply_image_mask,
        )

    ctf_t0 = time.time()
    ctf_half = config.compute_ctf_half(ctf_params)
    ctf2_over_nv_half = ctf_half**2 / noise_variance_half
    if synchronize_profile:
        _block_until_ready(ctf2_over_nv_half)
    if timer is not None:
        timer["ctf_s"] += time.time() - ctf_t0

    combined_processed = False
    if (
        combined_masked_preprocess
        and score_with_masked_images
        and not using_processed_cache
    ):
        combined_t0 = time.time()
        combined_halves = _try_process_masked_and_unmasked_half_together(experiment_dataset, batch)
        if combined_halves is not None:
            processed_score_half, processed_recon_half = combined_halves
            combined_processed = True
            if synchronize_profile:
                _block_until_ready(processed_score_half, processed_recon_half)
            if timer is not None:
                timer["combined_process_s"] += time.time() - combined_t0

    score_process_t0 = time.time()
    if not using_processed_cache and not combined_processed:
        processed_score_half = _process_half(score_with_masked_images)
    if synchronize_profile:
        _block_until_ready(processed_score_half)
    if timer is not None:
        if using_processed_cache:
            timer["processed_cache_gather_s"] += time.time() - score_process_t0
        elif not combined_processed:
            timer["score_process_s"] += time.time() - score_process_t0

    shift_score_t0 = time.time()
    score_weighted_half = processed_score_half * ctf_half / noise_variance_half
    shifted_score_half = _apply_half_translation_phases(score_weighted_half, translation_phases_half)
    if synchronize_profile:
        _block_until_ready(shifted_score_half)
    if timer is not None:
        timer["tile_shift_score_s"] += time.time() - shift_score_t0

    norm_t0 = time.time()
    batch_norm = jnp.sum(
        (jnp.abs(processed_score_half) ** 2 / noise_variance_half) * norm_half_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real
    if synchronize_profile:
        _block_until_ready(batch_norm)
    if timer is not None:
        timer["norm_s"] += time.time() - norm_t0

    if score_with_masked_images:
        recon_process_t0 = time.time()
        if not using_processed_cache and not combined_processed:
            processed_recon_half = _process_half(False)
        if synchronize_profile:
            _block_until_ready(processed_recon_half)
        if timer is not None:
            if using_processed_cache:
                timer["processed_cache_gather_s"] += time.time() - recon_process_t0
            elif not combined_processed:
                timer["recon_process_s"] += time.time() - recon_process_t0

        shift_recon_t0 = time.time()
        recon_weighted_half = processed_recon_half * ctf_half / noise_variance_half
        shifted_recon_half = _apply_half_translation_phases(recon_weighted_half, translation_phases_half)
        if synchronize_profile:
            _block_until_ready(shifted_recon_half)
        if timer is not None:
            timer["tile_shift_recon_s"] += time.time() - shift_recon_t0
    else:
        shifted_recon_half = shifted_score_half
    return (
        shifted_score_half,
        shifted_recon_half,
        batch_norm,
        ctf2_over_nv_half,
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
    packed_rotation_count = _exact_bucket_rotation_size(max_count, rotation_block_size)
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


def _select_local_reconstruction_pack(
    *,
    bucket: LocalBucketSpec,
    reconstruction_rotation_mask,
    probs_sum_t,
    reconstruct_significant_only: bool,
    compact_zero_posterior_rows: bool,
    rotation_block_size: int,
    transfer_profile: dict[str, float],
    pack_start_time: float,
) -> _LocalPackSelection:
    local_rotation_mask = np.asarray(bucket.local_rotation_mask, dtype=bool)
    probs_sum_t_np = None
    if reconstruct_significant_only:
        reconstruction_rotation_mask_np = np.asarray(reconstruction_rotation_mask, dtype=bool)
        transfer_profile["reconstruction_mask_to_host_s"] += time.time() - pack_start_time
    else:
        reconstruction_rotation_mask_np = local_rotation_mask

    if compact_zero_posterior_rows:
        transfer_t0 = time.time()
        probs_sum_t_np = np.asarray(probs_sum_t, dtype=np.float64)
        transfer_profile["mstep_posterior_sum_to_host_s"] += time.time() - transfer_t0
        take_indices, pack_mask, _, row_count = _build_nonzero_reconstruction_pack_indices(
            reconstruction_rotation_mask_np,
            local_rotation_mask,
            probs_sum_t_np,
            rotation_block_size,
        )
    elif reconstruct_significant_only:
        take_indices, pack_mask, _, row_count = _build_reconstruction_pack_indices(
            reconstruction_rotation_mask_np,
            local_rotation_mask,
            rotation_block_size,
        )
    else:
        take_indices = np.broadcast_to(
            np.arange(int(bucket.bucket_rotation_count), dtype=np.int32)[None, :],
            (int(bucket.image_indices.shape[0]), int(bucket.bucket_rotation_count)),
        )
        pack_mask = reconstruction_rotation_mask_np
        row_count = int(np.sum(np.asarray(bucket.actual_rotation_counts, dtype=np.int32), dtype=np.int64))

    if probs_sum_t_np is None:
        probs_sum_t_np = np.asarray(probs_sum_t, dtype=np.float64)
    return _LocalPackSelection(
        rotation_mask=reconstruction_rotation_mask_np,
        take_indices=take_indices,
        pack_mask=pack_mask,
        row_count=row_count,
        probs_sum_t=probs_sum_t_np,
    )


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
    accumulate_noise: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
    score_with_masked_images: bool = True,
    half_spectrum_scoring: bool = False,
    use_float64_scoring: bool = False,
    use_float64_normalization: bool = True,
    use_float64_projections: bool = False,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    image_corrections: np.ndarray | None = None,
    scale_corrections: np.ndarray | None = None,
    image_pre_shifts: np.ndarray | None = None,
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
    translation_prior_centers: np.ndarray | None = None,
):
    """Run exact local EM over per-image local hypothesis sets."""

    overall_t0 = time.time()
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    n_trans = int(local_layout.translation_grid.shape[0])
    n_images = int(local_layout.n_images)
    normalization_log_z_np = None
    if normalization_log_z is not None:
        normalization_log_z_np = np.asarray(normalization_log_z, dtype=np.float64)
        if normalization_log_z_np.shape != (n_images,):
            raise ValueError(
                "normalization_log_z must have shape "
                f"({n_images},), got {normalization_log_z_np.shape}",
            )
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
        debug_noise_dump_dir,
        debug_noise_dump_targets,
        debug_noise_dump_current_sizes,
        debug_noise_dump_iterations,
    ) = parse_debug_noise_component_dump_request()
    debug_score_dump_filter_matches = (
        debug_score_dump_dir is not None
        and (
            debug_score_dump_current_sizes is None
            or int(current_size or -1) in debug_score_dump_current_sizes
        )
        and (
            debug_score_dump_iterations is None
            or int(debug_iteration or -1) in debug_score_dump_iterations
        )
    )
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            mean,
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=current_size,
        )
    else:
        mean_for_proj = mean
        proj_volume_shape = volume_shape

    if use_float64_projections:
        mean_for_proj = jnp.asarray(mean_for_proj, dtype=jnp.complex128)

    if reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
    else:
        recon_volume_shape = volume_shape
    use_native_half_volume_mstep = os.environ.get(
        "RECOVAR_RELION_SPARSE_PASS2_HALF_VOLUME",
        "",
    ).lower() in {"1", "true", "yes", "on"}
    keep_half_volume_accumulators = (
        use_native_half_volume_mstep and _local_keep_half_volume_accumulators_enabled()
    )
    if use_native_half_volume_mstep:
        logger.info("Exact local M-step: using native half-volume RELION backprojection")
        if keep_half_volume_accumulators:
            logger.info("Exact local M-step: keeping packed half-volume accumulators")
        recon_accum_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)
    else:
        recon_accum_shape = recon_volume_shape
    recon_volume_size = int(np.prod(recon_accum_shape))

    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
    )
    use_window = window_spec.use_window
    window_indices = window_spec.score_indices
    recon_window_indices = window_spec.recon_indices
    n_windowed = window_spec.n_score
    n_recon_windowed = window_spec.n_recon
    projection_kwargs = window_spec.projection_kwargs()
    ## TODO: THE NEXT 100 LINES ARE JSUT DEFINING VARIABLES AND CONSTANTS. CLEAN THIS UP.
    spectrum_setup = _make_local_spectrum_setup(
        image_shape,
        n_half,
        noise_variance,
        window_spec,
        half_spectrum_scoring=half_spectrum_scoring,
    )
    half_weights = spectrum_setup.half_weights
    norm_half_weights = spectrum_setup.norm_half_weights
    half_weights_windowed = spectrum_setup.half_weights_windowed
    noise_variance_half = spectrum_setup.noise_variance_half
    materialize_projection_abs2 = spectrum_setup.materialize_projection_abs2

    Ft_y = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    log_evidence_per_image = np.empty(n_images, dtype=np.float32)
    best_log_score_per_image = np.empty(n_images, dtype=np.float32)
    max_posterior_per_image = np.empty(n_images, dtype=np.float32)
    rotation_posterior_sums = np.zeros(int(local_layout.n_global_rotations), dtype=np.float64)
    best_pose_rotations = np.empty((n_images, 3, 3), dtype=np.float32) if return_best_pose_details else None
    best_pose_translations = np.empty((n_images, local_layout.translation_grid.shape[1]), dtype=np.float32) if return_best_pose_details else None
    best_pose_rotation_ids = np.empty(n_images, dtype=np.int32) if return_best_pose_details else None

    noise_wsum = None
    noise_img_power = None
    noise_a2 = None
    noise_xa = None
    noise_sigma2_offset = jnp.asarray(0.0, dtype=jnp.float32)
    noise_sumw = jnp.asarray(0.0, dtype=jnp.float32)
    return_noise_split = noise_split_diagnostics_requested()
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
        noise_wsum = jnp.zeros(n_shells, dtype=jnp.float32)
        noise_img_power = jnp.zeros(n_shells, dtype=jnp.float32)
        noise_a2 = jnp.zeros(n_shells, dtype=jnp.float32)
        noise_xa = jnp.zeros(n_shells, dtype=jnp.float32)

    batch_backproject_enabled = _local_batch_backproject_enabled()
    big_jit_enabled = _local_big_jit_enabled()
    compact_zero_posterior_rows = _local_compact_zero_posterior_rows_enabled()
    combined_masked_preprocess = _local_combined_masked_preprocess_enabled()
    default_fused_score_mstep = (
        (max_significants is None or int(max_significants) <= 0)
        and normalization_log_z is None
    )
    fused_score_mstep_enabled = _local_fused_score_mstep_enabled(default_fused_score_mstep)
    timing = _LocalTiming()
    raw_cache_enabled = False
    processed_cache_enabled = False
    processed_score_half_cache = None
    processed_recon_half_cache = None
    processed_cache_real_space_pre_shift_applied = False
    preprocess_profile = _new_local_preprocess_timer()
    transfer_profile = _new_local_transfer_timer()
    big_jit_bucket_count = 0
    total_local_rotations = int(local_layout.total_local_rotations)
    collect_profile_stats = bool(return_profile)
    seen_global_rotations = (
        np.zeros(rotation_posterior_sums.shape[0], dtype=bool)
        if collect_profile_stats and rotation_posterior_sums.size
        else np.zeros(0, dtype=bool)
    )
    seen_nonzero_global_rotations = np.zeros_like(seen_global_rotations)
    seen_reconstruction_global_rotations = np.zeros_like(seen_global_rotations)
    total_padded_rotations = 0
    chunk_sizes = []
    chunk_local_rotations = []
    chunk_padded_rotations = []
    chunk_nonzero_posterior_rows = []
    chunk_reconstruction_rows = []
    chunk_significant_samples = []
    n_chunks = 0
    local_total_hypotheses = 0
    total_significant_samples = 0
    total_reconstruction_rows = 0
    max_hypotheses_per_microbatch = _exact_local_max_hypotheses_per_microbatch(
        max_hypotheses_per_microbatch,
        n_windowed,
    )
    mean_for_proj_big_jit = mean_for_proj
    projection_half_volume_big_jit = False
    if big_jit_enabled:
        # The big-JIT path keeps projection input in packed half-volume layout.
        mean_for_proj_big_jit = fourier_transform_utils.full_volume_to_half_volume(
            mean_for_proj,
            proj_volume_shape,
        ).reshape(-1)
        projection_half_volume_big_jit = True
    bucket_build_t0 = time.time()
    bucket_specs = bucket_local_hypothesis_layout(
        local_layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
    )
    timing.bucket_build_s += time.time() - bucket_build_t0

    raw_batch_cache = None
    ctf_param_cache = None
    raw_cache_requested = _local_raw_cache_enabled(
        n_images,
        image_shape,
        getattr(experiment_dataset, "dtype", np.float32),
    )
    processed_cache_requested = _local_processed_cache_enabled(
        n_images,
        image_shape,
        score_with_masked_images,
    )
    if raw_cache_requested or processed_cache_requested:
        raw_cache_t0 = time.time()
        raw_batch_cache, ctf_param_cache = _build_local_raw_cache(experiment_dataset, n_images)
        timing.raw_cache_build_s = time.time() - raw_cache_t0
        raw_cache_enabled = bool(raw_cache_requested)

        if processed_cache_requested:
            processed_cache_t0 = time.time()
            (
                processed_score_half_cache,
                processed_recon_half_cache,
                processed_cache_real_space_pre_shift_applied,
            ) = _build_processed_half_cache(
                experiment_dataset,
                raw_batch_cache,
                image_pre_shifts,
                n_images,
                score_with_masked_images,
            )
            if processed_score_half_cache is not None:
                processed_cache_enabled = True
                if return_profile:
                    _block_until_ready(processed_score_half_cache, processed_recon_half_cache)
            timing.processed_cache_build_s = time.time() - processed_cache_t0

        if not raw_cache_requested:
            raw_batch_cache = None

    phase_t0 = time.time()
    translation_phases_half = _half_translation_phase_table(
        local_layout.translation_grid,
        image_shape,
    )
    if return_profile:
        _block_until_ready(translation_phases_half)
    translation_phase_time = time.time() - phase_t0
    timing.preprocess_s += translation_phase_time
    preprocess_profile["translation_phase_s"] += translation_phase_time

    ## TODO: THIS IS VERY MESSY. CLEAN UP
    big_jit_mask_args = _local_big_jit_mask_args(experiment_dataset, image_shape)
    big_jit_mask_mode = big_jit_mask_args.mask_mode
    big_jit_image_mask_arg = big_jit_mask_args.image_mask

    big_jit_window_indices_arg = window_spec.score_or_full_indices(n_half)
    big_jit_recon_window_indices_arg = window_spec.recon_or_full_indices(n_half)

    for bucket in bucket_specs:
        n_chunks += 1
        if collect_profile_stats:
            chunk_sizes.append(int(bucket.image_indices.shape[0]))
            chunk_local_rotations.append(int(np.sum(bucket.actual_rotation_counts)))
            chunk_padded_rotations.append(int(bucket.image_indices.shape[0] * bucket.bucket_rotation_count))
            total_padded_rotations += int(bucket.image_indices.shape[0] * bucket.bucket_rotation_count)
            local_total_hypotheses += int(np.sum(bucket.actual_rotation_counts) * n_trans)
        fetch_t0 = time.time()
        if processed_cache_enabled:
            bucket_image_indices = np.asarray(bucket.image_indices, dtype=np.int32)
            batch_data = None
            ctf_params = ctf_param_cache[bucket_image_indices]
            fetched_indices = bucket_image_indices
        elif raw_batch_cache is None:
            batch_data, ctf_params, fetched_indices = _fetch_indexed_batch(experiment_dataset, bucket.image_indices)
        else:
            bucket_image_indices = np.asarray(bucket.image_indices, dtype=np.int32)
            batch_data = raw_batch_cache[bucket_image_indices]
            ctf_params = ctf_param_cache[bucket_image_indices]
            fetched_indices = bucket_image_indices
        timing.batch_fetch_s += time.time() - fetch_t0
        bucket = _reorder_bucket_to_indices(bucket, fetched_indices)
        batch_size = int(bucket.image_indices.shape[0])
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
        ## TODO: THIS IS INSANE BRANCHING LOGIC. HOW MANY OF THESE ARE USEFU?  CAN WE DELETE SOME/MANY OF THESE FLAGS?
        can_use_big_jit_bucket = _can_use_local_big_jit_bucket(
            big_jit_enabled=big_jit_enabled,
            processed_cache_enabled=processed_cache_enabled,
            batch_data=batch_data,
            materialize_projection_abs2=materialize_projection_abs2,
            debug_score_dump_filter_matches=debug_score_dump_filter_matches,
            debug_noise_dump_dir=debug_noise_dump_dir,
        )
        if can_use_big_jit_bucket:
            big_jit_t0 = time.time()
            unpadded_bucket = bucket
            unpadded_batch_size = batch_size
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
                jnp.asarray(bucket.local_sample_mask)
                if bucket.local_sample_mask is not None
                else jnp.ones((batch_size, int(bucket.bucket_rotation_count), n_trans), dtype=bool)
            )
            normalization_log_z_arg = (
                jnp.asarray(
                    pad_axis(normalization_log_z_np[bucket_image_indices], 0, batch_size, value=0),
                    dtype=jnp.float32,
                )
                if normalization_log_z_np is not None
                else jnp.zeros(batch_size, dtype=jnp.float32)
            )
            noise_args = _local_big_jit_noise_args(
                accumulate_noise=accumulate_noise,
                noise_wsum=noise_wsum,
                noise_img_power=noise_img_power,
                noise_a2=noise_a2,
                noise_xa=noise_xa,
                shell_indices_half=shell_indices_half,
                shell_indices_noise=shell_indices_noise,
                noise_variance_for_noise=noise_variance_for_noise,
                n_shells=n_shells,
                noise_variance_half=noise_variance_half,
            )

            projection_max_r_big_jit = window_spec.dense_big_jit_max_r()
            (
                Ft_y,
                Ft_ctf,
                noise_wsum,
                noise_img_power,
                noise_a2,
                noise_xa,
                noise_sigma2_offset,
                noise_sumw,
                batch_norm,
                log_Z,
                best_log_score,
                best_argmax,
                max_posterior,
                probs_sum_t,
                n_significant_samples,
                reconstruction_rotation_mask,
                reconstruction_row_count_jax,
            ) = run_local_bucket_big_jit(
                jnp.asarray(batch_data),
                jnp.asarray(ctf_params),
                mean_for_proj_big_jit,
                Ft_y,
                Ft_ctf,
                noise_args.noise_wsum,
                noise_args.noise_img_power,
                noise_args.noise_a2,
                noise_args.noise_xa,
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
                half_weights,
                norm_half_weights,
                big_jit_window_indices_arg,
                big_jit_recon_window_indices_arg,
                noise_args.shell_indices_half,
                noise_args.shell_indices_noise,
                noise_args.noise_variance_for_noise,
                jnp.asarray(bucket.local_rotations),
                jnp.asarray(bucket.local_rotation_log_prior),
                jnp.asarray(bucket.translation_log_prior),
                jnp.asarray(bucket.local_rotation_mask),
                sample_mask_arg,
                jnp.asarray(valid_image_mask),
                normalization_log_z_arg,
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
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                image_shape=image_shape,
                proj_volume_shape=proj_volume_shape,
                recon_volume_shape=recon_volume_shape,
                disc_type=disc_type,
                projection_half_volume=projection_half_volume_big_jit,
                projection_max_r=projection_max_r_big_jit,
                disable_adjoint_y=disable_adjoint_y,
                disable_adjoint_ctf=disable_adjoint_ctf,
                use_native_half_volume_mstep=use_native_half_volume_mstep,
                accumulate_noise=accumulate_noise,
                return_noise_split=return_noise_split,
                n_shells=noise_args.n_shells,
                has_normalization_log_z=normalization_log_z_np is not None,
            )
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
                    n_significant_samples,
                    reconstruction_rotation_mask,
                    reconstruction_row_count_jax,
                    noise_wsum,
                    noise_img_power,
                )
            timing.big_jit_bucket_s += time.time() - big_jit_t0
            big_jit_bucket_count += 1
            ## TODO: NEXT BLOCK OF CODE SEEMS VERY NON-PYTHONIC. CLEAN UP
            pack_t0 = time.time()
            reconstruction_pack = _local_big_jit_reconstruction_pack(
                bucket,
                reconstruction_rotation_mask,
                reconstruction_row_count_jax,
                unpadded_batch_size,
            )
            local_mask_np = reconstruction_pack.local_mask
            reconstruction_take_indices = reconstruction_pack.take_indices
            reconstruction_pack_mask_np = reconstruction_pack.pack_mask
            reconstruction_row_count = reconstruction_pack.row_count
            timing.pack_s += time.time() - pack_t0

            postprocess_t0 = time.time()
            transfer_t0 = time.time()
            best_rot_idx = np.asarray(best_argmax[:unpadded_batch_size] // n_trans, dtype=np.int32)
            best_trans_idx = np.asarray(best_argmax[:unpadded_batch_size] % n_trans, dtype=np.int32)
            transfer_profile["postprocess_argmax_to_host_s"] += time.time() - transfer_t0
            best_rotation_ids = np.take_along_axis(
                np.asarray(bucket.local_rotation_ids[:unpadded_batch_size], dtype=np.int32),
                best_rot_idx[:, None],
                axis=1,
            ).reshape(-1)
            if np.any(best_rotation_ids < 0):
                raise RuntimeError("exact local engine selected padded local rotation")
            hard_assignment[unpadded_bucket.image_indices] = (best_rotation_ids * n_trans + best_trans_idx).astype(np.int32)
            transfer_t0 = time.time()
            log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm[:unpadded_batch_size], axis=1), dtype=np.float64)
            log_z_np = np.asarray(log_Z[:unpadded_batch_size], dtype=np.float32)
            best_log_score_np = np.asarray(best_log_score[:unpadded_batch_size], dtype=np.float32)
            max_posterior_np = np.asarray(max_posterior[:unpadded_batch_size], dtype=np.float32)
            transfer_profile["postprocess_scores_to_host_s"] += time.time() - transfer_t0
            log_evidence_per_image[unpadded_bucket.image_indices] = log_z_np + log_score_offset.astype(np.float32)
            best_log_score_per_image[unpadded_bucket.image_indices] = best_log_score_np + log_score_offset.astype(np.float32)
            max_posterior_per_image[unpadded_bucket.image_indices] = max_posterior_np

            transfer_t0 = time.time()
            probs_sum_t_np = np.asarray(probs_sum_t[:unpadded_batch_size], dtype=np.float64)
            n_significant_samples_np = (
                np.asarray(n_significant_samples[:unpadded_batch_size], dtype=np.int32) if collect_profile_stats else None
            )
            transfer_profile["postprocess_posterior_to_host_s"] += time.time() - transfer_t0
            local_ids_np = np.asarray(bucket.local_rotation_ids[:unpadded_batch_size], dtype=np.int32)
            posterior_ids_np = (
                local_ids_np
                if bucket.local_rotation_posterior_ids is None
                else np.asarray(bucket.local_rotation_posterior_ids[:unpadded_batch_size], dtype=np.int32)
            )
            np.add.at(rotation_posterior_sums, posterior_ids_np[local_mask_np], probs_sum_t_np[local_mask_np])
            if collect_profile_stats:
                nonzero_mask = (probs_sum_t_np > 0.0) & local_mask_np
                chunk_nonzero_posterior_rows.append(int(np.count_nonzero(nonzero_mask)))
                chunk_significant_samples.append(int(np.sum(n_significant_samples_np, dtype=np.int64)))
                chunk_reconstruction_rows.append(int(reconstruction_row_count))
                total_significant_samples += int(np.sum(n_significant_samples_np, dtype=np.int64))
                total_reconstruction_rows += int(reconstruction_row_count)
            if seen_global_rotations.size:
                nonzero_mask = (probs_sum_t_np > 0.0) & local_mask_np
                seen_global_rotations[posterior_ids_np[local_mask_np]] = True
                seen_nonzero_global_rotations[posterior_ids_np[nonzero_mask]] = True
                packed_posterior_ids_np = np.take_along_axis(posterior_ids_np, reconstruction_take_indices, axis=1)
                seen_reconstruction_global_rotations[packed_posterior_ids_np[reconstruction_pack_mask_np]] = True
            if return_best_pose_details:
                best_pose_rotations[unpadded_bucket.image_indices] = np.take_along_axis(
                    np.asarray(bucket.local_rotations[:unpadded_batch_size], dtype=np.float32),
                    best_rot_idx[:, None, None, None],
                    axis=1,
                ).reshape(-1, 3, 3)
                best_pose_translations[unpadded_bucket.image_indices] = np.asarray(
                    local_layout.translation_grid,
                    dtype=np.float32,
                )[best_trans_idx]
                best_pose_rotation_ids[unpadded_bucket.image_indices] = best_rotation_ids.astype(np.int32, copy=False)
            timing.postprocess_s += time.time() - postprocess_t0

            host_stats_t0 = time.time()
            logger.debug(
                "Exact local big-JIT bucket: %d images, bucket_rot=%d, total_local_rot=%d",
                unpadded_batch_size,
                int(bucket.bucket_rotation_count),
                int(np.sum(unpadded_bucket.actual_rotation_counts)),
            )
            timing.host_stats_s += time.time() - host_stats_t0
            continue

        bucket_score_half_cache = None
        bucket_recon_half_cache = None
        if processed_cache_enabled:
            bucket_image_indices = np.asarray(bucket.image_indices, dtype=np.int32)
            bucket_score_half_cache = processed_score_half_cache[bucket_image_indices]
            bucket_recon_half_cache = processed_recon_half_cache[bucket_image_indices]

        preprocess_t0 = time.time()
        (
            shifted_half,
            shifted_recon_half,
            batch_norm,
            ctf2_over_nv_half,
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
            batch_size,
            n_trans,
            score_with_masked_images,
            image_pre_shifts=image_pre_shifts,
            processed_score_half=bucket_score_half_cache,
            processed_recon_half=bucket_recon_half_cache,
            real_space_pre_shift_applied_cache=processed_cache_real_space_pre_shift_applied,
            timer=preprocess_profile if return_profile else None,
            synchronize_profile=return_profile,
            combined_masked_preprocess=combined_masked_preprocess,
        )
        if scale_corrections is not None:
            batch_scale = jnp.asarray(scale_corrections[np.asarray(bucket.image_indices)])
        else:
            batch_scale = jnp.ones(batch_size, dtype=batch_norm.dtype)

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
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]

        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(bucket.image_indices)])
            lattice_half = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
                image_shape, voxel_size=1, scaled=True
            )
            phase_factors = jnp.exp(-2j * jnp.pi * (lattice_half @ batch_shifts.T)).T
            phase_expanded = jnp.repeat(phase_factors, n_trans, axis=0)
            shifted_half = shifted_half * phase_expanded
            shifted_recon_half = shifted_recon_half * phase_expanded
        shifted_half_with_dc = shifted_half
        ctf2_over_nv_half_with_dc = ctf2_over_nv_half

        if half_spectrum_scoring:
            dc_mask = make_shell_indices_half(image_shape) == 0
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

        if use_window:
            shifted_score = shifted_half[:, window_indices]
            shifted_recon = shifted_recon_half[:, recon_window_indices]
            ctf2_over_nv_score = ctf2_over_nv_half[:, window_indices]
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc[:, recon_window_indices]
            shifted_noise = shifted_half_with_dc[:, recon_window_indices]
        else:
            shifted_score = shifted_half
            shifted_recon = shifted_recon_half
            ctf2_over_nv_score = ctf2_over_nv_half
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
            shifted_noise = shifted_half_with_dc

        if use_float64_scoring:
            shifted_score = shifted_score.astype(jnp.complex128)
            shifted_recon = shifted_recon.astype(jnp.complex128)
            shifted_noise = shifted_noise.astype(jnp.complex128)
            ctf2_over_nv_score = ctf2_over_nv_score.astype(jnp.float64)
            ctf2_over_nv_recon = ctf2_over_nv_recon.astype(jnp.float64)
        else:
            shifted_score = shifted_score.astype(jnp.complex64)
            ctf2_over_nv_score = ctf2_over_nv_score.astype(jnp.float32)
        timing.preprocess_s += time.time() - preprocess_t0

        projection_t0 = time.time()
        # NOTE(local-projection-dedupe): do not retry per-bucket projection
        # dedupe here unless the real 5k duplicate factor changes materially.
        # We tried it repeatedly on the exact-local path and it is a bad trade:
        # after RELION-style reconstruction gating the measured projection
        # duplicate factor was only ~1.004-1.005, while the extra gather/shape
        # churn regressed the real 5k local run from ~76.7s to ~126.9s.
        flat_rotations = flatten_bucket_rotations(jnp.asarray(bucket.local_rotations))
        proj_half_flat, proj_abs2_half_flat = _compute_projections_block(
            mean_for_proj,
            flat_rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            return_abs2=materialize_projection_abs2,
            **projection_kwargs,
        )
        if use_window:
            proj_half = proj_half_flat[:, window_indices].reshape(batch_size, bucket.bucket_rotation_count, n_windowed)
            proj_weighted = proj_half * half_weights_windowed[None, None, :]
            proj_recon = proj_half_flat[:, recon_window_indices].reshape(
                batch_size,
                bucket.bucket_rotation_count,
                n_recon_windowed,
            )
            proj_for_noise = proj_recon
            if materialize_projection_abs2:
                proj_abs2 = proj_abs2_half_flat[:, window_indices].reshape(
                    batch_size,
                    bucket.bucket_rotation_count,
                    n_windowed,
                )
                proj_abs2_weighted = proj_abs2 * half_weights_windowed[None, None, :]
                proj_abs2_for_noise = proj_abs2_half_flat[:, recon_window_indices].reshape(
                    batch_size,
                    bucket.bucket_rotation_count,
                    n_recon_windowed,
                )
            else:
                proj_abs2_weighted = None
                proj_abs2_for_noise = None
        else:
            proj_half = proj_half_flat.reshape(batch_size, bucket.bucket_rotation_count, n_half)
            proj_weighted = proj_half * half_weights[None, None, :]
            proj_for_noise = proj_half
            if materialize_projection_abs2:
                proj_abs2 = proj_abs2_half_flat.reshape(batch_size, bucket.bucket_rotation_count, n_half)
                proj_abs2_weighted = proj_abs2 * half_weights[None, None, :]
                proj_abs2_for_noise = proj_abs2
            else:
                proj_abs2_weighted = None
                proj_abs2_for_noise = None
        if use_float64_scoring:
            proj_weighted = proj_weighted.astype(jnp.complex128)
            proj_for_noise = proj_for_noise.astype(jnp.complex128)
            if proj_abs2_weighted is not None:
                proj_abs2_weighted = proj_abs2_weighted.astype(jnp.float64)
            if proj_abs2_for_noise is not None:
                proj_abs2_for_noise = proj_abs2_for_noise.astype(jnp.float64)
        else:
            proj_weighted = proj_weighted.astype(jnp.complex64)
            if proj_abs2_weighted is not None:
                proj_abs2_weighted = proj_abs2_weighted.astype(jnp.float32)
        if return_profile:
            _block_until_ready(proj_weighted if proj_abs2_weighted is None else (proj_weighted, proj_abs2_weighted))
        timing.projection_s += time.time() - projection_t0

        shifted_score_split = shifted_score.reshape(batch_size, n_trans, -1)
        shifted_recon_split = shifted_recon.reshape(batch_size, n_trans, -1)
        ## TODO: ONCE AGIN: INSANE  BRANCIGN LOGIC. I WANT TO DELETE AS MANY OF THESE AS POSSIBLE.  HOW MANY OF THESE FLAGS ARE REALLY USEFUL?
        ## AND REMOVE AS MANY IF STATEMENTS. ANY FLAGS USELESS FOR OUR 5K PARITY RUN SHOULD BE REMOVED (UNLESS GREAT REASON TO STAY)
        can_use_fused_score_mstep = (
            fused_score_mstep_enabled
            and proj_abs2_weighted is None
            and normalization_log_z_np is None
            and not debug_score_dump_filter_matches
        )
        if can_use_fused_score_mstep:
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
                jnp.asarray(bucket.local_rotation_log_prior),
                jnp.asarray(bucket.translation_log_prior),
                jnp.asarray(bucket.local_rotation_mask),
                None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask),
                shifted_recon_split,
                ctf2_over_nv_recon,
                half_spectrum_scoring=half_spectrum_scoring,
                use_float64_normalization=use_float64_normalization,
                reconstruct_significant_only=reconstruct_significant_only,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
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
            fused_elapsed = time.time() - fused_t0
            timing.fused_score_mstep_s += fused_elapsed
        else:
            score_t0 = time.time()
            if proj_abs2_weighted is None:
                if half_spectrum_scoring:
                    scores = score_local_bucket_abs2_on_demand(
                        shifted_score_split,
                        ctf2_over_nv_score,
                        proj_weighted,
                        jnp.asarray(bucket.local_rotation_log_prior),
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
                        jnp.asarray(bucket.local_rotation_log_prior),
                        jnp.asarray(bucket.translation_log_prior),
                        jnp.asarray(bucket.local_rotation_mask),
                        None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask),
                    )
            else:
                scores = score_local_bucket(
                    shifted_score_split,
                    ctf2_over_nv_score,
                    proj_weighted,
                    proj_abs2_weighted,
                    jnp.asarray(bucket.local_rotation_log_prior),
                    jnp.asarray(bucket.translation_log_prior),
                    jnp.asarray(bucket.local_rotation_mask),
                    None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask),
                )
            if return_profile:
                _block_until_ready(scores)
            timing.score_s += time.time() - score_t0

            normalize_t0 = time.time()
            ## TODO: NEEDS A LOT OF CLEAN UP FOR THIS STUFF
            ## TODO: THIS IS AN INSANE BRANCH FOR EXAMPLE, AND INSANE THAT THERE ARE TWO FUNCTIONS FOR THIS. THIS SHOULD BE FIGURED OUT BY DTYPE. DUCK TYPING.
            bucket_log_z = (
                None
                if normalization_log_z_np is None
                else jnp.asarray(
                    normalization_log_z_np[np.asarray(bucket.image_indices)],
                    dtype=scores.real.dtype,
                )
            )
            log_Z, probs, best_log_score, best_argmax, max_posterior = normalize_local_scores_auto(
                scores,
                bucket_log_z,
                use_float64_normalization=use_float64_normalization,
            )
            if return_profile:
                _block_until_ready(log_Z, probs, best_log_score, best_argmax, max_posterior)
            timing.normalize_s += time.time() - normalize_t0

            significance_t0 = time.time()
            if reconstruct_significant_only:
                reconstruction_sample_mask, reconstruction_rotation_mask, n_significant_samples = compute_reconstruction_support(
                    probs,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
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
                ctf2_over_nv_score=ctf2_over_nv_score,
                proj_weighted=proj_weighted,
                proj_abs2_weighted=proj_abs2_weighted,
                dump_dir=debug_score_dump_dir,
                pending_targets=debug_score_dump_targets,
                requested_current_sizes=debug_score_dump_current_sizes,
                requested_iterations=debug_score_dump_iterations,
            )

            mstep_t0 = time.time()
            probs_sum_t = jnp.sum(probs, axis=-1)
            reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
            summed = compute_local_weighted_sums(reconstruction_probs, shifted_recon_split)
            ctf_probs = compute_local_ctf_sums(reconstruction_probs, ctf2_over_nv_recon)
            if return_profile:
                _block_until_ready(summed, ctf_probs, probs_sum_t, reconstruction_probs_sum_t)
            timing.mstep_s += time.time() - mstep_t0
            scores = None

        pack_t0 = time.time()
        pack_selection = _select_local_reconstruction_pack(
            bucket=bucket,
            reconstruction_rotation_mask=reconstruction_rotation_mask,
            probs_sum_t=probs_sum_t,
            reconstruct_significant_only=reconstruct_significant_only,
            compact_zero_posterior_rows=compact_zero_posterior_rows,
            rotation_block_size=rotation_block_size,
            transfer_profile=transfer_profile,
            pack_start_time=pack_t0,
        )
        reconstruction_rotation_mask_np = pack_selection.rotation_mask
        reconstruction_take_indices = pack_selection.take_indices
        reconstruction_pack_mask_np = pack_selection.pack_mask
        reconstruction_row_count = pack_selection.row_count
        probs_sum_t_np = pack_selection.probs_sum_t
        reconstruction_take_indices_jnp = jnp.asarray(reconstruction_take_indices, dtype=jnp.int32)
        reconstruction_pack_mask_jnp = jnp.asarray(reconstruction_pack_mask_np)
        packed_rotations_np = np.take_along_axis(
            np.asarray(bucket.local_rotations, dtype=np.float32),
            reconstruction_take_indices[:, :, None, None],
            axis=1,
        )
        packed_summed = jnp.take_along_axis(summed, reconstruction_take_indices_jnp[:, :, None], axis=1)
        packed_summed = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_summed, 0.0)
        packed_ctf_probs = jnp.take_along_axis(ctf_probs, reconstruction_take_indices_jnp[:, :, None], axis=1)
        packed_ctf_probs = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_ctf_probs, 0.0)
        packed_flat_rotations = None
        if not disable_adjoint_y or not disable_adjoint_ctf:
            packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_rotations_np))
        packed_summed_rows = flatten_bucket_rows(packed_summed)
        packed_ctf_rows = flatten_bucket_rows(packed_ctf_probs)
        timing.pack_s += time.time() - pack_t0
        ## TODO THIS CLEAN THIS INSSANE BRANCHING HERE
        Ft_y, Ft_ctf = _accumulate_local_adjoint_rows(
            packed_summed_rows=packed_summed_rows,
            packed_ctf_rows=packed_ctf_rows,
            packed_flat_rotations=packed_flat_rotations,
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
            recon_window_indices=recon_window_indices,
            image_shape=image_shape,
            recon_volume_shape=recon_volume_shape,
            use_window=use_window,
            use_native_half_volume_mstep=use_native_half_volume_mstep,
            current_size=current_size,
            batch_backproject_enabled=batch_backproject_enabled,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            return_profile=return_profile,
            timing=timing,
        )

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
            batch_img_power_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            batch_img_power_shells = batch_img_power_shells.at[shell_indices_half].add(batch_img_power)
            noise_img_power = noise_img_power + batch_img_power_shells
            noise_sumw = noise_sumw + jnp.sum(support_mass)

            shifted_noise_split = shifted_noise.reshape(batch_size, n_trans, -1)
            summed_masked_noise = compute_local_weighted_sums(reconstruction_probs, shifted_noise_split)
            debug_noise_dump_targets = maybe_write_debug_noise_component_dump(
                experiment_dataset=experiment_dataset,
                bucket=bucket,
                support_mass=support_mass,
                processed_noise_power_half=processed_noise_power_half,
                proj_for_noise=proj_for_noise,
                proj_abs2_for_noise=proj_abs2_for_noise,
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
            packed_proj_for_noise = jnp.take_along_axis(
                proj_for_noise,
                reconstruction_take_indices_jnp[:, :, None],
                axis=1,
            )
            packed_proj_for_noise = jnp.where(
                reconstruction_pack_mask_jnp[:, :, None],
                packed_proj_for_noise,
                0.0,
            )
            flat_proj_for_noise = flatten_bucket_rows(packed_proj_for_noise)
            if proj_abs2_for_noise is None:
                flat_proj_abs2_for_noise = jnp.abs(flat_proj_for_noise) ** 2
            else:
                packed_proj_abs2_for_noise = jnp.take_along_axis(
                    proj_abs2_for_noise,
                    reconstruction_take_indices_jnp[:, :, None],
                    axis=1,
                )
                packed_proj_abs2_for_noise = jnp.where(
                    reconstruction_pack_mask_jnp[:, :, None],
                    packed_proj_abs2_for_noise,
                    0.0,
                )
                flat_proj_abs2_for_noise = flatten_bucket_rows(packed_proj_abs2_for_noise)
            block_noise_shells, block_a2_shells, block_xa_shells = _compute_noise_block(
                flat_proj_for_noise,
                flat_proj_abs2_for_noise,
                flatten_bucket_rows(packed_summed_masked_noise),
                flatten_bucket_rows(packed_ctf_probs),
                noise_variance_for_noise,
                shell_indices_noise,
                n_shells,
                return_noise_split,
            )
            if return_profile:
                _block_until_ready(block_noise_shells)
            noise_wsum = noise_wsum + block_noise_shells
            if return_noise_split:
                noise_a2 = noise_a2 + block_a2_shells
                noise_xa = noise_xa + block_xa_shells
            noise_sigma2_offset = noise_sigma2_offset + noise_sumw_offset
            timing.noise_s += time.time() - noise_t0

        postprocess_t0 = time.time()
        transfer_t0 = time.time()
        best_rot_idx = np.asarray(best_argmax // n_trans, dtype=np.int32)
        best_trans_idx = np.asarray(best_argmax % n_trans, dtype=np.int32)
        transfer_profile["postprocess_argmax_to_host_s"] += time.time() - transfer_t0
        best_rotation_ids = np.take_along_axis(
            np.asarray(bucket.local_rotation_ids, dtype=np.int32),
            best_rot_idx[:, None],
            axis=1,
        ).reshape(-1)
        if np.any(best_rotation_ids < 0):
            raise RuntimeError("exact local engine selected padded local rotation")
        hard_assignment[bucket.image_indices] = (best_rotation_ids * n_trans + best_trans_idx).astype(np.int32)
        transfer_t0 = time.time()
        log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
        log_z_np = np.asarray(log_Z, dtype=np.float32)
        best_log_score_np = np.asarray(best_log_score, dtype=np.float32)
        max_posterior_np = np.asarray(max_posterior, dtype=np.float32)
        transfer_profile["postprocess_scores_to_host_s"] += time.time() - transfer_t0
        log_evidence_per_image[bucket.image_indices] = log_z_np + log_score_offset.astype(np.float32)
        best_log_score_per_image[bucket.image_indices] = best_log_score_np + log_score_offset.astype(np.float32)
        max_posterior_per_image[bucket.image_indices] = max_posterior_np

        transfer_t0 = time.time()
        if probs_sum_t_np is None:
            probs_sum_t_np = np.asarray(probs_sum_t, dtype=np.float64)
        n_significant_samples_np = (
            np.asarray(n_significant_samples, dtype=np.int32) if collect_profile_stats else None
        )
        transfer_profile["postprocess_posterior_to_host_s"] += time.time() - transfer_t0
        local_ids_np = np.asarray(bucket.local_rotation_ids, dtype=np.int32)
        posterior_ids_np = (
            local_ids_np
            if bucket.local_rotation_posterior_ids is None
            else np.asarray(bucket.local_rotation_posterior_ids, dtype=np.int32)
        )
        local_mask_np = np.asarray(bucket.local_rotation_mask, dtype=bool)
        np.add.at(rotation_posterior_sums, posterior_ids_np[local_mask_np], probs_sum_t_np[local_mask_np])
        if collect_profile_stats:
            nonzero_mask = (probs_sum_t_np > 0.0) & local_mask_np
            chunk_nonzero_posterior_rows.append(int(np.count_nonzero(nonzero_mask)))
            chunk_significant_samples.append(int(np.sum(n_significant_samples_np, dtype=np.int64)))
            chunk_reconstruction_rows.append(int(reconstruction_row_count))
            total_significant_samples += int(np.sum(n_significant_samples_np, dtype=np.int64))
            total_reconstruction_rows += int(reconstruction_row_count)
        if seen_global_rotations.size:
            nonzero_mask = (probs_sum_t_np > 0.0) & local_mask_np
            seen_global_rotations[posterior_ids_np[local_mask_np]] = True
            seen_nonzero_global_rotations[posterior_ids_np[nonzero_mask]] = True
            packed_rotation_ids_np = np.take_along_axis(
                np.asarray(bucket.local_rotation_ids, dtype=np.int32),
                reconstruction_take_indices,
                axis=1,
            )
            packed_posterior_ids_np = np.take_along_axis(posterior_ids_np, reconstruction_take_indices, axis=1)
            seen_reconstruction_global_rotations[packed_posterior_ids_np[reconstruction_pack_mask_np]] = True
        if return_best_pose_details:
            best_pose_rotations[bucket.image_indices] = np.take_along_axis(
                np.asarray(bucket.local_rotations, dtype=np.float32),
                best_rot_idx[:, None, None, None],
                axis=1,
            ).reshape(-1, 3, 3)
            best_pose_translations[bucket.image_indices] = np.asarray(local_layout.translation_grid, dtype=np.float32)[
                best_trans_idx
            ]
            best_pose_rotation_ids[bucket.image_indices] = best_rotation_ids.astype(np.int32, copy=False)
        timing.postprocess_s += time.time() - postprocess_t0

        host_stats_t0 = time.time()
        logger.debug(
            "Exact local bucket: %d images, bucket_rot=%d, total_local_rot=%d",
            batch_size,
            int(bucket.bucket_rotation_count),
            int(np.sum(bucket.actual_rotation_counts)),
        )
        timing.host_stats_s += time.time() - host_stats_t0

    final_accumulator_t0 = time.time()
    if use_native_half_volume_mstep:
        if os.environ.get("RECOVAR_RELION_SPARSE_PASS2_HALF_VOLUME_ENFORCE_X0", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            logger.info("Exact local M-step: enforcing RELION half-volume x=0 Hermitian plane")
            Ft_y = enforce_relion_half_volume_x0_hermitian(Ft_y, recon_volume_shape)
            Ft_ctf = enforce_relion_half_volume_x0_hermitian(Ft_ctf, recon_volume_shape)
        if not keep_half_volume_accumulators:
            Ft_y = fourier_transform_utils.half_volume_to_full_volume(Ft_y, recon_volume_shape).reshape(-1)
            Ft_ctf = fourier_transform_utils.half_volume_to_full_volume(Ft_ctf, recon_volume_shape).reshape(-1)

    if return_profile:
        _block_until_ready(Ft_y, Ft_ctf)
    timing.final_accumulator_s += time.time() - final_accumulator_t0

    stats_finalize_t0 = time.time()
    relion_stats = RelionStats(
        log_evidence_per_image=jnp.asarray(log_evidence_per_image),
        best_log_score_per_image=jnp.asarray(best_log_score_per_image),
        max_posterior_per_image=jnp.asarray(max_posterior_per_image),
        rotation_posterior_sums=jnp.asarray(rotation_posterior_sums, dtype=jnp.float32),
    )
    noise_stats = None
    if accumulate_noise:
        transfer_t0 = time.time()
        noise_sigma2_offset_value = float(np.asarray(noise_sigma2_offset, dtype=np.float64))
        noise_sumw_value = float(np.asarray(noise_sumw, dtype=np.float64))
        transfer_profile["final_noise_to_host_s"] += time.time() - transfer_t0
        noise_stats = NoiseStats(
            wsum_sigma2_noise=noise_wsum.astype(jnp.float32),
            wsum_img_power=noise_img_power.astype(jnp.float32),
            wsum_sigma2_offset=noise_sigma2_offset_value,
            sumw=noise_sumw_value,
            wsum_noise_a2=(noise_a2.astype(jnp.float32) if return_noise_split else None),
            wsum_noise_xa=(noise_xa.astype(jnp.float32) if return_noise_split else None),
        )
    timing.stats_finalize_s += time.time() - stats_finalize_t0

    if (
        debug_score_dump_filter_matches
        and debug_score_dump_targets
        and debug_score_dump_iterations is None
    ):
        logger.warning(
            "Requested local score dump indices were not observed in this dataset view: %s",
            sorted(debug_score_dump_targets),
        )

    if not return_profile:
        if return_best_pose_details:
            if accumulate_noise:
                return (
                    Ft_y,
                    Ft_ctf,
                    hard_assignment,
                    best_pose_rotations,
                    best_pose_translations,
                    best_pose_rotation_ids,
                    relion_stats,
                    noise_stats,
                )
            return Ft_y, Ft_ctf, hard_assignment, best_pose_rotations, best_pose_translations, best_pose_rotation_ids, relion_stats
        if accumulate_noise:
            return Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats
        return Ft_y, Ft_ctf, hard_assignment, relion_stats

    _block_until_ready(Ft_y, Ft_ctf)
    total_wall_time = time.time() - overall_t0
    profile_summary = {
        "local_engine_kind": np.array("exact_v1"),
        "big_jit_enabled": np.asarray(big_jit_enabled),
        "big_jit_bucket_count": np.int32(big_jit_bucket_count),
        "batch_backproject_enabled": np.asarray(batch_backproject_enabled),
        "compact_zero_posterior_rows": np.asarray(compact_zero_posterior_rows),
        "combined_masked_preprocess": np.asarray(combined_masked_preprocess),
        "fused_score_mstep_enabled": np.asarray(fused_score_mstep_enabled),
        "materialize_projection_abs2": np.asarray(materialize_projection_abs2),
        "keep_half_volume_accumulators": np.asarray(keep_half_volume_accumulators),
        "bucket_build_time_s": np.float64(timing.bucket_build_s),
        "raw_cache_build_time_s": np.float64(timing.raw_cache_build_s),
        "raw_cache_enabled": np.asarray(raw_cache_enabled),
        "processed_cache_build_time_s": np.float64(timing.processed_cache_build_s),
        "processed_cache_enabled": np.asarray(processed_cache_enabled),
        "batch_fetch_time_s": np.float64(timing.batch_fetch_s),
        "preprocess_time_s": np.float64(timing.preprocess_s),
        **_prefixed_timer_profile("preprocess_", preprocess_profile),
        **_prefixed_timer_profile("transfer_", transfer_profile),
        "transfer_total_to_host_s": np.float64(sum(transfer_profile.values())),
        **_local_timing_profile(timing),
        "em_time_s": np.float64(total_wall_time),
        "accounted_em_time_s": np.float64(timing.accounted_s()),
        "unattributed_em_time_s": np.float64(
            max(total_wall_time - timing.accounted_s(), 0.0)
        ),
        "n_chunks": np.int32(n_chunks),
        "chunk_sizes": np.asarray(chunk_sizes, dtype=np.int32),
        "chunk_local_rotations": np.asarray(chunk_local_rotations, dtype=np.int32),
        "chunk_padded_rotations": np.asarray(chunk_padded_rotations, dtype=np.int32),
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
            0.0 if not np.any(seen_global_rotations) else total_local_rotations / np.count_nonzero(seen_global_rotations)
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
        "local_pad_fraction": np.float64(
            0.0 if total_padded_rotations == 0 else 1.0 - total_local_rotations / total_padded_rotations
        ),
        "n_windowed": np.int32(n_windowed),
    }
    if return_best_pose_details:
        if accumulate_noise:
            return (
                Ft_y,
                Ft_ctf,
                hard_assignment,
                best_pose_rotations,
                best_pose_translations,
                best_pose_rotation_ids,
                relion_stats,
                noise_stats,
                profile_summary,
            )
        return (
            Ft_y,
            Ft_ctf,
            hard_assignment,
            best_pose_rotations,
            best_pose_translations,
            best_pose_rotation_ids,
            relion_stats,
            profile_summary,
        )
    if accumulate_noise:
        return Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats, profile_summary
    return Ft_y, Ft_ctf, hard_assignment, relion_stats, profile_summary
