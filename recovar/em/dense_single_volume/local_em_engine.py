"""Exact per-image local EM engine for RELION-mode local search."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_maybe_windowed as _adjoint_slice_volume_maybe_windowed,
)
from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
    make_scoring_half_image_weights,
    make_shell_indices_half,
)
from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    enforce_half_volume_x0,
    half_volume_accumulator_shape,
    half_volume_accumulators_to_full,
    relion_x_half_accumulators_to_full,
)
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    integer_pre_shifts_or_none,
    tiled_half_image_phase_factors,
)
from recovar.em.dense_single_volume.helpers.jax_runtime import block_until_ready as _block_until_ready
from recovar.em.dense_single_volume.helpers.preprocessing import (
    apply_half_translation_phases as _apply_half_translation_phases,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    half_translation_phase_table as _half_translation_phase_table,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    process_half_image,
    resolve_image_mask_for_half_preprocess,
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
    _sparse_big_jit_mstep_tensors_within_memory,
    _validate_native_half_batch,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_noise_block as _compute_noise_block,
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
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.dense_single_volume.local_big_jit import run_local_bucket_big_jit
from recovar.em.dense_single_volume.local_debug import (
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
    bucket_local_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_score_pass import (
    compute_reconstruction_support,
    fused_score_normalize_mstep_abs2_on_demand,
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
NVTX_DOMAIN_EM = "recovar_em"

# Keeps common 256^2 local-search buckets at two images without entering the
# three-image working set that previously exceeded memory.
EXACT_LOCAL_TARGET_ROW_PIXELS = 190_000_000
EXACT_LOCAL_TARGET_ROW_PIXELS_ENV = "RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS"
EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB = 4.0
EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV = "RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB"
# Disabled by default: on the 50k/256 local-search target this cache made the
# iteration slower by precomputing more spectra than the bucket schedule reuses.
# Upper bound for the extra M-step tensors materialized by the sparse big-JIT
# hybrid path. This path still packs rows before backprojection; the cap only
# guards the temporary fused summed/ctf tensor outputs.
EXACT_LOCAL_BIG_JIT_MIN_SIGNIFICANT_ROW_FRACTION = 0.25

_LOCAL_PREPROCESS_TIMER_KEYS = (
    "integer_shift_s",
    "translation_phase_s",
    "score_process_s",
    "recon_process_s",
    "ctf_s",
    "tile_shift_score_s",
    "tile_shift_recon_s",
    "norm_s",
    "cache_build_s",
    "cache_fetch_s",
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
    "batch_fetch_s",
    "preprocess_s",
)

_LOCAL_ACCOUNTED_TIMING_FIELDS = _LOCAL_ACCOUNTED_TIMING_SETUP_FIELDS + tuple(
    timing_attr for _, timing_attr in _LOCAL_TIMING_PROFILE_FIELDS
)


def _new_zero_timer(keys):
    return {key: 0.0 for key in keys}


class _LocalTiming(TimingAccumulator):
    """Mutable host-side timers for one exact-local EM call."""

    def __init__(self):
        super().__init__(_LOCAL_ACCOUNTED_TIMING_FIELDS)


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
    best_pose_rotations: np.ndarray | None = None
    best_pose_translations: np.ndarray | None = None
    best_pose_rotation_ids: np.ndarray | None = None


@dataclass
class _LocalProjectionBlock:
    proj_weighted: jnp.ndarray
    proj_for_noise: jnp.ndarray | None



def _local_em_return_tuple(
    Ft_y,
    Ft_ctf,
    hard_assignment,
    relion_stats,
    *,
    accumulate_noise: bool,
    return_profile: bool,
    return_best_pose_details: bool,
    best_pose_rotations=None,
    best_pose_translations=None,
    best_pose_rotation_ids=None,
    noise_stats=None,
    profile_summary=None,
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
        proj_half_flat, _ = _compute_relion_projector_projections_block(
            relion_projector_half,
            flat_rotations,
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            return_abs2=False,
            centered_rows=True,
        )
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
        proj_half_flat, _ = _compute_projections_block(
            mean_for_proj,
            flat_rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            return_abs2=False,
            **projection_kwargs,
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
) -> jnp.ndarray:
    """Project only packed reconstruction rows for local noise accumulation."""

    if (
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
        proj_half_flat, _ = _compute_projections_block(
            mean_for_proj,
            packed_flat_rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            return_abs2=False,
            **projection_kwargs,
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
    n_significant_samples_np = np.asarray(n_significant_samples, dtype=np.int32) if collect_profile_stats else None
    buffers.transfer_profile["postprocess_posterior_to_host_s"] += time.time() - transfer_t0

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


def _exact_local_max_hypotheses_per_microbatch(
    default: int | None,
    n_windowed: int,
    *,
    n_trans: int = 1,
    n_recon_windowed: int | None = None,
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
    target_row_pixels = int(os.environ.get(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, EXACT_LOCAL_TARGET_ROW_PIXELS))
    if target_row_pixels <= 0:
        raise ValueError(f"{EXACT_LOCAL_TARGET_ROW_PIXELS_ENV} must be positive")
    value = target_row_pixels // max(1, int(n_windowed))
    max_gb = float(os.environ.get(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB))
    if max_gb > 0.0:
        # The fused local M-step computes probabilities x shifted images with a
        # peak working set proportional to hypotheses * translations * recon
        # pixels. The old cap ignored translations and could still build ~32 GiB
        # buckets on 256-box pass-2 runs.
        n_recon = int(n_windowed if n_recon_windowed is None else n_recon_windowed)
        matmul_cap = int((max_gb * 1e9) // max(1, int(n_trans) * max(1, n_recon) * 4))
        value = min(value, matmul_cap)
    return int(max(512, min(65536, value)))


def _new_local_preprocess_timer():
    return _new_zero_timer(_LOCAL_PREPROCESS_TIMER_KEYS)


def _new_local_transfer_timer():
    return _new_zero_timer(_LOCAL_TRANSFER_TIMER_KEYS)


def _prefixed_timer_profile(prefix: str, timer: dict[str, float]) -> dict[str, np.float64]:
    return {f"{prefix}{key}": np.float64(value) for key, value in timer.items()}


def _local_timing_profile(timing: _LocalTiming) -> dict[str, np.float64]:
    return {
        output_key: np.float64(getattr(timing, timing_attr)) for output_key, timing_attr in _LOCAL_TIMING_PROFILE_FIELDS
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
    score_with_masked_images: bool,
    image_pre_shifts=None,
    processed_half_cache: _LocalProcessedHalfCache | None = None,
    timer: dict[str, float] | None = None,
    synchronize_profile: bool = False,
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
    accumulate_noise: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
    score_with_masked_images: bool = True,
    half_spectrum_scoring: bool = False,
    use_float64_scoring: bool = False,
    use_float64_normalization: bool = True,
    use_float64_projections: bool = False,
    projection_relion_texture_interp: bool = False,
    projection_force_jax: bool = False,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    image_corrections: np.ndarray | None = None,
    scale_corrections: np.ndarray | None = None,
    image_pre_shifts: np.ndarray | None = None,
    mstep_subtract_ctf_projection: bool = False,
    mstep_relion_x_half: bool = False,
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
):
    """Run exact local EM over per-image local hypothesis sets."""

    overall_t0 = time.time()
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    n_trans = int(local_layout.translation_grid.shape[0])
    n_images = int(local_layout.n_images)
    class_log_prior = float(class_log_prior)
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
        and (debug_score_dump_current_sizes is None or int(current_size or -1) in debug_score_dump_current_sizes)
        and (debug_score_dump_iterations is None or int(debug_iteration or -1) in debug_score_dump_iterations)
    )
    debug_noise_dump_filter_matches = (
        debug_noise_dump_dir is not None
        and (debug_noise_dump_current_sizes is None or int(current_size or -1) in debug_noise_dump_current_sizes)
        and (debug_noise_dump_iterations is None or int(debug_iteration or -1) in debug_noise_dump_iterations)
    )
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
            current_size=current_size,
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

    if reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
    else:
        recon_volume_shape = volume_shape
    if mstep_relion_x_half:
        logger.info("Exact local M-step: using RELION x-half BPref-layout backprojection")
    else:
        logger.info("Exact local M-step: using native half-volume backprojection")
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape)
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
    projection_kwargs = window_spec.projection_kwargs()
    projection_kwargs["relion_texture_interp"] = bool(projection_relion_texture_interp)
    projection_kwargs["force_jax"] = bool(projection_force_jax)
    projection_mode = _local_projection_mode(window_spec, projection_kwargs, relion_projector_half)

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )
    norm_half_weights = make_half_image_weights(image_shape)
    half_weights_windowed = window_spec.score_values(half_weights)
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()

    Ft_y = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    log_evidence_per_image = np.empty(n_images, dtype=np.float32)
    best_log_score_per_image = np.empty(n_images, dtype=np.float32)
    max_posterior_per_image = np.empty(n_images, dtype=np.float32)
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
    noise_a2 = None
    noise_xa = None
    noise_sigma2_offset = jnp.asarray(0.0, dtype=jnp.float32)
    noise_sumw = jnp.asarray(0.0, dtype=jnp.float32)
    return_noise_split = noise_split_diagnostics_requested()
    require_materialized_recon_projection = bool(
        mstep_subtract_ctf_projection
        or debug_score_dump_filter_matches
        or debug_noise_dump_filter_matches
        or return_noise_split
    )
    defer_local_noise_projection = (
        not require_materialized_recon_projection
        and window_spec.use_window
        and relion_projector_half is None
        and not bool(projection_kwargs.get("relion_texture_interp", False))
        and not bool(projection_kwargs.get("force_jax", False))
        and _indexed_projection_available()
    )
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
        noise_wsum = jnp.zeros(n_shells, dtype=jnp.float32)
        noise_img_power = jnp.zeros(n_shells, dtype=jnp.float32)
        noise_a2 = jnp.zeros(n_shells, dtype=jnp.float32)
        noise_xa = jnp.zeros(n_shells, dtype=jnp.float32)

    default_fused_score_mstep = (max_significants is None or int(max_significants) <= 0) and normalization_log_z is None
    fused_score_mstep_enabled = default_fused_score_mstep
    timing = _LocalTiming()
    raw_cache_enabled = False
    processed_half_cache_enabled = False
    preprocess_profile = _new_local_preprocess_timer()
    transfer_profile = _new_local_transfer_timer()
    big_jit_bucket_count = 0
    sparse_big_jit_bucket_count = 0
    total_local_rotations = int(local_layout.total_local_rotations)
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
    chunk_nonzero_posterior_rows = []
    chunk_reconstruction_rows = []
    chunk_significant_samples = []
    n_chunks = 0
    local_total_hypotheses = 0
    total_significant_samples = 0
    total_reconstruction_rows = 0
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
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
    )
    max_hypotheses_per_microbatch = _exact_local_max_hypotheses_per_microbatch(
        max_hypotheses_per_microbatch,
        n_windowed,
        n_trans=n_trans,
        n_recon_windowed=window_spec.n_recon,
    )
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
    processed_half_cache = None

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

    big_jit_image_mask_arg, big_jit_mask_mode = resolve_image_mask_for_half_preprocess(
        experiment_dataset,
        image_shape,
        require_mask=score_with_masked_images,
    )
    big_jit_image_mask_arg = jnp.asarray(big_jit_image_mask_arg)

    big_jit_window_indices_arg = window_spec.score_or_full_indices(n_half)
    big_jit_recon_window_indices_arg = window_spec.recon_or_full_indices(n_half)
    disabled_noise_wsum = jnp.zeros(1, dtype=jnp.float32)
    disabled_noise_img_power = jnp.zeros(1, dtype=jnp.float32)
    disabled_noise_a2 = jnp.zeros(1, dtype=jnp.float32)
    disabled_noise_xa = jnp.zeros(1, dtype=jnp.float32)
    disabled_noise_shell_indices = jnp.zeros(n_half, dtype=jnp.int32)

    local_support_rows = int(np.sum(local_layout.rotation_counts))
    significant_backprojection_candidate = (
        reconstruct_significant_only
        and n_images > 0
        and local_support_rows >= int(np.ceil(max(n_images, 1) / EXACT_LOCAL_BIG_JIT_MIN_SIGNIFICANT_ROW_FRACTION))
    )
    use_relion_projector = relion_projector_half is not None
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
        (not use_relion_projector)
        and not disable_big_jit_buckets
        and not debug_score_dump_filter_matches
        and not (accumulate_noise and debug_noise_dump_dir is not None)
        and not processed_half_cache_preferred
    )
    mean_for_proj_big_jit = mean_for_proj
    projection_half_volume_big_jit = False
    if use_big_jit_buckets and not projection_relion_texture_interp:
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

    for bucket in bucket_specs:
        n_chunks += 1
        if collect_profile_stats:
            chunk_sizes.append(int(bucket.image_indices.shape[0]))
            chunk_local_rotations.append(int(np.sum(bucket.actual_rotation_counts)))
            chunk_padded_rotations.append(int(bucket.image_indices.shape[0] * bucket.bucket_rotation_count))
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
        if use_big_jit_buckets and significant_backprojection_candidate:
            sparse_big_jit_backprojection = _sparse_big_jit_mstep_tensors_within_memory(
                image_count=max(batch_size, int(getattr(bucket, "bucket_image_count", batch_size))),
                rotation_count=int(bucket.bucket_rotation_count),
                n_recon_windowed=window_spec.n_recon,
                use_float64_scoring=use_float64_scoring,
            )

        if use_big_jit_buckets and (not significant_backprojection_candidate or sparse_big_jit_backprojection):
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
                shell_indices_half_arg = shell_indices_half
                shell_indices_noise_arg = shell_indices_noise
                noise_variance_for_noise_arg = noise_variance_for_noise
                n_shells_arg = n_shells
            else:
                noise_wsum_arg = disabled_noise_wsum
                noise_img_power_arg = disabled_noise_img_power
                noise_a2_arg = disabled_noise_a2
                noise_xa_arg = disabled_noise_xa
                shell_indices_half_arg = disabled_noise_shell_indices
                shell_indices_noise_arg = disabled_noise_shell_indices
                noise_variance_for_noise_arg = noise_variance_half
                n_shells_arg = 1

            projection_max_r_big_jit = window_spec.dense_big_jit_max_r()
            return_big_jit_mstep_tensors = sparse_big_jit_backprojection and (
                not disable_adjoint_y or not disable_adjoint_ctf
            )
            big_jit_disable_adjoint_y = disable_adjoint_y or return_big_jit_mstep_tensors
            big_jit_disable_adjoint_ctf = disable_adjoint_ctf or return_big_jit_mstep_tensors
            big_jit_result = run_local_bucket_big_jit(
                jnp.asarray(batch_data),
                jnp.asarray(ctf_params),
                mean_for_proj_big_jit,
                Ft_y,
                Ft_ctf,
                noise_wsum_arg,
                noise_img_power_arg,
                noise_a2_arg,
                noise_xa_arg,
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
                shell_indices_half_arg,
                shell_indices_noise_arg,
                noise_variance_for_noise_arg,
                jnp.asarray(bucket.local_rotations),
                local_rotation_log_prior_arg,
                jnp.asarray(bucket.translation_log_prior),
                jnp.asarray(bucket.local_rotation_mask),
                sample_mask_arg,
                jnp.asarray(valid_image_mask),
                normalization_log_z_arg,
                normalization_log_evidence_arg,
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
                projection_relion_texture_interp=bool(projection_relion_texture_interp),
                projection_force_jax=bool(projection_force_jax),
                mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
                mstep_relion_x_half=bool(mstep_relion_x_half),
                disable_adjoint_y=big_jit_disable_adjoint_y,
                disable_adjoint_ctf=big_jit_disable_adjoint_ctf,
                accumulate_noise=accumulate_noise,
                return_noise_split=return_noise_split,
                return_mstep_tensors=return_big_jit_mstep_tensors,
                n_shells=n_shells_arg,
                has_normalization_log_z=normalization_log_z_np is not None,
                has_normalization_log_evidence=normalization_log_evidence_np is not None,
            )
            if return_big_jit_mstep_tensors:
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
                ) = big_jit_result
                summed = None
                ctf_probs = None
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
                    *(() if summed is None else (summed, ctf_probs)),
                )
            timing.big_jit_bucket_s += time.time() - big_jit_t0
            big_jit_bucket_count += 1
            if sparse_big_jit_backprojection:
                sparse_big_jit_bucket_count += 1

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
            if sparse_big_jit_backprojection and (not disable_adjoint_y or not disable_adjoint_ctf):
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
                packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_rotations_np))
            else:
                probs_sum_t_np = None
                reconstruction_take_indices = np.broadcast_to(
                    np.arange(int(bucket.bucket_rotation_count), dtype=np.int32)[None, :],
                    (unpadded_batch_size, int(bucket.bucket_rotation_count)),
                )
                reconstruction_pack_mask_np = reconstruction_rotation_mask_np & local_mask_np
                reconstruction_row_count = int(np.asarray(reconstruction_row_count_jax, dtype=np.int32))
            timing.pack_s += time.time() - pack_t0

            if sparse_big_jit_backprojection and not disable_adjoint_y:
                adjoint_y_t0 = time.time()
                Ft_y = _adjoint_slice_volume_maybe_windowed(
                    flatten_bucket_rows(packed_summed),
                    recon_window_indices,
                    packed_flat_rotations,
                    Ft_y,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    True,
                    use_window=use_window,
                    max_r=float(current_size // 2) if use_window else None,
                    relion_x_half=bool(mstep_relion_x_half),
                )
                if return_profile:
                    _block_until_ready(Ft_y)
                timing.adjoint_y_s += time.time() - adjoint_y_t0

            if sparse_big_jit_backprojection and not disable_adjoint_ctf:
                adjoint_ctf_t0 = time.time()
                Ft_ctf = _adjoint_slice_volume_maybe_windowed(
                    flatten_bucket_rows(packed_ctf_probs),
                    recon_window_indices,
                    packed_flat_rotations,
                    Ft_ctf,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    True,
                    use_window=use_window,
                    max_r=float(current_size // 2) if use_window else None,
                    relion_x_half=bool(mstep_relion_x_half),
                )
                if return_profile:
                    _block_until_ready(Ft_ctf)
                timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

            postprocess_t0 = time.time()
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
                probs_sum_t=probs_sum_t[:unpadded_batch_size] if probs_sum_t_np is None else probs_sum_t_np,
                n_significant_samples=n_significant_samples[:unpadded_batch_size],
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
            continue

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
            score_with_masked_images,
            image_pre_shifts=image_pre_shifts,
            processed_half_cache=processed_half_cache,
            timer=preprocess_profile if return_profile else None,
            synchronize_profile=return_profile,
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
            phase_expanded = tiled_half_image_phase_factors(image_shape, batch_shifts, n_trans)
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
            materialize_recon_projection=not defer_local_noise_projection,
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
        can_use_fused_score_mstep = (
            fused_score_mstep_enabled
            and normalization_log_z_np is None
            and normalization_log_evidence_np is None
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
                local_rotation_log_prior,
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
            if reconstruct_significant_only:
                reconstruction_sample_mask, reconstruction_rotation_mask, n_significant_samples = (
                    compute_reconstruction_support(
                        probs,
                        adaptive_fraction=adaptive_fraction,
                        max_significants=max_significants,
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
            scores = None

        pack_t0 = time.time()
        probs_sum_t_np = None
        if reconstruct_significant_only:
            reconstruction_rotation_mask_np = np.asarray(reconstruction_rotation_mask, dtype=bool)
            transfer_profile["reconstruction_mask_to_host_s"] += time.time() - pack_t0
        else:
            reconstruction_rotation_mask_np = np.asarray(bucket.local_rotation_mask, dtype=bool)
        transfer_t0 = time.time()
        probs_sum_t_np = np.asarray(probs_sum_t, dtype=np.float64)
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
        packed_summed = jnp.take_along_axis(summed, reconstruction_take_indices_jnp[:, :, None], axis=1)
        packed_summed = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_summed, 0.0)
        packed_ctf_probs = jnp.take_along_axis(ctf_probs, reconstruction_take_indices_jnp[:, :, None], axis=1)
        packed_ctf_probs = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_ctf_probs, 0.0)
        packed_flat_rotations = None
        if not disable_adjoint_y or not disable_adjoint_ctf or (accumulate_noise and proj_for_noise is None):
            packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_rotations_np))
        timing.pack_s += time.time() - pack_t0

        if not disable_adjoint_y:
            adjoint_y_t0 = time.time()
            Ft_y = _adjoint_slice_volume_maybe_windowed(
                flatten_bucket_rows(packed_summed),
                recon_window_indices,
                packed_flat_rotations,
                Ft_y,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                True,
                use_window=use_window,
                max_r=float(current_size // 2) if use_window else None,
                relion_x_half=bool(mstep_relion_x_half),
            )
            if return_profile:
                _block_until_ready(Ft_y)
            timing.adjoint_y_s += time.time() - adjoint_y_t0

        if not disable_adjoint_ctf:
            adjoint_ctf_t0 = time.time()
            Ft_ctf = _adjoint_slice_volume_maybe_windowed(
                flatten_bucket_rows(packed_ctf_probs),
                recon_window_indices,
                packed_flat_rotations,
                Ft_ctf,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                True,
                use_window=use_window,
                max_r=float(current_size // 2) if use_window else None,
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
            if proj_for_noise is None:
                packed_proj_for_noise = _project_packed_noise_rows(
                    mean_for_proj=mean_for_proj,
                    packed_flat_rotations=packed_flat_rotations,
                    packed_rotation_count=packed_rotations_np.shape[1],
                    batch_size=batch_size,
                    image_shape=image_shape,
                    proj_volume_shape=proj_volume_shape,
                    disc_type=disc_type,
                    projection_kwargs=projection_kwargs,
                    window_spec=window_spec,
                    n_half=n_half,
                    precision_policy=precision_policy,
                    reconstruction_pack_mask_jnp=reconstruction_pack_mask_jnp,
                )
            else:
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
            flat_proj_abs2_for_noise = jnp.abs(flat_proj_for_noise) ** 2
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
            probs_sum_t=probs_sum_t if probs_sum_t_np is None else probs_sum_t_np,
            n_significant_samples=n_significant_samples,
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

    final_accumulator_t0 = time.time()
    Ft_y, Ft_ctf = enforce_half_volume_x0(
        Ft_y,
        Ft_ctf,
        recon_volume_shape,
        logger=logger,
        label="Exact local",
    )
    if return_half_volume_accumulators:
        logger.info("Exact local M-step: keeping native half-volume accumulators for downstream reconstruction")
    elif mstep_relion_x_half:
        Ft_y, Ft_ctf = relion_x_half_accumulators_to_full(Ft_y, Ft_ctf, recon_volume_shape)
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
        )
    timing.stats_finalize_s += time.time() - stats_finalize_t0

    if debug_score_dump_filter_matches and debug_score_dump_targets and debug_score_dump_iterations is None:
        logger.warning(
            "Requested local score dump indices were not observed in this dataset view: %s",
            sorted(debug_score_dump_targets),
        )
    if (
        debug_fused_posterior_dump_dir is not None
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
            best_pose_rotations=best_pose_rotations,
            best_pose_translations=best_pose_translations,
            best_pose_rotation_ids=best_pose_rotation_ids,
            noise_stats=noise_stats,
        )

    _block_until_ready(Ft_y, Ft_ctf)
    total_wall_time = time.time() - overall_t0
    profile_summary = {
        "big_jit_bucket_count": np.int32(big_jit_bucket_count),
        "sparse_big_jit_bucket_count": np.int32(sparse_big_jit_bucket_count),
        "fused_score_mstep_enabled": np.asarray(fused_score_mstep_enabled),
        "defer_local_noise_projection": np.asarray(defer_local_noise_projection),
        "bucket_build_time_s": np.float64(timing.bucket_build_s),
        "raw_cache_build_time_s": np.float64(timing.raw_cache_build_s),
        "raw_cache_enabled": np.asarray(raw_cache_enabled),
        "processed_half_cache_enabled": np.asarray(processed_half_cache_enabled),
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
        "local_pad_fraction": np.float64(
            0.0 if total_padded_rotations == 0 else 1.0 - total_local_rotations / total_padded_rotations
        ),
        "n_windowed": np.int32(n_windowed),
    }
    return _local_em_return_tuple(
        Ft_y,
        Ft_ctf,
        hard_assignment,
        relion_stats,
        accumulate_noise=accumulate_noise,
        return_profile=True,
        return_best_pose_details=return_best_pose_details,
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
        noise_stats=noise_stats,
        profile_summary=profile_summary,
    )
