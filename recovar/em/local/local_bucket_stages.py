"""Per-bucket stages of the exact local EM engine.

The projection, packed-noise projection, post-processing and adjoint windows
of one hypothesis bucket, the big-JIT argument and capacity planning around
them and the packed reconstruction/rotation gathers they share.
``run_local_em_exact`` drives every bucket through these stages.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.adjoint import adjoint_slice_volume_maybe_windowed as _adjoint_slice_volume_maybe_windowed
from recovar.em.helpers.deterministic_reduce import add_segment_sum
from recovar.em.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.helpers.fourier_window import centered_half_indices_to_fftw_half_indices
from recovar.em.helpers.projection import _validate_centered_relion_projector_pixel_indices
from recovar.em.helpers.projection import compute_noise_block as _compute_noise_block
from recovar.em.helpers.projection import compute_norm_residual_per_image as _compute_norm_residual_per_image
from recovar.em.helpers.projection import compute_projections_block as _compute_projections_block
from recovar.em.helpers.projection import (
    compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
)
from recovar.em.helpers.projection import (
    compute_scale_correction_terms_per_image as _compute_scale_correction_terms_per_image,
)
from recovar.em.helpers.projection import indexed_projection_available as _indexed_projection_available
from recovar.em.helpers.projection import project_indexed_half_spectrum as _project_indexed_half_spectrum
from recovar.em.helpers.shape_buckets import pad_axis, pad_batch_data_ctf_and_valid_mask
from recovar.em.local.flat_local_rows import (
    build_dense_to_flat_local_row_lookup,
    build_pool_flat_local_row_plan,
    encode_flat_local_row_plan,
    map_dense_local_rows_to_flat_rows,
)
from recovar.em.local.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.local.local_big_jit import _reconstruct_fixed_capacity_score_only_result, run_local_bucket_big_jit
from recovar.em.local.local_layout import LocalBucketSpec, _exact_bucket_rotation_size, _local_mstep_rotations
from recovar.em.refinement.projector_preparation import prepare_local_projector_slab
from recovar.em.scoring import compact_candidates
from recovar.em.sparse_pass2 import sparse_pass2_bucketed
from recovar.utils.nvtx_shim import nvtx

logger = logging.getLogger(__name__)


def _noise_wsum_initial_dtype(*, relion_exact_fine_diff2: bool, use_window: bool):
    """Match the initial carry to the direct-Wavg post-bucket dtype."""

    return jnp.float64 if relion_exact_fine_diff2 and use_window else jnp.float32


def _relion_exact_fine_full_to_compact_lookup(
    image_shape,
    current_size,
    n_half,
    window_spec,
):
    """Build RELION fine-score row mapping for cropped or full-size images."""
    compact_indices = (
        window_spec.score_indices_np
        if window_spec.use_window
        else np.arange(int(n_half), dtype=np.int32)
    )
    return sparse_pass2_bucketed._relion_cuda_fine_full_to_compact_lookup(
        image_shape,
        int(current_size),
        compact_indices,
    )


NVTX_DOMAIN_EM = "recovar_em"


EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS = 64_000_000


EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS_ENV = "RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS"


EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM = 512


EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM_ENV = "RECOVAR_EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM"


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
    best_pose_eulers_deg: np.ndarray | None = None


@dataclass(frozen=True)
class _FixedCapacityWholeScoreCallContext:
    """Host metadata retained while score calls execute in one boundary."""

    unpadded_bucket: LocalBucketSpec
    padded_bucket: LocalBucketSpec
    unpadded_batch_size: int


@dataclass
class _LocalProjectionBlock:
    proj_weighted: jnp.ndarray
    proj_for_noise: jnp.ndarray | None


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


def _noise_norm_capacity(n_images: int, *, enabled: bool) -> int:
    """Keep the global norm carry from specializing each packed noise bucket.

    This pads only a scalar per-image accumulator, never image/pose batches.
    Logical indices remain unchanged; publication crops the unused zero tail.
    """
    return ((n_images + 1023) // 1024) * 1024 if enabled else n_images


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


def encode_hard_assignment(rotation_ids, translation_indices, n_trans: int) -> np.ndarray:
    """Return ``rotation_id * n_trans + translation`` in int64.

    Fine local grids at high sampling orders carry rotation ids beyond the
    int32 range (fine HEALPix order 9 is about 3e3 psi steps times 3e6 pixels),
    so the flattened pose index must not be formed or stored in int32.
    """

    return np.asarray(rotation_ids, dtype=np.int64) * int(n_trans) + np.asarray(translation_indices, dtype=np.int64)


def validate_local_relion_projector_window(window_spec, image_shape) -> int | None:
    """Host-side check that every windowed index fits the local RELION projector crop.

    The local pass-2 sites gather ``score_indices``, ``projection_indices`` and
    ``recon_indices`` from a projector cropped to
    ``window_spec.relion_projector_output_size()``.  Inside the big-JIT path
    those indices are tracers, so the trace-time validation in
    ``compute_relion_projector_projections_block`` is skipped; the aliasing
    fixed in 9216a1b8f went undetected for that reason.  Validate the concrete
    NumPy copies once per call instead, and return the crop size used.
    """

    if not window_spec.use_window:
        return None
    projector_output_size = window_spec.relion_projector_output_size()
    if projector_output_size is None:
        return None
    for indices_np in (
        window_spec.score_indices_np,
        window_spec.projection_indices_np,
        window_spec.recon_indices_np,
    ):
        if indices_np is None:
            continue
        _validate_centered_relion_projector_pixel_indices(
            indices_np,
            image_shape=image_shape,
            projector_output_size=int(projector_output_size),
        )
    return int(projector_output_size)


def _relion_local_projector_flat(
    relion_projector_half,
    flat_rotations,
    *,
    image_shape,
    relion_projector_r_max,
    projection_padding_factor,
    projection_kwargs: dict,
    window_spec,
    projection_indices,
):
    """Project flat local rotations through the RELION projector slab at the window's pixel selection.

    ``projection_indices`` is the caller's window selection (``None`` without a
    window); the projector output size, the disk mask and the texture/floorf
    quirks come from ``projection_kwargs`` exactly as before.
    """

    if relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
    relion_projector_half = prepare_local_projector_slab(relion_projector_half)
    relion_texture_interp = projection_kwargs.get("relion_texture_interp")
    relion_acc_double_floorf_quirk = bool(projection_kwargs.get("relion_acc_double_floorf_quirk", False))
    mask_current_image_disk = bool(projection_kwargs.get("mask_current_image_disk", True))
    projector_kwargs = {}
    if window_spec.use_window and window_spec.max_r is not None:
        projector_kwargs["projector_output_size"] = int(window_spec.relion_projector_output_size())
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
        relion_acc_double_floorf_quirk=relion_acc_double_floorf_quirk,
        **projector_kwargs,
    )
    return proj_relion_flat


def _packed_reconstruction_rows(values, take_indices, pack_mask):
    """Gather the packed reconstruction rows of ``values`` (batch, rotation, ...) and zero the padding rows."""

    packed = jnp.take_along_axis(values, take_indices[:, :, None], axis=1)
    return jnp.where(pack_mask[:, :, None], packed, 0.0)


def _packed_bucket_rotations(bucket, reconstruction_take_indices, reconstruction_pack_mask_np, batch_rows=None, rotations_dtype=None):
    """Device copies of the packed take indices and pack mask, and the host rotations gathered along them.

    ``batch_rows`` limits the bucket rows to the unpadded batch (``None`` keeps
    every row); ``rotations_dtype`` casts the scoring rotations the way the
    source-VDAM path does (``None`` keeps their dtype).  Both the scoring and
    the M-step rotations are gathered.
    """

    take = reconstruction_take_indices[:, :, None, None]
    return (
        jnp.asarray(reconstruction_take_indices, dtype=jnp.int32),
        jnp.asarray(reconstruction_pack_mask_np),
        np.take_along_axis(np.asarray(bucket.local_rotations[:batch_rows], dtype=rotations_dtype), take, axis=1),
        np.take_along_axis(_local_mstep_rotations(bucket)[:batch_rows], take, axis=1),
    )


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
        proj_half_flat = _relion_local_projector_flat(
            relion_projector_half,
            flat_rotations,
            image_shape=image_shape,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            projection_kwargs=projection_kwargs,
            window_spec=window_spec,
            projection_indices=(
                (window_spec.projection_indices if materialize_recon_projection else window_spec.score_indices)
                if window_spec.use_window
                else None
            ),
        )
        compact_projection = window_spec.use_window
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
        proj_half_flat = proj_window_flat
        compact_projection = True
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
        compact_projection = False

    if window_spec.use_window:
        if compact_projection:
            if materialize_recon_projection:
                proj_half = proj_half_flat[..., window_spec.score_projection_take].reshape(
                    batch_size,
                    bucket_rotation_count,
                    window_spec.n_score,
                )
                proj_for_noise = proj_half_flat[..., window_spec.recon_projection_take].reshape(
                    batch_size,
                    bucket_rotation_count,
                    window_spec.n_recon,
                )
            else:
                proj_half = proj_half_flat.reshape(batch_size, bucket_rotation_count, window_spec.n_score)
                proj_for_noise = None
        else:
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
        flat_proj_for_noise = _relion_local_projector_flat(
            relion_projector_half,
            packed_flat_rotations,
            image_shape=image_shape,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            projection_kwargs=projection_kwargs,
            window_spec=window_spec,
            projection_indices=(
                (window_spec.recon_indices if window_spec.recon_indices is not None else window_spec.score_indices)
                if window_spec.use_window
                else None
            ),
        )
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


def _unpadded_bucket_rows(bucket, unpadded_batch_size: int) -> dict:
    """The bucket's rotation ids, mask, rotations, source Euler angles and posterior ids limited to the unpadded batch."""

    return dict(
        local_rotation_ids=bucket.local_rotation_ids[:unpadded_batch_size],
        local_rotation_mask=bucket.local_rotation_mask[:unpadded_batch_size],
        local_rotations=bucket.local_rotations[:unpadded_batch_size],
        local_source_eulers=(
            None if bucket.local_source_eulers is None else bucket.local_source_eulers[:unpadded_batch_size]
        ),
        local_rotation_posterior_ids=(
            None
            if bucket.local_rotation_posterior_ids is None
            else bucket.local_rotation_posterior_ids[:unpadded_batch_size]
        ),
    )


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
    host_prefix: bool = False,
    local_source_eulers=None,
):
    """Scatter one local bucket's host-side pose, posterior, and profile stats.

    With host_prefix, small device outputs retain their physical batch until
    their existing NumPy consumer. Slice and decode after that transfer so a
    changing tail does not compile separate device bookkeeping executables.
    """

    image_indices_np = np.asarray(image_indices, dtype=np.int32)
    local_rotation_ids_np = np.asarray(local_rotation_ids, dtype=np.int64)
    local_mask_np = np.asarray(local_rotation_mask, dtype=bool)

    def host_array(value, dtype=None):
        array = np.asarray(value, dtype=dtype)
        return array[:len(image_indices_np)] if host_prefix else array

    transfer_t0 = time.time()
    if host_prefix:
        best_argmax = host_array(best_argmax)
    best_rot_idx = np.asarray(best_argmax // n_trans, dtype=np.int32)
    best_trans_idx = np.asarray(best_argmax % n_trans, dtype=np.int32)
    buffers.transfer_profile["postprocess_argmax_to_host_s"] += time.time() - transfer_t0

    best_rotation_ids = np.take_along_axis(
        local_rotation_ids_np,
        best_rot_idx[:, None],
        axis=1,
    ).reshape(-1)
    if np.any(best_rotation_ids < 0):
        bad = np.flatnonzero(best_rotation_ids < 0)
        row = int(bad[0])
        ids_row = local_rotation_ids_np[row]
        raise RuntimeError(
            "exact local engine selected padded local rotation: "
            f"n_bad={int(bad.size)} of {int(best_rotation_ids.shape[0])} images; first image_index="
            f"{int(image_indices_np[row])} best_rot_slot={int(best_rot_idx[row])} n_slots={int(ids_row.shape[0])} "
            f"n_real_ids={int(np.count_nonzero(ids_row >= 0))} n_mask_true={int(np.count_nonzero(local_mask_np[row]))} "
            f"ids_min_max=({int(ids_row.min())}, {int(ids_row.max())})"
        )
    buffers.hard_assignment[image_indices_np] = encode_hard_assignment(best_rotation_ids, best_trans_idx, n_trans)

    transfer_t0 = time.time()
    log_score_offset = -0.5 * (
        np.squeeze(host_array(batch_norm, np.float64), axis=1)
        if host_prefix
        else np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
    )
    log_z_np = host_array(log_Z)
    best_log_score_np = host_array(best_log_score)
    max_posterior_np = host_array(max_posterior)
    buffers.transfer_profile["postprocess_scores_to_host_s"] += time.time() - transfer_t0
    buffers.log_evidence_per_image[image_indices_np] = log_z_np + log_score_offset
    buffers.best_log_score_per_image[image_indices_np] = best_log_score_np + log_score_offset
    buffers.max_posterior_per_image[image_indices_np] = max_posterior_np

    transfer_t0 = time.time()
    probs_sum_t_np = host_array(probs_sum_t, np.float64)
    collect_significant_counts = collect_profile_stats or buffers.significant_counts is not None
    n_significant_samples_np = (
        host_array(n_significant_samples, np.int32) if collect_significant_counts else None
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
        sample_mask_np = host_array(reconstruction_sample_mask, bool)
        for row, image_idx in enumerate(image_indices_np):
            valid_sample_mask = sample_mask_np[row] & local_mask_np[row, :, None]
            rot_rows, trans_cols = np.nonzero(valid_sample_mask)
            sample_ids = posterior_ids_np[row, rot_rows].astype(np.int64) * int(n_trans) + trans_cols.astype(np.int64)
            if np.any(sample_ids < 0):
                raise RuntimeError("local significant-sample collection encountered padded rotation ids")
            buffers.reconstruction_sample_indices_by_image[int(image_idx)] = sample_ids.astype(np.int64, copy=False)

    if buffers.best_pose_rotations is not None:
        buffers.best_pose_rotations[image_indices_np] = np.take_along_axis(
            np.asarray(local_rotations),
            best_rot_idx[:, None, None, None],
            axis=1,
        ).reshape(-1, 3, 3)
        buffers.best_pose_translations[image_indices_np] = np.asarray(translation_grid)[best_trans_idx]
        buffers.best_pose_rotation_ids[image_indices_np] = best_rotation_ids.astype(np.int64, copy=False)

    if buffers.best_pose_eulers_deg is not None:
        if local_source_eulers is None:
            raise ValueError("source Euler metadata was lost before local winner publication")
        buffers.best_pose_eulers_deg[image_indices_np] = np.take_along_axis(
            np.asarray(local_source_eulers, dtype=np.float64),
            best_rot_idx[:, None, None],
            axis=1,
        ).reshape(-1, 3)

    return significant_sample_count, int(reconstruction_row_count)


def _postprocess_fixed_capacity_whole_score_calls(
    contexts,
    final_carry,
    call_outputs,
    *,
    n_trans: int,
    translation_grid,
    stats_use_reconstruction_probs: bool,
    collect_profile_stats: bool,
    buffers: _LocalPostprocessBuffers,
) -> tuple[int, int]:
    """Scatter one-boundary score outputs with the mature host postprocessor."""

    contexts = tuple(contexts)
    call_outputs = tuple(call_outputs)
    if len(contexts) != len(call_outputs):
        raise RuntimeError(
            "fixed-capacity whole-local score output count does not match its call program"
        )
    total_significant_samples = 0
    total_reconstruction_rows = 0
    for context, call_output in zip(contexts, call_outputs, strict=True):
        result = _reconstruct_fixed_capacity_score_only_result(
            final_carry,
            call_output,
        )
        if any(value is not None for value in result[1:]):
            raise RuntimeError(
                "fixed-capacity whole-local score call returned a non-production topology"
            )
        batch_norm = result.core.batch_norm
        log_Z = result.core.log_Z
        best_log_score = result.core.best_log_score
        best_argmax = result.core.best_argmax
        max_posterior = result.core.max_posterior
        probs_sum_t = result.core.probs_sum_t
        reconstruction_probs_sum_t = result.core.reconstruction_probs_sum_t
        n_significant_samples = result.core.n_significant_samples
        reconstruction_sample_mask = result.core.reconstruction_sample_mask
        reconstruction_rotation_mask = result.core.reconstruction_rotation_mask
        reconstruction_row_count_jax = result.core.reconstruction_row_count
        bucket = context.padded_bucket
        unpadded_bucket = context.unpadded_bucket
        unpadded_batch_size = int(context.unpadded_batch_size)
        reconstruction_rotation_mask_np = np.asarray(
            reconstruction_rotation_mask[:unpadded_batch_size],
            dtype=bool,
        )
        local_mask_np = np.asarray(
            bucket.local_rotation_mask[:unpadded_batch_size],
            dtype=bool,
        )
        reconstruction_take_indices = np.broadcast_to(
            np.arange(int(bucket.bucket_rotation_count), dtype=np.int32)[None, :],
            (unpadded_batch_size, int(bucket.bucket_rotation_count)),
        )
        reconstruction_pack_mask_np = reconstruction_rotation_mask_np & local_mask_np
        reconstruction_row_count = int(
            np.asarray(reconstruction_row_count_jax, dtype=np.int32)
        )
        stats_probs_sum_t = (
            reconstruction_probs_sum_t
            if stats_use_reconstruction_probs
            else probs_sum_t
        )
        significant_sample_count, reconstruction_row_count = _postprocess_local_bucket(
            image_indices=unpadded_bucket.image_indices,
            **_unpadded_bucket_rows(bucket, unpadded_batch_size),
            translation_grid=translation_grid,
            n_trans=n_trans,
            best_argmax=best_argmax[:unpadded_batch_size],
            batch_norm=batch_norm[:unpadded_batch_size],
            log_Z=log_Z[:unpadded_batch_size],
            best_log_score=best_log_score[:unpadded_batch_size],
            max_posterior=max_posterior[:unpadded_batch_size],
            probs_sum_t=stats_probs_sum_t[:unpadded_batch_size],
            n_significant_samples=n_significant_samples[:unpadded_batch_size],
            reconstruction_sample_mask=reconstruction_sample_mask[:unpadded_batch_size],
            collect_profile_stats=collect_profile_stats,
            reconstruction_row_count=reconstruction_row_count,
            reconstruction_take_indices=reconstruction_take_indices,
            reconstruction_pack_mask=reconstruction_pack_mask_np,
            buffers=buffers,
        )
        if collect_profile_stats:
            total_significant_samples += significant_sample_count
            total_reconstruction_rows += int(reconstruction_row_count)
        logger.debug(
            "Exact local one-boundary score call: %d images, bucket_rot=%d, "
            "total_local_rot=%d",
            unpadded_batch_size,
            int(bucket.bucket_rotation_count),
            int(np.sum(unpadded_bucket.actual_rotation_counts)),
        )
    return total_significant_samples, total_reconstruction_rows


def _pad_local_big_jit_image_axis(bucket: LocalBucketSpec, batch_data, ctf_params):
    """Pad a local big-JIT bucket to its planned image shape class."""

    actual_batch_size = int(bucket.image_indices.shape[0])
    padded_batch_size = int(max(actual_batch_size, getattr(bucket, "bucket_image_count", actual_batch_size)))
    if actual_batch_size == padded_batch_size:
        return bucket, batch_data, ctf_params, np.ones(actual_batch_size, dtype=bool), actual_batch_size

    # pad_axis (np.pad) preserves the source array's own dtype -- do not
    # force float32 here, or every field of an already-correctly-built
    # (possibly float64) bucket gets silently truncated back to float32 on
    # this hot per-bucket padding path, same class of bug as
    # local_layout.py:bucket_local_hypothesis_layout.
    padded_rotations = pad_axis(bucket.local_rotations, 0, padded_batch_size, value=0)
    padded_rotations[actual_batch_size:] = np.eye(3, dtype=padded_rotations.dtype)
    padded_mstep_rotations = pad_axis(
        _local_mstep_rotations(bucket),
        0,
        padded_batch_size,
        value=0,
    )
    padded_mstep_rotations[actual_batch_size:] = np.eye(3, dtype=padded_mstep_rotations.dtype)
    padded_bucket = LocalBucketSpec(
        image_indices=np.asarray(bucket.image_indices, dtype=np.int32),
        bucket_image_count=padded_batch_size,
        bucket_rotation_count=int(bucket.bucket_rotation_count),
        actual_rotation_counts=pad_axis(bucket.actual_rotation_counts, 0, padded_batch_size, value=0).astype(np.int32),
        local_rotation_ids=pad_axis(bucket.local_rotation_ids, 0, padded_batch_size, value=-1).astype(np.int32),
        local_rotations=padded_rotations,
        local_mstep_rotations=padded_mstep_rotations,
        local_source_eulers=(
            None
            if bucket.local_source_eulers is None
            else pad_axis(bucket.local_source_eulers, 0, padded_batch_size, value=0)
        ),
        local_rotation_log_prior=pad_axis(
            bucket.local_rotation_log_prior,
            0,
            padded_batch_size,
            value=-1e30,
        ),
        local_rotation_mask=pad_axis(bucket.local_rotation_mask, 0, padded_batch_size, value=False).astype(bool),
        translation_log_prior=pad_axis(bucket.translation_log_prior, 0, padded_batch_size, value=0),
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


def _plan_flat_local_row_capacities(
    bucket_specs,
    *,
    rotation_block_size: int,
    exact_local_bucket_radix: int,
    stable_rectangular_capacity: bool = False,
) -> dict[tuple[int, int], int]:
    """Choose one packed-row shape for every existing dense bucket ABI.

    The stable policy deliberately reuses the mature rectangular ``B x R``
    ABI. Logical pool rows still occupy the exact same source-ordered prefix;
    only a score-inert physical tail is appended. The shared projector still
    evaluates that tail, but validity-aware fine CUDA returns ``+inf`` before
    pixel work while repeated iterations can reuse the same compiled shape.
    """

    capacities: dict[tuple[int, int], int] = {}
    for bucket in bucket_specs:
        physical_image_count = int(np.asarray(bucket.image_indices).shape[0])
        dense_batch_size = max(
            physical_image_count,
            int(getattr(bucket, "bucket_image_count", physical_image_count)),
        )
        dense_rotation_count = int(bucket.bucket_rotation_count)
        plan = build_pool_flat_local_row_plan(
            np.asarray(bucket.actual_rotation_counts, dtype=np.int32),
            dense_rotation_count,
            pool_size=3,
            rotation_block_size=rotation_block_size,
            exact_local_bucket_radix=exact_local_bucket_radix,
            dense_batch_size=dense_batch_size,
        )
        key = (dense_batch_size, dense_rotation_count)
        required_capacity = int(plan.packed_row_count)
        if stable_rectangular_capacity:
            required_capacity = dense_batch_size * dense_rotation_count
        capacities[key] = max(capacities.get(key, 0), required_capacity)
    return capacities


def _build_flat_local_row_argument(
    bucket: LocalBucketSpec,
    capacities: dict[tuple[int, int], int],
    *,
    dense_batch_size: int,
    rotation_block_size: int,
    exact_local_bucket_radix: int,
) -> np.ndarray:
    """Materialize one source-ordered packed plan at its shared static shape."""

    dense_rotation_count = int(bucket.bucket_rotation_count)
    key = (int(dense_batch_size), dense_rotation_count)
    if key not in capacities:
        raise ValueError(f"flat local row capacity is missing dense bucket ABI {key}")
    plan = build_pool_flat_local_row_plan(
        np.asarray(bucket.actual_rotation_counts, dtype=np.int32),
        dense_rotation_count,
        pool_size=3,
        rotation_block_size=rotation_block_size,
        exact_local_bucket_radix=exact_local_bucket_radix,
        packed_row_count=int(capacities[key]),
        dense_batch_size=int(dense_batch_size),
    )
    return encode_flat_local_row_plan(plan)


_FINE_JOB_BUCKET_QUANTUM_ENV = "RECOVAR_EXACT_FINE_JOB_BUCKET_QUANTUM"


_FINE_JOB_BUCKET_QUANTUM_DEFAULT = 65536


_FINE_JOB_MIN_CAPACITY = 4096


def _local_fine_candidate_mask(
    bucket: LocalBucketSpec,
    valid_image_mask,
) -> np.ndarray:
    """Return the canonical dense candidate mask used by both compact ABIs."""

    rotation_mask = np.asarray(bucket.local_rotation_mask, dtype=bool)
    valid_image_mask = np.asarray(valid_image_mask, dtype=bool)
    if rotation_mask.ndim != 2 or valid_image_mask.shape != (rotation_mask.shape[0],):
        raise ValueError(
            "fused fine scoring requires aligned bucket and valid-image axes"
        )
    n_translations = int(np.asarray(bucket.translation_log_prior).shape[1])
    candidate_mask = np.broadcast_to(
        rotation_mask[:, :, None],
        (*rotation_mask.shape, n_translations),
    ).copy()
    if bucket.local_sample_mask is not None:
        sample_mask = np.asarray(bucket.local_sample_mask, dtype=bool)
        if sample_mask.shape != candidate_mask.shape:
            raise ValueError(
                "fused fine sample mask does not match the dense source layout"
            )
        candidate_mask &= sample_mask
    candidate_mask &= valid_image_mask[:, None, None]
    return candidate_mask


def _plan_local_fine_job_capacities(
    bucket_specs,
) -> dict[tuple[int, int], int]:
    """Choose stable global-job capacity per physical big-JIT bucket ABI."""

    try:
        large_quantum = int(
            os.environ.get(
                _FINE_JOB_BUCKET_QUANTUM_ENV,
                str(_FINE_JOB_BUCKET_QUANTUM_DEFAULT),
            )
        )
    except ValueError as exc:
        raise ValueError(f"{_FINE_JOB_BUCKET_QUANTUM_ENV} must be an integer") from exc
    if large_quantum < _FINE_JOB_MIN_CAPACITY:
        raise ValueError(
            f"{_FINE_JOB_BUCKET_QUANTUM_ENV} must be at least "
            f"{_FINE_JOB_MIN_CAPACITY}"
        )

    required_by_abi: dict[tuple[int, int], int] = {}
    dense_capacity_by_abi: dict[tuple[int, int], int] = {}
    for bucket in bucket_specs:
        physical_image_count = int(np.asarray(bucket.image_indices).shape[0])
        dense_batch_size = max(
            physical_image_count,
            int(getattr(bucket, "bucket_image_count", physical_image_count)),
        )
        dense_rotation_count = int(bucket.bucket_rotation_count)
        candidate_mask = _local_fine_candidate_mask(
            bucket,
            np.ones(physical_image_count, dtype=bool),
        )
        required = int(np.count_nonzero(candidate_mask))
        key = (dense_batch_size, dense_rotation_count)
        required_by_abi[key] = max(required_by_abi.get(key, 0), required)
        dense_capacity_by_abi[key] = max(
            dense_capacity_by_abi.get(key, 0),
            dense_batch_size * dense_rotation_count * candidate_mask.shape[2],
        )

    return {
        key: min(
            dense_capacity_by_abi[key],
            max(
                _FINE_JOB_MIN_CAPACITY,
                _exact_bucket_rotation_size(
                    required,
                    5000,
                    large_bucket_quantum=large_quantum,
                ),
            ),
        )
        for key, required in required_by_abi.items()
    }


def _build_local_fused_pair_fine_arguments(
    bucket: LocalBucketSpec,
    flat_local_row_argument,
    valid_image_mask,
    *,
    fine_job_bucket_size: int | None = None,
) -> dict[str, np.ndarray | int]:
    """Map the mature compact-pair ABI onto packed local projection rows.

    Candidate order and pair padding come from the shared compact-pass-2 host
    encoder.  This local bridge adds only the projection-storage row mapping
    needed by the fused CUDA ABI; it does not define another pair layout.
    """

    rotation_mask = np.asarray(bucket.local_rotation_mask, dtype=bool)
    candidate_mask = _local_fine_candidate_mask(bucket, valid_image_mask)

    pair_arrays = compact_candidates.build_compact_pair_index_arrays(
        candidate_mask,
    )
    pair_mask = np.asarray(pair_arrays["pair_mask"], dtype=bool)
    local_rotation_rows = np.asarray(
        pair_arrays["local_rotation_row"],
        dtype=np.int32,
    )
    dense_to_flat = build_dense_to_flat_local_row_lookup(
        flat_local_row_argument,
        batch_size=rotation_mask.shape[0],
        dense_rotation_count=rotation_mask.shape[1],
    )
    reference_rows = map_dense_local_rows_to_flat_rows(
        dense_to_flat,
        local_rotation_rows,
        pair_mask,
    )
    reference_rows = np.where(pair_mask, reference_rows, -1).astype(
        np.int32,
        copy=False,
    )
    flat_jobs = compact_candidates.build_compact_fine_job_plan_from_pair_arrays(
        pair_arrays,
        dense_to_flat,
        job_bucket_size=fine_job_bucket_size,
    )
    return {
        **pair_arrays,
        **flat_jobs,
        "reference_row": reference_rows,
        "valid_pair_count": int(np.sum(pair_arrays["pair_counts"], dtype=np.int64)),
        "dense_candidate_capacity": int(candidate_mask.size),
    }


def _invoke_local_bucket_big_jit(*args, **kwargs):
    """Single shared numeric invocation for mature and fixed call views."""

    return run_local_bucket_big_jit(*args, **kwargs)


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
        # Reorder only -- preserve each field's own dtype (see
        # bucket_local_hypothesis_layout's docstring for why forcing float32
        # here would be wrong under double-precision scoring).
        local_rotations=np.asarray(bucket.local_rotations)[order],
        local_mstep_rotations=_local_mstep_rotations(bucket)[order],
        local_source_eulers=(None if bucket.local_source_eulers is None else bucket.local_source_eulers[order]),
        local_rotation_log_prior=np.asarray(bucket.local_rotation_log_prior)[order],
        local_rotation_mask=np.asarray(bucket.local_rotation_mask[order], dtype=bool),
        translation_log_prior=np.asarray(bucket.translation_log_prior)[order],
        local_rotation_posterior_ids=(
            None
            if bucket.local_rotation_posterior_ids is None
            else np.asarray(bucket.local_rotation_posterior_ids[order], dtype=np.int32)
        ),
        local_sample_mask=(
            None if bucket.local_sample_mask is None else np.asarray(bucket.local_sample_mask[order], dtype=bool)
        ),
    )


def _build_reconstruction_pack_indices(
    significant_rotation_mask: np.ndarray,
    local_rotation_mask: np.ndarray,
    rotation_block_size: int,
    *,
    exact_local_bucket_radix: int | None = None,
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
        exact_local_bucket_radix=exact_local_bucket_radix,
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
    *,
    exact_local_bucket_radix: int | None = None,
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
        exact_local_bucket_radix=exact_local_bucket_radix,
    )


def _return_local_big_jit_mstep_tensors(
    *,
    sparse_big_jit_backprojection: bool,
    source_faithful_bpref: bool,
    grouped_reconstruction: bool,
    reconstruct_significant_only: bool,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
) -> bool:
    """Keep grouped os0 BPref scatter outside the ungrouped fused kernel.

    The direct fused x-half call owns one accumulator.  A joint pseudo-halfset
    stream owns one accumulator per half, so the keep-all oversampling-zero
    path must return its physical operands and use the existing group-aware
    inline-projector scatter in the outer loop.
    """

    grouped_keep_all_source_mstep = bool(
        source_faithful_bpref
        and grouped_reconstruction
        and not reconstruct_significant_only
    )
    return bool(
        (sparse_big_jit_backprojection or grouped_keep_all_source_mstep)
        and (not disable_adjoint_y or not disable_adjoint_ctf)
    )


@nvtx.annotate("local.run_local_em_exact", color="purple", domain=NVTX_DOMAIN_EM)
def _accumulate_packed_noise_chunk(
    chunk_proj_for_noise,
    *,
    chunk_start,
    chunk_stop,
    defer_packed_mstep_reduction,
    packed_reconstruction_probs,
    shifted_noise_split,
    ctf2_over_nv_recon,
    packed_summed_masked_noise,
    packed_ctf_probs,
    noise_variance_for_noise,
    shell_indices_noise,
    n_shells,
    return_noise_split,
    batch_scale,
    scale_correction_pixel_mask,
    bucket_group_ids,
    block_noise_shells,
    block_a2_shells,
    block_xa_shells,
    block_norm_residual,
    noise_scale_xa,
    noise_scale_aa,
):
    """Add one packed rotation chunk to the exact local noise statistics.

    RELION's ``storeWeightedSums`` accumulates, per particle, the posterior
    weighted noise residual shells, the norm-correction residual and the
    group-scale ``XA``/``AA`` sums from the same projections. The chunk's
    weighted sums come from the packed posteriors when the M-step reduction
    is deferred, otherwise from the precomputed packed sums. Both projection
    sources of the exact local engine (deferred projection and cached
    projection rows) accumulate through this function. Returns the six
    updated accumulators.
    """

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
        noise_scale_xa = add_segment_sum(noise_scale_xa, bucket_group_ids, scale_xa_per_image.astype(noise_scale_xa.dtype))
        noise_scale_aa = add_segment_sum(noise_scale_aa, bucket_group_ids, scale_aa_per_image.astype(noise_scale_aa.dtype))
    return block_noise_shells, block_a2_shells, block_xa_shells, block_norm_residual, noise_scale_xa, noise_scale_aa
