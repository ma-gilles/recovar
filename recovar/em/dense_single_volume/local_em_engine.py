"""Exact per-image local EM engine for RELION-mode local search."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar import utils
import recovar.core as core
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.em_primitives import (
    _adjoint_slice_volume_half,
    _adjoint_slice_volume_windowed,
    _block_until_ready,
    _compute_noise_block,
    _compute_projections_block,
    make_half_image_weights,
    make_shell_indices_half,
)
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_indices_np
from recovar.em.dense_single_volume.helpers.types import NoiseStats, RelionStats
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.dense_single_volume.local_layout import (
    LocalBucketSpec,
    LocalHypothesisLayout,
    _exact_bucket_rotation_size,
    bucket_local_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_score_pass import (
    compute_reconstruction_support,
    normalize_local_scores,
    score_local_bucket,
)

logger = logging.getLogger(__name__)


def _fetch_indexed_batch(experiment_dataset, image_indices):
    batch_iter = experiment_dataset.iter_batches(
        len(image_indices),
        indices=np.asarray(image_indices),
        by_image=False,
    )
    batch_data, _, _, ctf_params, _, _, indices = next(batch_iter)
    return jnp.asarray(batch_data), ctf_params, np.asarray(indices)


def _reorder_bucket_to_indices(bucket: LocalBucketSpec, returned_indices: np.ndarray) -> LocalBucketSpec:
    if np.array_equal(returned_indices, bucket.image_indices):
        return bucket
    position = {int(idx): pos for pos, idx in enumerate(np.asarray(bucket.image_indices).tolist())}
    order = np.asarray([position[int(idx)] for idx in np.asarray(returned_indices).tolist()], dtype=np.int32)
    return LocalBucketSpec(
        image_indices=np.asarray(returned_indices, dtype=np.int32),
        bucket_rotation_count=int(bucket.bucket_rotation_count),
        actual_rotation_counts=np.asarray(bucket.actual_rotation_counts[order], dtype=np.int32),
        local_rotation_ids=np.asarray(bucket.local_rotation_ids[order], dtype=np.int32),
        local_rotations=np.asarray(bucket.local_rotations[order], dtype=np.float32),
        local_rotation_log_prior=np.asarray(bucket.local_rotation_log_prior[order], dtype=np.float32),
        local_rotation_mask=np.asarray(bucket.local_rotation_mask[order], dtype=bool),
        translation_log_prior=np.asarray(bucket.translation_log_prior[order], dtype=np.float32),
    )


def _prepare_local_exact_bucket(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance_half,
    translations,
    config,
    norm_half_weights,
    batch_size: int,
    n_trans: int,
    score_with_masked_images: bool,
):
    """Prepare score, reconstruction, and noise inputs for one local bucket.

    This keeps the exact-local path separate from the dense engine and avoids
    recomputing CTF / translation tiling scaffolding across masked, unmasked,
    and noise-specific preprocessing.
    """

    def _process_half(apply_image_mask: bool):
        process_half_fn = getattr(experiment_dataset, "process_images_half", None)
        if process_half_fn is not None:
            return process_half_fn(batch, apply_image_mask=apply_image_mask)
        processed_full = config.process_fn(batch, apply_image_mask=apply_image_mask)
        return fourier_transform_utils.full_image_to_half_image(processed_full, config.image_shape)

    ctf_half = config.compute_ctf_half(ctf_params)
    ctf2_over_nv_half = ctf_half**2 / noise_variance_half
    translations_tiled = jnp.repeat(translations[None], batch_size, axis=0).reshape(batch_size * n_trans, -1)

    processed_score_half = _process_half(score_with_masked_images)
    score_weighted_half = processed_score_half * ctf_half / noise_variance_half
    score_weighted_half_tiled = jnp.repeat(score_weighted_half[:, None, :], n_trans, axis=1).reshape(
        batch_size * n_trans, -1
    )
    shifted_score_half = core.translate_images(
        score_weighted_half_tiled,
        translations_tiled,
        config.image_shape,
        half_image=True,
    )
    batch_norm = jnp.sum(
        (jnp.abs(processed_score_half) ** 2 / noise_variance_half) * norm_half_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real
    if score_with_masked_images:
        processed_recon_half = _process_half(False)
        recon_weighted_half = processed_recon_half * ctf_half / noise_variance_half
        recon_weighted_half_tiled = jnp.repeat(recon_weighted_half[:, None, :], n_trans, axis=1).reshape(
            batch_size * n_trans, -1
        )
        shifted_recon_half = core.translate_images(
            recon_weighted_half_tiled,
            translations_tiled,
            config.image_shape,
            half_image=True,
        )
    else:
        shifted_recon_half = shifted_score_half
    return shifted_score_half, shifted_recon_half, batch_norm, ctf2_over_nv_half, processed_score_half


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


def _parse_debug_score_dump_request():
    """Return the optional debug score-dump request from the environment.

    This is intentionally debug-only and out of the public refinement API.
    It lets us dump a handful of current exact-local score tensors for direct
    RELION-vs-RECOVAR parity analysis without dragging heavyweight score-dump
    plumbing through the hot path.
    """

    dump_dir = os.environ.get("RECOVAR_LOCAL_SCORE_DUMP_DIR")
    dump_indices = os.environ.get("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES")
    if not dump_dir or not dump_indices:
        return None, set()
    targets = set()
    for token in dump_indices.replace(",", " ").split():
        token = token.strip()
        if token:
            targets.add(int(token))
    if not targets:
        return None, set()
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    return dump_path, targets


def _maybe_write_debug_score_dump(
    *,
    experiment_dataset,
    local_layout,
    bucket,
    image_pre_shifts,
    scores,
    probs,
    log_Z,
    best_log_score,
    max_posterior,
    reconstruction_sample_mask,
    reconstruction_rotation_mask,
    n_significant_samples,
    current_size,
    dump_dir: Path | None,
    pending_targets: set[int],
):
    """Dump one-image local score tensors for the requested original ids."""

    if dump_dir is None or not pending_targets:
        return pending_targets

    original_image_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(bucket.image_indices),
        dtype=np.int64,
    )
    target_rows = [row for row, original_idx in enumerate(original_image_indices.tolist()) if int(original_idx) in pending_targets]
    if not target_rows:
        return pending_targets

    scores_np = np.asarray(scores, dtype=np.float32)
    probs_np = np.asarray(probs, dtype=np.float32)
    log_Z_np = np.asarray(log_Z, dtype=np.float32)
    best_log_score_np = np.asarray(best_log_score, dtype=np.float32)
    max_posterior_np = np.asarray(max_posterior, dtype=np.float32)
    reconstruction_sample_mask_np = np.asarray(reconstruction_sample_mask, dtype=bool)
    reconstruction_rotation_mask_np = np.asarray(reconstruction_rotation_mask, dtype=bool)
    n_significant_samples_np = np.asarray(n_significant_samples, dtype=np.int32)

    for row in target_rows:
        original_idx = int(original_image_indices[row])
        local_idx = int(bucket.image_indices[row])
        actual_count = int(bucket.actual_rotation_counts[row])
        local_rotation_ids = np.asarray(bucket.local_rotation_ids[row, :actual_count], dtype=np.int32)
        local_rotation_matrices = np.asarray(bucket.local_rotations[row, :actual_count], dtype=np.float32)
        local_rotation_eulers = np.asarray(
            utils.R_to_relion(local_rotation_matrices, degrees=True),
            dtype=np.float32,
        )
        rotation_mask = np.asarray(bucket.local_rotation_mask[row, :actual_count], dtype=bool)
        rotation_log_prior = np.asarray(bucket.local_rotation_log_prior[row, :actual_count], dtype=np.float32)
        translation_log_prior = np.asarray(bucket.translation_log_prior[row], dtype=np.float32)
        total_scores = np.asarray(scores_np[row, :actual_count, :], dtype=np.float32)
        raw_scores = total_scores - rotation_log_prior[:, None] - translation_log_prior[None, :]
        raw_scores = np.where(rotation_mask[:, None], raw_scores, -np.inf)
        posterior = np.asarray(probs_np[row, :actual_count, :], dtype=np.float32)
        n_trans = int(translation_log_prior.shape[0])
        translation_indices = np.arange(n_trans, dtype=np.int32)
        best_score_flat = int(np.argmax(total_scores))
        best_score_rotation_index, best_score_translation_index = np.unravel_index(
            best_score_flat,
            total_scores.shape,
        )
        best_posterior_flat = int(np.argmax(posterior))
        best_posterior_rotation_index, best_posterior_translation_index = np.unravel_index(
            best_posterior_flat,
            posterior.shape,
        )
        reconstruction_sample_mask_row = np.asarray(
            reconstruction_sample_mask_np[row, :actual_count, :],
            dtype=bool,
        )
        reconstruction_rotation_mask_row = np.asarray(
            reconstruction_rotation_mask_np[row, :actual_count],
            dtype=bool,
        )

        dump_path = dump_dir / f"local_score_image_{original_idx}.npz"
        np.savez_compressed(
            dump_path,
            selected_global_image_indices=np.array([original_idx], dtype=np.int64),
            selected_local_image_indices=np.array([local_idx], dtype=np.int64),
            pass2_scores_raw=raw_scores[None, :, :],
            pass2_scores_total=total_scores[None, :, :],
            rotation_log_prior=rotation_log_prior[None, :],
            translation_log_prior=translation_log_prior[None, :],
            rotation_candidate_mask=rotation_mask[None, :],
            local_rotation_indices=local_rotation_ids,
            local_rotation_pixel_indices=(local_rotation_ids % int(local_layout.n_pixels)).astype(np.int64),
            local_rotation_psi_indices=(local_rotation_ids // int(local_layout.n_pixels)).astype(np.int64),
            local_rotation_eulers=local_rotation_eulers,
            local_rotation_matrices=local_rotation_matrices,
            translations=np.asarray(local_layout.translation_grid, dtype=np.float32),
            candidate_pose_rotation_indices=np.repeat(local_rotation_ids[:, None], n_trans, axis=1),
            candidate_pose_translation_indices=np.broadcast_to(
                translation_indices[None, :],
                (actual_count, n_trans),
            ),
            image_pre_shift=(
                np.asarray(image_pre_shifts[local_idx], dtype=np.float32)
                if image_pre_shifts is not None
                else np.array([], dtype=np.float32)
            ),
            posterior=posterior[None, :, :],
            reconstruction_sample_mask=reconstruction_sample_mask_row[None, :, :],
            reconstruction_rotation_mask=reconstruction_rotation_mask_row[None, :],
            n_significant_samples=np.array([int(n_significant_samples_np[row])], dtype=np.int32),
            max_posterior=np.array([float(max_posterior_np[row])], dtype=np.float32),
            log_Z=np.array([float(log_Z_np[row])], dtype=np.float32),
            best_score=np.array([float(best_log_score_np[row])], dtype=np.float32),
            best_score_rotation_local_index=np.array([int(best_score_rotation_index)], dtype=np.int32),
            best_score_translation_index=np.array([int(best_score_translation_index)], dtype=np.int32),
            best_score_rotation_global_id=np.array(
                [int(local_rotation_ids[int(best_score_rotation_index)])],
                dtype=np.int32,
            ),
            best_score_translation=np.asarray(
                local_layout.translation_grid[
                    int(best_score_translation_index) : int(best_score_translation_index) + 1
                ],
                dtype=np.float32,
            ),
            best_posterior_rotation_local_index=np.array([int(best_posterior_rotation_index)], dtype=np.int32),
            best_posterior_translation_index=np.array([int(best_posterior_translation_index)], dtype=np.int32),
            best_posterior_rotation_global_id=np.array(
                [int(local_rotation_ids[int(best_posterior_rotation_index)])],
                dtype=np.int32,
            ),
            best_posterior_translation=np.asarray(
                local_layout.translation_grid[
                    int(best_posterior_translation_index) : int(best_posterior_translation_index) + 1
                ],
                dtype=np.float32,
            ),
            current_size=np.array([int(current_size) if current_size is not None else -1], dtype=np.int32),
            n_rot=np.array([actual_count], dtype=np.int32),
            n_trans=np.array([n_trans], dtype=np.int32),
            grid_n_pixels=np.array([int(local_layout.n_pixels)], dtype=np.int32),
            grid_n_psi=np.array([int(local_layout.n_psi)], dtype=np.int32),
        )
        pending_targets.remove(original_idx)

    return pending_targets


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
    use_float64_projections: bool = False,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    image_corrections: np.ndarray | None = None,
    scale_corrections: np.ndarray | None = None,
    image_pre_shifts: np.ndarray | None = None,
    return_profile: bool = False,
    disable_adjoint_y: bool = False,
    disable_adjoint_ctf: bool = False,
    max_hypotheses_per_microbatch: int = 32768,
    reconstruct_significant_only: bool = False,
    adaptive_fraction: float = 0.999,
    max_significants: int = -1,
):
    """Run exact local EM over per-image local hypothesis sets."""

    overall_t0 = time.time()
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    n_trans = int(local_layout.translation_grid.shape[0])
    n_images = int(local_layout.n_images)
    debug_score_dump_dir, debug_score_dump_targets = _parse_debug_score_dump_request()

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
    # TODO(DENSE_ENGINE_BOUNDARY/E005): revisit half-volume accumulation for
    # local exact backprojection once we can prove the weighted local row sums
    # satisfy the Hermitian assumptions required for exact half-volume folding.
    recon_volume_size = int(np.prod(recon_volume_shape))

    use_window = current_size is not None and current_size < image_shape[0]
    if use_window:
        score_window_indices_np, n_windowed = make_fourier_window_indices_np(
            image_shape,
            int(current_size),
            square=square_window,
            include_dc=False,
        )
        recon_window_indices_np, n_recon_windowed = make_fourier_window_indices_np(
            image_shape,
            int(current_size),
            square=square_window,
            include_dc=True,
        )
        window_indices = jnp.asarray(score_window_indices_np, dtype=jnp.int32)
        recon_window_indices = jnp.asarray(recon_window_indices_np, dtype=jnp.int32)
    else:
        score_window_indices_np = None
        recon_window_indices_np = None
        window_indices = None
        recon_window_indices = None
        n_windowed = n_half
        n_recon_windowed = n_half

    if half_spectrum_scoring:
        half_weights = jnp.ones(n_half, dtype=jnp.float32)
    else:
        half_weights = make_half_image_weights(image_shape)
    norm_half_weights = make_half_image_weights(image_shape)
    half_weights_windowed = half_weights if window_indices is None else half_weights[window_indices]
    noise_variance_half = fourier_transform_utils.full_image_to_half_image(
        noise_variance.reshape(1, -1),
        image_shape,
    ).squeeze()

    Ft_y = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    log_evidence_per_image = np.empty(n_images, dtype=np.float32)
    best_log_score_per_image = np.empty(n_images, dtype=np.float32)
    max_posterior_per_image = np.empty(n_images, dtype=np.float32)
    rotation_posterior_sums = np.zeros(int(local_layout.n_global_rotations), dtype=np.float64)

    noise_wsum = None
    noise_img_power = None
    noise_sumw = 0.0
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        shell_indices_half = make_shell_indices_half(image_shape)
        shell_indices_noise = shell_indices_half if recon_window_indices is None else shell_indices_half[recon_window_indices]
        noise_variance_for_noise = (
            noise_variance_half if recon_window_indices is None else noise_variance_half[recon_window_indices]
        )
        noise_wsum = np.zeros(n_shells, dtype=np.float64)
        noise_img_power = np.zeros(n_shells, dtype=np.float64)

    bucket_build_time = 0.0
    batch_fetch_time = 0.0
    preprocess_time = 0.0
    projection_time = 0.0
    score_time = 0.0
    normalize_time = 0.0
    significance_time = 0.0
    postprocess_time = 0.0
    mstep_time = 0.0
    adjoint_y_time = 0.0
    adjoint_ctf_time = 0.0
    noise_time = 0.0
    host_stats_time = 0.0
    total_local_rotations = int(local_layout.total_local_rotations)
    seen_global_rotations = np.zeros(rotation_posterior_sums.shape[0], dtype=bool) if rotation_posterior_sums.size else np.zeros(0, dtype=bool)
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
    bucket_build_t0 = time.time()
    bucket_specs = bucket_local_hypothesis_layout(
        local_layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
    )
    bucket_build_time += time.time() - bucket_build_t0

    for bucket in bucket_specs:
        n_chunks += 1
        chunk_sizes.append(int(bucket.image_indices.shape[0]))
        chunk_local_rotations.append(int(np.sum(bucket.actual_rotation_counts)))
        chunk_padded_rotations.append(int(bucket.image_indices.shape[0] * bucket.bucket_rotation_count))
        total_padded_rotations += int(bucket.image_indices.shape[0] * bucket.bucket_rotation_count)
        local_total_hypotheses += int(np.sum(bucket.actual_rotation_counts) * n_trans)
        fetch_t0 = time.time()
        batch_data, ctf_params, fetched_indices = _fetch_indexed_batch(experiment_dataset, bucket.image_indices)
        batch_fetch_time += time.time() - fetch_t0
        bucket = _reorder_bucket_to_indices(bucket, fetched_indices)
        batch_size = int(bucket.image_indices.shape[0])

        preprocess_t0 = time.time()
        (
            shifted_half,
            shifted_recon_half,
            batch_norm,
            ctf2_over_nv_half,
            processed_score_half,
        ) = _prepare_local_exact_bucket(
            experiment_dataset,
            batch_data,
            ctf_params,
            noise_variance_half,
            local_layout.translation_grid,
            config,
            norm_half_weights,
            batch_size,
            n_trans,
            score_with_masked_images,
        )
        if image_corrections is not None:
            batch_corr = jnp.asarray(image_corrections[np.asarray(bucket.image_indices)])
            corr_expanded = jnp.repeat(batch_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            shifted_recon_half = shifted_recon_half * corr_expanded[:, None]
            batch_norm = batch_norm * (batch_corr**2)[:, None]

        if scale_corrections is not None:
            batch_scale = jnp.asarray(scale_corrections[np.asarray(bucket.image_indices)])
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]

        if image_pre_shifts is not None:
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
        preprocess_time += time.time() - preprocess_t0

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
        )
        if use_window:
            proj_half = proj_half_flat[:, window_indices].reshape(batch_size, bucket.bucket_rotation_count, n_windowed)
            proj_abs2 = proj_abs2_half_flat[:, window_indices].reshape(batch_size, bucket.bucket_rotation_count, n_windowed)
            proj_weighted = proj_half * half_weights_windowed[None, None, :]
            proj_abs2_weighted = proj_abs2 * half_weights_windowed[None, None, :]
            proj_recon = proj_half_flat[:, recon_window_indices].reshape(
                batch_size,
                bucket.bucket_rotation_count,
                n_recon_windowed,
            )
            proj_abs2_recon = proj_abs2_half_flat[:, recon_window_indices].reshape(
                batch_size,
                bucket.bucket_rotation_count,
                n_recon_windowed,
            )
            proj_for_noise = proj_recon
            proj_abs2_for_noise = proj_abs2_recon
        else:
            proj_half = proj_half_flat.reshape(batch_size, bucket.bucket_rotation_count, n_half)
            proj_abs2 = proj_abs2_half_flat.reshape(batch_size, bucket.bucket_rotation_count, n_half)
            proj_weighted = proj_half * half_weights[None, None, :]
            proj_abs2_weighted = proj_abs2 * half_weights[None, None, :]
            proj_for_noise = proj_half
            proj_abs2_for_noise = proj_abs2
        if use_float64_scoring:
            proj_weighted = proj_weighted.astype(jnp.complex128)
            proj_abs2_weighted = proj_abs2_weighted.astype(jnp.float64)
            proj_for_noise = proj_for_noise.astype(jnp.complex128)
            proj_abs2_for_noise = proj_abs2_for_noise.astype(jnp.float64)
        if return_profile:
            _block_until_ready(proj_weighted, proj_abs2_weighted)
        projection_time += time.time() - projection_t0

        score_t0 = time.time()
        shifted_score_split = shifted_score.reshape(batch_size, n_trans, -1)
        scores = score_local_bucket(
            shifted_score_split,
            ctf2_over_nv_score,
            proj_weighted,
            proj_abs2_weighted,
            jnp.asarray(bucket.local_rotation_log_prior),
            jnp.asarray(bucket.translation_log_prior),
            jnp.asarray(bucket.local_rotation_mask),
        )
        if return_profile:
            _block_until_ready(scores)
        score_time += time.time() - score_t0

        normalize_t0 = time.time()
        log_Z, probs, best_log_score, best_argmax, max_posterior = normalize_local_scores(scores)
        if return_profile:
            _block_until_ready(log_Z, probs, best_log_score, best_argmax, max_posterior)
        normalize_time += time.time() - normalize_t0

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
        significance_time += time.time() - significance_t0

        debug_score_dump_targets = _maybe_write_debug_score_dump(
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
            dump_dir=debug_score_dump_dir,
            pending_targets=debug_score_dump_targets,
        )

        mstep_t0 = time.time()
        shifted_recon_split = shifted_recon.reshape(batch_size, n_trans, -1)
        probs_sum_t = jnp.sum(probs, axis=-1)
        reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
        summed = compute_local_weighted_sums(reconstruction_probs, shifted_recon_split)
        ctf_probs = compute_local_ctf_sums(reconstruction_probs, ctf2_over_nv_recon)
        if return_profile:
            _block_until_ready(summed, ctf_probs, probs_sum_t, reconstruction_probs_sum_t)
        mstep_time += time.time() - mstep_t0

        reconstruction_rotation_mask_np = np.asarray(reconstruction_rotation_mask, dtype=bool)
        reconstruction_take_indices, reconstruction_pack_mask_np, reconstruction_counts_np, reconstruction_row_count = (
            _build_reconstruction_pack_indices(
                reconstruction_rotation_mask_np,
                np.asarray(bucket.local_rotation_mask, dtype=bool),
                rotation_block_size,
            )
        )
        reconstruction_take_indices_jnp = jnp.asarray(reconstruction_take_indices, dtype=jnp.int32)
        reconstruction_pack_mask_jnp = jnp.asarray(reconstruction_pack_mask_np)
        packed_rotations_np = np.take_along_axis(
            np.asarray(bucket.local_rotations, dtype=np.float32),
            reconstruction_take_indices[:, :, None, None],
            axis=1,
        )
        packed_rotation_ids_np = np.take_along_axis(
            np.asarray(bucket.local_rotation_ids, dtype=np.int32),
            reconstruction_take_indices,
            axis=1,
        )
        packed_summed = jnp.take_along_axis(summed, reconstruction_take_indices_jnp[:, :, None], axis=1)
        packed_summed = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_summed, 0.0)
        packed_ctf_probs = jnp.take_along_axis(ctf_probs, reconstruction_take_indices_jnp[:, :, None], axis=1)
        packed_ctf_probs = jnp.where(reconstruction_pack_mask_jnp[:, :, None], packed_ctf_probs, 0.0)

        if not disable_adjoint_y:
            adjoint_y_t0 = time.time()
            packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_rotations_np))
            if use_window:
                Ft_y = _adjoint_slice_volume_windowed(
                    flatten_bucket_rows(packed_summed),
                    recon_window_indices,
                    packed_flat_rotations,
                    Ft_y,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    False,
                )
            else:
                Ft_y = _adjoint_slice_volume_half(
                    flatten_bucket_rows(packed_summed),
                    packed_flat_rotations,
                    Ft_y,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    False,
                )
            if return_profile:
                _block_until_ready(Ft_y)
            adjoint_y_time += time.time() - adjoint_y_t0

        if not disable_adjoint_ctf:
            adjoint_ctf_t0 = time.time()
            packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_rotations_np))
            if use_window:
                Ft_ctf = _adjoint_slice_volume_windowed(
                    flatten_bucket_rows(packed_ctf_probs),
                    recon_window_indices,
                    packed_flat_rotations,
                    Ft_ctf,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    False,
                )
            else:
                Ft_ctf = _adjoint_slice_volume_half(
                    flatten_bucket_rows(packed_ctf_probs),
                    packed_flat_rotations,
                    Ft_ctf,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    False,
                )
            if return_profile:
                _block_until_ready(Ft_ctf)
            adjoint_ctf_time += time.time() - adjoint_ctf_t0

        if accumulate_noise:
            noise_t0 = time.time()
            batch_img_power = jnp.sum(jnp.abs(processed_score_half) ** 2, axis=0).astype(jnp.float32)
            batch_img_power_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            batch_img_power_shells = batch_img_power_shells.at[shell_indices_half].add(batch_img_power)
            noise_img_power += np.asarray(batch_img_power_shells, dtype=np.float64)
            noise_sumw += batch_size

            if half_spectrum_scoring:
                shifted_noise_split = shifted_noise.reshape(batch_size, n_trans, -1)
            else:
                shifted_noise_split = shifted_score_split
            summed_masked_noise = compute_local_weighted_sums(reconstruction_probs, shifted_noise_split)
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
            block_noise_shells, _, _ = _compute_noise_block(
                flatten_bucket_rows(packed_proj_for_noise),
                flatten_bucket_rows(packed_proj_abs2_for_noise),
                flatten_bucket_rows(packed_summed_masked_noise),
                flatten_bucket_rows(packed_ctf_probs),
                noise_variance_for_noise,
                shell_indices_noise,
                n_shells,
            )
            if return_profile:
                _block_until_ready(block_noise_shells)
            noise_wsum += np.asarray(block_noise_shells, dtype=np.float64)
            noise_time += time.time() - noise_t0

        postprocess_t0 = time.time()
        best_rot_idx = np.asarray(best_argmax // n_trans, dtype=np.int32)
        best_trans_idx = np.asarray(best_argmax % n_trans, dtype=np.int32)
        best_rotation_ids = np.take_along_axis(
            np.asarray(bucket.local_rotation_ids, dtype=np.int32),
            best_rot_idx[:, None],
            axis=1,
        ).reshape(-1)
        if np.any(best_rotation_ids < 0):
            raise RuntimeError("exact local engine selected padded local rotation")
        hard_assignment[bucket.image_indices] = (best_rotation_ids * n_trans + best_trans_idx).astype(np.int32)
        log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
        log_evidence_per_image[bucket.image_indices] = np.asarray(log_Z, dtype=np.float32) + log_score_offset.astype(np.float32)
        best_log_score_per_image[bucket.image_indices] = np.asarray(best_log_score, dtype=np.float32) + log_score_offset.astype(np.float32)
        max_posterior_per_image[bucket.image_indices] = np.asarray(max_posterior, dtype=np.float32)

        probs_sum_t_np = np.asarray(probs_sum_t, dtype=np.float64)
        n_significant_samples_np = np.asarray(n_significant_samples, dtype=np.int32)
        local_ids_np = np.asarray(bucket.local_rotation_ids, dtype=np.int32)
        local_mask_np = np.asarray(bucket.local_rotation_mask, dtype=bool)
        np.add.at(rotation_posterior_sums, local_ids_np[local_mask_np], probs_sum_t_np[local_mask_np])
        nonzero_mask = (probs_sum_t_np > 0.0) & local_mask_np
        chunk_nonzero_posterior_rows.append(int(np.count_nonzero(nonzero_mask)))
        chunk_significant_samples.append(int(np.sum(n_significant_samples_np, dtype=np.int64)))
        chunk_reconstruction_rows.append(int(reconstruction_row_count))
        total_significant_samples += int(np.sum(n_significant_samples_np, dtype=np.int64))
        total_reconstruction_rows += int(reconstruction_row_count)
        if seen_global_rotations.size:
            seen_global_rotations[local_ids_np[local_mask_np]] = True
            seen_nonzero_global_rotations[local_ids_np[nonzero_mask]] = True
            seen_reconstruction_global_rotations[packed_rotation_ids_np[reconstruction_pack_mask_np]] = True
        postprocess_time += time.time() - postprocess_t0

        host_stats_t0 = time.time()
        logger.debug(
            "Exact local bucket: %d images, bucket_rot=%d, total_local_rot=%d",
            batch_size,
            int(bucket.bucket_rotation_count),
            int(np.sum(bucket.actual_rotation_counts)),
        )
        host_stats_time += time.time() - host_stats_t0

    if return_profile:
        _block_until_ready(Ft_y, Ft_ctf)

    relion_stats = RelionStats(
        log_evidence_per_image=jnp.asarray(log_evidence_per_image),
        best_log_score_per_image=jnp.asarray(best_log_score_per_image),
        max_posterior_per_image=jnp.asarray(max_posterior_per_image),
        rotation_posterior_sums=jnp.asarray(rotation_posterior_sums, dtype=jnp.float32),
    )
    noise_stats = None
    if accumulate_noise:
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.asarray(noise_wsum, dtype=jnp.float32),
            wsum_img_power=jnp.asarray(noise_img_power, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(noise_sumw),
        )

    if debug_score_dump_dir is not None and debug_score_dump_targets:
        logger.warning(
            "Requested local score dump indices were not observed in this dataset view: %s",
            sorted(debug_score_dump_targets),
        )

    if not return_profile:
        if accumulate_noise:
            return Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats
        return Ft_y, Ft_ctf, hard_assignment, relion_stats

    _block_until_ready(Ft_y, Ft_ctf)
    total_wall_time = time.time() - overall_t0
    profile_summary = {
        "local_engine_kind": np.array("exact_v1"),
        "bucket_build_time_s": np.float64(bucket_build_time),
        "batch_fetch_time_s": np.float64(batch_fetch_time),
        "preprocess_time_s": np.float64(preprocess_time),
        "projection_time_s": np.float64(projection_time),
        "local_score_s": np.float64(score_time),
        "local_normalize_s": np.float64(normalize_time),
        "local_significance_s": np.float64(significance_time),
        "local_mstep_s": np.float64(mstep_time),
        "local_backproject_y_s": np.float64(adjoint_y_time),
        "local_backproject_ctf_s": np.float64(adjoint_ctf_time),
        "local_noise_s": np.float64(noise_time),
        "local_postprocess_s": np.float64(postprocess_time),
        "local_host_stats_s": np.float64(host_stats_time),
        "em_time_s": np.float64(total_wall_time),
        "accounted_em_time_s": np.float64(
            bucket_build_time
            + batch_fetch_time
            + preprocess_time
            + projection_time
            + score_time
            + normalize_time
            + significance_time
            + mstep_time
            + adjoint_y_time
            + adjoint_ctf_time
            + noise_time
            + postprocess_time
            + host_stats_time
        ),
        "unattributed_em_time_s": np.float64(
            max(
                total_wall_time
                - (
                    bucket_build_time
                    + batch_fetch_time
                    + preprocess_time
                    + projection_time
                    + score_time
                    + normalize_time
                    + significance_time
                    + mstep_time
                    + adjoint_y_time
                    + adjoint_ctf_time
                    + noise_time
                    + postprocess_time
                    + host_stats_time
                ),
                0.0,
            )
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
        "local_pad_fraction": np.float64(
            0.0 if total_padded_rotations == 0 else 1.0 - total_local_rotations / total_padded_rotations
        ),
        "n_windowed": np.int32(n_windowed),
    }
    if accumulate_noise:
        return Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats, profile_summary
    return Ft_y, Ft_ctf, hard_assignment, relion_stats, profile_summary
