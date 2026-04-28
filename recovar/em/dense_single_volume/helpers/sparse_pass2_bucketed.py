"""Bucketed batched implementation of sparse pass-2 oversampling.

Replaces the per-image Python loop in
``compute_pass2_stats_sparse`` with a shape-bucketed batched evaluation.

Background
----------
RELION's adaptive pass-2 evaluates the oversampled children of each
image's significant coarse (rotation, translation) pairs.  Because the
number of significant coarse rotations differs per image, a naive per-
image evaluation produces a different XLA shape for every call, leading
to catastrophic JIT recompilation when there are thousands of images.

This helper groups images by ``oversampled_rots.shape[0]`` (quantized
to a small set of bucket sizes via
``local_layout._exact_bucket_rotation_size``), pads each image's
oversampled rotations / log-priors / candidate masks to the bucket size,
and evaluates each bucket as a single GPU call with per-image
projections (analogous to the local-search exact engine).

The numerical contract matches the per-image reference path exactly:
identity-padded rotations are masked out via ``-inf`` log-prior and
``False`` (rot, trans) mask, so they contribute zero posterior mass and
do not perturb the M-step accumulators.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.em_primitives import (
    _adjoint_slice_volume_half,
    _adjoint_slice_volume_windowed,
    _compute_noise_block,
    _compute_projections_block,
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
    make_shell_indices_half,
)
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    integer_pre_shifts_or_none,
)
from recovar.em.dense_single_volume.helpers.preprocessing import process_half_image
from recovar.em.dense_single_volume.helpers.types import NoiseStats, RelionStats
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_weighted_sums,
    enforce_relion_half_volume_x0_hermitian,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size

logger = logging.getLogger(__name__)

_native_mstep_dump_counter = 0


def _maybe_dump_native_half_mstep(
    Ft_y_total,
    Ft_ctf_total,
    *,
    current_size,
    n_images,
    recon_volume_shape,
    stage,
):
    dump_dir = os.environ.get("RECOVAR_SPARSE_PASS2_NATIVE_DUMP_DIR")
    if not dump_dir:
        return

    global _native_mstep_dump_counter
    dump_idx = _native_mstep_dump_counter
    _native_mstep_dump_counter += 1

    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / f"native_half_mstep_{dump_idx:03d}_{stage}_n{int(n_images):04d}_cs{int(current_size):03d}.npz",
        Ft_y=np.asarray(Ft_y_total),
        Ft_ctf=np.asarray(Ft_ctf_total),
        current_size=np.int32(current_size),
        n_images=np.int32(n_images),
        recon_volume_shape=np.asarray(recon_volume_shape, dtype=np.int32),
        stage=np.asarray(stage),
    )


# ---------------------------------------------------------------------------
# Per-image hypothesis preparation
# ---------------------------------------------------------------------------


def _prepare_per_image_pass2_inputs(
    significant_sample_indices,
    n_coarse_rot,
    n_coarse_trans,
    nside_level,
    oversampling_order,
    n_fine_trans,
    fine_translation_parent,
    rotation_log_prior,
    random_perturbation,
):
    """Compute per-image oversampled rotations / parent maps / candidate masks.

    Mirrors the per-image branch in the legacy reference implementation in
    :func:`compute_pass2_stats_sparse_perimage_reference` exactly so the
    batched path is a strict per-image equivalent.
    """
    from recovar.em.sampling import get_oversampled_rotation_grid_from_samples

    n_images = len(significant_sample_indices)
    per_image_oversampled_rots = []
    per_image_parent_map = []
    per_image_oversampled_rot_indices = []
    per_image_unique_rot = []
    per_image_log_prior = []
    per_image_candidate_mask = []

    if rotation_log_prior is not None:
        rotation_log_prior_np = np.asarray(rotation_log_prior, dtype=np.float32)
    else:
        rotation_log_prior_np = None

    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if sig_samples is None:
            unique_rot = np.arange(n_coarse_rot, dtype=np.int32)
            use_full_candidate_mask = True
            coarse_rot = unique_rot
            coarse_trans = None
        else:
            sig_samples = np.asarray(sig_samples, dtype=np.int32).reshape(-1)
            if sig_samples.size == 0:
                raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")
            coarse_rot = sig_samples // n_coarse_trans
            coarse_trans = sig_samples % n_coarse_trans
            unique_rot = np.unique(coarse_rot)
            use_full_candidate_mask = False

        if unique_rot.size == 0:
            raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")

        oversampled_rots, parent_map, oversampled_rot_indices = get_oversampled_rotation_grid_from_samples(
            unique_rot,
            nside_level,
            oversampling_order=oversampling_order,
            random_perturbation=random_perturbation,
            return_rotation_indices=True,
        )
        oversampled_rots = np.asarray(oversampled_rots, dtype=np.float32)
        parent_map = np.asarray(parent_map, dtype=np.int32)
        oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int64)

        if rotation_log_prior_np is not None:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map]
        else:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=np.float32)

        if use_full_candidate_mask:
            candidate_mask = np.ones((oversampled_rots.shape[0], n_fine_trans), dtype=bool)
        else:
            sig_trans_by_rot = {
                int(rot_idx): set(coarse_trans[coarse_rot == rot_idx].tolist()) for rot_idx in unique_rot
            }
            candidate_mask = np.zeros((oversampled_rots.shape[0], n_fine_trans), dtype=bool)
            for parent_local_idx, coarse_rot_idx in enumerate(unique_rot):
                row_mask = parent_map == parent_local_idx
                valid_coarse_trans = sig_trans_by_rot[int(coarse_rot_idx)]
                col_mask = np.isin(fine_translation_parent, list(valid_coarse_trans))
                candidate_mask[row_mask, :] = col_mask[None, :]

        if not np.any(candidate_mask):
            raise ValueError(f"Image {image_idx} has no valid sparse pass-2 candidates after oversampling")

        per_image_oversampled_rots.append(oversampled_rots)
        per_image_parent_map.append(parent_map)
        per_image_oversampled_rot_indices.append(oversampled_rot_indices)
        per_image_unique_rot.append(unique_rot)
        per_image_log_prior.append(local_rotation_log_prior.astype(np.float32, copy=False))
        per_image_candidate_mask.append(candidate_mask)

    assert len(per_image_oversampled_rots) == n_images
    return {
        "oversampled_rots": per_image_oversampled_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
    }


# ---------------------------------------------------------------------------
# Bucket spec
# ---------------------------------------------------------------------------


def _bucket_pass2_inputs(
    per_image_inputs,
    n_fine_trans,
    rotation_block_size_for_quantization=5000,
    max_hypotheses_per_microbatch=2_000_000,
    max_images_per_microbatch=2048,
):
    """Group images into buckets that share a padded rotation count.

    Returns a list of dicts; each contains the padded per-image arrays
    needed to evaluate the bucket as one batched call.

    To avoid OOM when one bucket is very large
    (``bucket_size * n_images_in_bucket * n_fine_trans`` is the (B, R, T)
    score tensor footprint), we split each per-quantization-size group
    into chunks of at most ``max_hypotheses_per_microbatch /
    (bucket_size * n_fine_trans)`` images.  The bound ``2e6`` corresponds
    to ~32 MiB at float32 for a (B, R, T) tensor, well within budget for
    typical GPU memory.
    """
    n_images = len(per_image_inputs["oversampled_rots"])
    rotation_counts = np.array(
        [rots.shape[0] for rots in per_image_inputs["oversampled_rots"]],
        dtype=np.int64,
    )
    if n_images == 0:
        return []

    bucket_sizes = np.array(
        [_exact_bucket_rotation_size(int(count), rotation_block_size_for_quantization) for count in rotation_counts],
        dtype=np.int64,
    )

    # Group by bucket size, smaller buckets first
    processing_order = np.lexsort((rotation_counts, bucket_sizes)).astype(np.int64)
    unique_bucket_sizes = np.unique(bucket_sizes[processing_order])

    buckets = []
    for bucket_size in unique_bucket_sizes:
        bucket_size = int(bucket_size)
        bucket_image_indices = processing_order[bucket_sizes[processing_order] == bucket_size]
        # Chunk by max_hypotheses_per_microbatch and max_images_per_microbatch
        cap_by_hypotheses = max(
            1,
            int(max_hypotheses_per_microbatch) // max(1, bucket_size * int(n_fine_trans)),
        )
        max_per_chunk = max(1, min(int(max_images_per_microbatch), cap_by_hypotheses))
        for start in range(0, bucket_image_indices.shape[0], max_per_chunk):
            chunk = bucket_image_indices[start : start + max_per_chunk]
            buckets.append(
                {
                    "bucket_size": bucket_size,
                    "image_indices": np.asarray(chunk, dtype=np.int64),
                }
            )
    return buckets


def _build_bucket_arrays(
    bucket,
    per_image_inputs,
    n_fine_trans,
):
    """Stack/pad per-image arrays into batched bucket tensors."""
    bucket_size = int(bucket["bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    batch = int(image_indices.shape[0])

    # padded_rotations: identity-fill — projection of identity is harmless
    # because we mask via candidate_mask=False everywhere for padded rows.
    padded_rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (batch, bucket_size, 3, 3),
    ).copy()
    padded_log_prior = np.full((batch, bucket_size), -1e30, dtype=np.float32)
    padded_candidate_mask = np.zeros((batch, bucket_size, n_fine_trans), dtype=bool)
    padded_parent_map = np.full((batch, bucket_size), -1, dtype=np.int32)
    actual_counts = np.zeros(batch, dtype=np.int32)
    for row, image_idx in enumerate(image_indices.tolist()):
        rots = per_image_inputs["oversampled_rots"][image_idx]
        cnt = int(rots.shape[0])
        actual_counts[row] = cnt
        padded_rotations[row, :cnt] = rots
        padded_log_prior[row, :cnt] = per_image_inputs["log_prior"][image_idx]
        padded_candidate_mask[row, :cnt, :] = per_image_inputs["candidate_mask"][image_idx]
        padded_parent_map[row, :cnt] = per_image_inputs["parent_map"][image_idx]

    return {
        "image_indices": image_indices,
        "bucket_size": bucket_size,
        "actual_counts": actual_counts,
        "rotations": padded_rotations,
        "log_prior": padded_log_prior,
        "candidate_mask": padded_candidate_mask,
        "parent_map": padded_parent_map,
    }


# ---------------------------------------------------------------------------
# Scoring + normalization (per-bucket, supports (B, R, T) mask)
# ---------------------------------------------------------------------------


@jax.jit
def _score_pass2_bucket(
    shifted_score,  # (B, T, N) complex
    ctf2_over_nv_score,  # (B, N) real
    proj_weighted,  # (B, R, N) complex
    proj_abs2_weighted,  # (B, R, N) real
    rotation_log_prior,  # (B, R) real
    translation_log_prior,  # (B, T) real
    candidate_mask,  # (B, R, T) bool
):
    """Compute (B, R, T) scores; mask invalid (rot, trans) cells to -inf."""
    cross = -2.0 * jnp.einsum(
        "btn,brn->btr",
        jnp.conj(shifted_score),
        proj_weighted,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    cross = cross.swapaxes(1, 2)  # (B, R, T)
    norms = jnp.einsum(
        "bn,brn->br",
        ctf2_over_nv_score,
        proj_abs2_weighted,
        precision=jax.lax.Precision.HIGHEST,
    )  # (B, R)
    scores = -0.5 * (cross + norms[..., None])
    scores = scores + rotation_log_prior[:, :, None]
    scores = scores + translation_log_prior[:, None, :]
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return scores


@jax.jit
def _score_pass2_bucket_relion_gpu_diff2(
    shifted_corrected,  # (B, T, N) complex, image / (CTF * scale)
    corr_img_score,  # (B, N) real, Minvsigma2 * CTF^2 * scale^2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    rotation_log_prior,  # (B, R) real
    translation_log_prior,  # (B, T) real
    candidate_mask,  # (B, R, T) bool
):
    """RELION GPU-style direct ``diff2`` scoring for pass-2 diagnostics.

    RELION's CUDA fine-search kernel first corrects the image by dividing by
    CTF and scale, then accumulates ``|Fref - Fimg_corrected_shift|^2 *
    corr_img`` where ``corr_img = Minvsigma2 * CTF^2 * scale^2``.  This is
    algebraically equivalent to the CPU ``Frefctf - Fimg`` expression but has
    different float32 rounding.  We remove the image-only constant so the
    existing relative-score/log-evidence contract is unchanged.
    """

    weights = corr_img_score * half_weights[None, :]

    def score_one_image(shifted_tn, weights_n, proj_rn, rot_prior_r, trans_prior_t, mask_rt):
        image_constant_t = 0.5 * jnp.sum(
            (shifted_tn.real * shifted_tn.real + shifted_tn.imag * shifted_tn.imag) * weights_n[None, :],
            axis=-1,
        )

        def score_one_rotation(carry, proj_n):
            del carry
            diff_tn = proj_n[None, :] - shifted_tn
            diff2_t = 0.5 * jnp.sum(
                (diff_tn.real * diff_tn.real + diff_tn.imag * diff_tn.imag) * weights_n[None, :],
                axis=-1,
            )
            return None, -diff2_t + image_constant_t

        _, scores_rt = jax.lax.scan(score_one_rotation, None, proj_rn)
        scores_rt = scores_rt + rot_prior_r[:, None] + trans_prior_t[None, :]
        return jnp.where(mask_rt, scores_rt, -jnp.inf)

    return jax.vmap(score_one_image)(
        shifted_corrected,
        weights,
        proj_half,
        rotation_log_prior,
        translation_log_prior,
        candidate_mask,
    )


@jax.jit
def _normalize_pass2_bucket(scores):
    """Compute per-image normalization stats from (B, R, T) scores."""
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    log_shift = best_log_score[:, None, None]
    probs = jnp.exp((scores - log_shift).astype(jnp.float64))
    sum_exp = jnp.sum(probs.reshape(scores.shape[0], -1), axis=1)
    log_Z = best_log_score + jnp.log(sum_exp)
    probs = jnp.exp(scores - log_Z[:, None, None])
    best_argmax = jnp.argmax(flat, axis=1)
    max_posterior = jnp.exp(best_log_score - log_Z)
    return log_Z, probs, best_log_score, best_argmax, max_posterior


@jax.jit
def _normalize_pass2_bucket_with_log_z(scores, log_z):
    """Normalize sparse candidate scores with a precomputed full-grid log-Z."""
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    probs = jnp.exp(scores - log_z[:, None, None])
    best_argmax = jnp.argmax(flat, axis=1)
    max_posterior = jnp.exp(best_log_score - log_z)
    return log_z, probs, best_log_score, best_argmax, max_posterior


# ---------------------------------------------------------------------------
# Main bucketed driver
# ---------------------------------------------------------------------------


def _fetch_indexed_batch(experiment_dataset, image_indices):
    batch_iter = experiment_dataset.iter_batches(
        len(image_indices),
        indices=np.asarray(image_indices),
        by_image=False,
    )
    batch_data, _, _, ctf_params, _, _, indices = next(batch_iter)
    return jnp.asarray(batch_data), ctf_params, np.asarray(indices)


def _reorder_to_indices(image_indices_returned, requested_image_indices, *arrays):
    """Reorder per-image arrays so they match the order returned by the dataset."""
    if np.array_equal(image_indices_returned, requested_image_indices):
        return arrays
    position = {int(idx): pos for pos, idx in enumerate(np.asarray(requested_image_indices).tolist())}
    order = np.array([position[int(idx)] for idx in np.asarray(image_indices_returned).tolist()], dtype=np.int64)
    return tuple(arr[order] for arr in arrays)


def _parse_int_set_env(name):
    value = os.environ.get(name)
    if not value:
        return None
    out = set()
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def _maybe_dump_pass2_bucket(
    *,
    experiment_dataset,
    image_indices,
    per_image_inputs,
    current_size,
    n_fine_trans,
    fine_translations,
    scores,
    probs,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    shifted_score_split,
    ctf2_over_nv_score,
    proj_half,
    half_weights_used,
    window_indices,
    shifted_corrected_score_split=None,
):
    """Env-gated sparse pass-2 dump for RELION operand parity debugging."""
    dump_dir = os.environ.get("RECOVAR_PASS2_DUMP_DIR")
    if not dump_dir:
        return
    target_original_indices = _parse_int_set_env("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        target_original_indices = _parse_int_set_env("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return
    target_current_size = os.environ.get("RECOVAR_PASS2_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
    if original_indices_all is None:
        original_indices = local_indices
    else:
        original_indices = np.asarray(original_indices_all, dtype=np.int64)[local_indices]

    wanted_rows = [i for i, original_idx in enumerate(original_indices) if int(original_idx) in target_original_indices]
    if not wanted_rows:
        return

    os.makedirs(dump_dir, exist_ok=True)
    scores_np = np.asarray(scores, dtype=np.float64)
    probs_np = np.asarray(probs, dtype=np.float64)
    rot_prior_np = np.asarray(rotation_log_prior, dtype=np.float64)
    trans_prior_np = np.asarray(translation_log_prior, dtype=np.float64)
    mask_np = np.asarray(candidate_mask, dtype=bool)
    shifted_score_np = np.asarray(shifted_score_split)
    ctf2_np = np.asarray(ctf2_over_nv_score, dtype=np.float64)
    proj_np = np.asarray(proj_half)
    shifted_corrected_np = (
        None if shifted_corrected_score_split is None else np.asarray(shifted_corrected_score_split)
    )

    for row in wanted_rows:
        image_idx = int(local_indices[row])
        original_idx = int(original_indices[row])
        cnt = int(per_image_inputs["oversampled_rots"][image_idx].shape[0])
        scores_row = scores_np[row, :cnt, :]
        pre_prior = scores_row - rot_prior_np[row, :cnt, None] - trans_prior_np[row, None, :]
        out_path = os.path.join(
            dump_dir,
            f"pass2_orig{original_idx:06d}_cs{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        np.savez_compressed(
            out_path,
            original_index=np.int64(original_idx),
            local_index=np.int64(image_idx),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            n_fine_trans=np.int64(n_fine_trans),
            fine_translations=np.asarray(fine_translations, dtype=np.float32),
            rotations=np.asarray(per_image_inputs["oversampled_rots"][image_idx], dtype=np.float32),
            oversampled_rot_indices=np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64),
            parent_map=np.asarray(per_image_inputs["parent_map"][image_idx], dtype=np.int32),
            candidate_mask=mask_np[row, :cnt, :],
            scores_with_prior=scores_row,
            scores_pre_prior=pre_prior,
            probs=probs_np[row, :cnt, :],
            rotation_log_prior=rot_prior_np[row, :cnt],
            translation_log_prior=trans_prior_np[row],
            shifted_score=shifted_score_np[row],
            shifted_corrected=(
                shifted_corrected_np[row] if shifted_corrected_np is not None else np.empty((0,), dtype=np.complex64)
            ),
            ctf2_over_nv_score=ctf2_np[row],
            proj_half=proj_np[row, :cnt, :],
            half_weights=np.asarray(half_weights_used, dtype=np.float64),
            window_indices=(
                np.asarray(window_indices, dtype=np.int32) if window_indices is not None else np.empty((0,), dtype=np.int32)
            ),
        )


def _prepare_bucket_io(
    experiment_dataset,
    batch,
    ctf_params,
    image_indices,
    noise_variance_half,
    fine_translations,
    config,
    n_trans,
    score_with_masked_images,
    half_spectrum_scoring,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    return_direct_scoring_io=False,
):
    """Run preprocessing for a batch of images (translations tiled, CTF/noise ratios).

    Mirrors the ``run_em``/``_preprocess_batch`` pipeline exactly so the
    bucketed sparse pass-2 path is bit-for-bit identical to calling
    ``run_em`` per image.
    """
    image_shape = config.image_shape
    batch_size = int(batch.shape[0])
    integer_pre_shifts = integer_pre_shifts_or_none(image_pre_shifts, image_indices, batch=batch)
    real_space_pre_shift_applied = integer_pre_shifts is not None
    if real_space_pre_shift_applied:
        batch = apply_relion_integer_pre_shifts(batch, integer_pre_shifts)

    ctf_half = config.compute_ctf_half(ctf_params)
    ctf2_over_nv_half = ctf_half**2 / noise_variance_half

    # Raw processed half-spectrum images (BEFORE any per-image correction).
    # The score path uses masked images iff ``score_with_masked_images`` is True,
    # while the reconstruction path always uses the unmasked (raw) images.
    processed_score_half_raw = process_half_image(experiment_dataset, batch, config, score_with_masked_images)
    if score_with_masked_images:
        processed_recon_half_raw = process_half_image(experiment_dataset, batch, config, False)
    else:
        processed_recon_half_raw = processed_score_half_raw

    # batch_norm: (batch_size, 1) computed from RAW processed-score images
    # (matches ``_preprocess_batch`` line 156: ``processed`` is the raw masked).
    # We then scale by ``batch_corr**2`` if image_corrections is present, to
    # mirror ``run_em`` line 1069.
    norm_half_weights = make_half_image_weights(image_shape)
    batch_norm = jnp.sum(
        (jnp.abs(processed_score_half_raw) ** 2 / noise_variance_half) * norm_half_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real

    score_weighted_half = processed_score_half_raw * ctf_half / noise_variance_half
    recon_weighted_half = processed_recon_half_raw * ctf_half / noise_variance_half
    direct_corrected_score_half = processed_score_half_raw

    if scale_corrections is not None:
        batch_scale = jnp.asarray(np.asarray(scale_corrections)[np.asarray(image_indices)])
    else:
        batch_scale = jnp.ones(batch_size, dtype=ctf_half.dtype)

    # Per-image image corrections (matches run_em lines 1062-1069).
    if image_corrections is not None:
        batch_corr = jnp.asarray(np.asarray(image_corrections)[np.asarray(image_indices)])
        # Note: corrections are applied to the per-translation-tiled arrays in
        # run_em, but multiplication by a per-image scalar commutes with the
        # tiling and shifting so we apply it before tiling for efficiency.
        score_weighted_half = score_weighted_half * batch_corr[:, None]
        recon_weighted_half = recon_weighted_half * batch_corr[:, None]
        if return_direct_scoring_io:
            direct_raw_corr = batch_corr / batch_scale
            direct_corrected_score_half = direct_corrected_score_half * direct_raw_corr[:, None]
        # batch_norm scales by corr^2 to match run_em line 1069
        batch_norm = batch_norm * (batch_corr**2)[:, None]

    # Per-image scale correction on CTF^2/noise (matches run_em lines 1077-1079).
    if scale_corrections is not None:
        ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
        if return_direct_scoring_io:
            direct_corrected_score_half = direct_corrected_score_half / batch_scale[:, None]

    if return_direct_scoring_io:
        ctf_safe = jnp.abs(ctf_half) > 1e-8
        direct_corrected_score_half = jnp.where(
            ctf_safe,
            direct_corrected_score_half / ctf_half,
            direct_corrected_score_half,
        )

    # Per-image pre-centering (matches run_em lines 1087-1098): phase shift
    # in Fourier space.  Applied AFTER per-image scalar corrections.
    if image_pre_shifts is not None and not real_space_pre_shift_applied:
        batch_shifts = jnp.asarray(np.asarray(image_pre_shifts)[np.asarray(image_indices)])
        lattice_half = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
            image_shape,
            voxel_size=1,
            scaled=True,
        )
        # phase_factors: (batch, N_half) complex
        phase_factors = jnp.exp(-2j * jnp.pi * (lattice_half @ batch_shifts.T)).T
        score_weighted_half = score_weighted_half * phase_factors
        recon_weighted_half = recon_weighted_half * phase_factors
        if return_direct_scoring_io:
            direct_corrected_score_half = direct_corrected_score_half * phase_factors

    # Tile translations: (batch_size * n_trans, 2)
    translations_tiled = jnp.repeat(fine_translations[None], batch_size, axis=0).reshape(batch_size * n_trans, -1)
    score_weighted_tiled = jnp.repeat(score_weighted_half[:, None, :], n_trans, axis=1).reshape(
        batch_size * n_trans, -1
    )
    recon_weighted_tiled = jnp.repeat(recon_weighted_half[:, None, :], n_trans, axis=1).reshape(
        batch_size * n_trans, -1
    )
    shifted_score_half = core.translate_images(
        score_weighted_tiled,
        translations_tiled,
        image_shape,
        half_image=True,
    )
    if score_with_masked_images:
        shifted_recon_half = core.translate_images(
            recon_weighted_tiled,
            translations_tiled,
            image_shape,
            half_image=True,
        )
    else:
        shifted_recon_half = shifted_score_half

    shifted_corrected_score_half = None
    if return_direct_scoring_io:
        direct_corrected_score_tiled = jnp.repeat(direct_corrected_score_half[:, None, :], n_trans, axis=1).reshape(
            batch_size * n_trans,
            -1,
        )
        shifted_corrected_score_half = core.translate_images(
            direct_corrected_score_tiled,
            translations_tiled,
            image_shape,
            half_image=True,
        )

    # Save with-DC arrays for M-step / noise (RELION excludes DC from likelihood,
    # includes it in reconstruction weights).
    shifted_score_half_with_dc = shifted_score_half
    ctf2_over_nv_half_with_dc = ctf2_over_nv_half

    if half_spectrum_scoring:
        dc_shell_idx = make_shell_indices_half(image_shape)
        dc_mask = dc_shell_idx == 0
        shifted_score_half = jnp.where(dc_mask[None, :], 0.0, shifted_score_half)
        ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

    if use_float64_scoring:
        shifted_score_half = shifted_score_half.astype(jnp.complex128)
        shifted_recon_half = shifted_recon_half.astype(jnp.complex128)
        shifted_score_half_with_dc = shifted_score_half_with_dc.astype(jnp.complex128)
        ctf2_over_nv_half = ctf2_over_nv_half.astype(jnp.float64)
        ctf2_over_nv_half_with_dc = ctf2_over_nv_half_with_dc.astype(jnp.float64)
        if return_direct_scoring_io:
            shifted_corrected_score_half = shifted_corrected_score_half.astype(jnp.complex128)
    else:
        shifted_score_half = shifted_score_half.astype(jnp.complex64)
        ctf2_over_nv_half = ctf2_over_nv_half.astype(jnp.float32)
        if return_direct_scoring_io:
            shifted_corrected_score_half = shifted_corrected_score_half.astype(jnp.complex64)

    return (
        shifted_score_half,
        shifted_recon_half,
        batch_norm,
        ctf2_over_nv_half,
        ctf2_over_nv_half_with_dc,
        shifted_score_half_with_dc,
        processed_score_half_raw,  # used by noise_img_power (RAW, no corrections)
        shifted_corrected_score_half,
    )


def compute_pass2_stats_sparse_bucketed(
    experiment_dataset,
    volume,
    mean_variance,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    *,
    oversampling_order,
    current_size,
    translation_step,
    rotation_log_prior,
    score_with_masked_images,
    return_stats,
    translation_log_prior,
    accumulate_noise,
    half_spectrum_scoring,
    projection_padding_factor,
    reconstruction_padding_factor,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    translation_prior_centers=None,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation,
    normalization_log_z=None,
    rotation_block_size_for_quantization=5000,
):
    """Bucketed batched implementation of sparse pass-2 oversampling.

    Returns the same tuple as ``compute_pass2_stats_sparse``.
    """
    from recovar.em.sampling import (
        get_oversampled_translation_grid,
        rotation_grid_size,
    )

    n_images = experiment_dataset.n_units
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = rotation_grid_size(nside_level)

    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    # Recon volume layout: match the legacy per-image reference path by
    # default, but allow native RELION-style half-volume accumulation as an
    # isolated diagnostic for M-step parity against BackProjector.
    use_native_half_volume_mstep = os.environ.get(
        "RECOVAR_RELION_SPARSE_PASS2_HALF_VOLUME",
        "",
    ).lower() in {"1", "true", "yes", "on"}
    if reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
    else:
        recon_volume_shape = volume_shape
    recon_accum_shape = (
        fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)
        if use_native_half_volume_mstep
        else recon_volume_shape
    )
    recon_volume_size = int(np.prod(recon_accum_shape))
    recon_accum_dtype = experiment_dataset.dtype

    # Projection volume + padding
    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            volume,
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=current_size,
        )
    else:
        mean_for_proj = volume
        proj_volume_shape = volume_shape

    # Fine translations and prior mapping
    translations_np = np.asarray(translations, dtype=np.float32)
    if translation_step is None:
        unique_vals = np.unique(translations_np)
        diffs = np.diff(np.sort(unique_vals))
        diffs = diffs[diffs > 1e-6]
        translation_step = float(diffs.min()) if diffs.size else 1.0
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        translations_np,
        translation_step,
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=np.float32)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_trans = fine_translations.shape[0]

    translation_prior_centers_np = None
    if translation_prior_centers is not None:
        translation_prior_centers_np = np.asarray(translation_prior_centers, dtype=np.float32)
        if translation_prior_centers_np.ndim == 1:
            if translation_prior_centers_np.shape != (translations_np.shape[1],):
                raise ValueError(
                    "translation_prior_centers must have shape "
                    f"({translations_np.shape[1]},), got {translation_prior_centers_np.shape}",
                )
        elif translation_prior_centers_np.ndim == 2:
            if translation_prior_centers_np.shape != (n_images, translations_np.shape[1]):
                raise ValueError(
                    "translation_prior_centers must have shape "
                    f"({n_images}, {translations_np.shape[1]}) when image-specific, got "
                    f"{translation_prior_centers_np.shape}",
                )
        else:
            raise ValueError(
                f"translation_prior_centers must be 1D or 2D, got {translation_prior_centers_np.ndim} dimensions",
            )

    # Translation prior in the fine grid
    if translation_log_prior is None:
        fine_translation_prior_2d = None
    else:
        translation_log_prior_np = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior_np.ndim == 1:
            fine_tp = translation_log_prior_np[fine_translation_parent]
            fine_translation_prior_2d = np.broadcast_to(fine_tp, (n_images, n_fine_trans)).astype(
                np.float32, copy=False
            )
        elif translation_log_prior_np.ndim == 2:
            fine_translation_prior_2d = translation_log_prior_np[:, fine_translation_parent].astype(
                np.float32, copy=False
            )
        else:
            raise ValueError(
                f"translation_log_prior must be 1D or 2D, got {translation_log_prior_np.ndim} dimensions",
            )

    # Per-image hypothesis prep
    per_image_inputs = _prepare_per_image_pass2_inputs(
        significant_sample_indices,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=random_perturbation,
    )

    local_rot_counts = [int(rots.shape[0]) for rots in per_image_inputs["oversampled_rots"]]
    valid_candidate_counts = [int(np.asarray(m).sum()) for m in per_image_inputs["candidate_mask"]]

    use_relion_direct_diff2_scoring = os.environ.get("RECOVAR_RELION_DIRECT_DIFF2_SCORING", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    use_relion_adjoint_inverse = os.environ.get(
        "RECOVAR_RELION_SPARSE_PASS2_ADJOINT_INVERSE",
        "",
    ).lower() in {"1", "true", "yes", "on"}

    # Bucket
    buckets = _bucket_pass2_inputs(
        per_image_inputs,
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=rotation_block_size_for_quantization,
        max_hypotheses_per_microbatch=100_000 if use_relion_direct_diff2_scoring else 2_000_000,
    )

    logger.info(
        "Sparse pass-2 bucketing: %d images -> %d buckets (sizes: %s)",
        n_images,
        len(buckets),
        [b["bucket_size"] for b in buckets],
    )
    if use_native_half_volume_mstep:
        logger.info("Sparse pass-2 M-step: using native half-volume backprojection diagnostic")
    if use_relion_adjoint_inverse:
        logger.info("Sparse pass-2 M-step: using transposed rotations for RELION A.inv() diagnostic")

    # Output accumulators (volume_size matches what original returned: full N**3)
    Ft_y_total = jnp.zeros(recon_volume_size, dtype=recon_accum_dtype)
    Ft_ctf_total = jnp.zeros(recon_volume_size, dtype=recon_accum_dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    best_rotations = np.empty((n_images, 3, 3), dtype=np.float32)
    best_rotation_indices = np.empty(n_images, dtype=np.int64)

    log_evidence = np.empty(n_images, dtype=np.float32) if return_stats else None
    best_log_score = np.empty(n_images, dtype=np.float32) if return_stats else None
    max_posterior = np.empty(n_images, dtype=np.float32) if return_stats else None
    rotation_posterior_sums = np.zeros(n_coarse_rot, dtype=np.float64) if return_stats else None

    noise_wsum_total = None
    noise_img_power_total = None
    noise_sumw_total = 0.0
    noise_sigma2_offset_total = 0.0
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        noise_wsum_total = np.zeros(n_shells, dtype=np.float64)
        noise_img_power_total = np.zeros(n_shells, dtype=np.float64)

    # Forward-model config & half/window precomputes
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
    )
    use_window = window_spec.use_window
    window_indices_np = window_spec.score_indices_np
    window_indices = window_spec.score_indices
    recon_window_indices = window_spec.recon_indices
    n_windowed = window_spec.n_score
    n_recon_windowed = window_spec.n_recon

    if half_spectrum_scoring:
        half_weights = jnp.ones(n_half, dtype=jnp.float32)
    else:
        half_weights = make_half_image_weights(image_shape)
    half_weights_windowed = half_weights if window_indices is None else half_weights[window_indices]
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        half_weights_windowed = half_weights if window_indices is None else half_weights[window_indices]

    noise_variance_half = fourier_transform_utils.full_image_to_half_image(
        noise_variance.reshape(1, -1), image_shape
    ).squeeze()

    if accumulate_noise:
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        if use_window:
            shell_indices_noise = shell_indices_half[recon_window_indices]
            noise_variance_for_noise = noise_variance_half[recon_window_indices]
        else:
            shell_indices_noise = shell_indices_half
            noise_variance_for_noise = noise_variance_half

    normalization_log_z_np = None
    if normalization_log_z is not None:
        normalization_log_z_np = np.asarray(normalization_log_z, dtype=np.float64)
        if normalization_log_z_np.shape != (n_images,):
            raise ValueError(
                "normalization_log_z must have shape "
                f"({n_images},), got {normalization_log_z_np.shape}",
            )

    overall_t0 = time.time()

    for bucket_meta in buckets:
        bucket_arrays = _build_bucket_arrays(
            bucket_meta,
            per_image_inputs,
            n_fine_trans,
        )
        image_indices = bucket_arrays["image_indices"]
        bucket_size = bucket_arrays["bucket_size"]
        batch = int(image_indices.shape[0])

        # Fetch images (the dataset may reorder; we reorder our padded arrays
        # to match.)
        batch_data, ctf_params, fetched_indices = _fetch_indexed_batch(experiment_dataset, image_indices)
        # Reorder bucket arrays to match fetched_indices
        if not np.array_equal(np.asarray(fetched_indices), image_indices):
            (
                rotations,
                log_prior,
                candidate_mask,
                parent_map_padded,
                actual_counts,
            ) = _reorder_to_indices(
                np.asarray(fetched_indices),
                image_indices,
                bucket_arrays["rotations"],
                bucket_arrays["log_prior"],
                bucket_arrays["candidate_mask"],
                bucket_arrays["parent_map"],
                bucket_arrays["actual_counts"],
            )
            image_indices = np.asarray(fetched_indices)
        else:
            rotations = bucket_arrays["rotations"]
            log_prior = bucket_arrays["log_prior"]
            candidate_mask = bucket_arrays["candidate_mask"]
            parent_map_padded = bucket_arrays["parent_map"]
            actual_counts = bucket_arrays["actual_counts"]

        translation_sqdist_ang = None
        if translation_prior_centers_np is not None:
            if translation_prior_centers_np.ndim == 1:
                centers = np.broadcast_to(
                    translation_prior_centers_np[None, :],
                    (batch, translation_prior_centers_np.shape[0]),
                )
            else:
                centers = translation_prior_centers_np[image_indices]
            voxel = float(experiment_dataset.voxel_size if experiment_dataset.voxel_size > 0 else 1.0)
            translation_sqdist_ang = np.sum(
                ((fine_translations[None, :, :] - centers[:, None, :]) * voxel) ** 2,
                axis=-1,
                dtype=np.float64,
            )

        # Translation prior for this bucket (per-image)
        if fine_translation_prior_2d is None:
            bucket_translation_prior = jnp.zeros((batch, n_fine_trans), dtype=jnp.float32)
        else:
            bucket_translation_prior = jnp.asarray(fine_translation_prior_2d[image_indices], dtype=jnp.float32)

        # Preprocess
        (
            shifted_score_half,
            shifted_recon_half,
            batch_norm,
            ctf2_over_nv_half,
            ctf2_over_nv_half_with_dc,
            shifted_score_half_with_dc,
            processed_score_half_raw,
            shifted_corrected_score_half,
        ) = _prepare_bucket_io(
            experiment_dataset,
            batch_data,
            ctf_params,
            image_indices,
            noise_variance_half,
            fine_translations,
            config,
            n_fine_trans,
            score_with_masked_images,
            half_spectrum_scoring,
            image_corrections,
            scale_corrections,
            image_pre_shifts,
            use_float64_scoring,
            return_direct_scoring_io=use_relion_direct_diff2_scoring,
        )

        # Window gather (if applicable)
        if use_window:
            shifted_score = shifted_score_half[:, window_indices]
            shifted_recon = shifted_recon_half[:, recon_window_indices]
            ctf2_over_nv_score = ctf2_over_nv_half[:, window_indices]
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc[:, recon_window_indices]
            shifted_noise = shifted_score_half_with_dc[:, recon_window_indices]
            if use_relion_direct_diff2_scoring:
                shifted_corrected_score = shifted_corrected_score_half[:, window_indices]
        else:
            shifted_score = shifted_score_half
            shifted_recon = shifted_recon_half
            ctf2_over_nv_score = ctf2_over_nv_half
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
            shifted_noise = shifted_score_half_with_dc
            if use_relion_direct_diff2_scoring:
                shifted_corrected_score = shifted_corrected_score_half

        # Project (B*R, 3, 3) -> (B*R, n_half) -> reshape (B, R, n_half)
        flat_rotations = flatten_bucket_rotations(jnp.asarray(rotations))
        flat_backproject_rotations = (
            jnp.swapaxes(flat_rotations, -1, -2)
            if use_relion_adjoint_inverse
            else flat_rotations
        )
        projection_kwargs = window_spec.projection_kwargs(return_abs2=False if use_window else None)
        proj_half_flat, proj_abs2_half_flat = _compute_projections_block(
            mean_for_proj,
            flat_rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            **projection_kwargs,
        )
        if use_window:
            proj_half = proj_half_flat[:, window_indices].reshape(batch, bucket_size, n_windowed)
            proj_abs2 = jnp.abs(proj_half) ** 2
            proj_weighted = proj_half * half_weights_windowed[None, None, :]
            proj_abs2_weighted = proj_abs2 * half_weights_windowed[None, None, :]
            proj_for_noise = proj_half_flat[:, recon_window_indices].reshape(batch, bucket_size, n_recon_windowed)
            proj_abs2_for_noise = jnp.abs(proj_for_noise) ** 2
        else:
            proj_half = proj_half_flat.reshape(batch, bucket_size, n_half)
            proj_abs2 = proj_abs2_half_flat.reshape(batch, bucket_size, n_half)
            proj_weighted = proj_half * half_weights[None, None, :]
            proj_abs2_weighted = proj_abs2 * half_weights[None, None, :]
            proj_for_noise = proj_half
            proj_abs2_for_noise = proj_abs2

        if use_float64_scoring:
            proj_weighted = proj_weighted.astype(jnp.complex128)
            proj_abs2_weighted = proj_abs2_weighted.astype(jnp.float64)
            proj_for_noise = proj_for_noise.astype(jnp.complex128)
            proj_abs2_for_noise = proj_abs2_for_noise.astype(jnp.float64)
        else:
            proj_weighted = proj_weighted.astype(jnp.complex64)
            proj_abs2_weighted = proj_abs2_weighted.astype(jnp.float32)

        # Score: (B, R, T)
        shifted_score_split = shifted_score.reshape(batch, n_fine_trans, -1)
        shifted_corrected_score_split = None
        if use_relion_direct_diff2_scoring:
            shifted_corrected_score_split = shifted_corrected_score.reshape(batch, n_fine_trans, -1)
            direct_half_weights = half_weights_windowed if use_window else half_weights
            scores = _score_pass2_bucket_relion_gpu_diff2(
                shifted_corrected_score_split,
                ctf2_over_nv_score,
                proj_half,
                direct_half_weights,
                jnp.asarray(log_prior),
                bucket_translation_prior,
                jnp.asarray(candidate_mask),
            )
        else:
            scores = _score_pass2_bucket(
                shifted_score_split,
                ctf2_over_nv_score,
                proj_weighted,
                proj_abs2_weighted,
                jnp.asarray(log_prior),
                bucket_translation_prior,
                jnp.asarray(candidate_mask),
            )

        if normalization_log_z_np is None:
            log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = _normalize_pass2_bucket(scores)
        else:
            bucket_log_z = jnp.asarray(normalization_log_z_np[image_indices], dtype=scores.real.dtype)
            log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = (
                _normalize_pass2_bucket_with_log_z(scores, bucket_log_z)
            )

        _maybe_dump_pass2_bucket(
            experiment_dataset=experiment_dataset,
            image_indices=image_indices,
            per_image_inputs=per_image_inputs,
            current_size=current_size,
            n_fine_trans=n_fine_trans,
            fine_translations=fine_translations,
            scores=scores,
            probs=probs,
            rotation_log_prior=jnp.asarray(log_prior),
            translation_log_prior=bucket_translation_prior,
            candidate_mask=jnp.asarray(candidate_mask),
            shifted_score_split=shifted_score_split,
            ctf2_over_nv_score=ctf2_over_nv_score,
            proj_half=proj_half,
            half_weights_used=half_weights_windowed if use_window else half_weights,
            window_indices=window_indices_np,
            shifted_corrected_score_split=shifted_corrected_score_split,
        )

        # M-step accumulation: posterior-weighted sums per (image, rot)
        shifted_recon_split = shifted_recon.reshape(batch, n_fine_trans, -1)
        summed = compute_local_weighted_sums(probs, shifted_recon_split)  # (B, R, N)
        ctf_probs = compute_local_ctf_sums(probs, ctf2_over_nv_recon)  # (B, R, N)

        # Backproject (use flat_rotations + flat summed/ctf_probs).
        # Padded rotations contribute zero because their probs == 0
        # (candidate_mask=False -> score=-inf -> exp(-inf)=0).
        if use_window:
            Ft_y_total = _adjoint_slice_volume_windowed(
                flatten_bucket_rows(summed),
                recon_window_indices,
                flat_backproject_rotations,
                Ft_y_total,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                use_native_half_volume_mstep,
                float(current_size // 2),
            )
            Ft_ctf_total = _adjoint_slice_volume_windowed(
                flatten_bucket_rows(ctf_probs),
                recon_window_indices,
                flat_backproject_rotations,
                Ft_ctf_total,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                use_native_half_volume_mstep,
                float(current_size // 2),
            )
        else:
            Ft_y_total = _adjoint_slice_volume_half(
                flatten_bucket_rows(summed),
                flat_backproject_rotations,
                Ft_y_total,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                use_native_half_volume_mstep,
            )
            Ft_ctf_total = _adjoint_slice_volume_half(
                flatten_bucket_rows(ctf_probs),
                flat_backproject_rotations,
                Ft_ctf_total,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                use_native_half_volume_mstep,
            )

        # Noise accumulation
        if accumulate_noise:
            if translation_sqdist_ang is not None:
                translation_posterior = np.asarray(jnp.sum(probs, axis=1), dtype=np.float64)
                noise_sigma2_offset_total += float(
                    np.sum(translation_posterior * translation_sqdist_ang, dtype=np.float64)
                )
            # NOTE: noise_img_power uses RAW (un-corrected) processed images
            # to match run_em line 1144 (which recomputes processed_masked
            # from raw batch_data).
            batch_img_power = jnp.sum(jnp.abs(processed_score_half_raw) ** 2, axis=0).astype(jnp.float32)
            batch_img_power_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            batch_img_power_shells = batch_img_power_shells.at[shell_indices_half].add(batch_img_power)
            noise_img_power_total += np.asarray(batch_img_power_shells, dtype=np.float64)
            noise_sumw_total += float(batch)

            if half_spectrum_scoring:
                shifted_noise_split = shifted_noise.reshape(batch, n_fine_trans, -1)
            else:
                shifted_noise_split = shifted_score_split
            summed_masked_noise = compute_local_weighted_sums(probs, shifted_noise_split)
            block_noise_shells, _, _ = _compute_noise_block(
                flatten_bucket_rows(proj_for_noise),
                flatten_bucket_rows(proj_abs2_for_noise),
                flatten_bucket_rows(summed_masked_noise),
                flatten_bucket_rows(ctf_probs),
                noise_variance_for_noise,
                shell_indices_noise,
                n_shells,
            )
            noise_wsum_total += np.asarray(block_noise_shells, dtype=np.float64)

        # Decode best assignment and write per-image stats
        best_argmax_np = np.asarray(best_argmax, dtype=np.int64)
        best_rot_idx = best_argmax_np // n_fine_trans
        best_trans_idx = best_argmax_np % n_fine_trans

        # Sanity check: padded rotations should never be chosen (probs == 0 there).
        actual_counts_arr = np.asarray(actual_counts, dtype=np.int64)
        if np.any(best_rot_idx >= actual_counts_arr):
            bad = np.flatnonzero(best_rot_idx >= actual_counts_arr)
            raise RuntimeError(
                f"Bucket pass-2: best rotation index points into padding for images {bad.tolist()} "
                f"(best_rot_idx={best_rot_idx[bad].tolist()}, actual_counts={actual_counts_arr[bad].tolist()})"
            )

        for row, image_idx in enumerate(image_indices.tolist()):
            r = int(best_rot_idx[row])
            t = int(best_trans_idx[row])
            hard_assignment[image_idx] = r * n_fine_trans + t
            best_rotations[image_idx] = per_image_inputs["oversampled_rots"][image_idx][r]
            best_rotation_indices[image_idx] = per_image_inputs["oversampled_rot_indices"][image_idx][r]

        if return_stats:
            log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
            log_Z_np = np.asarray(log_Z, dtype=np.float64)
            best_log_score_np = np.asarray(best_log_score_bucket, dtype=np.float64)
            max_posterior_np = np.asarray(max_posterior_bucket, dtype=np.float32)
            for row, image_idx in enumerate(image_indices.tolist()):
                log_evidence[image_idx] = float(log_Z_np[row] + log_score_offset[row])
                best_log_score[image_idx] = float(best_log_score_np[row] + log_score_offset[row])
                max_posterior[image_idx] = float(max_posterior_np[row])

            # rotation_posterior_sums: scatter per (image, rot) probability mass back
            # to the parent coarse rotation indices.
            probs_sum_t = np.asarray(jnp.sum(probs, axis=-1), dtype=np.float64)  # (B, R)
            for row, image_idx in enumerate(image_indices.tolist()):
                cnt = int(actual_counts[row])
                if cnt == 0:
                    continue
                unique_rot_image = per_image_inputs["unique_rot"][image_idx]
                parent_map_image = per_image_inputs["parent_map"][image_idx]
                # Map each oversampled rot back to its coarse-grid rotation index.
                coarse_rot_indices = unique_rot_image[parent_map_image]
                np.add.at(rotation_posterior_sums, coarse_rot_indices, probs_sum_t[row, :cnt])

    em_wall = time.time() - overall_t0
    logger.info(
        "Sparse pass-2 (bucketed): %d images, %d buckets, %.2fs E+M; "
        "median local rot=%d, mean local rot=%.1f, median valid candidates/image=%d",
        n_images,
        len(buckets),
        em_wall,
        int(np.median(local_rot_counts)) if local_rot_counts else 0,
        float(np.mean(local_rot_counts)) if local_rot_counts else 0.0,
        int(np.median(valid_candidate_counts)) if valid_candidate_counts else 0,
    )

    if use_native_half_volume_mstep:
        _maybe_dump_native_half_mstep(
            Ft_y_total,
            Ft_ctf_total,
            current_size=current_size,
            n_images=n_images,
            recon_volume_shape=recon_volume_shape,
            stage="pre_x0",
        )
        if os.environ.get("RECOVAR_RELION_SPARSE_PASS2_HALF_VOLUME_ENFORCE_X0", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            logger.info("Sparse pass-2 M-step: enforcing RELION half-volume x=0 Hermitian plane")
            Ft_y_total = enforce_relion_half_volume_x0_hermitian(Ft_y_total, recon_volume_shape)
            Ft_ctf_total = enforce_relion_half_volume_x0_hermitian(Ft_ctf_total, recon_volume_shape)
        _maybe_dump_native_half_mstep(
            Ft_y_total,
            Ft_ctf_total,
            current_size=current_size,
            n_images=n_images,
            recon_volume_shape=recon_volume_shape,
            stage="post_x0",
        )
        # Keep the public return contract unchanged while testing whether the
        # RELION-style folded accumulation itself closes the Ft_ctf/Ft_y gap.
        Ft_y_total = fourier_transform_utils.half_volume_to_full_volume(
            Ft_y_total,
            recon_volume_shape,
        ).reshape(-1)
        Ft_ctf_total = fourier_transform_utils.half_volume_to_full_volume(
            Ft_ctf_total,
            recon_volume_shape,
        ).reshape(-1)

    best_translations = fine_translations[hard_assignment % n_fine_trans]

    merged_noise_stats = None
    if accumulate_noise:
        merged_noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.asarray(noise_wsum_total, dtype=jnp.float32),
            wsum_img_power=jnp.asarray(noise_img_power_total, dtype=jnp.float32),
            wsum_sigma2_offset=float(noise_sigma2_offset_total),
            sumw=float(noise_sumw_total),
        )

    if return_stats:
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.asarray(log_evidence),
            best_log_score_per_image=jnp.asarray(best_log_score),
            max_posterior_per_image=jnp.asarray(max_posterior),
            rotation_posterior_sums=jnp.asarray(rotation_posterior_sums, dtype=jnp.float32),
        )
        result = (
            Ft_y_total,
            Ft_ctf_total,
            hard_assignment,
            best_rotations,
            best_translations,
            best_rotation_indices,
            relion_stats,
        )
        if accumulate_noise:
            result = result + (merged_noise_stats,)
        return result

    result = (
        Ft_y_total,
        Ft_ctf_total,
        hard_assignment,
        best_rotations,
        best_translations,
        best_rotation_indices,
    )
    if accumulate_noise:
        result = result + (merged_noise_stats,)
    return result
