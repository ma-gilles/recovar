"""Normalization and group-scale capture at the sparse M-step boundary.

The score-dump owner supplies target selection and the shared capture directory.
This module records reconstruction-window operands and chunked Wavg intermediates;
production reductions and scheduling remain in ``sparse_pass2_bucketed``.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import bpref_diagnostics
from recovar.em.dense_single_volume.helpers.batch_fetch import original_image_indices
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag
from recovar.em.dense_single_volume.helpers.pass2_diagnostics import (
    _PASS2_DUMP_DIR_ENV,
    _pass2_dump_target_rows,
)


def _maybe_dump_norm_residual_inputs(
    *,
    experiment_dataset,
    image_indices,
    current_size,
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs,
    ctf2_over_nv_recon,
    posterior_probs,
    rotations_for_noise,
    noise_variance_for_noise,
    block_norm_residual,
    processed_score_half_for_noise,
    shell_indices_half,
    support_mass,
    relion_norm_high_shell,
    weighted_img_per_image,
    relion_score_translation_angles,
    recon_window_indices,
    score_window_indices,
    image_shape,
    bucket_scale_for_stats,
    scale_correction_pixel_mask,
    scale_shell_indices,
    bucket_group_ids,
    relion_wavg_atomic_diff2_rectangle=None,
    relion_wavg_atomic_rectangle_shell_indices=None,
):
    """Capture norm and group-scale AA inputs before any global reduction.

    This diagnostic is deliberately separate from the ordinary pass-2 dump:
    that dump records score-window projections, while the normalization update
    consumes the reconstruction/noise window and its squared projections.
    """

    if not parse_env_flag("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS", default=False):
        return 0
    dump_dir = os.environ.get(_PASS2_DUMP_DIR_ENV)
    if not dump_dir:
        raise ValueError("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS requires RECOVAR_PASS2_DUMP_DIR")
    target_iteration = os.environ.get("RECOVAR_PASS2_DUMP_ITERATION")
    context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
    if target_iteration and context_iteration != int(target_iteration):
        return 0
    target_rows = _pass2_dump_target_rows(
        experiment_dataset=experiment_dataset,
        image_indices=image_indices,
        current_size=current_size,
    )
    if target_rows.size == 0:
        return 0

    selected = jnp.asarray(target_rows, dtype=jnp.int32)
    raw_translated_recon = None
    raw_translated_wavg = None
    if relion_score_translation_angles is not None:
        from recovar import cuda_backproject

        recon_indices_jax = jnp.asarray(recon_window_indices, dtype=jnp.int32)
        translation_angles_jax = jnp.asarray(
            relion_score_translation_angles,
            dtype=jnp.float32,
        )
        raw_translated_recon = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(processed_score_half_for_noise)[selected][:, recon_indices_jax],
            translation_angles_jax,
            recon_indices_jax,
            image_shape,
        ).reshape(target_rows.size, translation_angles_jax.shape[0], -1)
        score_indices_jax = jnp.asarray(score_window_indices, dtype=jnp.int32)
        raw_translated_wavg = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(processed_score_half_for_noise)[selected][:, score_indices_jax],
            translation_angles_jax,
            score_indices_jax,
            image_shape,
        ).reshape(target_rows.size, translation_angles_jax.shape[0], -1)
    selected_proj_abs2 = jnp.asarray(proj_abs2_for_noise)[selected]
    selected_ctf_probs = jnp.asarray(ctf_probs)[selected]
    selected_posterior_probs = jnp.asarray(posterior_probs)[selected]
    selected_ctf2_over_nv = jnp.asarray(ctf2_over_nv_recon)[selected]
    selected_scale = jnp.asarray(bucket_scale_for_stats)[selected]
    selected_rotations = jnp.asarray(rotations_for_noise)[selected]
    scale_mask = jnp.asarray(scale_correction_pixel_mask, dtype=bool).reshape(-1)
    selected_noise = jnp.asarray(noise_variance_for_noise)
    norm_ctf_has_mass = selected_ctf_probs != 0.0
    norm_ctf_probs_raw = jnp.where(
        norm_ctf_has_mass,
        selected_ctf_probs * selected_noise[None, None, :],
        0.0,
    )
    norm_a2_terms = jnp.where(
        norm_ctf_has_mass,
        selected_proj_abs2 * norm_ctf_probs_raw,
        0.0,
    )
    selected_summed = jnp.asarray(summed_masked_noise)[selected]
    norm_cross_terms = jnp.where(
        selected_summed != 0.0,
        jnp.asarray(proj_for_noise)[selected] * jnp.conj(selected_summed),
        0.0,
    )
    norm_xa_terms = selected_noise[None, None, :] * norm_cross_terms.real
    norm_a2_per_image = jnp.sum(norm_a2_terms, axis=(1, 2)).astype(jnp.float32)
    norm_xa_per_image = jnp.sum(norm_xa_terms, axis=(1, 2)).astype(jnp.float32)
    ctf_has_mass = (selected_ctf_probs != 0.0) & scale_mask[None, None, :]
    ctf_probs_raw = jnp.where(
        ctf_has_mass,
        selected_ctf_probs * selected_noise[None, None, :],
        0.0,
    )
    aa_terms_before_scale = jnp.where(
        ctf_has_mass,
        selected_proj_abs2 * ctf_probs_raw,
        0.0,
    )
    safe_scale = jnp.maximum(
        selected_scale.astype(selected_proj_abs2.real.dtype),
        1e-30,
    )
    aa_terms = aa_terms_before_scale / (safe_scale[:, None, None] ** 2)
    aa_per_pixel = jnp.sum(aa_terms, axis=1)
    aa_per_image = jnp.sum(aa_terms, axis=(1, 2)).astype(jnp.float32)

    staged = jax.block_until_ready(
        (
            jnp.asarray(proj_for_noise)[selected],
            selected_proj_abs2,
            jnp.asarray(summed_masked_noise)[selected],
            selected_ctf_probs,
            selected_ctf2_over_nv,
            selected_posterior_probs,
            selected_rotations,
            jnp.asarray(block_norm_residual)[selected],
            jnp.asarray(processed_score_half_for_noise)[selected],
            jnp.asarray(support_mass)[selected],
            jnp.asarray(weighted_img_per_image)[selected],
            selected_scale,
            ctf_has_mass,
            ctf_probs_raw,
            aa_terms_before_scale,
            aa_terms,
            aa_per_pixel,
            aa_per_image,
            norm_ctf_has_mass,
            norm_ctf_probs_raw,
            norm_a2_terms,
            norm_cross_terms,
            norm_xa_terms,
            norm_a2_per_image,
            norm_xa_per_image,
            (
                jnp.empty((target_rows.size, 0, 0), dtype=jnp.complex64)
                if raw_translated_recon is None
                else raw_translated_recon
            ),
            (
                jnp.empty((target_rows.size, 0, 0), dtype=jnp.complex64)
                if raw_translated_wavg is None
                else raw_translated_wavg
            ),
        )
    )
    (
        proj_np,
        proj_abs2_np,
        summed_np,
        ctf_probs_np,
        ctf2_over_nv_np,
        posterior_probs_np,
        rotations_np,
        residual_np,
        processed_image_np,
        support_mass_np,
        weighted_image_power_np,
        scale_np,
        ctf_has_mass_np,
        ctf_probs_raw_np,
        aa_terms_before_scale_np,
        aa_terms_np,
        aa_per_pixel_np,
        aa_per_image_np,
        norm_ctf_has_mass_np,
        norm_ctf_probs_raw_np,
        norm_a2_terms_np,
        norm_cross_terms_np,
        norm_xa_terms_np,
        norm_a2_per_image_np,
        norm_xa_per_image_np,
        raw_translated_recon_np,
        raw_translated_wavg_np,
    ) = (np.asarray(value) for value in staged)
    noise_np = np.asarray(jax.block_until_ready(noise_variance_for_noise))
    shell_indices_np = np.asarray(jax.block_until_ready(shell_indices_half), dtype=np.int32)
    scale_shell_indices_np = np.asarray(
        jax.block_until_ready(scale_shell_indices),
        dtype=np.int32,
    )
    scale_mask_np = np.asarray(jax.block_until_ready(scale_mask), dtype=bool)
    relion_high_shell_np = (
        np.empty((0,), dtype=np.float32)
        if relion_norm_high_shell is None
        else np.asarray(jax.block_until_ready(relion_norm_high_shell), dtype=np.float32)
    )
    translation_angles_np = (
        np.empty((0, 2), dtype=np.float32)
        if relion_score_translation_angles is None
        else np.asarray(
            jax.block_until_ready(relion_score_translation_angles),
            dtype=np.float32,
        )
    )
    local_indices = np.asarray(image_indices, dtype=np.int64)
    group_ids_np = (
        np.zeros(local_indices.shape, dtype=np.int64)
        if bucket_group_ids is None
        else np.asarray(bucket_group_ids, dtype=np.int64)
    )
    atomic_rectangle_np = (
        None
        if relion_wavg_atomic_diff2_rectangle is None
        else np.asarray(relion_wavg_atomic_diff2_rectangle, dtype=np.float32)
    )
    atomic_rectangle_shells_np = (
        None
        if relion_wavg_atomic_rectangle_shell_indices is None
        else np.asarray(
            relion_wavg_atomic_rectangle_shell_indices,
            dtype=np.int32,
        ).reshape(-1)
    )
    if (atomic_rectangle_np is None) != (atomic_rectangle_shells_np is None):
        raise ValueError("ordinary Wavg rectangle values and shell labels must be supplied together")
    if atomic_rectangle_np is not None and atomic_rectangle_np.shape != (
        local_indices.size,
        atomic_rectangle_shells_np.size,
    ):
        raise ValueError("ordinary Wavg rectangle diff2 topology changed")
    original_indices = original_image_indices(experiment_dataset, local_indices)
    os.makedirs(dump_dir, exist_ok=True)
    context_half = int(bpref_diagnostics._bpref_contribution_context["half"])
    size_label = -1 if current_size is None else int(current_size)
    for selected_row, bucket_row in enumerate(target_rows.tolist()):
        original_index = int(original_indices[bucket_row])
        out_path = os.path.join(
            dump_dir,
            f"norm_residual_orig{original_index:06d}_half{context_half}_cs{size_label:03d}.npz",
        )
        aa_shells = np.zeros(int(np.max(scale_shell_indices_np, initial=-1)) + 1, dtype=np.float64)
        valid_scale_shell = (scale_shell_indices_np >= 0) & scale_mask_np & (scale_shell_indices_np < aa_shells.size)
        np.add.at(
            aa_shells,
            scale_shell_indices_np[valid_scale_shell],
            aa_per_pixel_np[selected_row, valid_scale_shell].astype(np.float64),
        )
        payload = dict(
            schema=np.asarray("recovar-k1-norm-residual-inputs-v3"),
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
            original_index=np.int64(original_index),
            local_index=np.int64(local_indices[bucket_row]),
            bucket_row=np.int64(bucket_row),
            current_size=np.int64(size_label),
            proj_for_noise=proj_np[selected_row],
            proj_abs2_for_noise=proj_abs2_np[selected_row],
            summed_masked_noise=summed_np[selected_row],
            ctf_probs=ctf_probs_np[selected_row],
            ctf2_over_nv_recon=ctf2_over_nv_np[selected_row],
            posterior_probs=posterior_probs_np[selected_row],
            rotations_for_noise=rotations_np[selected_row],
            relion_score_translation_angles=translation_angles_np,
            noise_variance_for_noise=noise_np,
            block_norm_residual=np.asarray(residual_np[selected_row]),
            processed_score_half_for_noise=processed_image_np[selected_row],
            shell_indices_half=shell_indices_np,
            support_mass=np.asarray(support_mass_np[selected_row]),
            relion_norm_high_shell=(
                relion_high_shell_np if relion_high_shell_np.size == 0 else np.asarray(relion_high_shell_np[bucket_row])
            ),
            weighted_img_per_image=np.asarray(weighted_image_power_np[selected_row]),
            group_id=np.int64(group_ids_np[bucket_row]),
            scale_for_stats=np.asarray(scale_np[selected_row]),
            scale_correction_pixel_mask=scale_mask_np,
            scale_shell_indices=scale_shell_indices_np,
            scale_ctf_has_mass=ctf_has_mass_np[selected_row],
            scale_ctf_probs_raw=ctf_probs_raw_np[selected_row],
            scale_aa_terms_before_scale=aa_terms_before_scale_np[selected_row],
            scale_aa_terms=aa_terms_np[selected_row],
            scale_aa_per_pixel=aa_per_pixel_np[selected_row],
            scale_aa_per_shell=aa_shells,
            scale_aa_per_image=np.asarray(aa_per_image_np[selected_row]),
            norm_ctf_has_mass=norm_ctf_has_mass_np[selected_row],
            norm_ctf_probs_raw=norm_ctf_probs_raw_np[selected_row],
            norm_a2_terms=norm_a2_terms_np[selected_row],
            norm_cross_terms=norm_cross_terms_np[selected_row],
            norm_xa_terms=norm_xa_terms_np[selected_row],
            norm_a2_per_image=np.asarray(norm_a2_per_image_np[selected_row]),
            norm_xa_per_image=np.asarray(norm_xa_per_image_np[selected_row]),
            raw_translated_recon=raw_translated_recon_np[selected_row],
            raw_translated_wavg=raw_translated_wavg_np[selected_row],
            recon_window_indices=np.asarray(recon_window_indices, dtype=np.int32),
            wavg_window_indices=np.asarray(score_window_indices, dtype=np.int32),
        )
        if atomic_rectangle_np is not None:
            rectangle_pixels = atomic_rectangle_np[bucket_row]
            valid_rectangle = atomic_rectangle_shells_np >= 0
            payload.update(
                wavg_diff2_atomic_rectangle_per_pixel=rectangle_pixels,
                wavg_diff2_atomic_rectangle_shell_indices=atomic_rectangle_shells_np,
                wavg_diff2_atomic_rectangle_per_image=np.float64(
                    np.sum(rectangle_pixels[valid_rectangle], dtype=np.float64)
                ),
            )
        np.savez_compressed(out_path, **payload)
    return int(target_rows.size)


def _write_chunked_scale_aa_dump(
    *,
    dump_dir,
    experiment_dataset,
    image_indices,
    target_rows,
    current_size,
    bucket_group_ids,
    bucket_scale_for_stats,
    scale_correction_pixel_mask,
    scale_shell_indices,
    chunk_ranges,
    posterior_mass_chunks,
    proj_abs2_sum_chunks,
    ctf_probs_raw_sum_chunks,
    xa_per_pixel_chunks,
    xa_per_image_chunks,
    norm_a2_per_pixel_chunks,
    norm_a2_per_image_chunks,
    norm_xa_per_pixel_chunks,
    norm_xa_per_image_chunks,
    aa_before_scale_per_pixel_chunks,
    aa_per_pixel_chunks,
    aa_per_image_chunks,
    posterior_probs_chunks=None,
    rotation_matrix_chunks=None,
    fine_translations=None,
    aa_feature_per_shell_chunks=None,
    aa_feature_shell_ids=None,
    atomic_xa_per_pixel=None,
    atomic_aa_per_pixel=None,
    atomic_diff2_per_pixel=None,
    atomic_diff2_rectangle=None,
    atomic_diff2_rectangle_shell_indices=None,
    noise_variance_for_noise=None,
    weighted_img_per_image=None,
    relion_norm_high_shell=None,
    norm_shifted_images=None,
):
    """Write compact Wavg XA/AA/diff2 intermediates for a target bucket."""

    target_rows = np.asarray(target_rows, dtype=np.int64)
    if target_rows.size == 0:
        return 0
    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)
    group_ids = np.asarray(bucket_group_ids, dtype=np.int64)[target_rows]
    scales = np.asarray(bucket_scale_for_stats, dtype=np.float32)[target_rows]
    mask = np.asarray(scale_correction_pixel_mask, dtype=bool).reshape(-1)
    shells = np.asarray(scale_shell_indices, dtype=np.int32).reshape(-1)
    chunk_ranges_np = np.asarray(chunk_ranges, dtype=np.int64)
    posterior_mass = np.stack(posterior_mass_chunks, axis=1)
    proj_abs2_sum = np.stack(proj_abs2_sum_chunks, axis=1)
    ctf_probs_raw_sum = np.stack(ctf_probs_raw_sum_chunks, axis=1)
    xa_per_pixel_by_chunk = np.stack(xa_per_pixel_chunks, axis=1)
    xa_per_image_by_chunk = np.stack(xa_per_image_chunks, axis=1)
    norm_a2_per_pixel_by_chunk = np.stack(norm_a2_per_pixel_chunks, axis=1)
    norm_a2_per_image_by_chunk = np.stack(norm_a2_per_image_chunks, axis=1)
    norm_xa_per_pixel_by_chunk = np.stack(norm_xa_per_pixel_chunks, axis=1)
    norm_xa_per_image_by_chunk = np.stack(norm_xa_per_image_chunks, axis=1)
    aa_before_scale = np.stack(aa_before_scale_per_pixel_chunks, axis=1)
    aa_per_pixel_by_chunk = np.stack(aa_per_pixel_chunks, axis=1)
    aa_per_image_by_chunk = np.stack(aa_per_image_chunks, axis=1)
    atomic_aa_per_pixel_np = None if atomic_aa_per_pixel is None else np.asarray(atomic_aa_per_pixel, dtype=np.float32)
    atomic_diff2_per_pixel_np = (
        None if atomic_diff2_per_pixel is None else np.asarray(atomic_diff2_per_pixel, dtype=np.float32)
    )
    atomic_xa_per_pixel_np = None if atomic_xa_per_pixel is None else np.asarray(atomic_xa_per_pixel, dtype=np.float32)
    atomic_diff2_rectangle_np = (
        None if atomic_diff2_rectangle is None else np.asarray(atomic_diff2_rectangle, dtype=np.float32)
    )
    atomic_diff2_rectangle_shells_np = (
        None
        if atomic_diff2_rectangle_shell_indices is None
        else np.asarray(atomic_diff2_rectangle_shell_indices, dtype=np.int32).reshape(-1)
    )
    if atomic_xa_per_pixel_np is not None and atomic_xa_per_pixel_np.shape != (
        local_indices.size,
        mask.size,
    ):
        raise ValueError("chunked scale-XA atomic pixel topology changed")
    if atomic_aa_per_pixel_np is not None and atomic_aa_per_pixel_np.shape != (
        local_indices.size,
        mask.size,
    ):
        raise ValueError("chunked scale-AA atomic pixel topology changed")
    if atomic_diff2_per_pixel_np is not None and atomic_diff2_per_pixel_np.shape != (
        local_indices.size,
        mask.size,
    ):
        raise ValueError("chunked Wavg diff2 atomic pixel topology changed")
    if (atomic_diff2_rectangle_np is None) != (atomic_diff2_rectangle_shells_np is None):
        raise ValueError("chunked Wavg rectangle values and shell labels must be supplied together")
    if atomic_diff2_rectangle_np is not None and atomic_diff2_rectangle_np.shape != (
        local_indices.size,
        atomic_diff2_rectangle_shells_np.size,
    ):
        raise ValueError("chunked Wavg rectangle diff2 topology changed")
    candidate_arrays_present = posterior_probs_chunks is not None or rotation_matrix_chunks is not None
    if candidate_arrays_present:
        if posterior_probs_chunks is None or rotation_matrix_chunks is None or fine_translations is None:
            raise ValueError("chunked scale-AA candidate capture is incomplete")
        posterior_probs = np.concatenate(posterior_probs_chunks, axis=1)
        rotation_matrices = np.concatenate(rotation_matrix_chunks, axis=1)
        fine_translations_np = np.asarray(fine_translations, dtype=np.float32)
        if posterior_probs.shape[:2] != rotation_matrices.shape[:2]:
            raise ValueError("chunked scale-AA candidate rotation topology changed")
        if posterior_probs.shape[2] != fine_translations_np.shape[0]:
            raise ValueError("chunked scale-AA candidate translation topology changed")
        if aa_feature_per_shell_chunks is None or aa_feature_shell_ids is None:
            raise ValueError("chunked scale-AA candidate shell features are missing")
        aa_feature_per_shell = np.concatenate(aa_feature_per_shell_chunks, axis=1)
        aa_feature_shell_ids_np = np.asarray(aa_feature_shell_ids, dtype=np.int32)
        if aa_feature_per_shell.shape[:2] != posterior_probs.shape[:2]:
            raise ValueError("chunked scale-AA candidate shell-feature topology changed")
        if aa_feature_per_shell.shape[2] != aa_feature_shell_ids_np.size:
            raise ValueError("chunked scale-AA candidate shell labels changed")
        norm_shifted_images_np = np.asarray(norm_shifted_images, dtype=np.complex64)
        if norm_shifted_images_np.shape != (
            target_rows.size,
            fine_translations_np.shape[0],
            mask.size,
        ):
            raise ValueError("chunked norm shifted-image topology changed")
    if not (
        posterior_mass.shape[:2]
        == proj_abs2_sum.shape[:2]
        == ctf_probs_raw_sum.shape[:2]
        == xa_per_pixel_by_chunk.shape[:2]
        == xa_per_image_by_chunk.shape[:2]
        == norm_a2_per_pixel_by_chunk.shape[:2]
        == norm_a2_per_image_by_chunk.shape[:2]
        == norm_xa_per_pixel_by_chunk.shape[:2]
        == norm_xa_per_image_by_chunk.shape[:2]
        == aa_before_scale.shape[:2]
        == aa_per_pixel_by_chunk.shape[:2]
        == aa_per_image_by_chunk.shape[:2]
        == (target_rows.size, chunk_ranges_np.shape[0])
    ):
        raise ValueError("chunked scale-AA capture topology changed")

    os.makedirs(dump_dir, exist_ok=True)
    context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
    context_half = int(bpref_diagnostics._bpref_contribution_context["half"])
    size_label = -1 if current_size is None else int(current_size)
    shell_count = int(np.max(shells, initial=-1)) + 1
    for selected_row, bucket_row in enumerate(target_rows.tolist()):
        xa_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        aa_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        aa_before_scale_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        ctf_probs_raw_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        proj_abs2_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        for chunk_index in range(chunk_ranges_np.shape[0]):
            xa_per_pixel = (xa_per_pixel + xa_per_pixel_by_chunk[selected_row, chunk_index]).astype(np.float32)
            aa_per_pixel = (aa_per_pixel + aa_per_pixel_by_chunk[selected_row, chunk_index]).astype(np.float32)
            aa_before_scale_per_pixel = (aa_before_scale_per_pixel + aa_before_scale[selected_row, chunk_index]).astype(
                np.float32
            )
            ctf_probs_raw_per_pixel = (ctf_probs_raw_per_pixel + ctf_probs_raw_sum[selected_row, chunk_index]).astype(
                np.float32
            )
            proj_abs2_per_pixel = (proj_abs2_per_pixel + proj_abs2_sum[selected_row, chunk_index]).astype(np.float32)
        aa_per_shell = np.zeros(shell_count, dtype=np.float64)
        valid = mask & (shells >= 0) & (shells < shell_count)
        np.add.at(
            aa_per_shell,
            shells[valid],
            aa_per_pixel[valid].astype(np.float64),
        )
        production_aa_total = float(np.sum(aa_per_image_by_chunk[selected_row], dtype=np.float64))
        production_xa_total = float(np.sum(xa_per_image_by_chunk[selected_row], dtype=np.float64))
        production_norm_a2_total = float(np.sum(norm_a2_per_image_by_chunk[selected_row], dtype=np.float64))
        production_norm_xa_total = float(np.sum(norm_xa_per_image_by_chunk[selected_row], dtype=np.float64))
        original_index = int(original_indices[bucket_row])
        out_path = os.path.join(
            dump_dir,
            f"scale_aa_chunked_orig{original_index:06d}_half{context_half}_cs{size_label:03d}.npz",
        )
        payload = dict(
            schema=np.asarray("recovar-k1-scale-xa-aa-chunked-v4"),
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
            original_index=np.int64(original_index),
            local_index=np.int64(local_indices[bucket_row]),
            bucket_row=np.int64(bucket_row),
            current_size=np.int64(size_label),
            group_id=np.int64(group_ids[selected_row]),
            scale_for_stats=np.float32(scales[selected_row]),
            chunk_ranges=chunk_ranges_np,
            posterior_mass_per_chunk=posterior_mass[selected_row],
            proj_abs2_sum_per_pixel_by_chunk=proj_abs2_sum[selected_row],
            ctf_probs_raw_sum_per_pixel_by_chunk=ctf_probs_raw_sum[selected_row],
            aa_before_scale_per_pixel_by_chunk=aa_before_scale[selected_row],
            aa_per_pixel_by_chunk=aa_per_pixel_by_chunk[selected_row],
            aa_per_image_by_chunk=aa_per_image_by_chunk[selected_row],
            scale_correction_pixel_mask=mask,
            scale_shell_indices=shells,
            proj_abs2_sum_per_pixel=proj_abs2_per_pixel,
            ctf_probs_raw_sum_per_pixel=ctf_probs_raw_per_pixel,
            scale_xa_per_pixel_by_chunk=xa_per_pixel_by_chunk[selected_row],
            scale_xa_per_image_by_chunk=xa_per_image_by_chunk[selected_row],
            scale_xa_per_pixel=xa_per_pixel,
            scale_xa_per_image=np.float64(production_xa_total),
            xa_pixel_sum_minus_production_total=np.float64(
                np.sum(xa_per_pixel, dtype=np.float64) - production_xa_total
            ),
            scale_aa_terms_before_scale_per_pixel=aa_before_scale_per_pixel,
            scale_aa_per_pixel=aa_per_pixel,
            scale_aa_per_shell=aa_per_shell,
            scale_aa_per_image=np.float64(production_aa_total),
            pixel_sum_minus_production_total=np.float64(np.sum(aa_per_pixel, dtype=np.float64) - production_aa_total),
            norm_a2_per_pixel_by_chunk=norm_a2_per_pixel_by_chunk[selected_row],
            norm_a2_per_image_by_chunk=norm_a2_per_image_by_chunk[selected_row],
            norm_a2_per_image=np.float64(production_norm_a2_total),
            norm_xa_per_pixel_by_chunk=norm_xa_per_pixel_by_chunk[selected_row],
            norm_xa_per_image_by_chunk=norm_xa_per_image_by_chunk[selected_row],
            norm_xa_per_image=np.float64(production_norm_xa_total),
            norm_residual_per_image=np.float64(production_norm_a2_total - 2.0 * production_norm_xa_total),
            noise_variance_for_noise=np.asarray(noise_variance_for_noise, dtype=np.float32),
            weighted_img_per_image=np.float64(np.asarray(weighted_img_per_image, dtype=np.float64)[bucket_row]),
            relion_norm_high_shell=np.float64(np.asarray(relion_norm_high_shell, dtype=np.float64)[bucket_row]),
        )
        if atomic_xa_per_pixel_np is not None:
            atomic_xa_pixels = atomic_xa_per_pixel_np[bucket_row]
            atomic_xa_shells = np.zeros(shell_count, dtype=np.float64)
            np.add.at(
                atomic_xa_shells,
                shells[valid],
                atomic_xa_pixels[valid].astype(np.float64),
            )
            payload.update(
                scale_xa_atomic_per_pixel=atomic_xa_pixels,
                scale_xa_atomic_per_shell=atomic_xa_shells,
                scale_xa_atomic_per_image=np.float64(np.sum(atomic_xa_pixels, dtype=np.float64)),
            )
        if atomic_aa_per_pixel_np is not None:
            atomic_pixels = atomic_aa_per_pixel_np[bucket_row]
            atomic_shells = np.zeros(shell_count, dtype=np.float64)
            np.add.at(
                atomic_shells,
                shells[valid],
                atomic_pixels[valid].astype(np.float64),
            )
            payload.update(
                scale_aa_atomic_per_pixel=atomic_pixels,
                scale_aa_atomic_per_shell=atomic_shells,
                scale_aa_atomic_per_image=np.float64(np.sum(atomic_pixels, dtype=np.float64)),
            )
        if atomic_diff2_per_pixel_np is not None:
            atomic_diff2_pixels = atomic_diff2_per_pixel_np[bucket_row]
            atomic_diff2_shells = np.zeros(shell_count, dtype=np.float64)
            valid_current_size = (shells >= 0) & (shells < shell_count)
            np.add.at(
                atomic_diff2_shells,
                shells[valid_current_size],
                atomic_diff2_pixels[valid_current_size].astype(np.float64),
            )
            payload.update(
                wavg_diff2_atomic_per_pixel=atomic_diff2_pixels,
                wavg_diff2_atomic_per_shell=atomic_diff2_shells,
                wavg_diff2_atomic_per_image=np.float64(np.sum(atomic_diff2_pixels, dtype=np.float64)),
            )
        if atomic_diff2_rectangle_np is not None:
            rectangle_pixels = atomic_diff2_rectangle_np[bucket_row]
            rectangle_shells = np.zeros(shell_count, dtype=np.float64)
            valid_rectangle = (atomic_diff2_rectangle_shells_np >= 0) & (atomic_diff2_rectangle_shells_np < shell_count)
            np.add.at(
                rectangle_shells,
                atomic_diff2_rectangle_shells_np[valid_rectangle],
                rectangle_pixels[valid_rectangle].astype(np.float64),
            )
            payload.update(
                wavg_diff2_atomic_rectangle_per_pixel=rectangle_pixels,
                wavg_diff2_atomic_rectangle_shell_indices=atomic_diff2_rectangle_shells_np,
                wavg_diff2_atomic_rectangle_per_shell=rectangle_shells,
                wavg_diff2_atomic_rectangle_per_image=np.float64(
                    np.sum(rectangle_pixels[valid_rectangle], dtype=np.float64)
                ),
            )
        if candidate_arrays_present:
            payload.update(
                candidate_posterior_probs=np.asarray(posterior_probs[selected_row], dtype=np.float32),
                candidate_rotation_matrices=np.asarray(rotation_matrices[selected_row], dtype=np.float32),
                fine_translations=fine_translations_np,
                candidate_aa_feature_per_shell=np.asarray(
                    aa_feature_per_shell[selected_row],
                    dtype=np.float32,
                ),
                candidate_aa_feature_shell_ids=aa_feature_shell_ids_np,
                norm_shifted_images=np.asarray(
                    norm_shifted_images_np[selected_row],
                    dtype=np.complex64,
                ),
            )
        np.savez_compressed(out_path, **payload)
    return int(target_rows.size)
