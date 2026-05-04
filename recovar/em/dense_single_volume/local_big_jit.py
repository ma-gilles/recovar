"""Large JIT bucket kernel for exact local EM.

This module is intentionally separate from ``local_em_engine`` so the Python
orchestration can stay thin while the numeric bucket hot path is compiled as a
single unit.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.core.padding as padding
from recovar.core import mask as core_mask
from recovar.em.dense_single_volume.helpers.adjoint import (
    batch_adjoint_slice_volume_maybe_windowed as _batch_adjoint_slice_volume_maybe_windowed,
)
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.image_shifts import tiled_half_image_phase_factors
from recovar.em.dense_single_volume.helpers.oversampling import _find_significant_mask_full_sort
from recovar.em.dense_single_volume.helpers.projection import (
    DEFAULT_PROJECTION_MAX_R,
    project_half_spectrum,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_noise_block as _compute_noise_block,
)


def _apply_integer_pre_shifts(images, shifts):
    """Apply RELION-style integer real-space pre-shifts with zero fill."""

    images = jnp.asarray(images)
    shifts = jnp.asarray(shifts, dtype=jnp.int32)
    height, width = images.shape[-2:]
    yy = jnp.arange(height, dtype=jnp.int32)[:, None]
    xx = jnp.arange(width, dtype=jnp.int32)[None, :]

    def _shift_one(image, shift):
        dx = shift[0]
        dy = shift[1]
        src_y = yy - dy
        src_x = xx - dx
        valid = (src_y >= 0) & (src_y < height) & (src_x >= 0) & (src_x < width)
        src_y = jnp.clip(src_y, 0, height - 1)
        src_x = jnp.clip(src_x, 0, width - 1)
        return jnp.where(valid, image[src_y, src_x], 0)

    return jax.vmap(_shift_one)(images, shifts)


def _preprocess_half(
    batch,
    image_mask,
    config,
    *,
    apply_image_mask: bool,
    mask_mode: str,
):
    images = jnp.asarray(batch)
    if apply_image_mask:
        if mask_mode == "relion_background_fill":
            images = core_mask.apply_relion_soft_image_mask(images, image_mask)
        elif mask_mode == "multiply":
            images = images * jnp.asarray(image_mask)
        elif mask_mode == "none":
            pass
        else:
            raise ValueError(f"unknown image mask mode {mask_mode!r}")
    images = images * jnp.asarray(config.data_multiplier, dtype=images.dtype)
    return padding.padded_rfft(images, int(config.grid_size), int(config.padding))


def _score_normalize_mstep(
    shifted_score_split,
    ctf2_over_nv_score,
    proj_weighted,
    half_weights,
    rotation_log_prior,
    translation_log_prior,
    rotation_mask,
    sample_mask,
    valid_image_mask,
    normalization_log_z,
    shifted_recon_split,
    ctf2_over_nv_recon,
    *,
    has_normalization_log_z: bool,
    half_spectrum_scoring: bool,
    use_float64_normalization: bool,
    reconstruct_significant_only: bool,
    adaptive_fraction: float,
    max_significants: int,
):
    """Score, normalize, and form M-step tensors inside the fused bucket JIT."""

    cross = (
        -2.0
        * jnp.einsum(
            "btn,brn->btr",
            jnp.conj(shifted_score_split),
            proj_weighted,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.swapaxes(1, 2)
    if half_spectrum_scoring:
        weighted_abs2 = jnp.abs(proj_weighted) ** 2
    else:
        weighted_abs2 = (jnp.abs(proj_weighted) ** 2) / half_weights[None, None, :]
    norms = jnp.einsum(
        "bn,brn->br",
        ctf2_over_nv_score,
        weighted_abs2,
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = -0.5 * (cross + norms[..., None])
    scores = scores + rotation_log_prior[:, :, None]
    scores = scores + translation_log_prior[:, None, :]
    scores = jnp.where(rotation_mask[:, :, None] & sample_mask, scores, -jnp.inf)
    scores = jnp.where(valid_image_mask[:, None, None], scores, 0.0)

    flat_scores = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    if has_normalization_log_z:
        log_Z = normalization_log_z.astype(scores.real.dtype)
    else:
        log_shift = best_log_score[:, None, None]
        if use_float64_normalization:
            shifted_exp = jnp.exp((scores - log_shift).astype(jnp.float64))
        else:
            shifted_exp = jnp.exp(scores - log_shift)
        sum_exp = jnp.sum(shifted_exp.reshape(scores.shape[0], -1), axis=1)
        log_Z = best_log_score + jnp.log(sum_exp)
    probs = jnp.exp(scores - log_Z[:, None, None])
    probs = jnp.where(valid_image_mask[:, None, None], probs, 0.0)
    best_argmax = jnp.argmax(flat_scores, axis=1)
    max_posterior = jnp.exp(best_log_score - log_Z)
    best_log_score = jnp.where(valid_image_mask, best_log_score, -jnp.inf)
    best_argmax = jnp.where(valid_image_mask, best_argmax, 0)
    max_posterior = jnp.where(valid_image_mask, max_posterior, 0.0)

    if reconstruct_significant_only:
        flat_probs = probs.reshape(probs.shape[0], -1)
        significant_flat, n_significant_samples = _find_significant_mask_full_sort(
            flat_probs,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )
        reconstruction_sample_mask = significant_flat.reshape(probs.shape)
        reconstruction_sample_mask = reconstruction_sample_mask & valid_image_mask[:, None, None]
        reconstruction_rotation_mask = jnp.any(reconstruction_sample_mask, axis=-1)
        n_significant_samples = jnp.where(valid_image_mask, n_significant_samples, 0)
        reconstruction_probs = jnp.where(reconstruction_sample_mask, probs, 0.0)
    else:
        reconstruction_sample_mask = (rotation_mask[:, :, None] & sample_mask) & valid_image_mask[:, None, None]
        reconstruction_rotation_mask = jnp.any(reconstruction_sample_mask, axis=-1)
        n_significant_samples = jnp.sum(reconstruction_sample_mask, axis=(1, 2)).astype(jnp.int32)
        reconstruction_probs = jnp.where(reconstruction_sample_mask, probs, 0.0)

    probs_sum_t = jnp.sum(probs, axis=-1)
    reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
    summed = jnp.matmul(reconstruction_probs, shifted_recon_split)
    ctf_probs = reconstruction_probs_sum_t[..., None] * ctf2_over_nv_recon[:, None, :]
    return (
        log_Z,
        best_log_score,
        best_argmax,
        max_posterior,
        reconstruction_rotation_mask,
        n_significant_samples,
        reconstruction_probs,
        probs_sum_t,
        reconstruction_probs_sum_t,
        summed,
        ctf_probs,
    )


def _adjoint_local_mstep_volumes(
    flat_summed,
    flat_ctf_probs,
    recon_window_indices,
    flat_rotations,
    Ft_y,
    Ft_ctf,
    image_shape,
    recon_volume_shape,
    disc_type,
    *,
    use_window: bool,
    max_r,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    relion_x_half_mstep: bool,
):
    """Apply enabled exact-local M-step adjoints without duplicating window branches."""

    # B.2 fix: at 256³ the stacked-adjoint path requests ~32 GiB working set
    # (Ft_y + Ft_ctf at 512³ × complex128 doubled by jnp.stack, then materialized
    # by the JIT'd backproject). Split into separate adjoint calls when the
    # padded volume is large enough that the stack peak alone would exceed
    # what the bucket-jit allocator can serve without OOM. Threshold at
    # padded_size**3 * 16 * 2 > 4 GiB → padded_size >= 256 (i.e. recovar
    # grid_size >= 128 with PADDING_FACTOR=2).
    padded_size = (
        int(np.asarray(recon_volume_shape).flat[0])
        if hasattr(recon_volume_shape, "flat")
        else (recon_volume_shape[0] if isinstance(recon_volume_shape, (list, tuple)) else int(recon_volume_shape))
    )
    split_adjoints = padded_size >= 384  # i.e. recovar grid_size >= 192
    if not disable_adjoint_y and not disable_adjoint_ctf and not split_adjoints:
        updated_volumes = _batch_adjoint_slice_volume_maybe_windowed(
            jnp.stack([flat_summed, flat_ctf_probs], axis=0),
            recon_window_indices,
            flat_rotations,
            jnp.stack([Ft_y, Ft_ctf], axis=0),
            image_shape,
            recon_volume_shape,
            disc_type,
            True,
            True,
            use_window=use_window,
            max_r=max_r,
            relion_x_half=relion_x_half_mstep,
        )
        return updated_volumes[0], updated_volumes[1]
    if not disable_adjoint_y and not disable_adjoint_ctf and split_adjoints:
        # Force the separate-adjoint fallback path even when both are enabled,
        # to halve the peak memory footprint.
        Ft_y = _batch_adjoint_slice_volume_maybe_windowed(
            flat_summed[None, :, :],
            recon_window_indices,
            flat_rotations,
            Ft_y[None, :],
            image_shape,
            recon_volume_shape,
            disc_type,
            True,
            True,
            use_window=use_window,
            max_r=max_r,
            relion_x_half=relion_x_half_mstep,
        )[0]
        Ft_ctf = _batch_adjoint_slice_volume_maybe_windowed(
            flat_ctf_probs[None, :, :],
            recon_window_indices,
            flat_rotations,
            Ft_ctf[None, :],
            image_shape,
            recon_volume_shape,
            disc_type,
            True,
            True,
            use_window=use_window,
            max_r=max_r,
            relion_x_half=relion_x_half_mstep,
        )[0]
        return Ft_y, Ft_ctf
    if not disable_adjoint_y:
        Ft_y = _batch_adjoint_slice_volume_maybe_windowed(
            flat_summed[None, :, :],
            recon_window_indices,
            flat_rotations,
            Ft_y[None, :],
            image_shape,
            recon_volume_shape,
            disc_type,
            True,
            True,
            use_window=use_window,
            max_r=max_r,
            relion_x_half=relion_x_half_mstep,
        )[0]
    if not disable_adjoint_ctf:
        Ft_ctf = _batch_adjoint_slice_volume_maybe_windowed(
            flat_ctf_probs[None, :, :],
            recon_window_indices,
            flat_rotations,
            Ft_ctf[None, :],
            image_shape,
            recon_volume_shape,
            disc_type,
            True,
            True,
            use_window=use_window,
            max_r=max_r,
            relion_x_half=relion_x_half_mstep,
        )[0]
    return Ft_y, Ft_ctf


def _project_local_half_spectrum(
    mean_for_proj,
    flat_rotations,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    projection_half_volume: bool,
    projection_max_r,
    projection_relion_texture_interp: bool,
    projection_force_jax: bool,
):
    """Project local candidates with the requested exact-local interpolation contract."""

    max_r = DEFAULT_PROJECTION_MAX_R if projection_max_r == "auto" else projection_max_r
    return project_half_spectrum(
        mean_for_proj,
        flat_rotations,
        image_shape,
        proj_volume_shape,
        disc_type,
        half_volume=projection_half_volume,
        max_r=max_r,
        relion_texture_interp=projection_relion_texture_interp,
        force_jax=projection_force_jax,
    )


@partial(
    jax.jit,
    static_argnames=(
        "mask_mode",
        "score_with_masked_images",
        "apply_integer_pre_shift",
        "apply_fourier_pre_shift",
        "half_spectrum_scoring",
        "use_float64_scoring",
        "use_float64_normalization",
        "use_window",
        "reconstruct_significant_only",
        "adaptive_fraction",
        "max_significants",
        "image_shape",
        "proj_volume_shape",
        "recon_volume_shape",
        "disc_type",
        "projection_half_volume",
        "projection_max_r",
        "projection_relion_texture_interp",
        "projection_force_jax",
        "mstep_subtract_ctf_projection",
        "mstep_relion_x_half",
        "disable_adjoint_y",
        "disable_adjoint_ctf",
        "accumulate_noise",
        "return_noise_split",
        "n_shells",
        "has_normalization_log_z",
        "has_normalization_log_evidence",
    ),
)
def run_local_bucket_big_jit(
    batch,
    ctf_params,
    mean_for_proj,
    Ft_y,
    Ft_ctf,
    noise_wsum,
    noise_img_power,
    noise_a2,
    noise_xa,
    noise_sigma2_offset,
    noise_sumw,
    image_mask,
    integer_pre_shifts,
    fourier_pre_shifts,
    image_corrections,
    image_only_corrections,
    scale_corrections,
    translation_sqdist_ang,
    noise_variance_half,
    translation_phases_half,
    half_weights,
    norm_half_weights,
    window_indices,
    recon_window_indices,
    shell_indices_half,
    shell_indices_noise,
    noise_variance_for_noise,
    local_rotations,
    rotation_log_prior,
    translation_log_prior,
    rotation_mask,
    sample_mask,
    valid_image_mask,
    normalization_log_z,
    normalization_log_evidence,
    config,
    *,
    mask_mode: str,
    score_with_masked_images: bool,
    apply_integer_pre_shift: bool,
    apply_fourier_pre_shift: bool,
    half_spectrum_scoring: bool,
    use_float64_scoring: bool,
    use_float64_normalization: bool,
    use_window: bool,
    reconstruct_significant_only: bool,
    adaptive_fraction: float,
    max_significants: int,
    image_shape,
    proj_volume_shape,
    recon_volume_shape,
    disc_type: str,
    projection_half_volume: bool,
    projection_max_r,
    projection_relion_texture_interp: bool,
    projection_force_jax: bool,
    mstep_subtract_ctf_projection: bool,
    mstep_relion_x_half: bool,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    accumulate_noise: bool,
    return_noise_split: bool,
    n_shells: int,
    has_normalization_log_z: bool,
    has_normalization_log_evidence: bool,
):
    """Run one exact-local bucket in a single compiled numeric boundary.

    The caller only enters this path for raw real-space image batches that can
    use native half-rFFT preprocessing. Debug dump paths that need intermediate
    tensors stay in ``local_em_engine``.
    """

    if apply_integer_pre_shift:
        batch = _apply_integer_pre_shifts(batch, integer_pre_shifts)

    ctf_half = config.compute_ctf_half(ctf_params)
    ctf2_over_nv_half = ctf_half**2 / noise_variance_half

    processed_score_half = _preprocess_half(
        batch,
        image_mask,
        config,
        apply_image_mask=score_with_masked_images,
        mask_mode=mask_mode,
    )
    if score_with_masked_images:
        processed_recon_half = _preprocess_half(
            batch,
            image_mask,
            config,
            apply_image_mask=False,
            mask_mode=mask_mode,
        )
    else:
        processed_recon_half = processed_score_half

    score_weighted_half = processed_score_half * ctf_half / noise_variance_half
    recon_weighted_half = processed_recon_half * ctf_half / noise_variance_half
    shifted_half = (score_weighted_half[:, None, :] * translation_phases_half[None, :, :]).reshape(
        processed_score_half.shape[0] * translation_phases_half.shape[0],
        processed_score_half.shape[1],
    )
    shifted_recon_half = (recon_weighted_half[:, None, :] * translation_phases_half[None, :, :]).reshape(
        processed_recon_half.shape[0] * translation_phases_half.shape[0],
        processed_recon_half.shape[1],
    )
    batch_norm = jnp.sum(
        (jnp.abs(processed_score_half) ** 2 / noise_variance_half) * norm_half_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real

    batch_size = processed_score_half.shape[0]
    n_trans = translation_phases_half.shape[0]
    batch_scale = scale_corrections.astype(batch_norm.dtype)
    batch_corr = image_corrections.astype(batch_norm.dtype)
    image_only_corr = image_only_corrections.astype(batch_norm.dtype)
    valid_image_mask = valid_image_mask.astype(bool)
    corr_expanded = jnp.repeat(batch_corr, n_trans)
    shifted_half = shifted_half * corr_expanded[:, None]
    shifted_recon_half = shifted_recon_half * corr_expanded[:, None]
    batch_norm = batch_norm * (image_only_corr**2)[:, None]
    ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]

    if apply_fourier_pre_shift:
        phase_expanded = tiled_half_image_phase_factors(image_shape, fourier_pre_shifts, n_trans)
        shifted_half = shifted_half * phase_expanded
        shifted_recon_half = shifted_recon_half * phase_expanded

    shifted_half_with_dc = shifted_half
    ctf2_over_nv_half_with_dc = ctf2_over_nv_half
    if half_spectrum_scoring:
        dc_mask = fourier_transform_utils.get_grid_of_radial_distances_real(image_shape, rounded=True).reshape(-1) == 0
        shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
        ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

    if use_window:
        shifted_score = shifted_half[:, window_indices]
        shifted_recon = shifted_recon_half[:, recon_window_indices]
        shifted_noise = shifted_half_with_dc[:, recon_window_indices]
        ctf2_over_nv_score = ctf2_over_nv_half[:, window_indices]
        ctf2_over_nv_recon = ctf2_over_nv_half_with_dc[:, recon_window_indices]
        score_half_weights = half_weights[window_indices]
    else:
        shifted_score = shifted_half
        shifted_recon = shifted_recon_half
        shifted_noise = shifted_half_with_dc
        ctf2_over_nv_score = ctf2_over_nv_half
        ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
        score_half_weights = half_weights

    flat_rotations = local_rotations.reshape(local_rotations.shape[0] * local_rotations.shape[1], 3, 3)
    proj_half_flat = _project_local_half_spectrum(
        mean_for_proj,
        flat_rotations,
        image_shape,
        proj_volume_shape,
        disc_type,
        projection_half_volume=projection_half_volume,
        projection_max_r=projection_max_r,
        projection_relion_texture_interp=projection_relion_texture_interp,
        projection_force_jax=projection_force_jax,
    )
    if use_window:
        proj_half = proj_half_flat[:, window_indices].reshape(
            batch_size,
            local_rotations.shape[1],
            window_indices.shape[0],
        )
        proj_for_noise = proj_half_flat[:, recon_window_indices].reshape(
            batch_size,
            local_rotations.shape[1],
            recon_window_indices.shape[0],
        )
    else:
        proj_half = proj_half_flat.reshape(batch_size, local_rotations.shape[1], -1)
        proj_for_noise = proj_half

    proj_weighted = proj_half * score_half_weights[None, None, :]
    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)
    (
        shifted_score,
        shifted_recon,
        shifted_noise,
        ctf2_over_nv_score,
        ctf2_over_nv_recon,
        proj_weighted,
        proj_for_noise,
    ) = precision_policy.cast_local_big_jit_inputs(
        shifted_score,
        shifted_recon,
        shifted_noise,
        ctf2_over_nv_score,
        ctf2_over_nv_recon,
        proj_weighted,
        proj_for_noise,
    )

    shifted_score_split = shifted_score.reshape(batch_size, n_trans, -1)
    shifted_recon_split = shifted_recon.reshape(batch_size, n_trans, -1)
    effective_normalization_log_z = normalization_log_z
    effective_has_normalization_log_z = has_normalization_log_z
    if has_normalization_log_evidence:
        normalization_dtype = jnp.float64 if use_float64_normalization else batch_norm.dtype
        log_score_offset = (-0.5 * jnp.squeeze(batch_norm, axis=1)).astype(normalization_dtype)
        effective_normalization_log_z = normalization_log_evidence.astype(normalization_dtype) - log_score_offset
        effective_has_normalization_log_z = True
    (
        log_Z,
        best_log_score,
        best_argmax,
        max_posterior,
        reconstruction_rotation_mask,
        n_significant_samples,
        reconstruction_probs,
        probs_sum_t,
        reconstruction_probs_sum_t,
        summed,
        ctf_probs,
    ) = _score_normalize_mstep(
        shifted_score_split,
        ctf2_over_nv_score,
        proj_weighted,
        score_half_weights,
        rotation_log_prior,
        translation_log_prior,
        rotation_mask,
        sample_mask,
        valid_image_mask,
        effective_normalization_log_z,
        shifted_recon_split,
        ctf2_over_nv_recon,
        has_normalization_log_z=effective_has_normalization_log_z,
        half_spectrum_scoring=half_spectrum_scoring,
        use_float64_normalization=use_float64_normalization,
        reconstruct_significant_only=reconstruct_significant_only,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
    )
    if mstep_subtract_ctf_projection:
        # RELION's VDAM/--grad storeWeightedSums backprojects
        # (Fimg_shift_nomask - Frefctf) * CTF / sigma2.
        frefctf_weighted = proj_for_noise * ctf2_over_nv_recon[:, None, :]
        summed = summed - reconstruction_probs_sum_t[..., None] * frefctf_weighted

    flat_summed = summed.reshape(batch_size * local_rotations.shape[1], summed.shape[-1])
    flat_ctf_probs = ctf_probs.reshape(batch_size * local_rotations.shape[1], ctf_probs.shape[-1])
    Ft_y, Ft_ctf = _adjoint_local_mstep_volumes(
        flat_summed,
        flat_ctf_probs,
        recon_window_indices,
        flat_rotations,
        Ft_y,
        Ft_ctf,
        image_shape,
        recon_volume_shape,
        disc_type,
        use_window=use_window,
        max_r=projection_max_r,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        relion_x_half_mstep=mstep_relion_x_half,
    )

    if accumulate_noise:
        support_mass = jnp.sum(reconstruction_probs.reshape(batch_size, -1), axis=1).astype(jnp.float32)
        support_mass = jnp.where(valid_image_mask, support_mass, 0.0)
        translation_posterior = jnp.sum(reconstruction_probs, axis=1).astype(jnp.float32)
        noise_sumw_offset = jnp.sum(translation_posterior * translation_sqdist_ang.astype(jnp.float32))
        processed_noise_power_half = processed_score_half * image_only_corr[:, None]
        batch_img_power = jnp.sum(
            (jnp.abs(processed_noise_power_half) ** 2) * support_mass[:, None],
            axis=0,
        ).astype(jnp.float32)
        batch_img_power_shells = jnp.zeros(n_shells, dtype=jnp.float32)
        batch_img_power_shells = batch_img_power_shells.at[shell_indices_half].add(batch_img_power)
        noise_img_power = noise_img_power + batch_img_power_shells
        noise_sumw = noise_sumw + jnp.sum(support_mass)

        shifted_noise_split = shifted_noise.reshape(batch_size, n_trans, -1)
        summed_masked_noise = jnp.matmul(reconstruction_probs, shifted_noise_split)
        flat_summed_masked_noise = summed_masked_noise.reshape(
            batch_size * local_rotations.shape[1],
            summed_masked_noise.shape[-1],
        )
        flat_proj_for_noise = proj_for_noise.reshape(batch_size * local_rotations.shape[1], proj_for_noise.shape[-1])
        flat_proj_abs2_for_noise = jnp.abs(flat_proj_for_noise) ** 2
        block_noise_shells, block_a2_shells, block_xa_shells = _compute_noise_block(
            flat_proj_for_noise,
            flat_proj_abs2_for_noise,
            flat_summed_masked_noise,
            flat_ctf_probs,
            noise_variance_for_noise,
            shell_indices_noise,
            n_shells,
            return_noise_split,
        )
        noise_wsum = noise_wsum + block_noise_shells
        if return_noise_split:
            noise_a2 = noise_a2 + block_a2_shells
            noise_xa = noise_xa + block_xa_shells
        noise_sigma2_offset = noise_sigma2_offset + noise_sumw_offset

    reconstruction_row_count = jnp.sum(reconstruction_rotation_mask & rotation_mask).astype(jnp.int32)
    return (
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
        reconstruction_row_count,
    )
