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
from recovar.em.dense_single_volume.helpers.half_spectrum import bin_shell_values_jax
from recovar.em.dense_single_volume.helpers.image_shifts import (
    half_image_phase_factors,
    tiled_half_image_phase_factors,
)
from recovar.em.dense_single_volume.helpers.oversampling import _find_significant_mask_full_sort
from recovar.em.dense_single_volume.helpers.projection import (
    DEFAULT_PROJECTION_MAX_R,
    compute_relion_projector_projections_block,
    project_half_spectrum,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_noise_block as _compute_noise_block,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_norm_residual_per_image as _compute_norm_residual_per_image,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_scale_correction_terms_per_image as _compute_scale_correction_terms_per_image,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_translate_wavg_norm_images,
    _relion_cuda_fine_diff2_to_scores,
    _relion_cuda_pixel_correction_from_rfloat_ctf,
    _relion_cuda_powerclass_highres_xi2_half,
    _relion_cuda_powerclass_highres_xi2_half_atomic,
    _relion_cuda_powerclass_highres_norm_units,
    _relion_cuda_powerclass_spectrum_norm_units,
    _relion_f32_fine_reconstruction_probs,
    _relion_wavg_rectangle_triplet_terms,
    _relion_wavg_sequential_triplet_terms,
)
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_mstep_sums,
    compute_local_weighted_sums,
)


def _validate_relion_exact_fine_diff2_preconditions(
    *,
    relion_exact_fine_diff2: bool,
    relion_exact_bpref_operands: bool,
    use_relion_cuda_preprocess: bool,
) -> None:
    """Validate inputs shared by compact-window and full-size fine scoring."""

    if relion_exact_fine_diff2 and not (
        relion_exact_bpref_operands and use_relion_cuda_preprocess
    ):
        raise ValueError(
            "RELION exact fine diff2 requires exact operands and CUDA preprocessing"
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


def _centered_rfft2_per_image(images):
    """Run the RELION/cuFFT packed transform with one plan per particle."""

    images = jnp.asarray(images, dtype=jnp.float32)

    def _transform(image):
        shifted = jnp.fft.fftshift(image, axes=(-2, -1))
        transformed = jnp.fft.rfft2(shifted, axes=(-2, -1))
        return jnp.fft.fftshift(transformed, axes=(-2,)).reshape(-1)

    return jax.lax.map(_transform, images)


def _norm_correction_image_power_mass(
    support_mass,
    shell_indices_half,
    valid_image_mask,
    projection_max_r,
    *,
    shell_count: int,
    include_unweighted_high_shell: bool = True,
):
    """RELION normcorr image-power mass on valid Fourier shells only."""

    support_mass = jnp.asarray(support_mass)
    shell_indices_half = jnp.asarray(shell_indices_half)
    valid_shell = (shell_indices_half >= 0) & (shell_indices_half < int(shell_count))
    weighted_mass = jnp.where(valid_shell[None, :], support_mass[:, None], 0.0)
    if projection_max_r == "auto" or projection_max_r is None:
        return weighted_mass
    full_mass = jnp.asarray(valid_image_mask, dtype=support_mass.dtype)
    unmodeled_shell = valid_shell & (shell_indices_half > int(projection_max_r))
    high_shell_mass = full_mass if include_unweighted_high_shell else jnp.zeros_like(full_mass)
    return jnp.where(unmodeled_shell[None, :], high_shell_mass[:, None], weighted_mass)


def _norm_correction_image_power_per_image(
    processed_noise_power_half,
    support_mass,
    shell_indices_half,
    valid_image_mask,
    projection_max_r,
    *,
    shell_count: int,
    image_shape,
    current_size,
    include_unweighted_high_shell: bool = True,
):
    """Return RELION norm-correction image power for one local class.

    Low/current shells carry this class's posterior support.  RELION adds the
    unweighted high-shell ``power_img`` term once per particle after its class
    loop, using CUDA ``powerClass`` divide-before-square float32 arithmetic.
    ``include_unweighted_high_shell`` lets K-class orchestration assign that
    shared term to exactly one class while preserving the single-class default.
    """

    processed_noise_power_half = jnp.asarray(processed_noise_power_half)
    pixel_power = jnp.abs(processed_noise_power_half) ** 2
    power_mass = _norm_correction_image_power_mass(
        support_mass,
        shell_indices_half,
        valid_image_mask,
        projection_max_r,
        shell_count=shell_count,
        include_unweighted_high_shell=include_unweighted_high_shell,
    )
    per_image = jnp.sum(pixel_power * power_mass, axis=-1).astype(jnp.float32)
    if (
        current_size is None
        or projection_max_r == "auto"
        or projection_max_r is None
        or not include_unweighted_high_shell
    ):
        return per_image

    shell_indices_half = jnp.asarray(shell_indices_half)
    valid_shell = (shell_indices_half >= 0) & (shell_indices_half < int(shell_count))
    unmodeled_shell = valid_shell & (shell_indices_half > int(projection_max_r))
    generic_high = jnp.sum(
        jnp.where(unmodeled_shell[None, :], pixel_power, 0.0),
        axis=-1,
    ).astype(jnp.float32)
    relion_high = _relion_cuda_powerclass_highres_norm_units(
        processed_noise_power_half,
        image_shape=image_shape,
        current_size=current_size,
    )
    full_mass = jnp.asarray(valid_image_mask, dtype=jnp.float32)
    per_image = jax.lax.optimization_barrier(per_image)
    return per_image + full_mass * (relion_high - generic_high)


def _noise_image_power_shells_and_per_image(
    processed_noise_power_half,
    support_mass,
    shell_indices_half,
    valid_image_mask,
    projection_max_r,
    *,
    shell_count: int,
    image_shape,
    current_size,
    include_unweighted_high_shell: bool = True,
    use_relion_cuda_powerclass_spectrum: bool = False,
):
    """Return RELION's noise-spectrum and norm-correction image power.

    Current-model shells are weighted by the retained posterior mass. RELION
    accumulates ``power_img`` once per valid particle above that window, so the
    high-shell spectrum must not be scaled by a Pmax/significance mass smaller
    than one.
    """

    processed_noise_power_half = jnp.asarray(processed_noise_power_half)
    pixel_power = jnp.abs(processed_noise_power_half) ** 2
    power_mass = _norm_correction_image_power_mass(
        support_mass,
        shell_indices_half,
        valid_image_mask,
        projection_max_r,
        shell_count=shell_count,
        include_unweighted_high_shell=include_unweighted_high_shell,
    )
    weighted_half = jnp.sum(pixel_power * power_mass, axis=0).astype(jnp.float32)
    shells = bin_shell_values_jax(weighted_half, shell_indices_half, shell_count)
    if (
        use_relion_cuda_powerclass_spectrum
        and current_size is not None
        and projection_max_r not in ("auto", None)
        and include_unweighted_high_shell
    ):
        exact_spectra = _relion_cuda_powerclass_spectrum_norm_units(
            processed_noise_power_half,
            image_shape=image_shape,
            current_size=current_size,
        )
        exact_spectra = jnp.where(valid_image_mask[:, None], exact_spectra, 0.0)
        exact_shells = jnp.sum(exact_spectra, axis=0).astype(jnp.float32)
        shell_ids = jnp.arange(shell_count, dtype=jnp.int32)
        shells = jnp.where(shell_ids > int(projection_max_r), exact_shells, shells)
    per_image = _norm_correction_image_power_per_image(
        processed_noise_power_half,
        support_mass,
        shell_indices_half,
        valid_image_mask,
        projection_max_r,
        shell_count=shell_count,
        image_shape=image_shape,
        current_size=current_size,
        include_unweighted_high_shell=include_unweighted_high_shell,
    )
    return shells, per_image


def _relion_wavg_direct_triplet_shells(
    processed_score_half,
    relion_score_translation_angles,
    rectangle_indices,
    exact_positions,
    rectangle_shell_indices,
    recon_window_indices,
    proj_for_noise,
    ctf_rfloat_half,
    batch_scale,
    reconstruction_probs,
    valid_image_mask,
    *,
    image_shape,
    shell_count,
    cutoff_shell,
    return_per_image_cutoff=False,
):
    """Return RELION Wavg shells and optional per-image cutoff triplets."""

    from recovar import cuda_backproject

    raw_rectangle = _relion_cuda_translate_wavg_norm_images(
        processed_score_half,
        relion_score_translation_angles,
        rectangle_indices,
        image_shape,
    )
    raw_exact = raw_rectangle[:, :, exact_positions]
    raw_ctf_exact = jnp.asarray(ctf_rfloat_half, dtype=jnp.float64)[:, recon_window_indices]
    exact_terms = _relion_wavg_sequential_triplet_terms(
        proj_for_noise,
        raw_ctf_exact,
        batch_scale,
        raw_exact,
        reconstruction_probs,
    )
    rectangle_terms = _relion_wavg_rectangle_triplet_terms(
        exact_terms,
        raw_rectangle,
        reconstruction_probs,
        exact_positions,
    )
    atomic = cuda_backproject.relion_wavg_rotation_atomic_triplet_add_f32(
        rectangle_terms,
        jnp.zeros(rectangle_terms.shape[:1] + rectangle_terms.shape[2:], dtype=jnp.float32),
    )
    atomic = jnp.where(valid_image_mask[:, None, None], atomic, 0.0)
    atomic_f64 = atomic.astype(jnp.float64)
    pixel_triplets = jnp.sum(atomic_f64, axis=0)
    shell_triplets = jnp.stack(
        [
            bin_shell_values_jax(pixel_triplets[:, component], rectangle_shell_indices, shell_count)
            for component in range(3)
        ],
        axis=0,
    )
    if return_per_image_cutoff:
        cutoff_mask = jnp.asarray(rectangle_shell_indices) == int(cutoff_shell)

        def _add_pixel(pixel, value):
            return value + jnp.where(cutoff_mask[pixel], atomic_f64[:, pixel, :], 0.0)

        per_image_cutoff = jax.lax.fori_loop(
            0,
            int(atomic.shape[1]),
            _add_pixel,
            jnp.zeros((atomic.shape[0], 3), dtype=jnp.float64),
        )
    else:
        per_image_cutoff = jnp.zeros((1, 3), dtype=jnp.float64)
    return shell_triplets, per_image_cutoff


def _exact_local_mstep_should_split_adjoints(recon_volume_shape, *arrays) -> bool:
    """Return whether exact-local y/CTF adjoints must run separately."""

    padded_size = (
        int(np.asarray(recon_volume_shape).flat[0])
        if hasattr(recon_volume_shape, "flat")
        else (recon_volume_shape[0] if isinstance(recon_volume_shape, (list, tuple)) else int(recon_volume_shape))
    )
    if padded_size >= 384:  # i.e. recovar grid_size >= 192
        return True
    dtypes = {np.dtype(getattr(array, "dtype")) for array in arrays if array is not None}
    return len(dtypes) > 1


def _score_normalize_support(
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
    reconstruction_probability_threshold=None,
    *,
    has_normalization_log_z: bool,
    has_reconstruction_probability_threshold: bool = False,
    half_spectrum_scoring: bool,
    use_float64_normalization: bool,
    reconstruct_significant_only: bool,
    use_relion_f32_fine_posterior: bool = False,
    adaptive_fraction: float,
    max_significants: int,
    scores_override=None,
):
    """Score, normalize, and form posterior support inside the fused bucket JIT."""

    if scores_override is None:
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
    else:
        scores = jnp.asarray(scores_override, dtype=jnp.float32)
    valid_sample_mask = rotation_mask[:, :, None]
    if sample_mask is not None:
        valid_sample_mask = valid_sample_mask & sample_mask
    scores = jnp.where(valid_sample_mask, scores, -jnp.inf)
    scores = jnp.where(valid_image_mask[:, None, None], scores, -jnp.inf)
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    # Keep the score tensor as an explicit float32 rounding boundary before
    # the max/log-sum-exp reductions. Without this barrier, asking the same
    # fused JIT to return scores for diagnostics can change XLA fusion and
    # flip near-tied VDAM winners. The sparse RELION-exact path uses the same
    # boundary before posterior normalization.
    scores = jax.lax.optimization_barrier(scores)

    flat_scores = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    row_has_score = jnp.isfinite(best_log_score) & valid_image_mask
    if has_normalization_log_z:
        raw_log_Z = normalization_log_z.astype(scores.real.dtype)
        row_has_mass = row_has_score & jnp.isfinite(raw_log_Z)
        log_Z = jnp.where(row_has_mass, raw_log_Z, 0.0)
    else:
        log_shift = jnp.where(row_has_score, best_log_score, 0.0)[:, None, None]
        if use_float64_normalization:
            shifted_exp = jnp.exp((scores - log_shift).astype(jnp.float64))
        else:
            shifted_exp = jnp.exp(scores - log_shift)
        shifted_exp = jnp.where(row_has_score[:, None, None] & jnp.isfinite(shifted_exp), shifted_exp, 0.0)
        sum_exp = jnp.sum(shifted_exp.reshape(scores.shape[0], -1), axis=1)
        row_has_mass = row_has_score & jnp.isfinite(sum_exp) & (sum_exp > 0.0)
        safe_sum_exp = jnp.where(row_has_mass, sum_exp, 1.0)
        log_Z = jnp.where(row_has_mass, best_log_score + jnp.log(safe_sum_exp), 0.0)
    scores_for_probs = jnp.where(row_has_mass[:, None, None], scores, -jnp.inf)
    probs = jnp.exp(scores_for_probs - log_Z[:, None, None])
    probs = jnp.where(row_has_mass[:, None, None] & jnp.isfinite(probs), probs, 0.0)
    best_argmax = jnp.argmax(flat_scores, axis=1)
    max_posterior = jnp.exp(best_log_score - log_Z)
    best_log_score = jnp.where(row_has_mass, best_log_score, -jnp.inf)
    best_argmax = jnp.where(row_has_mass, best_argmax, 0)
    max_posterior = jnp.where(row_has_mass & jnp.isfinite(max_posterior), max_posterior, 0.0)
    reconstruction_image_mask = valid_image_mask & row_has_mass

    if reconstruct_significant_only and use_relion_f32_fine_posterior:
        if has_reconstruction_probability_threshold:
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
        ) = _relion_f32_fine_reconstruction_probs(
            scores,
            adaptive_fraction=adaptive_fraction,
        )
        reconstruction_sample_mask = (
            reconstruction_sample_mask
            & reconstruction_image_mask[:, None, None]
        )
        reconstruction_probs = jnp.where(
            reconstruction_sample_mask,
            reconstruction_probs,
            jnp.float32(0.0),
        )
        reconstruction_rotation_mask = jnp.any(
            reconstruction_sample_mask,
            axis=-1,
        )
        n_significant_samples = jnp.where(
            reconstruction_image_mask,
            n_significant_samples,
            0,
        )
        max_posterior = jnp.max(
            reconstruction_probs.reshape(reconstruction_probs.shape[0], -1),
            axis=1,
        )
    elif reconstruct_significant_only:
        if has_reconstruction_probability_threshold:
            threshold = reconstruction_probability_threshold.astype(probs.dtype).reshape((probs.shape[0], 1, 1))
            reconstruction_sample_mask = (probs > 0.0) & (probs >= threshold)
            n_significant_samples = jnp.sum(
                reconstruction_sample_mask.reshape(probs.shape[0], -1),
                axis=1,
            ).astype(jnp.int32)
        else:
            flat_probs = probs.reshape(probs.shape[0], -1)
            significant_flat, n_significant_samples = _find_significant_mask_full_sort(
                flat_probs,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            reconstruction_sample_mask = significant_flat.reshape(probs.shape)
        reconstruction_sample_mask = reconstruction_sample_mask & reconstruction_image_mask[:, None, None]
        reconstruction_rotation_mask = jnp.any(reconstruction_sample_mask, axis=-1)
        n_significant_samples = jnp.where(reconstruction_image_mask, n_significant_samples, 0)
        reconstruction_probs = jnp.where(reconstruction_sample_mask, probs, 0.0)
    else:
        if sample_mask is None:
            reconstruction_sample_mask = jnp.broadcast_to(
                (rotation_mask & reconstruction_image_mask[:, None])[:, :, None],
                probs.shape,
            )
        else:
            reconstruction_sample_mask = valid_sample_mask & reconstruction_image_mask[:, None, None]
        reconstruction_rotation_mask = jnp.any(reconstruction_sample_mask, axis=-1)
        n_significant_samples = jnp.sum(reconstruction_sample_mask, axis=(1, 2)).astype(jnp.int32)
        reconstruction_probs = jnp.where(reconstruction_sample_mask, probs, 0.0)

    probs_sum_t = jnp.sum(probs, axis=-1)
    reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
    return (
        log_Z,
        scores,
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
    )


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
    reconstruction_probability_threshold=None,
    *,
    has_normalization_log_z: bool,
    has_reconstruction_probability_threshold: bool = False,
    half_spectrum_scoring: bool,
    use_float64_normalization: bool,
    reconstruct_significant_only: bool,
    use_relion_f32_fine_posterior: bool = False,
    adaptive_fraction: float,
    max_significants: int,
    sequential_translation_reduction: bool = False,
    scores_override=None,
):
    """Score, normalize, and form M-step tensors inside the fused bucket JIT."""

    (
        log_Z,
        scores,
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
    ) = _score_normalize_support(
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
        reconstruction_probability_threshold,
        has_normalization_log_z=has_normalization_log_z,
        has_reconstruction_probability_threshold=has_reconstruction_probability_threshold,
        half_spectrum_scoring=half_spectrum_scoring,
        use_float64_normalization=use_float64_normalization,
        reconstruct_significant_only=reconstruct_significant_only,
        use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        scores_override=scores_override,
    )
    summed, ctf_probs = compute_local_mstep_sums(
        reconstruction_probs,
        shifted_recon_split,
        ctf2_over_nv_recon,
        relion_x_half=sequential_translation_reduction,
        default_probs_sum_t=reconstruction_probs_sum_t,
        sequential_translation_reduction=sequential_translation_reduction,
    )
    return (
        log_Z,
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
        scores,
        probs,
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
    split_adjoints = _exact_local_mstep_should_split_adjoints(
        recon_volume_shape,
        flat_summed,
        flat_ctf_probs,
        Ft_y,
        Ft_ctf,
    )
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
    relion_projector_half,
    flat_rotations,
    projection_pixel_indices,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    projection_half_volume: bool,
    projection_max_r,
    relion_projector_output_size: int,
    projection_relion_texture_interp: bool,
    projection_force_jax: bool,
    projection_mask_current_image_disk: bool = True,
    use_relion_projector: bool,
    relion_projector_r_max: int,
    projection_padding_factor: int,
):
    """Project local candidates with the requested exact-local interpolation contract."""

    if use_relion_projector:
        projector_kwargs = {}
        if int(relion_projector_output_size) > 0:
            projector_kwargs["projector_output_size"] = int(relion_projector_output_size)
        if projection_pixel_indices is not None:
            projector_kwargs["pixel_indices"] = projection_pixel_indices
        proj_half, _ = compute_relion_projector_projections_block(
            relion_projector_half,
            flat_rotations,
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            mask_current_image_disk=projection_mask_current_image_disk,
            **projector_kwargs,
        )
        return proj_half

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
    # Ft_y and Ft_ctf are loop-carried M-step accumulators.  Donating them
    # lets XLA update multi-GB full-Nyquist BPref buffers in place instead of
    # allocating a same-sized output for every local-search bucket.
    donate_argnums=(4, 5),
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
        "use_relion_f32_fine_posterior",
        "adaptive_fraction",
        "max_significants",
        "image_shape",
        "proj_volume_shape",
        "recon_volume_shape",
        "disc_type",
        "projection_half_volume",
        "projection_max_r",
        "mstep_max_r",
        "use_compact_relion_projector_projection",
        "use_relion_projection_cache",
        "relion_projector_output_size",
        "projection_relion_texture_interp",
        "projection_force_jax",
        "projection_mask_current_image_disk",
        "relion_exact_bpref_operands",
        "relion_exact_fine_diff2",
        "relion_cuda_preprocess_radius",
        "relion_cuda_preprocess_cosine_width",
        "mstep_subtract_ctf_projection",
        "mstep_relion_x_half",
        "relion_sequential_mstep_reduction",
        "disable_adjoint_y",
        "disable_adjoint_ctf",
        "accumulate_noise",
        "accumulate_scale_correction",
        "return_noise_split",
        "return_mstep_tensors",
        "return_deferred_mstep_inputs",
        "return_deferred_noise_inputs",
        "n_shells",
        "norm_current_size",
        "include_unweighted_norm_high_shell",
        "has_normalization_log_z",
        "has_normalization_log_evidence",
        "has_reconstruction_probability_threshold",
        "score_only",
        "use_relion_projector",
        "relion_projector_r_max",
        "projection_padding_factor",
        "return_debug_arrays",
        "return_debug_scores",
        "return_debug_operands",
    ),
)
def run_local_bucket_big_jit(
    batch,
    ctf_params,
    ctf_rfloat_half,
    inverse_noise_rfloat_cast_half,
    corr_img_rfloat_square_half,
    mean_for_proj,
    relion_projector_half,
    Ft_y,
    Ft_ctf,
    noise_wsum,
    noise_img_power,
    noise_a2,
    noise_xa,
    noise_scale_xa,
    noise_scale_aa,
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
    relion_score_translation_angles,
    half_weights,
    norm_half_weights,
    window_indices,
    relion_fine_full_to_compact,
    recon_window_indices,
    relion_wavg_rectangle_indices,
    relion_wavg_exact_positions,
    relion_wavg_rectangle_shell_indices,
    mstep_recon_window_indices,
    shell_indices_half,
    shell_indices_noise,
    noise_variance_for_noise,
    scale_correction_pixel_mask,
    projection_pixel_indices,
    projection_score_take_indices,
    projection_recon_take_indices,
    relion_projection_cache,
    relion_projection_cache_id_map,
    local_rotation_ids_for_projection_cache,
    local_rotations,
    local_mstep_rotations,
    rotation_log_prior,
    translation_log_prior,
    rotation_mask,
    sample_mask,
    valid_image_mask,
    group_ids,
    normalization_log_z,
    normalization_log_evidence,
    reconstruction_probability_threshold,
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
    use_relion_f32_fine_posterior: bool = False,
    adaptive_fraction: float,
    max_significants: int,
    image_shape,
    proj_volume_shape,
    recon_volume_shape,
    disc_type: str,
    projection_half_volume: bool,
    projection_max_r,
    mstep_max_r,
    use_compact_relion_projector_projection: bool,
    use_relion_projection_cache: bool,
    relion_projector_output_size: int,
    projection_relion_texture_interp: bool,
    projection_force_jax: bool,
    projection_mask_current_image_disk: bool = True,
    relion_exact_bpref_operands: bool = False,
    relion_exact_fine_diff2: bool = False,
    relion_cuda_preprocess_radius: float = 0.0,
    relion_cuda_preprocess_cosine_width: float = 0.0,
    mstep_subtract_ctf_projection: bool,
    mstep_relion_x_half: bool,
    relion_sequential_mstep_reduction: bool = False,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    accumulate_noise: bool,
    accumulate_scale_correction: bool,
    return_noise_split: bool,
    return_mstep_tensors: bool,
    return_deferred_mstep_inputs: bool,
    return_deferred_noise_inputs: bool,
    n_shells: int,
    norm_current_size: int | None,
    include_unweighted_norm_high_shell: bool,
    has_normalization_log_z: bool,
    has_normalization_log_evidence: bool,
    has_reconstruction_probability_threshold: bool,
    score_only: bool = False,
    use_relion_projector: bool = False,
    relion_projector_r_max: int = 0,
    projection_padding_factor: int = 1,
    return_debug_arrays: bool = False,
    return_debug_scores: bool = False,
    return_debug_operands: bool = False,
):
    """Run one exact-local bucket in a single compiled numeric boundary.

    The caller only enters this path for raw real-space image batches that can
    use native half-rFFT preprocessing. Debug dumps may request bucket-local
    score/probability tensors and, for targeted operand dumps, the already
    computed projection/preprocessing operands.
    """

    if score_only and (
        (not disable_adjoint_y)
        or (not disable_adjoint_ctf)
        or accumulate_noise
        or mstep_subtract_ctf_projection
        or return_mstep_tensors
        or return_deferred_mstep_inputs
    ):
        raise ValueError("score_only local big-JIT requires disabled adjoints, no noise, and no M-step outputs")
    if return_deferred_mstep_inputs and (
        return_mstep_tensors
        or accumulate_noise
        or mstep_subtract_ctf_projection
        or (not disable_adjoint_y)
        or (not disable_adjoint_ctf)
    ):
        raise ValueError(
            "deferred local big-JIT M-step returns posterior/preprocessed inputs only; "
            "disable in-kernel adjoints, residual subtraction, full M-step tensors, and in-kernel noise"
        )

    use_relion_cuda_preprocess = bool(
        relion_exact_bpref_operands and relion_cuda_preprocess_radius > 0.0
    )
    _validate_relion_exact_fine_diff2_preconditions(
        relion_exact_fine_diff2=relion_exact_fine_diff2,
        relion_exact_bpref_operands=relion_exact_bpref_operands,
        use_relion_cuda_preprocess=use_relion_cuda_preprocess,
    )
    if use_relion_cuda_preprocess and relion_cuda_preprocess_cosine_width <= 0.0:
        raise ValueError("RELION CUDA preprocessing requires a positive cosine width")
    if use_relion_cuda_preprocess and apply_fourier_pre_shift:
        raise ValueError("RELION CUDA preprocessing requires integral real-space pre-shifts")
    if apply_integer_pre_shift and not use_relion_cuda_preprocess:
        batch = _apply_integer_pre_shifts(batch, integer_pre_shifts)

    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)
    if relion_exact_bpref_operands:
        inverse_noise_half = jnp.asarray(
            inverse_noise_rfloat_cast_half,
            dtype=jnp.float32,
        )
        ctf_rfloat_half = jnp.asarray(ctf_rfloat_half, dtype=jnp.float64)
        ctf_half = ctf_rfloat_half.astype(jnp.float32)
        weighted_ctf_half = ctf_half * inverse_noise_half[None, :]
        # BPref evaluates (Minvsigma2 * CTF) * CTF in XFLOAT, while the
        # fine-score corr_img squares the RFLOAT CTF before the final XFLOAT
        # cast. Keep the two demonstrated RELION orders separate.
        ctf2_over_nv_recon_half = weighted_ctf_half * ctf_half
        ctf2_over_nv_score_half = jnp.asarray(
            corr_img_rfloat_square_half,
            dtype=jnp.float32,
        )
        noise_variance_half = jnp.asarray(
            noise_variance_half, dtype=jnp.float32
        )
    else:
        ctf_half = config.compute_ctf_half(ctf_params).astype(
            precision_policy.score_real_dtype
        )
        noise_variance_half = noise_variance_half.astype(
            precision_policy.score_real_dtype
        )
        inverse_noise_half = None
        weighted_ctf_half = None
        ctf2_over_nv_recon_half = ctf_half**2 / noise_variance_half
        ctf2_over_nv_score_half = ctf2_over_nv_recon_half
    translation_phases_half = translation_phases_half.astype(precision_policy.score_complex_dtype)
    if relion_score_translation_angles is not None:
        relion_score_translation_angles = relion_score_translation_angles.astype(jnp.float32)
    if use_relion_cuda_preprocess:
        from recovar import cuda_backproject

        normalized_images, masked_images = cuda_backproject.relion_preprocess_real_f32(
            jnp.asarray(batch, dtype=jnp.float32),
            jnp.asarray(image_only_corrections, dtype=jnp.float32),
            jnp.asarray(integer_pre_shifts, dtype=jnp.int32),
            float(relion_cuda_preprocess_radius),
            float(relion_cuda_preprocess_cosine_width),
            apply_mask=score_with_masked_images,
        )
        score_images = masked_images if score_with_masked_images else normalized_images
        processed_score_half = _centered_rfft2_per_image(score_images).astype(
            precision_policy.score_complex_dtype
        )
        if score_only:
            processed_recon_half = None
        elif score_with_masked_images:
            processed_recon_half = _centered_rfft2_per_image(normalized_images).astype(
                precision_policy.score_complex_dtype
            )
        else:
            processed_recon_half = processed_score_half
    else:
        processed_score_half = _preprocess_half(
            batch,
            image_mask,
            config,
            apply_image_mask=score_with_masked_images,
            mask_mode=mask_mode,
        ).astype(precision_policy.score_complex_dtype)
        if score_only:
            processed_recon_half = None
        elif score_with_masked_images:
            processed_recon_half = _preprocess_half(
                batch,
                image_mask,
                config,
                apply_image_mask=False,
                mask_mode=mask_mode,
            ).astype(precision_policy.score_complex_dtype)
        else:
            processed_recon_half = processed_score_half

    batch_size = processed_score_half.shape[0]
    n_trans = translation_phases_half.shape[0]
    materialize_shifted_noise = not (return_deferred_mstep_inputs and not return_deferred_noise_inputs)

    def _translate_score_weighted_half(weighted_half, pixel_indices):
        if relion_score_translation_angles is not None:
            from recovar import cuda_backproject

            return cuda_backproject.relion_translate_score_f32(
                jnp.asarray(weighted_half, dtype=jnp.complex64),
                relion_score_translation_angles,
                jnp.asarray(pixel_indices, dtype=jnp.int32),
                image_shape,
            )
        translation_phases = translation_phases_half[:, pixel_indices]
        return (weighted_half[:, None, :] * translation_phases[None, :, :]).reshape(
            batch_size * n_trans,
            pixel_indices.shape[0],
        )

    def _translate_weighted_half_window(processed_half, pixel_indices):
        if relion_exact_bpref_operands:
            weighted_half = (
                processed_half[:, pixel_indices]
                * weighted_ctf_half[:, pixel_indices]
            )
        else:
            weighted_half = (
                processed_half[:, pixel_indices]
                * ctf_half[:, pixel_indices]
                / noise_variance_half[pixel_indices]
            )
        translation_phases = translation_phases_half[:, pixel_indices]
        return (weighted_half[:, None, :] * translation_phases[None, :, :]).reshape(
            batch_size * n_trans,
            pixel_indices.shape[0],
        )

    if use_window:
        if relion_exact_bpref_operands:
            score_weighted_half = (
                processed_score_half[:, window_indices]
                * weighted_ctf_half[:, window_indices]
            )
        else:
            score_weighted_half = (
                processed_score_half[:, window_indices]
                * ctf_half[:, window_indices]
                / noise_variance_half[window_indices]
            )
        shifted_score = _translate_score_weighted_half(score_weighted_half, window_indices)
        ctf2_over_nv_score = ctf2_over_nv_score_half[:, window_indices]
        score_half_weights = half_weights[window_indices]
        if not score_only:
            shifted_recon = _translate_weighted_half_window(processed_recon_half, recon_window_indices)
            if materialize_shifted_noise:
                shifted_noise = _translate_weighted_half_window(processed_score_half, recon_window_indices)
            else:
                shifted_noise = jnp.zeros((1, 1), dtype=shifted_score.dtype)
            ctf2_over_nv_recon = ctf2_over_nv_recon_half[:, recon_window_indices]
    else:
        score_weighted_half = (
            processed_score_half * weighted_ctf_half
            if relion_exact_bpref_operands
            else processed_score_half * ctf_half / noise_variance_half
        )
        shifted_half = _translate_score_weighted_half(
            score_weighted_half,
            jnp.arange(processed_score_half.shape[1], dtype=jnp.int32),
        )
        if not score_only:
            recon_weighted_half = (
                processed_recon_half * weighted_ctf_half
                if relion_exact_bpref_operands
                else processed_recon_half * ctf_half / noise_variance_half
            )
            shifted_recon_half = (recon_weighted_half[:, None, :] * translation_phases_half[None, :, :]).reshape(
                batch_size * n_trans,
                processed_recon_half.shape[1],
            )
    score_power_over_noise = (
        jnp.abs(processed_score_half) ** 2 * inverse_noise_half[None, :]
        if relion_exact_bpref_operands
        else jnp.abs(processed_score_half) ** 2 / noise_variance_half
    )
    batch_norm = jnp.sum(
        score_power_over_noise * norm_half_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real

    batch_scale = scale_corrections.astype(batch_norm.dtype)
    batch_corr = image_corrections.astype(batch_norm.dtype)
    image_only_corr = image_only_corrections.astype(batch_norm.dtype)
    valid_image_mask = valid_image_mask.astype(bool)
    applied_corr = batch_scale if use_relion_cuda_preprocess else batch_corr
    corr_expanded = jnp.repeat(applied_corr, n_trans)
    if use_window:
        shifted_score = shifted_score * corr_expanded[:, None]
        if not score_only:
            shifted_recon = shifted_recon * corr_expanded[:, None]
            if materialize_shifted_noise:
                shifted_noise = shifted_noise * corr_expanded[:, None]
    else:
        shifted_half = shifted_half * corr_expanded[:, None]
        if not score_only:
            shifted_recon_half = shifted_recon_half * corr_expanded[:, None]
    if not use_relion_cuda_preprocess:
        batch_norm = batch_norm * (image_only_corr**2)[:, None]
    ctf2_over_nv_score_half = ctf2_over_nv_score_half * (batch_scale**2)[:, None]
    ctf2_over_nv_recon_half = ctf2_over_nv_recon_half * (batch_scale**2)[:, None]
    if use_window:
        ctf2_over_nv_score = ctf2_over_nv_score * (batch_scale**2)[:, None]
        if not score_only:
            ctf2_over_nv_recon = ctf2_over_nv_recon * (batch_scale**2)[:, None]

    if apply_fourier_pre_shift:
        if use_window:
            pre_shift_phases = half_image_phase_factors(image_shape, fourier_pre_shifts)
            score_phase_expanded = jnp.repeat(pre_shift_phases[:, window_indices], n_trans, axis=0)
            shifted_score = shifted_score * score_phase_expanded
            if not score_only:
                recon_phase_expanded = jnp.repeat(pre_shift_phases[:, recon_window_indices], n_trans, axis=0)
                shifted_recon = shifted_recon * recon_phase_expanded
                if materialize_shifted_noise:
                    shifted_noise = shifted_noise * recon_phase_expanded
        else:
            phase_expanded = tiled_half_image_phase_factors(image_shape, fourier_pre_shifts, n_trans)
            shifted_half = shifted_half * phase_expanded
            if not score_only:
                shifted_recon_half = shifted_recon_half * phase_expanded

    if half_spectrum_scoring:
        dc_mask = fourier_transform_utils.get_grid_of_radial_distances_real(image_shape, rounded=True).reshape(-1) == 0
        if use_window:
            score_dc_mask = dc_mask[window_indices]
            shifted_score = jnp.where(score_dc_mask[None, :], 0.0, shifted_score)
            ctf2_over_nv_score = jnp.where(score_dc_mask[None, :], 0.0, ctf2_over_nv_score)
        else:
            shifted_half_with_dc = shifted_half
            if not score_only:
                ctf2_over_nv_score_half_with_dc = ctf2_over_nv_score_half
                ctf2_over_nv_recon_half_with_dc = ctf2_over_nv_recon_half
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            ctf2_over_nv_score_half = jnp.where(
                dc_mask[None, :], 0.0, ctf2_over_nv_score_half
            )
            ctf2_over_nv_recon_half = jnp.where(
                dc_mask[None, :], 0.0, ctf2_over_nv_recon_half
            )

    if not use_window:
        if not half_spectrum_scoring:
            shifted_half_with_dc = shifted_half
            if not score_only:
                ctf2_over_nv_score_half_with_dc = ctf2_over_nv_score_half
                ctf2_over_nv_recon_half_with_dc = ctf2_over_nv_recon_half
        shifted_score = shifted_half
        ctf2_over_nv_score = ctf2_over_nv_score_half
        score_half_weights = half_weights
        if not score_only:
            shifted_recon = shifted_recon_half
            if return_deferred_mstep_inputs and not return_deferred_noise_inputs:
                shifted_noise = jnp.zeros((1, 1), dtype=shifted_half_with_dc.dtype)
            else:
                shifted_noise = shifted_half_with_dc
            ctf2_over_nv_recon = ctf2_over_nv_recon_half_with_dc

    flat_rotations = local_rotations.reshape(local_rotations.shape[0] * local_rotations.shape[1], 3, 3)
    flat_mstep_rotations = local_mstep_rotations.reshape(
        local_mstep_rotations.shape[0] * local_mstep_rotations.shape[1],
        3,
        3,
    )
    if use_relion_projection_cache:
        safe_rotation_ids = jnp.maximum(local_rotation_ids_for_projection_cache.reshape(-1), 0)
        cache_rows = relion_projection_cache_id_map[safe_rotation_ids]
        proj_half_flat = relion_projection_cache[cache_rows]
    else:
        proj_half_flat = _project_local_half_spectrum(
            mean_for_proj,
            relion_projector_half,
            flat_rotations,
            projection_pixel_indices if use_compact_relion_projector_projection else None,
            image_shape,
            proj_volume_shape,
            disc_type,
            projection_half_volume=projection_half_volume,
            projection_max_r=projection_max_r,
            relion_projector_output_size=relion_projector_output_size,
            projection_relion_texture_interp=projection_relion_texture_interp,
            projection_force_jax=projection_force_jax,
            projection_mask_current_image_disk=projection_mask_current_image_disk,
            use_relion_projector=use_relion_projector,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
        )
    if use_window:
        if use_compact_relion_projector_projection:
            proj_half = proj_half_flat[:, projection_score_take_indices].reshape(
                batch_size,
                local_rotations.shape[1],
                projection_score_take_indices.shape[0],
            )
        else:
            proj_half = proj_half_flat[:, window_indices].reshape(
                batch_size,
                local_rotations.shape[1],
                window_indices.shape[0],
            )
        if not score_only:
            if use_compact_relion_projector_projection:
                if return_deferred_mstep_inputs and not accumulate_noise and not return_debug_operands:
                    proj_for_noise = jnp.zeros((1, 1, 1), dtype=proj_half.dtype)
                else:
                    proj_for_noise = proj_half_flat[:, projection_recon_take_indices].reshape(
                        batch_size,
                        local_rotations.shape[1],
                        projection_recon_take_indices.shape[0],
                    )
            else:
                if return_deferred_mstep_inputs and not accumulate_noise and not return_debug_operands:
                    proj_for_noise = jnp.zeros((1, 1, 1), dtype=proj_half.dtype)
                else:
                    proj_for_noise = proj_half_flat[:, recon_window_indices].reshape(
                        batch_size,
                        local_rotations.shape[1],
                        recon_window_indices.shape[0],
                    )
    else:
        proj_half = proj_half_flat.reshape(batch_size, local_rotations.shape[1], -1)
        if not score_only:
            if return_deferred_mstep_inputs and not accumulate_noise and not return_debug_operands:
                proj_for_noise = jnp.zeros((1, 1, 1), dtype=proj_half.dtype)
            else:
                proj_for_noise = proj_half

    proj_weighted = proj_half * score_half_weights[None, None, :]

    direct_scores = None
    if relion_exact_fine_diff2:
        from recovar import cuda_backproject

        pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
            scale_corrections[:, None],
            ctf_rfloat_half,
        )
        corrected_score = (
            processed_score_half[:, window_indices] * pixel_correction[:, window_indices]
        ).astype(jnp.complex64)
        direct_weight = (
            ctf2_over_nv_score * score_half_weights[None, :]
        ).astype(jnp.float32)
        direct_highres = _relion_cuda_powerclass_highres_xi2_half_atomic(
            processed_score_half,
            image_shape=image_shape,
            current_size=norm_current_size,
        )
        direct_diff2 = cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
            jnp.asarray(proj_half, dtype=jnp.complex64),
            corrected_score,
            jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
            direct_weight,
            jnp.asarray(relion_fine_full_to_compact, dtype=jnp.int32),
            direct_highres,
            current_size=norm_current_size,
        )
        direct_candidate_mask = rotation_mask[:, :, None]
        if sample_mask is not None:
            direct_candidate_mask = direct_candidate_mask & sample_mask
        direct_scores = _relion_cuda_fine_diff2_to_scores(
            direct_diff2,
            rotation_log_prior[:, :, None],
            translation_log_prior[:, None, :],
            direct_candidate_mask,
        )

    def _append_debug_outputs(
        result,
        debug_scores,
        debug_probs,
        shifted_score_split,
        ctf2_over_nv_score,
        proj_weighted,
        shifted_recon_split=None,
        ctf2_over_nv_recon=None,
        proj_for_noise=None,
        wavg_cutoff_triplet=None,
    ):
        if not return_debug_arrays:
            return result
        debug_scores_for_return = (
            debug_scores
            if return_debug_scores
            else jnp.zeros((1, 1, 1), dtype=debug_probs.dtype)
        )
        if not return_debug_operands:
            return result + (debug_scores_for_return, debug_probs)
        debug_shifted_recon = (
            shifted_recon_split
            if shifted_recon_split is not None
            else jnp.zeros((1, 1, 1), dtype=shifted_score_split.dtype)
        )
        debug_ctf2_over_nv_recon = (
            ctf2_over_nv_recon
            if ctf2_over_nv_recon is not None
            else jnp.zeros((1, 1), dtype=ctf2_over_nv_score.dtype)
        )
        debug_proj_for_noise = (
            proj_for_noise
            if proj_for_noise is not None
            else jnp.zeros((1, 1, 1), dtype=proj_weighted.dtype)
        )
        debug_wavg_cutoff_triplet = (
            wavg_cutoff_triplet
            if wavg_cutoff_triplet is not None
            else jnp.zeros((1, 3), dtype=jnp.float64)
        )
        return result + (
            debug_scores_for_return,
            debug_probs,
            shifted_score_split,
            debug_shifted_recon,
            ctf2_over_nv_score,
            debug_ctf2_over_nv_recon,
            proj_weighted,
            debug_proj_for_noise,
            debug_wavg_cutoff_triplet,
        )

    if score_only:
        shifted_score = shifted_score.astype(precision_policy.score_complex_dtype)
        ctf2_over_nv_score = ctf2_over_nv_score.astype(precision_policy.score_real_dtype)
        proj_weighted = proj_weighted.astype(precision_policy.score_complex_dtype)
        shifted_score_split = shifted_score.reshape(batch_size, n_trans, -1)
        effective_normalization_log_z = normalization_log_z
        effective_has_normalization_log_z = has_normalization_log_z
        if has_normalization_log_evidence:
            normalization_dtype = jnp.float64 if use_float64_normalization else batch_norm.dtype
            log_score_offset = (-0.5 * jnp.squeeze(batch_norm, axis=1)).astype(normalization_dtype)
            effective_normalization_log_z = normalization_log_evidence.astype(normalization_dtype) - log_score_offset
            effective_has_normalization_log_z = True
        (
            log_Z,
            debug_scores,
            debug_probs,
            best_log_score,
            best_argmax,
            max_posterior,
            reconstruction_sample_mask,
            reconstruction_rotation_mask,
            n_significant_samples,
            _reconstruction_probs,
            probs_sum_t,
            reconstruction_probs_sum_t,
        ) = _score_normalize_support(
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
            reconstruction_probability_threshold,
            has_normalization_log_z=effective_has_normalization_log_z,
            has_reconstruction_probability_threshold=has_reconstruction_probability_threshold,
            half_spectrum_scoring=half_spectrum_scoring,
            use_float64_normalization=use_float64_normalization,
            reconstruct_significant_only=reconstruct_significant_only,
            use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            scores_override=direct_scores,
        )
        reconstruction_row_count = jnp.sum(reconstruction_rotation_mask & rotation_mask).astype(jnp.int32)
        result = (
            Ft_y,
            Ft_ctf,
            noise_wsum,
            noise_img_power,
            noise_a2,
            noise_xa,
            noise_scale_xa,
            noise_scale_aa,
            jnp.zeros((batch_size,), dtype=jnp.float32),
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
            reconstruction_row_count,
        )
        return _append_debug_outputs(
            result,
            debug_scores,
            debug_probs,
            shifted_score_split,
            ctf2_over_nv_score,
            proj_weighted,
        )
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
    if return_deferred_mstep_inputs:
        (
            log_Z,
            debug_scores,
            debug_probs,
            best_log_score,
            best_argmax,
            max_posterior,
            reconstruction_sample_mask,
            reconstruction_rotation_mask,
            n_significant_samples,
            reconstruction_probs,
            probs_sum_t,
            reconstruction_probs_sum_t,
        ) = _score_normalize_support(
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
            reconstruction_probability_threshold,
            has_normalization_log_z=effective_has_normalization_log_z,
            has_reconstruction_probability_threshold=has_reconstruction_probability_threshold,
            half_spectrum_scoring=half_spectrum_scoring,
            use_float64_normalization=use_float64_normalization,
            reconstruct_significant_only=reconstruct_significant_only,
            use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            scores_override=direct_scores,
        )
        reconstruction_row_count = jnp.sum(reconstruction_rotation_mask & rotation_mask).astype(jnp.int32)
        if return_deferred_noise_inputs:
            shifted_noise_for_return = shifted_noise.reshape(batch_size, n_trans, -1)
            processed_score_half_for_return = processed_score_half
        else:
            shifted_noise_for_return = jnp.zeros((1, 1, 1), dtype=shifted_recon_split.dtype)
            processed_score_half_for_return = jnp.zeros((1, 1), dtype=processed_score_half.dtype)
        result = (
            Ft_y,
            Ft_ctf,
            noise_wsum,
            noise_img_power,
            noise_a2,
            noise_xa,
            noise_scale_xa,
            noise_scale_aa,
            jnp.zeros((batch_size,), dtype=jnp.float32),
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
            reconstruction_row_count,
            reconstruction_probs,
            shifted_recon_split,
            ctf2_over_nv_recon,
            shifted_noise_for_return,
            processed_score_half_for_return,
        )
        return _append_debug_outputs(
            result,
            debug_scores,
            debug_probs,
            shifted_score_split,
            ctf2_over_nv_score,
            proj_weighted,
            shifted_recon_split=shifted_recon_split,
            ctf2_over_nv_recon=ctf2_over_nv_recon,
            proj_for_noise=proj_for_noise,
        )
    (
        log_Z,
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
        debug_scores,
        debug_probs,
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
        reconstruction_probability_threshold=reconstruction_probability_threshold,
        has_normalization_log_z=effective_has_normalization_log_z,
        has_reconstruction_probability_threshold=has_reconstruction_probability_threshold,
        half_spectrum_scoring=half_spectrum_scoring,
        use_float64_normalization=use_float64_normalization,
        reconstruct_significant_only=reconstruct_significant_only,
        use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        sequential_translation_reduction=relion_sequential_mstep_reduction,
        scores_override=direct_scores,
    )
    source_ordered_vdam_mstep = bool(
        relion_sequential_mstep_reduction
        and mstep_subtract_ctf_projection
        and relion_exact_bpref_operands
        and use_relion_cuda_preprocess
        and not apply_fourier_pre_shift
        and relion_score_translation_angles is not None
    )
    source_ordered_vdam_scattered = bool(
        source_ordered_vdam_mstep
        and not return_mstep_tensors
        and not disable_adjoint_y
        and not disable_adjoint_ctf
    )
    if source_ordered_vdam_mstep:
        from recovar import cuda_backproject

        bpref_pixel_indices = (
            recon_window_indices
            if use_window
            else jnp.arange(processed_recon_half.shape[1], dtype=jnp.int32)
        )
        bpref_ctf = (
            ctf_half[:, bpref_pixel_indices] * batch_scale[:, None]
        ).astype(jnp.float32)
        bpref_minvsigma2 = jnp.broadcast_to(
            inverse_noise_half[bpref_pixel_indices][None, :],
            bpref_ctf.shape,
        )
        if source_ordered_vdam_scattered:
            Ft_y, Ft_ctf, ctf_probs = cuda_backproject.relion_vdam_mstep_fused_x_half(
                Ft_y,
                Ft_ctf,
                jnp.asarray(
                    processed_recon_half[:, bpref_pixel_indices],
                    dtype=jnp.complex64,
                ),
                bpref_ctf,
                jnp.asarray(bpref_minvsigma2, dtype=jnp.float32),
                jnp.asarray(reconstruction_probs, dtype=jnp.float32),
                jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
                jnp.asarray(mstep_recon_window_indices, dtype=jnp.int32),
                jnp.asarray(proj_for_noise, dtype=jnp.complex64),
                jnp.asarray(local_mstep_rotations, dtype=jnp.float32),
                image_shape,
                recon_volume_shape,
                float(mstep_max_r),
            )
        else:
            summed, ctf_probs = cuda_backproject.relion_vdam_mstep_sums_f32(
                jnp.asarray(
                    processed_recon_half[:, bpref_pixel_indices],
                    dtype=jnp.complex64,
                ),
                bpref_ctf,
                jnp.asarray(bpref_minvsigma2, dtype=jnp.float32),
                jnp.asarray(reconstruction_probs, dtype=jnp.float32),
                jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
                jnp.asarray(bpref_pixel_indices, dtype=jnp.int32),
                jnp.asarray(proj_for_noise, dtype=jnp.complex64),
                image_shape,
            )
    elif mstep_subtract_ctf_projection:
        # RELION's VDAM/--grad storeWeightedSums backprojects
        # (Fimg_shift_nomask - Frefctf) * CTF / sigma2.
        frefctf_weighted = proj_for_noise * ctf2_over_nv_recon[:, None, :]
        frefctf_delta = jnp.where(
            reconstruction_probs_sum_t[..., None] != 0.0,
            reconstruction_probs_sum_t[..., None] * frefctf_weighted,
            0.0,
        )
        summed = summed - frefctf_delta

    flat_ctf_probs = ctf_probs.reshape(batch_size * local_rotations.shape[1], ctf_probs.shape[-1])
    if not source_ordered_vdam_scattered:
        flat_summed = summed.reshape(batch_size * local_rotations.shape[1], summed.shape[-1])
        Ft_y, Ft_ctf = _adjoint_local_mstep_volumes(
            flat_summed,
            flat_ctf_probs,
            mstep_recon_window_indices,
            flat_mstep_rotations,
            Ft_y,
            Ft_ctf,
            image_shape,
            recon_volume_shape,
            disc_type,
            use_window=use_window,
            max_r=mstep_max_r,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            relion_x_half_mstep=mstep_relion_x_half,
        )

    bucket_norm_correction = jnp.zeros((batch_size,), dtype=jnp.float32)
    debug_wavg_cutoff_triplet = jnp.zeros((1, 3), dtype=jnp.float64)
    if accumulate_noise:
        support_mass = jnp.sum(reconstruction_probs.reshape(batch_size, -1), axis=1).astype(jnp.float32)
        support_mass = jnp.where(valid_image_mask, support_mass, 0.0)
        translation_posterior = jnp.sum(reconstruction_probs, axis=1).astype(jnp.float32)
        noise_sumw_offset = jnp.sum(translation_posterior * translation_sqdist_ang.astype(jnp.float32))
        processed_noise_power_half = processed_score_half * image_only_corr[:, None]
        batch_img_power_shells, batch_img_power_per_image = _noise_image_power_shells_and_per_image(
            processed_noise_power_half,
            support_mass,
            shell_indices_half,
            valid_image_mask,
            projection_max_r,
            shell_count=n_shells,
            image_shape=image_shape,
            current_size=norm_current_size,
            include_unweighted_high_shell=include_unweighted_norm_high_shell,
            use_relion_cuda_powerclass_spectrum=relion_exact_fine_diff2,
        )
        noise_sumw = noise_sumw + jnp.sum(support_mass)

        shifted_noise_split = shifted_noise.reshape(batch_size, n_trans, -1)
        shifted_noise_split = jnp.where(support_mass[:, None, None] != 0.0, shifted_noise_split, 0.0)
        summed_masked_noise = compute_local_weighted_sums(
            reconstruction_probs,
            shifted_noise_split,
        )
        flat_summed_masked_noise = summed_masked_noise.reshape(
            batch_size * local_rotations.shape[1],
            summed_masked_noise.shape[-1],
        )
        flat_proj_for_noise = proj_for_noise.reshape(batch_size * local_rotations.shape[1], proj_for_noise.shape[-1])
        proj_abs2_for_norm = jnp.abs(proj_for_noise) ** 2
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
        use_relion_wavg_cutoff = bool(
            relion_exact_fine_diff2 and use_window and norm_current_size is not None
        )
        if use_relion_wavg_cutoff:
            direct_triplet_shells, debug_wavg_cutoff_triplet = _relion_wavg_direct_triplet_shells(
                processed_score_half,
                relion_score_translation_angles,
                relion_wavg_rectangle_indices,
                relion_wavg_exact_positions,
                relion_wavg_rectangle_shell_indices,
                recon_window_indices,
                proj_for_noise,
                ctf_rfloat_half,
                batch_scale,
                reconstruction_probs,
                valid_image_mask,
                image_shape=image_shape,
                shell_count=n_shells,
                cutoff_shell=int(norm_current_size) // 2,
                return_per_image_cutoff=return_debug_operands,
            )
            cutoff_mask = jnp.arange(n_shells, dtype=jnp.int32) == int(norm_current_size) // 2
            block_noise_shells = jnp.where(cutoff_mask, direct_triplet_shells[2], block_noise_shells)
            batch_img_power_shells = jnp.where(cutoff_mask, 0.0, batch_img_power_shells)
            if return_noise_split:
                block_a2_shells = jnp.where(cutoff_mask, direct_triplet_shells[1], block_a2_shells)
                block_xa_shells = jnp.where(cutoff_mask, direct_triplet_shells[0], block_xa_shells)
        noise_wsum = noise_wsum + block_noise_shells
        noise_img_power = noise_img_power + batch_img_power_shells
        if return_noise_split:
            noise_a2 = noise_a2 + block_a2_shells
            noise_xa = noise_xa + block_xa_shells
        if accumulate_scale_correction:
            scale_xa_per_image, scale_aa_per_image = _compute_scale_correction_terms_per_image(
                proj_for_noise,
                proj_abs2_for_norm,
                summed_masked_noise,
                ctf_probs,
                noise_variance_for_noise,
                batch_scale,
                scale_correction_pixel_mask,
            )
            scale_xa_per_image = jnp.where(valid_image_mask, scale_xa_per_image, 0.0)
            scale_aa_per_image = jnp.where(valid_image_mask, scale_aa_per_image, 0.0)
            noise_scale_xa = noise_scale_xa.at[group_ids].add(scale_xa_per_image)
            noise_scale_aa = noise_scale_aa.at[group_ids].add(scale_aa_per_image)
        noise_sigma2_offset = noise_sigma2_offset + noise_sumw_offset
        bucket_norm_correction = batch_img_power_per_image + _compute_norm_residual_per_image(
            proj_for_noise,
            proj_abs2_for_norm,
            summed_masked_noise,
            ctf_probs,
            noise_variance_for_noise,
        )
        bucket_norm_correction = jnp.where(valid_image_mask, bucket_norm_correction, 0.0).astype(jnp.float32)

    reconstruction_row_count = jnp.sum(reconstruction_rotation_mask & rotation_mask).astype(jnp.int32)
    if return_mstep_tensors:
        result = (
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
            reconstruction_row_count,
            summed,
            ctf_probs,
        )
        return _append_debug_outputs(
            result,
            debug_scores,
            debug_probs,
            shifted_score_split,
            ctf2_over_nv_score,
            proj_weighted,
            shifted_recon_split=shifted_recon_split,
            ctf2_over_nv_recon=ctf2_over_nv_recon,
            proj_for_noise=proj_for_noise,
            wavg_cutoff_triplet=debug_wavg_cutoff_triplet,
        )
    result = (
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
        reconstruction_row_count,
    )
    return _append_debug_outputs(
        result,
        debug_scores,
        debug_probs,
        shifted_score_split,
        ctf2_over_nv_score,
        proj_weighted,
        shifted_recon_split=shifted_recon_split,
        ctf2_over_nv_recon=ctf2_over_nv_recon,
        proj_for_noise=proj_for_noise,
        wavg_cutoff_triplet=debug_wavg_cutoff_triplet,
    )
