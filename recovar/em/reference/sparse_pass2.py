"""Independent per-image dense calculation for sparse pass-2 comparisons and fallback."""

import logging

import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.oversampling import _map_translation_log_prior_to_fine_grid
from recovar.em.helpers.types import SparsePass2Output, make_noise_stats, make_relion_stats

# Preserve the category consumed by existing run-log collectors.
logger = logging.getLogger("recovar.em.helpers.oversampling")


def _compute_pass2_stats_sparse_perimage_reference(
    experiment_dataset,
    volume,
    mean_variance,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    oversampling_order=1,
    current_size=None,
    translation_step=None,
    *,
    rotation_log_prior=None,
    score_with_masked_images=False,
    return_stats=False,
    translation_log_prior=None,
    accumulate_noise=False,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    translation_prior_centers=None,
    use_float64_scoring=False,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation=0.0,
    normalization_log_z=None,
    normalization_other_score_log_z=None,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    relion_half_volume_mstep=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
):
    """Per-image reference implementation for sparse pass-2.

    Runs ``run_em`` once per image with ``image_batch_size=1`` and the
    image's oversampled rotation/translation grid.  This is correct but
    causes a JIT recompile per image when the rotation count varies, so
    it serves numerical comparisons and the admitted full-grid fallback in
    :func:`recovar.em.sparse_pass2.dispatch.compute_pass2_stats_sparse`.
    Ordinary sparse passes use the bucketed implementation.
    """
    from recovar.em.dense.em_engine import run_em
    from recovar.em.sampling import (
        get_oversampled_rotation_grid_from_samples,
        get_oversampled_translation_grid,
        rotation_grid_size,
    )
    if normalization_log_z is not None:
        raise NotImplementedError(
            "normalization_log_z is only implemented for the bucketed sparse pass-2 path",
        )
    if normalization_other_score_log_z is not None:
        raise NotImplementedError(
            "normalization_other_score_log_z is only implemented for the bucketed sparse pass-2 path",
        )

    n_images = experiment_dataset.n_units
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = rotation_grid_size(nside_level)
    recon_vol_size = experiment_dataset.volume_size * reconstruction_padding_factor**3
    Ft_y_total = jnp.zeros(recon_vol_size, dtype=experiment_dataset.dtype)
    Ft_ctf_total = jnp.zeros(recon_vol_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    score_dtype = np.float64 if use_float64_scoring else np.float32
    best_rotations = np.empty((n_images, 3, 3), dtype=score_dtype)
    best_rotation_indices = np.empty(n_images, dtype=np.int64)

    log_evidence = None
    best_log_score = None
    max_posterior = None
    rotation_posterior_sums = None
    if return_stats:
        # K-class assignment depends on small inter-class score deltas after
        # adding a large image-power offset. Keep these in float64 like dense
        # run_em.
        log_evidence = np.empty(n_images, dtype=np.float64)
        best_log_score = np.empty(n_images, dtype=np.float64)
        max_posterior = np.empty(n_images, dtype=score_dtype)
        rotation_posterior_sums = np.zeros(n_coarse_rot, dtype=np.float64)

    # Noise accumulators (additive across per-image calls)
    noise_wsum_total = None
    noise_img_power_total = None
    noise_sumw_total = 0.0
    noise_sigma2_offset_total = 0.0
    if accumulate_noise:
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        noise_wsum_total = np.zeros(n_shells, dtype=np.float64)
        noise_img_power_total = np.zeros(n_shells, dtype=np.float64)

    translations_np = np.asarray(translations, dtype=score_dtype)
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
    fine_translations = np.asarray(fine_translations, dtype=score_dtype)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_trans = fine_translations.shape[0]
    fine_translation_prior = _map_translation_log_prior_to_fine_grid(
        translation_log_prior,
        fine_translation_parent,
    )
    local_rot_counts = []
    valid_candidate_counts = []

    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if sig_samples is None:
            coarse_rot = np.arange(n_coarse_rot, dtype=np.int32)
            coarse_trans = None
            unique_rot = coarse_rot
            use_full_candidate_mask = True
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
            dtype=score_dtype,
        )
        oversampled_rots = np.asarray(oversampled_rots, dtype=score_dtype)
        parent_map = np.asarray(parent_map, dtype=np.int32)
        oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int64)
        local_rotation_log_prior = None
        if rotation_log_prior is not None:
            rotation_log_prior = np.asarray(rotation_log_prior, dtype=score_dtype)
            local_rotation_log_prior = rotation_log_prior[unique_rot][parent_map]

        if use_full_candidate_mask:
            candidate_mask = np.ones(
                (oversampled_rots.shape[0], n_fine_trans),
                dtype=bool,
            )
        else:
            sig_trans_by_rot = {
                int(rot_idx): set(coarse_trans[coarse_rot == rot_idx].tolist()) for rot_idx in unique_rot
            }
            candidate_mask = np.zeros(
                (oversampled_rots.shape[0], n_fine_trans),
                dtype=bool,
            )
            for parent_local_idx, coarse_rot_idx in enumerate(unique_rot):
                row_mask = parent_map == parent_local_idx
                valid_coarse_trans = sig_trans_by_rot[int(coarse_rot_idx)]
                col_mask = np.isin(fine_translation_parent, list(valid_coarse_trans))
                candidate_mask[row_mask, :] = col_mask[None, :]

        if not np.any(candidate_mask):
            raise ValueError(f"Image {image_idx} has no valid sparse pass-2 candidates after oversampling")

        local_rot_counts.append(int(oversampled_rots.shape[0]))
        valid_candidate_counts.append(int(candidate_mask.sum()))

        run_em_outputs = run_em(
            experiment_dataset,
            volume,
            mean_variance,
            noise_variance,
            oversampled_rots,
            fine_translations,
            disc_type,
            image_batch_size=1,
            rotation_block_size=min(5000, max(1, oversampled_rots.shape[0])),
            current_size=current_size,
            rotation_log_prior=local_rotation_log_prior,
            translation_log_prior=(
                None
                if fine_translation_prior is None
                else np.asarray(fine_translation_prior[image_idx : image_idx + 1], dtype=np.float32)
                if np.asarray(fine_translation_prior).ndim == 2
                else fine_translation_prior
            ),
            image_indices=np.array([image_idx], dtype=np.int32),
            rotation_translation_mask=candidate_mask,
            score_with_masked_images=score_with_masked_images,
            return_stats=return_stats,
            accumulate_noise=accumulate_noise,
            half_spectrum_scoring=half_spectrum_scoring,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            translation_prior_centers=(
                None
                if translation_prior_centers is None
                else np.asarray(translation_prior_centers[image_idx : image_idx + 1], dtype=np.float32)
                if np.asarray(translation_prior_centers).ndim == 2
                else np.asarray(translation_prior_centers, dtype=np.float32)
            ),
            use_float64_scoring=use_float64_scoring,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            relion_half_volume_mstep=relion_half_volume_mstep,
            relion_firstiter_score_mode=relion_firstiter_score_mode,
            relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
        )

        ha_i = run_em_outputs.hard_assignments
        Ft_y_i = run_em_outputs.Ft_y
        Ft_ctf_i = run_em_outputs.Ft_ctf
        stats_i = run_em_outputs.stats
        noise_stats_i = run_em_outputs.noise_stats

        if return_stats:
            log_evidence[image_idx] = float(np.asarray(stats_i.log_evidence_per_image)[0])
            best_log_score[image_idx] = float(np.asarray(stats_i.best_log_score_per_image)[0])
            max_posterior[image_idx] = float(np.asarray(stats_i.max_posterior_per_image)[0])
            np.add.at(
                rotation_posterior_sums,
                unique_rot[parent_map],
                np.asarray(stats_i.rotation_posterior_sums, dtype=np.float64),
            )

        if accumulate_noise and noise_stats_i is not None:
            noise_wsum_total += np.asarray(noise_stats_i.wsum_sigma2_noise, dtype=np.float64)
            noise_img_power_total += np.asarray(noise_stats_i.wsum_img_power, dtype=np.float64)
            noise_sigma2_offset_total += float(getattr(noise_stats_i, "wsum_sigma2_offset", 0.0))
            noise_sumw_total += noise_stats_i.sumw

        Ft_y_total = Ft_y_total + Ft_y_i
        Ft_ctf_total = Ft_ctf_total + Ft_ctf_i

        best_idx = int(np.asarray(ha_i)[0])
        rot_idx = best_idx // n_fine_trans
        hard_assignment[image_idx] = best_idx
        best_rotations[image_idx] = oversampled_rots[rot_idx]
        best_rotation_indices[image_idx] = oversampled_rot_indices[rot_idx]

    logger.info(
        "Sparse pass 2: median local rotations=%d, mean local rotations=%.1f, median valid candidates/image=%d",
        int(np.median(local_rot_counts)) if local_rot_counts else 0,
        float(np.mean(local_rot_counts)) if local_rot_counts else 0.0,
        int(np.median(valid_candidate_counts)) if valid_candidate_counts else 0,
    )

    best_translations = fine_translations[hard_assignment % n_fine_trans]

    merged_noise_stats = None
    if accumulate_noise:
        merged_noise_stats = make_noise_stats(
            wsum_sigma2_noise=noise_wsum_total,
            wsum_img_power=noise_img_power_total,
            wsum_sigma2_offset=noise_sigma2_offset_total,
            sumw=noise_sumw_total,
        )

    relion_stats = None
    if return_stats:
        relion_stats = make_relion_stats(
            log_evidence_per_image=log_evidence,
            best_log_score_per_image=best_log_score,
            max_posterior_per_image=max_posterior,
            rotation_posterior_sums=rotation_posterior_sums,
        )
    return SparsePass2Output(
        Ft_y_total,
        Ft_ctf_total,
        hard_assignment,
        best_rotations,
        best_translations,
        best_rotation_indices,
        relion_stats=relion_stats,
        noise_stats=merged_noise_stats if accumulate_noise else None,
    )

