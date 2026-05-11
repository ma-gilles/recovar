"""Exact local-search iteration body.

Encapsulates ``_run_local_search_iteration`` and its supporting helpers
(``_LocalSearchIterationResult``, ``_unpack_local_search_engine_outputs``,
``_pack_local_search_iteration_result``, ``_precompute_exact_local_fine_grid_enabled``)
out of ``iteration_loop.py``. The main loop in iteration_loop.py imports them back
under their underscored names so existing test monkeypatches at
``iteration_loop._run_local_search_iteration`` continue to bind correctly.

Patched symbols that live in iteration_loop's namespace
(``build_local_hypothesis_layout``, ``run_local_em_exact``,
``_estimate_relion_em_batch_sizes``) are accessed via the iteration_loop module
reference (lazy import inside the function) so ``monkeypatch.setattr(iteration_loop, ...)``
calls in the existing test suite remain effective.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from recovar.em.dense_single_volume.helpers.local_search import (
    _local_search_engine_rotation_block_size,
)
from recovar.em.dense_single_volume.helpers.types import NoiseStats, RelionStats
from recovar.em.dense_single_volume.k_class import run_local_k_class_em
from recovar.em.sampling import build_local_search_grid_metadata

logger = logging.getLogger(__name__)


# Mirror iteration_loop's constant locally so the helper has a stable home.
EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS = 3_000_000


def _precompute_exact_local_fine_grid_enabled(healpix_order: int) -> bool:
    """Return whether exact local search should materialize the fine grid once."""
    from recovar.em.sampling import rotation_grid_size

    return rotation_grid_size(int(healpix_order)) <= EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS


@dataclass
class _LocalSearchIterationResult:
    Ft_y: object
    Ft_ctf: object
    hard_assignment: object
    relion_stats: RelionStats
    noise_stats: NoiseStats | None = None
    profile_summary: dict | None = None
    best_pose_rotations: object | None = None
    best_pose_translations: object | None = None
    best_pose_rotation_ids: object | None = None
    class_assignments: np.ndarray | None = None
    class_posterior_sums: np.ndarray | None = None


def _unpack_local_search_engine_outputs(
    engine_outputs,
    *,
    accumulate_noise: bool,
    return_profile: bool,
    return_best_pose_details: bool,
    class_details=None,
) -> _LocalSearchIterationResult:
    cursor = 0
    Ft_y, Ft_ctf, hard_assignment = engine_outputs[cursor : cursor + 3]
    cursor += 3
    best_pose_rotations = best_pose_translations = best_pose_rotation_ids = None
    if return_best_pose_details:
        best_pose_rotations, best_pose_translations, best_pose_rotation_ids = engine_outputs[cursor : cursor + 3]
        cursor += 3
    relion_stats = engine_outputs[cursor]
    cursor += 1
    noise_stats = engine_outputs[cursor] if accumulate_noise else None
    cursor += int(accumulate_noise)
    profile_summary = engine_outputs[cursor] if return_profile else None
    if class_details is None:
        class_assignments = class_posterior_sums = None
    else:
        class_assignments, class_posterior_sums = class_details
    return _LocalSearchIterationResult(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        hard_assignment=hard_assignment,
        relion_stats=relion_stats,
        noise_stats=noise_stats,
        profile_summary=profile_summary,
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
        class_assignments=class_assignments,
        class_posterior_sums=class_posterior_sums,
    )


def _pack_local_search_iteration_result(
    result: _LocalSearchIterationResult,
    *,
    accumulate_noise: bool,
    return_profile: bool,
    return_best_pose_details: bool,
    return_class_details: bool,
):
    output = [result.Ft_y, result.Ft_ctf, result.hard_assignment]
    if return_best_pose_details:
        output.extend(
            [
                result.best_pose_rotations,
                result.best_pose_translations,
                result.best_pose_rotation_ids,
            ],
        )
    output.append(result.relion_stats)
    if accumulate_noise:
        output.append(result.noise_stats)
    if return_profile:
        output.append(result.profile_summary)
    if return_class_details:
        if result.class_assignments is None or result.class_posterior_sums is None:
            raise ValueError("return_class_details=True requires class_log_priors")
        output.extend([result.class_assignments, result.class_posterior_sums])
    return tuple(output)


def _run_local_search_iteration(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    prior_rotations,
    rotation_grid_rotations,
    rotation_grid_eulers,
    healpix_order,
    sigma_rot,
    sigma_psi,
    translations,
    prior_translations,
    sigma_offset_angstrom,
    offset_range_pixels,
    disc_type,
    image_batch_size,
    rotation_block_size,
    current_size,
    *,
    accumulate_noise=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    use_float64_scoring=False,
    use_float64_projections=False,
    do_gridding_correction=False,
    square_window=False,
    half_spectrum_scoring=False,
    projection_relion_texture_interp=False,
    projection_force_jax=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    mstep_subtract_ctf_projection=False,
    mstep_relion_x_half=False,
    return_half_volume_accumulators=False,
    score_with_masked_images=True,
    return_profile=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    adaptive_fraction=0.999,
    max_significants=-1,
    reconstruct_significant_only=True,
    translation_prior_reference_translations=None,
    debug_iteration=None,
    pass2_layout=None,
    return_best_pose_details=False,
    normalization_log_z=None,
    translation_prior_centers=None,
    rotation_grid_random_perturbation=0.0,
    rotation_grid_angular_sampling_deg=None,
    class_log_priors=None,
    return_class_details=False,
):
    """Run exact local search over image-specific rotation neighborhoods."""
    # Indirection through the iteration_loop module so test monkeypatches that
    # target ``iteration_loop.build_local_hypothesis_layout``,
    # ``iteration_loop.run_local_em_exact``, etc. continue to win at the call
    # site even though this function lives in a sibling module.
    from recovar.em.dense_single_volume import iteration_loop as _il

    requested_image_batch_size = int(image_batch_size)
    requested_rotation_block_size = int(rotation_block_size)
    rotation_block_size = _local_search_engine_rotation_block_size(rotation_block_size)
    prior_rotations = np.asarray(prior_rotations, dtype=np.float32)
    if prior_rotations.ndim == 3:
        n_prior = prior_rotations.shape[0]
    elif prior_rotations.ndim == 2 and prior_rotations.shape[1] == 3:
        n_prior = prior_rotations.shape[0]
    else:
        raise ValueError(f"prior_rotations must have shape (n,3,3) or (n,3), got {prior_rotations.shape}")
    if prior_translations is None:
        prior_translations = np.zeros(
            (n_prior, np.asarray(translations).shape[1]),
            dtype=np.float32,
        )
    else:
        prior_translations = np.asarray(prior_translations, dtype=np.float32).reshape(
            -1,
            np.asarray(translations).shape[1],
        )

    if pass2_layout is None:
        metadata_t0 = time.time()
        # RELION local priors remain factorized in canonical direction/psi index
        # space even when the scored trial rotations have been perturbed.
        local_grid_metadata = build_local_search_grid_metadata(healpix_order)
        metadata_build_time = time.time() - metadata_t0

        layout_t0 = time.time()
        local_layout = _il.build_local_hypothesis_layout(
            prior_rotations,
            rotation_grid_rotations,
            sigma_rot,
            sigma_psi,
            healpix_order,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            # Match the grouped RELION-mode path: local translation priors use the
            # learned/model sigma, not the older range/3 override.
            None,
            experiment_dataset.voxel_size,
            grid_metadata=local_grid_metadata,
            translation_prior_reference_translations=translation_prior_reference_translations,
            rotation_grid_random_perturbation=rotation_grid_random_perturbation,
            rotation_grid_angular_sampling_deg=rotation_grid_angular_sampling_deg,
        )
        selector_time = time.time() - layout_t0
    else:
        local_layout = pass2_layout
        metadata_build_time = 0.0
        selector_time = 0.0

    if class_log_priors is not None:
        local_n_classes = int(np.asarray(class_log_priors).size)
    else:
        local_n_classes = 1
    # Exact local K-class invokes the per-class local kernels sequentially
    # (probe pass per class, then M-step per class), so K is not a simultaneous
    # tensor dimension for the shifted-image/projection tiles here.
    local_kernel_classes = 1
    local_rotation_count = (
        int(np.max(np.asarray(local_layout.rotation_counts, dtype=np.int64)))
        if int(np.asarray(local_layout.rotation_counts).size)
        else 1
    )
    local_batch_plan = _il._estimate_relion_em_batch_sizes(
        requested_image_batch_size=image_batch_size,
        requested_rotation_block_size=rotation_block_size,
        n_rot=max(1, local_rotation_count),
        n_trans=max(1, int(np.asarray(local_layout.translation_grid).shape[0])),
        image_shape=experiment_dataset.image_shape,
        volume_shape=experiment_dataset.volume_shape,
        padding_factor=max(int(projection_padding_factor), int(reconstruction_padding_factor), 1),
        n_classes=local_kernel_classes,
    )
    if (
        local_batch_plan.image_batch_size != image_batch_size
        or local_batch_plan.rotation_block_size != rotation_block_size
    ):
        logger.info(
            "Local search memory batch sizing: requested image_batch_size=%d rotation_block_size=%d; "
            "using image_batch_size=%d rotation_block_size=%d "
            "(local_rot_max=%d n_trans=%d K=%d effective_kernel_K=%d, translation_tile=%.2f/%.2f GB, "
            "projection_tile=%.2f/%.2f GB, persistent_est=%.2f GB, usable_est=%.2f GB, "
            "gpu_used_est=%.2f GB)",
            requested_image_batch_size,
            requested_rotation_block_size,
            local_batch_plan.image_batch_size,
            local_batch_plan.rotation_block_size,
            local_rotation_count,
            int(np.asarray(local_layout.translation_grid).shape[0]),
            local_n_classes,
            local_kernel_classes,
            local_batch_plan.translation_tile_gb,
            local_batch_plan.translation_tile_budget_gb,
            local_batch_plan.projection_block_gb,
            local_batch_plan.projection_budget_gb,
            local_batch_plan.persistent_estimate_gb,
            local_batch_plan.usable_estimate_gb,
            local_batch_plan.gpu_used_estimate_gb,
        )
    image_batch_size = local_batch_plan.image_batch_size
    rotation_block_size = local_batch_plan.rotation_block_size

    if class_log_priors is not None:
        if return_profile:
            raise NotImplementedError("K-class local search does not yet emit local profile summaries")
        if disable_adjoint_y or disable_adjoint_ctf:
            raise NotImplementedError("K-class local search does not support adjoint ablation flags")
        if normalization_log_z is not None:
            raise NotImplementedError("K-class local search requires evidence-space normalization, not pass-2 log_z")
        if projection_force_jax:
            raise NotImplementedError("K-class local search does not yet plumb projection_force_jax")
        k_class_result = run_local_k_class_em(
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            local_layout,
            disc_type,
            class_log_priors=class_log_priors,
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            current_size=current_size,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            score_with_masked_images=score_with_masked_images,
            half_spectrum_scoring=half_spectrum_scoring,
            use_float64_scoring=use_float64_scoring,
            use_float64_normalization=use_float64_scoring,
            use_float64_projections=use_float64_projections,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            reconstruct_significant_only=reconstruct_significant_only,
            adaptive_fraction=adaptive_fraction,
            max_significants=-1,
            debug_iteration=debug_iteration,
            translation_prior_centers=translation_prior_centers,
        )
        class_details = (
            np.asarray(k_class_result.class_assignments, dtype=np.int32),
            np.asarray(k_class_result.class_posterior_sums, dtype=np.float64),
        )
        engine_outputs = (
            k_class_result.Ft_y,
            k_class_result.Ft_ctf,
            np.asarray(k_class_result.pose_assignments, dtype=np.int32),
            k_class_result.best_pose_rotations,
            k_class_result.best_pose_translations,
            k_class_result.best_pose_rotation_ids,
            k_class_result.stats,
            k_class_result.aggregate_noise_stats,
        )
    else:
        class_details = None
        engine_outputs = _il.run_local_em_exact(
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            local_layout,
            disc_type,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            current_size=current_size,
            accumulate_noise=accumulate_noise,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            score_with_masked_images=score_with_masked_images,
            half_spectrum_scoring=half_spectrum_scoring,
            projection_relion_texture_interp=projection_relion_texture_interp,
            projection_force_jax=projection_force_jax,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            use_float64_scoring=use_float64_scoring,
            use_float64_normalization=use_float64_scoring,
            use_float64_projections=use_float64_projections,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            mstep_subtract_ctf_projection=mstep_subtract_ctf_projection,
            mstep_relion_x_half=mstep_relion_x_half,
            return_half_volume_accumulators=return_half_volume_accumulators,
            return_profile=return_profile,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            reconstruct_significant_only=reconstruct_significant_only,
            adaptive_fraction=adaptive_fraction,
            # RELION's maximum_significants cap is used to define the coarse pass-1
            # adaptive support. In pass 2, the reconstruction threshold is governed
            # by adaptive_fraction only; do not reapply the cap here.
            max_significants=-1,
            debug_iteration=debug_iteration,
            return_best_pose_details=return_best_pose_details,
            normalization_log_z=normalization_log_z,
            translation_prior_centers=translation_prior_centers,
        )

    result = _unpack_local_search_engine_outputs(
        engine_outputs,
        accumulate_noise=accumulate_noise,
        return_profile=return_profile,
        return_best_pose_details=return_best_pose_details,
        class_details=class_details,
    )

    if return_profile and result.profile_summary is not None:
        result.profile_summary = dict(result.profile_summary)
        result.profile_summary["metadata_build_time_s"] = np.float64(metadata_build_time)
        result.profile_summary["selector_time_s"] = np.float64(selector_time)
        result.profile_summary["translation_prior_time_s"] = np.float64(0.0)

    return _pack_local_search_iteration_result(
        result,
        accumulate_noise=accumulate_noise,
        return_profile=return_profile,
        return_best_pose_details=return_best_pose_details,
        return_class_details=return_class_details,
    )
