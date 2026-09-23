"""Build and execute one exact local-search iteration.

Construct image-specific pose neighborhoods, apply the batch memory budget,
dispatch the single-class or K-class kernel, and return named statistics to the
refinement controller. Dependencies are imported from their owning modules.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np


from recovar.em.classification.k_class import run_local_k_class_em
from recovar.em.helpers.batch_planning import _estimate_relion_em_batch_sizes
from recovar.em.helpers.types import LocalEMResult, NoiseStats, RelionStats
from recovar.em.local.local_em_engine import run_local_em_exact
from recovar.em.local.local_layout import _local_search_engine_rotation_block_size, build_local_hypothesis_layout
from recovar.em.sampling import build_local_search_grid_metadata
from recovar.em.sparse_pass2.resident_local_pass2 import (
    RESIDENT_LOCAL_SEARCH_ENV,
    compute_local_search_resident,
    resident_local_search_requested,
)

logger = logging.getLogger("recovar.em.local.local_search_iteration")


# Mirror iteration_loop's constant locally so the helper has a stable home.
EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS = 3_000_000
EXACT_LOCAL_XHALF_BATCH_GUARD_ENV = "RECOVAR_LOCAL_XHALF_BATCH_GUARD"




def _precompute_exact_local_fine_grid_enabled(healpix_order: int, symmetry: str = "C1") -> bool:
    """Return whether exact local search should materialize the fine grid once."""
    from recovar.em.sampling import rotation_grid_size

    return rotation_grid_size(int(healpix_order), symmetry) <= EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS


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
    class_assignments: np.ndarray | None = None
    class_posterior_sums: np.ndarray | None = None
    class_full_posterior_sums: np.ndarray | None = None
    best_pose_eulers_deg: np.ndarray | None = None


def _run_local_search_iteration(
    experiment_dataset,
    mean,
    noise_variance,
    prior_rotations,
    rotation_grid_rotations,
    healpix_order,
    sigma_rot,
    sigma_psi,
    translations,
    prior_translations,
    sigma_offset_angstrom,
    disc_type,
    image_batch_size,
    rotation_block_size,
    current_size,
    *,
    reconstruction_current_size=None,
    accumulate_noise=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    use_float64_scoring=False,
    use_float64_projections=False,
    do_gridding_correction=False,
    square_window=False,
    half_spectrum_scoring=False,
    relion_exact_score_translation=False,
    projection_relion_texture_interp=False,
    projection_relion_acc_double_floorf_quirk=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    image_corrections=None,
    scale_corrections=None,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    image_pre_shifts=None,
    mstep_relion_x_half=False,
    return_profile=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    adaptive_fraction=0.999,
    max_significants=-1,
    reconstruct_significant_only=True,
    translation_prior_reference_translations=None,
    debug_iteration=None,
    debug_pass_label=None,
    pass2_layout=None,
    return_best_pose_details=False,
    normalization_log_evidence=None,
    translation_prior_centers=None,
    rotation_grid_random_perturbation=0.0,
    rotation_grid_angular_sampling_deg=None,
    local_parent_oversampling_order: int = 0,
    class_log_priors=None,
    return_reconstruction_sample_indices=False,
    apply_max_significants_to_support=False,
    stats_use_reconstruction_probs=False,
    score_only=False,
    source_faithful_spectrum_norm=False,
    relion_translation_angle_scale=1.0,
    rotation_grid_mstep_rotations=None,
    generate_relion_mstep_rotations=False,
    symmetry: str = "C1",
    batch_size_planner=None,
) -> _LocalSearchIterationResult:
    """Run exact local search and return named halfset statistics and pose fields.

    ``debug_pass_label`` is diagnostic-only and forwarded verbatim to
    ``run_local_em_exact``: pass a distinct label per call site whenever a
    caller invokes this function more than once for the same image at the
    same ``current_size``/``debug_iteration`` (e.g. local search's pass-1
    "parent" probe vs. its pass-2 fine call), or the later call's
    ``RECOVAR_LOCAL_SCORE_DUMP_*`` output silently overwrites the earlier
    one at the same path.

    Optional fields are None when their corresponding return flags are disabled.
    Arrays retain the engine's layouts and identities; profile metadata is copied
    and augmented with this wrapper's timings.
    """
    requested_image_batch_size = int(image_batch_size)
    requested_rotation_block_size = int(rotation_block_size)
    rotation_block_size = _local_search_engine_rotation_block_size(rotation_block_size)
    # Keep the local-search hypothesis grid (rotations/translations/priors)
    # genuinely double precision end to end when either flag requests it;
    # default stays float32 to match RELION's accelerated-GPU precision.
    local_layout_dtype = np.float64 if (use_float64_scoring or use_float64_projections) else np.float32
    prior_rotations = np.asarray(prior_rotations, dtype=local_layout_dtype)
    if prior_rotations.ndim == 3:
        n_prior = prior_rotations.shape[0]
    elif prior_rotations.ndim == 2 and prior_rotations.shape[1] == 3:
        n_prior = prior_rotations.shape[0]
    else:
        raise ValueError(f"prior_rotations must have shape (n,3,3) or (n,3), got {prior_rotations.shape}")
    if prior_translations is None:
        prior_translations = np.zeros(
            (n_prior, np.asarray(translations).shape[1]),
            dtype=local_layout_dtype,
        )
    else:
        prior_translations = np.asarray(prior_translations, dtype=local_layout_dtype).reshape(
            -1,
            np.asarray(translations).shape[1],
        )

    if pass2_layout is None:
        metadata_t0 = time.time()
        # RELION local priors remain factorized in canonical direction/psi index
        # space even when the scored trial rotations have been perturbed.
        local_grid_metadata = build_local_search_grid_metadata(healpix_order, **({"symmetry": symmetry} if symmetry != "C1" else {}))
        metadata_build_time = time.time() - metadata_t0

        layout_t0 = time.time()
        layout_kwargs = {}
        if int(local_parent_oversampling_order) > 0:
            layout_kwargs["local_parent_oversampling_order"] = int(local_parent_oversampling_order)
        if rotation_grid_mstep_rotations is not None:
            layout_kwargs["rotation_grid_mstep_rotations"] = rotation_grid_mstep_rotations
        if bool(generate_relion_mstep_rotations):
            layout_kwargs["generate_relion_mstep_rotations"] = True
        local_layout = build_local_hypothesis_layout(
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
            rotation_log_prior=None,
            rotation_grid_random_perturbation=rotation_grid_random_perturbation,
            rotation_grid_angular_sampling_deg=rotation_grid_angular_sampling_deg,
            dtype=local_layout_dtype,
            **layout_kwargs,
        )
        selector_time = time.time() - layout_t0
    else:
        local_layout = pass2_layout
        metadata_build_time = 0.0
        selector_time = 0.0

    if class_log_priors is not None:
        if source_faithful_spectrum_norm:
            raise ValueError("RELION source-faithful spectrum normalization is fresh K=1-only")
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
    # The RELION x-half pass-2 path projects and backprojects only the active
    # Fourier window. Budget it against that window by default; keep the older
    # full-spectrum guard available as an emergency rollback knob for OOM
    # triage on smaller GPUs.
    local_batch_planning_current_size = current_size
    if relion_projector_half is not None and mstep_relion_x_half and not score_only:
        xhalf_guard_mode = os.environ.get(EXACT_LOCAL_XHALF_BATCH_GUARD_ENV, "windowed").strip().lower()
        if xhalf_guard_mode in {"", "full", "full_spectrum", "full-spectrum", "conservative"}:
            local_batch_planning_current_size = None
        elif xhalf_guard_mode in {"window", "windowed", "compact", "current_size", "current-size"}:
            local_batch_planning_current_size = current_size
        else:
            raise ValueError(
                f"{EXACT_LOCAL_XHALF_BATCH_GUARD_ENV} must be 'full' or 'windowed', got {xhalf_guard_mode!r}"
            )
    local_n_trans = max(1, int(np.asarray(local_layout.translation_grid).shape[0]))
    if batch_size_planner is None:
        local_batch_plan = _estimate_relion_em_batch_sizes(
            requested_image_batch_size=image_batch_size,
            requested_rotation_block_size=rotation_block_size,
            n_rot=max(1, local_rotation_count),
            n_trans=local_n_trans,
            image_shape=experiment_dataset.image_shape,
            volume_shape=experiment_dataset.volume_shape,
            padding_factor=max(int(projection_padding_factor), int(reconstruction_padding_factor), 1),
            n_classes=local_kernel_classes,
            current_size=local_batch_planning_current_size,
            use_float64_scoring=use_float64_scoring,
        )
        planned_image_batch_size = local_batch_plan.image_batch_size
        planned_rotation_block_size = local_batch_plan.rotation_block_size
    else:
        # The enclosing RELION loop may have qualified a compact K=1
        # Projector/BPref lifetime and bound that policy into its planner.
        # Reusing that callable here keeps the actual local rotation count
        # without silently falling back to the obsolete full-cube estimate.
        # Never grow beyond the already-approved outer local-search sizes.
        planned_image_batch_size, planned_rotation_block_size = batch_size_planner(
            max(1, local_rotation_count),
            local_n_trans,
            classes=local_kernel_classes,
            image_shape_for_batch=experiment_dataset.image_shape,
            current_size_for_batch=local_batch_planning_current_size,
        )
        planned_image_batch_size = min(
            image_batch_size,
            max(1, int(planned_image_batch_size)),
        )
        planned_rotation_block_size = min(
            rotation_block_size,
            max(1, int(planned_rotation_block_size)),
        )
        local_batch_plan = None
    if (
        planned_image_batch_size != image_batch_size
        or planned_rotation_block_size != rotation_block_size
    ) and local_batch_plan is not None:
        logger.info(
            "Local search memory batch sizing: requested image_batch_size=%d rotation_block_size=%d; "
            "using image_batch_size=%d rotation_block_size=%d "
            "(local_rot_max=%d n_trans=%d K=%d effective_kernel_K=%d, score_pixels=%d, "
            "translation_tile=%.2f/%.2f GB, "
            "projection_tile=%.2f/%.2f GB, persistent_est=%.2f GB, usable_est=%.2f GB, "
            "gpu_used_est=%.2f GB)",
            requested_image_batch_size,
            requested_rotation_block_size,
            planned_image_batch_size,
            planned_rotation_block_size,
            local_rotation_count,
            int(np.asarray(local_layout.translation_grid).shape[0]),
            local_n_classes,
            local_kernel_classes,
            local_batch_plan.score_pixel_count,
            local_batch_plan.translation_tile_gb,
            local_batch_plan.translation_tile_budget_gb,
            local_batch_plan.projection_block_gb,
            local_batch_plan.projection_budget_gb,
            local_batch_plan.persistent_estimate_gb,
            local_batch_plan.usable_estimate_gb,
            local_batch_plan.gpu_used_estimate_gb,
        )
    elif (
        planned_image_batch_size != image_batch_size
        or planned_rotation_block_size != rotation_block_size
    ):
        logger.info(
            "Local search caller-qualified batch sizing: requested "
            "image_batch_size=%d rotation_block_size=%d; using "
            "image_batch_size=%d rotation_block_size=%d "
            "(local_rot_max=%d n_trans=%d)",
            requested_image_batch_size,
            requested_rotation_block_size,
            planned_image_batch_size,
            planned_rotation_block_size,
            local_rotation_count,
            local_n_trans,
        )
    image_batch_size = planned_image_batch_size
    rotation_block_size = planned_rotation_block_size

    if class_log_priors is not None:
        if resident_local_search_requested():
            raise NotImplementedError(
                f"{RESIDENT_LOCAL_SEARCH_ENV}=1 selects the device-resident local pass 2, "
                "which is K=1 only; K-class local search keeps the exact local engine"
            )
        if return_reconstruction_sample_indices:
            raise NotImplementedError("K-class local search does not return reconstruction sample indices")
        if score_only:
            raise NotImplementedError("K-class local search does not support score_only")
        if return_profile:
            raise NotImplementedError("K-class local search does not yet emit local profile summaries")
        if disable_adjoint_y or disable_adjoint_ctf:
            raise NotImplementedError("K-class local search does not support adjoint ablation flags")
        if normalization_log_evidence is not None:
            raise NotImplementedError("K-class local search does not support external evidence normalization")
        k_class_result = run_local_k_class_em(
            experiment_dataset,
            mean,
            noise_variance,
            local_layout,
            disc_type,
            class_log_priors=class_log_priors,
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            current_size=current_size,
            reconstruction_current_size=reconstruction_current_size,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            half_spectrum_scoring=half_spectrum_scoring,
            relion_exact_score_translation=relion_exact_score_translation,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            use_float64_scoring=use_float64_scoring,
            use_float64_normalization=True,
            use_float64_projections=use_float64_projections,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            group_ids=group_ids,
            scale_correction_group_count=scale_correction_group_count,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior,
            image_pre_shifts=image_pre_shifts,
            mstep_relion_x_half=mstep_relion_x_half,
            reconstruct_significant_only=reconstruct_significant_only,
            adaptive_fraction=adaptive_fraction,
            max_significants=-1,
            stats_use_reconstruction_probs=stats_use_reconstruction_probs,
            class_posterior_sums_from_noise=bool(reconstruct_significant_only and accumulate_noise),
            debug_iteration=debug_iteration,
            translation_prior_centers=translation_prior_centers,
            **({"symmetry_label": symmetry} if symmetry != "C1" else {}),
        )
        use_noise_class_sums = bool(reconstruct_significant_only and accumulate_noise)
        class_mstep_posterior_sums = (
            getattr(k_class_result, "class_mstep_posterior_sums", None) if use_noise_class_sums else None
        )
        if class_mstep_posterior_sums is None:
            class_mstep_posterior_sums = k_class_result.class_posterior_sums
        class_details = (
            np.asarray(k_class_result.class_assignments, dtype=np.int32),
            np.asarray(class_mstep_posterior_sums, dtype=np.float64),
            np.asarray(k_class_result.class_posterior_sums, dtype=np.float64),
        )
        engine_outputs = LocalEMResult(
            Ft_y=k_class_result.Ft_y,
            Ft_ctf=k_class_result.Ft_ctf,
            hard_assignments=np.asarray(k_class_result.pose_assignments, dtype=np.int32),
            stats=k_class_result.stats,
            best_pose_rotations=k_class_result.best_pose_rotations if return_best_pose_details else None,
            best_pose_translations=k_class_result.best_pose_translations if return_best_pose_details else None,
            best_pose_eulers_deg=getattr(k_class_result, "best_pose_eulers_deg", None)
            if return_best_pose_details
            else None,
            noise_stats=k_class_result.aggregate_noise_stats if accumulate_noise else None,
        )
    elif (
        resident_local_search_requested()
        and not score_only
        and current_size is not None
        and int(current_size) < int(experiment_dataset.image_shape[0])
    ):
        # The device-resident local pass 2 (T12). Only the fine pass is routed
        # here: the pass-1 parent probe selects pass 2's candidate set with
        # RELION's ``maximum_significants`` cap, which the segmented float32
        # posterior does not implement, so routing it would change the support
        # rather than only its layout. The boundary is logged, not silent.
        class_details = None
        logger.info(
            "%s=1: running the device-resident local fine pass 2 "
            "(image_batch_size=%d and rotation_block_size=%d are unused by this path; "
            "its capacity plan is sized from the projection byte budget)",
            RESIDENT_LOCAL_SEARCH_ENV,
            image_batch_size,
            rotation_block_size,
        )
        engine_outputs = compute_local_search_resident(
            experiment_dataset,
            mean,
            noise_variance,
            local_layout,
            disc_type,
            current_size=current_size,
            reconstruction_current_size=reconstruction_current_size,
            accumulate_noise=accumulate_noise,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            half_spectrum_scoring=half_spectrum_scoring,
            relion_exact_score_translation=relion_exact_score_translation,
            projection_relion_texture_interp=projection_relion_texture_interp,
            projection_relion_acc_double_floorf_quirk=projection_relion_acc_double_floorf_quirk,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            use_float64_scoring=use_float64_scoring,
            use_float64_projections=use_float64_projections,
            square_window=square_window,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            group_ids=group_ids,
            scale_correction_group_count=scale_correction_group_count,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior,
            image_pre_shifts=image_pre_shifts,
            mstep_relion_x_half=mstep_relion_x_half,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            reconstruct_significant_only=reconstruct_significant_only,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants if apply_max_significants_to_support else -1,
            return_best_pose_details=return_best_pose_details,
            return_reconstruction_sample_indices=return_reconstruction_sample_indices,
            return_profile=return_profile,
            stats_use_reconstruction_probs=stats_use_reconstruction_probs,
            translation_prior_centers=translation_prior_centers,
            normalization_log_evidence=normalization_log_evidence,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            relion_translation_angle_scale=relion_translation_angle_scale,
            class_log_priors=None,
            score_only=score_only,
        )
    else:
        class_details = None
        if resident_local_search_requested():
            if score_only:
                logger.info(
                    "%s=1: the pass-1 parent probe keeps the exact local engine "
                    "(its RELION maximum_significants cap is outside the segmented "
                    "posterior's contract, and changing it would change pass 2's support)",
                    RESIDENT_LOCAL_SEARCH_ENV,
                )
            else:
                logger.info(
                    "%s=1: this pass scores at current_size=%s, the full image box "
                    "(RELION's final all-data iteration), where the exact local engine "
                    "scores the whole centred half and RELION's radial support does "
                    "not; the choice between them is a scientific decision, so this "
                    "iteration keeps the exact local engine",
                    RESIDENT_LOCAL_SEARCH_ENV,
                    current_size,
                )
        engine_outputs = run_local_em_exact(
            experiment_dataset,
            mean,
            noise_variance,
            local_layout,
            disc_type,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            current_size=current_size,
            reconstruction_current_size=reconstruction_current_size,
            accumulate_noise=accumulate_noise,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            half_spectrum_scoring=half_spectrum_scoring,
            relion_exact_score_translation=relion_exact_score_translation,
            projection_relion_texture_interp=projection_relion_texture_interp,
            projection_relion_acc_double_floorf_quirk=projection_relion_acc_double_floorf_quirk,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            use_float64_scoring=use_float64_scoring,
            # Keep posterior/log-Z reductions in float64 even when score/projection
            # tensors stay float32 for throughput. The dtype policy default is
            # float64 normalization, and the significance threshold is sensitive
            # to small log-sum-exp changes near RELION's 0.999 cutoff.
            use_float64_normalization=True,
            use_float64_projections=use_float64_projections,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            group_ids=group_ids,
            scale_correction_group_count=scale_correction_group_count,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior,
            image_pre_shifts=image_pre_shifts,
            mstep_relion_x_half=mstep_relion_x_half,
            return_profile=return_profile,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            reconstruct_significant_only=reconstruct_significant_only,
            adaptive_fraction=adaptive_fraction,
            # RELION's maximum_significants cap is used to define the coarse
            # adaptive support. In pass 2, the reconstruction threshold is
            # governed by adaptive_fraction only; do not reapply the cap there.
            max_significants=max_significants if apply_max_significants_to_support else -1,
            debug_iteration=debug_iteration,
            debug_pass_label=debug_pass_label,
            return_best_pose_details=return_best_pose_details,
            normalization_log_evidence=normalization_log_evidence,
            translation_prior_centers=translation_prior_centers,
            return_reconstruction_sample_indices=return_reconstruction_sample_indices,
            stats_use_reconstruction_probs=stats_use_reconstruction_probs,
            score_only=score_only,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            relion_translation_angle_scale=relion_translation_angle_scale,
            **({"symmetry_label": symmetry} if symmetry != "C1" else {}),
        )

    if class_details is None:
        class_assignments = class_posterior_sums = class_full_posterior_sums = None
    else:
        class_assignments, class_posterior_sums, class_full_posterior_sums = class_details
    result = _LocalSearchIterationResult(
        Ft_y=engine_outputs.Ft_y,
        Ft_ctf=engine_outputs.Ft_ctf,
        hard_assignment=engine_outputs.hard_assignments,
        relion_stats=engine_outputs.stats,
        noise_stats=engine_outputs.noise_stats,
        profile_summary=engine_outputs.profile if return_profile else None,
        best_pose_rotations=engine_outputs.best_pose_rotations,
        best_pose_translations=engine_outputs.best_pose_translations,
        best_pose_eulers_deg=engine_outputs.best_pose_eulers_deg,
        class_assignments=class_assignments,
        class_posterior_sums=class_posterior_sums,
        class_full_posterior_sums=class_full_posterior_sums,
    )

    if return_profile and result.profile_summary is not None:
        result.profile_summary = dict(result.profile_summary)
        result.profile_summary["metadata_build_time_s"] = np.float64(metadata_build_time)
        result.profile_summary["selector_time_s"] = np.float64(selector_time)
        result.profile_summary["translation_prior_time_s"] = np.float64(0.0)

    return result
