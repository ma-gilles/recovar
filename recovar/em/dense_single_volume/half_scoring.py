"""Dispatch dense and local scoring for one refinement half-set.

The iteration controller owns scheduling, state transitions, result-container
lifetime and device offloading. These adapters choose existing engine routes,
prepare per-half arguments and populate caller-owned result slots. Diagnostic
BPref scopes surround the same calls as before; no array copies are introduced
by the ownership boundary.
"""

import logging
import os

import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject as _cuda_backproject_diagnostics
from recovar import utils
from recovar.em.dense_single_volume import parity_dump as _parity_dump
from recovar.em.dense_single_volume.batch_planning import _plan_kclass_adaptive_grid_batch_sizes
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.dense_single_volume.firstiter_cc import (
    _build_firstiter_cc_pass2_grids,
    _score_kclass_firstiter_cc_pass2,
)
from recovar.em.dense_single_volume.helpers.dtype_policy import (
    _diagnostic_float64_pass2_matches,
    _local_search_precision_flags,
)
from recovar.em.dense_single_volume.helpers.half_volume_mstep import relion_backprojector_volume_shape
from recovar.em.dense_single_volume.k_class import run_dense_k_class_em, run_dense_k_class_em_adaptive
from recovar.em.dense_single_volume.local_debug import (
    log_local_adaptive_support,
    log_local_denominator_support,
)
from recovar.em.dense_single_volume.local_layout import (
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_search_iteration import _run_local_search_iteration
from recovar.em.dense_single_volume.score_outputs import (
    HalfScoreResult,
    PerHalfOutputs,
    _collapse_fine_pose_assignments_to_coarse,
    _collapse_single_class_stats_to_coarse,
    _scatter_dense_k_class_result,
    _select_single_class_accumulator,
)
from recovar.em.dense_single_volume.scoring_policy import (
    _DENSE_EM_STATIC_KWARGS,
    _LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
    _LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT_ENV,
    _LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV,
    _LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV,
    PADDING_FACTOR,
    PROJECTION_PADDING_FACTOR,
    RELION_ACC_DOUBLE_FLOORF_QUIRK,
    RELION_ADAPTIVE_FRACTION,
    RELION_FOURIER_WINDOW_SQUARE,
    _dense_global_scoring_dtype,
    _k1_relion_x_half_mstep_enabled,
    _k1_skip_significance_pruning_enabled,
    _k_class_relion_half_volume_mstep_enabled,
    _k_class_relion_x_half_mstep_enabled,
    _local_adaptive_pass2_denominator_support_mode,
    _local_adaptive_pass2_full_parent_enabled,
    _local_adaptive_pass2_rotation_only_enabled,
)
from recovar.em.sampling import (
    apply_relion_translation_perturbation,
    build_local_search_grid_metadata,
    relion_angular_sampling_deg,
    rotation_grid_size,
)

logger = logging.getLogger(__name__)


def _expand_significant_samples_to_full_parent_translations(
    significant_sample_indices,
    n_parent_translations: int,
):
    """Expand significant parent rotation ids to every parent translation."""

    n_parent_translations = int(n_parent_translations)
    if n_parent_translations <= 0:
        raise ValueError(f"n_parent_translations must be positive, got {n_parent_translations}")
    expanded = []
    parent_translations = np.arange(n_parent_translations, dtype=np.int64)
    for samples in significant_sample_indices:
        if samples is None:
            expanded.append(None)
            continue
        samples_np = np.asarray(samples, dtype=np.int64).reshape(-1)
        if samples_np.size == 0:
            expanded.append(samples_np)
            continue
        parent_rotations = np.unique(samples_np // n_parent_translations).astype(np.int64, copy=False)
        expanded_samples = (parent_rotations[:, None] * n_parent_translations + parent_translations[None, :]).reshape(
            -1
        )
        expanded.append(expanded_samples.astype(np.int64, copy=False))
    return expanded


def _score_half_dense(
    *,
    k: int,
    experiment_dataset,
    means_k,
    mean_variance,
    noise_variance_k,
    effective_rotations,
    current_translations,
    base_translations,
    current_healpix_order: int,
    state,
    random_perturbation,
    disc_type,
    image_batch_size: int,
    rotation_log_prior_k,
    class_rotation_log_prior_k,
    translation_log_prior,
    translation_search_base,
    trans_prior_center_for_engine,
    image_corrections_k,
    scale_corrections_k,
    firstiter_score_mode_this_iter: str,
    firstiter_winner_take_all_this_iter: bool,
    cs_for_engine,
    model_current_size_for_engine=None,
    class_log_priors,
    k_class_enabled: bool,
    relion_firstiter_cc_this_iter: bool,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    safe_batch_sizes,
    max_significants,
    # Output lists are owned by the caller and mutated in place:
    outputs: "PerHalfOutputs",
    group_ids_k=None,
    group_count_k=None,
    scale_correction_data_vs_prior=None,
    # Mode-specific overrides (adaptive site sets these; single-pass uses
    # the defaults):
    k_class_image_batch_size_override: int | None = None,
    k_class_rotation_block_size_override: int | None = None,
    significance_image_batch_size_override: int | None = None,
    significance_rotation_block_size_override: int | None = None,
    firstiter_coarse_current_size: int | None = None,
    firstiter_fine_current_size: int | None = None,
    firstiter_log_label: str = "(non-adaptive site) ",
    firstiter_updates_em_kwargs_ibs: bool = False,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    return_best_pose_details: bool = True,
    bpref_device_signature_active: bool = False,
    debug_iteration: int | None = None,
    coarse_rotation_ids=None,
    preserve_bpref_particle_order: bool = False,
    source_faithful_spectrum_norm: bool = False,
) -> HalfScoreResult:
    """Dense (non-local-search) E+M scoring for one half-set.

    Used by both the single-pass (``else``) and adaptive-2-pass
    (``elif use_adaptive``) branches of the half-set loop. The two modes
    differ in five places, all controlled by trailing parameters:

    1. ``k_class_image_batch_size_override`` /
       ``k_class_rotation_block_size_override`` — adaptive overrides
       em_kwargs ibs/rbs to K-class values before firstiter_cc check.
    2. ``significance_*_override`` — adaptive pass 1 may use a smaller
       Fourier window than pass 2, so it needs its own memory-sized batches.
    3. ``firstiter_coarse_current_size`` / ``firstiter_fine_current_size``
       — adaptive passes ``coarse_cs`` / ``cs_for_engine`` through to the
       adaptive 2-pass engine; single-pass omits them.
    4. ``firstiter_log_label`` — single-pass uses
       ``"(non-adaptive site) "`` for the routing log message.
    5. ``firstiter_updates_em_kwargs_ibs`` — adaptive overrides
       em_kwargs["image_batch_size"] with the firstiter clamp; single-pass
       leaves em_kwargs untouched.

    Stores K-class summaries and explicit best poses in ``outputs``. The
    caller records the common payload from the returned ``HalfScoreResult``.
    ``safe_batch_sizes`` is the closure-bound batch sizer from
    ``_run_relion_iteration_loop``.
    """

    safe_ibs, safe_rbs = safe_batch_sizes(
        effective_rotations.shape[0],
        current_translations.shape[0],
        current_size_for_batch=cs_for_engine,
    )
    em_kwargs = {
        **_DENSE_EM_STATIC_KWARGS,
        "image_batch_size": safe_ibs,
        "rotation_block_size": safe_rbs,
        "current_size": cs_for_engine,
        "rotation_log_prior": rotation_log_prior_k,
        "translation_log_prior": translation_log_prior,
        "image_corrections": image_corrections_k,
        "scale_corrections": scale_corrections_k,
        "group_ids": group_ids_k,
        "scale_correction_group_count": group_count_k,
        "scale_correction_data_vs_prior": scale_correction_data_vs_prior,
        "image_pre_shifts": translation_search_base,
        "translation_prior_centers": trans_prior_center_for_engine,
        "relion_firstiter_score_mode": firstiter_score_mode_this_iter,
        "relion_firstiter_winner_take_all": firstiter_winner_take_all_this_iter,
    }
    if model_current_size_for_engine is not None:
        em_kwargs["reconstruction_current_size"] = model_current_size_for_engine
    if preserve_bpref_particle_order and k_class_enabled:
        raise ValueError("RELION BPref particle-order preservation is K=1-only")
    if preserve_bpref_particle_order:
        em_kwargs["preserve_bpref_particle_order"] = True
    if source_faithful_spectrum_norm:
        em_kwargs["source_faithful_spectrum_norm"] = True
    diagnostic_float64_pass2 = _diagnostic_float64_pass2_matches(debug_iteration)
    if diagnostic_float64_pass2:
        logger.info(
            "Diagnostic genuine-float64 adaptive pass 2 at iteration %d; pass 1 and prior boundaries remain f32",
            int(debug_iteration),
        )
    if k_class_image_batch_size_override is not None:
        em_kwargs["image_batch_size"] = k_class_image_batch_size_override
    if k_class_rotation_block_size_override is not None:
        em_kwargs["rotation_block_size"] = k_class_rotation_block_size_override
    if class_rotation_log_prior_k is not None:
        em_kwargs["rotation_log_prior"] = None
        em_kwargs["class_rotation_log_prior"] = class_rotation_log_prior_k
    if relion_projector_half is not None:
        em_kwargs["relion_projector_half"] = relion_projector_half
        em_kwargs["relion_projector_r_max"] = relion_projector_r_max
    logger.info(
        "Dense half-set projector handoff: supplied_ppref=%s state_oversampling=%d",
        relion_projector_half is not None,
        int(state.adaptive_oversampling),
    )
    if relion_firstiter_cc_this_iter:
        # Shared first-iteration inputs; means, layouts and pose IDs remain route-specific.
        firstiter_kwargs = {
            "logger": logger,
            "experiment_dataset": experiment_dataset,
            "mean_variance": mean_variance,
            "noise_variance_k": noise_variance_k,
            "effective_rotations": effective_rotations,
            "current_translations": current_translations,
            "base_translations": base_translations,
            "current_healpix_order": current_healpix_order,
            "state": state,
            "random_perturbation": random_perturbation,
            "disc_type": disc_type,
            "class_log_priors": class_log_priors,
            "image_batch_size": image_batch_size,
            "safe_batch_sizes": safe_batch_sizes,
            "coarse_current_size": firstiter_coarse_current_size,
            "fine_current_size": firstiter_fine_current_size,
            "update_em_kwargs_image_batch_size": firstiter_updates_em_kwargs_ibs,
            "bpref_device_signature_active": bpref_device_signature_active,
            "debug_iteration": debug_iteration,
        }

    if k_class_enabled:
        if disable_adjoint_y or disable_adjoint_ctf:
            raise NotImplementedError("K-class refine does not support adjoint ablation flags")
        # K-class should use RELION's x-half BackProjector accumulator layout,
        # matching the K=1 parity path.  The old full-volume and native
        # half-volume paths remain available as diagnostics via
        # RECOVAR_K_CLASS_RELION_X_HALF_MSTEP=0 together with the legacy
        # RECOVAR_K_CLASS_FULL_VOLUME_MSTEP / RECOVAR_K_CLASS_HALF_VOLUME_MSTEP
        # switches.
        k_class_relion_x_half_mstep = _k_class_relion_x_half_mstep_enabled()
        em_kwargs["mstep_relion_x_half"] = bool(k_class_relion_x_half_mstep)
        em_kwargs["relion_half_volume_mstep"] = (
            False if k_class_relion_x_half_mstep else _k_class_relion_half_volume_mstep_enabled()
        )
        k_class_mstep_full_half_axis_this_score = None
        rot_pmap_for_collapse = None
        trans_pmap_for_collapse = None
        n_trans_fine_for_collapse = None
        fine_rotations_for_pose = None
        adaptive_os_local = 0
        # STRICT-PARITY: at iter 1 with --firstiter_cc, route through the
        # adaptive 2-pass engine with normalized-CC scoring. Pass 2 retains the
        # oversampled children of the single best coarse class/pose, matching
        # RELION's firstiter-CC binarized coarse support.
        if relion_firstiter_cc_this_iter:
            (
                k_class_result,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                n_trans_fine_for_collapse,
                adaptive_os_local,
            ) = _score_kclass_firstiter_cc_pass2(
                mean=means_k,
                image_shape_k=experiment_dataset.image_shape,
                em_kwargs=em_kwargs,
                log_label=firstiter_log_label,
                coarse_rotation_ids=coarse_rotation_ids,
                **firstiter_kwargs,
            )
            k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis
        elif firstiter_coarse_current_size is not None and int(state.adaptive_oversampling) > 0:
            adaptive_os_local = int(state.adaptive_oversampling)
            (
                coarse_rot,
                coarse_trans,
                fine_rot,
                fine_trans,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                fine_mstep_rot,
            ) = _build_firstiter_cc_pass2_grids(
                effective_rotations,
                current_translations,
                base_translations,
                int(current_healpix_order),
                adaptive_os_local,
                float(state.translation_step),
                random_perturbation,
                return_mstep_rotations=True,
                **(
                    {"coarse_rotation_ids": coarse_rotation_ids}
                    if coarse_rotation_ids is not None
                    else {}
                ),
            )
            coarse_translation_phase_source = apply_relion_translation_perturbation(
                np.asarray(base_translations, dtype=np.float64),
                float(random_perturbation),
                float(state.translation_step),
            )
            n_trans_fine_for_collapse = int(fine_trans.shape[0])
            adaptive_em_kwargs = dict(em_kwargs)
            n_classes_local = int(np.asarray(means_k).shape[0]) if np.asarray(means_k).ndim >= 2 else 1
            grid_batch_plan = _plan_kclass_adaptive_grid_batch_sizes(
                coarse_rotations=coarse_rot,
                coarse_translations=coarse_trans,
                fine_rotations=fine_rot,
                fine_translations=fine_trans,
                n_classes=n_classes_local,
                image_shape=experiment_dataset.image_shape,
                coarse_current_size=firstiter_coarse_current_size,
                fine_current_size=firstiter_fine_current_size,
                safe_batch_sizes=safe_batch_sizes,
            )
            adaptive_em_kwargs["image_batch_size"] = grid_batch_plan.pass2_image_batch_size
            adaptive_em_kwargs["rotation_block_size"] = grid_batch_plan.pass2_rotation_block_size
            significance_image_batch_size_override = grid_batch_plan.significance_image_batch_size
            significance_rotation_block_size_override = grid_batch_plan.significance_rotation_block_size
            logger.info(
                "RELION adaptive K-class grid batch sizing: "
                "coarse image_batch_size=%d rotation_block_size=%d; "
                "fine image_batch_size=%d rotation_block_size=%d",
                significance_image_batch_size_override,
                significance_rotation_block_size_override,
                adaptive_em_kwargs["image_batch_size"],
                adaptive_em_kwargs["rotation_block_size"],
            )
            # ``RECOVAR_K_CLASS_DENSE_PASS2=1`` swaps K-class adaptive
            # oversampling from sparse-bucketed pass-2 to dense pass-2.
            # Diagnostic: tests whether the sparse-bucket reduction order
            # carries a structural bias vs the dense in-place reduction.
            kclass_sparse_pass2 = not bool(
                os.environ.get("RECOVAR_K_CLASS_DENSE_PASS2", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            adaptive_em_kwargs["sparse_pass2"] = kclass_sparse_pass2
            logger.info(
                "RELION adaptive K-class routing through run_dense_k_class_em_adaptive "
                "(oversampling=%d, pass2_backend=%s, fine_mstep_prune=%s)",
                adaptive_os_local,
                "sparse" if kclass_sparse_pass2 else "dense",
                bool(kclass_sparse_pass2),
            )
            k_class_result = run_dense_k_class_em_adaptive(
                experiment_dataset,
                means_k,
                mean_variance,
                noise_variance_k,
                coarse_rot,
                coarse_trans,
                fine_rot,
                fine_trans,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                disc_type,
                class_log_priors=class_log_priors,
                accumulate_noise=True,
                adaptive_fraction=RELION_ADAPTIVE_FRACTION,
                max_significants=-1 if max_significants is None else int(max_significants),
                relion_fine_mstep_prune=bool(kclass_sparse_pass2),
                significance_image_batch_size=significance_image_batch_size_override,
                significance_rotation_block_size=significance_rotation_block_size_override,
                coarse_current_size=firstiter_coarse_current_size,
                fine_current_size=firstiter_fine_current_size,
                coarse_healpix_order=int(current_healpix_order),
                oversampling_order=int(adaptive_os_local),
                fine_mstep_rotations_override=(fine_mstep_rot if kclass_sparse_pass2 else None),
                return_best_pose_details=return_best_pose_details,
                bpref_device_signature_active=bpref_device_signature_active,
                debug_iteration=debug_iteration,
                **adaptive_em_kwargs,
            )
            k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis
        else:
            dense_em_kwargs = dict(em_kwargs)
            # The direct dense K-class wrapper delegates to run_em, which does
            # not implement RELION x-half accumulators. Keep that branch on its
            # historical layout and avoid tagging its full-volume output as
            # x-half-expanded.
            dense_em_kwargs.pop("mstep_relion_x_half", None)
            dense_em_kwargs.pop("group_ids", None)
            dense_em_kwargs.pop("scale_correction_group_count", None)
            dense_em_kwargs.pop("scale_correction_data_vs_prior", None)
            # Exact fine-Gaussian scoring is implemented only by sparse pass 2.
            # A non-adaptive dense iteration has no fine pass to select.
            dense_em_kwargs.pop("relion_exact_fine_gaussian", None)
            dense_em_kwargs.pop("reconstruction_current_size", None)
            k_class_result = run_dense_k_class_em(
                experiment_dataset,
                means_k,
                mean_variance,
                noise_variance_k,
                effective_rotations,
                current_translations,
                disc_type,
                class_log_priors=class_log_priors,
                accumulate_noise=True,
                return_best_pose_details=return_best_pose_details,
                **dense_em_kwargs,
            )
            k_class_mstep_full_half_axis_this_score = None
        ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = _scatter_dense_k_class_result(
            k_class_result,
            k=k,
            effective_rotations=effective_rotations,
            rot_pmap_for_collapse=rot_pmap_for_collapse,
            adaptive_os_local=adaptive_os_local,
            outputs=outputs,
            require_best_pose_details=return_best_pose_details,
            pose_dtype=_dense_global_scoring_dtype(),
        )
        coarse_ha_k = None
        if trans_pmap_for_collapse is not None and n_trans_fine_for_collapse is not None:
            coarse_ha_k = _collapse_fine_pose_assignments_to_coarse(
                ha_k,
                rot_parent_map=rot_pmap_for_collapse,
                trans_parent_map=trans_pmap_for_collapse,
                n_trans_coarse=current_translations.shape[0],
                n_trans_fine=n_trans_fine_for_collapse,
            )
        return HalfScoreResult(
            ha=ha_k,
            Ft_y=Ft_y_k,
            Ft_ctf=Ft_ctf_k,
            em_stats=em_stats_k,
            noise_stats=noise_stats_k,
            coarse_ha=coarse_ha_k,
            significant_counts=(
                None
                if k_class_result.significant_counts is None
                else np.asarray(k_class_result.significant_counts, dtype=np.int32)
            ),
            profile_summary=k_class_result.profile_summary,
            mstep_full_half_axis=k_class_mstep_full_half_axis_this_score,
            mstep_accumulator_shape=getattr(k_class_result, "mstep_accumulator_shape", None),
        )

    if int(state.adaptive_oversampling) > 0:
        if disable_adjoint_y or disable_adjoint_ctf:
            raise NotImplementedError("K=1 adaptive oversampling does not support adjoint ablation flags")
        adaptive_os_local = int(state.adaptive_oversampling)
        k1_relion_x_half_mstep = _k1_relion_x_half_mstep_enabled()
        means_single = jnp.asarray(means_k)[None, :]
        rot_pmap_for_collapse = None
        trans_pmap_for_collapse = None
        n_trans_fine_for_collapse = None
        fine_rotations_for_pose = None
        if relion_firstiter_cc_this_iter:
            (
                k1_adaptive_result,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                n_trans_fine_for_collapse,
                adaptive_os_local,
            ) = _score_kclass_firstiter_cc_pass2(
                mean=means_single,
                image_shape_k=experiment_dataset.image_shape,
                em_kwargs=(
                    {**em_kwargs, "mstep_relion_x_half": True}
                    if k1_relion_x_half_mstep
                    else em_kwargs
                ),
                log_label="K=1 ",
                **firstiter_kwargs,
            )
        else:
            (
                coarse_rot,
                coarse_trans,
                fine_rot,
                fine_trans,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                fine_mstep_rot,
            ) = _build_firstiter_cc_pass2_grids(
                effective_rotations,
                current_translations,
                base_translations,
                int(current_healpix_order),
                adaptive_os_local,
                float(state.translation_step),
                random_perturbation,
                return_mstep_rotations=True,
                **(
                    {"coarse_rotation_ids": coarse_rotation_ids}
                    if coarse_rotation_ids is not None
                    else {}
                ),
            )
            coarse_translation_phase_source = apply_relion_translation_perturbation(
                np.asarray(base_translations, dtype=np.float64),
                float(random_perturbation),
                float(state.translation_step),
            )
            n_trans_fine_for_collapse = int(fine_trans.shape[0])
            fine_rotations_for_pose = fine_rot
            adaptive_em_kwargs = dict(em_kwargs)
            k1_sparse_pass2 = not bool(
                os.environ.get("RECOVAR_K1_DENSE_PASS2", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            k1_skip_significance_pruning = _k1_skip_significance_pruning_enabled()
            adaptive_em_kwargs["sparse_pass2"] = k1_sparse_pass2
            if group_ids_k is not None:
                adaptive_em_kwargs["group_ids"] = group_ids_k
            if k1_relion_x_half_mstep:
                adaptive_em_kwargs["mstep_relion_x_half"] = True
            logger.info(
                "RELION adaptive K=1 routing through run_dense_k_class_em_adaptive "
                "(oversampling=%d, pass2_backend=%s, skip_significance_pruning=%s, "
                "fine_mstep_prune=%s, relion_x_half_mstep=%s, supplied_ppref=%s, "
                "engine_ppref=%s)",
                adaptive_os_local,
                "sparse" if k1_sparse_pass2 else "dense",
                bool(k1_skip_significance_pruning),
                bool(k1_sparse_pass2),
                bool(k1_relion_x_half_mstep),
                relion_projector_half is not None,
                adaptive_em_kwargs.get("relion_projector_half") is not None,
            )
            k1_adaptive_result = run_dense_k_class_em_adaptive(
                experiment_dataset,
                means_single,
                mean_variance,
                noise_variance_k,
                coarse_rot,
                coarse_trans,
                fine_rot,
                fine_trans,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                disc_type,
                class_log_priors=class_log_priors,
                accumulate_noise=True,
                adaptive_fraction=RELION_ADAPTIVE_FRACTION,
                max_significants=-1 if max_significants is None else int(max_significants),
                skip_significance_pruning=k1_skip_significance_pruning,
                relion_fine_mstep_prune=bool(k1_sparse_pass2),
                significance_image_batch_size=significance_image_batch_size_override,
                significance_rotation_block_size=significance_rotation_block_size_override,
                coarse_current_size=firstiter_coarse_current_size,
                fine_current_size=firstiter_fine_current_size,
                coarse_healpix_order=int(current_healpix_order),
                oversampling_order=int(adaptive_os_local),
                fine_mstep_rotations_override=(fine_mstep_rot if k1_sparse_pass2 else None),
                return_best_pose_details=return_best_pose_details,
                bpref_device_signature_active=bpref_device_signature_active,
                debug_iteration=debug_iteration,
                pass2_use_float64_scoring=True if diagnostic_float64_pass2 else None,
                pass2_use_float64_projections=True if diagnostic_float64_pass2 else None,
                coarse_translation_phase_source=coarse_translation_phase_source,
                **adaptive_em_kwargs,
            )
        ha_k = np.asarray(k1_adaptive_result.pose_assignments, dtype=np.int32)
        Ft_y_k = _select_single_class_accumulator(k1_adaptive_result.Ft_y, label="Ft_y")
        Ft_ctf_k = _select_single_class_accumulator(k1_adaptive_result.Ft_ctf, label="Ft_ctf")
        em_stats_k = _collapse_single_class_stats_to_coarse(
            k1_adaptive_result.stats,
            rot_parent_map=rot_pmap_for_collapse,
            n_rot_coarse=effective_rotations.shape[0],
            dtype=_dense_global_scoring_dtype(),
        )
        noise_stats_k = k1_adaptive_result.aggregate_noise_stats
        if noise_stats_k is None and k1_adaptive_result.noise_stats is not None:
            noise_stats_k = k1_adaptive_result.noise_stats[0]
        if noise_stats_k is None:
            raise RuntimeError("K=1 adaptive path did not return noise statistics")
        coarse_ha_k = None
        if trans_pmap_for_collapse is not None and n_trans_fine_for_collapse is not None:
            coarse_ha_k = _collapse_fine_pose_assignments_to_coarse(
                ha_k,
                rot_parent_map=rot_pmap_for_collapse,
                trans_parent_map=trans_pmap_for_collapse,
                n_trans_coarse=current_translations.shape[0],
                n_trans_fine=n_trans_fine_for_collapse,
            )
        if return_best_pose_details:
            if (
                k1_adaptive_result.best_pose_rotations is None
                or k1_adaptive_result.best_pose_translations is None
            ):
                raise RuntimeError("K=1 adaptive path did not return best pose details")
            pose_dtype = _dense_global_scoring_dtype()
            best_rots = np.asarray(k1_adaptive_result.best_pose_rotations, dtype=pose_dtype)
            outputs.best_pose_rotations[k] = best_rots
            outputs.best_pose_rotation_eulers[k] = utils.R_to_relion(best_rots, degrees=True).astype(pose_dtype)
            outputs.best_pose_translations[k] = np.asarray(k1_adaptive_result.best_pose_translations, dtype=pose_dtype)
        if fine_rotations_for_pose is None and rot_pmap_for_collapse is not None:
            fine_rotations_for_pose = _build_firstiter_cc_pass2_grids(
                effective_rotations,
                current_translations,
                base_translations,
                int(current_healpix_order),
                adaptive_os_local,
                float(state.translation_step),
                random_perturbation,
                **(
                    {"coarse_rotation_ids": coarse_rotation_ids}
                    if coarse_rotation_ids is not None
                    else {}
                ),
            )[2]
        fine_rotation_eulers_for_pose = None
        if fine_rotations_for_pose is not None and _parity_dump.is_active():
            fine_rotation_eulers_for_pose = utils.R_to_relion(
                np.asarray(fine_rotations_for_pose, dtype=np.float32),
                degrees=True,
            ).astype(np.float32)
        return HalfScoreResult(
            ha=ha_k,
            Ft_y=Ft_y_k,
            Ft_ctf=Ft_ctf_k,
            em_stats=em_stats_k,
            noise_stats=noise_stats_k,
            best_pose_rotations=outputs.best_pose_rotations[k],
            best_pose_rotation_eulers=outputs.best_pose_rotation_eulers[k],
            best_pose_translations=outputs.best_pose_translations[k],
            coarse_ha=coarse_ha_k,
            pose_rotations=fine_rotations_for_pose,
            pose_rotation_eulers=fine_rotation_eulers_for_pose,
            significant_counts=(
                None
                if k1_adaptive_result.significant_counts is None
                else np.asarray(k1_adaptive_result.significant_counts, dtype=np.int32)
            ),
            profile_summary=k1_adaptive_result.profile_summary,
            mstep_full_half_axis=k1_adaptive_result.mstep_full_half_axis,
            mstep_accumulator_shape=getattr(k1_adaptive_result, "mstep_accumulator_shape", None),
        )

    if group_ids_k is not None:
        raise RuntimeError(
            "RELION native group-scale correction requires the sparse/adaptive or local "
            "M-step; direct dense K=1 does not accumulate group XA/AA statistics"
        )
    direct_em_kwargs = dict(em_kwargs)
    direct_em_kwargs.pop("group_ids", None)
    direct_em_kwargs.pop("scale_correction_group_count", None)
    direct_em_kwargs.pop("scale_correction_data_vs_prior", None)
    # Exact fine-Gaussian scoring is implemented only by sparse pass 2.  This
    # branch is the single dense pass used when adaptive oversampling is off.
    direct_em_kwargs.pop("relion_exact_fine_gaussian", None)
    direct_em_kwargs.pop("reconstruction_current_size", None)
    em_result = run_em(
        experiment_dataset,
        means_k,
        mean_variance,
        noise_variance_k,
        effective_rotations,
        current_translations,
        disc_type,
        return_stats=True,
        accumulate_noise=True,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        **direct_em_kwargs,
    )
    return HalfScoreResult(
        ha=em_result.hard_assignments,
        Ft_y=em_result.Ft_y,
        Ft_ctf=em_result.Ft_ctf,
        em_stats=em_result.stats,
        noise_stats=em_result.noise_stats,
        mstep_accumulator_shape=None,
    )


def _score_half_dense_in_bpref_scope(
    *,
    bpref_device_signature_active: bool,
    **kwargs,
) -> HalfScoreResult:
    """Keep all authoritative dense-half work outside diagnostic CUDA scope."""

    with _cuda_backproject_diagnostics.bpref_device_signature_scope(False):
        return _score_half_dense(
            bpref_device_signature_active=bpref_device_signature_active,
            **kwargs,
        )


def _local_translation_prior_reference_translations(
    *,
    current_translations,
    base_translations,
    replay_prior_translations,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, str, bool]:
    """Choose a local-search translation-prior grid compatible with scoring."""

    current = np.asarray(current_translations, dtype=dtype)
    base = np.asarray(base_translations, dtype=dtype)
    if replay_prior_translations is not None:
        candidate = np.asarray(replay_prior_translations, dtype=dtype)
        source = "replay"
    else:
        candidate = base
        source = "base"

    if candidate.shape == current.shape:
        return candidate, source, False
    if base.shape == current.shape:
        return base, "base", True
    return current, "current", True


def _relion_coarse_significant_counts(significant_sample_indices):
    """Count explicit retained pass-1 samples using RELION metadata semantics."""

    if any(indices is None for indices in significant_sample_indices):
        return None
    return np.asarray(
        [np.asarray(indices).size for indices in significant_sample_indices],
        dtype=np.int32,
    )


def _score_half_local(
    *,
    k: int,
    experiment_dataset,
    means_k,
    mean_variance,
    noise_variance_k,
    previous_best_rotation_eulers_k,
    local_search_rotations,
    local_search_rotation_eulers,
    local_search_order: int,
    sigma_rot,
    sigma_psi,
    current_translations,
    base_translations,
    trans_prior_center,
    trans_prior_center_for_engine,
    current_sigma_offset_angstrom: float,
    current_translation_range: float,
    disc_type,
    cs_for_engine,
    model_current_size_for_engine=None,
    local_pass1_current_size,
    image_corrections_k,
    scale_corrections_k,
    translation_search_base,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    max_significants,
    iteration: int,
    debug_iteration: int | None = None,
    save_intermediates_dir,
    local_search_random_perturbation,
    local_search_angular_sampling_deg,
    local_parent_oversampling_order: int,
    local_search_translation_prior_mode: str,
    replay_prior_translations,
    class_log_priors,
    k_class_enabled: bool,
    collect_local_search_profile: bool,
    diagnostic_score_only: bool,
    safe_batch_sizes,
    # Output lists are owned by the caller and mutated in place:
    outputs: "PerHalfOutputs",
    local_profile_history,
    local_search_mstep_rotations=None,
    group_ids_k=None,
    group_count_k=None,
    scale_correction_data_vs_prior=None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    source_faithful_spectrum_norm: bool = False,
) -> HalfScoreResult:
    """Local-search E+M scoring for one half-set.

    Sizes the per-chunk M-step batches against the cone-restricted
    rotation count (not the full HEALPix grid) so chunk_size doesn't
    collapse at high HEALPix orders. Routes through
    ``_run_local_search_iteration`` which itself handles K-class /
    K=1 internally via ``return_class_details=k_class_enabled``.

    Caller handles ``noise_stats_per_half[k]``, ``pose_rotations[k] = None``,
    and ``coarse_ha[k] = ha_k`` from the returned ``HalfScoreResult``.
    """

    # RELION's convertAllSquaredDifferencesToWeights uses mymodel.pdf_direction
    # only when orientational_prior_mode == NOPRIOR. Local searches run through
    # PRIOR_ROTTILT_PSI and score the local direction/psi priors in the
    # hypothesis layout, so adding the learned global direction prior here
    # biases both support selection and final weights.
    relion_local_rotation_log_prior_k = None
    if diagnostic_score_only and k_class_enabled:
        raise NotImplementedError("score-only local-search diagnostics are currently K=1-only")
    if source_faithful_spectrum_norm and k_class_enabled:
        raise ValueError("RELION source-faithful spectrum normalization is fresh K=1-only")

    reconstruction_current_size_for_engine = (
        cs_for_engine
        if model_current_size_for_engine is None
        else model_current_size_for_engine
    )

    # For local search the per-chunk M-step only sees the cone-restricted
    # rotation set (typically a few thousand rotations per image with high
    # overlap across the chunk) rather than the full ~10⁶-rotation grid at
    # healpix order 5+. Estimate per-image cone size from
    #     fraction = (sigma_cutoff * sigma_rot / pi)^2
    # (spherical cap area as a fraction of full SO(3) volume; good to
    # within ~30% for reasonable cones). Use that for an effective rotation
    # count equal to ``chunk_size * cone_size`` with a 2x safety factor.
    cone_radius = 3.0 * float(sigma_rot)  # sigma_cutoff=3.0
    cone_fraction = max(
        (cone_radius / float(np.pi)) ** 2,
        1.0 / float(rotation_grid_size(local_search_order)),
    )
    est_cone_rots = int(np.ceil(rotation_grid_size(local_search_order) * cone_fraction))
    eff_n_rot = max(64, 2 * est_cone_rots)
    local_n_trans = int(current_translations.shape[0])
    if int(local_parent_oversampling_order) > 0:
        local_n_trans *= int(4 ** int(local_parent_oversampling_order))
    local_debug_iteration = iteration + 1 if debug_iteration is None else int(debug_iteration)
    parent_use_float64_scoring, parent_use_float64_projections = _local_search_precision_flags(
        local_debug_iteration,
        pass_index=1,
        static_em_kwargs=_DENSE_EM_STATIC_KWARGS,
    )
    fine_use_float64_scoring, fine_use_float64_projections = _local_search_precision_flags(
        local_debug_iteration,
        pass_index=2,
        static_em_kwargs=_DENSE_EM_STATIC_KWARGS,
    )
    # Adaptive pass-2 (fine, oversampled) hypothesis layout precision; see
    # ``parent_local_layout_dtype`` below for the matching pass-1 value.
    fine_local_layout_dtype = (
        np.float64 if (fine_use_float64_scoring or fine_use_float64_projections) else np.float32
    )
    if fine_use_float64_scoring or fine_use_float64_projections:
        logger.info(
            "Local-search precision iteration %d: pass1 scoring/projections=%s/%s "
            "pass2 scoring/projections=%s/%s",
            local_debug_iteration,
            parent_use_float64_scoring,
            parent_use_float64_projections,
            fine_use_float64_scoring,
            fine_use_float64_projections,
        )

    safe_ibs, safe_rbs = safe_batch_sizes(
        eff_n_rot,
        local_n_trans,
        image_shape_for_batch=experiment_dataset.image_shape,
        current_size_for_batch=cs_for_engine,
    )
    logger.info(
        "Local search batch sizing: cone_radius=%.3f rad (%.2f deg), est_cone_rots=%d, eff_n_rot=%d "
        "n_trans=%d → image_batch_size=%d, rotation_block_size=%d",
        cone_radius,
        np.rad2deg(cone_radius),
        est_cone_rots,
        eff_n_rot,
        local_n_trans,
        safe_ibs,
        safe_rbs,
    )
    # Keep the local-search hypothesis grid genuinely double precision end to
    # end when either flag requests it; default stays float32 to match
    # RELION's accelerated-GPU precision.
    parent_local_layout_dtype = (
        np.float64 if (parent_use_float64_scoring or parent_use_float64_projections) else np.float32
    )
    translation_prior_reference_translations = np.asarray(
        current_translations, dtype=parent_local_layout_dtype
    )
    if local_search_translation_prior_mode == "coarse":
        translation_prior_reference_translations, prior_grid_source, prior_grid_shape_mismatch = (
            _local_translation_prior_reference_translations(
                current_translations=current_translations,
                base_translations=base_translations,
                replay_prior_translations=replay_prior_translations,
                dtype=parent_local_layout_dtype,
            )
        )
        if prior_grid_shape_mismatch:
            logger.warning(
                "RELION mode: local translation prior grid from replay/base did not match scoring grid; "
                "using %s grid shape=%s for scoring grid shape=%s",
                prior_grid_source,
                translation_prior_reference_translations.shape,
                np.asarray(current_translations).shape,
            )
        logger.info(
            "RELION mode: local translation prior uses coarse %s grid (n=%d) while scoring perturbed translations",
            prior_grid_source,
            translation_prior_reference_translations.shape[0],
        )
    if int(local_parent_oversampling_order) > 0:
        logger.info(
            "RELION local search: expanding translations by oversampling_order=%d (coarse n=%d -> fine n=%d)",
            int(local_parent_oversampling_order),
            int(current_translations.shape[0]),
            int(local_n_trans),
        )
    # Shared operands and options for parent, denominator and final scoring.
    # Pass-specific precision, support, reconstruction and profiling stay below.
    common_local_kwargs = {
        "projection_padding_factor": PROJECTION_PADDING_FACTOR,
        "reconstruction_padding_factor": PADDING_FACTOR,
        "relion_projector_half": relion_projector_half,
        "relion_projector_r_max": relion_projector_r_max,
        "do_gridding_correction": True,
        "square_window": RELION_FOURIER_WINDOW_SQUARE,
        "half_spectrum_scoring": True,
        "image_corrections": image_corrections_k,
        "scale_corrections": scale_corrections_k,
        "group_ids": group_ids_k,
        "scale_correction_group_count": group_count_k,
        "scale_correction_data_vs_prior": scale_correction_data_vs_prior,
        "image_pre_shifts": translation_search_base,
        "score_with_masked_images": True,
        "adaptive_fraction": RELION_ADAPTIVE_FRACTION,
        "max_significants": max_significants,
        "translation_prior_reference_translations": translation_prior_reference_translations,
        "translation_prior_centers": trans_prior_center_for_engine,
        "source_faithful_spectrum_norm": source_faithful_spectrum_norm,
    }
    pass2_layout = None
    relion_significant_counts_k = None
    local_adaptive_pass2_parent_mode = "none"
    local_adaptive_pass2_denominator_layout = None
    local_normalization_log_evidence = None
    if int(local_parent_oversampling_order) > 0 and not k_class_enabled:
        local_adaptive_pass2_full_parent = _local_adaptive_pass2_full_parent_enabled()
        local_adaptive_pass2_rotation_only = _local_adaptive_pass2_rotation_only_enabled()
        local_adaptive_pass2_denominator_mode = _local_adaptive_pass2_denominator_support_mode()
        local_adaptive_pass2_parent_mode = "full_parent" if local_adaptive_pass2_full_parent else "pruned_parent"
        parent_prior_translations = trans_prior_center
        if parent_prior_translations is None:
            parent_prior_translations = np.zeros(
                (np.asarray(previous_best_rotation_eulers_k).shape[0], np.asarray(current_translations).shape[1]),
                dtype=parent_local_layout_dtype,
            )
        parent_order = int(local_search_order) - int(local_parent_oversampling_order)
        if parent_order < 0:
            raise ValueError(
                "local_search_order must be >= local_parent_oversampling_order; "
                f"got {local_search_order} and {local_parent_oversampling_order}",
            )
        parent_grid_metadata = build_local_search_grid_metadata(parent_order)
        parent_layout = build_local_hypothesis_layout(
            previous_best_rotation_eulers_k,
            None,
            sigma_rot,
            sigma_psi,
            parent_order,
            current_translations,
            parent_prior_translations,
            current_sigma_offset_angstrom,
            None,
            experiment_dataset.voxel_size,
            grid_metadata=parent_grid_metadata,
            translation_prior_reference_translations=translation_prior_reference_translations,
            rotation_log_prior=relion_local_rotation_log_prior_k,
            rotation_grid_random_perturbation=local_search_random_perturbation,
            rotation_grid_angular_sampling_deg=relion_angular_sampling_deg(parent_order, adaptive_oversampling=0),
            dtype=parent_local_layout_dtype,
        )
        parent_local_rot_max = (
            int(np.max(np.asarray(parent_layout.rotation_counts, dtype=np.int64)))
            if int(np.asarray(parent_layout.rotation_counts).size)
            else 1
        )
        parent_ibs, parent_rbs = safe_batch_sizes(
            max(64, parent_local_rot_max),
            int(current_translations.shape[0]),
        )
        logger.info(
            "RELION local adaptive pass 1: parent_order=%d local_rot_max=%d n_trans=%d current_size=%s",
            parent_order,
            parent_local_rot_max,
            int(current_translations.shape[0]),
            local_pass1_current_size,
        )
        logger.info("RELION local adaptive pass 1: using manual supplied-PPref interpolation")
        parent_outputs = _run_local_search_iteration(
            experiment_dataset,
            means_k,
            mean_variance,
            noise_variance_k,
            previous_best_rotation_eulers_k,
            None,
            None,
            parent_order,
            sigma_rot,
            sigma_psi,
            current_translations,
            trans_prior_center,
            current_sigma_offset_angstrom,
            current_translation_range,
            disc_type,
            image_batch_size=parent_ibs,
            rotation_block_size=parent_rbs,
            current_size=local_pass1_current_size,
            accumulate_noise=False,
            projection_relion_texture_interp=False,
            projection_relion_acc_double_floorf_quirk=RELION_ACC_DOUBLE_FLOORF_QUIRK,
            use_float64_scoring=parent_use_float64_scoring,
            use_float64_projections=parent_use_float64_projections,
            relion_exact_score_translation=bool(
                _DENSE_EM_STATIC_KWARGS["relion_exact_fine_gaussian"]
                and not parent_use_float64_scoring
            ),
            return_profile=True,
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            reconstruct_significant_only=True,
            debug_iteration=local_debug_iteration,
            debug_pass_label="pass1_parent",
            pass2_layout=parent_layout,
            return_best_pose_details=False,
            rotation_log_prior=relion_local_rotation_log_prior_k,
            return_reconstruction_sample_indices=True,
            apply_max_significants_to_support=True,
            score_only=True,
            **common_local_kwargs,
        )
        parent_profile = parent_outputs.profile_summary
        significant_sample_indices = parent_profile["reconstruction_sample_indices_by_image"]
        pruned_parent_significant_sample_indices = significant_sample_indices
        # RELION's rlnNrOfSignificantSamples records the number of retained
        # coarse hypotheses from pass 1, not the number of fine hypotheses
        # used for reconstruction in pass 2. Preserve this before any
        # diagnostic expansion of the pass-2 parent support.
        relion_significant_counts_k = _relion_coarse_significant_counts(
            pruned_parent_significant_sample_indices
        )
        if relion_significant_counts_k is None:
            logger.warning(
                "RELION local adaptive pass 1 did not return explicit retained support; "
                "rlnNrOfSignificantSamples-compatible counts are unavailable"
            )
        if local_adaptive_pass2_full_parent:
            significant_sample_indices = [None] * len(significant_sample_indices)
            logger.info(
                "RELION local adaptive pass 2: expanding all parent samples; "
                "set %s=0 or %s=1 for diagnostic pruned-parent support",
                _LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV,
                _LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT_ENV,
            )
        elif local_adaptive_pass2_rotation_only:
            significant_sample_indices = _expand_significant_samples_to_full_parent_translations(
                significant_sample_indices,
                int(current_translations.shape[0]),
            )
            local_adaptive_pass2_parent_mode = "significant_rotation_full_translation"
            logger.info(
                "RELION local adaptive pass 2 diagnostic: expanding significant parent rotations to all "
                "parent translations via %s=1",
                _LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV,
            )
        pass2_layout = build_local_adaptive_pass2_hypothesis_layout(
            parent_layout,
            significant_sample_indices,
            parent_order,
            oversampling_order=int(local_parent_oversampling_order),
            random_perturbation=float(local_search_random_perturbation),
            dtype=fine_local_layout_dtype,
        )
        if local_adaptive_pass2_denominator_mode is not None:
            if local_adaptive_pass2_denominator_mode == "full_parent":
                denominator_significant_sample_indices = [None] * len(pruned_parent_significant_sample_indices)
            elif local_adaptive_pass2_denominator_mode == "rotation_only":
                denominator_significant_sample_indices = _expand_significant_samples_to_full_parent_translations(
                    pruned_parent_significant_sample_indices,
                    int(current_translations.shape[0]),
                )
            else:  # Defensive only; parser restricts values.
                raise AssertionError(f"unexpected denominator mode {local_adaptive_pass2_denominator_mode!r}")
            local_adaptive_pass2_denominator_layout = build_local_adaptive_pass2_hypothesis_layout(
                parent_layout,
                denominator_significant_sample_indices,
                parent_order,
                oversampling_order=int(local_parent_oversampling_order),
                random_perturbation=float(local_search_random_perturbation),
                dtype=fine_local_layout_dtype,
            )
            log_local_denominator_support(
                logger,
                local_adaptive_pass2_denominator_layout,
                local_adaptive_pass2_denominator_mode,
                _LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
            )
        log_local_adaptive_support(
            logger, parent_layout, significant_sample_indices, current_translations, pass2_layout
        )
    elif int(local_parent_oversampling_order) > 0:
        local_adaptive_pass2_parent_mode = "k_class_parent_expanded"
        logger.info(
            "Adaptive local coarse-pair masking is currently K=1-only; K-class local search keeps the existing parent-expanded support"
        )
    local_relion_x_half_mstep = (
        _k_class_relion_x_half_mstep_enabled()
        if k_class_enabled
        else _k1_relion_x_half_mstep_enabled()
    )
    if diagnostic_score_only:
        local_relion_x_half_mstep = False
    if local_relion_x_half_mstep:
        logger.info(
            "RELION local %s M-step: using x-half BPref-layout backprojection",
            "K-class" if k_class_enabled else "K=1",
        )
    if local_adaptive_pass2_denominator_layout is not None:
        logger.info(
            "RELION local adaptive pass 2 diagnostic: running score-only broad-denominator probe"
        )
        local_debug_env_names = [
            name
            for name in os.environ
            if name.startswith("RECOVAR_LOCAL_SCORE_DUMP_")
            or name.startswith("RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_")
            or name.startswith("RECOVAR_LOCAL_NOISE_COMPONENT_DUMP_")
        ]
        saved_local_debug_env = {name: os.environ.pop(name) for name in local_debug_env_names}
        try:
            denominator_outputs = _run_local_search_iteration(
                experiment_dataset,
                means_k,
                mean_variance,
                noise_variance_k,
                previous_best_rotation_eulers_k,
                local_search_rotations,
                local_search_rotation_eulers,
                local_search_order,
                sigma_rot,
                sigma_psi,
                current_translations,
                trans_prior_center,
                current_sigma_offset_angstrom,
                current_translation_range,
                disc_type,
                image_batch_size=safe_ibs,
                rotation_block_size=safe_rbs,
                current_size=cs_for_engine,
                reconstruction_current_size=reconstruction_current_size_for_engine,
                accumulate_noise=False,
                use_float64_scoring=fine_use_float64_scoring,
                use_float64_projections=fine_use_float64_projections,
                relion_exact_score_translation=bool(
                    _DENSE_EM_STATIC_KWARGS["relion_exact_fine_gaussian"]
                    and not fine_use_float64_scoring
                ),
                return_profile=False,
                disable_adjoint_y=True,
                disable_adjoint_ctf=True,
                reconstruct_significant_only=False,
                debug_iteration=None,
                pass2_layout=local_adaptive_pass2_denominator_layout,
                return_best_pose_details=False,
                rotation_grid_random_perturbation=local_search_random_perturbation,
                rotation_grid_angular_sampling_deg=relion_angular_sampling_deg(
                    local_search_order,
                    adaptive_oversampling=0,
                ),
                score_only=True,
                **common_local_kwargs,
            )
        finally:
            os.environ.update(saved_local_debug_env)
        denominator_stats = denominator_outputs.relion_stats
        local_normalization_log_evidence = np.asarray(
            denominator_stats.log_evidence_per_image,
            dtype=np.float64,
        )
        logger.info(
            "RELION local adaptive pass 2 diagnostic: broad-denominator evidence ready "
            "(finite=%d/%d)",
            int(np.count_nonzero(np.isfinite(local_normalization_log_evidence))),
            int(local_normalization_log_evidence.size),
        )
    # RELION's accelerated local-search loop still executes the symbolic
    # second pass when adaptive_oversampling == 0. In that case
    # convertAllSquaredDifferencesToWeights sets significant_weight to the
    # minimum fine-pass weight, so storeWeightedSums keeps all local
    # candidates. Do not apply the 0.999 significant-support prune on
    # this os0 path.
    local_reconstruct_significant_only = int(local_parent_oversampling_order) > 0
    local_accumulate_noise = not diagnostic_score_only
    local_disable_adjoint_y = bool(disable_adjoint_y or diagnostic_score_only)
    local_disable_adjoint_ctf = bool(disable_adjoint_ctf or diagnostic_score_only)
    logger.info(
        "RELION local fine pass 2: supplied-PPref interpolation follows "
        "RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP (default texture)"
    )
    local_outputs = _run_local_search_iteration(
        experiment_dataset,
        means_k,
        mean_variance,
        noise_variance_k,
        previous_best_rotation_eulers_k,
        local_search_rotations,
        local_search_rotation_eulers,
        local_search_order,
        sigma_rot,
        sigma_psi,
        current_translations,
        trans_prior_center,
        current_sigma_offset_angstrom,
        current_translation_range,
        disc_type,
        image_batch_size=safe_ibs,
        rotation_block_size=safe_rbs,
        current_size=cs_for_engine,
        reconstruction_current_size=reconstruction_current_size_for_engine,
        accumulate_noise=local_accumulate_noise,
        # RELION's local adaptive path is intentionally hybrid: parent pass 1
        # uses the manual supplied-PPref projector above, while fine pass 2
        # follows the user-switchable texture default.
        projection_relion_texture_interp=None,
        projection_relion_acc_double_floorf_quirk=RELION_ACC_DOUBLE_FLOORF_QUIRK,
        use_float64_scoring=fine_use_float64_scoring,
        use_float64_projections=fine_use_float64_projections,
        relion_exact_score_translation=bool(
            _DENSE_EM_STATIC_KWARGS["relion_exact_fine_gaussian"]
            and not fine_use_float64_scoring
        ),
        mstep_relion_x_half=local_relion_x_half_mstep,
        return_profile=collect_local_search_profile,
        disable_adjoint_y=local_disable_adjoint_y,
        disable_adjoint_ctf=local_disable_adjoint_ctf,
        reconstruct_significant_only=local_reconstruct_significant_only,
        stats_use_reconstruction_probs=local_reconstruct_significant_only,
        debug_iteration=local_debug_iteration,
        debug_pass_label="pass2_final",
        pass2_layout=pass2_layout,
        return_best_pose_details=True,
        normalization_log_evidence=local_normalization_log_evidence,
        rotation_grid_random_perturbation=local_search_random_perturbation,
        rotation_grid_angular_sampling_deg=local_search_angular_sampling_deg,
        local_parent_oversampling_order=local_parent_oversampling_order,
        rotation_log_prior=None if pass2_layout is not None else relion_local_rotation_log_prior_k,
        class_log_priors=class_log_priors if k_class_enabled else None,
        return_class_details=k_class_enabled,
        # The engine count describes fine-pass reconstruction support. It is
        # deliberately not exposed as RELION's coarse pass-1 metadata count.
        return_significant_counts=False,
        score_only=diagnostic_score_only,
        rotation_grid_mstep_rotations=local_search_mstep_rotations,
        generate_relion_mstep_rotations=True,
        **common_local_kwargs,
    )
    Ft_y_k = local_outputs.Ft_y
    Ft_ctf_k = local_outputs.Ft_ctf
    ha_k = local_outputs.hard_assignment
    best_rots_k = local_outputs.best_pose_rotations
    best_trans_k = local_outputs.best_pose_translations
    _best_rot_ids_k = local_outputs.best_pose_rotation_ids
    em_stats_k = local_outputs.relion_stats
    noise_stats_k = local_outputs.noise_stats
    if collect_local_search_profile:
        local_profile_k = local_outputs.profile_summary
        profile_row = dict(local_profile_k)
        profile_row["iteration"] = np.int32(iteration)
        profile_row["half_index"] = np.int32(k)
        profile_row["local_adaptive_pass2_parent_mode"] = local_adaptive_pass2_parent_mode
        profile_row["local_adaptive_pass2_full_parent"] = np.bool_(local_adaptive_pass2_parent_mode == "full_parent")
        profile_row["diagnostic_score_only"] = np.bool_(diagnostic_score_only)
        local_profile_history.append(profile_row)
        if save_intermediates_dir is not None:
            np.savez_compressed(
                os.path.join(
                    save_intermediates_dir,
                    f"it{iteration:03d}_half{k + 1}_local_profile.npz",
                ),
                **local_profile_k,
            )
    if k_class_enabled:
        class_assignments_k = local_outputs.class_assignments
        class_posterior_sums_k = local_outputs.class_posterior_sums
        class_full_posterior_sums_k = local_outputs.class_full_posterior_sums
        outputs.class_assignments[k] = np.asarray(class_assignments_k, dtype=np.int32)
        outputs.class_posterior[k] = np.asarray(class_posterior_sums_k, dtype=np.float64)
        if outputs.class_full_posterior is not None:
            outputs.class_full_posterior[k] = np.asarray(class_full_posterior_sums_k, dtype=np.float64)
    pose_dtype = _dense_global_scoring_dtype()
    outputs.best_pose_rotations[k] = np.asarray(best_rots_k, dtype=pose_dtype)
    outputs.best_pose_rotation_eulers[k] = utils.R_to_relion(
        np.asarray(best_rots_k),
        degrees=True,
    ).astype(pose_dtype)
    outputs.best_pose_translations[k] = np.asarray(best_trans_k, dtype=pose_dtype)
    return HalfScoreResult(
        ha=ha_k,
        Ft_y=Ft_y_k,
        Ft_ctf=Ft_ctf_k,
        em_stats=em_stats_k,
        noise_stats=noise_stats_k,
        best_pose_rotations=outputs.best_pose_rotations[k],
        best_pose_rotation_eulers=outputs.best_pose_rotation_eulers[k],
        best_pose_translations=outputs.best_pose_translations[k],
        significant_counts=relion_significant_counts_k,
        mstep_full_half_axis=0 if local_relion_x_half_mstep else None,
        mstep_accumulator_shape=(
            # Must match the current-size BPref grid allocated by the local
            # engine above; downstream join/reconstruct calls infer layout
            # from this shape.
            relion_backprojector_volume_shape(
                experiment_dataset.volume_shape,
                PADDING_FACTOR,
                current_size=reconstruction_current_size_for_engine,
            )
            if local_relion_x_half_mstep
            else None
        ),
    )


def _score_half_local_in_bpref_scope(
    *,
    bpref_device_signature_active: bool,
    **kwargs,
) -> HalfScoreResult:
    """Run local scoring with device-capture flags disabled or fail closed."""

    if bpref_device_signature_active:
        raise RuntimeError(
            "BPref device signature capture is supported only by sparse adaptive pass 2"
        )
    with _cuda_backproject_diagnostics.bpref_device_signature_scope(False):
        return _score_half_local(**kwargs)
