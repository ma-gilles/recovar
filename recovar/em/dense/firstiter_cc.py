"""RELION first-iteration adaptive winner-take-all dispatch.

Batch budgets and adaptive pass plans are owned by ``batch_planning``.
"""

from __future__ import annotations

import logging

import numpy as np

from recovar.em.classification.k_class import _sparse_pass2_selected, run_dense_k_class_em_adaptive
from recovar.em.helpers.batch_planning import (
    _plan_kclass_adaptive_grid_batch_sizes,
    _safe_dense_k_class_rotation_block_size,
    _safe_firstiter_cc_image_batch_size,
)
from recovar.em.helpers.oversampling import build_adaptive_pass2_grids
from recovar.em.sampling import (
    apply_relion_translation_perturbation,
)


def _score_kclass_firstiter_cc_pass2(
    *,
    logger: logging.Logger,
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance_k,
    effective_rotations,
    current_translations,
    base_translations,
    current_healpix_order: int,
    state,
    random_perturbation,
    disc_type,
    class_log_priors,
    image_batch_size: int,
    image_shape_k,
    em_kwargs: dict,
    safe_batch_sizes=None,
    coarse_current_size: int | None = None,
    fine_current_size: int | None = None,
    log_label: str = "",
    update_em_kwargs_image_batch_size: bool = False,
    bpref_device_signature_active: bool = False,
    debug_iteration: int | None = None,
    coarse_rotation_ids=None,
):
    """RELION iter-1 ``--firstiter_cc`` adaptive two-pass dispatch.

    Build coarse/fine grids and invoke the K-class engine with normalized-CC
    scoring through the global coarse winner subset. The winner-take-all
    policy applies to M-step support as well as reported Pmax.

    The K-class and K=1 adaptive scoring branches share this dispatcher.
    ``update_em_kwargs_image_batch_size`` controls whether the batch clamp
    also updates the caller's dictionary; the engine receives a clamped copy
    either way. Coarse/fine size overrides are forwarded only when supplied.
    ``log_label`` identifies the caller in routing messages.

    Return ``(k_class_result, rot_pmap, trans_pmap, n_trans_fine, adaptive_os)``.
    """

    adaptive_os_local = int(state.adaptive_oversampling)
    (
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        fine_mstep_rot,
    ) = build_adaptive_pass2_grids(
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
    n_classes = int(np.asarray(mean).shape[0]) if np.asarray(mean).ndim >= 2 else 1
    firstiter_significance_image_batch_size = None
    firstiter_significance_rotation_block_size = None
    firstiter_sparse_pass2 = _sparse_pass2_selected("RECOVAR_K_CLASS_DENSE_PASS2")
    if safe_batch_sizes is not None:
        batch_plan = _plan_kclass_adaptive_grid_batch_sizes(
            coarse_rotations=coarse_rot,
            coarse_translations=coarse_trans,
            fine_rotations=fine_rot,
            fine_translations=fine_trans,
            n_classes=n_classes,
            image_shape=image_shape_k,
            coarse_current_size=coarse_current_size if coarse_current_size is not None else em_kwargs.get("current_size"),
            fine_current_size=fine_current_size if fine_current_size is not None else em_kwargs.get("current_size"),
            safe_batch_sizes=safe_batch_sizes,
        )
        if firstiter_sparse_pass2:
            requested_firstiter_image_batch_size = int(em_kwargs.get("image_batch_size", image_batch_size))
            firstiter_image_batch_size = min(
                requested_firstiter_image_batch_size,
                _safe_firstiter_cc_image_batch_size(
                    fine_trans.shape[0],
                    image_shape_k,
                ),
            )
            firstiter_rotation_block_size = min(
                int(em_kwargs.get("rotation_block_size", batch_plan.pass2_rotation_block_size)),
                _safe_dense_k_class_rotation_block_size(
                    fine_trans.shape[0],
                    firstiter_image_batch_size,
                ),
            )
        else:
            firstiter_image_batch_size = batch_plan.pass2_image_batch_size
            firstiter_rotation_block_size = batch_plan.pass2_rotation_block_size
        firstiter_significance_image_batch_size = batch_plan.significance_image_batch_size
        firstiter_significance_rotation_block_size = batch_plan.significance_rotation_block_size
        logger.info(
            "STRICT-PARITY: iter-1 K-class adaptive batch sizing "
            "coarse image_batch_size=%d rotation_block_size=%d; "
            "fine image_batch_size=%d rotation_block_size=%d (%s pass2)",
            firstiter_significance_image_batch_size,
            firstiter_significance_rotation_block_size,
            firstiter_image_batch_size,
            firstiter_rotation_block_size,
            "sparse" if firstiter_sparse_pass2 else "dense",
        )
    else:
        requested_firstiter_image_batch_size = int(em_kwargs.get("image_batch_size", image_batch_size))
        firstiter_image_batch_size = min(
            requested_firstiter_image_batch_size,
            _safe_firstiter_cc_image_batch_size(
                fine_trans.shape[0],
                image_shape_k,
            ),
        )
        firstiter_rotation_block_size = int(em_kwargs.get("rotation_block_size", 5000))
        if firstiter_image_batch_size != requested_firstiter_image_batch_size:
            logger.info(
                "STRICT-PARITY: clamping iter-1 winner-take-all image_batch_size from %d to %d",
                requested_firstiter_image_batch_size,
                firstiter_image_batch_size,
            )
    if update_em_kwargs_image_batch_size:
        em_kwargs["image_batch_size"] = firstiter_image_batch_size
    firstiter_em_kwargs = dict(em_kwargs)
    firstiter_em_kwargs["image_batch_size"] = firstiter_image_batch_size
    firstiter_em_kwargs["rotation_block_size"] = firstiter_rotation_block_size
    firstiter_em_kwargs["sparse_pass2"] = firstiter_sparse_pass2
    logger.info(
        "STRICT-PARITY %srouting iter-1 K-class through %s run_dense_k_class_em_adaptive "
        "(oversampling=%d, relion_x_half_mstep=%s, best_coarse_subset=True)",
        log_label,
        "sparse" if firstiter_sparse_pass2 else "dense",
        adaptive_os_local,
        bool(firstiter_em_kwargs.get("mstep_relion_x_half", False)),
    )
    extra: dict = {}
    if coarse_current_size is not None:
        extra["coarse_current_size"] = coarse_current_size
    if fine_current_size is not None:
        extra["fine_current_size"] = fine_current_size
    k_class_result = run_dense_k_class_em_adaptive(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance_k,
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        disc_type,
        class_log_priors=class_log_priors,
        accumulate_noise=True,
        return_best_pose_details=True,
        firstiter_cc_pass2_only_best_coarse=True,
        skip_significance_pruning=False,
        relion_fine_mstep_prune=True,
        significance_image_batch_size=firstiter_significance_image_batch_size,
        significance_rotation_block_size=firstiter_significance_rotation_block_size,
        coarse_healpix_order=int(current_healpix_order),
        coarse_rotation_ids=coarse_rotation_ids,
        oversampling_order=int(adaptive_os_local),
        fine_mstep_rotations_override=(fine_mstep_rot if firstiter_sparse_pass2 else None),
        bpref_device_signature_active=bpref_device_signature_active,
        debug_iteration=debug_iteration,
        coarse_translation_phase_source=(
            coarse_translation_phase_source if n_classes == 1 else None
        ),
        **extra,
        **firstiter_em_kwargs,
    )
    return k_class_result, rot_pmap, trans_pmap, int(fine_trans.shape[0]), adaptive_os_local
