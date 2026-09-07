"""RELION first-iteration coarse/fine grids and their parent maps.

Batch budgets and adaptive pass plans are owned by ``batch_planning``.
"""

from __future__ import annotations

import numpy as np

from recovar.em.sampling import (
    apply_relion_translation_perturbation,
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
)

def _build_firstiter_cc_pass2_grids(
    coarse_rotations,
    coarse_translations,
    base_translations,
    coarse_healpix_order: int,
    adaptive_oversampling: int,
    translation_step_px: float,
    random_perturbation: float,
    *,
    return_mstep_rotations: bool = False,
    coarse_rotation_ids=None,
):
    """Build (coarse, fine, parent_map) pose grids for K-class iter-1 firstiter_cc adaptive engine.

    Mirrors run_k_class_parity.py's adaptive 2-pass grid construction (lines 832-855).
    With ``adaptive_oversampling==0`` returns identity parent maps + coarse-as-fine
    so the engine still goes through the firstiter_cc_pass2_only_best_coarse logic
    but the fine grid is the coarse grid (1 child per parent). With
    ``adaptive_oversampling>0`` builds the proper HEALPix-subdivided fine rotation
    grid (8x children per parent at order=1) and oversampled translation grid
    (4x children per parent at order=1), applies RELION SamplingPerturbation to
    the fine translation grid, returns parent_maps that index from fine to
    coarse.
    """
    coarse_rot_np = np.asarray(coarse_rotations, dtype=np.float32)
    coarse_trans_np = np.asarray(coarse_translations, dtype=np.float32)
    base_translations_f64 = np.asarray(base_translations, dtype=np.float64)
    if int(adaptive_oversampling) <= 0:
        n_rot = int(coarse_rot_np.shape[0])
        n_trans = int(coarse_trans_np.shape[0])
        rot_parent_map = np.arange(n_rot, dtype=np.int64)
        trans_parent_map = np.arange(n_trans, dtype=np.int64)
        fine_translations = apply_relion_translation_perturbation(
            base_translations_f64,
            float(random_perturbation),
            float(translation_step_px),
        )
        outputs = (
            coarse_rot_np,
            coarse_trans_np,
            coarse_rot_np,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
        )
        if return_mstep_rotations:
            return (*outputs, coarse_rot_np.copy())
        return outputs

    adaptive_os = int(adaptive_oversampling)
    all_coarse_rot_indices = (
        np.arange(int(coarse_rot_np.shape[0]), dtype=np.int64)
        if coarse_rotation_ids is None
        else np.asarray(coarse_rotation_ids, dtype=np.int64)
    )
    if all_coarse_rot_indices.shape != (int(coarse_rot_np.shape[0]),):
        raise ValueError(
            "coarse_rotation_ids must identify every supplied coarse rotation exactly once"
        )
    fine_rotation_outputs = get_oversampled_rotation_grid_from_samples(
        all_coarse_rot_indices,
        parent_nside_level=int(coarse_healpix_order),
        oversampling_order=adaptive_os,
        random_perturbation=float(random_perturbation),
        return_mstep_rotations=return_mstep_rotations,
    )
    fine_rotations, rot_parent_map = fine_rotation_outputs[:2]
    fine_mstep_rotations = fine_rotation_outputs[2] if return_mstep_rotations else None
    fine_rotations = np.asarray(fine_rotations, dtype=np.float32)
    if fine_mstep_rotations is not None:
        fine_mstep_rotations = np.asarray(fine_mstep_rotations, dtype=np.float32)
    rot_parent_map = np.asarray(rot_parent_map, dtype=np.int64)

    fine_base_translations, trans_parent_map = get_oversampled_translation_grid(
        base_translations_f64,
        float(translation_step_px),
        oversampling_order=adaptive_os,
    )
    fine_translations = apply_relion_translation_perturbation(
        fine_base_translations,
        float(random_perturbation),
        float(translation_step_px),
    )
    trans_parent_map = np.asarray(trans_parent_map, dtype=np.int64)

    outputs = (
        coarse_rot_np,
        coarse_trans_np,
        fine_rotations,
        fine_translations,
        rot_parent_map,
        trans_parent_map,
    )
    if return_mstep_rotations:
        return (*outputs, fine_mstep_rotations)
    return outputs
