"""RELION ``--firstiter_cc`` iter-1 helpers.

Extracted verbatim from ``iteration_loop.py``: bucket-batch caps that
defend the dense K-class reconstruction tensor footprint at iter-1, plus
the (coarse, fine, parent map) builder shared by the K-class STRICT-PARITY
routes.
"""

from __future__ import annotations

import numpy as np

from recovar.em.sampling import (
    apply_relion_translation_perturbation,
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
)

# Mirrors iteration_loop's module-level constants so monkeypatches at either
# module level still bind correctly.
RELION_FIRSTITER_RECON_COMPLEX_BUDGET = 256_000_000
RELION_DENSE_K_CLASS_HYPOTHESES_BUDGET = 8_000_000


def _safe_firstiter_cc_image_batch_size(n_trans, image_shape):
    """Cap dense K-class reconstruction batches by the temporary footprint.

    ``prepare_reconstruction_batch`` materializes a
    ``batch_size × n_trans × n_half`` complex tensor before any class or
    pose masking can trim anything.  The generic score-tensor budget does
    not account for that temporary, so dense K-class runs that keep
    ``score_with_masked_images=True`` need a separate clamp.  The
    first-iteration winner-take-all route is the most obvious case, but
    the same bound also protects later dense K-class iterations that reuse
    the same reconstruction path.
    """

    n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    return max(1, RELION_FIRSTITER_RECON_COMPLEX_BUDGET // max(int(n_trans) * n_half, 1))


def _safe_dense_k_class_rotation_block_size(n_trans, image_batch_size):
    """Cap dense K-class rotation buckets by a microbatch hypothesis budget.

    The dense K-class adaptive probe still has to evaluate a dense (batch,
    rotation, translation) tensor in its big-JIT path.  RELION's own
    bucketed pass2 code keeps similar hypothesis tensors under a ~2e6
    per-microbatch ceiling, so mirror that bound here to keep the probe
    memory-safe without changing the score math.
    """

    return max(
        64,
        RELION_DENSE_K_CLASS_HYPOTHESES_BUDGET // max(int(image_batch_size) * max(int(n_trans), 1), 1),
    )


def _build_firstiter_cc_pass2_grids(
    coarse_rotations,
    coarse_translations,
    base_translations,
    coarse_healpix_order: int,
    adaptive_oversampling: int,
    translation_step_px: float,
    random_perturbation: float,
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
    if int(adaptive_oversampling) <= 0:
        n_rot = int(coarse_rot_np.shape[0])
        n_trans = int(coarse_trans_np.shape[0])
        rot_parent_map = np.arange(n_rot, dtype=np.int64)
        trans_parent_map = np.arange(n_trans, dtype=np.int64)
        return (
            coarse_rot_np,
            coarse_trans_np,
            coarse_rot_np,
            coarse_trans_np,
            rot_parent_map,
            trans_parent_map,
        )

    adaptive_os = int(adaptive_oversampling)
    all_coarse_rot_indices = np.arange(int(coarse_rot_np.shape[0]), dtype=np.int64)
    fine_rotations, rot_parent_map = get_oversampled_rotation_grid_from_samples(
        all_coarse_rot_indices,
        parent_nside_level=int(coarse_healpix_order),
        oversampling_order=adaptive_os,
        random_perturbation=float(random_perturbation),
    )
    fine_rotations = np.asarray(fine_rotations, dtype=np.float32)
    rot_parent_map = np.asarray(rot_parent_map, dtype=np.int64)

    fine_base_translations, trans_parent_map = get_oversampled_translation_grid(
        np.asarray(base_translations, dtype=np.float32),
        float(translation_step_px),
        oversampling_order=adaptive_os,
    )
    fine_translations = apply_relion_translation_perturbation(
        fine_base_translations,
        float(random_perturbation),
        float(translation_step_px),
    ).astype(np.float32)
    trans_parent_map = np.asarray(trans_parent_map, dtype=np.int64)

    return (
        coarse_rot_np,
        coarse_trans_np,
        fine_rotations,
        fine_translations,
        rot_parent_map,
        trans_parent_map,
    )
