"""K=1 BnB EM driver: support selection -> LocalHypothesisLayout -> run_local_em_exact.

This is the public Phase-2 engine: call ``run_bnb_em_k1`` exactly where
you would call ``run_em``. Internally it:
  1. Runs ``select_bnb_support_fixed_grid_k1`` over the global pose grid.
  2. Packs the survivors into a ``LocalHypothesisLayout``.
  3. Calls ``run_local_em_exact`` for the exact final E-step + M-step + noise
     accumulation at the requested ``current_size``.

The return tuple is shaped to match ``run_em`` (when ``return_stats=True,
accumulate_noise=True``): ``(new_mean, hard_assignment, Ft_y, Ft_ctf,
em_stats, noise_stats)``. K-class is not supported in Phase 2.
"""

from __future__ import annotations

import logging
import time

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact
from recovar.em.dense_single_volume.local_layout import bucket_local_hypothesis_layout

from .hierarchical_support import select_bnb_support_hierarchical_k1
from .layout import build_bnb_local_layout
from .options import BranchBoundOptions
from .per_image_engine import run_paper_faithful_bnb_em_k1
from .support import select_bnb_support_fixed_grid_k1

logger = logging.getLogger(__name__)


def run_bnb_em_k1(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    rotations: np.ndarray,
    translations: jnp.ndarray,
    disc_type: str,
    *,
    current_size: int | None,
    options: BranchBoundOptions,
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    image_corrections: np.ndarray | None = None,
    scale_corrections: np.ndarray | None = None,
    image_pre_shifts: np.ndarray | None = None,
    translation_prior_centers: np.ndarray | None = None,
    accumulate_noise: bool = True,
    return_best_pose_details: bool = True,
    half_spectrum_scoring: bool = False,
    use_float64_scoring: bool = False,
    use_float64_projections: bool = False,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    score_with_masked_images: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
    prior_rotations: np.ndarray | None = None,
    prior_translations: np.ndarray | None = None,
):
    """Phase-2 BnB driver for K=1 EM refinement on a fixed global pose grid.

    Parameters mirror those of ``run_em`` / ``run_local_em_exact`` so this
    can drop into ``_score_half_bnb_k1`` (added in Phase 4). For Phase 2 we
    expose a minimal kwargs surface; additional options (projection
    padding, M-step ablations, K-class) are deferred to later phases.
    """
    t0 = time.time()

    if options.subdivision_mode == "paper_faithful":
        return run_paper_faithful_bnb_em_k1(
            experiment_dataset, mean, mean_variance, noise_variance,
            current_size=current_size,
            options=options,
            disc_type=disc_type,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            translation_prior_centers=translation_prior_centers,
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            half_spectrum_scoring=half_spectrum_scoring,
            score_with_masked_images=score_with_masked_images,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            prior_rotations=prior_rotations,
            prior_translations=prior_translations,
        )

    if options.subdivision_mode == "axis_angle_hierarchical":
        support, rotations_final, translations_final = select_bnb_support_hierarchical_k1(
            experiment_dataset,
            mean,
            noise_variance,
            max_shift_px=float(options.max_shift_px),
            current_size=current_size,
            options=options,
            disc_type=disc_type,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
        )
        rotations_for_layout = rotations_final
        translations_for_layout = translations_final
        # Priors: hierarchical paths build their own pose grid, so caller-
        # supplied per-rotation/per-translation priors don't apply 1:1. Pass
        # uniform priors for now; per-class prior plumbing is a follow-up.
        rotation_log_prior_for_layout = None
        translation_log_prior_for_layout = None
    else:
        support = select_bnb_support_fixed_grid_k1(
            experiment_dataset,
            mean,
            noise_variance,
            np.asarray(rotations, dtype=np.float32),
            jnp.asarray(translations),
            current_size=current_size,
            options=options,
            disc_type=disc_type,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
        )
        rotations_for_layout = np.asarray(rotations, dtype=np.float32)
        translations_for_layout = np.asarray(translations, dtype=np.float32)
        rotation_log_prior_for_layout = rotation_log_prior
        translation_log_prior_for_layout = translation_log_prior

    t_support = time.time() - t0

    layout = build_bnb_local_layout(
        support,
        rotations_for_layout,
        translations_for_layout,
        rotation_log_prior=rotation_log_prior_for_layout,
        translation_log_prior=translation_log_prior_for_layout,
    )
    bucketed = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
    )

    t_layout = time.time() - t0 - t_support

    logger.info(
        "BnB engine_k1: support survivors mean=%.1f max=%d (n_images=%d) "
        "[support %.2fs, layout %.2fs]",
        support.diagnostics.candidates_final_mean,
        support.diagnostics.candidates_final_max,
        len(support.image_indices),
        t_support, t_layout,
    )

    # Delegate the final E-step and M-step to the local engine. Padding
    # factors default to 2 to match `iteration_loop.PROJECTION_PADDING_FACTOR`
    # so downstream `join_halves_at_low_resolution` receives the expected
    # 256³-style Ft_y shape (not the unpadded 128³).
    return run_local_em_exact(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        layout,
        disc_type,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        current_size=current_size,
        accumulate_noise=accumulate_noise,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        half_spectrum_scoring=half_spectrum_scoring,
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
        score_with_masked_images=score_with_masked_images,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        translation_prior_centers=translation_prior_centers,
        return_best_pose_details=return_best_pose_details,
    )
