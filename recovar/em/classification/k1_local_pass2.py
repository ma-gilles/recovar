"""K=1 adaptive pass 2 through the exact local engine (opt-in, shape-stable route).

The compact engine (``sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed``)
stays the default and the numerical oracle.  With ``RECOVAR_K1_PASS2_ENGINE=local``
the same significant coarse samples are laid out as a per-image local hypothesis
set (``build_pass2_hypothesis_layout``) and scored, normalised and back-projected
by ``run_local_em_exact`` in its flat-row, stable-capacity, stable-Fourier-window
configuration, which is the shape-stable K=1 topology the InitialModel driver
already runs.

The candidate set, rotation matrices, M-step rotations, fine translations and
priors are taken from the same pass-2 grids the compact engine receives, so both
routes score identical hypotheses; pose ids are the compact engine's
parent-major fine-grid ids, so the caller's parent maps collapse them unchanged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import numpy as np

from recovar.em.helpers.batch_planning import _estimate_relion_em_batch_sizes
from recovar.em.helpers.env_flags import parse_env_flag
from recovar.em.local.local_layout import LocalHypothesisLayout, build_pass2_hypothesis_layout
from recovar.em.sampling import get_oversampled_translation_grid, infer_translation_step

logger = logging.getLogger(__name__)

K1_PASS2_ENGINE_ENV = "RECOVAR_K1_PASS2_ENGINE"
_FLAT_ROWS_ENV = "RECOVAR_K1_LOCAL_PASS2_FLAT_ROWS"
_STABLE_ROW_CAPACITY_ENV = "RECOVAR_K1_LOCAL_PASS2_STABLE_ROW_CAPACITY"
_PACKED_PROJECTION_ENV = "RECOVAR_K1_LOCAL_PASS2_PACKED_PROJECTION"
_STABLE_WINDOWS_ENV = "RECOVAR_K1_LOCAL_PASS2_STABLE_WINDOWS"
_FUSED_PAIR_SCORE_ENV = "RECOVAR_K1_LOCAL_PASS2_FUSED_PAIR_SCORE"
_UNIFY_BUCKET_SIZES_ENV = "RECOVAR_K1_LOCAL_PASS2_UNIFY_BUCKET_SIZES"


def k1_local_pass2_engine_selected() -> bool:
    """Return whether the K=1 adaptive pass 2 runs on the exact local engine."""
    value = os.environ.get(K1_PASS2_ENGINE_ENV, "compact").strip().lower()
    if value not in {"compact", "local"}:
        raise ValueError(f"{K1_PASS2_ENGINE_ENV} must be 'compact' or 'local', got {value!r}")
    return value == "local"


def k1_local_pass2_execution_flags() -> dict:
    """Shape-stable local-engine modes for the K=1 route (all default on but fused pairs)."""
    flat_rows = parse_env_flag(_FLAT_ROWS_ENV, default=True)
    return {
        "_flat_local_rows_enabled": flat_rows,
        "_stable_flat_row_capacity_enabled": flat_rows
        and parse_env_flag(_STABLE_ROW_CAPACITY_ENV, default=True),
        "_packed_local_projection_enabled": flat_rows
        and parse_env_flag(_PACKED_PROJECTION_ENV, default=True),
        "stable_fourier_window_shapes": parse_env_flag(_STABLE_WINDOWS_ENV, default=True),
        "fused_pair_fine_score": flat_rows and parse_env_flag(_FUSED_PAIR_SCORE_ENV, default=False),
        "unify_local_bucket_sizes": parse_env_flag(_UNIFY_BUCKET_SIZES_ENV, default=True),
    }


def coarse_prior_from_pass2_prior(prior, *, n_rot_coarse: int, children_per_parent: int):
    """Return the coarse-grid rotation prior behind a parent-major fine-grid prior.

    ``_run_sparse_k_class_adaptive_pass2`` receives the rotation prior already
    broadcast onto the fine grid (``prior[rot_parent_map]``).  The layout builder
    indexes the prior by coarse rotation id, so take the first child of every
    parent, which carries exactly the parent's value.
    """
    if prior is None:
        return None
    prior_np = np.asarray(prior)
    n_fine = int(n_rot_coarse) * int(children_per_parent)
    if prior_np.shape[-1] == int(n_rot_coarse):
        return prior_np
    if prior_np.shape[-1] != n_fine:
        raise ValueError(
            f"rotation prior must span {n_rot_coarse} coarse or {n_fine} fine rotations, "
            f"got {prior_np.shape}"
        )
    first_children = np.arange(int(n_rot_coarse), dtype=np.int64) * int(children_per_parent)
    return prior_np[..., first_children]


def align_layout_to_pass2_grids(
    layout: LocalHypothesisLayout,
    *,
    fine_rotations: np.ndarray,
    fine_mstep_rotations: np.ndarray | None,
    fine_translations: np.ndarray,
    fine_source_eulers: np.ndarray | None,
    children_per_parent: int,
    n_rot_coarse: int,
) -> LocalHypothesisLayout:
    """Give the layout the compact engine's parent-major fine ids and its exact grids.

    ``build_pass2_hypothesis_layout`` stores nearest fine-HEALPix ids and
    regenerates child rotations; the compact engine indexes the caller's
    ``fine_rotations`` (row ``parent * children + child``) and uses the caller's
    perturbed fine translations and M-step rotations.  Every image block lists
    its parents in ascending order with the children contiguous, so the child
    offset of row ``r`` is ``r mod children``.  The regenerated rotations must
    equal the gathered rows; a mismatch means the two grids disagree and the
    route fails closed.
    """
    children = int(children_per_parent)
    n_rows = int(layout.rotation_ids_flat.shape[0])
    if n_rows % children:
        raise ValueError("pass-2 layout rows are not whole parent blocks")
    parents = np.asarray(layout.rotation_posterior_ids_flat, dtype=np.int64)
    if parents.shape != (n_rows,):
        raise ValueError("pass-2 layout lacks per-row coarse parent ids")
    if n_rows and (parents.min() < 0 or parents.max() >= int(n_rot_coarse)):
        raise ValueError("pass-2 layout parent ids are outside the coarse grid")
    ids = parents * children + (np.arange(n_rows, dtype=np.int64) % children)
    fine_rotations = np.asarray(fine_rotations)
    if fine_rotations.shape != (int(n_rot_coarse) * children, 3, 3):
        raise ValueError(
            f"fine rotations must be the full parent-major children grid, got {fine_rotations.shape}"
        )
    gathered = fine_rotations[ids].astype(layout.rotations_flat.dtype, copy=False)
    if not np.array_equal(gathered, np.asarray(layout.rotations_flat)):
        raise RuntimeError(
            "pass-2 layout rotations differ from the caller's fine rotation grid; "
            "the local route cannot claim the compact engine's candidate set"
        )
    updates = dict(
        rotation_ids_flat=ids,
        rotations_flat=gathered,
        translation_grid=np.asarray(fine_translations, dtype=layout.translation_grid.dtype),
    )
    if fine_mstep_rotations is not None:
        updates["mstep_rotations_flat"] = np.asarray(fine_mstep_rotations)[ids].astype(
            layout.rotations_flat.dtype, copy=False
        )
    if fine_source_eulers is not None:
        updates["source_eulers_flat"] = np.asarray(fine_source_eulers, dtype=np.float64)[ids]
    if updates["translation_grid"].shape != layout.translation_grid.shape:
        raise ValueError(
            "fine translation grid shape mismatch: "
            f"{updates['translation_grid'].shape} vs {layout.translation_grid.shape}"
        )
    return replace(layout, **updates)


def run_k1_local_adaptive_pass2(
    experiment_dataset,
    mean,
    noise_variance,
    coarse_translations_np,
    sig_sample_indices,
    disc_type: str,
    *,
    n_rot_coarse: int,
    healpix_order: int,
    oversampling_order: int,
    random_perturbation: float,
    fine_rotations_np,
    fine_mstep_rotations_np,
    rot_parent_map_np,
    fine_translations_np,
    trans_parent_map_np,
    fine_source_eulers,
    class_log_prior: float,
    common: dict,
    engine_kwargs: dict,
    accumulate_noise: bool,
    return_best_pose_details: bool,
    mstep_accumulator_shape,
):
    """Score the K=1 sparse pass 2 on the exact local engine and return a K-class result."""
    from recovar.em.classification.k_class import run_local_k_class_em

    if not common.get("relion_x_half_mstep", False):
        raise NotImplementedError("the local K=1 pass-2 route requires the RELION x-half M-step")
    if common.get("relion_firstiter_score_mode", "gaussian") != "gaussian" or common.get(
        "relion_firstiter_winner_take_all", False
    ):
        raise NotImplementedError("the local K=1 pass-2 route supports the Gaussian pass 2 only")
    if not common.get("relion_exact_fine_gaussian", True):
        raise NotImplementedError("the local K=1 pass-2 route requires exact RELION fine scoring")
    if common.get("relion_f32_normalization_sum_weight") is not None:
        raise NotImplementedError("the local K=1 pass-2 route does not reuse the coarse normalisation")
    children = 8 ** int(oversampling_order)
    n_rot_coarse = int(n_rot_coarse)
    n_coarse_trans = int(np.asarray(coarse_translations_np).shape[0])
    rot_parent_map_np = np.asarray(rot_parent_map_np, dtype=np.int64)
    if not np.array_equal(rot_parent_map_np, np.repeat(np.arange(n_rot_coarse), children)):
        raise ValueError("the local K=1 pass-2 route needs the parent-major full children grid")
    fine_translations_np = np.asarray(fine_translations_np)
    n_fine_trans = int(fine_translations_np.shape[0])
    translation_step = infer_translation_step(np.asarray(coarse_translations_np, dtype=np.float64))
    _, expected_trans_parent = get_oversampled_translation_grid(
        np.asarray(coarse_translations_np, dtype=np.float64),
        float(translation_step),
        oversampling_order=int(oversampling_order),
    )
    if not np.array_equal(np.asarray(expected_trans_parent, dtype=np.int64), np.asarray(trans_parent_map_np, dtype=np.int64)):
        raise ValueError("fine translation parent map differs from the oversampled coarse grid")

    dtype = np.float64 if common.get("use_float64_scoring", False) else np.float32
    rotation_prior = coarse_prior_from_pass2_prior(
        common.get("rotation_log_prior"),
        n_rot_coarse=n_rot_coarse,
        children_per_parent=children,
    )
    if rotation_prior is not None:
        rotation_prior = np.asarray(rotation_prior, dtype=dtype)
        if rotation_prior.ndim != 1:
            raise NotImplementedError("the local K=1 pass-2 route takes a shared rotation prior only")
    translation_prior = common.get("translation_log_prior")
    layout_prior_kwargs = {}
    if translation_prior is not None:
        translation_prior = np.asarray(translation_prior, dtype=dtype)
        if translation_prior.shape[-1] == n_fine_trans:
            layout_prior_kwargs["fine_translation_log_prior"] = translation_prior
        elif translation_prior.shape[-1] == n_coarse_trans:
            layout_prior_kwargs["translation_log_prior"] = translation_prior
        else:
            raise ValueError(f"translation prior spans neither grid: {translation_prior.shape}")

    layout = build_pass2_hypothesis_layout(
        sig_sample_indices,
        n_rot_coarse,
        n_coarse_trans,
        int(healpix_order),
        np.asarray(coarse_translations_np, dtype=dtype),
        oversampling_order=int(oversampling_order),
        translation_step=float(translation_step),
        rotation_log_prior=rotation_prior,
        random_perturbation=float(random_perturbation),
        allow_empty=True,
        dtype=dtype,
        **layout_prior_kwargs,
    )
    layout = align_layout_to_pass2_grids(
        layout,
        fine_rotations=np.asarray(fine_rotations_np, dtype=dtype),
        fine_mstep_rotations=None if fine_mstep_rotations_np is None else np.asarray(fine_mstep_rotations_np, dtype=dtype),
        fine_translations=fine_translations_np,
        fine_source_eulers=fine_source_eulers,
        children_per_parent=children,
        n_rot_coarse=n_rot_coarse,
    )

    padding_factor = max(
        int(common.get("projection_padding_factor", 1)),
        int(common.get("reconstruction_padding_factor", 1)),
        1,
    )
    batch_plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=int(engine_kwargs.get("image_batch_size", 64)),
        requested_rotation_block_size=int(engine_kwargs.get("rotation_block_size", 4096)),
        n_rot=max(1, int(np.max(layout.rotation_counts)) if layout.n_images else 1),
        n_trans=n_fine_trans,
        image_shape=experiment_dataset.image_shape,
        volume_shape=experiment_dataset.volume_shape,
        padding_factor=padding_factor,
        n_classes=1,
        current_size=common.get("current_size"),
    )
    execution_flags = k1_local_pass2_execution_flags()
    prune = bool(common.get("relion_fine_mstep_prune", False))
    logger.info(
        "K=1 pass 2 on the exact local engine: images=%d rows=%d (median/max per image %d/%d) "
        "fine_trans=%d image_batch_size=%d rotation_block_size=%d flags=%s",
        layout.n_images,
        layout.total_local_rotations,
        int(np.median(layout.rotation_counts)) if layout.n_images else 0,
        int(np.max(layout.rotation_counts)) if layout.n_images else 0,
        n_fine_trans,
        batch_plan.image_batch_size,
        batch_plan.rotation_block_size,
        {key.lstrip("_"): value for key, value in execution_flags.items()},
    )
    mean_np = np.asarray(mean)
    means = mean_np if mean_np.ndim == 4 else mean_np[None]
    result = run_local_k_class_em(
        experiment_dataset,
        means,
        noise_variance,
        layout,
        disc_type,
        class_log_priors=[float(class_log_prior)],
        accumulate_noise=bool(accumulate_noise),
        return_best_pose_details=bool(return_best_pose_details),
        stats_use_reconstruction_probs=prune,
        image_batch_size=int(batch_plan.image_batch_size),
        rotation_block_size=int(batch_plan.rotation_block_size),
        current_size=common.get("current_size"),
        reconstruction_current_size=common.get("reconstruction_current_size"),
        projection_padding_factor=int(common.get("projection_padding_factor", 1)),
        reconstruction_padding_factor=int(common.get("reconstruction_padding_factor", 1)),
        score_with_masked_images=bool(common.get("score_with_masked_images", False)),
        half_spectrum_scoring=bool(common.get("half_spectrum_scoring", False)),
        use_float64_scoring=bool(common.get("use_float64_scoring", False)),
        use_float64_normalization=True,
        use_float64_projections=bool(engine_kwargs.get("use_float64_projections", False)),
        do_gridding_correction=bool(common.get("do_gridding_correction", False)),
        square_window=bool(common.get("square_window", False)),
        recon_exact_radius=bool(engine_kwargs.get("recon_exact_radius", True)),
        image_corrections=common.get("image_corrections"),
        scale_corrections=common.get("scale_corrections"),
        group_ids=common.get("group_ids"),
        scale_correction_group_count=common.get("scale_correction_group_count"),
        scale_correction_data_vs_prior=common.get("scale_correction_data_vs_prior"),
        image_pre_shifts=common.get("image_pre_shifts"),
        translation_prior_centers=common.get("translation_prior_centers"),
        mstep_subtract_ctf_projection=bool(common.get("mstep_subtract_ctf_projection", False)),
        mstep_relion_x_half=True,
        reconstruct_significant_only=prune,
        adaptive_fraction=float(common.get("adaptive_fraction", 0.999)),
        max_significants=-1,
        relion_f32_fine_posterior=bool(common.get("relion_f32_fine_posterior", False)),
        relion_projector_half=common.get("relion_projector_half"),
        relion_projector_r_max=common.get("relion_projector_r_max"),
        projection_mask_current_image_disk=bool(common.get("projection_mask_current_image_disk", True)),
        relion_exact_bpref_operands=True,
        relion_exact_fine_diff2=True,
        relion_exact_score_translation=True,
        relion_wavg_sequential_cuda=True,
        preserve_bpref_particle_order=bool(common.get("preserve_bpref_particle_order", False)),
        include_unweighted_norm_high_shell=True,
        source_faithful_spectrum_norm=bool(common.get("source_faithful_spectrum_norm", False)),
        debug_iteration=engine_kwargs.get("debug_iteration"),
        **execution_flags,
    )
    return result._replace(
        mstep_full_half_axis=0,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )
