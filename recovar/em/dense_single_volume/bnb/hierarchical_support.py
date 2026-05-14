"""Paper-faithful hierarchical BnB pose-refinement selector for K=1.

Implements cryoSPARC's "Subdivision scheme" (Punjani 2017 Suppl Note 2):
the axis-angle Cartesian grid and the 2D shift Cartesian grid are
subdivided per BnB stage, and the score upper bound at radius L =
12 · 2^j is used to prune candidates.

The fixed-grid Phase-2 ``select_bnb_support_fixed_grid_k1`` runs progressive
L pruning over whatever (rotations, translations) grid the caller passes
(typically the existing recovar HEALPix grid). This module *builds its own*
pose tree from scratch following the paper schedule:

  stage 0:  axis-angle spacing = 24 deg, shift spacing = 5 px, L = 12.
  stage j:  spacing halved per axis, L = min(2*L_{j-1}, L_max).

Per-image survivor masks evolve through subdivision: when image i kept the
(parent_axis, parent_shift) cell at stage j-1, at stage j it inherits all
8 * 4 = 32 children of that cell as fresh candidates.

The grid itself is shared across all images at every stage — we take the
UNION of all images' surviving parent cells before subdividing. This keeps
the implementation simple and lets us reuse the dense
``_score_rotation_block`` scoring kernel. A per-image-ragged variant
(separate axis-angle / shift grid per image) is a follow-up speed
optimisation that doesn't change the algorithm's correctness.
"""

from __future__ import annotations

import logging
import time

import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
from recovar.em.dense_single_volume.helpers.preprocessing import preprocess_batch
from recovar.em.dense_single_volume.helpers.projection import compute_projections_block
from recovar.em.dense_single_volume.helpers.scoring import _score_rotation_block
from recovar.reconstruction import noise as noise_utils


def _to_half_noise(noise_variance, image_shape):
    """Promote a raw (image_size,) noise variance to packed-half shape."""
    nv = jnp.asarray(noise_variance)
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    if int(nv.shape[-1]) == n_half:
        return nv
    return noise_utils.to_batched_half_pixel_noise(nv, image_shape).squeeze()

from .axis_angle_grid import (
    AxisAngleGridLevel,
    make_initial_axis_angle_grid,
    subdivide_axis_angle_cells,
)
from .bounds import compute_high_model_pmax_per_image
from .diagnostics import BnBDiagnostics, BnBStageDiagnostics
from .frequency import (
    make_bnb_frequency_schedule,
    make_bnb_high_indices_np,
    make_bnb_low_window_spec,
)
from .options import BranchBoundOptions
from .pruning import prune_by_tail_mass_and_caps
from .shift_grid import ShiftGridLevel, make_initial_shift_grid, subdivide_shift_cells
from .support import BnBSupportResult

logger = logging.getLogger(__name__)


def _expand_sample_mask_to_children(
    parent_mask: np.ndarray,
    axis_parent_ids: np.ndarray,
    shift_parent_ids: np.ndarray,
) -> np.ndarray:
    """For each image, expand surviving (parent_axis, parent_shift) cells to
    all (child_axis, child_shift) pairs.

    Parameters
    ----------
    parent_mask : (n_images, n_axis_parent, n_shift_parent) bool
    axis_parent_ids : (n_axis_child,) int32 — parent of each child axis cell.
    shift_parent_ids : (n_shift_child,) int32 — parent of each child shift cell.

    Returns
    -------
    child_mask : (n_images, n_axis_child, n_shift_child) bool
    """
    # Gather parent_mask along the axis dim: result[i, c_a, p_s] = parent_mask[i, parent_of_a, p_s]
    parent_axis = np.asarray(axis_parent_ids, dtype=np.int64)
    parent_shift = np.asarray(shift_parent_ids, dtype=np.int64)
    # Broadcast: (n_images, n_axis_child, n_shift_parent) ← parent_mask[:, parent_axis, :]
    gathered_axis = parent_mask[:, parent_axis, :]
    # Then gather along the shift dim: (n_images, n_axis_child, n_shift_child)
    return gathered_axis[:, :, parent_shift]


def _gather_grid(level, *, kind: str):
    if kind == "axis":
        return np.asarray(level.rotations, dtype=np.float32), np.asarray(
            level.parent_ids if level.parent_ids is not None else np.arange(level.n_cells),
            dtype=np.int32,
        )
    elif kind == "shift":
        return np.asarray(level.centers, dtype=np.float32), np.asarray(
            level.parent_ids if level.parent_ids is not None else np.arange(level.n_cells),
            dtype=np.int32,
        )
    raise ValueError(kind)


def _score_at_low_freq(
    experiment_dataset,
    mean,
    rotations_global,
    translations_global,
    noise_variance,
    L: int,
    disc_type: str,
    image_batch_size: int,
    rotation_block_size: int,
    image_indices: np.ndarray,
) -> np.ndarray:
    """Dense score (n_images, n_rot, n_trans) at radius L. Shared grid across images."""
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    low_window = make_bnb_low_window_spec(image_shape, L, n_half)
    half_weights = make_half_image_weights(image_shape)
    precision_policy = DensePrecisionPolicy()

    n_rot = int(rotations_global.shape[0])
    n_trans = int(np.asarray(translations_global).shape[0])
    n_images_total = int(image_indices.shape[0])

    scores = np.empty((n_images_total, n_rot, n_trans), dtype=np.float32)
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    start = 0
    for batch in experiment_dataset.iter_batches(
        image_batch_size, indices=image_indices, by_image=False,
    ):
        batch_data = batch[0]
        ctf_params = batch[3]
        batch_size = int(jnp.asarray(batch_data).shape[0])
        end = start + batch_size

        shifted_half, batch_norm, ctf2_over_nv_half = preprocess_batch(
            experiment_dataset, jnp.asarray(batch_data), ctf_params,
            _to_half_noise(noise_variance, image_shape),
            jnp.asarray(translations_global), config, False,
        )
        shifted_windowed = low_window.score_values(shifted_half)
        ctf2_windowed = low_window.score_values(ctf2_over_nv_half)

        scores_batch = np.empty((batch_size, n_rot, n_trans), dtype=np.float32)
        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = min(r0 + rotation_block_size, n_rot)
            rot_block = rotations_global[r0:r1]
            if rot_block.shape[0] == 0:
                continue
            proj_half, proj_abs2_half = compute_projections_block(
                mean, rot_block, image_shape, volume_shape, disc_type,
                return_abs2=True,
            )
            block_scores = _score_rotation_block(
                low_window,
                shifted_score=shifted_windowed,
                batch_norm=batch_norm,
                score_weight=ctf2_windowed,
                proj_half=proj_half,
                proj_abs2_half=proj_abs2_half,
                half_weights=half_weights,
                n_images=batch_size,
                n_trans=n_trans,
                image_shape=image_shape,
                volume_shape=volume_shape,
                score_mode="gaussian",
                precision_policy=precision_policy,
            )
            scores_batch[:, r0:r1, :] = np.asarray(block_scores)[:, : (r1 - r0), :]

        scores[start:end] = scores_batch
        start = end

    return scores


def select_bnb_support_hierarchical_k1(
    experiment_dataset,
    mean,
    noise_variance,
    *,
    max_shift_px: float,
    current_size: int | None,
    options: BranchBoundOptions,
    disc_type: str = "linear_interp",
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    image_indices: np.ndarray | None = None,
) -> tuple[BnBSupportResult, np.ndarray, np.ndarray]:
    """Paper-faithful axis-angle + shift hierarchical BnB pose-refinement selector.

    Returns
    -------
    support : BnBSupportResult
        ``sample_mask_per_image`` indexes into the FINAL stage's
        (rotations, translations) — see the second and third return values.
    rotations_final : (n_rot_final, 3, 3) float32
        Final rotation grid (axis-angle subdivided to depth = n_subdivisions).
    translations_final : (n_trans_final, 2) float32
        Final shift grid in pixels.
    """
    image_shape = experiment_dataset.image_shape

    if image_indices is None:
        image_indices = np.arange(experiment_dataset.n_units, dtype=np.int32)
    else:
        image_indices = np.asarray(image_indices, dtype=np.int32)
    n_images = int(image_indices.shape[0])

    diag = BnBDiagnostics()
    L_schedule = make_bnb_frequency_schedule(current_size, image_shape, options)
    diag.L_schedule = np.asarray(L_schedule, dtype=np.int32)

    # Initial grids.
    axis_level: AxisAngleGridLevel = make_initial_axis_angle_grid(
        np.deg2rad(float(options.initial_angular_spacing_deg)),
    )
    shift_level: ShiftGridLevel = make_initial_shift_grid(
        float(options.initial_shift_spacing_px),
        max_shift_px=float(max_shift_px),
    )

    n_axis = axis_level.n_cells
    n_shift = shift_level.n_cells
    sample_mask = np.ones((n_images, n_axis, n_shift), dtype=bool)

    final_score_indices_np, _ = make_fourier_window_indices_np(
        image_shape,
        current_size if current_size is not None else image_shape[0],
        square=False, include_dc=False,
    )

    half_weights = make_half_image_weights(image_shape)
    omitted_mass = np.zeros(n_images, dtype=np.float32)
    best_score_upper = np.full(n_images, -np.inf, dtype=np.float32)

    n_subdivisions = int(options.n_subdivisions)
    L_max = L_schedule[-1]

    # Evaluation pass 0 ... n_subdivisions (= n_subdivisions+1 passes).
    for stage in range(n_subdivisions + 1):
        t_stage = time.time()
        L_idx = min(stage, len(L_schedule) - 1)
        L = L_schedule[L_idx]
        rotations_global = np.asarray(axis_level.rotations, dtype=np.float32)
        translations_global = np.asarray(shift_level.centers, dtype=np.float32)
        n_axis = axis_level.n_cells
        n_shift = shift_level.n_cells

        # High band relative to current refinement support.
        low_score_indices_np, _ = make_fourier_window_indices_np(
            image_shape, 2 * L, square=False, include_dc=False,
        )
        high_indices_np = make_bnb_high_indices_np(
            final_score_indices_np, low_score_indices_np,
        )

        # Score active candidates at L (dense, shared grid). Inactive
        # candidates get score -inf after this step via sample_mask.
        s_low = _score_at_low_freq(
            experiment_dataset, mean, rotations_global, translations_global,
            noise_variance, L, disc_type,
            image_batch_size, rotation_block_size, image_indices,
        )

        # Per-batch P^max over the active rotation cover (use global axis grid).
        config = ForwardModelConfig.from_dataset(
            experiment_dataset, disc_type=disc_type,
            process_fn=experiment_dataset.process_images,
        )
        pmax_per_image = np.empty(n_images, dtype=np.float32)
        start = 0
        for batch in experiment_dataset.iter_batches(
            image_batch_size, indices=image_indices, by_image=False,
        ):
            batch_data = batch[0]
            ctf_params = batch[3]
            batch_size = int(jnp.asarray(batch_data).shape[0])
            end = start + batch_size

            _, _, ctf2_over_nv_half = preprocess_batch(
                experiment_dataset, jnp.asarray(batch_data), ctf_params,
                _to_half_noise(noise_variance, image_shape),
                jnp.asarray(translations_global), config, False,
            )
            if high_indices_np.size == 0:
                pmax_per_image[start:end] = 0.0
            else:
                pmax_batch = compute_high_model_pmax_per_image(
                    mean,
                    rotations_global,
                    ctf2_over_nv_half,
                    half_weights,
                    jnp.asarray(high_indices_np, dtype=jnp.int32),
                    image_shape=image_shape,
                    volume_shape=experiment_dataset.volume_shape,
                    disc_type=disc_type,
                    rotation_block_size=rotation_block_size,
                    use_rms_ctf_approximation=(options.ctf_bound_mode == "cryosparc_rms"),
                    rms_ctf_squared=float(getattr(options, "rms_ctf_squared", 0.5)),
                    noise_variance_half=jnp.asarray(noise_variance) if options.ctf_bound_mode == "cryosparc_rms" else None,
                )
                pmax_per_image[start:end] = np.asarray(pmax_batch)
            start = end

        delta_H = pmax_per_image + float(options.tau_sigma) * np.sqrt(
            np.maximum(pmax_per_image, 0.0),
        )
        upper = s_low + delta_H[:, None, None].astype(s_low.dtype)
        best_score_upper = np.max(upper.reshape(n_images, -1), axis=1).astype(np.float32)

        # Final stage: don't prune — hand off to run_local_em_exact at
        # current_size.
        if stage == n_subdivisions:
            n_kept_per_image = sample_mask.reshape(n_images, -1).sum(axis=1)
            diag.append_stage(BnBStageDiagnostics(
                stage=stage,
                L=int(L),
                angular_spacing_deg=float(np.rad2deg(axis_level.spacing_rad)),
                shift_spacing_px=float(shift_level.spacing_px),
                n_active_rotations=int(sample_mask.any(axis=2).sum() / max(1, n_images)),
                n_active_shifts=int(sample_mask.any(axis=1).sum() / max(1, n_images)),
                n_active_joint=int(n_kept_per_image.mean()),
                n_survivors_mean=float(n_kept_per_image.mean()),
                n_survivors_max=int(n_kept_per_image.max()),
                pmax_high_mean=float(pmax_per_image.mean()),
                high_correction_mean=float(delta_H.mean()),
                cap_applied_count=int(0),
                omitted_mass_upper_mean=0.0,
                omitted_mass_upper_max=0.0,
            ))
            diag.timing[f"stage_{stage}_L{L}_s"] = time.time() - t_stage
            break

        # Prune. Inactive candidates get -inf so they cannot win.
        sample_mask, omitted_mass_stage, cap_applied_stage = prune_by_tail_mass_and_caps(
            sample_mask, upper, options,
        )
        omitted_mass = np.maximum(omitted_mass, omitted_mass_stage)

        n_kept_per_image = sample_mask.reshape(n_images, -1).sum(axis=1)
        diag.append_stage(BnBStageDiagnostics(
            stage=stage,
            L=int(L),
            angular_spacing_deg=float(np.rad2deg(axis_level.spacing_rad)),
            shift_spacing_px=float(shift_level.spacing_px),
            n_active_rotations=int(sample_mask.any(axis=2).sum() / max(1, n_images)),
            n_active_shifts=int(sample_mask.any(axis=1).sum() / max(1, n_images)),
            n_active_joint=int(n_kept_per_image.mean()),
            n_survivors_mean=float(n_kept_per_image.mean()),
            n_survivors_max=int(n_kept_per_image.max()),
            pmax_high_mean=float(pmax_per_image.mean()),
            high_correction_mean=float(delta_H.mean()),
            cap_applied_count=int(cap_applied_stage.sum()),
            omitted_mass_upper_mean=float(omitted_mass_stage.mean()),
            omitted_mass_upper_max=float(omitted_mass_stage.max()),
        ))
        diag.timing[f"stage_{stage}_L{L}_s"] = time.time() - t_stage

        # ---- Subdivide ----
        # Union of all images' surviving axis-angle cells; ditto shifts.
        any_axis_alive = sample_mask.any(axis=(0, 2))
        any_shift_alive = sample_mask.any(axis=(0, 1))
        axis_surv_ids = np.flatnonzero(any_axis_alive).astype(np.int32)
        shift_surv_ids = np.flatnonzero(any_shift_alive).astype(np.int32)

        if axis_surv_ids.size == 0 or shift_surv_ids.size == 0:
            logger.warning(
                "Hierarchical BnB exhausted survivors at stage %d (L=%d); stopping.",
                stage, L,
            )
            break

        axis_level_new = subdivide_axis_angle_cells(
            axis_level, surviving_ids=axis_surv_ids,
        )
        shift_level_new = subdivide_shift_cells(
            shift_level, surviving_ids=shift_surv_ids,
        )

        # Expand sample_mask via parent-id lookup. Note: axis dedup may
        # produce children that map to several distinct parents; we
        # accept the "any parent kept" semantics by gathering directly.
        # First restrict parent-axis dim of sample_mask to surviving parents
        # so the gather indices match the subdivision's parent-id space.
        compressed_mask = sample_mask[:, axis_surv_ids, :][:, :, shift_surv_ids]
        # Rebase parent_ids to the compressed-parent indexing.
        # axis_level_new.parent_ids is in original axis_level coords (since
        # subdivide_axis_angle_cells stored surv_idx); we need it remapped
        # to 0..len(axis_surv_ids)-1 for compressed_mask.
        axis_parent_map = -np.ones(axis_level.n_cells, dtype=np.int32)
        axis_parent_map[axis_surv_ids] = np.arange(axis_surv_ids.size, dtype=np.int32)
        shift_parent_map = -np.ones(shift_level.n_cells, dtype=np.int32)
        shift_parent_map[shift_surv_ids] = np.arange(shift_surv_ids.size, dtype=np.int32)

        axis_child_parents_compressed = axis_parent_map[axis_level_new.parent_ids]
        shift_child_parents_compressed = shift_parent_map[shift_level_new.parent_ids]

        # Children whose parent isn't in the compressed set (-1) shouldn't
        # exist because axis_level_new only subdivides surviving parents.
        assert np.all(axis_child_parents_compressed >= 0)
        assert np.all(shift_child_parents_compressed >= 0)

        new_sample_mask = _expand_sample_mask_to_children(
            compressed_mask,
            axis_child_parents_compressed,
            shift_child_parents_compressed,
        )

        axis_level = axis_level_new
        shift_level = shift_level_new
        sample_mask = new_sample_mask

    rotations_final = np.asarray(axis_level.rotations, dtype=np.float32)
    translations_final = np.asarray(shift_level.centers, dtype=np.float32)
    n_kept_per_image = sample_mask.reshape(n_images, -1).sum(axis=1)
    diag.candidates_final_mean = float(n_kept_per_image.mean())
    diag.candidates_final_max = int(n_kept_per_image.max())

    rotation_survivor_mask = sample_mask.any(axis=2)

    return BnBSupportResult(
        image_indices=image_indices,
        n_rotations_global=int(axis_level.n_cells),
        n_translations=int(shift_level.n_cells),
        sample_mask_per_image=sample_mask,
        rotation_survivor_mask_per_image=rotation_survivor_mask,
        omitted_mass_upper=omitted_mass,
        best_score_upper=best_score_upper,
        diagnostics=diag,
    ), rotations_final, translations_final
