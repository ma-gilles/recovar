"""Paper-faithful per-image-ragged BnB engine for K=1.

Implements cryoSPARC's actual hierarchical refinement (Suppl Note 2):
each image carries its own (axis-angle, shift) candidate cells; cells are
scored, bounded, pruned per image; survivors subdivide 8-axis × 4-shift;
repeat for n_subdivisions stages. After the final stage, retained
per-image cells are packed into a ``LocalHypothesisLayout`` and handed to
``run_local_em_exact`` for the exact final E-step + M-step + noise.

Key contrast with the Phase-2 ``select_bnb_support_fixed_grid_k1``: that
mode took a fixed pose grid (e.g. recovar's HEALPix order 3, 36864
rotations) and progressively pruned per stage WITHOUT subdividing. This
mode actually subdivides — per-image candidate count stays bounded
(~hundreds) while spacing halves each stage, which is what makes
cryoSPARC BnB asymptotically faster than dense+local at scale.

Score path uses ``per_image_score.score_per_image_at_low_freq`` which
loops one image at a time; that's the simple version — a bucketed
pad-to-max variant is a follow-up if Python-loop overhead dominates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_half_image_weights,
)
from recovar.em.dense_single_volume.helpers.preprocessing import preprocess_batch
from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact
from recovar.em.dense_single_volume.local_layout import (
    LocalHypothesisLayout,
    bucket_local_hypothesis_layout,
)

from .bounds import compute_high_model_pmax_per_image
from .diagnostics import BnBDiagnostics, BnBStageDiagnostics
from .frequency import make_bnb_frequency_schedule, make_bnb_high_indices_np
from .options import BranchBoundOptions
from .per_image_score import _to_half_noise, score_per_image_at_low_freq
from .per_image_score_bucketed import score_per_image_at_low_freq_bucketed
from .per_image_state import (
    PerImageBnBPoseState,
    initialize_per_image_state,
    subdivide_per_image_state,
)

logger = logging.getLogger(__name__)


def _prune_per_image(
    sample_mask: np.ndarray,
    upper: np.ndarray,
    options: BranchBoundOptions,
) -> tuple[np.ndarray, float, bool]:
    """Apply margin + caps + floor pruning to a single image's (n_axis, n_shift)
    upper-score grid. Returns (new_mask, omitted_mass_upper, cap_applied)."""

    if options.score_margin is not None:
        tau = float(options.score_margin)
    else:
        tau = -np.log(max(options.posterior_tail_tol, 1e-300))

    flat_u = np.where(sample_mask, upper, -np.inf)
    if not np.any(np.isfinite(flat_u)):
        return sample_mask.copy(), 0.0, False

    best = float(flat_u.max())
    keep = flat_u >= (best - tau)
    cap_applied = False

    n_axis, n_shift = sample_mask.shape
    # Orientation cap: per image, at most max_orientation_fraction × n_axis (with floor).
    rot_upper = np.where(keep, upper, -np.inf).max(axis=1)
    n_keep_rot = max(
        int(options.min_orientations_per_image),
        int(np.ceil(options.max_orientation_fraction * n_axis)),
    )
    n_keep_rot = min(n_keep_rot, n_axis)
    if n_keep_rot < n_axis:
        thresh = np.partition(rot_upper, n_axis - n_keep_rot)[n_axis - n_keep_rot]
        rot_keep = rot_upper >= thresh
        if rot_keep.sum() < (sample_mask.any(axis=1)).sum():
            cap_applied = True
        keep = keep & rot_keep[:, None]

    trans_upper = np.where(keep, upper, -np.inf).max(axis=0)
    n_keep_shift = max(
        int(options.min_shifts_per_image),
        int(np.ceil(options.max_shift_fraction * n_shift)),
    )
    n_keep_shift = min(n_keep_shift, n_shift)
    if n_keep_shift < n_shift:
        thresh_t = np.partition(trans_upper, n_shift - n_keep_shift)[n_shift - n_keep_shift]
        trans_keep = trans_upper >= thresh_t
        if trans_keep.sum() < (sample_mask.any(axis=0)).sum():
            cap_applied = True
        keep = keep & trans_keep[None, :]

    floor = int(options.min_joint_candidates_per_image)
    if floor > 0 and keep.sum() < floor:
        # Restore top-K by upper score within the active set.
        active_flat = np.where(sample_mask, upper, -np.inf).reshape(-1)
        if np.any(np.isfinite(active_flat)):
            k = min(floor, sample_mask.sum())
            if k > 0:
                top_thresh = np.partition(active_flat, active_flat.size - k)[active_flat.size - k]
                keep = ((active_flat >= top_thresh).reshape(sample_mask.shape) & sample_mask)

    ceiling = int(options.max_joint_candidates_per_image)
    if ceiling > 0 and keep.sum() > ceiling:
        kept_flat = np.where(keep, upper, -np.inf).reshape(-1)
        thresh_top = np.partition(kept_flat, kept_flat.size - ceiling)[kept_flat.size - ceiling]
        keep = (kept_flat >= thresh_top).reshape(sample_mask.shape)

    if not np.any(keep):
        return sample_mask.copy(), 1.0, cap_applied

    u_max = float(np.where(keep, upper, -np.inf).max())
    kept_sum = float(np.exp(np.where(keep, upper, -np.inf) - u_max).sum())
    pruned_active = sample_mask & ~keep
    pruned_sum = (
        float(np.exp(np.where(pruned_active, upper, -np.inf) - u_max).sum())
        if pruned_active.any()
        else 0.0
    )
    rho = pruned_sum / max(kept_sum + pruned_sum, 1e-30)
    return keep, rho, cap_applied


def _build_local_layout_from_per_image_state(
    state: PerImageBnBPoseState,
    image_indices: np.ndarray,
) -> LocalHypothesisLayout:
    """Pack per-image-ragged state into a ``LocalHypothesisLayout``.

    Strategy: use the union of all images' shifts as the shared
    ``translation_grid``. For each image, its sample_mask along the shift
    axis is mapped onto the union via lookup. Per-image rotation lists go
    into the layout's flat per-image rotation arrays.
    """
    n_images = state.n_images
    # Build the shared shift grid: union of all per-image shift cells.
    # Round to a small precision to dedup cells that came from independent
    # subdivision but landed at numerically equal positions.
    quant = 1e-5
    shift_keys: list[tuple[int, int]] = []
    shift_index_per_image: list[np.ndarray] = []
    for i in range(n_images):
        for s in state.shift_cells[i]:
            key = (int(round(float(s[0]) / quant)), int(round(float(s[1]) / quant)))
            shift_keys.append(key)
    unique_shift_keys = sorted(set(shift_keys))
    key_to_idx = {k: idx for idx, k in enumerate(unique_shift_keys)}
    union_shifts = np.asarray(
        [(k[0] * quant, k[1] * quant) for k in unique_shift_keys],
        dtype=np.float32,
    )
    n_trans_union = int(union_shifts.shape[0])

    # Per-image: map each local shift_id to its union index.
    for i in range(n_images):
        per_image_idx = np.empty(state.shift_cells[i].shape[0], dtype=np.int32)
        for j, s in enumerate(state.shift_cells[i]):
            key = (int(round(float(s[0]) / quant)), int(round(float(s[1]) / quant)))
            per_image_idx[j] = key_to_idx[key]
        shift_index_per_image.append(per_image_idx)

    # Per-image rotation lists.
    rotation_offsets = np.zeros(n_images + 1, dtype=np.int64)
    rotation_counts = np.zeros(n_images, dtype=np.int32)
    rotation_ids_parts: list[np.ndarray] = []
    rotations_parts: list[np.ndarray] = []
    sample_mask_parts: list[np.ndarray] = []

    rotation_id_offset = 0
    for i in range(n_images):
        n_axis_i = state.axis_cells[i].shape[0]
        rotation_counts[i] = n_axis_i
        rotation_offsets[i + 1] = rotation_offsets[i] + n_axis_i

        # Per-image rotation ids are unique per image (no cross-image
        # rotation sharing in the per-image-ragged hierarchy). Use a
        # running counter so rotation_ids_flat is unique within the layout.
        ids = np.arange(rotation_id_offset, rotation_id_offset + n_axis_i, dtype=np.int32)
        rotation_id_offset += n_axis_i
        rotation_ids_parts.append(ids)
        rotations_parts.append(state.axis_rotations[i])

        # Build per-image (n_axis_i, n_trans_union) sample mask.
        sm_i = np.zeros((n_axis_i, n_trans_union), dtype=bool)
        local_shift_idx = shift_index_per_image[i]
        for s_local in range(state.shift_cells[i].shape[0]):
            sm_i[:, local_shift_idx[s_local]] = state.sample_mask[i][:, s_local]
        sample_mask_parts.append(sm_i)

    rotation_ids_flat = (
        np.concatenate(rotation_ids_parts, axis=0)
        if rotation_ids_parts
        else np.zeros(0, dtype=np.int32)
    )
    rotations_flat = (
        np.concatenate(rotations_parts, axis=0)
        if rotations_parts
        else np.zeros((0, 3, 3), dtype=np.float32)
    )
    rotation_log_priors_flat = np.zeros(rotation_id_offset, dtype=np.float32)
    sample_mask_flat = (
        np.concatenate(sample_mask_parts, axis=0)
        if sample_mask_parts
        else np.zeros((0, n_trans_union), dtype=bool)
    )
    translation_log_priors = np.zeros((n_images, n_trans_union), dtype=np.float32)

    return LocalHypothesisLayout(
        n_global_rotations=int(rotation_id_offset),
        n_pixels=0,
        n_psi=0,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=rotation_counts,
        translation_grid=union_shifts,
        translation_log_priors=translation_log_priors,
        rotation_posterior_ids_flat=None,
        sample_mask_flat=sample_mask_flat,
    )


def run_paper_faithful_bnb_em_k1(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    *,
    current_size: int | None,
    options: BranchBoundOptions,
    disc_type: str = "linear_interp",
    image_batch_size: int = 187,
    rotation_block_size: int = 5000,
    image_corrections: np.ndarray | None = None,
    scale_corrections: np.ndarray | None = None,
    image_pre_shifts: np.ndarray | None = None,
    translation_prior_centers: np.ndarray | None = None,
    accumulate_noise: bool = True,
    return_best_pose_details: bool = True,
    half_spectrum_scoring: bool = False,
    score_with_masked_images: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
):
    """Paper-faithful K=1 BnB driver.

    Returns the same tuple shape as ``run_local_em_exact``.
    """
    image_shape = experiment_dataset.image_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    n_images = int(experiment_dataset.n_units)
    image_indices = np.arange(n_images, dtype=np.int32)

    diag = BnBDiagnostics()

    # Initialise per-image state with the shared coarse grid.
    state = initialize_per_image_state(
        n_images=n_images,
        initial_angular_spacing_deg=float(options.initial_angular_spacing_deg),
        initial_shift_spacing_px=float(options.initial_shift_spacing_px),
        max_shift_px=float(options.max_shift_px),
    )

    L_schedule = make_bnb_frequency_schedule(current_size, image_shape, options)
    diag.L_schedule = np.asarray(L_schedule, dtype=np.int32)
    L_max = L_schedule[-1]

    half_weights = make_half_image_weights(image_shape)
    final_score_indices_np, _ = make_fourier_window_indices_np(
        image_shape,
        current_size if current_size is not None else image_shape[0],
        square=False, include_dc=False,
    )

    n_subdivisions = int(options.n_subdivisions)

    omitted_mass = np.zeros(n_images, dtype=np.float32)

    # Iterate stages: at stage j, score current per-image candidates at L_j,
    # apply bound, prune, then subdivide for stage j+1.
    for stage in range(n_subdivisions + 1):
        L_idx = min(stage, len(L_schedule) - 1)
        L = L_schedule[L_idx]
        t_stage = time.time()

        # High band relative to current refinement support.
        low_score_indices_np, _ = make_fourier_window_indices_np(
            image_shape, 2 * L, square=False, include_dc=False,
        )
        high_indices_np = make_bnb_high_indices_np(
            final_score_indices_np, low_score_indices_np,
        )

        # Score per image at L. The bucketed kernel amortises per-image
        # JAX kernel launches; the per-image-loop variant is the simple
        # reference (also useful for small N where bucketing overhead
        # dominates).
        score_kernel = getattr(options, "score_kernel", "bucketed")
        if score_kernel == "bucketed":
            per_image_scores = score_per_image_at_low_freq_bucketed(
                experiment_dataset, mean, noise_variance, state, image_indices,
                L=L, disc_type=disc_type, image_batch_size=image_batch_size,
                axis_quantum=int(getattr(options, "bucketed_axis_quantum", 64)),
                shift_quantum=int(getattr(options, "bucketed_shift_quantum", 8)),
            )
        else:
            per_image_scores = score_per_image_at_low_freq(
                experiment_dataset, mean, noise_variance, state, image_indices,
                L=L, disc_type=disc_type, image_batch_size=image_batch_size,
            )

        # Per-image P^max and Δ_iH using each image's own axis_rotations.
        # For the bound, we need per-image projections of mean at the
        # image's axis_rotations, then weighted by C^2/sigma^2 × h_l. We
        # iterate per image (same dataset batches).
        delta_H = np.zeros(n_images, dtype=np.float32)
        if high_indices_np.size > 0:
            # Compute Pmax for each image via per-image projections.
            config = ForwardModelConfig.from_dataset(
                experiment_dataset, disc_type=disc_type,
                process_fn=experiment_dataset.process_images,
            )
            nv_half = _to_half_noise(noise_variance, image_shape)
            image_idx_to_local = {int(g): i for i, g in enumerate(image_indices)}
            for batch in experiment_dataset.iter_batches(
                image_batch_size, indices=image_indices, by_image=False,
            ):
                batch_data = batch[0]
                ctf_params = batch[3]
                batch_global = np.asarray(batch[5], dtype=np.int32)
                batch_size = int(jnp.asarray(batch_data).shape[0])

                # ctf2_over_nv per image: just need per-image (1, n_half).
                # Use a degenerate single-shift translations to satisfy
                # preprocess_batch's API.
                _, _, ctf2_over_nv_half = preprocess_batch(
                    experiment_dataset, jnp.asarray(batch_data), ctf_params,
                    nv_half,
                    jnp.zeros((1, 2), dtype=jnp.float32),
                    config, False,
                )

                for b_local in range(batch_size):
                    global_id = int(batch_global[b_local])
                    i_local = image_idx_to_local.get(global_id)
                    if i_local is None:
                        continue
                    axis_rots_i = state.axis_rotations[i_local]
                    if axis_rots_i.shape[0] == 0:
                        delta_H[i_local] = 0.0
                        continue
                    pmax_i = compute_high_model_pmax_per_image(
                        mean,
                        axis_rots_i,
                        ctf2_over_nv_half[b_local : b_local + 1],
                        half_weights,
                        jnp.asarray(high_indices_np, dtype=jnp.int32),
                        image_shape=image_shape,
                        volume_shape=experiment_dataset.volume_shape,
                        disc_type=disc_type,
                        rotation_block_size=rotation_block_size,
                    )
                    pmax_val = float(np.asarray(pmax_i)[0])
                    delta_H[i_local] = pmax_val + float(options.tau_sigma) * float(np.sqrt(max(pmax_val, 0.0)))

        # Apply bound and (if not the final stage) prune per image.
        omitted_mass_stage = np.zeros(n_images, dtype=np.float32)
        cap_applied_stage = np.zeros(n_images, dtype=bool)
        if stage < n_subdivisions:
            for i in range(n_images):
                if state.axis_cells[i].shape[0] == 0 or state.shift_cells[i].shape[0] == 0:
                    continue
                upper_i = per_image_scores[i] + delta_H[i]
                new_mask, rho_i, cap_i = _prune_per_image(state.sample_mask[i], upper_i, options)
                state.sample_mask[i] = new_mask
                omitted_mass_stage[i] = rho_i
                cap_applied_stage[i] = cap_i
            omitted_mass = np.maximum(omitted_mass, omitted_mass_stage)

        cand_counts = state.per_image_candidate_counts()
        diag.append_stage(BnBStageDiagnostics(
            stage=stage,
            L=int(L),
            angular_spacing_deg=float(np.rad2deg(state.axis_spacing_rad)),
            shift_spacing_px=float(state.shift_spacing_px),
            n_active_rotations=int(state.per_image_axis_counts().mean()),
            n_active_shifts=int(state.per_image_shift_counts().mean()),
            n_active_joint=int(cand_counts.mean()) if cand_counts.size else 0,
            n_survivors_mean=float(cand_counts.mean()) if cand_counts.size else 0.0,
            n_survivors_max=int(cand_counts.max()) if cand_counts.size else 0,
            pmax_high_mean=float(delta_H.mean()),
            high_correction_mean=float(delta_H.mean()),
            cap_applied_count=int(cap_applied_stage.sum()),
            omitted_mass_upper_mean=float(omitted_mass_stage.mean()),
            omitted_mass_upper_max=float(omitted_mass_stage.max()),
        ))
        diag.timing[f"stage_{stage}_L{L}_s"] = time.time() - t_stage
        logger.info(
            "Paper-faithful BnB stage %d/%d: L=%d, ax_spacing=%.3f deg, "
            "shift_spacing=%.4f px, mean_candidates=%.1f, max=%d, "
            "pmax_mean=%.3f, omitted_mass_mean=%.2e, caps_fired=%d, %.1fs",
            stage, n_subdivisions, L,
            np.rad2deg(state.axis_spacing_rad), state.shift_spacing_px,
            float(cand_counts.mean()) if cand_counts.size else 0.0,
            int(cand_counts.max()) if cand_counts.size else 0,
            float(delta_H.mean()),
            float(omitted_mass_stage.mean()),
            int(cap_applied_stage.sum()),
            time.time() - t_stage,
        )

        if stage == n_subdivisions:
            break
        state = subdivide_per_image_state(state)

    diag.candidates_final_mean = float(state.per_image_candidate_counts().mean())
    diag.candidates_final_max = int(state.per_image_candidate_counts().max())
    logger.info(
        "Paper-faithful BnB done: final mean=%d max=%d candidates/image, "
        "axis_spacing=%.3f deg, shift_spacing=%.4f px",
        int(diag.candidates_final_mean), diag.candidates_final_max,
        np.rad2deg(state.axis_spacing_rad), state.shift_spacing_px,
    )

    # Build LocalHypothesisLayout from final per-image state.
    layout = _build_local_layout_from_per_image_state(state, image_indices)
    bucketed = bucket_local_hypothesis_layout(
        layout, image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
    )
    del bucketed  # bucket_local_hypothesis_layout pre-warms the bucket cache

    return run_local_em_exact(
        experiment_dataset, mean, mean_variance, noise_variance,
        layout, disc_type,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        current_size=current_size,
        accumulate_noise=accumulate_noise,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        half_spectrum_scoring=half_spectrum_scoring,
        score_with_masked_images=score_with_masked_images,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        translation_prior_centers=translation_prior_centers,
        return_best_pose_details=return_best_pose_details,
    )
