"""Per-image support selection for cryoSPARC-style BnB (Phase 2: fixed grid).

The Phase-2 selector operates on a *fixed global* pose grid (no axis-angle /
shift subdivision yet — that lands in Phase 3). It runs the cryoSPARC
progressive-L pruning loop over that grid, returning per-image survivors as
a boolean mask. The downstream engine packs those survivors into a
``LocalHypothesisLayout`` and hands them to ``run_local_em_exact``.

Algorithm:
    active = all (r, t) pairs survive initially
    for L in L_schedule (excluding L_max):
        score_low[i, r, t] = recovar Gaussian score over |k| <= L
        pmax[i]           = max_r 1/2 sum_{k in H(L)} h_k C^2/sigma^2 |Y(r)|^2
        delta_H[i]        = pmax[i] + tau * sqrt(pmax[i])
        U[i, r, t]        = score_low + delta_H[i]
        active            = prune_by_tail_mass(active, U, options)
    # Final stage handled by run_local_em_exact at current_size.

For Phase 2 we keep the pruning rule simple: per-image score margin
``tau = -log(posterior_tail_tol)`` plus the cryoSPARC 12.5%/25% caps with
``min_orientations_per_image`` / ``min_shifts_per_image`` floors. Phase 5
will tighten this with explicit omitted-mass diagnostics and per-image
fallback wiring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
    make_fourier_window_spec,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_half_image_weights,
)
from recovar.em.dense_single_volume.helpers.preprocessing import preprocess_batch
from recovar.em.dense_single_volume.helpers.scoring import _score_rotation_block

from .bounds import (
    compute_high_model_pmax_per_image,
    cryosparc_score_upper_correction,
)
from .diagnostics import BnBDiagnostics, BnBStageDiagnostics
from .frequency import (
    make_bnb_frequency_schedule,
    make_bnb_high_indices_np,
    make_bnb_low_window_spec,
)
from .hierarchical_support import _to_half_noise
from .options import BranchBoundOptions
from .pruning import prune_by_tail_mass_and_caps

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BnBSupportResult:
    """Per-image surviving (rotation, translation) candidates after BnB."""

    image_indices: np.ndarray
    """(n_images_in_batch,) int32 — global image indices in this BnB pass."""

    n_rotations_global: int
    n_translations: int

    sample_mask_per_image: np.ndarray
    """(n_images, n_rotations_global, n_translations) bool — True if (r, t)
    survives for this image. May be sparse for small survivor counts."""

    rotation_survivor_mask_per_image: np.ndarray
    """(n_images, n_rotations_global) bool — any t survives for this (i, r)."""

    omitted_mass_upper: np.ndarray
    """(n_images,) — upper bound on pruned posterior mass per image (1 = stage 0)."""

    best_score_upper: np.ndarray
    """(n_images,) — max upper score over kept candidates at final pruning stage."""

    diagnostics: BnBDiagnostics


# NOTE: an earlier draft kept a separate `_score_active_low_frequency` helper
# that returned a (n_images, n_rot, n_trans) float32 tensor. At 100k images,
# 36864 rotations, 49 translations that's 723 GB — host OOM on Slurm 8235089.
# The streaming version inlined in `select_bnb_support_fixed_grid_k1` keeps
# only the per-batch (batch_size, n_rot, n_trans) ~1.3 GB tensor alive.


def _prune_by_tail_mass_and_caps(
    sample_mask: np.ndarray,
    upper_scores: np.ndarray,
    options: BranchBoundOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-2 shim that delegates to ``pruning.prune_by_tail_mass_and_caps``.

    Returned tuple is ``(new_mask, omitted_mass_upper)``; cap-applied
    diagnostics are dropped here for back-compat. New callers should use
    ``pruning.prune_by_tail_mass_and_caps`` directly to also receive the
    ``cap_applied_per_image`` vector.
    """
    new_mask, rho, _cap_applied = prune_by_tail_mass_and_caps(
        sample_mask, upper_scores, options,
    )
    return new_mask, rho


def select_bnb_support_fixed_grid_k1(
    experiment_dataset,
    mean,
    noise_variance,
    rotations_global: np.ndarray,
    translations_global: jnp.ndarray,
    *,
    current_size: int | None,
    options: BranchBoundOptions,
    disc_type: str = "linear_interp",
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    image_indices: np.ndarray | None = None,
) -> BnBSupportResult:
    """Phase-2 BnB support selection on a fixed global rotation/translation grid.

    Returns the per-image survivor mask after the progressive-L pruning loop.
    The downstream engine packs this into a ``LocalHypothesisLayout`` and
    hands it to ``run_local_em_exact`` for the exact final E-step and M-step.

    The final L_max stage is *not* pruned here — that's the exact final E-step
    handled by the local engine. Pruning at L_max would risk dropping the MAP
    pose. We exit the BnB loop once L reaches L_max.
    """
    image_shape = experiment_dataset.image_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    n_rot = int(rotations_global.shape[0])
    n_trans = int(np.asarray(translations_global).shape[0])

    if image_indices is None:
        image_indices = np.arange(experiment_dataset.n_units, dtype=np.int32)
    else:
        image_indices = np.asarray(image_indices, dtype=np.int32)
    n_images = int(image_indices.shape[0])

    diag = BnBDiagnostics()

    L_schedule = make_bnb_frequency_schedule(current_size, image_shape, options)
    diag.L_schedule = np.asarray(L_schedule, dtype=np.int32)

    # Start with everything active.
    sample_mask = np.ones((n_images, n_rot, n_trans), dtype=bool)
    omitted_mass = np.zeros(n_images, dtype=np.float32)
    best_score_upper = np.full(n_images, -np.inf, dtype=np.float32)

    half_weights = make_half_image_weights(image_shape)
    final_score_indices_np, _ = make_fourier_window_indices_np(
        image_shape,
        current_size if current_size is not None else image_shape[0],
        square=False, include_dc=False,
    )

    diag.candidates_initial_mean = float(sample_mask.sum() / max(1, n_images))

    # Stop pruning *before* the final L_max stage; the exact local engine
    # handles L_max itself.
    L_max = L_schedule[-1]
    for stage_idx, L in enumerate(L_schedule):
        if L >= L_max:
            # Final stage: no more pruning here. Survivors are handed off.
            break

        t_stage = time.time()

        # Build high band indices = final \ low.
        low_score_indices_np, _ = make_fourier_window_indices_np(
            image_shape, 2 * L, square=False, include_dc=False,
        )
        high_indices_np = make_bnb_high_indices_np(
            final_score_indices_np, low_score_indices_np,
        )
        if high_indices_np.size == 0:
            # Nothing left in the high band — bound is exactly 0, no pruning.
            logger.debug("BnB stage %d (L=%d): empty high band, skipping bound.", stage_idx, L)
            continue

        # Compute P^max_H per image over the global rotation cover.
        # ctf2_over_nv_half is per-batch — but the rotation max only depends
        # on the model and CTF, not on the image phase. So we need per-image
        # CTF/noise. Iterate over batches the same way the scorer does.
        from recovar.em.dense_single_volume.helpers.preprocessing import preprocess_batch
        from recovar.em.dense_single_volume.helpers.projection import (
            compute_projections_block,
        )

        config = ForwardModelConfig.from_dataset(
            experiment_dataset, disc_type=disc_type,
            process_fn=experiment_dataset.process_images,
        )

        pmax_per_image = np.empty(n_images, dtype=np.float32)
        use_rms = options.ctf_bound_mode == "cryosparc_rms"
        if use_rms:
            # Phase-6 optimisation: RMS-CTF + shared noise spectrum gives a
            # single P^max for the whole batch — compute once and broadcast.
            # Skips the per-batch CTF preprocess + projection redo loop.
            pmax_shared = float(np.asarray(compute_high_model_pmax_per_image(
                mean,
                rotations_global,
                jnp.zeros((1, jnp.asarray(half_weights).shape[0]), dtype=jnp.float32),  # ignored in RMS mode
                half_weights,
                jnp.asarray(high_indices_np, dtype=jnp.int32),
                image_shape=image_shape,
                volume_shape=experiment_dataset.volume_shape,
                disc_type=disc_type,
                rotation_block_size=rotation_block_size,
                use_rms_ctf_approximation=True,
                rms_ctf_squared=float(getattr(options, "rms_ctf_squared", 0.5)),
                noise_variance_half=jnp.asarray(_to_half_noise(noise_variance, image_shape)),
            ))[0])
            pmax_per_image[:] = pmax_shared
        else:
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
                    translations_global, config, False,
                )
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
                    use_rms_ctf_approximation=False,
                )
                pmax_per_image[start:end] = np.asarray(pmax_batch)
                start = end

        delta_H = pmax_per_image + float(options.tau_sigma) * np.sqrt(
            np.maximum(pmax_per_image, 0.0),
        )

        # Stream score+prune per image batch. Materialising the full
        # (n_images, n_rot, n_trans) score tensor (e.g. 100k × 36864 × 49 × 4 B
        # = 723 GB) would OOM the host. Per-batch is bounded at
        # batch_size × n_rot × n_trans × 4 B ~ 1.3 GB for batch_size=187,
        # n_rot=36864, n_trans=49 — comfortable.
        omitted_mass_stage = np.zeros(n_images, dtype=np.float32)
        cap_applied_stage = np.zeros(n_images, dtype=bool)
        stage_best_upper = np.full(n_images, -np.inf, dtype=np.float32)

        low_window = make_bnb_low_window_spec(image_shape, L, n_half)
        translations_j = jnp.asarray(translations_global)
        precision_policy = DensePrecisionPolicy()
        n_rot_blocks_score = (n_rot + rotation_block_size - 1) // rotation_block_size

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
                translations_j, config, False,
            )
            shifted_windowed = low_window.score_values(shifted_half)
            ctf2_windowed = low_window.score_values(ctf2_over_nv_half)

            # Per-batch scores at L: (batch_size, n_rot, n_trans) float32.
            scores_batch = np.empty((batch_size, n_rot, n_trans), dtype=np.float32)
            for b in range(n_rot_blocks_score):
                r0 = b * rotation_block_size
                r1 = min(r0 + rotation_block_size, n_rot)
                rot_block = rotations_global[r0:r1]
                if rot_block.shape[0] == 0:
                    continue
                proj_half, proj_abs2_half = compute_projections_block(
                    mean, rot_block, image_shape,
                    experiment_dataset.volume_shape, disc_type,
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
                    volume_shape=experiment_dataset.volume_shape,
                    score_mode="gaussian",
                    precision_policy=precision_policy,
                )
                scores_batch[:, r0:r1, :] = np.asarray(block_scores)[:, : (r1 - r0), :]

            upper_batch = scores_batch + delta_H[start:end, None, None].astype(scores_batch.dtype)
            stage_best_upper[start:end] = np.max(
                upper_batch.reshape(batch_size, -1), axis=1,
            )

            active_batch = sample_mask[start:end]
            new_active_batch, rho_batch, cap_batch = prune_by_tail_mass_and_caps(
                active_batch, upper_batch, options,
            )
            sample_mask[start:end] = new_active_batch
            omitted_mass_stage[start:end] = rho_batch
            cap_applied_stage[start:end] = cap_batch

            start = end

        best_score_upper = stage_best_upper.astype(np.float32)
        omitted_mass = np.maximum(omitted_mass, omitted_mass_stage)

        n_kept_per_image = sample_mask.reshape(n_images, -1).sum(axis=1)
        diag.append_stage(BnBStageDiagnostics(
            stage=stage_idx,
            L=int(L),
            angular_spacing_deg=float("nan"),  # axis-angle grid hooks up in Phase 5+
            shift_spacing_px=float("nan"),
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
        diag.timing[f"stage_{stage_idx}_L{L}_s"] = time.time() - t_stage

    n_kept_per_image = sample_mask.reshape(n_images, -1).sum(axis=1)
    diag.candidates_final_mean = float(n_kept_per_image.mean())
    diag.candidates_final_max = int(n_kept_per_image.max())
    rotation_survivor_mask = sample_mask.any(axis=2)

    return BnBSupportResult(
        image_indices=image_indices,
        n_rotations_global=n_rot,
        n_translations=n_trans,
        sample_mask_per_image=sample_mask,
        rotation_survivor_mask_per_image=rotation_survivor_mask,
        omitted_mass_upper=omitted_mass,
        best_score_upper=best_score_upper.astype(np.float32),
        diagnostics=diag,
    )
