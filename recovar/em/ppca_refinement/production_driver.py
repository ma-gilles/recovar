"""Production iteration driver for ``recovar ppca-refine`` (Phase A.6).

Bypasses the M5 callback split (block_provider → image_stats →
backprojector) for the production path. The driver walks halfsets ×
image batches × rotation blocks, runs :func:`fused_dense_pose_ppca_block`
per block, accumulates ``AugmentedPPCAStats`` per halfset, calls
:func:`solve_augmented_ppca_mstep` per halfset, halfset-combines, and
returns the updated state.

Single-iter contract: the function does ONE EM iteration. The EM-loop
driver (``run_pose_marginal_em_loop``) wraps it with iteration index +
prior-recompute schedule + diagnostics logging.

Limitations of this initial production driver
---------------------------------------------
* Dense pose-mode only. Sparse / local-pose production driver lands at
  Phase A.3 (analogous structure but with rotation-bucketed flat layout).
* Single fixed ``rotation_grid`` shared across all images (matches
  ``--pose-mode dense``). Per-image local neighborhoods (Mode B) need a
  per-image layout — sparse path.
* No JIT recompilation guard: each call re-traces. Rotation-block
  bucketing for stable JIT shapes is a perf-only follow-up.
* ``y_norm`` includes the half-spec Hermitian pair count via the
  Parseval weights — same convention as the legacy ``_e_step_half_inner``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.em.ppca_refinement.dense_engine import fused_dense_pose_ppca_block
from recovar.em.ppca_refinement.iterations import IterationOpts
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
from recovar.ppca import AugmentedPPCAStats, solve_augmented_ppca_mstep
from recovar.ppca.ppca import _tri_size

__all__ = [
    "run_pose_marginal_iteration_dense_production",
]


def _theta_aug_half_fourier(mu_real, W_real, volume_shape):
    """Convert real-space (μ, W) to flat-Fourier ``theta_aug [P, vol_size]``
    suitable for ``slice_volume`` (full-volume convention).

    Input:
        mu_real: (D, D, D) real32
        W_real:  (q, D, D, D) real32
    Output:
        theta_aug_fourier: (P, vol_size) complex64
    """
    q = W_real.shape[0]
    P = q + 1
    vol_size = int(np.prod(volume_shape))
    mu_f = ftu.get_dft3(jnp.asarray(mu_real, dtype=jnp.float32)).reshape(vol_size)
    W_f = jnp.stack(
        [ftu.get_dft3(jnp.asarray(W_real[k], dtype=jnp.float32)).reshape(vol_size) for k in range(q)],
        axis=0,
    )
    return jnp.concatenate([mu_f[None, :], W_f], axis=0)


def _project_theta_aug_to_block(theta_aug_full, rotations_block, image_shape, volume_shape, disc_type):
    """Project ``theta_aug [P, vol_size]`` onto a rotation block, returning
    ``proj_aug [R, P, F]`` half-spectrum images.

    Loops over P (small) and uses ``core.slice_volume`` for each
    component. Could be batched via ``vmap``; for simplicity we keep
    the explicit loop here.
    """
    P = theta_aug_full.shape[0]
    R = rotations_block.shape[0]
    proj_per_p = []
    for p in range(P):
        proj = core.slice_volume(
            theta_aug_full[p],
            rotations_block,
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
        )
        # proj shape: (R, F)
        proj_per_p.append(proj)
    return jnp.stack(proj_per_p, axis=1)  # (R, P, F)


def _build_Y1_for_block(
    images_full_fourier_batch,  # (B, full_F) complex64
    ctf_full_batch,  # (B, full_F) real32
    noise_var_full_batch,  # (B, full_F) real32
    translation_grid,  # (T, 2) real32
    image_shape,
):
    """Pre-shift CTF-weighted whitened images: Y1[b, t, f] = (C·y·phase_t) / σ²

    Inputs are full-spectrum; we convert to half-spec at the end.
    """
    B, full_F = images_full_fourier_batch.shape
    T = translation_grid.shape[0]
    # Apply C/σ² weighting first (pre-translation, pre-half-spec).
    cy_over_var = ctf_full_batch * images_full_fourier_batch / jnp.maximum(noise_var_full_batch, 1e-12)
    # Vmap over translations using batch_trans_translate_images-style.
    # We do an explicit Python loop over T (T is small, e.g. 1-9).
    Y1_per_t = []
    for t_idx in range(T):
        translation = translation_grid[t_idx]
        # core.translate_images expects (batch, image_size) full-spec.
        translated = core.translate_images(
            cy_over_var,
            jnp.broadcast_to(translation[None, :], (B, 2)),
            image_shape,
            half_image=False,
        )
        Y1_per_t.append(translated)
    Y1_full = jnp.stack(Y1_per_t, axis=1)  # (B, T, full_F)
    # Convert to half-spec.
    Y1_half = jax.vmap(
        jax.vmap(lambda im: ftu.full_image_to_half_image(im, image_shape)),
    )(Y1_full)
    return Y1_half  # (B, T, half_F)


def _per_image_y_norm(images_full_fourier_batch, noise_var_full_batch):
    """y_norm[b] = Σ_f |y[b, f]|² / σ²[b, f] using full-spec Parseval."""
    return jnp.sum(
        jnp.abs(images_full_fourier_batch) ** 2 / jnp.maximum(noise_var_full_batch, 1e-12),
        axis=-1,
    ).real.astype(jnp.float32)


def run_pose_marginal_iteration_dense_production(
    state: PoseMarginalPPCAEMState,
    cryo,
    *,
    rotation_grid,  # (R_total, 3, 3) real32
    translation_grid,  # (T, 2) real32 — translation offsets in pixels
    halfset_indices: tuple[Any, Any],  # (idx_h0, idx_h1) int32 arrays
    mask,  # (D, D, D) real32
    masks=None,
    pc_mask_assignment=None,
    mean_mask_idx: int = 0,
    image_batch_size: int = 32,
    rotation_block_size: int = 64,
    disc_type_project: str = "linear_interp",
    disc_type_backproject: str = "linear_interp",
    opts: IterationOpts = IterationOpts(),
):
    """One full EM iteration: fused E-step + M-step + halfset combine.

    Returns ``(new_state, iter_diagnostics)``.

    The driver walks halfsets in order (0, 1). For each halfset:
      1. Build theta_aug from state.mu_score, state.W_score (the SHARED
         scoring volume — both halves use it for the E-step).
      2. Initialize per-halfset rhs/lhs_tri accumulators.
      3. Walk image batches × rotation blocks; run fused engine; rhs/lhs_tri
         accumulate in place.
      4. Wrap into AugmentedPPCAStats; call solve_augmented_ppca_mstep.
      5. Record (mu_h, W_h).

    After both halves: halfset combine via the simple mean (M10 helper);
    return updated state.

    The mean prior used in the M-step is ``state.mean_prior`` (caller
    is responsible for recomputing it via ``make_mean_prior_provider``
    on the appropriate iteration per the §7.4 schedule).
    """
    image_shape = cryo.image_shape
    volume_shape = (cryo.grid_size, cryo.grid_size, cryo.grid_size)
    full_F = int(np.prod(image_shape))
    half_image_shape = ftu.image_shape_to_half_image_shape(image_shape)
    half_F = int(np.prod(half_image_shape))
    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol = int(np.prod(half_vs))

    q = state.W_score.shape[0]
    P = q + 1
    tri_size = _tri_size(P)

    # Build theta_aug once — shared across halfsets (the E-step uses the
    # combined scoring volume).
    theta_aug_full = _theta_aug_half_fourier(state.mu_score, state.W_score, volume_shape)
    # (P, vol_size) complex64

    R_total = rotation_grid.shape[0]
    rotation_grid_jax = jnp.asarray(rotation_grid, dtype=jnp.float32)
    translation_grid_jax = jnp.asarray(translation_grid, dtype=jnp.float32)

    # Per-halfset accumulators.
    new_mu_half: list[Any] = []
    new_W_half: list[Any] = []
    iter_diag = {
        "iteration_log_evidence": [0.0, 0.0],
        "iteration_n_significant_mean": [0.0, 0.0],
        "iteration_pmax_mean": [0.0, 0.0],
    }

    for half_idx, idx_array in enumerate((halfset_indices[0], halfset_indices[1])):
        rhs_acc = jnp.zeros((P, half_vol), dtype=jnp.complex64)
        lhs_tri_acc = jnp.zeros((tri_size, half_vol), dtype=jnp.float32)
        sum_logZ = 0.0
        sum_pmax = 0.0
        sum_nsig = 0
        n_total = 0

        idx_array_np = np.asarray(idx_array)
        for start in range(0, len(idx_array_np), image_batch_size):
            batch_global_indices = idx_array_np[start : start + image_batch_size]
            B = len(batch_global_indices)
            if B == 0:
                continue

            # Pull batch from cryo.image_source.
            images_full = []
            for it_chunk, global_idx_chunk, _ in cryo.image_source.get_dataset_subset_generator(
                batch_size=B,
                subset_indices=batch_global_indices,
            ):
                images_full.append(np.asarray(it_chunk))
                break  # one batch
            images_full_np = images_full[0].reshape(B, full_F)
            images_full = jnp.asarray(images_full_np, dtype=jnp.complex64)

            # CTF (full-spec real) per image.
            ctf_params = jnp.asarray(cryo.CTF_params[batch_global_indices])
            ctf_full = cryo.ctf_evaluator(ctf_params, image_shape, cryo.voxel_size)
            ctf_full = jnp.asarray(ctf_full, dtype=jnp.float32).reshape(B, full_F)

            # Noise (full-spec real) per image.
            noise_full_np = cryo.noise.get(batch_global_indices)
            noise_full = jnp.asarray(noise_full_np, dtype=jnp.float32).reshape(B, full_F)

            # ctf2_over_noise per image, half-spec.
            ctf2_over_noise_full = (ctf_full**2) / jnp.maximum(noise_full, 1e-12)
            ctf2_over_noise_half = jax.vmap(
                lambda im: ftu.full_image_to_half_image(im.astype(jnp.complex64), image_shape).real
            )(ctf2_over_noise_full).astype(jnp.float32)

            # y_norm per image (full-spec Parseval).
            y_norm = _per_image_y_norm(images_full, noise_full)

            # Y1 per (b, t, f) half-spec.
            Y1_half = _build_Y1_for_block(
                images_full,
                ctf_full,
                noise_full,
                translation_grid_jax,
                image_shape,
            )

            # Walk rotation blocks.
            for r_start in range(0, R_total, rotation_block_size):
                r_stop = min(r_start + rotation_block_size, R_total)
                rotations_block = rotation_grid_jax[r_start:r_stop]
                R_block = rotations_block.shape[0]

                # Project theta_aug onto this rotation block.
                proj_aug_block = _project_theta_aug_to_block(
                    theta_aug_full,
                    rotations_block,
                    image_shape,
                    volume_shape,
                    disc_type_project,
                )  # (R_block, P, half_F)

                # Run fused engine.
                rhs_acc, lhs_tri_acc, diag = fused_dense_pose_ppca_block(
                    Y1_half,
                    proj_aug_block,
                    ctf2_over_noise_half,
                    y_norm,
                    rotations_block,
                    image_shape,
                    volume_shape,
                    rhs_acc,
                    lhs_tri_acc,
                    significance_threshold=opts.significance_threshold,
                    disc_type_backproject=disc_type_backproject,
                )
                sum_logZ += float(jnp.sum(diag.logZ))
                sum_pmax += float(jnp.sum(diag.pmax))
                sum_nsig += int(jnp.sum(diag.n_significant_per_image))
                n_total += B

        # M-step for this halfset.
        # rhs_acc / lhs_tri_acc are in [P/tri, half_vol] layout; the
        # M-step wrapper expects [half_vol, P] / [half_vol, tri].
        rhs_for_mstep = jnp.asarray(rhs_acc).T  # (half_vol, P)
        lhs_tri_for_mstep = jnp.asarray(lhs_tri_acc).T  # (half_vol, tri)
        stats = AugmentedPPCAStats(
            rhs=rhs_for_mstep,
            lhs_tri=lhs_tri_for_mstep,
            n_images=n_total,
            log_likelihood=sum_logZ,
        )
        mu_h, W_h = solve_augmented_ppca_mstep(
            stats,
            mean_prior=state.mean_prior,
            W_prior=state.W_prior,
            mask=mask,
            masks=masks,
            pc_mask_assignment=pc_mask_assignment,
            mean_mask_idx=mean_mask_idx,
            maxiter=opts.pcg_maxiter,
            tol=opts.pcg_tol,
            theta_init=(state.mu_half[half_idx], state.W_half[half_idx])
            if state.mu_half[half_idx] is not None and state.W_half[half_idx] is not None
            else None,
        )
        new_mu_half.append(mu_h)
        new_W_half.append(W_h)

        if n_total > 0:
            iter_diag["iteration_log_evidence"][half_idx] = sum_logZ
            iter_diag["iteration_n_significant_mean"][half_idx] = sum_nsig / n_total
            iter_diag["iteration_pmax_mean"][half_idx] = sum_pmax / n_total

    # Halfset combine — simple mean for now (M10).
    mu_score_new = 0.5 * (new_mu_half[0] + new_mu_half[1])
    W_score_new = 0.5 * (new_W_half[0] + new_W_half[1])

    new_state = state.replace(
        mu_half=(new_mu_half[0], new_mu_half[1]),
        W_half=(new_W_half[0], new_W_half[1]),
        mu_score=mu_score_new,
        W_score=W_score_new,
    )
    return new_state, iter_diag
