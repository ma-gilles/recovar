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
    "run_pose_marginal_iteration_sparse_production",
    "build_local_neighborhood_layout",
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

    Uses ``core.batch_slice_volume`` to vmap slice_volume over the
    leading P axis (or use the CUDA batched kernel when available) — a
    single fused kernel call instead of P launches.
    """
    return jnp.transpose(
        core.batch_slice_volume(
            theta_aug_full,  # [P, vol_size]
            rotations_block,  # [R, 3, 3]
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
        ),  # → [P, R, F]
        (1, 0, 2),  # → [R, P, F]
    )


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
    halfset_combiner=None,
    prior_recompute_fn=None,
    iteration_index: int = 0,
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

    After both halves: halfset combine via ``halfset_combiner`` (default
    simple mean — see :mod:`recovar.em.ppca_refinement.halfset_combine`
    for the 40 Å low-resolution-join variant); return updated state.

    Mean prior recompute (§7.4 schedule)
    ------------------------------------
    When ``prior_recompute_fn`` is provided, it is called BEFORE the
    M-step at iterations matching the
    :func:`~recovar.em.ppca_refinement.iterations._should_recompute_prior`
    check on ``opts.pc_prior_config``. The callback signature is
    ``(state) -> mean_prior_per_voxel`` (matches
    :func:`recovar.em.ppca_refinement.prior_provider.make_mean_prior_provider`).
    The returned ``mean_prior`` replaces ``state.mean_prior`` for the
    M-step in this iteration.
    """
    # Lazy imports to avoid circular dependency on iterations.py.
    from recovar.em.ppca_refinement.halfset_combine import mean_halfset_combine
    from recovar.em.ppca_refinement.iterations import _should_recompute_prior
    from recovar.ppca import PCPriorConfig

    if halfset_combiner is None:
        halfset_combiner = mean_halfset_combine

    # Optional mean-prior recompute per §7.4 schedule.
    pc_prior_cfg = opts.pc_prior_config or PCPriorConfig()
    mean_prior_for_mstep = state.mean_prior
    if prior_recompute_fn is not None and _should_recompute_prior(iteration_index, pc_prior_cfg):
        mean_prior_for_mstep = prior_recompute_fn(state)

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
            mean_prior=mean_prior_for_mstep,
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

    # Halfset combine via the supplied combiner (default = simple mean).
    mu_score_new = halfset_combiner(new_mu_half[0], new_mu_half[1], "mu")
    W_score_new = halfset_combiner(new_W_half[0], new_W_half[1], "W")

    new_state = state.replace(
        mu_half=(new_mu_half[0], new_mu_half[1]),
        W_half=(new_W_half[0], new_W_half[1]),
        mu_score=mu_score_new,
        W_score=W_score_new,
        mean_prior=mean_prior_for_mstep,
    )
    iter_diag["prior_recomputed"] = mean_prior_for_mstep is not state.mean_prior
    return new_state, iter_diag


# ===========================================================================
# Sparse production driver — Mode B local-pose around current poses
# ===========================================================================


def _random_rotation_perturbation(rng, sigma_rad: float):
    """Sample a rotation matrix close to identity with std ``sigma_rad``."""
    axis = rng.standard_normal(3).astype(np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    axis = axis / axis_norm
    angle = float(rng.normal(0.0, sigma_rad))
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)).astype(np.float32)


def build_local_neighborhood_layout(
    cryo,
    image_indices: np.ndarray,
    halfset_idx: int,
    *,
    n_local_rotations: int = 8,
    sigma_rad: float = 0.05,
    translation_grid=None,
    seed: int = 0,
):
    """Build per-image local rotation neighborhoods around the current
    ``cryo.rotation_matrices`` for the sparse engine.

    For each image i in ``image_indices``, generate ``n_local_rotations``
    perturbed rotations around the image's current rotation
    (``cryo.rotation_matrices[i]``). Returns a flat per-hypothesis layout
    indexable by ``image_id``.

    Mode B (local-pose refinement) per CLAUDE.md §9.3: the layout is
    rebuilt every iteration from the current ``state.pose_estimates``
    (or the initial dataset rotations on iter 0).

    This is a lightweight neighborhood builder. The full
    ``LocalHypothesisLayout`` from
    ``recovar.em.dense_single_volume.local_layout`` adds bucketed-JIT
    optimizations; this version is simpler and JIT-friendly per call.

    Returns
    -------
    rotations_per_hyp : ``[Nh, 3, 3]`` real32
    translations_per_hyp : ``[Nh, 2]`` real32
    image_id : ``[Nh]`` int32 — local image index (0..len(image_indices)-1)
    image_indices : ``[Nh]`` int32 — global image index for backprojection
    """
    rng = np.random.default_rng(seed + halfset_idx)
    if translation_grid is None:
        translation_grid = np.zeros((1, 2), dtype=np.float32)
    T = translation_grid.shape[0]
    n_images = len(image_indices)
    Nh = n_images * n_local_rotations * T

    rotations = np.empty((Nh, 3, 3), dtype=np.float32)
    translations = np.empty((Nh, 2), dtype=np.float32)
    image_id = np.empty((Nh,), dtype=np.int32)
    image_indices_per_hyp = np.empty((Nh,), dtype=np.int32)

    h = 0
    for local_i, global_i in enumerate(image_indices):
        base_R = np.asarray(cryo.rotation_matrices[global_i], dtype=np.float32)
        for _ in range(n_local_rotations):
            dR = _random_rotation_perturbation(rng, sigma_rad)
            R = (dR @ base_R).astype(np.float32)
            for t in range(T):
                rotations[h] = R
                translations[h] = translation_grid[t]
                image_id[h] = local_i
                image_indices_per_hyp[h] = global_i
                h += 1
    return rotations, translations, image_id, image_indices_per_hyp


def run_pose_marginal_iteration_sparse_production(
    state: PoseMarginalPPCAEMState,
    cryo,
    *,
    halfset_indices: tuple[Any, Any],
    mask,
    masks=None,
    pc_mask_assignment=None,
    mean_mask_idx: int = 0,
    n_local_rotations: int = 8,
    local_sigma_rad: float = 0.05,
    translation_grid=None,
    image_batch_size: int = 32,
    disc_type_project: str = "linear_interp",
    disc_type_backproject: str = "linear_interp",
    halfset_combiner=None,
    prior_recompute_fn=None,
    iteration_index: int = 0,
    opts: IterationOpts = IterationOpts(),
    layout_seed: int = 0,
):
    """One full EM iteration with the sparse fused engine (--pose-mode local).

    Mirrors ``run_pose_marginal_iteration_dense_production`` but uses
    per-image local rotation neighborhoods (Mode B) and the fused sparse
    engine (Phase A.3).

    Returns ``(new_state, iter_diagnostics)``.
    """
    from recovar.em.ppca_refinement.halfset_combine import mean_halfset_combine
    from recovar.em.ppca_refinement.iterations import _should_recompute_prior
    from recovar.em.ppca_refinement.sparse_engine import (
        SparseHypothesisLayout,
        fused_sparse_pose_ppca_block,
    )
    from recovar.ppca import PCPriorConfig

    if halfset_combiner is None:
        halfset_combiner = mean_halfset_combine

    pc_prior_cfg = opts.pc_prior_config or PCPriorConfig()
    mean_prior_for_mstep = state.mean_prior
    if prior_recompute_fn is not None and _should_recompute_prior(iteration_index, pc_prior_cfg):
        mean_prior_for_mstep = prior_recompute_fn(state)

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

    theta_aug_full = _theta_aug_half_fourier(state.mu_score, state.W_score, volume_shape)

    if translation_grid is None:
        translation_grid = np.zeros((1, 2), dtype=np.float32)
    translation_grid_jax = jnp.asarray(translation_grid, dtype=jnp.float32)

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

            # Pull image batch + CTF + noise.
            images_full = []
            for it_chunk, _, _ in cryo.image_source.get_dataset_subset_generator(
                batch_size=B,
                subset_indices=batch_global_indices,
            ):
                images_full.append(np.asarray(it_chunk))
                break
            images_full_np = images_full[0].reshape(B, full_F)
            images_full_jax = jnp.asarray(images_full_np, dtype=jnp.complex64)
            ctf_params = jnp.asarray(cryo.CTF_params[batch_global_indices])
            ctf_full = jnp.asarray(
                cryo.ctf_evaluator(ctf_params, image_shape, cryo.voxel_size),
                dtype=jnp.float32,
            ).reshape(B, full_F)
            noise_full = jnp.asarray(cryo.noise.get(batch_global_indices), dtype=jnp.float32).reshape(B, full_F)

            # Per-image y_norm (full-spec Parseval).
            y_norm_per_image = _per_image_y_norm(images_full_jax, noise_full)

            # ctf2_over_noise (half-spec).
            ctf2_over_noise_full = (ctf_full**2) / jnp.maximum(noise_full, 1e-12)
            ctf2_over_noise_half = jax.vmap(
                lambda im: ftu.full_image_to_half_image(im.astype(jnp.complex64), image_shape).real
            )(ctf2_over_noise_full).astype(jnp.float32)

            # Build local rotation neighborhood for this batch.
            rot_per_hyp_np, trans_per_hyp_np, image_id_np, _ = build_local_neighborhood_layout(
                cryo,
                batch_global_indices,
                halfset_idx=half_idx,
                n_local_rotations=n_local_rotations,
                sigma_rad=local_sigma_rad,
                translation_grid=translation_grid,
                seed=layout_seed + iteration_index,
            )
            Nh = rot_per_hyp_np.shape[0]
            rotations_per_hyp = jnp.asarray(rot_per_hyp_np, dtype=jnp.float32)

            # Per-hypothesis Y1 (translate + weight) and proj_aug.
            # Y1[h] = (C_{i(h)} · y_{i(h)} · phase_{t(h)}) / σ²_{i(h)}.
            # Build by mapping each hypothesis to its image's pre-weighted
            # full-spec image, then translating + half-spec converting.
            cy_over_var_full = ctf_full * images_full_jax / jnp.maximum(noise_full, 1e-12)  # [B, full_F]
            # Per-hypothesis full-spec image (pre-translation):
            cy_per_hyp = cy_over_var_full[image_id_np]  # [Nh, full_F]
            # Apply per-hypothesis translation.
            translations_per_hyp_jax = jnp.asarray(trans_per_hyp_np, dtype=jnp.float32)
            translated = core.translate_images(
                cy_per_hyp,
                translations_per_hyp_jax,
                image_shape,
                half_image=False,
            )
            Y1_half = jax.vmap(lambda im: ftu.full_image_to_half_image(im, image_shape))(translated).astype(
                jnp.complex64
            )  # [Nh, half_F]

            # Per-hypothesis ctf2_over_noise + y_norm (broadcast from image).
            ctf2_per_hyp = ctf2_over_noise_half[image_id_np]  # [Nh, half_F]
            y_norm_per_hyp = y_norm_per_image[image_id_np]  # [Nh]

            # Project theta_aug onto per-hypothesis rotations.
            proj_per_p = []
            for p in range(P):
                proj = core.slice_volume(
                    theta_aug_full[p],
                    rotations_per_hyp,
                    image_shape,
                    volume_shape,
                    disc_type_project,
                    half_image=True,
                )
                proj_per_p.append(proj)
            proj_aug_per_hyp = jnp.stack(proj_per_p, axis=1)  # [Nh, P, half_F]

            layout = SparseHypothesisLayout(
                Y1=Y1_half,
                proj_aug=proj_aug_per_hyp,
                ctf2_over_noise=ctf2_per_hyp,
                y_norm=y_norm_per_hyp,
                pose_log_prior=None,
                image_id=jnp.asarray(image_id_np, dtype=jnp.int32),
                n_images=B,
            )

            rhs_acc, lhs_tri_acc, diag = fused_sparse_pose_ppca_block(
                layout,
                rotations_per_hyp,
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

        # M-step per halfset.
        rhs_for_mstep = jnp.asarray(rhs_acc).T
        lhs_tri_for_mstep = jnp.asarray(lhs_tri_acc).T
        stats = AugmentedPPCAStats(
            rhs=rhs_for_mstep,
            lhs_tri=lhs_tri_for_mstep,
            n_images=n_total,
            log_likelihood=sum_logZ,
        )
        mu_h, W_h = solve_augmented_ppca_mstep(
            stats,
            mean_prior=mean_prior_for_mstep,
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

    # Halfset combine.
    mu_score_new = halfset_combiner(new_mu_half[0], new_mu_half[1], "mu")
    W_score_new = halfset_combiner(new_W_half[0], new_W_half[1], "W")

    new_state = state.replace(
        mu_half=(new_mu_half[0], new_mu_half[1]),
        W_half=(new_W_half[0], new_W_half[1]),
        mu_score=mu_score_new,
        W_score=W_score_new,
        mean_prior=mean_prior_for_mstep,
    )
    iter_diag["prior_recomputed"] = mean_prior_for_mstep is not state.mean_prior
    return new_state, iter_diag
