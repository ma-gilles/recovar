"""Augmented [μ, W] PCG M-step wrapper (Milestone 2).

The legacy ``recovar.ppca.ppca._pcg_hard_mstep`` already accepts ``q`` as a
runtime argument — every internal helper threads it through, including
``_mstep_AL_solve_fourier``, ``_mstep_batched_rfft``,
``_mstep_batched_irfft``, the ``scatter`` / ``gather`` reshape pair, and
the ``unpack_fn`` partial. So the augmented [μ, W] M-step does not require
refactoring the legacy function — it requires:

  1. Building a ``reg_diag_aug = [mean_reg | W_reg]`` of shape
     ``[half_vol, q+1]``, with component 0 carrying the homogeneous mean
     prior precision and components 1..q carrying ``1 / W_prior``.
  2. Calling ``_pcg_hard_mstep(..., q=q+1, ...)`` with the augmented stats.
  3. Splitting the returned ``[q+1, *vs]`` array into ``(μ_real, W_real)``.

Multi-mask support is preserved: callers pass ``masks`` (M, D, D, D) and
``pc_mask_assignment`` (q,) for the loading components; the wrapper
prepends ``mean_mask_idx`` to obtain the augmented assignment of length
``q+1``.

The gridding-kernel correction ``K(x) = sinc²(x/D)`` is unchanged —
``_pcg_hard_mstep`` already bakes it into the operator, so the returned
``μ`` and ``W`` are mask-zeroed and K-deconvolved without any post-hoc
deconvolution by callers.

See ``recovar/em/ppca_refinement/CLAUDE.md`` §6 for the math contract.
"""

from __future__ import annotations

import jax.numpy as jnp

from .pose_accumulators import AugmentedPPCAStats
from .ppca import _pcg_hard_mstep, _tri_size, unpack_tri_to_full

__all__ = ["solve_augmented_ppca_mstep"]


_DEFAULT_REG_FLOOR = 1e-16


def solve_augmented_ppca_mstep(
    stats: AugmentedPPCAStats,
    *,
    mean_prior,
    W_prior,
    mask,
    masks=None,
    pc_mask_assignment=None,
    mean_mask_idx: int = 0,
    maxiter: int = 20,
    tol: float = 1e-4,
    theta_init=None,
    reg_floor: float = _DEFAULT_REG_FLOOR,
):
    """Augmented [μ, W] PCG M-step.

    Parameters
    ----------
    stats:
        :class:`AugmentedPPCAStats` with ``rhs [half_vol, q+1]`` and
        ``lhs_tri [half_vol, (q+1)(q+2)/2]``.
    mean_prior:
        ``[half_vol]`` real. **Variance** of the mean's homogeneous
        Gaussian prior in Fourier space; the regularizer is
        ``1 / (mean_prior + reg_floor)``. Larger ``mean_prior`` ⇒ weaker
        regularization on ``μ``.
    W_prior:
        ``[half_vol, q]`` real. **Variance** of the loading prior per
        voxel and per PC; the regularizer is
        ``1 / (W_prior + reg_floor)``. See CLAUDE.md §7.
    mask:
        ``(D, D, D)`` real. Single-mask support gather. Used as the
        support indicator when ``masks`` is ``None``; ignored otherwise
        (the union of ``masks`` defines the support).
    masks:
        Optional ``(M, D, D, D)`` real. Per-PC mask bank. When provided
        with ``pc_mask_assignment``, enables multi-mask projection.
    pc_mask_assignment:
        Optional ``(q,)`` int. Mask index per loading component. The mean
        component is assigned to ``mean_mask_idx`` automatically (the
        wrapper prepends it).
    mean_mask_idx:
        Mask index for ``μ`` in multi-mask mode. Default 0.
    maxiter, tol:
        Forwarded to the legacy CG.
    theta_init:
        Optional ``(μ_real, W_real)`` warmstart. ``μ_real`` shape
        ``(D, D, D)``; ``W_real`` shape ``(q, D, D, D)``.
    reg_floor:
        Floor added to priors before inversion to avoid division by
        zero. Default ``1e-16``.

    Returns
    -------
    mu_real: ``(D, D, D)`` real32. Mask-zeroed, K-deconvolved μ.
    W_real:  ``(q, D, D, D)`` real32. Mask-zeroed, K-deconvolved loadings.
    """
    rhs = jnp.asarray(stats.rhs)
    lhs_tri = jnp.asarray(stats.lhs_tri)
    mean_prior_arr = jnp.asarray(mean_prior)
    W_prior_arr = jnp.asarray(W_prior)

    half_vol, q = W_prior_arr.shape
    p = q + 1

    if rhs.shape != (half_vol, p):
        raise ValueError(f"stats.rhs shape {rhs.shape} mismatches expected ({half_vol}, {p}).")
    expected_tri = _tri_size(p)
    if lhs_tri.shape != (half_vol, expected_tri):
        raise ValueError(
            f"stats.lhs_tri shape {lhs_tri.shape} mismatches expected ({half_vol}, {expected_tri}=tri({p}))."
        )
    if mean_prior_arr.shape != (half_vol,):
        raise ValueError(f"mean_prior shape {mean_prior_arr.shape} != ({half_vol},).")

    # reg_diag_aug[:, 0] = 1 / (mean_prior + ε); reg_diag_aug[:, 1:] = 1 / (W_prior + ε)
    mean_reg = 1.0 / (mean_prior_arr + reg_floor)  # [half_vol]
    W_reg = 1.0 / (W_prior_arr + reg_floor)  # [half_vol, q]
    reg_diag_aug = jnp.concatenate([mean_reg[:, None], W_reg], axis=1).astype(jnp.float32)  # [half_vol, p]

    # Multi-mask augmentation: prepend mean_mask_idx to pc_mask_assignment.
    masks_kwarg = None
    assignment_kwarg = None
    if masks is not None:
        if pc_mask_assignment is None:
            raise ValueError("masks provided but pc_mask_assignment is None.")
        assignment = jnp.asarray(pc_mask_assignment, dtype=jnp.int32)
        if assignment.shape != (q,):
            raise ValueError(f"pc_mask_assignment shape {assignment.shape} != ({q},).")
        assignment_kwarg = jnp.concatenate([jnp.array([mean_mask_idx], dtype=jnp.int32), assignment])
        masks_kwarg = jnp.asarray(masks)

    # Warmstart.
    W0_real = None
    if theta_init is not None:
        mu_init, W_init = theta_init
        mu_init_arr = jnp.asarray(mu_init, dtype=jnp.float32)
        W_init_arr = jnp.asarray(W_init, dtype=jnp.float32)
        if mu_init_arr.ndim != 3:
            raise ValueError(f"theta_init[0] (μ) must be 3D, got {mu_init_arr.shape}.")
        if W_init_arr.ndim != 4 or W_init_arr.shape[0] != q:
            raise ValueError(f"theta_init[1] (W) must be (q={q}, *vs), got {W_init_arr.shape}.")
        W0_real = jnp.concatenate([mu_init_arr[None], W_init_arr], axis=0)

    vs = mask.shape if masks is None else masks_kwarg.shape[1:]

    # NOTE: pass the bare ``unpack_tri_to_full`` here, not a partial. Internal
    # helpers (``_mstep_AL_solve_fourier``, ``_mstep_apply_fourier_op``) call
    # ``unpack_fn(lhs_tri_chunk, q)`` with ``q`` as a positional second arg;
    # a partial pre-binding ``basis_size`` would collide with that.
    theta_real = _pcg_hard_mstep(
        lhs_tri,
        rhs,
        reg_diag_aug,
        jnp.asarray(mask),
        vs,
        p,
        unpack_tri_to_full,
        W0_real=W0_real,
        maxiter=maxiter,
        tol=tol,
        masks=masks_kwarg,
        pc_mask_assignment=assignment_kwarg,
    )  # [p, *vs]

    mu_real = theta_real[0]
    W_real = theta_real[1:]
    return mu_real, W_real
