"""Preconditioned conjugate gradient mean estimation with support mask.

Solves the support-constrained Wiener system:

    H_Omega z = g,    H_Omega = R* C R + lambda W_Omega

where:
    C = F^{-1} diag(d) F           (Wiener filter in Fourier space)
    W_Omega = diag(w|_Omega)        (real-space weighting on support)
    w = |k|^{-2}                    (optional spatial prior weight)
    R injects the support Omega into the full grid
    d = regularized CTF diagonal (accumulated across all images)

PCG preconditioner (circulant + diagonal sandwich):

    M^{-1} = M_J^{-1/2} . M_0^{-1} . M_J^{-1/2}

where:
    M_0^{-1} r = R* F^{-1}[ F(Rr) / (d + lambda * wbar) ]
    M_J = alpha*I + lambda*W_Omega
    alpha = mean(d),  wbar = mean(w on support)

This captures both the spectral weighting d and the local amplitude
variation from w at the cost of one FFT pair + pointwise multiplies
per CG iteration.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu

logger = logging.getLogger(__name__)


def _matvec(
    u: jnp.ndarray, d: jnp.ndarray, mask: jnp.ndarray, lam: float = 0.0, w_support: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Apply  H_Omega u = R* F^{-1} diag(d) F R u  +  lambda * W_Omega u.

    All arrays are 3D volumes (real space for u/mask, Fourier for d).
    """
    Pu = mask * u
    HPu = mask * ftu.get_idft3(d * ftu.get_dft3(Pu)).real
    if lam > 0 and w_support is not None:
        HPu = HPu + lam * w_support * Pu
    return HPu


def build_preconditioner(
    d: jnp.ndarray,
    mask: jnp.ndarray,
    lam: float = 0.0,
    w_support: Optional[jnp.ndarray] = None,
) -> callable:
    """Build the sandwich preconditioner M^{-1} = M_J^{-1/2} M_0^{-1} M_J^{-1/2}.

    Parameters
    ----------
    d : 3D array (Fourier space)
        Regularized CTF diagonal.
    mask : 3D array (real space)
        Support mask (binary or soft).
    lam : float
        Regularization strength for the spatial prior.
    w_support : 3D array (real space), optional
        Spatial weighting w(x) restricted to support. If None, w=1.

    Returns
    -------
    apply_Minv : callable
        r -> M^{-1} r  (one FFT pair + pointwise multiplies)
    """
    alpha = float(jnp.mean(d))

    if w_support is not None and lam > 0:
        n_support = float(jnp.sum(mask > 0.5))
        wbar = float(jnp.sum(w_support * (mask > 0.5))) / max(n_support, 1.0)

        # M_J = alpha*I + lambda*W_Omega  →  M_J^{-1/2}
        mj_diag = alpha + lam * w_support
        mj_inv_sqrt = jnp.where(mask > 0.5, 1.0 / jnp.sqrt(jnp.maximum(mj_diag, 1e-12)), 0.0)

        # M_0^{-1}: Fourier-space inverse of (d + lambda*wbar)
        m0_inv_diag = 1.0 / jnp.maximum(d + lam * wbar, jnp.max(d) * 1e-10)
    else:
        wbar = 0.0
        mj_inv_sqrt = jnp.where(mask > 0.5, 1.0 / jnp.sqrt(jnp.maximum(alpha, 1e-12)), 0.0)
        m0_inv_diag = 1.0 / jnp.maximum(d, jnp.max(d) * 1e-10)

    @jax.jit
    def apply_Minv(r):
        # Step 1: M_J^{-1/2}
        t = mj_inv_sqrt * r

        # Step 2: M_0^{-1} — one FFT pair
        q = mask * ftu.get_idft3(m0_inv_diag * ftu.get_dft3(mask * t)).real

        # Step 3: M_J^{-1/2} again
        return mj_inv_sqrt * q

    logger.info("Preconditioner: alpha=%.2e, wbar=%.2e, lam=%.2e", alpha, wbar, lam)
    return apply_Minv


def pcg_mean(
    d: jnp.ndarray,
    rhs_fourier: jnp.ndarray,
    mask: jnp.ndarray,
    lam: float = 0.0,
    w_support: Optional[jnp.ndarray] = None,
    x0: Optional[jnp.ndarray] = None,
    maxiter: int = 20,
    tol: float = 1e-4,
    precondition: bool = True,
) -> Tuple[jnp.ndarray, list]:
    """Solve the masked Wiener system via PCG.

    Parameters
    ----------
    d : 3D array (Fourier space)
        Regularized CTF diagonal (d_reg from accumulation).
    rhs_fourier : 3D array (Fourier space)
        Right-hand side c = A*b (weighted back-projection in Fourier).
    mask : 3D array (real space)
        Support mask.
    lam : float
        Regularization strength for spatial prior (|k|^{-2} weighting).
    w_support : 3D array (real space), optional
        |k|^{-2} weighting restricted to support. If None, no spatial prior.
    x0 : 3D array (real space), optional
        Initial guess. Default: masked Wiener solution.
    maxiter : int
        Maximum CG iterations (10-20 recommended with warmstart).
    tol : float
        Relative residual tolerance.
    precondition : bool
        Use the sandwich preconditioner.

    Returns
    -------
    x : 3D array (real space)
        Solution restricted to support.
    residuals : list of float
        Relative residual ||r||/||b|| per iteration.
    """
    # Compile matvec
    mv = jax.jit(lambda u: _matvec(u, d, mask, lam, w_support))

    # Build preconditioner
    if precondition:
        apply_Minv = build_preconditioner(d, mask, lam, w_support)
    else:
        apply_Minv = lambda r: r

    # RHS in real space
    rhs = mask * ftu.get_idft3(rhs_fourier).real

    # Initial guess
    if x0 is None:
        x = mask * ftu.get_idft3(rhs_fourier / jnp.maximum(d, jnp.max(d) * 1e-10)).real
    else:
        x = mask * x0

    # PCG iteration
    r = rhs - mv(x)
    z = apply_Minv(r)
    p = z.copy()
    rz = float(jnp.sum(r * z))
    b_norm = float(jnp.sqrt(jnp.sum(rhs * rhs)))
    residuals = []

    for it in range(maxiter):
        Ap = mv(p)
        pAp = float(jnp.sum(p * Ap))
        if pAp < 1e-30:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rr = float(jnp.sqrt(jnp.sum(r * r))) / max(b_norm, 1e-30)
        residuals.append(rr)

        if rr < tol:
            logger.info("PCG converged iter %d  rr=%.2e", it + 1, rr)
            break

        z = apply_Minv(r)
        rz_new = float(jnp.sum(r * z))
        beta = rz_new / max(abs(rz), 1e-30)
        p = z + beta * p
        rz = rz_new

        if (it + 1) % 5 == 0:
            logger.info("PCG iter %3d: rr=%.2e", it + 1, rr)

    if residuals and residuals[-1] >= tol:
        logger.info("PCG maxiter=%d rr=%.2e", maxiter, residuals[-1])

    return x, residuals


# =====================================================================
# Multi-column PCG for the PPCA M-step
#
# Problem:
#   Minimize over W supported on Omega:
#     Q(W) = sum_xi  hat{W}(xi)^H  D(xi)  hat{W}(xi)  -  2 Re tr[ hat{W}(xi)^H hat{R}(xi) ]
#   where D(xi) = L(xi) + Lambda(xi) is the q×q regularized LHS per Fourier voxel.
#
# Variables:
#   W = [w_1 | ... | w_q]  in R^{N x q},  N = D^3 voxels
#   hat{W}(xi) = DFT(W)(xi) in C^q  per Fourier voxel xi
#   L(xi) = sum_n |CTF_n(xi)|^2 E[z_n z_n^T | y_n]   (q×q, real symmetric, >=0)
#   Lambda(xi) = diag(1/prior(xi))                      (q×q, real diagonal, >0)
#   hat{R}(xi) = sum_n CTF_n(xi) y_centered_n(xi) E[z_n|y_n]^*   (q-vector, complex)
#   P = diag(mask)   (N×N real diagonal, the support projection)
#
# Normal equations (support-constrained):
#   H_Omega W = G
#   where:
#     H_Omega W  =  P  F^{-1}[ D(xi) . F(P W)(xi) ]       (operator)
#     G          =  P  F^{-1}[ hat{R} ]                     (RHS)
#
# H_Omega is SPD on the support (proven: D(xi) SPD for all xi, P is a projection).
#
# Matvec cost: one FFT pair + one per-voxel q×q matmul per CG iteration.
# =====================================================================


def _mstep_matvec(
    W_real: jnp.ndarray,
    lhs_fourier: jnp.ndarray,
    reg_diag: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Apply H_Omega to W (all q columns jointly). All in half-volume (rfft).

    Steps:
      1. PW = P * W                          — mask in real space
      2. PW_hat = rfft3(PW)                  — half-volume DFT
      3. HPW_hat = D(xi) * PW_hat(xi)        — q×q matmul per half-voxel
      4. HPW = irfft3(HPW_hat)               — real output guaranteed
      5. return P * HPW                       — mask again
    """
    q = W_real.shape[0]
    vs = W_real.shape[1:]
    half_vol_size = lhs_fourier.shape[0]

    # Step 1: mask
    PW = mask[None] * W_real

    # Step 2: rfft3 → half-volume
    PW_half = ftu.get_dft3_real(PW).reshape(q, half_vol_size)

    # Step 3: per-voxel q×q multiply in half-volume
    HPW_half = jnp.einsum("vij,vj->vi", lhs_fourier, PW_half.T) + reg_diag * PW_half.T

    # Step 4: irfft3 → real (guaranteed, no .real needed)
    HPW = ftu.get_idft3_real(HPW_half.T.reshape(q, *ftu.get_real_fft_packed_shape(vs)), vs)

    # Step 5: mask
    return mask[None] * HPW


def _mstep_preconditioner(
    lhs_fourier: jnp.ndarray,
    reg_diag: jnp.ndarray,
    mask: jnp.ndarray,
    volume_shape: tuple,
):
    """Preconditioner: unmasked q×q solve per Fourier voxel.

    M_0^{-1} r = P F^{-1}[ D(xi)^{-1} F(P r)(xi) ]

    where D(xi) = L(xi) + Lambda(xi) is the q×q regularized LHS.
    This is the exact inverse of the unmasked operator — the best
    circulant preconditioner for the masked problem.

    Cost: one FFT pair + one q×q solve per voxel per CG iteration.
    """
    q = reg_diag.shape[1]
    vol_size = reg_diag.shape[0]
    vs = volume_shape

    # D(xi) = LHS(xi) + diag(reg(xi))  per half-voxel
    half_vol_size = lhs_fourier.shape[0]
    D = lhs_fourier + jnp.eye(q)[None] * reg_diag[:, :, None]  # (half_vol, q, q)

    # Precompute D^{-1} for all half-voxels
    D_inv = jnp.linalg.inv(D)  # (half_vol, q, q)
    half_vs = ftu.get_real_fft_packed_shape(vs)

    def apply_Minv(R_real):
        q_ = R_real.shape[0]

        # Mask → rfft3 → half-volume
        R_half = ftu.get_dft3_real(mask[None] * R_real).reshape(q_, half_vol_size)

        # Per-voxel q×q solve in half-volume
        R_solved = jnp.einsum("vij,vj->vi", D_inv, R_half.T)  # (half_vol, q)

        # irfft3 → real → mask
        return mask[None] * ftu.get_idft3_real(R_solved.T.reshape(q_, *half_vs), vs)

    return apply_Minv


def pcg_mstep(
    lhs_fourier: jnp.ndarray,
    rhs_fourier: jnp.ndarray,
    reg_diag: jnp.ndarray,
    mask: jnp.ndarray,
    volume_shape: tuple,
    W0_real: Optional[jnp.ndarray] = None,
    maxiter: int = 20,
    tol: float = 1e-4,
    precondition: bool = True,
) -> Tuple[jnp.ndarray, list]:
    """Solve the masked PPCA M-step for all q columns jointly via PCG.

    Everything in half-volume (rfft-packed) — no full-volume expansion.

    Parameters
    ----------
    lhs_fourier : (half_vol, q, q) — accumulated LHS in half-volume Fourier
    rhs_fourier : (half_vol, q) complex — accumulated RHS in half-volume
    reg_diag : (half_vol, q) real — diagonal regularization 1/W_prior
    mask : 3D array (D, D, D) real — support mask
    volume_shape : tuple — full real-space shape (D, D, D)
    W0_real : (q, D, D, D) real, optional — warmstart
    maxiter, tol : CG parameters
    precondition : use block-diagonal preconditioner

    Returns
    -------
    W_real : (q, D, D, D) real — solution
    residuals : list of float
    """
    q = rhs_fourier.shape[1]
    half_vol_size = rhs_fourier.shape[0]
    vs = volume_shape
    half_vs = ftu.get_real_fft_packed_shape(vs)

    mask = jnp.asarray(mask).reshape(vs)

    def mv(W):
        return _mstep_matvec(W, lhs_fourier, reg_diag, mask)

    if precondition:
        apply_Minv = _mstep_preconditioner(lhs_fourier, reg_diag, mask, vs)
    else:
        apply_Minv = lambda r: r

    # RHS: irfft3(rhs_half) → real, masked
    rhs_real = mask[None] * ftu.get_idft3_real(rhs_fourier.T.reshape(q, *half_vs), vs)

    # Initial guess
    if W0_real is not None:
        W = mask[None] * jnp.asarray(W0_real).reshape(q, *vs)
    else:
        diag_idx = jnp.arange(q)
        d_diag = lhs_fourier[:, diag_idx, diag_idx] + reg_diag
        W_half = rhs_fourier / jnp.maximum(d_diag, jnp.max(d_diag, axis=0, keepdims=True) * 1e-10)
        W = mask[None] * ftu.get_idft3_real(W_half.T.reshape(q, *half_vs), vs)

    # PCG
    r = rhs_real - mv(W)
    z = apply_Minv(r)
    p = z.copy()
    rz = float(jnp.sum(r * z))
    b_norm = float(jnp.sqrt(jnp.sum(rhs_real**2)))
    residuals = []

    for it in range(maxiter):
        Ap = mv(p)
        pAp = float(jnp.sum(p * Ap))
        if pAp < 1e-30:
            break
        alpha_cg = rz / pAp
        W = W + alpha_cg * p
        r = r - alpha_cg * Ap

        rr = float(jnp.sqrt(jnp.sum(r**2))) / max(b_norm, 1e-30)
        residuals.append(rr)

        if rr < tol:
            logger.info("PCG M-step converged iter %d  rr=%.2e", it + 1, rr)
            break

        z = apply_Minv(r)
        rz_new = float(jnp.sum(r * z))
        beta = rz_new / max(abs(rz), 1e-30)
        p = z + beta * p
        rz = rz_new

        if (it + 1) % 5 == 0:
            logger.info("PCG M-step iter %3d: rr=%.2e", it + 1, rr)

    if residuals and residuals[-1] >= tol:
        logger.info("PCG M-step maxiter=%d rr=%.2e", maxiter, residuals[-1])

    return W, residuals
