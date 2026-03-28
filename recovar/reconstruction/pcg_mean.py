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
