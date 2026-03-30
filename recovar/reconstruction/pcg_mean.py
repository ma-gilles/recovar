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


_MATVEC_CHUNK = 100_000  # voxels per chunk for memory-bounded matvec


def _half_vol_lhs_matvec(
    W_half: jnp.ndarray,
    lhs_tri: jnp.ndarray,
    reg_diag: jnp.ndarray,
    q: int,
    unpack_fn=None,
) -> jnp.ndarray:
    """Chunked LHS multiply in half-volume: result = (LHS + diag(reg)) @ W.

    W_half: (half_vol, q) complex
    Returns: (half_vol, q) complex
    """
    half_vol_size = lhs_tri.shape[0]
    is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
    result = reg_diag * W_half  # (half_vol, q) — reg term

    for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
        i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
        if is_tri:
            lhs_chunk = unpack_fn(lhs_tri[i0:i1], q)
        else:
            lhs_chunk = lhs_tri[i0:i1]
        result = result.at[i0:i1].add(jnp.einsum("vij,vj->vi", lhs_chunk, W_half[i0:i1]))
    return result


_PC_BATCH = 4  # PCs per batch for irfft/rfft to limit memory


def _batched_mask_irfft_rfft(W_half, mask, volume_shape):
    """irfft → mask → rfft, batched over PCs to limit memory.

    W_half: (half_vol, q) complex
    Returns: (half_vol, q) complex
    """
    q = W_half.shape[1]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        batch = W_half[:, j0:j1].T.reshape(j1 - j0, *half_vs)
        real_batch = ftu.get_idft3_real(batch, volume_shape)
        masked = mask[None] * real_batch
        parts.append(ftu.get_dft3_real(masked).reshape(j1 - j0, -1).T)
    return jnp.concatenate(parts, axis=1)


def _mstep_matvec_half(
    W_half: jnp.ndarray,
    lhs_tri: jnp.ndarray,
    reg_diag: jnp.ndarray,
    mask: jnp.ndarray,
    volume_shape: tuple,
    soft_penalty_weight: Optional[jnp.ndarray] = None,
    soft_penalty_lam: float = 0.0,
    unpack_fn=None,
) -> jnp.ndarray:
    """Apply H_Omega: input/output are (half_vol, q) complex.

    Converts to real for mask (batched over PCs), LHS multiply in half-vol.
    H W = rfft(P * irfft((LHS + reg) * rfft(P * irfft(W))))
    """
    q = W_half.shape[1]
    use_soft = soft_penalty_weight is not None and soft_penalty_lam > 0

    # Step 1: irfft → mask → rfft (batched over PCs)
    if use_soft:
        PW_half = W_half
    else:
        PW_half = _batched_mask_irfft_rfft(W_half, mask, volume_shape)

    # Step 2: LHS multiply in half-volume (chunked over voxels)
    HPW_half = _half_vol_lhs_matvec(PW_half, lhs_tri, reg_diag, q, unpack_fn)

    # Step 3: irfft → mask → rfft (batched over PCs)
    if use_soft:
        # Need W_real for penalty — batch it
        half_vs = ftu.get_real_fft_packed_shape(volume_shape)
        result = jnp.empty_like(HPW_half)
        for j0 in range(0, q, _PC_BATCH):
            j1 = min(j0 + _PC_BATCH, q)
            nb = j1 - j0
            hpw_r = ftu.get_idft3_real(HPW_half[:, j0:j1].T.reshape(nb, *half_vs), volume_shape)
            w_r = ftu.get_idft3_real(W_half[:, j0:j1].T.reshape(nb, *half_vs), volume_shape)
            out_r = hpw_r + soft_penalty_lam * soft_penalty_weight[None] * w_r
            result = result.at[:, j0:j1].set(ftu.get_dft3_real(out_r).reshape(nb, -1).T)
        return result
    else:
        return _batched_mask_irfft_rfft(HPW_half, mask, volume_shape)


def _mstep_preconditioner(
    lhs_fourier: jnp.ndarray,
    reg_diag: jnp.ndarray,
    mask: jnp.ndarray,
    volume_shape: tuple,
    soft_penalty_weight: Optional[jnp.ndarray] = None,
    soft_penalty_lam: float = 0.0,
):
    """Preconditioner: unmasked q×q solve per Fourier voxel.

    Hard mask mode:
      M_0^{-1} r = P F^{-1}[ D(xi)^{-1} F(P r)(xi) ]
    Soft penalty mode:
      M_0^{-1} r = F^{-1}[ (D(xi) + λ_avg I)^{-1} F(r)(xi) ]
      where λ_avg = λ * mean(w) averages the spatially-varying penalty.

    Cost: one FFT pair + one q×q solve per voxel per CG iteration.
    """
    q = reg_diag.shape[1]
    vs = volume_shape
    use_soft = soft_penalty_weight is not None and soft_penalty_lam > 0

    half_vol_size = lhs_fourier.shape[0]

    # D(xi) = LHS(xi) + diag(reg(xi))  per half-voxel
    D = lhs_fourier + jnp.eye(q)[None] * reg_diag[:, :, None]  # (half_vol, q, q)

    if use_soft:
        # Average penalty strength across volume for circulant approximation
        lam_avg = soft_penalty_lam * float(jnp.mean(soft_penalty_weight))
        D = D + lam_avg * jnp.eye(q)[None]

    # Precompute D^{-1} for all half-voxels
    D_inv = jnp.linalg.inv(D)  # (half_vol, q, q)
    half_vs = ftu.get_real_fft_packed_shape(vs)

    def apply_Minv(R_real):
        q_ = R_real.shape[0]

        if use_soft:
            R_half = ftu.get_dft3_real(R_real).reshape(q_, half_vol_size)
        else:
            R_half = ftu.get_dft3_real(mask[None] * R_real).reshape(q_, half_vol_size)

        # Per-voxel q×q solve in half-volume
        R_solved = jnp.einsum("vij,vj->vi", D_inv, R_half.T)  # (half_vol, q)

        # irfft3 → real
        result = ftu.get_idft3_real(R_solved.T.reshape(q_, *half_vs), vs)
        if use_soft:
            return result
        else:
            return mask[None] * result

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
    soft_penalty_lam: float = 0.0,
    unpack_fn=None,
) -> Tuple[jnp.ndarray, list]:
    """Solve the masked PPCA M-step for all q columns jointly via PCG.

    Memory-efficient: LHS can be upper-tri packed (half_vol, tri_sz) and is
    unpacked in chunks during matvec.  No full (half_vol, q, q) materialized.

    Parameters
    ----------
    lhs_fourier : (half_vol, q, q) or (half_vol, tri_sz) — accumulated LHS.
        If tri_sz, pass ``unpack_fn`` to convert chunks to full q×q.
    rhs_fourier : (half_vol, q) complex — accumulated RHS in half-volume
    reg_diag : (half_vol, q) real — diagonal regularization 1/W_prior
    mask : 3D array (D, D, D) real — support mask
    volume_shape : tuple — full real-space shape (D, D, D)
    W0_real : (q, D, D, D) real, optional — warmstart
    maxiter, tol : CG parameters
    precondition : use block-diagonal preconditioner
    soft_penalty_lam : float — if >0, use soft penalty instead of hard mask
    unpack_fn : callable(tri_chunk, q) -> full_chunk, for upper-tri LHS

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
    use_soft = soft_penalty_lam > 0
    soft_penalty_weight = (1.0 - mask) ** 2 if use_soft else None

    is_tri = lhs_fourier.ndim == 2 and lhs_fourier.shape[1] != q

    # --- All CG state in half-volume (half_vol, q) complex ---
    # Saves ~2× memory vs full real-space (q, D, D, D).
    #
    # rfft inner product weights: DC/Nyquist cols weight 1, rest weight 2.
    # <x,y>_real = sum_ξ w(ξ) Re(conj(x_h(ξ)) y_h(ξ)) / N³
    rfft_w = 2 * jnp.ones(half_vs, dtype=jnp.float32)
    rfft_w = rfft_w.at[:, :, 0].set(1)
    N = vs[0]
    if N % 2 == 0:
        rfft_w = rfft_w.at[:, :, -1].set(1)
    rfft_w = rfft_w.reshape(-1) / (N**3)  # (half_vol,)

    def _ip(a, b):
        """Weighted inner product in half-volume = real-space inner product."""
        return float(jnp.sum(rfft_w[:, None] * jnp.real(jnp.conj(a) * b)))

    def mv_half(W_h):
        """Matvec in half-volume."""
        return _mstep_matvec_half(
            W_h,
            lhs_fourier,
            reg_diag,
            mask,
            vs,
            soft_penalty_weight=soft_penalty_weight,
            soft_penalty_lam=soft_penalty_lam,
            unpack_fn=unpack_fn,
        )

    # Chunked preconditioner in half-volume
    if precondition:

        def apply_Minv_half(R_h):
            """Preconditioner: D^{-1} R in half-volume, chunked."""
            if not use_soft:
                R_h_masked = _batched_mask_irfft_rfft(R_h, mask, vs)
            else:
                R_h_masked = R_h
            R_solved = jnp.empty((half_vol_size, q), dtype=R_h.dtype)
            for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
                i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
                if is_tri:
                    lc = unpack_fn(lhs_fourier[i0:i1], q)
                else:
                    lc = lhs_fourier[i0:i1]
                Dc = lc.at[:, jnp.arange(q), jnp.arange(q)].add(reg_diag[i0:i1])
                R_solved = R_solved.at[i0:i1].set(jnp.linalg.solve(Dc, R_h_masked[i0:i1, :, None])[..., 0])
            if not use_soft:
                return _batched_mask_irfft_rfft(R_solved, mask, vs)
            return R_solved
    else:
        apply_Minv_half = lambda r: r

    # RHS in half-volume
    if use_soft:
        rhs_half = rhs_fourier
    else:
        rhs_half = _batched_mask_irfft_rfft(rhs_fourier, mask, vs)

    # Initial guess in half-volume (chunked per-voxel solve + mask)
    if W0_real is not None:
        W0 = jnp.asarray(W0_real).reshape(q, *vs)
        if not use_soft:
            W0 = mask[None] * W0
        W_h = ftu.get_dft3_real(W0).reshape(q, -1).T
    else:
        W_solved_parts = []
        for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
            i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
            if is_tri:
                lhs_chunk = unpack_fn(lhs_fourier[i0:i1], q)
            else:
                lhs_chunk = lhs_fourier[i0:i1]
            D_chunk = lhs_chunk.at[:, jnp.arange(q), jnp.arange(q)].add(reg_diag[i0:i1])
            W_solved_parts.append(jnp.linalg.solve(D_chunk, rhs_fourier[i0:i1, :, None])[..., 0])
        W_solved = jnp.concatenate(W_solved_parts, axis=0)
        W_h = _batched_mask_irfft_rfft(W_solved, mask, vs)

    # PCG in half-volume with weighted inner products
    r = rhs_half - mv_half(W_h)
    z = apply_Minv_half(r)
    p = z.copy()
    rz = _ip(r, z)
    b_norm = float(jnp.sqrt(_ip(rhs_half, rhs_half)))
    residuals = []

    for it in range(maxiter):
        Ap = mv_half(p)
        pAp = _ip(p, Ap)
        if pAp < 1e-30:
            break
        alpha_cg = rz / pAp
        W_h = W_h + alpha_cg * p
        r = r - alpha_cg * Ap

        rr = float(jnp.sqrt(_ip(r, r))) / max(b_norm, 1e-30)
        residuals.append(rr)

        if rr < tol:
            logger.info("PCG M-step converged iter %d  rr=%.2e", it + 1, rr)
            break

        z = apply_Minv_half(r)
        rz_new = _ip(r, z)
        beta = rz_new / max(abs(rz), 1e-30)
        p = z + beta * p
        rz = rz_new

        if (it + 1) % 5 == 0:
            logger.info("PCG M-step iter %3d: rr=%.2e", it + 1, rr)

    if residuals and residuals[-1] >= tol:
        logger.info("PCG M-step maxiter=%d rr=%.2e", maxiter, residuals[-1])

    # Convert back to real space
    W_real = ftu.get_idft3_real(W_h.T.reshape(q, *half_vs), vs)
    return W_real, residuals
