"""Solver/preconditioner variants for the masked PPCA M-step.

This module implements several alternative formulations for
    H_Omega W = G
where H_Omega is the masked normal-equations operator from the PPCA M-step.

All solvers accept the same inputs as pcg_mstep in pcg_mean.py and return
(W_real, info_dict) where info_dict contains convergence diagnostics.

Variants implemented:
  1. baseline_circulant  — current PCG with circulant (unmasked-inverse) preconditioner
  2. no_precond          — plain CG (no preconditioner)
  3. reduced_coord       — CG in reduced (mask-only) coordinates
  4. reduced_circulant   — reduced-coord CG with scatter→FFT→D^{-1}→iFFT→gather precond
  5. reduced_jacobi      — reduced-coord CG with block-Jacobi (q×q per masked voxel)
  6. soft_penalty        — soft-penalty relaxation of the mask constraint
  7. two_level           — reduced circulant + low-freq coarse correction
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu

logger = logging.getLogger(__name__)

_MATVEC_CHUNK = 100_000
_PC_BATCH = 4


# =====================================================================
# Shared utilities
# =====================================================================

def _batched_mask_irfft_rfft(W_half, mask, volume_shape):
    """irfft → mask → rfft, batched over PCs to limit memory."""
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


def _half_vol_lhs_matvec(W_half, lhs_tri, reg_diag, q, unpack_fn=None):
    """Chunked LHS multiply in half-volume: (LHS + diag(reg)) @ W."""
    half_vol_size = lhs_tri.shape[0]
    is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
    result = reg_diag * W_half

    for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
        i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
        if is_tri:
            lhs_chunk = unpack_fn(lhs_tri[i0:i1], q)
        else:
            lhs_chunk = lhs_tri[i0:i1]
        result = result.at[i0:i1].add(
            jnp.einsum("vij,vj->vi", lhs_chunk, W_half[i0:i1])
        )
    return result


def _build_rfft_weights(volume_shape):
    """Inner-product weights for half-volume rfft representation."""
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    N = volume_shape[0]
    rfft_w = 2 * jnp.ones(half_vs, dtype=jnp.float32)
    rfft_w = rfft_w.at[:, :, 0].set(1)
    if N % 2 == 0:
        rfft_w = rfft_w.at[:, :, -1].set(1)
    return rfft_w.reshape(-1) / (N ** 3)


def _half_ip(a, b, rfft_w):
    """Weighted inner product in half-volume = real-space inner product."""
    return float(jnp.sum(rfft_w[:, None] * jnp.real(jnp.conj(a) * b)))


def _real_ip(a, b):
    """Plain real-space inner product for (q, D, D, D) arrays."""
    return float(jnp.sum(a * b))


def _masked_real_ip(a, b, mask_flat):
    """Inner product restricted to masked voxels. a, b: (n_mask, q)."""
    return float(jnp.sum(a * b))


def _initial_guess_mask_projected(lhs_tri, rhs_fourier, reg_diag, mask,
                                   volume_shape, q, unpack_fn=None):
    """Per-voxel solve → mask project (mask projection heuristic)."""
    half_vol_size = rhs_fourier.shape[0]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)

    W_solved_parts = []
    for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
        i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
        is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
        if is_tri:
            lhs_chunk = unpack_fn(lhs_tri[i0:i1], q)
        else:
            lhs_chunk = lhs_tri[i0:i1]
        D_chunk = lhs_chunk.at[:, jnp.arange(q), jnp.arange(q)].add(reg_diag[i0:i1])
        W_solved_parts.append(
            jnp.linalg.solve(D_chunk, rhs_fourier[i0:i1, :, None])[..., 0]
        )
    W_solved = jnp.concatenate(W_solved_parts, axis=0)
    W_real = ftu.get_idft3_real(
        W_solved.T.reshape(q, *half_vs), volume_shape
    )
    return mask[None] * W_real


# =====================================================================
# Matvec operators
# =====================================================================

def _matvec_half_vol(W_half, lhs_tri, reg_diag, mask, volume_shape,
                     q, unpack_fn=None):
    """Full-grid matvec in half-volume: P F^{-1} D F P."""
    PW_half = _batched_mask_irfft_rfft(W_half, mask, volume_shape)
    HPW_half = _half_vol_lhs_matvec(PW_half, lhs_tri, reg_diag, q, unpack_fn)
    return _batched_mask_irfft_rfft(HPW_half, mask, volume_shape)


def _matvec_reduced_v2(x_mask, lhs_tri, reg_diag, mask_idx, volume_shape,
                       q, unpack_fn=None):
    """Reduced-coordinate matvec (correct version).

    x_mask: (n_mask, q) real — unknowns at masked voxels.
    Returns: (n_mask, q) real.

    Steps:
      1. Scatter x_mask to full grid: X_full[mask_idx] = x_mask
      2. rfft3(X_full) → X_half (half_vol, q) complex
      3. (LHS + reg) @ X_half per voxel → HX_half
      4. irfft3(HX_half) → HX_full real
      5. Gather: HX_full[mask_idx] → result
    """
    vol_size = int(np.prod(volume_shape))
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    half_vol_size = int(np.prod(half_vs))

    # Step 1-2: Scatter + rfft3, batched over PCs
    X_half_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        nb = j1 - j0
        X_full = jnp.zeros((nb, vol_size), dtype=jnp.float32)
        X_full = X_full.at[:, mask_idx].set(
            jnp.asarray(x_mask[:, j0:j1].T, dtype=jnp.float32)
        )
        X_full = X_full.reshape(nb, *volume_shape)
        X_half_parts.append(
            ftu.get_dft3_real(X_full).reshape(nb, half_vol_size).T
        )
    X_half = jnp.concatenate(X_half_parts, axis=1)  # (half_vol, q) complex

    # Step 3: LHS multiply in half-volume
    HX_half = _half_vol_lhs_matvec(X_half, lhs_tri, reg_diag, q, unpack_fn)

    # Step 4-5: irfft3 + gather, batched over PCs
    result_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        nb = j1 - j0
        HX_real = ftu.get_idft3_real(
            HX_half[:, j0:j1].T.reshape(nb, *half_vs), volume_shape
        )  # (nb, D, D, D) real
        HX_flat = HX_real.reshape(nb, vol_size)
        result_parts.append(HX_flat[:, mask_idx].T)  # (n_mask, nb)
    return jnp.concatenate(result_parts, axis=1)  # (n_mask, q)


# =====================================================================
# Preconditioners
# =====================================================================

def _precond_circulant_half(lhs_tri, reg_diag, mask, volume_shape, q,
                            unpack_fn=None):
    """Circulant (unmasked-inverse) preconditioner in half-volume.

    M^{-1} r = P F^{-1}[ D(xi)^{-1} F(P r)(xi) ]
    """
    half_vol_size = lhs_tri.shape[0]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)

    # Precompute D^{-1} for all half-voxels
    D_inv_parts = []
    for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
        i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
        is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
        if is_tri:
            lhs_chunk = unpack_fn(lhs_tri[i0:i1], q)
        else:
            lhs_chunk = lhs_tri[i0:i1]
        Dc = lhs_chunk.at[:, jnp.arange(q), jnp.arange(q)].add(reg_diag[i0:i1])
        D_inv_parts.append(jnp.linalg.inv(Dc))
    D_inv = jnp.concatenate(D_inv_parts, axis=0)  # (half_vol, q, q)

    def apply(R_half):
        """R_half: (half_vol, q) complex → (half_vol, q) complex."""
        R_masked = _batched_mask_irfft_rfft(R_half, mask, volume_shape)
        R_solved = jnp.einsum("vij,vj->vi", D_inv, R_masked)
        return _batched_mask_irfft_rfft(R_solved, mask, volume_shape)

    return apply, D_inv


def _precond_circulant_reduced(D_inv, mask_idx, volume_shape, q):
    """Circulant preconditioner in reduced coordinates.

    scatter → rfft → D^{-1} → irfft → gather.
    """
    vol_size = int(np.prod(volume_shape))
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    half_vol_size = int(np.prod(half_vs))

    def apply(r_mask):
        """r_mask: (n_mask, q) real → (n_mask, q) real."""
        # Scatter
        parts_half = []
        for j0 in range(0, q, _PC_BATCH):
            j1 = min(j0 + _PC_BATCH, q)
            nb = j1 - j0
            R_full = jnp.zeros((nb, vol_size), dtype=jnp.float32)
            R_full = R_full.at[:, mask_idx].set(
                jnp.asarray(r_mask[:, j0:j1].T, dtype=jnp.float32)
            )
            R_full = R_full.reshape(nb, *volume_shape)
            parts_half.append(
                ftu.get_dft3_real(R_full).reshape(nb, half_vol_size).T
            )
        R_half = jnp.concatenate(parts_half, axis=1)  # (half_vol, q)

        # D^{-1} per voxel
        R_solved = jnp.einsum("vij,vj->vi", D_inv, R_half)

        # irfft + gather
        result_parts = []
        for j0 in range(0, q, _PC_BATCH):
            j1 = min(j0 + _PC_BATCH, q)
            nb = j1 - j0
            out_real = ftu.get_idft3_real(
                R_solved[:, j0:j1].T.reshape(nb, *half_vs), volume_shape
            )
            result_parts.append(out_real.reshape(nb, vol_size)[:, mask_idx].T)
        return jnp.concatenate(result_parts, axis=1)

    return apply


def _precond_block_jacobi_reduced(lhs_tri, reg_diag, mask_idx, volume_shape,
                                   q, unpack_fn=None):
    """Block-Jacobi preconditioner: q×q block per masked voxel.

    The diagonal block at masked voxel x for the operator
      H_Omega = P F^{-1} D F P
    is:
      diag_block(x) = (1/N³) sum_xi D(xi)

    This is because for the (j,k) entry at voxel x:
      [H]_{x,j; x,k} = (1/N³) sum_xi D_{jk}(xi) |mask(x)|²

    For binary mask, |mask(x)|²=1 on support, so:
      block(x) = (1/N³) sum_xi D(xi)  [same for all masked voxels]

    This is a single q×q matrix inverted once.
    """
    half_vol_size = lhs_tri.shape[0]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    N = volume_shape[0]

    # Compute mean D = (1/N³) sum_xi D(xi)  using rfft weights
    rfft_w = _build_rfft_weights(volume_shape)  # (half_vol,)

    # Weighted sum of D(xi) over half-voxels
    D_mean = jnp.zeros((q, q), dtype=jnp.float32)
    is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
    for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
        i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
        if is_tri:
            lhs_chunk = unpack_fn(lhs_tri[i0:i1], q)
        else:
            lhs_chunk = lhs_tri[i0:i1]
        Dc = lhs_chunk.at[:, jnp.arange(q), jnp.arange(q)].add(reg_diag[i0:i1])
        # Weighted sum: sum_v w(v) * D(v)
        D_mean += jnp.einsum("v,vij->ij", rfft_w[i0:i1], Dc)

    # D_mean is now the average D over the full grid (N³ normalization in rfft_w)
    D_mean_inv = jnp.linalg.inv(D_mean)

    logger.info("Block-Jacobi: D_mean cond=%.2e", float(jnp.linalg.cond(D_mean)))

    def apply(r_mask):
        """r_mask: (n_mask, q) real → (n_mask, q) real."""
        return r_mask @ D_mean_inv.T

    return apply


def _precond_diagonal_jacobi_reduced(lhs_tri, reg_diag, mask_idx, volume_shape,
                                      q, unpack_fn=None):
    """Scalar Jacobi: only the q diagonal entries of the mean D block."""
    half_vol_size = lhs_tri.shape[0]
    rfft_w = _build_rfft_weights(volume_shape)
    is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q

    diag_sum = jnp.zeros(q, dtype=jnp.float32)
    for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
        i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
        if is_tri:
            lhs_chunk = unpack_fn(lhs_tri[i0:i1], q)
        else:
            lhs_chunk = lhs_tri[i0:i1]
        D_diag = lhs_chunk[:, jnp.arange(q), jnp.arange(q)] + reg_diag[i0:i1]
        diag_sum += jnp.einsum("v,vj->j", rfft_w[i0:i1], D_diag)

    inv_diag = 1.0 / jnp.maximum(diag_sum, 1e-12)

    def apply(r_mask):
        return r_mask * inv_diag[None, :]

    return apply


# =====================================================================
# Soft-penalty matvec and preconditioner
# =====================================================================

def _matvec_soft_penalty_half(W_half, lhs_tri, reg_diag, mask, volume_shape,
                               soft_lam, q, unpack_fn=None):
    """Matvec for soft-penalty formulation in half-volume.

    H_soft W = F^{-1}[ (LHS+reg) F(W) ] + λ (1-mask)² W
    """
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    half_vol_size = lhs_tri.shape[0]
    soft_weight = (1.0 - mask) ** 2  # (D,D,D)

    # LHS multiply in half-volume
    HW_half = _half_vol_lhs_matvec(W_half, lhs_tri, reg_diag, q, unpack_fn)

    # Add soft penalty: need to go to real space, multiply, come back
    result_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        nb = j1 - j0
        hw_r = ftu.get_idft3_real(
            HW_half[:, j0:j1].T.reshape(nb, *half_vs), volume_shape
        )
        w_r = ftu.get_idft3_real(
            W_half[:, j0:j1].T.reshape(nb, *half_vs), volume_shape
        )
        out_r = hw_r + soft_lam * soft_weight[None] * w_r
        result_parts.append(
            ftu.get_dft3_real(out_r).reshape(nb, -1).T
        )
    return jnp.concatenate(result_parts, axis=1)


def _precond_soft_penalty_half(lhs_tri, reg_diag, soft_lam, mask,
                                volume_shape, q, unpack_fn=None):
    """Preconditioner for soft penalty: (D + λ_avg I)^{-1} per Fourier voxel."""
    half_vol_size = lhs_tri.shape[0]
    soft_weight = (1.0 - mask) ** 2
    lam_avg = soft_lam * float(jnp.mean(soft_weight))
    is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q

    D_inv_parts = []
    for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
        i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
        if is_tri:
            lhs_chunk = unpack_fn(lhs_tri[i0:i1], q)
        else:
            lhs_chunk = lhs_tri[i0:i1]
        Dc = lhs_chunk.at[:, jnp.arange(q), jnp.arange(q)].add(
            reg_diag[i0:i1] + lam_avg
        )
        D_inv_parts.append(jnp.linalg.inv(Dc))
    D_inv = jnp.concatenate(D_inv_parts, axis=0)

    def apply(R_half):
        return jnp.einsum("vij,vj->vi", D_inv, R_half)

    return apply


# =====================================================================
# Two-level preconditioner: circulant + low-frequency coarse correction
# =====================================================================

def _build_coarse_basis(mask, volume_shape, n_coarse=8):
    """Build a small coarse basis of low-frequency functions on the mask.

    Returns basis: (n_coarse, n_mask) real, orthonormal on the mask.
    Uses the lowest-frequency 3D cosine modes restricted to the mask.
    """
    D = volume_shape[0]
    mask_flat = mask.reshape(-1)
    mask_idx = jnp.where(mask_flat > 0.5)[0]
    n_mask = mask_idx.shape[0]

    # Generate low-freq modes: constant + 3 linear + ... up to n_coarse
    # Use 3D DCT-like modes: cos(pi*k*x/D) for k=0,1,2,...
    coords = jnp.mgrid[:D, :D, :D].reshape(3, -1).T  # (D³, 3)
    coords_mask = coords[mask_idx]  # (n_mask, 3)
    # Normalize to [0, 1]
    coords_norm = coords_mask / D

    # Build modes: products of 1D cosines
    # Mode (a,b,c): cos(pi*a*x) * cos(pi*b*y) * cos(pi*c*z)
    # Use low-frequency triples sorted by a²+b²+c²
    from itertools import product as cart_product
    max_k = 3
    triples = sorted(cart_product(range(max_k), repeat=3),
                     key=lambda t: sum(x**2 for x in t))

    modes = []
    for a, b, c in triples[:n_coarse]:
        mode = (jnp.cos(jnp.pi * a * coords_norm[:, 0]) *
                jnp.cos(jnp.pi * b * coords_norm[:, 1]) *
                jnp.cos(jnp.pi * c * coords_norm[:, 2]))
        modes.append(mode)

    basis = jnp.stack(modes, axis=0)  # (n_coarse, n_mask)

    # Orthonormalize via QR
    Q, _ = jnp.linalg.qr(basis.T)  # (n_mask, n_coarse)
    return Q.T  # (n_coarse, n_mask)


def _precond_two_level_reduced(fine_precond_apply, matvec_reduced_fn,
                                coarse_basis, q):
    """Two-level preconditioner: fine (circulant) + coarse correction.

    M^{-1}_{2L} r = M^{-1}_fine r + Z (Z^T A Z)^{-1} Z^T (r - A M^{-1}_fine r)

    where Z = coarse_basis ⊗ I_q (Kronecker), and A is the reduced operator.

    For efficiency, we precompute (Z^T A Z)^{-1} as a small dense matrix.
    """
    n_coarse = coarse_basis.shape[0]
    n_mask = coarse_basis.shape[1]
    sz = n_coarse * q  # size of coarse problem

    # Build coarse operator Z^T A Z by applying A to each coarse basis vector
    # Z has shape (n_mask * q, n_coarse * q) = (n_mask, q) × (n_coarse, q)
    # Z[:, i*q+j] = coarse_basis[i, :] at component j, 0 elsewhere

    AZ_cols = []
    for i in range(n_coarse):
        for j in range(q):
            # Build column: (n_mask, q) with coarse_basis[i] in column j
            x = jnp.zeros((n_mask, q), dtype=jnp.float32)
            x = x.at[:, j].set(coarse_basis[i])
            Ax = matvec_reduced_fn(x)  # (n_mask, q)
            AZ_cols.append(Ax)

    # Z^T A Z: (sz, sz) matrix
    ZtAZ = jnp.zeros((sz, sz), dtype=jnp.float32)
    for ii in range(sz):
        for jj in range(sz):
            i1, j1 = ii // q, ii % q
            # Z^T (AZ)_jj = sum over mask of coarse_basis[i1, :] * (AZ_jj)[:, j1]
            ZtAZ = ZtAZ.at[ii, jj].set(
                jnp.sum(coarse_basis[i1] * AZ_cols[jj][:, j1])
            )

    ZtAZ_inv = jnp.linalg.inv(ZtAZ)
    logger.info("Two-level: coarse operator cond=%.2e, size=%d",
                float(jnp.linalg.cond(ZtAZ)), sz)

    def apply(r_mask):
        """r_mask: (n_mask, q) real → (n_mask, q) real."""
        # Fine correction
        y = fine_precond_apply(r_mask)

        # Coarse correction: Z (Z^T A Z)^{-1} Z^T (r - A y)
        Ay = matvec_reduced_fn(y)
        defect = r_mask - Ay

        # Z^T defect: (sz,)
        Zt_defect = jnp.zeros(sz, dtype=jnp.float32)
        for i in range(n_coarse):
            for j in range(q):
                idx = i * q + j
                Zt_defect = Zt_defect.at[idx].set(
                    jnp.sum(coarse_basis[i] * defect[:, j])
                )

        # Solve coarse system
        coarse_sol = ZtAZ_inv @ Zt_defect  # (sz,)

        # Prolongate: Z * coarse_sol → (n_mask, q)
        correction = jnp.zeros_like(r_mask)
        for i in range(n_coarse):
            for j in range(q):
                idx = i * q + j
                correction = correction.at[:, j].add(
                    coarse_sol[idx] * coarse_basis[i]
                )

        return y + correction

    return apply


# =====================================================================
# CG solver (generic)
# =====================================================================

def _cg_solve(matvec, precond, rhs, x0, maxiter, tol, ip_fn, label="CG"):
    """Generic preconditioned CG.

    All arguments operate on the same vector space (whatever shape).
    matvec(x) → Ax
    precond(r) → M^{-1} r
    ip_fn(a, b) → float (inner product)
    """
    t_start = time.time()

    r = rhs - matvec(x0)
    z = precond(r)
    p = z.copy()
    rz = ip_fn(r, z)
    b_norm = float(jnp.sqrt(ip_fn(rhs, rhs)))
    x = x0

    residuals = []
    timings = []

    for it in range(maxiter):
        Ap = matvec(p)
        pAp = ip_fn(p, Ap)
        if pAp < 1e-30:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rr = float(jnp.sqrt(ip_fn(r, r))) / max(b_norm, 1e-30)
        residuals.append(rr)
        timings.append(time.time() - t_start)

        if rr < tol:
            logger.info("%s converged iter %d  rr=%.2e", label, it + 1, rr)
            break

        z = precond(r)
        rz_new = ip_fn(r, z)
        beta = rz_new / max(abs(rz), 1e-30)
        p = z + beta * p
        rz = rz_new

        if (it + 1) % 5 == 0:
            logger.info("%s iter %3d: rr=%.2e  t=%.1fs", label, it + 1, rr,
                       timings[-1])

    if residuals and residuals[-1] >= tol:
        logger.info("%s maxiter=%d rr=%.2e", label, maxiter, residuals[-1])

    total_time = time.time() - t_start
    return x, {"residuals": residuals, "timings": timings,
               "total_time": total_time, "n_iters": len(residuals)}


# =====================================================================
# Public solver functions
# =====================================================================

def solve_baseline_circulant(lhs_tri, rhs_fourier, reg_diag, mask,
                              volume_shape, W0_real=None, maxiter=50,
                              tol=1e-6, unpack_fn=None):
    """Option 1: Current baseline — PCG with circulant preconditioner."""
    q = rhs_fourier.shape[1]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    rfft_w = _build_rfft_weights(volume_shape)
    mask = jnp.asarray(mask).reshape(volume_shape)

    def mv(W_h):
        return _matvec_half_vol(W_h, lhs_tri, reg_diag, mask, volume_shape,
                                q, unpack_fn)

    precond_apply, _ = _precond_circulant_half(
        lhs_tri, reg_diag, mask, volume_shape, q, unpack_fn
    )

    # RHS
    rhs_half = _batched_mask_irfft_rfft(rhs_fourier, mask, volume_shape)

    # Initial guess
    if W0_real is not None:
        W0 = mask[None] * jnp.asarray(W0_real).reshape(q, *volume_shape)
        W_h = ftu.get_dft3_real(W0).reshape(q, -1).T
    else:
        W_init = _initial_guess_mask_projected(
            lhs_tri, rhs_fourier, reg_diag, mask, volume_shape, q, unpack_fn
        )
        W_h = ftu.get_dft3_real(W_init).reshape(q, -1).T

    ip = lambda a, b: _half_ip(a, b, rfft_w)

    W_h, info = _cg_solve(mv, precond_apply, rhs_half, W_h, maxiter, tol,
                           ip, "Baseline-Circulant")

    # Convert to real
    W_real = ftu.get_idft3_real(W_h.T.reshape(q, *half_vs), volume_shape)
    info["label"] = "baseline_circulant"
    return W_real, info


def solve_no_precond(lhs_tri, rhs_fourier, reg_diag, mask, volume_shape,
                     W0_real=None, maxiter=50, tol=1e-6, unpack_fn=None):
    """Option 2: Plain CG with no preconditioner."""
    q = rhs_fourier.shape[1]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    rfft_w = _build_rfft_weights(volume_shape)
    mask = jnp.asarray(mask).reshape(volume_shape)

    def mv(W_h):
        return _matvec_half_vol(W_h, lhs_tri, reg_diag, mask, volume_shape,
                                q, unpack_fn)

    identity = lambda r: r

    rhs_half = _batched_mask_irfft_rfft(rhs_fourier, mask, volume_shape)

    if W0_real is not None:
        W0 = mask[None] * jnp.asarray(W0_real).reshape(q, *volume_shape)
        W_h = ftu.get_dft3_real(W0).reshape(q, -1).T
    else:
        W_init = _initial_guess_mask_projected(
            lhs_tri, rhs_fourier, reg_diag, mask, volume_shape, q, unpack_fn
        )
        W_h = ftu.get_dft3_real(W_init).reshape(q, -1).T

    ip = lambda a, b: _half_ip(a, b, rfft_w)

    W_h, info = _cg_solve(mv, identity, rhs_half, W_h, maxiter, tol,
                           ip, "No-Precond")

    W_real = ftu.get_idft3_real(W_h.T.reshape(q, *half_vs), volume_shape)
    info["label"] = "no_precond"
    return W_real, info


def solve_reduced_coord(lhs_tri, rhs_fourier, reg_diag, mask, volume_shape,
                        W0_real=None, maxiter=50, tol=1e-6, unpack_fn=None):
    """Option 3: CG in reduced (mask-only) coordinates, no preconditioner."""
    q = rhs_fourier.shape[1]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    mask = jnp.asarray(mask).reshape(volume_shape)
    mask_flat = mask.reshape(-1)
    mask_idx = jnp.where(mask_flat > 0.5)[0]
    n_mask = mask_idx.shape[0]
    vol_size = int(np.prod(volume_shape))

    logger.info("Reduced-coord: n_mask=%d / %d (%.1f%%)",
                n_mask, vol_size, 100 * n_mask / vol_size)

    def mv(x_mask):
        return _matvec_reduced_v2(x_mask, lhs_tri, reg_diag, mask_idx,
                                  volume_shape, q, unpack_fn)

    identity = lambda r: r

    # RHS: irfft(rhs_fourier) gathered at mask
    rhs_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        nb = j1 - j0
        rhs_r = ftu.get_idft3_real(
            rhs_fourier[:, j0:j1].T.reshape(nb, *half_vs), volume_shape
        )
        rhs_parts.append(rhs_r.reshape(nb, vol_size)[:, mask_idx].T)
    rhs_mask = jnp.concatenate(rhs_parts, axis=1)  # (n_mask, q)

    # Initial guess
    if W0_real is not None:
        W0 = jnp.asarray(W0_real).reshape(q, *volume_shape)
        x0 = W0.reshape(q, vol_size)[:, mask_idx].T  # (n_mask, q)
    else:
        W_init = _initial_guess_mask_projected(
            lhs_tri, rhs_fourier, reg_diag, mask, volume_shape, q, unpack_fn
        )
        x0 = W_init.reshape(q, vol_size)[:, mask_idx].T

    ip = lambda a, b: _masked_real_ip(a, b, mask_flat)

    x_mask, info = _cg_solve(mv, identity, rhs_mask, x0, maxiter, tol,
                              ip, "Reduced-NoPrecond")

    # Scatter back to full volume
    W_real = jnp.zeros((q, vol_size), dtype=jnp.float32)
    W_real = W_real.at[:, mask_idx].set(jnp.asarray(x_mask.T, dtype=jnp.float32))
    W_real = W_real.reshape(q, *volume_shape)
    info["label"] = "reduced_coord"
    info["n_mask"] = int(n_mask)
    return W_real, info


def solve_reduced_circulant(lhs_tri, rhs_fourier, reg_diag, mask,
                             volume_shape, W0_real=None, maxiter=50,
                             tol=1e-6, unpack_fn=None):
    """Option 4: Reduced-coord CG with circulant preconditioner."""
    q = rhs_fourier.shape[1]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    mask = jnp.asarray(mask).reshape(volume_shape)
    mask_flat = mask.reshape(-1)
    mask_idx = jnp.where(mask_flat > 0.5)[0]
    n_mask = mask_idx.shape[0]
    vol_size = int(np.prod(volume_shape))

    logger.info("Reduced-circulant: n_mask=%d / %d (%.1f%%)",
                n_mask, vol_size, 100 * n_mask / vol_size)

    def mv(x_mask):
        return _matvec_reduced_v2(x_mask, lhs_tri, reg_diag, mask_idx,
                                  volume_shape, q, unpack_fn)

    # Build preconditioner
    _, D_inv = _precond_circulant_half(
        lhs_tri, reg_diag, mask, volume_shape, q, unpack_fn
    )
    precond_apply = _precond_circulant_reduced(D_inv, mask_idx, volume_shape, q)

    # RHS
    rhs_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        nb = j1 - j0
        rhs_r = ftu.get_idft3_real(
            rhs_fourier[:, j0:j1].T.reshape(nb, *half_vs), volume_shape
        )
        rhs_parts.append(rhs_r.reshape(nb, vol_size)[:, mask_idx].T)
    rhs_mask = jnp.concatenate(rhs_parts, axis=1)

    # Initial guess
    if W0_real is not None:
        W0 = jnp.asarray(W0_real).reshape(q, *volume_shape)
        x0 = W0.reshape(q, vol_size)[:, mask_idx].T
    else:
        W_init = _initial_guess_mask_projected(
            lhs_tri, rhs_fourier, reg_diag, mask, volume_shape, q, unpack_fn
        )
        x0 = W_init.reshape(q, vol_size)[:, mask_idx].T

    ip = lambda a, b: _masked_real_ip(a, b, mask_flat)

    x_mask, info = _cg_solve(mv, precond_apply, rhs_mask, x0, maxiter, tol,
                              ip, "Reduced-Circulant")

    W_real = jnp.zeros((q, vol_size), dtype=jnp.float32)
    W_real = W_real.at[:, mask_idx].set(jnp.asarray(x_mask.T, dtype=jnp.float32))
    W_real = W_real.reshape(q, *volume_shape)
    info["label"] = "reduced_circulant"
    info["n_mask"] = int(n_mask)
    return W_real, info


def solve_reduced_jacobi(lhs_tri, rhs_fourier, reg_diag, mask, volume_shape,
                          W0_real=None, maxiter=50, tol=1e-6,
                          unpack_fn=None, block=True):
    """Option 5: Reduced-coord CG with block-Jacobi (or scalar Jacobi)."""
    q = rhs_fourier.shape[1]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    mask = jnp.asarray(mask).reshape(volume_shape)
    mask_flat = mask.reshape(-1)
    mask_idx = jnp.where(mask_flat > 0.5)[0]
    n_mask = mask_idx.shape[0]
    vol_size = int(np.prod(volume_shape))

    def mv(x_mask):
        return _matvec_reduced_v2(x_mask, lhs_tri, reg_diag, mask_idx,
                                  volume_shape, q, unpack_fn)

    if block:
        precond_apply = _precond_block_jacobi_reduced(
            lhs_tri, reg_diag, mask_idx, volume_shape, q, unpack_fn
        )
        lbl = "Reduced-BlockJacobi"
    else:
        precond_apply = _precond_diagonal_jacobi_reduced(
            lhs_tri, reg_diag, mask_idx, volume_shape, q, unpack_fn
        )
        lbl = "Reduced-DiagJacobi"

    # RHS
    rhs_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        nb = j1 - j0
        rhs_r = ftu.get_idft3_real(
            rhs_fourier[:, j0:j1].T.reshape(nb, *half_vs), volume_shape
        )
        rhs_parts.append(rhs_r.reshape(nb, vol_size)[:, mask_idx].T)
    rhs_mask = jnp.concatenate(rhs_parts, axis=1)

    if W0_real is not None:
        W0 = jnp.asarray(W0_real).reshape(q, *volume_shape)
        x0 = W0.reshape(q, vol_size)[:, mask_idx].T
    else:
        W_init = _initial_guess_mask_projected(
            lhs_tri, rhs_fourier, reg_diag, mask, volume_shape, q, unpack_fn
        )
        x0 = W_init.reshape(q, vol_size)[:, mask_idx].T

    ip = lambda a, b: _masked_real_ip(a, b, mask_flat)

    x_mask, info = _cg_solve(mv, precond_apply, rhs_mask, x0, maxiter, tol,
                              ip, lbl)

    W_real = jnp.zeros((q, vol_size), dtype=jnp.float32)
    W_real = W_real.at[:, mask_idx].set(jnp.asarray(x_mask.T, dtype=jnp.float32))
    W_real = W_real.reshape(q, *volume_shape)
    info["label"] = "reduced_block_jacobi" if block else "reduced_diag_jacobi"
    info["n_mask"] = int(n_mask)
    return W_real, info


def solve_soft_penalty(lhs_tri, rhs_fourier, reg_diag, mask, volume_shape,
                       soft_lam=100.0, W0_real=None, maxiter=50, tol=1e-6,
                       unpack_fn=None):
    """Option 6: Soft penalty — replace hard mask with λ||(1-mask)*W||²."""
    q = rhs_fourier.shape[1]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    rfft_w = _build_rfft_weights(volume_shape)
    mask = jnp.asarray(mask).reshape(volume_shape)
    half_vol_size = rhs_fourier.shape[0]

    def mv(W_h):
        return _matvec_soft_penalty_half(
            W_h, lhs_tri, reg_diag, mask, volume_shape, soft_lam, q, unpack_fn
        )

    precond_apply = _precond_soft_penalty_half(
        lhs_tri, reg_diag, soft_lam, mask, volume_shape, q, unpack_fn
    )

    # RHS: no mask projection (soft penalty doesn't project)
    rhs_half = rhs_fourier

    # Initial guess: per-voxel solve (no mask projection for soft)
    if W0_real is not None:
        W_h = ftu.get_dft3_real(
            jnp.asarray(W0_real).reshape(q, *volume_shape)
        ).reshape(q, -1).T
    else:
        W_solved_parts = []
        for i0 in range(0, half_vol_size, _MATVEC_CHUNK):
            i1 = min(i0 + _MATVEC_CHUNK, half_vol_size)
            is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
            if is_tri:
                lhs_chunk = unpack_fn(lhs_tri[i0:i1], q)
            else:
                lhs_chunk = lhs_tri[i0:i1]
            D_chunk = lhs_chunk.at[:, jnp.arange(q), jnp.arange(q)].add(
                reg_diag[i0:i1]
            )
            W_solved_parts.append(
                jnp.linalg.solve(D_chunk, rhs_fourier[i0:i1, :, None])[..., 0]
            )
        W_h = jnp.concatenate(W_solved_parts, axis=0)

    ip = lambda a, b: _half_ip(a, b, rfft_w)

    W_h, info = _cg_solve(mv, precond_apply, rhs_half, W_h, maxiter, tol,
                           ip, f"Soft-Penalty(λ={soft_lam})")

    W_real = ftu.get_idft3_real(W_h.T.reshape(q, *half_vs), volume_shape)
    info["label"] = f"soft_penalty_lam{soft_lam}"
    info["soft_lam"] = soft_lam
    return W_real, info


def solve_reduced_two_level(lhs_tri, rhs_fourier, reg_diag, mask,
                             volume_shape, W0_real=None, maxiter=50,
                             tol=1e-6, unpack_fn=None, n_coarse=8):
    """Option 7: Two-level — reduced circulant + low-freq coarse correction."""
    q = rhs_fourier.shape[1]
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    mask = jnp.asarray(mask).reshape(volume_shape)
    mask_flat = mask.reshape(-1)
    mask_idx = jnp.where(mask_flat > 0.5)[0]
    n_mask = mask_idx.shape[0]
    vol_size = int(np.prod(volume_shape))

    logger.info("Two-level: n_mask=%d, n_coarse=%d, q=%d",
                n_mask, n_coarse, q)

    def mv(x_mask):
        return _matvec_reduced_v2(x_mask, lhs_tri, reg_diag, mask_idx,
                                  volume_shape, q, unpack_fn)

    # Fine preconditioner: circulant in reduced coords
    _, D_inv = _precond_circulant_half(
        lhs_tri, reg_diag, mask, volume_shape, q, unpack_fn
    )
    fine_apply = _precond_circulant_reduced(D_inv, mask_idx, volume_shape, q)

    # Coarse basis
    coarse_basis = _build_coarse_basis(mask, volume_shape, n_coarse)

    # Two-level preconditioner
    precond_apply = _precond_two_level_reduced(
        fine_apply, mv, coarse_basis, q
    )

    # RHS
    rhs_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        nb = j1 - j0
        rhs_r = ftu.get_idft3_real(
            rhs_fourier[:, j0:j1].T.reshape(nb, *half_vs), volume_shape
        )
        rhs_parts.append(rhs_r.reshape(nb, vol_size)[:, mask_idx].T)
    rhs_mask = jnp.concatenate(rhs_parts, axis=1)

    if W0_real is not None:
        W0 = jnp.asarray(W0_real).reshape(q, *volume_shape)
        x0 = W0.reshape(q, vol_size)[:, mask_idx].T
    else:
        W_init = _initial_guess_mask_projected(
            lhs_tri, rhs_fourier, reg_diag, mask, volume_shape, q, unpack_fn
        )
        x0 = W_init.reshape(q, vol_size)[:, mask_idx].T

    ip = lambda a, b: _masked_real_ip(a, b, mask_flat)

    x_mask, info = _cg_solve(mv, precond_apply, rhs_mask, x0, maxiter, tol,
                              ip, "Two-Level")

    W_real = jnp.zeros((q, vol_size), dtype=jnp.float32)
    W_real = W_real.at[:, mask_idx].set(jnp.asarray(x_mask.T, dtype=jnp.float32))
    W_real = W_real.reshape(q, *volume_shape)
    info["label"] = "two_level"
    info["n_mask"] = int(n_mask)
    info["n_coarse"] = n_coarse
    return W_real, info


# =====================================================================
# Mask projection baseline (no CG — just per-voxel solve + mask)
# =====================================================================

def solve_mask_projection(lhs_tri, rhs_fourier, reg_diag, mask,
                           volume_shape, unpack_fn=None):
    """Mask-projection heuristic: per-voxel solve → mask. No iterations."""
    q = rhs_fourier.shape[1]
    mask = jnp.asarray(mask).reshape(volume_shape)
    t0 = time.time()
    W_real = _initial_guess_mask_projected(
        lhs_tri, rhs_fourier, reg_diag, mask, volume_shape, q, unpack_fn
    )
    dt = time.time() - t0
    info = {
        "label": "mask_projection",
        "residuals": [],
        "timings": [dt],
        "total_time": dt,
        "n_iters": 0,
    }
    return W_real, info
