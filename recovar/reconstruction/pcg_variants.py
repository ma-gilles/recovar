"""Masked PPCA M-step solver with soft boundary penalty and gridding.

Solves:
  min_w  w^H G D G w  -  2 Re(w^H G r)  +  λ Σ_x α(x) |w(x)|²

where:
  D(ξ) = LHS(ξ) + Λ(ξ)   per-voxel q×q in Fourier (data + regularization)
  G(x) = sinc²(x/D) per axis in real space (gridding / interpolation kernel)
  α(x) = smooth penalty weight (0 in core, ramp in collar, 1 outside mask)

The operator is:  H w = G · iFFT[ D · FFT[ G · w ] ]  +  λ α · w
                        ~~~real~~~ ~~~~Fourier~~~~ ~~~real~~~    ~~~real~~~

Variables live on a generous hard outer support (reduced coordinates).
"""

from __future__ import annotations

import logging
import time

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

import recovar.core.fourier_transform_utils as ftu

logger = logging.getLogger(__name__)

_MATVEC_CHUNK = 100_000
_PC_BATCH = 4


# -- Gridding kernel (real-space) -------------------------------------------

def compute_gridding_kernel_real(volume_shape, order=1):
    """Real-space gridding kernel G(x) = sinc²(x₁/D)·sinc²(x₂/D)·sinc²(x₃/D).

    This is the blurring introduced by trilinear interpolation during
    Fourier-slice extraction.  order=1 → sinc², order=0 → sinc.

    Returns (D,D,D) float32.
    """
    D = volume_shape[0]
    # Real-space coordinates: centered at 0, range [-D/2, D/2-1]
    coords = np.arange(D, dtype=np.float32) - D / 2
    normed = coords / D  # normalised so sinc argument matches RELION convention

    def sinc(a):
        safe = np.where(np.abs(a) < 1e-8, 1.0, a)
        return np.where(np.abs(a) < 1e-8, 1.0,
                        np.sin(np.pi * safe) / (np.pi * safe))

    kern = sinc if order == 0 else lambda x: sinc(x) ** 2
    Gx = kern(normed)
    return (Gx[:, None, None] * Gx[None, :, None] * Gx[None, None, :]).astype(np.float32)


# -- Alpha weight -----------------------------------------------------------

def build_alpha_weight(binary_mask, collar_width=5, outer_dilate=3):
    """Smooth penalty weight α(x) from a binary mask.

    Returns (alpha, outer_support) where:
      alpha = 0 in core, cosine ramp 0→1 in collar, 1 outside mask.
      outer_support = mask dilated by (collar_width + outer_dilate).
    """
    bm = np.asarray(binary_mask > 0.5, dtype=bool)
    d_in = distance_transform_edt(bm).astype(np.float32)
    d_out = distance_transform_edt(~bm).astype(np.float32)
    signed = d_out - d_in  # negative inside, positive outside

    alpha = np.where(signed > 0, 1.0, 0.0).astype(np.float32)
    collar = (signed > -collar_width) & (signed <= 0)
    alpha[collar] = 0.5 * (1 + np.cos(np.pi * signed[collar] / collar_width))

    outer = binary_dilation(bm, iterations=outer_dilate + collar_width)
    return alpha.astype(np.float32), outer.astype(np.float32)


# -- LHS multiply in half-volume -------------------------------------------

def _lhs_matvec_half(W_half, lhs_tri, reg_diag, q, unpack_fn=None):
    """(LHS + diag(reg)) @ W  in half-volume, chunked."""
    n = lhs_tri.shape[0]
    is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
    out = reg_diag * W_half
    for i0 in range(0, n, _MATVEC_CHUNK):
        i1 = min(i0 + _MATVEC_CHUNK, n)
        L = unpack_fn(lhs_tri[i0:i1], q) if is_tri else lhs_tri[i0:i1]
        out = out.at[i0:i1].add(jnp.einsum("vij,vj->vi", L, W_half[i0:i1]))
    return out


# -- Scatter / gather helpers -----------------------------------------------

def _scatter_to_real(x_sup, support_idx, volume_shape, q):
    """(n_support, q) → (q, D, D, D) real via scatter."""
    vol = int(np.prod(volume_shape))
    full = jnp.zeros((q, vol), dtype=jnp.float32)
    full = full.at[:, support_idx].set(
        jnp.asarray(x_sup.T, dtype=jnp.float32))
    return full.reshape(q, *volume_shape)


def _gather_from_real(W_real, support_idx, q):
    """(q, D, D, D) real → (n_support, q)."""
    vol = W_real.shape[1] * W_real.shape[2] * W_real.shape[3]
    return W_real.reshape(q, vol)[:, support_idx].T


# -- Core matvec -----------------------------------------------------------

def _matvec(x_sup, lhs_tri, reg_diag, alpha_sup, support_idx,
            volume_shape, lam, q, G_real, unpack_fn):
    """Operator:  G · iFFT[ D · FFT[ G · w ] ]  +  λ α · w

    x_sup: (n_support, q) real.
    G_real: (D,D,D) real-space gridding kernel (or None → skip gridding).
    """
    hvs = ftu.get_real_fft_packed_shape(volume_shape)
    hv = int(np.prod(hvs))

    # scatter to full grid, apply G in real space, FFT
    W_real = _scatter_to_real(x_sup, support_idx, volume_shape, q)
    if G_real is not None:
        W_real = G_real[None] * W_real

    # rfft, batched over PCs
    W_half_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        W_half_parts.append(
            ftu.get_dft3_real(W_real[j0:j1]).reshape(j1 - j0, hv).T)
    W_half = jnp.concatenate(W_half_parts, axis=1)  # (half_vol, q)

    # D multiply in Fourier
    DW = _lhs_matvec_half(W_half, lhs_tri, reg_diag, q, unpack_fn)

    # iFFT, apply G, gather
    result_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        nb = j1 - j0
        real = ftu.get_idft3_real(
            DW[:, j0:j1].T.reshape(nb, *hvs), volume_shape)
        if G_real is not None:
            real = G_real[None] * real
        vol = int(np.prod(volume_shape))
        result_parts.append(real.reshape(nb, vol)[:, support_idx].T)
    out = jnp.concatenate(result_parts, axis=1)

    # penalty
    return out + lam * alpha_sup[:, None] * x_sup


# -- CG solver -------------------------------------------------------------

def _cg(matvec, rhs, x0, maxiter, tol, label="CG", recompute_every=10):
    """Plain CG with periodic residual recomputation (float32 stability)."""
    t0 = time.time()
    ip = lambda a, b: float(jnp.sum(a * b))

    r = rhs - matvec(x0)
    p = r.copy()
    rr = ip(r, r)
    b2 = max(ip(rhs, rhs), 1e-30)
    x = x0
    residuals, timings = [], []

    for it in range(maxiter):
        Ap = matvec(p)
        pAp = ip(p, Ap)
        if pAp < 1e-30:
            break
        alpha = rr / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        if (it + 1) % recompute_every == 0:
            r = rhs - matvec(x)
        rr_new = ip(r, r)
        rel = float(jnp.sqrt(rr_new / b2))
        residuals.append(rel)
        timings.append(time.time() - t0)
        if rel < tol:
            logger.info("%s converged iter %d rr=%.2e", label, it + 1, rel)
            break
        p = r + (rr_new / max(rr, 1e-30)) * p
        rr = rr_new
        if (it + 1) % 10 == 0:
            logger.info("%s iter %d rr=%.2e t=%.1fs",
                        label, it + 1, rel, timings[-1])

    if residuals and residuals[-1] >= tol:
        logger.info("%s maxiter=%d rr=%.2e", label, maxiter, residuals[-1])
    return x, {"residuals": residuals, "timings": timings,
               "total_time": time.time() - t0, "n_iters": len(residuals)}


# -- Public API -------------------------------------------------------------

def solve(lhs_tri, rhs_fourier, reg_diag, mask, volume_shape,
          lam=100.0, collar_width=5, outer_dilate=3,
          W0_real=None, maxiter=50, tol=1e-4,
          unpack_fn=None, use_gridding=True):
    """Soft-alpha M-step with optional gridding in the objective.

    Operator:  H w = G · iFFT[ D · FFT[ G · w ] ]  +  λ α · w
    where G(x) = sinc²(x/D) per axis acts in real space.

    When use_gridding=True the solution is the deconvolved volume —
    no post-processing gridding correction needed.
    """
    q = rhs_fourier.shape[1]
    hvs = ftu.get_real_fft_packed_shape(volume_shape)
    hv = int(np.prod(hvs))
    vol = int(np.prod(volume_shape))

    mask_np = np.asarray(mask).reshape(volume_shape)
    alpha, outer = build_alpha_weight(mask_np, collar_width, outer_dilate)
    sup_idx = jnp.where(jnp.asarray(outer).ravel() > 0.5)[0]
    n_sup = sup_idx.shape[0]
    alpha_sup = jnp.asarray(alpha).ravel()[sup_idx]

    G_real = jnp.asarray(compute_gridding_kernel_real(volume_shape)) \
        if use_gridding else None

    logger.info("solve: %s n_sup=%d (%.1f%%) collar=%d λ=%.0f grid=%s",
                volume_shape, n_sup, 100 * n_sup / vol,
                collar_width, lam, use_gridding)

    mv = lambda x: _matvec(x, lhs_tri, reg_diag, alpha_sup, sup_idx,
                           volume_shape, lam, q, G_real, unpack_fn)

    # RHS:  G · iFFT[ rhs ]  gathered at support
    rhs_real_parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        nb = j1 - j0
        rhs_r = ftu.get_idft3_real(
            rhs_fourier[:, j0:j1].T.reshape(nb, *hvs), volume_shape)
        if G_real is not None:
            rhs_r = G_real[None] * rhs_r
        rhs_real_parts.append(rhs_r.reshape(nb, vol)[:, sup_idx].T)
    rhs_sup = jnp.concatenate(rhs_real_parts, axis=1)

    # Initial guess
    if W0_real is not None:
        x0 = _gather_from_real(jnp.asarray(W0_real).reshape(q, *volume_shape),
                               sup_idx, q)
    else:
        # Per-voxel Fourier solve → iFFT → gather
        parts = []
        for i0 in range(0, hv, _MATVEC_CHUNK):
            i1 = min(i0 + _MATVEC_CHUNK, hv)
            is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
            L = unpack_fn(lhs_tri[i0:i1], q) if is_tri else lhs_tri[i0:i1]
            Dc = L.at[:, jnp.arange(q), jnp.arange(q)].add(reg_diag[i0:i1])
            parts.append(
                jnp.linalg.solve(Dc, rhs_fourier[i0:i1, :, None])[..., 0])
        W0h = jnp.concatenate(parts, axis=0)
        W0r = ftu.get_idft3_real(W0h.T.reshape(q, *hvs), volume_shape)
        x0 = _gather_from_real(W0r, sup_idx, q)

    x, info = _cg(mv, rhs_sup, x0, maxiter, tol,
                   f"SA(λ={lam},c={collar_width},g={use_gridding})")

    W = _scatter_to_real(x, sup_idx, volume_shape, q)
    info.update(n_support=int(n_sup), collar_width=collar_width,
                lam=lam, use_gridding=use_gridding)
    return W, info
