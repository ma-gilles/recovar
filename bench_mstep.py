#!/usr/bin/env python3
"""PPCA M-step solver comparison.

All solvers work in real space with standard dot-product inner product.
See docs/math/masked_mstep.md for formulations.

Methods:
  1. naive:          per-voxel (A+Λ)^{-1}d → K^{-1} → mask (post-processing)
  2. hard:           V=EZ, reduced coords, K in operator
  3. hard+precond:   same + circulant preconditioner
  4. soft:           full grid, K + μα² penalty
  5. soft+precond:   same + block preconditioner

Core operator (all CG methods):
  K · iFFT[(A + Λ) · FFT[K · V]]
Both data (A) and prior (Λ) act on KV.  This ensures the naive solution
V = K^{-1}(A+Λ)^{-1}d is the exact unmasked optimum.

Defaults: 128³, q=10, 50k images, 20 EM iters.
Prior: (1/NPC) * gt.get_fourier_variances(), radially averaged.
Mask: moving GT mask.  collar=4%.  μ=100.
"""
import argparse, json, logging, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

if "--float64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "True"

import jax, jax.numpy as jnp
print(f"JAX: {jax.devices()}, x64={jax.config.x64_enabled}", flush=True)
assert any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in jax.devices())

try:
    from recovar.cuda_backproject import _ensure_ffi; _ensure_ffi()
except Exception: pass

from recovar.ppca import ppca
from recovar.ppca.ppca_scale_sweep import _load_simulated_dataset, _with_trailing_separator
import recovar.core.fourier_transform_utils as ftu
from recovar.core.mask import make_moving_gt_mask
from recovar.output import metrics
from recovar.reconstruction import regularization
from recovar.utils import batch_make_radial_image
from scipy.ndimage import distance_transform_edt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bench")

_CHUNK = 100_000
_PC_BATCH = 4

# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def compute_G(vs):
    D = vs[0]; c = np.arange(D, dtype=np.float32) - D / 2; n = c / D
    def sinc(a):
        s = np.where(np.abs(a) < 1e-8, 1.0, a)
        return np.where(np.abs(a) < 1e-8, 1.0, np.sin(np.pi * s) / (np.pi * s))
    g = sinc(n) ** 2
    return jnp.array((g[:, None, None] * g[None, :, None] * g[None, None, :]).astype(np.float32))


def build_alpha(mask_bin, collar):
    bm = np.asarray(mask_bin > 0.5, dtype=bool)
    d_in = distance_transform_edt(bm).astype(np.float32)
    d_out = distance_transform_edt(~bm).astype(np.float32)
    signed = d_out - d_in
    alpha = np.where(signed > 0, 1.0, 0.0).astype(np.float32)
    cz = (signed > -collar) & (signed <= 0)
    alpha[cz] = 0.5 * (1 + np.cos(np.pi * signed[cz] / collar))
    return jnp.array(alpha)


def _batched_rfft(V, vs):
    """(q, D, D, D) real → (half_vol, q) complex."""
    q = V.shape[0]; hvs = ftu.get_real_fft_packed_shape(vs); hv = int(np.prod(hvs))
    parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        parts.append(ftu.get_dft3_real(V[j0:j1]).reshape(j1 - j0, hv).T)
    return jnp.concatenate(parts, axis=1)


def _batched_irfft(W_h, vs, q):
    """(half_vol, q) complex → (q, D, D, D) real."""
    hvs = ftu.get_real_fft_packed_shape(vs)
    parts = []
    for j0 in range(0, q, _PC_BATCH):
        j1 = min(j0 + _PC_BATCH, q)
        parts.append(ftu.get_idft3_real(W_h[:, j0:j1].T.reshape(j1 - j0, *hvs), vs))
    return jnp.concatenate(parts, axis=0)


def _A_mul_fourier(W_h, lhs_tri, q, unpack_fn):
    """A(ξ) * W(ξ), chunked. (hv, q) → (hv, q)."""
    hv = W_h.shape[0]; out = jnp.zeros_like(W_h)
    is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
    for i0 in range(0, hv, _CHUNK):
        i1 = min(i0 + _CHUNK, hv)
        L = unpack_fn(lhs_tri[i0:i1], q) if is_tri else lhs_tri[i0:i1]
        out = out.at[i0:i1].set(jnp.einsum("vij,vj->vi", L, W_h[i0:i1]))
    return out


def _AL_solve_fourier(W_h, lhs_tri, reg_diag, q, unpack_fn, lhs_scale=1.0):
    """(scale*A(ξ) + Λ(ξ))^{-1} * W(ξ), chunked."""
    hv = W_h.shape[0]; is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
    parts = []
    for i0 in range(0, hv, _CHUNK):
        i1 = min(i0 + _CHUNK, hv)
        L = unpack_fn(lhs_tri[i0:i1], q) if is_tri else lhs_tri[i0:i1]
        if lhs_scale != 1.0:
            L = lhs_scale * L
        D = L.at[:, jnp.arange(q), jnp.arange(q)].add(reg_diag[i0:i1])
        parts.append(jnp.linalg.solve(D, W_h[i0:i1, :, None])[..., 0])
    return jnp.concatenate(parts, axis=0)


def _apply_fourier_op(V_real, lhs_tri, reg_diag, q, vs, unpack_fn, G=None):
    """Apply K iFFT[(A+Λ) FFT[K V]] in real space.

    V_real: (q, D, D, D).  Returns (q, D, D, D).
    When G (gridding kernel K) is provided, BOTH data and prior act on KV:
      K · iFFT[(A + Λ) · FFT[KV]]
    Without G: iFFT[(A + Λ) · FFT[V]].
    """
    KV = G[None] * V_real if G is not None else V_real
    KV_h = _batched_rfft(KV, vs)
    AKV_h = _A_mul_fourier(KV_h, lhs_tri, q, unpack_fn)
    result_h = AKV_h + reg_diag * KV_h   # (A + Λ) applied to KV
    result = _batched_irfft(result_h, vs, q)
    return G[None] * result if G is not None else result


# ═══════════════════════════════════════════════════════════════════════
# Krylov solvers (real space, standard dot product)
# ═══════════════════════════════════════════════════════════════════════

def _dot(a, b):
    return float(jnp.sum(a * b))


def _cg(matvec, b, x0, maxiter, tol, precond=None, label="CG"):
    """Preconditioned CG. All vectors are flat real arrays."""
    t0 = time.time()
    logger.info("%s: dtype=%s size=%d", label, x0.dtype, x0.size)
    x = x0.copy()
    r = b - matvec(x)
    z = precond(r) if precond else r
    p = z.copy(); rz = _dot(r, z)
    b2 = max(_dot(b, b), 1e-30)
    residuals = []
    for it in range(maxiter):
        Ap = matvec(p); pAp = _dot(p, Ap)
        if pAp < 1e-30: break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rr = _dot(r, r); rel = float(jnp.sqrt(rr / b2))
        residuals.append(rel)
        if rel < tol:
            logger.info("%s converged %d rr=%.2e", label, it+1, rel); break
        z = precond(r) if precond else r
        rz_new = _dot(r, z)
        p = z + (rz_new / max(abs(rz), 1e-30)) * p; rz = rz_new
        if (it + 1) % 10 == 0:
            logger.info("%s %d rr=%.2e t=%.1fs", label, it+1, rel, time.time()-t0)
    return x, {"residuals": residuals, "n_iters": len(residuals), "total_time": time.time()-t0}


def _minres(matvec, b, x0, maxiter, tol, precond=None, label="MINRES"):
    """Preconditioned MINRES (Paige-Saunders). All vectors are flat real arrays.

    Adapted from scipy.sparse.linalg.minres.
    """
    t0 = time.time()
    logger.info("%s: dtype=%s size=%d", label, x0.dtype, x0.size)
    n = x0.size
    x = x0.copy()
    b2 = max(float(jnp.sqrt(_dot(b, b))), 1e-30)

    r1 = b - matvec(x)
    y = precond(r1) if precond else r1
    beta1 = _dot(r1, y)
    if beta1 < 0:
        logger.warning("%s: <r, M^{-1}r>=%.2e < 0", label, beta1)
        return x, {"residuals": [], "n_iters": 0, "total_time": 0}
    beta1 = float(jnp.sqrt(beta1))

    oldb = 0.0; beta = beta1; dbar = 0.0; epsln = 0.0; phibar = beta1
    rhs1 = beta1; rhs2 = 0.0
    tnorm2 = 0.0; cs = -1.0; sn = 0.0
    w = jnp.zeros_like(x); w2 = jnp.zeros_like(x)
    r2 = r1.copy()
    residuals = []

    for it in range(maxiter):
        s = 1.0 / beta
        v = s * y
        y = matvec(v)
        if it > 0:
            y = y - (beta / oldb) * r1
        alpha = _dot(v, y)
        y = y - (alpha / beta) * r2
        r1 = r2.copy(); r2 = y.copy()
        y = precond(r2) if precond else r2
        oldb = beta
        beta = _dot(r2, y)
        if beta < 0:
            logger.warning("%s iter %d: <r,z>=%.2e<0", label, it+1, beta)
            break
        beta = float(jnp.sqrt(beta))
        tnorm2 = tnorm2 + alpha**2 + oldb**2 + beta**2

        if it == 0:
            if beta / beta1 <= 10 * jnp.finfo(x.dtype).eps:
                break

        oldeps = epsln
        delta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha
        epsln = sn * beta
        dbar = -cs * beta

        gamma = float(jnp.sqrt(gbar**2 + beta**2))
        gamma = max(gamma, 1e-30)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar

        denom = 1.0 / gamma
        w1 = w2.copy()
        w2 = w.copy()
        w = (v - oldeps * w1 - delta * w2) * denom
        x = x + phi * w

        rel = abs(phibar) / b2
        residuals.append(rel)
        if rel < tol:
            logger.info("%s converged %d rr=%.2e", label, it+1, rel); break
        if (it + 1) % 10 == 0:
            logger.info("%s %d rr=%.2e t=%.1fs", label, it+1, rel, time.time()-t0)

    return x, {"residuals": residuals, "n_iters": len(residuals), "total_time": time.time()-t0}


# ═══════════════════════════════════════════════════════════════════════
# Hard mask solver (real space, reduced coordinates V=EZ)
# See docs/math/masked_mstep.md § Hard mask
# ═══════════════════════════════════════════════════════════════════════

def solve_hard(lhs_tri, rhs_h, reg_diag, mask, vs, q,
               G, maxiter, tol, unpack_fn, precondition, W0_real,
               use_float64=False, krylov_fn=None):
    sup = jnp.where(jnp.asarray(mask).ravel() > 0.5)[0]
    n_sup = sup.shape[0]; vol = int(np.prod(vs))
    N = vol
    k_eff_sq = float(jnp.sum(G ** 2)) / N
    logger.info("hard: n_sup=%d (%.1f%%) k_eff²=%.4f precond=%s",
                n_sup, 100*n_sup/vol, k_eff_sq, precondition)

    dt = jnp.float64 if use_float64 else jnp.float32
    if use_float64:
        lhs_tri = lhs_tri.astype(jnp.complex128 if jnp.iscomplexobj(lhs_tri) else jnp.float64)
        reg_diag = reg_diag.astype(jnp.float64)
        rhs_h = rhs_h.astype(jnp.complex128)
        G = G.astype(jnp.float64)

    def scatter(Z_flat):
        """(n_sup*q,) → (q, D, D, D)"""
        Z = Z_flat.reshape(n_sup, q)
        f = jnp.zeros((q, vol), dtype=dt)
        return f.at[:, sup].set(Z.T).reshape(q, *vs)

    def gather(V):
        """(q, D, D, D) → (n_sup*q,)"""
        return V.reshape(q, vol)[:, sup].T.ravel()

    def matvec(Z_flat):
        V = scatter(Z_flat)
        result = _apply_fourier_op(V, lhs_tri, reg_diag, q, vs, unpack_fn, G)
        return gather(result)

    # RHS: E^T K iFFT[d]
    d_real = _batched_irfft(rhs_h, vs, q)
    rhs_flat = gather(G[None] * d_real).astype(dt)

    # Preconditioner: E^T iFFT[(k_eff²(A+Λ))^{-1} FFT[EZ]]
    if precondition:
        prec_reg = k_eff_sq * reg_diag if G is not None else reg_diag
        def precond(Z_flat):
            V = scatter(Z_flat)
            V_h = _batched_rfft(V, vs)
            S_h = _AL_solve_fourier(V_h, lhs_tri, prec_reg, q, unpack_fn, lhs_scale=k_eff_sq)
            return gather(_batched_irfft(S_h, vs, q))
    else:
        precond = None

    # Initial guess: per-voxel Wiener → K^{-1} → gather
    if W0_real is not None:
        x0 = gather(jnp.asarray(W0_real, dtype=dt).reshape(q, *vs))
    else:
        W0_h = _AL_solve_fourier(rhs_h, lhs_tri, reg_diag, q, unpack_fn)
        W0_r = _batched_irfft(W0_h, vs, q) / jnp.maximum(G[None], 0.01)
        x0 = gather(W0_r).astype(dt)

    _solve = krylov_fn or _cg
    Z_flat, info = _solve(matvec, rhs_flat, x0, maxiter, tol, precond, "hard")
    W_real = scatter(Z_flat).astype(jnp.float32)
    return W_real, info


# ═══════════════════════════════════════════════════════════════════════
# Soft mask solver (real space, full grid)
# See docs/math/masked_mstep.md § Soft mask
# ═══════════════════════════════════════════════════════════════════════

def solve_soft(lhs_tri, rhs_h, reg_diag, alpha, vs, q,
               G, mu, maxiter, tol, unpack_fn, precondition, W0_real,
               use_float64=False, krylov_fn=None):
    """Soft mask: min_V (KV)*A(KV) + V*ΛV + μ||αV||².  CG/MINRES in real space.

    See docs/math/masked_mstep.md § Soft mask.

    Preconditioner (when precondition=True):
      Additive block preconditioner splitting interior I vs outside O:
        P^{-1} = E_I P_I^{-1} E_I^T + E_O D_O^{-1} E_O^T
      where P_I^{-1} is the circulant hard-mask preconditioner on I,
      and D_O^{-1} is a pointwise q×q block solve on O using the
      Fourier-averaged A and Λ.
      See docs/math/masked_mstep.md § Soft mask preconditioner.
    """
    vol = int(np.prod(vs)); N = vol
    alpha_sq = (alpha ** 2).ravel()
    eps_mask = 1e-6

    dt = jnp.float64 if use_float64 else jnp.float32
    if use_float64:
        lhs_tri = lhs_tri.astype(jnp.complex128 if jnp.iscomplexobj(lhs_tri) else jnp.float64)
        reg_diag = reg_diag.astype(jnp.float64)
        rhs_h = rhs_h.astype(jnp.complex128)
        G = G.astype(jnp.float64)
        alpha_sq = alpha_sq.astype(jnp.float64)

    # Interior/outside split
    alpha_flat = alpha.ravel()
    I_idx = jnp.where(alpha_flat < 1.0 - eps_mask)[0]
    O_idx = jnp.where(alpha_flat >= 1.0 - eps_mask)[0]
    n_I = I_idx.shape[0]; n_O = O_idx.shape[0]
    G_flat = G.ravel()
    kI2 = float(jnp.sum(G_flat[I_idx] ** 2)) / max(n_I, 1)

    logger.info("soft: mu=%g n_I=%d (%.1f%%) n_O=%d kI²=%.4f precond=%s",
                mu, n_I, 100*n_I/vol, n_O, kI2, precondition)

    def matvec(V_flat):
        V = V_flat.reshape(q, *vs)
        result = _apply_fourier_op(V, lhs_tri, reg_diag, q, vs, unpack_fn, G)
        result = result + mu * alpha_sq.reshape(vs)[None] * V
        return result.ravel().astype(dt)

    # RHS: K iFFT[d]
    d_real = _batched_irfft(rhs_h, vs, q)
    rhs_flat = (G[None] * d_real).ravel().astype(dt)

    if precondition:
        # --- Precompute outside block diagonal D_O(x) ---
        # Weighted Fourier mean of A and Λ
        hvs = ftu.get_real_fft_packed_shape(vs); hv = int(np.prod(hvs))
        rfft_w = 2 * jnp.ones(hvs, dtype=dt)
        rfft_w = rfft_w.at[:, :, 0].set(1)
        if vs[0] % 2 == 0:
            rfft_w = rfft_w.at[:, :, -1].set(1)
        rfft_w = rfft_w.reshape(-1) / N

        # A_bar: weighted mean of A(ξ) over Fourier voxels → (q, q)
        is_tri = lhs_tri.ndim == 2 and lhs_tri.shape[1] != q
        A_bar = jnp.zeros((q, q), dtype=lhs_tri.dtype)
        for i0 in range(0, hv, _CHUNK):
            i1 = min(i0 + _CHUNK, hv)
            L = unpack_fn(lhs_tri[i0:i1], q) if is_tri else lhs_tri[i0:i1]
            A_bar = A_bar + jnp.einsum("v,vij->ij", rfft_w[i0:i1], L)
        A_bar = A_bar.real.astype(dt)

        # Λ_bar: weighted mean of diag(Λ) → (q,)
        L_bar = jnp.sum(rfft_w[:, None] * reg_diag.real, axis=0).astype(dt)

        # D_O(x) = K(x)² (A_bar + diag(Λ_bar)) + μ α(x)² I_q
        # Prior also gets K² weighting (consistent with K·(A+Λ)·K operator)
        K_O_sq = G_flat[O_idx] ** 2  # (n_O,)
        alpha_O_sq = alpha_sq[O_idx]  # (n_O,)
        # D_O: (n_O, q, q)
        D_O = (K_O_sq[:, None, None] * (A_bar[None, :, :] + jnp.diag(L_bar)[None, :, :])
               + mu * alpha_O_sq[:, None, None] * jnp.eye(q, dtype=dt)[None, :, :])
        # Cholesky factorize
        D_O_chol = jnp.linalg.cholesky(D_O)

        def apply_D_O_inv(R_O):
            """(n_O, q) → (n_O, q) via Cholesky solve."""
            return jax.scipy.linalg.cho_solve((D_O_chol, True), R_O[:, :, None])[:, :, 0]

        # --- Interior hard preconditioner: circulant on I ---
        prec_reg_I = kI2 * reg_diag if G is not None else reg_diag
        def apply_P_I_inv(R_I):
            """(n_I, q) → (n_I, q): scatter → FFT → (kI²(A+Λ))^{-1} → iFFT → gather."""
            V = jnp.zeros((q, vol), dtype=dt)
            V = V.at[:, I_idx].set(R_I.T)
            V = V.reshape(q, *vs)
            V_h = _batched_rfft(V, vs)
            S_h = _AL_solve_fourier(V_h, lhs_tri, prec_reg_I, q, unpack_fn, lhs_scale=kI2)
            S = _batched_irfft(S_h, vs, q)
            return S.reshape(q, vol)[:, I_idx].T

        # --- Additive block preconditioner ---
        def precond(V_flat):
            """P^{-1} = E_I P_I^{-1} E_I^T + E_O D_O^{-1} E_O^T"""
            V = V_flat.reshape(q, vol)
            R_I = V[:, I_idx].T  # (n_I, q)
            R_O = V[:, O_idx].T  # (n_O, q)

            Z_I = apply_P_I_inv(R_I)   # (n_I, q)
            Z_O = apply_D_O_inv(R_O)    # (n_O, q)

            out = jnp.zeros((q, vol), dtype=dt)
            out = out.at[:, I_idx].set(Z_I.T)
            out = out.at[:, O_idx].set(Z_O.T)
            return out.ravel()
    else:
        precond = None

    # Initial guess: zero for cold start, previous iterate for warmstart
    if W0_real is not None:
        x0 = jnp.asarray(W0_real, dtype=dt).reshape(q, *vs).ravel()
    else:
        x0 = jnp.zeros(q * vol, dtype=dt)

    _solve = krylov_fn or _cg
    V_flat, info = _solve(matvec, rhs_flat, x0, maxiter, tol, precond, "soft")
    W_real = V_flat.reshape(q, *vs).astype(jnp.float32)
    return W_real, info


# ═══════════════════════════════════════════════════════════════════════
# Solver wrappers (mstep_solver_fn interface for ppca.EM)
# ═══════════════════════════════════════════════════════════════════════

def make_hard_solver(precondition=False, use_float64=False, krylov_fn=None,
                     override_tol=None, use_grid_correction=True):
    all_res = []
    def fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
           maxiter=20, tol=1e-4, unpack_fn=None):
        _tol = override_tol if override_tol is not None else tol
        q = rhs.shape[1]
        G = compute_G(vol_shape) if use_grid_correction else jnp.ones(vol_shape, dtype=jnp.float32)
        W, info = solve_hard(lhs, rhs, reg, mask, vol_shape, q,
                             G, maxiter, _tol, unpack_fn, precondition, W0_real,
                             use_float64=use_float64, krylov_fn=krylov_fn)
        all_res.append(info.get("residuals", []))
        return W, info
    return fn, all_res


def make_soft_solver(mu=100, collar=5, precondition=False, use_float64=False,
                     krylov_fn=None, override_tol=None, use_grid_correction=True):
    all_res = []
    def fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
           maxiter=20, tol=1e-4, unpack_fn=None):
        _tol = override_tol if override_tol is not None else tol
        q = rhs.shape[1]
        G = compute_G(vol_shape) if use_grid_correction else jnp.ones(vol_shape, dtype=jnp.float32)
        mask_bin = jnp.asarray(mask) > 0.5
        alpha = build_alpha(mask_bin, collar)
        W, info = solve_soft(lhs, rhs, reg, alpha, vol_shape, q,
                             G, mu, maxiter, _tol, unpack_fn, precondition, W0_real,
                             use_float64=use_float64, krylov_fn=krylov_fn)
        all_res.append(info.get("residuals", []))
        return W, info
    return fn, all_res


# ═══════════════════════════════════════════════════════════════════════
# Prior, EM runner, main
# ═══════════════════════════════════════════════════════════════════════

def make_prior(gt, npc, vs):
    fv = gt.get_fourier_variances(contrasted=False)
    per_pc = fv / npc
    radial = regularization.batch_average_over_shells(
        jnp.array(per_pc).reshape(1, -1), vs, 0)
    img = batch_make_radial_image(radial, vs, True).T
    W_prior = np.array(jnp.tile(img, (1, npc)))
    logger.info("Prior: (1/%d)*fourier_var, median=%.2e", npc, float(np.median(per_pc)))
    return W_prior


def run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mask_arr,
           use_pcg, solver_fn, pcg_maxiter, label, n_iter,
           use_gridding_correction=True,
           contrast_mode="none", contrast_grid=None):
    logger.info("=== %s ===", label)
    t0 = time.time()
    U, S, W, ez, sm, idata = ppca.EM(
        cryos, gt_mean, W_init.copy(), W_prior,
        U_gt=U_gt, S_gt=s_gt**2, EM_iter=n_iter,
        use_whitening=False, sparse_PCA=False,
        disc_type_mean="cubic", disc_type="linear_interp",
        return_iteration_data=True, use_pcg_mean=use_pcg,
        volume_mask=mask_arr, pcg_maxiter=pcg_maxiter,
        use_gridding_correction=use_gridding_correction, mstep_solver_fn=solver_fn,
        contrast_mode=contrast_mode, contrast_grid=contrast_grid)
    dt = time.time() - t0
    _, rv, _ = metrics.get_all_variance_scores(U, U_gt, s_gt**2)
    logger.info("  RelVar=%.4f time=%.0fs", rv[-1], dt)
    em_data = []
    for d in idata:
        if isinstance(d, dict):
            em_data.append({k: d.get(k) for k in
                ["Neg_LL_Total", "Neg_LL_Data", "Neg_LL_Prior", "Rel_Var_Explained"]})
    vs = cryos[0].volume_shape if hasattr(cryos, '__len__') else cryos.volume_shape
    q = W_init.shape[1]
    W_real = np.array(ftu.get_idft3(W.T.reshape(q, *vs)).real)
    return {"label": label, "relvar": float(rv[-1]),
            "relvar_per_pc": [float(x) for x in rv],
            "time": dt, "em_data": em_data, "W_real": W_real}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--n-images", type=int, default=10000)
    parser.add_argument("--em-iters", type=int, default=20)
    parser.add_argument("--cg-maxiter", type=int, default=50)
    parser.add_argument("--mu", type=float, default=100.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--solver", type=str, default="cg", choices=["cg", "minres"])
    parser.add_argument("--mask-dilate", type=int, default=0,
        help="Extra dilation of binary mask before passing to solvers (match soft mask ramp)")
    parser.add_argument("--dataset-dir", type=str, default=None,
        help="Override dataset directory")
    parser.add_argument("--contrast-std", type=float, default=0.0,
        help="Per-image contrast std (0 = no contrast variation)")
    parser.add_argument("--noise-level", type=float, default=1.0,
        help="Noise level for dataset generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-grid-correction", action="store_true",
        help="Set G=1 (identity) for all methods — isolate effect of gridding kernel K")
    parser.add_argument("--contrast-mode", type=str, default="none",
        choices=["none", "profile", "marginalize"],
        help="Contrast handling: none (c=1), profile (MAP), marginalize (quadrature)")
    parser.add_argument("--methods", type=str,
        default="hard")
    args = parser.parse_args()

    gs = args.grid_size; npc = args.n_pcs
    collar = max(3, round(0.04 * gs))
    methods = [m.strip() for m in args.methods.split(",")]
    krylov_fn = _minres if args.solver == "minres" else _cg

    use_K = not args.no_grid_correction
    suffix = "_f64" if args.float64 else "_f32"
    if args.solver != "cg": suffix += f"_{args.solver}"
    if args.mask_dilate > 0: suffix += f"_dil{args.mask_dilate}"
    if not use_K: suffix += "_noK"
    if args.contrast_mode != "none": suffix += f"_{args.contrast_mode}"
    base_dir = "/scratch/gpfs/GILLES/mg6942/tmp/convergence_tests"
    out_dir = f"{base_dir}/mstep_{gs}_{npc}pc{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    # Generate or load dataset
    if args.dataset_dir:
        ds_dir = args.dataset_dir
    else:
        # Auto-generate from B-factored volumes
        vol_path = f"/scratch/gpfs/GILLES/mg6942/tmp/ppca_bfac60_n1_128/true_volumes"
        tag = f"bench_{gs}_n{args.noise_level}_c{args.contrast_std}_{args.n_images}"
        ds_dir = f"/scratch/gpfs/GILLES/mg6942/tmp/{tag}/test_dataset"
        if not os.path.exists(os.path.join(ds_dir, "particles.star")):
            from recovar.simulation import simulator
            voxel_size = 4.25 * 128 / gs  # source vols are 128³ at 4.25 Å/px
            logger.info("Generating dataset: %s (voxel_size=%.2f)", ds_dir, voxel_size)
            simulator.generate_synthetic_dataset(
                ds_dir, voxel_size=voxel_size, volumes_path_root=vol_path,
                n_images=args.n_images, grid_size=gs,
                noise_level=args.noise_level, noise_model="radial1",
                contrast_std=args.contrast_std, noise_scale_std=0.0,
                dataset_params_option="dataset1", disc_type="linear_interp",
                trailing_zero_format_in_vol_name=True,
                put_extra_particles=False, percent_outliers=0.0)
        else:
            logger.info("Dataset exists: %s", ds_dir)
    cryos, sim_info, gt, nv = _load_simulated_dataset(
        _with_trailing_separator(ds_dir), gs, args.n_images, lazy=False)
    vs = gt.volume_shape

    U_gt_all, s_gt_all, _ = gt.get_vol_svd()
    U_gt, s_gt = U_gt_all[:, :npc], s_gt_all[:npc]
    gt_mean = gt.get_mean()
    W_prior = make_prior(gt, npc, vs)

    real_vols = [np.asarray(ftu.get_idft3(gt.volumes[i].reshape(vs)).real)
                 for i in range(gt.volumes.shape[0])]
    mov_soft, mov_bin = make_moving_gt_mask(real_vols, vs)
    if args.mask_dilate > 0:
        from scipy.ndimage import binary_dilation
        mov_bin = binary_dilation(mov_bin, iterations=args.mask_dilate)
        from recovar.core.mask import soften_volume_mask
        mov_soft = soften_volume_mask(mov_bin, kern_rad=3)
    mask = np.array(mov_soft, dtype=np.float32)
    n_mask = int(np.sum(mov_bin))
    contrast_grid = np.linspace(0.3, 1.7, 16) if args.contrast_mode != "none" else None
    logger.info("Setup: %d³ q=%d mask=%.1f%% collar=%d μ=%g solver=%s f64=%s K=%s contrast=%s",
                gs, npc, 100*n_mask/np.prod(vs), collar, args.mu, args.solver, args.float64, use_K, args.contrast_mode)

    np.random.seed(args.seed)
    W_init = jnp.array(np.random.randn(gt_mean.shape[0], npc).astype(np.float32) * 0.01)

    results = {}

    # When a CG solver has K inside its operator, the solution is already
    # deblurred — the EM must NOT apply griddingCorrect again (double correction).
    # Naive doesn't have K in its solve, so it needs the EM's K^{-1} post-processing.
    em_K_naive = use_K          # naive needs EM to apply K^{-1}
    em_K_solver = False         # CG/ADMM with K handle it internally

    if "naive" in methods:
        r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                   mask, False, None, 20, "naive", args.em_iters,
                   use_gridding_correction=em_K_naive,
                   contrast_mode=args.contrast_mode, contrast_grid=contrast_grid)
        r["cg_residuals"] = []
        results["naive"] = r

    if "hard" in methods:
        fn, res = make_hard_solver(precondition=False, use_float64=args.float64,
                                   krylov_fn=krylov_fn, override_tol=args.tol,
                                   use_grid_correction=use_K)
        r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                   mask, False, fn, args.cg_maxiter, "hard", args.em_iters,
                   use_gridding_correction=em_K_solver if use_K else False,
                   contrast_mode=args.contrast_mode, contrast_grid=contrast_grid)
        r["cg_residuals"] = [list(x) for x in res]
        results["hard"] = r

    if "hard_precond" in methods:
        fn, res = make_hard_solver(precondition=True, use_float64=args.float64,
                                   krylov_fn=krylov_fn, override_tol=args.tol,
                                   use_grid_correction=use_K)
        r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                   mask, False, fn, args.cg_maxiter, "hard+precond", args.em_iters,
                   use_gridding_correction=em_K_solver if use_K else False,
                   contrast_mode=args.contrast_mode, contrast_grid=contrast_grid)
        r["cg_residuals"] = [list(x) for x in res]
        results["hard+precond"] = r

    if "soft" in methods:
        fn, res = make_soft_solver(mu=args.mu, collar=collar, precondition=False,
                                   use_float64=args.float64, krylov_fn=krylov_fn,
                                   override_tol=args.tol, use_grid_correction=use_K)
        r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                   mask, False, fn, args.cg_maxiter, "soft", args.em_iters,
                   use_gridding_correction=em_K_solver if use_K else False,
                   contrast_mode=args.contrast_mode, contrast_grid=contrast_grid)
        r["cg_residuals"] = [list(x) for x in res]
        results["soft"] = r

    if "soft_precond" in methods:
        fn, res = make_soft_solver(mu=args.mu, collar=collar, precondition=True,
                                   use_float64=args.float64, krylov_fn=krylov_fn,
                                   override_tol=args.tol, use_grid_correction=use_K)
        r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                   mask, False, fn, args.cg_maxiter, "soft+precond", args.em_iters,
                   use_gridding_correction=em_K_solver if use_K else False,
                   contrast_mode=args.contrast_mode, contrast_grid=contrast_grid)
        r["cg_residuals"] = [list(x) for x in res]
        results["soft+precond"] = r

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"{gs}³ q={npc} {args.n_images} imgs {args.em_iters} EM collar={collar} μ={args.mu}")
    print(f"solver={args.solver} f64={args.float64} tol={args.tol}")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'RelVar':>8} {'Time':>8}")
    print("-" * 40)
    for n, r in results.items():
        print(f"{n:<20} {r['relvar']:8.4f} {r['time']:8.0f}s")

    for n, r in results.items():
        print(f"\n--- {n} ---")
        for i, d in enumerate(r["em_data"]):
            print(f"  {i:2d} RV={d.get('Rel_Var_Explained',''):>10} "
                  f"LL_D={d.get('Neg_LL_Data',''):>14} LL_P={d.get('Neg_LL_Prior',''):>12}")

    save = {k: {kk: vv for kk, vv in v.items() if kk != "W_real"}
            for k, v in results.items()}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(save, f, indent=2, default=str)

    # ── Plots ──
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax = axes[0]
        for n, r in results.items():
            rvs = [d["Rel_Var_Explained"] for d in r["em_data"]
                   if d.get("Rel_Var_Explained") is not None]
            if rvs: ax.plot(range(1, len(rvs)+1), rvs, "o-", label=n, ms=3)
        ax.set_xlabel("EM iter"); ax.set_ylabel("RelVar")
        ax.set_title("RelVar"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = axes[1]
        for n, r in results.items():
            ll = [d["Neg_LL_Data"] for d in r["em_data"]
                  if d.get("Neg_LL_Data") is not None]
            if ll: ax.plot(range(1, len(ll)+1), ll, "o-", label=n, ms=3)
        ax.set_xlabel("EM iter"); ax.set_ylabel("Neg LL (data)")
        ax.set_title("Data LL"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = axes[2]
        for n, r in results.items():
            if r["cg_residuals"]:
                ax.semilogy(r["cg_residuals"][0], label=f"{n} (iter 0)", alpha=0.7)
                if len(r["cg_residuals"]) > 1:
                    ax.semilogy(r["cg_residuals"][-1], "--", label=f"{n} (last)", alpha=0.5)
        ax.set_xlabel(f"{args.solver.upper()} iter"); ax.set_ylabel("Residual")
        ax.set_title(f"{args.solver.upper()} convergence"); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

        plt.suptitle(f"{gs}³ q={npc} collar={collar} μ={args.mu} {args.solver} f64={args.float64}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=150); plt.close()

        # Eigenvector slices
        n_show = min(3, npc)
        mnames = list(results.keys())
        W_gt = U_gt[:, :n_show] * s_gt[:n_show]
        W_gt_real = np.array(ftu.get_idft3(W_gt.T.reshape(n_show, *vs)).real)
        mid = gs // 2
        fig, axes = plt.subplots(n_show, len(mnames)+1,
                                  figsize=(3*(len(mnames)+1), 3*n_show))
        if n_show == 1: axes = axes[None, :]
        for k in range(n_show):
            gt_slice = W_gt_real[k, mid]
            gt_norm = max(float(np.max(np.abs(gt_slice))), 1e-10)
            axes[k, 0].imshow(gt_slice / gt_norm, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[k, 0].set_title(f"GT PC{k}", fontsize=8); axes[k, 0].axis("off")
            for j, n in enumerate(mnames):
                pc_slice = results[n]["W_real"][k, mid]
                pc_norm = max(float(np.max(np.abs(pc_slice))), 1e-10)
                axes[k, j+1].imshow(pc_slice / pc_norm, cmap="RdBu_r", vmin=-1, vmax=1)
                axes[k, j+1].set_title(f"{n} PC{k}", fontsize=7); axes[k, j+1].axis("off")
        plt.suptitle(f"Eigenvectors: {gs}³ q={npc}", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "slices.png"), dpi=150); plt.close()
        print(f"\nPlots: {out_dir}/{{convergence,slices}}.png")
    except Exception as e:
        import traceback; print(f"Plot failed: {e}"); traceback.print_exc()

    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
