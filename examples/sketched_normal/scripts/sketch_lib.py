"""Shared library: the matrix-free proximal R4SVD iteration with optional
radial D-prior damping computed from GT volumes.

Knobs:
  method:  soft (SVT), hard (top-k projection)
  prior:   none, radial_D (damping from shell-averaged power of (vol-mean))
  init:    cold, gt
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.core import linalg
from recovar.output import metrics
from recovar.reconstruction import regularization

# -------------------- radial D prior --------------------


def compute_radial_D2(gt_results, volume_shape, eps_rel=1e-4):
    """Shell-averaged |sqrt(p_k)(vol_k - mean)|^2 → radial D^2 at each Fourier voxel.

    Regularizer is ||D X||_F^2 with D^2(|k|) = 1 / (shell_power(|k|) + eps).
    No rescaling, no normalization — the scale is the natural one implied by the
    ground-truth heterogeneity Fourier power, same spirit as PPCA's diag(1/W_prior).
    """
    mean = np.asarray(gt_results.get_mean()).astype(np.complex64)
    vols = np.stack([np.asarray(gt_results.volumes[k]).astype(np.complex64) for k in range(len(gt_results.volumes))])
    probs = np.asarray(gt_results.get_probs_of_state()).astype(np.float32)
    diff = np.sqrt(probs)[:, None] * (vols - mean[None, :])  # (K, V)
    power = (np.abs(diff) ** 2).sum(axis=0)  # (V,)
    shells = np.asarray(regularization.batch_average_over_shells(jnp.asarray(power[None, :]), volume_shape, 0))[0]
    eps = eps_rel * float(shells.max())
    D2_radial_shells = 1.0 / (shells + eps)
    D2_full = np.asarray(utils.batch_make_radial_image(jnp.asarray(D2_radial_shells[None, :]), volume_shape, True))[
        0
    ].reshape(-1)
    return D2_full.astype(np.float32), shells.astype(np.float32), D2_radial_shells.astype(np.float32)


def apply_D2_to_basis(U_real, D2_fourier, volume_shape, V_SIZE):
    """Apply D^2 (radial, Fourier-diagonal) to each real-space column of U.
    X ∈ R^(V, r) real → DFT each column → multiply by D2_fourier → IDFT → real part.
    """
    if U_real.shape[1] == 0:
        return np.zeros_like(U_real)
    r = U_real.shape[1]
    U_f = np.asarray(linalg.batch_dft3(jnp.asarray(U_real, dtype=jnp.float32), volume_shape, r))  # (V, r) complex
    U_f_weighted = U_f * D2_fourier[:, None]
    U_w_f = U_f_weighted.T.reshape(r, *volume_shape)
    U_w_real = np.asarray(ftu.get_idft3(jnp.asarray(U_w_f))).real.reshape(r, V_SIZE).T.astype(np.float32)
    return U_w_real


# Alias — semantically clearer when the diagonal is not D² specifically.
apply_diag_fourier = apply_D2_to_basis


class DWrappedOperator:
    """Wrap an op computing G(X)=A*A X - A*b (real vol-space, Q right-multipled)
    so it instead computes G_Y(Y) = D^{-1}(A*A D^{-1} Y - A*b), i.e. the gradient
    in the change-of-variables Y = D X. D is Fourier-diagonal (per-voxel weight).

    Presents the same .right_matvec / .left_matvec interface as SketchedNormalOperator.
    """

    def __init__(self, op, D_inv_fourier, volume_shape, V_SIZE):
        self.op = op
        self.D_inv = np.asarray(D_inv_fourier, dtype=np.float32)
        self.vs = volume_shape
        self.V = V_SIZE

    def _apply_Dinv(self, U_real):
        if U_real.shape[1] == 0:
            return U_real
        return apply_diag_fourier(U_real, self.D_inv, self.vs, self.V)

    def right_matvec(self, U, s, V, Q):
        # U is Y-space real basis; convert to X via D^{-1}, run raw op, then D^{-1} on output voxel axis
        U_X = self._apply_Dinv(U) if U.shape[1] else U
        out_X = np.asarray(self.op.right_matvec(U_X, s, V, Q), dtype=np.float32)
        return self._apply_Dinv(out_X)

    def left_matvec(self, U, s, V, S):
        U_X = self._apply_Dinv(U) if U.shape[1] else U
        # S has shape (sketch_rank, vol_size); rows are volumes → apply D^{-1} per row via transpose trick
        S_X = self._apply_Dinv(S.T).T
        return np.asarray(self.op.left_matvec(U_X, s, V, S_X), dtype=np.float32)


# -------------------- operator wrapping --------------------


@dataclass
class SketchedIterationCfg:
    block_size: int = 15
    max_rank: int = 30
    n_power: int = 1
    target_rank: int = 10
    delta: float = 1.0
    method: str = "hard"  # "hard" | "soft"
    lam: float = 0.0  # only used when method == "soft"
    prior_mode: str = "none"  # "none" | "radial_D"
    record_per_k: tuple = (1, 2, 5, 10)


def _orth(X):
    if X.size == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    Q, _ = np.linalg.qr(X, mode="reduced")
    return Q.astype(np.float32)


class SketchedSolver:
    """Wraps SketchedNormalOperator + optional radial D prior in real volume space."""

    def __init__(self, op, volume_shape, V_SIZE, n, left_scale, D2_fourier=None):
        self.op = op
        self.vs = volume_shape
        self.V = V_SIZE
        self.n = n
        self.left_scale = left_scale
        self.D2 = D2_fourier  # (V,) real, or None

    def estimate_lipschitz(self, include_prior=False, n_power=30, seed=0):
        """Estimate σ_max(A*A [+ D²]) via power iteration on a rank-1 probe.

        A*A·v is obtained as G(X=v·u^T)·u − G(X=0)·u where u ∈ R^n is fixed
        unit. The probe v is **initialized from A*b direction** (i.e., from
        G(0)·u) so it's seeded where the data gradient actually lives,
        rather than a random Gaussian — in cryo-EM this makes a 15× difference
        because A*A has its dominant eigenvalues in the low-frequency shells
        that A*b populates.
        """
        rng = np.random.default_rng(seed)
        u = rng.normal(size=(self.n,)).astype(np.float32)
        u = u / np.linalg.norm(u)
        U0 = np.zeros((self.V, 0), np.float32)
        s0 = np.zeros((0,), np.float32)
        V0 = np.zeros((self.n, 0), np.float32)
        G0_u = self._right(U0, s0, V0, u[:, None]).ravel()  # -A*b · u
        # Seed probe from A*b direction: v = -G(0)·u / ||...||
        v = -G0_u
        vnorm = float(np.linalg.norm(v))
        if vnorm < 1e-30:
            v = rng.normal(size=(self.V,)).astype(np.float32)
            vnorm = float(np.linalg.norm(v))
        v = v / max(vnorm, 1e-30)
        lam = 0.0
        for _ in range(n_power):
            G_u = self._right(v[:, None], np.array([1.0], np.float32), u[:, None], u[:, None]).ravel()
            AtAv = G_u - G0_u
            if include_prior and self.D2 is not None:
                D2v = apply_D2_to_basis(v[:, None], self.D2, self.vs, self.V).ravel()
                AtAv = AtAv + D2v
            lam = float(np.linalg.norm(AtAv))
            v = AtAv / max(lam, 1e-30)
        return lam

    def grad_minus_Atb_V(self, V_img):
        """Return ∇f(0) · V_img = -(A^T b) · V_img.  One operator call.  Cached per outer iter."""
        U0 = np.zeros((self.V, 0), np.float32)
        s0 = np.zeros((0,), np.float32)
        V0 = np.zeros((self.n, 0), np.float32)
        return self._right(U0, s0, V0, V_img)

    def evaluate_f_smooth(self, U, s, V_img, use_prior, GV=None, G0V=None, return_caches=False):
        """Evaluate f(X) = (1/2)||AX-b||^2 [+ (1/2)||DX||^2]  UP TO a constant in X.

        Quadratic identity:  f(X) - const = (1/2) <X, ∇f(X) + ∇f(0)>  where ∇f(0) = -A^T b.

        Note V_img columns are assumed orthonormal (V_img^T V_img = I_rank); true after
        sketched SVT since we return V_new from the SVD of B.
        """
        if len(s) == 0:
            val = 0.0
            if return_caches:
                return val, None, None
            return val
        if GV is None:
            GV = self._right(U, s, V_img, V_img)  # (V, rank) = (A^T A X - A^T b) V
        GV_total = GV
        if use_prior and self.D2 is not None:
            # ∇_prior(X) · V = D² X · V = D² U diag(s)  (since V^T V = I_rank)
            GV_total = GV + apply_D2_to_basis(U, self.D2, self.vs, self.V) * s
        if G0V is None:
            G0V = self.grad_minus_Atb_V(V_img)  # (V, rank)
        # <X, ∇f(X)> = Σ_i s_i <U_i, (GV_total)_i>
        inner_x = float(np.sum(s.astype(np.float64) * (U.astype(np.float64) * GV_total.astype(np.float64)).sum(axis=0)))
        # <X, ∇f(0)> = Σ_i s_i <U_i, (G0V)_i>
        inner_0 = float(np.sum(s.astype(np.float64) * (U.astype(np.float64) * G0V.astype(np.float64)).sum(axis=0)))
        val = 0.5 * (inner_x + inner_0)
        if return_caches:
            return val, GV, G0V
        return val

    def grad_inner_with(self, U_old, s_old, V_old, U_new, s_new, V_new, use_prior):
        """Return <∇f(X_old), X_new>  for the Armijo linearization term.

        One right_matvec call (∇f(X_old) · V_new), plus prior correction.
        """
        # ∇f_data(X_old) · V_new = A^T A X_old V_new - A^T b V_new
        G = self._right(U_old, s_old, V_old, V_new)  # (V, rank_new)
        if use_prior and self.D2 is not None and len(s_old):
            # D² X_old V_new = D² U_old diag(s_old) (V_old^T V_new)
            VtV = V_old.T @ V_new  # (rank_old, rank_new)
            D2U_old = apply_D2_to_basis(U_old, self.D2, self.vs, self.V)  # (V, rank_old)
            G = G + (D2U_old * s_old) @ VtV
        # <X_new, G> = Σ_i s_new_i <U_new_i, G_i>
        return float(np.sum(s_new.astype(np.float64) * (U_new.astype(np.float64) * G.astype(np.float64)).sum(axis=0)))

    @staticmethod
    def frob_sq_diff(U_a, s_a, V_a, U_b, s_b, V_b):
        """||X_a - X_b||²_F with X = U diag(s) V^T and orthonormal U,V columns."""
        norm_a = float(np.sum(s_a.astype(np.float64) ** 2)) if len(s_a) else 0.0
        norm_b = float(np.sum(s_b.astype(np.float64) ** 2)) if len(s_b) else 0.0
        if len(s_a) == 0 or len(s_b) == 0:
            return norm_a + norm_b
        UtU = U_a.T @ U_b  # (r_a, r_b)
        VtV = V_a.T @ V_b  # (r_a, r_b)
        # <X_a, X_b>_F = trace((s_a U_a^T U_b s_b) (V_b^T V_a)) = sum_{ij} s_a_i s_b_j (UtU)_ij (VtV)_ij
        cross = float(
            np.sum(
                s_a[:, None].astype(np.float64)
                * s_b[None, :].astype(np.float64)
                * UtU.astype(np.float64)
                * VtV.astype(np.float64)
            )
        )
        return norm_a + norm_b - 2.0 * cross

    def step_with_backtracking(
        self,
        U,
        s,
        V,
        cfg: SketchedIterationCfg,
        rng,
        delta_state,
        armijo_c=0.9,
        shrink=0.5,
        grow=1.5,
        max_retries=12,
        f_old=None,
        GV_old_cache=None,
        G0V_old_cache=None,
    ):
        """Proximal-gradient step with Armijo backtracking on the smooth part of f.

        Returns (U_new, s_new, V_new, sb, delta_used, new_state, f_new, caches).
        state := {"delta": float}
        caches: (GV_new, G0V_new) to reuse as (GV_old, G0V_old) on the next outer iter.

        Accept if:  f(X_new) ≤ f(X_old) + <∇f(X_old), X_new - X_old> + (1/(2δ·c)) ||X_new - X_old||²_F
        (where c=armijo_c < 1 gives a strict sufficient-decrease condition, reflecting
         the Lipschitz-bound proximal descent lemma with safety factor).
        """
        use_prior = cfg.prior_mode == "radial_D"
        delta = float(delta_state["delta"])

        # f(X_old), once. Need GV_old and G0V_old caches.
        if f_old is None:
            f_old, GV_old_cache, G0V_old_cache = self.evaluate_f_smooth(
                U, s, V, use_prior, GV=GV_old_cache, G0V=G0V_old_cache, return_caches=True
            )

        last_trial = None
        for trial in range(max_retries):
            cfg_trial = SketchedIterationCfg(
                block_size=cfg.block_size,
                max_rank=cfg.max_rank,
                n_power=cfg.n_power,
                target_rank=cfg.target_rank,
                delta=delta,
                method=cfg.method,
                lam=cfg.lam,
                prior_mode=cfg.prior_mode,
                record_per_k=cfg.record_per_k,
            )
            # Use a fresh child rng so each retry has independent sketches (avoids
            # deterministic failure at a bad δ).
            rng_trial = np.random.default_rng(rng.integers(0, 2**31 - 1))
            U_new, s_new, V_new, sb = self.step(U, s, V, cfg_trial, rng_trial)
            if len(s_new) == 0:
                # Empty iterate: treat as "accept with huge shrink" so we don't loop
                delta_state["delta"] = max(delta * shrink, 1e-8)
                return (
                    U_new,
                    s_new,
                    V_new,
                    sb,
                    delta,
                    delta_state,
                    f_old,
                    (None, None),
                    {"trials": trial + 1, "accepted": False},
                )

            f_new, GV_new, G0V_new = self.evaluate_f_smooth(U_new, s_new, V_new, use_prior, return_caches=True)
            inner = self.grad_inner_with(U, s, V, U_new, s_new, V_new, use_prior)
            # <∇f(X_old), X_old>  from cached GV_old_cache (plus prior term if enabled)
            if len(s):
                GV_old_total = GV_old_cache
                if use_prior and self.D2 is not None:
                    GV_old_total = GV_old_cache + apply_D2_to_basis(U, self.D2, self.vs, self.V) * s
                grad_dot_xold = float(
                    np.sum(s.astype(np.float64) * (U.astype(np.float64) * GV_old_total.astype(np.float64)).sum(axis=0))
                )
            else:
                grad_dot_xold = 0.0
            lin_term = inner - grad_dot_xold  # = <∇f(X_old), X_new - X_old>
            diff_sq = self.frob_sq_diff(U_new, s_new, V_new, U, s, V)
            quad_term = diff_sq / (2.0 * delta * armijo_c)  # safety factor shrinks accepted δ
            rhs = f_old + lin_term + quad_term
            ok = f_new <= rhs + 1e-12 * max(abs(rhs), abs(f_new), 1.0)
            last_trial = {
                "delta": delta,
                "f_old": f_old,
                "f_new": f_new,
                "lin": lin_term,
                "quad": quad_term,
                "diff_sq": diff_sq,
                "rhs": rhs,
                "ok": bool(ok),
            }
            if ok:
                # Optionally grow δ for next outer iter
                # Grow only if the quadratic safety is loose (f_new well below rhs).
                slack = rhs - f_new
                if slack > 0.5 * quad_term and quad_term > 0:
                    new_delta = delta * grow
                else:
                    new_delta = delta
                delta_state["delta"] = new_delta
                info = {"trials": trial + 1, "accepted": True, "last": last_trial}
                return U_new, s_new, V_new, sb, delta, delta_state, f_new, (GV_new, G0V_new), info
            # Shrink δ and retry
            delta = delta * shrink
            if delta < 1e-10:
                break
        # Bailed out: take last step anyway, note failure
        delta_state["delta"] = max(delta, 1e-10)
        info = {"trials": max_retries, "accepted": False, "last": last_trial}
        return U_new, s_new, V_new, sb, delta, delta_state, f_new, (GV_new, G0V_new), info

    def _right(self, U, s, V, Q):
        return np.asarray(self.op.right_matvec(U, s, V, Q), dtype=np.float32)

    def _left(self, U, s, V, S):
        return np.asarray(self.op.left_matvec(U, s, V, S), dtype=np.float32) / self.left_scale

    def _prior_right(self, U, s, V, Q):
        """D^2 X @ Q = (D^2 U) diag(s) V^T Q — real-space column-wise radial weighting."""
        if self.D2 is None or len(s) == 0:
            return np.zeros((self.V, Q.shape[1]), dtype=np.float32)
        D2U = apply_D2_to_basis(U, self.D2, self.vs, self.V)
        return (D2U * s) @ (V.T @ Q)

    def _prior_left(self, U, s, V, S):
        """S @ (D^2 X) = S @ (D^2 U diag(s) V^T)."""
        if self.D2 is None or len(s) == 0:
            return np.zeros((S.shape[0], self.n), dtype=np.float32)
        D2U = apply_D2_to_basis(U, self.D2, self.vs, self.V)
        return (S @ (D2U * s)) @ V.T

    def apply_Zr(self, U, s, V, delta, use_prior, Q):
        """Right sketch of Z = X - delta*(G(X) [+ D^2 X])."""
        XQ = (U * s) @ (V.T @ Q) if len(s) else np.zeros((self.V, Q.shape[1]), np.float32)
        out = XQ - delta * self._right(U, s, V, Q)
        if use_prior and self.D2 is not None and len(s):
            out -= delta * self._prior_right(U, s, V, Q)
        return out

    def apply_Zl(self, U, s, V, delta, use_prior, S):
        """Left sketch of Z = X - delta*(G(X) [+ D^2 X])."""
        SX = (S @ (U * s)) @ V.T if len(s) else np.zeros((S.shape[0], self.n), np.float32)
        out = SX - delta * self._left(U, s, V, S)
        if use_prior and self.D2 is not None and len(s):
            out -= delta * self._prior_left(U, s, V, S)
        return out

    def step(self, U, s, V, cfg: SketchedIterationCfg, rng):
        delta = cfg.delta
        use_prior = cfg.prior_mode == "radial_D"
        basis = U.copy()
        B = self.apply_Zl(U, s, V, delta, use_prior, basis.T) if basis.size else np.zeros((0, self.n), np.float32)
        while basis.shape[1] < cfg.max_rank:
            blk = min(cfg.block_size, cfg.max_rank - basis.shape[1])
            Om = rng.normal(size=(self.n, blk)).astype(np.float32)
            Y = self.apply_Zr(U, s, V, delta, use_prior, Om)
            if basis.size:
                Y = Y - basis @ (basis.T @ Y)
            for _ in range(cfg.n_power):
                T = _orth(Y)
                W_ = self.apply_Zl(U, s, V, delta, use_prior, T.T).T
                Y = self.apply_Zr(U, s, V, delta, use_prior, W_)
                if basis.size:
                    Y = Y - basis @ (basis.T @ Y)
            Q0 = _orth(Y)
            if basis.size and Q0.size:
                Q0 = _orth(Q0 - basis @ (basis.T @ Q0))
            if Q0.size == 0:
                break
            B0 = self.apply_Zl(U, s, V, delta, use_prior, Q0.T)
            basis = np.concatenate([basis, Q0], axis=1)
            B = np.vstack([B, B0])
        if basis.shape[1] == 0:
            return (
                np.zeros((self.V, 0), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((self.n, 0), np.float32),
                np.zeros((0,), np.float32),
            )
        try:
            Ub, sb, Vhb = np.linalg.svd(B, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fall back to a regularized SVD via eigen-decomp of B B^T
            BBt = B @ B.T
            w, Ub2 = np.linalg.eigh(BBt.astype(np.float64))
            w = np.clip(w, 0.0, None)
            sb = np.sqrt(w[::-1]).astype(np.float32)
            Ub = Ub2[:, ::-1].astype(np.float32)
            inv = np.zeros_like(sb)
            inv[sb > 0] = 1.0 / sb[sb > 0]
            Vhb = (Ub.T @ B) * inv[:, None]
        if cfg.method == "hard":
            keep_idx = np.arange(min(cfg.target_rank, len(sb)))
            if len(keep_idx) == 0:
                return (
                    np.zeros((self.V, 0), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((self.n, 0), np.float32),
                    sb,
                )
            U_new = (basis @ Ub[:, keep_idx]).astype(np.float32)
            s_new = sb[keep_idx].astype(np.float32)
            V_new = Vhb[keep_idx, :].T.astype(np.float32)
        else:
            keep = sb > cfg.lam * delta
            if not np.any(keep):
                return (
                    np.zeros((self.V, 0), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((self.n, 0), np.float32),
                    sb,
                )
            U_new = (basis @ Ub[:, keep]).astype(np.float32)
            s_new = (sb[keep] - cfg.lam * delta).astype(np.float32)
            V_new = Vhb[keep, :].T.astype(np.float32)
        return U_new, s_new, V_new, sb


# -------------------- metrics --------------------


def relvar_k_real(U_real, k, volume_shape, V_SIZE, U_gt_f, s_gt):
    """Legacy metric: cumulative over first-k learned PCs, linear s_gt weighting,
    denominator sum(s_gt).  Kept for back-compat with older JSONs.

    Upper bound at rank k equals (sum s_gt[:k]) / (sum s_gt), i.e. linear cumfrac.
    """
    if U_real.shape[1] == 0:
        return 0.0
    r = min(k, U_real.shape[1])
    Uf = np.asarray(linalg.batch_dft3(jnp.asarray(U_real[:, :r], dtype=jnp.float32), volume_shape, r)) / np.sqrt(V_SIZE)
    rv = metrics.captured_variance(Uf, U_gt_f, s_gt)
    return float(np.asarray(metrics.relative_variance_from_captured_variance(rv, s_gt))[-1])


def relvar_k_s2_full(U_real, k, volume_shape, V_SIZE, U_gt_f, s_gt):
    """Variance metric: fraction of TOTAL GT variance (sum s_gt²) captured by
    the rank-k learned subspace against the FULL GT basis.  Comparable across
    methods that use different basis sizes.

    Definition:  sum_j ||P_learned U_gt_j||² · s_gt_j² / sum_m s_gt_m²,
    where P_learned is projection onto span(U_real[:, :k]).
    """
    if U_real.shape[1] == 0:
        return 0.0
    r = min(k, U_real.shape[1])
    Uf = np.asarray(linalg.batch_dft3(jnp.asarray(U_real[:, :r], dtype=jnp.float32), volume_shape, r)) / np.sqrt(V_SIZE)
    # <Uf_i, U_gt_j> → coefficient squared weighted by s_gt_j² summed over i and j
    M = np.conj(Uf.T) @ np.asarray(U_gt_f)  # (r, K)
    coeff_sq = (np.abs(M) ** 2).sum(axis=0)  # (K,) projection energy on each GT dir
    s2 = np.asarray(s_gt).astype(np.float64) ** 2
    num = float((coeff_sq.astype(np.float64) * s2).sum())
    den = float(s2.sum())
    return num / max(den, 1e-30)


def build_gt_factors(gt_results, sim_info, volume_shape, V_SIZE):
    U_gt_f, s_gt_full, Vh_gt = gt_results.get_vol_svd()
    probs = np.asarray(gt_results.get_probs_of_state())
    assign = np.asarray(sim_info["image_assignment"])
    U_gt_f_arr = np.asarray(U_gt_f).astype(np.complex64)
    K = U_gt_f_arr.shape[1]
    U_gt_real = (
        np.asarray(ftu.get_idft3(U_gt_f_arr.T.reshape(K, *volume_shape))).real.reshape(K, V_SIZE).T.astype(np.float32)
    )
    s_gt = np.asarray(s_gt_full).astype(np.float32)
    V_gt_img = (np.asarray(Vh_gt) / np.sqrt(probs)[None, :]).T[assign].real.astype(np.float32)
    return dict(U_gt_f=U_gt_f_arr, s_gt=s_gt, U_gt_real=U_gt_real, V_gt_img=V_gt_img, probs=probs, assign=assign)


def run_iterations(
    solver: SketchedSolver,
    cfg: SketchedIterationCfg,
    U0,
    s0,
    V0,
    n_iter,
    gt,
    volume_shape,
    V_SIZE,
    seed=1,
    log_every=1,
    logfn=print,
):
    U, s, V = U0.copy(), s0.copy(), V0.copy()
    rng = np.random.default_rng(seed)
    s_sq_total = float(np.sum(gt["s_gt"].astype(np.float64) ** 2))
    history = []
    t0 = time.time()
    for it in range(n_iter):
        U, s, V, sb = solver.step(U, s, V, cfg, rng)
        row = {"it": it + 1, "rank": int(len(s)), "top_sv": float(sb[0]) if len(sb) else 0.0, "t": time.time() - t0}
        for k in cfg.record_per_k:
            row[f"rv@{k}"] = relvar_k_real(U, k, volume_shape, V_SIZE, gt["U_gt_f"], gt["s_gt"])
            row[f"rv_s2@{k}"] = relvar_k_s2_full(U, k, volume_shape, V_SIZE, gt["U_gt_f"], gt["s_gt"])
        history.append(row)
        if it < 5 or (it + 1) % log_every == 0 or it == n_iter - 1:
            msg = (
                f"  it={row['it']:3d} rank={row['rank']:2d} top={row['top_sv']:.3e} "
                + " ".join(f"{k}={row[k]:.4f}" for k in row if k.startswith("rv@"))
                + f" t={row['t']:.1f}s"
            )
            logfn(msg)
    final = history[-1] if history else {}
    return U, s, V, history, final


# End of run_iterations (legacy fixed-δ runner).


def run_iterations_backtracking(
    solver: SketchedSolver,
    cfg: SketchedIterationCfg,
    U0,
    s0,
    V0,
    n_iter,
    gt,
    volume_shape,
    V_SIZE,
    seed=1,
    log_every=1,
    logfn=print,
    delta_init=0.1,
    armijo_c=0.9,
    shrink=0.5,
    grow=1.5,
    max_retries=10,
    delta_min=1e-8,
    delta_max=1e3,
):
    """Outer loop using solver.step_with_backtracking.

    Per outer iter:
      - At most `max_retries` sketched-step evaluations (each also does 2-3 extra matvecs
        for f & gradient).
      - Armijo accept condition uses the standard proximal-gradient descent lemma
        (safety factor = armijo_c).
      - δ grows by `grow` when there's slack, shrinks by `shrink` on reject.

    Records δ_used and number of trials per iter in history.
    """
    U, s, V = U0.copy(), s0.copy(), V0.copy()
    rng = np.random.default_rng(seed)
    history = []
    delta_state = {"delta": float(delta_init)}
    f_cache = None
    GV_cache = None
    G0V_cache = None
    t0 = time.time()
    for it in range(n_iter):
        # Clamp δ to safe range
        delta_state["delta"] = float(np.clip(delta_state["delta"], delta_min, delta_max))
        U_new, s_new, V_new, sb, delta_used, delta_state, f_new, caches, info = solver.step_with_backtracking(
            U,
            s,
            V,
            cfg,
            rng,
            delta_state,
            armijo_c=armijo_c,
            shrink=shrink,
            grow=grow,
            max_retries=max_retries,
            f_old=f_cache,
            GV_old_cache=GV_cache,
            G0V_old_cache=G0V_cache,
        )
        U, s, V = U_new, s_new, V_new
        f_cache = f_new
        GV_cache, G0V_cache = caches  # reuse at X_k+1 on next iter
        row = {
            "it": it + 1,
            "rank": int(len(s)),
            "top_sv": float(sb[0]) if len(sb) else 0.0,
            "t": time.time() - t0,
            "delta_used": float(delta_used),
            "delta_next": float(delta_state["delta"]),
            "trials": int(info.get("trials", 1)),
            "accepted": bool(info.get("accepted", False)),
            "f_smooth": float(f_new) if f_new is not None else None,
        }
        for k in cfg.record_per_k:
            row[f"rv@{k}"] = relvar_k_real(U, k, volume_shape, V_SIZE, gt["U_gt_f"], gt["s_gt"])
            row[f"rv_s2@{k}"] = relvar_k_s2_full(U, k, volume_shape, V_SIZE, gt["U_gt_f"], gt["s_gt"])
        history.append(row)
        if it < 5 or (it + 1) % log_every == 0 or it == n_iter - 1:
            msg = (
                f"  it={row['it']:3d} rank={row['rank']:2d} top={row['top_sv']:.3e} "
                f"δ_used={row['delta_used']:.2e} δ_next={row['delta_next']:.2e} "
                f"trials={row['trials']} f={row['f_smooth']:.4e} "
                + " ".join(f"{k}={row[k]:.4f}" for k in row if k.startswith("rv@"))
                + f" t={row['t']:.1f}s"
            )
            logfn(msg)
    final = history[-1] if history else {}
    return U, s, V, history, final


def _u_y_to_u_x(U_Y, D_inv_fourier, volume_shape, V_SIZE):
    """Convert Y-space basis to X-space via D^{-1}, then re-orthonormalize."""
    if U_Y.shape[1] == 0:
        return U_Y
    U_X_raw = apply_diag_fourier(U_Y, D_inv_fourier, volume_shape, V_SIZE)
    Q, _ = np.linalg.qr(U_X_raw, mode="reduced")
    return Q.astype(np.float32)


def run_iterations_backtracking_dmetric(
    raw_op,
    D2_fourier,
    volume_shape,
    V_SIZE,
    n,
    left_scale,
    cfg: SketchedIterationCfg,
    U0_X,
    s0,
    V0,
    n_iter,
    gt,
    *,
    seed=1,
    log_every=1,
    logfn=print,
    delta_init=0.1,
    armijo_c=0.9,
    shrink=0.5,
    grow=1.5,
    max_retries=10,
    delta_min=1e-8,
    delta_max=1e3,
):
    """D-metric nuclear-norm variant.

    Substitutes Y = D X with D Fourier-diagonal (D² = D2_fourier), giving
        min (1/2) ||A D^{-1} Y − b||² + (1/2) ||Y||² + λ ||Y||_*.
    Standard SVT prox on Y; the L2 term is the natural metric.

    Logs rv@k + rv_s2@k in X-space (converting U_Y → U_X via D^{-1} per iter).
    """
    D2 = np.asarray(D2_fourier, dtype=np.float32)
    D_fourier = np.sqrt(np.clip(D2, 0.0, None)).astype(np.float32)
    D_floor = float(D_fourier.max()) * 1e-8
    D_safe = np.maximum(D_fourier, D_floor)
    D_inv_fourier = (1.0 / D_safe).astype(np.float32)

    wrapped = DWrappedOperator(raw_op, D_inv_fourier, volume_shape, V_SIZE)
    D2_Y = np.ones(V_SIZE, dtype=np.float32)  # identity L2 in Y-space
    solver = SketchedSolver(wrapped, volume_shape, V_SIZE, n, left_scale, D2_fourier=D2_Y)

    cfg_y = SketchedIterationCfg(
        block_size=cfg.block_size,
        max_rank=cfg.max_rank,
        n_power=cfg.n_power,
        target_rank=cfg.target_rank,
        delta=cfg.delta,
        method=cfg.method,
        lam=cfg.lam,
        prior_mode="radial_D",  # turns on the L2 ||Y||² term via D2_Y=ones
        record_per_k=cfg.record_per_k,
    )

    if U0_X.shape[1] > 0:
        U0_Y_raw = apply_diag_fourier(U0_X, D_fourier, volume_shape, V_SIZE)
        Q, _ = np.linalg.qr(U0_Y_raw, mode="reduced")
        U_Y = Q.astype(np.float32)
    else:
        U_Y = U0_X.copy()
    s = s0.copy()
    V = V0.copy()

    rng = np.random.default_rng(seed)
    history = []
    delta_state = {"delta": float(delta_init)}
    f_cache = None
    GV_cache = None
    G0V_cache = None
    t0 = time.time()
    for it in range(n_iter):
        delta_state["delta"] = float(np.clip(delta_state["delta"], delta_min, delta_max))
        U_new, s_new, V_new, sb, delta_used, delta_state, f_new, caches, info = solver.step_with_backtracking(
            U_Y, s, V, cfg_y, rng, delta_state,
            armijo_c=armijo_c, shrink=shrink, grow=grow, max_retries=max_retries,
            f_old=f_cache, GV_old_cache=GV_cache, G0V_old_cache=G0V_cache,
        )
        U_Y, s, V = U_new, s_new, V_new
        f_cache = f_new
        GV_cache, G0V_cache = caches

        U_X_orth = _u_y_to_u_x(U_Y, D_inv_fourier, volume_shape, V_SIZE)
        row = {
            "it": it + 1,
            "rank": int(len(s)),
            "top_sv": float(sb[0]) if len(sb) else 0.0,
            "t": time.time() - t0,
            "delta_used": float(delta_used),
            "delta_next": float(delta_state["delta"]),
            "trials": int(info.get("trials", 1)),
            "accepted": bool(info.get("accepted", False)),
            "f_smooth": float(f_new) if f_new is not None else None,
        }
        for k in cfg_y.record_per_k:
            row[f"rv@{k}"] = relvar_k_real(U_X_orth, k, volume_shape, V_SIZE, gt["U_gt_f"], gt["s_gt"])
            row[f"rv_s2@{k}"] = relvar_k_s2_full(U_X_orth, k, volume_shape, V_SIZE, gt["U_gt_f"], gt["s_gt"])
        history.append(row)
        if it < 5 or (it + 1) % log_every == 0 or it == n_iter - 1:
            msg = (
                f"  [D] it={row['it']:3d} rank={row['rank']:2d} top={row['top_sv']:.3e} "
                f"δ_used={row['delta_used']:.2e} δ_next={row['delta_next']:.2e} "
                f"trials={row['trials']} f={row['f_smooth']:.4e} "
                + " ".join(f"{k}={row[k]:.4f}" for k in row if k.startswith("rv@") and not k.startswith("rv_"))
                + f" t={row['t']:.1f}s"
            )
            logfn(msg)

    if len(s) > 0:
        U_X_raw = apply_diag_fourier(U_Y, D_inv_fourier, volume_shape, V_SIZE)
        X_scaled = U_X_raw * s
        Qx, Rx = np.linalg.qr(X_scaled, mode="reduced")
        M = Rx @ V.T
        Um, sm, Vhm = np.linalg.svd(M, full_matrices=False)
        U_X_final = (Qx @ Um).astype(np.float32)
        s_X_final = sm.astype(np.float32)
        V_X_final = Vhm.T.astype(np.float32)
    else:
        U_X_final = np.zeros((V_SIZE, 0), np.float32)
        s_X_final = np.zeros((0,), np.float32)
        V_X_final = np.zeros((n, 0), np.float32)

    final = history[-1] if history else {}
    return U_X_final, s_X_final, V_X_final, history, final
