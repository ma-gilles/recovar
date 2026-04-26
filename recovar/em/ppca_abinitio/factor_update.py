"""Factor update for the PPCA-ab-initio v0 loop (Stage 1C).

Per spec Section 8.3:

- `s` is **strictly fixed** (Q2). The update changes only `U`.
- The objective is the expected complete-data NLL with respect to
  `U`, with the latent posterior `(γ, m, Hinv)` from the current
  E-step. We take K (~3) gradient steps via `jax.value_and_grad`
  on a closure that consumes `(U_half, posterior_block_data)` and
  returns a scalar.
- After the gradient steps, we apply the real-O(q) gauge-fix chain
  from `recovar.em.ppca_abinitio.half_volume`:

      U_band = radial_band_limit_half(U_raw, volume_shape, k_max)
      U_new  = real_volume_orthonormalize_half(U_band, weights, N_full)

  There is no Hermitian-projection step — the half-volume rfft
  layout makes Hermitian symmetry structural.

The expected complete-data NLL has the form

    L(U) = sum_{i, g} γ_{i,g} *
            { ||y_i - CTF_i A_g μ - CTF_i A_g U m_{i,g}||²/σ²
              + tr( (CTF_i A_g U) Σ_{α | i,g} (CTF_i A_g U)^H )/σ² }
         + λ ||U||²

where `Σ_{α|i,g} = H_{i,g}^{-1}`. The trace term is the
contribution of the posterior covariance and is what makes this
an EM update rather than a hard-assignment update. For v0 we
include both the `m m^T` term (via the post_mean) and the `Hinv`
term.

For v0 simplicity the loss is computed in **half-image** layout
with rfft Hermitian weights and full vmap over (image, rotation,
translation). This is fine at toy size; later phases may want a
streaming variant via `iter_posterior_blocks`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu

from .half_volume import (
    make_half_volume_weights,
    project_to_real_volume_subspace_batch,
    radial_band_limit_half,
    real_volume_orthonormalize_half,
)
from .posterior import (
    _preprocess_batch_to_half,
    _slice_mu_half,
    _slice_U_half,
    make_half_image_weights,
    score_from_half_image_projections,
)
from .types import PPCAInit


def compute_W_prior_half(U_half, s, volume_shape, eps_rel: float = 1e-3):
    """Per-voxel, per-PC Wiener-style regularizer in half-volume layout.

    The closed-form M-step solves `(M_v + R_v) U[:, v] = B_v` per voxel.
    `R_v` is the regularization. The default uses a scalar ridge
    `λ I_q`. This function builds an alternative regularizer:

        R_v = diag(1 / W_v)  where  W_v = shell_avg(|U|² · s)

    The motivation is that cryo-EM signal is shell-stratified — DC and
    low-frequency shells carry orders of magnitude more signal than
    high-frequency shells. A scalar ridge cannot express this; the
    shell-averaged Wiener prior does.

    This is REGULARIZATION ONLY, not eigenvalue estimation. The
    eigenvalues `s` themselves are still treated as a fixed
    hyperparameter (frozen at `s = 1` flat by default). See
    `docs/math/ppca_abinitio_clean_algorithm.md` and the Phase 1
    ablation experiment in `docs/math/ppca_abinitio_status_*.md`.

    Parameters
    ----------
    U_half : (q, half_vol_size) complex
        Current factor estimate.
    s : (q,) real
        Eigenvalues — used only as a per-PC scale; with `s = 1` flat
        the prior reduces to a per-PC shell average of |U|².
    volume_shape : 3-tuple
    eps_rel : float
        Floor for `W_v` as a fraction of the per-PC mean — prevents
        the regularizer from blowing up at empty / outer shells where
        the running estimate of |U|² is near zero.

    Returns
    -------
    W_prior : (q, half_vol_size) real
        Floored shell-averaged signal-variance prior.
    """
    W_sq = jnp.abs(U_half) ** 2 * s[:, None]

    radial = ftu.get_grid_of_radial_distances_real(volume_shape).reshape(-1)
    n_shells = max(1, volume_shape[0] // 2 - 1)
    labels = jnp.minimum(radial, n_shells - 1)

    def _shell_avg_one(w_sq_pc):
        sums = jnp.bincount(labels, weights=w_sq_pc, length=n_shells)
        counts = jnp.bincount(labels, length=n_shells)
        safe_counts = jnp.where(counts > 0, counts, 1)
        per_shell = jnp.where(counts > 0, sums / safe_counts, 0.0)
        return per_shell[labels]

    W_prior = jax.vmap(_shell_avg_one)(W_sq)
    per_pc_mean = jnp.mean(W_prior, axis=1, keepdims=True)
    floor = jnp.maximum(eps_rel * per_pc_mean, 1e-30)
    return jnp.maximum(W_prior, floor)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def _expected_nll_half(
    U_half,
    mu_half,
    s,
    rotations,
    image_shape,
    volume_shape,
    shifted_half,
    ctf2_over_nv_half,
    weights_half,
    log_resp,
    post_mean,
    post_Hinv,
    *,
    ridge_lambda: float = 0.0,
):
    """Expected complete-data NLL of the PPCA model w.r.t. `U` only.

    Parameters
    ----------
    U_half : (q, half_volume_size) complex128
    mu_half : (half_volume_size,) complex128
    s : (q,) float64
    shifted_half : (n_img, n_trans, half_image_size) complex128
        S_t (CTF * y / σ²) — preprocessed batch state.
    ctf2_over_nv_half : (n_img, half_image_size) float64
    weights_half : (half_image_size,) float64 — rfft Hermitian weights
    log_resp : (n_img, n_rot, n_trans) float64
    post_mean : (n_img, n_rot, n_trans, q) float64
    post_Hinv : (n_img, n_rot, q, q) float64

    Returns
    -------
    loss : scalar float64
    """
    del s  # In the frozen-posterior M-step, the latent prior term is constant in U.

    mean_proj_half = _slice_mu_half(mu_half, rotations, image_shape, volume_shape).astype(jnp.complex128)
    u_proj_half = _slice_U_half(U_half, rotations, image_shape, volume_shape).astype(jnp.complex128)

    gamma = jnp.exp(jnp.asarray(log_resp))  # (n_img, n_rot, n_trans)
    mean_proj = mean_proj_half[None, :, None, :]  # (1, n_rot, 1, n_half)

    # Candidate mean projection under the frozen posterior means.
    pred_het = jnp.einsum("irtk,rkp->irtp", post_mean, u_proj_half)
    pred = mean_proj + pred_het  # (n_img, n_rot, n_trans, n_half)

    shifted_conj_w = jnp.conj(shifted_half) * weights_half[None, None, :]
    cross = -2.0 * jnp.einsum("itp,irtp->irt", shifted_conj_w, pred).real

    ctf2_w = ctf2_over_nv_half * weights_half[None, :]
    pred_abs2 = jnp.abs(pred) ** 2
    norm = jnp.einsum("ip,irtp->irt", ctf2_w, pred_abs2)

    # Trace term from the frozen posterior covariance.
    u_Hinv_u = jnp.einsum("rkp,irkl,rlp->irp", jnp.conj(u_proj_half), post_Hinv, u_proj_half).real
    cov = jnp.einsum("ip,irp->ir", ctf2_w, u_Hinv_u)[:, :, None]

    nll = 0.5 * jnp.sum(gamma * (cross + norm + cov))

    # Ridge prior on U (per Section 8.3.4)
    if ridge_lambda > 0.0:
        nll = nll + ridge_lambda * jnp.sum(jnp.abs(U_half) ** 2).real

    return nll


# ---------------------------------------------------------------------------
# Gradient step + projection chain
# ---------------------------------------------------------------------------


def _build_loss_closure(
    mu_half,
    s,
    rotations,
    image_shape,
    volume_shape,
    shifted_half,
    ctf2_over_nv_half,
    weights_half,
    log_resp,
    post_mean,
    post_Hinv,
    ridge_lambda: float,
):
    def loss_fn(U_half):
        return _expected_nll_half(
            U_half,
            mu_half,
            s,
            rotations,
            image_shape,
            volume_shape,
            shifted_half,
            ctf2_over_nv_half,
            weights_half,
            log_resp,
            post_mean,
            post_Hinv,
            ridge_lambda=ridge_lambda,
        )

    return loss_fn


def _project_factor(U_half, volume_shape, k_max, weights_half_volume):
    """Apply the half-volume projection chain from spec Section 8.3.3:

    U_real_proj = project_to_real_volume_subspace(U_raw, volume_shape)
    U_band      = radial_band_limit_half(U_real_proj, volume_shape, k_max)
    U_new       = real_volume_orthonormalize_half(U_band, weights, N_full)

    The first step (`project_to_real_volume_subspace_batch`) is
    load-bearing: `jax.value_and_grad` produces free complex
    gradients that don't respect the half-volume rfft layout's
    Hermitian-symmetry constraint, so the post-gradient `U` has
    random imaginary content in conjugate-symmetric pairs. Without
    this projection, the subsequent Cholesky orthonormalization
    rotates `U` wildly. Caught by the oracle-init factor-update
    diagnostic test on this branch.
    """
    N_full = int(np.prod(volume_shape))
    U_proj = project_to_real_volume_subspace_batch(U_half, volume_shape)
    U_band = radial_band_limit_half(U_proj, volume_shape, k_max)
    return real_volume_orthonormalize_half(U_band, weights_half_volume, N_full)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def update_factor_one_outer_step(
    config,
    init: PPCAInit,
    batch_full,
    rotations,
    translations,
    ctf_params,
    noise_variance_full,
    *,
    inner_steps: int = 3,
    lr: float = 1e-2,
    k_max: float | None = None,
    ridge_lambda: float = 1e-4,
) -> PPCAInit:
    """Run one outer factor-update step.

    1. E-step: compute responsibilities and posterior moments under
       the current `(mu, U, s)`.
    2. Inner gradient steps on `U` only, with `s` and `mu` frozen.
       The loss is the expected complete-data NLL evaluated under
       the **frozen** (γ, m, Hinv) — i.e. these are NOT recomputed
       inside the inner loop. Per spec Section 8.3.2, this is the
       Generalized-EM form.
    3. Apply the real-O(q) gauge-fix chain (band-limit +
       orthonormalize). No Hermitian projection because the
       half-volume layout makes that structural.
    """
    image_shape = config.image_shape
    volume_shape = config.volume_shape
    weights_half_image = make_half_image_weights(image_shape)
    weights_half_volume = make_half_volume_weights(volume_shape)

    # E-step: snapshot the current posterior moments
    mean_proj_half = _slice_mu_half(init.mu, rotations, image_shape, volume_shape).astype(jnp.complex128)
    u_proj_half = _slice_U_half(init.U, rotations, image_shape, volume_shape).astype(jnp.complex128)
    shifted_half, ctf2_over_nv_half, _ctf_half = _preprocess_batch_to_half(
        config, batch_full, translations, ctf_params, noise_variance_full
    )
    stats = score_from_half_image_projections(
        mean_proj_half, u_proj_half, init.s, shifted_half, ctf2_over_nv_half, weights_half_image
    )

    # Build loss closure with the FROZEN posterior moments
    loss_fn = _build_loss_closure(
        mu_half=init.mu,
        s=init.s,
        rotations=rotations,
        image_shape=image_shape,
        volume_shape=volume_shape,
        shifted_half=shifted_half,
        ctf2_over_nv_half=ctf2_over_nv_half,
        weights_half=weights_half_image,
        log_resp=stats.log_resp,
        post_mean=stats.post_mean,
        post_Hinv=stats.post_Hinv,
        ridge_lambda=ridge_lambda,
    )
    grad_fn = jax.value_and_grad(loss_fn)

    U = init.U
    for _step in range(inner_steps):
        _val, grad_U = grad_fn(U)
        # JAX uses Wirtinger calculus: for a real-valued loss `f(z)`
        # with complex `z`, `jax.grad f(z)` returns the conjugate of
        # the descent direction. The actual steepest-descent step is
        # `z - lr * conj(grad)`. This was caught by the
        # gradient-direction diagnostic on this branch.
        U = U - lr * grad_U.conj()

    # Real-volume gauge fix
    if k_max is None:
        k_max = float(volume_shape[0]) / 4.0
    U_new = _project_factor(U, volume_shape, k_max, weights_half_volume)

    return PPCAInit(
        mu=init.mu,
        U=U_new,
        s=init.s,  # strictly frozen per Q2
        volume_shape=init.volume_shape,
    )


# ---------------------------------------------------------------------------
# Stage 1D — full soft M-step (per spec Section 11.6)
# ---------------------------------------------------------------------------


def update_factor_full_ecm(
    config,
    init: PPCAInit,
    batch_full,
    rotations,
    translations,
    ctf_params,
    noise_variance_full,
    *,
    max_inner_steps: int = 200,
    lr: float = 1e-4,
    grad_norm_tol: float = 1e-4,
    k_max: float | None = None,
    ridge_lambda: float = 1e-4,
    line_search: bool = True,
    line_search_shrink: float = 0.5,
    line_search_min_lr_frac: float = 1e-4,
):
    """Stage 1D / "full soft M-step" factor update.

    Differs from `update_factor_one_outer_step` in two ways:

    1. **Inner-loop convergence**: instead of taking a fixed K=3
       gradient steps, we run the inner loop until the gradient norm
       drops below `grad_norm_tol` (or `max_inner_steps` is reached).
       At convergence, the result is the *local minimum of the
       expected complete-data NLL with respect to U*, which is what
       the proper soft M-step computes (modulo a closed-form solve
       vs. an iterative one).

    2. **Backtracking line search**: each step shrinks the learning
       rate until the loss actually decreases. This makes the
       inner loop monotone in the loss, which is what spec
       Section 11.5 criterion 5 requires for the GEM objective.

    The closed-form per-voxel ECM solve is more efficient at scale
    but requires per-voxel q×q linear algebra over the rotation set;
    the iterative version below produces the same fixed point at
    much smaller code volume and is what v0 ships.

    Returns
    -------
    out : PPCAInit
        Updated factor with mu and s preserved.
    info : dict
        `n_inner_steps`, `final_grad_norm`, `final_loss`, and
        `converged` (True if grad norm dropped below tolerance).
    """
    image_shape = config.image_shape
    volume_shape = config.volume_shape
    weights_half_image = make_half_image_weights(image_shape)
    weights_half_volume = make_half_volume_weights(volume_shape)

    # E-step snapshot
    mean_proj_half = _slice_mu_half(init.mu, rotations, image_shape, volume_shape).astype(jnp.complex128)
    u_proj_half = _slice_U_half(init.U, rotations, image_shape, volume_shape).astype(jnp.complex128)
    shifted_half, ctf2_over_nv_half, _ctf_half = _preprocess_batch_to_half(
        config, batch_full, translations, ctf_params, noise_variance_full
    )
    stats = score_from_half_image_projections(
        mean_proj_half,
        u_proj_half,
        init.s,
        shifted_half,
        ctf2_over_nv_half,
        weights_half_image,
    )

    loss_fn = _build_loss_closure(
        mu_half=init.mu,
        s=init.s,
        rotations=rotations,
        image_shape=image_shape,
        volume_shape=volume_shape,
        shifted_half=shifted_half,
        ctf2_over_nv_half=ctf2_over_nv_half,
        weights_half=weights_half_image,
        log_resp=stats.log_resp,
        post_mean=stats.post_mean,
        post_Hinv=stats.post_Hinv,
        ridge_lambda=ridge_lambda,
    )
    grad_fn = jax.value_and_grad(loss_fn)

    U = init.U
    cur_lr = float(lr)
    cur_loss, _ = grad_fn(U)
    cur_loss = float(cur_loss)
    initial_loss = cur_loss
    final_grad_norm = float("inf")
    converged = False
    step_idx = 0

    for step_idx in range(1, max_inner_steps + 1):
        cur_loss, grad_U = grad_fn(U)
        cur_loss = float(cur_loss)
        gn = float(jnp.sqrt(jnp.sum(jnp.abs(grad_U) ** 2)).real)
        final_grad_norm = gn
        if gn < grad_norm_tol:
            converged = True
            break

        # Wirtinger descent direction (see comment in
        # update_factor_one_outer_step): conj(grad), not grad.
        descent_dir = grad_U.conj()
        if line_search:
            trial_lr = cur_lr
            min_trial_lr = float(line_search_min_lr_frac * lr)
            while trial_lr > min_trial_lr:
                U_trial = U - trial_lr * descent_dir
                trial_loss = float(loss_fn(U_trial))
                if trial_loss < cur_loss:
                    U = U_trial
                    cur_loss = trial_loss
                    cur_lr = trial_lr
                    break
                trial_lr *= line_search_shrink
            else:
                # No step decreases the loss → declare convergence
                converged = True
                break
        else:
            U = U - cur_lr * descent_dir

    # Final gauge fix
    if k_max is None:
        k_max = float(volume_shape[0]) / 4.0
    U_new = _project_factor(U, volume_shape, k_max, weights_half_volume)

    out = PPCAInit(
        mu=init.mu,
        U=U_new,
        s=init.s,
        volume_shape=init.volume_shape,
    )
    info = {
        "n_inner_steps": int(step_idx),
        "final_grad_norm": float(final_grad_norm),
        "initial_loss": float(initial_loss),
        "final_loss": float(cur_loss),
        "loss_decrease": float(initial_loss - cur_loss),
        "converged": bool(converged),
    }
    return out, info


# ===========================================================================
# Closed-form M-step for U  (Tipping & Bishop 1999, pose-marginal, NEAREST)
# ===========================================================================
#
# WHAT THIS COMPUTES
# ------------------
# Given the OLD posterior moments (gamma, m, Hinv) from one E-step
# of the score kernel, this returns the U that minimizes the expected
# complete-data NLL with respect to U (with mu and s held fixed).
# That is the canonical PPCA M-step (Tipping & Bishop 1999, §3),
# extended from per-image to pose-marginal cryo-EM.
#
# WHY IT'S A DIRECT BLOCK SOLVE
# -----------------------------
# Because v0 ab-initio uses NEAREST discretization throughout
# (simulator + score kernel + mean update + this M-step), the slice
# operator A_g[pixel, voxel] is binary: each pixel hits exactly one
# voxel. As a consequence,
#
#     A_g^T diag(CTF_i^2 / sigma_i^2) A_g
#
# is EXACTLY diagonal in the voxel basis (zero off-voxel coupling).
# The M-step's normal equation
#
#     [ sum_{i,g,t} gamma_{igt} · A_g^T diag(CTF^2/sigma^2) A_g
#       (·) C_{igt} ]   +   lambda · I  =  RHS
#
# therefore decomposes voxel-by-voxel into independent q×q linear
# systems. No CG, no preconditioner, no nullspace handling — just
# `vmap(jnp.linalg.solve)` over voxels.
#
# DERIVATION SKETCH (full version in docs/math/ppca_closed_form_mstep.md)
# ----------------------------------------------------------------------
# Forward model in image space (per image i, pose (g, t)):
#
#     y_i = CTF_i · S_t · A_R · (mu + U alpha_i) + eps_i
#
# Expected complete-data NLL with respect to U (drop U-independent
# terms; with E[alpha] = m, E[alpha alpha^T] = C = Hinv + m m^T):
#
#     Q(U) = -1/2 * sum_{igt} gamma_{igt} *
#                E[ ||y - CTF*A_g*mu - CTF*A_g*U*alpha||^2 / sigma^2 ]
#          = const
#            + sum_{igt} gamma_igt * Re tr( m · U^* · A_g^T (CTF/s2) (y - CTF·A_g·mu) )
#            - 1/2 * sum_{igt} gamma_igt * tr( U^* · A_g^T (CTF^2/s2) A_g · U · C )
#            - 1/2 * lambda * ||U||^2
#
# Wirtinger derivative with respect to U^* (set to zero):
#
#     sum_{igt} gamma · A_g^T (CTF^2/s2) A_g · U · C  +  lambda · U
#       =  sum_{igt} gamma · A_g^T (CTF/s2 · (y - CTF·A_g·mu)) · m^T
#
# Per-voxel (because A_g is binary under nearest):
#
#     M[v] · U[v, :]  =  B[v, :]
#
# with
#
#     M[v]_kl = sum_{igt} gamma_igt · Psi_{ig}[v] · C_{igt}[k, l]   + lambda * delta_kl
#     B[v, k] = sum_{igt} gamma_igt · m_{igt, k} · b_{ig}[v]
#
# where
#
#     Psi_{ig}[v] = adj_slice_g(  w_p · (CTF_i^2 / sigma_i^2)_p  )[v]
#     b_{ig}[v]   = adj_slice_g(  w_p · ( shifted_half_{it,p}
#                                          - (CTF^2/sigma^2)_p · mean_proj_{g,p} )  )[v]
#
# - `w_p` are the rfft Hermitian half-image weights (1 at DC/Nyquist
#   columns, 2 elsewhere). They appear because the score kernel and
#   the M-step compute inner products in the FULL-image sense, and
#   the half-image identity is `<a,b>_full = sum_half w * conj(a)*b`
#   for Hermitian arrays.
# - The "residual image" `shifted_half - (CTF^2/sigma^2) · mean_proj`
#   is exactly what the score kernel's `_build_b` computes (b1 - b2).
# - Both Psi and b live in the half-image basis BEFORE adj_slice.
#   We accumulate them per pose with a weighted sum over images, then
#   adj_slice once per (k, l) (for M) or per k (for B).
#
# REPLACES
# --------
# `update_factor_one_outer_step` (gradient descent) and
# `update_factor_full_ecm` (gradient with line search). Both retained
# only for parity testing — do not call from new code.
# ===========================================================================


def _update_eigenvalues(gamma, m, Hinv, n_img):
    """Tipping-Bishop eigenvalue update from E-step moments.

    s_new[k] = (1/n_img) Σ_{i,g,t} γ_{i,g,t} (m²_{i,g,t,k} + H⁻¹_{i,g,kk})

    γ sums to 1 over (g,t) per image, so the effective sample size is n_img.
    """
    m2 = jnp.sum(gamma[..., None] * m**2, axis=(1, 2))  # (n_img, q)
    Hinv_diag = jnp.diagonal(Hinv, axis1=-2, axis2=-1)  # (n_img, n_rot, q)
    gamma_rot = jnp.sum(gamma, axis=2)  # (n_img, n_rot) — sum over translations
    cov_term = jnp.sum(gamma_rot[..., None] * Hinv_diag, axis=1)  # (n_img, q)
    s_new = jnp.mean(m2 + cov_term, axis=0)  # (q,)
    return jnp.maximum(s_new, 1e-10)


def _orthonormalize_and_update_s(U_half, s, weights, volume_size):
    """Orthonormalize U and absorb the gauge transform into s.

    Given U_raw with non-identity weighted Gram G = L L^H:
    1. U_orth = L^{-1} U_raw  (orthonormal rows)
    2. Λ_new = L Λ L^H  (absorb the inverse transform into s)
    3. Eigendecompose Λ_new = V D V^T, apply V^T to U_orth

    The result (V^T L^{-1} U_raw, diag(D)) represents the same
    covariance as (U_raw, diag(s)) but with orthonormal rows and
    a diagonal latent covariance.
    """
    q = U_half.shape[0]
    Uw = U_half * weights[None, :].astype(U_half.dtype)
    G = (Uw @ U_half.conj().T).real / float(volume_size)  # (q, q)
    G = G + 1e-12 * jnp.eye(q, dtype=G.dtype)
    L = jnp.linalg.cholesky(G)

    # Absorb L into s: Λ_new = L diag(s) L^T
    Lambda_new = L * s[None, :] @ L.T  # (q, q)

    # Eigendecompose to get new diagonal s and rotation
    eigvals, V = jnp.linalg.eigh(Lambda_new)
    # eigh returns ascending order; flip to descending
    eigvals = eigvals[::-1]
    V = V[:, ::-1]

    # U_orth = L^{-1} U_raw, then rotate by V^T
    U_orth = jax.scipy.linalg.solve_triangular(L, U_half, lower=True)
    U_new = V.T @ U_orth

    return U_new, jnp.maximum(eigvals, 1e-10)


def update_factor_closed_form(
    config,
    init: PPCAInit,
    batch_full,
    rotations,
    translations,
    ctf_params,
    noise_variance_full,
    *,
    W_prior=None,
    ridge_lambda: float = 1e-4,
    update_s: bool = False,
) -> PPCAInit:
    """Closed-form per-voxel M-step for the PPCA factor U.

    Pose-marginal extension of `recovar/heterogeneity/ppca.py::M_step`
    adapted to the half-volume rfft layout. Exact under the v0
    nearest-discretization design choice. See the module-level
    "WHAT THIS COMPUTES" comment immediately above for the math.

    The full call costs:
      1. one E-step via `score_from_half_image_projections`,
      2. one weighted sum-over-images per pose for `M_im` and `B_im`,
      3. q*q + q half-volume `adjoint_slice_volume` calls,
      4. `vmap(jnp.linalg.solve)` over the V_half voxels of a q×q
         system per voxel.

    When ``update_s=False`` (default): no orthonormalization, s frozen.
    When ``update_s=True``: Tipping-Bishop eigenvalue update from the
    E-step moments, then joint orthonormalization that absorbs the
    gauge transform into s (see ``_orthonormalize_and_update_s``).

    No PCG, no preconditioner, no learning rate, no line search, and
    no band-limit. After the direct solve we project back to the
    real-volume half-spectrum subspace but do NOT orthonormalize,
    because with frozen s the Cholesky whitening would change the
    represented covariance U diag(s) U^H.  The E-step and all
    downstream metrics handle non-orthonormal U correctly.

    Parameters
    ----------
    config, init, batch_full, rotations, translations, ctf_params,
    noise_variance_full : same as the legacy gradient M-steps.
    ridge_lambda : float
        Tikhonov regularizer added to the diagonal of every per-voxel
        ``M[v]``. Default ``1e-4`` is small (just enough to keep the
        per-voxel solve well-conditioned at voxels with very low CTF²
        coverage); the per-voxel system is well-conditioned almost
        everywhere, so the ridge mostly only matters in the corners
        of the half-volume that no rotation touches.

    Returns
    -------
    PPCAInit with U updated; mu, s, volume_shape unchanged.
    """
    from recovar.core.slicing import adjoint_slice_volume

    image_shape = config.image_shape
    volume_shape = config.volume_shape
    weights_half_image = make_half_image_weights(image_shape)

    # -----------------------------------------------------------------
    # 1. E-STEP — get gamma, m, Hinv for the current (mu, U, s).
    # -----------------------------------------------------------------
    # The score kernel handles the latent-z marginalization (gives us
    # the Gaussian posterior of alpha given y, g, t) and the discrete
    # pose marginalization (gives us responsibilities gamma over the
    # fixed (rotation, translation) grid).
    mean_proj_half = _slice_mu_half(init.mu, rotations, image_shape, volume_shape).astype(jnp.complex128)
    u_proj_half = _slice_U_half(init.U, rotations, image_shape, volume_shape).astype(jnp.complex128)
    shifted_half, ctf2_over_nv_half, _ctf_half = _preprocess_batch_to_half(
        config, batch_full, translations, ctf_params, noise_variance_full
    )
    stats = score_from_half_image_projections(
        mean_proj_half,
        u_proj_half,
        init.s,
        shifted_half,
        ctf2_over_nv_half,
        weights_half_image,
    )
    gamma = jnp.exp(stats.log_resp)  # (n_img, n_rot, n_trans)  responsibilities
    m = stats.post_mean  # (n_img, n_rot, n_trans, q)  E[alpha | y, g, t]
    Hinv = stats.post_Hinv  # (n_img, n_rot, q, q)        Cov(alpha | y, g)

    n_img, n_rot, n_trans, q = m.shape
    half_volume_size = init.U.shape[-1]

    # Phase 8 streaming-compatible accumulators: no per-image
    # intermediates above (n_img, n_rot, q²) get materialized.
    # Mathematically identical to the prior `C = Hinv + mm^T` path
    # but uses the linearity of the (i, t)-contraction to feed
    # XLA contractions that reduce over i directly.

    # ---- ctf2_w stays the same (rfft Hermitian weights baked in) ----
    ctf2_w = ctf2_over_nv_half * weights_half_image[None, :]  # (n_img, n_half_image)

    # ---- M_im[g, k, l, p] = sum_{i,t} γ_{igt} · C_{igt,kl} · ctf2_w[i,p] ----
    # Split into Hinv part (translation-independent) and mm^T part.
    gamma_sum_t = jnp.sum(gamma, axis=2)  # (n_img, n_rot)
    # Hinv contribution: sum_i (sum_t γ_{igt}) · Hinv_{ig,kl} · ctf2_w[i,p]
    M_im_Hinv = jnp.einsum("ig,igkl,ip->gklp", gamma_sum_t, Hinv, ctf2_w)
    # m m^T contribution: sum_{i,t} γ_{igt} · m_{igt,k} · m_{igt,l} · ctf2_w[i,p]
    M_im_mm = jnp.einsum("igt,igtk,igtl,ip->gklp", gamma, m, m, ctf2_w)
    M_im = M_im_Hinv + M_im_mm  # (n_rot, q, q, n_half_image)

    # ---- B_im[g, k, p] : per-pose residual accumulator ----
    # Math:  residual_{igt,p} = shifted_half[i,t,p] - ctf2_over_nv[i,p] · mean_proj[g,p]
    #        B_im[g,k,p]      = sum_{i,t} γ_{igt} · m_{igt,k} · w_p · residual_{igt,p}
    # Split into shifted-part (Part1) and mean-part (Part2).
    shifted_w = shifted_half * weights_half_image[None, None, :]  # (n_img, n_trans, n_half_image)
    # Fused over i,t in one contraction → no (n_img, n_rot, n_trans, q) intermediate.
    Part1 = jnp.einsum("igt,igtk,itp->gkp", gamma, m, shifted_w)  # (n_rot, q, n_half_image)

    # weight_B_t[i,g,k] = sum_t γ_{igt} m_{igt,k} : single fused einsum.
    weight_B_t = jnp.einsum("igt,igtk->igk", gamma, m)  # (n_img, n_rot, q)
    M_const = jnp.einsum("igk,ip->gkp", weight_B_t, ctf2_w)  # (n_rot, q, n_half_image)
    Part2 = mean_proj_half[:, None, :] * M_const
    B_im = Part1 - Part2  # (n_rot, q, n_half_image)

    # -----------------------------------------------------------------
    # 3. BACK-PROJECT EACH ACCUMULATOR INTO THE HALF-VOLUME.
    # -----------------------------------------------------------------
    # Under nearest discretization, adj_slice[v] = sum over pixels
    # that map to voxel v (each pixel maps to exactly one voxel). So:
    #
    #     M[v]_kl = adj_slice( M_im[:, k, l, :] )[v]
    #     B[v, k] = adj_slice( B_im[:, k,    :] )[v]

    def _adj_one(im_per_rot):
        # (n_rot, n_half_image) -> (half_volume_size,)
        return adjoint_slice_volume(
            im_per_rot,
            rotations,
            image_shape,
            volume_shape,
            "nearest",  # binary A_g => exact diagonal solve below
            half_image=True,
            half_volume=True,
        ).reshape(-1)

    # B_voxel: (q, half_volume_size)
    B_voxel = jax.vmap(_adj_one, in_axes=1)(B_im)

    # M_voxel: (q, q, half_volume_size). vmap over the (q*q) middle axis.
    M_im_flat = M_im.reshape(n_rot, q * q, M_im.shape[-1])  # (n_rot, q*q, n_half_image)
    M_voxel_flat = jax.vmap(_adj_one, in_axes=1)(M_im_flat)  # (q*q, half_volume_size)
    M_voxel = M_voxel_flat.reshape(q, q, half_volume_size)  # (q, q, V_half)

    # M[v] should be Hermitian PSD by construction (sum of γ·Ψ·C with
    # C ≽ 0 and Ψ ≥ 0). Symmetrize to clean up numerical asymmetry
    # from the float64 GEMMs above.
    M_voxel = 0.5 * (M_voxel + jnp.conj(jnp.swapaxes(M_voxel, 0, 1)))

    # -----------------------------------------------------------------
    # 4. PER-VOXEL DIRECT SOLVE.
    # -----------------------------------------------------------------
    if W_prior is not None:
        reg = jax.vmap(jnp.diag)(1.0 / (W_prior.T + 1e-16))
        M_per_voxel = jnp.moveaxis(M_voxel, -1, 0) + reg
    else:
        eye_q = jnp.eye(q, dtype=jnp.complex128)
        M_per_voxel = jnp.moveaxis(M_voxel + ridge_lambda * eye_q[:, :, None], -1, 0)

    B_per_voxel = jnp.moveaxis(B_voxel, -1, 0)

    U_per_voxel = jax.vmap(jnp.linalg.solve)(M_per_voxel, B_per_voxel)
    U_new = jnp.moveaxis(U_per_voxel, 0, -1).astype(jnp.complex128)
    U_new = project_to_real_volume_subspace_batch(U_new, volume_shape)

    if not update_s:
        # With s frozen, L^{-1} whitening would change the represented
        # covariance U diag(s) U^H.  Skip orthonormalization.
        return PPCAInit(
            mu=init.mu,
            U=U_new,
            s=init.s,
            volume_shape=init.volume_shape,
        )

    # --- Joint orthonormalization + eigenvalue update ---
    # Orthonormalize U, then absorb the gauge transform into s so the
    # represented covariance U diag(s) U^H is preserved.
    s_new = _update_eigenvalues(gamma, m, Hinv, n_img)
    weights_half_volume = make_half_volume_weights(volume_shape)
    U_new, s_new = _orthonormalize_and_update_s(U_new, s_new, weights_half_volume, int(np.prod(volume_shape)))
    return PPCAInit(
        mu=init.mu,
        U=U_new,
        s=s_new,
        volume_shape=init.volume_shape,
    )
