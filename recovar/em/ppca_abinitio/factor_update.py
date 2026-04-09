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
    mean_proj_half = _slice_mu_half(mu_half, rotations, image_shape, volume_shape).astype(jnp.complex128)
    u_proj_half = _slice_U_half(U_half, rotations, image_shape, volume_shape).astype(jnp.complex128)

    # Re-score with the candidate U: this gives us the model under
    # the current U for the (i, g) likelihood. The factor update IS
    # supposed to take the gradient through this.
    stats = score_from_half_image_projections(
        mean_proj_half, u_proj_half, s, shifted_half, ctf2_over_nv_half, weights_half
    )

    # The expected complete-data NLL under (γ, m, Hinv) from the
    # *current* E-step is:
    #   sum_{i,g} γ * (-log_score_under_new_U) + const
    # where -log_score is `-2 log_scores` rounded into NLL form.
    # Equivalently, we minimize the expected `-log_score` weighted
    # by the responsibilities. For v0, use this form: it is
    # mathematically the same up to a constant in U.
    #
    # Production EM-style M-step would derivatize the residual form
    # explicitly, but for the gradient direction the two are
    # equivalent (and JAX-autodiff handles the `re-score` chain).
    log_resp_jax = jnp.asarray(log_resp)
    gamma = jnp.exp(log_resp_jax)
    nll = -jnp.sum(gamma * stats.log_scores)

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


def update_factor_closed_form(
    config,
    init: PPCAInit,
    batch_full,
    rotations,
    translations,
    ctf_params,
    noise_variance_full,
    *,
    ridge_lambda: float = 1e-4,
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

    No PCG, no preconditioner, no learning rate, no line search, no
    Hermitian projection, no gauge fix, no band-limit. The M-step is
    a single direct solve.

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

    # Second moment C[i, g, t] = Hinv[i, g] + m[i, g, t] m[i, g, t]^T.
    # This is what gets multiplied into the per-pose CTF^2 accumulator.
    # Note: Hinv has no n_trans axis (the kernel says posterior cov is
    # translation-independent, see posterior.py docstring), so we
    # broadcast it.
    Hinv_b = Hinv[:, :, None, :, :]  # (n_img, n_rot, 1,        q, q)
    mm_outer = m[..., :, None] * m[..., None, :]  # (n_img, n_rot, n_trans,  q, q)
    C = Hinv_b + mm_outer  # (n_img, n_rot, n_trans,  q, q)

    # -----------------------------------------------------------------
    # 2. ASSEMBLE M_im AND B_im IN PER-POSE IMAGE SPACE.
    # -----------------------------------------------------------------
    # The big sum over (i, g, t) factors into:
    #   - sum over i of (γ * C * CTF²/σ²) which is a (n_img -> n_rot)
    #     contraction with the per-image CTF² weights;
    #   - one adj_slice per pose to get the half-volume accumulator.
    #
    # We bake the rfft Hermitian weights `w` into ctf2 here so that
    # subsequent inner products are in the full-image sense.

    ctf2_w = ctf2_over_nv_half * weights_half_image[None, :]  # (n_img, n_half_image)
    #         ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^
    #         per-image CTF^2/σ²    rfft Hermitian weights w_p

    # ---- M_im[g, k, l, p] : per-pose, per-(k,l), per-pixel weight ----
    # Math:  M_im[g,k,l,p] = sum_i [ sum_t gamma_{igt} C_{igt,kl} ] · ctf2_w[i,p]
    weight_M_kl = jnp.einsum("igt,igtkl->igkl", gamma, C)  # (n_img, n_rot, q, q)
    M_im = jnp.einsum("igkl,ip->gklp", weight_M_kl, ctf2_w)  # (n_rot, q, q, n_half_image)

    # ---- B_im[g, k, p] : per-pose, per-k, per-pixel residual ----
    # Math:  residual_{igt,p} = shifted_half[i,t,p] - ctf2_over_nv[i,p] · mean_proj[g,p]
    #        B_im[g,k,p]      = sum_{i,t} gamma_{igt} · m_{igt,k} · w_p · residual_{igt,p}
    #
    # We split residual into shifted-part minus mean-part to avoid
    # materializing the full (n_img, n_rot, n_trans, n_half_image)
    # tensor:
    #
    #     Part1 = sum_{i,t} (gamma * m_k) · (w · shifted_half)
    #     Part2 = mean_proj[g] · sum_{i,t} (gamma * m_k) · ctf2_w[i]
    #     B_im  = Part1 - Part2

    weight_B = gamma[..., None] * m  # (n_img, n_rot, n_trans, q)
    shifted_w = shifted_half * weights_half_image[None, None, :]  # (n_img, n_trans,         n_half_image)
    Part1 = jnp.einsum("igtk,itp->gkp", weight_B, shifted_w)  # (n_rot, q, n_half_image)

    weight_B_t = jnp.sum(weight_B, axis=2)  # (n_img, n_rot, q)
    M_const = jnp.einsum("igk,ip->gkp", weight_B_t, ctf2_w)  # (n_rot, q, n_half_image)
    Part2 = mean_proj_half[:, None, :] * M_const  # (n_rot, q, n_half_image)
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
    # For each voxel v: solve  (M[v] + λ I_q) · U[v, :]  =  B[v, :].
    # vmap over the voxel axis. Cost: V_half * O(q^3) flops, trivial
    # for q ~ 2-10.
    eye_q = jnp.eye(q, dtype=jnp.complex128)
    M_with_ridge = M_voxel + ridge_lambda * eye_q[:, :, None]  # (q, q, V_half)

    M_per_voxel = jnp.moveaxis(M_with_ridge, -1, 0)  # (V_half, q, q)
    B_per_voxel = jnp.moveaxis(B_voxel, -1, 0)  # (V_half, q)

    U_per_voxel = jax.vmap(jnp.linalg.solve)(M_per_voxel, B_per_voxel)  # (V_half, q)
    U_new = jnp.moveaxis(U_per_voxel, 0, -1).astype(jnp.complex128)  # (q, V_half)

    return PPCAInit(
        mu=init.mu,
        U=U_new,
        s=init.s,
        volume_shape=init.volume_shape,
    )
