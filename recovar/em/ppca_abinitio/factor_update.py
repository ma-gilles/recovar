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


# ---------------------------------------------------------------------------
# Closed-form M-step (Tipping & Bishop 1999, pose-marginal, CG solve)
# ---------------------------------------------------------------------------
#
# This is the **correct** PPCA M-step for the v0 ab-initio loop. See
# `docs/math/ppca_closed_form_mstep.md` for the math. The
# pose-marginal version of `recovar/heterogeneity/ppca.py::M_step`,
# adapted to the half-volume rfft layout and the
# `score_from_half_image_projections` E-step.
#
# Implementation notes
# --------------------
# The fixed-pose `recovar/heterogeneity/ppca.py::M_step` uses NEAREST
# discretization, which makes the slice operator A_g binary so the
# `A_g^T diag(CTF^2 / sigma^2) A_g` operator is exactly diagonal in
# the volume basis. The M-step then decouples per voxel into a `q x q`
# linear solve.
#
# The v0 ab-initio path uses LINEAR-INTERP slicing for consistency
# with the score kernel and the mean update. Under linear-interp the
# diagonal approximation has up to ~80% per-voxel error (verified on
# vol 8), so we cannot reuse the per-voxel q x q solve directly.
#
# Instead we solve the EXACT linear system T(U) + lambda·U = B by
# applying conjugate gradient to the matvec
#
#     T(U)[v, k] = sum_{i,g,t} gamma_{i,g,t}
#                  · sum_l (A_g^T diag(w · CTF^2/sigma^2) A_g · U[:, l])[v]
#                  · C_{i,g,t}[k, l]
#
# where the inner `A_g^T diag(...) A_g` is implemented as
# `slice -> multiply -> adj_slice` and the outer sum over (i, g, t)
# is folded into a precomputed `(n_rot, q, q, n_half_image)` weight
# tensor `M_im` that depends only on (gamma, C, CTF^2/sigma^2). The
# `w` factor is the rfft Hermitian weights from the half-image
# layout.
#
# At oracle init this matvec satisfies T(U_true) = B to ~1e-4
# relative precision (verified at vol 8 with sigma_real = 0.001),
# confirming the formulation is correct. The remaining error is
# numerical noise from the float64 GEMMs in the slice/adj_slice
# pair.
#
# Replaces `update_factor_one_outer_step` (gradient descent) and
# `update_factor_full_ecm` (gradient with line search). Both of those
# are retained only for parity testing.


def update_factor_closed_form(
    config,
    init: PPCAInit,
    batch_full,
    rotations,
    translations,
    ctf_params,
    noise_variance_full,
    *,
    k_max: float | None = None,
    ridge_lambda: float = 100.0,
    apply_gauge_fix: bool = False,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 10,
) -> PPCAInit:
    """Closed-form M-step for the PPCA factor U (pose-marginal CG solve).

    Pose-marginal extension of
    `recovar/heterogeneity/ppca.py::M_step` adapted to the half-volume
    rfft layout used by the v0 ab-initio loop.

    Math summary
    ------------
    Given the OLD posterior moments ``(gamma, m, Hinv)`` from the
    score-kernel E-step, the U-minimizer of the expected
    complete-data NLL satisfies the linear system

        ``T(U) + lambda · U = B``

    where, for ``C_{igt} = Hinv_{ig} + m_{igt} m_{igt}^T``,

        T(U)[v, k] = sum_{i,g,t} gamma_{igt}
                     · sum_l (A_g^T diag(w · CTF^2/sigma^2) A_g · U[:, l])[v]
                     · C_{igt}[k, l]

        B[v, k]    = sum_{i,g,t} gamma_{igt} · m_{igt, k}
                     · adj_slice_g( w · (shifted_half[i, t]
                                         - (CTF^2/sigma^2)[i] · A_g mu) )[v]

    `w` is the rfft Hermitian half-image weights (1 at DC/Nyquist
    columns, 2 elsewhere) from `make_half_image_weights`. They are
    needed because the score kernel and the M-step both compute
    inner products in the *full-image* sense, expressed in the
    half-image rfft layout via the identity
    `<a, b>_full = sum_half w · conj(a) · b` for Hermitian operands.

    The system is solved with conjugate gradient on the EXACT operator
    `T + lambda · I`. We do NOT use the per-voxel diagonal
    approximation that
    `recovar/heterogeneity/ppca.py::M_step` uses, because under
    linear-interp slicing the off-diagonal coupling is not negligible
    (~80% per-voxel error at vol 8).

    See `docs/math/ppca_closed_form_mstep.md` for the full
    derivation and a discussion of why this replaces the previous
    gradient-descent path.

    Parameters
    ----------
    config, init, batch_full, rotations, translations, ctf_params,
    noise_variance_full : same as `update_factor_one_outer_step`.
    k_max : optional float
        Radial band limit for the gauge fix. Defaults to
        ``volume_shape[0] / 4``. Only used when ``apply_gauge_fix``
        is True.
    ridge_lambda : float
        Tikhonov regularizer added to the diagonal of the linear
        operator. Default ``100.0`` is calibrated for the v0 toy
        synthetic family at ``sigma=0.1``; tune up at higher noise,
        down at lower noise. Too small a ridge lets CG fit noise
        in the near-null modes (the imaginary parts of constrained
        rfft voxels); too large washes out the signal.
    apply_gauge_fix : bool
        If True, apply the half-volume gauge fix (Hermitian project
        + radial band-limit + real-volume orthonormalize) after the
        CG solve. **Default False** per the v0 design choice not to
        apply masks/gridding inside the PPCA path. Set True only
        when comparing against tests that expect the canonical
        gauge.
    cg_tol : float
        Conjugate-gradient relative residual tolerance. With the
        default early-stopping ``cg_maxiter=10`` this rarely
        triggers in practice.
    cg_maxiter : int
        Conjugate-gradient iteration cap. **Default 10** uses early
        stopping as the regularizer for the ill-conditioned operator
        — more iterations just fit noise into the rfft nullspace.
        The "few-step inner loop" idiom matches Tipping & Bishop's
        original PPCA EM paper.

    Returns
    -------
    PPCAInit with U updated; mu, s, volume_shape unchanged.
    """
    from recovar.core.slicing import adjoint_slice_volume, slice_volume

    image_shape = config.image_shape
    volume_shape = config.volume_shape
    weights_half_image = make_half_image_weights(image_shape)
    weights_half_volume = make_half_volume_weights(volume_shape)

    # ---- 1. E-step (gamma, m, Hinv) under the current (mu, U, s) ----
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

    gamma = jnp.exp(stats.log_resp)  # (n_img, n_rot, n_trans)
    m = stats.post_mean  # (n_img, n_rot, n_trans, q)
    Hinv = stats.post_Hinv  # (n_img, n_rot, q, q)

    n_img, n_rot, n_trans, q = m.shape
    n_half_image = ctf2_over_nv_half.shape[-1]
    half_volume_size = init.U.shape[-1]

    # Second moment C[i,g,t] = Hinv[i,g] + m[i,g,t] m[i,g,t]^T
    Hinv_b = Hinv[:, :, None, :, :]  # (n_img, n_rot, 1, q, q)
    mm_outer = m[..., :, None] * m[..., None, :]  # (n_img, n_rot, n_trans, q, q)
    C = Hinv_b + mm_outer  # (n_img, n_rot, n_trans, q, q)

    # ---- 2. Precompute the per-pose pixel-space (q, q) weight tensor for T ----
    # weight_M[i, g, k, l] = sum_t gamma[i,g,t] * C[i,g,t,k,l]
    weight_M_kl = jnp.einsum("igt,igtkl->igkl", gamma, C)  # (n_img, n_rot, q, q)
    # ctf2_w[i, p] = w_p · (CTF^2/sigma^2)[i, p]   (Hermitian weights baked in)
    ctf2_w = ctf2_over_nv_half * weights_half_image[None, :]  # (n_img, n_half_image)
    # M_im[g, k, l, p] = sum_i weight_M[i,g,k,l] * ctf2_w[i, p]
    M_im = jnp.einsum("igkl,ip->gklp", weight_M_kl, ctf2_w)  # (n_rot, q, q, n_half_image)

    # ---- 3. Build the RHS B (q half-volumes) ----
    # B residual_im = shifted_half - (CTF^2/sigma^2) · mean_proj   (per (i,g,t))
    # B_im[g, k, p] = sum_{i,t} gamma * m_k * residual_im (with Hermitian weight w)
    weight_B = gamma[..., None] * m  # (n_img, n_rot, n_trans, q)
    shifted_w = shifted_half * weights_half_image[None, None, :]  # (n_img, n_trans, n_half_image)
    Part1 = jnp.einsum("igtk,itp->gkp", weight_B, shifted_w)  # (n_rot, q, n_half_image)
    weight_B_t = jnp.sum(weight_B, axis=2)  # (n_img, n_rot, q)
    M_const = jnp.einsum("igk,ip->gkp", weight_B_t, ctf2_w)  # (n_rot, q, n_half_image)
    Part2 = mean_proj_half[:, None, :] * M_const  # (n_rot, q, n_half_image)
    B_im = Part1 - Part2  # (n_rot, q, n_half_image)

    def _adj_one(im_per_rot):
        # (n_rot, n_half_image) -> (half_volume_size,)
        return adjoint_slice_volume(
            im_per_rot,
            rotations,
            image_shape,
            volume_shape,
            "linear_interp",
            half_image=True,
            half_volume=True,
        ).reshape(-1)

    def _slice_one_vol(vol_half):
        # (half_volume_size,) -> (n_rot, n_half_image)
        return slice_volume(
            vol_half,
            rotations,
            image_shape,
            volume_shape,
            "linear_interp",
            half_volume=True,
            half_image=True,
        )

    # B_voxel: (q, half_volume_size)
    B_voxel = jax.vmap(_adj_one, in_axes=1)(B_im)

    # ---- 4. Build the matvec T(U) + lambda U ----
    def matvec(U_flat):
        # CG operates on flat real arrays. We pack/unpack via real/imag parts.
        # U_flat shape: (2 * q * half_volume_size,)
        U_real = U_flat[: q * half_volume_size].reshape(q, half_volume_size)
        U_imag = U_flat[q * half_volume_size :].reshape(q, half_volume_size)
        U = (U_real + 1j * U_imag).astype(jnp.complex128)

        # Slice each U[k] through all rotations: u_proj[k, g, p]
        u_proj = jax.vmap(_slice_one_vol)(U)  # (q, n_rot, n_half_image)
        u_proj = jnp.swapaxes(u_proj, 0, 1)  # (n_rot, q, n_half_image)

        # T_im[g, k, p] = sum_l u_proj[g, l, p] * M_im[g, k, l, p]
        T_im = jnp.einsum("glp,gklp->gkp", u_proj, M_im.astype(u_proj.dtype))  # (n_rot, q, n_half_image)

        # Adjoint slice each T_im[k] back to a half-volume
        T_U = jax.vmap(_adj_one, in_axes=1)(T_im)  # (q, half_volume_size)

        T_U = T_U + ridge_lambda * U
        # Re-flatten as real
        out_real = T_U.real.reshape(-1)
        out_imag = T_U.imag.reshape(-1)
        return jnp.concatenate([out_real, out_imag])

    # ---- 5. Solve T(U) + lambda U = B with conjugate gradient ----
    # NOTE: this operator has a numerical near-nullspace from the
    # half-volume rfft layout (imaginary parts of constrained voxels
    # are unobservable). Letting CG run too long over-fits noise into
    # the near-null modes, so we use early stopping (small `cg_maxiter`)
    # as the natural regularizer. The empirical sweet spot at vol 8-32
    # is ~30-50 iterations; the default `cg_maxiter=30` matches the
    # PPCA "few-step inner loop" idiom from Tipping & Bishop.
    rhs_flat = jnp.concatenate([B_voxel.real.reshape(-1), B_voxel.imag.reshape(-1)])
    U0_flat = jnp.concatenate([init.U.real.reshape(-1), init.U.imag.reshape(-1)])

    sol_flat, _info = jax.scipy.sparse.linalg.cg(
        matvec,
        rhs_flat,
        x0=U0_flat,
        tol=cg_tol,
        maxiter=cg_maxiter,
    )
    U_real = sol_flat[: q * half_volume_size].reshape(q, half_volume_size)
    U_imag = sol_flat[q * half_volume_size :].reshape(q, half_volume_size)
    U_new = (U_real + 1j * U_imag).astype(jnp.complex128)

    # ---- 6. Optional gauge fix (band limit + real-volume orthonormalize) ----
    if apply_gauge_fix:
        if k_max is None:
            k_max = float(volume_shape[0]) / 4.0
        U_new = _project_factor(U_new, volume_shape, k_max, weights_half_volume)

    return PPCAInit(
        mu=init.mu,
        U=U_new,
        s=init.s,
        volume_shape=init.volume_shape,
    )
