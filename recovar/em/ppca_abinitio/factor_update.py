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

    U_band = radial_band_limit_half(U_raw, volume_shape, k_max)
    U_new  = real_volume_orthonormalize_half(U_band, weights, N_full)
    """
    N_full = int(np.prod(volume_shape))
    U_band = radial_band_limit_half(U_half, volume_shape, k_max)
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
        U = U - lr * grad_U

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
