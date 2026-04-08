"""Posterior calibration test (per spec Section 7.4 / reviewer Section 7).

The reviewer's flagship test for the posterior helper. Score parity
proves the *log_scores* are right; this test proves that
`post_mean` and `post_Hinv` are real Bayesian posterior moments,
not just Cholesky outputs that happen to give a correct score.

Setup
-----

Generate `n_img` images at the **true** pose (a single fixed
rotation, identity translation) from the model

    y_i = U_g · α_i + ε_i,    α_i ~ N(0, diag(s)),  ε_i ~ N(0, σ² I)

so that the posterior of α | y, g_true is exactly N(m_i, H_inv_i)
with `m_i` and `H_inv_i` produced by the kernel.

For each image, compute the squared Mahalanobis distance

    d_i² = (α_true_i - m_i)^T H_i (α_true_i - m_i)

Under the model, `α_true | y, g_true ~ N(m, H_inv)`, so
`d² = (α_true - m)^T H (α_true - m) ~ χ²_q` independently across
images.

The 90% quantile of χ²_q is `chi2.ppf(0.9, q)`. The empirical
fraction of images with `d² ≤ chi2.ppf(0.9, q)` should lie in
`[0.85, 0.95]` for `n_img = 256` (a few standard errors window).

If `m` and `H_inv` are NOT the true posterior, the empirical
coverage will systematically deviate. This is the only test that
can detect a "score-correct, posterior-wrong" bug.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.posterior import (
    make_half_image_weights,
    score_from_half_image_projections,
)

pytestmark = pytest.mark.unit

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


IMAGE_SHAPE = (8, 8)
N_FULL = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)


# ---------------------------------------------------------------------------
# Inputs in real space, FT to full layout, then convert to half-image
# ---------------------------------------------------------------------------


def _real_volume_to_full_ft(real_img):
    return jnp.asarray(ftu.get_dft2(jnp.asarray(real_img)).reshape(-1), dtype=jnp.complex128)


def _make_true_pose_inputs(rng, q, n_img, sigma_real=1.0):
    """Generate `n_img` synthetic images at one fixed pose under the model

        y_i = U_g μ_g + U_g · alpha_i + epsilon_i,
        alpha_i ~ N(0, diag(s)), epsilon_i ~ N(0, sigma_real² · I_pixels).

    Everything is built in *real space* and then transformed to the
    centered FT layout, so the resulting full-image FT vectors are
    Hermitian by construction.

    Returns
    -------
    mean_proj_full : (1, N_FULL) complex128 — single rotation
    u_proj_full   : (1, q, N_FULL) complex128
    s             : (q,) float64
    batch_full    : (n_img, N_FULL) complex128
    alpha_true    : (n_img, q) float64 — ground-truth latents
    noise_var_fourier : float — Fourier-space variance per pixel
                        (= sigma_real² · N_FULL, see Section 4.3)
    """
    # Mean: a smooth real-space "blob"
    H, W = IMAGE_SHAPE
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, H),
        np.linspace(-1, 1, W),
        indexing="ij",
    )
    mean_real = np.exp(-(yy**2 + xx**2) * 4.0)

    # Build q PCs as orthogonal real-space basis functions (sinusoids)
    pcs_real = []
    for k in range(q):
        kx = (k % 3) + 1
        ky = (k // 3) + 1
        pc = np.cos(np.pi * kx * yy) * np.cos(np.pi * ky * xx)
        pcs_real.append(pc)
    U_real = np.stack(pcs_real)  # (q, H, W)

    # Latent prior variances
    s = (1.0 / (np.arange(q) + 1.0)).astype(np.float64)  # decreasing

    # Sample alpha and build images in real space
    alpha_true = (rng.standard_normal((n_img, q)) * np.sqrt(s)).astype(np.float64)
    epsilons = (sigma_real * rng.standard_normal((n_img, H, W))).astype(np.float64)
    images_real = mean_real[None] + np.einsum("nq,qhw->nhw", alpha_true, U_real) + epsilons

    # Convert to FT layout
    mean_full = _real_volume_to_full_ft(mean_real).reshape(1, -1)  # (1, N_FULL)
    u_full = jnp.stack([_real_volume_to_full_ft(U_real[k]) for k in range(q)])  # (q, N_FULL)
    u_full = u_full.reshape(1, q, -1)  # (1, q, N_FULL)
    batch_full = jnp.stack([_real_volume_to_full_ft(images_real[i]) for i in range(n_img)])

    # Fourier-space noise variance per pixel: sigma_real² * N_FULL
    noise_var_fourier = sigma_real**2 * N_FULL
    return mean_full, u_full, jnp.asarray(s, dtype=jnp.float64), batch_full, alpha_true, noise_var_fourier


# ---------------------------------------------------------------------------
# Half-image kernel call
# ---------------------------------------------------------------------------


def _call_kernel(mean_proj_full, u_proj_full, s, batch_full, noise_var_fourier):
    n_img = batch_full.shape[0]
    n_rot = mean_proj_full.shape[0]
    q = u_proj_full.shape[1]
    weights_half = make_half_image_weights(IMAGE_SHAPE)

    mean_proj_half = ftu.full_image_to_half_image(jnp.asarray(mean_proj_full), IMAGE_SHAPE)
    u_proj_half = ftu.full_image_to_half_image(
        jnp.asarray(u_proj_full).reshape(n_rot * q, N_FULL), IMAGE_SHAPE
    ).reshape(n_rot, q, N_HALF)

    batch_half = ftu.full_image_to_half_image(jnp.asarray(batch_full), IMAGE_SHAPE)
    # The kernel expects `shifted_half = S_t (CTF * y / sigma^2)` with the
    # CTF/noise weighting already baked in (mirrors compute_bLambdainvPU_terms
    # at recovar/em/heterogeneity.py:127). With CTF=1 and identity translation,
    # this is just `batch_half / noise_var_fourier`.
    shifted_half = (batch_half / noise_var_fourier)[:, None, :]
    ctf2_over_nv_half = jnp.full((n_img, N_HALF), 1.0 / noise_var_fourier, dtype=jnp.float64)

    return score_from_half_image_projections(
        mean_proj_half=mean_proj_half.astype(jnp.complex128),
        u_proj_half=u_proj_half.astype(jnp.complex128),
        s=jnp.asarray(s, dtype=jnp.float64),
        shifted_half=shifted_half.astype(jnp.complex128),
        ctf2_over_nv_half=ctf2_over_nv_half,
        weights_half=weights_half,
    )


# ---------------------------------------------------------------------------
# The calibration test
# ---------------------------------------------------------------------------


def test_posterior_ellipsoid_coverage_at_true_pose():
    """For 256 samples drawn from the model at one fixed pose, the
    empirical 90% Mahalanobis-ellipsoid coverage of α_true under
    the posterior `(m, H_inv)` should lie in [0.85, 0.95]."""
    if scipy_stats is None:
        pytest.skip("scipy not available")

    rng = np.random.default_rng(0xCA1B)
    q = 2
    n_img = 256
    mean_full, u_full, s, batch_full, alpha_true, noise_var_fourier = _make_true_pose_inputs(
        rng, q=q, n_img=n_img, sigma_real=1.0
    )

    stats = _call_kernel(mean_full, u_full, s, batch_full, noise_var_fourier)
    # Single rotation, single translation: drop those axes.
    m = np.asarray(stats.post_mean)[:, 0, 0, :]  # (n_img, q)
    H_inv = np.asarray(stats.post_Hinv)[:, 0, :, :]  # (n_img, q, q)

    # H = inv(H_inv); compute Mahalanobis squared distance
    # d² = (α_true - m)^T H (α_true - m)
    diff = alpha_true - m  # (n_img, q)
    d_sq = np.empty(n_img, dtype=np.float64)
    for i in range(n_img):
        H = np.linalg.inv(H_inv[i])
        d_sq[i] = float(diff[i] @ H @ diff[i])

    # χ²_q 90% quantile
    threshold = float(scipy_stats.chi2.ppf(0.9, df=q))
    coverage = float(np.mean(d_sq <= threshold))

    assert 0.85 <= coverage <= 0.95, (
        f"Empirical 90% ellipsoid coverage = {coverage:.3f}, "
        f"expected in [0.85, 0.95]. Threshold (χ²_{q} @ 0.9) = {threshold:.3f}. "
        "If this fails, the posterior moments produced by the kernel are not "
        "real Bayesian posterior moments — score parity alone does not catch this."
    )


def test_posterior_ellipsoid_coverage_at_50_percent():
    """Same test at the 50% level. The 50% quantile is more
    sensitive to systematic posterior bias."""
    if scipy_stats is None:
        pytest.skip("scipy not available")

    rng = np.random.default_rng(0xCA1C)
    q = 2
    n_img = 256
    mean_full, u_full, s, batch_full, alpha_true, noise_var_fourier = _make_true_pose_inputs(
        rng, q=q, n_img=n_img, sigma_real=1.0
    )
    stats = _call_kernel(mean_full, u_full, s, batch_full, noise_var_fourier)
    m = np.asarray(stats.post_mean)[:, 0, 0, :]
    H_inv = np.asarray(stats.post_Hinv)[:, 0, :, :]

    diff = alpha_true - m
    d_sq = np.empty(len(diff), dtype=np.float64)
    for i in range(len(diff)):
        H = np.linalg.inv(H_inv[i])
        d_sq[i] = float(diff[i] @ H @ diff[i])

    threshold = float(scipy_stats.chi2.ppf(0.5, df=q))
    coverage = float(np.mean(d_sq <= threshold))
    assert 0.42 <= coverage <= 0.58, f"Empirical 50% ellipsoid coverage = {coverage:.3f}, expected in [0.42, 0.58]."
