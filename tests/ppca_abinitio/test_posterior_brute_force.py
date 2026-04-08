"""Independent dense-numpy verification of `score_from_half_image_projections`.

`test_posterior_matches_production.py` already verifies that the new
half-image kernel matches the existing production score assembly.
That test transitively depends on the audited correctness of
`compute_bHb_terms`. This test is the **independent** verification:
it computes the marginal log-likelihood, posterior mean, and
posterior covariance directly from the dense Σ_y formulation in
float64 numpy, and compares against the kernel.

Setup:
- Real-derived "projections" (random real images FT'd via the
  centered backward FFT) so the rfft inner products are clean.
- Identity CTF, σ²=1, single (0,0) translation.
- Tiny dimensions (q ≤ 3, image_shape (4,4)) so a dense Σ_y of
  shape (16, 16) can be inverted by hand without numerical pain.

Tolerance is `rtol=1e-10` in float64.
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


IMAGE_SHAPE = (4, 4)
N_FULL = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)


# ---------------------------------------------------------------------------
# Brute-force dense Σ_y reference
# ---------------------------------------------------------------------------


def _brute_force_posterior_full(mean_proj_full, u_proj_full, s, batch_full, sigma2=1.0):
    """For each (image i, rotation r), compute the dense posterior
    moments and the marginal log-likelihood up to a g-independent
    constant.

    Inputs are *full-image* (n_*, N_FULL) complex128. The kernel
    being tested operates on half-image inputs but the math is
    equivalent (the half-image weights make the inner products
    match the full-image sums exactly).

    Returns
    -------
    log_scores : (n_img, n_rot) float64
        `-2 log p_het(y|g) - constant`, where the constant is
        absorbed into the per-image normalization later.
    post_mean : (n_img, n_rot, q) float64
        `H^{-1} b` where b = U_g^T (y - μ_g) / σ², H = diag(1/s)
        + U_g^T U_g / σ². For real-derived inputs both are real.
    post_Hinv : (n_img, n_rot, q, q) float64
        H^{-1}.
    """
    mean_proj = np.asarray(mean_proj_full, dtype=np.complex128)
    u_proj = np.asarray(u_proj_full, dtype=np.complex128)
    s = np.asarray(s, dtype=np.float64)
    batch = np.asarray(batch_full, dtype=np.complex128)

    n_img = batch.shape[0]
    n_rot = mean_proj.shape[0]
    n_pc = u_proj.shape[1]
    inv_s_diag = np.diag(1.0 / s)

    log_scores_neg2 = np.empty((n_img, n_rot), dtype=np.float64)
    post_mean = np.empty((n_img, n_rot, n_pc), dtype=np.float64)
    post_Hinv = np.empty((n_img, n_rot, n_pc, n_pc), dtype=np.float64)

    for r in range(n_rot):
        U = u_proj[r].T  # (N_FULL, q)
        UhU = (U.conj().T @ U).real  # (q, q) real-symmetric
        H = inv_s_diag + UhU / sigma2
        H_inv = np.linalg.inv(H)
        sign, logdet_H = np.linalg.slogdet(H)
        assert sign > 0

        mu = mean_proj[r]
        for i in range(n_img):
            resid = batch[i] - mu
            homog_term = (resid.conj() @ resid).real / sigma2  # ||y - mu||^2 / sigma^2
            b = (U.conj().T @ resid).real / sigma2  # (q,) real for Hermitian inputs
            m = H_inv @ b
            bHb = float(b @ m)

            # -2 log p_het = ||y-mu||^2/σ² - bHb + log det H + const(g)
            log_scores_neg2[i, r] = homog_term - bHb + logdet_H

            post_mean[i, r] = m
            post_Hinv[i, r] = H_inv

    log_scores = -0.5 * log_scores_neg2  # convert to log-prob form (matches kernel)
    return log_scores, post_mean, post_Hinv


# ---------------------------------------------------------------------------
# Half-image kernel call (mirrors test_posterior_matches_production helper)
# ---------------------------------------------------------------------------


def _call_kernel(mean_proj_full, u_proj_full, s, batch_full):
    n_img = batch_full.shape[0]
    n_rot = mean_proj_full.shape[0]
    q = u_proj_full.shape[1]
    weights_half = make_half_image_weights(IMAGE_SHAPE)

    mean_proj_half = ftu.full_image_to_half_image(jnp.asarray(mean_proj_full), IMAGE_SHAPE)
    u_proj_half = ftu.full_image_to_half_image(
        jnp.asarray(u_proj_full).reshape(n_rot * q, N_FULL), IMAGE_SHAPE
    ).reshape(n_rot, q, N_HALF)

    batch_half = ftu.full_image_to_half_image(jnp.asarray(batch_full), IMAGE_SHAPE)
    shifted_half = batch_half[:, None, :]  # n_trans=1
    ctf2_over_nv_half = jnp.ones((n_img, N_HALF), dtype=jnp.float64)

    return score_from_half_image_projections(
        mean_proj_half=mean_proj_half.astype(jnp.complex128),
        u_proj_half=u_proj_half.astype(jnp.complex128),
        s=jnp.asarray(s, dtype=jnp.float64),
        shifted_half=shifted_half.astype(jnp.complex128),
        ctf2_over_nv_half=ctf2_over_nv_half,
        weights_half=weights_half,
    )


# ---------------------------------------------------------------------------
# Real-derived inputs
# ---------------------------------------------------------------------------


def _real_derived_full(rng):
    real = rng.standard_normal(IMAGE_SHAPE).astype(np.float64)
    ft = np.asarray(ftu.get_dft2(jnp.asarray(real)))
    return jnp.asarray(ft.reshape(-1), dtype=jnp.complex128)


def _make_inputs(rng, n_rot, n_pc, n_img):
    mean_proj = jnp.stack([_real_derived_full(rng) for _ in range(n_rot)])
    u_proj = jnp.stack([jnp.stack([_real_derived_full(rng) for _ in range(n_pc)]) for _ in range(n_rot)])
    u_proj = 0.1 * u_proj
    s = jnp.asarray(0.5 + rng.uniform(size=n_pc), dtype=jnp.float64)
    batch = jnp.stack([_real_derived_full(rng) for _ in range(n_img)])
    return mean_proj, u_proj, s, batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_kernel_log_scores_match_brute_force_dense_reference():
    rng = np.random.default_rng(0xC0FFEE)
    n_rot, n_pc, n_img = 3, 2, 4
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    ref_log_scores, _ref_m, _ref_Hinv = _brute_force_posterior_full(mean_proj, u_proj, s, batch)
    stats = _call_kernel(mean_proj, u_proj, s, batch)

    kernel_log_scores = np.asarray(stats.log_scores)[..., 0]  # drop n_trans=1

    # Both differ from -0.5 * (full -2 log p_het) by an additive
    # g-independent constant. Subtract per-image mean.
    ref_centered = ref_log_scores - ref_log_scores.mean(axis=1, keepdims=True)
    kernel_centered = kernel_log_scores - kernel_log_scores.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(kernel_centered, ref_centered, rtol=1e-10, atol=1e-12)


def test_kernel_post_mean_matches_brute_force():
    """post_mean = H^{-1} b should equal the dense H^{-1} b at every
    (i, r). Tolerance 1e-10 in float64."""
    rng = np.random.default_rng(0xBAD1DEA)
    n_rot, n_pc, n_img = 3, 3, 4
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    _, ref_m, _ = _brute_force_posterior_full(mean_proj, u_proj, s, batch)
    stats = _call_kernel(mean_proj, u_proj, s, batch)
    kernel_m = np.asarray(stats.post_mean)[..., 0, :]  # drop n_trans=1, shape (n_img, n_rot, q)

    np.testing.assert_allclose(kernel_m, ref_m, rtol=1e-10, atol=1e-12)


def test_kernel_post_Hinv_matches_brute_force():
    """post_Hinv = H^{-1} should equal the dense inverse at every (i, r)."""
    rng = np.random.default_rng(0xFEEDFACE)
    n_rot, n_pc, n_img = 3, 3, 4
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    _, _, ref_Hinv = _brute_force_posterior_full(mean_proj, u_proj, s, batch)
    stats = _call_kernel(mean_proj, u_proj, s, batch)
    kernel_Hinv = np.asarray(stats.post_Hinv)  # (n_img, n_rot, q, q)
    np.testing.assert_allclose(kernel_Hinv, ref_Hinv, rtol=1e-10, atol=1e-12)


def test_kernel_log_resp_normalization():
    """log_resp must sum to 1 in each image (over rot × trans)."""
    rng = np.random.default_rng(7)
    n_rot, n_pc, n_img = 4, 2, 3
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    stats = _call_kernel(mean_proj, u_proj, s, batch)
    log_resp = np.asarray(stats.log_resp)
    resp = np.exp(log_resp)
    sums = resp.reshape(n_img, -1).sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-12)
