"""Posterior helper parity against the production full-image score assembly.

The half-image kernel `score_from_half_image_projections` and the
existing full-image production path
(`compute_dot_products_eqx + compute_CTFed_proj_norms_eqx -
compute_bHb_terms`) must produce the same `log_scores` up to a
g-independent additive constant.

This is the load-bearing parity test for `posterior.py`. It uses
real-derived volumes to keep all rfft inner products real, and
identity rotation / single trivial translation to keep the test
free of slicing/translation conventions. Slicing is bypassed
entirely: the test feeds pre-built half-image projections directly
to the kernel, then converts the same data to full-image and
runs the production assembly on it.

Tolerance is `rtol=1e-10` in float64.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
import recovar.em.core as em_core
import recovar.em.heterogeneity as hetero
from recovar.em.ppca_abinitio.posterior import (
    make_half_image_weights,
    score_from_half_image_projections,
)

pytestmark = pytest.mark.unit


IMAGE_SHAPE = (8, 8)
N_FULL = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)


# ---------------------------------------------------------------------------
# Identity-CTF / identity-process forward model config
# ---------------------------------------------------------------------------


def _identity_ctf(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(batch, apply_image_mask=False):
    return batch


class _TinyConfig(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    _ctf: object = eqx.field(static=True)
    _process: object = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, ctf_params, *, half_image=False):
        full = self._ctf(ctf_params, self.image_shape, self.voxel_size)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, batch, apply_image_mask=False):
        return self._process(batch, apply_image_mask=apply_image_mask)


def _make_config():
    return _TinyConfig(
        image_shape=IMAGE_SHAPE,
        _ctf=_identity_ctf,
        _process=_identity_process,
        voxel_size=1.0,
    )


# ---------------------------------------------------------------------------
# Real-derived inputs (Hermitian by construction in full layout)
# ---------------------------------------------------------------------------


def _real_derived_full(rng):
    """Random real image of shape IMAGE_SHAPE, FT'd via the centered
    backward FFT, returned as a flat (N_FULL,) complex128 vector."""
    real = rng.standard_normal(IMAGE_SHAPE).astype(np.float64)
    ft = np.asarray(ftu.get_dft2(jnp.asarray(real)))
    return jnp.asarray(ft.reshape(-1), dtype=jnp.complex128)


def _real_derived_full_batch(rng, n):
    return jnp.stack([_real_derived_full(rng) for _ in range(n)])


# ---------------------------------------------------------------------------
# Production score assembly (full-image, mirrors E_with_precompute)
# ---------------------------------------------------------------------------


def _assemble_production_residual(mean_proj_full, u_proj_full, s, batch_full, n_trans=1):
    """Reproduce the score assembly inside E_with_precompute exactly,
    in float64. Returns `(n_img, n_rot, n_trans)`.

    Per `tests/ppca_abinitio/test_score_matches_e_step_residual_ref.py`
    this assembly equals `-2 log p_het(y|g)` up to a g-independent
    constant.
    """
    config = _make_config()
    n_img = batch_full.shape[0]
    trans = jnp.zeros((n_trans, 2), dtype=jnp.float64)
    ctf_params = jnp.zeros((n_img, 9), dtype=jnp.float64)
    noise_var = jnp.ones(N_FULL, dtype=jnp.float64)

    residuals = em_core.compute_dot_products_eqx(
        config,
        jnp.asarray(mean_proj_full),
        jnp.asarray(batch_full),
        trans,
        ctf_params,
        noise_var,
    )
    bHb = hetero.compute_bHb_terms(
        jnp.asarray(mean_proj_full),
        jnp.asarray(u_proj_full),
        jnp.asarray(s),
        jnp.asarray(batch_full),
        trans,
        ctf_params,
        _identity_ctf,
        noise_var,
        1.0,
        IMAGE_SHAPE,
        _identity_process,
    )
    residuals = residuals - bHb

    proj_squared = jnp.abs(jnp.asarray(mean_proj_full)) ** 2
    proj_norms = em_core.compute_CTFed_proj_norms_eqx(
        config,
        proj_squared,
        ctf_params,
        noise_var,
    )
    return np.asarray(residuals + proj_norms[..., None])


# ---------------------------------------------------------------------------
# Half-image kernel call (mirrors score_and_posterior_moments_eqx, but
# fed pre-sliced data so the test bypasses slice_volume)
# ---------------------------------------------------------------------------


def _call_half_image_kernel(mean_proj_full, u_proj_full, s, batch_full, n_trans=1):
    """Convert full-image inputs to half-image and run the kernel.

    Translation is fixed to (0, 0) and CTF is identity, matching the
    production-parity test setup.
    """
    n_img = batch_full.shape[0]
    n_rot = mean_proj_full.shape[0]
    q = u_proj_full.shape[1]
    weights_half = make_half_image_weights(IMAGE_SHAPE)

    mean_proj_half = ftu.full_image_to_half_image(jnp.asarray(mean_proj_full), IMAGE_SHAPE)
    u_proj_half = ftu.full_image_to_half_image(
        jnp.asarray(u_proj_full).reshape(n_rot * q, N_FULL), IMAGE_SHAPE
    ).reshape(n_rot, q, N_HALF)

    # Identity CTF, σ²=1, no translation: shifted_half = batch_half (broadcast over n_trans)
    batch_half = ftu.full_image_to_half_image(jnp.asarray(batch_full), IMAGE_SHAPE)
    shifted_half = jnp.broadcast_to(batch_half[:, None, :], (n_img, n_trans, N_HALF))
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
# Test inputs
# ---------------------------------------------------------------------------


def _make_inputs(rng, n_rot, n_pc, n_img):
    mean_proj = _real_derived_full_batch(rng, n_rot)  # (n_rot, N_FULL)
    u_rows = jnp.stack([_real_derived_full_batch(rng, n_pc) for _ in range(n_rot)])  # (n_rot, n_pc, N_FULL)
    # Damp U to keep H well-conditioned
    u_rows = 0.1 * u_rows
    s = jnp.asarray(0.5 + rng.uniform(size=n_pc), dtype=jnp.float64)
    batch = _real_derived_full_batch(rng, n_img)
    return mean_proj, u_rows, s, batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_half_image_kernel_log_scores_match_production_assembly():
    """log_scores from the half-image kernel must equal -0.5 * residual
    from the production full-image assembly, up to a per-image
    additive constant."""
    rng = np.random.default_rng(0xC0DE)
    n_rot, n_pc, n_img = 4, 2, 5
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    # Production residual: -2 log p_het up to const, shape (n_img, n_rot, n_trans=1)
    prod_residual = _assemble_production_residual(mean_proj, u_proj, s, batch, n_trans=1)
    prod_log_scores = -0.5 * prod_residual  # (n_img, n_rot, 1)

    # Half-image kernel
    stats = _call_half_image_kernel(mean_proj, u_proj, s, batch, n_trans=1)
    half_log_scores = np.asarray(stats.log_scores)  # (n_img, n_rot, 1)

    # Subtract per-image mean to remove the g-independent constant
    prod_centered = prod_log_scores - prod_log_scores.mean(axis=(1, 2), keepdims=True)
    half_centered = half_log_scores - half_log_scores.mean(axis=(1, 2), keepdims=True)

    np.testing.assert_allclose(half_centered, prod_centered, rtol=1e-10, atol=1e-12)


def test_half_image_kernel_log_scores_higher_q():
    """Same parity at q=4."""
    rng = np.random.default_rng(0xBEEF)
    n_rot, n_pc, n_img = 3, 4, 6
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    prod_residual = _assemble_production_residual(mean_proj, u_proj, s, batch)
    prod_log_scores = -0.5 * prod_residual

    stats = _call_half_image_kernel(mean_proj, u_proj, s, batch)
    half_log_scores = np.asarray(stats.log_scores)

    prod_centered = prod_log_scores - prod_log_scores.mean(axis=(1, 2), keepdims=True)
    half_centered = half_log_scores - half_log_scores.mean(axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(half_centered, prod_centered, rtol=1e-10, atol=1e-12)


def test_half_image_kernel_zero_u_matches_homogeneous():
    """With U=0, the heterogeneous correction is constant in g, so
    the centered log_scores should match the centered homogeneous
    score `-0.5 * ||y - mu_g||^2`."""
    rng = np.random.default_rng(0xDEAD)
    n_rot, n_pc, n_img = 4, 2, 4
    mean_proj, _, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)
    u_zero = jnp.zeros((n_rot, n_pc, N_FULL), dtype=jnp.complex128)

    stats = _call_half_image_kernel(mean_proj, u_zero, s, batch)
    half_log_scores = np.asarray(stats.log_scores)[..., 0]  # drop n_trans

    # Homogeneous reference: -0.5 * ||y - mu_g||^2 (full Hermitian inner product)
    homog_ref = np.empty((n_img, n_rot), dtype=np.float64)
    mean_np = np.asarray(mean_proj)
    batch_np = np.asarray(batch)
    for i in range(n_img):
        for r in range(n_rot):
            d = batch_np[i] - mean_np[r]
            homog_ref[i, r] = -0.5 * (d.conj() @ d).real

    half_centered = half_log_scores - half_log_scores.mean(axis=1, keepdims=True)
    homog_centered = homog_ref - homog_ref.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(half_centered, homog_centered, rtol=1e-10, atol=1e-12)


def test_post_mean_is_real_for_real_derived_inputs():
    """For real-derived inputs, post_mean is computed by Cholesky on
    a real Gram and a real b vector, so it should be real-valued
    (the dtype is float64 already; this test pins that no NaN/Inf
    appears and the imaginary part — if anywhere — is < 1e-10)."""
    rng = np.random.default_rng(0xF00D)
    n_rot, n_pc, n_img = 3, 2, 4
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    stats = _call_half_image_kernel(mean_proj, u_proj, s, batch)
    pm = np.asarray(stats.post_mean)
    assert pm.dtype == np.float64, f"post_mean dtype = {pm.dtype}, expected float64"
    assert np.all(np.isfinite(pm))


def test_post_Hinv_is_positive_definite():
    """H is built as a real symmetric matrix and inverted; Hinv must
    be positive-definite (eigenvalues > 0) at every (i, r)."""
    rng = np.random.default_rng(0xCAFE)
    n_rot, n_pc, n_img = 3, 3, 4
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    stats = _call_half_image_kernel(mean_proj, u_proj, s, batch)
    Hinv = np.asarray(stats.post_Hinv)  # (n_img, n_rot, q, q)
    for i in range(n_img):
        for r in range(n_rot):
            eigs = np.linalg.eigvalsh(Hinv[i, r])
            assert float(np.min(eigs)) > 0, f"Hinv at (i={i}, r={r}) not PD: eigs={eigs}"
