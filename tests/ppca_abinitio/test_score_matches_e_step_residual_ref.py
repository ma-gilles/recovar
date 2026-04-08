"""Production-path score parity (per spec Section 13.1).

Asserts that the score actually assembled inside `E_with_precompute`
from the three production pieces

    compute_dot_products_eqx        (cross term + ||y||^2 / sigma^2)
    compute_CTFed_proj_norms_eqx    (||CTF . proj||^2 / sigma^2)
    compute_bHb_terms               (heterogeneous correction)

agrees with a dense, brute-force reference for `-2 log p_het(y | g)`
up to a g-independent additive constant.

This test extends the audit P1 result (which verified
`compute_bHb_terms` in isolation) to the *assembled* production score
that the new posterior helper will be required to match. It tests
existing code only — no new module needed.

The brute-force reference uses a single zero translation to keep the
phase-shift convention out of the comparison; translation handling
is exercised by tests/unit/test_em_heterogeneity_numerical.py and by
the existing P1 translation-invariance test.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.em.core as em_core
import recovar.em.heterogeneity as hetero

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Identity CTF / process_images callbacks (float64)
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (4, 4)
IMG_SZ = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]


def _identity_ctf(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(batch, apply_image_mask=False):
    return batch


# ---------------------------------------------------------------------------
# Minimal ForwardModelConfig stand-in
#
# E_with_precompute calls compute_dot_products_eqx and
# compute_CTFed_proj_norms_eqx with a ForwardModelConfig built from a
# CryoEMDataset. For unit testing we need a tiny stand-in that exposes
# the right attributes (compute_ctf, process_fn, image_shape) so that
# the @eqx.filter_jit closures can run on synthetic inputs.
# ---------------------------------------------------------------------------


import equinox as eqx


class _TinyConfig(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    _ctf_callable: callable = eqx.field(static=True)
    _process_callable: callable = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, ctf_params):
        return self._ctf_callable(ctf_params, self.image_shape, self.voxel_size)

    def process_fn(self, batch, apply_image_mask=False):
        return self._process_callable(batch, apply_image_mask=apply_image_mask)


def _make_tiny_config():
    return _TinyConfig(
        image_shape=IMAGE_SHAPE,
        _ctf_callable=_identity_ctf,
        _process_callable=_identity_process,
        voxel_size=1.0,
    )


# ---------------------------------------------------------------------------
# Synthetic inputs (real-valued, cast to complex128, identity CTF)
# ---------------------------------------------------------------------------


def _make_inputs(rng, n_rot, n_pc, n_img):
    mean_proj = rng.standard_normal((n_rot, IMG_SZ)).astype(np.float64).astype(np.complex128)
    u_proj = (0.1 * rng.standard_normal((n_rot, n_pc, IMG_SZ))).astype(np.float64).astype(np.complex128)
    s = (0.5 + rng.uniform(size=n_pc)).astype(np.float64)
    batch = rng.standard_normal((n_img, IMG_SZ)).astype(np.float64).astype(np.complex128)
    return mean_proj, u_proj, s, batch


# ---------------------------------------------------------------------------
# Brute-force reference: -2 log p_het(y | g) up to g-independent const
# ---------------------------------------------------------------------------


def _brute_force_neg_2_log_p_het(mean_proj, u_proj, s, batch, sigma2):
    """Compute -2 log p_het(y | g) - C, where C is g-independent.

    Setup: identity CTF, no translation, scalar Sigma = sigma2 * I.

    Per spec Section 4.5, with Sigma_y = Sigma + U diag(s) U^H,

        -2 log p_het(y|g) = (y - mu_g)^H Sigma_y^{-1} (y - mu_g)
                            + log det Sigma_y + N log(2 pi)
                          = ||y - mu_g||^2 / sigma^2
                            - b^H H^{-1} b + log det H
                            + (g-independent constant: log det Sigma + sum log s + ...)

    The function returns the first two g-dependent terms (homogeneous
    residual minus the heterogeneous correction). Per-image
    normalization in the test then removes any remaining additive
    constant.
    """
    mean_proj = np.asarray(mean_proj, dtype=np.complex128)
    u_proj = np.asarray(u_proj, dtype=np.complex128)
    s = np.asarray(s, dtype=np.float64)
    batch = np.asarray(batch, dtype=np.complex128)

    n_img = batch.shape[0]
    n_rot = mean_proj.shape[0]
    inv_s_diag = np.diag(1.0 / s)

    out = np.empty((n_img, n_rot), dtype=np.float64)
    for r in range(n_rot):
        U = u_proj[r].T  # (image_size, n_pc)
        UhU = U.conj().T @ U
        H = inv_s_diag + UhU / sigma2
        sign, logdet_H = np.linalg.slogdet(H)
        assert sign > 0
        mu = mean_proj[r]
        for i in range(n_img):
            resid = batch[i] - mu
            homog_term = (resid.conj() @ resid).real / sigma2
            b = (U.conj().T @ resid) / sigma2
            bHb = (b.conj() @ np.linalg.solve(H, b)).real
            # heterogeneous score = homog_term - (bHb - log det H)
            #                     = homog_term - bHb + log det H
            out[i, r] = homog_term - bHb + logdet_H
    return out


# ---------------------------------------------------------------------------
# Production-path assembly (mirrors E_with_precompute lines 99-175)
# ---------------------------------------------------------------------------


def _assemble_production_residual(mean_proj, u_proj, s, batch):
    """Reproduce the score assembly inside E_with_precompute, exactly.

    Returns
    -------
    residual : (n_img, n_rot, 1) float64
        The value E_with_precompute writes to `residuals` after the
        bHb subtraction and the proj-norm addition. By construction
        this equals -2 log p_het(y|g) up to a g-independent constant.
    """
    config = _make_tiny_config()
    n_img = batch.shape[0]
    n_rot = mean_proj.shape[0]
    trans = jnp.zeros((1, 2), dtype=jnp.float64)
    ctf_params = jnp.zeros((n_img, 9), dtype=jnp.float64)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float64)

    # Step 1 (e_step.py:99): residuals = compute_dot_products_eqx
    residuals = em_core.compute_dot_products_eqx(
        config,
        jnp.asarray(mean_proj),
        jnp.asarray(batch),
        trans,
        ctf_params,
        noise_var,
    )
    # Step 2 (e_step.py:139): residuals -= compute_bHb_terms
    bHb = hetero.compute_bHb_terms(
        jnp.asarray(mean_proj),
        jnp.asarray(u_proj),
        jnp.asarray(s),
        jnp.asarray(batch),
        trans,
        ctf_params,
        _identity_ctf,
        noise_var,
        1.0,
        IMAGE_SHAPE,
        _identity_process,
    )
    residuals = residuals - bHb
    # Step 3 (e_step.py:155 then 167-175): production squares the
    # complex projections to |proj|^2 before calling
    # compute_CTFed_proj_norms_eqx, so that
    #   CTFs @ |proj|^2.T   ==   sum_k (CTF[i,k]^2 / sigma^2[k]) * |proj[r,k]|^2
    #                       ==   ||CTF . proj_r||^2 / sigma^2.
    # Without the square, the matmul collapses to a complex sum that
    # has nothing to do with the projection norm.
    proj_squared = jnp.abs(jnp.asarray(mean_proj)) ** 2
    proj_norms = em_core.compute_CTFed_proj_norms_eqx(
        config,
        proj_squared,
        ctf_params,
        noise_var,
    )
    residuals = residuals + proj_norms[..., None]
    return np.asarray(residuals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_production_assembly_matches_brute_force_neg_2_log_p_het():
    """Production score assembly equals -2 log p_het(y|g) up to a
    g-independent constant.

    After per-image mean-removal both quantities should agree to
    machine precision in float64.
    """
    rng = np.random.default_rng(0xC0DECAFE)
    n_rot, n_pc, n_img = 4, 2, 5
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    # Production path
    prod = _assemble_production_residual(mean_proj, u_proj, s, batch)
    assert prod.shape == (n_img, n_rot, 1)
    prod = prod[..., 0]  # drop the n_trans=1 axis

    # Brute-force reference
    ref = _brute_force_neg_2_log_p_het(mean_proj, u_proj, s, batch, sigma2=1.0)

    # Subtract per-image mean from both to remove any g-independent
    # constant (the production assembly drops `log det Sigma`,
    # `sum log s`, `N log 2pi`; the brute-force keeps `log det H`
    # but no additional terms).
    prod_centered = prod - prod.mean(axis=1, keepdims=True)
    ref_centered = ref - ref.mean(axis=1, keepdims=True)

    np.testing.assert_allclose(prod_centered, ref_centered, rtol=1e-10, atol=1e-12)


def test_production_assembly_matches_brute_force_with_higher_q():
    """Same parity check at q=4, n_rot=3, n_img=6."""
    rng = np.random.default_rng(0xDEADBEEF)
    n_rot, n_pc, n_img = 3, 4, 6
    mean_proj, u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)

    prod = _assemble_production_residual(mean_proj, u_proj, s, batch)[..., 0]
    ref = _brute_force_neg_2_log_p_het(mean_proj, u_proj, s, batch, sigma2=1.0)

    prod_centered = prod - prod.mean(axis=1, keepdims=True)
    ref_centered = ref - ref.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(prod_centered, ref_centered, rtol=1e-10, atol=1e-12)


def test_production_assembly_at_zero_u_matches_homogeneous():
    """With U = 0, the assembled production residual must equal the
    homogeneous squared residual ||y - mu_g||^2 / sigma^2 plus the
    `sum log s` term that compute_bHb_terms emits when U=0
    (verified by audit P1: U=0 returns sum log s).

    After per-image normalization the constant cancels and we should
    recover exactly the homogeneous residual.
    """
    rng = np.random.default_rng(0xFEEDFACE)
    n_rot, n_pc, n_img = 3, 2, 4
    mean_proj, _u_proj, s, batch = _make_inputs(rng, n_rot, n_pc, n_img)
    u_zero = np.zeros((n_rot, n_pc, IMG_SZ), dtype=np.complex128)

    prod = _assemble_production_residual(mean_proj, u_zero, s, batch)[..., 0]

    # Homogeneous reference: ||y - mu_g||^2 / sigma^2
    homog_ref = np.empty((n_img, n_rot), dtype=np.float64)
    for i in range(n_img):
        for r in range(n_rot):
            d = batch[i] - mean_proj[r]
            homog_ref[i, r] = (d.conj() @ d).real

    prod_centered = prod - prod.mean(axis=1, keepdims=True)
    homog_centered = homog_ref - homog_ref.mean(axis=1, keepdims=True)

    np.testing.assert_allclose(prod_centered, homog_centered, rtol=1e-10, atol=1e-12)
