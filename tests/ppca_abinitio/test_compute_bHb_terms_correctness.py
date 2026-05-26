"""Prerequisite test P1 for the PPCA-ab-initio v0 plan.

Verifies that recovar.em.heterogeneity.compute_bHb_terms — the
low-rank score correction used inside E_with_precompute — actually
computes the formula it claims to compute.

Per the audit in docs/math/plan_ppca_abinitio_v0.md (Section "Audit
1"), the function has never been numerically verified. Its existing
tests in tests/unit/test_em_heterogeneity_numerical.py only check
shape, finiteness, and dtype. The PPCA-ab-initio plan pins to this
function as its parity oracle, so we must verify its correctness
*before* any new code is written on top of it.

The contract being tested
-------------------------
With Σ_y = Σ + U diag(s) U^H and  b = U^H Σ^{-1} (y - μ_g),
H = diag(1/s) + U^H Σ^{-1} U,  the function must return

    bHb_correction[i, r, t] = b^H H^{-1} b - log det H

up to floating-point error. The caller (E_with_precompute) subtracts
this from the homogeneous squared residual (y-μ_g)^H Σ^{-1} (y-μ_g)
to obtain -2 log p_het(y | g) up to a g-independent constant.

Test setup
----------
We feed real-valued inputs cast to complex128 / float64. For real
inputs the conjugations and `.real` calls inside the function are
no-ops, and the linear algebra reduces to a clean real-valued
problem we can compute densely in float64 numpy. This isolates the
math from the Hermitian-symmetry assumption that production cryo-EM
relies on.

We deliberately run the function in float64 mode (verified
end-to-end by tests/ppca_abinitio/test_compute_bHb_terms_dtype.py).
The production E-step runs the same function at float32 — that path
is documented separately by the dtype test, and its precision floor
(~1e-4) is exactly why the PPCA-ab-initio v0 spec mandates float64
for the new helper.

Identity CTF and unit noise variance further isolate the math from
the CTF/noise weighting (which is exercised by the existing
test_bLambdainvPU_noise_scaling_mean_term test).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.em.heterogeneity as hetero

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stand-in CTF / process_images callbacks (mirror tests/unit fixtures)
# ---------------------------------------------------------------------------


def _identity_ctf_f64(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(batch, apply_image_mask=False):
    return batch


# ---------------------------------------------------------------------------
# Brute-force reference implementation in float64 numpy
# ---------------------------------------------------------------------------


def _brute_force_bHb(mean_proj, u_proj, s, images, sigma2):
    """Compute  b^H H^{-1} b - log det H  for every (image, rotation).

    Parameters
    ----------
    mean_proj : (n_rot, image_size) complex
    u_proj    : (n_rot, n_pc, image_size) complex
    s         : (n_pc,) real
    images    : (n_img, image_size) complex
    sigma2    : scalar real (assumes Σ = sigma2 · I)

    Returns
    -------
    out : (n_img, n_rot) real, dtype float64
    """
    mean_proj = np.asarray(mean_proj, dtype=np.complex128)
    u_proj = np.asarray(u_proj, dtype=np.complex128)
    s = np.asarray(s, dtype=np.float64)
    images = np.asarray(images, dtype=np.complex128)

    n_img = images.shape[0]
    n_rot = mean_proj.shape[0]
    inv_s_diag = np.diag(1.0 / s)

    out = np.empty((n_img, n_rot), dtype=np.float64)
    for r in range(n_rot):
        # U_r columns live along the last axis of u_proj[r]
        U = u_proj[r].T  # (image_size, n_pc)
        UhU = U.conj().T @ U  # (n_pc, n_pc) Hermitian
        H = inv_s_diag + UhU / sigma2
        sign, logdet_H = np.linalg.slogdet(H)
        assert sign > 0, f"H not positive-definite for rot {r}"

        mu = mean_proj[r]
        for i in range(n_img):
            resid = images[i] - mu
            b = (U.conj().T @ resid) / sigma2  # (n_pc,)
            Hinv_b = np.linalg.solve(H, b)
            bHb = (b.conj() @ Hinv_b).real
            out[i, r] = bHb - logdet_H
    return out


# ---------------------------------------------------------------------------
# Test inputs — small, real-valued, identity CTF, σ² = 1
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (4, 4)  # 16-pixel "images"
IMG_SZ = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]


def _make_real_inputs(rng, n_rot, n_pc, n_img):
    """Real-valued inputs cast into the complex128 dtype the function expects.

    With purely real inputs the function's internal `.real` calls are
    no-ops, so the test is a clean check on the linear-algebra formula.
    """
    mean_proj_real = rng.standard_normal((n_rot, IMG_SZ)).astype(np.float64)
    u_proj_real = (0.1 * rng.standard_normal((n_rot, n_pc, IMG_SZ))).astype(np.float64)
    s = (0.5 + rng.uniform(size=n_pc)).astype(np.float64)  # in [0.5, 1.5]
    images_real = rng.standard_normal((n_img, IMG_SZ)).astype(np.float64)

    mean_proj_c = mean_proj_real.astype(np.complex128)
    u_proj_c = u_proj_real.astype(np.complex128)
    images_c = images_real.astype(np.complex128)
    return mean_proj_c, u_proj_c, s, images_c, mean_proj_real, u_proj_real, images_real


def _call_function(mean_proj, u_proj, s, batch, n_trans):
    """Call compute_bHb_terms in float64 mode with identity CTF, σ²=1, n_trans
    translations at origin.

    Float64 propagation through this function is verified by
    test_compute_bHb_terms_dtype.py — if that test starts failing,
    these correctness tolerances will become unreachable.
    """
    n_img = batch.shape[0]
    trans = jnp.zeros((n_trans, 2), dtype=jnp.float64)
    ctf_params = jnp.zeros((n_img, 9), dtype=jnp.float64)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float64)
    return hetero.compute_bHb_terms(
        jnp.asarray(mean_proj),
        jnp.asarray(u_proj),
        jnp.asarray(s, dtype=jnp.float64),
        jnp.asarray(batch),
        trans,
        ctf_params,
        _identity_ctf_f64,
        noise_var,
        1.0,
        IMAGE_SHAPE,
        _identity_process,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bHb_matches_brute_force_single_translation():
    """Single (0,0) translation: function output must match brute-force formula."""
    rng = np.random.default_rng(0xC0FFEE)
    n_rot, n_pc, n_img = 3, 2, 4
    mean_c, u_c, s, batch_c, mean_r, u_r, batch_r = _make_real_inputs(rng, n_rot, n_pc, n_img)

    # Reference: brute force in float64
    expected = _brute_force_bHb(mean_r, u_r, s, batch_r, sigma2=1.0)

    # Function under test (float32 internally)
    out = _call_function(mean_c, u_c, s, batch_c, n_trans=1)
    out = np.asarray(out)
    assert out.shape == (n_img, n_rot, 1), f"unexpected shape {out.shape}"

    actual = out[..., 0]  # drop the n_trans=1 axis
    # Float64 path: tight tolerance well above the float64 precision floor
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_bHb_translation_invariance_at_origin():
    """All translations are (0,0): all n_trans entries should be identical
    (and equal the brute-force, single-translation reference)."""
    rng = np.random.default_rng(0xBA5EBA11)
    n_rot, n_pc, n_img, n_trans = 2, 3, 3, 4
    mean_c, u_c, s, batch_c, mean_r, u_r, batch_r = _make_real_inputs(rng, n_rot, n_pc, n_img)

    expected = _brute_force_bHb(mean_r, u_r, s, batch_r, sigma2=1.0)

    out = np.asarray(_call_function(mean_c, u_c, s, batch_c, n_trans=n_trans))
    assert out.shape == (n_img, n_rot, n_trans)

    for t in range(n_trans):
        np.testing.assert_allclose(out[..., t], expected, rtol=1e-10, atol=1e-12)


def test_bHb_changes_correctly_with_s():
    """Sanity: shrinking s should *increase* log det H (since H ≈ diag(1/s))
    and therefore *decrease* the returned bHb-correction value, all else equal.

    This pins the sign of the log-det term, which the source variable
    name `log_det_H` actively misleads about (it is actually -log det H
    in the local code; see the audit)."""
    rng = np.random.default_rng(7)
    mean_c, u_c, s, batch_c, *_ = _make_real_inputs(rng, n_rot=2, n_pc=2, n_img=2)

    out_ref = np.asarray(_call_function(mean_c, u_c, s, batch_c, n_trans=1))
    out_small_s = np.asarray(_call_function(mean_c, u_c, 0.1 * s, batch_c, n_trans=1))

    # Smaller s → 1/s larger → det H larger → log det H larger → returned
    # (bHb - log det H) smaller. Verify on average across (i, r).
    assert out_small_s.mean() < out_ref.mean(), (
        "Shrinking s should reduce the bHb-correction term on average; "
        "if this fails, the sign of the log-det contribution is wrong."
    )


def test_bHb_zero_u_returns_minus_log_det_diag_inv_s():
    """With U = 0, b = 0 and H = diag(1/s), so the function should return
    -log det H = -sum log(1/s) = sum log s, independent of (i, r, t)."""
    rng = np.random.default_rng(13)
    n_rot, n_pc, n_img = 2, 3, 2
    mean_c, _u_c, s, batch_c, *_ = _make_real_inputs(rng, n_rot, n_pc, n_img)
    u_zero = np.zeros((n_rot, n_pc, IMG_SZ), dtype=np.complex128)

    out = np.asarray(_call_function(mean_c, u_zero, s, batch_c, n_trans=1))
    expected_scalar = float(np.sum(np.log(s.astype(np.float64))))
    np.testing.assert_allclose(out, expected_scalar, rtol=1e-12, atol=1e-14)
