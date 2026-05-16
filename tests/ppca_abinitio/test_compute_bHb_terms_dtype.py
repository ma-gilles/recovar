"""Prerequisite test P2 for the PPCA-ab-initio v0 plan.

Audits dtype propagation through recovar.em.heterogeneity's
low-rank score path. Per the audit in
docs/math/plan_ppca_abinitio_v0.md (Section "Audit 2"), the entire
production E-step path runs at complex64 / float32, even though
jax_enable_x64 is True globally. This is because:

  - CryoEMDataset.__init__ defaults to dtype=np.complex64
  - E_with_precompute creates `projections` and `u_projections`
    explicitly as np.complex64
  - compute_UPLambdainvPU uses `.real` of those complex64 outer
    products, producing float32

The PPCA-ab-initio v0 spec requires float64 throughout so that the
parity test against an independent dense reference can be tightened
to ~1e-10 (the float32 path can only achieve ~1e-4, which is below
the precision needed to detect the kind of subtle bug we are trying
to catch).

This test does NOT require any code change to pass — it documents
the *current* behavior. If a future change breaks dtype propagation
(e.g. silently downcasting float64 inputs), this test fires.

What the test asserts
---------------------
1. With complex64 / float32 inputs, the function returns float32.
   (Documents the current production behavior.)
2. With complex128 / float64 inputs, the function returns float64
   AND the underlying compute_UPLambdainvPU and
   compute_bLambdainvPU_terms also return float64 — i.e. there is
   no hidden downcast inside the implementation.

If assertion (2) fails, the audit was incomplete and there is a
hidden float32 cast we need to find before writing the new helper.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.em.heterogeneity as hetero

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stand-in CTF / process_images callbacks parameterized by dtype
# ---------------------------------------------------------------------------


def _make_identity_ctf(real_dtype):
    def _identity_ctf(CTF_params, image_shape, voxel_size):
        n = CTF_params.shape[0]
        sz = int(np.prod(image_shape))
        return jnp.ones((n, sz), dtype=real_dtype)

    return _identity_ctf


def _identity_process(batch, apply_image_mask=False):
    return batch


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (4, 4)
IMG_SZ = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]


def _make_inputs(rng, complex_dtype, real_dtype, n_rot=2, n_pc=2, n_img=2):
    mean = (rng.standard_normal((n_rot, IMG_SZ)) + 1j * rng.standard_normal((n_rot, IMG_SZ))).astype(complex_dtype)
    u = (0.05 * (rng.standard_normal((n_rot, n_pc, IMG_SZ)) + 1j * rng.standard_normal((n_rot, n_pc, IMG_SZ)))).astype(
        complex_dtype
    )
    s = jnp.asarray(0.5 + rng.uniform(size=n_pc), dtype=real_dtype)
    batch = (rng.standard_normal((n_img, IMG_SZ)) + 1j * rng.standard_normal((n_img, IMG_SZ))).astype(complex_dtype)
    trans = jnp.zeros((1, 2), dtype=real_dtype)
    ctf_params = jnp.zeros((n_img, 9), dtype=real_dtype)
    noise_var = jnp.ones(IMG_SZ, dtype=real_dtype)
    return (
        jnp.asarray(mean),
        jnp.asarray(u),
        s,
        jnp.asarray(batch),
        trans,
        ctf_params,
        noise_var,
    )


# ---------------------------------------------------------------------------
# (1) Document the current float32 path
# ---------------------------------------------------------------------------


def test_bHb_float32_path_returns_float32():
    """Production-default complex64 / float32 inputs ⇒ float32 output.

    Pins the current behavior. If this changes (e.g. someone fixes the
    dtype path to upcast everywhere), this test will fire and we can
    decide whether the change is intentional.
    """
    rng = np.random.default_rng(0)
    mean, u, s, batch, trans, ctf_params, noise_var = _make_inputs(
        rng, complex_dtype=np.complex64, real_dtype=np.float32
    )
    out = hetero.compute_bHb_terms(
        mean,
        u,
        s,
        batch,
        trans,
        ctf_params,
        _make_identity_ctf(np.float32),
        noise_var,
        1.0,
        IMAGE_SHAPE,
        _identity_process,
    )
    assert out.dtype == jnp.float32, f"expected float32 in production path, got {out.dtype}"


def test_UPLambdainvPU_float32_path_returns_float32():
    """compute_UPLambdainvPU is the source of the float32 cast: `.real` of
    a complex64 product is float32 regardless of CTF / noise dtype."""
    rng = np.random.default_rng(1)
    u = (rng.standard_normal((2, 2, IMG_SZ)) + 1j * rng.standard_normal((2, 2, IMG_SZ))).astype(np.complex64)
    CTF = jnp.ones((2, IMG_SZ), dtype=jnp.float32)
    noise = jnp.ones((2, IMG_SZ), dtype=jnp.float32)
    H = hetero.compute_UPLambdainvPU(jnp.asarray(u), CTF, noise)
    assert H.dtype == jnp.float32, f"expected float32 H, got {H.dtype}"


# ---------------------------------------------------------------------------
# (2) Float64 propagation — the assertion the PPCA spec depends on
# ---------------------------------------------------------------------------


def test_bHb_float64_inputs_yield_float64_output():
    """The PPCA-ab-initio v0 spec requires that, given complex128 / float64
    inputs, compute_bHb_terms returns float64 with no hidden downcast.

    If this fails, the audit was incomplete and we have to find the
    silent cast before building anything on top of this function.
    """
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("jax x64 not enabled in this run")

    rng = np.random.default_rng(2)
    mean, u, s, batch, trans, ctf_params, noise_var = _make_inputs(
        rng, complex_dtype=np.complex128, real_dtype=np.float64
    )
    out = hetero.compute_bHb_terms(
        mean,
        u,
        s,
        batch,
        trans,
        ctf_params,
        _make_identity_ctf(np.float64),
        noise_var,
        1.0,
        IMAGE_SHAPE,
        _identity_process,
    )
    assert out.dtype == jnp.float64, (
        f"expected float64 output for float64 inputs, got {out.dtype}. "
        "There is a silent downcast somewhere in the heterogeneity score path. "
        "Locate it before building PPCA-ab-initio on top of this function."
    )


def test_UPLambdainvPU_float64_inputs_yield_float64():
    """The complex128 → .real cast must produce float64, not float32."""
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("jax x64 not enabled in this run")
    rng = np.random.default_rng(3)
    u = (rng.standard_normal((2, 2, IMG_SZ)) + 1j * rng.standard_normal((2, 2, IMG_SZ))).astype(np.complex128)
    CTF = jnp.ones((2, IMG_SZ), dtype=jnp.float64)
    noise = jnp.ones((2, IMG_SZ), dtype=jnp.float64)
    H = hetero.compute_UPLambdainvPU(jnp.asarray(u), CTF, noise)
    assert H.dtype == jnp.float64, f"expected float64 H from float64 inputs, got {H.dtype}"


def test_bLambdainvPU_float64_inputs_yield_float64():
    """compute_bLambdainvPU_terms must also propagate float64."""
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("jax x64 not enabled in this run")
    rng = np.random.default_rng(4)
    n_rot, n_pc, n_img, n_trans = 2, 2, 2, 1
    mean = (rng.standard_normal((n_rot, IMG_SZ)) + 1j * rng.standard_normal((n_rot, IMG_SZ))).astype(np.complex128)
    u = (rng.standard_normal((n_rot, n_pc, IMG_SZ)) + 1j * rng.standard_normal((n_rot, n_pc, IMG_SZ))).astype(
        np.complex128
    )
    images = (rng.standard_normal((n_img, IMG_SZ)) + 1j * rng.standard_normal((n_img, IMG_SZ))).astype(np.complex128)
    trans = jnp.zeros((n_trans, 2), dtype=jnp.float64)
    CTF = jnp.ones((n_img, IMG_SZ), dtype=jnp.float64)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float64)
    b = hetero.compute_bLambdainvPU_terms(
        jnp.asarray(mean), jnp.asarray(u), jnp.asarray(images), trans, CTF, noise_var, IMAGE_SHAPE
    )
    assert b.dtype == jnp.float64, f"expected float64 b, got {b.dtype}"
