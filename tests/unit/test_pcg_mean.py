"""Tests for PCG mean estimation with support mask."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.reconstruction.pcg_mean import _matvec, pcg_mean

pytestmark = pytest.mark.unit


def _make_test_data(gs=32, d_ratio=1000.0, mask_radius=0.5, lam=0.0):
    """Build synthetic mean-estimation problem with known ground truth."""
    x = np.linspace(-1, 1, gs)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    vol_gt = np.exp(-(X**2 + Y**2 + Z**2) / 0.15).astype(np.float32)
    mask = ((X**2 + Y**2 + Z**2) < mask_radius**2).astype(np.float32)

    freq = np.fft.fftshift(np.fft.fftfreq(gs))
    Kx, Ky, Kz = np.meshgrid(freq, freq, freq, indexing="ij")
    k = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    d = (d_ratio * np.exp(-200 * k**2) + 1.0).astype(np.float32)

    vol_gt_ft = ftu.get_dft3(jnp.array(vol_gt))
    c = jnp.array(d) * vol_gt_ft

    w_support = jnp.array(mask) if lam > 0 else None

    return dict(d=jnp.array(d), c=c, mask=jnp.array(mask), vol_gt=vol_gt, w_support=w_support, lam=lam)


def test_pcg_converges_to_wiener():
    """Without mask, PCG should recover the Wiener-filtered volume."""
    gs = 32
    dd = _make_test_data(gs=gs, d_ratio=100.0, mask_radius=2.0)  # mask covers everything
    x, res = pcg_mean(dd["d"], dd["c"], jnp.ones((gs, gs, gs)), maxiter=50, tol=1e-6)
    wiener = ftu.get_idft3(dd["c"] / jnp.maximum(dd["d"], 1e-6)).real
    err = float(jnp.max(jnp.abs(x - wiener)))
    assert err < 1e-4, f"PCG should match Wiener without mask, err={err}"


def test_pcg_converges():
    """PCG should converge to tolerance."""
    dd = _make_test_data(gs=32, d_ratio=1000.0)
    _, res = pcg_mean(dd["d"], dd["c"], dd["mask"], maxiter=100, tol=1e-5, precondition=True)
    assert res[-1] < 1e-5, f"PCG did not converge: rr={res[-1]}"


def test_preconditioner_reduces_iterations():
    """Preconditioned CG should need fewer iterations than unpreconditioned."""
    dd = _make_test_data(gs=32, d_ratio=1000.0)
    _, res_nop = pcg_mean(dd["d"], dd["c"], dd["mask"], maxiter=200, tol=1e-5, precondition=False)
    _, res_pcg = pcg_mean(dd["d"], dd["c"], dd["mask"], maxiter=200, tol=1e-5, precondition=True)
    assert len(res_pcg) < len(res_nop), f"Precond {len(res_pcg)} >= no-precond {len(res_nop)}"


def test_sandwich_preconditioner_with_lambda():
    """Sandwich M_J^{-1/2} M_0^{-1} M_J^{-1/2} with lambda converges faster."""
    dd = _make_test_data(gs=32, d_ratio=1000.0, lam=1.0)
    _, res_m0 = pcg_mean(dd["d"], dd["c"], dd["mask"], lam=0.0, maxiter=100, tol=1e-5, precondition=True)
    _, res_mj = pcg_mean(
        dd["d"], dd["c"], dd["mask"], lam=dd["lam"], w_support=dd["w_support"], maxiter=100, tol=1e-5, precondition=True
    )
    assert len(res_mj) <= len(res_m0), f"Sandwich {len(res_mj)} > M0-only {len(res_m0)}"


def test_warmstart():
    """Warmstart from previous solution should converge in very few iterations."""
    dd = _make_test_data(gs=32, d_ratio=1000.0)
    x_cold, _ = pcg_mean(dd["d"], dd["c"], dd["mask"], maxiter=100, tol=1e-6, precondition=True)
    _, res_warm = pcg_mean(dd["d"], dd["c"], dd["mask"], x0=x_cold, maxiter=20, tol=1e-6, precondition=True)
    assert len(res_warm) <= 3, f"Warmstart needed {len(res_warm)} iters (expected <=3)"


def test_matvec_is_spd():
    """H_Omega should be symmetric positive definite on the support."""
    dd = _make_test_data(gs=16, d_ratio=100.0)
    rng = np.random.default_rng(42)
    u = jnp.array(rng.normal(size=(16, 16, 16)).astype(np.float32)) * dd["mask"]
    v = jnp.array(rng.normal(size=(16, 16, 16)).astype(np.float32)) * dd["mask"]

    Hu = _matvec(u, dd["d"], dd["mask"])
    Hv = _matvec(v, dd["d"], dd["mask"])

    # Symmetry: <u, Hv> = <Hu, v>
    uHv = float(jnp.sum(u * Hv))
    Huv = float(jnp.sum(Hu * v))
    np.testing.assert_allclose(uHv, Huv, rtol=1e-5, err_msg="H not symmetric")

    # Positive definite: <u, Hu> > 0
    uHu = float(jnp.sum(u * Hu))
    assert uHu > 0, f"H not positive definite: <u,Hu>={uHu}"


def test_solution_respects_mask():
    """PCG solution should be zero outside the support mask."""
    dd = _make_test_data(gs=32, d_ratio=100.0, mask_radius=0.3)
    x, _ = pcg_mean(dd["d"], dd["c"], dd["mask"], maxiter=50, tol=1e-5)
    outside = float(jnp.max(jnp.abs(x * (1 - dd["mask"]))))
    assert outside < 1e-10, f"Solution nonzero outside mask: {outside}"


def test_f64_tighter_convergence():
    """Float64 achieves tighter residual than float32."""
    import os

    os.environ["JAX_ENABLE_X64"] = "1"
    jax.config.update("jax_enable_x64", True)

    dd = _make_test_data(gs=16, d_ratio=100.0)
    d64 = dd["d"].astype(jnp.float64)
    c64 = dd["c"].astype(jnp.complex128)
    mask64 = dd["mask"].astype(jnp.float64)

    _, res32 = pcg_mean(dd["d"], dd["c"], dd["mask"], maxiter=50, tol=1e-8, precondition=True)
    _, res64 = pcg_mean(d64, c64, mask64, maxiter=50, tol=1e-12, precondition=True)

    assert res64[-1] < res32[-1] * 1e-3, f"f64 ({res64[-1]:.2e}) should be >1000× tighter than f32 ({res32[-1]:.2e})"
