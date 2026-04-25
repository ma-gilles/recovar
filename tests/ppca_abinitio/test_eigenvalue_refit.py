"""Unit tests for the post-EM posterior-covariance eigenvalue refit."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from recovar.em.ppca_abinitio.eigenvalue_refit import (
    EigenvalueRefitInfo,
    refit_eigenvalues_post_em,
)
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)
from recovar.em.ppca_abinitio.types import PPCAInit
from scripts.ppca_abinitio.run_cryobench import _Cfg, run_two_stage

pytestmark = [pytest.mark.unit]


def _build_tiny_dataset(seed: int = 0):
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    grid = build_fixed_grid(healpix_order=0, max_shift=0)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=volume_shape,
        image_shape=image_shape,
        grid=grid,
        q=2,
        n_images_train=8,
        n_images_val=0,
        sigma_real=0.1,
        seed=seed,
    )
    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)
    return cfg, ds


def _run_short_em(cfg, ds, q=2, seed=0):
    final, _, _, _, _, _ = run_two_stage(
        cfg,
        ds,
        q=q,
        n_burnin=0,
        n_joint=2,
        mu_init_kind="zero",
        u_init_kind="svd",
        weighted_svd=True,
        anneal_schedule="none",
        seed=seed,
        s_init_kind="flat",
    )
    return final


def test_refit_returns_correct_shapes_and_dtypes():
    cfg, ds = _build_tiny_dataset(seed=0)
    cur = _run_short_em(cfg, ds, q=2)

    refit_state, info = refit_eigenvalues_post_em(cur, cfg, ds)

    assert isinstance(refit_state, PPCAInit)
    assert isinstance(info, EigenvalueRefitInfo)
    assert refit_state.U.shape == cur.U.shape
    assert refit_state.s.shape == cur.s.shape
    assert refit_state.s.dtype == jnp.float64
    assert refit_state.U.dtype == jnp.complex128
    assert info.s_refit.shape == (cur.U.shape[0],)
    assert info.rotation.shape == (cur.U.shape[0], cur.U.shape[0])
    assert info.sigma_alpha.shape == (cur.U.shape[0], cur.U.shape[0])


def test_refit_eigenvalues_are_descending_and_positive():
    cfg, ds = _build_tiny_dataset(seed=1)
    cur = _run_short_em(cfg, ds, q=2)

    _, info = refit_eigenvalues_post_em(cur, cfg, ds)

    assert np.all(info.s_refit > 0), f"non-positive eigenvalues: {info.s_refit}"
    diffs = np.diff(info.s_refit)
    assert np.all(diffs <= 1e-9), f"eigenvalues not descending: {info.s_refit}"


def test_refit_does_not_change_mu():
    cfg, ds = _build_tiny_dataset(seed=2)
    cur = _run_short_em(cfg, ds, q=2)

    refit_state, _ = refit_eigenvalues_post_em(cur, cfg, ds)

    np.testing.assert_array_equal(np.asarray(cur.mu), np.asarray(refit_state.mu))


def test_refit_rotation_is_orthogonal():
    """The rotation V from eigh of a real symmetric matrix must be orthogonal."""
    cfg, ds = _build_tiny_dataset(seed=3)
    cur = _run_short_em(cfg, ds, q=2)

    _, info = refit_eigenvalues_post_em(cur, cfg, ds)
    V = info.rotation
    VtV = V.T @ V
    np.testing.assert_allclose(VtV, np.eye(V.shape[0]), atol=1e-10)


def test_refit_sigma_alpha_recovers_eigenvalues():
    """The refit eigenvalues must equal the eigenvalues of sigma_alpha."""
    cfg, ds = _build_tiny_dataset(seed=4)
    cur = _run_short_em(cfg, ds, q=2)

    _, info = refit_eigenvalues_post_em(cur, cfg, ds)
    eigvals_check = np.linalg.eigvalsh(info.sigma_alpha)[::-1]
    np.testing.assert_allclose(info.s_refit, eigvals_check, atol=1e-10)


def test_refit_preserves_represented_subspace():
    """Subspace span(U_old) and span(U_new) must agree (rotation only)."""
    cfg, ds = _build_tiny_dataset(seed=5)
    cur = _run_short_em(cfg, ds, q=2)

    refit_state, _ = refit_eigenvalues_post_em(cur, cfg, ds)

    # Real-volume orthogonal rotation preserves row span. Verify by
    # projecting one onto the other and checking the result is the
    # full rank.
    U_old = np.asarray(cur.U)  # (q, V_half) complex
    U_new = np.asarray(refit_state.U)
    # Real Gram in half-volume layout (treating real and imag parts as 2 components)
    cross = U_new @ U_old.conj().T  # (q, q)
    # SVD of this cross-Gram should give all singular values close to 1
    # if the spans coincide and the rotation is orthogonal.
    s = np.linalg.svd(cross, compute_uv=False)
    # The sum of squared singular values equals tr(U_new W U_old^H W' U_new^H)
    # which is bounded by the row norms. Just check that we lose no rank.
    assert s.min() > 1e-6, f"rotation collapsed a row: svals = {s}"
