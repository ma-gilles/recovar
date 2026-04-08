"""Tests for the learning-stage metrics added to metrics.py.

Pins `fourier_relative_error_mu`, `oracle_fsc_gt`,
`projector_frobenius_error`, `principal_angles_deg`, and the two
embedding-error helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.ppca_abinitio.half_volume import (
    make_half_volume_weights,
    real_volume_to_half,
)
from recovar.em.ppca_abinitio.metrics import (
    embedding_error_marginal,
    embedding_error_oracle,
    fourier_relative_error_mu,
    oracle_fsc_gt,
    principal_angles_deg,
    projector_frobenius_error,
)

pytestmark = pytest.mark.unit


VOLUME_SHAPE = (8, 8, 8)
N_FULL = 8 * 8 * 8


def _real_to_half(real_vol):
    return real_volume_to_half(jnp.asarray(real_vol), VOLUME_SHAPE)


# ---------------------------------------------------------------------------
# fourier_relative_error_mu
# ---------------------------------------------------------------------------


def test_fre_zero_when_estimates_match():
    rng = np.random.default_rng(0)
    mu_real = rng.standard_normal(VOLUME_SHAPE).astype(np.float64)
    mu_half = _real_to_half(mu_real)
    err = fourier_relative_error_mu(mu_half, mu_half)
    assert err == pytest.approx(0.0, abs=1e-12)


def test_fre_with_weights_matches_real_space_relative_error():
    rng = np.random.default_rng(1)
    mu_true_real = rng.standard_normal(VOLUME_SHAPE).astype(np.float64)
    perturbation = 0.3 * rng.standard_normal(VOLUME_SHAPE).astype(np.float64)
    mu_est_real = mu_true_real + perturbation

    mu_true_half = _real_to_half(mu_true_real)
    mu_est_half = _real_to_half(mu_est_real)
    weights = make_half_volume_weights(VOLUME_SHAPE)

    err_weighted = fourier_relative_error_mu(mu_est_half, mu_true_half, weights_half=weights)

    # In real space:
    real_err = float(np.linalg.norm(mu_est_real - mu_true_real) / np.linalg.norm(mu_true_real))
    np.testing.assert_allclose(err_weighted, real_err, rtol=1e-10)


def test_fre_unweighted_is_finite():
    rng = np.random.default_rng(2)
    mu_t = _real_to_half(rng.standard_normal(VOLUME_SHAPE).astype(np.float64))
    mu_e = _real_to_half(rng.standard_normal(VOLUME_SHAPE).astype(np.float64))
    err = fourier_relative_error_mu(mu_e, mu_t)
    assert np.isfinite(err)
    assert err > 0


# ---------------------------------------------------------------------------
# oracle_fsc_gt
# ---------------------------------------------------------------------------


def test_oracle_fsc_self_is_one():
    rng = np.random.default_rng(0)
    mu_half = _real_to_half(rng.standard_normal(VOLUME_SHAPE).astype(np.float64))
    fsc = oracle_fsc_gt(mu_half, mu_half, VOLUME_SHAPE)
    # Self-FSC should be 1 in every shell (or 0 in empty shells)
    nonzero_shells = fsc[fsc != 0]
    np.testing.assert_allclose(nonzero_shells, 1.0, rtol=1e-10)


def test_oracle_fsc_independent_volumes_is_close_to_zero():
    rng = np.random.default_rng(1)
    mu_a = _real_to_half(rng.standard_normal(VOLUME_SHAPE).astype(np.float64))
    mu_b = _real_to_half(rng.standard_normal(VOLUME_SHAPE).astype(np.float64))
    fsc = oracle_fsc_gt(mu_a, mu_b, VOLUME_SHAPE)
    # Average should be near zero — use a generous tolerance because
    # at small box sizes the per-shell sample count is tiny.
    assert abs(float(np.mean(fsc))) < 0.5


# ---------------------------------------------------------------------------
# projector_frobenius_error and principal_angles_deg
# ---------------------------------------------------------------------------


def _make_random_real_basis(rng, q):
    """q random orthogonal real volumes (rows of (q, N)) returned in
    half-volume layout."""
    pcs_real = rng.standard_normal((q,) + VOLUME_SHAPE).astype(np.float64)
    flat = pcs_real.reshape(q, -1)
    Q, _ = np.linalg.qr(flat.T)
    flat_orth = Q.T[:q]
    rows = []
    for k in range(q):
        rows.append(_real_to_half(flat_orth[k].reshape(VOLUME_SHAPE)))
    return jnp.stack(rows)


def test_projector_error_is_zero_for_same_basis():
    rng = np.random.default_rng(0)
    U = _make_random_real_basis(rng, q=3)
    err = projector_frobenius_error(U, U, VOLUME_SHAPE)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_projector_error_is_zero_under_orthogonal_recombination():
    """Multiplying U by an orthogonal q×q matrix changes the basis
    but not the row span — the projector error must remain zero."""
    rng = np.random.default_rng(1)
    U = np.asarray(_make_random_real_basis(rng, q=3))
    R_qq = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    U_rot = R_qq @ U  # rotate the basis
    err = projector_frobenius_error(U_rot, U, VOLUME_SHAPE)
    assert err == pytest.approx(0.0, abs=1e-8)


def test_projector_error_is_positive_for_different_spans():
    rng = np.random.default_rng(2)
    U_a = _make_random_real_basis(rng, q=2)
    U_b = _make_random_real_basis(rng, q=2)
    err = projector_frobenius_error(U_a, U_b, VOLUME_SHAPE)
    assert err > 0.5  # for two random q=2 spans in dim 512, should be substantial


def test_principal_angles_zero_for_same_basis():
    rng = np.random.default_rng(0)
    U = _make_random_real_basis(rng, q=3)
    angles = principal_angles_deg(U, U, VOLUME_SHAPE)
    np.testing.assert_allclose(angles, 0.0, atol=1e-6)


def test_principal_angles_in_range():
    rng = np.random.default_rng(1)
    U_a = _make_random_real_basis(rng, q=2)
    U_b = _make_random_real_basis(rng, q=2)
    angles = principal_angles_deg(U_a, U_b, VOLUME_SHAPE)
    assert np.all(angles >= -1e-9)
    assert np.all(angles <= 90.0 + 1e-9)


# ---------------------------------------------------------------------------
# embedding errors
# ---------------------------------------------------------------------------


def test_embedding_error_oracle_is_zero_when_post_mean_equals_alpha():
    n_img, n_rot, n_trans, q = 8, 3, 2, 2
    rng = np.random.default_rng(0)
    alpha_true = rng.standard_normal((n_img, q)).astype(np.float64)
    r_true = rng.integers(0, n_rot, size=n_img)
    t_true = rng.integers(0, n_trans, size=n_img)

    post_mean = np.zeros((n_img, n_rot, n_trans, q), dtype=np.float64)
    for i in range(n_img):
        post_mean[i, r_true[i], t_true[i]] = alpha_true[i]

    err = embedding_error_oracle(post_mean, alpha_true, r_true, t_true)
    assert err == pytest.approx(0.0, abs=1e-12)


def test_embedding_error_oracle_invariant_under_orthogonal_rotation():
    """If we rotate the post_mean entries by a real orthogonal R, the
    Procrustes alignment should undo the rotation and the error
    should not change."""
    n_img, n_rot, n_trans, q = 8, 3, 2, 2
    rng = np.random.default_rng(1)
    alpha_true = rng.standard_normal((n_img, q)).astype(np.float64)
    r_true = rng.integers(0, n_rot, size=n_img)
    t_true = rng.integers(0, n_trans, size=n_img)

    pm = np.zeros((n_img, n_rot, n_trans, q), dtype=np.float64)
    for i in range(n_img):
        pm[i, r_true[i], t_true[i]] = alpha_true[i]

    R = np.linalg.qr(rng.standard_normal((q, q)))[0]
    pm_rotated = pm.copy()
    for i in range(n_img):
        pm_rotated[i, r_true[i], t_true[i]] = alpha_true[i] @ R

    err_orig = embedding_error_oracle(pm, alpha_true, r_true, t_true)
    err_rotated = embedding_error_oracle(pm_rotated, alpha_true, r_true, t_true)
    np.testing.assert_allclose(err_orig, err_rotated, atol=1e-10)


def test_embedding_error_marginal_zero_with_oracle_responsibilities():
    """If gamma is one-hot at the true pose and post_mean[true] is
    alpha_true, the marginal embedding equals alpha_true."""
    n_img, n_rot, n_trans, q = 6, 3, 2, 2
    rng = np.random.default_rng(2)
    alpha_true = rng.standard_normal((n_img, q)).astype(np.float64)
    r_true = rng.integers(0, n_rot, size=n_img)
    t_true = rng.integers(0, n_trans, size=n_img)

    pm = np.zeros((n_img, n_rot, n_trans, q), dtype=np.float64)
    log_resp = np.full((n_img, n_rot, n_trans), -np.inf, dtype=np.float64)
    for i in range(n_img):
        pm[i, r_true[i], t_true[i]] = alpha_true[i]
        log_resp[i, r_true[i], t_true[i]] = 0.0

    err = embedding_error_marginal(pm, log_resp, alpha_true)
    assert err == pytest.approx(0.0, abs=1e-12)
