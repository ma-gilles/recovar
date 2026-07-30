import argparse
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from recovar.commands import pipeline
import recovar.core.fourier_transform_utils as ftu
from recovar.core import linalg
from recovar.ppca.w_regularization import w_prior_precision, w_prior_quadratic
from recovar.solvar.solvar import (
    _state_from_loadings,
    loadings_from_state,
    make_loading_from_basis,
    make_random_loading,
    solvar_image_losses,
)


pytestmark = pytest.mark.unit


def _weights(image_shape):
    return np.asarray(
        jnp.tile(linalg.half_spectrum_last_axis_weights(image_shape[1], dtype=jnp.float32), image_shape[0])
    )


def test_solvar_ls_loss_matches_direct_low_rank_covariance_terms():
    image_shape = (4, 4)
    rng = np.random.default_rng(0)
    y = rng.normal(size=(3, image_shape[0] * (image_shape[1] // 2 + 1))).astype(np.float32)
    Z = rng.normal(size=(3, 2, y.shape[1])).astype(np.float32)

    got = np.asarray(solvar_image_losses(jnp.asarray(y), jnp.asarray(Z), image_shape, objective="ls", apply_mean_terms=True))
    w_sqrt = np.sqrt(_weights(image_shape)).astype(np.float32)
    expected = []
    for i in range(y.shape[0]):
        y_w = y[i] * w_sqrt
        Z_cols = (Z[i] * w_sqrt[None, :]).T
        cov = Z_cols @ Z_cols.T
        direct = np.linalg.norm(np.outer(y_w, y_w) - cov - np.eye(y.shape[1], dtype=np.float32), ord="fro") ** 2
        expected.append(direct - y.shape[1])
    np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_solvar_mle_loss_matches_direct_woodbury_covariance():
    image_shape = (4, 4)
    rng = np.random.default_rng(1)
    y = rng.normal(size=(2, image_shape[0] * (image_shape[1] // 2 + 1))).astype(np.float32)
    Z = rng.normal(size=(2, 3, y.shape[1])).astype(np.float32)

    got = np.asarray(solvar_image_losses(jnp.asarray(y), jnp.asarray(Z), image_shape, objective="mle", apply_mean_terms=True))
    w_sqrt = np.sqrt(_weights(image_shape)).astype(np.float32)
    expected = []
    for i in range(y.shape[0]):
        y_w = y[i] * w_sqrt
        Z_cols = (Z[i] * w_sqrt[None, :]).T
        cov = Z_cols @ Z_cols.T + np.eye(y.shape[1], dtype=np.float32)
        expected.append(float(y_w @ np.linalg.solve(cov, y_w) + np.linalg.slogdet(cov)[1]))
    np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_solvar_mle_loss_has_finite_gradient():
    image_shape = (4, 4)
    y = jnp.arange(24, dtype=jnp.float32).reshape(2, 12) / 20.0
    Z = jnp.arange(48, dtype=jnp.float32).reshape(2, 2, 12) / 30.0

    def loss_fn(projected_basis):
        return jnp.sum(solvar_image_losses(y, projected_basis, image_shape, objective="mle"))

    grad = jax.grad(loss_fn)(Z)
    assert jnp.all(jnp.isfinite(grad))


def test_complex_adam_step_descends_real_quadratic():
    W = jnp.array([1.0 + 2.0j], dtype=jnp.complex64)
    objective = lambda z: jnp.sum(jnp.real(jnp.conj(z) * z))
    grad = jax.grad(objective)(W)
    grad = jax.tree.map(jnp.conj, grad)

    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(W)
    updates, _ = optimizer.update(grad, opt_state, W)
    next_W = optax.apply_updates(W, updates)

    assert objective(next_W) < objective(W)


def test_make_loading_from_basis_packs_covariance_square_root():
    volume_shape = (4, 4, 4)
    real_basis = np.zeros((2, *volume_shape), dtype=np.float32)
    real_basis[0, 1, 1, 1] = 1.0
    real_basis[1, 2, 2, 2] = 1.0
    u_full_cols = np.asarray(ftu.get_dft3(real_basis).reshape(2, -1).T)
    s = np.array([4.0, 9.0], dtype=np.float32)

    W_half = make_loading_from_basis(u_full_cols, s, 2, volume_shape)
    W_full = np.asarray(ftu.half_volume_to_full_volume(W_half.T, volume_shape).T)
    np.testing.assert_allclose(W_full, u_full_cols * np.sqrt(s)[None, :], rtol=1e-6, atol=1e-6)

    W_half_rows = make_loading_from_basis(u_full_cols.T, s, 2, volume_shape)
    np.testing.assert_allclose(W_half_rows, W_half, rtol=1e-6, atol=1e-6)


def test_w_prior_regularization_helper_matches_ppca_convention():
    W = jnp.array([[1.0 + 2.0j, 3.0], [4.0, 5.0j]], dtype=jnp.complex64)
    W_prior = jnp.array([[2.0, 4.0], [8.0, 16.0]], dtype=jnp.float32)
    expected_precision = 1.0 / W_prior
    np.testing.assert_allclose(np.asarray(w_prior_precision(W_prior)), np.asarray(expected_precision))
    expected_quad = jnp.sum(expected_precision * (jnp.abs(W) ** 2))
    np.testing.assert_allclose(np.asarray(w_prior_quadratic(W, W_prior)), np.asarray(expected_quad))


def test_pipeline_solvar_cli_plumbing():
    parser = pipeline.add_args(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "particles.mrcs",
            "-o",
            "out",
            "--mask",
            "sphere",
            "--poses",
            "poses.pkl",
            "--ctf",
            "ctf.pkl",
            "--use-solvar",
            "--solvar-objective",
            "ls",
            "--solvar-zdim",
            "4",
            "--solvar-iters",
            "2",
            "--solvar-init",
            "covariance",
        ]
    )
    assert args.use_solvar is True
    assert args.solvar_objective == "ls"
    assert args.solvar_zdim == 4
    assert args.solvar_init == "covariance"
    assert args.solvar_warm_start_n_pcs == 0
    assert args.solvar_iters == 2

    assert pipeline._resolve_solvar_zdim(SimpleNamespace(solvar_zdim=None, zdim=[6, 10])) == 6
    assert pipeline._resolve_solvar_warm_start_n_pcs(SimpleNamespace(solvar_warm_start_n_pcs=0), 20) == 50
    assert pipeline._resolve_solvar_warm_start_n_pcs(SimpleNamespace(solvar_warm_start_n_pcs=64), 20) == 64
    with pytest.raises(ValueError):
        pipeline._resolve_solvar_warm_start_n_pcs(SimpleNamespace(solvar_warm_start_n_pcs=8), 20)


def test_state_from_loadings_reconstructs_same_covariance():
    volume_shape = (6, 6, 6)
    rank = 2
    W_half = jnp.asarray(make_random_loading(volume_shape, rank, seed=3, init_scale=1.0))

    state = _state_from_loadings(W_half, volume_shape)
    W_recon = loadings_from_state(state)

    cov_orig = np.asarray(W_half) @ np.asarray(W_half).conj().T
    cov_recon = np.asarray(W_recon) @ np.asarray(W_recon).conj().T
    np.testing.assert_allclose(cov_recon, cov_orig, rtol=1e-4, atol=1e-3)


def test_state_from_loadings_u_is_orthonormal_in_real_space():
    volume_shape = (6, 6, 6)
    rank = 3
    W_half = jnp.asarray(make_random_loading(volume_shape, rank, seed=4, init_scale=1.0))
    vol_size = int(np.prod(volume_shape))
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)

    state = _state_from_loadings(W_half, volume_shape)
    U_real = np.asarray(ftu.get_idft3_real(np.asarray(state.U).T.reshape(rank, *half_shape), volume_shape))
    gram = U_real.reshape(rank, -1) @ U_real.reshape(rank, -1).T
    np.testing.assert_allclose(gram, np.eye(rank) / vol_size, rtol=1e-4, atol=1e-6)


