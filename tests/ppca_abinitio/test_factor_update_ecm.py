"""Tests for the Stage 1D `update_factor_full_ecm` variant."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.factor_update import (
    update_factor_full_ecm,
    update_factor_one_outer_step,
)
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    half_real_space_gram,
    make_half_volume_weights,
)
from recovar.em.ppca_abinitio.init import init_truth_perturbed
from recovar.em.ppca_abinitio.metrics import projector_frobenius_error
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


VOLUME_SHAPE = (8, 8, 8)
IMAGE_SHAPE = (8, 8)
N_FULL = int(np.prod(VOLUME_SHAPE))


def _identity_ctf(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(batch, apply_image_mask=False):
    return batch


class _SyntheticConfig(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, ctf_params, *, half_image=False):
        full = _identity_ctf(ctf_params, self.image_shape, self.voxel_size)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, batch, apply_image_mask=False):
        return _identity_process(batch, apply_image_mask=apply_image_mask)


def _make_dataset_and_init():
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=128,
        n_images_val=32,
        sigma_real=0.2,
        seed=0,
    )
    cfg = _SyntheticConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.3,
        seed=0,
    )
    return ds, cfg, init


def test_ecm_returns_valid_init_and_info():
    ds, cfg, init = _make_dataset_and_init()
    out, info = update_factor_full_ecm(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        max_inner_steps=20,
        lr=1e-2,
        grad_norm_tol=1e-4,
    )
    assert out.U.shape == init.U.shape
    assert out.U.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(jnp.real(out.U)))
    assert jnp.all(jnp.isfinite(jnp.imag(out.U)))

    assert "n_inner_steps" in info
    assert "final_grad_norm" in info
    assert "initial_loss" in info
    assert "final_loss" in info
    assert "loss_decrease" in info
    assert "converged" in info
    assert info["n_inner_steps"] >= 1


def test_ecm_preserves_s_and_mu():
    ds, cfg, init = _make_dataset_and_init()
    out, _ = update_factor_full_ecm(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        max_inner_steps=20,
        lr=1e-2,
    )
    np.testing.assert_array_equal(np.asarray(out.s), np.asarray(init.s))
    np.testing.assert_array_equal(np.asarray(out.mu), np.asarray(init.mu))


def test_ecm_output_is_real_space_orthonormal():
    ds, cfg, init = _make_dataset_and_init()
    out, _ = update_factor_full_ecm(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        max_inner_steps=20,
        lr=1e-2,
    )
    weights = make_half_volume_weights(VOLUME_SHAPE)
    G = np.asarray(half_real_space_gram(out.U, weights, N_FULL))
    np.testing.assert_allclose(G, np.eye(2), rtol=1e-9, atol=1e-9)


def test_ecm_loss_is_monotone_decreasing_with_line_search():
    """The line search makes the inner loop loss-monotone — `final_loss
    <= initial_loss` always."""
    ds, cfg, init = _make_dataset_and_init()
    _, info = update_factor_full_ecm(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        max_inner_steps=20,
        lr=1e-2,
        line_search=True,
    )
    assert info["final_loss"] <= info["initial_loss"] + 1e-9


def test_ecm_at_least_as_good_as_1c_on_projector_error():
    """The ECM (full inner-loop convergence with line search) should
    not be much WORSE than the fixed-K Stage 1C update on the
    projector error metric. v0 tolerates 0.1 absolute slack."""
    ds, cfg, init = _make_dataset_and_init()

    out_1c = update_factor_one_outer_step(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        inner_steps=3,
        lr=1e-3,
        k_max=2.5,
    )
    err_1c = projector_frobenius_error(out_1c.U, ds.U_half_true, VOLUME_SHAPE)

    out_1d, _ = update_factor_full_ecm(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        max_inner_steps=50,
        lr=1e-2,
        k_max=2.5,
        line_search=True,
    )
    err_1d = projector_frobenius_error(out_1d.U, ds.U_half_true, VOLUME_SHAPE)

    assert err_1d <= err_1c + 0.1, f"1D projector error {err_1d:.4f} much worse than 1C {err_1c:.4f}"
