"""Tests for the residual-PCA baseline."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.baselines import residual_pca_baseline
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    half_real_space_gram,
    make_half_volume_weights,
)
from recovar.em.ppca_abinitio.metrics import projector_frobenius_error
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)
from recovar.em.ppca_abinitio.types import PPCAInit

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


def _make_config():
    return _SyntheticConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)


def _make_dataset(seed=0, sigma=0.2, n_train=128, n_val=32):
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    return make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=n_train,
        n_images_val=n_val,
        sigma_real=sigma,
        seed=seed,
    )


def test_residual_pca_baseline_returns_valid_init():
    ds = _make_dataset(seed=0)
    cfg = _make_config()
    out = residual_pca_baseline(
        cfg,
        ds.mu_half_true,
        1e-6,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        q=2,
    )
    assert isinstance(out, PPCAInit)
    assert out.q == 2
    assert out.U.dtype == jnp.complex128
    assert out.s.dtype == jnp.float64
    assert jnp.all(jnp.isfinite(jnp.real(out.U)))
    assert jnp.all(jnp.isfinite(jnp.imag(out.U)))


def test_residual_pca_baseline_mu_unchanged():
    """The baseline updates U only — mu must come back exactly equal."""
    ds = _make_dataset(seed=0)
    cfg = _make_config()
    out = residual_pca_baseline(
        cfg,
        ds.mu_half_true,
        1e-6,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        q=2,
    )
    np.testing.assert_array_equal(np.asarray(out.mu), np.asarray(ds.mu_half_true))


def test_residual_pca_baseline_output_is_real_space_orthonormal():
    ds = _make_dataset(seed=0)
    cfg = _make_config()
    out = residual_pca_baseline(
        cfg,
        ds.mu_half_true,
        1e-6,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        q=2,
    )
    weights = make_half_volume_weights(VOLUME_SHAPE)
    G = np.asarray(half_real_space_gram(out.U, weights, N_FULL))
    np.testing.assert_allclose(G, np.eye(2), rtol=1e-9, atol=1e-9)


def test_residual_pca_baseline_produces_finite_projector_error():
    """The baseline doesn't have to be GOOD — just finite and well-defined.
    The PPCA-vs-baseline comparison happens at the script level."""
    ds = _make_dataset(seed=0)
    cfg = _make_config()
    out = residual_pca_baseline(
        cfg,
        ds.mu_half_true,
        1e-6,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        q=2,
    )
    err = projector_frobenius_error(out.U, ds.U_half_true, VOLUME_SHAPE)
    assert np.isfinite(err)
    assert 0 <= err <= 4.0  # for q=2 the max ||P_a - P_b||_F is 2*sqrt(2)
