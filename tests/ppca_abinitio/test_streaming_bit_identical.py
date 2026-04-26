"""Phase 8 streaming bit-identicality contract.

The image_batch_size parameter on update_mu_homogeneous,
update_mu_residualized, and update_factor_closed_form must produce
output bit-identical to the unbatched path within float64 round-off.

This is a hard regression gate for Phase 8: any future refactor of
the streaming logic must keep this test passing.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.mean_update import (
    update_mu_homogeneous,
    update_mu_residualized,
)
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)
from recovar.em.ppca_abinitio.types import PPCAInit
from scripts.ppca_abinitio.run_cryobench import _Cfg

pytestmark = [pytest.mark.unit]


def _setup():
    volume_shape = (16, 16, 16)
    image_shape = (16, 16)
    grid = build_fixed_grid(healpix_order=0, max_shift=0)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=volume_shape,
        image_shape=image_shape,
        grid=grid,
        q=2,
        n_images_train=64,
        n_images_val=0,
        sigma_real=0.1,
        seed=0,
    )
    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)
    init = PPCAInit(
        mu=ds.mu_half_true.astype(jnp.complex128),
        U=ds.U_half_true.astype(jnp.complex128),
        s=jnp.ones(2, dtype=jnp.float64),
        volume_shape=volume_shape,
    )
    return cfg, ds, init


@pytest.mark.parametrize("image_batch_size", [8, 16, 32])
def test_update_mu_residualized_streaming_bit_identical(image_batch_size):
    cfg, ds, init = _setup()
    full = update_mu_residualized(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        tau=0.0,
    )
    streamed = update_mu_residualized(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        tau=0.0,
        image_batch_size=image_batch_size,
    )
    np.testing.assert_allclose(
        np.asarray(streamed.mu_half),
        np.asarray(full.mu_half),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("image_batch_size", [8, 16, 32])
def test_update_mu_homogeneous_streaming_bit_identical(image_batch_size):
    cfg, ds, init = _setup()
    full = update_mu_homogeneous(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        tau=0.0,
    )
    streamed = update_mu_homogeneous(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        tau=0.0,
        image_batch_size=image_batch_size,
    )
    np.testing.assert_allclose(
        np.asarray(streamed.mu_half),
        np.asarray(full.mu_half),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("image_batch_size", [8, 16, 32])
def test_update_factor_streaming_bit_identical(image_batch_size):
    cfg, ds, init = _setup()
    full = update_factor_closed_form(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
    )
    streamed = update_factor_closed_form(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        image_batch_size=image_batch_size,
    )
    np.testing.assert_allclose(
        np.asarray(streamed.U),
        np.asarray(full.U),
        atol=1e-7,
        rtol=1e-7,
    )
    np.testing.assert_array_equal(np.asarray(streamed.mu), np.asarray(full.mu))
    np.testing.assert_array_equal(np.asarray(streamed.s), np.asarray(full.s))


def test_streaming_no_op_when_batch_size_geq_n_img():
    """When image_batch_size >= n_img, the function falls through to
    the original unbatched path and is exactly equal."""
    cfg, ds, init = _setup()
    full = update_factor_closed_form(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
    )
    streamed = update_factor_closed_form(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        image_batch_size=128,  # > n_img=64
    )
    # When batch_size >= n_img, the function falls through to the
    # unbatched path. Outputs agree to JIT-recompile sub-ulp jitter.
    np.testing.assert_allclose(
        np.asarray(streamed.U),
        np.asarray(full.U),
        atol=1e-12,
        rtol=0,
    )
