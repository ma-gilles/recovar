"""Tests for `recovar.em.ppca_abinitio.factor_update`.

Pins the gradient + projection chain:
- output is finite, has the right shape and dtype
- output rows are real-space orthonormal (Gram-from-I < 1e-9)
- s is strictly preserved
- single iter from a perturbed init reduces the projector error
- random_lowpass init does not blow up
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.factor_update import (
    _expected_nll_half,
    update_factor_one_outer_step,
)
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    half_real_space_gram,
    make_half_volume_weights,
)
from recovar.em.ppca_abinitio.init import init_random_lowpass, init_truth_perturbed
from recovar.em.ppca_abinitio.metrics import projector_frobenius_error
from recovar.em.ppca_abinitio.posterior import (
    _preprocess_batch_to_half,
    _slice_mu_half,
    _slice_U_half,
    make_half_image_weights,
    score_from_half_image_projections,
)
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


def _make_dataset(seed=0, sigma=0.2, n_train=128, n_val=64):
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factor_update_returns_valid_init():
    ds = _make_dataset(seed=0)
    cfg = _make_config()
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.3,
        seed=0,
    )
    out = update_factor_one_outer_step(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        inner_steps=2,
        lr=1e-3,
        k_max=2.5,
    )
    assert isinstance(out, PPCAInit)
    assert out.U.shape == init.U.shape
    assert out.U.dtype == jnp.complex128
    assert out.mu.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(jnp.real(out.U)))
    assert jnp.all(jnp.isfinite(jnp.imag(out.U)))


def test_factor_update_preserves_s_strictly():
    """Per Q2, s must be literally constant during 1C."""
    ds = _make_dataset(seed=0)
    cfg = _make_config()
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.3,
        seed=0,
    )
    out = update_factor_one_outer_step(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        inner_steps=2,
        lr=1e-3,
        k_max=2.5,
    )
    np.testing.assert_array_equal(np.asarray(out.s), np.asarray(init.s))


def test_factor_update_preserves_mu_strictly():
    """update_factor_one_outer_step does not touch mu."""
    ds = _make_dataset(seed=0)
    cfg = _make_config()
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.3,
        seed=0,
    )
    out = update_factor_one_outer_step(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        inner_steps=2,
        lr=1e-3,
        k_max=2.5,
    )
    np.testing.assert_array_equal(np.asarray(out.mu), np.asarray(init.mu))


def test_factor_update_output_is_real_space_orthonormal():
    """The gauge-fix step at the end of the update must produce
    real-space orthonormal rows: (1/N) (U·w) U^H = I_q to ~1e-9."""
    ds = _make_dataset(seed=0)
    cfg = _make_config()
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.3,
        seed=0,
    )
    out = update_factor_one_outer_step(
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
    weights = make_half_volume_weights(VOLUME_SHAPE)
    G = np.asarray(half_real_space_gram(out.U, weights, N_FULL))
    np.testing.assert_allclose(G, np.eye(out.q), rtol=1e-9, atol=1e-9)


def test_factor_update_reduces_projector_error_from_truth_perturbed_init():
    """The first outer step from a truth-perturbed init must
    *reduce* the projector Frobenius error against U_true."""
    ds = _make_dataset(seed=0, sigma=0.2)
    cfg = _make_config()
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.3,
        seed=0,
    )
    err_before = projector_frobenius_error(init.U, ds.U_half_true, VOLUME_SHAPE)
    out = update_factor_one_outer_step(
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
    err_after = projector_frobenius_error(out.U, ds.U_half_true, VOLUME_SHAPE)
    assert err_after < err_before, f"factor update did not reduce projector error: {err_before:.4f} → {err_after:.4f}"


def test_factor_update_random_lowpass_does_not_blow_up():
    """Stress test: starting from random low-pass U, the update must
    not produce NaN/Inf and must preserve the gauge fix."""
    ds = _make_dataset(seed=1)
    cfg = _make_config()
    init = init_random_lowpass(volume_shape=VOLUME_SHAPE, q=2, k_max=2.5, s_init=ds.s_true, seed=0)
    init = PPCAInit(mu=ds.mu_half_true, U=init.U, s=ds.s_true, volume_shape=VOLUME_SHAPE)
    out = update_factor_one_outer_step(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        inner_steps=2,
        lr=1e-3,
        k_max=2.5,
    )
    assert jnp.all(jnp.isfinite(jnp.real(out.U)))
    assert jnp.all(jnp.isfinite(jnp.imag(out.U)))
    weights = make_half_volume_weights(VOLUME_SHAPE)
    G = np.asarray(half_real_space_gram(out.U, weights, N_FULL))
    np.testing.assert_allclose(G, np.eye(out.q), rtol=1e-9, atol=1e-9)


def test_expected_nll_uses_frozen_posterior_moments():
    """Zeroing the frozen posterior moments must change the Stage 1C loss."""
    ds = _make_dataset(seed=0, sigma=0.2, n_train=32, n_val=8)
    cfg = _make_config()
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.3,
        seed=0,
    )

    weights_half = make_half_image_weights(IMAGE_SHAPE)
    mean_proj_half = _slice_mu_half(init.mu, ds.rotations, IMAGE_SHAPE, VOLUME_SHAPE).astype(jnp.complex128)
    u_proj_half = _slice_U_half(init.U, ds.rotations, IMAGE_SHAPE, VOLUME_SHAPE).astype(jnp.complex128)
    shifted_half, ctf2_over_nv_half, _ = _preprocess_batch_to_half(
        cfg, ds.batch_full, ds.translations, ds.ctf_params, ds.noise_variance_full
    )
    stats = score_from_half_image_projections(
        mean_proj_half, u_proj_half, init.s, shifted_half, ctf2_over_nv_half, weights_half
    )

    loss_ref = float(
        _expected_nll_half(
            init.U,
            init.mu,
            init.s,
            ds.rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            shifted_half,
            ctf2_over_nv_half,
            weights_half,
            stats.log_resp,
            stats.post_mean,
            stats.post_Hinv,
        )
    )
    loss_no_mean = float(
        _expected_nll_half(
            init.U,
            init.mu,
            init.s,
            ds.rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            shifted_half,
            ctf2_over_nv_half,
            weights_half,
            stats.log_resp,
            jnp.zeros_like(stats.post_mean),
            stats.post_Hinv,
        )
    )
    loss_no_cov = float(
        _expected_nll_half(
            init.U,
            init.mu,
            init.s,
            ds.rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            shifted_half,
            ctf2_over_nv_half,
            weights_half,
            stats.log_resp,
            stats.post_mean,
            jnp.zeros_like(stats.post_Hinv),
        )
    )

    assert abs(loss_ref - loss_no_mean) > 1e-6
    assert abs(loss_ref - loss_no_cov) > 1e-6
