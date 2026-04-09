"""Test the posterior helper with a non-trivial CTF.

All other ppca_abinitio tests use identity CTF. This test exercises
the posterior kernel with a real `recovar.core.ctf.CTFEvaluator`
populated with realistic SPA defocus values, and verifies the
kernel still produces sensible scores and posterior moments.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.core.ctf import CTFEvaluator, CTFMode
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.posterior import (
    score_and_posterior_moments_eqx,
)
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


VOLUME_SHAPE = (8, 8, 8)
IMAGE_SHAPE = (8, 8)


def _identity_process(batch, apply_image_mask=False):
    return batch


class _RealCTFConfig(eqx.Module):
    """Forward-model config that uses a real CTFEvaluator."""

    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)
    _ctf: object = eqx.field(static=True)
    _process: object = eqx.field(static=True)

    def compute_ctf(self, ctf_params, *, half_image=False):
        return self._ctf(ctf_params, self.image_shape, self.voxel_size, half_image=half_image)

    def process_fn(self, batch, apply_image_mask=False):
        return self._process(batch, apply_image_mask=apply_image_mask)


def _build_real_ctf_params(rng, n_images):
    """Build per-image SPA CTF params with defocus uniform in [1, 3] um.

    Parameter layout per recovar.core.ctf.CTFParamIndex:
        [DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT, BFACTOR, CONTRAST]
    """
    DFU = rng.uniform(10000, 30000, size=n_images).astype(np.float64)  # 1-3 um in Å
    DFV = DFU + rng.uniform(-1000, 1000, size=n_images)
    DFANG = rng.uniform(0, 360, size=n_images)
    VOLT = np.full(n_images, 300.0, dtype=np.float64)  # 300 kV
    CS = np.full(n_images, 2.7, dtype=np.float64)  # mm
    W = np.full(n_images, 0.07, dtype=np.float64)
    PHASE = np.zeros(n_images, dtype=np.float64)
    BFACTOR = np.zeros(n_images, dtype=np.float64)
    CONTRAST = np.ones(n_images, dtype=np.float64)
    params = np.stack([DFU, DFV, DFANG, VOLT, CS, W, PHASE, BFACTOR, CONTRAST], axis=-1)
    return jnp.asarray(params, dtype=jnp.float64)


def test_kernel_runs_with_real_ctf_and_produces_finite_output():
    rng = np.random.default_rng(0)
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=0.2,
        seed=0,
    )

    ctf_eval = CTFEvaluator(mode=CTFMode.SPA)
    cfg = _RealCTFConfig(
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        voxel_size=2.0,  # 2 Å/pixel
        _ctf=ctf_eval,
        _process=_identity_process,
    )

    n_img = ds.n_img
    ctf_params = _build_real_ctf_params(rng, n_img)

    stats = score_and_posterior_moments_eqx(
        cfg,
        ds.mu_half_true,
        ds.U_half_true,
        ds.s_true,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ctf_params,
        ds.noise_variance_full,
    )

    assert jnp.all(jnp.isfinite(stats.log_scores)), "non-finite log_scores"
    assert jnp.all(jnp.isfinite(stats.post_mean)), "non-finite post_mean"
    assert jnp.all(jnp.isfinite(stats.post_Hinv)), "non-finite post_Hinv"

    # log_resp must still normalize per image
    sums = np.exp(np.asarray(stats.log_resp).reshape(n_img, -1)).sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-12)


def test_kernel_real_ctf_post_Hinv_is_positive_definite():
    """The H matrix is built from CTF² weighting; with non-trivial
    CTF (which has zeros at certain frequencies), the H eigenvalues
    must still be positive (the diag(1/s) regularizer ensures this)."""
    rng = np.random.default_rng(1)
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=2,
        n_images_val=1,
        sigma_real=0.2,
        seed=1,
    )

    ctf_eval = CTFEvaluator(mode=CTFMode.SPA)
    cfg = _RealCTFConfig(
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        voxel_size=2.0,
        _ctf=ctf_eval,
        _process=_identity_process,
    )
    n_img = ds.n_img
    ctf_params = _build_real_ctf_params(rng, n_img)
    stats = score_and_posterior_moments_eqx(
        cfg,
        ds.mu_half_true,
        ds.U_half_true,
        ds.s_true,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ctf_params,
        ds.noise_variance_full,
    )
    Hinv = np.asarray(stats.post_Hinv)
    for i in range(n_img):
        for r in range(min(5, ds.n_rot)):
            eigs = np.linalg.eigvalsh(Hinv[i, r])
            assert eigs.min() > 0, f"Hinv at (i={i}, r={r}) not PD: eigs={eigs}"


def test_kernel_real_ctf_differs_from_identity_ctf():
    """A sanity check: the score with real defocused CTF should
    differ from the score with identity CTF on the same inputs."""
    rng = np.random.default_rng(2)
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=0.2,
        seed=2,
    )

    ctf_eval_real = CTFEvaluator(mode=CTFMode.SPA)
    cfg_real = _RealCTFConfig(
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        voxel_size=2.0,
        _ctf=ctf_eval_real,
        _process=_identity_process,
    )

    n_img = ds.n_img
    ctf_params_real = _build_real_ctf_params(rng, n_img)

    stats_real = score_and_posterior_moments_eqx(
        cfg_real,
        ds.mu_half_true,
        ds.U_half_true,
        ds.s_true,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ctf_params_real,
        ds.noise_variance_full,
    )

    # Identity CTF for comparison (use the standard SyntheticConfig)
    def _id_ctf(p, sh, vs, *, half_image=False):
        n = p.shape[0]
        sz = int(np.prod(sh))
        full = jnp.ones((n, sz), dtype=jnp.float64)
        if half_image:
            return ftu.full_image_to_half_image(full, sh)
        return full

    class _IdConfig(eqx.Module):
        image_shape: tuple = eqx.field(static=True)
        volume_shape: tuple = eqx.field(static=True)
        voxel_size: float = eqx.field(static=True)

        def compute_ctf(self, p, *, half_image=False):
            return _id_ctf(p, self.image_shape, self.voxel_size, half_image=half_image)

        def process_fn(self, b, apply_image_mask=False):
            return b

    cfg_id = _IdConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=2.0)
    stats_id = score_and_posterior_moments_eqx(
        cfg_id,
        ds.mu_half_true,
        ds.U_half_true,
        ds.s_true,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ctf_params_real,  # ctf_params are still passed but ignored by id ctf
        ds.noise_variance_full,
    )

    # log_scores must differ between real and identity CTF
    diff = float(jnp.max(jnp.abs(stats_real.log_scores - stats_id.log_scores)))
    assert diff > 1e-3, (
        f"real CTF and identity CTF produce identical scores ({diff:.2e}); the CTF is not actually being applied."
    )
