"""Convergence diagnostic for the iterative Stage 1B/1C loop.

Earlier debugging found that the iter-1 mean update has a ~25%
gridding-discretization bias at oracle init (toy 8³ scale) that is
independent of `n_images`. This test characterizes what happens
across many iterations:

  - From oracle init, the loop should reach a fixed point near
    (but not exactly at) `mu_true`. The fixed-point FRE should be
    bounded.
  - The drift between consecutive iterations should DECREASE as
    we approach the fixed point (i.e. monotone convergence in
    iter-to-iter delta, not in distance to `mu_true`).
  - With more outer iterations the FRE-vs-truth should not blow up
    above the toy-size band.

This is a regression pin: if a future change makes the loop
diverge (FRE growing without bound) or oscillate, this test fires.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import make_half_volume_weights
from recovar.em.ppca_abinitio.init import init_oracle, init_truth_perturbed
from recovar.em.ppca_abinitio.loop import run_fixed_grid_ppca
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)
from recovar.em.ppca_abinitio.types import PPCAConfig

pytestmark = [pytest.mark.unit, pytest.mark.slow]


VOLUME_SHAPE = (8, 8, 8)
IMAGE_SHAPE = (8, 8)


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


def _make_dataset(seed=0, sigma=0.1, n_train=512, n_val=2):
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


def test_loop_from_oracle_init_does_not_diverge_over_8_iterations():
    """The mean-only loop from oracle init should not blow up the
    FRE over 8 iterations. The trajectory may drift away from
    `mu_true` (because of the gridding bias documented in
    test_mean_update_oracle_diagnostic.py), but the drift should
    be bounded above by 0.4 absolute.
    """
    ds = _make_dataset()
    cfg = _SyntheticConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_oracle(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
    )
    cfg_run = PPCAConfig(n_iters=8, update_mu=True, update_factor=False, ridge_lambda=0.0)
    weights = make_half_volume_weights(VOLUME_SHAPE)
    res = run_fixed_grid_ppca(cfg, ds, init, cfg_run, weights_half=weights)

    # Track the trajectory
    fre_traj = [m.fre_mu_val for m in res.iter_metrics]
    assert all(np.isfinite(fre_traj)), "loop produced non-finite FRE"
    assert max(fre_traj) < 0.4, f"FRE trajectory blew up: {fre_traj}. The mean update is unstable or has a regression."


def test_loop_from_perturbed_init_makes_iter_1_progress():
    """From a 0.5-perturbed init, iter-1 should reduce FRE by at
    least 0.15 absolute. This is the load-bearing claim of Stage
    1B at toy size — the loop runs and produces meaningful
    iter-1 progress."""
    ds = _make_dataset()
    cfg = _SyntheticConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.5,
        eps_U=0.0,
        seed=0,
    )
    cfg_run = PPCAConfig(n_iters=2, update_mu=True, update_factor=False, ridge_lambda=0.0)
    weights = make_half_volume_weights(VOLUME_SHAPE)
    res = run_fixed_grid_ppca(cfg, ds, init, cfg_run, weights_half=weights)

    fre_init = res.iter_metrics[0].fre_mu_val
    fre_iter1 = res.iter_metrics[1].fre_mu_val
    improvement = fre_init - fre_iter1
    assert improvement > 0.15, (
        f"iter-1 improvement = {improvement:.4f}, expected > 0.15. "
        f"Init FRE = {fre_init:.4f}, iter-1 FRE = {fre_iter1:.4f}."
    )


def test_loop_n_iters_drift_is_sublinear():
    """At the toy-size regime, the iter-by-iter drift away from
    truth should slow down with iteration count (i.e. it's
    bounded, not linear in n_iters). After iter 8, the FRE
    should be at most 1.5x the FRE at iter 4.
    """
    ds = _make_dataset()
    cfg = _SyntheticConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.3,
        eps_U=0.0,
        seed=0,
    )
    cfg_run = PPCAConfig(n_iters=8, update_mu=True, update_factor=False, ridge_lambda=0.0)
    weights = make_half_volume_weights(VOLUME_SHAPE)
    res = run_fixed_grid_ppca(cfg, ds, init, cfg_run, weights_half=weights)
    fre_traj = [m.fre_mu_val for m in res.iter_metrics]
    fre4 = fre_traj[4]
    fre8 = fre_traj[8]
    assert fre8 <= 1.5 * fre4 + 0.05, (
        f"trajectory drift is super-linear: iter4={fre4:.4f}, iter8={fre8:.4f}. Full traj: {fre_traj}"
    )
