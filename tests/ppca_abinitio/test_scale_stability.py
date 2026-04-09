"""Pin the cross-volume stability of the fixed-grid PPCA loop.

The mean update has a documented gridding-discretization bias at
oracle init (~0.24 FRE), characterized in
`test_mean_update_oracle_diagnostic.py` at vol 8. This file extends
that characterization to vol 12, 16, 24, 32 and pins the claim that:

  1. The bias is roughly volume-INVARIANT (does not shrink with N).
  2. The loop is stable at every size (no NaN, no blow-up).
  3. The iter-to-iter drift is bounded (the loop reaches a fixed point
     in the documented [0.20, 0.32] band).
  4. From a perturbed init, iter 1 makes meaningful progress.

These tests run the actual code (not loading the JSON artifact) so a
regression in `mean_update.py` or the slicer is caught by tests, not
by re-running the scale-validation script. The artifact in
`docs/math/ppca_abinitio_artifacts/scale_validation_v0.json` is the
v0 reference snapshot kept for the PR description.

Marked `slow` — vol 32 needs a GPU and ~1s wall time.
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


def _identity_ctf(p, sh, vs):
    n = p.shape[0]
    sz = int(np.prod(sh))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(b, apply_image_mask=False):
    return b


class _SynthCfg(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, p, *, half_image=False):
        full = _identity_ctf(p, self.image_shape, self.voxel_size)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, b, apply_image_mask=False):
        return _identity_process(b, apply_image_mask=apply_image_mask)


def _run_one(volume_size, n_images, init_kind, n_iters):
    vs = volume_size
    image_shape = (vs, vs)
    volume_shape = (vs, vs, vs)
    grid = build_fixed_grid(healpix_order=1, max_shift=1)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=volume_shape,
        image_shape=image_shape,
        grid=grid,
        q=2,
        n_images_train=n_images,
        n_images_val=max(8, n_images // 8),
        sigma_real=0.1,
        seed=0,
    )
    cfg = _SynthCfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)
    if init_kind == "oracle":
        init = init_oracle(
            mu_half_true=ds.mu_half_true,
            U_half_true=ds.U_half_true,
            s_true=ds.s_true,
            volume_shape=volume_shape,
        )
    else:
        init = init_truth_perturbed(
            mu_half_true=ds.mu_half_true,
            U_half_true=ds.U_half_true,
            s_true=ds.s_true,
            volume_shape=volume_shape,
            eps_mu=0.3,
            eps_U=0.0,
            seed=0,
        )
    cfg_run = PPCAConfig(n_iters=n_iters, update_mu=True, update_factor=False, ridge_lambda=0.0)
    weights = make_half_volume_weights(volume_shape)
    return run_fixed_grid_ppca(cfg, ds, init, cfg_run, weights_half=weights)


@pytest.mark.parametrize("volume_size,n_images", [(12, 256), (16, 256), (24, 384), (32, 512)])
def test_loop_oracle_init_stable_across_volume_sizes(volume_size, n_images):
    """Across vol 12, 16, 24, 32, oracle-init loop must:
    - produce no NaN
    - reach FRE in the structural-bias band [0.18, 0.32] within 3 iters
    - not blow up beyond 0.4 at any iteration
    """
    res = _run_one(volume_size, n_images, "oracle", n_iters=3)
    fre_traj = [m.fre_mu_val for m in res.iter_metrics]
    assert all(np.isfinite(fre_traj)), f"non-finite fre_traj at vol={volume_size}: {fre_traj}"
    assert max(fre_traj) < 0.4, f"loop blew up at vol={volume_size}: {fre_traj}"
    assert 0.18 <= fre_traj[-1] <= 0.32, (
        f"vol={volume_size}: final FRE {fre_traj[-1]:.4f} outside structural band [0.18, 0.32]. "
        f"Full traj: {fre_traj}. Either an improvement (lower) or a regression (higher) — "
        f"investigate before relaxing this band."
    )


def test_perturbed_init_iter1_makes_progress_at_vol_16():
    """At vol 16 from a 0.3-perturbed init, iter 1 must reduce FRE
    by at least 0.04 absolute (the perturbation is 0.30, the fixed
    point is ~0.24, so the reduction is bounded above by ~0.06)."""
    res = _run_one(16, 512, "perturbed", n_iters=2)
    fre_traj = [m.fre_mu_val for m in res.iter_metrics]
    assert fre_traj[1] < fre_traj[0] - 0.04, (
        f"perturbed-init iter 1 made too little progress: {fre_traj}. Expected fre_traj[1] < fre_traj[0] - 0.04."
    )
