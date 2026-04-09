"""Pin the cross-volume stability of the fixed-grid PPCA loop.

History: an earlier revision of this file pinned a "structural
gridding bias" of FRE ~0.24 across all volume sizes. After the
2026-04-09 switch to NEAREST discretization throughout the v0
ab-initio path (forward model = inversion model), that "structural"
bias dropped to FRE ~0.01-0.02. The bias was a linear-interp
artifact, not a fundamental property of the slice operator.

What this file pins now (post-nearest):

  1. From oracle init the loop preserves mu to FRE < 0.05 across
     vol 12, 16, 24, 32. (Was [0.18, 0.32]; now < 0.05.)
  2. The loop is stable at every size (no NaN, no blow-up).
  3. The iter-to-iter drift is bounded.
  4. From a perturbed init, iter 1 makes meaningful progress.

These tests run the actual code (not loading the JSON artifact) so a
regression in `mean_update.py` or the slicer is caught by tests, not
by re-running the scale-validation script. The artifact in
`docs/math/ppca_abinitio_artifacts/scale_validation_v0.json` is a
historical snapshot from BEFORE the nearest-disc switch and should
NOT be used as the current reference.

Marked `slow` — vol 32 needs a GPU.
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
    """Across vol 12, 16, 24, 32, oracle-init mean-only loop must:
      - produce no NaN
      - keep FRE small (< 0.05) — i.e. mu stays near truth.

    Under NEAREST discretization the slice/adjoint operators are an
    exact pair, so the mean update at oracle init is essentially a
    no-op modulo data noise. The "structural 0.24 bias" pinned by an
    earlier revision of this test was a linear-interp artifact and
    is gone now.
    """
    res = _run_one(volume_size, n_images, "oracle", n_iters=3)
    fre_traj = [m.fre_mu_val for m in res.iter_metrics]
    assert all(np.isfinite(fre_traj)), f"non-finite fre_traj at vol={volume_size}: {fre_traj}"
    assert max(fre_traj) < 0.05, (
        f"vol={volume_size}: max FRE {max(fre_traj):.4f} > 0.05. "
        f"Full traj: {fre_traj}. The mean update should preserve mu near oracle "
        f"under nearest discretization. If this fires, the slicer or adjoint may "
        f"have regressed."
    )


def test_perturbed_init_iter1_makes_progress_at_vol_16():
    """At vol 16 from a 0.3-perturbed init, iter 1 must reduce FRE
    by at least 0.20 absolute. With nearest discretization the mean
    update is essentially exact, so a single iteration brings the
    mu from a 0.30-perturbed init back to within ~0.02 of truth.
    """
    res = _run_one(16, 512, "perturbed", n_iters=2)
    fre_traj = [m.fre_mu_val for m in res.iter_metrics]
    assert fre_traj[1] < fre_traj[0] - 0.20, (
        f"perturbed-init iter 1 made too little progress: {fre_traj}. Expected fre_traj[1] < fre_traj[0] - 0.20."
    )
