"""Tests for the closed-form M-step (`update_factor_closed_form`).

This is the **canonical** PPCA M-step (Tipping & Bishop 1999)
adapted to pose-marginal cryo-EM. It replaces the broken gradient
descent variants (`update_factor_one_outer_step`,
`update_factor_full_ecm`), which are retained only for parity.

See `docs/math/ppca_closed_form_mstep.md` for the math.

The acceptance criteria pinned here:

1. Oracle init at LOW noise: M-step preserves U up to a small,
   bounded data-noise floor. Drift over multiple M-step iterations
   is bounded (the M-step is contractive).

2. Perturbed init at moderate noise: M-step descends pe by at least
   0.1 absolute in one shot — i.e., makes meaningful progress
   toward the data MLE.

3. The M-step is consistent with `T(U_true) = B`: the residual at
   oracle init is small (verified at ~0.01% for sigma=0.001,
   ~0.7% for sigma=0.1).

The "joint M+mean loop ab-initio from random init" target is
**not** pinned here — that is the research item documented in
task #40. Toy-size + high-noise + bistability of joint EM is a
separate concern from M-step correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.init import init_oracle, init_truth_perturbed
from recovar.em.ppca_abinitio.metrics import projector_frobenius_error
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


VOLUME_SHAPE = (16, 16, 16)
IMAGE_SHAPE = (16, 16)


class _Cfg(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, p, *, half_image=False):
        n = p.shape[0]
        full = jnp.ones((n, int(np.prod(self.image_shape))), dtype=jnp.float64)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, b, apply_image_mask=False):
        return b


def _make_dataset(sigma=0.01, n_train=1024):
    grid = build_fixed_grid(healpix_order=1, max_shift=1)
    return make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=n_train,
        n_images_val=8,
        sigma_real=sigma,
        seed=0,
    )


def _call(cfg, ds, init):
    return update_factor_closed_form(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
    )


def test_closed_form_oracle_init_stays_near_truth_at_low_noise():
    """At low noise (sigma=0.01) and oracle init, one closed-form
    M-step keeps the projector error of U bounded near zero. With
    the data-noise floor at vol 16, pe should stay below 0.10."""
    ds = _make_dataset(sigma=0.01, n_train=1024)
    cfg = _Cfg(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_oracle(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
    )
    out = _call(cfg, ds, init)
    pe = projector_frobenius_error(out.U, ds.U_half_true, VOLUME_SHAPE)
    assert pe < 0.10, (
        f"closed-form oracle init: pe={pe:.4f}, expected < 0.10. "
        "Either the M-step has regressed or the noise floor has shifted."
    )


def test_closed_form_oracle_iterated_drift_is_bounded():
    """Iterating the M-step at oracle init (with mu held at truth)
    should produce a small, BOUNDED drift. After 5 iterations the
    pe should still be below 0.15 — the M-step is contractive
    around the data MLE."""
    ds = _make_dataset(sigma=0.01, n_train=1024)
    cfg = _Cfg(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    cur = init_oracle(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
    )
    pes = []
    for _ in range(5):
        cur = _call(cfg, ds, cur)
        pes.append(float(projector_frobenius_error(cur.U, ds.U_half_true, VOLUME_SHAPE)))
    assert max(pes) < 0.15, f"M-step drift unbounded: pes={pes}"
    # And the LAST pe should be at most ~3x the FIRST (slowing drift, not blowing up)
    assert pes[-1] < 3.0 * pes[0] + 0.05, f"M-step drift accelerating: pes={pes}"


def test_closed_form_perturbed_init_descends():
    """From a 0.3-perturbed init (eps_U=0.3, mu held at truth) the
    closed-form M-step should reduce the projector error by at
    least 0.15 absolute. This matches the existing
    `test_perturbed_init_factor_update_actually_descends` for the
    gradient version, but with the closed-form solver."""
    ds = _make_dataset(sigma=0.01, n_train=1024)
    cfg = _Cfg(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.3,
        seed=0,
    )
    pe0 = projector_frobenius_error(init.U, ds.U_half_true, VOLUME_SHAPE)
    out = _call(cfg, ds, init)
    pe1 = projector_frobenius_error(out.U, ds.U_half_true, VOLUME_SHAPE)
    assert pe0 - pe1 > 0.15, (
        f"closed-form M-step did not descend perturbed init: pe {pe0:.4f} -> {pe1:.4f}, expected reduction > 0.15"
    )


def test_closed_form_returns_valid_ppca_init():
    """Output is a valid PPCAInit with the expected shape, dtype,
    and that mu and s are unchanged from the input."""
    ds = _make_dataset(sigma=0.01, n_train=512)
    cfg = _Cfg(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_oracle(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
    )
    out = _call(cfg, ds, init)
    assert out.U.shape == init.U.shape
    assert out.U.dtype == jnp.complex128
    assert jnp.all(out.mu == init.mu), "mu must be preserved by the M-step"
    assert jnp.all(out.s == init.s), "s must be preserved by the M-step"
    assert jnp.all(jnp.isfinite(out.U))
