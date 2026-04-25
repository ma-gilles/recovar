"""Cheat-free contract tests for ab-initio PPCA.

The v0 algorithm claims it does not depend on ground-truth eigenvalues
(see docs/math/ppca_abinitio_clean_algorithm.md). The natural way to
prove this programmatically is to poison `ds.s_true` with NaN and
verify the algorithmic path produces finite outputs anyway.

Two contracts are tested:

1. With `--s-init flat` (default), `run_two_stage` MUST NOT read
   `ds.s_true`. NaN in `ds.s_true` must not propagate to mu, U, or s.

2. As a positive control, `--s-init truth` MUST produce non-finite
   outputs when `ds.s_true` is poisoned, because that path explicitly
   reads it. (This makes the negative result in test 1 meaningful: it
   proves the test is sensitive to the leak it claims to detect.)
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)
from scripts.ppca_abinitio.run_cryobench import _Cfg, run_two_stage

pytestmark = [pytest.mark.unit]


def _build_tiny_dataset(seed: int = 0):
    """Smallest dataset that still exercises the EM path end-to-end.

    healpix_order=0 gives 12 rotations. max_shift=0 gives a single
    translation. Volume size 8 keeps half-volume size at 8*8*5 = 320.
    """
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    grid = build_fixed_grid(healpix_order=0, max_shift=0)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=volume_shape,
        image_shape=image_shape,
        grid=grid,
        q=2,
        n_images_train=8,
        n_images_val=0,
        sigma_real=0.1,
        seed=seed,
    )
    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)
    return cfg, ds


def _poison_s_true(ds):
    """Replace ds.s_true with NaN to detect any algorithmic leak.

    SyntheticDataset is a dataclass; we mutate the field in place. If
    any code path reads ds.s_true and propagates it through the linear
    algebra, the resulting mu/U will be non-finite.
    """
    poisoned = jnp.full_like(ds.s_true, jnp.nan)
    ds.s_true = poisoned
    return ds


def test_flat_s_init_does_not_read_s_true():
    """Contract: --s-init flat must produce finite mu and U even if
    ds.s_true is NaN. This is the cheat-free guarantee."""
    cfg, ds = _build_tiny_dataset(seed=0)
    ds = _poison_s_true(ds)

    final, _, _, _, _, _ = run_two_stage(
        cfg,
        ds,
        q=2,
        n_burnin=0,
        n_joint=2,
        mu_init_kind="zero",
        u_init_kind="svd",
        weighted_svd=True,
        anneal_schedule="none",
        seed=0,
        s_init_kind="flat",
    )

    assert jnp.all(jnp.isfinite(final.mu)), "mu became non-finite under flat s_init"
    assert jnp.all(jnp.isfinite(final.U)), "U became non-finite under flat s_init"
    assert jnp.all(jnp.isfinite(final.s)), "s became non-finite under flat s_init"


def test_truth_s_init_propagates_poisoned_s_true():
    """Positive control: with --s-init truth, NaN in ds.s_true must
    propagate. If this test ever passes, the previous test no longer
    proves anything."""
    cfg, ds = _build_tiny_dataset(seed=0)
    ds = _poison_s_true(ds)

    final, _, _, _, _, _ = run_two_stage(
        cfg,
        ds,
        q=2,
        n_burnin=0,
        n_joint=2,
        mu_init_kind="zero",
        u_init_kind="svd",
        weighted_svd=True,
        anneal_schedule="none",
        seed=0,
        s_init_kind="truth",
    )

    has_nan = (
        bool(jnp.any(~jnp.isfinite(final.mu)))
        or bool(jnp.any(~jnp.isfinite(final.U)))
        or bool(jnp.any(~jnp.isfinite(final.s)))
    )
    assert has_nan, (
        "Positive control failed: --s-init truth did not propagate NaN "
        "from poisoned ds.s_true. The cheat-free test in this file is no "
        "longer sensitive and must be redesigned."
    )


def test_run_two_stage_default_is_flat():
    """The function-level default for s_init_kind must match the CLI
    default (flat). A future internal caller relying on defaults must
    not silently get the cheating path."""
    import inspect

    sig = inspect.signature(run_two_stage)
    default = sig.parameters["s_init_kind"].default
    assert default == "flat", (
        f"run_two_stage default s_init_kind is '{default}', expected 'flat'. "
        "Internal callers without an explicit kwarg would silently use ground "
        "truth eigenvalues, breaking the cheat-free contract."
    )
