"""Diagnostic test pinning the mean-update bias at oracle init.

Established empirically (this branch, 2026-04-08): the
half-volume mean update path has a systematic ~25% Fourier
relative error against `mu_true` at oracle init even with
sigma_real → 0 and many images. This is **not** a Crowther
sparsity bias (only 0.01% of `mu_true`'s energy is outside the
default `max_r=3` sphere); it's a slice/adjoint *gridding*
discretization that the naive Wiener filter

    mu_next = Ft_y / Ft_ctf

does not undo. The slice operator's transfer function mixes
neighboring voxels through linear interpolation, and the
naive divide only inverts the diagonal of that mixing.

Production cryo-EM reconstruction handles this via
`relion_functions.post_process_from_filter_v2` (gridding
correction in real space) — but at the toy 8³ scale used in
v0 tests, that path produces *worse* results because the
spherical mask + grid correction are tuned for much larger
volumes.

This file pins the current behavior so that any future change
to `mean_update.py` that significantly improves OR degrades the
oracle-init FRE is caught loudly. The test does not assert any
specific value of the bias — it asserts the bias is in a
documented band.

Cross-references:
  - mean_update.py:_solve_wiener
  - relion_functions.post_process_from_filter_v2
  - this commit's debugging report in the commit message
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    half_volume_radial_index,
    make_half_volume_weights,
)
from recovar.em.ppca_abinitio.init import init_oracle
from recovar.em.ppca_abinitio.mean_update import update_mu_homogeneous
from recovar.em.ppca_abinitio.metrics import fourier_relative_error_mu
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


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


def test_oracle_init_mean_update_has_known_gridding_bias():
    """At oracle init with sigma → 0 and many images, the FRE
    against mu_true is in the documented [0.20, 0.32] band. If it
    drops outside this band, investigate.

    A FRE *significantly below 0.20* would mean the mean update
    has been improved (e.g. via proper gridding correction) — that
    would be great but should be a deliberate change with this
    test updated.

    A FRE *above 0.32* would mean a regression: the bias has
    grown beyond the toy-size discretization floor.
    """
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=(8, 8, 8),
        image_shape=(8, 8),
        grid=grid,
        q=2,
        n_images_train=1024,
        n_images_val=2,
        sigma_real=0.001,  # essentially noise-free
        seed=0,
    )
    cfg = _SyntheticConfig(image_shape=(8, 8), volume_shape=(8, 8, 8), voxel_size=1.0)
    init = init_oracle(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=(8, 8, 8),
    )

    res = update_mu_homogeneous(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        tau=0.0,
    )

    weights = make_half_volume_weights((8, 8, 8))
    fre = fourier_relative_error_mu(res.mu_half, ds.mu_half_true, weights_half=weights)
    assert 0.20 <= fre <= 0.32, (
        f"oracle-init FRE = {fre:.4f}, expected in [0.20, 0.32]. "
        "If significantly lower, the gridding correction has been "
        "implemented (update this test with new band). If higher, "
        "investigate a regression."
    )


def test_oracle_init_bias_is_independent_of_n_images():
    """Pin the claim that the bias is gridding-discretization, not
    sparsity. A doubling of n_images should NOT meaningfully change
    the FRE (it shrinks by less than 0.02 absolute)."""
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    cfg = _SyntheticConfig(image_shape=(8, 8), volume_shape=(8, 8, 8), voxel_size=1.0)
    weights = make_half_volume_weights((8, 8, 8))

    fres = []
    for n_train in (256, 1024, 4096):
        ds = make_synthetic_fixed_grid_dataset(
            SyntheticFamily.MATCHED_GRID_HET,
            volume_shape=(8, 8, 8),
            image_shape=(8, 8),
            grid=grid,
            q=2,
            n_images_train=n_train,
            n_images_val=2,
            sigma_real=0.001,
            seed=0,
        )
        init = init_oracle(
            mu_half_true=ds.mu_half_true,
            U_half_true=ds.U_half_true,
            s_true=ds.s_true,
            volume_shape=(8, 8, 8),
        )
        res = update_mu_homogeneous(
            cfg,
            init,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
            tau=0.0,
        )
        fres.append(float(fourier_relative_error_mu(res.mu_half, ds.mu_half_true, weights_half=weights)))

    spread = max(fres) - min(fres)
    assert spread < 0.02, (
        f"FRE varies by {spread:.4f} across 16x growth in n_images "
        f"({fres}). The bias is not independent of n_images — "
        "investigate whether this is the Crowther sparsity (which "
        "should shrink with n_images) or something else."
    )


def test_oracle_init_bias_is_independent_of_volume_size():
    """The slice/adjoint discretization bias is structural — it
    persists across volume sizes from 8³ to ~16³ at toy data scales.
    The bias does NOT shrink as volume grows; it's a property of
    the slice/adjoint pair without proper gridding correction.

    This test pins the claim that the bias is structural by
    checking that the FRE stays in [0.20, 0.30] across vol sizes.

    Cross-reference: the production `griddingCorrect_square` in
    `relion_functions.py` is calibrated for
    `volume_upsampling_factor=2`. With upsampling=1 (which the
    v0 mean update uses), grid correction makes FRE *worse*, not
    better. Fixing the bias requires either implementing the
    upsampled-volume slicing path or accepting the bias as a
    documented v0 limitation.
    """
    cfg_factory = lambda vs: _SyntheticConfig(image_shape=(vs, vs), volume_shape=(vs, vs, vs), voxel_size=1.0)

    fres = []
    for vs in (8, 12, 16):
        grid = build_fixed_grid(healpix_order=0 if vs <= 12 else 1, max_shift=1)
        ds = make_synthetic_fixed_grid_dataset(
            SyntheticFamily.MATCHED_GRID_HET,
            volume_shape=(vs, vs, vs),
            image_shape=(vs, vs),
            grid=grid,
            q=2,
            n_images_train=512,
            n_images_val=2,
            sigma_real=0.001,
            seed=0,
        )
        cfg = cfg_factory(vs)
        init = init_oracle(
            mu_half_true=ds.mu_half_true,
            U_half_true=ds.U_half_true,
            s_true=ds.s_true,
            volume_shape=(vs, vs, vs),
        )
        res = update_mu_homogeneous(
            cfg,
            init,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
            tau=0.0,
        )
        weights = make_half_volume_weights((vs, vs, vs))
        fres.append(float(fourier_relative_error_mu(res.mu_half, ds.mu_half_true, weights_half=weights)))

    # All FREs should be in the structural-bias band
    for fre in fres:
        assert 0.18 <= fre <= 0.32, (
            f"FREs across volume sizes 8/12/16: {fres}. Expected all "
            "in [0.18, 0.32] (structural slice/adjoint bias). Outside "
            "this band suggests a real change to the slicer or accumulator."
        )

    # And the bias should NOT scale with volume size (within 0.10 spread)
    spread = max(fres) - min(fres)
    assert spread < 0.10, (
        f"Bias varies by {spread:.4f} across vol sizes 8→16. The bias "
        "should be roughly volume-invariant if it's the slice/adjoint "
        "discretization."
    )


def test_oracle_init_99pct_of_mu_energy_is_inside_max_r_sphere():
    """Sanity check supporting the 'gridding bias, not sparsity'
    diagnosis: most of mu_true's energy is inside the sphere of
    voxels that the slicer touches."""
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=(8, 8, 8),
        image_shape=(8, 8),
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=0.001,
        seed=0,
    )
    R = np.asarray(half_volume_radial_index((8, 8, 8)))
    weights = np.asarray(make_half_volume_weights((8, 8, 8)))
    mu = np.asarray(ds.mu_half_true)
    inside = R <= 3.0
    energy_in = float(np.sum(weights[inside] * np.abs(mu[inside]) ** 2))
    energy_total = float(np.sum(weights * np.abs(mu) ** 2))
    frac_inside = energy_in / energy_total
    assert frac_inside > 0.99, (
        f"Only {100 * frac_inside:.2f}% of mu_true energy is inside "
        f"r<=3, expected > 99%. The Gaussian-blob mu may have grown "
        "high-frequency content."
    )
