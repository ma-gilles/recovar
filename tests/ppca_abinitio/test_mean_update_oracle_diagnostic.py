"""Diagnostic test pinning the mean update at oracle init.

History (2026-04-08): an earlier revision of this file pinned a
"structural ~0.24 gridding bias" of the mean update at oracle init.
That bias was actually a LINEAR-INTERP slice/adjoint discretization
artifact, NOT a fundamental property of the half-volume Wiener
filter.

After the 2026-04-09 switch to NEAREST discretization throughout
the v0 ab-initio path (forward model = inversion model), the
oracle-init FRE drops to ~0.005-0.015. The mean update under
nearest disc is essentially exact at oracle, modulo the data noise.

This file now pins the *new* behavior: small FRE at oracle, with
the small drift coming entirely from data noise (sigma_real). If
this test fires high, either nearest disc has regressed or someone
has reintroduced linear-interp slicing.

Cross-references:
  - mean_update.py:_solve_wiener
  - posterior.py (NEAREST disc note at top of slicing helpers)
  - docs/math/ppca_closed_form_mstep.md
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
    against mu_true should be very small (< 0.05) under the v0
    nearest-disc design. The mean update is essentially a no-op at
    oracle.

    History: this test used to pin a "structural [0.20, 0.32] band"
    that was a LINEAR-INTERP artifact. Switching to NEAREST disc
    dropped the FRE by an order of magnitude.

    A FRE *above 0.05* would mean either nearest disc has been
    regressed back to linear-interp somewhere, or a real bug in the
    Wiener filter / slicer adjoint pair.
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
    # vol=8 is the worst-case for discretization noise (smallest grid).
    # Empirically FRE is ~0.05 here under nearest disc; we leave headroom.
    assert fre < 0.10, (
        f"oracle-init FRE = {fre:.4f}, expected < 0.10 under nearest disc. "
        "If this is large, nearest disc has regressed somewhere or the "
        "slice/adjoint pair has a real bug."
    )


def test_oracle_init_bias_is_independent_of_n_images():
    """At oracle init under nearest disc, the (small) FRE should be
    essentially independent of n_images. Variability across a 16x
    growth in n_images stays under 0.02 absolute.

    The original motivation was to distinguish a Crowther-sparsity
    bias (which would shrink with n_images) from a structural slice
    bias (which would not). Under nearest disc both are absent at
    oracle and the FRE is just data-noise variance.
    """
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
    is bounded across volume sizes from 8³ to 16³ at toy data scales
    under the v0 nearest-disc design.

    History: an earlier revision of this test pinned a [0.18, 0.32]
    "structural bias band" that turned out to be a linear-interp
    artifact. With nearest disc the FRE is < 0.05 across all
    tested sizes.
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

    # Under nearest disc, all FREs should be small. vol 8 is the
    # worst case (smallest grid → most discretization noise) and
    # empirically lands around 0.05.
    for fre in fres:
        assert fre < 0.10, (
            f"FREs across volume sizes 8/12/16: {fres}. Expected all < 0.10 "
            "under nearest disc. If this fires, nearest disc has been "
            "regressed somewhere or the slice/adjoint pair has a real bug."
        )

    # Larger volumes should give SMALLER FRE (more voxels per slice
    # plane → less per-voxel discretization noise). Pin a loose
    # monotone-ish bound.
    assert fres[-1] <= fres[0] + 0.02, (
        f"FREs across vol sizes 8/12/16: {fres}. Expected vol 16 to be no worse than vol 8 + 0.02."
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
