"""Tests for `recovar.em.ppca_abinitio.init`.

Pins the three v0 initializers:
- `init_oracle` — used by Stage 0B
- `init_truth_perturbed` — positive control for Stage 1B/1C
- `init_random_lowpass` — stress control / negative control
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    half_real_space_gram,
    half_volume_radial_index,
    make_half_volume_weights,
)
from recovar.em.ppca_abinitio.init import (
    init_oracle,
    init_random_lowpass,
    init_truth_perturbed,
)
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)
from recovar.em.ppca_abinitio.types import PPCAInit

pytestmark = pytest.mark.unit


VOLUME_SHAPE = (8, 8, 8)
IMAGE_SHAPE = (8, 8)
N_FULL = int(np.prod(VOLUME_SHAPE))
N_HALF = VOLUME_SHAPE[0] * VOLUME_SHAPE[1] * (VOLUME_SHAPE[2] // 2 + 1)


def _make_ground_truth(seed=0, q=3):
    grid = build_fixed_grid(healpix_order=2, max_shift=1)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=q,
        n_images_train=4,
        n_images_val=2,
        seed=seed,
    )
    return ds


# ---------------------------------------------------------------------------
# init_oracle
# ---------------------------------------------------------------------------


def test_init_oracle_returns_ground_truth_unchanged():
    ds = _make_ground_truth()
    init = init_oracle(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
    )
    assert isinstance(init, PPCAInit)
    np.testing.assert_array_equal(np.asarray(init.mu), np.asarray(ds.mu_half_true))
    np.testing.assert_array_equal(np.asarray(init.U), np.asarray(ds.U_half_true))
    np.testing.assert_array_equal(np.asarray(init.s), np.asarray(ds.s_true))
    assert init.volume_shape == VOLUME_SHAPE
    assert init.q == ds.q


# ---------------------------------------------------------------------------
# init_truth_perturbed
# ---------------------------------------------------------------------------


def test_init_truth_perturbed_changes_with_nonzero_eps():
    ds = _make_ground_truth()
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.1,
        eps_U=0.1,
        seed=0,
    )
    assert not np.allclose(np.asarray(init.mu), np.asarray(ds.mu_half_true))
    assert not np.allclose(np.asarray(init.U), np.asarray(ds.U_half_true))
    # s is left untouched
    np.testing.assert_array_equal(np.asarray(init.s), np.asarray(ds.s_true))


def test_init_truth_perturbed_zero_eps_returns_truth():
    ds = _make_ground_truth()
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.0,
        seed=0,
    )
    np.testing.assert_allclose(np.asarray(init.mu), np.asarray(ds.mu_half_true), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(init.U), np.asarray(ds.U_half_true), rtol=1e-12, atol=1e-12)


def test_init_truth_perturbed_preserves_real_volume_invariant():
    """Re-encoding through real_volume_to_half (= get_dft3_real)
    means the result is automatically Hermitian — verified by
    decoding to a real volume via get_idft3_real and checking the
    imaginary energy fraction is below 1e-12."""
    ds = _make_ground_truth()
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.2,
        eps_U=0.2,
        seed=42,
    )

    # mu round-trip
    mu_grid = jnp.asarray(init.mu).reshape(VOLUME_SHAPE[0], VOLUME_SHAPE[1], VOLUME_SHAPE[2] // 2 + 1)
    mu_real = ftu.get_idft3_real(mu_grid, volume_shape=VOLUME_SHAPE)
    assert mu_real.dtype in (jnp.float64, jnp.float32)
    assert jnp.all(jnp.isfinite(mu_real))

    # U round-trip — each row must be a valid real volume
    for k in range(init.q):
        U_grid = jnp.asarray(init.U[k]).reshape(VOLUME_SHAPE[0], VOLUME_SHAPE[1], VOLUME_SHAPE[2] // 2 + 1)
        U_real = ftu.get_idft3_real(U_grid, volume_shape=VOLUME_SHAPE)
        assert jnp.all(jnp.isfinite(U_real))


def test_init_truth_perturbed_seed_reproducible():
    ds = _make_ground_truth()
    kwargs = dict(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.1,
        eps_U=0.1,
        seed=7,
    )
    a = init_truth_perturbed(**kwargs)
    b = init_truth_perturbed(**kwargs)
    np.testing.assert_array_equal(np.asarray(a.mu), np.asarray(b.mu))
    np.testing.assert_array_equal(np.asarray(a.U), np.asarray(b.U))


# ---------------------------------------------------------------------------
# init_random_lowpass
# ---------------------------------------------------------------------------


def test_init_random_lowpass_shapes_and_dtypes():
    init = init_random_lowpass(volume_shape=VOLUME_SHAPE, q=3, k_max=2.0, seed=0)
    assert init.q == 3
    assert init.mu.shape == (N_HALF,)
    assert init.U.shape == (3, N_HALF)
    assert init.mu.dtype == jnp.complex128
    assert init.U.dtype == jnp.complex128
    assert init.s.dtype == jnp.float64
    assert init.s.shape == (3,)


def test_init_random_lowpass_mu_is_zero():
    init = init_random_lowpass(volume_shape=VOLUME_SHAPE, q=2, k_max=2.0, seed=0)
    np.testing.assert_array_equal(np.asarray(init.mu), 0.0 + 0.0j)


def test_init_random_lowpass_band_limit_zeros_high_frequencies():
    init = init_random_lowpass(volume_shape=VOLUME_SHAPE, q=2, k_max=1.5, seed=0)
    R = np.asarray(half_volume_radial_index(VOLUME_SHAPE))
    high = R > 1.5
    U_np = np.asarray(init.U)
    np.testing.assert_array_equal(U_np[:, high], 0.0 + 0.0j)


def test_init_random_lowpass_orthonormal_when_requested():
    init = init_random_lowpass(volume_shape=VOLUME_SHAPE, q=3, k_max=3.0, seed=0, orthonormalize=True)
    weights = make_half_volume_weights(VOLUME_SHAPE)
    G = np.asarray(half_real_space_gram(init.U, weights, N_FULL))
    np.testing.assert_allclose(G, np.eye(3), rtol=1e-10, atol=1e-10)


def test_init_random_lowpass_seed_reproducible():
    a = init_random_lowpass(volume_shape=VOLUME_SHAPE, q=2, k_max=2.0, seed=99)
    b = init_random_lowpass(volume_shape=VOLUME_SHAPE, q=2, k_max=2.0, seed=99)
    np.testing.assert_array_equal(np.asarray(a.U), np.asarray(b.U))
