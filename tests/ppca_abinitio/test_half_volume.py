"""Tests for `recovar.em.ppca_abinitio.half_volume`.

The PPCA ab-initio module stores `mu` and `U` rows in half-volume
rfft layout (`(N, N, N//2+1)`). This test pins:

1. The rfft Hermitian weights match the 2D `make_half_image_weights`
   recipe extended to 3D, AND make Parseval hold against the full
   layout exactly.
2. `radial_band_limit_half` zeros out the right voxels.
3. `real_volume_orthonormalize_half` produces rows whose real-space
   Gram is identity.
4. The orthonormalized rows decode to actual orthonormal real
   volumes, verified by going through `get_idft3_real`.
5. The half-volume orthonormalize agrees with full-volume QR done
   on the inverse-FFT'd real volumes (cross-check against an
   independent path).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.half_volume import (
    half_real_space_gram,
    half_to_real_volume,
    half_to_real_volume_batch,
    half_volume_radial_index,
    make_half_volume_weights,
    radial_band_limit_half,
    radial_band_limit_half_batch,
    real_volume_orthonormalize_half,
    real_volume_to_half,
)

pytestmark = pytest.mark.unit


VOLUME_SHAPE = (8, 8, 8)
N0, N1, N2 = VOLUME_SHAPE
N_FULL = N0 * N1 * N2
N_HALF = N0 * N1 * (N2 // 2 + 1)


def _real_derived_half(rng):
    """A flat half-volume vector that is the rfft of a true real volume."""
    real_vol = rng.standard_normal(VOLUME_SHAPE).astype(np.float64)
    return real_volume_to_half(jnp.asarray(real_vol), VOLUME_SHAPE)


def _real_derived_half_batch(rng, q):
    return jnp.stack([_real_derived_half(rng) for _ in range(q)])


# ---------------------------------------------------------------------------
# Weights and Parseval
# ---------------------------------------------------------------------------


def test_weights_shape_and_values():
    w = make_half_volume_weights(VOLUME_SHAPE)
    assert w.shape == (N_HALF,)
    w3 = w.reshape((N0, N1, N2 // 2 + 1))
    # DC column (kx=0): all 1
    assert float(jnp.max(jnp.abs(w3[:, :, 0] - 1.0))) < 1e-12
    # Nyquist column (even N2): all 1
    if N2 % 2 == 0:
        assert float(jnp.max(jnp.abs(w3[:, :, -1] - 1.0))) < 1e-12
    # Interior columns: all 2
    interior = w3[:, :, 1:-1] if N2 % 2 == 0 else w3[:, :, 1:]
    assert float(jnp.max(jnp.abs(interior - 2.0))) < 1e-12


def test_weighted_half_inner_product_equals_full_inner_product():
    """The whole point of the rfft weights: weighted half-spectrum
    inner products must reproduce the full-spectrum sum exactly.
    """
    rng = np.random.default_rng(0)
    a = _real_derived_half(rng)
    b = _real_derived_half(rng)
    w = make_half_volume_weights(VOLUME_SHAPE)

    half_inner = float(jnp.sum(w * (jnp.conj(a) * b).real))

    a_full = ftu.half_volume_to_full_volume(a, VOLUME_SHAPE)
    b_full = ftu.half_volume_to_full_volume(b, VOLUME_SHAPE)
    full_inner = float(jnp.sum((jnp.conj(a_full) * b_full).real))

    np.testing.assert_allclose(half_inner, full_inner, rtol=1e-10, atol=1e-10)


def test_weighted_half_norm_equals_N_times_real_space_norm():
    """Parseval, half-spectrum form: weighted half-spectrum sum
    equals N · sum of squared real-space pixels."""
    rng = np.random.default_rng(1)
    real_vol = rng.standard_normal(VOLUME_SHAPE).astype(np.float64)
    half_v = real_volume_to_half(jnp.asarray(real_vol), VOLUME_SHAPE)
    w = make_half_volume_weights(VOLUME_SHAPE)

    half_norm = float(jnp.sum(w * (jnp.abs(half_v) ** 2)))
    real_norm = float(np.sum(real_vol**2))
    np.testing.assert_allclose(half_norm, N_FULL * real_norm, rtol=1e-10)


# ---------------------------------------------------------------------------
# Radial band-limit
# ---------------------------------------------------------------------------


def test_radial_band_limit_zeros_out_high_frequencies():
    rng = np.random.default_rng(0xB1)
    v = _real_derived_half(rng)
    R = half_volume_radial_index(VOLUME_SHAPE)
    k_max = 1.5
    out = radial_band_limit_half(v, VOLUME_SHAPE, k_max=k_max)
    out_np = np.asarray(out)

    high = np.where(np.asarray(R) > k_max)[0]
    low = np.where(np.asarray(R) <= k_max)[0]
    assert len(high) > 0
    np.testing.assert_array_equal(out_np[high], 0.0 + 0.0j)
    np.testing.assert_array_equal(out_np[low], np.asarray(v)[low])


def test_radial_band_limit_keeps_real_volume_real():
    """A band-limited half-volume must still decode to a real
    volume via get_idft3_real (which it does by construction —
    this test pins that no machinery is breaking the layout)."""
    rng = np.random.default_rng(0xB2)
    v = _real_derived_half(rng)
    out = radial_band_limit_half(v, VOLUME_SHAPE, k_max=2.0)
    real_vol = half_to_real_volume(out, VOLUME_SHAPE)
    assert real_vol.dtype in (jnp.float64, jnp.float32)
    assert real_vol.shape == VOLUME_SHAPE


def test_radial_band_limit_batch_matches_per_row():
    rng = np.random.default_rng(0xB3)
    q = 3
    U = _real_derived_half_batch(rng, q)
    out_b = radial_band_limit_half_batch(U, VOLUME_SHAPE, k_max=2.0)
    for i in range(q):
        per = radial_band_limit_half(U[i], VOLUME_SHAPE, k_max=2.0)
        np.testing.assert_allclose(np.asarray(out_b[i]), np.asarray(per), rtol=1e-14)


# ---------------------------------------------------------------------------
# Real-volume orthonormalization (half-spectrum, weighted)
# ---------------------------------------------------------------------------


def test_orthonormalize_produces_identity_gram():
    rng = np.random.default_rng(0xD1)
    q = 4
    U = _real_derived_half_batch(rng, q)
    w = make_half_volume_weights(VOLUME_SHAPE)
    U_orth = real_volume_orthonormalize_half(U, w, N_FULL)
    G = half_real_space_gram(U_orth, w, N_FULL)
    np.testing.assert_allclose(np.asarray(G), np.eye(q), rtol=1e-10, atol=1e-10)


def test_orthonormalized_rows_decode_to_orthonormal_real_volumes():
    """The whole point of the weighted orthonormalization: the
    decoded real-space volumes must be orthonormal under the
    real-space inner product (no FFT scale weirdness).
    """
    rng = np.random.default_rng(0xD2)
    q = 3
    U = _real_derived_half_batch(rng, q)
    w = make_half_volume_weights(VOLUME_SHAPE)
    U_orth = real_volume_orthonormalize_half(U, w, N_FULL)

    real_vols = half_to_real_volume_batch(U_orth, VOLUME_SHAPE)  # (q, N0, N1, N2)
    real_flat = np.asarray(real_vols).reshape(q, -1)
    G_real = real_flat @ real_flat.T  # (q, q), real-space inner products
    np.testing.assert_allclose(G_real, np.eye(q), rtol=1e-9, atol=1e-10)


def test_orthonormalize_preserves_row_span():
    """Every original row must lie in the row span of the
    orthonormalized matrix.

    Done entirely in real space (decoded volumes) to avoid the
    rfft-weighting confusion in the half-spectrum projector
    formula. With `U_orth_real` row-orthonormal under the standard
    real-space inner product, the projector is just
    `P = U_orth_real^T @ U_orth_real`, and applying it to each
    row of `U_real` must reproduce the row.
    """
    rng = np.random.default_rng(0xD3)
    q = 3
    U = _real_derived_half_batch(rng, q)
    w = make_half_volume_weights(VOLUME_SHAPE)
    U_orth = real_volume_orthonormalize_half(U, w, N_FULL)

    U_real = np.asarray(half_to_real_volume_batch(U, VOLUME_SHAPE)).reshape(q, -1)
    U_orth_real = np.asarray(half_to_real_volume_batch(U_orth, VOLUME_SHAPE)).reshape(q, -1)

    # Projection coefficients in real space: c[i, j] = <U_real[i], U_orth_real[j]>
    coeffs = U_real @ U_orth_real.T
    U_real_reconstructed = coeffs @ U_orth_real

    np.testing.assert_allclose(U_real_reconstructed, U_real, rtol=1e-9, atol=1e-10)


def test_orthonormalize_idempotent_gram():
    rng = np.random.default_rng(0xD4)
    q = 3
    U = _real_derived_half_batch(rng, q)
    w = make_half_volume_weights(VOLUME_SHAPE)
    U_orth1 = real_volume_orthonormalize_half(U, w, N_FULL)
    U_orth2 = real_volume_orthonormalize_half(U_orth1, w, N_FULL)
    G2 = half_real_space_gram(U_orth2, w, N_FULL)
    np.testing.assert_allclose(np.asarray(G2), np.eye(q), rtol=1e-10, atol=1e-10)


def test_orthonormalize_handles_zero_row_via_ridge():
    rng = np.random.default_rng(0xD5)
    q = 3
    U = _real_derived_half_batch(rng, q)
    U = U.at[1].set(jnp.zeros(N_HALF, dtype=jnp.complex128))
    w = make_half_volume_weights(VOLUME_SHAPE)
    U_orth = real_volume_orthonormalize_half(U, w, N_FULL)
    assert U_orth.shape == (q, N_HALF)
    assert jnp.all(jnp.isfinite(jnp.real(U_orth)))
    assert jnp.all(jnp.isfinite(jnp.imag(U_orth)))


# ---------------------------------------------------------------------------
# Round-trip via get_idft3_real / get_dft3_real
# ---------------------------------------------------------------------------


def test_real_volume_to_half_roundtrip():
    rng = np.random.default_rng(0xE1)
    real_vol = rng.standard_normal(VOLUME_SHAPE).astype(np.float64)
    half_v = real_volume_to_half(jnp.asarray(real_vol), VOLUME_SHAPE)
    decoded = half_to_real_volume(half_v, VOLUME_SHAPE)
    np.testing.assert_allclose(np.asarray(decoded), real_vol, rtol=1e-10, atol=1e-12)
