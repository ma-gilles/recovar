"""Half-volume / half-image utilities for the PPCA ab-initio module.

The PPCA ab-initio v0 module stores `mu` and `U` rows in **rfft-packed
half-volume layout** (`(N, N, N//2+1)`, complex128). In this layout the
Hermitian symmetry of FTs of real volumes is structural, not enforced
— there is no projection-back step. The user-driven correction
(2026-04-08) replaced an earlier `enforce_real_volume_ft` design with
this layout for that reason.

Key facts about half-spectrum inner products
--------------------------------------------

For two real-valued volumes `a, b` of shape `(N, N, N)` with
`norm="backward"` FFT:

    <a, b>_real           =  sum_x a(x) b(x)
    N * <a, b>_real       =  Re[ sum_k conj(a_full[k]) b_full[k] ]   (Parseval)
                          =  Re[ sum_k_half w(k) conj(a_half[k]) b_half[k] ]

where `w(k)` is the rfft Hermitian weight: `2` for interior packed-axis
columns (which represent `(k, -k)` pairs), `1` for the DC column
(`kx=0`) and the Nyquist column (`kx=N/2`, even-N only) which are
self-conjugate. This module's `make_half_volume_weights(...)` returns
that array.

Real-space orthonormality of the rows of a `(q, N_half)` half-volume
matrix `U_half` is therefore

    G_ij  =  Re[ sum_k w(k) conj(U_i[k]) U_j[k] ]  =  N * delta_ij

i.e. `G / N = I_q`. The orthonormalization function below performs the
matching Cholesky-whitening on the weighted Gram.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu

# ---------------------------------------------------------------------------
# rfft Hermitian weights — 3D analog of engine_v2.make_half_image_weights
# ---------------------------------------------------------------------------


def make_half_volume_weights(volume_shape: Sequence[int]) -> jnp.ndarray:
    """Return `(N_half,)` Hermitian weights for half-volume inner products.

    Mirrors `recovar/em/dense_single_volume/engine_v2.py:56:make_half_image_weights`
    in 3D. For a volume of shape `(N0, N1, N2)`, the rfft-packed
    half-volume has shape `(N0, N1, N2//2+1)`. The weights are:

      - `2` everywhere by default (each interior pixel represents itself
        and its conjugate partner in the negative `kx` half-axis);
      - `1` on the packed-axis column `kx = 0` (DC column — no conjugate
        partner);
      - `1` on the packed-axis column `kx = N2/2` if `N2` is even
        (Nyquist column — self-conjugate).

    The first two axes (`kz`, `ky`) get the full doubled weight regardless
    of position because the conjugate partner of `(kz, ky, kx>0)` is
    `(-kz, -ky, -kx)` which lives in the negative-kx half — already
    folded out of the rfft.
    """
    N0, N1, N2 = (int(s) for s in volume_shape)
    N2_half = N2 // 2 + 1
    w = 2.0 * jnp.ones((N0, N1, N2_half), dtype=jnp.float64)
    w = w.at[:, :, 0].set(1.0)
    if N2 % 2 == 0:
        w = w.at[:, :, -1].set(1.0)
    return w.reshape(-1)


# ---------------------------------------------------------------------------
# Radial mask in the half-volume rfft layout
# ---------------------------------------------------------------------------


def half_volume_radial_index(volume_shape: Sequence[int]) -> jnp.ndarray:
    """Return `(N_half,)` float64 radial frequency (in cycles / box) for
    every voxel in the half-volume rfft layout.

    Layout (matching `get_dft3_real`):
      - first two axes are fftshift'd, so `kz, ky` are centered around 0;
      - last axis is rfft-packed, so `kx` runs from `0` to `N2//2`.
    """
    N0, N1, N2 = (int(s) for s in volume_shape)
    kz = jnp.fft.fftshift(jnp.fft.fftfreq(N0)) * N0  # centered: -N0/2 .. N0/2-1
    ky = jnp.fft.fftshift(jnp.fft.fftfreq(N1)) * N1
    kx = jnp.arange(N2 // 2 + 1, dtype=jnp.float64)  # packed: 0 .. N2/2
    KZ, KY, KX = jnp.meshgrid(kz.astype(jnp.float64), ky.astype(jnp.float64), kx, indexing="ij")
    R = jnp.sqrt(KZ**2 + KY**2 + KX**2)
    return R.reshape(-1)


def radial_band_limit_half(v_flat_half: jnp.ndarray, volume_shape: Sequence[int], k_max: float) -> jnp.ndarray:
    """Zero out half-volume entries whose radial frequency exceeds `k_max`."""
    R = half_volume_radial_index(volume_shape)
    mask = (R <= float(k_max)).astype(v_flat_half.dtype)
    return v_flat_half * mask


def radial_band_limit_half_batch(U_flat_half: jnp.ndarray, volume_shape: Sequence[int], k_max: float) -> jnp.ndarray:
    R = half_volume_radial_index(volume_shape)
    mask = (R <= float(k_max)).astype(U_flat_half.dtype)
    return U_flat_half * mask[None, :]


# ---------------------------------------------------------------------------
# Real-space orthonormalization on half-volume rows
# ---------------------------------------------------------------------------


def real_volume_orthonormalize_half(
    U_flat_half: jnp.ndarray,
    weights: jnp.ndarray,
    volume_size: int,
    *,
    ridge: float = 1e-12,
) -> jnp.ndarray:
    """Orthonormalize the rows of `U_flat_half` so that the corresponding
    real volumes are orthonormal in the real-space inner product.

    Parameters
    ----------
    U_flat_half : (q, N_half) complex128
        Rows are flat half-volume rfft vectors.
    weights : (N_half,) float64
        Hermitian rfft weights from `make_half_volume_weights`.
    volume_size : int
        Total number of real-space voxels (`prod(volume_shape)`,
        **not** `N_half`). This is the Parseval normalization.
    ridge : float
        Tikhonov ridge added to the weighted Gram before Cholesky.
        Inert in the well-conditioned regime; protects against
        rank-deficiency from `radial_band_limit_half` zeroing out
        a row.

    Returns
    -------
    U_new : (q, N_half) complex128
        Rows whose real-space Gram is `(1/volume_size) · I_q` weighted-
        Hermitian: `(U_new * w) @ U_new^H / volume_size = I_q`.
    """
    if U_flat_half.ndim != 2:
        raise ValueError(f"expected (q, N_half), got shape {U_flat_half.shape}")
    if weights.ndim != 1 or weights.shape[0] != U_flat_half.shape[1]:
        raise ValueError(f"weights shape {weights.shape} incompatible with U_flat_half {U_flat_half.shape}")

    q = U_flat_half.shape[0]
    Uw = U_flat_half * weights[None, :].astype(U_flat_half.dtype)
    G = (Uw @ U_flat_half.conj().T).real / float(volume_size)  # (q, q) real-symm
    G = G + ridge * jnp.eye(q, dtype=G.dtype)
    L = jnp.linalg.cholesky(G)
    U_new = jax.scipy.linalg.solve_triangular(L, U_flat_half, lower=True)
    return U_new


# ---------------------------------------------------------------------------
# Diagnostics — used by the projection / orthonormalize tests
# ---------------------------------------------------------------------------


def half_real_space_gram(U_flat_half: jnp.ndarray, weights: jnp.ndarray, volume_size: int) -> jnp.ndarray:
    """Compute the real-space Gram of the rows of `U_flat_half` in the
    half-volume layout. Should equal `I_q` after orthonormalization."""
    Uw = U_flat_half * weights[None, :].astype(U_flat_half.dtype)
    return (Uw @ U_flat_half.conj().T).real / float(volume_size)


def half_to_real_volume(v_flat_half: jnp.ndarray, volume_shape: Sequence[int]) -> jnp.ndarray:
    """Decode a half-volume flat vector back to its real-space volume.

    Uses `get_idft3_real` (`recovar/core/fourier_transform_utils.py:464`).
    Returns a real-valued `(N0, N1, N2)` array.
    """
    N0, N1, N2 = (int(s) for s in volume_shape)
    half_grid = v_flat_half.reshape((N0, N1, N2 // 2 + 1))
    return ftu.get_idft3_real(half_grid, volume_shape=tuple(volume_shape))


def half_to_real_volume_batch(U_flat_half: jnp.ndarray, volume_shape: Sequence[int]) -> jnp.ndarray:
    """Decode each row of a `(q, N_half)` half-volume matrix to a real
    volume. Returns `(q, N0, N1, N2)` real array.
    """
    return jax.vmap(lambda v: half_to_real_volume(v, volume_shape))(U_flat_half)


def real_volume_to_half(real_vol: jnp.ndarray, volume_shape: Sequence[int]) -> jnp.ndarray:
    """Forward FT a real-space volume into half-volume rfft layout, flat.

    Inverse of `half_to_real_volume`.
    """
    half_grid = ftu.get_dft3_real(real_vol.reshape(tuple(volume_shape)))
    return half_grid.reshape(-1)
