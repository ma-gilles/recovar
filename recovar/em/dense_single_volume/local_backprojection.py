"""Exact local sufficient-statistics accumulation helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils


@jax.jit
def compute_local_weighted_sums(probs, shifted):
    """Compute weighted image sums for one exact local bucket.

    probs: (B, R, T)
    shifted: (B, T, N)
    returns: (B, R, N)
    """

    return jnp.matmul(probs, shifted)


@jax.jit
def compute_local_ctf_sums(probs, ctf2_over_nv):
    """Compute weighted CTF^2/noise sums for one exact local bucket."""

    probs_sum_t = jnp.sum(probs, axis=-1)  # (B, R)
    return compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv)


@jax.jit
def compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv):
    """Compute weighted CTF^2/noise sums from precomputed rotation posterior sums."""

    return jnp.where(
        probs_sum_t[..., None] != 0.0,
        probs_sum_t[..., None] * ctf2_over_nv[:, None, :],
        0.0,
    )


@jax.jit
def flatten_bucket_rows(values):
    """Flatten a bucket's per-image rows into one row-major batch."""

    return values.reshape(values.shape[0] * values.shape[1], values.shape[-1])


@jax.jit
def flatten_bucket_rotations(rotations):
    """Flatten a bucket's per-image rotations into one row-major batch."""

    return rotations.reshape(rotations.shape[0] * rotations.shape[1], 3, 3)


def enforce_relion_half_volume_x0_hermitian(volume_flat, full_volume_shape):
    """Match RELION BackProjector::enforceHermitianSymmetry on x=0 plane."""

    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(full_volume_shape)
    vol = jnp.asarray(volume_flat).reshape(half_shape)
    n0, n1, _ = half_shape
    i0 = jnp.arange(n0, dtype=jnp.int32)
    i1 = jnp.arange(n1, dtype=jnp.int32)
    # RELION pairs logical Xmipp-origin coordinates (z, y) with (-z, -y).
    # In RECOVAR's centered array convention this is (N - (N % 2) - i) % N;
    # odd RELION BPref grids therefore use N-1-i, not the unshifted -i.
    p0 = (n0 - (n0 % 2) - i0) % n0
    p1 = (n1 - (n1 % 2) - i1) % n1
    plane = vol[:, :, 0]
    partner = jnp.conj(plane[p0[:, None], p1[None, :]])
    summed = plane + partner
    self_partner = (p0[:, None] == i0[:, None]) & (p1[None, :] == i1[None, :])
    plane = jnp.where(self_partner, plane, summed)
    return vol.at[:, :, 0].set(plane).reshape(-1)


def enforce_relion_half_volume_x0_hermitian_host(volume_flat, full_volume_shape):
    """Host implementation of RELION x=0 Hermitian-plane enforcement.

    The device path updates a single plane with ``.at[..., 0].set(...)`` but may
    still allocate another full packed half-volume.  Large RELION BPref grids
    already repack through host memory downstream, so handling the plane update
    here avoids a transient GPU allocation without changing the arithmetic.
    """

    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(full_volume_shape)
    host = np.asarray(jax.device_get(volume_flat))
    if isinstance(volume_flat, np.ndarray) or not host.flags.writeable:
        host = host.copy()
    vol = host.reshape(half_shape)
    n0, n1, _ = half_shape
    i0 = np.arange(n0, dtype=np.int32)
    i1 = np.arange(n1, dtype=np.int32)
    p0 = (n0 - (n0 % 2) - i0) % n0
    p1 = (n1 - (n1 % 2) - i1) % n1
    plane = vol[:, :, 0]
    partner = np.conj(plane[np.ix_(p0, p1)])
    summed = plane + partner
    self_partner = (p0[:, None] == i0[:, None]) & (p1[None, :] == i1[None, :])
    vol[:, :, 0] = np.where(self_partner, plane, summed)
    return vol.reshape(-1)
