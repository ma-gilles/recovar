"""Exact local sufficient-statistics accumulation helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

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
    p0 = (-i0) % n0
    p1 = (-i1) % n1
    plane = vol[:, :, 0]
    partner = jnp.conj(plane[p0[:, None], p1[None, :]])
    summed = plane + partner
    self_partner = (p0[:, None] == i0[:, None]) & (p1[None, :] == i1[None, :])
    plane = jnp.where(self_partner, plane, summed)
    return vol.at[:, :, 0].set(plane).reshape(-1)
