"""Exact local sufficient-statistics accumulation helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


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
    return probs_sum_t[..., None] * ctf2_over_nv[:, None, :]


@jax.jit
def flatten_bucket_rows(values):
    """Flatten a bucket's per-image rows into one row-major batch."""

    return values.reshape(values.shape[0] * values.shape[1], values.shape[-1])


@jax.jit
def flatten_bucket_rotations(rotations):
    """Flatten a bucket's per-image rotations into one row-major batch."""

    return rotations.reshape(rotations.shape[0] * rotations.shape[1], 3, 3)
