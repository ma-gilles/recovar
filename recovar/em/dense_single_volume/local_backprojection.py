"""Exact local sufficient-statistics accumulation helpers."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils


_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION_ENV = (
    "RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION"
)


def relion_x_half_sequential_translation_reduction_enabled() -> bool:
    """Return whether the diagnostic RELION-order translation reduction is enabled."""

    raw = os.environ.get(_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION_ENV)
    return raw is not None and raw.strip().lower() not in {"", "0", "false", "no", "off"}


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
def compute_relion_f32_sequential_mstep_sums(probs, shifted, ctf2_over_nv):
    """Reduce translations in RELION GPU float32 order for an x-half diagnostic.

    RELION's GPU backprojector visits translations in increasing index order
    inside each orientation/pixel thread, carrying both the complex numerator
    and positive CTF/noise denominator in ``XFLOAT`` (float32 in the reference
    build).  Keep this separate from the production GEMM reduction so the
    diagnostic cannot change default behavior.
    """

    probs_f32 = jnp.asarray(probs, dtype=jnp.float32)
    shifted_c64 = jnp.asarray(shifted, dtype=jnp.complex64)
    ctf2_f32 = jnp.asarray(ctf2_over_nv, dtype=jnp.float32)
    batch, n_rot, n_trans = probs_f32.shape
    n_pixels = shifted_c64.shape[-1]
    numerator0 = jnp.zeros((batch, n_rot, n_pixels), dtype=jnp.complex64)
    denominator0 = jnp.zeros((batch, n_rot, n_pixels), dtype=jnp.float32)

    def add_translation(trans_idx, carry):
        numerator, denominator = carry
        weight = probs_f32[:, :, trans_idx, None]
        numerator = numerator + weight * shifted_c64[:, None, trans_idx, :]
        denominator = denominator + weight * ctf2_f32[:, None, :]
        return numerator, denominator

    return jax.lax.fori_loop(0, n_trans, add_translation, (numerator0, denominator0))


def compute_local_mstep_sums(
    probs,
    shifted,
    ctf2_over_nv,
    *,
    relion_x_half: bool,
    default_probs_sum_t=None,
    sequential_translation_reduction: bool | None = None,
):
    """Compute numerator/denominator sums, optionally using the x-half diagnostic."""

    use_sequential_reduction = (
        relion_x_half_sequential_translation_reduction_enabled()
        if sequential_translation_reduction is None
        else bool(sequential_translation_reduction)
    )
    if relion_x_half and use_sequential_reduction:
        return compute_relion_f32_sequential_mstep_sums(probs, shifted, ctf2_over_nv)
    denominator = (
        compute_local_ctf_sums(probs, ctf2_over_nv)
        if default_probs_sum_t is None
        else compute_local_ctf_sums_from_probs_sum_t(default_probs_sum_t, ctf2_over_nv)
    )
    return compute_local_weighted_sums(probs, shifted), denominator


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
