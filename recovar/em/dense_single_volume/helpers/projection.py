"""Shared projection and noise kernels for dense/local EM paths."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from recovar import core

DEFAULT_PROJECTION_MAX_R = object()


def project_half_spectrum(
    volume,
    rotations_block,
    image_shape,
    volume_shape,
    disc_type,
    *,
    half_volume: bool = False,
    max_r=DEFAULT_PROJECTION_MAX_R,
):
    """Forward-slice one rotation block into half-spectrum image layout."""
    kwargs = {"half_image": True}
    if half_volume:
        kwargs["half_volume"] = True
    if max_r is not DEFAULT_PROJECTION_MAX_R and max_r != "auto":
        kwargs["max_r"] = max_r
    return core.slice_volume(
        volume,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        **kwargs,
    )


def compute_projections_block(
    volume,
    rotations_block,
    image_shape,
    volume_shape,
    disc_type,
    *,
    max_r=DEFAULT_PROJECTION_MAX_R,
    return_abs2: bool = True,
):
    """Forward-slice one rotation block and optionally compute ``|proj|^2``."""
    proj_half = project_half_spectrum(
        volume,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        max_r=max_r,
    )
    # TODO: WE SHOULD THINK ABOUT WHETHER STORING SQUARES IS WORTH IT.
    proj_abs2_half = jnp.abs(proj_half) ** 2 if return_abs2 else None
    return proj_half, proj_abs2_half


@partial(jax.jit, static_argnums=(6, 7))
def compute_noise_block(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    shell_count,
    return_split: bool = True,
):
    # TODO: QUESTION? Projections are unweighted by half_weights. Confirm that
    # this matches RELION and the math docs.
    # TODO: Revisit whether carrying both projections and |projection|^2 is
    # worth the memory cost.
    # TODO: Confirm whether Hermitian weights should participate in shell noise
    # binning even if RELION omits them.
    """Accumulate RELION-style posterior-weighted noise for one rotation block."""
    ctf_probs_raw = ctf_probs * noise_variance_half
    a2 = jnp.sum(proj_abs2_half * ctf_probs_raw, axis=0)

    cross = jnp.sum(proj_half * jnp.conj(summed_masked), axis=0)
    xa = noise_variance_half * cross.real
    block_noise = a2 - 2.0 * xa

    noise_shells = jnp.zeros(shell_count, dtype=jnp.float32)
    noise_shells = noise_shells.at[shell_indices].add(block_noise.astype(jnp.float32))
    if not return_split:
        zeros = jnp.zeros(shell_count, dtype=jnp.float32)
        return noise_shells, zeros, zeros
    a2_shells = jnp.zeros(shell_count, dtype=jnp.float32)
    a2_shells = a2_shells.at[shell_indices].add(a2.astype(jnp.float32))
    xa_shells = jnp.zeros(shell_count, dtype=jnp.float32)
    xa_shells = xa_shells.at[shell_indices].add(xa.astype(jnp.float32))
    return noise_shells, a2_shells, xa_shells
