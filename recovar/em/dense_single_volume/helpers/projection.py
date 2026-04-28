"""Shared projection and noise kernels for dense/local EM paths."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from recovar import core

DEFAULT_PROJECTION_MAX_R = object()


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
    ## TODO: QUESTION? Projections (unweighted by half_weights). IS THIS RIGHT? ARE DOCS WRONG? I THOUGHT RELION DID NOT USE WEIGHTS AT ALL?

    ## TODO: SHOULD WE REALLY BE KEEPING AROUDN BOTH PROJ AND |PROJ|^2 THROUGHOUT CODE? SEEMS WASTEFUL IN MEMORY?

    """Accumulate RELION-style posterior-weighted noise for one rotation block.

    Uses the decomposition::

        E_w[|CTF*proj - img|^2] = E_w[|CTF*proj|^2] - 2*Re(E_w[conj(img)*CTF*proj]) + |img|^2
                                 =     A2            -           2*XA                  + P_img

    ``P_img`` is handled by the caller (image-only, no rotation dependence).
    This function computes the ``A2 - 2*XA`` contribution from one rotation
    block, binned to resolution shells.

    The key identity: since CTF is real-valued,
    ``conj(raw_img_shifted) * CTF = conj(shifted_half) * sigma2``,
    so the XA GEMM output ``P @ shifted_masked`` can be reused.

    Parameters
    ----------
    proj_half : (rot_block, N) complex
        Projections (unweighted by half_weights).
    proj_abs2_half : (rot_block, N) float
        ``|proj|^2``.
    summed_masked : (rot_block, N) complex
        ``P @ shifted_masked_half`` -- masked-image M-step GEMM output.
    ctf_probs : (rot_block, N) float
        ``probs_sum_t.T @ (CTF^2 / noise_variance)`` -- already computed
        for Ft_ctf.
    noise_variance_half : (N,) float
        Per-pixel noise variance in half-spectrum layout.
    shell_indices : (N,) int32
        Radial shell index per half-spectrum pixel.
    shell_count : int (static)
        Number of resolution shells.

    Returns
    -------
    noise_shells : (shell_count,) float
        ``sum_{k in shell} (A2(k) - 2*XA(k))`` contribution from this block.
    a2_shells, xa_shells : (shell_count,) float
        Diagnostic split terms. Returned as zeros when ``return_split`` is
        false, avoiding two extra scatter reductions in normal runs.
    """
    ctf_probs_raw = ctf_probs * noise_variance_half
    A2 = jnp.sum(proj_abs2_half * ctf_probs_raw, axis=0)

    cross = jnp.sum(proj_half * jnp.conj(summed_masked), axis=0)
    XA = noise_variance_half * cross.real

    block_noise = A2 - 2.0 * XA

    ## TODO: IS THIS REALLY WHAT RELION DOES? WHY ARE STORING THE MIDDLE TERMS LIKE A2 AND XA?

    # Bin to resolution shells (no Hermitian weights -- matching RELION)
    ## TODO SHOULD THERE BE HERMITIAN WEIGHTS AT ALL, EVEN IF RELION USED THEM? NOT COMPLEETELY SURE, TRIPLE CHECK
    noise_shells = jnp.zeros(shell_count, dtype=jnp.float32)
    noise_shells = noise_shells.at[shell_indices].add(block_noise.astype(noise_shells.dtype))
    if not return_split:
        zeros = jnp.zeros(shell_count, dtype=jnp.float32)
        return noise_shells, zeros, zeros
    a2_shells = jnp.zeros(shell_count, dtype=jnp.float32)
    a2_shells = a2_shells.at[shell_indices].add(A2.astype(a2_shells.dtype))
    xa_shells = jnp.zeros(shell_count, dtype=jnp.float32)
    xa_shells = xa_shells.at[shell_indices].add(XA.astype(xa_shells.dtype))
    return noise_shells, a2_shells, xa_shells


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
    """Forward-slice one rotation block in half-spectrum layout."""

    if max_r is DEFAULT_PROJECTION_MAX_R:
        proj_half = core.slice_volume(
            volume,
            rotations_block,
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
        )
    else:
        proj_half = core.slice_volume(
            volume,
            rotations_block,
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
            max_r=max_r,
        )
    ## TODO: WE SHOULD THINK ABOUT WHETHER STORING SQUARES IS WORTH IT.
    proj_abs2_half = jnp.abs(proj_half) ** 2 if return_abs2 else None
    return proj_half, proj_abs2_half


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
    """Forward-slice one rotation block and return only the half-spectrum projection."""

    if max_r is DEFAULT_PROJECTION_MAX_R or max_r == "auto":
        return core.slice_volume(
            volume,
            rotations_block,
            image_shape,
            volume_shape,
            disc_type,
            half_volume=half_volume,
            half_image=True,
        )
    return core.slice_volume(
        volume,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        half_volume=half_volume,
        half_image=True,
        max_r=max_r,
    )
