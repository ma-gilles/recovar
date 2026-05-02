"""Shared projection and noise kernels for dense/local EM paths."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from recovar import core
from recovar.core import fourier_transform_utils as ftu

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
    relion_texture_interp: bool = True,
):
    """Forward-slice one rotation block into half-spectrum image layout."""
    kwargs = {
        "half_image": True,
        "relion_texture_interp": relion_texture_interp,
    }
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


def _safe_read_relion_projector(data, zi, yi, xi, valid):
    zdim, ydim, xdim = data.shape
    zi_clip = jnp.clip(zi, 0, zdim - 1)
    yi_clip = jnp.clip(yi, 0, ydim - 1)
    xi_clip = jnp.clip(xi, 0, xdim - 1)
    values = data[zi_clip, yi_clip, xi_clip]
    return jnp.where(valid, values, jnp.zeros_like(values))


def _project_relion_projector_one(data, rotation, coords, output_max_r):
    """Project RELION ``Projector::data`` with Projector::project semantics."""

    zdim, ydim, xdim = data.shape
    start_y = -(ydim // 2)
    start_z = -(zdim // 2)
    ref_max_r = xdim - 2
    ref_max_r2 = jnp.asarray(ref_max_r * ref_max_r, dtype=jnp.float32)
    output_max_r2 = jnp.asarray(output_max_r * output_max_r, dtype=jnp.float32)

    x = coords[:, 0].astype(jnp.float32)
    y = coords[:, 1].astype(jnp.float32)
    in_output = x * x + y * y <= output_max_r2

    # RELION uses A.inv(); for rotation matrices this is A.T.
    ainv = jnp.swapaxes(rotation, -1, -2)
    xp = ainv[0, 0] * x + ainv[0, 1] * y
    yp = ainv[1, 0] * x + ainv[1, 1] * y
    zp = ainv[2, 0] * x + ainv[2, 1] * y

    in_ref = xp * xp + yp * yp + zp * zp <= ref_max_r2
    use_conj = xp < 0.0
    xp = jnp.where(use_conj, -xp, xp)
    yp = jnp.where(use_conj, -yp, yp)
    zp = jnp.where(use_conj, -zp, zp)

    x0 = jnp.floor(xp).astype(jnp.int32)
    y0 = jnp.floor(yp).astype(jnp.int32)
    z0 = jnp.floor(zp).astype(jnp.int32)
    fx = xp - x0.astype(jnp.float32)
    fy = yp - y0.astype(jnp.float32)
    fz = zp - z0.astype(jnp.float32)

    yi0 = y0 - start_y
    zi0 = z0 - start_z
    valid = (
        in_output
        & in_ref
        & (x0 >= 0)
        & (x0 + 1 < xdim)
        & (yi0 >= 0)
        & (yi0 + 1 < ydim)
        & (zi0 >= 0)
        & (zi0 + 1 < zdim)
    )

    d000 = _safe_read_relion_projector(data, zi0, yi0, x0, valid)
    d001 = _safe_read_relion_projector(data, zi0, yi0, x0 + 1, valid)
    d010 = _safe_read_relion_projector(data, zi0, yi0 + 1, x0, valid)
    d011 = _safe_read_relion_projector(data, zi0, yi0 + 1, x0 + 1, valid)
    d100 = _safe_read_relion_projector(data, zi0 + 1, yi0, x0, valid)
    d101 = _safe_read_relion_projector(data, zi0 + 1, yi0, x0 + 1, valid)
    d110 = _safe_read_relion_projector(data, zi0 + 1, yi0 + 1, x0, valid)
    d111 = _safe_read_relion_projector(data, zi0 + 1, yi0 + 1, x0 + 1, valid)

    dx00 = d000 + (d001 - d000) * fx
    dx01 = d100 + (d101 - d100) * fx
    dx10 = d010 + (d011 - d010) * fx
    dx11 = d110 + (d111 - d110) * fx
    dxy0 = dx00 + (dx10 - dx00) * fy
    dxy1 = dx01 + (dx11 - dx01) * fy
    values = dxy0 + (dxy1 - dxy0) * fz
    return jnp.where(use_conj, jnp.conj(values), values)


def project_relion_projector_data(
    projector_data,
    rotations_block,
    image_shape,
    projector_shape,
    *,
    max_r,
):
    """Forward-project RELION ``Projector::data`` into RECOVAR half-image order."""

    data = jnp.asarray(projector_data).reshape(tuple(int(x) for x in projector_shape))
    coords = jnp.asarray(ftu.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size=1, scaled=False))
    output_max_r = image_shape[0] // 2 - 1 if max_r is DEFAULT_PROJECTION_MAX_R or max_r is None else int(max_r)
    return jax.vmap(lambda rot: _project_relion_projector_one(data, rot, coords, output_max_r))(rotations_block)


def compute_projections_block(
    volume,
    rotations_block,
    image_shape,
    volume_shape,
    disc_type,
    *,
    max_r=DEFAULT_PROJECTION_MAX_R,
    return_abs2: bool = True,
    relion_texture_interp: bool = True,
    relion_projector_shape=None,
):
    """Forward-slice one rotation block and optionally compute ``|proj|^2``.

    Dense scoring and noise accumulation need ``|proj|^2`` repeatedly enough to
    materialize it. Exact-local paths can pass ``return_abs2=False`` and compute
    norms on demand when that saves memory.
    """
    if relion_projector_shape is not None:
        proj_half = project_relion_projector_data(
            volume,
            rotations_block,
            image_shape,
            relion_projector_shape,
            max_r=max_r,
        )
    else:
        proj_half = project_half_spectrum(
            volume,
            rotations_block,
            image_shape,
            volume_shape,
            disc_type,
            max_r=max_r,
            relion_texture_interp=relion_texture_interp,
        )
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
    """Accumulate RELION-style posterior-weighted noise for one rotation block.

    Uses the decomposition::

        E_w[|CTF*proj - img|^2] = E_w[|CTF*proj|^2] - 2*Re(E_w[conj(img)*CTF*proj]) + |img|^2
                                 =     A2            -           2*XA                  + P_img

    ``P_img`` is handled by the caller (image-only, no rotation dependence).
    This function computes the ``A2 - 2*XA`` contribution from one rotation
    block, binned to resolution shells. Inputs are un-Hermitian-weighted packed
    half spectra because RELION's noise update bins over its FFTW half-plane
    convention directly.
    """
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
