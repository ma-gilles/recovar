"""Projection and noise primitives shared by dense/global and local EM paths."""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar import core
from recovar.cuda_backproject import cuda_available as _cuda_projection_available
from recovar.cuda_backproject import project_indexed
from recovar.em.dense_single_volume.helpers.half_spectrum import bin_shell_values_jax

DEFAULT_PROJECTION_MAX_R = object()
_RELION_PROJECTOR_TEXTURE_ENV = "RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP"


@partial(jax.jit, static_argnums=(2, 3, 4))
def project_relion_projector_half_spectrum(
    volume_relion_half,
    rotations_block,
    image_shape,
    r_max: int,
    padding_factor: int = 1,
):
    """Forward-project RELION Projector storage into full half-image layout.

    ``volume_relion_half`` is RELION's ``Projector::data`` array, not
    recovar's centered full Fourier volume. This path is used by InitialModel
    parity code where RELION's pass-1/pass-2 scores must consume the exact
    ``PPref`` representation.
    """

    from recovar.core.relion_project import relion_project_half

    image_size = int(image_shape[0])
    project_one = lambda R: relion_project_half(
        volume_relion_half,
        R,
        image_size,
        int(r_max),
        int(padding_factor),
    )
    proj_fftw = jax.vmap(project_one)(rotations_block)

    return proj_fftw.reshape((rotations_block.shape[0], -1))


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def project_relion_projector_half_spectrum_centered_rows(
    volume_relion_half,
    rotations_block,
    image_shape,
    r_max: int,
    padding_factor: int = 1,
    projector_output_size: int | None = None,
) -> jnp.ndarray:
    """Project RELION ``PPref`` data and return recovar-centered row order.

    ``relion_project_half`` is a raw port of RELION's Projector and consumes
    RELION's FFTW-row projector matrix directly.  The dense E-step scorer
    supplies the same rotation matrices used by RECOVAR's centered half-spectrum
    scoring path, whose image-plane convention is the transpose at this
    handoff.  Apply that conversion here, then shift rows into RECOVAR's
    centered order.
    """

    image_size = int(image_shape[0])
    projector_image_size = int(r_max) * 2 if projector_output_size is None else int(projector_output_size)
    if projector_image_size <= 0 or projector_image_size > image_size:
        projector_image_size = image_size
    projector_rotations = jnp.swapaxes(rotations_block, -1, -2)
    proj_fftw = project_relion_projector_half_spectrum(
        volume_relion_half,
        projector_rotations,
        (projector_image_size, projector_image_size),
        int(r_max),
        int(padding_factor),
    ).reshape((rotations_block.shape[0], projector_image_size, projector_image_size // 2 + 1))
    if projector_image_size == image_size:
        row_order = jnp.fft.fftshift(jnp.arange(image_size, dtype=jnp.int32))
        return proj_fftw[:, row_order, :].reshape((rotations_block.shape[0], -1))

    crop_rows = jnp.arange(projector_image_size, dtype=jnp.int32)
    crop_ky = jnp.where(
        crop_rows <= projector_image_size // 2,
        crop_rows,
        crop_rows - projector_image_size,
    )
    full_rows = crop_ky + image_size // 2
    crop_cols = jnp.arange(projector_image_size // 2 + 1, dtype=jnp.int32)
    full_indices = (full_rows[:, None] * (image_size // 2 + 1) + crop_cols[None, :]).reshape(-1)
    proj_full = jnp.zeros(
        (rotations_block.shape[0], image_size * (image_size // 2 + 1)),
        dtype=proj_fftw.dtype,
    )
    return proj_full.at[:, full_indices].set(proj_fftw.reshape((rotations_block.shape[0], -1)))


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def project_relion_projector_half_spectrum_centered_rows_at_indices(
    volume_relion_half,
    rotations_block,
    image_shape,
    r_max: int,
    padding_factor: int = 1,
    projector_output_size: int | None = None,
    pixel_indices=None,
) -> jnp.ndarray:
    """Project RELION ``PPref`` data and gather centered-row half-image pixels.

    This is equivalent to ``project_relion_projector_half_spectrum_centered_rows(
    ...)[..., pixel_indices]`` but avoids building the full centered half-image
    when RELION's current image size only needs a cropped Fourier window.
    """

    image_size = int(image_shape[0])
    projector_image_size = int(r_max) * 2 if projector_output_size is None else int(projector_output_size)
    if projector_image_size <= 0 or projector_image_size > image_size:
        projector_image_size = image_size

    projector_rotations = jnp.swapaxes(rotations_block, -1, -2)
    proj_fftw = project_relion_projector_half_spectrum(
        volume_relion_half,
        projector_rotations,
        (projector_image_size, projector_image_size),
        int(r_max),
        int(padding_factor),
    ).reshape((rotations_block.shape[0], projector_image_size, projector_image_size // 2 + 1))

    indices = jnp.asarray(pixel_indices, dtype=jnp.int32)
    full_x_half = image_size // 2 + 1
    full_rows = indices // full_x_half
    cols = indices - full_rows * full_x_half
    if projector_image_size == image_size:
        row_order = jnp.fft.fftshift(jnp.arange(image_size, dtype=jnp.int32))
        projector_rows = row_order[full_rows]
    else:
        ky = full_rows - image_size // 2
        projector_rows = jnp.where(ky >= 0, ky, ky + projector_image_size)
    projector_x_half = projector_image_size // 2 + 1
    projector_flat_indices = projector_rows * projector_x_half + cols
    return proj_fftw.reshape((rotations_block.shape[0], -1))[:, projector_flat_indices]


def _validate_centered_relion_projector_pixel_indices(
    pixel_indices,
    *,
    image_shape,
    projector_output_size: int,
) -> None:
    """Fail early if compact RELION projector indices cannot live in the crop."""

    indices = np.asarray(pixel_indices, dtype=np.int64)
    if indices.size == 0:
        return
    image_size = int(image_shape[0])
    full_x_half = image_size // 2 + 1
    rows = indices // full_x_half
    cols = indices - rows * full_x_half
    ky = rows - image_size // 2
    projector_x_half = int(projector_output_size) // 2 + 1
    min_ky = -(int(projector_output_size) // 2 - 1)
    max_ky = int(projector_output_size) // 2
    valid = (ky >= min_ky) & (ky <= max_ky) & (cols >= 0) & (cols < projector_x_half)
    if not np.all(valid):
        bad = indices[~valid][:8].tolist()
        raise ValueError(
            "centered RELION projector compact indices exceed projector crop "
            f"(image_shape={tuple(image_shape)}, projector_output_size={int(projector_output_size)}, "
            f"bad_indices={bad})"
        )


def relion_projector_half_to_texture_full(volume_relion_half: jax.Array) -> jax.Array:
    """Embed RELION ``Projector::data[z,y,x>=0]`` for CUDA texture staging.

    The CUDA texture projector only stages the non-negative model-x half from
    the centered full volume.  Consequently the negative-x half can remain
    zero: RELION handles negative projected x by flipping all coordinates and
    conjugating the sampled positive-x value.
    """

    volume_relion_half = jnp.asarray(volume_relion_half)
    pad_z, pad_y, half_x = volume_relion_half.shape
    if pad_z != pad_y or pad_z % 2 != 1 or half_x != pad_z // 2 + 1:
        raise ValueError(
            "RELION texture projection expects odd Projector::data shape "
            f"(pad, pad, pad//2+1), got {volume_relion_half.shape}",
        )
    center = pad_z // 2
    full = jnp.zeros((pad_z, pad_z, pad_z), dtype=volume_relion_half.dtype)
    return full.at[center:, :, :].set(jnp.transpose(volume_relion_half, (2, 1, 0)))


def _relion_projector_texture_enabled(
    volume_relion_half,
    *,
    r_max: int,
    padding_factor: int,
    enabled: bool | None = None,
) -> bool:
    if enabled is None:
        token = os.environ.get(_RELION_PROJECTOR_TEXTURE_ENV, "0").strip().lower()
        if token in {"0", "false", "no", "off"}:
            return False
        if token not in {"1", "true", "yes", "on"}:
            raise ValueError(f"Unsupported {_RELION_PROJECTOR_TEXTURE_ENV}={token!r}")
    elif not bool(enabled):
        return False
    shape = tuple(int(value) for value in volume_relion_half.shape)
    expected_pad = 2 * (int(float(padding_factor) * float(r_max) + 0.5) + 1) + 1
    return (
        _cuda_projection_available()
        and jnp.dtype(volume_relion_half.dtype) == jnp.dtype(jnp.complex64)
        and len(shape) == 3
        and shape == (expected_pad, expected_pad, expected_pad // 2 + 1)
    )


def _texture_centered_crop_to_full(
    projection_crop,
    *,
    image_shape,
    projector_output_size: int,
):
    """Scatter a centered even-size CUDA projection into the full image box."""

    image_size = int(image_shape[0])
    crop_size = int(projector_output_size)
    crop = projection_crop.reshape((projection_crop.shape[0], crop_size, crop_size // 2 + 1))
    crop_rows = jnp.arange(crop_size, dtype=jnp.int32)
    crop_ky = jnp.where(crop_rows == 0, crop_size // 2, crop_rows - crop_size // 2)
    crop_cols = jnp.arange(crop_size // 2 + 1, dtype=jnp.int32)
    output_radius = crop_size // 2
    output_disk = crop_ky[:, None] ** 2 + crop_cols[None, :] ** 2 <= output_radius**2
    # RELION clips projections to min(PPref.mdlMaxR, image_half_width-1).
    # The texture kernel already enforces the PPref/model sphere; apply the
    # independent current-image disk here before embedding the crop.
    crop = jnp.where(output_disk[None, :, :], crop, jnp.zeros((), dtype=crop.dtype))
    if crop_size == image_size:
        return crop.reshape((projection_crop.shape[0], -1))
    # Row zero is the even-box Nyquist row (+N/2 == -N/2); remaining rows
    # proceed from -N/2+1 through +N/2-1 in centered order.
    full_rows = crop_ky + image_size // 2
    full_indices = (full_rows[:, None] * (image_size // 2 + 1) + crop_cols[None, :]).reshape(-1)
    full = jnp.zeros(
        (projection_crop.shape[0], image_size * (image_size // 2 + 1)),
        dtype=projection_crop.dtype,
    )
    return full.at[:, full_indices].set(crop.reshape((projection_crop.shape[0], -1)))


def _project_relion_projector_texture(
    volume_relion_half,
    rotations_block,
    image_shape,
    *,
    r_max: int,
    projector_output_size: int,
):
    """Project one RELION ``PPref`` block with RELION's CUDA texture arithmetic."""

    projector_full = relion_projector_half_to_texture_full(volume_relion_half)
    pad_size = int(projector_full.shape[0])
    projection_crop = project_half_spectrum(
        projector_full.reshape(-1),
        rotations_block,
        (int(projector_output_size), int(projector_output_size)),
        (pad_size, pad_size, pad_size),
        "linear_interp",
        max_r=float(r_max),
        relion_texture_interp=True,
    )
    return _texture_centered_crop_to_full(
        projection_crop,
        image_shape=image_shape,
        projector_output_size=int(projector_output_size),
    )


def compute_relion_projector_projections_block(
    volume_relion_half,
    rotations_block,
    image_shape,
    *,
    r_max: int,
    padding_factor: int = 1,
    return_abs2: bool = True,
    centered_rows: bool = False,
    dense_scale: bool = False,
    projector_output_size: int | None = None,
    pixel_indices=None,
    relion_texture_interp: bool | None = None,
):
    """Project precomputed RELION ``PPref`` data for one rotation block.

    Strict parity defaults to the manual/JAX interpolator.  The CUDA texture
    path remains available with ``RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP=1``
    for diagnostics; it currently changes global K=1/K-class winners and
    degrades RELION map FSC on the qualified parity fixtures.
    """

    image_size = int(image_shape[0])
    resolved_output_size = int(r_max) * 2 if projector_output_size is None else int(projector_output_size)
    if resolved_output_size <= 0 or resolved_output_size > image_size:
        resolved_output_size = image_size
    use_texture = _relion_projector_texture_enabled(
        volume_relion_half,
        r_max=int(r_max),
        padding_factor=int(padding_factor),
        enabled=relion_texture_interp,
    )

    if use_texture:
        if not centered_rows and pixel_indices is not None:
            raise ValueError("pixel_indices are only supported with centered_rows=True")
        if pixel_indices is not None and not isinstance(pixel_indices, jax.core.Tracer):
            _validate_centered_relion_projector_pixel_indices(
                pixel_indices,
                image_shape=image_shape,
                projector_output_size=resolved_output_size,
            )
        proj_centered = _project_relion_projector_texture(
            volume_relion_half,
            rotations_block,
            image_shape,
            r_max=int(r_max),
            projector_output_size=resolved_output_size,
        )
        if centered_rows:
            proj_half = proj_centered if pixel_indices is None else proj_centered[:, pixel_indices]
        else:
            proj_half = jnp.fft.ifftshift(
                proj_centered.reshape((proj_centered.shape[0], image_size, image_size // 2 + 1)),
                axes=1,
            ).reshape((proj_centered.shape[0], -1))

    elif pixel_indices is not None:
        if not centered_rows:
            raise ValueError("pixel_indices are only supported with centered_rows=True")
        if resolved_output_size < image_size and not isinstance(pixel_indices, jax.core.Tracer):
            _validate_centered_relion_projector_pixel_indices(
                pixel_indices,
                image_shape=image_shape,
                projector_output_size=resolved_output_size,
            )
        proj_half = project_relion_projector_half_spectrum_centered_rows_at_indices(
            volume_relion_half,
            rotations_block,
            image_shape,
            int(r_max),
            int(padding_factor),
            projector_output_size,
            pixel_indices,
        )
    elif centered_rows:
        proj_half = project_relion_projector_half_spectrum_centered_rows(
            volume_relion_half,
            rotations_block,
            image_shape,
            int(r_max),
            int(padding_factor),
            projector_output_size,
        )
    else:
        proj_half = project_relion_projector_half_spectrum(
            volume_relion_half,
            rotations_block,
            image_shape,
            int(r_max),
            int(padding_factor),
        )
    if dense_scale:
        token = (os.environ.get("RECOVAR_DENSE_MEANS_SCALE") or "-N2").strip()
        n = int(image_shape[0])
        scale = {"-N2": -(n**2), "N2": float(n**2)}.get(token)
        if scale is None:
            raise ValueError(f"Unsupported RECOVAR_DENSE_MEANS_SCALE={token!r}")
        proj_half = proj_half * scale
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
    relion_texture_interp: bool = True,
    force_jax: bool = False,
):
    """Forward-slice one rotation block into half-spectrum image layout."""
    if force_jax:
        order = core.decide_order(disc_type)
        if order > 1:
            raise ValueError("force_jax projection is only supported for nearest/linear interpolation")
        from recovar.core import relion_interp

        resolved_max_r = core._default_max_r(image_shape) if max_r is DEFAULT_PROJECTION_MAX_R else max_r
        return relion_interp.project(
            volume,
            rotations_block,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=True,
            max_r=resolved_max_r,
        )

    kwargs = {
        "half_image": True,
        "relion_texture_interp": relion_texture_interp,
    }
    if half_volume:
        kwargs["half_volume"] = True
    if max_r is not DEFAULT_PROJECTION_MAX_R:
        kwargs["max_r"] = max_r
    return core.slice_volume(
        volume,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        **kwargs,
    )


def project_indexed_half_spectrum(
    volume,
    pixel_indices,
    rotations_block,
    image_shape,
    volume_shape,
    disc_type,
    *,
    half_volume: bool = False,
    max_r=DEFAULT_PROJECTION_MAX_R,
):
    """Forward-slice selected packed half-spectrum pixels into compact rows."""

    order = core.decide_order(disc_type)
    if order > 1:
        raise ValueError("indexed projection is only supported for nearest/linear interpolation")
    return project_indexed(
        volume,
        pixel_indices,
        rotations_block,
        image_shape,
        volume_shape,
        order=order,
        half_volume=half_volume,
        half_image=True,
        max_r=None if max_r is DEFAULT_PROJECTION_MAX_R else max_r,
    )


def indexed_projection_available() -> bool:
    """Return whether the CUDA indexed projection path can be used."""

    return _cuda_projection_available()


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
    force_jax: bool = False,
):
    """Forward-slice one rotation block and optionally compute ``|proj|^2``.

    Dense scoring and noise accumulation need ``|proj|^2`` repeatedly enough to
    materialize it. Exact-local paths can pass ``return_abs2=False`` and compute
    norms on demand when that saves memory.
    """
    proj_half = project_half_spectrum(
        volume,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        max_r=max_r,
        relion_texture_interp=relion_texture_interp,
        force_jax=force_jax,
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
    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half, 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    a2 = jnp.sum(a2_terms, axis=0)

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    cross = jnp.sum(cross_terms, axis=0)
    xa = jnp.where(cross.real != 0.0, noise_variance_half * cross.real, 0.0)
    block_noise = a2 - 2.0 * xa

    noise_shells = bin_shell_values_jax(block_noise.astype(jnp.float32), shell_indices, shell_count)
    if not return_split:
        zeros = jnp.zeros(shell_count, dtype=jnp.float32)
        return noise_shells, zeros, zeros
    a2_shells = bin_shell_values_jax(a2.astype(jnp.float32), shell_indices, shell_count)
    xa_shells = bin_shell_values_jax(xa.astype(jnp.float32), shell_indices, shell_count)
    return noise_shells, a2_shells, xa_shells


@jax.jit
def compute_norm_residual_per_image(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
):
    """Return RELION norm-correction residual terms per image.

    This is the same ``A2 - 2*XA`` contribution as :func:`compute_noise_block`,
    but summed per image instead of binned over shells.  The caller adds the
    image-power term once per image.
    """

    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half[None, None, :], 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    a2_per_image = jnp.sum(a2_terms, axis=(1, 2))

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    xa_terms = noise_variance_half[None, None, :] * cross_terms.real
    xa_per_image = jnp.sum(xa_terms, axis=(1, 2))
    return (a2_per_image - 2.0 * xa_per_image).astype(jnp.float32)


@jax.jit
def compute_scale_correction_terms_per_image(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    old_scale,
):
    """Return RELION group-scale XA/AA sufficient statistics per image.

    The inputs are the same retained M-step support tensors used by
    ``compute_norm_residual_per_image``.  The current E-step tensors already
    include the old group scale in ``XA`` and ``AA``; RELION's scale update
    accumulators divide those factors back out before summing by group.
    """

    safe_scale = jnp.maximum(jnp.asarray(old_scale, dtype=proj_abs2_half.real.dtype), 1e-30)
    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half[None, None, :], 0.0)
    aa_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    aa_per_image = jnp.sum(aa_terms, axis=(1, 2)) / (safe_scale**2)

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    xa_terms = noise_variance_half[None, None, :] * cross_terms.real
    xa_per_image = jnp.sum(xa_terms, axis=(1, 2)) / safe_scale
    return xa_per_image.astype(jnp.float32), aa_per_image.astype(jnp.float32)
