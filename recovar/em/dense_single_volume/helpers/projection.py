"""Projection and noise primitives shared by dense/global and local EM paths."""

from __future__ import annotations

import math
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar import core
from recovar.cuda_backproject import cuda_available as _cuda_projection_available
from recovar.cuda_backproject import (
    project_indexed,
    relion_projector_half_texture_f32,
    relion_projector_persistent_half_texture_f32,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import bin_shell_values_jax

DEFAULT_PROJECTION_MAX_R = object()
_RELION_PROJECTOR_TEXTURE_ENV = "RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP"


def compact_relion_projector_half_for_centered_indices(
    projector_half,
    pixel_indices,
    image_shape,
    *,
    r_max: int,
    padding_factor: int,
):
    """Materialize the smallest host PPref slab covering score pixels.

    RELION's first-iteration normalized-CC pass scores a small square Fourier
    window even when ``Projector::data`` was built for the full image box.
    Copy only the centered y/z region and nonnegative-x prefix that can be
    sampled by those pixels, retaining the standard one-voxel interpolation
    halo.  The returned radius is strictly larger than every consumed image
    radius, so the compact texture's model-sphere cutoff remains inactive for
    the complete score window.

    This helper intentionally accepts a NumPy host array.  Compacting after an
    eager JAX transfer would leave the full projector resident on the device,
    defeating the memory bound this operation provides.
    """

    if not isinstance(projector_half, np.ndarray):
        raise TypeError("RELION projector compaction requires a NumPy host array")
    if projector_half.ndim != 3:
        raise ValueError(
            "RELION projector compaction expects one (z, y, x-half) slab, "
            f"got {projector_half.shape}",
        )
    r_max = int(r_max)
    padding_factor = int(padding_factor)
    if r_max <= 0 or padding_factor <= 0:
        raise ValueError(
            f"r_max and padding_factor must be positive, got {r_max} and {padding_factor}",
        )
    padded_r_max = r_max * padding_factor
    expected_size = 2 * (padded_r_max + 1) + 1
    expected_shape = (expected_size, expected_size, padded_r_max + 2)
    if projector_half.shape != expected_shape:
        raise ValueError(
            "RELION projector shape does not match r_max/padding_factor: "
            f"got {projector_half.shape}, expected {expected_shape}",
        )

    image_height, image_width = (int(value) for value in image_shape)
    if (
        image_height <= 0
        or image_width <= 0
        or image_height != image_width
        or image_height % 2
        or image_width % 2
    ):
        raise ValueError(f"expected a positive even square image shape, got {image_shape}")
    indices = np.asarray(pixel_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("RELION projector compaction requires at least one pixel index")
    image_half_width = image_width // 2 + 1
    if np.any(indices < 0) or np.any(indices >= image_height * image_half_width):
        raise ValueError("RELION projector compaction pixel indices exceed the half image")
    rows = indices // image_half_width
    columns = indices - rows * image_half_width
    centered_rows = rows - image_height // 2
    max_radius_squared = int(np.max(centered_rows * centered_rows + columns * columns))
    # ``isqrt(max_r2) + 1`` is a strict radius bound, including when the
    # farthest score pixel lies exactly on an integer-radius shell.
    compact_r_max = min(r_max, max(1, math.isqrt(max_radius_squared) + 1))
    if compact_r_max == r_max:
        return projector_half, r_max

    compact_padded_r_max = compact_r_max * padding_factor
    source_center = padded_r_max + 1
    compact_center = compact_padded_r_max + 1
    start = source_center - compact_center
    stop = source_center + compact_center + 1
    compact = np.ascontiguousarray(
        projector_half[
            start:stop,
            start:stop,
            : compact_padded_r_max + 2,
        ],
    )
    expected_compact_shape = (
        2 * compact_center + 1,
        2 * compact_center + 1,
        compact_padded_r_max + 2,
    )
    if compact.shape != expected_compact_shape:
        raise RuntimeError(
            "internal RELION projector compaction shape mismatch: "
            f"got {compact.shape}, expected {expected_compact_shape}",
        )
    return compact, compact_r_max


def select_relion_projector_half_for_class(
    value,
    class_index: int,
    n_classes: int,
):
    """Select one RELION projector before transferring it to the device.

    Production projector slabs are NumPy arrays.  Preserve a host view for
    both singleton reshape and K-class indexing so only the selected 3-D slab
    reaches ``jnp.asarray`` in its consumer.  A traced JAX value may reshape
    inside its enclosing compilation.  Reject eager 4-D JAX arrays because
    even a logically shape-only reshape can allocate a second device buffer.
    """

    if value is None:
        return None
    if isinstance(value, (np.ndarray, jax.Array, jax.core.Tracer)):
        value_array = value
    else:
        value_array = np.asarray(value)
    if value_array.ndim >= 4 and int(value_array.shape[0]) == int(n_classes):
        if isinstance(value_array, jax.Array) and not isinstance(
            value_array,
            jax.core.Tracer,
        ):
            raise ValueError(
                "RELION projector class selection must occur before eager "
                "device transfer; pass the NumPy host array or select inside "
                "an enclosing jax.jit trace",
            )
        if int(n_classes) == 1:
            if int(class_index) != 0:
                raise IndexError("a singleton RELION projector only has class index 0")
            if isinstance(value_array, jax.core.Tracer):
                return jnp.reshape(value_array, value_array.shape[1:])
            return np.reshape(value_array, value_array.shape[1:])
        return value_array[int(class_index)]
    return value


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
    def project_one(rotation):
        return relion_project_half(
            volume_relion_half,
            rotation,
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
        token = os.environ.get(_RELION_PROJECTOR_TEXTURE_ENV, "1").strip().lower()
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


def _texture_centered_crop_at_indices(
    projection_crop,
    pixel_indices,
    *,
    image_shape,
    projector_output_size: int,
):
    """Gather centered full-image pixels directly from a CUDA projection crop."""

    image_size = int(image_shape[0])
    crop_size = int(projector_output_size)
    full_x_half = image_size // 2 + 1
    crop_x_half = crop_size // 2 + 1
    indices = jnp.asarray(pixel_indices, dtype=jnp.int32)
    full_rows = indices // full_x_half
    cols = indices - full_rows * full_x_half

    if crop_size == image_size:
        crop_rows = full_rows
        ky = jnp.where(
            full_rows == 0,
            crop_size // 2,
            full_rows - crop_size // 2,
        )
    else:
        ky = full_rows - image_size // 2
        crop_rows = jnp.where(
            ky == crop_size // 2,
            0,
            ky + crop_size // 2,
        )
    crop_indices = crop_rows * crop_x_half + cols
    selected = projection_crop.reshape((projection_crop.shape[0], -1))[:, crop_indices]
    output_disk = ky * ky + cols * cols <= (crop_size // 2) ** 2
    return jnp.where(output_disk[None, :], selected, jnp.zeros((), dtype=selected.dtype))


def _project_relion_projector_texture(
    volume_relion_half,
    rotations_block,
    image_shape,
    *,
    r_max: int,
    padding_factor: int,
    projector_output_size: int,
    pixel_indices=None,
    persistent_texture=None,
):
    """Project one RELION ``PPref`` block with RELION's CUDA texture arithmetic."""

    if persistent_texture is None:
        projection_crop = relion_projector_half_texture_f32(
            jnp.asarray(volume_relion_half, dtype=jnp.complex64),
            jnp.asarray(rotations_block, dtype=jnp.float32),
            current_size=int(projector_output_size),
            padding_factor=int(padding_factor),
            projector_max_r=int(r_max),
        )
    else:
        projection_crop = relion_projector_persistent_half_texture_f32(
            persistent_texture,
            jnp.asarray(rotations_block, dtype=jnp.float32),
            current_size=int(projector_output_size),
            padding_factor=int(padding_factor),
            projector_max_r=int(r_max),
        )
    if pixel_indices is not None:
        return _texture_centered_crop_at_indices(
            projection_crop,
            pixel_indices,
            image_shape=image_shape,
            projector_output_size=int(projector_output_size),
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
    persistent_texture=None,
):
    """Project precomputed RELION ``PPref`` data for one rotation block.

    Strict parity defaults to RELION's CUDA texture interpolator when the
    custom CUDA projector is available.  Set
    ``RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP=0`` to force the manual/JAX
    diagnostic fallback.
    """

    image_size = int(image_shape[0])
    resolved_output_size = int(r_max) * 2 if projector_output_size is None else int(projector_output_size)
    if resolved_output_size <= 0 or resolved_output_size > image_size:
        resolved_output_size = image_size
    use_texture = persistent_texture is not None
    if not use_texture:
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
        texture_kwargs = {
            "r_max": int(r_max),
            "padding_factor": int(padding_factor),
            "projector_output_size": resolved_output_size,
        }
        if centered_rows and pixel_indices is not None:
            texture_kwargs["pixel_indices"] = pixel_indices
        if persistent_texture is not None:
            texture_kwargs["persistent_texture"] = persistent_texture
        proj_centered = _project_relion_projector_texture(
            volume_relion_half,
            rotations_block,
            image_shape,
            **texture_kwargs,
        )
        if centered_rows:
            proj_half = proj_centered
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
    scale_correction_pixel_mask=None,
):
    """Return RELION group-scale XA/AA sufficient statistics per image.

    The inputs are the same retained M-step support tensors used by
    ``compute_norm_residual_per_image``.  The current E-step tensors already
    include the old group scale in ``XA`` and ``AA``; RELION's scale update
    accumulators divide those factors back out before summing by group.
    """

    safe_scale = jnp.maximum(jnp.asarray(old_scale, dtype=proj_abs2_half.real.dtype), 1e-30)
    ctf_has_mass = ctf_probs != 0.0
    scale_pixel_mask = None
    if scale_correction_pixel_mask is not None:
        scale_pixel_mask = jnp.asarray(scale_correction_pixel_mask, dtype=bool).reshape(-1)
        ctf_has_mass = ctf_has_mass & scale_pixel_mask[None, None, :]
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half[None, None, :], 0.0)
    aa_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    aa_per_image = jnp.sum(aa_terms, axis=(1, 2)) / (safe_scale**2)

    cross_has_mass = summed_masked != 0.0
    if scale_pixel_mask is not None:
        cross_has_mass = cross_has_mass & scale_pixel_mask[None, None, :]
    cross_terms = jnp.where(cross_has_mass, proj_half * jnp.conj(summed_masked), 0.0)
    xa_terms = noise_variance_half[None, None, :] * cross_terms.real
    xa_per_image = jnp.sum(xa_terms, axis=(1, 2)) / safe_scale
    return xa_per_image.astype(jnp.float32), aa_per_image.astype(jnp.float32)


def relion_scale_correction_pixel_mask(data_vs_prior, shell_indices, *, n_shells=None):
    """Return RELION's ``data_vs_prior > 3`` scale-statistic pixel mask."""

    indices = jnp.asarray(shell_indices, dtype=jnp.int32).reshape(-1)
    if n_shells is None:
        n_shells = int(np.asarray(data_vs_prior).size) if data_vs_prior is not None else int(np.max(indices))
    valid_shell = (indices >= 0) & (indices < int(n_shells))
    if data_vs_prior is None:
        return valid_shell
    dvp = jnp.asarray(data_vs_prior).reshape(-1)
    if dvp.size == 0:
        return jnp.zeros_like(indices, dtype=bool)
    safe_indices = jnp.clip(indices, 0, dvp.size - 1)
    return valid_shell & (indices < dvp.size) & (dvp[safe_indices] > 3.0)
