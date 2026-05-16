"""RELION-compatible image pre-shift helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils


def integer_pre_shifts_or_none(image_pre_shifts, image_indices, *, batch=None, atol: float = 1e-6):
    """Return rounded integer pre-shifts when all selected shifts are integral.

    RELION rounds ``old_offset`` and applies that integer offset to the
    real-space image with zero fill before FFT.  Non-integral shifts use
    Fourier phases for tests and non-RELION callers.
    """

    if image_pre_shifts is None:
        return None
    if batch is not None:
        batch_ndim = getattr(batch, "ndim", None)
        if batch_ndim is None:
            batch_ndim = np.asarray(batch).ndim
        if batch_ndim != 3:
            return None
    shifts = np.asarray(image_pre_shifts, dtype=np.float32)[np.asarray(image_indices)]
    if shifts.size == 0:
        return shifts.astype(np.int32).reshape(0, 2)
    rounded = np.rint(shifts)
    if not np.allclose(shifts, rounded, rtol=0.0, atol=atol):
        return None
    return rounded.astype(np.int32)


def apply_relion_integer_pre_shifts(batch, integer_shifts):
    """Apply RELION's zero-filled integer real-space pre-shifts.

    RELION's accelerated SPA path uses an out-of-place translate kernel with
    ``out[y + dy, x + dx] = in[y, x]`` for in-bounds pixels and zeros
    elsewhere.  ``integer_shifts`` stores ``(dx, dy)`` in pixels.
    """

    shifts = np.asarray(integer_shifts, dtype=np.int32)
    if shifts.size == 0:
        return np.asarray(batch)

    images = np.asarray(batch)
    if images.ndim != 3:
        raise ValueError(
            "RELION integer pre-shifts expect real-space images with shape "
            f"(batch, H, W), got {images.shape}",
        )
    if shifts.shape != (images.shape[0], 2):
        raise ValueError(
            "integer_shifts must have shape "
            f"({images.shape[0]}, 2), got {shifts.shape}",
        )

    out = np.zeros_like(images)
    height, width = images.shape[-2:]
    for row, (dx, dy) in enumerate(shifts.tolist()):
        src_x0 = max(0, -dx)
        src_x1 = width - max(0, dx)
        src_y0 = max(0, -dy)
        src_y1 = height - max(0, dy)
        if src_x0 >= src_x1 or src_y0 >= src_y1:
            continue
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        out[row, dst_y0:dst_y1, dst_x0:dst_x1] = images[row, src_y0:src_y1, src_x0:src_x1]

    return out


def half_image_phase_factors(image_shape, shifts):
    """Return packed-half Fourier phase factors for per-image pre-shifts."""

    lattice_half = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
        image_shape,
        voxel_size=1,
        scaled=True,
    )
    shifts = jnp.asarray(shifts, dtype=jnp.float32)
    return jnp.exp(-2j * jnp.pi * (lattice_half @ shifts.T)).T


def tiled_half_image_phase_factors(image_shape, shifts, n_trans: int):
    """Return phase factors expanded across translation-tiled image rows."""

    return jnp.repeat(
        half_image_phase_factors(image_shape, shifts),
        int(n_trans),
        axis=0,
    )
