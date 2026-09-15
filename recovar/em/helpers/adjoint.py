"""Shared adjoint-slice wrappers for dense and local EM engines."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from recovar import core


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def adjoint_slice_volume_windowed(
    windowed_half,
    window_indices,
    rotations_block,
    volume,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
    max_r=None,
    relion_x_half=False,
):
    """Scatter a windowed half-spectrum into a full half-grid and adjoint-slice."""

    return core.adjoint_slice_volume_indexed(
        windowed_half,
        window_indices,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volume=volume,
        half_image=half_image,
        half_volume=half_volume,
        max_r=max_r,
        relion_x_half=relion_x_half,
    )


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def batch_adjoint_slice_volume_windowed(
    windowed_halves,
    window_indices,
    rotations_block,
    volumes,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
    max_r=None,
    relion_x_half=False,
):
    """Batched indexed adjoint-slice for windowed half-spectrum blocks."""

    return core.batch_adjoint_slice_volume_indexed(
        windowed_halves,
        window_indices,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volumes=volumes,
        half_image=half_image,
        half_volume=half_volume,
        max_r=max_r,
        relion_x_half=relion_x_half,
    )


def adjoint_slice_volume_maybe_windowed(
    half_block,
    window_indices,
    rotations_block,
    volume,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
    *,
    use_window: bool,
    max_r=None,
    relion_x_half=False,
):
    """Adjoint-slice either a full half-grid or an indexed Fourier window."""

    if use_window or relion_x_half:
        if window_indices is None:
            n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
            window_indices = jnp.arange(n_half, dtype=jnp.int32)
        return adjoint_slice_volume_windowed(
            half_block,
            window_indices,
            rotations_block,
            volume,
            image_shape,
            volume_shape,
            disc_type,
            half_image,
            half_volume,
            max_r,
            relion_x_half,
        )
    return adjoint_slice_volume_half(
        half_block,
        rotations_block,
        volume,
        image_shape,
        volume_shape,
        disc_type,
        half_image,
        half_volume,
    )


def batch_adjoint_slice_volume_maybe_windowed(
    half_blocks,
    window_indices,
    rotations_block,
    volumes,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
    *,
    use_window: bool,
    max_r=None,
    relion_x_half=False,
):
    """Batched adjoint-slice either full half-grids or indexed Fourier windows."""

    if use_window or relion_x_half:
        if window_indices is None:
            n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
            window_indices = jnp.arange(n_half, dtype=jnp.int32)
        return batch_adjoint_slice_volume_windowed(
            half_blocks,
            window_indices,
            rotations_block,
            volumes,
            image_shape,
            volume_shape,
            disc_type,
            half_image,
            half_volume,
            max_r,
            relion_x_half,
        )
    return batch_adjoint_slice_volume_half(
        half_blocks,
        rotations_block,
        volumes,
        image_shape,
        volume_shape,
        disc_type,
        half_image,
        half_volume,
    )


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def adjoint_slice_volume_half(
    half_block,
    rotations_block,
    volume,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
):
    """Adjoint-slice a half-spectrum block into the volume accumulator."""

    return core.adjoint_slice_volume(
        half_block,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volume=volume,
        half_image=half_image,
        half_volume=half_volume,
    )


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def batch_adjoint_slice_volume_half(
    half_blocks,
    rotations_block,
    volumes,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
):
    """Batched adjoint-slice half-spectrum blocks into volume accumulators."""

    return core.batch_adjoint_slice_volume(
        half_blocks,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volumes=volumes,
        half_image=half_image,
        half_volume=half_volume,
    )
