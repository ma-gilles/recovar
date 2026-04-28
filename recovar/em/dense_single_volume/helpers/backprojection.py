"""Shared half-spectrum adjoint-slice dispatch helpers."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from recovar import core


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
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
):
    """Scatter a windowed half-spectrum into a volume accumulator."""

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
    )


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
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
):
    """Batched indexed adjoint-slice for windowed half-spectrum rows."""

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
    """Adjoint-slice half-spectrum rows into one volume accumulator."""

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
    """Batched adjoint-slice half-spectrum rows into volume accumulators."""

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


def batch_adjoint_half_spectrum(
    half_blocks,
    rotations,
    volumes,
    image_shape,
    volume_shape,
    disc_type,
    *,
    use_window: bool,
    window_indices=None,
    half_volume: bool = False,
    max_r="auto",
):
    """Accumulate half-spectrum rows into one or more volume accumulators."""

    if use_window:
        if max_r == "auto":
            return batch_adjoint_slice_volume_windowed(
                half_blocks,
                window_indices,
                rotations,
                volumes,
                image_shape,
                volume_shape,
                disc_type,
                True,
                half_volume,
            )
        return batch_adjoint_slice_volume_windowed(
            half_blocks,
            window_indices,
            rotations,
            volumes,
            image_shape,
            volume_shape,
            disc_type,
            True,
            half_volume,
            max_r,
        )

    return batch_adjoint_slice_volume_half(
        half_blocks,
        rotations,
        volumes,
        image_shape,
        volume_shape,
        disc_type,
        True,
        half_volume,
    )


def accumulate_adjoint_pair(
    left_half,
    right_half,
    rotations,
    left_volume,
    right_volume,
    window_indices,
    image_shape,
    volume_shape,
    disc_type,
    *,
    use_window: bool,
    disable_left: bool,
    disable_right: bool,
    half_volume: bool = False,
    max_r="auto",
):
    """Accumulate two optional adjoint-slice streams with one shared dispatch."""

    if disable_left and disable_right:
        return left_volume, right_volume

    if not disable_left and not disable_right:
        updated = batch_adjoint_half_spectrum(
            jnp.stack([left_half, right_half], axis=0),
            rotations,
            jnp.stack([left_volume, right_volume], axis=0),
            image_shape,
            volume_shape,
            disc_type,
            use_window=use_window,
            window_indices=window_indices,
            half_volume=half_volume,
            max_r=max_r,
        )
        return updated[0], updated[1]

    if not disable_left:
        left_volume = batch_adjoint_half_spectrum(
            left_half[None, :, :],
            rotations,
            left_volume[None, :],
            image_shape,
            volume_shape,
            disc_type,
            use_window=use_window,
            window_indices=window_indices,
            half_volume=half_volume,
            max_r=max_r,
        )[0]
        return left_volume, right_volume

    right_volume = batch_adjoint_half_spectrum(
        right_half[None, :, :],
        rotations,
        right_volume[None, :],
        image_shape,
        volume_shape,
        disc_type,
        use_window=use_window,
        window_indices=window_indices,
        half_volume=half_volume,
        max_r=max_r,
    )[0]
    return left_volume, right_volume
