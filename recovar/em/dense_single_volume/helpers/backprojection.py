"""Shared half-spectrum adjoint-slice dispatch helpers."""

from __future__ import annotations

import jax.numpy as jnp

from recovar import core


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
            return core.batch_adjoint_slice_volume_indexed(
                half_blocks,
                window_indices,
                rotations,
                image_shape,
                volume_shape,
                disc_type,
                volumes=volumes,
                half_image=True,
                half_volume=half_volume,
            )
        return core.batch_adjoint_slice_volume_indexed(
            half_blocks,
            window_indices,
            rotations,
            image_shape,
            volume_shape,
            disc_type,
            volumes=volumes,
            half_image=True,
            half_volume=half_volume,
            max_r=max_r,
        )

    return core.batch_adjoint_slice_volume(
        half_blocks,
        rotations,
        image_shape,
        volume_shape,
        disc_type,
        volumes=volumes,
        half_image=True,
        half_volume=half_volume,
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
