"""Shared helpers for RELION-style half-volume M-step accumulation."""

from __future__ import annotations

import logging

import recovar.core.fourier_transform_utils as fourier_transform_utils

from recovar.em.dense_single_volume.local_backprojection import (
    enforce_relion_half_volume_x0_hermitian,
)


def half_volume_accumulator_shape(recon_volume_shape):
    """Return the packed half-volume accumulator shape."""

    return fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)


def enforce_half_volume_x0(Ft_y, Ft_ctf, recon_volume_shape, *, logger: logging.Logger, label: str):
    """Apply RELION x=0 Hermitian-plane enforcement to half-volume accumulators."""

    logger.info("%s M-step: enforcing RELION half-volume x=0 Hermitian plane", label)
    return (
        enforce_relion_half_volume_x0_hermitian(Ft_y, recon_volume_shape),
        enforce_relion_half_volume_x0_hermitian(Ft_ctf, recon_volume_shape),
    )


def half_volume_accumulators_to_full(Ft_y, Ft_ctf, recon_volume_shape):
    """Convert half-volume M-step accumulators back to the public full-volume contract."""

    return (
        fourier_transform_utils.half_volume_to_full_volume(Ft_y, recon_volume_shape).reshape(-1),
        fourier_transform_utils.half_volume_to_full_volume(Ft_ctf, recon_volume_shape).reshape(-1),
    )


def relion_x_half_volume_to_full(volume_flat, recon_volume_shape):
    """Expand a RELION-layout ``(z, y, xhalf)`` accumulator to RECOVAR full layout.

    RELION's BackProjector packs the Fourier x-axis and stores arrays in
    public order ``(z, y, xhalf)``. RECOVAR's public full accumulator order is
    ``(x, y, z)``, so expansion is a last-axis Hermitian unpack in RELION
    layout followed by a ``(2, 1, 0)`` transpose.
    """

    relion_full = fourier_transform_utils.half_volume_to_full_volume(
        volume_flat,
        recon_volume_shape,
    ).reshape(recon_volume_shape)
    return relion_full.transpose(2, 1, 0).reshape(-1)


def relion_x_half_accumulators_to_full(Ft_y, Ft_ctf, recon_volume_shape):
    """Convert RELION ``(z, y, xhalf)`` accumulators to RECOVAR full volumes."""

    return (
        relion_x_half_volume_to_full(Ft_y, recon_volume_shape),
        relion_x_half_volume_to_full(Ft_ctf, recon_volume_shape),
    )
