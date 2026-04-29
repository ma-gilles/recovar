"""Shared helpers for RELION-style half-volume M-step accumulation."""

from __future__ import annotations

import logging

import recovar.core.fourier_transform_utils as fourier_transform_utils

from recovar.em.dense_single_volume.helpers.env_flags import parse_env_bool
from recovar.em.dense_single_volume.local_backprojection import (
    enforce_relion_half_volume_x0_hermitian,
)

_HALF_VOLUME_MSTEP_ENV = "RECOVAR_RELION_SPARSE_PASS2_HALF_VOLUME"
_ENFORCE_X0_ENV = "RECOVAR_RELION_SPARSE_PASS2_HALF_VOLUME_ENFORCE_X0"


def native_half_volume_mstep_enabled() -> bool:
    """Return whether native half-volume M-step accumulation is enabled."""

    return parse_env_bool(_HALF_VOLUME_MSTEP_ENV, default=False)


def half_volume_accumulator_shape(recon_volume_shape):
    """Return the packed half-volume accumulator shape."""

    return fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)


def enforce_half_volume_x0_if_requested(Ft_y, Ft_ctf, recon_volume_shape, *, logger: logging.Logger, label: str):
    """Apply RELION x=0 Hermitian-plane enforcement when requested."""

    if not parse_env_bool(_ENFORCE_X0_ENV, default=False):
        return Ft_y, Ft_ctf
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
