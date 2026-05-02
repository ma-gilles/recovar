"""Shared helpers for RELION-style half-volume M-step accumulation."""

from __future__ import annotations

import logging

import jax.numpy as jnp

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


def _half_volume_dense_contract_weights(recon_volume_shape):
    """Weights that undo RELION Hermitian-pair summing for dense-EM callers."""

    n0, n1, n2 = (int(v) for v in recon_volume_shape)
    half_shape = half_volume_accumulator_shape(recon_volume_shape)

    i0 = jnp.arange(n0, dtype=jnp.int32)[:, None, None]
    i1 = jnp.arange(n1, dtype=jnp.int32)[None, :, None]
    kz = jnp.arange(half_shape[-1], dtype=jnp.int32)[None, None, :]

    partner_i0 = (n0 - (n0 % 2) - i0) % n0
    partner_i1 = (n1 - (n1 % 2) - i1) % n1
    self_xy = (partner_i0 == i0) & (partner_i1 == i1)
    self_kz = kz == 0
    if n2 % 2 == 0:
        self_kz = self_kz | (kz == n2 // 2)

    return jnp.where(self_xy & self_kz, 1.0, 0.5)


def _scale_half_volume_to_dense_contract(accumulator, recon_volume_shape):
    """Convert RELION-summed half-volume accumulators to dense-EM scale."""

    half_shape = half_volume_accumulator_shape(recon_volume_shape)
    grid = jnp.asarray(accumulator).reshape(half_shape)
    weights = _half_volume_dense_contract_weights(recon_volume_shape).astype(grid.real.dtype)
    return (grid * weights).reshape(-1)


def half_volume_accumulators_to_full(Ft_y, Ft_ctf, recon_volume_shape, *, contract: str = "dense"):
    """Convert half-volume M-step accumulators back to the public full-volume contract."""

    if contract == "dense":
        Ft_y = _scale_half_volume_to_dense_contract(Ft_y, recon_volume_shape)
        Ft_ctf = _scale_half_volume_to_dense_contract(Ft_ctf, recon_volume_shape)
    elif contract != "relion_sum":
        raise ValueError(f"unknown half-volume M-step contract {contract!r}")

    return (
        fourier_transform_utils.half_volume_to_full_volume(Ft_y, recon_volume_shape).reshape(-1),
        fourier_transform_utils.half_volume_to_full_volume(Ft_ctf, recon_volume_shape).reshape(-1),
    )
