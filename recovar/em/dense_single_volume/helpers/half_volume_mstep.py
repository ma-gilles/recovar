"""Shared helpers for RELION-style half-volume M-step accumulation."""

from __future__ import annotations

import logging
import os

import jax
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils

from recovar.em.dense_single_volume.local_backprojection import (
    enforce_relion_half_volume_x0_hermitian,
    enforce_relion_half_volume_x0_hermitian_host,
)


_RELION_X_HALF_TO_NATIVE_HALF_MIN_VOXELS = 200_000_000
_RELION_X_HALF_FULL_HOST_MIN_VOXELS = 100_000_000
_RELION_X_HALF_HOST_X0_MIN_VOXELS = 200_000_000
_RELION_X_HALF_MSTEP_DOUBLE_ENV = "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE"
_RELION_X_HALF_HOST_X0_ENV = "RECOVAR_RELION_X_HALF_HOST_X0"


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def relion_x_half_mstep_double_enabled() -> bool:
    """Return whether RELION x-half M-step accumulates with double precision.

    RELION stores its ``BackProjector`` accumulators in single precision.
    Double accumulators are useful as a diagnostic toggle when isolating
    boundary-shell differences, but they are not the default production path.
    """

    return _env_enabled(_RELION_X_HALF_MSTEP_DOUBLE_ENV, default=False)


def relion_x_half_mstep_accumulator_dtypes(dataset_dtype, *, use_relion_x_half_mstep: bool):
    """Return ``(Ft_y dtype, Ft_ctf dtype)`` for an M-step accumulator."""

    base_dtype = np.dtype(dataset_dtype)
    if use_relion_x_half_mstep:
        if relion_x_half_mstep_double_enabled():
            return np.dtype(np.complex128), np.dtype(np.float64)
        return np.dtype(np.complex64), np.dtype(np.float32)
    return base_dtype, base_dtype


def _large_relion_x_half_to_native_half_enabled(full_voxels: int) -> bool:
    """Return whether large RELION x-half accumulators should stay half-packed."""

    raw = os.environ.get("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    min_voxels_raw = os.environ.get("RECOVAR_RELION_X_HALF_TO_NATIVE_HALF_MIN_VOXELS")
    min_voxels = _RELION_X_HALF_TO_NATIVE_HALF_MIN_VOXELS
    if min_voxels_raw is not None:
        try:
            min_voxels = int(min_voxels_raw)
        except ValueError:
            min_voxels = _RELION_X_HALF_TO_NATIVE_HALF_MIN_VOXELS
    return int(full_voxels) >= int(min_voxels)


def _large_relion_x_half_full_host_enabled(full_voxels: int) -> bool:
    """Return whether large RELION x-half full expansion should run on host."""

    raw = os.environ.get("RECOVAR_RELION_X_HALF_FULL_HOST")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    min_voxels_raw = os.environ.get("RECOVAR_RELION_X_HALF_FULL_HOST_MIN_VOXELS")
    min_voxels = _RELION_X_HALF_FULL_HOST_MIN_VOXELS
    if min_voxels_raw is not None:
        try:
            min_voxels = int(min_voxels_raw)
        except ValueError:
            min_voxels = _RELION_X_HALF_FULL_HOST_MIN_VOXELS
    return int(full_voxels) >= int(min_voxels)


def _large_relion_x_half_host_x0_enabled(full_voxels: int) -> bool:
    """Return whether x=0 plane enforcement should run on host for large grids."""

    raw = os.environ.get(_RELION_X_HALF_HOST_X0_ENV)
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    min_voxels_raw = os.environ.get("RECOVAR_RELION_X_HALF_HOST_X0_MIN_VOXELS")
    min_voxels = _RELION_X_HALF_HOST_X0_MIN_VOXELS
    if min_voxels_raw is not None:
        try:
            min_voxels = int(min_voxels_raw)
        except ValueError:
            min_voxels = _RELION_X_HALF_HOST_X0_MIN_VOXELS
    return int(full_voxels) >= int(min_voxels)


def half_volume_accumulator_shape(recon_volume_shape):
    """Return the packed half-volume accumulator shape."""

    return fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)


def relion_backprojector_r_max(volume_shape, current_size=None):
    """Return RELION ``BackProjector`` support radius for ``initZeros``."""

    volume_shape = tuple(int(v) for v in volume_shape)
    if len(volume_shape) != 3 or len(set(volume_shape)) != 1:
        raise ValueError(f"RELION BackProjector requires a cubic 3-D volume_shape, got {volume_shape}")
    ori_half = volume_shape[0] // 2
    if current_size is None or int(current_size) < 0:
        r_max = ori_half
    else:
        r_max = int(current_size) // 2
    return min(int(r_max), int(ori_half))


def relion_backprojector_volume_shape(volume_shape, padding_factor, current_size=None):
    """Return RELION's odd BPref accumulator grid for ``BackProjector::initZeros``."""

    padding_factor = float(padding_factor)
    if padding_factor <= 0:
        raise ValueError(f"padding_factor must be positive, got {padding_factor!r}")
    r_max = relion_backprojector_r_max(volume_shape, current_size=current_size)
    pad_size = 2 * (int(padding_factor * float(r_max) + 0.5) + 1) + 1
    return (int(pad_size), int(pad_size), int(pad_size))


def enforce_half_volume_x0(
    Ft_y,
    Ft_ctf,
    recon_volume_shape,
    *,
    logger: logging.Logger,
    label: str,
    force_host: bool = False,
):
    """Apply RELION x=0 Hermitian-plane enforcement to half-volume accumulators."""

    logger.info("%s M-step: enforcing RELION half-volume x=0 Hermitian plane", label)
    full_voxels = int(np.prod(recon_volume_shape))
    if force_host or _large_relion_x_half_host_x0_enabled(full_voxels):
        logger.info(
            "%s M-step: using host x=0 Hermitian enforcement for large RELION half-volume "
            "accumulators (shape=%s, full_voxels=%d)",
            label,
            tuple(recon_volume_shape),
            full_voxels,
        )
        return (
            enforce_relion_half_volume_x0_hermitian_host(Ft_y, recon_volume_shape),
            enforce_relion_half_volume_x0_hermitian_host(Ft_ctf, recon_volume_shape),
        )
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


def relion_x_half_volume_to_full(volume_flat, recon_volume_shape, *, force_host: bool = False):
    """Expand a RELION-layout ``(z, y, xhalf)`` accumulator to RECOVAR full layout.

    RELION's BackProjector packs the Fourier x-axis and stores arrays in
    public order ``(z, y, xhalf)``. RECOVAR's public full accumulator order is
    ``(x, y, z)``, so expansion is a last-axis Hermitian unpack in RELION
    layout followed by a ``(2, 1, 0)`` transpose.
    """

    if force_host or _large_relion_x_half_full_host_enabled(int(np.prod(recon_volume_shape))):
        logging.getLogger(__name__).info(
            "RELION x-half M-step: expanding large accumulator to public full layout on host "
            "(shape=%s, full_voxels=%d)",
            tuple(recon_volume_shape),
            int(np.prod(recon_volume_shape)),
        )
        return _relion_x_half_volume_to_full_host(volume_flat, recon_volume_shape).reshape(-1)

    relion_full = fourier_transform_utils.half_volume_to_full_volume(
        volume_flat,
        recon_volume_shape,
    ).reshape(recon_volume_shape)
    return relion_full.transpose(2, 1, 0).reshape(-1)


def relion_x_half_volume_to_native_half(volume_flat, recon_volume_shape):
    """Repack RELION ``(z, y, xhalf)`` to RECOVAR native ``(x, y, zhalf)``."""

    return _relion_x_half_volume_to_native_half_host(volume_flat, recon_volume_shape).reshape(-1)


def relion_x_half_volume_to_public_layout(volume_flat, recon_volume_shape, *, force_host: bool = False):
    """Convert RELION x-half to the downstream public accumulator layout.

    Normal grids keep the historical full-volume contract.  Very large grids
    are repacked into RECOVAR's native packed half-volume layout, which the
    RELION FSC/tau2/reconstruction path already accepts by shape inference.
    """

    if _large_relion_x_half_to_native_half_enabled(int(np.prod(recon_volume_shape))):
        logging.getLogger(__name__).info(
            "RELION x-half M-step: repacking large accumulator to RECOVAR native half layout "
            "(shape=%s, full_voxels=%d)",
            tuple(recon_volume_shape),
            int(np.prod(recon_volume_shape)),
        )
        return relion_x_half_volume_to_native_half(volume_flat, recon_volume_shape)
    return relion_x_half_volume_to_full(volume_flat, recon_volume_shape, force_host=force_host)


def relion_x_half_accumulators_to_public_layout(
    Ft_y,
    Ft_ctf,
    recon_volume_shape,
    *,
    force_host: bool = False,
):
    """Convert RELION ``(z, y, xhalf)`` accumulators for downstream consumers."""

    return (
        relion_x_half_volume_to_public_layout(Ft_y, recon_volume_shape, force_host=force_host),
        relion_x_half_volume_to_public_layout(Ft_ctf, recon_volume_shape, force_host=force_host),
    )


def relion_x_half_accumulators_to_full(Ft_y, Ft_ctf, recon_volume_shape):
    """Convert RELION ``(z, y, xhalf)`` accumulators to RECOVAR full volumes."""

    return (
        relion_x_half_volume_to_full(Ft_y, recon_volume_shape),
        relion_x_half_volume_to_full(Ft_ctf, recon_volume_shape),
    )


def _relion_x_half_volume_to_native_half_host(volume_flat, recon_volume_shape):
    """Host implementation of the RELION x-half to RECOVAR native-half repack."""

    recon_volume_shape = tuple(int(v) for v in recon_volume_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)
    half_grid = np.asarray(jax.device_get(volume_flat)).reshape(half_shape)

    n0, n1, n2 = recon_volume_shape
    ic2 = n2 // 2
    packed_idx = np.asarray(
        fourier_transform_utils.get_real_fft_packed_last_axis_indices(n2),
        dtype=np.intp,
    )

    native_half = np.empty(half_shape, dtype=half_grid.dtype)
    native_half[packed_idx, :, :] = half_grid[packed_idx, :, :].transpose(2, 1, 0)

    if n2 % 2 == 0:
        redundant = np.arange(1, ic2, dtype=np.intp)
    else:
        redundant = np.arange(0, ic2, dtype=np.intp)
    if redundant.size:
        partner_i0 = ((n0 - (n0 % 2) - np.arange(n0)) % n0).astype(np.intp, copy=False)
        partner_i1 = ((n1 - (n1 % 2) - np.arange(n1)) % n1).astype(np.intp, copy=False)
        source_cols = ic2 - redundant
        conjugate_source = np.conj(
            half_grid[partner_i0[packed_idx][:, None], partner_i1[None, :], :]
        )
        native_half[redundant, :, :] = conjugate_source[:, :, source_cols].transpose(2, 1, 0)

    return np.ascontiguousarray(native_half)


def _relion_x_half_volume_to_full_host(volume_flat, recon_volume_shape):
    """Host implementation of RELION x-half to public full-layout expansion."""

    recon_volume_shape = tuple(int(v) for v in recon_volume_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)
    half_grid = np.asarray(jax.device_get(volume_flat)).reshape(half_shape)

    n0, n1, n2 = recon_volume_shape
    ic2 = n2 // 2
    packed_idx = np.asarray(
        fourier_transform_utils.get_real_fft_packed_last_axis_indices(n2),
        dtype=np.intp,
    )
    relion_full = np.zeros(recon_volume_shape, dtype=half_grid.dtype)
    relion_full[:, :, packed_idx] = half_grid

    if n2 % 2 == 0:
        redundant = np.arange(1, ic2, dtype=np.intp)
    else:
        redundant = np.arange(0, ic2, dtype=np.intp)
    if redundant.size:
        partner_i0 = ((n0 - (n0 % 2) - np.arange(n0)) % n0).astype(np.intp, copy=False)
        partner_i1 = ((n1 - (n1 % 2) - np.arange(n1)) % n1).astype(np.intp, copy=False)
        source_cols = ic2 - redundant
        conj_partner = np.conj(half_grid[np.ix_(partner_i0, partner_i1, np.arange(half_shape[2]))])
        relion_full[:, :, redundant] = conj_partner[:, :, source_cols]

    return np.ascontiguousarray(relion_full.transpose(2, 1, 0))
