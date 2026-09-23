"""Shared helpers for RELION-style half-volume M-step accumulation."""

from __future__ import annotations

import logging
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils

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


def _relion_x_half_mstep_double_enabled() -> bool:
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
        if _relion_x_half_mstep_double_enabled():
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


def _relion_backprojector_r_max(volume_shape, current_size=None):
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
    r_max = _relion_backprojector_r_max(volume_shape, current_size=current_size)
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


def _validated_relion_right_operators(symmetry_label: str, symmetry_operators):
    """Return identity-inclusive RELION right operators in source order."""

    from recovar.em.symmetry import parse_rotational_symmetry, rotational_operators

    parsed = parse_rotational_symmetry(symmetry_label)
    if symmetry_operators is None:
        operators = rotational_operators(parsed.label, dtype=np.float64)
    else:
        operators = np.asarray(symmetry_operators, dtype=np.float64)
    expected_shape = (parsed.operator_count, 3, 3)
    if operators.shape != expected_shape:
        raise ValueError(
            f"RELION {parsed.label} requires right operators with shape {expected_shape}, "
            f"got {operators.shape}"
        )
    if not np.all(np.isfinite(operators)):
        raise ValueError(f"RELION {parsed.label} right operators must be finite")
    if not np.array_equal(operators[0], np.eye(3, dtype=np.float64)):
        raise ValueError(f"RELION {parsed.label} right operators must contain exact identity first")
    if not np.allclose(
        operators @ np.swapaxes(operators, -1, -2),
        np.eye(3, dtype=np.float64)[None, :, :],
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError(f"RELION {parsed.label} right operators must be orthogonal")
    if not np.allclose(np.linalg.det(operators), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"RELION {parsed.label} right operators must be proper rotations")
    return parsed.label, np.ascontiguousarray(operators)


def finalize_half_volume_bpref(
    Ft_y,
    Ft_ctf,
    recon_volume_shape,
    *,
    logger: logging.Logger,
    label: str,
    symmetry_label: str = "C1",
    symmetry_operators=None,
    relion_x_half: bool,
    force_host: bool | None = None,
):
    """Finalize half-volume BPref accumulators before layout conversion.

    C1 deliberately calls the historical x=0 helper verbatim.  This keeps
    existing C1 output bitwise stable.  Non-C1 currently requires RELION's
    odd ``(z, y, xhalf)`` BPref layout and uses a streamed CUDA finalizer that
    fuses x=0 Hermitian enforcement with ordered point-group accumulation.
    Large float32 accumulators use bounded output ranges copied to host;
    the complex input stays in place and is never split into device copies.
    """

    if not isinstance(symmetry_label, str) or not symmetry_label.strip():
        raise ValueError("symmetry_label must be a nonempty RELION point-group label")
    normalized_label = symmetry_label.strip().upper()
    if normalized_label == "C1":
        if symmetry_operators is not None:
            identity = np.asarray(symmetry_operators)
            if identity.shape != (1, 3, 3) or not np.array_equal(identity[0], np.eye(3)):
                raise ValueError("C1 symmetry_operators must contain exact identity only")
        return enforce_half_volume_x0(
            Ft_y,
            Ft_ctf,
            recon_volume_shape,
            logger=logger,
            label=label,
            **({"force_host": force_host} if force_host is not None else {}),
        )

    canonical_label, right_operators = _validated_relion_right_operators(
        normalized_label,
        symmetry_operators,
    )
    if not relion_x_half:
        raise NotImplementedError(
            f"{label} requested {canonical_label} point-group symmetry for a native half-volume "
            "accumulator; non-C1 reconstruction symmetry requires RELION x-half BPref storage"
        )

    recon_volume_shape = tuple(int(value) for value in recon_volume_shape)
    if len(recon_volume_shape) != 3 or len(set(recon_volume_shape)) != 1:
        raise ValueError(
            "RELION point-group BPref symmetry requires a cubic accumulator, "
            f"got {recon_volume_shape}"
        )
    if any(value <= 0 or value % 2 == 0 for value in recon_volume_shape):
        raise ValueError(
            "RELION point-group BPref symmetry requires an odd positive accumulator grid, "
            f"got {recon_volume_shape}"
        )

    data = jnp.asarray(Ft_y).reshape(-1)
    weight = jnp.asarray(Ft_ctf).reshape(-1)
    if data.dtype == jnp.dtype(jnp.complex64):
        operator_dtype = np.float32
        expected_weight_dtype = jnp.dtype(jnp.float32)
    elif data.dtype == jnp.dtype(jnp.complex128):
        operator_dtype = np.float64
        expected_weight_dtype = jnp.dtype(jnp.float64)
    else:
        raise TypeError(
            "RELION point-group BPref data must be complex64 or complex128, "
            f"got {data.dtype}"
        )
    if weight.dtype != expected_weight_dtype:
        raise TypeError(
            "RELION point-group BPref weight precision must match the data component, "
            f"got data={data.dtype}, weight={weight.dtype}"
        )

    support_radius = recon_volume_shape[0] // 2 - 1
    logger.info(
        "%s M-step: enforcing RELION x=0 and %s point-group symmetry on CUDA "
        "(operators=%d, support_radius=%d)",
        label,
        canonical_label,
        right_operators.shape[0],
        support_radius,
    )
    from recovar.em.cuda import kernels as em_cuda_kernels

    use_host = (
        (data.dtype == jnp.dtype(jnp.complex64)
         and _large_relion_x_half_host_x0_enabled(int(np.prod(recon_volume_shape))))
        if force_host is None else bool(force_host)
    )
    finalize = (
        em_cuda_kernels.relion_point_group_symmetrise_bpref_host
        if use_host else em_cuda_kernels.relion_point_group_symmetrise_bpref
    )
    return finalize(
        data,
        weight,
        jnp.asarray(right_operators, dtype=operator_dtype),
        recon_volume_shape,
        support_radius,
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


def crop_relion_x_half_accumulator(
    values,
    physical_volume_shape,
    logical_volume_shape,
):
    """Crop a centered physical BPref capacity to its logical odd cube.

    RELION's x-half layout is ``(z, y, x>=0)``.  The signed z/y axes are
    centered and therefore crop symmetrically; the packed x axis starts at
    zero and keeps its logical prefix.  No arithmetic is performed here.
    """

    physical_volume_shape = tuple(int(value) for value in physical_volume_shape)
    logical_volume_shape = tuple(int(value) for value in logical_volume_shape)
    if (
        len(physical_volume_shape) != 3
        or len(logical_volume_shape) != 3
        or len(set(physical_volume_shape)) != 1
        or len(set(logical_volume_shape)) != 1
        or logical_volume_shape[0] > physical_volume_shape[0]
        or physical_volume_shape[0] % 2 == 0
        or logical_volume_shape[0] % 2 == 0
    ):
        raise ValueError(
            "stable RELION BPref cropping requires nested odd cubic shapes, "
            f"got physical={physical_volume_shape}, logical={logical_volume_shape}"
        )
    physical_size = physical_volume_shape[0]
    logical_size = logical_volume_shape[0]
    physical_half_width = physical_size // 2 + 1
    logical_half_width = logical_size // 2 + 1
    expected_size = physical_size * physical_size * physical_half_width
    if int(values.size) != expected_size:
        raise ValueError(
            f"RELION x-half accumulator has {values.size} entries, expected {expected_size}"
        )
    start = (physical_size - logical_size) // 2
    grid = values.reshape((physical_size, physical_size, physical_half_width))
    return grid[
        start : start + logical_size,
        start : start + logical_size,
        :logical_half_width,
    ].reshape(-1)


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


def enforce_relion_half_volume_x0_hermitian(volume_flat, full_volume_shape):
    """Match RELION BackProjector::enforceHermitianSymmetry on x=0 plane."""

    return _enforce_relion_half_volume_x0_hermitian_jit(
        volume_flat,
        tuple(int(value) for value in full_volume_shape),
    )


@partial(jax.jit, static_argnames=("full_volume_shape",))
def _enforce_relion_half_volume_x0_hermitian_jit(volume_flat, full_volume_shape):
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(full_volume_shape)
    vol = jnp.asarray(volume_flat).reshape(half_shape)
    n0, n1, _ = half_shape
    i0 = jnp.arange(n0, dtype=jnp.int32)
    i1 = jnp.arange(n1, dtype=jnp.int32)
    # RELION pairs logical Xmipp-origin coordinates (z, y) with (-z, -y).
    # In RECOVAR's centered array convention this is (N - (N % 2) - i) % N;
    # odd RELION BPref grids therefore use N-1-i, not the unshifted -i.
    p0 = (n0 - (n0 % 2) - i0) % n0
    p1 = (n1 - (n1 % 2) - i1) % n1
    plane = vol[:, :, 0]
    partner = jnp.conj(plane[p0[:, None], p1[None, :]])
    summed = plane + partner
    self_partner = (p0[:, None] == i0[:, None]) & (p1[None, :] == i1[None, :])
    plane = jnp.where(self_partner, plane, summed)
    return vol.at[:, :, 0].set(plane).reshape(-1)


def enforce_relion_half_volume_x0_hermitian_host(volume_flat, full_volume_shape):
    """Host implementation of RELION x=0 Hermitian-plane enforcement.

    The device path updates a single plane with ``.at[..., 0].set(...)`` but may
    still allocate another full packed half-volume.  Large RELION BPref grids
    already repack through host memory downstream, so handling the plane update
    here avoids a transient GPU allocation without changing the arithmetic.
    """

    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(full_volume_shape)
    host = np.asarray(jax.device_get(volume_flat))
    if isinstance(volume_flat, np.ndarray) or not host.flags.writeable:
        host = host.copy()
    vol = host.reshape(half_shape)
    n0, n1, _ = half_shape
    i0 = np.arange(n0, dtype=np.int32)
    i1 = np.arange(n1, dtype=np.int32)
    p0 = (n0 - (n0 % 2) - i0) % n0
    p1 = (n1 - (n1 % 2) - i1) % n1
    plane = vol[:, :, 0]
    partner = np.conj(plane[np.ix_(p0, p1)])
    summed = plane + partner
    self_partner = (p0[:, None] == i0[:, None]) & (p1[None, :] == i1[None, :])
    vol[:, :, 0] = np.where(self_partner, plane, summed)
    return vol.reshape(-1)


def finalize_split_relion_x_half_bpref(
    Ft_y_real,
    Ft_y_imag,
    Ft_ctf,
    recon_volume_shape,
    *,
    logger: logging.Logger,
    label: str,
    symmetry_label: str,
    symmetry_operators=None,
):
    """Finalize a deferred RELION BPref while preserving split inputs.

    This is the large-grid continuation of the exact split first-iteration
    replay.  C1 supplies identity alone and therefore performs only RELION's
    x=0 enforcement.  Every point group is emitted to bounded host ranges, so
    no full complex data buffer or full finalized output pair is allocated on
    the device.
    """

    canonical_label, right_operators = _validated_relion_right_operators(
        symmetry_label,
        symmetry_operators,
    )
    recon_volume_shape = tuple(int(value) for value in recon_volume_shape)
    if len(recon_volume_shape) != 3 or len(set(recon_volume_shape)) != 1:
        raise ValueError(
            "RELION point-group BPref symmetry requires a cubic accumulator, "
            f"got {recon_volume_shape}"
        )
    if any(value <= 0 or value % 2 == 0 for value in recon_volume_shape):
        raise ValueError(
            "RELION point-group BPref symmetry requires an odd positive accumulator grid, "
            f"got {recon_volume_shape}"
        )

    data_real = jnp.asarray(Ft_y_real).reshape(-1)
    data_imag = jnp.asarray(Ft_y_imag).reshape(-1)
    weight = jnp.asarray(Ft_ctf).reshape(-1)
    for field, value in (
        ("real data", data_real),
        ("imaginary data", data_imag),
        ("weight", weight),
    ):
        if value.dtype != jnp.dtype(jnp.float32):
            raise TypeError(
                f"RELION split point-group {field} accumulator must be float32, got {value.dtype}"
            )

    support_radius = recon_volume_shape[0] // 2 - 1
    logger.info(
        "%s M-step: enforcing RELION x=0 and %s point-group symmetry from split "
        "accumulators (operators=%d, support_radius=%d)",
        label,
        canonical_label,
        right_operators.shape[0],
        support_radius,
    )
    from recovar.em.cuda import kernels as em_cuda_kernels

    return em_cuda_kernels.relion_point_group_symmetrise_bpref_split_host(
        data_real,
        data_imag,
        weight,
        jnp.asarray(right_operators, dtype=np.float32),
        recon_volume_shape,
        support_radius,
    )
