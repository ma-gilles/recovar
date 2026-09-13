"""Layout bridges between dense EM full ``(N, N, N)`` Fourier volumes and
RELION BackProjector centered half-complex slabs."""

from __future__ import annotations

import numpy as np


def _bp_slab(arr: np.ndarray, r_max: int, c: int) -> np.ndarray:
    """Slice a centered full volume into a RELION BPref slab (full half-complex or cropped)."""
    if r_max >= c:
        return np.concatenate([arr[:, :, c:], arr[:, :, :1]], axis=2)
    half_ps = r_max + 1
    return arr[c - half_ps : c + half_ps + 1, c - half_ps : c + half_ps + 1, c : c + half_ps + 1]


def _as_centered_bpref_source(
    values: np.ndarray,
    *,
    ori_size: int,
    r_max: int,
    padding_factor: int,
) -> tuple[np.ndarray, int, int]:
    """Return ``(centered cube, center, effective radius)`` for BPref slicing.

    Dense EM historically returned an original-box full cube. The shared
    RELION x-half M-step returns its current-size odd BackProjector cube after
    conversion to the public full layout. Both encode the same centered
    support and must feed the one BPref slab conversion below.
    """
    arr = np.asarray(values)
    full_size = int(ori_size) * int(padding_factor)
    if arr.size == full_size**3:
        return arr.reshape(full_size, full_size, full_size), full_size // 2, int(r_max)

    effective_radius = int(float(padding_factor) * float(r_max) + 0.5)
    compact_size = 2 * (effective_radius + 1) + 1
    if arr.size == compact_size**3:
        return (
            arr.reshape(compact_size, compact_size, compact_size),
            compact_size // 2,
            effective_radius,
        )
    raise ValueError(
        "expected either an original-box centered Fourier cube of size "
        f"{full_size**3} or a current-size BackProjector cube of size {compact_size**3}; got shape {arr.shape}"
    )


def _centered_bpref_sources(Ft_y, Ft_ctf, *, ori_size: int, r_max: int, padding_factor: int):
    """Return ``(data cube, weight cube, center, radius)`` for the dense and RELION-x-half BPref converters.

    Both accumulators must encode the same centered support; the converters differ only in axis order.
    """
    if padding_factor not in (1, 2):
        raise NotImplementedError(f"padding_factor must be 1 or 2, got {padding_factor}")
    if r_max < 0:
        raise ValueError(f"r_max must be non-negative, got {r_max}")

    data_cube, data_center, data_radius = _as_centered_bpref_source(
        Ft_y,
        ori_size=ori_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    weight_cube, weight_center, weight_radius = _as_centered_bpref_source(
        Ft_ctf,
        ori_size=ori_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    if (data_center, data_radius) != (weight_center, weight_radius):
        raise ValueError("data and weight accumulators use different centered layouts")
    return data_cube, weight_cube, data_center, data_radius


def _bpref_slab_outputs(bp_data: np.ndarray, bp_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cast a BPref slab pair to RELION double precision; clamp denormal weights (``updateSSNRarrays`` aborts on (0, 1e-20])."""
    bp_weight_f64 = np.asarray(bp_weight.real, dtype=np.float64).copy()
    bp_weight_f64[np.abs(bp_weight_f64) < 1e-15] = 0.0
    return np.asarray(bp_data, dtype=np.complex128).copy(), bp_weight_f64


def run_em_output_to_bpref(
    Ft_y: np.ndarray,
    Ft_ctf: np.ndarray,
    ori_size: int,
    r_max: int,
    padding_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense EM accumulators ``(N,N,N)`` to RELION BPref slab (full half-complex or low-freq crop)."""
    data_cube, weight_cube, center, radius = _centered_bpref_sources(
        Ft_y,
        Ft_ctf,
        ori_size=ori_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    return _bpref_slab_outputs(_bp_slab(data_cube, radius, center), _bp_slab(weight_cube, radius, center))


def relion_bpref_frame_scales(ori_size: int) -> tuple[float, float]:
    """``(-N², N⁴)`` — RECOVAR unnormalised-FFT → RELION BPref frame."""
    n = float(ori_size)
    return -(n**2), n**4


def relion_x_public_output_to_bpref(
    Ft_y: np.ndarray,
    Ft_ctf: np.ndarray,
    ori_size: int,
    r_max: int,
    padding_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the shared RELION-x-half public-layout conversion.

    The shared M-step expands native RELION ``(z, y, xhalf)`` storage to a
    full cube and transposes it to RECOVAR's public ``(x, y, z)`` order.
    InitialModel consumes a native BPref again, so undo that transpose before
    selecting the positive-x slab.  The generic dense converter must remain
    unchanged because its input is already a centered RECOVAR Fourier cube.
    """

    data_cube, weight_cube, center, radius = _centered_bpref_sources(
        Ft_y,
        Ft_ctf,
        ori_size=ori_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    return _bpref_slab_outputs(
        _bp_slab(data_cube.transpose(2, 1, 0), radius, center),
        _bp_slab(weight_cube.transpose(2, 1, 0), radius, center),
    )
