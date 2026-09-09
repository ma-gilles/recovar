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


def run_em_output_to_bpref(
    Ft_y: np.ndarray,
    Ft_ctf: np.ndarray,
    ori_size: int,
    r_max: int,
    padding_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense EM accumulators ``(N,N,N)`` to RELION BPref slab (full half-complex or low-freq crop)."""
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
    bp_data = _bp_slab(data_cube, data_radius, data_center)
    bp_weight = _bp_slab(weight_cube, weight_radius, weight_center)

    # Clamp denormal weights to 0 (RELION ``updateSSNRarrays`` aborts on (0, 1e-20]).
    bp_weight_f64 = np.asarray(bp_weight.real, dtype=np.float64).copy()
    bp_weight_f64[np.abs(bp_weight_f64) < 1e-15] = 0.0
    return np.asarray(bp_data, dtype=np.complex128).copy(), bp_weight_f64


def bpref_to_run_em_output(
    bp_data: np.ndarray,
    bp_weight: np.ndarray,
    ori_size: int,
    r_max: int,
    padding_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed a RELION BPref slab back into RECOVAR's centered full layout."""
    if padding_factor != 1:
        raise NotImplementedError("padding_factor must be 1")
    if r_max < 0:
        raise ValueError(f"r_max must be non-negative, got {r_max}")

    N = int(ori_size)
    c = N // 2
    Fy = np.zeros((N, N, N), dtype=np.complex128)
    Fc = np.zeros((N, N, N), dtype=np.float64)
    data = np.asarray(bp_data, dtype=np.complex128)
    weight = np.asarray(bp_weight, dtype=np.float64)

    if r_max >= c:
        expected = (N, N, c + 1)
        if data.shape != expected or weight.shape != expected:
            raise ValueError(f"full-resolution BPref shape must be {expected}, got {data.shape} and {weight.shape}")
        Fy[:, :, c:] = data[:, :, :-1]
        Fy[:, :, :1] = data[:, :, -1:]
        Fc[:, :, c:] = weight[:, :, :-1]
        Fc[:, :, :1] = weight[:, :, -1:]
    else:
        half_ps = r_max + 1
        expected = (2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1)
        if data.shape != expected or weight.shape != expected:
            raise ValueError(f"cropped BPref shape must be {expected}, got {data.shape} and {weight.shape}")
        sl = (
            slice(c - half_ps, c + half_ps + 1),
            slice(c - half_ps, c + half_ps + 1),
            slice(c, c + half_ps + 1),
        )
        Fy[sl] = data
        Fc[sl] = weight

    return Fy, Fc


def relion_bpref_frame_scales(ori_size: int) -> tuple[float, float]:
    """``(-N², N⁴)`` — RECOVAR unnormalised-FFT → RELION BPref frame."""
    n = float(ori_size)
    return -(n**2), n**4
