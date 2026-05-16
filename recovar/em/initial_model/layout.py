"""Layout bridges between dense EM full ``(N, N, N)`` Fourier volumes and
RELION BackProjector centered half-complex slabs."""

from __future__ import annotations

import numpy as np


def _as_centered_full_volume(values: np.ndarray, ori_size: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape == (ori_size, ori_size, ori_size):
        return arr
    if arr.size != ori_size**3:
        raise ValueError(f"expected full centered Fourier volume of size {ori_size**3}, got shape {arr.shape}")
    return arr.reshape(ori_size, ori_size, ori_size)


def _bp_slab(arr: np.ndarray, r_max: int, c: int) -> np.ndarray:
    """Slice a centered full volume into a RELION BPref slab (full half-complex or cropped)."""
    if r_max >= c:
        return np.concatenate([arr[:, :, c:], arr[:, :, :1]], axis=2)
    half_ps = r_max + 1
    return arr[c - half_ps : c + half_ps + 1, c - half_ps : c + half_ps + 1, c : c + half_ps + 1]


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

    N = int(ori_size) * int(padding_factor)
    c = N // 2
    bp_data = _bp_slab(_as_centered_full_volume(Ft_y, N), r_max, c)
    bp_weight = _bp_slab(_as_centered_full_volume(Ft_ctf, N), r_max, c)

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
