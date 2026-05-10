"""Layout bridges for InitialModel/VDAM accumulators.

The dense EM kernels accumulate Fourier volumes in RECOVAR's centered full
``(N, N, N)`` layout. RELION's VDAM M-step consumes BackProjector slabs in a
centered half-complex layout. Keep this conversion isolated so the E-step
adapter and M-step code do not duplicate indexing conventions.
"""

from __future__ import annotations

import numpy as np


def _as_centered_full_volume(values: np.ndarray, ori_size: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape == (ori_size, ori_size, ori_size):
        return arr
    if arr.size != ori_size**3:
        raise ValueError(
            f"expected a full centered Fourier volume of size {ori_size**3}, got shape {arr.shape}",
        )
    return arr.reshape(ori_size, ori_size, ori_size)


def run_em_output_to_bpref(
    Ft_y: np.ndarray,
    Ft_ctf: np.ndarray,
    ori_size: int,
    r_max: int,
    padding_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense EM accumulators to RELION BackProjector layout.

    Parameters
    ----------
    Ft_y, Ft_ctf
        Full centered Fourier volumes, either flat or shaped ``(N, N, N)``.
    ori_size
        Unpadded box size.
    r_max
        Current-resolution shell used by RELION's BackProjector.
    padding_factor
        Only ``1`` is supported for GUI InitialModel parity.

    Returns
    -------
    bp_data, bp_weight
        Complex data and real weights in RELION's centered half-complex slab
        layout. For a cropped current size this is the low-frequency slab
        around DC; for full-resolution ``r_max >= N/2`` this returns the full
        half-complex volume.
    """
    if padding_factor not in (1, 2):
        raise NotImplementedError(
            f"InitialModel BPref conversion currently supports padding_factor 1 or 2, got {padding_factor}"
        )
    if r_max < 0:
        raise ValueError(f"r_max must be non-negative, got {r_max}")

    N = int(ori_size) * int(padding_factor)
    c = N // 2
    Fy = _as_centered_full_volume(Ft_y, N)
    Fc = _as_centered_full_volume(Ft_ctf, N)

    if r_max >= c:
        # Full half-complex x axis: kx = 0..N/2 maps to centered indices
        # [c, c+1, ..., N-1, 0].
        bp_data = np.concatenate([Fy[:, :, c:], Fy[:, :, :1]], axis=2)
        bp_weight = np.concatenate([Fc[:, :, c:], Fc[:, :, :1]], axis=2)
    else:
        half_ps = r_max + 1
        bp_data = Fy[
            c - half_ps : c + half_ps + 1,
            c - half_ps : c + half_ps + 1,
            c : c + half_ps + 1,
        ]
        bp_weight = Fc[
            c - half_ps : c + half_ps + 1,
            c - half_ps : c + half_ps + 1,
            c : c + half_ps + 1,
        ]

    # Clamp denormal-range weight values to 0. recovar accumulates the BPref
    # weight in float32 inside the JAX scatter; for shells where scatter never
    # actually fires the result should be exact 0, but float32 rounding around
    # the denormal boundary can leave residual values in the (0, 1e-20) band.
    # RELION's `BackProjector::updateSSNRarrays` requires sigma2 to be either
    # exactly 0 or strictly > 1e-20 — anything in between aborts with
    # `BackProjector::reconstruct: ERROR: unexpectedly small, yet non-zero
    # sigma2 value, this should not happen...` (seen at iter-7 of K=4
    # nr_iter=10 where compounding drift drove some shells into this range).
    # Threshold 1e-15 gives a safety margin above RELION's 1e-20 cutoff.
    bp_weight_f64 = np.asarray(bp_weight.real, dtype=np.float64).copy()
    bp_weight_f64[np.abs(bp_weight_f64) < 1e-15] = 0.0
    return (
        np.asarray(bp_data, dtype=np.complex128).copy(),
        bp_weight_f64,
    )


def bpref_to_run_em_output(
    bp_data: np.ndarray,
    bp_weight: np.ndarray,
    ori_size: int,
    r_max: int,
    padding_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed a RELION BackProjector slab into RECOVAR's centered full layout."""
    if padding_factor != 1:
        raise NotImplementedError("InitialModel BPref conversion currently supports only padding_factor=1")
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
        Fy[
            c - half_ps : c + half_ps + 1,
            c - half_ps : c + half_ps + 1,
            c : c + half_ps + 1,
        ] = data
        Fc[
            c - half_ps : c + half_ps + 1,
            c - half_ps : c + half_ps + 1,
            c : c + half_ps + 1,
        ] = weight

    return Fy, Fc


def relion_bpref_frame_scales(ori_size: int) -> tuple[float, float]:
    """Return RECOVAR full-FFT to RELION BPref frame scales.

    The dense kernels use RECOVAR's unnormalised Fourier convention. RELION's
    VDAM BackProjector primitives consume native RELION-frame BPref arrays.
    Existing InitialModel diagnostics pin the bridge as ``bp_data *= -N^2``
    and ``bp_weight *= N^4``.
    """
    n = float(ori_size)
    return -(n**2), n**4
