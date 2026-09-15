"""NumPy FSC reporting shared by EM validation and reconstruction replay.

Independent of production scoring and free of backend/environment initialization.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def shell_fsc(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return canonical RECOVAR FSC shells, excluding Nyquist edges."""
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or len(set(a.shape)) != 1:
        return np.asarray([], dtype=np.float64)

    n = int(a.shape[0])
    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)
    freqs = np.fft.fftfreq(n) * n
    z, y, x = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    shells = np.rint(np.sqrt(x * x + y * y + z * z)).astype(np.int32).ravel()
    product = (fa * np.conj(fb)).ravel()
    numerator = np.bincount(shells, weights=np.real(product))
    lhs_power = np.bincount(shells, weights=(np.abs(fa) ** 2).ravel())
    rhs_power = np.bincount(shells, weights=(np.abs(fb) ** 2).ravel())
    denom = np.sqrt(lhs_power * rhs_power)
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denom, out=out, where=denom > 0.0)
    return out[: n // 2 - 1]



def normalized_fsc_auc(values: Any, axis: Any | None = None) -> float:
    """Integrate an FSC curve over a normalized shell/radius axis."""
    fsc = np.asarray(values, dtype=np.float64).reshape(-1)
    if fsc.size == 0:
        return float("nan")

    if axis is None:
        x = np.arange(fsc.size, dtype=np.float64)
    else:
        x = np.asarray(axis, dtype=np.float64).reshape(-1)
        if x.size != fsc.size:
            return float("nan")

    finite = np.isfinite(fsc) & np.isfinite(x)
    if finite.size:
        finite[0] = False  # Shell 0/DC is excluded from the existing FSC shell summaries.
    x = x[finite]
    y = fsc[finite]
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return float(y[0])

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0.0 or not math.isfinite(span):
        return float(np.mean(y))
    x_norm = (x - x[0]) / span
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x_norm))

