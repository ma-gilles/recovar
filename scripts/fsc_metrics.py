"""NumPy FSC reporting shared by EM validation and reconstruction replay.

Independent of production scoring and free of backend/environment initialization.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def array_difference_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    """Summarize an array difference after float64 conversion."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.size == 0:
        raise ValueError("metric topology mismatch")
    residual = candidate - reference
    return {
        "count": int(candidate.size),
        "relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(reference), np.finfo(float).tiny)),
        "median_abs": float(np.median(np.abs(residual))),
        "p95_abs": float(np.percentile(np.abs(residual), 95)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def centered_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return centered correlation, or NaN for incompatible or constant arrays."""
    a = np.asarray(lhs, dtype=np.float64).reshape(-1)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        return float("nan")
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.dot(a, b) / denom)


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
