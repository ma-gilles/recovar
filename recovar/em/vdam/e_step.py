"""VDAM E-step helpers (RELION conventions: pad=1, Minvsigma2[0]=0, half-complex w=1).

Production E-step wiring is in ``dense_adapter.py``; this module exposes the
small convention helpers and a posterior container used by tests/debugging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def minvsigma2_with_dc_zero(sigma2_per_group: np.ndarray) -> np.ndarray:
    """``1/sigma2_noise`` with DC shell zeroed (matches RELION's expectation preprocessing)."""
    sigma2 = np.asarray(sigma2_per_group, dtype=np.float64)
    if sigma2.ndim not in (1, 2):
        raise ValueError("sigma2 must be (n_shells,) or (G, n_shells)")
    inv = np.zeros_like(sigma2)
    nz = sigma2 > 0
    inv[nz] = 1.0 / sigma2[nz]
    (inv.__setitem__(0, 0.0) if sigma2.ndim == 1 else inv.__setitem__((slice(None), 0), 0.0))
    return inv


def hermitian_weights_relion(ori_size: int) -> np.ndarray:
    """RELION's all-ones half-complex weights (parity convention; see make_half_image_weights for correct doubling)."""
    if ori_size < 2:
        raise ValueError("ori_size must be >= 2")
    return np.ones((ori_size, ori_size // 2 + 1), dtype=np.float64)


def fourier_crop_half(image_half: np.ndarray, current_size: int) -> np.ndarray:
    """Crop ``(ori_size, ori_size/2+1)`` half-complex to ``(current_size, current_size/2+1)`` (``windowFourierTransform``)."""
    if image_half.ndim != 2:
        raise ValueError("image_half must be 2D (ori_size, ori_size/2+1)")
    ori_size = image_half.shape[0]
    if image_half.shape[1] != ori_size // 2 + 1:
        raise ValueError(f"image_half expected (N, N/2+1), got {image_half.shape}")
    if current_size > ori_size:
        raise ValueError(f"current_size={current_size} > ori_size={ori_size}")
    if current_size < 2 or current_size % 2:
        raise ValueError(f"current_size={current_size} must be even and >= 2")
    if current_size == ori_size:
        return np.ascontiguousarray(image_half)

    half_cs = current_size // 2
    out_y = current_size
    out_x = current_size // 2 + 1
    out = np.zeros((out_y, out_x), dtype=image_half.dtype)
    out[:half_cs, :out_x] = image_half[:half_cs, :out_x]
    out[half_cs:, :out_x] = image_half[ori_size - (out_y - half_cs) :, :out_x]
    return out


@dataclass
class VdamPosterior:
    """E-step batch output: posteriors + per-image Pmax/nr_significant/argmax summaries."""

    weights: np.ndarray
    pmax: np.ndarray
    nr_significant: np.ndarray
    best_class: np.ndarray
    best_euler: np.ndarray
    best_trans: np.ndarray


def build_posterior_summary(weights: np.ndarray, significance_threshold: float = 1e-8) -> VdamPosterior:
    """Build a ``VdamPosterior`` from a ``(N, K, n_rot, n_trans)`` weight tensor (argmax → opaque indices)."""
    if weights.ndim != 4:
        raise ValueError(f"weights must be 4D (N, K, n_rot, n_trans), got {weights.shape}")
    N = weights.shape[0]
    flat = weights.reshape(N, -1)
    argmax = flat.argmax(axis=1)
    K, n_rot, n_trans = weights.shape[1:]
    best_class = argmax // (n_rot * n_trans)
    rest = argmax % (n_rot * n_trans)
    best_rot_idx = rest // n_trans
    best_trans_idx = rest % n_trans
    return VdamPosterior(
        weights=weights,
        pmax=flat.max(axis=1),
        nr_significant=(flat > significance_threshold).sum(axis=1),
        best_class=best_class,
        best_euler=np.column_stack([best_rot_idx.astype(np.float64), np.zeros(N), np.zeros(N)]),
        best_trans=np.column_stack([best_trans_idx.astype(np.float64), np.zeros(N)]),
    )
