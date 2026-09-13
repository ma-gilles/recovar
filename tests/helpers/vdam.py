"""VDAM test inputs, independent NumPy conventions and error measurements."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numpy_rnd_unif_factory(seed: int) -> Callable[[int], float]:
    """Deterministic NumPy-backed ``rnd_unif`` for tests (not bit-exact to RELION)."""
    rng = np.random.default_rng(seed)

    def _rnd(_call_idx: int) -> float:
        return float(rng.random())

    return _rnd


def relative_metrics(left, right):
    left, right = np.asarray(left, np.complex128), np.asarray(right, np.complex128)
    delta = np.abs(left - right)
    tiny = np.finfo(np.float64).tiny
    return np.array(
        [
            np.linalg.norm(delta.ravel()) / max(np.linalg.norm(left.ravel()), np.linalg.norm(right.ravel()), tiny),
            np.max(delta) / max(np.max(np.abs(left)), np.max(np.abs(right)), tiny),
            np.mean(delta) / max(np.mean(np.abs(left)), np.mean(np.abs(right)), tiny),
        ]
    )


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
