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
