"""Pure-JAX 3D discrete wavelet transform.

Drop-in replacement for ``jaxwt.wavedec3`` / ``jaxwt.waverec3``.
All computation uses :func:`jax.lax.conv_general_dilated` and
:func:`jax.lax.conv_transpose` — fully JIT-compilable, GPU-native,
and handles complex data natively.

Filter coefficients are obtained from ``pywt.Wavelet`` (pure-Python
lookup, no heavy C extension at runtime).
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pywt

# ---------------------------------------------------------------------------
# Filter management
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def get_filter_bank(
    wavelet: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(dec_lo, dec_hi, rec_lo, rec_hi)`` as 1-D numpy arrays."""
    w = pywt.Wavelet(wavelet)
    return tuple(np.array(f, dtype=np.float64) for f in w.filter_bank)


_COMBO_IDX = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])


def _build_3d_filters(f_lo: np.ndarray, f_hi: np.ndarray, dtype) -> jnp.ndarray:
    """Build 8 separable 3-D filters via outer products.

    Computation done in numpy (free at JIT trace time), cast to JAX once.
    Returns shape ``(8, K, K, K)``.
    """
    lo = np.asarray(f_lo[::-1], dtype=np.float64)
    hi = np.asarray(f_hi[::-1], dtype=np.float64)
    pair = np.stack([lo, hi])  # (2, K)
    f0 = pair[_COMBO_IDX[:, 0]]  # (8, K)
    f1 = pair[_COMBO_IDX[:, 1]]
    f2 = pair[_COMBO_IDX[:, 2]]
    filts = f0[:, :, None, None] * f1[:, None, :, None] * f2[:, None, None, :]
    return jnp.array(filts, dtype=dtype)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def _pad3d(data: jnp.ndarray, filt_len: int, mode: str) -> jnp.ndarray:
    """Pad the last 3 spatial dims of ``(N, C, D, H, W)`` for analysis."""
    if mode == "zero":
        mode = "constant"

    pad_base = max(0, (2 * filt_len - 3) // 2)

    pads = []
    for ax in [-3, -2, -1]:
        p = pad_base
        extra = data.shape[ax] % 2  # pad to even length
        pads.append((pad_base, p + extra))

    # Leading dims (N, C) get no padding
    full_pad = [(0, 0)] * (data.ndim - 3) + pads
    return jnp.pad(data, full_pad, mode)


def _trim_reconstruction(res: jnp.ndarray, filt_len: int, target_shape: Optional[Tuple[int, ...]]) -> jnp.ndarray:
    """Remove conv_transpose padding in a single slice op.

    Computes start/end offsets for all 3 spatial axes, then applies one
    combined ``lax.slice`` (via Python indexing) to avoid 6 separate ops.
    """
    p = max(0, (2 * filt_len - 3) // 2)
    # Compute per-axis (start, end) trim for the last 3 dims of (N, C, D, H, W)
    starts = [0, 0]  # N, C unchanged
    limits = [res.shape[0], res.shape[1]]
    for ax_idx in range(3):
        dim = ax_idx + 2  # axes 2, 3, 4
        sz = res.shape[dim]
        ps, pe = p, p
        if target_shape is not None:
            pred = sz - (ps + pe)
            if pred == target_shape[ax_idx] + 1:
                pe += 1
        starts.append(ps)
        limits.append(sz - pe)
    return jax.lax.slice(res, starts, limits)


# ---------------------------------------------------------------------------
# Analysis (forward DWT)
# ---------------------------------------------------------------------------

# Band key ordering — matches jaxwt convention
_DETAIL_KEYS = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")


def wavedec3(
    data: jnp.ndarray,
    wavelet: Union[str, pywt.Wavelet],
    mode: str = "symmetric",
    level: Optional[int] = None,
    axes: Tuple[int, int, int] = (-3, -2, -1),
    precision: str = "highest",
) -> List[Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    """Three-dimensional multi-level wavelet decomposition.

    Drop-in replacement for ``jaxwt.wavedec3``.  Pure JAX, JIT-safe,
    handles complex data natively.

    Parameters
    ----------
    data : jnp.ndarray
        Input with at least 3 dimensions.  The first dimension is treated
        as a batch axis.
    wavelet : str or pywt.Wavelet
        Wavelet name (e.g. ``"db1"``, ``"sym4"``, ``"coif2"``).
    mode : str
        Padding mode: ``"symmetric"``, ``"reflect"``, or ``"zero"``.
    level : int or None
        Decomposition depth.  ``None`` → maximum.
    axes : tuple
        Must be ``(-3, -2, -1)`` (other orderings not yet supported).
    precision : str
        JAX precision: ``"fastest"``, ``"high"``, or ``"highest"``.
    """
    if isinstance(wavelet, pywt.Wavelet):
        wavelet_name = wavelet.name
    else:
        wavelet_name = wavelet
    dec_lo, dec_hi, _, _ = get_filter_bank(wavelet_name)
    filt_len = len(dec_lo)

    if axes != (-3, -2, -1):
        raise NotImplementedError("Only axes=(-3, -2, -1) is supported")

    # Ensure (N, 1, D, H, W)
    orig_ndim = data.ndim
    if data.ndim == 3:
        data = data[None, None, :, :, :]
    elif data.ndim >= 4:
        # Fold leading dims into batch, add channel=1
        batch_shape = data.shape[:-3]
        data = data.reshape(-1, *data.shape[-3:])[:, None, :, :, :]
    else:
        raise ValueError("wavedec3 requires at least 3 input dimensions")

    dec_filt = _build_3d_filters(dec_lo, dec_hi, data.dtype)[:, None]  # (8,1,K,K,K)
    jax_precision = jax.lax.Precision(precision)

    if level is None:
        level = pywt.dwtn_max_level(list(data.shape[-3:]), pywt.Wavelet(wavelet_name))

    result: List[Union[jnp.ndarray, Dict[str, jnp.ndarray]]] = []
    approx = data  # (N, 1, D, H, W)
    for _ in range(level):
        approx = _pad3d(approx, filt_len, mode)
        res = jax.lax.conv_general_dilated(
            lhs=approx,
            rhs=dec_filt,
            padding="VALID",
            window_strides=[2, 2, 2],
            dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
            precision=jax_precision,
        )
        # res: (N, 8, D', H', W') — index channel axis directly (no split)
        approx = res[:, 0:1, :, :, :]  # keep channel dim for next level
        result.append({k: res[:, i + 1, :, :, :] for i, k in enumerate(_DETAIL_KEYS)})

    # Approx always has shape (N, 1, D, H, W) — squeeze channel dim
    result.append(approx[:, 0, :, :, :])
    result.reverse()

    # Unfold batch dims back to original leading shape
    if orig_ndim >= 4:

        def _unfold(arr):
            return arr.reshape(*batch_shape, *arr.shape[1:])

        result = jax.tree.map(_unfold, result)

    return result


# ---------------------------------------------------------------------------
# Synthesis (inverse DWT)
# ---------------------------------------------------------------------------


def waverec3(
    coeffs: List[Union[jnp.ndarray, Dict[str, jnp.ndarray]]],
    wavelet: Union[str, pywt.Wavelet],
    axes: Tuple[int, int, int] = (-3, -2, -1),
    precision: str = "highest",
) -> jnp.ndarray:
    """Three-dimensional multi-level wavelet reconstruction.

    Drop-in replacement for ``jaxwt.waverec3``.

    Parameters
    ----------
    coeffs : list
        Output of :func:`wavedec3`.
    wavelet : str or pywt.Wavelet
        Must match the wavelet used for decomposition.
    """
    if isinstance(wavelet, pywt.Wavelet):
        wavelet_name = wavelet.name
    else:
        wavelet_name = wavelet
    _, _, rec_lo, rec_hi = get_filter_bank(wavelet_name)
    filt_len = len(rec_lo)

    if axes != (-3, -2, -1):
        raise NotImplementedError("Only axes=(-3, -2, -1) is supported")

    # Detect and normalise leading dimensions
    approx = coeffs[0]
    squeeze_batch = approx.ndim == 3  # single volume, no batch dim
    batch_shape = None

    if approx.ndim == 3:
        # Single volume — add batch dim
        coeffs = jax.tree.map(lambda a: a[None], coeffs)
        approx = coeffs[0]
    elif approx.ndim > 4:
        batch_shape = approx.shape[:-3]
        coeffs = jax.tree.map(lambda a: a.reshape(-1, *a.shape[len(batch_shape) :]), coeffs)
        approx = coeffs[0]

    dtype = approx.dtype
    rec_filt = _build_3d_filters(rec_lo, rec_hi, dtype)[None]  # (1,8,K,K,K)
    jax_precision = jax.lax.Precision(precision)

    res = approx
    detail_dicts = coeffs[1:]

    for c_pos, d in enumerate(detail_dicts):
        # Stack approx + 7 details into (N, 8, D, H, W) with one op
        res = jnp.stack([res] + [d[k] for k in _DETAIL_KEYS], axis=1)
        res = jax.lax.conv_transpose(
            lhs=res,
            rhs=rec_filt,
            padding="VALID",
            strides=[2, 2, 2],
            dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
            precision=jax_precision,
        )
        if c_pos + 1 < len(detail_dicts):
            target = detail_dicts[c_pos + 1][_DETAIL_KEYS[0]].shape[-3:]
        else:
            target = None
        res = _trim_reconstruction(res, filt_len, target)
        res = res[:, 0, :, :, :]  # back to (N, D, H, W) for next level

    if squeeze_batch:
        res = res.squeeze(0)
    elif batch_shape is not None:
        res = res.reshape(*batch_shape, *res.shape[1:])

    return res
