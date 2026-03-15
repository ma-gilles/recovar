"""Canonical index normalization for cryo-EM datasets.

All functions that accept ``indices`` arguments (boolean masks, integer
arrays, or ``None``) should call :func:`normalize_indices` to validate and
convert them to a consistent ``int32`` array representation.
"""

from __future__ import annotations

import numpy as np


def normalize_indices(
    values,
    n_total: int,
    *,
    name: str = "indices",
    allow_none: bool = False,
):
    """Normalize int/bool indices to an int32 array with bounds checking.

    Parameters
    ----------
    values : array-like, bool mask, or None
        The indices to normalize.
    n_total : int
        Total number of items (used for bool-mask validation and range checks).
    name : str
        Human-readable label for error messages.
    allow_none : bool
        If *True*, ``None`` input returns ``None`` instead of raising.

    Returns
    -------
    np.ndarray[int32] or None
        Validated, 1-D integer index array.
    """
    if values is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must not be None")

    arr = np.asarray(values)

    # --- Boolean mask path ---
    if arr.dtype == bool:
        if arr.ndim != 1:
            raise ValueError(f"{name} boolean mask must be 1D")
        if arr.size != int(n_total):
            raise ValueError(
                f"{name} boolean mask length {arr.size} must match total size {int(n_total)}"
            )
        return np.flatnonzero(arr).astype(np.int32, copy=False)

    # --- Integer index path ---
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must be integer indices or boolean mask")

    arr = arr.astype(np.int64, copy=False).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.int32, copy=False)

    if np.any(arr < 0):
        raise IndexError(f"{name} contains negative values")
    if np.any(arr >= int(n_total)):
        raise IndexError(
            f"{name} contains out-of-range values for total size {int(n_total)}"
        )

    return arr.astype(np.int32, copy=False)
