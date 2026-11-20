"""Small triangular-matrix helpers shared by the PPCA code."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _tri_size(size: int) -> int:
    """Return the number of upper-triangular entries in a square matrix."""
    return int(size) * (int(size) + 1) // 2


def unpack_tri_to_full(lhs_tri, basis_size: int):
    """Unpack row-major upper-triangle storage to a full symmetric matrix.

    The upper triangle is mirrored without conjugation.
    """
    basis_size = int(basis_size)
    expected = _tri_size(basis_size)
    if int(lhs_tri.shape[-1]) != expected:
        raise ValueError(f"lhs_tri last dim {lhs_tri.shape[-1]} != tri({basis_size})={expected}")
    tri_i, tri_j = np.triu_indices(basis_size)
    shape = lhs_tri.shape[:-1] + (basis_size, basis_size)
    out = jnp.zeros(shape, dtype=lhs_tri.dtype)
    out = out.at[..., tri_i, tri_j].set(lhs_tri)
    out = out.at[..., tri_j, tri_i].set(lhs_tri)
    return out
