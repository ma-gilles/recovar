"""Small triangular-matrix helpers shared by PPCA refinement code."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _tri_size(size: int) -> int:
    """Return the number of upper-triangular entries in a square matrix."""
    return int(size) * (int(size) + 1) // 2


def pack_upper_tri(matrix):
    """Pack ``(..., p, p)`` into row-major upper-triangular storage."""
    p = int(matrix.shape[-1])
    if matrix.shape[-2] != p:
        raise ValueError(f"matrix must be square in its last two axes, got {matrix.shape}")
    if p == 0:
        return jnp.zeros(matrix.shape[:-2] + (0,), dtype=matrix.dtype)
    tri_i, tri_j = np.triu_indices(p)
    return matrix[..., tri_i, tri_j]


def unpack_tri_to_full(lhs_tri, basis_size: int, *, hermitian: bool = False):
    """Unpack row-major upper-triangle storage to a full matrix.

    The historical fixed-pose PPCA helper mirrored the upper triangle
    symmetrically without conjugation. New complex augmented normal-equation
    code can request Hermitian fill via ``hermitian=True``.
    """
    basis_size = int(basis_size)
    expected = _tri_size(basis_size)
    if int(lhs_tri.shape[-1]) != expected:
        raise ValueError(f"lhs_tri last dim {lhs_tri.shape[-1]} != tri({basis_size})={expected}")
    tri_i, tri_j = np.triu_indices(basis_size)
    shape = lhs_tri.shape[:-1] + (basis_size, basis_size)
    out = jnp.zeros(shape, dtype=lhs_tri.dtype)
    out = out.at[..., tri_i, tri_j].set(lhs_tri)
    lower = jnp.conj(lhs_tri) if hermitian else lhs_tri
    out = out.at[..., tri_j, tri_i].set(lower)
    return out
