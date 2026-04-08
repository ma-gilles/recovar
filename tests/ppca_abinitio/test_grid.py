"""Tests for `recovar.em.ppca_abinitio.grid.build_fixed_grid`.

Pins:
- v0 enforces `healpix_order <= 2` (Q4 / Section 5.1).
- Output dtypes are float64.
- Translation grid contains the (0,0) origin.
- Rotation matrices are orthogonal.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.types import FixedGridSpec

pytestmark = pytest.mark.unit


def test_order_2_max_shift_1_shapes_and_dtypes():
    g = build_fixed_grid(healpix_order=2, max_shift=1, shift_step=1)
    assert isinstance(g, FixedGridSpec)
    # Order 2: nside=4, npix=192. n_in_planes ≈ 24 (round(360/(360/(6*4)))).
    # We don't pin the exact n_rot — just that it's nontrivial and float64.
    assert g.rotations.dtype == jnp.float64
    assert g.translations.dtype == jnp.float64
    assert g.rotations.shape[1:] == (3, 3)
    assert g.translations.shape[1] == 2
    assert g.n_rot > 100
    assert g.n_trans >= 1


def test_order_3_rejected_by_default():
    with pytest.raises(ValueError, match="exceeds v0 maximum"):
        build_fixed_grid(healpix_order=3, max_shift=1)


def test_order_3_allowed_with_explicit_override():
    """Phase 4 escape hatch."""
    g = build_fixed_grid(healpix_order=3, max_shift=1, enforce_v0_limits=False)
    assert g.n_rot > 1000  # order 3 is much bigger than order 2


def test_translation_grid_contains_origin():
    g = build_fixed_grid(healpix_order=2, max_shift=2, shift_step=1)
    has_origin = bool(jnp.any(jnp.all(g.translations == 0, axis=1)))
    assert has_origin


def test_translation_grid_respects_max_shift():
    g = build_fixed_grid(healpix_order=2, max_shift=2)
    radii = np.linalg.norm(np.asarray(g.translations), axis=1)
    assert float(np.max(radii)) <= 2.0 + 1e-9


def test_rotations_are_orthogonal():
    g = build_fixed_grid(healpix_order=2, max_shift=1)
    R = np.asarray(g.rotations)
    # R @ R.T == I, det(R) == 1 for proper rotations
    for r in R[: min(20, len(R))]:
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-6)


def test_max_shift_zero_gives_only_origin():
    g = build_fixed_grid(healpix_order=2, max_shift=0, shift_step=1)
    assert g.n_trans == 1
    np.testing.assert_array_equal(np.asarray(g.translations), np.array([[0.0, 0.0]]))


def test_negative_max_shift_rejected():
    with pytest.raises(ValueError, match="max_shift must be >= 0"):
        build_fixed_grid(healpix_order=2, max_shift=-1)
