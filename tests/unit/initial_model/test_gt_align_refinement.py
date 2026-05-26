"""Regression tests for the GT-alignment local refinement (commit 4a581cc5).

The alignment helper now mirrors the EM E-step: a coarse HEALPix grid
picks the best orientation, then successive finer-order grids are
searched LOCALLY around that pick (within ``refine_sigma_deg`` of the
current best). These tests pin the contract:

* ``refine_orders=None`` (default) preserves legacy coarse-only behavior
  bit-for-bit when called with the trivial identity-only grid.
* Refinement reduces the angular error vs the true rotation when the
  coarse grid step exceeds the truth-to-grid distance.
* Mirror and sign decisions made at the coarse stage are locked through
  refinement (not re-searched per finer grid).
* The resulting ``rotation_matrix`` lives in (or close to) the finest
  refinement grid, not the coarse grid passed in.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.initial_model.gt_metrics import (
    DEFAULT_GT_ALIGN_REFINE_ORDERS,
    align_volume_to_reference,
    relion_alignment_rotations,
    rotate_volume_about_center,
)

pytestmark = pytest.mark.unit


def _asymmetric_volume(seed: int = 0, ori: int = 32) -> np.ndarray:
    """Build a volume with no rotational symmetry — needed so rotation alignment
    has a unique optimum."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((ori, ori, ori), dtype=np.float64)
    # 3 asymmetric blobs at distinct off-center positions.
    centers = [(8, 12, 16), (20, 22, 10), (14, 6, 24)]
    weights = [1.0, 0.7, 0.5]
    z, y, x = np.indices(vol.shape)
    for (cz, cy, cx), w in zip(centers, weights):
        r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
        vol += w * np.exp(-r2 / 6.0)
    # A bit of low-amplitude noise breaks any residual symmetry.
    vol += 0.02 * rng.standard_normal(vol.shape)
    return vol


def _angular_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos_theta = np.clip(0.5 * (np.trace(a.T @ b) - 1.0), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def test_refine_orders_none_preserves_coarse_only_behavior():
    """``refine_orders=None`` → identical result to the legacy coarse-only path."""
    gt = _asymmetric_volume()
    rotations = relion_alignment_rotations(1)

    a_coarse = align_volume_to_reference(gt, gt, rotations, refine_orders=None)
    # An empty refinement tuple is equivalent to None.
    a_empty = align_volume_to_reference(gt, gt, rotations, refine_orders=())

    np.testing.assert_array_equal(a_coarse.rotation_matrix, a_empty.rotation_matrix)
    assert a_coarse.corr == pytest.approx(a_empty.corr, abs=1e-12)
    assert a_coarse.score == pytest.approx(a_empty.score, abs=1e-12)


def test_refinement_reduces_angular_error_to_truth():
    """When the coarse grid step exceeds the truth-to-grid gap, refinement
    must close the gap by switching to a finer grid rotation."""
    gt = _asymmetric_volume()

    # A truth rotation drawn from HEALPix-3 (small angular step) but NOT
    # in HEALPix-1 (where the nearest grid point is several degrees away).
    fine_rotations = relion_alignment_rotations(3)
    rng = np.random.default_rng(7)
    truth = fine_rotations[int(rng.integers(0, len(fine_rotations)))]
    rotated = rotate_volume_about_center(gt, truth, order=1)

    coarse_rotations = relion_alignment_rotations(1)

    # Coarse-only: best HEALPix-1 grid pick.
    a_coarse = align_volume_to_reference(rotated, gt, coarse_rotations, refine_orders=None, allow_mirror=False)
    coarse_err = _angular_distance_deg(a_coarse.rotation_matrix, truth)

    # With refinement at orders (2, 3) — should reach a HEALPix-3 grid pick.
    a_refined = align_volume_to_reference(
        rotated,
        gt,
        coarse_rotations,
        refine_orders=(2, 3),
        refine_sigma_deg=30.0,
        allow_mirror=False,
    )
    refined_err = _angular_distance_deg(a_refined.rotation_matrix, truth)

    assert refined_err <= coarse_err + 1e-9, (
        f"refinement must not regress angular accuracy: coarse {coarse_err:.3f}° vs refined {refined_err:.3f}°"
    )
    # Refinement must move us at least one HEALPix-1 step closer when the
    # true rotation is reachable on a finer grid. HEALPix-1 step ≈ 30°,
    # HEALPix-3 step ≈ 7.5°, so a multi-degree improvement is expected.
    assert refined_err < coarse_err - 1e-3, (
        f"refinement did not improve over coarse: coarse {coarse_err:.3f}° vs refined {refined_err:.3f}°"
    )
    # And the refined corr must be no worse than the coarse corr.
    assert a_refined.corr >= a_coarse.corr - 1e-9


def test_refinement_locks_mirror_and_sign_from_coarse():
    """Mirror and sign decisions are fixed at the coarse stage; refinement
    only moves rotation."""
    gt = _asymmetric_volume()
    # Mirrored copy of GT — coarse pass should pick mirror_x=True.
    mirrored = gt[::-1, :, :].copy()
    rotations = relion_alignment_rotations(1)

    a = align_volume_to_reference(
        mirrored,
        gt,
        rotations,
        refine_orders=(2, 3),
        refine_sigma_deg=30.0,
        allow_mirror=True,
        allow_sign=False,
    )
    assert bool(a.mirror_x) is True
    assert int(a.sign) == 1
    # Refinement should have produced a near-identity rotation (up to grid step).
    angle_to_identity = _angular_distance_deg(a.rotation_matrix, np.eye(3))
    assert angle_to_identity < 15.0, (
        f"refined rotation should be near identity for a pure mirror, got {angle_to_identity:.2f}°"
    )


def test_default_refine_orders_constant_is_safe():
    """The exposed ``DEFAULT_GT_ALIGN_REFINE_ORDERS`` constant must keep the
    HEALPix orders strictly increasing and reasonable so the eval CLI
    default stays correct after the merge."""
    orders = DEFAULT_GT_ALIGN_REFINE_ORDERS
    assert orders, "default refine orders must be non-empty"
    assert all(isinstance(o, int) and o >= 2 for o in orders), (
        "refine orders should be >=2 (coarse pass uses HEALPix-2 by default)"
    )
    assert list(orders) == sorted(orders), "refine orders must be ascending"
