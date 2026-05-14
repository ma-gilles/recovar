"""Phase-3 unit tests for axis-angle and shift Cartesian grids.

Covers:
- Initial grid coverage and SO(3) ball culling.
- Subdivision: 8 orientation children, 4 shift children per parent cell.
- Quaternion dedup at the SO(3) boundary.
- Paper-faithful spacing schedule (24 deg / 5 px halved per subdivision).
- Rodrigues round-trip (matrix -> axis-angle -> matrix).
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.bnb.axis_angle_grid import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    make_initial_axis_angle_grid,
    subdivide_axis_angle_cells,
)
from recovar.em.dense_single_volume.bnb.shift_grid import (
    make_initial_shift_grid,
    subdivide_shift_cells,
)


def test_rodrigues_identity_at_zero():
    R = axis_angle_to_matrix(np.zeros(3))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


def test_rodrigues_quarter_rotation_around_z():
    a = np.array([0.0, 0.0, np.pi / 2])
    R = axis_angle_to_matrix(a)
    expected = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(R, expected, atol=1e-6)


def test_rodrigues_is_orthogonal_det1():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((20, 3)) * 0.5
    R = axis_angle_to_matrix(a)
    for i in range(20):
        np.testing.assert_allclose(R[i] @ R[i].T, np.eye(3), atol=1e-5)
        assert np.linalg.det(R[i]) > 0.0


def test_quaternion_canonical_positive_scalar():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((20, 3)) * 0.4
    q = axis_angle_to_quaternion(a)
    assert np.all(q[:, 0] >= 0.0), "Canonical quaternion must have non-negative scalar"
    norms = np.linalg.norm(q, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_initial_grid_lies_inside_so3_ball():
    grid = make_initial_axis_angle_grid(np.deg2rad(24.0))
    norms = np.linalg.norm(grid.centers_axis_angle, axis=1)
    assert np.all(norms <= np.pi + np.sqrt(3.0) * grid.spacing_rad / 2.0 + 1e-6)


def test_subdivision_eight_children_at_half_spacing():
    """Every interior parent produces exactly 8 children, spacing halves."""
    grid0 = make_initial_axis_angle_grid(np.deg2rad(24.0))
    # Pick a few interior cells (far from |a|=pi) so dedup doesn't fire.
    interior_mask = (
        np.linalg.norm(grid0.centers_axis_angle, axis=1)
        < np.pi - np.sqrt(3.0) * grid0.spacing_rad
    )
    interior_idx = np.flatnonzero(interior_mask)[:5]
    grid1 = subdivide_axis_angle_cells(grid0, surviving_ids=interior_idx)

    # Every interior parent must contribute 8 children.
    assert grid1.n_cells == 8 * len(interior_idx), (
        f"Expected 8x{len(interior_idx)}={8 * len(interior_idx)} children, got {grid1.n_cells}"
    )
    # Spacing halves.
    np.testing.assert_allclose(grid1.spacing_rad, grid0.spacing_rad / 2.0)


def test_subdivision_paper_schedule_seven_steps():
    """24 deg -> 0.1875 deg after 7 subdivisions (paper's stated precision)."""
    initial_deg = 24.0
    grid = make_initial_axis_angle_grid(np.deg2rad(initial_deg))
    # Use a tiny survivor set so the test stays fast.
    keep = np.arange(min(2, grid.n_cells))
    for j in range(7):
        grid = subdivide_axis_angle_cells(grid, surviving_ids=keep)
        # Pick one child to continue refining; this keeps the test cheap.
        keep = np.arange(min(1, grid.n_cells))
    np.testing.assert_allclose(
        np.rad2deg(grid.spacing_rad),
        initial_deg / 128.0,
        atol=1e-6,
    )
    # Paper stated 0.18 deg final precision; we match 0.1875 to that.
    assert abs(np.rad2deg(grid.spacing_rad) - 0.1875) < 1e-4


def test_subdivision_quaternion_dedup_at_pi_boundary():
    """Subdivision near |a|=pi must dedup antipodal quaternion pairs."""
    # Construct a parent grid that sits right at |a|=pi.
    grid0 = make_initial_axis_angle_grid(np.deg2rad(24.0))
    parent_idx = int(np.argmax(np.linalg.norm(grid0.centers_axis_angle, axis=1)))
    # Subdivide just this one cell.
    grid1 = subdivide_axis_angle_cells(grid0, surviving_ids=np.asarray([parent_idx]))
    # After dedup we expect <= 8 children; some may collapse to existing ones.
    assert grid1.n_cells <= 8
    # All retained children must still be unique by quaternion.
    keys = np.round(grid1.quaternions / 1e-5).astype(np.int64)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    assert np.all(counts == 1), "Duplicate quaternion survived dedup"


def test_initial_shift_grid_covers_disc():
    grid = make_initial_shift_grid(5.0, max_shift_px=10.0)
    norms = np.linalg.norm(grid.centers, axis=1)
    assert np.all(norms <= 10.0 + 1e-6)
    # Origin must be in the grid.
    has_origin = np.any(np.all(np.abs(grid.centers) < 1e-6, axis=1))
    assert has_origin, "Origin shift (0, 0) missing"


def test_shift_subdivision_four_children_at_half_spacing():
    grid0 = make_initial_shift_grid(5.0, max_shift_px=10.0)
    interior = np.arange(min(4, grid0.n_cells))
    grid1 = subdivide_shift_cells(grid0, surviving_ids=interior)
    assert grid1.n_cells == 4 * len(interior)
    np.testing.assert_allclose(grid1.spacing_px, grid0.spacing_px / 2.0)


def test_shift_subdivision_paper_schedule():
    """5 px -> 5/128 ~ 0.039 px after 7 subdivisions."""
    grid = make_initial_shift_grid(5.0, max_shift_px=10.0)
    keep = np.arange(min(1, grid.n_cells))
    for _ in range(7):
        grid = subdivide_shift_cells(grid, surviving_ids=keep)
        keep = np.arange(min(1, grid.n_cells))
    np.testing.assert_allclose(grid.spacing_px, 5.0 / 128.0, atol=1e-7)
    assert abs(grid.spacing_px - 0.0390625) < 1e-6


def test_shift_subdivision_offsets_are_at_quarter_parent_spacing():
    """Children sit at parent +- spacing/4 in each axis."""
    grid0 = make_initial_shift_grid(5.0, max_shift_px=15.0)
    # Pick the origin cell.
    origin_idx = int(np.argmin(np.linalg.norm(grid0.centers, axis=1)))
    grid1 = subdivide_shift_cells(grid0, surviving_ids=np.asarray([origin_idx]))
    expected = 1.25 * np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
    # Sort both for comparison.
    actual_sorted = np.asarray(sorted(grid1.centers.tolist()))
    expected_sorted = np.asarray(sorted(expected.tolist()))
    np.testing.assert_allclose(actual_sorted, expected_sorted, atol=1e-6)


def test_axis_angle_subdivision_offsets_are_at_quarter_parent_spacing():
    """Axis-angle children sit at parent +- spacing/4 in each axis."""
    # Start from a single, interior cell (origin).
    grid0 = make_initial_axis_angle_grid(np.deg2rad(24.0))
    origin_idx = int(np.argmin(np.linalg.norm(grid0.centers_axis_angle, axis=1)))
    grid1 = subdivide_axis_angle_cells(grid0, surviving_ids=np.asarray([origin_idx]))
    # 8 children at parent +- spacing/4 in each axis.
    expected_offsets = 0.25 * grid0.spacing_rad * np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    )
    parent_center = grid0.centers_axis_angle[origin_idx].astype(np.float64)
    expected = parent_center[None, :] + expected_offsets
    # Sort both.
    actual_sorted = np.asarray(sorted(grid1.centers_axis_angle.astype(np.float64).tolist()))
    expected_sorted = np.asarray(sorted(expected.tolist()))
    np.testing.assert_allclose(actual_sorted, expected_sorted, atol=1e-6)
