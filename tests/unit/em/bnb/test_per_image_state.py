"""Phase-7 unit tests for per-image-ragged BnB pose state."""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.bnb.per_image_state import (
    PerImageBnBPoseState,
    initialize_per_image_state,
    subdivide_per_image_state,
)


def test_initialize_per_image_state_shapes():
    state = initialize_per_image_state(
        n_images=3,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=5.0,
        max_shift_px=5.0,
    )
    assert state.n_images == 3
    n_axis = state.axis_cells[0].shape[0]
    n_shift = state.shift_cells[0].shape[0]
    for i in range(3):
        assert state.axis_cells[i].shape == (n_axis, 3)
        assert state.axis_rotations[i].shape == (n_axis, 3, 3)
        assert state.shift_cells[i].shape == (n_shift, 2)
        assert state.sample_mask[i].shape == (n_axis, n_shift)
        assert state.sample_mask[i].all()
    np.testing.assert_allclose(state.axis_spacing_rad, np.deg2rad(24.0))
    np.testing.assert_allclose(state.shift_spacing_px, 5.0)


def test_subdivide_with_all_alive_grows_8x_and_4x():
    state = initialize_per_image_state(
        n_images=2,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=5.0,
        max_shift_px=5.0,
    )
    n_axis = state.axis_cells[0].shape[0]
    n_shift = state.shift_cells[0].shape[0]
    new_state = subdivide_per_image_state(state)
    for i in range(2):
        assert new_state.axis_cells[i].shape[0] == 8 * n_axis
        assert new_state.shift_cells[i].shape[0] == 4 * n_shift
        # All-alive parent → all children alive.
        assert new_state.sample_mask[i].all()
    np.testing.assert_allclose(new_state.axis_spacing_rad, state.axis_spacing_rad / 2)
    np.testing.assert_allclose(new_state.shift_spacing_px, state.shift_spacing_px / 2)


def test_subdivide_only_propagates_alive_cells():
    state = initialize_per_image_state(
        n_images=1,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=5.0,
        max_shift_px=5.0,
    )
    # Kill all but one (axis=0, shift=0) pair for image 0.
    state.sample_mask[0][:] = False
    state.sample_mask[0][0, 0] = True

    new_state = subdivide_per_image_state(state)
    # Only one parent axis, one parent shift -> 8 axis children × 4 shift children.
    assert new_state.axis_cells[0].shape[0] == 8
    assert new_state.shift_cells[0].shape[0] == 4
    assert new_state.sample_mask[0].shape == (8, 4)
    assert new_state.sample_mask[0].all()


def test_subdivide_per_image_independence():
    """Image 1 keeps a different parent set from image 0; subdivision is independent."""
    state = initialize_per_image_state(
        n_images=2,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=5.0,
        max_shift_px=5.0,
    )
    state.sample_mask[0][:] = False
    state.sample_mask[0][0, 0] = True   # image 0: parent (0, 0)
    state.sample_mask[1][:] = False
    state.sample_mask[1][1, 2] = True   # image 1: parent (1, 2)

    new_state = subdivide_per_image_state(state)

    # Each image has 8 axis children × 4 shift children but the cells differ
    # because they came from different parent positions.
    assert new_state.axis_cells[0].shape == (8, 3)
    assert new_state.axis_cells[1].shape == (8, 3)
    assert not np.allclose(new_state.axis_cells[0], new_state.axis_cells[1])

    # Children sit at parent +- spacing/4 in each axis.
    parent_axis_0 = state.axis_cells[0][0]
    parent_axis_1 = state.axis_cells[1][1]
    expected_offset = 0.25 * state.axis_spacing_rad
    np.testing.assert_allclose(
        np.linalg.norm(new_state.axis_cells[0] - parent_axis_0, axis=1),
        np.linalg.norm(np.array([
            [-expected_offset, -expected_offset, -expected_offset],
            [-expected_offset, -expected_offset,  expected_offset],
            [-expected_offset,  expected_offset, -expected_offset],
            [-expected_offset,  expected_offset,  expected_offset],
            [ expected_offset, -expected_offset, -expected_offset],
            [ expected_offset, -expected_offset,  expected_offset],
            [ expected_offset,  expected_offset, -expected_offset],
            [ expected_offset,  expected_offset,  expected_offset],
        ]), axis=1),
        atol=1e-6,
    )


def test_subdivide_paper_schedule_seven_steps_one_image():
    """24°/5px → 0.1875°/0.039px after 7 subdivisions, single tracked path."""
    state = initialize_per_image_state(
        n_images=1,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=5.0,
        max_shift_px=5.0,
    )
    # Track a single (axis, shift) parent each stage to keep the test cheap.
    state.sample_mask[0][:] = False
    state.sample_mask[0][0, 0] = True
    for _ in range(7):
        state = subdivide_per_image_state(state)
        # After each subdivision keep just one axis × one shift child.
        state.sample_mask[0][:] = False
        state.sample_mask[0][0, 0] = True
    np.testing.assert_allclose(np.rad2deg(state.axis_spacing_rad), 24.0 / 128.0)
    np.testing.assert_allclose(state.shift_spacing_px, 5.0 / 128.0)


def test_per_image_candidate_counts():
    state = initialize_per_image_state(
        n_images=2,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=5.0,
        max_shift_px=5.0,
    )
    n_axis = state.axis_cells[0].shape[0]
    n_shift = state.shift_cells[0].shape[0]
    np.testing.assert_array_equal(
        state.per_image_candidate_counts(),
        np.full(2, n_axis * n_shift, dtype=np.int64),
    )

    # Kill half the candidates for image 0 (kill rows from n_axis//2 onward).
    n_killed = n_axis - (n_axis // 2)  # integer arith on odd n_axis
    state.sample_mask[0][n_axis // 2 :] = False
    counts = state.per_image_candidate_counts()
    assert counts[0] == (n_axis - n_killed) * n_shift
    assert counts[1] == n_axis * n_shift
