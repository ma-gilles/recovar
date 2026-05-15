"""Tests for cone-from-prior initialization in paper-faithful BnB."""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.bnb.axis_angle_grid import (
    axis_angle_to_matrix,
)
from recovar.em.dense_single_volume.bnb.per_image_state import (
    _local_axis_cells_in_cone,
    _local_shift_cells_in_disc,
    _rotation_matrix_to_axis_angle,
    initialize_per_image_state_from_priors,
)


def test_rotation_to_axis_angle_inverse_at_identity():
    """R = I → axis-angle = 0."""
    R = np.eye(3, dtype=np.float32)
    a = _rotation_matrix_to_axis_angle(R[None])[0]
    np.testing.assert_allclose(a, np.zeros(3), atol=1e-6)


def test_rotation_to_axis_angle_inverse_at_quarter_turn_z():
    """90° around z → axis-angle = (0, 0, π/2)."""
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    a = _rotation_matrix_to_axis_angle(R[None])[0]
    np.testing.assert_allclose(a, [0.0, 0.0, np.pi / 2], atol=1e-6)


def test_rotation_to_axis_angle_round_trip():
    """Rodrigues inverse round-trips for random rotations within |a| < π."""
    rng = np.random.default_rng(7)
    a = rng.uniform(-1.5, 1.5, size=(20, 3))  # |a| ~ < π for typical samples
    R = axis_angle_to_matrix(a)
    a_recovered = _rotation_matrix_to_axis_angle(R)
    # Check round-trip via the rotation, not the axis-angle (different
    # axis-angle vectors can give the same R for |a| > π/2).
    R2 = axis_angle_to_matrix(a_recovered)
    np.testing.assert_allclose(R, R2, atol=1e-5)


def test_local_axis_cells_in_cone_centered_at_origin():
    """Cone of radius 22.5° at spacing 11.25° → ~few cells inside."""
    cone = np.deg2rad(22.5)
    spacing = np.deg2rad(11.25)
    cells = _local_axis_cells_in_cone(np.zeros(3), spacing, cone)
    norms = np.linalg.norm(cells, axis=1)
    assert np.all(norms <= cone + 0.5 * np.sqrt(3.0) * spacing + 1e-6)
    # Origin cell should be present (within numerical tolerance).
    assert np.any(np.linalg.norm(cells, axis=1) < spacing / 2)


def test_local_axis_cells_in_cone_offset():
    """Cone offset from origin contains the offset point."""
    center = np.array([0.5, -0.3, 0.1])
    cone = np.deg2rad(22.5)
    spacing = np.deg2rad(11.25)
    cells = _local_axis_cells_in_cone(center, spacing, cone)
    # All cells should be within cone radius of center.
    d = np.linalg.norm(cells - center, axis=1)
    assert np.all(d <= cone + 0.5 * np.sqrt(3.0) * spacing + 1e-6)
    # Center should be near at least one cell.
    assert np.min(d) < spacing


def test_local_shift_cells_in_disc():
    center = np.array([1.5, -2.0])
    disc = 5.0
    spacing = 2.5
    cells = _local_shift_cells_in_disc(center, spacing, disc)
    d = np.linalg.norm(cells - center, axis=1)
    assert np.all(d <= disc + 0.5 * np.sqrt(2.0) * spacing + 1e-6)
    assert np.min(d) < spacing


def test_initialize_per_image_state_from_priors_shapes():
    """Per-image init with priors gives one cone per image."""
    rng = np.random.default_rng(11)
    n_images = 3
    R = np.zeros((n_images, 3, 3), dtype=np.float32)
    for i in range(n_images):
        a = rng.uniform(-0.5, 0.5, size=3)  # near identity
        R[i] = axis_angle_to_matrix(a)
    t = rng.uniform(-2, 2, size=(n_images, 2)).astype(np.float32)

    state = initialize_per_image_state_from_priors(
        prior_rotations=R, prior_translations=t,
        cone_radius_deg=22.5, shift_radius_px=5.0,
        cells_across_diameter=4,
    )
    assert state.n_images == n_images
    for i in range(n_images):
        assert state.axis_cells[i].shape[1] == 3
        assert state.axis_rotations[i].shape[-2:] == (3, 3)
        assert state.shift_cells[i].shape[1] == 2
        assert state.sample_mask[i].shape == (
            state.axis_cells[i].shape[0],
            state.shift_cells[i].shape[0],
        )
        assert state.sample_mask[i].all()


def test_initialize_per_image_state_from_priors_size_dramatically_smaller():
    """Cone init produces ~30-50 cells per image, vs ~500+ for full SO(3)."""
    rng = np.random.default_rng(42)
    R = np.eye(3, dtype=np.float32)[None].repeat(2, axis=0)
    t = np.zeros((2, 2), dtype=np.float32)

    state_cone = initialize_per_image_state_from_priors(
        prior_rotations=R, prior_translations=t,
        cone_radius_deg=22.5, shift_radius_px=5.0,
        cells_across_diameter=4,
    )

    from recovar.em.dense_single_volume.bnb.per_image_state import (
        initialize_per_image_state,
    )
    state_full = initialize_per_image_state(
        n_images=2, initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=5.0, max_shift_px=5.0,
    )

    n_cone = state_cone.axis_cells[0].shape[0]
    n_full = state_full.axis_cells[0].shape[0]
    print(f"cone init: {n_cone} cells per image; full SO(3) init: {n_full} cells per image")
    # Cone should give MUCH fewer cells than the full cube.
    assert n_cone < n_full / 5, f"cone init not small enough: {n_cone} vs {n_full}"


def test_cone_init_independent_per_image():
    """Two images with different priors get different cells."""
    rng = np.random.default_rng(0)
    R0 = axis_angle_to_matrix(np.array([0.0, 0.0, 0.0]))
    R1 = axis_angle_to_matrix(np.array([0.5, -0.2, 0.3]))  # ~30 deg
    R = np.stack([R0, R1], axis=0).astype(np.float32)
    t = np.zeros((2, 2), dtype=np.float32)

    state = initialize_per_image_state_from_priors(
        prior_rotations=R, prior_translations=t,
        cone_radius_deg=15.0, shift_radius_px=2.0,
        cells_across_diameter=4,
    )
    # Image 0's cells centered near origin in axis-angle space.
    assert np.linalg.norm(state.axis_cells[0].mean(axis=0)) < np.deg2rad(15.0)
    # Image 1's cells centered near R1's axis-angle position.
    a1 = _rotation_matrix_to_axis_angle(R1[None])[0]
    assert np.linalg.norm(state.axis_cells[1].mean(axis=0) - a1) < np.deg2rad(15.0)
    # The two images' axis cells should NOT overlap (cones are 30° apart).
    pairwise_min = min(
        np.linalg.norm(state.axis_cells[0][:, None, :] - state.axis_cells[1][None, :, :], axis=2).min(),
        1.0,
    )
    assert pairwise_min > 0.0, "Cones should be disjoint when priors are 30° apart"
