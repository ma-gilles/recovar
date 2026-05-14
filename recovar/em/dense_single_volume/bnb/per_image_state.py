"""Per-image-ragged candidate state for paper-faithful cryoSPARC BnB.

Each image carries its own list of axis-angle cells (with the corresponding
rotation matrices) and shift cells, plus a (n_axis_i, n_shift_i) joint
sample mask. After each pruning step the per-image candidate count stays
roughly constant (cryoSPARC caps: 12.5% rotations, 25% shifts), but cell
spacing halves with each subdivision step (8 axis children × 4 shift
children per surviving cell — Suppl §"Subdivision scheme").

This is the structural fix for the scaling test on 100k 256² where the
``fixed_grid`` mode scored the full 36864-rotation HEALPix grid at every
L-stage. With per-image-ragged state the scoring count is bounded by the
per-image candidate cardinality, NOT the global grid size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .axis_angle_grid import (
    AxisAngleGridLevel,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    make_initial_axis_angle_grid,
)
from .shift_grid import ShiftGridLevel, make_initial_shift_grid


@dataclass
class PerImageBnBPoseState:
    """Per-image axis-angle and shift candidate cells plus joint survivor mask.

    Conventions:
        - ``axis_cells[i]``: (n_axis_i, 3) axis-angle vectors (units: radians)
          for image ``i``'s currently-active rotations.
        - ``axis_rotations[i]``: (n_axis_i, 3, 3) corresponding SO(3) matrices.
        - ``shift_cells[i]``: (n_shift_i, 2) shift vectors in pixels.
        - ``sample_mask[i]``: (n_axis_i, n_shift_i) bool — True if the joint
          (axis, shift) candidate is still active for image ``i``.
        - ``axis_spacing_rad`` / ``shift_spacing_px``: current cell spacing,
          shared across images (subdivision is in lockstep).
    """

    axis_cells: list[np.ndarray]
    axis_rotations: list[np.ndarray]
    shift_cells: list[np.ndarray]
    sample_mask: list[np.ndarray]
    axis_spacing_rad: float
    shift_spacing_px: float

    @property
    def n_images(self) -> int:
        return len(self.axis_cells)

    def per_image_candidate_counts(self) -> np.ndarray:
        """Return (n_images,) int — number of active (axis, shift) pairs per image."""
        return np.asarray(
            [int(m.sum()) for m in self.sample_mask],
            dtype=np.int64,
        )

    def per_image_axis_counts(self) -> np.ndarray:
        return np.asarray(
            [m.any(axis=1).sum() for m in self.sample_mask],
            dtype=np.int64,
        )

    def per_image_shift_counts(self) -> np.ndarray:
        return np.asarray(
            [m.any(axis=0).sum() for m in self.sample_mask],
            dtype=np.int64,
        )


def initialize_per_image_state(
    n_images: int,
    initial_angular_spacing_deg: float = 24.0,
    initial_shift_spacing_px: float = 5.0,
    max_shift_px: float = 10.0,
) -> PerImageBnBPoseState:
    """Stage-0 state: every image starts with the same shared coarse cube.

    Each image gets its own copy of the initial axis-angle and shift grids,
    so that subsequent per-image pruning and subdivision can diverge
    cleanly. (We could share the underlying arrays via views to save
    memory at stage 0, but they get rewritten at the first subdivision.)
    """
    axis_grid: AxisAngleGridLevel = make_initial_axis_angle_grid(
        np.deg2rad(float(initial_angular_spacing_deg)),
    )
    shift_grid: ShiftGridLevel = make_initial_shift_grid(
        float(initial_shift_spacing_px), max_shift_px=float(max_shift_px),
    )

    axis_cells = [axis_grid.centers_axis_angle.copy() for _ in range(n_images)]
    axis_rotations = [axis_grid.rotations.copy() for _ in range(n_images)]
    shift_cells = [shift_grid.centers.copy() for _ in range(n_images)]
    sample_mask = [
        np.ones((axis_grid.n_cells, shift_grid.n_cells), dtype=bool)
        for _ in range(n_images)
    ]

    return PerImageBnBPoseState(
        axis_cells=axis_cells,
        axis_rotations=axis_rotations,
        shift_cells=shift_cells,
        sample_mask=sample_mask,
        axis_spacing_rad=axis_grid.spacing_rad,
        shift_spacing_px=shift_grid.spacing_px,
    )


_AXIS_CHILD_OFFSETS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float64,
)
_SHIFT_CHILD_OFFSETS = np.array(
    [[sx, sy] for sx in (-1, 1) for sy in (-1, 1)],
    dtype=np.float64,
)


def _expand_axis_cells(parent_centers: np.ndarray, parent_spacing: float) -> np.ndarray:
    """Return (n_parents * 8, 3) array of axis-angle child centers."""
    parent = np.asarray(parent_centers, dtype=np.float64)  # (P, 3)
    offsets = 0.25 * float(parent_spacing) * _AXIS_CHILD_OFFSETS  # (8, 3)
    children = (parent[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return children


def _expand_shift_cells(parent_centers: np.ndarray, parent_spacing: float) -> np.ndarray:
    """Return (n_parents * 4, 2) array of shift child centers."""
    parent = np.asarray(parent_centers, dtype=np.float64)  # (P, 2)
    offsets = 0.25 * float(parent_spacing) * _SHIFT_CHILD_OFFSETS  # (4, 2)
    children = (parent[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    return children


def subdivide_per_image_state(state: PerImageBnBPoseState) -> PerImageBnBPoseState:
    """Subdivide every image's surviving (axis, shift) cells.

    For image ``i``:
      - Surviving axis cell ids = unique parents of any active joint pair.
      - Surviving shift cell ids = analogous.
      - Each surviving axis cell expands to 8 children at half spacing.
      - Each surviving shift cell expands to 4 children at half spacing.
      - The new sample mask is True for all (axis_child, shift_child) pairs
        whose (axis_parent, shift_parent) cell was active.

    Note: we do NOT take the union across images. Each image gets its own
    new (axis_cells, shift_cells) lists. This is the load-bearing
    correctness fix versus the prior shared-grid hierarchical mode.
    """
    new_axis_cells: list[np.ndarray] = []
    new_axis_rotations: list[np.ndarray] = []
    new_shift_cells: list[np.ndarray] = []
    new_sample_mask: list[np.ndarray] = []

    for i in range(state.n_images):
        mask_i = state.sample_mask[i]
        axis_alive = mask_i.any(axis=1)
        shift_alive = mask_i.any(axis=0)
        axis_surv_ids = np.flatnonzero(axis_alive).astype(np.int32)
        shift_surv_ids = np.flatnonzero(shift_alive).astype(np.int32)

        if axis_surv_ids.size == 0 or shift_surv_ids.size == 0:
            # Image has no surviving joint pair; carry forward an empty
            # state. The driver's fallback should pick this up and either
            # restore the top candidate or hand off to dense.
            new_axis_cells.append(np.zeros((0, 3), dtype=np.float32))
            new_axis_rotations.append(np.zeros((0, 3, 3), dtype=np.float32))
            new_shift_cells.append(np.zeros((0, 2), dtype=np.float32))
            new_sample_mask.append(np.zeros((0, 0), dtype=bool))
            continue

        parent_axis = state.axis_cells[i][axis_surv_ids]
        parent_shift = state.shift_cells[i][shift_surv_ids]
        child_axis = _expand_axis_cells(parent_axis, state.axis_spacing_rad)
        child_shift = _expand_shift_cells(parent_shift, state.shift_spacing_px)

        # Build (n_parents_axis * 8, 3, 3) rotation matrices via Rodrigues.
        child_rotations = axis_angle_to_matrix(child_axis)

        # New sample mask: child (8 * j_axis_local + ca, 4 * j_shift_local + cs)
        # is True iff parent (j_axis_local, j_shift_local) was active.
        compressed = mask_i[axis_surv_ids][:, shift_surv_ids]  # (P_ax, P_sh) bool
        # Repeat each parent True 8x along axis dim, 4x along shift dim.
        new_mask = np.repeat(np.repeat(compressed, 8, axis=0), 4, axis=1)
        # That gives shape (8*P_ax, 4*P_sh) — same as len(child_axis), len(child_shift).

        new_axis_cells.append(child_axis.astype(np.float32))
        new_axis_rotations.append(child_rotations.astype(np.float32))
        new_shift_cells.append(child_shift.astype(np.float32))
        new_sample_mask.append(new_mask)

    return PerImageBnBPoseState(
        axis_cells=new_axis_cells,
        axis_rotations=new_axis_rotations,
        shift_cells=new_shift_cells,
        sample_mask=new_sample_mask,
        axis_spacing_rad=state.axis_spacing_rad / 2.0,
        shift_spacing_px=state.shift_spacing_px / 2.0,
    )
