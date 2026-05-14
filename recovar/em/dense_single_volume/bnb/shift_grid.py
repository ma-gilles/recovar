"""Paper-faithful 2D Cartesian shift grid for cryoSPARC BnB translation refinement.

cryoSPARC subdivides shifts on a 2D Cartesian grid: each cell splits into 4
children at half spacing. Initial spacing is 5 px; after 7 subdivisions,
spacing is 5/128 ~ 0.039 px, matching the paper's stated final precision.

Provides:
- ``ShiftGridLevel`` dataclass.
- ``make_initial_shift_grid(spacing_px, max_shift_px)`` — 2D Cartesian grid
  within a disc of radius ``max_shift_px``.
- ``subdivide_shift_cells(parent_level, surviving_ids)`` — 4 children per cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShiftGridLevel:
    """One subdivision stage of the 2D shift Cartesian grid."""

    level: int
    spacing_px: float
    centers: np.ndarray
    """(n_cells, 2) float32 — (x, y) shift centers in pixels."""

    parent_ids: np.ndarray | None
    """(n_cells,) int32 — index into parent level's centers, or None for root."""

    @property
    def n_cells(self) -> int:
        return int(self.centers.shape[0])

    @property
    def cell_area(self) -> float:
        return float(self.spacing_px ** 2)


def make_initial_shift_grid(
    spacing_px: float,
    *,
    max_shift_px: float,
) -> ShiftGridLevel:
    """Initial Cartesian grid covering the disc |t| <= max_shift_px."""
    spacing = float(spacing_px)
    if spacing <= 0:
        raise ValueError(f"spacing_px must be positive, got {spacing}")
    if max_shift_px < spacing / 2:
        raise ValueError(
            f"max_shift_px={max_shift_px} too small for spacing={spacing}",
        )

    max_k = int(np.ceil(max_shift_px / spacing))
    coords = np.arange(-max_k, max_k + 1, dtype=np.float64) * spacing
    g = np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1).reshape(-1, 2)
    keep = np.linalg.norm(g, axis=1) <= max_shift_px + 1e-6
    centers = g[keep].astype(np.float32, copy=False)
    return ShiftGridLevel(
        level=0,
        spacing_px=spacing,
        centers=centers,
        parent_ids=None,
    )


_SHIFT_CHILD_OFFSETS = np.array(
    [[sx, sy] for sx in (-1, 1) for sy in (-1, 1)],
    dtype=np.float64,
)


def subdivide_shift_cells(
    parent: ShiftGridLevel,
    *,
    surviving_ids: np.ndarray | None = None,
) -> ShiftGridLevel:
    """Expand each surviving parent cell into 4 child cells at half spacing.

    Children are placed at parent_center +- spacing/4 in each pixel
    coordinate, giving 4 cells of side length spacing/2.
    """
    if surviving_ids is None:
        surv_idx = np.arange(parent.n_cells, dtype=np.int32)
    else:
        surv_idx = np.asarray(surviving_ids, dtype=np.int32)

    parent_centers = parent.centers[surv_idx].astype(np.float64)
    child_spacing = float(parent.spacing_px / 2.0)
    offsets = 0.25 * float(parent.spacing_px) * _SHIFT_CHILD_OFFSETS

    child_centers = (
        parent_centers[:, None, :] + offsets[None, :, :]
    ).reshape(-1, 2).astype(np.float32, copy=False)
    parent_ids_for_children = np.repeat(surv_idx, 4).astype(np.int32, copy=False)

    return ShiftGridLevel(
        level=parent.level + 1,
        spacing_px=child_spacing,
        centers=child_centers,
        parent_ids=parent_ids_for_children,
    )
