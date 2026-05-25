"""Paper-faithful axis-angle Cartesian grid for cryoSPARC BnB rotation refinement.

cryoSPARC's BnB (Punjani 2017 Suppl Note 2, "Subdivision scheme") uses a
Cartesian grid in axis-angle space, NOT HEALPix. Each cell subdivides by a
factor of 2 in each of the three axis-angle dimensions, giving 8 children
per parent. Initial spacing is 24 deg; after 7 subdivisions, spacing is
24 / 128 = 0.1875 deg, matching the paper's stated final precision.

Provides:
- ``AxisAngleGridLevel`` dataclass — one stage of the hierarchy.
- ``axis_angle_to_matrix`` (Rodrigues) and ``axis_angle_to_quaternion``.
- ``make_initial_axis_angle_grid(spacing_rad)`` — cube [-pi, pi]^3 culled to
  the closed SO(3) ball |a| <= pi + sqrt(3) * spacing/2.
- ``subdivide_axis_angle_cells(centers, spacing)`` — 8 children per cell with
  quaternion-canonical dedup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisAngleGridLevel:
    """One subdivision stage of the axis-angle Cartesian grid."""

    level: int
    spacing_rad: float
    centers_axis_angle: np.ndarray
    """(n_cells, 3) float32 — axis-angle vectors at cell centers."""

    rotations: np.ndarray
    """(n_cells, 3, 3) float32 — SO(3) matrix via Rodrigues."""

    quaternions: np.ndarray
    """(n_cells, 4) float32 — canonical (positive-scalar) quaternion (scalar first)."""

    parent_ids: np.ndarray | None
    """(n_cells,) int32 — index into parent level's ``centers_axis_angle``, or
    None for the root level."""

    @property
    def n_cells(self) -> int:
        return int(self.centers_axis_angle.shape[0])

    @property
    def cell_volume(self) -> float:
        """Volume of each Cartesian cell in axis-angle space."""
        return float(self.spacing_rad ** 3)


def axis_angle_to_matrix(a: np.ndarray) -> np.ndarray:
    """Rodrigues' formula. Input ``a`` has shape (..., 3); output (..., 3, 3)."""
    a = np.asarray(a, dtype=np.float64)
    flat = a.reshape(-1, 3)
    n = flat.shape[0]
    theta = np.linalg.norm(flat, axis=1)
    out = np.empty((n, 3, 3), dtype=np.float64)
    small = theta < 1e-12
    # Small-angle: R ≈ I + [a]_×
    for i in np.where(small)[0]:
        out[i] = np.eye(3, dtype=np.float64) + _skew(flat[i])
    for i in np.where(~small)[0]:
        u = flat[i] / theta[i]
        K = _skew(u)
        out[i] = (
            np.eye(3, dtype=np.float64)
            + np.sin(theta[i]) * K
            + (1.0 - np.cos(theta[i])) * (K @ K)
        )
    return out.reshape(*a.shape[:-1], 3, 3).astype(np.float32)


def axis_angle_to_quaternion(a: np.ndarray, *, canonical: bool = True) -> np.ndarray:
    """Return scalar-first unit quaternion. Optionally canonical (q[0] >= 0)."""
    a = np.asarray(a, dtype=np.float64)
    flat = a.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=1)
    half = 0.5 * theta
    # sin(theta/2) / theta -> 1/2 as theta -> 0; use a safe series for small theta.
    sinc_half = np.where(theta > 1e-12, np.sin(half) / np.maximum(theta, 1e-30), 0.5)
    q = np.empty((flat.shape[0], 4), dtype=np.float64)
    q[:, 0] = np.cos(half)
    q[:, 1:] = flat * sinc_half[:, None]
    if canonical:
        sign = np.where(q[:, 0] < 0, -1.0, 1.0)
        q = q * sign[:, None]
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-30)
    return q.reshape(*a.shape[:-1], 4).astype(np.float32)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def _dedup_by_quaternion(
    centers: np.ndarray,
    quaternions: np.ndarray,
    parent_ids: np.ndarray | None,
    *,
    quat_tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Remove duplicate cells whose canonical quaternions match within ``quat_tol``.

    Returns (unique_centers, unique_quaternions, unique_parent_ids). For tied
    duplicates we keep the entry with the smallest index — which, when called
    after subdivision, preserves the order of children expansion.
    """
    keys = np.round(quaternions / quat_tol).astype(np.int64)
    # np.unique with axis=0 returns sorted-unique with the index of first occurrence.
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx_sorted = np.sort(idx)
    if parent_ids is None:
        return centers[idx_sorted], quaternions[idx_sorted], None
    return centers[idx_sorted], quaternions[idx_sorted], parent_ids[idx_sorted]


def make_initial_axis_angle_grid(
    spacing_rad: float,
    *,
    dedup_quat_tol: float = 1e-5,
) -> AxisAngleGridLevel:
    """Initial uniform Cartesian grid in [-pi, pi]^3, culled to the SO(3) ball.

    Each cell has side length ``spacing_rad``. We keep cells whose center
    norm satisfies |a| <= pi + sqrt(3) * spacing / 2 so the boundary band is
    covered. Antipodal duplicates at |a| = pi are removed by canonical
    quaternion dedup.
    """
    spacing = float(spacing_rad)
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")

    # Cell centers along each axis: midpoints of bins covering [-pi, pi].
    # Number of bins per axis ~ 2pi / spacing.
    half_extent = np.pi + 0.5 * np.sqrt(3.0) * spacing
    # Coordinates centered at 0, stepping by ``spacing``.
    max_k = int(np.ceil(half_extent / spacing))
    coords = (np.arange(-max_k, max_k + 1, dtype=np.float64)) * spacing
    g = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1).reshape(-1, 3)

    # Cull to the closed SO(3) ball plus a half-cell buffer.
    norms = np.linalg.norm(g, axis=1)
    keep = norms <= half_extent + 1e-9
    centers = g[keep]

    quaternions = axis_angle_to_quaternion(centers)
    centers, quaternions, _ = _dedup_by_quaternion(centers, quaternions, None, quat_tol=dedup_quat_tol)
    rotations = axis_angle_to_matrix(centers)

    return AxisAngleGridLevel(
        level=0,
        spacing_rad=spacing,
        centers_axis_angle=centers.astype(np.float32, copy=False),
        rotations=rotations,
        quaternions=quaternions,
        parent_ids=None,
    )


_CHILD_OFFSETS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float64,
)


def subdivide_axis_angle_cells(
    parent: AxisAngleGridLevel,
    *,
    surviving_ids: np.ndarray | None = None,
    dedup_quat_tol: float = 1e-5,
) -> AxisAngleGridLevel:
    """Expand each surviving parent cell into 8 child cells at half spacing.

    Children are placed at parent_center +- spacing/4 in each axis-angle
    coordinate, giving 8 cells of side length spacing/2.

    If ``surviving_ids`` is None, every parent cell is expanded; otherwise
    only the survivors are.

    Duplicates produced by the SO(3) boundary topology (q ~ -q at |a|=pi) are
    removed via canonical-quaternion dedup.
    """
    if surviving_ids is None:
        surv_idx = np.arange(parent.n_cells, dtype=np.int32)
    else:
        surv_idx = np.asarray(surviving_ids, dtype=np.int32)

    parent_centers = parent.centers_axis_angle[surv_idx].astype(np.float64)
    child_spacing = float(parent.spacing_rad / 2.0)
    # offsets at +/- spacing/4: that places children at corners of subdividing
    # cube of side spacing/2 centred on parent.
    offsets = 0.25 * float(parent.spacing_rad) * _CHILD_OFFSETS  # (8, 3)

    child_centers = (
        parent_centers[:, None, :] + offsets[None, :, :]
    ).reshape(-1, 3)
    parent_ids_for_children = np.repeat(surv_idx, 8)

    quaternions = axis_angle_to_quaternion(child_centers)
    child_centers, quaternions, parent_ids_for_children = _dedup_by_quaternion(
        child_centers, quaternions, parent_ids_for_children, quat_tol=dedup_quat_tol,
    )
    rotations = axis_angle_to_matrix(child_centers)

    return AxisAngleGridLevel(
        level=parent.level + 1,
        spacing_rad=child_spacing,
        centers_axis_angle=child_centers.astype(np.float32, copy=False),
        rotations=rotations,
        quaternions=quaternions,
        parent_ids=parent_ids_for_children.astype(np.int32, copy=False),
    )
