"""Exact identity checks between RELION and RECOVAR coarse grids.

These helpers align complete coarse grids by physical rotation/translation
identity and return exact bijections, so a comparison never has to infer
correspondence from already-reduced scores. They make no map-quality claim;
map conclusions remain gated by shellwise FSC/FSC-AUC.
"""

from __future__ import annotations

import numpy as np


def rotation_bijection(relion_matrices, recovar_matrices) -> np.ndarray:
    """Map each RELION rotation row to its bitwise-identical RECOVAR row."""

    relion = np.asarray(relion_matrices, dtype=np.float32)
    recovar = np.asarray(recovar_matrices, dtype=np.float32)
    if relion.shape != recovar.shape or relion.ndim != 3 or relion.shape[1:] != (3, 3):
        raise ValueError(f"rotation topology mismatch: {relion.shape} != {recovar.shape}")
    lookup: dict[bytes, int] = {}
    for index, matrix in enumerate(recovar):
        key = matrix.tobytes()
        if key in lookup:
            raise ValueError("RECOVAR rotation grid contains duplicate matrices")
        lookup[key] = index
    try:
        mapping = np.asarray([lookup[matrix.T.tobytes()] for matrix in relion], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("RELION rotation has no bitwise RECOVAR transpose match") from exc
    if np.unique(mapping).size != mapping.size:
        raise ValueError("rotation mapping is not bijective")
    return mapping


def translation_bijection(
    relion_phase_xyz,
    recovar_translations,
    *,
    image_size: int,
    tolerance_px: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map RELION SoA phase coefficients to RECOVAR pixel translations."""

    phases = np.asarray(relion_phase_xyz, dtype=np.float32).reshape(3, -1).T
    recovar = np.asarray(recovar_translations, dtype=np.float32)
    if recovar.shape != (phases.shape[0], 2):
        raise ValueError(f"translation topology mismatch: {phases.shape} vs {recovar.shape}")
    relion_xy = -phases[:, :2].astype(np.float64) * float(image_size) / (2.0 * np.pi)
    distances = np.linalg.norm(relion_xy[:, None, :] - recovar[None, :, :], axis=2)
    mapping = np.argmin(distances, axis=1).astype(np.int64)
    matched = distances[np.arange(mapping.size), mapping]
    if np.unique(mapping).size != mapping.size:
        raise ValueError("translation mapping is not bijective")
    max_error = float(np.max(matched))
    if max_error > float(tolerance_px):
        raise ValueError(f"translation identity error {max_error:.9g}px exceeds {tolerance_px:.9g}px")
    return mapping, relion_xy, max_error


def align_relion_surface(relion_positive_scores, rotation_map, translation_map) -> np.ndarray:
    relion = np.asarray(relion_positive_scores, dtype=np.float32)
    rotation_map = np.asarray(rotation_map, dtype=np.int64)
    translation_map = np.asarray(translation_map, dtype=np.int64)
    if relion.shape != (rotation_map.size, translation_map.size):
        raise ValueError("score surface does not match identity grids")
    aligned = np.empty_like(relion)
    aligned[np.ix_(rotation_map, translation_map)] = relion
    return aligned
