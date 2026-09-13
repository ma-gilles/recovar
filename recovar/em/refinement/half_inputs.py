"""Mutable per-half pose and correction inputs for refinement and replay."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def optional_half_arrays(values, *, dtype=None):
    """Return optional per-half arrays, preserving precision by default.

    Float32 sources remain float32; higher-precision state is not narrowed
    unless the caller supplies an explicit dtype.
    """
    if values is None:
        return [None, None]
    return [
        np.asarray(values[0], dtype=dtype) if values[0] is not None else None,
        np.asarray(values[1], dtype=dtype) if values[1] is not None else None,
    ]


def _optional_group_count_half_pair(values):
    """Return an optional explicit group cardinality for each half-set."""
    if values is None:
        return [None, None]
    arr = np.asarray(values).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    if arr.size != 2:
        raise ValueError(
            f"init_group_count must be a scalar or contain exactly two values; got shape {np.asarray(values).shape}"
        )
    counts = []
    for value in arr:
        if value is None:
            counts.append(None)
            continue
        count = int(value)
        if count < 0 or float(value) != float(count):
            raise ValueError(f"init_group_count values must be non-negative integers, got {value!r}")
        counts.append(count)
    return counts


def _logged_half_arrays(values, *, label: str):
    """Normalize per-half correction arrays and log summary statistics."""
    per_half = optional_half_arrays(values)
    for k, arr in enumerate(per_half):
        if arr is None:
            continue
        if arr.size:
            logger.info(
                "RELION mode: %s half-%d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f (%d images)",
                label,
                k + 1,
                arr.mean(),
                arr.std(),
                arr.min(),
                arr.max(),
                len(arr),
            )
        else:
            logger.info("RELION mode: %s half-%d: empty", label, k + 1)
    return per_half


@dataclass
class HalfInputState:
    """Mutable per-half inputs carried across replay and local-search iterations."""

    previous_best_translations: list
    previous_best_rotation_eulers: list
    image_corrections: list
    scale_corrections: list
    group_ids: list
    group_count: list

    @classmethod
    def from_initial_values(
        cls,
        *,
        previous_best_translations,
        previous_best_rotation_eulers,
        image_corrections,
        scale_corrections,
        group_ids=None,
        group_count=None,
    ):
        return cls(
            previous_best_translations=optional_half_arrays(previous_best_translations),
            previous_best_rotation_eulers=optional_half_arrays(previous_best_rotation_eulers),
            image_corrections=_logged_half_arrays(
                image_corrections,
                label="image_corrections",
            ),
            scale_corrections=_logged_half_arrays(
                scale_corrections,
                label="scale_corrections",
            ),
            group_ids=optional_half_arrays(group_ids, dtype=np.int64),
            group_count=_optional_group_count_half_pair(group_count),
        )


def _normalize_sigma_offset_per_half(values):
    """Return a strict two-element float list for half-specific sigma offsets."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError(
            f"translation_sigma_angstrom_per_half must contain exactly two values; got shape {np.asarray(values).shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("translation_sigma_angstrom_per_half must be finite")
    return [float(arr[0]), float(arr[1])]


def _as_sigma_offset_half_pair(values):
    """Return a scalar or explicit pair as a strict two-half sigma list."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    return _normalize_sigma_offset_per_half(arr)


def _mean_sigma_offset_per_half(values):
    per_half = _normalize_sigma_offset_per_half(values)
    if per_half is None:
        return None
    return float(0.5 * (per_half[0] + per_half[1]))
