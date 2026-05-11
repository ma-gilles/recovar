"""RELION parity-replay helpers.

Extracted verbatim from ``iteration_loop.py``: replay-iteration index mapping,
per-half float32 normalizers, and the ``_RelionHalfInputState`` dataclass that
carries per-half image corrections / scale corrections / direction priors
through the iteration loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def _replay_control_model_iteration(init_relion_iteration: int, loop_iteration: int) -> int:
    """Return the RELION model.star index whose control state governs this replay step."""
    return int(init_relion_iteration) + int(loop_iteration) + 1


def _optional_float32_half_pair(values):
    """Return optional per-half arrays normalized to float32."""
    if values is None:
        return [None, None]
    return [
        np.asarray(values[0], dtype=np.float32) if values[0] is not None else None,
        np.asarray(values[1], dtype=np.float32) if values[1] is not None else None,
    ]


def _normalize_logged_float32_half_pair(values, *, label: str):
    """Normalize per-half correction arrays and log summary statistics."""
    per_half = _optional_float32_half_pair(values)
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
class _RelionHalfInputState:
    """Mutable per-half inputs carried across replay and local-search iterations."""

    previous_best_translations: list
    previous_best_rotation_eulers: list
    image_corrections: list
    scale_corrections: list

    @classmethod
    def from_initial_values(
        cls,
        *,
        previous_best_translations,
        previous_best_rotation_eulers,
        image_corrections,
        scale_corrections,
    ):
        return cls(
            previous_best_translations=_optional_float32_half_pair(previous_best_translations),
            previous_best_rotation_eulers=_optional_float32_half_pair(previous_best_rotation_eulers),
            image_corrections=_normalize_logged_float32_half_pair(
                image_corrections,
                label="image_corrections",
            ),
            scale_corrections=_normalize_logged_float32_half_pair(
                scale_corrections,
                label="scale_corrections",
            ),
        )
