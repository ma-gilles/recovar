"""Deterministic VDAM test inputs and shared error measurements."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numpy_rnd_unif_factory(seed: int) -> Callable[[int], float]:
    """Deterministic NumPy-backed ``rnd_unif`` for tests (not bit-exact to RELION)."""
    rng = np.random.default_rng(seed)

    def _rnd(_call_idx: int) -> float:
        return float(rng.random())

    return _rnd


def relative_metrics(left, right):
    left, right = np.asarray(left, np.complex128), np.asarray(right, np.complex128)
    delta = np.abs(left - right)
    tiny = np.finfo(np.float64).tiny
    return np.array(
        [
            np.linalg.norm(delta.ravel()) / max(np.linalg.norm(left.ravel()), np.linalg.norm(right.ravel()), tiny),
            np.max(delta) / max(np.max(np.abs(left)), np.max(np.abs(right)), tiny),
            np.mean(delta) / max(np.mean(np.abs(left)), np.mean(np.abs(right)), tiny),
        ]
    )

