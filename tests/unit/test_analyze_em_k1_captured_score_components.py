from __future__ import annotations

import numpy as np

from scripts.analyze_em_k1_captured_score_components import (
    decompose_captured_residual,
)


def test_reference_norm_counterfactual_dominates() -> None:
    rows = np.arange(4, dtype=np.float64)[:, None]
    cols = np.arange(3, dtype=np.float64)[None, :]
    norm = np.broadcast_to(3.0 * rows, (4, 3))
    cross = np.broadcast_to(0.1 * cols, (4, 3))
    total = norm + cross + 7.0
    report = decompose_captured_residual(total, norm, cross)
    assert report["reference_norm_dominated"]
    assert not report["cross_dominated"]
    assert report["closure"]["max_abs"] < 1e-12
    assert (
        report["counterfactual_energy_removal_fraction"]["reference_norm"]
        > 0.99
    )


def test_cross_counterfactual_dominates() -> None:
    rows = np.arange(4, dtype=np.float64)[:, None]
    cols = np.arange(3, dtype=np.float64)[None, :]
    norm = np.broadcast_to(0.1 * rows, (4, 3))
    cross = np.broadcast_to(4.0 * cols, (4, 3))
    total = norm + cross - 11.0
    report = decompose_captured_residual(total, norm, cross)
    assert report["cross_dominated"]
    assert not report["reference_norm_dominated"]
    assert report["closure"]["max_abs"] < 1e-12


def test_closure_reports_unexplained_component() -> None:
    rows = np.arange(4, dtype=np.float64)[:, None]
    cols = np.arange(3, dtype=np.float64)[None, :]
    norm = np.broadcast_to(rows, (4, 3))
    cross = np.broadcast_to(cols, (4, 3))
    interaction = rows * cols
    report = decompose_captured_residual(
        norm + cross + interaction,
        norm,
        cross,
    )
    assert report["closure"]["max_abs"] > 0.0
