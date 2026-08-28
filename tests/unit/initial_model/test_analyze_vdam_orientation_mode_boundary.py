from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_vdam_orientation_mode_boundary import (
    OrientationModeError,
    summarize_orientation_modes,
)

pytestmark = pytest.mark.unit


def test_orientation_modes_report_winner_spacing_and_shared_rows():
    eulers = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    posterior = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    report = summarize_orientation_modes(
        local_eulers=eulers,
        total_scores=scores,
        posterior=posterior,
        modes={"native-a": eulers[1], "native-b": eulers[1], "candidate": eulers[0]},
        translation_index=1,
    )

    assert report["global_winner"]["rotation_row"] == 1
    assert report["labels_by_local_rotation_row"]["1"] == ["native-a", "native-b"]
    candidate = next(row for row in report["modes"] if row["label"] == "candidate")
    assert candidate["winner_minus_mode_score"] == 2.0
    assert candidate["is_global_winner"] is False


def test_orientation_modes_preserve_exact_ties():
    eulers = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    scores = np.array([[4.0], [4.0]], dtype=np.float32)
    report = summarize_orientation_modes(
        local_eulers=eulers,
        total_scores=scores,
        posterior=np.array([[0.5], [0.5]], dtype=np.float32),
        modes={"first": eulers[0], "second": eulers[1]},
        translation_index=0,
    )

    assert report["global_winner"]["rotation_row"] == 0
    second = next(row for row in report["modes"] if row["label"] == "second")
    assert second["exact_score_tie_with_global_winner"] is True
    assert second["is_global_winner"] is False


def test_orientation_mode_rejects_state_outside_captured_support():
    with pytest.raises(OrientationModeError, match="misses local support"):
        summarize_orientation_modes(
            local_eulers=np.array([[0.0, 0.0, 0.0]]),
            total_scores=np.array([[1.0]], dtype=np.float32),
            posterior=np.array([[1.0]], dtype=np.float32),
            modes={"outside": np.array([90.0, 90.0, 90.0])},
            translation_index=0,
        )
