from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_vdam_orientation_mode_boundary import (
    OrientationModeError,
    summarize_native_orientation_modes,
    summarize_orientation_modes,
)

pytestmark = pytest.mark.unit


def test_orientation_modes_report_winner_spacing_and_shared_rows():
    eulers = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    posterior = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    report = summarize_orientation_modes(
        local_eulers=eulers,
        raw_scores=scores,
        total_scores=scores,
        posterior=posterior,
        rotation_log_prior=np.zeros(2, dtype=np.float32),
        translation_log_prior=np.zeros(2, dtype=np.float32),
        modes={"native-a": eulers[1], "native-b": eulers[1], "candidate": eulers[0]},
        translation_index=1,
    )

    assert report["global_winner"]["rotation_row"] == 1
    assert report["labels_by_local_rotation_row"]["1"] == ["native-a", "native-b"]
    candidate = next(row for row in report["modes"] if row["label"] == "candidate")
    assert candidate["winner_minus_mode_score"] == 2.0
    assert candidate["winner_minus_mode_components"]["raw_log_score"] == 2.0
    assert candidate["is_global_winner"] is False


def test_orientation_modes_preserve_exact_ties():
    eulers = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    scores = np.array([[4.0], [4.0]], dtype=np.float32)
    report = summarize_orientation_modes(
        local_eulers=eulers,
        raw_scores=scores,
        total_scores=scores,
        posterior=np.array([[0.5], [0.5]], dtype=np.float32),
        rotation_log_prior=np.zeros(2, dtype=np.float32),
        translation_log_prior=np.zeros(1, dtype=np.float32),
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
            raw_scores=np.array([[1.0]], dtype=np.float32),
            total_scores=np.array([[1.0]], dtype=np.float32),
            posterior=np.array([[1.0]], dtype=np.float32),
            rotation_log_prior=np.zeros(1, dtype=np.float32),
            translation_log_prior=np.zeros(1, dtype=np.float32),
            modes={"outside": np.array([90.0, 90.0, 90.0])},
            translation_index=0,
        )


def test_native_orientation_modes_decompose_cross_engine_gap():
    candidate = summarize_orientation_modes(
        local_eulers=np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        raw_scores=np.array([[4.0], [1.0]], dtype=np.float32),
        total_scores=np.array([[2.0], [1.0]], dtype=np.float32),
        posterior=np.array([[0.7], [0.3]], dtype=np.float32),
        rotation_log_prior=np.array([-2.0, 0.0], dtype=np.float32),
        translation_log_prior=np.zeros(1, dtype=np.float32),
        modes={"winner": np.array([10.0, 20.0, 30.0]), "alternate": np.array([40.0, 50.0, 60.0])},
        translation_index=0,
    )
    report = summarize_native_orientation_modes(
        native_candidate_rotation_rows=np.array([0, 1]),
        native_candidate_translation_indices=np.array([0, 0]),
        native_raw_diff2=np.array([10.0, 14.0]),
        native_rotation_log_prior=np.array([-3.0, 0.0]),
        native_translation_log_prior=np.zeros(2),
        native_posterior=np.array([0.6, 0.4]),
        candidate_boundary=candidate,
    )

    alternate = next(row for row in report["modes"] if row["label"] == "alternate")
    assert report["native_global_winner"]["local_rotation_row"] == 0
    assert alternate["native_winner_minus_mode_score"] == 1.0
    assert alternate["candidate_winner_minus_mode_score"] == 1.0
    assert alternate["candidate_minus_native_winner_gap"] == 0.0
    assert alternate["native_winner_minus_mode_components"] == {
        "raw_log_score": 4.0,
        "rotation_log_prior": -3.0,
        "translation_log_prior": 0.0,
    }


def test_native_orientation_modes_reject_ambiguous_mapping():
    candidate = summarize_orientation_modes(
        local_eulers=np.array([[10.0, 20.0, 30.0]]),
        raw_scores=np.array([[1.0]], dtype=np.float32),
        total_scores=np.array([[1.0]], dtype=np.float32),
        posterior=np.array([[1.0]], dtype=np.float32),
        rotation_log_prior=np.zeros(1, dtype=np.float32),
        translation_log_prior=np.zeros(1, dtype=np.float32),
        modes={"mode": np.array([10.0, 20.0, 30.0])},
        translation_index=0,
    )
    with pytest.raises(OrientationModeError, match="maps to 2 candidates"):
        summarize_native_orientation_modes(
            native_candidate_rotation_rows=np.array([0, 0]),
            native_candidate_translation_indices=np.array([0, 0]),
            native_raw_diff2=np.array([1.0, 2.0]),
            native_rotation_log_prior=np.zeros(2),
            native_translation_log_prior=np.zeros(2),
            native_posterior=np.array([0.6, 0.4]),
            candidate_boundary=candidate,
        )
