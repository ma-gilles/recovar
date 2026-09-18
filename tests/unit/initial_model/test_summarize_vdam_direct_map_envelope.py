from __future__ import annotations

import pytest

from scripts.summarize_vdam_direct_map_envelope import (
    PAIR_SCHEMA,
    DirectMapEnvelopeError,
    summarize_direct_map_envelope,
)


def _report(values: list[float], passes: list[bool], *, threshold: float = 0.999):
    checkpoints = list(range(len(values)))
    return {
        "schema": PAIR_SCHEMA,
        "K": 1,
        "thresholds": {
            "minimum_class_assignment_accuracy": 0.999,
            "minimum_per_class_fsc_auc": threshold,
        },
        "checkpoints": checkpoints,
        "iterations": [
            {
                "iteration": iteration,
                "minimum_matched_fsc_auc": value,
                "pass": passed,
            }
            for iteration, (value, passed) in enumerate(zip(values, passes, strict=True))
        ],
    }


def test_direct_map_envelope_accepts_any_complete_native_mode() -> None:
    report = summarize_direct_map_envelope(
        [
            _report([1.0, 0.998, 0.997], [True, False, False]),
            _report([1.0, 0.9995, 0.9985], [True, True, False]),
        ]
    )

    assert report["result"] == "fail"
    assert report["first_failure_iteration"] == 2
    assert report["failure_count"] == 1
    assert report["iterations"][1]["matching_native_repeat_indices"] == [2]
    assert report["iterations"][1]["candidate_best_native_minimum_fsc_auc"] == pytest.approx(
        0.9995
    )


def test_direct_map_envelope_rejects_mixed_thresholds() -> None:
    with pytest.raises(DirectMapEnvelopeError, match="K or thresholds differ"):
        summarize_direct_map_envelope(
            [
                _report([1.0], [True]),
                _report([1.0], [True], threshold=0.99),
            ]
        )
