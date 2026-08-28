"""Focused tests for the VDAM direct-map relative-L2 panel summarizer."""

from __future__ import annotations

import copy

import pytest

from scripts.summarize_vdam_map_relative_l2_panel import (
    MapRelativeL2PanelError,
    summarize_map_relative_l2_panel,
)

pytestmark = pytest.mark.unit


def _report(candidate: int, *, nearest: int = 0, within: bool = True):
    distances = [4.0, 5.0, 6.0, 7.0]
    distances[nearest] = 1.0
    return {
        "schema": "recovar.vdam_map_relative_l2_envelope.v1",
        "suite_id": "suite",
        "case_id": "case",
        "candidate_root": f"/candidate-{candidate}",
        "native_roots": [f"/native-{index}" for index in range(4)],
        "native_repeat_count": 4,
        "checkpoint_count": 1,
        "checkpoints": [
            {
                "iteration": 1,
                "candidate_native_relative_l2": distances,
                "candidate_best_native_relative_l2": 1.0,
                "native_repeat_relative_l2": [2.0, 3.0, 4.0, 2.0, 3.0, 2.0],
                "native_repeat_max_relative_l2": 4.0,
                "candidate_best_over_native_repeat_max_relative_l2": 0.25,
                "candidate_within_native_repeat_relative_l2_envelope": within,
            }
        ],
    }


def test_summarizes_complete_inside_panel_and_nearest_repeat_coverage() -> None:
    report = summarize_map_relative_l2_panel(
        [_report(index, nearest=index % 2) for index in range(4)]
    )
    assert report["status"] == "pass"
    assert report["outside_candidate_checkpoint_count"] == 0
    assert report["maximum_finite_candidate_over_native_repeat_envelope_ratio"] == 0.25
    row = report["checkpoints"][0]
    assert row["nearest_native_repeat_by_candidate"] == [1, 2, 1, 2]
    assert row["nearest_native_repeat_coverage"] == [1, 2]


def test_reports_one_candidate_checkpoint_outside() -> None:
    reports = [_report(index) for index in range(4)]
    reports[2]["checkpoints"][0][
        "candidate_within_native_repeat_relative_l2_envelope"
    ] = False
    report = summarize_map_relative_l2_panel(reports)
    assert report["status"] == "fail"
    assert report["first_iteration_outside_native_repeat_envelope"] == 1
    assert report["outside_candidate_checkpoint_count"] == 1


def test_rejects_mixed_native_envelope() -> None:
    reports = [_report(index) for index in range(4)]
    reports[3] = copy.deepcopy(reports[3])
    reports[3]["checkpoints"][0]["native_repeat_max_relative_l2"] = 5.0
    with pytest.raises(MapRelativeL2PanelError, match="native envelope differs"):
        summarize_map_relative_l2_panel(reports)
