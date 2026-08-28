from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_vdam_mstep_repeat_panel import SCHEMA as PANEL_SCHEMA
from scripts.summarize_vdam_mstep_repeat_panels import (
    RAW_ACCUMULATOR_STAGES,
    summarize_repeat_panels,
)


def _write_panel(path: Path, *, directory_tag: str, native: float, ratios: tuple[float, float]) -> None:
    stages = {
        stage: {
            "cross_arm_a_over_native_repeat": ratios[0],
            "cross_arm_b_over_native_repeat": ratios[1],
        }
        for stage in RAW_ACCUMULATOR_STAGES
    }
    path.write_text(
        json.dumps(
            {
                "schema": PANEL_SCHEMA,
                "status": "complete",
                "iteration": 58,
                "directories": {"native_a": f"/{directory_tag}/native-a"},
                "native_repeat": {
                    stage: {"relative_l2": native} for stage in stages
                },
                "native_floor_ratios": stages,
            }
        )
    )


def test_summary_reports_distribution_and_raw_native_floor_gate(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_panel(first, directory_tag="first", native=0.1, ratios=(0.25, 0.5))
    _write_panel(second, directory_tag="second", native=0.3, ratios=(0.75, 1.25))

    report = summarize_repeat_panels([first, second], iteration=58)

    assert report["panel_count"] == 2
    stage = report["stages"][RAW_ACCUMULATOR_STAGES[0]]
    assert stage["native_repeat_relative_l2"]["median"] == pytest.approx(0.2)
    assert stage["cross_over_native_repeat"]["median"] == pytest.approx(0.625)
    assert stage["cross_over_native_repeat"]["within_native_floor_count"] == 3
    gate = report["raw_accumulator_gate"]
    assert gate["comparison_count"] == 16
    assert gate["within_native_floor_count"] == 12
    assert gate["all_within_native_floor"] is False
    assert all(len(source["sha256"]) == 64 for source in report["source_reports"])


@pytest.mark.parametrize("mutation", ["status", "iteration"])
def test_summary_rejects_unqualified_panel(tmp_path, mutation):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_panel(first, directory_tag="first", native=0.1, ratios=(0.25, 0.5))
    _write_panel(second, directory_tag="second", native=0.3, ratios=(0.75, 1.25))
    payload = json.loads(second.read_text())
    payload[mutation] = "failed" if mutation == "status" else 57
    second.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=f"expected {mutation}"):
        summarize_repeat_panels([first, second], iteration=58)


def test_summary_requires_independent_panels(tmp_path):
    panel = tmp_path / "panel.json"
    _write_panel(panel, directory_tag="only", native=0.1, ratios=(0.25, 0.5))

    with pytest.raises(ValueError, match="at least two"):
        summarize_repeat_panels([panel], iteration=58)
    with pytest.raises(ValueError, match="paths must be unique"):
        summarize_repeat_panels([panel, panel], iteration=58)
