"""Tests for the bounded K=1 coarse operand report envelope."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts import analyze_k1_coarse_operand_boundary_v3 as analyzer


def _run_report(
    tmp_path: Path,
    monkeypatch,
    *,
    physical_iteration: int,
    validations: dict[str, bool],
) -> tuple[dict, list[int]]:
    native_directory = tmp_path / "capture"
    recovar_directory = tmp_path / "coarse"
    analysis_directory = tmp_path / "analysis"
    native_directory.mkdir()
    recovar_directory.mkdir()
    analysis_directory.mkdir()

    stack_index = 79
    original_index = 78
    (native_directory / f"part2767_stack{stack_index}.p1-v2.bin").touch()
    (native_directory / f"part2767_stack{stack_index}.p1-op-v2.bin").touch()
    (recovar_directory / f"significance_orig{original_index:06d}_it003_cs080.npz").touch()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "case_id": 22,
                "physical_iteration": physical_iteration,
                "targets": [
                    {
                        "stack_index_one_based": stack_index,
                        "original_index_zero_based": original_index,
                    }
                ],
            }
        )
    )
    for label, ready in validations.items():
        filename = {
            "components": "components_validation.json",
            "operands": "operand_validation.json",
        }[label]
        (analysis_directory / filename).write_text(
            json.dumps(
                {
                    "status": "accepted" if ready else "rejected",
                    "classification_ready": ready,
                }
            )
        )

    observed_iterations: list[int] = []

    def fake_compare(*_args, physical_iteration: int):
        observed_iterations.append(physical_iteration)
        return {"stack_index_one_based": stack_index}

    output_path = tmp_path / "report.json"
    monkeypatch.setattr(analyzer, "_compare", fake_compare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_k1_coarse_operand_boundary_v3.py",
            "--native-directory",
            str(native_directory),
            "--recovar-directory",
            str(recovar_directory),
            "--selection-json",
            str(selection_path),
            "--output-json",
            str(output_path),
        ],
    )
    analyzer.main()
    return json.loads(output_path.read_text()), observed_iterations


def test_report_uses_selection_physical_iteration_and_validation_gates(
    tmp_path: Path,
    monkeypatch,
):
    report, observed_iterations = _run_report(
        tmp_path,
        monkeypatch,
        physical_iteration=3,
        validations={"components": False, "operands": True},
    )

    assert observed_iterations == [3]
    assert report["physical_iteration"] == 3
    assert report["classification_ready"] is False
    assert report["capture_validation"]["components"]["status"] == "rejected"
    assert report["capture_validation"]["operands"]["status"] == "accepted"


def test_report_rejects_missing_capture_validations(tmp_path: Path, monkeypatch):
    report, _ = _run_report(
        tmp_path,
        monkeypatch,
        physical_iteration=3,
        validations={},
    )

    assert report["classification_ready"] is False
    assert report["capture_validation"]["components"]["status"] == "missing"
    assert report["capture_validation"]["operands"]["status"] == "missing"


def test_operand_panel_uses_active_overlap_in_requested_order():
    selected, positions, operand_order, recovar_only, native_only = (
        analyzer._matched_operand_rotation_panel(
            np.asarray([534, 27288, 9], dtype=np.int64),
            np.asarray([19000, 534, 9], dtype=np.int64),
            np.asarray([True, True, False]),
        )
    )

    np.testing.assert_array_equal(selected, [534])
    np.testing.assert_array_equal(positions, [0])
    np.testing.assert_array_equal(operand_order, [1])
    assert recovar_only == [27288]
    assert native_only == [19000]
