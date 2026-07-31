from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_em_k1_serialized_restart_boundary as analyzer


def test_classifies_complete_serialized_restart_closure() -> None:
    assert (
        analyzer.classify_serialized_restart(
            qualified=True,
            dominated=14,
            absolute_gate_passed=14,
            expected_particles=14,
        )
        == analyzer.CLASSIFICATION
    )


def test_classification_preserves_absolute_gate_failure() -> None:
    assert (
        analyzer.classify_serialized_restart(
            qualified=True,
            dominated=14,
            absolute_gate_passed=13,
            expected_particles=14,
        )
        == (
            "serialized_restart_removes_majority_residual_but_not_"
            "absolute_score_gates"
        )
    )


def test_classification_rejects_zero_particle_effect() -> None:
    assert (
        analyzer.classify_serialized_restart(
            qualified=True,
            dominated=0,
            absolute_gate_passed=0,
            expected_particles=14,
        )
        == "serialized_restart_does_not_remove_fresh_process_residual"
    )


def test_classification_fails_closed_on_unqualified_inputs() -> None:
    assert (
        analyzer.classify_serialized_restart(
            qualified=False,
            dominated=14,
            absolute_gate_passed=14,
            expected_particles=14,
        )
        == "serialized_restart_inputs_not_qualified"
    )


def test_absolute_score_gate_uses_fixed_inclusive_p95_and_strict_max() -> None:
    at_gate = {
        "swapped_centered_p95_abs": analyzer.SCORE_P95_MAX_ABS,
        "swapped_centered_max_abs": (
            analyzer.SCORE_MAX_ABS_STRICTLY_BELOW - 1.0e-12
        ),
    }
    assert analyzer._absolute_gate(at_gate)
    at_gate["swapped_centered_max_abs"] = (
        analyzer.SCORE_MAX_ABS_STRICTLY_BELOW
    )
    assert not analyzer._absolute_gate(at_gate)


def test_direct_operand_validation_requires_all_fixed_gates(tmp_path) -> None:
    path = tmp_path / "operand.json"
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "classification_ready": True,
                "fixed_metric": {
                    "evaluated_particles": 14,
                    "expected_particles": 14,
                    "reference_replay_passed": 14,
                    "cross_replay_p95_passed": 14,
                    "cross_replay_max_passed": 14,
                    "production_diff2_centered_replay_p95_passed": 14,
                    "production_diff2_centered_replay_max_passed": 14,
                },
            }
        )
    )

    report = analyzer._validate_operand_report(path)

    assert report["status"] == "pass"


def test_direct_operand_validation_fails_closed(tmp_path) -> None:
    path = tmp_path / "operand.json"
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "classification_ready": True,
                "fixed_metric": {
                    "evaluated_particles": 14,
                    "expected_particles": 14,
                    "reference_replay_passed": 14,
                    "cross_replay_p95_passed": 14,
                    "cross_replay_max_passed": 14,
                    "production_diff2_centered_replay_p95_passed": 13,
                    "production_diff2_centered_replay_max_passed": 14,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="fixed gates"):
        analyzer._validate_operand_report(path)


def test_explicit_fresh_capture_requires_complete_qualification_tuple() -> None:
    assert not analyzer._validate_fresh_capture_arguments(
        fresh_capture_directory=None,
        fresh_component_validation_json=None,
        fresh_operand_validation_json=None,
    )
    assert analyzer._validate_fresh_capture_arguments(
        fresh_capture_directory=Path("/fresh"),
        fresh_component_validation_json=Path("/component.json"),
        fresh_operand_validation_json=Path("/operand.json"),
    )

    with pytest.raises(ValueError, match="supplied together"):
        analyzer._validate_fresh_capture_arguments(
            fresh_capture_directory=Path("/fresh"),
            fresh_component_validation_json=Path("/component.json"),
            fresh_operand_validation_json=None,
        )


def test_explicit_fresh_capture_replaces_parent_particle_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fresh_directory = tmp_path / "fresh"
    restart_directory = tmp_path / "restart"
    rows = [
        {
            "group": "control",
            "stack_index_one_based": index + 1,
            "original_index_zero_based": index,
            "artifact_paths": {
                "component": str(tmp_path / f"parent_{index}.bin"),
                "recovar": str(tmp_path / f"recovar_{index}.npz"),
            },
            "artifact_sha256": {
                "component": f"sha:parent_{index}.bin",
                "recovar": f"sha:recovar_{index}.npz",
            },
        }
        for index in range(14)
    ]
    parent = {
        "particles": rows,
        "shell_partition_metric": {
            "classification": analyzer.SHELL_PARTITION_CLASSIFICATION
        },
    }
    component_validation = {
        "status": "pass",
        "fixed_metric": {"evaluated_particles": 14},
    }
    operand_fixed_metric = {
        "evaluated_particles": 14,
        "expected_particles": 14,
        "reference_replay_passed": 14,
        "cross_replay_p95_passed": 14,
        "cross_replay_max_passed": 14,
        "production_diff2_centered_replay_p95_passed": 14,
        "production_diff2_centered_replay_max_passed": 14,
    }
    operand_validation = {
        "status": "pass",
        "classification_ready": True,
        "fixed_metric": operand_fixed_metric,
    }

    monkeypatch.setattr(analyzer, "_validate_parent", lambda _path: parent)
    monkeypatch.setattr(
        analyzer,
        "_validate_component_report",
        lambda _path: component_validation,
    )
    monkeypatch.setattr(
        analyzer,
        "_validate_operand_report",
        lambda _path: operand_validation,
    )
    monkeypatch.setattr(
        analyzer,
        "_sha256",
        lambda path: f"sha:{Path(path).name}",
    )

    def component_by_stack(directory: Path) -> dict[int, SimpleNamespace]:
        prefix = "fresh" if directory == fresh_directory else "restart"
        return {
            index + 1: SimpleNamespace(
                stack_index=index + 1,
                part_id=index + 101,
                header=[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 8, 4, 3, 96],
                path=directory / f"{prefix}_{index}.bin",
                sha256=f"sha:{prefix}_{index}.bin",
                raw_kind=prefix,
            )
            for index in range(14)
        }

    monkeypatch.setattr(analyzer, "_component_by_stack", component_by_stack)
    monkeypatch.setattr(
        analyzer,
        "_load_recovar",
        lambda path: {
            "original_index": int(path.stem.removeprefix("recovar_")),
            "scores": np.asarray([0.0]),
            "rotation_ids": np.asarray([0]),
            "sha256": f"sha:{path.name}",
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_mapped_selected_raw",
        lambda component, recovar: (
            np.asarray([1.0 if component.raw_kind == "fresh" else 2.0]),
            {"exact": True},
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "reference_swap_counterfactual",
        lambda _fresh, _restart: {
            "live_reference_dominated": True,
            "counterfactual_energy_removal_fraction": 1.0,
            "swapped_centered_p95_abs": 0.0,
            "swapped_centered_max_abs": 0.0,
        },
    )

    report = analyzer.build_report(
        parent_analysis_json=tmp_path / "parent.json",
        restart_capture_directory=restart_directory,
        restart_component_validation_json=tmp_path / "restart_component.json",
        restart_operand_validation_json=tmp_path / "restart_operand.json",
        source_optimiser_star=tmp_path / "run_it000_optimiser.star",
        fresh_capture_directory=fresh_directory,
        fresh_component_validation_json=tmp_path / "fresh_component.json",
        fresh_operand_validation_json=tmp_path / "fresh_operand.json",
    )

    assert report["classification"] == analyzer.CLASSIFICATION
    assert report["fixed_metric"]["serialized_restart_dominated"] == 14
    assert report["fresh_capture_source"]["mode"] == (
        "explicit_same_allocation_fresh_capture"
    )
    assert all(
        Path(row["artifact_paths"]["fresh_component"]).parent
        == fresh_directory
        for row in report["particles"]
    )
