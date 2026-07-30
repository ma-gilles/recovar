from __future__ import annotations

import json

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
