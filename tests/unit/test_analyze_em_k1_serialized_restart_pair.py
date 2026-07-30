from __future__ import annotations

import hashlib
import json

import pytest

from scripts import analyze_em_k1_serialized_restart_pair as analyzer


@pytest.mark.parametrize(
    ("iteration0", "iteration1", "classification", "interpretation"),
    [
        (
            True,
            True,
            "both_serialized_restart_points_close_score_and_map_gates",
            "not_preserved_by_either_serialized_restart",
        ),
        (
            True,
            False,
            "only_iteration0_restart_closes_score_and_map_gates",
            "replaying_iteration1_from_serialized_it0",
        ),
        (
            False,
            True,
            "only_iteration1_restart_closes_score_and_map_gates",
            "direct_iteration1_restart_specific",
        ),
        (
            False,
            False,
            "neither_serialized_restart_point_closes_score_and_map_gates",
            "serialized_restart_hypothesis_rejected",
        ),
    ],
)
def test_classifies_restart_pair(
    iteration0: bool,
    iteration1: bool,
    classification: str,
    interpretation: str,
) -> None:
    observed, causal = analyzer.classify_restart_pair(
        iteration0_accepted=iteration0,
        iteration1_accepted=iteration1,
    )

    assert observed == classification
    assert interpretation in causal


def _write_arm(tmp_path, label: str, *, passed: bool):
    score_path = tmp_path / f"{label}_score.json"
    score = {
        "schema": "em-k1-serialized-restart-boundary-v1",
        "status": "complete",
        "classification_ready": True,
        "classification": (
            analyzer.SCORE_CLASSIFICATION
            if passed
            else (
                "serialized_restart_removes_majority_residual_but_not_"
                "absolute_score_gates"
            )
        ),
        "fixed_metric": {
            "evaluated_particles": 14,
            "expected_particles": 14,
            "serialized_restart_dominated": 14,
            "absolute_score_gate_passed": 14 if passed else 13,
        },
    }
    score_path.write_text(json.dumps(score))
    score_sha = hashlib.sha256(score_path.read_bytes()).hexdigest()
    map_path = tmp_path / f"{label}_map.json"
    maps = {
        "schema": "em-k1-serialized-restart-map-fsc-v2",
        "status": "complete",
        "classification_ready": True,
        "classification": analyzer.MAP_CLASSIFICATION,
        "overall_intervention_accepted": passed,
        "fixed_metric": {
            "parity_strictly_improved": 3,
            "gt_nondegraded": 3,
            "evaluated_maps": 3,
            "expected_maps": 3,
            "score_boundary_passed": passed,
        },
        "score_boundary": {
            "path": str(score_path.resolve()),
            "sha256": score_sha,
            "passed": passed,
        },
    }
    map_path.write_text(json.dumps(maps))
    return score_path, map_path


def test_build_report_preserves_fixed_two_arm_denominator(tmp_path) -> None:
    it0_score, it0_map = _write_arm(tmp_path, "it0", passed=True)
    it1_score, it1_map = _write_arm(tmp_path, "it1", passed=False)

    report = analyzer.build_report(
        iteration0_score_path=it0_score,
        iteration0_map_path=it0_map,
        iteration1_score_path=it1_score,
        iteration1_map_path=it1_map,
    )

    assert (
        report["classification"]
        == "only_iteration0_restart_closes_score_and_map_gates"
    )
    assert report["fixed_metric"] == {
        "evaluated_restart_arms": 2,
        "expected_restart_arms": 2,
        "score_gate_passed_arms": 1,
        "map_parity_gate_passed_arms": 2,
        "map_gt_gate_passed_arms": 2,
        "overall_passed_arms": 1,
    }
    assert report["scorecard_change_admissible"] is False


def test_map_report_requires_hash_linked_score(tmp_path) -> None:
    score_path, map_path = _write_arm(tmp_path, "arm", passed=True)
    payload = json.loads(map_path.read_text())
    payload["score_boundary"]["sha256"] = "0" * 64
    map_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hash-linked"):
        analyzer._load_map(
            map_path,
            expected_score_path=score_path,
            expected_score_passed=True,
        )


def test_map_report_replays_overall_acceptance(tmp_path) -> None:
    score_path, map_path = _write_arm(tmp_path, "arm", passed=False)
    payload = json.loads(map_path.read_text())
    payload["overall_intervention_accepted"] = True
    map_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not replay"):
        analyzer._load_map(
            map_path,
            expected_score_path=score_path,
            expected_score_passed=False,
        )
