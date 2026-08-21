from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import analyze_em_k1_serialized_restart_maps as analyzer


def test_classifies_all_map_improvement_without_gt_regression() -> None:
    assert (
        analyzer.classify_map_effect(
            parity_improved=3,
            gt_nondegraded=3,
            expected_maps=3,
            merged_parity_improved=True,
            merged_gt_nondegraded=True,
        )
        == analyzer.CLASSIFICATION
    )


def test_classification_preserves_gt_regression() -> None:
    assert (
        analyzer.classify_map_effect(
            parity_improved=3,
            gt_nondegraded=2,
            expected_maps=3,
            merged_parity_improved=True,
            merged_gt_nondegraded=True,
        )
        == (
            "serialized_restart_improves_all_case22_iteration2_map_"
            "parity_but_regresses_gt_fsc_auc"
        )
    )


def test_classification_reports_merged_only_improvement() -> None:
    assert (
        analyzer.classify_map_effect(
            parity_improved=1,
            gt_nondegraded=3,
            expected_maps=3,
            merged_parity_improved=True,
            merged_gt_nondegraded=True,
        )
        == (
            "serialized_restart_improves_case22_iteration2_merged_"
            "map_parity_only"
        )
    )


def test_classification_reports_no_parity_improvement() -> None:
    assert (
        analyzer.classify_map_effect(
            parity_improved=0,
            gt_nondegraded=3,
            expected_maps=3,
            merged_parity_improved=False,
            merged_gt_nondegraded=True,
        )
        == (
            "serialized_restart_does_not_improve_case22_iteration2_"
            "map_parity"
        )
    )


def _score_report() -> dict:
    return {
        "schema": "em-k1-serialized-restart-boundary-v1",
        "status": "complete",
        "classification_ready": True,
        "classification": analyzer.SCORE_BOUNDARY_CLASSIFICATION,
        "fixed_metric": {
            "evaluated_particles": 14,
            "expected_particles": 14,
            "serialized_restart_dominated": 14,
            "absolute_score_gate_passed": 14,
        },
    }


def test_score_report_requires_fixed_14_of_14_gate(tmp_path) -> None:
    path = tmp_path / "score.json"
    path.write_text(json.dumps(_score_report()))

    report = analyzer._validate_score_report(path)

    assert report["classification"] == analyzer.SCORE_BOUNDARY_CLASSIFICATION


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("classification_ready", False, "not classification-ready"),
        ("classification", "different", "did not pass"),
    ],
)
def test_score_report_fails_closed_on_unqualified_input(
    tmp_path,
    field,
    value,
    match,
) -> None:
    report = _score_report()
    report[field] = value
    path = tmp_path / "score.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match=match):
        analyzer._validate_score_report(path)


def test_score_report_fails_closed_on_partial_fixed_metric(tmp_path) -> None:
    report = _score_report()
    report["fixed_metric"]["absolute_score_gate_passed"] = 13
    path = tmp_path / "score.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="14/14"):
        analyzer._validate_score_report(path)


def test_map_paths_use_iteration_two_and_explicit_engine_roots(tmp_path) -> None:
    paths = analyzer._paths(
        recovar_root=tmp_path / "rec",
        fresh_relion_root=tmp_path / "fresh",
        restart_relion_root=tmp_path / "restart",
        relion_iteration=2,
    )

    assert paths["recovar"]["merged"] == (
        tmp_path / "rec" / "recovar" / "final_merged.mrc"
    )
    assert paths["fresh_relion"]["half1"] == (
        tmp_path / "fresh" / "relion" / "run_it002_half1_class001.mrc"
    )
    assert paths["restart_relion"]["half2"] == (
        tmp_path / "restart" / "relion" / "run_it002_half2_class001.mrc"
    )


def test_build_report_counts_fixed_fsc_auc_gates(
    tmp_path,
    monkeypatch,
) -> None:
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(_score_report()))
    recovar_root = tmp_path / "rec"
    fresh_root = tmp_path / "fresh"
    restart_root = tmp_path / "restart"
    gt_path = tmp_path / "reference_gt.mrc"
    paths = analyzer._paths(
        recovar_root=recovar_root,
        fresh_relion_root=fresh_root,
        restart_relion_root=restart_root,
        relion_iteration=2,
    )
    for arm_paths in paths.values():
        for path in arm_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    gt_path.touch()

    rng = np.random.default_rng(7)
    gt = rng.normal(size=(16, 16, 16))
    noises = {
        label: np.random.default_rng(seed).normal(size=gt.shape)
        for label, seed in (("half1", 1), ("half2", 2))
    }
    arrays = {
        paths["recovar"]["half1"]: gt,
        paths["recovar"]["half2"]: gt,
        paths["recovar"]["merged"]: gt,
        paths["fresh_relion"]["half1"]: gt + 2.0 * noises["half1"],
        paths["fresh_relion"]["half2"]: gt + 2.0 * noises["half2"],
        paths["restart_relion"]["half1"]: gt + 0.5 * noises["half1"],
        paths["restart_relion"]["half2"]: gt + 0.5 * noises["half2"],
        gt_path: gt,
    }

    def load(path):
        return arrays[path]

    monkeypatch.setattr(analyzer, "_load_recovar_volume", load)
    monkeypatch.setattr(analyzer, "_load_relion_volume", load)

    report = analyzer.build_report(
        score_analysis_json=score_path,
        recovar_root=recovar_root,
        fresh_relion_root=fresh_root,
        restart_relion_root=restart_root,
        gt_volume=gt_path,
        relion_iteration=2,
    )

    assert report["classification"] == analyzer.CLASSIFICATION
    assert report["overall_intervention_accepted"] is True
    assert report["fixed_metric"] == {
        "parity_strictly_improved": 3,
        "gt_nondegraded": 3,
        "evaluated_maps": 3,
        "expected_maps": 3,
        "score_boundary_passed": True,
    }
    for label in analyzer.MAP_LABELS:
        row = report["comparisons"][label]
        assert row["restart_minus_fresh_parity_fsc_auc"] > 0.0
        assert row["restart_minus_fresh_gt_fsc_auc"] > 0.0
        assert row["fresh_vs_restart_relion"]["fsc"]

    failed_score = _score_report()
    failed_score["fixed_metric"]["absolute_score_gate_passed"] = 13
    failed_score["classification"] = (
        "serialized_restart_removes_majority_residual_but_not_"
        "absolute_score_gates"
    )
    score_path.write_text(json.dumps(failed_score))
    diagnostic = analyzer.build_report(
        score_analysis_json=score_path,
        recovar_root=recovar_root,
        fresh_relion_root=fresh_root,
        restart_relion_root=restart_root,
        gt_volume=gt_path,
        relion_iteration=2,
        require_score_boundary_pass=False,
    )

    assert diagnostic["classification"] == analyzer.CLASSIFICATION
    assert diagnostic["overall_intervention_accepted"] is False
    assert diagnostic["fixed_metric"]["score_boundary_passed"] is False
    assert diagnostic["score_boundary"]["required_for_map_evaluation"] is False
