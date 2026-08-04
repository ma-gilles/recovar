from __future__ import annotations

import pytest

from scripts import analyze_em_k4_raw_diff2_parity as phase_a_analyzer
from scripts import route_em_k4_phase_a as router


def _phase_a_report(*, raw_exact: bool, score_exact: bool) -> dict:
    raw_classification = (
        phase_a_analyzer.PASS_CLASSIFICATION
        if raw_exact
        else "exact_device_k4_raw_diff2_mismatch__raw_diff2"
    )
    score_classification = (
        phase_a_analyzer.PASS_SCORE_CLASSIFICATION
        if score_exact
        else "exact_device_k4_score_path_mismatch__rotation_prior"
    )
    return {
        "schema": phase_a_analyzer.SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": raw_classification,
        "accepted": raw_exact,
        "scorecard_change_admissible": False,
        "support": {"exact": True},
        "score_path": {
            "classification": score_classification,
            "accepted": score_exact,
        },
    }


def test_raw_mismatch_routes_only_to_bounded_operand_freeze() -> None:
    report = router.build_causal_route(
        _phase_a_report(raw_exact=False, score_exact=False)
    )

    assert report["route"] == router.RAW_MISMATCH_ROUTE
    assert report["phase_b_raw_operand_freeze_required"] is True
    assert report["claims"]["class1_raw_boundary_resolved"] is False
    assert report["scorecard_change_admissible"] is False


def test_raw_match_but_score_mismatch_routes_to_prior_boundary() -> None:
    report = router.build_causal_route(
        _phase_a_report(raw_exact=True, score_exact=False)
    )

    assert report["route"] == router.SCORE_PATH_MISMATCH_ROUTE
    assert report["phase_b_raw_operand_freeze_required"] is False
    assert report["claims"]["class1_raw_boundary_resolved"] is True
    assert report["claims"]["class1_score_path_resolved"] is False


def test_class1_score_match_still_requires_joint_k4_capture() -> None:
    report = router.build_causal_route(
        _phase_a_report(raw_exact=True, score_exact=True)
    )

    assert report["route"] == router.JOINT_POSTERIOR_ROUTE
    assert report["claims"]["class1_score_path_resolved"] is True
    assert report["claims"]["all_class_tuple_and_score_boundary_resolved"] is False
    assert report["claims"]["joint_class_pose_normalization_resolved"] is False
    assert report["claims"]["joint_significance_resolved"] is False
    assert report["claims"]["map_parity_established"] is False
    assert report["scorecard_change_admissible"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "running", "incomplete"),
        ("classification_ready", False, "not ready"),
        ("scorecard_change_admissible", True, "must not authorize"),
    ],
)
def test_rejects_nonterminal_or_overclaimed_reports(
    field: str,
    value: object,
    message: str,
) -> None:
    phase_a = _phase_a_report(raw_exact=True, score_exact=True)
    phase_a[field] = value

    with pytest.raises(ValueError, match=message):
        router.build_causal_route(phase_a)


def test_rejects_inconsistent_raw_acceptance_flag() -> None:
    phase_a = _phase_a_report(raw_exact=True, score_exact=True)
    phase_a["accepted"] = False

    with pytest.raises(ValueError, match="raw classification"):
        router.build_causal_route(phase_a)


def test_rejects_inconsistent_score_path_acceptance_flag() -> None:
    phase_a = _phase_a_report(raw_exact=True, score_exact=True)
    phase_a["score_path"]["accepted"] = False

    with pytest.raises(ValueError, match="score-path classification"):
        router.build_causal_route(phase_a)
