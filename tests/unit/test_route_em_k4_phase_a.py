from __future__ import annotations

import pytest

from scripts import analyze_em_k4_raw_diff2_parity as phase_a_analyzer
from scripts import route_em_k4_phase_a as router


def _phase_a_report(
    *,
    raw_exact: bool,
    score_exact: bool,
    inert_score_operand_mismatch: bool = False,
) -> dict:
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
            "saved_score_replay": {
                "bitwise_exact": score_exact
                or inert_score_operand_mismatch
            },
            "native_vs_recovar_replayed_combined_score": {
                "bitwise_exact": score_exact
                or inert_score_operand_mismatch
            },
            "maximum_tie_sets_exact": score_exact
            or inert_score_operand_mismatch,
        },
    }


def _wrapped_phase_a_report(
    *,
    raw_exact: bool,
    score_exact: bool,
    inert_score_operand_mismatch: bool = False,
    wrapper_classification: str | None = None,
) -> dict:
    comparison = _phase_a_report(
        raw_exact=raw_exact,
        score_exact=score_exact,
        inert_score_operand_mismatch=inert_score_operand_mismatch,
    )
    comparison.pop("schema")
    comparison.pop("status")
    comparison.pop("classification_ready")
    comparison.pop("scorecard_change_admissible")
    if wrapper_classification is None:
        wrapper_classification = (
            router.WRAPPED_RAW_EXACT
            if raw_exact
            else router.WRAPPED_RAW_MISMATCH
        )
    accepted = wrapper_classification == router.WRAPPED_RAW_EXACT
    return {
        "schema": router.WRAPPED_PHASE_A_SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": wrapper_classification,
        "accepted_raw_boundary_exact": accepted,
        "phase_b_required": (
            wrapper_classification == router.WRAPPED_RAW_MISMATCH
        ),
        "scorecard_change_admissible": False,
        "correlation_used": False,
        "cross_engine": comparison,
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


def test_inert_prior_operand_mismatch_routes_to_joint_posterior() -> None:
    report = router.build_causal_route(
        _phase_a_report(
            raw_exact=True,
            score_exact=False,
            inert_score_operand_mismatch=True,
        )
    )

    assert report["route"] == router.INERT_OPERAND_JOINT_POSTERIOR_ROUTE
    assert report["phase_a"]["class1_score_effect_exact"] is True
    assert report["claims"]["class1_score_path_resolved"] is False
    assert report["claims"]["class1_combined_score_effect_resolved"] is True
    assert report["claims"][
        "class1_operand_mismatch_arithmetically_inert"
    ] is True
    assert report["claims"]["joint_class_pose_normalization_resolved"] is False
    assert report["scorecard_change_admissible"] is False


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


def test_wrapped_raw_match_but_score_mismatch_routes_to_prior_boundary() -> None:
    report = router.build_causal_route(
        _wrapped_phase_a_report(raw_exact=True, score_exact=False)
    )

    assert report["route"] == router.SCORE_PATH_MISMATCH_ROUTE
    assert report["phase_a"]["wrapper_classification"] == (
        router.WRAPPED_RAW_EXACT
    )
    assert report["phase_a"]["preconditions_passed"] is True
    assert report["phase_b_raw_operand_freeze_required"] is False


def test_wrapped_raw_mismatch_routes_to_bounded_operand_freeze() -> None:
    report = router.build_causal_route(
        _wrapped_phase_a_report(raw_exact=False, score_exact=False)
    )

    assert report["route"] == router.RAW_MISMATCH_ROUTE
    assert report["phase_b_raw_operand_freeze_required"] is True


def test_wrapped_inert_operand_mismatch_routes_to_joint_posterior() -> None:
    report = router.build_causal_route(
        _wrapped_phase_a_report(
            raw_exact=True,
            score_exact=False,
            inert_score_operand_mismatch=True,
        )
    )

    assert report["route"] == router.INERT_OPERAND_JOINT_POSTERIOR_ROUTE
    assert report["phase_a"]["wrapper_classification"] == (
        router.WRAPPED_RAW_EXACT
    )
    assert report["claims"][
        "class1_operand_mismatch_arithmetically_inert"
    ] is True


def test_wrapped_capture_rejection_does_not_authorize_operand_phase() -> None:
    phase_a = _wrapped_phase_a_report(
        raw_exact=False,
        score_exact=False,
        wrapper_classification="rejected_native_capture_inertness",
    )
    phase_a["cross_engine"]["accepted"] = False

    report = router.build_causal_route(phase_a)

    assert report["route"] == router.PRECONDITION_FAILURE_ROUTE
    assert report["phase_a"]["preconditions_passed"] is False
    assert report["phase_b_raw_operand_freeze_required"] is False
    assert report["claims"]["class1_raw_boundary_resolved"] is False


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


def test_rejects_inconsistent_wrapped_phase_b_decision() -> None:
    phase_a = _wrapped_phase_a_report(
        raw_exact=False,
        score_exact=False,
    )
    phase_a["phase_b_required"] = False

    with pytest.raises(ValueError, match="Phase-B decision"):
        router.build_causal_route(phase_a)
