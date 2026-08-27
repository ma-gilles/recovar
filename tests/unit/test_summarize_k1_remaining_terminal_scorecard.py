import json

import pytest

from scripts.summarize_k1_remaining_terminal_scorecard import (
    ScorecardError,
    build_scorecard,
)


def _baseline(case_id: int, final: float) -> dict:
    return {
        "case_id": case_id,
        "fixture": f"case_{case_id}",
        "numbered_iteration_count": 2,
        "minimum_numbered_merged_cross_engine_fsc_auc": 0.999,
        "last_numbered_merged_cross_engine_fsc_auc": 0.999,
        "topology_status": "pass",
        "final_merged_cross_engine_fsc_auc": final,
        "final_recovar_minus_relion_merged_gt_fsc_auc": 0.0,
        "status": "fail_final_cross_engine",
    }


def _report(final: float, *, numbered: float = 0.999, gt_delta: float = 0.0, topology=None) -> dict:
    return {
        "numbered_iterations": [
            {"cross_engine": {"merged": {"signed_fsc_auc": numbered}}},
            {"cross_engine": {"merged": {"signed_fsc_auc": numbered + 1e-5}}},
        ],
        "final": {
            "cross_engine": {"merged": {"signed_fsc_auc": final}},
            "merged_gt_fsc_auc_delta": gt_delta,
        },
        "topology_failures": [] if topology is None else topology,
    }


def test_build_scorecard_promotes_only_complete_fixed_gate_passes(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "em_k1_remaining3_terminal_scorecard_v1",
                "fixed_suite_score": {
                    "pass": 32,
                    "fail": 2,
                    "fixed_denominator": 34,
                },
                "thresholds": {
                    "merged_cross_engine_fsc_auc_min": 0.995,
                    "recovar_minus_relion_merged_gt_fsc_auc_min": -0.002,
                },
                "cases": [_baseline(4, 0.99), _baseline(5, 0.98)],
            }
        )
    )
    case4 = tmp_path / "case4.json"
    case5 = tmp_path / "case5.json"
    case4.write_text(json.dumps(_report(0.996)))
    case5.write_text(json.dumps(_report(0.9949)))

    report = build_scorecard(
        baseline_json=baseline_path,
        replacements={4: case4, 5: case5},
        fixed_total=34,
        baseline_passing=32,
    )

    assert report["score"] == "33/34"
    assert report["baseline_schema"] == "em_k1_remaining3_terminal_scorecard_v1"
    assert report["status"] == "incomplete"
    assert report["cases"][0]["status"] == "pass"
    assert report["cases"][0]["final_merged_cross_engine_fsc_auc_change"] == pytest.approx(0.006)
    assert report["cases"][1]["status"] == "fail"


@pytest.mark.parametrize(
    "replacement",
    [
        _report(0.996, numbered=0.994),
        _report(0.996, gt_delta=-0.0021),
        _report(0.996, topology=[{"iteration": 2}]),
    ],
)
def test_build_scorecard_preserves_numbered_gt_and_topology_gates(tmp_path, replacement):
    baseline_path = tmp_path / "baseline.json"
    replacement_path = tmp_path / "replacement.json"
    baseline_path.write_text(json.dumps([_baseline(4, 0.99)]))
    replacement_path.write_text(json.dumps(replacement))

    report = build_scorecard(
        baseline_json=baseline_path,
        replacements={4: replacement_path},
        fixed_total=34,
        baseline_passing=33,
    )

    assert report["score"] == "33/34"
    assert report["cases"][0]["status"] == "fail"


def test_build_scorecard_rejects_unknown_replacement_case(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    replacement_path = tmp_path / "replacement.json"
    baseline_path.write_text(json.dumps([_baseline(4, 0.99)]))
    replacement_path.write_text(json.dumps(_report(0.996)))

    with pytest.raises(ScorecardError, match="absent from baseline"):
        build_scorecard(
            baseline_json=baseline_path,
            replacements={5: replacement_path},
            fixed_total=34,
            baseline_passing=33,
        )


def test_build_scorecard_rejects_changed_frozen_threshold(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "thresholds": {
                    "merged_cross_engine_fsc_auc_min": 0.99,
                    "recovar_minus_relion_merged_gt_fsc_auc_min": -0.002,
                },
                "cases": [_baseline(4, 0.99)],
            }
        )
    )

    with pytest.raises(ScorecardError, match="frozen thresholds"):
        build_scorecard(
            baseline_json=baseline_path,
            replacements={},
            fixed_total=34,
            baseline_passing=33,
        )
