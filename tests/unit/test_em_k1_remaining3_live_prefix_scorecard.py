import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCORECARD_PATH = REPO_ROOT / "docs" / "math" / "em_k1_remaining3_live_prefix_scorecard_v1.json"
MARKDOWN_PATH = REPO_ROOT / "docs" / "math" / "em_relion_parity_scorecard.md"


@pytest.mark.unit
def test_remaining3_live_prefix_scorecard_has_fixed_scope_and_passing_gates():
    scorecard = json.loads(SCORECARD_PATH.read_text())

    assert scorecard["schema"] == "em_k1_remaining3_live_prefix_scorecard_v1"
    assert scorecard["completion_claim"] is False
    assert scorecard["counts"] == {"pass": 3, "fail": 0, "fixed_denominator": 3}
    assert scorecard["frozen_terminal_score"] == {
        "pass": 31,
        "fixed_denominator": 34,
        "changed_by_this_prefix_scorecard": False,
    }
    assert scorecard["thresholds"] == {
        "merged_cross_engine_fsc_auc_min": 0.995,
        "recovar_minus_relion_merged_gt_fsc_auc_min": -0.002,
    }

    cases = scorecard["cases"]
    assert [case["case_id"] for case in cases] == [4, 5, 10]
    latest_iterations = [case["latest_audited_relion_iteration"] for case in cases]
    assert all(current >= floor for current, floor in zip(latest_iterations, [4, 5, 8]))
    assert [case["controller_latest_audited_relion_iteration"] for case in cases] == latest_iterations
    for case in cases:
        assert case["status"] == "pass"
        assert case["topology_status"] == "pass"
        assert case["controller_topology_status"] == "pass"
        assert case["merged_cross_engine_fsc_auc"] >= 0.995
        assert case["recovar_minus_relion_merged_gt_fsc_auc"] >= -0.002
        assert len(case["audit_json_sha256"]) == 64
        int(case["audit_json_sha256"], 16)
        assert len(case["controller_audit_json_sha256"]) == 64
        int(case["controller_audit_json_sha256"], 16)
    assert all(case["frozen_baseline_same_iteration"] is None for case in cases)


@pytest.mark.unit
def test_remaining3_live_prefix_markdown_is_manual_and_does_not_promote_terminal_score():
    markdown = MARKDOWN_PATH.read_text()
    manual_start = markdown.index("<!-- BEGIN MANUAL POST-SNAPSHOT DIAGNOSTICS -->")
    live_gate = markdown.index("## Current exact-BPref candidate prefix gate")

    assert live_gate > manual_start
    assert "3 / 3 remaining K=1 cases pass their latest sealed numbered prefix" in markdown
    assert "31 / 34 until complete autonomous trajectories and final maps pass" in markdown
    for case_id in (4, 5, 10):
        assert f"| [x] | `k1-{case_id:02d}` |" in markdown
