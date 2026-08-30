from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import summarize_vdam_relion_parity_scorecard as scorecard_mod


@pytest.fixture
def scorecard() -> dict:
    value = json.loads(scorecard_mod.DEFAULT_SCORECARD.read_text())
    for case in value["cases"]:
        case.update(result="not_run", checkpoint_results={}, evidence=None)
    value["history"] = [value["history"][0]]
    value["current_snapshot"] = {
        "id": value["history"][0]["id"],
        "counts": {"pass": 0, "fail": 0, "not_run": 12},
        "evidence": None,
    }
    return value


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _set_counts(value: dict) -> None:
    counts = {name: 0 for name in ("pass", "fail", "not_run")}
    for case in value["cases"]:
        counts[case["result"]] += 1
    value["current_snapshot"]["counts"] = counts
    value["history"][-1]["counts"] = counts


def _evaluate_first_case(value: dict, *, cross: float = 0.9995, gt_delta: float = -0.001) -> None:
    case = value["cases"][0]
    case["checkpoint_results"] = {str(i): "pass" for i in scorecard_mod.REQUIRED_CHECKPOINTS}
    case["evidence"] = {
        "source_head": "1" * 40,
        "report_sha256": "2" * 64,
        "recovar_job": "101",
        "relion_job": "102",
        "audit_job": "103",
        "same_physical_gpu": True,
        "correlation_used": False,
        "exact_schedule": True,
        "exact_artifact_topology": True,
        "final_cross_engine_fsc_auc": cross,
        "final_gt_fsc_auc_delta": gt_delta,
    }
    case["result"] = "pass" if cross >= 0.999 and gt_delta >= -0.002 else "fail"
    _set_counts(value)


def test_checked_scorecard_is_valid() -> None:
    loaded = scorecard_mod.load_and_validate()
    assert loaded["frozen_denominator"] == 12
    assert loaded["current_snapshot"]["counts"] == {"pass": 12, "fail": 0, "not_run": 0}


def test_definition_digest_is_reproducible(scorecard: dict) -> None:
    assert scorecard_mod.frozen_case_definitions_sha256(scorecard["cases"]) == (
        scorecard_mod.FROZEN_CASE_DEFINITIONS_SHA256
    )


def test_suite_reuses_twelve_distinct_fixed_em_fixtures(scorecard: dict) -> None:
    source_ids = [case["definition"]["source_em_case_id"] for case in scorecard["cases"]]
    assert len(source_ids) == len(set(source_ids)) == 12


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("nr_iter", 9),
        ("nr_classes", 2),
        ("padding_factor", 2),
        ("tau2_fudge", 1.0),
        ("source_em_case_id", "k1-34"),
    ],
)
def test_frozen_definition_changes_require_new_suite_version(
    tmp_path: Path, scorecard: dict, field: str, replacement: object
) -> None:
    scorecard["cases"][0]["definition"][field] = replacement
    with pytest.raises(ValueError, match="frozen VDAM case definitions changed"):
        scorecard_mod.load_and_validate(_write(tmp_path, scorecard))


def test_case_denominator_cannot_shrink(tmp_path: Path, scorecard: dict) -> None:
    scorecard["cases"].pop()
    with pytest.raises(ValueError, match="case denominator mismatch"):
        scorecard_mod.load_and_validate(_write(tmp_path, scorecard))


def test_acceptance_threshold_cannot_move(tmp_path: Path, scorecard: dict) -> None:
    scorecard["acceptance_contract"]["cross_engine_fsc_auc_min"] = 0.99
    with pytest.raises(ValueError, match="acceptance contract changed"):
        scorecard_mod.load_and_validate(_write(tmp_path, scorecard))


def test_not_run_case_cannot_claim_evidence(tmp_path: Path, scorecard: dict) -> None:
    scorecard["cases"][0]["evidence"] = {"report_sha256": "2" * 64}
    with pytest.raises(ValueError, match="not-run case has evidence"):
        scorecard_mod.load_and_validate(_write(tmp_path, scorecard))


def test_passing_evaluated_case_replays_fixed_gates(tmp_path: Path, scorecard: dict) -> None:
    _evaluate_first_case(scorecard)
    loaded = scorecard_mod.load_and_validate(_write(tmp_path, scorecard))
    assert loaded["cases"][0]["result"] == "pass"


@pytest.mark.parametrize(
    ("cross", "gt_delta"),
    [(0.998999, -0.001), (0.9995, -0.002001)],
)
def test_evaluated_case_fails_below_either_map_gate(
    tmp_path: Path, scorecard: dict, cross: float, gt_delta: float
) -> None:
    _evaluate_first_case(scorecard, cross=cross, gt_delta=gt_delta)
    loaded = scorecard_mod.load_and_validate(_write(tmp_path, scorecard))
    assert loaded["cases"][0]["result"] == "fail"


def test_checkpoint_failure_forces_case_failure(tmp_path: Path, scorecard: dict) -> None:
    _evaluate_first_case(scorecard)
    scorecard["cases"][0]["checkpoint_results"]["2"] = "fail"
    scorecard["cases"][0]["result"] = "fail"
    _set_counts(scorecard)
    loaded = scorecard_mod.load_and_validate(_write(tmp_path, scorecard))
    assert loaded["cases"][0]["result"] == "fail"


def test_false_pass_is_rejected(tmp_path: Path, scorecard: dict) -> None:
    _evaluate_first_case(scorecard, cross=0.998, gt_delta=0.0)
    scorecard["cases"][0]["result"] = "pass"
    _set_counts(scorecard)
    with pytest.raises(ValueError, match="does not replay its fixed gates"):
        scorecard_mod.load_and_validate(_write(tmp_path, scorecard))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("same_physical_gpu", False, "not on one physical GPU"),
        ("correlation_used", True, "correlation cannot gate"),
        ("exact_schedule", False, "schedule mismatch"),
        ("exact_artifact_topology", False, "artifact topology mismatch"),
    ],
)
def test_evidence_must_preserve_parity_topology(
    tmp_path: Path, scorecard: dict, field: str, replacement: object, message: str
) -> None:
    _evaluate_first_case(scorecard)
    scorecard["cases"][0]["evidence"][field] = replacement
    with pytest.raises(ValueError, match=message):
        scorecard_mod.load_and_validate(_write(tmp_path, scorecard))


def test_current_counts_must_replay_cases(tmp_path: Path, scorecard: dict) -> None:
    scorecard["current_snapshot"]["counts"] = {"pass": 1, "fail": 0, "not_run": 11}
    with pytest.raises(ValueError, match="current counts do not replay cases"):
        scorecard_mod.load_and_validate(_write(tmp_path, scorecard))


def test_future_history_rows_need_immutable_evidence(tmp_path: Path, scorecard: dict) -> None:
    row = copy.deepcopy(scorecard["history"][0])
    row["id"] = "vdam-k1-fixed12-next"
    scorecard["history"].append(row)
    scorecard["current_snapshot"]["id"] = row["id"]
    with pytest.raises(ValueError, match="missing immutable evidence"):
        scorecard_mod.load_and_validate(_write(tmp_path, scorecard))


def test_renderer_exposes_fixed_denominator_and_metric_policy(scorecard: dict) -> None:
    rendered = scorecard_mod.render_markdown(scorecard_mod.load_and_validate())
    assert "Legacy, non-authoritative short-prefix suite" in rendered
    assert "vdam_relion_parity_dashboard.md" in rendered
    assert "12 / 12 passing" in rendered
    assert "iterations `0`, `1`, `2`, `4`, and `8`" in rendered
    assert "Map correlation is not computed or gated" in rendered
    assert rendered.count("| `vdam-") == 12


def test_cli_check_accepts_checked_markdown() -> None:
    result = subprocess.run(
        [sys.executable, str(scorecard_mod.__file__), "--check"],
        cwd=scorecard_mod.REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
