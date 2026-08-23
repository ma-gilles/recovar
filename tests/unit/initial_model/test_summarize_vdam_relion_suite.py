from __future__ import annotations

import json

import pytest

from scripts.summarize_vdam_relion_suite import SummaryError, summarize


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _suite(tmp_path):
    path = tmp_path / "suite.json"
    _write_json(
        path,
        {
            "suite_id": "parameter-suite",
            "acceptance_contract": {"required_checkpoints": [0, 1, 3]},
            "cases": [
                {"id": "case-1", "name": "one"},
                {"id": "case-2", "name": "two"},
            ],
        },
    )
    return path


def _case(root, case_id, *, result="pass", source_head="1" * 40):
    case_root = root / case_id
    _write_json(
        case_root / "trajectory_audit.json",
        {
            "schema": "recovar.vdam_relion_fsc_trajectory_audit.v1",
            "suite_id": "parameter-suite",
            "case_id": case_id,
            "result": result,
            "checkpoints": [{"iteration": value} for value in (0, 1, 3)],
            "minimum_cross_engine_fsc_auc": 0.9995,
            "minimum_recovar_minus_relion_gt_fsc_auc": -0.0001,
        },
    )
    _write_json(case_root / "run_provenance.json", {"git_head": source_head, "slurm_job_id": "123"})
    _write_json(case_root / "recovar" / "recovar.timing.json", {"external_wall_s": 20.0})
    _write_json(case_root / "relion" / "relion.timing.json", {"external_wall_s": 10.0})


def test_summarize_complete_suite_tracks_quality_runtime_and_failures(tmp_path):
    suite = _suite(tmp_path)
    root = tmp_path / "reports"
    _case(root, "case-1")
    _case(root, "case-2", result="fail")

    result = summarize(suite, root)

    assert result["result"] == "fail"
    assert result["counts"] == {"pass": 1, "fail": 1, "total": 2}
    assert result["failure_case_ids"] == ["case-2"]
    assert result["cases"][0]["runtime_ratio_recovar_over_relion"] == pytest.approx(2.0)


def test_summarize_rejects_mixed_source_heads(tmp_path):
    suite = _suite(tmp_path)
    root = tmp_path / "reports"
    _case(root, "case-1")
    _case(root, "case-2", source_head="2" * 40)

    with pytest.raises(SummaryError, match="mixed source heads"):
        summarize(suite, root)
