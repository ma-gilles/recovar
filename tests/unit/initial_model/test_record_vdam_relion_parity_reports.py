from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.record_vdam_relion_parity_reports import discover_reports, record_reports
from scripts.summarize_vdam_relion_parity_scorecard import DEFAULT_SCORECARD, load_and_validate


def _write_report(root: Path, *, case_id: str = "vdam-01", source_id: str = "k1-11") -> Path:
    case_root = root / case_id
    case_root.mkdir(parents=True)
    checkpoints = []
    for iteration in (0, 1, 2, 4, 8):
        checkpoints.append(
            {
                "iteration": iteration,
                "pass": True,
                "cross_engine": {"fsc_auc": 0.9995},
                "recovar_minus_relion_gt_fsc_auc": -0.001,
            }
        )
    report = {
        "schema": "recovar.vdam_relion_fsc_trajectory_audit.v1",
        "case_id": case_id,
        "source_em_case_id": source_id,
        "result": "pass",
        "checkpoints": checkpoints,
        "same_physical_gpu": True,
        "correlation_used": False,
        "artifact_topology_exact": True,
    }
    report_path = case_root / "trajectory_audit.json"
    report_path.write_text(json.dumps(report))
    (case_root / "run_provenance.json").write_text(
        json.dumps({"git_head": "1" * 40, "slurm_job_id": "123_1"})
    )
    return report_path


def test_record_reports_updates_case_counts_and_writes_hash_bound_ledger(tmp_path):
    report_root = tmp_path / "reports"
    report = _write_report(report_root)
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text(DEFAULT_SCORECARD.read_text())
    ledger = tmp_path / "ledger.json"

    updated = record_reports(
        scorecard_path,
        {"vdam-01": report},
        snapshot_id="test-snapshot",
        ledger_path=ledger,
        source_head="2" * 40,
    )
    scorecard_path.write_text(json.dumps(updated))
    validated = load_and_validate(scorecard_path)

    assert validated["current_snapshot"]["counts"] == {"pass": 1, "fail": 0, "not_run": 11}
    assert validated["cases"][0]["evidence"]["audit_job"] == "123_1"
    assert ledger.is_file()
    assert validated["history"][-1]["evidence"]["ledger_sha256"]


def test_discover_reports_rejects_duplicate_case_ids(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_report(first)
    _write_report(second)

    with pytest.raises(ValueError, match="duplicate trajectory audit"):
        discover_reports([first, second])
