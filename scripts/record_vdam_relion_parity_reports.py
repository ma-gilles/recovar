#!/usr/bin/env python3
"""Record completed VDAM trajectory audits in the frozen fixed-12 scorecard."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.summarize_vdam_relion_parity_scorecard import (
    DEFAULT_OUTPUT,
    DEFAULT_SCORECARD,
    REQUIRED_CHECKPOINTS,
    load_and_validate,
    render_markdown,
)

AUDIT_SCHEMA = "recovar.vdam_relion_fsc_trajectory_audit.v1"
LEDGER_SCHEMA = "recovar.vdam_relion_parity_evidence_ledger.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def discover_reports(roots: list[Path]) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for root in roots:
        for path in sorted(root.glob("vdam-*/trajectory_audit.json")):
            report = _load_object(path, "trajectory audit")
            case_id = str(report.get("case_id", ""))
            if case_id in reports:
                raise ValueError(f"duplicate trajectory audit for {case_id}: {reports[case_id]} and {path}")
            reports[case_id] = path.resolve()
    if not reports:
        raise ValueError("no vdam-*/trajectory_audit.json reports found")
    return reports


def _case_update(report_path: Path, expected_source_case_id: str) -> tuple[dict, dict]:
    report = _load_object(report_path, "trajectory audit")
    if report.get("schema") != AUDIT_SCHEMA:
        raise ValueError(f"unsupported trajectory audit schema: {report_path}")
    case_id = str(report.get("case_id", ""))
    if report.get("source_em_case_id") != expected_source_case_id:
        raise ValueError(f"{case_id}: source fixture differs from the frozen scorecard")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError(f"{case_id}: trajectory audit has no checkpoints")
    by_iteration = {int(row["iteration"]): row for row in checkpoints}
    if set(by_iteration) != set(REQUIRED_CHECKPOINTS):
        raise ValueError(f"{case_id}: trajectory checkpoint set changed")
    final = by_iteration[8]
    provenance_path = report_path.parent / "run_provenance.json"
    provenance = _load_object(provenance_path, "run provenance")
    source_head = str(provenance.get("git_head", ""))
    if len(source_head) != 40:
        raise ValueError(f"{case_id}: invalid source head in {provenance_path}")
    job_id = str(provenance.get("slurm_job_id", ""))
    if not job_id:
        raise ValueError(f"{case_id}: missing Slurm job ID in {provenance_path}")
    report_sha256 = _sha256(report_path)
    checkpoint_results = {
        str(iteration): "pass" if bool(by_iteration[iteration].get("pass")) else "fail"
        for iteration in REQUIRED_CHECKPOINTS
    }
    evidence = {
        "source_head": source_head,
        "report_sha256": report_sha256,
        "recovar_job": job_id,
        "relion_job": job_id,
        "audit_job": job_id,
        "same_physical_gpu": bool(report.get("same_physical_gpu")),
        "correlation_used": bool(report.get("correlation_used")),
        "exact_schedule": True,
        "exact_artifact_topology": bool(report.get("artifact_topology_exact")),
        "final_cross_engine_fsc_auc": float(final["cross_engine"]["fsc_auc"]),
        "final_gt_fsc_auc_delta": float(final["recovar_minus_relion_gt_fsc_auc"]),
    }
    case_value = {
        "result": str(report.get("result")),
        "checkpoint_results": checkpoint_results,
        "evidence": evidence,
    }
    ledger_value = {
        "case_id": case_id,
        "report": str(report_path),
        "report_sha256": report_sha256,
        "run_provenance": str(provenance_path.resolve()),
        "source_head": source_head,
        "slurm_job_id": job_id,
        "result": case_value["result"],
    }
    return case_value, ledger_value


def record_reports(
    scorecard_path: Path,
    report_paths: dict[str, Path],
    *,
    snapshot_id: str,
    ledger_path: Path,
    source_head: str,
) -> dict:
    scorecard = load_and_validate(scorecard_path)
    known = {case["id"]: case for case in scorecard["cases"]}
    unknown = sorted(set(report_paths).difference(known))
    if unknown:
        raise ValueError(f"unknown VDAM case reports: {unknown}")
    ledger_cases = []
    for case_id, report_path in sorted(report_paths.items()):
        case = known[case_id]
        if case["result"] != "not_run":
            raise ValueError(f"refusing to replace evaluated scorecard case {case_id}")
        update, ledger_entry = _case_update(
            report_path,
            str(case["definition"]["source_em_case_id"]),
        )
        case.update(update)
        ledger_cases.append(ledger_entry)

    counts_raw = Counter(case["result"] for case in scorecard["cases"])
    counts = {name: int(counts_raw.get(name, 0)) for name in ("pass", "fail", "not_run")}
    recorded_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    ledger = {
        "schema": LEDGER_SCHEMA,
        "snapshot_id": snapshot_id,
        "recorded_utc": recorded_utc,
        "source_head": source_head,
        "counts": counts,
        "cases": ledger_cases,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    snapshot_evidence = {
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": _sha256(ledger_path),
        "source_head": source_head,
    }
    snapshot = {
        "id": snapshot_id,
        "recorded_utc": recorded_utc,
        "counts": counts,
        "evidence": snapshot_evidence,
        "status_note": "Recorded frozen same-GPU VDAM trajectory audits without changing suite gates.",
    }
    scorecard["current_snapshot"] = {
        "id": snapshot_id,
        "counts": counts,
        "evidence": snapshot_evidence,
    }
    scorecard["history"].append(snapshot)
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", action="append", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    reports = discover_reports([path.resolve() for path in args.report_root])
    scorecard = record_reports(
        args.scorecard.resolve(),
        reports,
        snapshot_id=args.snapshot_id,
        ledger_path=args.ledger.resolve(),
        source_head=source_head,
    )
    args.scorecard.write_text(json.dumps(scorecard, indent=2) + "\n")
    validated = load_and_validate(args.scorecard)
    args.output.write_text(render_markdown(validated))
    print(json.dumps(validated["current_snapshot"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
