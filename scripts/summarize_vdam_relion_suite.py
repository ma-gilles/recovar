#!/usr/bin/env python3
"""Summarize a complete VDAM parity suite root without mutating its definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "recovar.vdam_relion_suite_summary.v1"
AUDIT_SCHEMA = "recovar.vdam_relion_fsc_trajectory_audit.v1"


class SummaryError(RuntimeError):
    """Raised when suite evidence is missing, mixed, or malformed."""


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SummaryError(f"{label} must contain a JSON object: {path}")
    return value


def summarize(suite_path: Path, report_root: Path) -> dict[str, Any]:
    suite = _load(suite_path, "suite")
    case_definitions = suite.get("cases")
    if not isinstance(case_definitions, list) or not case_definitions:
        raise SummaryError("suite contains no cases")
    required_checkpoints = tuple(int(value) for value in suite["acceptance_contract"]["required_checkpoints"])
    rows = []
    source_heads = set()
    for case in case_definitions:
        case_id = str(case["id"])
        case_root = report_root / case_id
        audit_path = case_root / "trajectory_audit.json"
        provenance_path = case_root / "run_provenance.json"
        if not audit_path.is_file() or not provenance_path.is_file():
            raise SummaryError(f"{case_id}: missing audit or provenance evidence")
        audit = _load(audit_path, "trajectory audit")
        provenance = _load(provenance_path, "run provenance")
        if audit.get("schema") != AUDIT_SCHEMA or audit.get("suite_id") != suite.get("suite_id"):
            raise SummaryError(f"{case_id}: audit schema or suite identity differs")
        if audit.get("case_id") != case_id:
            raise SummaryError(f"{case_id}: audit case identity differs")
        checkpoints = tuple(int(row["iteration"]) for row in audit.get("checkpoints", ()))
        if checkpoints != required_checkpoints:
            raise SummaryError(f"{case_id}: checkpoint topology differs from the frozen suite")
        source_head = str(provenance.get("git_head", ""))
        if len(source_head) != 40:
            raise SummaryError(f"{case_id}: invalid source head")
        source_heads.add(source_head)
        recovar_timing = _load(case_root / "recovar" / "recovar.timing.json", "RECOVAR timing")
        relion_timing = _load(case_root / "relion" / "relion.timing.json", "RELION timing")
        recovar_wall_s = float(recovar_timing["external_wall_s"])
        relion_wall_s = float(relion_timing["external_wall_s"])
        rows.append(
            {
                "case_id": case_id,
                "name": str(case["name"]),
                "result": str(audit["result"]),
                "minimum_cross_engine_fsc_auc": float(audit["minimum_cross_engine_fsc_auc"]),
                "minimum_recovar_minus_relion_gt_fsc_auc": float(
                    audit["minimum_recovar_minus_relion_gt_fsc_auc"]
                ),
                "recovar_wall_s": recovar_wall_s,
                "relion_wall_s": relion_wall_s,
                "runtime_ratio_recovar_over_relion": recovar_wall_s / relion_wall_s,
                "slurm_job_id": str(provenance.get("slurm_job_id", "")),
                "source_head": source_head,
            }
        )
    if len(source_heads) != 1:
        raise SummaryError(f"suite reports contain mixed source heads: {sorted(source_heads)}")
    failures = [row["case_id"] for row in rows if row["result"] != "pass"]
    return {
        "schema": SCHEMA,
        "suite_id": suite["suite_id"],
        "source_head": next(iter(source_heads)),
        "result": "pass" if not failures else "fail",
        "counts": {"pass": len(rows) - len(failures), "fail": len(failures), "total": len(rows)},
        "failure_case_ids": failures,
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.suite.resolve(), args.report_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
