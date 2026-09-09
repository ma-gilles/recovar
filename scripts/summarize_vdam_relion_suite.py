#!/usr/bin/env python3
"""Summarize a complete VDAM parity suite root without mutating its definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA = "recovar.vdam_relion_suite_summary.v1"
AUDIT_SCHEMAS = {
    "recovar.vdam_relion_fsc_trajectory_audit.v1",
    "recovar.vdam_relion_real_data_trajectory_audit.v2",
}
REPEATABILITY_ENVELOPE_SCHEMA = "recovar.vdam_real_repeatability_envelope.v1"


class SummaryError(RuntimeError):
    """Raised when suite evidence is missing, mixed, or malformed."""


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SummaryError(f"{label} must contain a JSON object: {path}")
    return value


def _runtime(case_root: Path, provenance: dict[str, Any]) -> tuple[float, float]:
    recovar_path = case_root / "recovar" / "recovar.timing.json"
    relion_path = case_root / "relion" / "relion.timing.json"
    if recovar_path.is_file() and relion_path.is_file():
        recovar_wall_s = float(_load(recovar_path, "RECOVAR timing")["external_wall_s"])
        relion_wall_s = float(_load(relion_path, "RELION timing")["external_wall_s"])
    else:
        try:
            recovar_wall_s = float(provenance["recovar_wall_s"])
            relion_wall_s = float(provenance["relion_wall_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise SummaryError("missing paired runtime evidence") from error
    if recovar_wall_s <= 0 or relion_wall_s <= 0:
        raise SummaryError("paired runtimes must be positive")
    return recovar_wall_s, relion_wall_s


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
        if audit.get("schema") not in AUDIT_SCHEMAS or audit.get("suite_id") != suite.get("suite_id"):
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
        recovar_wall_s, relion_wall_s = _runtime(case_root, provenance)
        row = {
            "case_id": case_id,
            "name": str(case["name"]),
            "dataset": case.get("dataset"),
            "strict_point_reference_result": str(audit["result"]),
            "minimum_cross_engine_fsc_auc": float(audit["minimum_cross_engine_fsc_auc"]),
            "recovar_wall_s": recovar_wall_s,
            "relion_wall_s": relion_wall_s,
            "runtime_ratio_recovar_over_relion": recovar_wall_s / relion_wall_s,
            "slurm_job_id": str(provenance.get("slurm_job_id", "")),
            "source_head": source_head,
        }
        gt_delta = audit.get("minimum_recovar_minus_relion_gt_fsc_auc")
        if gt_delta is not None:
            row["minimum_recovar_minus_relion_gt_fsc_auc"] = float(gt_delta)
        particle_states = [
            checkpoint.get("particle_state")
            for checkpoint in audit.get("checkpoints", ())
            if checkpoint.get("particle_state") is not None
        ]
        if particle_states:
            row.update(
                maximum_divergent_particle_count=max(
                    int(state["divergent_particle_count"]) for state in particle_states
                ),
                maximum_pmax_absolute_error_p95=max(
                    float(state["pmax_absolute_error"]["p95"]) for state in particle_states
                ),
                maximum_pmax_absolute_error=max(
                    float(state["pmax_absolute_error"]["max"]) for state in particle_states
                ),
            )
        envelope_path = case_root / "repeatability_envelope.json"
        calibrated_result = str(audit["result"])
        if envelope_path.is_file():
            envelope = _load(envelope_path, "repeatability envelope")
            if envelope.get("schema") != REPEATABILITY_ENVELOPE_SCHEMA:
                raise SummaryError(f"{case_id}: repeatability envelope schema differs")
            if envelope.get("strict_point_reference_is_preserved") is not True:
                raise SummaryError(f"{case_id}: repeatability envelope rewrites strict status")
            if envelope.get("strict_point_reference_result") != audit.get("result"):
                raise SummaryError(f"{case_id}: strict audit and repeatability envelope disagree")
            calibrated_result = str(envelope["repeatability_calibrated_result"])
            row["repeatability_envelope"] = str(envelope_path)
        row["repeatability_calibrated_result"] = calibrated_result
        row["result"] = calibrated_result
        rows.append(row)
    if len(source_heads) != 1:
        raise SummaryError(f"suite reports contain mixed source heads: {sorted(source_heads)}")
    strict_failures = [
        row["case_id"] for row in rows if row["strict_point_reference_result"] != "pass"
    ]
    failures = [row["case_id"] for row in rows if row["repeatability_calibrated_result"] != "pass"]
    runtime_ratios = [row["runtime_ratio_recovar_over_relion"] for row in rows]
    return {
        "schema": SCHEMA,
        "suite_id": suite["suite_id"],
        "source_head": next(iter(source_heads)),
        "result": "pass" if not failures else "fail",
        "counts": {"pass": len(rows) - len(failures), "fail": len(failures), "total": len(rows)},
        "failure_case_ids": failures,
        "strict_point_reference_result": "pass" if not strict_failures else "fail",
        "strict_point_reference_counts": {
            "pass": len(rows) - len(strict_failures),
            "fail": len(strict_failures),
            "total": len(rows),
        },
        "strict_point_reference_failure_case_ids": strict_failures,
        "repeatability_calibrated_result": "pass" if not failures else "fail",
        "minimum_cross_engine_fsc_auc": min(row["minimum_cross_engine_fsc_auc"] for row in rows),
        "runtime_ratio_recovar_over_relion": {
            "min": min(runtime_ratios),
            "median": median(runtime_ratios),
            "max": max(runtime_ratios),
        },
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
