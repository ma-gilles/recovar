#!/usr/bin/env python3
"""Combine direct candidate/native VDAM map audits into a mode envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PAIR_SCHEMA = "recovar.vdam_kclass_trajectory_audit.v1"
SCHEMA = "recovar.vdam_direct_map_envelope.v1"


class DirectMapEnvelopeError(RuntimeError):
    """Raised when pair reports cannot define one native map envelope."""


def summarize_direct_map_envelope(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise DirectMapEnvelopeError("map envelope requires at least two native pair reports")
    reference = reports[0]
    if reference.get("schema") != PAIR_SCHEMA:
        raise DirectMapEnvelopeError(f"unsupported pair schema: {reference.get('schema')!r}")
    checkpoints = tuple(int(value) for value in reference.get("checkpoints", ()))
    if not checkpoints or checkpoints != tuple(sorted(set(checkpoints))):
        raise DirectMapEnvelopeError("pair checkpoints must be sorted, unique, and nonempty")
    k_classes = int(reference.get("K", 0))
    thresholds = reference.get("thresholds")
    rows_by_report = []
    for index, report in enumerate(reports, start=1):
        if report.get("schema") != PAIR_SCHEMA:
            raise DirectMapEnvelopeError(f"native pair {index} schema differs")
        if int(report.get("K", 0)) != k_classes or report.get("thresholds") != thresholds:
            raise DirectMapEnvelopeError(f"native pair {index} K or thresholds differ")
        if tuple(int(value) for value in report.get("checkpoints", ())) != checkpoints:
            raise DirectMapEnvelopeError(f"native pair {index} checkpoints differ")
        rows = report.get("iterations", ())
        if tuple(int(row["iteration"]) for row in rows) != checkpoints:
            raise DirectMapEnvelopeError(f"native pair {index} iteration rows differ")
        rows_by_report.append(rows)

    envelope_rows = []
    for offset, iteration in enumerate(checkpoints):
        rows = [report_rows[offset] for report_rows in rows_by_report]
        values = np.asarray([row["minimum_matched_fsc_auc"] for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise DirectMapEnvelopeError(f"iteration {iteration} contains non-finite FSC-AUC")
        matching = [index for index, row in enumerate(rows, start=1) if bool(row["pass"])]
        envelope_rows.append(
            {
                "iteration": iteration,
                "pass": bool(matching),
                "matching_native_repeat_indices": matching,
                "candidate_native_minimum_fsc_auc": values.tolist(),
                "candidate_best_native_minimum_fsc_auc": float(np.max(values)),
            }
        )

    failures = [row for row in envelope_rows if not row["pass"]]
    return {
        "schema": SCHEMA,
        "result": "fail" if failures else "pass",
        "scope": "diagnostic candidate map coverage; not a frozen-suite promotion",
        "K": k_classes,
        "thresholds": thresholds,
        "native_repeat_count": len(reports),
        "checkpoints": list(checkpoints),
        "first_failure_iteration": None if not failures else failures[0]["iteration"],
        "failure_count": len(failures),
        "minimum_best_native_fsc_auc": float(
            min(row["candidate_best_native_minimum_fsc_auc"] for row in envelope_rows)
        ),
        "iterations": envelope_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-report", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text()) for path in args.pair_report]
    summary = summarize_direct_map_envelope(reports)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
