#!/usr/bin/env python3
"""Summarize the fixed case-4/5/10 K=1 treatment prefix gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "recovar.em.k1_selected_treatment_prefix.v1"
MOVEMENT_SCHEMA = "recovar.em.k1_autonomous_boundary_movement.v1"
FIXED_CASES = (4, 5, 10)


def _load_case(case_id: int, path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema") != MOVEMENT_SCHEMA:
        raise ValueError(f"case {case_id}: unexpected movement schema in {path}")
    arms = report.get("arms", {})
    if set(arms) != {"baseline", "candidate"}:
        raise ValueError(f"case {case_id}: movement report must contain baseline and candidate")
    gates = report.get("gates", {})
    if not gates or not all(isinstance(value, bool) for value in gates.values()):
        raise ValueError(f"case {case_id}: movement gates are missing or non-boolean")

    baseline = arms["baseline"]
    candidate = arms["candidate"]
    return {
        "case": case_id,
        "source_report": str(path.resolve()),
        "source_classification": report.get("classification"),
        "all_boundary_gates_pass": all(gates.values()),
        "gates": gates,
        "baseline": {
            "pmax_relative_l2": baseline["pmax"]["relative_l2"],
            "support_mismatches": baseline["support"]["mismatch_count"],
            "pose_outliers": baseline["pose_error_gt_0p01_deg_count"],
            "translation_outliers": baseline[
                "translation_error_gt_0p01_angstrom_count"
            ],
            "merged_signed_fsc_auc": baseline["merged_signed_fsc_auc"],
        },
        "candidate": {
            "pmax_relative_l2": candidate["pmax"]["relative_l2"],
            "support_mismatches": candidate["support"]["mismatch_count"],
            "pose_outliers": candidate["pose_error_gt_0p01_deg_count"],
            "translation_outliers": candidate[
                "translation_error_gt_0p01_angstrom_count"
            ],
            "merged_signed_fsc_auc": candidate["merged_signed_fsc_auc"],
        },
        "movement": report.get("movement", {}),
    }


def summarize(case_paths: dict[int, Path]) -> dict[str, Any]:
    if tuple(sorted(case_paths)) != FIXED_CASES:
        raise ValueError(
            f"fixed prefix gate requires cases {FIXED_CASES}, got {tuple(sorted(case_paths))}"
        )
    cases = [_load_case(case_id, case_paths[case_id]) for case_id in FIXED_CASES]
    passed = sum(item["all_boundary_gates_pass"] for item in cases)
    return {
        "schema": SCHEMA,
        "metric_policy": (
            "Fixed case-4/5/10 source-row particle state and signed non-DC "
            "FSC-AUC; correlation is not computed"
        ),
        "fixed_cases": list(FIXED_CASES),
        "counts": {"passed": passed, "total": len(cases)},
        "terminal_run_eligible": passed == len(cases),
        "cases": cases,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# K=1 selected-treatment iteration-2 prefix gate",
        "",
        "The denominator is fixed at cases 4, 5, and 10. Correlation is not computed.",
        "",
        "| Case | Gates | Pmax rel-L2 base → candidate | Support base → candidate | Pose base → candidate | Translation base → candidate | Merged signed FSC-AUC base → candidate |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["cases"]:
        baseline = item["baseline"]
        candidate = item["candidate"]
        lines.append(
            f"| {item['case']} | {'✓' if item['all_boundary_gates_pass'] else '✗'} | "
            f"{baseline['pmax_relative_l2']:.9g} → {candidate['pmax_relative_l2']:.9g} | "
            f"{baseline['support_mismatches']} → {candidate['support_mismatches']} | "
            f"{baseline['pose_outliers']} → {candidate['pose_outliers']} | "
            f"{baseline['translation_outliers']} → {candidate['translation_outliers']} | "
            f"{baseline['merged_signed_fsc_auc']:.13f} → "
            f"{candidate['merged_signed_fsc_auc']:.13f} |"
        )
    counts = report["counts"]
    lines.extend(
        [
            "",
            f"Prefix boundary gates: **{counts['passed']}/{counts['total']}**.",
            "",
            "This prefix report selects whether to spend on terminal runs; only the fixed terminal FSC/FSC-AUC scorecard can promote the treatment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="ID=REPORT.json",
        help="Fixed case movement report; repeat for cases 4, 5, and 10",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--require-all-pass", action="store_true")
    args = parser.parse_args()

    case_paths: dict[int, Path] = {}
    for value in args.case:
        if "=" not in value:
            parser.error(f"invalid --case {value!r}; expected ID=REPORT.json")
        raw_case, raw_path = value.split("=", 1)
        case_id = int(raw_case)
        if case_id in case_paths:
            parser.error(f"duplicate case {case_id}")
        case_paths[case_id] = Path(raw_path)

    report = summarize(case_paths)
    for path in (args.output_json, args.output_markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if args.require_all_pass and not report["terminal_run_eligible"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
