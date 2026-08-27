#!/usr/bin/env python3
"""Build the fixed K=1 score from replacement terminal FSC reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ScorecardError(RuntimeError):
    """Raised when a replacement report cannot support a fixed-score claim."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScorecardError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ScorecardError(f"{label} must be finite, got {result}")
    return result


def _replacement_spec(value: str) -> tuple[int, Path]:
    case_text, separator, path_text = value.partition("=")
    if not separator or not case_text.isdigit() or not path_text:
        raise argparse.ArgumentTypeError("replacement must have CASE_ID=/absolute/report.json form")
    path = Path(path_text)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("replacement report path must be absolute")
    return int(case_text), path


def _replacement_entry(
    baseline: dict[str, Any],
    report_path: Path,
    *,
    merged_threshold: float,
    gt_delta_threshold: float,
) -> dict[str, Any]:
    if not report_path.is_file():
        raise ScorecardError(f"missing replacement FSC report: {report_path}")
    report = json.loads(report_path.read_text())
    numbered = report.get("numbered_iterations")
    if not isinstance(numbered, list) or not numbered:
        raise ScorecardError(f"{report_path} has no numbered trajectory")
    final = report.get("final")
    if not isinstance(final, dict):
        raise ScorecardError(f"{report_path} has no complete final product")
    topology_failures = report.get("topology_failures")
    if not isinstance(topology_failures, list):
        raise ScorecardError(f"{report_path}:topology_failures must be a list")

    numbered_merged = [
        _finite_number(
            row["cross_engine"]["merged"]["signed_fsc_auc"],
            label=f"{report_path}:numbered[{index}].merged",
        )
        for index, row in enumerate(numbered)
    ]
    final_merged = _finite_number(
        final["cross_engine"]["merged"]["signed_fsc_auc"],
        label=f"{report_path}:final.merged",
    )
    gt_delta = _finite_number(
        final["merged_gt_fsc_auc_delta"],
        label=f"{report_path}:final.merged_gt_fsc_auc_delta",
    )
    numbered_pass = min(numbered_merged) >= merged_threshold
    topology_pass = not topology_failures
    final_cross_pass = final_merged >= merged_threshold
    final_gt_pass = gt_delta >= gt_delta_threshold
    passed = numbered_pass and topology_pass and final_cross_pass and final_gt_pass
    return {
        **baseline,
        "source": "replacement",
        "status": "pass" if passed else "fail",
        "numbered_status": "pass" if numbered_pass else "fail",
        "topology_status": "pass" if topology_pass else "fail",
        "final_cross_engine_status": "pass" if final_cross_pass else "fail",
        "final_gt_status": "pass" if final_gt_pass else "fail",
        "numbered_iteration_count": len(numbered),
        "minimum_numbered_merged_cross_engine_fsc_auc": min(numbered_merged),
        "last_numbered_merged_cross_engine_fsc_auc": numbered_merged[-1],
        "final_merged_cross_engine_fsc_auc": final_merged,
        "final_recovar_minus_relion_merged_gt_fsc_auc": gt_delta,
        "baseline_final_merged_cross_engine_fsc_auc": baseline[
            "final_merged_cross_engine_fsc_auc"
        ],
        "final_merged_cross_engine_fsc_auc_change": (
            final_merged - float(baseline["final_merged_cross_engine_fsc_auc"])
        ),
        "audit_json": str(report_path.resolve()),
        "audit_json_sha256": _sha256(report_path),
        "topology_failures": topology_failures,
    }


def build_scorecard(
    *,
    baseline_json: Path,
    replacements: dict[int, Path],
    fixed_total: int = 34,
    baseline_passing: int = 31,
) -> dict[str, Any]:
    baseline_json = baseline_json.resolve()
    baseline_payload = json.loads(baseline_json.read_text())
    baseline_rows = (
        baseline_payload.get("cases")
        if isinstance(baseline_payload, dict)
        else baseline_payload
    )
    if not isinstance(baseline_rows, list) or not baseline_rows:
        raise ScorecardError("baseline JSON must contain a non-empty case list")
    baseline_by_case: dict[int, dict[str, Any]] = {}
    for row in baseline_rows:
        case_id = int(row["case_id"])
        if case_id in baseline_by_case:
            raise ScorecardError(f"duplicate baseline case {case_id}")
        baseline_by_case[case_id] = dict(row)
    unknown = sorted(set(replacements) - set(baseline_by_case))
    if unknown:
        raise ScorecardError(f"replacement cases are absent from baseline: {unknown}")
    if baseline_passing < 0 or fixed_total < baseline_passing + len(baseline_rows):
        raise ScorecardError("fixed total and baseline passing count are inconsistent")

    merged_threshold = 0.995
    gt_delta_threshold = -0.002
    if isinstance(baseline_payload, dict):
        frozen_score = baseline_payload.get("fixed_suite_score")
        if frozen_score is not None:
            observed_score = {
                "pass": baseline_passing,
                "fail": len(baseline_rows),
                "fixed_denominator": fixed_total,
            }
            if frozen_score != observed_score:
                raise ScorecardError(
                    f"frozen score {frozen_score} disagrees with requested {observed_score}"
                )
        frozen_thresholds = baseline_payload.get("thresholds")
        expected_thresholds = {
            "merged_cross_engine_fsc_auc_min": merged_threshold,
            "recovar_minus_relion_merged_gt_fsc_auc_min": gt_delta_threshold,
        }
        if frozen_thresholds is not None and frozen_thresholds != expected_thresholds:
            raise ScorecardError(
                f"frozen thresholds {frozen_thresholds} disagree with {expected_thresholds}"
            )
    cases = []
    for case_id, baseline in sorted(baseline_by_case.items()):
        replacement = replacements.get(case_id)
        if replacement is None:
            cases.append(
                {
                    **baseline,
                    "source": "baseline",
                    "status": "fail",
                    "numbered_status": "pass",
                    "topology_status": baseline.get("topology_status", "unknown"),
                    "final_cross_engine_status": "fail",
                    "final_gt_status": (
                        "pass"
                        if float(baseline["final_recovar_minus_relion_merged_gt_fsc_auc"])
                        >= gt_delta_threshold
                        else "fail"
                    ),
                }
            )
        else:
            cases.append(
                _replacement_entry(
                    baseline,
                    replacement.resolve(),
                    merged_threshold=merged_threshold,
                    gt_delta_threshold=gt_delta_threshold,
                )
            )

    newly_passing = sum(row["status"] == "pass" for row in cases)
    passing = baseline_passing + newly_passing
    return {
        "schema": "recovar.em.k1_remaining_terminal_scorecard.v2",
        "metric_policy": (
            "Signed shellwise FSC and normalized non-DC FSC-AUC; correlation is not used."
        ),
        "thresholds": {
            "merged_cross_engine_fsc_auc_min": merged_threshold,
            "recovar_minus_relion_merged_gt_fsc_auc_min": gt_delta_threshold,
            "exact_numbered_topology_required": True,
        },
        "fixed_total": fixed_total,
        "baseline_passing": baseline_passing,
        "newly_passing_remaining_cases": newly_passing,
        "passing": passing,
        "failing": fixed_total - passing,
        "score": f"{passing}/{fixed_total}",
        "status": "pass" if passing == fixed_total else "incomplete",
        "baseline_json": str(baseline_json),
        "baseline_json_sha256": _sha256(baseline_json),
        "baseline_schema": (
            baseline_payload.get("schema")
            if isinstance(baseline_payload, dict)
            else "bare_case_list"
        ),
        "cases": cases,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Fixed K=1 score: {report['score']}",
        "",
        "| Done | Case | Fixture | Numbered min | Final merged | GT delta | Status |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["cases"]:
        checked = "x" if row["status"] == "pass" else " "
        lines.append(
            f"| [{checked}] | {row['case_id']} | {row['fixture']} | "
            f"{row['minimum_numbered_merged_cross_engine_fsc_auc']:.12f} | "
            f"{row['final_merged_cross_engine_fsc_auc']:.12f} | "
            f"{row['final_recovar_minus_relion_merged_gt_fsc_auc']:+.12f} | "
            f"{row['status']} |"
        )
    lines.extend(
        [
            "",
            "Thresholds are frozen at merged FSC-AUC >= 0.995, merged GT delta >= -0.002, and exact numbered topology.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--replacement", action="append", default=[], type=_replacement_spec)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    replacements: dict[int, Path] = {}
    for case_id, path in args.replacement:
        if case_id in replacements:
            raise ScorecardError(f"duplicate replacement case {case_id}")
        replacements[case_id] = path

    report = build_scorecard(
        baseline_json=args.baseline_json,
        replacements=replacements,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
