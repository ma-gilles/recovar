#!/usr/bin/env python3
"""Report all fixed RECOVAR/RELION EM parity panels in one table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_em_k4_causal_boundary_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K4_CAUSAL_SCORECARD,
)
from scripts.summarize_em_k4_causal_boundary_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k4_causal,
)
from scripts.summarize_em_relion_parity_scorecard import (  # noqa: E402
    DEFAULT_FIXTURE_MANIFEST,
    DEFAULT_K4_SNAPSHOT,
    DEFAULT_SCORECARD,
    load_and_validate,
    load_and_validate_fixture_manifest,
    load_and_validate_k4_snapshot,
    sha256_file,
)

SCHEMA = "recovar.em_parity_progress.v1"


def _panel(
    panel_id: str,
    label: str,
    passed: int,
    evaluated: int,
    denominator: int,
    *,
    scoring: bool,
) -> dict[str, object]:
    if not (0 <= passed <= evaluated <= denominator):
        raise ValueError(f"{panel_id}: invalid counts passed={passed} evaluated={evaluated} denominator={denominator}")
    return {
        "id": panel_id,
        "label": label,
        "passed": passed,
        "evaluated": evaluated,
        "denominator": denominator,
        "rate_percent": round(100.0 * passed / denominator, 1),
        "scoring": scoring,
    }


def _input_record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT.resolve())),
        "sha256": sha256_file(path),
    }


def build_progress(
    *,
    scorecard_path: Path = DEFAULT_SCORECARD,
    fixture_manifest_path: Path = DEFAULT_FIXTURE_MANIFEST,
    k4_snapshot_path: Path = DEFAULT_K4_SNAPSHOT,
    k4_causal_path: Path = DEFAULT_K4_CAUSAL_SCORECARD,
) -> dict[str, object]:
    """Validate every fixed source and return the consolidated progress report."""

    scorecard = load_and_validate(scorecard_path)
    load_and_validate_fixture_manifest(fixture_manifest_path, scorecard)
    k4_snapshot = load_and_validate_k4_snapshot(k4_snapshot_path)
    k4_causal = load_and_validate_k4_causal(k4_causal_path)

    k1_counts = scorecard["current_snapshot"]["counts"]
    k1_denominator = scorecard["frozen_denominator"]
    k1_evaluated = k1_counts["pass"] + k1_counts["fail"]
    k1_topology_passed = sum(case["intermediate_result"] == "pass" for case in scorecard["cases"])
    k4_direct_denominator = k4_snapshot["direct_fsc_auc_checks_total"]
    k4_iteration_denominator = k4_snapshot["numbered_iterations"]
    k4_causal_summary = k4_causal["summary"]

    panels = [
        _panel(
            "k1_strict",
            "K=1 strict FSC/FSC-AUC",
            k1_counts["pass"],
            k1_evaluated,
            k1_denominator,
            scoring=True,
        ),
        _panel(
            "k1_topology",
            "K=1 topology",
            k1_topology_passed,
            k1_evaluated,
            k1_denominator,
            scoring=False,
        ),
        _panel(
            "k1_evaluated",
            "K=1 evaluated",
            k1_evaluated,
            k1_evaluated,
            k1_denominator,
            scoring=False,
        ),
        _panel(
            "k4_direct",
            "K=4 direct per-class FSC-AUC",
            k4_snapshot["direct_fsc_auc_checks_passed"],
            k4_direct_denominator,
            k4_direct_denominator,
            scoring=True,
        ),
        _panel(
            "k4_all_class",
            "K=4 all-class iterations",
            k4_snapshot["iterations_all_classes_passed"],
            k4_iteration_denominator,
            k4_iteration_denominator,
            scoring=True,
        ),
        _panel(
            "k4_causal",
            "K=4 exact-device causal boundary",
            k4_causal_summary["pass"],
            k4_causal_summary["evaluated"],
            k4_causal["frozen_denominator"],
            scoring=False,
        ),
    ]
    return {
        "schema": SCHEMA,
        "metric_policy": (
            "K=1 and K=4 quality panels use shellwise FSC/FSC-AUC; "
            "correlation is not used. The K=4 causal panel is non-scoring."
        ),
        "scorecard_change_admissible": False,
        "panels": panels,
        "k1_strict_history": [snapshot["counts"]["pass"] for snapshot in scorecard["history"]],
        "inputs": {
            "k1_scorecard": _input_record(scorecard_path),
            "k1_fixture_manifest": _input_record(fixture_manifest_path),
            "k4_trajectory_snapshot": _input_record(k4_snapshot_path),
            "k4_causal_scorecard": _input_record(k4_causal_path),
        },
    }


def render_markdown(progress: dict[str, object]) -> str:
    """Render the consolidated fixed panels as a compact PR-ready table."""

    lines = [
        "| Fixed panel | Passed | Evaluated | Denominator | Rate | Scoring |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for panel in progress["panels"]:
        scoring = "yes" if panel["scoring"] else "no"
        lines.append(
            f"| {panel['label']} | **{panel['passed']}** | "
            f"{panel['evaluated']} | {panel['denominator']} | "
            f"{panel['rate_percent']:.1f}% | {scoring} |"
        )
    history = " → ".join(str(value) for value in progress["k1_strict_history"])
    lines.extend(
        [
            "",
            f"K=1 strict progress on the unchanged denominator: **{history}**.",
            "",
            str(progress["metric_policy"]),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    progress = build_progress()
    if args.format == "json":
        rendered = json.dumps(progress, indent=2, sort_keys=True) + "\n"
    else:
        rendered = render_markdown(progress)
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
