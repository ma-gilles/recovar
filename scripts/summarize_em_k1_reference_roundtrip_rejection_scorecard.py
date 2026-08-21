#!/usr/bin/env python3
"""Validate and render the rejected K=1 reference-roundtrip experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = (
    REPO_ROOT
    / "docs"
    / "math"
    / "em_k1_reference_roundtrip_rejection_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT
    / "docs"
    / "math"
    / "em_k1_reference_roundtrip_rejection_scorecard.md"
)
SCHEMA = "recovar.em_k1_reference_roundtrip_rejection_scorecard.v1"
SUITE_ID = "k1-case22-reference-roundtrip-preintervention-control"
CLASSIFICATION = (
    "reference_roundtrip_experiment_rejected_by_preintervention_and_gt_fsc_gates"
)
FROZEN_DENOMINATOR = 9
CASE_IDS = (
    "precontrol-half1-byte-identity",
    "precontrol-half2-byte-identity",
    "baseline-component-validator",
    "roundtrip-component-validator",
    "serialized-component-validator",
    "baseline-roundtrip-normalization-identity",
    "serialized-score-boundary",
    "serialized-map-parity",
    "serialized-map-gt-nondegradation",
)
EXPECTED_RESULTS = (
    "fail",
    "fail",
    "fail",
    "fail",
    "fail",
    "fail",
    "pass",
    "pass",
    "fail",
)
EXPECTED_EVIDENCE_SHA256 = {
    "post_terminal_audit": (
        "ad7288ae5a5bd86cba7830b9eb6cf2d6e9b2f4d1cb2040f4e34eb75955a95daf"
    ),
    "baseline_to_roundtrip_operand_boundary": (
        "484a2ed1efae5aee22d90a3a4f813aae1ac062146c410a97aa3e2af181d247b0"
    ),
    "serialized_restart_retention_score_boundary": (
        "2ec32f492ba307e860156095c2fb3f391850e06c082f71f57c3652b4c5a2dce8"
    ),
    "serialized_restart_map_fsc": (
        "1522a1955bbc5e4a909f077d4dee42291890cf85962b66e2e395a7f75ba2a07b"
    ),
    "baseline_component_validation": (
        "c3f3e9bda61ed5888b6704d777be8d768e33602fc7cd9c6cc737d783ae2d5d1e"
    ),
    "roundtrip_component_validation": (
        "4ea71e4c0fcc7c5bb4211584054912bde8a5952c070e4d388cf0ddf9cab88cc9"
    ),
    "serialized_component_validation": (
        "4288e60c655892fec6a3cbaf2f369673b45d3e3b837b160e99d2861be8d1c999"
    ),
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce its rejected status."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(
        scorecard.get("classification") == CLASSIFICATION,
        "classification changed",
    )
    _require(
        scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR,
        "frozen denominator changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("correlation_used") is False
        and scorecard.get("fsc_auc_evaluated") is True,
        "non-scoring metric policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("job_id") == 11848603
        and contract.get("job_state") == "FAILED"
        and contract.get("job_exit_code") == "1:0"
        and contract.get("accepted") is False
        and contract.get("science_complete_marker_present") is False,
        "terminal Slurm acceptance contract changed",
    )

    evidence = scorecard.get("evidence")
    _require(
        isinstance(evidence, dict)
        and set(evidence) == set(EXPECTED_EVIDENCE_SHA256),
        "fixed evidence identity changed",
    )
    for name, expected_digest in EXPECTED_EVIDENCE_SHA256.items():
        record = evidence[name]
        evidence_path = record.get("path")
        digest = record.get("sha256")
        _require(
            isinstance(evidence_path, str) and Path(evidence_path).is_absolute(),
            f"{name}: evidence path must be absolute",
        )
        _require(
            isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{name}: invalid SHA-256",
        )
        _require(digest == expected_digest, f"{name}: evidence SHA-256 changed")

    cases = scorecard.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == FROZEN_DENOMINATOR,
        "cases do not preserve the frozen denominator",
    )
    _require(
        tuple(case.get("id") for case in cases) == CASE_IDS,
        "fixed case identity/order changed",
    )
    for case, expected_result in zip(cases, EXPECTED_RESULTS, strict=True):
        _require(case.get("result") == expected_result, "fixed result changed")
        _require(
            case.get("checked") is (expected_result == "pass"),
            "checkbox state changed",
        )

    summary = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"pass": 2, "evaluated": 9}, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the fixed rejected-experiment panel."""

    summary = scorecard["summary"]
    lines = [
        "# K=1 reference-roundtrip rejection scorecard",
        "",
        "This fixed-denominator diagnostic records every predeclared control and",
        "acceptance gate from the rejected case-22 reference-roundtrip experiment.",
        "It cannot change the FSC/FSC-AUC quality scorecards.",
        "",
        (
            f"Accepted gates: **{summary['pass']} / "
            f"{scorecard['frozen_denominator']}**."
        ),
        "",
        "| Checked | Case | Result | Observation |",
        "| --- | --- | ---: | --- |",
    ]
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(
            f"| {check} | `{case['id']}` | {case['result']} | "
            f"{case['observed']} |"
        )
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            "Immutable evidence:",
            "",
        ]
    )
    for name, record in scorecard["evidence"].items():
        lines.append(
            f"- `{name}`: `{record['path']}` (SHA-256 `{record['sha256']}`)"
        )
    lines.extend(
        [
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            (
                "pixi run python "
                "scripts/summarize_em_k1_reference_roundtrip_rejection_scorecard.py "
                "--check"
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scorecard = load_and_validate(args.scorecard)
    rendered = render_markdown(scorecard)
    if args.check:
        target = DEFAULT_MARKDOWN if args.output is None else args.output
        if target.read_text() != rendered:
            raise SystemExit(f"{target} is stale; regenerate it")
    elif args.output is not None:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
