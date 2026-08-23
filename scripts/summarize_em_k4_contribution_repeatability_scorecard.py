#!/usr/bin/env python3
"""Validate and render the fixed K=4 contribution-repeatability scorecard."""

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
    / "em_k4_contribution_repeatability_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT / "docs" / "math" / "em_k4_contribution_repeatability_scorecard.md"
)
SCHEMA = "recovar.em_k4_contribution_repeatability_scorecard.v1"
SUITE_ID = "k4-it2-h1-orig53722-class1-same-observer-repeatability"
CLASSIFICATION = "same_observer_archives_do_not_repeat_bit_for_bit"
FROZEN_DENOMINATOR = 3
CASE_IDS = (
    "pass2-archive-byte-equality",
    "contribution-archive-byte-equality",
    "device-signature-archive-byte-equality",
)
EXPECTED_FAILED_ARRAY_COUNTS = (5, 14, 2)
EXPECTED_EVIDENCE_SHA256 = {
    "strict_audit_report": (
        "9c791cfe7de4bc17b391ee55c896e9451db466618a9f82ab59e2393928d54b7f"
    ),
    "strict_audit_complete": (
        "ad355ce2ef184297fa8e5b005f152cb0b5fa67f3b1b4e610c22987f7ea1ee9db"
    ),
    "observed_pass2": (
        "a654eb32963659b0c7641410bde4216f270e1f9901683b463ba198d480584afc"
    ),
    "observed_contribution": (
        "a7bbd6c00a40c5a77cd3b0129aae6a177ea7f76365de1b1a324c24e978f488c3"
    ),
    "observed_device_signature": (
        "63c14a7ee4fedf8b2a62ed366e7847c9c7a5643c107fae7cc1981b3a1ba10934"
    ),
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce its fixed denominator."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(
        scorecard.get("classification") == CLASSIFICATION,
        "repeatability classification changed",
    )
    _require(
        scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR,
        "frozen denominator changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("correlation_used") is False
        and scorecard.get("fsc_auc_evaluated") is False,
        "non-scoring metric policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("producer_job_id") == 11847462
        and contract.get("producer_state") == "FAILED"
        and contract.get("audit_job_id") == 11847542
        and contract.get("audit_state") == "COMPLETED"
        and contract.get("accepted") is False,
        "terminal Slurm acceptance contract changed",
    )

    evidence = scorecard.get("evidence")
    _require(
        isinstance(evidence, dict) and set(evidence) == set(EXPECTED_EVIDENCE_SHA256),
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
    for case, expected_failed_arrays in zip(
        cases,
        EXPECTED_FAILED_ARRAY_COUNTS,
        strict=True,
    ):
        _require(
            case.get("result") == "fail" and case.get("checked") is False,
            "strict byte result changed",
        )
        _require(
            case.get("failed_array_count") == expected_failed_arrays,
            "failed-array count changed",
        )

    summary = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"pass": 0, "evaluated": 3}, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the checked repeatability scorecard as a compact table."""

    summary = scorecard["summary"]
    lines = [
        "# K=4 contribution repeatability scorecard",
        "",
        "This fixed-denominator diagnostic tests strict same-observer archive",
        "repeatability. It cannot change the FSC/FSC-AUC quality scorecards.",
        "",
        f"Strict byte equality: **{summary['pass']} / {scorecard['frozen_denominator']}**.",
        "",
        "| Checked | Case | Archive | Result | Failed arrays |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(
            f"| {check} | `{case['id']}` | `{case['archive']}` | "
            f"{case['result']} | {case['failed_array_count']} |"
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
                "scripts/summarize_em_k4_contribution_repeatability_scorecard.py "
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
