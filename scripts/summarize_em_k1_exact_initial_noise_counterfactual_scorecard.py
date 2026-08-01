#!/usr/bin/env python3
"""Validate and render the fixed rejected exact-initial-noise scorecard."""

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
    / "em_k1_exact_initial_noise_counterfactual_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT
    / "docs"
    / "math"
    / "em_k1_exact_initial_noise_counterfactual_scorecard.md"
)
SCHEMA = "recovar.em_k1_exact_initial_noise_counterfactual_scorecard.v1"
SUITE_ID = "k1-case22-exact-relion-initial-noise-counterfactual"
CLASSIFICATION = "exact_initial_noise_bootstrap_rejected_under_fixed_score_and_fsc_gates"
FROZEN_DENOMINATOR = 24
SCIENCE_DENOMINATOR = 20
PROVENANCE_DENOMINATOR = 4
STACKS = (35, 252, 348, 591, 683, 1100, 1522, 1640, 1767, 2124, 2322, 2330, 2846, 2994)
CASE_IDS = (
    *(f"score-exact-noise-stack-{stack:04d}" for stack in STACKS),
    "map-parity-beyond-floor-half1",
    "map-parity-beyond-floor-half2",
    "map-parity-beyond-floor-merged",
    "map-gt-nondegradation-half1",
    "map-gt-nondegradation-half2",
    "map-gt-nondegradation-merged",
    "provenance-science-arms-zero-same-gpu",
    "provenance-analysis-job-terminal-zero",
    "provenance-independent-report-byte-identical",
    "provenance-pinned-evidence-no-correlation",
)
EXPECTED_EVIDENCE_SHA256 = {
    "primary_report": "19855c6cff4a639af1f4c55c59ea9b70107d6de44f3ccf52c2e65e4a90796a26",
    "independent_report": "19855c6cff4a639af1f4c55c59ea9b70107d6de44f3ccf52c2e65e4a90796a26",
    "completion_seal": "19c89bf7ad40206320779a95d22eafee54234dc485b2f1f3075e54466256f868",
    "output_manifest": "863ab6174ce13f531147a115c5919247caa0671281f9a805fbb81f50f8de027a",
    "predeclaration": "3b13dd3aff9113aebd3714750b7e2c1cf6f9a4e0619a49ac4e06fd65c7455619",
    "analysis_launcher": "00f12eab856cbc129f4849c00c1e008e3b07a083560f9dab64812ad863a06b71",
    "analyzer": "0d1feabca2d0a756226775916a7b2a292043e6aee272f58a878811ce90228f0d",
    "science_failure_log": "6717a3905540808fcfb2fdc1457512d85b67a4e6e7d296ef476de107484a5269",
    "audit_record": "22a12abe444d7d3a3f604a754edb3194e24308afd88449991e6baa620faddf6e",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce its frozen result."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(scorecard.get("classification") == CLASSIFICATION, "classification changed")
    _require(scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR, "frozen denominator changed")
    _require(
        scorecard.get("science_denominator") == SCIENCE_DENOMINATOR
        and scorecard.get("provenance_denominator") == PROVENANCE_DENOMINATOR,
        "sub-denominator changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("correlation_used") is False,
        "non-scoring metric policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("science_owner_job_id") == 11853352
        and contract.get("science_owner_state") == "FAILED"
        and contract.get("science_owner_exit_code") == "2:0"
        and contract.get("science_arms_exit_status_zero") is True
        and contract.get("analysis_job_id") == 11856679
        and contract.get("analysis_state") == "COMPLETED"
        and contract.get("analysis_exit_code") == "0:0"
        and contract.get("candidate_source_commit")
        == "8fc7c4c9c25d78ac7b0ab3ee84fb3774aa6bbcb5"
        and contract.get("accepted") is False,
        "terminal acceptance contract changed",
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
    _require(tuple(case.get("id") for case in cases) == CASE_IDS, "fixed case identity/order changed")
    for index, case in enumerate(cases):
        expected_pass = index >= SCIENCE_DENOMINATOR
        _require(
            case.get("result") == ("pass" if expected_pass else "fail")
            and case.get("checked") is expected_pass,
            "fixed exact-noise result changed",
        )

    summary = {"pass": sum(case["result"] == "pass" for case in cases), "evaluated": len(cases)}
    science_summary = {
        "pass": sum(case["result"] == "pass" for case in cases[:SCIENCE_DENOMINATOR]),
        "evaluated": SCIENCE_DENOMINATOR,
    }
    provenance_summary = {
        "pass": sum(case["result"] == "pass" for case in cases[SCIENCE_DENOMINATOR:]),
        "evaluated": PROVENANCE_DENOMINATOR,
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(scorecard.get("science_summary") == science_summary, "science summary changed")
    _require(scorecard.get("provenance_summary") == provenance_summary, "provenance summary changed")
    _require(summary == {"pass": 4, "evaluated": 24}, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the rejected candidate as a checked fixed-denominator table."""

    lines = [
        "# K=1 exact RELION initial-noise counterfactual scorecard",
        "",
        "This fixed-denominator same-A100 diagnostic is non-scoring.",
        "Map acceptance uses FSC/FSC-AUC; correlation is forbidden.",
        "",
        f"Accepted gates: **{scorecard['summary']['pass']} / {scorecard['frozen_denominator']}**.",
        f"Science gates: **{scorecard['science_summary']['pass']} / {scorecard['science_denominator']}**.",
        f"Provenance gates: **{scorecard['provenance_summary']['pass']} / {scorecard['provenance_denominator']}**.",
        "",
        "| Checked | Fixed gate | Result |",
        "| --- | --- | ---: |",
    ]
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(f"| {check} | `{case['id']}` | {case['result']} |")
    observations = scorecard["key_observations"]
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            f"Score gates: **{observations['score_gates_passed']}**.",
            f"Parity FSC-AUC gains beyond the control floor: **{observations['map_parity_beyond_control_floor']}**.",
            f"GT FSC-AUC nondegradation: **{observations['map_gt_nondegraded']}**.",
            "",
            "The exact bootstrap leaves all score captures unchanged and slightly",
            "regresses both parity and GT FSC-AUC, so it is rejected.",
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k1_exact_initial_noise_counterfactual_scorecard.py --check",
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
