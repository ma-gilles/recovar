#!/usr/bin/env python3
"""Validate and render the fixed K=1 live-noise counterfactual scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs/math/em_k1_live_noise_counterfactual_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs/math/em_k1_live_noise_counterfactual_scorecard.md"
SCHEMA = "recovar.em_k1_live_noise_counterfactual_scorecard.v1"
SUITE_ID = "k1-case22-live-relion-binary64-noise-counterfactual"
CLASSIFICATION = "live_noise_improves_score_and_parity_but_regresses_gt"
FROZEN_DENOMINATOR = 24
SCIENCE_DENOMINATOR = 20
PROVENANCE_DENOMINATOR = 4
EXPECTED_SUMMARY = {"pass": 21, "evaluated": 24}
EXPECTED_SCIENCE_SUMMARY = {"pass": 17, "evaluated": 20}
EXPECTED_PROVENANCE_SUMMARY = {"pass": 4, "evaluated": 4}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_EVIDENCE_SHA256 = {
    "primary_report": "c75a0855e73bf617bfff01c0a81ca54f022be9393a0ec7071425a88ee4e9a63c",
    "independent_report": "c75a0855e73bf617bfff01c0a81ca54f022be9393a0ec7071425a88ee4e9a63c",
    "completion_seal": "03a6c5f57b400aba3549cf4aa60782491ea3398d242c1d51b6badef228394f0d",
    "predeclaration": "ade30bfc5ffe97f5853a02c26ba0dac4fbf53ce53c89eafb422ccaf6532ec909",
    "analysis_launcher": "0dca62651bb5b1ed93041c3be53b97bd41203b4b87b1c68bcf60a6dad62f0cfa",
    "output_manifest": "3ea81fcc18a4529524ebcc955434011c46902aac8a01f3420ac269ff6ff45e87",
    "audit_record": "84d4c074120e9f3c7c7cef1cf0edcc8bc2d4ad392f4feaf613259d37b29f9fe8",
    "serialized_a_walltime": "5505a7306e536eef18c2df09cbcb7276734ead83cb8d8f07353949d1fcb262c6",
    "serialized_b_walltime": "6bcdbd16bbcd3d7bd6628b63b287d0066564597008e8aacc844033f6c727ec98",
    "live_noise_walltime": "e096967a010c49eee61ff993842bc9cd67dd788a0a3c4e713f1025477718aba2",
}
EXPECTED_CASE_RESULTS = (
    *((f"score-live-noise-stack-{stack}", "pass") for stack in (
        "0035", "0252", "0348", "0591", "0683", "1100", "1522",
        "1640", "1767", "2124", "2322", "2330", "2846", "2994",
    )),
    ("map-parity-beyond-floor-half1", "pass"),
    ("map-parity-beyond-floor-half2", "pass"),
    ("map-parity-beyond-floor-merged", "pass"),
    ("map-gt-nondegradation-half1", "fail"),
    ("map-gt-nondegradation-half2", "fail"),
    ("map-gt-nondegradation-merged", "fail"),
    ("provenance-science-arms-zero-same-gpu", "pass"),
    ("provenance-analysis-job-terminal-zero", "pass"),
    ("provenance-independent-report-byte-identical", "pass"),
    ("provenance-pinned-evidence-no-correlation", "pass"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(scorecard.get("classification") == CLASSIFICATION, "classification changed")
    _require(scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR, "frozen denominator changed")
    _require(scorecard.get("science_denominator") == SCIENCE_DENOMINATOR, "science denominator changed")
    _require(scorecard.get("provenance_denominator") == PROVENANCE_DENOMINATOR, "provenance denominator changed")
    _require(scorecard.get("correlation_used") is False, "correlation policy changed")
    _require(scorecard.get("scorecard_change_admissible") is False, "scorecard policy changed")
    contract = scorecard.get("acceptance_contract", {})
    _require(
        contract.get("science_owner_job_id") == 11852265
        and contract.get("science_owner_state") == "FAILED"
        and contract.get("science_owner_exit_code") == "1:0"
        and contract.get("science_arms_exit_status_zero") is True
        and contract.get("analysis_job_id") == 11855557
        and contract.get("analysis_state") == "COMPLETED"
        and contract.get("analysis_exit_code") == "0:0"
        and contract.get("accepted") is False,
        "terminal acceptance contract changed",
    )
    evidence = scorecard.get("evidence", {})
    _require(set(evidence) == set(EXPECTED_EVIDENCE_SHA256), "evidence set changed")
    for name, record in evidence.items():
        _require(Path(record["path"]).is_absolute(), f"{name}: path is not absolute")
        _require(SHA256_RE.fullmatch(record["sha256"]) is not None, f"{name}: invalid SHA-256")
        _require(record["sha256"] == EXPECTED_EVIDENCE_SHA256[name], f"{name}: evidence digest changed")
    _require(
        evidence["primary_report"]["sha256"] == evidence["independent_report"]["sha256"],
        "independent report digest changed",
    )
    cases = scorecard.get("cases")
    _require(isinstance(cases, list) and len(cases) == FROZEN_DENOMINATOR, "cases changed denominator")
    _require(len({case["id"] for case in cases}) == FROZEN_DENOMINATOR, "case identities are not unique")
    _require(
        tuple((case["id"], case["result"]) for case in cases) == EXPECTED_CASE_RESULTS,
        "fixed case identities or results changed",
    )
    for case in cases:
        _require(case["result"] in {"pass", "fail"}, "invalid case result")
        _require(case["checked"] is (case["result"] == "pass"), "checkbox state changed")
    summary = {"pass": sum(case["result"] == "pass" for case in cases), "evaluated": len(cases)}
    science_cases = cases[:SCIENCE_DENOMINATOR]
    provenance_cases = cases[SCIENCE_DENOMINATOR:]
    science_summary = {
        "pass": sum(case["result"] == "pass" for case in science_cases),
        "evaluated": len(science_cases),
    }
    provenance_summary = {
        "pass": sum(case["result"] == "pass" for case in provenance_cases),
        "evaluated": len(provenance_cases),
    }
    _require(summary == EXPECTED_SUMMARY and scorecard.get("summary") == summary, "fixed result changed")
    _require(
        science_summary == EXPECTED_SCIENCE_SUMMARY
        and scorecard.get("science_summary") == science_summary,
        "science result changed",
    )
    _require(
        provenance_summary == EXPECTED_PROVENANCE_SUMMARY
        and scorecard.get("provenance_summary") == provenance_summary,
        "provenance result changed",
    )
    return scorecard


def render_markdown(scorecard: dict) -> str:
    lines = [
        "# K=1 live RELION binary64-noise counterfactual scorecard",
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
            "The live-noise treatment improves RECOVAR-to-RELION FSC-AUC but regresses",
            "GT FSC-AUC for half 1, half 2, and merged maps, so it is rejected.",
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k1_live_noise_counterfactual_scorecard.py --check",
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
    rendered = render_markdown(load_and_validate(args.scorecard))
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
