#!/usr/bin/env python3
"""Validate and render the fixed K=4 deterministic candidate scorecard."""

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
    / "em_k4_deterministic_contribution_repeatability_candidate_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT
    / "docs"
    / "math"
    / "em_k4_deterministic_contribution_repeatability_candidate_scorecard.md"
)
SCHEMA = "recovar.em_k4_deterministic_contribution_repeatability_candidate_scorecard.v1"
SUITE_ID = "k4-softmask2-it2-h1-same-observer-repeatability-candidate"
CLASSIFICATION = "same_observer_archives_repeat_bit_for_bit"
SOURCE_COMMIT = "e98a5f333cc789f1e2511da58b95b974c6fe6636"
CUDA_SHA256 = "567f8d8af5a45a0a3501f30ee22bc8b012243de7edc1ad7464cfadb197bbea69"
FROZEN_DENOMINATOR = 3
CASE_IDS = (
    "candidate-pass2-archive-byte-equality",
    "candidate-contribution-archive-byte-equality",
    "candidate-device-signature-archive-byte-equality",
)
EXPECTED_EVIDENCE_SHA256 = {
    "strict_audit_report": "59e37fa9155075822ccbf1994d562dcab949cad9a49ea4261911184d12ed22ce",
    "completion_seal": "071cc32789470d39f4a4045f5f796f99fcababe81d8eec60c6640186971672a0",
    "science_output_manifest": "3430b96dc2a17a55751b9fa5eea82f599a9a69708b93f5fe32c0d22e41d277de",
    "static_input_manifest": "0a224d9a646f1ef33ca9642b5804f94ece203d267930db61c83e067fd37d221d",
    "launcher": "cbf39909660218e8ceeceea5d3617829ef62abed4d7a7e61bcd9e2899bd7b125",
    "predeclaration": "3a6b01f6a70fcd6eb420f61f4d6b0e9e850e7aa6bc1bbab084b9fc9336c921d8",
    "post_terminal_audit": "859827b619c212ab7267dae9c471a221fbb893a50c5a83e114da2b9c7af592aa",
    "independent_manifest_verification": "15f57de1363d004d1eac624e7850ba5b8b607f12c62fb5922bcc9e813a75689c",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce its frozen identity."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(scorecard.get("classification") == CLASSIFICATION, "classification changed")
    _require(scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR, "frozen denominator changed")
    _require(
        scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("correlation_used") is False
        and scorecard.get("fsc_auc_evaluated") is False,
        "non-scoring metric policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("producer_job_id") == 11854999
        and contract.get("producer_state") == "COMPLETED"
        and contract.get("producer_exit_code") == "0:0"
        and contract.get("source_commit") == SOURCE_COMMIT
        and contract.get("cuda_library_sha256") == CUDA_SHA256
        and contract.get("repeatability_accepted") is True
        and contract.get("production_integration_accepted") is False
        and contract.get("quality_gate_job_id") == 11854692,
        "terminal or quality-contingent acceptance contract changed",
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
    for case in cases:
        _require(
            case.get("result") == "pass"
            and case.get("checked") is True
            and case.get("failed_array_count") == 0,
            "strict candidate byte result changed",
        )

    summary = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"pass": 3, "evaluated": 3}, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the checked candidate scorecard as a compact table."""

    summary = scorecard["summary"]
    lines = [
        "# K=4 deterministic contribution-repeatability candidate scorecard",
        "",
        "This fixed-denominator panel records strict same-observer candidate",
        "repeatability. It is non-scoring and remains contingent on the separate",
        "K=4 FSC/FSC-AUC and GT-quality A/B.",
        "",
        f"Strict byte equality: **{summary['pass']} / {scorecard['frozen_denominator']}**.",
        "Published baseline retained: **0 / 3**.",
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
            "The implementation is not production-accepted until the quality A/B passes.",
            "No correlation, tolerance, scale, sign, threshold, map, or FSC claim is used.",
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            (
                "pixi run python scripts/"
                "summarize_em_k4_deterministic_contribution_repeatability_candidate_scorecard.py "
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
