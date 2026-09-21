#!/usr/bin/env python3
"""Validate and render the fixed K=4 exact-device causal scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = (
    REPO_ROOT / "docs" / "math" / "em_k4_causal_boundary_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT / "docs" / "math" / "em_k4_causal_boundary_scorecard.md"
)
SCHEMA = "recovar.em_k4_causal_boundary_scorecard.v1"
SUITE_ID = "k4-it2-class1-exact-device-owner-pair"
FROZEN_DENOMINATOR = 4
CASE_IDS = (
    "native-target-operand-replay",
    "fixed-target-raw-diff2",
    "global-raw-diff2",
    "global-combined-score",
)
VALID_RESULTS = {"pass", "fail"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_CLASSIFICATION = (
    "global_raw_and_score_paths_differ_but_fixed_target_closes"
)
EXPECTED_TARGET_GPU_UUID = "GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30"
EXPECTED_EVIDENCE_SHA256 = {
    "completion_report": (
        "963e9b6b315368ae9a8201b73624163129f92e5626fff4089ad8fe3ce6516552"
    ),
    "raw_report": (
        "f19dbd316eb654d0c38d6a334cb052ef5181e201789c576bbece4f604849e214"
    ),
    "operand_report": (
        "151194b5412949aa450453e469edc6371ff25256e883b6d641d79ee1919476dd"
    ),
    "pair_report": (
        "be047c3ad90220c88834ed0995339bceab65f4cf3d7358fabdf9bac28ebd142c"
    ),
    "raw_capture": (
        "ccbdc9040da463f479784e3ad270fd76bb5817006742f43c96f9b053bf9d6eef"
    ),
    "operand_capture": (
        "93322e2b98ca11e626f178007f39cf8d6137655fdffd5239907cd2321459270f"
    ),
}


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
        scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR,
        "frozen denominator changed",
    )
    _require(
        scorecard.get("metric_policy")
        == (
            "Fixed bitwise exact-device causal gates only; no fitted scale, "
            "sign, threshold, map metric, or correlation."
        ),
        "metric policy changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False,
        "causal scorecard cannot authorize a production score change",
    )
    _require(
        scorecard.get("classification") == EXPECTED_CLASSIFICATION,
        "owner-pair classification changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(isinstance(contract, dict), "acceptance contract is missing")
    _require(
        contract.get("same_physical_gpu") is True,
        "same-physical-GPU gate changed",
    )
    _require(
        contract.get("target_gpu_uuid") == EXPECTED_TARGET_GPU_UUID,
        "target physical GPU changed",
    )
    _require(
        contract.get("grid_correction") == "unset/default-off",
        "grid-correction policy changed",
    )
    _require(
        contract.get("forced_final_all_data_after_nonconvergence") is False,
        "invalid forced final all-data is enabled",
    )
    _require(
        (
            contract.get("raw_science_owner_job_id"),
            contract.get("operand_science_owner_job_id"),
            contract.get("dependency_bound_audit_job_id"),
        )
        == (11790517, 11812925, 11812941),
        "science/audit owner jobs changed",
    )

    evidence = scorecard.get("evidence")
    _require(isinstance(evidence, dict) and evidence, "evidence is missing")
    _require(
        set(evidence) == set(EXPECTED_EVIDENCE_SHA256),
        "fixed evidence identity changed",
    )
    for name, record in evidence.items():
        _require(isinstance(record, dict), f"{name}: invalid evidence record")
        evidence_path = record.get("path")
        digest = record.get("sha256")
        _require(
            isinstance(evidence_path, str)
            and Path(evidence_path).is_absolute(),
            f"{name}: evidence path must be absolute",
        )
        _require(
            isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{name}: invalid SHA-256",
        )
        _require(
            digest == EXPECTED_EVIDENCE_SHA256[name],
            f"{name}: evidence SHA-256 changed",
        )

    cases = scorecard.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == FROZEN_DENOMINATOR,
        "cases do not preserve the frozen denominator",
    )
    case_ids = tuple(case.get("id") for case in cases)
    _require(case_ids == CASE_IDS, "fixed case identity/order changed")
    _require(len(set(case_ids)) == len(case_ids), "case IDs are not unique")
    for case in cases:
        result = case.get("result")
        _require(result in VALID_RESULTS, f"{case.get('id')}: invalid result")
        _require(
            case.get("checked") is (result == "pass"),
            f"{case.get('id')}: checkmark disagrees with result",
        )
        _require(
            isinstance(case.get("name"), str) and case["name"],
            f"{case.get('id')}: missing name",
        )
        _require(
            isinstance(case.get("observed"), str) and case["observed"],
            f"{case.get('id')}: missing observation",
        )

    counts = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "fail": sum(case["result"] == "fail" for case in cases),
    }
    expected_summary = {
        **counts,
        "evaluated": counts["pass"] + counts["fail"],
    }
    _require(
        scorecard.get("summary") == expected_summary,
        "recorded summary does not match fixed cases",
    )
    _require(
        expected_summary["evaluated"] == FROZEN_DENOMINATOR,
        "not all fixed cases are evaluated",
    )
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the checked scorecard as a compact human-readable table."""

    summary = scorecard["summary"]
    lines = [
        "# K=4 exact-device causal boundary scorecard",
        "",
        "This is a non-scoring, fixed-denominator diagnostic. It cannot change",
        "the frozen K=1 or K=4 FSC/FSC-AUC scorecards.",
        "",
        (
            f"Fixed causal score: **{summary['pass']} / "
            f"{scorecard['frozen_denominator']} passing** "
            f"({summary['evaluated']} / "
            f"{scorecard['frozen_denominator']} evaluated)."
        ),
        "",
        "| Checked | Gate | Result | Observation |",
        "| --- | --- | --- | --- |",
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
            (
                "Metric policy: "
                f"{scorecard['metric_policy']}"
            ),
            "",
            "Immutable evidence:",
            "",
        ]
    )
    for name, record in scorecard["evidence"].items():
        lines.append(
            f"- `{name}`: `{record['path']}` "
            f"(SHA-256 `{record['sha256']}`)"
        )
    lines.extend(
        [
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            (
                "pixi run python "
                "scripts/summarize_em_k4_causal_boundary_scorecard.py "
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
