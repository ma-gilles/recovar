#!/usr/bin/env python3
"""Validate and render the fixed RECOVAR K=4 all-class repeatability panel."""

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
    / "em_k4_allclass_recovar_repeatability_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT
    / "docs"
    / "math"
    / "em_k4_allclass_recovar_repeatability_scorecard.md"
)
SCHEMA = "recovar.em_k4_allclass_recovar_repeatability_scorecard.v1"
SUITE_ID = "k4-it2-fixed-a100-recovar-allclass-pass2-repeatability"
CLASSIFICATION = "all_observed_pass2_fields_exact"
FROZEN_DENOMINATOR = 9
EXPECTED_CASES = (
    "arm-a-valid",
    "arm-b-valid",
    "identity-exact",
    "geometry-and-candidate-tuples-exact",
    "raw-diff2-exact",
    "priors-exact",
    "unnormalized-scores-exact",
    "joint-posterior-exact",
    "global-significant-support-exact",
)
EXPECTED_ARTIFACT_SHA256 = (
    "32c0318a70f71437fab705cbd31fa6f7ef47ef542556954059d02628129f5b97",
    "7c5e4dcbde0c5a2e3cfce238d737ac0a0525ae6e7f92b79a3c3cc404082516c7",
    "986077d552326bebab005058068a83c5bc9725ec5ae61cb1ccd3896e2dae8535",
    "4d80fc4b3cb0bccb8f7f772274ed2b2462d0544579b66180330a64b434c1402f",
)
EXPECTED_EVIDENCE_SHA256 = {
    "repeatability_report": "3e2341222a1a2e00a014995245709f8c5383eed9d611adcf23128bbc34f7f4cd",
    "launcher": "b4b0bb886ae39df51c0cde8a99c599ddbb9615049780829dc38e10544f8e7fa9",
    "predeclaration": "935b032ad097816a5b6943a1a160c8ffc00adbfd47f0013f9a3a7897004ecca2",
    "submission": "4948b7220138bdfa0fb91fbd6c59ade980fa9bdbe8afaad9a92b394ecb73e79f",
    "science_manifest": "34b57e868fff2a9cdaab0b8dad3a77fbe313912e02c4883dd90793d1bc6bc294",
    "static_manifest": "dcea997986165acdd5826e4eadbf468c54b3e98685b7c15cdfa78fc4a74dff91",
    "science_log": "2b9d6c6251f35b9066bf9d43df05fd4e5cfd3b56c32871be389d6052e0948580",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the scorecard and enforce its fixed same-device evidence."""

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
        and scorecard.get("cross_engine_parity_established") is False
        and scorecard.get("stable_recovar_boundary_established") is True,
        "scope or metric policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("slurm_job_id") == 11994138
        and contract.get("state") == "COMPLETED"
        and contract.get("exit_code") == "0:0"
        and contract.get("elapsed") == "01:17:58"
        and contract.get("node") == "della-l07g2"
        and contract.get("gpu_uuid") == "GPU-f3e94635-d095-bea9-dbe3-26e91dd3ea27"
        and contract.get("recovar_science_commit")
        == "223e7e81188e3d63217605245d08999c26a86d5b"
        and contract.get("accepted") is True,
        "terminal Slurm acceptance contract changed",
    )

    boundary = scorecard.get("boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("iteration") == 2
        and boundary.get("current_size") == 38
        and boundary.get("target_original_index_zero_based") == 53722
        and boundary.get("target_stack_index_one_based") == 53723
        and boundary.get("joint_probability_mass_arm_a") == 0.9999999999999997
        and boundary.get("joint_probability_mass_arm_b") == 0.9999999999999997
        and boundary.get("active_candidate_count") == 247232
        and boundary.get("significant_candidate_count") == 66986
        and tuple(boundary.get("class_artifact_sha256", ()))
        == EXPECTED_ARTIFACT_SHA256,
        "repeatability boundary changed",
    )

    cases = scorecard.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == FROZEN_DENOMINATOR,
        "cases do not preserve the frozen denominator",
    )
    _require(
        tuple(case.get("id") for case in cases) == EXPECTED_CASES,
        "fixed gate identity/order changed",
    )
    _require(
        all(case.get("result") == "pass" and case.get("checked") is True for case in cases),
        "fixed repeatability result changed",
    )
    summary = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"pass": 9, "evaluated": 9}, "fixed result changed")

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
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the fixed repeatability panel as a compact checked table."""

    summary = scorecard["summary"]
    boundary = scorecard["boundary"]
    lines = [
        "# K=4 RECOVAR all-class repeatability scorecard",
        "",
        "This fixed-denominator panel compares two independent RECOVAR",
        "iteration-2 four-class captures on one physical A100. It establishes",
        "a stable RECOVAR-side boundary only; it does not establish cross-engine parity.",
        "",
        f"Exact repeatability gates: **{summary['pass']} / {scorecard['frozen_denominator']}**.",
        "",
        "| Checked | Gate | Result |",
        "| --- | --- | --- |",
    ]
    for case in scorecard["cases"]:
        lines.append(f"| [x] | `{case['id']}` | pass |")
    lines.extend(
        [
            "",
            (
                f"The fixed target contains {boundary['active_candidate_count']:,} active "
                f"and {boundary['significant_candidate_count']:,} significant class-pose "
                "tuples; every observed identity, tuple, score, prior, posterior, and "
                "support array is byte-exact across the two arms."
            ),
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            "Immutable evidence:",
            "",
        ]
    )
    for name, record in scorecard["evidence"].items():
        lines.append(f"- `{name}`: `{record['path']}` (SHA-256 `{record['sha256']}`)")
    lines.extend(
        [
            "",
            "Code references:",
            "",
            "- `scripts/analyze_em_k4_allclass_recovar_repeatability.py`",
            "- `scripts/summarize_em_k4_allclass_recovar_repeatability_scorecard.py`",
            "- `scripts/report_em_parity_progress.py`",
            "",
            "To validate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k4_allclass_recovar_repeatability_scorecard.py --check",
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
