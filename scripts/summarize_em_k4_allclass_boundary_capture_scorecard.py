#!/usr/bin/env python3
"""Validate and render the fixed RECOVAR K=4 all-class boundary capture."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = (
    REPO_ROOT / "docs" / "math" / "em_k4_allclass_boundary_capture_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT / "docs" / "math" / "em_k4_allclass_boundary_capture_scorecard.md"
)
SCHEMA = "recovar.em_k4_allclass_boundary_capture_scorecard.v1"
SUITE_ID = "k4-it2-fixed-a100-recovar-allclass-pass2-boundary"
CLASSIFICATION = "recovar_four_class_joint_posterior_boundary_complete"
FROZEN_DENOMINATOR = 4
EXPECTED_CLASSES = {
    1: (2968, 109184, 38982, "32c0318a70f71437fab705cbd31fa6f7ef47ef542556954059d02628129f5b97"),
    2: (2432, 65952, 14076, "7c5e4dcbde0c5a2e3cfce238d737ac0a0525ae6e7f92b79a3c3cc404082516c7"),
    3: (2096, 64704, 11804, "986077d552326bebab005058068a83c5bc9725ec5ae61cb1ccd3896e2dae8535"),
    4: (392, 7392, 2124, "4d80fc4b3cb0bccb8f7f772274ed2b2462d0544579b66180330a64b434c1402f"),
}
EXPECTED_EVIDENCE_SHA256 = {
    "capture_report": "45beed43d823191ca6ad2358cd3965cde80ffc534b67a5e127b3f9028f4f3d03",
    "launcher": "ee4ea47352b4bb95ac0d72b6f71f9879bdb655c26fe02837d0519d011b607cc3",
    "predeclaration": "ca1bb1dcabf3b54052ff8d1defce78508c45d355d6901c672a62e889bccaf7eb",
    "postterminal_audit": "870157ef201f4517655c2f14b7637fabae9f13187bfb95308fffaa7aa20425ac",
    "science_manifest": "8272520263a3e5a6edc9164e6ea45821bb411b1ca9483a8e105231435116c784",
    "static_manifest": "253d5598b98f029a4bcc0eb5c8e6f23f88d56129490d0cd725fd5e60b5fcd45d",
    "wrapper_manifest": "147a0aa0bd1f83ff247a9fa36322c0b59f9eb4834cf6505d9c96e2fdf02494e3",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked capture scorecard and enforce immutable boundaries."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(scorecard.get("classification") == CLASSIFICATION, "classification changed")
    _require(
        scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR,
        "frozen denominator changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("correlation_used") is False
        and scorecard.get("cross_engine_parity_established") is False,
        "scope or metric policy changed",
    )
    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("slurm_job_id") == 11987097
        and contract.get("state") == "COMPLETED"
        and contract.get("exit_code") == "0:0"
        and contract.get("gpu_uuid") == "GPU-f3e94635-d095-bea9-dbe3-26e91dd3ea27"
        and contract.get("accepted") is True,
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
        tuple(case.get("class_one_based") for case in cases) == tuple(EXPECTED_CLASSES),
        "fixed class identity/order changed",
    )
    for case in cases:
        class_id = case["class_one_based"]
        rotations, active, significant, digest = EXPECTED_CLASSES[class_id]
        _require(
            case.get("id") == f"k4-it2-class{class_id}-recovar-boundary"
            and case.get("result") == "pass"
            and case.get("checked") is True
            and case.get("rotation_count") == rotations
            and case.get("active_candidate_count") == active
            and case.get("significant_candidate_count") == significant
            and case.get("artifact_sha256") == digest,
            f"class {class_id}: fixed capture result changed",
        )
    boundary = scorecard.get("boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("iteration") == 2
        and boundary.get("current_size") == 38
        and boundary.get("target_original_index_zero_based") == 53722
        and boundary.get("target_stack_index_one_based") == 53723
        and boundary.get("translations_per_rotation") == 116
        and boundary.get("joint_probability_mass") == 0.9999999999999997
        and boundary.get("joint_significant_support_mass") == 0.999000044117344
        and boundary.get("replayed_joint_log_normalizer") == -1.8714483132983646
        and boundary.get("probability_replay_max_abs") == 4.336808689942018e-19,
        "joint posterior boundary telemetry changed",
    )
    summary = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"pass": 4, "evaluated": 4}, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the checked all-class capture as a compact report."""

    summary = scorecard["summary"]
    boundary = scorecard["boundary"]
    lines = [
        "# K=4 RECOVAR all-class boundary capture scorecard",
        "",
        "This fixed-denominator panel records completion of the RECOVAR side of",
        "one K=4 class-pose boundary. It is non-scoring and does not establish",
        "cross-engine parity.",
        "",
        f"Captured classes: **{summary['pass']} / {scorecard['frozen_denominator']}**.",
        "",
        "| Checked | Class | Rotations | Active tuples | Significant tuples | Result |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in scorecard["cases"]:
        lines.append(
            f"| [x] | {case['class_one_based']} | {case['rotation_count']} | "
            f"{case['active_candidate_count']} | {case['significant_candidate_count']} | pass |"
        )
    lines.extend(
        [
            "",
            (
                f"The fixed iteration-{boundary['iteration']} target is stack "
                f"{boundary['target_stack_index_one_based']} at current size "
                f"{boundary['current_size']}; stored joint probabilities replay "
                f"within {boundary['probability_replay_max_abs']:.17g} maximum "
                "absolute error."
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
            "- `scripts/summarize_em_k4_allclass_boundary_capture_scorecard.py`",
            "- `scripts/report_em_parity_progress.py`",
            "",
            "To validate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k4_allclass_boundary_capture_scorecard.py --check",
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
