#!/usr/bin/env python3
"""Validate and render the fixed K=1 shared-checkpoint FP64 scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs/math/em_k1_shared_checkpoint_fp64_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs/math/em_k1_shared_checkpoint_fp64_scorecard.md"
SCHEMA = "recovar.em_k1_shared_checkpoint_fp64_scorecard.v1"
SUITE_ID = "k1-case22-shared-checkpoint-fp64-reference"
CLASSIFICATION = "shared_checkpoint_fp64_reference_rejected"
FROZEN_DENOMINATOR = 34
EXPECTED_SUMMARY = {"pass": 16, "evaluated": 34}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_EVIDENCE_SHA256 = {
    "fixed_audit": "84bc7606ab570fccaebc1e0549ac2dda5a4a7a0c70f43eca8be41eef721faf25",
    "audit_complete": "0185283f0a6719160ca6dc86a78fe60b967c5926971a2c75b257ee057bb2205a",
    "compute_complete": "e9e9bdb5ed2dced986614824dbf3c3bdd6707f6ed21859453ae6161b56331576",
    "snapshot_compute_audit": "4e3c762d68680187c1549ae42474145f59b5b0f5c22268eeb074a95416a659e2",
    "predeclaration": "560ae299790fa22d55e2f121fbe6615e6fedf06d2332efbb1aa138ba7c85896d",
    "audit_declaration": "1054da001071054b76226dac7c3eb8a630b227d91a816d07f053f43b1695452b",
}
EXPECTED_CASE_RESULTS = (
    ("snapshot-half1-float32-roundtrip", "pass"),
    ("snapshot-half2-float32-roundtrip", "pass"),
    ("component-structural-normal_a", "pass"),
    ("component-structural-normal_b", "pass"),
    ("component-structural-fp64_reference", "pass"),
    ("geometry-raw_input", "pass"),
    ("geometry-rotation_keys", "pass"),
    ("geometry-local_rotation_indices", "pass"),
    ("geometry-euler_matrices", "pass"),
    ("geometry-translation_values", "pass"),
    ("preprocess-norm-correction", "pass"),
    ("preprocess-masked-real", "pass"),
    ("preprocess-masked-fourier-pre-optics", "pass"),
    ("preprocess-masked-fourier-post-optics", "pass"),
    *(
        (f"score-beyond-floor-stack-{stack}", "fail")
        for stack in (
            "0035",
            "0252",
            "0348",
            "0591",
            "0683",
            "1100",
            "1522",
            "1640",
            "1767",
            "2124",
            "2322",
            "2330",
            "2846",
            "2994",
        )
    ),
    ("map-parity-beyond-floor-half1", "fail"),
    ("map-gt-nondegradation-half1", "pass"),
    ("map-parity-beyond-floor-half2", "fail"),
    ("map-gt-nondegradation-half2", "fail"),
    ("map-parity-beyond-floor-merged", "fail"),
    ("map-gt-nondegradation-merged", "pass"),
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
    _require(scorecard.get("correlation_used") is False, "correlation policy changed")
    _require(scorecard.get("scorecard_change_admissible") is False, "scorecard policy changed")
    contract = scorecard.get("acceptance_contract", {})
    _require(
        contract.get("compute_job_id") == 11850133
        and contract.get("compute_state") == "COMPLETED"
        and contract.get("compute_exit_code") == "0:0"
        and contract.get("accepted") is False,
        "terminal acceptance contract changed",
    )
    evidence = scorecard.get("evidence", {})
    _require(set(evidence) == set(EXPECTED_EVIDENCE_SHA256), "evidence set changed")
    for name, record in evidence.items():
        _require(Path(record["path"]).is_absolute(), f"{name}: path is not absolute")
        _require(SHA256_RE.fullmatch(record["sha256"]) is not None, f"{name}: invalid SHA-256")
        _require(record["sha256"] == EXPECTED_EVIDENCE_SHA256[name], f"{name}: evidence digest changed")
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
    _require(summary == EXPECTED_SUMMARY and scorecard.get("summary") == summary, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    lines = [
        "# K=1 shared-checkpoint binary64-reference scorecard",
        "",
        "This fixed-denominator control-floor diagnostic is non-scoring.",
        "Map acceptance uses FSC/FSC-AUC; correlation is forbidden.",
        "",
        f"Accepted gates: **{scorecard['summary']['pass']} / {scorecard['frozen_denominator']}**.",
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
            f"Live-reference-dominated score cases: **{observations['score_live_reference_dominated']}**.",
            f"Parity FSC-AUC gains beyond the control floor: **{observations['map_parity_beyond_control_floor']}**.",
            f"GT FSC-AUC nondegradation: **{observations['map_gt_nondegraded']}**.",
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k1_shared_checkpoint_fp64_scorecard.py --check",
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
