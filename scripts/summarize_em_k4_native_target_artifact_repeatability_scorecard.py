#!/usr/bin/env python3
"""Validate and render the fixed native K=4 target-artifact panel."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_k4_native_target_artifact_repeatability_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "math" / "em_k4_native_target_artifact_repeatability_scorecard.md"
SCHEMA = "recovar.em_k4_native_target_artifact_repeatability_scorecard.v1"
SUITE_ID = "k4-it2-native-classes2-4-target-artifact-repeatability"
CLASSIFICATION = "accepted_target_artifact_repeatability"
FROZEN_DENOMINATOR = 32
PER_CLASS_GATES = (
    "artifact_validators_passed",
    "bpref_bytes_exact",
    "dispatch_bytes_exact",
    "dispatch_row_count_exact",
    "fine_score_bytes_exact",
    "hard_pose_class_shift_exact",
    "map_fsc_auc_at_least_threshold",
    "runtime_replay_and_no_fatal_exact",
    "target_state_exact",
    "topology_exact",
)
EXPECTED_CASES = tuple(
    f"class{class_one_based}_{gate}" for class_one_based in (2, 3, 4) for gate in PER_CLASS_GATES
) + ("immutable_static_provenance_exact", "target_gpu_exact")
EXPECTED_EVIDENCE_SHA256 = {
    "repeatability_report": "da59157b92956fca4095b87d2dce850cc53d1e21e4e3321474d12bd651f3c4b8",
    "analysis_completion": "acd30c07408c1475ced6eac107ac04abf3c8ccb1feea55d78fff5c645ffbc088",
    "launcher": "363304a62ab0077eef1aae97567650ad50a4c8313c0beb8182aace4f70664177",
    "analyzer": "fb7b65c3579c66288ad78e091a1be99618ee847aa3e775f8a553e89421947ed2",
    "predeclaration": "b8a47b6ff22c24c9df75d23200ae46579ae75e8994baf47e3c11f03b5e638bc8",
    "submission": "1274fd53e7c40904b7dd1bbd743d10f4098d5ad6a38769927e7c730bdfdbc432",
    "science_manifest": "f582f642c3ccdae0251ec21eb548e44d440bb89c6b20789765427c82fcd81be0",
    "static_manifest": "5812d510f8c158a524d826f4914ffb3223644c2a927636d5e80fbedbd5952d0c",
    "science_stdout": "51bf46819f0505854adfe2ce2c3a2a4e10c687c2081789e8ed85f62500dc4138",
    "science_stderr": "933fae46b1152e45cbeabdd939d7ccb7e4ed5f11cb762253aab15f50dbeb333d",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the scorecard and enforce its fixed target-local evidence."""

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
        and scorecard.get("target_local_artifact_use_allowed") is True
        and scorecard.get("allclass_cross_engine_attribution_allowed") is False,
        "scope or metric policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("slurm_job_id") == 11996846
        and contract.get("state") == "COMPLETED"
        and contract.get("exit_code") == "0:0"
        and contract.get("elapsed") == "00:25:32"
        and contract.get("node") == "della-l07g2"
        and contract.get("gpu_uuid") == "GPU-f3e94635-d095-bea9-dbe3-26e91dd3ea27"
        and contract.get("native_source_commit") == "17a97690c79d28f0e413fca6540c63f944e22868"
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
        and boundary.get("classes_one_based") == [2, 3, 4]
        and boundary.get("map_fsc_auc_threshold") == 0.999999
        and boundary.get("minimum_signed_normalized_non_dc_fsc_auc") == 0.9999999794616498,
        "target boundary changed",
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
    _require(summary == {"pass": 32, "evaluated": 32}, "fixed result changed")

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
    """Render the fixed target-artifact panel as a checked table."""

    summary = scorecard["summary"]
    boundary = scorecard["boundary"]
    lines = [
        "# K=4 native target-artifact repeatability scorecard",
        "",
        "This fixed-denominator panel admits native classes 2--4 artifacts for",
        "target-local analysis only. Broad all-class attribution remains prohibited.",
        "",
        f"Fixed admission gates: **{summary['pass']} / {scorecard['frozen_denominator']}**.",
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
                "Minimum signed normalized non-DC FSC-AUC over the 12 preserved "
                f"class-map comparisons: `{boundary['minimum_signed_normalized_non_dc_fsc_auc']}` "
                f"(threshold `{boundary['map_fsc_auc_threshold']}`)."
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
            "To validate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k4_native_target_artifact_repeatability_scorecard.py --check",
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
