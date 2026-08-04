#!/usr/bin/env python3
"""Validate and render the fixed fresh-dispatch K=1 causal A/B panel."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = (
    REPO_ROOT / "docs" / "math" / "em_k1_fresh_dispatch_causal_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT / "docs" / "math" / "em_k1_fresh_dispatch_causal_scorecard.md"
)
SCHEMA = "recovar.em_k1_fresh_dispatch_causal_scorecard.v1"
SUITE_ID = "k1-fresh-physical-dispatch-order-case22-case26-ab"
CLASSIFICATION = (
    "fresh_dispatch_order_not_supported_as_standalone_fix_in_case22_or_case26"
)
CASE_IDS = ("k1-22", "k1-26")
EXPECTED_REPORTS = {
    "k1-22": {
        "job_id": 11928434,
        "state": "FAILED",
        "exit_code": "1:0",
        "gpu_uuid": "GPU-9f3ef2b2-6c59-a421-d927-697c58b38eb5",
        "control": 0.8260705143704488,
        "treatment": 0.8260839225848037,
        "delta": 1.3408214354915238e-05,
        "arm_fsc": 0.9999995598380529,
        "evidence": {
            "primary_report": "a6f74876a8c3249a1d604fa5a1f2261456a0c14cee89228f367aaa75de9a7476",
            "latent_report": "bc51728113bed57045e8071df9fcc0625376531f8163ff4c20db1b4ede1a2d1e",
            "order_transform": "79dc9bdbd1067494b709abe7cc788b048278159203b9296eae10a78a4f3f69e8",
            "predeclaration": "94f33f4f260b2cb1a88e63ab7b1e0beab819148a1a6d6c0a441e3f81f74bb99e",
            "static_inputs": "fa0f3b6cfd28d6f3322591c61616b20262cdc58cab6b66cb7e9212417ca6911c",
            "terminal_analysis_error": "afa2c8b25a39e13eec2dde7e135ca2ed495a40cff265fae7dbee540fe1168547",
        },
    },
    "k1-26": {
        "job_id": 11928437,
        "state": "FAILED",
        "exit_code": "2:0",
        "gpu_uuid": "GPU-e61dd352-2cd2-c1ae-ea85-6bfe203572cb",
        "control": 0.9633280568660826,
        "treatment": 0.9632747165209523,
        "delta": -5.334034513027053e-05,
        "arm_fsc": 0.9999917227463765,
        "evidence": {
            "primary_report": "e72342d504417e383c7c1c9209d203e0ccdfe3220babcab560dedf237c532336",
            "latent_report": "c7002cb8f214bb5f8dccc7ccc898a9ea113b221d735c168183ca1878dd304dab",
            "order_transform": "583d7ffca010fd0b9dd4ff57194144e0edf3ae5ca991a337d2867fa2d8f37535",
            "predeclaration": "750bf4ffe787bf594e9b496005b3464b4da21fc3968be82812151c892dba6ae9",
            "static_inputs": "7cfc02c9cfbfbed05b011a94cd0bf66f4ed13183a220a8f36176200dc03aac28",
            "science_completion": "6e3a71a111507a3c3a0fa15dd8c12aa5fc1c1090685879f34da36b0330649512",
        },
    },
}
SOURCE_COMMIT = "e2893cb3ef27061a77e39545f13edba202b952a1"
SOURCE_TREE = "ee01cf18b8b001bb293f45d034556ad304409afe"
LATENT_CLASSIFICATION = "treatment_has_mixed_but_net_latent_movement_toward_relion"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked panel and enforce its immutable negative result."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(
        scorecard.get("classification") == CLASSIFICATION,
        "classification changed",
    )
    _require(
        scorecard.get("frozen_denominator") == len(CASE_IDS),
        "frozen denominator changed",
    )
    _require(
        scorecard.get("correlation_used") is False
        and scorecard.get("fsc_auc_evaluated") is True
        and scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("production_order_fix_accepted") is False,
        "metric or production policy changed",
    )

    cases = scorecard.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == len(CASE_IDS),
        "cases do not preserve the frozen denominator",
    )
    _require(
        tuple(case.get("id") for case in cases) == CASE_IDS,
        "fixed case identity/order changed",
    )
    for case in cases:
        case_id = case["id"]
        expected = EXPECTED_REPORTS[case_id]
        producer = case.get("producer")
        _require(isinstance(producer, dict), f"{case_id}: producer record missing")
        _require(
            producer.get("slurm_job_id") == expected["job_id"]
            and producer.get("state") == expected["state"]
            and producer.get("exit_code") == expected["exit_code"]
            and producer.get("gpu_uuid") == expected["gpu_uuid"],
            f"{case_id}: producer identity changed",
        )
        _require(
            case.get("source_commit") == SOURCE_COMMIT
            and case.get("source_tree") == SOURCE_TREE,
            f"{case_id}: source identity changed",
        )
        _require(
            case.get("evaluation_result") == "complete"
            and case.get("standalone_rescue_result") == "fail"
            and case.get("causal_support") is False
            and case.get("checked") is True
            and case.get("latent_classification") == LATENT_CLASSIFICATION
            and case.get("latent_causal_acceptance") is False
            and case.get("transform_controls_all_pass") is True
            and case.get("grid_correction") == "unset/default-off"
            and case.get("forced_final_all_data_after_nonconvergence") is False,
            f"{case_id}: causal or policy result changed",
        )
        _require(
            case.get("control_final_merged_cross_fsc_auc") == expected["control"]
            and case.get("treatment_final_merged_cross_fsc_auc")
            == expected["treatment"]
            and case.get("treatment_minus_control_final_merged_cross_fsc_auc")
            == expected["delta"]
            and case.get("control_vs_treatment_final_merged_fsc_auc")
            == expected["arm_fsc"],
            f"{case_id}: fixed FSC-AUC result changed",
        )
        evidence = case.get("evidence")
        _require(
            isinstance(evidence, dict)
            and set(evidence) == set(expected["evidence"]),
            f"{case_id}: evidence set changed",
        )
        for name, expected_sha in expected["evidence"].items():
            record = evidence[name]
            evidence_path = record.get("path")
            digest = record.get("sha256")
            _require(
                isinstance(evidence_path, str) and Path(evidence_path).is_absolute(),
                f"{case_id}/{name}: evidence path must be absolute",
            )
            _require(
                isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
                f"{case_id}/{name}: invalid SHA-256",
            )
            _require(
                digest == expected_sha,
                f"{case_id}/{name}: evidence SHA-256 changed",
            )

    case22, case26 = cases
    _require(
        case22.get("material_fsc_auc_delta_min") == 0.0001
        and case22.get("materially_improved") is False
        and case22.get("control_reproduces_strict_failure") is True
        and case22.get("control_reproduces_topology_failure") is True
        and case22.get("treatment_passes_strict_fsc") is False
        and case22.get("treatment_passes_topology") is False,
        "k1-22 fixed causal gates changed",
    )
    _require(
        case26.get("strictly_improved") is False
        and case26.get("control_reproduces_strict_failure") is True
        and case26.get("control_preserves_topology") is True
        and case26.get("treatment_passes_strict_fsc") is False
        and case26.get("treatment_preserves_topology") is True,
        "k1-26 fixed causal gates changed",
    )

    summary = {
        "evaluations_complete": sum(
            case["evaluation_result"] == "complete" for case in cases
        ),
        "standalone_rescues": sum(
            case["standalone_rescue_result"] == "pass" for case in cases
        ),
        "not_supported": sum(not case["causal_support"] for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(
        summary
        == {
            "evaluations_complete": 2,
            "standalone_rescues": 0,
            "not_supported": 2,
            "evaluated": 2,
        },
        "fixed dispatch causal summary changed",
    )
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the checked fixed-GPU A/B conclusions."""

    summary = scorecard["summary"]
    lines = [
        "# K=1 fresh physical-dispatch causal scorecard",
        "",
        "This fixed-denominator diagnostic distinguishes completed causal",
        "evaluations from successful treatment rescues. It is non-scoring.",
        "Map gates use signed FSC/FSC-AUC; correlation is forbidden.",
        "",
        f"Evaluated: **{summary['evaluations_complete']} / {scorecard['frozen_denominator']}**.",
        f"Standalone rescues: **{summary['standalone_rescues']} / {scorecard['frozen_denominator']}**.",
        "",
        "| Checked | Case | Control final FSC-AUC | Dispatch final FSC-AUC | Delta | Strict rescue | Topology | Conclusion |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        topology = (
            "fail"
            if case["id"] == "k1-22"
            else ("pass" if case["treatment_preserves_topology"] else "fail")
        )
        lines.append(
            f"| {check} | `{case['id']}` | "
            f"{case['control_final_merged_cross_fsc_auc']:.9f} | "
            f"{case['treatment_final_merged_cross_fsc_auc']:.9f} | "
            f"{case['treatment_minus_control_final_merged_cross_fsc_auc']:+.9f} | "
            f"{case['standalone_rescue_result']} | {topology} | not supported |"
        )
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            "Both A/Bs verified the order transform and retained mixed latent",
            "movement, but neither closed the fixed strict FSC gate. Case 22",
            "also retained its topology failure; case 26's final FSC-AUC",
            "decreased. Particle order remains a structural invariant and",
            "possible mediator, not an accepted standalone production fix.",
            "",
            "The producer Slurm states are preserved exactly: case 22 ended",
            "after a post-science analysis exception and was reanalyzed from",
            "sealed arm outputs; case 26 intentionally exited nonzero after",
            "recording a complete negative causal result.",
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k1_fresh_dispatch_causal_scorecard.py --check",
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
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
