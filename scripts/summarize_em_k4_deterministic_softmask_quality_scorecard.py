#!/usr/bin/env python3
"""Validate and render the fixed K=4 deterministic soft-mask quality A/B."""

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
    / "em_k4_deterministic_softmask_quality_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT
    / "docs"
    / "math"
    / "em_k4_deterministic_softmask_quality_scorecard.md"
)
SCHEMA = "recovar.em_k4_deterministic_softmask_quality_scorecard.v1"
SUITE_ID = "k4-softmask2-full15-quality-ab"
CLASSIFICATION = "quality_and_topology_preserved__production_integration_accepted"
SOURCE_COMMIT = "e98a5f333cc789f1e2511da58b95b974c6fe6636"
FROZEN_DENOMINATOR = 7
CASE_IDS = (
    "direct-pass-count-not-lower",
    "all-class-iteration-count-not-lower",
    "gt-delta-pass-count-not-lower",
    "class-agreement-pass-count-not-lower",
    "direct-nondegradation-60-of-60",
    "gt-nondegradation-60-of-60",
    "cohort-and-provenance-4-of-4",
)
EXPECTED_EVIDENCE_SHA256 = {
    "quality_report": "bf3ff3d6c3087c0bedcc62c48ff9e5eb7af95472c60f60d65fbcae65a68b6aa5",
    "shellwise_repeatability": "0cc13b667cca88276cdeb4cc1e59e42d26c71b059a9ff972a20004bfe5c816bd",
    "control_topology": "23cac9f6459aafb4a93912ba9750676fa97bf79c6d72ef48d0d54d9243afa0b7",
    "treatment_topology": "5d3f138a1e6b5ae7fb942cb9b78c09bbc4d5309334cb1552c0402c12ba3c2409",
    "analysis_complete": "ac3c72ee9cb139169ff96d71e2c6b4ca5fcfdd3cab3750940f690e235908a6d8",
    "analysis_manifest": "ded291d723efb10ba979770181f1aef6974d2a204f2cf3efc94f830a6ad39f98",
    "science_complete": "a42de5ae86f0c090e321f1dcaf3549cb18c849c516a814d9c8af0fe68f8f61a3",
    "recovered_walltimes": "fb7a7434df9ccc2e94334383b1fd56b9220f4f3e60b46db1410aab22977a1ee3",
    "science_predeclaration": "5eacf1f27b194b11a6c11109705d08f6be7975adfdba3ce7dfa7fd0a00e11af1",
    "analysis_predeclaration": "371a5b6bd0db38cdc0713e6e95cfda18dfa415f0c2918a4ef85b6ec58d226b4f",
    "analysis_launcher": "12537369288f5787c3566c975daa1b59ec5d488c10519c80de434c634ebbc4a6",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked A/B scorecard and enforce its frozen identity."""

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
        and scorecard.get("fsc_auc_evaluated") is True,
        "quality metric policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("science_job_id") == 11854692
        and contract.get("science_job_state") == "FAILED_POST_ARM_LAUNCHER_HASH_STEP"
        and contract.get("science_arms_completed") is True
        and contract.get("analysis_job_id") == 11893992
        and contract.get("analysis_job_state") == "COMPLETED"
        and contract.get("analysis_exit_code") == "0:0"
        and contract.get("source_commit") == SOURCE_COMMIT
        and contract.get("quality_accepted") is True
        and contract.get("topology_accepted") is True
        and contract.get("independent_repeatability_accepted") is True
        and contract.get("production_integration_accepted") is True
        and contract.get("same_physical_gpu") is True
        and contract.get("grid_correction") == "unset/default-off"
        and contract.get("forced_final_all_data_after_nonconvergence") is False,
        "terminal or integration acceptance contract changed",
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

    arms = scorecard.get("arms")
    _require(isinstance(arms, dict) and set(arms) == {"control", "treatment"}, "arm identity changed")
    for arm in arms.values():
        _require(
            arm.get("direct_fsc_auc") == {"passed": 41, "evaluated": 60}
            and arm.get("all_class_iterations") == {"passed": 9, "evaluated": 15}
            and arm.get("gt_delta") == {"passed": 60, "evaluated": 60}
            and arm.get("class_agreement") == {"passed": 15, "evaluated": 15},
            "fixed arm quality counts changed",
        )
    _require(
        arms["control"].get("minimum_direct_fsc_auc") == 0.9909648289183515
        and arms["treatment"].get("minimum_direct_fsc_auc") == 0.9905422936627672
        and arms["control"].get("minimum_gt_delta") == -0.0003450886080695881
        and arms["treatment"].get("minimum_gt_delta") == -0.0004546970772076331
        and arms["control"].get("minimum_class_agreement") == 0.99276
        and arms["treatment"].get("minimum_class_agreement") == 0.99203,
        "fixed arm minima changed",
    )

    paired = scorecard.get("treatment_minus_control")
    _require(
        isinstance(paired, dict)
        and paired.get("direct_fsc_auc")
        == {
            "passed": 60,
            "evaluated": 60,
            "minimum": -0.0007612990230934091,
            "mean": -2.924142069163545e-05,
        }
        and paired.get("recovar_gt_fsc_auc")
        == {
            "passed": 60,
            "evaluated": 60,
            "minimum": -0.000109608469138045,
            "mean": -3.307056120961891e-06,
        },
        "fixed paired nondegradation result changed",
    )

    performance = scorecard.get("performance_observation")
    _require(
        performance
        == {
            "control_wall_s": 27659,
            "treatment_wall_s": 27684,
            "treatment_minus_control_percent": 0.0903864926425395,
            "formal_acceptance_gate": False,
        },
        "fixed performance observation changed",
    )

    cases = scorecard.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == FROZEN_DENOMINATOR,
        "cases do not preserve the frozen denominator",
    )
    _require(tuple(case.get("id") for case in cases) == CASE_IDS, "fixed case identity/order changed")
    _require(
        all(case.get("result") == "pass" and case.get("checked") is True for case in cases),
        "quality acceptance result changed",
    )
    summary = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"pass": 7, "evaluated": 7}, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the fixed quality acceptance as a compact checked table."""

    summary = scorecard["summary"]
    control = scorecard["arms"]["control"]
    treatment = scorecard["arms"]["treatment"]
    paired = scorecard["treatment_minus_control"]
    perf = scorecard["performance_observation"]
    lines = [
        "# K=4 deterministic soft-mask quality A/B scorecard",
        "",
        "This fixed-denominator panel records the predeclared full-trajectory",
        "quality and provenance gates for the deterministic CUDA soft-mask reduction.",
        "The published K=4 score remains unchanged.",
        "",
        f"Quality acceptance: **{summary['pass']} / {scorecard['frozen_denominator']}**.",
        "",
        "| Checked | Gate | Result | Observation |",
        "| --- | --- | --- | --- |",
    ]
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(
            f"| {check} | `{case['id']}` | {case['result']} | {case['observed']} |"
        )
    lines.extend(
        [
            "",
            "| Fixed quality panel | Control | Treatment | Change |",
            "| --- | ---: | ---: | ---: |",
            (
                "| direct per-class FSC-AUC | "
                f"{control['direct_fsc_auc']['passed']}/{control['direct_fsc_auc']['evaluated']} | "
                f"{treatment['direct_fsc_auc']['passed']}/{treatment['direct_fsc_auc']['evaluated']} | 0 |"
            ),
            (
                "| all-class iterations | "
                f"{control['all_class_iterations']['passed']}/{control['all_class_iterations']['evaluated']} | "
                f"{treatment['all_class_iterations']['passed']}/{treatment['all_class_iterations']['evaluated']} | 0 |"
            ),
            (
                "| GT-delta panels | "
                f"{control['gt_delta']['passed']}/{control['gt_delta']['evaluated']} | "
                f"{treatment['gt_delta']['passed']}/{treatment['gt_delta']['evaluated']} | 0 |"
            ),
            (
                "| class-agreement iterations | "
                f"{control['class_agreement']['passed']}/{control['class_agreement']['evaluated']} | "
                f"{treatment['class_agreement']['passed']}/{treatment['class_agreement']['evaluated']} | 0 |"
            ),
            "",
            (
                "All treatment-minus-control nondegradation panels pass: "
                f"direct **{paired['direct_fsc_auc']['passed']}/"
                f"{paired['direct_fsc_auc']['evaluated']}** (minimum "
                f"`{paired['direct_fsc_auc']['minimum']:.12g}`), GT **"
                f"{paired['recovar_gt_fsc_auc']['passed']}/"
                f"{paired['recovar_gt_fsc_auc']['evaluated']}** (minimum "
                f"`{paired['recovar_gt_fsc_auc']['minimum']:.12g}`)."
            ),
            "",
            (
                "Observed whole-arm wall time is "
                f"`{perf['control_wall_s']} s -> {perf['treatment_wall_s']} s` "
                f"(`{perf['treatment_minus_control_percent']:+.4f}%`); this was "
                "recorded but was not one of the seven formal acceptance gates."
            ),
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            "Both independent FSC/topology analyses, shellwise arrays, and pair",
            "reports reproduce exactly. Correlation is not computed.",
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            (
                "pixi run python scripts/"
                "summarize_em_k4_deterministic_softmask_quality_scorecard.py --check"
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
