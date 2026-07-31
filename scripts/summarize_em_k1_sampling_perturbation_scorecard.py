#!/usr/bin/env python3
"""Validate and render the fixed K=1 sampling-perturbation A/B scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_k1_sampling_perturbation_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "math" / "em_k1_sampling_perturbation_scorecard.md"
SCHEMA = "recovar.em_k1_sampling_perturbation_scorecard.v1"
SUITE_ID = "k1-case22-sameallocation-continuation-sampling-perturbation-ab"
STACK_IDS = (35, 252, 348, 591, 683, 1100, 1522, 1640, 1767, 2124, 2322, 2330, 2846, 2994)
MAP_LABELS = ("half1", "half2", "merged")
SCORE_MAP_CASE_IDS = (
    *(f"sampling-perturb-score-stack-{stack_id:04d}" for stack_id in STACK_IDS),
    *(f"sampling-perturb-map-parity-{label}" for label in MAP_LABELS),
    *(f"sampling-perturb-map-gt-{label}" for label in MAP_LABELS),
    "sampling-perturb-overall",
)
GEOMETRY_CASE_IDS = (
    "sampling-perturb-geometry-raw-input",
    "sampling-perturb-geometry-rotation-keys",
    "sampling-perturb-geometry-local-rotation-indices",
    "sampling-perturb-geometry-euler-matrices",
    "sampling-perturb-geometry-translations",
)
SCORE_MAP_GROUP_DENOMINATORS = {
    "score": 14,
    "map-parity": 3,
    "map-gt": 3,
    "overall-arm": 1,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_CLASSIFICATION = (
    "sampling_perturbation_restore_collapses_geometry_score_and_map_gaps_but_"
    "star_precision_prevents_identity_and_signed_gate_closure"
)
EXPECTED_EVIDENCE_SHA256 = {
    "fail_closed_result": "408152e7f24842406988ad29ee207e54c219415a934f8965dd935339beb56c1e",
    "baseline_operand": "520c4eaba86f0f9641963420433463084f8e78e7e59dee52bb106241e48e3aed",
    "fresh_arm_complete": "4bd4c2744a892b26a8bced026cf199b99075a04cbecbbd7bc1294c2f459b7443",
    "stock_arm_complete": "5907fd45036ccd8ff2a6f2be9a2ce81ff0510f1ec9f04aa899b8f2566e626917",
    "treatment_arm_complete": "43e33bfccb03a4f8e563fa7ef6bf080ec10b2c363d6e956f474ab6122db3ad97",
    "treatment_operand": "34fae25a8f921890a9052984e8c0712b7a2c9355424b4b7756829ede857f3768",
    "stock_score": "92f4e753cc055e25c02c782d409c1730c1c9ec5423609abcfcc0b8d18c77d495",
    "treatment_score": "2c8459682de3f91149c429f839d569f737bb07c89406856924bb6040d8674b06",
    "stock_map": "1268660a21693fb664f3e8f56f8cca55ae464bbf6dca06c7025fd1e7f617481d",
    "treatment_map": "96362f26457a154c515f25730ad97b8db9eeb12914e3cf2f4df97ea56432e6b3",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _transition(baseline: str, treatment: str) -> str:
    if baseline == "fail" and treatment == "pass":
        return "improved"
    if baseline == "pass" and treatment == "pass":
        return "retained"
    if baseline == "pass" and treatment == "fail":
        return "regressed"
    return "unchanged-fail"


def _validate_cases(
    cases: object,
    expected_ids: tuple[str, ...],
    expected_pairs: tuple[tuple[str, str], ...],
) -> tuple[dict[str, int], dict[str, int]]:
    _require(isinstance(cases, list) and len(cases) == len(expected_ids), "case denominator changed")
    _require(tuple(case.get("id") for case in cases) == expected_ids, "case identity/order changed")
    baseline_pass = 0
    treatment_pass = 0
    transitions = {name: 0 for name in ("improved", "retained", "regressed", "unchanged-fail")}
    for case, expected_pair in zip(cases, expected_pairs, strict=True):
        pair = (case.get("baseline_result"), case.get("treatment_result"))
        _require(pair == expected_pair, f"{case.get('id')}: fixed result pair changed")
        _require(case.get("result") == pair[1], f"{case.get('id')}: result is not treatment result")
        _require(case.get("checked") is (pair[1] == "pass"), f"{case.get('id')}: checkmark changed")
        transition = _transition(*pair)
        _require(case.get("transition") == transition, f"{case.get('id')}: transition changed")
        _require(isinstance(case.get("name"), str) and case["name"], f"{case.get('id')}: missing name")
        _require(
            isinstance(case.get("observed"), str) and case["observed"],
            f"{case.get('id')}: missing observation",
        )
        baseline_pass += pair[0] == "pass"
        treatment_pass += pair[1] == "pass"
        transitions[transition] += 1
    summary = {
        "baseline_pass": baseline_pass,
        "treatment_pass": treatment_pass,
        "evaluated": len(expected_ids),
        "denominator": len(expected_ids),
        "paired_gain": treatment_pass - baseline_pass,
    }
    return summary, transitions


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce both fixed paired denominators."""

    scorecard = json.loads(Path(path).read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(scorecard.get("classification") == EXPECTED_CLASSIFICATION, "classification changed")
    _require(scorecard.get("correlation_used") is False, "correlation is forbidden")
    _require(scorecard.get("scorecard_change_admissible") is False, "causal panel cannot change quality scores")
    _require(
        scorecard.get("score_map_group_denominators") == SCORE_MAP_GROUP_DENOMINATORS,
        "score/map group denominators changed",
    )
    contract = scorecard.get("acceptance_contract", {})
    _require(
        contract
        == {
            "science_job_id": 11842188,
            "science_job_terminal_state": "FAILED",
            "science_job_exit_code": "1:0",
            "same_physical_gpu": True,
            "gpu_uuid": "GPU-eb1c5b04-20c1-b6c9-16e6-b3dc87905bd7",
            "all_arms_complete": True,
            "post_terminal_analysis_complete": True,
            "predeclared_hypothesis_accepted": False,
            "grid_correction": "unset/default-off",
            "forced_final_all_data_after_nonconvergence": False,
        },
        "acceptance contract changed",
    )

    evidence = scorecard.get("evidence")
    _require(isinstance(evidence, dict) and set(evidence) == set(EXPECTED_EVIDENCE_SHA256), "evidence changed")
    for name, expected_digest in EXPECTED_EVIDENCE_SHA256.items():
        record = evidence[name]
        _require(isinstance(record, dict), f"{name}: invalid evidence record")
        _require(Path(record.get("path", "")).is_absolute(), f"{name}: evidence path is not absolute")
        digest = record.get("sha256")
        _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, f"{name}: invalid SHA-256")
        _require(digest == expected_digest, f"{name}: evidence SHA-256 changed")

    score_pairs = (
        *(("fail", "fail") for _ in STACK_IDS),
        *(("fail", "fail") for _ in MAP_LABELS),
        *(("pass", "fail") for _ in MAP_LABELS),
        ("fail", "fail"),
    )
    score_summary, score_transitions = _validate_cases(
        scorecard.get("score_map_cases"),
        SCORE_MAP_CASE_IDS,
        score_pairs,
    )
    geometry_pairs = (
        ("pass", "pass"),
        ("pass", "pass"),
        ("pass", "pass"),
        ("fail", "fail"),
        ("fail", "fail"),
    )
    geometry_summary, geometry_transitions = _validate_cases(
        scorecard.get("geometry_cases"),
        GEOMETRY_CASE_IDS,
        geometry_pairs,
    )
    _require(scorecard.get("score_map_summary") == score_summary, "score/map summary does not replay cases")
    _require(scorecard.get("score_map_transitions") == score_transitions, "score/map transitions changed")
    _require(scorecard.get("geometry_summary") == geometry_summary, "geometry summary does not replay cases")
    _require(scorecard.get("geometry_transitions") == geometry_transitions, "geometry transitions changed")
    _require(
        score_summary
        == {"baseline_pass": 3, "treatment_pass": 0, "evaluated": 21, "denominator": 21, "paired_gain": -3},
        "fixed score/map result changed",
    )
    _require(
        geometry_summary
        == {"baseline_pass": 3, "treatment_pass": 3, "evaluated": 5, "denominator": 5, "paired_gain": 0},
        "fixed geometry result changed",
    )
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render paired metrics and checked cases."""

    score = scorecard["score_map_summary"]
    geometry = scorecard["geometry_summary"]
    lines = [
        "# K=1 continuation sampling-perturbation A/B scorecard",
        "",
        "This fixed-denominator causal diagnostic is non-scoring.",
        "",
        "| Fixed paired panel | Stock | Treatment | Gain |",
        "| --- | ---: | ---: | ---: |",
        f"| Geometry identity | {geometry['baseline_pass']}/5 | {geometry['treatment_pass']}/5 | {geometry['paired_gain']:+d} |",
        f"| Score/map gates | {score['baseline_pass']}/21 | {score['treatment_pass']}/21 | {score['paired_gain']:+d} |",
        "",
        "| Checked | Geometry case | Stock | Treatment | Transition | Observation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in scorecard["geometry_cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(
            f"| {check} | `{case['id']}` | {case['baseline_result']} | "
            f"{case['treatment_result']} | {case['transition']} | {case['observed']} |"
        )
    lines.extend(
        [
            "",
            "| Checked | Score/map case | Stock | Treatment | Transition | Observation |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in scorecard["score_map_cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(
            f"| {check} | `{case['id']}` | {case['baseline_result']} | "
            f"{case['treatment_result']} | {case['transition']} | {case['observed']} |"
        )
    lines.extend(["", f"Classification: `{scorecard['classification']}`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_markdown(load_and_validate(args.scorecard))
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            raise SystemExit(f"stale generated scorecard: {args.output}")
    else:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
