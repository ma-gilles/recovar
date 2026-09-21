#!/usr/bin/env python3
"""Validate and render the fixed K=1 normalization-roundtrip scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_k1_norm_roundtrip_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "math" / "em_k1_norm_roundtrip_scorecard.md"
SCHEMA = "recovar.em_k1_norm_roundtrip_scorecard.v1"
SUITE_ID = "k1-case22-sameallocation-continuation-normalization-roundtrip-ab"
STACK_IDS = (35, 252, 348, 591, 683, 1100, 1522, 1640, 1767, 2124, 2322, 2330, 2846, 2994)
MAP_LABELS = ("half1", "half2", "merged")
PREPROCESS_CASE_IDS = (
    "norm-roundtrip-preprocess-correction",
    "norm-roundtrip-preprocess-normalized-shifted-real",
)
GEOMETRY_CASE_IDS = (
    "norm-roundtrip-geometry-raw-input",
    "norm-roundtrip-geometry-rotation-keys",
    "norm-roundtrip-geometry-local-rotation-indices",
    "norm-roundtrip-geometry-euler-matrices",
    "norm-roundtrip-geometry-translations",
)
SCORE_MAP_CASE_IDS = (
    *(f"norm-roundtrip-score-stack-{stack_id:04d}" for stack_id in STACK_IDS),
    *(f"norm-roundtrip-map-parity-{label}" for label in MAP_LABELS),
    *(f"norm-roundtrip-map-gt-{label}" for label in MAP_LABELS),
    "norm-roundtrip-overall",
)
SCORE_MAP_GROUP_DENOMINATORS = {
    "score": 14,
    "map-parity": 3,
    "map-gt": 3,
    "overall-arm": 1,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_CLASSIFICATION = "normalization_roundtrip_closes_preprocessing_exactness_but_score_map_panel_is_retained"
EXPECTED_EVIDENCE_SHA256 = {
    "sampling_baseline_terminal": "22fefd3a3597403c747ba832493792a405634fa0cddeddc773295e79f798edee",
    "normalization_terminal": "d667f6d40ec384a8f3e0294d15ae7471f711e6c64fa9ac22ed188148241f1bbd",
    "fresh_arm_complete": "4bd4c2744a892b26a8bced026cf199b99075a04cbecbbd7bc1294c2f459b7443",
    "stock_arm_complete": "5907fd45036ccd8ff2a6f2be9a2ce81ff0510f1ec9f04aa899b8f2566e626917",
    "treatment_arm_complete": "24ad391a53c613ae6bbf156a45dcce9636249b1f79f8707542565d7119254bc5",
    "treatment_operand": "b9824c7142c73d767896528e95abc9478aeda7738ac04b0f23d42f2239aa030d",
    "stock_score": "fb8d0481791befb9eb249972e7dab00833af2c831be96770d053c68d386971b3",
    "treatment_score": "86ecd8727e3a32f5611aa3867e1a08f2ea0b0fce894659a59b9b3f696502eb5e",
    "stock_map": "26fc06d76330fdb313cffe0e9723b645ffd2ec0f42ad1c323e1ff6c95cf26267",
    "treatment_map": "f9a0e17982715ce93b4ddceb311d54ae73a5fccc241ef66a6a059d5f5bf9158f",
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
    return (
        {
            "baseline_pass": baseline_pass,
            "treatment_pass": treatment_pass,
            "evaluated": len(expected_ids),
            "denominator": len(expected_ids),
            "paired_gain": treatment_pass - baseline_pass,
        },
        transitions,
    )


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce all fixed denominators."""

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
    _require(
        scorecard.get("acceptance_contract")
        == {
            "science_job_id": 11845465,
            "science_job_terminal_state": "COMPLETED",
            "science_job_exit_code": "0:0",
            "same_physical_gpu": True,
            "gpu_uuid": "GPU-a1de512c-f178-a5e1-6c95-c54c6d07c9f3",
            "all_arms_complete": True,
            "predeclared_primary_hypothesis_accepted": True,
            "launcher_gate_valid": True,
            "independent_replay_complete": True,
            "score_map_metric_observational": True,
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

    preprocess_summary, preprocess_transitions = _validate_cases(
        scorecard.get("preprocess_cases"),
        PREPROCESS_CASE_IDS,
        (("fail", "pass"), ("fail", "pass")),
    )
    geometry_summary, geometry_transitions = _validate_cases(
        scorecard.get("geometry_cases"),
        GEOMETRY_CASE_IDS,
        tuple(("pass", "pass") for _ in GEOMETRY_CASE_IDS),
    )
    score_pairs = (
        *(("pass", "pass") for _ in STACK_IDS),
        *(("pass", "pass") for _ in MAP_LABELS),
        *(("fail", "fail") for _ in MAP_LABELS),
        ("fail", "fail"),
    )
    score_summary, score_transitions = _validate_cases(
        scorecard.get("score_map_cases"),
        SCORE_MAP_CASE_IDS,
        score_pairs,
    )
    for name, summary, transitions in (
        ("preprocess", preprocess_summary, preprocess_transitions),
        ("geometry", geometry_summary, geometry_transitions),
        ("score_map", score_summary, score_transitions),
    ):
        _require(scorecard.get(f"{name}_summary") == summary, f"{name} summary does not replay cases")
        _require(scorecard.get(f"{name}_transitions") == transitions, f"{name} transitions changed")
    _require(
        preprocess_summary
        == {"baseline_pass": 0, "treatment_pass": 2, "evaluated": 2, "denominator": 2, "paired_gain": 2},
        "fixed preprocessing result changed",
    )
    _require(
        geometry_summary
        == {"baseline_pass": 5, "treatment_pass": 5, "evaluated": 5, "denominator": 5, "paired_gain": 0},
        "fixed geometry result changed",
    )
    _require(
        score_summary
        == {"baseline_pass": 17, "treatment_pass": 17, "evaluated": 21, "denominator": 21, "paired_gain": 0},
        "fixed score/map result changed",
    )
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render paired metrics and checked cases."""

    summaries = {name: scorecard[f"{name}_summary"] for name in ("preprocess", "geometry", "score_map")}
    lines = [
        "# K=1 continuation normalization-roundtrip scorecard",
        "",
        "This fixed-denominator causal diagnostic is non-scoring.",
        "",
        "| Fixed paired panel | Sampling-only baseline | Normalization treatment | Gain |",
        "| --- | ---: | ---: | ---: |",
        f"| Preprocessing exactness | {summaries['preprocess']['baseline_pass']}/2 | {summaries['preprocess']['treatment_pass']}/2 | {summaries['preprocess']['paired_gain']:+d} |",
        f"| Geometry identity | {summaries['geometry']['baseline_pass']}/5 | {summaries['geometry']['treatment_pass']}/5 | {summaries['geometry']['paired_gain']:+d} |",
        f"| Score/map gates | {summaries['score_map']['baseline_pass']}/21 | {summaries['score_map']['treatment_pass']}/21 | {summaries['score_map']['paired_gain']:+d} |",
    ]
    for key, title in (
        ("preprocess_cases", "Preprocessing case"),
        ("geometry_cases", "Geometry case"),
        ("score_map_cases", "Score/map case"),
    ):
        lines.extend(
            [
                "",
                f"| Checked | {title} | Baseline | Treatment | Transition | Observation |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for case in scorecard[key]:
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
