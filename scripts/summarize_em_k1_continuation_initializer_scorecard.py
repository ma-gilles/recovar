#!/usr/bin/env python3
"""Validate and render the fixed K=1 continuation-initializer A/B scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_k1_continuation_initializer_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "math" / "em_k1_continuation_initializer_scorecard.md"
SCHEMA = "recovar.em_k1_continuation_initializer_scorecard.v1"
SUITE_ID = "k1-case22-sameallocation-continuation-resolution-initializer-ab"
FROZEN_DENOMINATOR = 21
STACK_IDS = (
    35,
    252,
    348,
    591,
    683,
    1100,
    1522,
    1640,
    1767,
    2124,
    2322,
    2330,
    2846,
    2994,
)
MAP_LABELS = ("half1", "half2", "merged")
GROUP_DENOMINATORS = {
    "score": 14,
    "map-parity": 3,
    "map-gt": 3,
    "overall-arm": 1,
}
CASE_IDS = (
    *(f"continuation-init-score-stack-{stack_id:04d}" for stack_id in STACK_IDS),
    *(f"continuation-init-map-parity-{label}" for label in MAP_LABELS),
    *(f"continuation-init-map-gt-{label}" for label in MAP_LABELS),
    "continuation-init-overall",
)
VALID_RESULTS = {"pass", "fail"}
VALID_TRANSITIONS = {"improved", "retained", "regressed", "unchanged-fail"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
METRIC_POLICY = (
    "Fixed same-allocation 14-particle score gates, 3 signed shellwise "
    "FSC-AUC parity gates, 3 GT FSC-AUC non-degradation gates, and 1 "
    "overall arm gate, evaluated before and after one predeclared RELION "
    "continuation-initializer change; no fitted tolerance, scale, sign, "
    "shell boundary, or correlation."
)
EXPECTED_CLASSIFICATION = (
    "resolution_initializer_changes_iteration2_geometry_but_is_not_sufficient_for_score_or_map_parity"
)
EXPECTED_CAUSAL_INTERPRETATION = (
    "continuation_divergence_contains_additional_process_resident_state_beyond_resolution_initializer"
)
EXPECTED_CONTRACT = {
    "same_physical_gpu": True,
    "gpu_uuid": "GPU-eb1c5b04-20c1-b6c9-16e6-b3dc87905bd7",
    "science_job_id": 11840907,
    "science_job_terminal_state": "FAILED",
    "science_job_exit_code": "1:0",
    "predeclared_hypothesis_accepted": False,
    "all_arms_and_analysis_reports_complete": True,
    "recovar_commit": "d676d9d8ac4ca91b4e74b6b29ebe817dd889bc36",
    "stock_relion_commit": "ed53c60d83125902c456b7fc5461c78c3966b306",
    "patched_relion_commit": "c5e1280db23c1202adc4b72c38985f52300bf93f",
    "stock_relion_binary_sha256": ("a274dda1b0b40478ddd7f2b81d144bec20510db369225f365f1be7d27ac45309"),
    "patched_relion_binary_sha256": ("552109b733171a6a48feee98e4eb629e2fe97122a4dbdd86bfb37afdb3ad8133"),
    "launcher_sha256": ("9cec5a7cd9f91b7af4adfb10be53cc378b6dd4aca9a40a535d43c5567dc73ae3"),
    "predeclaration_sha256": ("e5cad8349d7c4194b9d2647062ddf46e9e881d65b52aea9419d76b6078241c67"),
    "fresh_iteration2_current_size": 60,
    "stock_continuation_iteration2_current_size": 58,
    "patched_continuation_iteration2_current_size": 60,
    "fresh_iteration2_entry_resolution_angstrom": 30.2222,
    "stock_continuation_iteration2_entry_resolution_angstrom": 32.0,
    "patched_continuation_iteration2_entry_resolution_angstrom": 30.2222,
    "fresh_iteration2_entry_no_gain_iterations": 1,
    "stock_continuation_iteration2_entry_no_gain_iterations": 2,
    "patched_continuation_iteration2_entry_no_gain_iterations": 1,
    "grid_correction": "unset/default-off",
    "forced_final_all_data_after_nonconvergence": False,
}
EXPECTED_EVIDENCE_SHA256 = {
    "fail_closed_result": "9fe9e2b60a9a2d2368cf10df8a46f590f559c7e7b9b0a8db2ef570c7db7a4b9c",
    "fresh_arm_complete": "4bd4c2744a892b26a8bced026cf199b99075a04cbecbbd7bc1294c2f459b7443",
    "stock_arm_complete": "5907fd45036ccd8ff2a6f2be9a2ce81ff0510f1ec9f04aa899b8f2566e626917",
    "patched_arm_complete": "43e33bfccb03a4f8e563fa7ef6bf080ec10b2c363d6e956f474ab6122db3ad97",
    "stock_score": "686b170a1e96d3426647e06a27f7cd003ec46ef1dda3dc322c1d4237c1b917f2",
    "stock_map": "bb03847af2ecd7dcad4fa683d10b3bfe9a3cf30bb0b788b193017ab8dff03c50",
    "patched_score": "42bfc5c437faebcf50aaa42b8646b43e0237f7c00c498464a3add4146fa08376",
    "patched_map": "dff4f487a6d6154d7929bdaf5f8447d0021e794f4ffbd39db1c94f74c1d44158",
    "science_inputs": "bb130eaa93ba138bdf21881c157bcf5b4096e3c7001802da9471966400deb66c",
    "stdout": "b1fe28e8a7fcdb5e21f43f864c7cc636bbcff56bbb6fd5032a1f3d9fb5895458",
    "stderr": "64002ec91f04e9c74e11b772f3a753fa8aa6ea39a46e615b98140cea1e373c24",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _case_group(case_id: str) -> str:
    if "-score-stack-" in case_id:
        return "score"
    if "-map-parity-" in case_id:
        return "map-parity"
    if "-map-gt-" in case_id:
        return "map-gt"
    return "overall-arm"


def _transition(baseline: str, treatment: str) -> str:
    if baseline == "fail" and treatment == "pass":
        return "improved"
    if baseline == "pass" and treatment == "pass":
        return "retained"
    if baseline == "pass" and treatment == "fail":
        return "regressed"
    return "unchanged-fail"


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce every frozen identity/count."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(
        scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR,
        "frozen denominator changed",
    )
    _require(
        scorecard.get("group_denominators") == GROUP_DENOMINATORS,
        "group denominators changed",
    )
    _require(scorecard.get("metric_policy") == METRIC_POLICY, "metric policy changed")
    _require(
        scorecard.get("scorecard_change_admissible") is False,
        "diagnostic scorecard cannot authorize a production score change",
    )
    _require(
        scorecard.get("correlation_used") is False,
        "correlation is forbidden in the causal scorecard",
    )
    _require(
        scorecard.get("fsc_auc_evaluated") is True,
        "the fixed map FSC-AUC gates were not evaluated",
    )
    _require(
        scorecard.get("classification") == EXPECTED_CLASSIFICATION,
        "intervention classification changed",
    )
    _require(
        scorecard.get("causal_interpretation") == EXPECTED_CAUSAL_INTERPRETATION,
        "causal interpretation changed",
    )
    _require(
        scorecard.get("acceptance_contract") == EXPECTED_CONTRACT,
        "acceptance contract changed",
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
            isinstance(evidence_path, str) and Path(evidence_path).is_absolute(),
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

    grouped_counts = {group: {"baseline_pass": 0, "treatment_pass": 0, "evaluated": 0} for group in GROUP_DENOMINATORS}
    baseline_pass = 0
    treatment_pass = 0
    transitions = {transition: 0 for transition in VALID_TRANSITIONS}
    for case, expected_id in zip(cases, CASE_IDS, strict=True):
        baseline = case.get("baseline_result")
        treatment = case.get("treatment_result")
        group = case.get("group")
        _require(baseline in VALID_RESULTS, f"{expected_id}: invalid baseline result")
        _require(treatment in VALID_RESULTS, f"{expected_id}: invalid treatment result")
        _require(
            case.get("result") == treatment,
            f"{expected_id}: result must equal treatment result",
        )
        _require(
            case.get("checked") is (treatment == "pass"),
            f"{expected_id}: checkmark disagrees with treatment result",
        )
        expected_transition = _transition(baseline, treatment)
        _require(
            case.get("transition") in VALID_TRANSITIONS and case["transition"] == expected_transition,
            f"{expected_id}: transition disagrees with arm results",
        )
        _require(group in GROUP_DENOMINATORS, f"{expected_id}: invalid group")
        _require(
            group == _case_group(expected_id),
            f"{expected_id}: fixed group changed",
        )
        expected_pair = ("pass", "pass") if group == "map-gt" else ("fail", "fail")
        _require(
            (baseline, treatment) == expected_pair,
            f"{expected_id}: fixed stock/patched outcome changed",
        )
        _require(
            isinstance(case.get("name"), str) and case["name"],
            f"{expected_id}: missing name",
        )
        _require(
            isinstance(case.get("observed"), str) and case["observed"],
            f"{expected_id}: missing observation",
        )
        grouped_counts[group]["baseline_pass"] += baseline == "pass"
        grouped_counts[group]["treatment_pass"] += treatment == "pass"
        grouped_counts[group]["evaluated"] += 1
        baseline_pass += baseline == "pass"
        treatment_pass += treatment == "pass"
        transitions[expected_transition] += 1

    for group, denominator in GROUP_DENOMINATORS.items():
        _require(
            grouped_counts[group]["evaluated"] == denominator,
            f"{group}: evaluated denominator changed",
        )
    _require(
        scorecard.get("grouped_summary") == grouped_counts,
        "recorded grouped summary does not replay cases",
    )
    expected_baseline = {
        "pass": baseline_pass,
        "fail": FROZEN_DENOMINATOR - baseline_pass,
        "evaluated": FROZEN_DENOMINATOR,
    }
    expected_treatment = {
        "pass": treatment_pass,
        "fail": FROZEN_DENOMINATOR - treatment_pass,
        "evaluated": FROZEN_DENOMINATOR,
    }
    _require(
        scorecard.get("baseline_summary") == expected_baseline,
        "recorded baseline summary does not replay cases",
    )
    _require(
        scorecard.get("treatment_summary") == expected_treatment,
        "recorded treatment summary does not replay cases",
    )
    _require(
        scorecard.get("transition_summary") == transitions,
        "recorded transition summary does not replay cases",
    )
    _require(
        scorecard.get("paired_gain") == treatment_pass - baseline_pass,
        "recorded paired gain does not replay cases",
    )
    expected_two_arm = {
        "pass": baseline_pass + treatment_pass,
        "fail": 2 * FROZEN_DENOMINATOR - baseline_pass - treatment_pass,
        "evaluated": 2 * FROZEN_DENOMINATOR,
        "denominator": 2 * FROZEN_DENOMINATOR,
    }
    _require(
        scorecard.get("two_arm_summary") == expected_two_arm,
        "recorded two-arm summary does not replay cases",
    )
    _require(expected_baseline["pass"] == 3, "stock baseline must remain 3/21")
    _require(expected_treatment["pass"] == 3, "patched treatment must remain 3/21")
    _require(scorecard["paired_gain"] == 0, "paired gain must remain zero")
    _require(
        transitions
        == {
            "improved": 0,
            "retained": 3,
            "regressed": 0,
            "unchanged-fail": 18,
        },
        "fixed transition counts changed",
    )
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the fixed paired scorecard as counts plus checked cases."""

    baseline = scorecard["baseline_summary"]
    treatment = scorecard["treatment_summary"]
    lines = [
        "# K=1 continuation resolution-initializer A/B scorecard",
        "",
        "This is a non-scoring, fixed-denominator causal diagnostic. It cannot",
        "change the frozen K=1 or K=4 FSC/FSC-AUC quality scorecards.",
        "",
        (
            f"Fixed paired score: stock **{baseline['pass']} / "
            f"{scorecard['frozen_denominator']}** → patched "
            f"**{treatment['pass']} / {scorecard['frozen_denominator']}** "
            f"(gain **+{scorecard['paired_gain']}**; "
            f"{scorecard['two_arm_summary']['pass']} / "
            f"{scorecard['two_arm_summary']['denominator']} total arm checks "
            "passing)."
        ),
        "",
        "| Gate group | Stock pass | Patched pass | Evaluated | Denominator |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for group, denominator in scorecard["group_denominators"].items():
        grouped = scorecard["grouped_summary"][group]
        lines.append(
            f"| `{group}` | {grouped['baseline_pass']} | "
            f"{grouped['treatment_pass']} | {grouped['evaluated']} | "
            f"{denominator} |"
        )
    lines.extend(
        [
            "",
            "| Checked | Gate | Stock | Patched | Transition | Observation |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(
            f"| {check} | `{case['id']}` | {case['baseline_result']} | "
            f"{case['treatment_result']} | {case['transition']} | "
            f"{case['observed']} |"
        )
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            f"Causal interpretation: `{scorecard['causal_interpretation']}`.",
            "",
            f"Metric policy: {scorecard['metric_policy']}",
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
            "To validate and regenerate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k1_continuation_initializer_scorecard.py --check",
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
