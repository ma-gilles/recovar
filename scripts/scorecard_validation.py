"""Common checks for fixed historical baseline/treatment scorecards.

Case inventories, expected outcomes and rendering stay in each report script.
Keep this module free of scientific-runtime imports so these scripts remain
usable with standard Python, through direct paths or ``python -m``.
"""

from __future__ import annotations


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def paired_transition(baseline: str, treatment: str) -> str:
    if baseline == "fail" and treatment == "pass":
        return "improved"
    if baseline == "pass" and treatment == "pass":
        return "retained"
    if baseline == "pass" and treatment == "fail":
        return "regressed"
    return "unchanged-fail"


def validate_paired_cases(
    cases: object,
    expected_ids: tuple[str, ...],
    expected_pairs: tuple[tuple[str, str], ...],
) -> tuple[dict[str, int], dict[str, int]]:
    require(isinstance(cases, list) and len(cases) == len(expected_ids), "case denominator changed")
    require(tuple(case.get("id") for case in cases) == expected_ids, "case identity/order changed")
    baseline_pass = 0
    treatment_pass = 0
    transitions = {name: 0 for name in ("improved", "retained", "regressed", "unchanged-fail")}
    for case, expected_pair in zip(cases, expected_pairs, strict=True):
        pair = (case.get("baseline_result"), case.get("treatment_result"))
        require(pair == expected_pair, f"{case.get('id')}: fixed result pair changed")
        require(case.get("result") == pair[1], f"{case.get('id')}: result is not treatment result")
        require(case.get("checked") is (pair[1] == "pass"), f"{case.get('id')}: checkmark changed")
        transition = paired_transition(*pair)
        require(case.get("transition") == transition, f"{case.get('id')}: transition changed")
        require(isinstance(case.get("name"), str) and case["name"], f"{case.get('id')}: missing name")
        require(
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
