#!/usr/bin/env python3
"""Validate and render the fixed K=1 serialized-restart causal scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_k1_serialized_restart_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "math" / "em_k1_serialized_restart_scorecard.md"
SCHEMA = "recovar.em_k1_serialized_restart_scorecard.v1"
SUITE_ID = "k1-case22-sameallocation-serialized-restart-pair"
FROZEN_DENOMINATOR = 42
ARMS = ("iteration0-restart", "iteration1-restart")
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
    "score": 28,
    "map-parity": 6,
    "map-gt": 6,
    "overall-arm": 2,
}
CASE_IDS = tuple(
    case_id
    for arm in ARMS
    for case_id in (
        *(f"{arm}-score-stack-{stack_id:04d}" for stack_id in STACK_IDS),
        *(f"{arm}-map-parity-{label}" for label in MAP_LABELS),
        *(f"{arm}-map-gt-{label}" for label in MAP_LABELS),
        f"{arm}-overall",
    )
)
VALID_RESULTS = {"pass", "fail"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
METRIC_POLICY = (
    "Fixed same-allocation 28-particle score gates, 6 signed shellwise "
    "FSC-AUC parity gates, 6 GT FSC-AUC non-degradation gates, and 2 "
    "overall restart-arm gates; no fitted tolerance, scale, sign, shell "
    "boundary, or correlation."
)
EXPECTED_CLASSIFICATION = "only_iteration0_restart_closes_score_and_map_gates"
EXPECTED_CAUSAL_INTERPRETATION = "case22_recovery_requires_replaying_iteration1_from_serialized_it0"
EXPECTED_EVIDENCE_SHA256 = {
    "science_complete": ("56fd70c2eb72e750bd36762ecb2bd62bfcc17c9487a775869e40c722e186446c"),
    "restart_pair": ("ef05a9a55d1d339d61f0d354ae344caca128fb436b8acb3b609f1063c54e8ed0"),
    "restart0_score": ("5fc76eca4cb90c93e4b5412b1de8e7ca679ef27a855e075242ee84a58514c370"),
    "restart0_map": ("e206ff91e54460118f7411d2fd1680074fdb1f0856b6fe6a20528ce7d2e9d2a7"),
    "restart1_score": ("e074f54cf830aa4495d831a12a993a7ad594e3548d4188403df0a61c2bbd5c25"),
    "restart1_map": ("4da6ca5af66f03bed4fccecd5641f2a8e93ab39720b2f36265c74223a561be66"),
    "fresh_it0_control": ("bbf9eca502ae0a9a23f8b6291c3feeaf98ca48874144cf2206706639af3fb016"),
    "restart0_iteration1_control": ("f816fcdd9b92e11bcda59ca38b8b62ec365c12db65a01ca6a4ba95915fc663f9"),
    "analysis_manifest": ("6ae2ae809b8fb2b1f2e2212c96dd72210ae2342bbc93b5aee3a3af06585cc688"),
    "arm_manifest": ("33e3f6891e684fcd0cd8dd7a5d1763a3ce8e0c7d058c3a6ddda51691cdd03bf1"),
}
EXPECTED_CONTRACT = {
    "same_physical_gpu": True,
    "gpu_uuid": "GPU-eb1c5b04-20c1-b6c9-16e6-b3dc87905bd7",
    "science_job_id": 11839040,
    "recovar_commit": "d676d9d8ac4ca91b4e74b6b29ebe817dd889bc36",
    "relion_commit": "ed53c60d83125902c456b7fc5461c78c3966b306",
    "relion_binary_sha256": ("a274dda1b0b40478ddd7f2b81d144bec20510db369225f365f1be7d27ac45309"),
    "launcher_sha256": ("6ef245eb663ae55eac0d6e4eae07de42f131713929056c1266d3b4620e74ff01"),
    "predeclaration_sha256": ("f8a6f1867d271dd69d489940f108f8311ed03aef599e3fe669acfc734e625ad2"),
    "grid_correction": "unset/default-off",
    "forced_final_all_data_after_nonconvergence": False,
    "restart0_controls": ("--firstiter_cc --ini_high 30 --auto_iter_max 2 --pool 3"),
    "restart1_controls": "--auto_iter_max 2 --pool 3",
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
    _require(
        scorecard.get("metric_policy") == METRIC_POLICY,
        "metric policy changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False,
        "causal scorecard cannot authorize a production score change",
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
        "restart-pair classification changed",
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
    grouped_counts = {group: {"pass": 0, "fail": 0, "evaluated": 0} for group in GROUP_DENOMINATORS}
    for case, expected_id in zip(cases, CASE_IDS, strict=True):
        result = case.get("result")
        group = case.get("group")
        _require(result in VALID_RESULTS, f"{case.get('id')}: invalid result")
        _require(
            case.get("checked") is (result == "pass"),
            f"{case.get('id')}: checkmark disagrees with result",
        )
        _require(group in GROUP_DENOMINATORS, f"{case.get('id')}: invalid group")
        _require(
            case.get("arm") == next(arm for arm in ARMS if expected_id.startswith(f"{arm}-")),
            f"{case.get('id')}: restart arm changed",
        )
        _require(
            group == _case_group(expected_id),
            f"{case.get('id')}: fixed group changed",
        )
        _require(
            isinstance(case.get("name"), str) and case["name"],
            f"{case.get('id')}: missing name",
        )
        _require(
            isinstance(case.get("observed"), str) and case["observed"],
            f"{case.get('id')}: missing observation",
        )
        grouped_counts[group][result] += 1
        grouped_counts[group]["evaluated"] += 1

    for group, denominator in GROUP_DENOMINATORS.items():
        _require(
            grouped_counts[group]["evaluated"] == denominator,
            f"{group}: evaluated denominator changed",
        )
    _require(
        scorecard.get("grouped_summary") == grouped_counts,
        "recorded grouped summary does not replay cases",
    )
    counts = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "fail": sum(case["result"] == "fail" for case in cases),
    }
    expected_summary = {
        **counts,
        "evaluated": counts["pass"] + counts["fail"],
    }
    _require(
        scorecard.get("summary") == expected_summary,
        "recorded summary does not match fixed cases",
    )
    _require(
        expected_summary["evaluated"] == FROZEN_DENOMINATOR,
        "not all fixed cases are evaluated",
    )
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the fixed scorecard as grouped counts plus checked cases."""

    summary = scorecard["summary"]
    lines = [
        "# K=1 case-22 serialized-restart causal scorecard",
        "",
        "This is a non-scoring, fixed-denominator diagnostic. It cannot change",
        "the frozen K=1 or K=4 FSC/FSC-AUC quality scorecards.",
        "",
        (
            f"Fixed causal score: **{summary['pass']} / "
            f"{scorecard['frozen_denominator']} passing** "
            f"({summary['evaluated']} / "
            f"{scorecard['frozen_denominator']} evaluated)."
        ),
        "",
        "| Gate group | Passed | Evaluated | Denominator |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group, denominator in scorecard["group_denominators"].items():
        grouped = scorecard["grouped_summary"][group]
        lines.append(f"| `{group}` | {grouped['pass']} | {grouped['evaluated']} | {denominator} |")
    lines.extend(
        [
            "",
            "| Checked | Arm | Gate | Result | Observation |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(f"| {check} | `{case['arm']}` | `{case['id']}` | {case['result']} | {case['observed']} |")
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            (f"Causal interpretation: `{scorecard['causal_interpretation']}`."),
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
            ("pixi run python scripts/summarize_em_k1_serialized_restart_scorecard.py --check"),
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
