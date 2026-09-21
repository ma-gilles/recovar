#!/usr/bin/env python3
"""Validate and render the fixed K=4 preprocessing replay scorecard."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_k4_preprocess_replay_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "math" / "em_k4_preprocess_replay_scorecard.md"
SCHEMA = "recovar.em_k4_preprocess_replay_scorecard.v1"
SUITE_ID = "k4-it2-h1-orig53722-class1-preprocess-replay"
FROZEN_DENOMINATOR = 9
MATERIAL_RELATIVE_L2_THRESHOLD = 5.0e-7
CLASSIFICATION = "softmask_background_reduction_drift_within_fixed_material_floor"
STAGES = (
    "normalized_shifted_real",
    "masked_real",
    "masked_fourier",
)
CASE_IDS = tuple(
    f"{stage_name}-repeat-{repeat_index}"
    for stage_name in ("normalized-shifted", "masked-real", "masked-fourier")
    for repeat_index in range(1, 4)
)
EXPECTED_EXACT = (True, True, True, False, False, False, False, False, False)
EXPECTED_EVIDENCE_SHA256 = {
    "report": "2059de0e8487e2b7dc7f13f94fffe87bdb801c17ccc377924ec297ea783de146",
    "replay_arrays": "123c51379bd563d9b22f45d7a797dc3cc6949f93d4f71ec376c347d71429fd74",
    "sealed_input": "98c8642d7b85645f6416aa834eef931d3561e3db651111cd5d22cbd6ff7e5c0b",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce its fixed denominator."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(
        scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR,
        "frozen denominator changed",
    )
    _require(
        scorecard.get("material_relative_l2_threshold") == MATERIAL_RELATIVE_L2_THRESHOLD,
        "fixed material threshold changed",
    )
    _require(
        scorecard.get("classification") == CLASSIFICATION,
        "preprocessing classification changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("correlation_used") is False
        and scorecard.get("fsc_auc_evaluated") is False,
        "non-scoring metric policy changed",
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
        tuple(case.get("id") for case in cases) == CASE_IDS,
        "fixed case identity/order changed",
    )
    for case_index, case in enumerate(cases):
        stage_index, comparison_index = divmod(case_index, 3)
        _require(case.get("stage") == STAGES[stage_index], "case stage changed")
        _require(
            case.get("comparison_to_first_execution") == comparison_index + 1,
            "comparison identity changed",
        )
        _require(
            case.get("bitwise_equal") is EXPECTED_EXACT[case_index],
            "bitwise result changed",
        )
        relative_l2 = case.get("relative_l2")
        max_abs = case.get("max_abs")
        _require(
            isinstance(relative_l2, (int, float)) and math.isfinite(relative_l2) and relative_l2 >= 0.0,
            "invalid relative-L2",
        )
        _require(
            isinstance(max_abs, (int, float)) and math.isfinite(max_abs) and max_abs >= 0.0,
            "invalid maximum absolute error",
        )
        expected_within = relative_l2 <= MATERIAL_RELATIVE_L2_THRESHOLD
        _require(
            case.get("within_material_floor") is expected_within,
            "material-floor result disagrees with relative-L2",
        )
        _require(
            case.get("checked") is expected_within,
            "checkmark disagrees with material-floor result",
        )

    summary = {
        "bitwise_equal": sum(case["bitwise_equal"] for case in cases),
        "within_material_floor": sum(case["within_material_floor"] for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"bitwise_equal": 3, "within_material_floor": 9, "evaluated": 9}, "fixed replay result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the checked replay scorecard as a compact table."""

    summary = scorecard["summary"]
    lines = [
        "# K=4 preprocessing replay scorecard",
        "",
        "This fixed-denominator diagnostic localizes numerical repeatability.",
        "It cannot change the frozen K=1 or K=4 FSC/FSC-AUC scorecards.",
        "",
        (
            f"Bitwise exact: **{summary['bitwise_equal']} / "
            f"{scorecard['frozen_denominator']}**. Within fixed material "
            f"floor: **{summary['within_material_floor']} / "
            f"{scorecard['frozen_denominator']}**."
        ),
        "",
        "| Checked | Case | Stage | Exact | Relative L2 | Max abs |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        exact = "yes" if case["bitwise_equal"] else "no"
        lines.append(
            f"| {check} | `{case['id']}` | `{case['stage']}` | {exact} | "
            f"{case['relative_l2']:.9g} | {case['max_abs']:.9g} |"
        )
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            (f"Fixed material relative-L2 threshold: `{scorecard['material_relative_l2_threshold']:.9g}`."),
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
            ("pixi run python scripts/summarize_em_k4_preprocess_replay_scorecard.py --check"),
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
