#!/usr/bin/env python3
"""Validate and render the fixed K=1 restart particle-order scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_k1_restart_particle_order_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "math" / "em_k1_restart_particle_order_scorecard.md"
SCHEMA = "recovar.em_k1_restart_particle_order_scorecard.v1"
SUITE_ID = "k1-case22-restart-particle-order-restore-ab"
CLASSIFICATION = (
    "serialized_iteration1_state_closes_case22_score_and_map_gates_when_iteration1_particle_order_is_restored"
)
FROZEN_DENOMINATOR_PER_ARM = 20
TREATMENT_DENOMINATOR = 40
STACKS = (35, 252, 348, 591, 683, 1100, 1522, 1640, 1767, 2124, 2322, 2330, 2846, 2994)
CASE_IDS = (
    *(f"score-stack-{stack:04d}" for stack in STACKS),
    "parity-fsc-auc-half1",
    "parity-fsc-auc-half2",
    "parity-fsc-auc-merged",
    "gt-fsc-auc-nondegradation-half1",
    "gt-fsc-auc-nondegradation-half2",
    "gt-fsc-auc-nondegradation-merged",
)
EXPECTED_EVIDENCE_SHA256 = {
    "primary_report": "e54ab9f917bdc1f6a66c68e38f84356b4d64eb3a2e1f6d143464f9811a2bd7d1",
    "completion_record": "2c14512581f2b7edc5e774ba7d0aac696263eb18a6091697ad6bb819e008e827",
    "science_output_manifest": "247187905f938ae4aa52e8e6f280d2543e51c361f8d9eef409b47d798fe93c3f",
    "science_input_manifest": "6f7394b51837bc138829f4ffd1058175857de6af254006b707b0ddfc20069792",
    "predeclaration": "6b2ec0a04584a5bda81c39cc848abc327fbf7ded330e767dfc64d22f63f05e43",
    "launcher": "692cddced6b99f5fcb37c8c6f31bceee31345d6353327031cda32a81b342b0c2",
    "science_summarizer": "457392eb7f86021b8478fc5d88584d428685791994ce09421da334381a0a0ee9",
    "post_terminal_audit": "da98ef81dfc99bb0128d043c7b63cde0471b75e7667ddd24634ebe68d05a9dda",
    "stock_score_report": "e747cf832dd08ca4f572d32a6c57e77a30e3d51b396943caa4eb1598eafed134",
    "stock_map_report": "0d8d4de189f6d3282ecdfbf6226e532f5bb7dcde1c0d6d35e3ac575e7fc63c8f",
    "restored_a_score_report": "ebd63dbc7c6e2a3edea19b909fce4aca7a836e5866f4e1ef9fdd5f6dd8f6d1fd",
    "restored_a_map_report": "0e568430e138a1efcdab683c42b49ef38a8201d3a97739a7f7374f7e59caa87d",
    "restored_b_score_report": "eba9060b06ec58e6f20696fa5477b3e7e2d85b7d4837cf74dc8afa0d71d9bc75",
    "restored_b_map_report": "116d7aad5888e64d8f2d4468930d5a6213887ba31b748a27dd17d988b830d52e",
}
EXPECTED_REPEATABILITY = {
    "half1": 0.9999999999872374,
    "half2": 0.9999999999871069,
    "merged": 0.9999999999932139,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce its immutable result."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(scorecard.get("classification") == CLASSIFICATION, "classification changed")
    _require(
        scorecard.get("frozen_denominator_per_arm") == FROZEN_DENOMINATOR_PER_ARM
        and scorecard.get("treatment_denominator") == TREATMENT_DENOMINATOR,
        "frozen denominator changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("correlation_used") is False
        and scorecard.get("fsc_auc_evaluated") is True
        and scorecard.get("production_relion_patch_accepted") is False,
        "metric or production policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("producer_job_id") == 11905073
        and contract.get("producer_state") == "COMPLETED"
        and contract.get("producer_exit_code") == "0:0"
        and contract.get("same_physical_gpu") is True
        and contract.get("analysis_commit") == "6e692d7bead4ce3997f003c8438b980ba3723d75"
        and contract.get("diagnostic_relion_source_commit") == "c1f598337ccefc5dc13afa84ee0d40dc71507880"
        and contract.get("diagnostic_relion_binary_sha256")
        == "38732db0c7eda4a28b5ecaa9aad0fe6eb61d5161c7f6dd95ff0112883454e6d1"
        and contract.get("dispatch_controls_passed") == 4
        and contract.get("dispatch_controls_evaluated") == 4
        and contract.get("fresh_dispatch_rows_per_iteration") == 3000
        and contract.get("stock_treatment_marker_count") == 0
        and contract.get("restored_a_treatment_marker_count") == 1
        and contract.get("restored_b_treatment_marker_count") == 1
        and contract.get("restored_decisions_agree") is True
        and contract.get("accepted") is True
        and contract.get("grid_correction") == "unset/default-off"
        and contract.get("forced_final_all_data_after_nonconvergence") is False,
        "terminal acceptance contract changed",
    )
    _require(
        scorecard.get("restored_repeatability_fsc_auc") == EXPECTED_REPEATABILITY,
        "repeatability FSC-AUC changed",
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
        isinstance(cases, list) and len(cases) == FROZEN_DENOMINATOR_PER_ARM,
        "cases do not preserve the frozen denominator",
    )
    _require(tuple(case.get("id") for case in cases) == CASE_IDS, "fixed case identity/order changed")
    for index, case in enumerate(cases):
        expected_stock = "fail" if index >= 17 else "pass"
        _require(
            case.get("stock_result") == expected_stock
            and case.get("restored_a_result") == "pass"
            and case.get("restored_b_result") == "pass"
            and case.get("checked") is True,
            "fixed particle-order result changed",
        )

    summary = {
        "stock": {
            "pass": sum(case["stock_result"] == "pass" for case in cases),
            "evaluated": len(cases),
        },
        "restored_a": {
            "pass": sum(case["restored_a_result"] == "pass" for case in cases),
            "evaluated": len(cases),
        },
        "restored_b": {
            "pass": sum(case["restored_b_result"] == "pass" for case in cases),
            "evaluated": len(cases),
        },
        "restored_combined": {
            "pass": sum(case[arm] == "pass" for case in cases for arm in ("restored_a_result", "restored_b_result")),
            "evaluated": 2 * len(cases),
        },
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(
        summary
        == {
            "stock": {"pass": 17, "evaluated": 20},
            "restored_a": {"pass": 20, "evaluated": 20},
            "restored_b": {"pass": 20, "evaluated": 20},
            "restored_combined": {"pass": 40, "evaluated": 40},
        }
        and scorecard.get("paired_gain_per_repeat") == 3,
        "fixed particle-order summary changed",
    )
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the fixed same-A100 stock/restored table."""

    summary = scorecard["summary"]
    lines = [
        "# K=1 restart particle-order restoration scorecard",
        "",
        "This fixed-denominator same-A100 diagnostic is non-scoring.",
        "Map acceptance uses FSC/FSC-AUC; correlation is forbidden.",
        "",
        (f"Stock restart: **{summary['stock']['pass']} / {scorecard['frozen_denominator_per_arm']}**."),
        (f"Restored repeats: **{summary['restored_combined']['pass']} / {scorecard['treatment_denominator']}**."),
        "Paired gain: **+3 / 20 per restored repeat**.",
        "",
        "| Checked | Fixed gate | Stock | Restored A | Restored B |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(
            f"| {check} | `{case['id']}` | {case['stock_result']} | "
            f"{case['restored_a_result']} | {case['restored_b_result']} |"
        )
    repeatability = scorecard["restored_repeatability_fsc_auc"]
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            "Dispatch controls: **4 / 4**; each dispatch contains the exact",
            "3,000-particle permutation. Restored A and B use the fresh",
            "iteration-1 order; stock does not.",
            "",
            "Restored A/B FSC-AUC repeatability:",
            "",
            f"- half 1: `{repeatability['half1']}`",
            f"- half 2: `{repeatability['half2']}`",
            f"- merged: `{repeatability['merged']}`",
            "",
            "The intervention is diagnostic and is not a production RELION patch.",
            "It causally identifies stock restart reordering as the source of its",
            "three GT FSC-AUC failures.",
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k1_restart_particle_order_scorecard.py --check",
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
