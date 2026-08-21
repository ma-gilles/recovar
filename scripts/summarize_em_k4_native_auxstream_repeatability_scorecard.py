#!/usr/bin/env python3
"""Validate and render the fixed K=4 native aux-stream scorecard."""

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
    / "em_k4_native_auxstream_repeatability_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT / "docs" / "math" / "em_k4_native_auxstream_repeatability_scorecard.md"
)
SCHEMA = "recovar.em_k4_native_auxstream_repeatability_scorecard.v1"
SUITE_ID = "k4-it2-fixed-a100-native-auxstream-observer-repeatability"
CLASSIFICATION = "native_auxstream_observer_not_byte_repeatable"
FROZEN_DENOMINATOR = 13
EXPECTED_GATE_RESULTS = {
    "artifact_bytes_exact": "fail",
    "auxiliary_reduction_stream_markers_exact": "pass",
    "capture_headers_mpi2_openmp1_exact": "pass",
    "capture_validators_passed": "pass",
    "class_map_fsc_auc_at_least_threshold": "pass",
    "current_size_and_sampling_topology_exact": "pass",
    "dispatch_bytes_exact": "pass",
    "dispatch_row_count_exact": "pass",
    "no_fatal_runtime_pattern": "pass",
    "particle_count_exact": "pass",
    "powerclass_stream_markers_exact": "pass",
    "target_state_exact": "pass",
    "thread_replay_markers_exact": "pass",
}
EXPECTED_EVIDENCE_SHA256 = {
    "analysis_result": (
        "bc68328ef3c97f996018c4992ad69a3748c47eaca98f44e5b8c062f8c0fb8a57"
    ),
    "analyzer": (
        "814b9785e068713852951a901e26d3dc07fc17ad612df743ffc0b5c1870a952a"
    ),
    "science_completion": (
        "6463dea3c1b879097d27ae9e9338b3a18159c0e286421d6e6d27ad14d2fc5d9e"
    ),
    "analysis_completion": (
        "2e679318eb603d75fdd2f73105096257ad5274118fdee89a7e7ae9530dcfbfcf"
    ),
    "science_manifest": (
        "bd3625b4c3cb57e272395c15687c9c3787481d9712a1f61fdaec569d142ae09b"
    ),
    "static_manifest": (
        "05707af0b37847264215238d7e72049a97acc996568048c2f0e93297193b86e8"
    ),
}
EXPECTED_CLASS_FSC_AUC = (
    0.9999999895278622,
    0.9999999846443698,
    0.9999999749836932,
    0.9999999778593218,
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce its fixed evidence and denominator."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(
        scorecard.get("classification") == CLASSIFICATION,
        "repeatability classification changed",
    )
    _require(
        scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR,
        "frozen denominator changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("correlation_used") is False
        and scorecard.get("fsc_auc_evaluated") is True,
        "non-scoring metric policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("science_job_id") == 11988750
        and contract.get("science_state") == "COMPLETED"
        and contract.get("science_exit_code") == "0:0"
        and contract.get("audit_job_id") == 11988757
        and contract.get("audit_state") == "FAILED"
        and contract.get("postterminal_analysis_completed") is True
        and contract.get("accepted") is False,
        "terminal Slurm acceptance contract changed",
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
        tuple(case.get("id") for case in cases) == tuple(EXPECTED_GATE_RESULTS),
        "fixed gate identity/order changed",
    )
    for case in cases:
        expected_result = EXPECTED_GATE_RESULTS[case["id"]]
        _require(case.get("result") == expected_result, f"{case['id']}: result changed")
        _require(
            case.get("checked") is (expected_result == "pass"),
            f"{case['id']}: checked state changed",
        )

    maps = scorecard.get("class_map_fsc_auc")
    _require(
        isinstance(maps, dict)
        and maps.get("threshold") == 0.999999
        and tuple(maps.get("values", ())) == EXPECTED_CLASS_FSC_AUC,
        "signed class-map FSC-AUC telemetry changed",
    )
    localization = scorecard.get("first_unequal_boundary")
    _require(
        isinstance(localization, dict)
        and localization.get("boundary") == "preprocessed_image_then_native_fine_score"
        and localization.get("fine_score_unequal") == 16735
        and localization.get("fine_score_total") == 109184
        and localization.get("fine_score_max_abs") == 0.0001220703125
        and localization.get("preprocessed_real_unequal") == 460
        and localization.get("preprocessed_real_total") == 1520
        and localization.get("preprocessed_real_max_abs") == 8.381903171539307e-09
        and localization.get("preprocessed_imag_unequal") == 470
        and localization.get("preprocessed_imag_total") == 1520
        and localization.get("preprocessed_imag_max_abs") == 3.725290298461914e-09,
        "first unequal boundary telemetry changed",
    )

    summary = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"pass": 12, "evaluated": 13}, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the checked native aux-stream scorecard as a compact report."""

    summary = scorecard["summary"]
    boundary = scorecard["first_unequal_boundary"]
    maps = scorecard["class_map_fsc_auc"]
    lines = [
        "# K=4 native aux-stream repeatability scorecard",
        "",
        "This fixed-denominator same-A100 diagnostic tests whether native CUDA",
        "auxiliary streams make a selected RELION observer byte-repeatable. It is",
        "non-scoring and cannot change the frozen FSC/FSC-AUC quality scorecards.",
        "",
        f"Fixed gates: **{summary['pass']} / {scorecard['frozen_denominator']}**.",
        "",
        "| Checked | Fixed gate | Result |",
        "| --- | --- | ---: |",
    ]
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(f"| {check} | `{case['id']}` | {case['result']} |")
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            (
                "The first captured unequal boundary was preprocessing followed by "
                "native fine scoring: "
                f"{boundary['preprocessed_real_unequal']}/"
                f"{boundary['preprocessed_real_total']} real and "
                f"{boundary['preprocessed_imag_unequal']}/"
                f"{boundary['preprocessed_imag_total']} imaginary values differed, "
                f"then {boundary['fine_score_unequal']}/"
                f"{boundary['fine_score_total']} raw `diff2` values differed "
                f"(maximum absolute delta {boundary['fine_score_max_abs']})."
            ),
            "",
            (
                "Signed normalized non-DC class-map FSC-AUC values were "
                + ", ".join(f"{value:.16f}" for value in maps["values"])
                + f" (threshold {maps['threshold']})."
            ),
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
            "Code references:",
            "",
            "- `scripts/summarize_em_k4_native_auxstream_repeatability_scorecard.py`",
            "- `scripts/report_em_parity_progress.py`",
            "",
            "To validate:",
            "",
            "```bash",
            (
                "pixi run python "
                "scripts/summarize_em_k4_native_auxstream_repeatability_scorecard.py "
                "--check"
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
