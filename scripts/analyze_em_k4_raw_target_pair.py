#!/usr/bin/env python3
"""Synthesize the fixed K=4 raw-score and native-target operand audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.analyze_em_k4_authoritative_native_scores import (
    EXPECTED_SUPPORT,
    TARGET_GPU_UUID,
    TARGET_RECOVAR_ROTATION,
    TARGET_TRANSLATIONS,
)
from scripts.analyze_em_k4_native_target_operands import (
    PASS_CLASSIFICATION as OPERAND_PASS_CLASSIFICATION,
)
from scripts.analyze_em_k4_raw_diff2_parity import (
    PASS_CLASSIFICATION as RAW_PASS_CLASSIFICATION,
)
from scripts.analyze_em_k4_raw_diff2_parity import (
    PASS_SCORE_CLASSIFICATION,
)

SCHEMA = "relion-k4-it2-raw-target-pair-v1"
RAW_SCHEMA = "relion-k4-it2-raw-diff2-parity-v2"
OPERAND_SCHEMA = "relion-k4-it2-native-target-operand-audit-v1"
EXPECTED_RECOVAR_JOB_ID = 11_790_517
EXPECTED_OPERAND_JOB_ID = 11_790_787
EXPECTED_NATIVE_JOB_ID = 11_787_017
SHARED_INPUTS = ("factor", "fine_score", "recovar_pass2", "native_completion")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_raw_target_pair(
    *,
    native_operand_accepted: bool,
    global_raw_accepted: bool,
    global_score_accepted: bool,
    target_raw_bitwise_exact: bool,
) -> tuple[str, str]:
    """Return the predeclared joint classification and next causal boundary."""

    _require(
        not global_raw_accepted or target_raw_bitwise_exact,
        "global raw-cost parity cannot pass while the fixed target differs",
    )
    if not native_operand_accepted:
        return (
            "native_target_operand_gate_rejected",
            "repair_or_repeat_native_target_operand_capture_before_inference",
        )
    if global_raw_accepted and global_score_accepted:
        return (
            "native_operand_raw_and_combined_score_paths_bitwise_close",
            "continue_downstream_of_combined_score_generation",
        )
    if global_raw_accepted:
        return (
            "raw_costs_close_but_combined_score_path_differs",
            "localize_prior_minimum_and_saved_score_operation_order",
        )
    if global_score_accepted:
        return (
            "raw_costs_differ_but_combined_score_path_bitwise_closes",
            "treat_raw_residual_as_combined_score_inert_and_continue_downstream",
        )
    if target_raw_bitwise_exact:
        return (
            "global_raw_and_score_paths_differ_but_fixed_target_closes",
            "stratify_global_raw_mismatches_and_capture_a_representative_row",
        )
    return (
        "fixed_target_raw_cost_differs_after_native_operand_replay",
        "capture_recovar_per_pixel_operands_for_the_same_fixed_target",
    )


def _validate_raw_report(
    report: dict[str, Any],
    *,
    expected_recovar_job_id: int,
) -> dict[str, bool]:
    _require(report.get("schema") == RAW_SCHEMA, "K4 raw-score schema changed")
    _require(
        report.get("status") == "complete"
        and report.get("classification_ready") is True,
        "K4 raw-score report is incomplete",
    )
    fixed = report.get("fixed_contract", {})
    _require(
        int(fixed.get("native_slurm_job_id")) == EXPECTED_NATIVE_JOB_ID
        and int(fixed.get("recovar_slurm_job_id"))
        == expected_recovar_job_id
        and fixed.get("target_gpu_uuid") == TARGET_GPU_UUID
        and int(fixed.get("expected_support")) == EXPECTED_SUPPORT
        and int(fixed.get("target_recovar_rotation"))
        == TARGET_RECOVAR_ROTATION
        and tuple(int(value) for value in fixed.get("target_translations", ()))
        == TARGET_TRANSLATIONS,
        "K4 raw-score fixed contract changed",
    )
    raw_accepted = bool(
        report.get("classification") == RAW_PASS_CLASSIFICATION
    )
    _require(
        report.get("accepted") is raw_accepted,
        "K4 raw-score acceptance does not replay",
    )
    score = report.get("score_path", {})
    score_accepted = bool(
        score.get("classification") == PASS_SCORE_CLASSIFICATION
    )
    _require(
        score.get("accepted") is score_accepted,
        "K4 combined-score acceptance does not replay",
    )
    target = report.get("target", {})
    records = target.get("records", ())
    _require(
        len(records) == len(TARGET_TRANSLATIONS),
        "K4 raw-score target denominator changed",
    )
    observed_translations = tuple(
        int(record.get("translation_id")) for record in records
    )
    _require(
        observed_translations == TARGET_TRANSLATIONS,
        "K4 raw-score target translations changed",
    )
    target_raw_exact = all(
        int(record["native_raw_diff2_bits"])
        == int(record["recovar_raw_diff2_bits"])
        for record in records
    )
    _require(
        bool(target.get("native_raw_diff2_tied"))
        == (
            int(records[0]["native_raw_diff2_bits"])
            == int(records[1]["native_raw_diff2_bits"])
        )
        and bool(target.get("recovar_raw_diff2_tied"))
        == (
            int(records[0]["recovar_raw_diff2_bits"])
            == int(records[1]["recovar_raw_diff2_bits"])
        ),
        "K4 raw-score target tie summary does not replay",
    )
    return {
        "global_raw_accepted": raw_accepted,
        "global_score_accepted": score_accepted,
        "target_raw_bitwise_exact": target_raw_exact,
    }


def _validate_operand_report(
    report: dict[str, Any],
    *,
    expected_operand_job_id: int,
) -> bool:
    _require(
        report.get("schema") == OPERAND_SCHEMA,
        "K4 native target-operand schema changed",
    )
    _require(
        report.get("status") == "complete"
        and report.get("classification_ready") is True,
        "K4 native target-operand report is incomplete",
    )
    fixed = report.get("fixed_contract", {})
    _require(
        int(fixed.get("native_slurm_job_id")) == EXPECTED_NATIVE_JOB_ID
        and int(fixed.get("operand_slurm_job_id"))
        == expected_operand_job_id
        and fixed.get("target_gpu_uuid") == TARGET_GPU_UUID
        and int(fixed.get("target_recovar_rotation"))
        == TARGET_RECOVAR_ROTATION
        and tuple(int(value) for value in fixed.get("target_translations", ()))
        == TARGET_TRANSLATIONS,
        "K4 native target-operand fixed contract changed",
    )
    accepted = bool(
        report.get("classification") == OPERAND_PASS_CLASSIFICATION
    )
    _require(
        report.get("accepted") is accepted,
        "K4 native target-operand acceptance does not replay",
    )
    return accepted


def _validate_shared_inputs(
    raw_report: dict[str, Any],
    operand_report: dict[str, Any],
) -> None:
    raw_inputs = raw_report.get("inputs", {})
    operand_inputs = operand_report.get("inputs", {})
    for name in SHARED_INPUTS:
        raw_input = raw_inputs.get(name, {})
        operand_input = operand_inputs.get(name, {})
        raw_path = raw_input.get("path")
        operand_path = operand_input.get("path")
        raw_sha = raw_input.get("sha256")
        operand_sha = operand_input.get("sha256")
        _require(
            isinstance(raw_path, str)
            and bool(raw_path)
            and isinstance(operand_path, str)
            and bool(operand_path)
            and isinstance(raw_sha, str)
            and len(raw_sha) == 64
            and raw_sha == operand_sha
            and Path(raw_path).resolve() == Path(operand_path).resolve(),
            f"K4 reports do not share the same hash-linked {name}",
        )


def build_report(
    *,
    raw_report_path: Path,
    operand_report_path: Path,
    expected_recovar_job_id: int = EXPECTED_RECOVAR_JOB_ID,
    expected_operand_job_id: int = EXPECTED_OPERAND_JOB_ID,
) -> dict[str, Any]:
    raw_report = json.loads(raw_report_path.read_text())
    operand_report = json.loads(operand_report_path.read_text())
    raw_gates = _validate_raw_report(
        raw_report,
        expected_recovar_job_id=expected_recovar_job_id,
    )
    native_operand_accepted = _validate_operand_report(
        operand_report,
        expected_operand_job_id=expected_operand_job_id,
    )
    _validate_shared_inputs(raw_report, operand_report)
    classification, next_boundary = classify_raw_target_pair(
        native_operand_accepted=native_operand_accepted,
        **raw_gates,
    )
    gates = {
        "native_target_operand_replay": native_operand_accepted,
        "global_raw_diff2": raw_gates["global_raw_accepted"],
        "global_combined_score": raw_gates["global_score_accepted"],
        "fixed_target_raw_diff2": raw_gates["target_raw_bitwise_exact"],
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": classification,
        "next_causal_boundary": next_boundary,
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed four-boundary K4 exact-device diagnostic; bitwise native "
            "operand replay, global raw diff2, global combined score, and "
            "fixed-target raw diff2 gates; no fitted tolerance, FSC claim, "
            "or correlation"
        ),
        "fixed_metric": {
            "evaluated_boundaries": len(gates),
            "expected_boundaries": 4,
            "passed_boundaries": sum(int(value) for value in gates.values()),
            "gates": gates,
        },
        "fixed_contract": {
            "native_slurm_job_id": EXPECTED_NATIVE_JOB_ID,
            "recovar_raw_diff2_slurm_job_id": expected_recovar_job_id,
            "native_target_operand_slurm_job_id": expected_operand_job_id,
            "target_gpu_uuid": TARGET_GPU_UUID,
            "expected_support": EXPECTED_SUPPORT,
            "target_recovar_rotation": TARGET_RECOVAR_ROTATION,
            "target_translations": list(TARGET_TRANSLATIONS),
        },
        "inputs": {
            "raw_report": {
                "path": str(raw_report_path.resolve()),
                "sha256": _sha256(raw_report_path),
            },
            "operand_report": {
                "path": str(operand_report_path.resolve()),
                "sha256": _sha256(operand_report_path),
            },
        },
        "shared_inputs": {
            name: raw_report["inputs"][name] for name in SHARED_INPUTS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--operand-report", type=Path, required=True)
    parser.add_argument(
        "--expected-recovar-job-id",
        type=int,
        default=EXPECTED_RECOVAR_JOB_ID,
    )
    parser.add_argument(
        "--expected-operand-job-id",
        type=int,
        default=EXPECTED_OPERAND_JOB_ID,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        raw_report_path=args.raw_report,
        operand_report_path=args.operand_report,
        expected_recovar_job_id=args.expected_recovar_job_id,
        expected_operand_job_id=args.expected_operand_job_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "fixed_metric": report["fixed_metric"],
                "next_causal_boundary": report["next_causal_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
