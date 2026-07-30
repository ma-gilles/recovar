#!/usr/bin/env python3
"""Audit authoritative native K=4 operands in the exact rotation-mapped frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k4_authoritative_native_scores import (
    EXPECTED_CLASS,
    EXPECTED_CURRENT_SIZE,
    EXPECTED_ITERATION,
    EXPECTED_PARTICLE_ID,
    EXPECTED_ROTATIONS,
    EXPECTED_STACK,
    TARGET_GPU_UUID,
    TARGET_RECOVAR_ROTATION,
    TARGET_TRANSLATIONS,
    _read_allocation_table,
    _rotation_permutation,
    _validate_completion,
)
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_operand_capture import (
    load_fine_operand_capture,
    validate_capture,
)
from scripts.validate_relion_fine_score_capture import (
    ACTIVE,
    load_fine_score_capture,
)

SCHEMA = "relion-k4-it2-native-target-operand-audit-v1"
COMPLETION_SCHEMA = "relion_k4_it2_native_target_operand_capture_v1"
PASS_CLASSIFICATION = "native_k4_target_operands_match_sealed_score_rows_bitwise"
NATIVE_SCIENCE_JOB_ID = 11_787_017
EXPECTED_NATIVE_TARGET_ROTATION = 1_210


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_native_target_operands(
    *,
    rotation_frame_exact: bool,
    translations_exact: bool,
    score_raw_diff2_bitwise_exact: bool,
    production_replay_bitwise_exact: bool,
    target_tie_exact: bool,
) -> str:
    gates = {
        "rotation_frame": rotation_frame_exact,
        "translations": translations_exact,
        "score_raw_diff2": score_raw_diff2_bitwise_exact,
        "production_replay": production_replay_bitwise_exact,
        "target_tie": target_tie_exact,
    }
    failures = [name for name, passed in gates.items() if not passed]
    if not failures:
        return PASS_CLASSIFICATION
    return "native_k4_target_operand_mismatch__" + "__".join(failures)


def _validate_target_operand_completion(
    path: Path,
    *,
    expected_job_id: int,
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == COMPLETION_SCHEMA,
        "unexpected native target-operand completion schema",
    )
    _require(
        report.get("status") == "complete",
        "native target-operand capture is incomplete",
    )
    _require(
        int(report.get("slurm_job_id")) == expected_job_id,
        "native target-operand Slurm identity changed",
    )
    _require(
        int(report.get("native_rotation_local"))
        == EXPECTED_NATIVE_TARGET_ROTATION,
        "native target-operand rotation changed",
    )
    _require(
        tuple(int(value) for value in report.get("target_translations", ()))
        == TARGET_TRANSLATIONS,
        "native target-operand translations changed",
    )
    _require(
        report.get("grid_correction") == "unset_default_off"
        and report.get("final_all_data_after_max_iter") == "unset",
        "native target-operand grid/finalization contract changed",
    )
    _require(
        report.get("scorecard_change_admissible") is False,
        "native target-operand completion incorrectly permits a scorecard change",
    )
    return report


def _comparison(
    *,
    factor_path: Path,
    fine_score_path: Path,
    fine_operand_path: Path,
    recovar_pass2_path: Path,
) -> dict[str, Any]:
    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(fine_score_path)
    operand = load_fine_operand_capture(fine_operand_path)
    with np.load(recovar_pass2_path, allow_pickle=False) as archive:
        recovar = {key: np.asarray(archive[key]) for key in archive.files}
    required_recovar = {
        "original_index",
        "class_index",
        "current_size",
        "rotations",
    }
    _require(
        required_recovar.issubset(recovar),
        "RECOVAR pass-2 identity/rotation fields are absent",
    )
    _require(
        int(recovar["original_index"]) == 53_722
        and int(recovar["class_index"]) == 0
        and int(recovar["current_size"]) == EXPECTED_CURRENT_SIZE,
        "RECOVAR pass-2 particle/class/current-size identity changed",
    )
    operand_validation = validate_capture(
        operand,
        expected_stack=EXPECTED_STACK,
        expected_class=EXPECTED_CLASS,
        expected_translations=TARGET_TRANSLATIONS,
    )
    _require(factor.geometry_only, "expected geometry-only BPref capture")
    _require(factor.stack_index == EXPECTED_STACK, "factor stack changed")
    _require(
        score.header[4:8]
        == (
            EXPECTED_ITERATION,
            EXPECTED_CLASS,
            EXPECTED_PARTICLE_ID,
            EXPECTED_STACK,
        ),
        "native fine-score identity changed",
    )
    _require(
        (
            operand.iteration,
            operand.class_one_based,
            operand.particle_id,
            operand.stack_index,
        )
        == (
            EXPECTED_ITERATION,
            EXPECTED_CLASS,
            EXPECTED_PARTICLE_ID,
            EXPECTED_STACK,
        ),
        "native fine-operand identity changed",
    )

    native_rotations = (
        np.asarray(factor.rotations["matrix"], dtype=np.float32)
        .reshape(-1, 3, 3)
        .transpose(0, 2, 1)
    )
    native_to_recovar = _rotation_permutation(
        native_rotations,
        np.asarray(recovar["rotations"], dtype=np.float32),
    )
    inverse_target = np.flatnonzero(
        native_to_recovar == TARGET_RECOVAR_ROTATION
    )
    _require(
        inverse_target.size == 1,
        "target RECOVAR rotation does not have one native match",
    )
    expected_native_rotation = int(inverse_target[0])
    _require(
        expected_native_rotation == EXPECTED_NATIVE_TARGET_ROTATION,
        "exact rotation mapping no longer selects native target row 1210",
    )
    rotation_frame_exact = bool(
        np.all(
            np.asarray(operand.candidates["rotation_local"], dtype=np.int64)
            == expected_native_rotation
        )
    )

    translations = tuple(
        int(value)
        for value in np.asarray(
            operand.candidates["translation_id"],
            dtype=np.int64,
        )
    )
    translations_exact = translations == TARGET_TRANSLATIONS
    active = (score.candidates["flags"] & ACTIVE) != 0
    records = []
    for operand_row, translation in enumerate(TARGET_TRANSLATIONS):
        matches = np.flatnonzero(
            active
            & (
                np.asarray(
                    score.candidates["rotation_local"],
                    dtype=np.int64,
                )
                == expected_native_rotation
            )
            & (
                np.asarray(
                    score.candidates["translation_id"],
                    dtype=np.int64,
                )
                == translation
            )
        )
        _require(
            matches.size == 1,
            f"target translation {translation} does not have one fine-score row",
        )
        score_value = np.float32(
            score.candidates[int(matches[0])]["raw_diff2"]
        )
        production = np.float32(
            operand.candidates[operand_row]["production_raw_diff2"]
        )
        replay = np.float32(
            operand.candidates[operand_row]["replay_raw_diff2"]
        )
        records.append(
            {
                "translation_id": translation,
                "fine_score_raw_diff2": float(score_value),
                "fine_score_raw_diff2_bits": int(score_value.view(np.uint32)),
                "operand_production_raw_diff2": float(production),
                "operand_production_raw_diff2_bits": int(
                    production.view(np.uint32)
                ),
                "operand_replay_raw_diff2": float(replay),
                "operand_replay_raw_diff2_bits": int(replay.view(np.uint32)),
                "score_to_production_bitwise_exact": bool(
                    score_value.view(np.uint32)
                    == production.view(np.uint32)
                ),
                "production_to_replay_bitwise_exact": bool(
                    production.view(np.uint32) == replay.view(np.uint32)
                ),
            }
        )

    score_raw_exact = all(
        row["score_to_production_bitwise_exact"] for row in records
    )
    production_replay_exact = all(
        row["production_to_replay_bitwise_exact"] for row in records
    )
    target_tie_exact = bool(
        records[0]["fine_score_raw_diff2_bits"]
        == records[1]["fine_score_raw_diff2_bits"]
        == records[0]["operand_production_raw_diff2_bits"]
        == records[1]["operand_production_raw_diff2_bits"]
    )
    classification = classify_native_target_operands(
        rotation_frame_exact=rotation_frame_exact,
        translations_exact=translations_exact,
        score_raw_diff2_bitwise_exact=score_raw_exact,
        production_replay_bitwise_exact=production_replay_exact,
        target_tie_exact=target_tie_exact,
    )
    return {
        "classification": classification,
        "accepted": classification == PASS_CLASSIFICATION,
        "rotation_mapping": {
            "count": EXPECTED_ROTATIONS,
            "recovar_target_rotation_row": TARGET_RECOVAR_ROTATION,
            "expected_native_target_rotation_local": (
                expected_native_rotation
            ),
            "operand_rotation_frame_exact": rotation_frame_exact,
        },
        "translations_exact": translations_exact,
        "score_raw_diff2_bitwise_exact": score_raw_exact,
        "production_replay_bitwise_exact": production_replay_exact,
        "target_tie_exact": target_tie_exact,
        "records": records,
        "operand_validation": operand_validation,
    }


def build_report(
    *,
    factor_path: Path,
    fine_score_path: Path,
    fine_operand_path: Path,
    recovar_pass2_path: Path,
    native_completion_path: Path,
    operand_completion_path: Path,
    operand_allocation_path: Path,
    expected_operand_job_id: int,
) -> dict[str, Any]:
    native_completion = _validate_completion(
        native_completion_path,
        expected_job_id=NATIVE_SCIENCE_JOB_ID,
    )
    operand_completion = _validate_target_operand_completion(
        operand_completion_path,
        expected_job_id=expected_operand_job_id,
    )
    allocation = _read_allocation_table(operand_allocation_path)
    comparison = _comparison(
        factor_path=factor_path,
        fine_score_path=fine_score_path,
        fine_operand_path=fine_operand_path,
        recovar_pass2_path=recovar_pass2_path,
    )
    inputs = {
        "factor": factor_path,
        "fine_score": fine_score_path,
        "fine_operand": fine_operand_path,
        "recovar_pass2": recovar_pass2_path,
        "native_completion": native_completion_path,
        "operand_completion": operand_completion_path,
        "operand_allocation": operand_allocation_path,
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": comparison.pop("classification"),
        "accepted": comparison.pop("accepted"),
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed exact-device K4 target-operand provenance gate; "
            "bitwise raw-cost and CUDA-replay checks only; no fitted scale, "
            "sign, correlation, or map acceptance claim"
        ),
        "fixed_contract": {
            "native_slurm_job_id": NATIVE_SCIENCE_JOB_ID,
            "operand_slurm_job_id": expected_operand_job_id,
            "target_gpu_uuid": TARGET_GPU_UUID,
            "expected_rotations": EXPECTED_ROTATIONS,
            "target_recovar_rotation": TARGET_RECOVAR_ROTATION,
            "expected_native_target_rotation": (
                EXPECTED_NATIVE_TARGET_ROTATION
            ),
            "target_translations": list(TARGET_TRANSLATIONS),
        },
        "hardware": {
            "allocation": allocation,
            "target_gpu_present": True,
        },
        "native_completion": {
            "status": native_completion["status"],
            "slurm_job_id": native_completion["slurm_job_id"],
        },
        "operand_completion": {
            "status": operand_completion["status"],
            "slurm_job_id": operand_completion["slurm_job_id"],
        },
        **comparison,
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in inputs.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--fine-operand", type=Path, required=True)
    parser.add_argument("--recovar-pass2", type=Path, required=True)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--operand-completion", type=Path, required=True)
    parser.add_argument("--operand-allocation", type=Path, required=True)
    parser.add_argument("--expected-operand-job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        factor_path=args.factor,
        fine_score_path=args.fine_score,
        fine_operand_path=args.fine_operand,
        recovar_pass2_path=args.recovar_pass2,
        native_completion_path=args.native_completion,
        operand_completion_path=args.operand_completion,
        operand_allocation_path=args.operand_allocation,
        expected_operand_job_id=args.expected_operand_job_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "classification": report["classification"],
            }
        )
    )


if __name__ == "__main__":
    main()
