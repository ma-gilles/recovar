#!/usr/bin/env python3
"""Compare authoritative native and RECOVAR K=4 raw fine-pass costs."""

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
    EXPECTED_SUPPORT,
    TARGET_GPU_UUID,
    TARGET_RECOVAR_ROTATION,
    TARGET_TRANSLATIONS,
    _read_allocation_table,
    _rotation_permutation,
    _validate_completion,
    float32_metric,
)
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import (
    ACTIVE,
    load_fine_score_capture,
)

SCHEMA = "relion-k4-it2-raw-diff2-parity-v1"
PASS_CLASSIFICATION = "exact_device_k4_raw_diff2_and_common_min_bitwise_match"
RECOVAR_CAPTURE_HEAD = "ec68f651a4408ed14ed7ebce0ddf3d54a74e0d41"
RECOVAR_CAPTURE_SCHEMA = "recovar-k4-it2-selected-raw-diff2-job-v1"
NATIVE_SCIENCE_JOB_ID = 11_787_017


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> np.float32:
    return np.asarray(np.uint32(value)).view(np.float32)[()]


def classify_raw_diff2_parity(
    *,
    support_exact: bool,
    common_min_bitwise_exact: bool,
    raw_diff2_bitwise_exact: bool,
    centered_pre_prior_bitwise_exact: bool,
    native_target_tied: bool,
    recovar_target_tied: bool,
) -> str:
    gates = {
        "support": support_exact,
        "common_min": common_min_bitwise_exact,
        "raw_diff2": raw_diff2_bitwise_exact,
        "centered_pre_prior": centered_pre_prior_bitwise_exact,
        "native_target_tie": native_target_tied,
        "recovar_target_tie": recovar_target_tied,
    }
    failures = [name for name, passed in gates.items() if not passed]
    if not failures:
        return PASS_CLASSIFICATION
    return "exact_device_k4_raw_diff2_mismatch__" + "__".join(failures)


def _validate_recovar_completion(
    path: Path,
    *,
    expected_job_id: int,
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == RECOVAR_CAPTURE_SCHEMA,
        "unexpected RECOVAR raw-diff2 completion schema",
    )
    _require(report.get("status") == "complete", "RECOVAR raw-diff2 capture is incomplete")
    _require(
        int(report.get("slurm_job_id")) == expected_job_id,
        "RECOVAR raw-diff2 Slurm identity changed",
    )
    _require(
        report.get("integration_head") == RECOVAR_CAPTURE_HEAD,
        "RECOVAR raw-diff2 source commit changed",
    )
    _require(
        report.get("gpu_uuid") == TARGET_GPU_UUID,
        "RECOVAR raw-diff2 GPU UUID changed",
    )
    _require(
        report.get("grid_correction") == "unset_default_off"
        and report.get("final_all_data_after_max_iter") == "unset",
        "RECOVAR raw-diff2 grid/finalization contract changed",
    )
    _require(
        report.get("scorecard_change_admissible") is False,
        "RECOVAR raw-diff2 completion incorrectly permits a scorecard change",
    )
    return report


def _comparison(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_pass2_path: Path,
) -> dict[str, Any]:
    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(fine_score_path)
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
    with np.load(recovar_pass2_path, allow_pickle=False) as archive:
        recovar = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "original_index",
        "class_index",
        "current_size",
        "rotations",
        "candidate_mask",
        "relion_raw_diff2",
        "relion_min_diff2",
    }
    _require(required.issubset(recovar), "RECOVAR raw-diff2 artifact schema is incomplete")
    _require(int(recovar["original_index"]) == 53_722, "RECOVAR particle changed")
    _require(
        int(recovar["class_index"]) == 0
        and int(recovar["current_size"]) == EXPECTED_CURRENT_SIZE,
        "RECOVAR class/current-size identity changed",
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

    candidates = score.candidates
    active = (candidates["flags"] & ACTIVE) != 0
    native_rotation = np.asarray(candidates["rotation_local"], dtype=np.int64)
    native_translation = np.asarray(candidates["translation_id"], dtype=np.int64)
    mapped_rotation = native_to_recovar[native_rotation]
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    recovar_raw_table = np.asarray(recovar["relion_raw_diff2"], dtype=np.float32)
    _require(
        recovar_raw_table.shape == candidate_mask.shape,
        "RECOVAR raw diff2 and candidate-mask shapes differ",
    )
    _require(
        np.all(np.isfinite(recovar_raw_table[candidate_mask])),
        "RECOVAR active raw diff2 contains non-finite values",
    )

    native_support = np.zeros(candidate_mask.shape, dtype=bool)
    native_support[
        mapped_rotation[active],
        native_translation[active],
    ] = True
    _require(
        np.count_nonzero(native_support) == EXPECTED_SUPPORT
        and np.count_nonzero(candidate_mask) == EXPECTED_SUPPORT,
        "fixed active-support denominator changed",
    )
    support_exact = bool(np.array_equal(native_support, candidate_mask))

    native_raw = np.asarray(candidates["raw_diff2"][active], dtype=np.float32)
    recovar_raw = np.asarray(
        recovar_raw_table[
            mapped_rotation[active],
            native_translation[active],
        ],
        dtype=np.float32,
    )
    native_min = _float32_from_bits(score.header[18])
    recovar_min = np.float32(recovar["relion_min_diff2"])
    native_centered = np.subtract(native_min, native_raw, dtype=np.float32)
    recovar_centered = np.subtract(recovar_min, recovar_raw, dtype=np.float32)
    common_min_exact = bool(
        native_min.view(np.uint32) == recovar_min.view(np.uint32)
    )
    raw_metric = float32_metric(native_raw, recovar_raw)
    centered_metric = float32_metric(native_centered, recovar_centered)

    inverse_target = np.flatnonzero(
        native_to_recovar == TARGET_RECOVAR_ROTATION
    )
    _require(
        inverse_target.size == 1,
        "target RECOVAR rotation does not have one native match",
    )
    target_native_rotation = int(inverse_target[0])
    target_records = []
    for translation in TARGET_TRANSLATIONS:
        matches = np.flatnonzero(
            active
            & (native_rotation == target_native_rotation)
            & (native_translation == translation)
        )
        _require(
            matches.size == 1,
            f"target translation {translation} does not have one native row",
        )
        native_value = np.float32(candidates[int(matches[0])]["raw_diff2"])
        recovar_value = np.float32(
            recovar_raw_table[TARGET_RECOVAR_ROTATION, translation]
        )
        target_records.append(
            {
                "translation_id": translation,
                "native_raw_diff2": float(native_value),
                "native_raw_diff2_bits": int(native_value.view(np.uint32)),
                "recovar_raw_diff2": float(recovar_value),
                "recovar_raw_diff2_bits": int(recovar_value.view(np.uint32)),
                "delta_recovar_minus_native": float(
                    np.float64(recovar_value) - np.float64(native_value)
                ),
            }
        )
    native_target_tied = bool(
        target_records[0]["native_raw_diff2_bits"]
        == target_records[1]["native_raw_diff2_bits"]
    )
    recovar_target_tied = bool(
        target_records[0]["recovar_raw_diff2_bits"]
        == target_records[1]["recovar_raw_diff2_bits"]
    )
    classification = classify_raw_diff2_parity(
        support_exact=support_exact,
        common_min_bitwise_exact=common_min_exact,
        raw_diff2_bitwise_exact=raw_metric["bitwise_exact"],
        centered_pre_prior_bitwise_exact=centered_metric["bitwise_exact"],
        native_target_tied=native_target_tied,
        recovar_target_tied=recovar_target_tied,
    )
    return {
        "classification": classification,
        "accepted": classification == PASS_CLASSIFICATION,
        "rotation_mapping": {
            "count": EXPECTED_ROTATIONS,
            "bitwise_exact_bijection": True,
            "native_target_rotation_local": target_native_rotation,
            "recovar_target_rotation_row": TARGET_RECOVAR_ROTATION,
        },
        "support": {
            "native_active_count": int(np.count_nonzero(native_support)),
            "recovar_active_count": int(np.count_nonzero(candidate_mask)),
            "exact": support_exact,
        },
        "common_min_diff2": {
            "native": float(native_min),
            "native_bits": int(native_min.view(np.uint32)),
            "recovar": float(recovar_min),
            "recovar_bits": int(recovar_min.view(np.uint32)),
            "bitwise_exact": common_min_exact,
        },
        "raw_diff2": raw_metric,
        "centered_pre_prior": centered_metric,
        "target": {
            "records": target_records,
            "native_raw_diff2_tied": native_target_tied,
            "recovar_raw_diff2_tied": recovar_target_tied,
        },
    }


def build_report(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_pass2_path: Path,
    native_completion_path: Path,
    recovar_completion_path: Path,
    recovar_allocation_path: Path,
    expected_recovar_job_id: int,
) -> dict[str, Any]:
    native_completion = _validate_completion(
        native_completion_path,
        expected_job_id=NATIVE_SCIENCE_JOB_ID,
    )
    recovar_completion = _validate_recovar_completion(
        recovar_completion_path,
        expected_job_id=expected_recovar_job_id,
    )
    allocation = _read_allocation_table(recovar_allocation_path)
    comparison = _comparison(
        factor_path=factor_path,
        fine_score_path=fine_score_path,
        recovar_pass2_path=recovar_pass2_path,
    )
    inputs = {
        "factor": factor_path,
        "fine_score": fine_score_path,
        "recovar_pass2": recovar_pass2_path,
        "native_completion": native_completion_path,
        "recovar_completion": recovar_completion_path,
        "recovar_allocation": recovar_allocation_path,
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": comparison.pop("classification"),
        "accepted": comparison.pop("accepted"),
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed exact-device K4 iteration-2 raw-cost diagnostic; "
            "bitwise rotation/support/common-min/raw/centered-score gates; "
            "no fitted scale, sign, or correlation; no map acceptance claim"
        ),
        "fixed_contract": {
            "native_slurm_job_id": NATIVE_SCIENCE_JOB_ID,
            "recovar_slurm_job_id": expected_recovar_job_id,
            "target_gpu_uuid": TARGET_GPU_UUID,
            "expected_support": EXPECTED_SUPPORT,
            "expected_rotations": EXPECTED_ROTATIONS,
            "target_recovar_rotation": TARGET_RECOVAR_ROTATION,
            "target_translations": list(TARGET_TRANSLATIONS),
            "recovar_capture_head": RECOVAR_CAPTURE_HEAD,
        },
        "hardware": {
            "allocation": allocation,
            "target_gpu_present": True,
        },
        "native_completion": {
            "status": native_completion["status"],
            "slurm_job_id": native_completion["slurm_job_id"],
        },
        "recovar_completion": {
            "status": recovar_completion["status"],
            "slurm_job_id": recovar_completion["slurm_job_id"],
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
    parser.add_argument("--recovar-pass2", type=Path, required=True)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--recovar-completion", type=Path, required=True)
    parser.add_argument("--recovar-allocation", type=Path, required=True)
    parser.add_argument("--expected-recovar-job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        factor_path=args.factor,
        fine_score_path=args.fine_score,
        recovar_pass2_path=args.recovar_pass2,
        native_completion_path=args.native_completion,
        recovar_completion_path=args.recovar_completion,
        recovar_allocation_path=args.recovar_allocation,
        expected_recovar_job_id=args.expected_recovar_job_id,
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
