#!/usr/bin/env python3
"""Audit the exact-device authoritative K=4 iteration-2 native score capture.

This gate compares RELION's native fine-score table to the frozen RECOVAR
pass-2 table after constructing a bitwise-exact rotation permutation.  It is
a fixed-target operand diagnostic, not a map-quality or scorecard gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import (
    ACTIVE,
    load_fine_score_capture,
)

SCHEMA = "relion-k4-it2-exact-device-native-score-audit-v1"
PASS_CLASSIFICATION = "exact_device_authoritative_native_and_recovar_target_match_after_exact_rotation_permutation"
TARGET_GPU_UUID = "GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30"
RECOVAR_PASS2_SHA256 = "3c4c566b6f2fce613f4d5869d2d3ccf53a2bcd1b3c26e5a32138588464049485"
EXPECTED_SUPPORT = 109_184
EXPECTED_ROTATIONS = 2_968
EXPECTED_STACK = 53_723
EXPECTED_PARTICLE_ID = 48_584
EXPECTED_CLASS = 1
EXPECTED_ITERATION = 2
EXPECTED_CURRENT_SIZE = 38
TARGET_RECOVAR_ROTATION = 2_626
TARGET_TRANSLATIONS = (80, 82)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def float32_metric(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    """Return fixed scale-sensitive float32 comparison metrics."""

    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    _require(lhs.shape == rhs.shape, "float32 metric shapes differ")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    denominator = max(
        float(np.linalg.norm(rhs.astype(np.float64))),
        float(np.finfo(np.float64).tiny),
    )
    return {
        "count": int(lhs.size),
        "bitwise_exact": bool(np.array_equal(lhs.view(np.uint32), rhs.view(np.uint32))),
        "bitwise_mismatch_count": int(np.count_nonzero(lhs.view(np.uint32) != rhs.view(np.uint32))),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "relative_l2_over_rhs": float(np.linalg.norm(delta) / denominator),
    }


def classify_target_parity(
    *,
    support_exact: bool,
    winner_exact: bool,
    max_tie_key_sets_exact: bool,
    native_raw_diff2_tied: bool,
    recovar_scores_tied: bool,
    cross_engine_target_scores_bitwise_exact: bool,
) -> str:
    """Classify the six predeclared exact target gates."""

    gates = {
        "support": support_exact,
        "winner": winner_exact,
        "max_ties": max_tie_key_sets_exact,
        "native_raw_tie": native_raw_diff2_tied,
        "recovar_tie": recovar_scores_tied,
        "cross_engine_target": cross_engine_target_scores_bitwise_exact,
    }
    failures = [name for name, passed in gates.items() if not passed]
    if not failures:
        return PASS_CLASSIFICATION
    if len(failures) == 1:
        return f"exact_device_k4_target_{failures[0]}_mismatch"
    return "exact_device_k4_target_mixed_mismatch__" + "__".join(failures)


def _read_allocation_table(path: Path) -> list[dict[str, str]]:
    rows = []
    for raw_line in path.read_text().splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        _require(len(fields) == 3, f"invalid allocation row: {raw_line}")
        rows.append({"uuid": fields[0], "name": fields[1], "pci_bus_id": fields[2]})
    _require(len(rows) == 2, "exact-device job did not receive exactly two GPUs")
    targets = [row for row in rows if row["uuid"] == TARGET_GPU_UUID]
    _require(len(targets) == 1, "required exact GPU UUID is absent or duplicated")
    _require("A100" in targets[0]["name"], "required exact GPU is not an A100")
    return rows


def _validate_completion(path: Path, *, expected_job_id: int) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == "relion_k4_it2_authoritative_native_capture_v1",
        "unexpected science-completion schema",
    )
    _require(report.get("status") == "complete", "science is incomplete")
    _require(
        int(report.get("slurm_job_id")) == expected_job_id,
        "science-completion job identity changed",
    )
    _require(
        report.get("sampling_perturbation") == 0.27053284645080566,
        "authoritative perturbation changed",
    )
    _require(
        report.get("scorecard_change_admissible") is False,
        "science completion incorrectly permits a scorecard change",
    )
    _require(
        report.get("grid_correction") == "unset_default_off" and report.get("final_all_data_after_max_iter") == "unset",
        "grid/finalization contract changed",
    )
    return report


def _validate_state(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == "relion_k4_it2_authoritative_translation_grid_validation_v1",
        "unexpected translation-grid validation schema",
    )
    _require(
        report.get("status") == "accepted"
        and report.get("classification") == "native_capture_matches_uninterrupted_iteration2_translation_grid",
        "authoritative translation-grid validation did not pass",
    )
    _require(
        report.get("translation_ids") == list(TARGET_TRANSLATIONS),
        "target translation IDs changed",
    )
    _require(
        float(report.get("max_abs_pixels")) <= 2.0e-6,
        "native translation grid exceeds the fixed pixel gate",
    )
    _require(
        report.get("phase_capture_sha256") == RECOVAR_PASS2_SHA256,
        "frozen RECOVAR pass-2 hash changed in state validation",
    )
    return report


def _rotation_permutation(
    native_rotations: np.ndarray,
    recovar_rotations: np.ndarray,
) -> np.ndarray:
    native = np.asarray(native_rotations, dtype=np.float32)
    recovar = np.asarray(recovar_rotations, dtype=np.float32)
    _require(native.shape == recovar.shape, "rotation table shapes differ")
    _require(
        native.shape == (EXPECTED_ROTATIONS, 3, 3),
        "fixed K4 rotation topology changed",
    )
    lookup = {matrix.tobytes(): index for index, matrix in enumerate(recovar)}
    _require(
        len(lookup) == EXPECTED_ROTATIONS,
        "RECOVAR rotation matrices are not unique",
    )
    permutation = np.asarray(
        [lookup.get(matrix.tobytes(), -1) for matrix in native],
        dtype=np.int64,
    )
    _require(
        np.all(permutation >= 0),
        "one or more native rotations lack a bitwise RECOVAR match",
    )
    _require(
        np.unique(permutation).size == EXPECTED_ROTATIONS,
        "native-to-RECOVAR rotation mapping is not bijective",
    )
    return permutation


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
        "fine-score identity changed",
    )
    _require(
        _sha256(recovar_pass2_path) == RECOVAR_PASS2_SHA256,
        "frozen RECOVAR pass-2 artifact hash changed",
    )
    with np.load(recovar_pass2_path, allow_pickle=False) as archive:
        recovar = {key: np.asarray(archive[key]) for key in archive.files}
    _require(int(recovar["original_index"]) == 53_722, "RECOVAR particle changed")
    _require(
        int(recovar["class_index"]) == 0 and int(recovar["current_size"]) == EXPECTED_CURRENT_SIZE,
        "RECOVAR class/current-size identity changed",
    )

    native_rotations = np.asarray(factor.rotations["matrix"], dtype=np.float32).reshape(-1, 3, 3).transpose(0, 2, 1)
    recovar_rotations = np.asarray(recovar["rotations"], dtype=np.float32)
    native_to_recovar = _rotation_permutation(
        native_rotations,
        recovar_rotations,
    )

    candidates = score.candidates
    active = (candidates["flags"] & ACTIVE) != 0
    active_indices = np.flatnonzero(active)
    native_rotation = np.asarray(candidates["rotation_local"], dtype=np.int64)
    native_translation = np.asarray(candidates["translation_id"], dtype=np.int64)
    _require(
        np.all((native_rotation >= 0) & (native_rotation < EXPECTED_ROTATIONS)),
        "native rotation index is out of bounds",
    )
    mapped_rotation = native_to_recovar[native_rotation]
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    _require(
        np.all((native_translation >= 0) & (native_translation < candidate_mask.shape[1])),
        "native translation index is out of bounds",
    )
    mapped_keys = np.column_stack((mapped_rotation[active], native_translation[active]))
    _require(
        np.unique(mapped_keys, axis=0).shape[0] == mapped_keys.shape[0],
        "mapped active native candidates are not unique",
    )
    native_support = np.zeros(candidate_mask.shape, dtype=bool)
    native_support[mapped_rotation[active], native_translation[active]] = True
    _require(
        np.count_nonzero(native_support) == EXPECTED_SUPPORT and np.count_nonzero(candidate_mask) == EXPECTED_SUPPORT,
        "fixed active-support denominator changed",
    )
    intersection = int(np.count_nonzero(native_support & candidate_mask))
    union = int(np.count_nonzero(native_support | candidate_mask))
    support_exact = bool(np.array_equal(native_support, candidate_mask))

    native_combined = np.asarray(candidates["combined_preexponent"][active], dtype=np.float32)
    recovar_combined = np.asarray(
        recovar["scores_with_prior"][
            mapped_rotation[active],
            native_translation[active],
        ],
        dtype=np.float32,
    )
    native_orientation_prior = np.asarray(candidates["orientation_log_prior"][active], dtype=np.float32)
    recovar_orientation_prior = np.asarray(recovar["rotation_log_prior"][mapped_rotation[active]], dtype=np.float32)
    native_translation_prior = np.asarray(candidates["translation_log_prior"][active], dtype=np.float32)
    recovar_translation_prior = np.asarray(
        recovar["translation_log_prior"][native_translation[active]],
        dtype=np.float32,
    )

    native_winner_candidate = int(active_indices[np.argmax(native_combined)])
    native_winner_mapped = (
        int(mapped_rotation[native_winner_candidate]),
        int(native_translation[native_winner_candidate]),
    )
    recovar_scores = np.asarray(recovar["scores_with_prior"], dtype=np.float32)
    recovar_winner = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(np.where(candidate_mask, recovar_scores, -np.inf))),
            candidate_mask.shape,
        )
    )
    winner_exact = native_winner_mapped == recovar_winner

    native_max = np.max(native_combined)
    native_max_keys = {tuple(int(value) for value in key) for key in mapped_keys[native_combined == native_max]}
    recovar_max = np.max(recovar_scores[candidate_mask])
    recovar_max_keys = {
        tuple(int(value) for value in row) for row in np.argwhere(candidate_mask & (recovar_scores == recovar_max))
    }
    max_tie_key_sets_exact = native_max_keys == recovar_max_keys

    inverse_target = np.flatnonzero(native_to_recovar == TARGET_RECOVAR_ROTATION)
    _require(
        inverse_target.size == 1,
        "target RECOVAR rotation does not have one native match",
    )
    target_native_rotation = int(inverse_target[0])
    target_records = []
    for translation in TARGET_TRANSLATIONS:
        matches = np.flatnonzero(
            active & (native_rotation == target_native_rotation) & (native_translation == translation)
        )
        _require(
            matches.size == 1,
            f"target translation {translation} does not have one native row",
        )
        row = candidates[int(matches[0])]
        native_raw = np.asarray(row["raw_diff2"], dtype=np.float32)
        native_total = np.asarray(row["combined_preexponent"], dtype=np.float32)
        recovar_total = np.asarray(
            recovar_scores[TARGET_RECOVAR_ROTATION, translation],
            dtype=np.float32,
        )
        target_records.append(
            {
                "translation_id": translation,
                "native_sparse_index": int(row["sparse_index"]),
                "native_rotation_local": target_native_rotation,
                "recovar_rotation_row": TARGET_RECOVAR_ROTATION,
                "native_raw_diff2": float(native_raw),
                "native_raw_diff2_bits": int(native_raw.view(np.uint32)),
                "native_combined_preexponent": float(native_total),
                "native_combined_preexponent_bits": int(native_total.view(np.uint32)),
                "recovar_score_with_prior": float(recovar_total),
                "recovar_score_with_prior_bits": int(recovar_total.view(np.uint32)),
            }
        )
    native_raw_tied = target_records[0]["native_raw_diff2_bits"] == target_records[1]["native_raw_diff2_bits"]
    recovar_tied = (
        target_records[0]["recovar_score_with_prior_bits"] == target_records[1]["recovar_score_with_prior_bits"]
    )
    target_cross_exact = all(
        row["native_combined_preexponent_bits"] == row["recovar_score_with_prior_bits"] for row in target_records
    )
    classification = classify_target_parity(
        support_exact=support_exact,
        winner_exact=winner_exact,
        max_tie_key_sets_exact=max_tie_key_sets_exact,
        native_raw_diff2_tied=native_raw_tied,
        recovar_scores_tied=recovar_tied,
        cross_engine_target_scores_bitwise_exact=target_cross_exact,
    )
    return {
        "classification": classification,
        "accepted": classification == PASS_CLASSIFICATION,
        "rotation_mapping": {
            "count": int(native_to_recovar.size),
            "bitwise_exact_bijection": True,
            "native_target_rotation_local": target_native_rotation,
            "recovar_target_rotation_row": TARGET_RECOVAR_ROTATION,
            "native_row_2626_maps_to_recovar_row": int(native_to_recovar[2626]),
        },
        "support": {
            "native_active_count": int(np.count_nonzero(native_support)),
            "recovar_active_count": int(np.count_nonzero(candidate_mask)),
            "intersection": intersection,
            "union": union,
            "jaccard": float(intersection / union),
            "exact": support_exact,
        },
        "scores": {
            "combined_preexponent_vs_scores_with_prior": float32_metric(
                native_combined,
                recovar_combined,
            ),
            "orientation_log_prior": float32_metric(
                native_orientation_prior,
                recovar_orientation_prior,
            ),
            "translation_log_prior": float32_metric(
                native_translation_prior,
                recovar_translation_prior,
            ),
        },
        "winner": {
            "native_rotation_local": int(candidates[native_winner_candidate]["rotation_local"]),
            "native_mapped_recovar_key": list(native_winner_mapped),
            "recovar_key": list(recovar_winner),
            "exact": winner_exact,
            "native_max_tie_count": len(native_max_keys),
            "recovar_max_tie_count": len(recovar_max_keys),
            "max_tie_key_sets_exact": max_tie_key_sets_exact,
        },
        "target": {
            "records": target_records,
            "native_raw_diff2_tied": native_raw_tied,
            "recovar_scores_tied": recovar_tied,
            "cross_engine_combined_scores_bitwise_exact": target_cross_exact,
        },
    }


def build_report(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_pass2_path: Path,
    state_validation_path: Path,
    allocation_table_path: Path,
    science_completion_path: Path,
    expected_job_id: int,
) -> dict[str, Any]:
    """Build the exact-device fixed-target K4 report."""

    allocation = _read_allocation_table(allocation_table_path)
    completion = _validate_completion(
        science_completion_path,
        expected_job_id=expected_job_id,
    )
    state = _validate_state(state_validation_path)
    comparison = _comparison(
        factor_path=factor_path,
        fine_score_path=fine_score_path,
        recovar_pass2_path=recovar_pass2_path,
    )
    inputs = {
        "factor": factor_path,
        "fine_score": fine_score_path,
        "recovar_pass2": recovar_pass2_path,
        "state_validation": state_validation_path,
        "allocation_table": allocation_table_path,
        "science_completion": science_completion_path,
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": comparison.pop("classification"),
        "accepted": comparison.pop("accepted"),
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed exact-device K4 iteration-2 candidate diagnostic; "
            "bitwise rotation mapping, exact support/winner/tie sets, and "
            "bitwise target scores; no map acceptance claim; no correlation"
        ),
        "fixed_contract": {
            "slurm_job_id": expected_job_id,
            "target_gpu_uuid": TARGET_GPU_UUID,
            "expected_support": EXPECTED_SUPPORT,
            "expected_rotations": EXPECTED_ROTATIONS,
            "target_recovar_rotation": TARGET_RECOVAR_ROTATION,
            "target_translations": list(TARGET_TRANSLATIONS),
            "recovar_pass2_sha256": RECOVAR_PASS2_SHA256,
        },
        "hardware": {
            "allocation": allocation,
            "target_gpu_present": True,
        },
        "science_completion": completion,
        "state_validation": {
            "classification": state["classification"],
            "max_abs_pixels": state["max_abs_pixels"],
        },
        **comparison,
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in inputs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--recovar-pass2", type=Path, required=True)
    parser.add_argument("--state-validation", type=Path, required=True)
    parser.add_argument("--allocation-table", type=Path, required=True)
    parser.add_argument("--science-completion", type=Path, required=True)
    parser.add_argument("--expected-job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        factor_path=args.factor,
        fine_score_path=args.fine_score,
        recovar_pass2_path=args.recovar_pass2,
        state_validation_path=args.state_validation,
        allocation_table_path=args.allocation_table,
        science_completion_path=args.science_completion,
        expected_job_id=args.expected_job_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"accepted": report["accepted"], "classification": report["classification"]}))


if __name__ == "__main__":
    main()
