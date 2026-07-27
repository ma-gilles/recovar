#!/usr/bin/env python3
"""Classify a target-only RELION passive-capture winner flip.

This diagnostic is intentionally non-scorecard evidence.  It requires a
strictly rejected identity-safe inertness report, then tests whether the
control and capture translations are adjacent candidates with exact raw-score
ties in both captured RELION and RECOVAR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.parse_relion_dump_dir import parse_dump_dir

INERTNESS_SCHEMA = "em_relion_iteration1_particle_state_inertness_v3"
REPORT_SCHEMA = "em_relion_capture_target_tie_classification_v1"
MAPPING_MAX_ABS_ANGSTROM = 5e-6
EXPECTED_MISMATCH_FIELD = "rlnOriginYAngst"
EXPECTED_EXACT_FIELDS = (
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnOriginXAngst",
    "rlnClassNumber",
    "rlnMaxValueProbDistribution",
    "rlnNrOfSignificantSamples",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_float(value: Any, *, field: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{field} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _validate_inertness(inertness: dict[str, Any]) -> dict[str, Any]:
    _require(inertness.get("schema") == INERTNESS_SCHEMA, "unknown inertness schema")
    _require(inertness.get("status") == "rejected", "capture inertness must be rejected")
    _require(inertness.get("scorecard_change_admissible") is False, "inertness policy changed")
    strict_gate = inertness.get("strict_gate")
    _require(isinstance(strict_gate, dict), "inertness strict gate is missing")
    _require(strict_gate.get("particle_count_exact") is True, "particle count gate failed")
    _require(strict_gate.get("sampling_perturbation_exact") is True, "perturbation gate failed")
    _require(
        strict_gate.get("all_half_map_fsc_auc_at_least_threshold") is True,
        "half-map FSC-AUC gate failed",
    )
    _require(
        strict_gate.get("all_particle_fields_exact") is False,
        "expected a particle-field rejection",
    )
    _require(inertness.get("perturbation_exact") is True, "perturbation exact flag changed")
    _require(
        inertness.get("particle_count") == inertness.get("expected_particle_count"),
        "particle count does not match expectation",
    )
    fields = inertness.get("fields")
    _require(isinstance(fields, dict), "inertness field results are missing")
    for field in EXPECTED_EXACT_FIELDS:
        result = fields.get(field)
        _require(isinstance(result, dict), f"missing inertness field {field}")
        _require(
            result.get("exact") is True
            and result.get("mismatch_count") == 0
            and float(result.get("max_abs", 0.0)) == 0.0,
            f"inertness field {field} is not exact",
        )
    mismatch = fields.get(EXPECTED_MISMATCH_FIELD)
    _require(isinstance(mismatch, dict), f"missing inertness field {EXPECTED_MISMATCH_FIELD}")
    _require(
        mismatch.get("exact") is False and mismatch.get("mismatch_count") == 1,
        "OriginY must contain exactly one mismatch",
    )
    examples = mismatch.get("mismatch_examples")
    _require(isinstance(examples, list) and len(examples) == 1, "OriginY mismatch identity is missing")
    target = inertness.get("target")
    _require(isinstance(target, dict), "inertness target is missing")
    _require(
        examples[0].get("image_identity") == target.get("image_identity"),
        "OriginY mismatch is not the requested target",
    )
    _require(
        target.get("mismatch_fields") == [EXPECTED_MISMATCH_FIELD],
        "target mismatch fields changed",
    )
    return target


def _translation_index(
    translations_pixels: np.ndarray,
    target_xy_angstrom: np.ndarray,
    *,
    pixel_size: float,
    label: str,
) -> tuple[int, float]:
    translations_angstrom = np.asarray(translations_pixels, dtype=np.float64) * pixel_size
    errors = np.max(np.abs(translations_angstrom - target_xy_angstrom[None, :]), axis=1)
    index = int(np.argmin(errors))
    error = float(errors[index])
    _require(
        error <= MAPPING_MAX_ABS_ANGSTROM,
        f"{label} translation does not map within {MAPPING_MAX_ABS_ANGSTROM} Angstrom",
    )
    _require(
        int(np.count_nonzero(errors <= MAPPING_MAX_ABS_ANGSTROM)) == 1,
        f"{label} translation mapping is not unique",
    )
    return index, error


def classify(
    inertness: dict[str, Any],
    recovar: dict[str, np.ndarray],
    relion: dict[str, np.ndarray],
    *,
    pixel_size: float,
) -> dict[str, Any]:
    target = _validate_inertness(inertness)
    _require(pixel_size > 0.0 and math.isfinite(pixel_size), "pixel size must be positive and finite")

    translations = np.asarray(recovar["fine_translations"], dtype=np.float64)
    scores = np.asarray(recovar["scores_pre_prior"], dtype=np.float64)
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    _require(
        translations.ndim == 2 and translations.shape[1] == 2,
        "RECOVAR fine translations must have shape (T, 2)",
    )
    _require(
        scores.ndim == 2 and candidate_mask.shape == scores.shape,
        "RECOVAR scores and mask must have the same two-dimensional shape",
    )
    _require(
        scores.shape[1] == translations.shape[0],
        "RECOVAR score translation axis does not match fine translations",
    )
    original_index = int(np.asarray(recovar["original_index"]).item())
    _require(
        original_index == target.get("original_index_zero_based"),
        "RECOVAR target original index does not match inertness",
    )

    control_fields = target.get("control_fields")
    capture_fields = target.get("capture_fields")
    _require(
        isinstance(control_fields, dict) and isinstance(capture_fields, dict),
        "target control/capture fields are missing",
    )
    control_xy = np.array(
        [
            _finite_float(control_fields["rlnOriginXAngst"], field="control OriginX"),
            _finite_float(control_fields["rlnOriginYAngst"], field="control OriginY"),
        ],
        dtype=np.float64,
    )
    capture_xy = np.array(
        [
            _finite_float(capture_fields["rlnOriginXAngst"], field="capture OriginX"),
            _finite_float(capture_fields["rlnOriginYAngst"], field="capture OriginY"),
        ],
        dtype=np.float64,
    )
    control_trans, control_mapping_error = _translation_index(
        translations, control_xy, pixel_size=pixel_size, label="control"
    )
    capture_trans, capture_mapping_error = _translation_index(
        translations, capture_xy, pixel_size=pixel_size, label="capture"
    )
    _require(control_trans != capture_trans, "control and capture map to the same translation")

    raw_scores = np.asarray(relion["raw_scores"], dtype=np.float64).reshape(-1)
    raw_rot = np.asarray(relion["raw_rot_idx"], dtype=np.int64).reshape(-1)
    raw_trans = np.asarray(relion["raw_trans_idx"], dtype=np.int64).reshape(-1)
    _require(
        raw_scores.size == raw_rot.size == raw_trans.size and raw_scores.size > 0,
        "RELION raw candidate arrays have inconsistent sizes",
    )
    argmax_index = int(np.asarray(relion["argmax_index"]).item())
    _require(0 <= argmax_index < raw_scores.size, "RELION selected index is out of bounds")
    capture_rot = int(raw_rot[argmax_index])
    _require(
        int(raw_trans[argmax_index]) == capture_trans,
        "RELION selected translation does not match captured target metadata",
    )
    control_key = (capture_rot, control_trans)
    capture_key = (capture_rot, capture_trans)

    def relion_row(key: tuple[int, int]) -> int:
        rows = np.flatnonzero((raw_rot == key[0]) & (raw_trans == key[1]))
        _require(rows.size == 1, f"RELION candidate {list(key)} is not unique")
        return int(rows[0])

    control_relion_row = relion_row(control_key)
    capture_relion_row = relion_row(capture_key)
    control_relion_score = float(raw_scores[control_relion_row])
    capture_relion_score = float(raw_scores[capture_relion_row])
    _require(
        control_relion_score == capture_relion_score,
        "captured RELION control/capture candidates are not an exact raw-score tie",
    )
    _require(
        raw_scores[argmax_index] == capture_relion_score,
        "RELION selected score does not match the captured candidate",
    )

    for key in (control_key, capture_key):
        _require(
            0 <= key[0] < scores.shape[0] and 0 <= key[1] < scores.shape[1],
            f"RECOVAR candidate {list(key)} is out of bounds",
        )
        _require(candidate_mask[key], f"RECOVAR candidate {list(key)} is not in support")
        _require(math.isfinite(float(scores[key])), f"RECOVAR candidate {list(key)} is non-finite")
    control_recovar_score = float(scores[control_key])
    capture_recovar_score = float(scores[capture_key])
    _require(
        control_recovar_score == capture_recovar_score,
        "RECOVAR control/capture candidates are not an exact raw-score tie",
    )
    masked_scores = np.where(candidate_mask, scores, -np.inf)
    recovar_flat_winner = int(np.argmax(masked_scores.reshape(-1)))
    recovar_winner = tuple(int(value) for value in np.unravel_index(recovar_flat_winner, scores.shape))
    _require(
        recovar_winner == capture_key,
        "RECOVAR first-index winner does not match the capture candidate",
    )

    delta_xy = capture_xy - control_xy
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "classification": "observer_sensitive_exact_tie_winner_flip",
        "classification_rule": (
            "strict passive capture changed only the requested target translation; "
            "control and capture map uniquely to supported adjacent candidates; "
            "captured RELION and RECOVAR score both candidates exactly equally"
        ),
        "target": {
            "original_index_zero_based": original_index,
            "image_identity": target["image_identity"],
            "control_key": list(control_key),
            "capture_key": list(capture_key),
            "control_xy_angstrom": control_xy.tolist(),
            "capture_xy_angstrom": capture_xy.tolist(),
            "capture_minus_control_xy_angstrom": delta_xy.tolist(),
            "translation_step_distance_angstrom": float(np.linalg.norm(delta_xy)),
            "control_mapping_max_abs_error_angstrom": control_mapping_error,
            "capture_mapping_max_abs_error_angstrom": capture_mapping_error,
        },
        "exact_raw_score_tie": {
            "relion": True,
            "recovar": True,
        },
        "raw_scores": {
            "relion": {
                "at_control_key": control_relion_score,
                "at_capture_key": capture_relion_score,
                "at_control_key_hex": control_relion_score.hex(),
                "at_capture_key_hex": capture_relion_score.hex(),
            },
            "recovar": {
                "at_control_key": control_recovar_score,
                "at_capture_key": capture_recovar_score,
                "at_control_key_hex": control_recovar_score.hex(),
                "at_capture_key_hex": capture_recovar_score.hex(),
            },
        },
        "diagnostic_admissibility": {
            "capture_inertness_passed": False,
            "target_flip_localized": True,
            "reason": (
                "the exact tie localizes the rejected observer effect but cannot "
                "qualify captured RELION as a scorecard oracle"
            ),
        },
        "quality_metric_policy": {
            "map_gate": "shellwise FSC/FSC-AUC from the bound inertness report",
            "correlation_computed": False,
        },
        "scorecard_change_admissible": False,
        "next_step": (
            "Do not change production tie ordering from this rejected capture; "
            "treat the target winner as observer-sensitive at exact score equality."
        ),
    }


def _load_recovar(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _load_relion(dump_dir: Path) -> dict[str, np.ndarray]:
    payload = parse_dump_dir(dump_dir)
    raw_costs = np.asarray(
        payload["pass1_firstiter_cc_exp_Mweight_raw_preonehot"], dtype=np.float64
    )
    return {
        "raw_scores": -raw_costs,
        "raw_rot_idx": np.asarray(payload["pass1_firstiter_cc_raw_rot_idx"]),
        "raw_trans_idx": np.asarray(payload["pass1_firstiter_cc_raw_trans_idx"]),
        "argmax_index": np.asarray(payload["pass1_firstiter_cc_argmin_index"], dtype=np.int64),
    }


def _clean_repo_head(repo: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1"], text=True)
    _require(not status, "analyzer repository is dirty")
    return head


def build_report(
    *,
    repo: Path,
    inertness_json: Path,
    recovar_npz: Path,
    relion_dump_dir: Path,
    pixel_size: float,
) -> dict[str, Any]:
    provenance = relion_dump_dir.parent.parent / "provenance"
    dump_manifests = sorted(provenance.glob("relion_dump_*.sha256"))
    _require(len(dump_manifests) == 1, "expected exactly one RELION dump manifest")
    dump_manifest = dump_manifests[0]
    subprocess.check_call(["sha256sum", "-c", str(dump_manifest)])
    report = classify(
        json.loads(inertness_json.read_text()),
        _load_recovar(recovar_npz),
        _load_relion(relion_dump_dir),
        pixel_size=pixel_size,
    )
    report["inputs"] = {
        "inertness_json": {
            "path": str(inertness_json.resolve()),
            "sha256": _sha256(inertness_json),
        },
        "recovar_npz": {
            "path": str(recovar_npz.resolve()),
            "sha256": _sha256(recovar_npz),
        },
        "relion_dump_manifest": {
            "path": str(dump_manifest.resolve()),
            "sha256": _sha256(dump_manifest),
            "verified": True,
        },
        "analyzer_repo_head": _clean_repo_head(repo),
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--inertness-json", required=True, type=Path)
    parser.add_argument("--recovar-npz", required=True, type=Path)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--pixel-size", required=True, type=float)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = build_report(
        repo=args.repo,
        inertness_json=args.inertness_json,
        recovar_npz=args.recovar_npz,
        relion_dump_dir=args.relion_dump_dir,
        pixel_size=args.pixel_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": report["classification"], **report["target"]}, indent=2))


if __name__ == "__main__":
    main()
