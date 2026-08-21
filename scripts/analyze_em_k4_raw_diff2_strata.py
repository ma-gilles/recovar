#!/usr/bin/env python3
"""Stratify exact-device K=4 raw-cost mismatches on the fixed active support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k4_authoritative_native_scores import (
    EXPECTED_ROTATIONS,
    EXPECTED_SUPPORT,
    _rotation_permutation,
)
from scripts.analyze_em_k4_raw_diff2_parity import SCHEMA as RAW_REPORT_SCHEMA
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import (
    ACTIVE,
    load_fine_score_capture,
)

SCHEMA = "relion-k4-it2-raw-diff2-strata-v1"
PASS_CLASSIFICATION = "global_raw_diff2_is_bitwise_exact"
MISMATCH_CLASSIFICATION = (
    "global_raw_diff2_mismatches_stratified_with_fixed_representative"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    _require(values64.ndim == 1 and values64.size > 0, "empty quantile input")
    return {
        "minimum": float(np.min(values64)),
        "p50": float(np.quantile(values64, 0.50)),
        "p95": float(np.quantile(values64, 0.95)),
        "p99": float(np.quantile(values64, 0.99)),
        "maximum": float(np.max(values64)),
    }


def _ordered_float32_bits(values: np.ndarray) -> np.ndarray:
    """Map float32 bit patterns to monotone adjacent integer codes."""

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    sign = (bits & np.uint32(0x80000000)) != 0
    return np.where(
        sign,
        np.bitwise_not(bits),
        bits | np.uint32(0x80000000),
    ).astype(np.uint64)


def summarize_raw_diff2_strata(
    *,
    native_raw: np.ndarray,
    recovar_raw: np.ndarray,
    recovar_rotation: np.ndarray,
    translation: np.ndarray,
    native_candidate_index: np.ndarray,
) -> dict[str, Any]:
    """Return fixed mismatch strata and a deterministic representative row."""

    native = np.asarray(native_raw, dtype=np.float32)
    recovar = np.asarray(recovar_raw, dtype=np.float32)
    rotations = np.asarray(recovar_rotation, dtype=np.int64)
    translations = np.asarray(translation, dtype=np.int64)
    candidate_indices = np.asarray(native_candidate_index, dtype=np.int64)
    shape = native.shape
    _require(
        native.ndim == 1
        and recovar.shape == shape
        and rotations.shape == shape
        and translations.shape == shape
        and candidate_indices.shape == shape,
        "aligned raw-cost arrays must be one-dimensional and equal length",
    )
    _require(native.size > 0, "aligned raw-cost arrays are empty")
    _require(
        np.all(np.isfinite(native)) and np.all(np.isfinite(recovar)),
        "aligned raw costs must be finite",
    )
    _require(
        np.all(native >= 0) and np.all(recovar >= 0),
        "aligned squared-difference costs must be nonnegative",
    )
    _require(
        np.all(rotations >= 0)
        and np.all(translations >= 0)
        and np.all(candidate_indices >= 0),
        "aligned candidate identities must be nonnegative",
    )
    _require(
        np.unique(candidate_indices).size == candidate_indices.size,
        "native candidate indices must be unique",
    )

    mismatch = native.view(np.uint32) != recovar.view(np.uint32)
    mismatch_indices = np.flatnonzero(mismatch)
    delta = recovar.astype(np.float64) - native.astype(np.float64)
    mismatch_delta = delta[mismatch]
    mismatch_abs = np.abs(mismatch_delta)
    native_ordered = _ordered_float32_bits(native)
    recovar_ordered = _ordered_float32_bits(recovar)
    mismatch_ulp = np.abs(
        recovar_ordered[mismatch].astype(np.int64)
        - native_ordered[mismatch].astype(np.int64)
    )
    rotation_ids, rotation_counts = np.unique(
        rotations[mismatch],
        return_counts=True,
    )
    translation_ids, translation_counts = np.unique(
        translations[mismatch],
        return_counts=True,
    )

    report: dict[str, Any] = {
        "active_count": int(native.size),
        "bitwise_match_count": int(native.size - mismatch_indices.size),
        "bitwise_mismatch_count": int(mismatch_indices.size),
        "bitwise_exact": bool(mismatch_indices.size == 0),
        "signed_mismatch_counts": {
            "recovar_lower": int(np.count_nonzero(mismatch_delta < 0)),
            "equal_numeric_nonbitwise": int(
                np.count_nonzero(mismatch_delta == 0)
            ),
            "recovar_higher": int(np.count_nonzero(mismatch_delta > 0)),
        },
        "rotation_strata": {
            "mismatching_rotation_count": int(rotation_ids.size),
            "maximum_mismatches_in_one_rotation": (
                int(np.max(rotation_counts)) if rotation_counts.size else 0
            ),
        },
        "translation_strata": {
            "mismatching_translation_count": int(translation_ids.size),
            "maximum_mismatches_in_one_translation": (
                int(np.max(translation_counts))
                if translation_counts.size
                else 0
            ),
        },
        "absolute_delta_quantiles_nonzero": (
            _quantiles(mismatch_abs) if mismatch_abs.size else None
        ),
        "float32_ulp_distance_quantiles_nonzero": (
            _quantiles(mismatch_ulp) if mismatch_ulp.size else None
        ),
        "representative": None,
    }
    if mismatch_indices.size == 0:
        return report

    # Choose the largest absolute raw-cost residual.  Ties are resolved by
    # the native capture's stable candidate row, never by post-hoc identity.
    maximum_abs = np.max(mismatch_abs)
    tied_positions = mismatch_indices[mismatch_abs == maximum_abs]
    representative_index = int(
        tied_positions[np.argmin(candidate_indices[tied_positions])]
    )
    native_value = native[representative_index]
    recovar_value = recovar[representative_index]
    report["representative"] = {
        "selection_rule": (
            "maximum_absolute_delta_then_lowest_native_candidate_index"
        ),
        "native_candidate_index": int(
            candidate_indices[representative_index]
        ),
        "recovar_rotation_row": int(rotations[representative_index]),
        "translation_id": int(translations[representative_index]),
        "native_raw_diff2": float(native_value),
        "native_raw_diff2_bits": int(native_value.view(np.uint32)),
        "recovar_raw_diff2": float(recovar_value),
        "recovar_raw_diff2_bits": int(recovar_value.view(np.uint32)),
        "delta_recovar_minus_native": float(delta[representative_index]),
        "absolute_delta": float(abs(delta[representative_index])),
        "float32_ulp_distance": int(
            abs(
                int(recovar_ordered[representative_index])
                - int(native_ordered[representative_index])
            )
        ),
    }
    return report


def _validated_input(raw_report: dict[str, Any], name: str) -> Path:
    record = raw_report.get("inputs", {}).get(name, {})
    path_text = record.get("path")
    expected_sha256 = record.get("sha256")
    _require(
        isinstance(path_text, str)
        and bool(path_text)
        and isinstance(expected_sha256, str)
        and len(expected_sha256) == 64,
        f"raw report does not bind input {name}",
    )
    path = Path(path_text).resolve()
    _require(path.is_file(), f"raw report input {name} is missing")
    _require(
        _sha256(path) == expected_sha256,
        f"raw report input {name} hash changed",
    )
    return path


def build_report(*, raw_report_path: Path) -> dict[str, Any]:
    raw_report = json.loads(raw_report_path.read_text())
    _require(
        raw_report.get("schema") == RAW_REPORT_SCHEMA,
        "unexpected K4 raw-score report schema",
    )
    _require(
        raw_report.get("status") == "complete"
        and raw_report.get("classification_ready") is True,
        "K4 raw-score report is incomplete",
    )
    _require(
        raw_report.get("support", {}).get("exact") is True,
        "K4 raw-score support is not exact",
    )
    factor_path = _validated_input(raw_report, "factor")
    fine_score_path = _validated_input(raw_report, "fine_score")
    recovar_pass2_path = _validated_input(raw_report, "recovar_pass2")

    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(fine_score_path)
    with np.load(recovar_pass2_path, allow_pickle=False) as archive:
        recovar = {key: np.asarray(archive[key]) for key in archive.files}
    _require(
        {"rotations", "candidate_mask", "relion_raw_diff2"}.issubset(recovar),
        "RECOVAR raw-cost artifact schema is incomplete",
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
    active_candidate_indices = np.flatnonzero(active)
    native_rotation = np.asarray(
        candidates["rotation_local"][active],
        dtype=np.int64,
    )
    translation = np.asarray(
        candidates["translation_id"][active],
        dtype=np.int64,
    )
    mapped_rotation = native_to_recovar[native_rotation]
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    recovar_raw_table = np.asarray(
        recovar["relion_raw_diff2"],
        dtype=np.float32,
    )
    _require(
        candidate_mask.shape == recovar_raw_table.shape,
        "RECOVAR candidate mask and raw-cost table differ",
    )
    native_support = np.zeros(candidate_mask.shape, dtype=bool)
    native_support[mapped_rotation, translation] = True
    _require(
        int(np.count_nonzero(native_support)) == EXPECTED_SUPPORT
        and int(np.count_nonzero(candidate_mask)) == EXPECTED_SUPPORT
        and np.array_equal(native_support, candidate_mask),
        "fixed exact active support changed",
    )
    _require(
        native_to_recovar.size == EXPECTED_ROTATIONS,
        "fixed rotation denominator changed",
    )

    strata = summarize_raw_diff2_strata(
        native_raw=np.asarray(candidates["raw_diff2"][active]),
        recovar_raw=recovar_raw_table[mapped_rotation, translation],
        recovar_rotation=mapped_rotation,
        translation=translation,
        native_candidate_index=active_candidate_indices,
    )
    _require(
        strata["active_count"] == EXPECTED_SUPPORT,
        "fixed active denominator changed",
    )
    raw_metric = raw_report.get("raw_diff2", {})
    _require(
        raw_metric.get("count") == EXPECTED_SUPPORT
        and raw_metric.get("bitwise_mismatch_count")
        == strata["bitwise_mismatch_count"]
        and raw_metric.get("bitwise_exact") is strata["bitwise_exact"],
        "raw mismatch strata do not replay the parent raw-score report",
    )
    accepted = bool(strata["bitwise_exact"])
    classification = (
        PASS_CLASSIFICATION if accepted else MISMATCH_CLASSIFICATION
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": classification,
        "accepted": accepted,
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed exact-device K4 iteration-2 active-support raw-cost "
            "stratification; bitwise mismatch denominator, signed counts, "
            "absolute-delta quantiles, rotation/translation strata, and "
            "maximum-absolute-delta representative with stable row tie-break; "
            "no fitted tolerance, scale, sign, FSC claim, or correlation"
        ),
        "fixed_contract": {
            "expected_support": EXPECTED_SUPPORT,
            "expected_rotations": EXPECTED_ROTATIONS,
            "raw_report_schema": RAW_REPORT_SCHEMA,
        },
        "strata": strata,
        "inputs": {
            "raw_report": {
                "path": str(raw_report_path.resolve()),
                "sha256": _sha256(raw_report_path),
            },
            "factor": raw_report["inputs"]["factor"],
            "fine_score": raw_report["inputs"]["fine_score"],
            "recovar_pass2": raw_report["inputs"]["recovar_pass2"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(raw_report_path=args.raw_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "classification": report["classification"],
                "mismatches": report["strata"]["bitwise_mismatch_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
