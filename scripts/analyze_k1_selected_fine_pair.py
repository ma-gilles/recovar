#!/usr/bin/env python3
"""Compare two mapped native/RECOVAR fine rotations at every translation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .validate_relion_bpref_factor_capture import load_factor_capture
    from .validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture
else:
    from validate_relion_bpref_factor_capture import load_factor_capture  # type: ignore[no-redef]
    from validate_relion_fine_score_capture import (  # type: ignore[no-redef]
        ACTIVE,
        load_fine_score_capture,
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


def _parse_mapping(value: str) -> tuple[int, int]:
    try:
        native, recovar = value.split(":", maxsplit=1)
        return int(native), int(recovar)
    except ValueError as error:
        raise argparse.ArgumentTypeError("rotation mapping must be NATIVE:RECOVAR") from error


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape and left.size > 0, "metric arrays differ in shape or are empty")
    delta = right.astype(np.float64) - left.astype(np.float64)
    denominator = float(np.linalg.norm(left.astype(np.float64)))
    return {
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2_over_native": float(np.linalg.norm(delta) / denominator) if denominator else 0.0,
    }


def analyze(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_path: Path,
    mappings: list[tuple[int, int]],
    translation: int,
) -> dict[str, object]:
    _require(len(mappings) == 2, "exactly two rotation mappings are required")
    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(fine_score_path)
    _require(factor.stack_index == score.stack_index, "native particle identity differs")
    with np.load(recovar_path, allow_pickle=False) as archive:
        recovar = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "schema",
        "original_index",
        "rotation_rows_global",
        "rotations",
        "candidate_mask",
        "scores_pre_prior",
        "scores_with_prior",
        "probs",
        "rotation_log_prior",
        "translation_log_prior",
        "posterior_max",
    }
    _require(required <= set(recovar), f"RECOVAR dump misses {sorted(required - set(recovar))}")
    _require(str(recovar["schema"]) == "recovar.em.k1_pass2_selected_rotations.v1", "schema changed")
    global_rows = np.asarray(recovar["rotation_rows_global"], dtype=np.int64)
    _require(global_rows.size == 2, "RECOVAR dump does not contain exactly two rotations")
    _require(0 <= translation < recovar["candidate_mask"].shape[1], "translation is outside dump")

    native_active = score.candidates[(score.candidates["flags"] & ACTIVE) != 0]
    native_weight_sum = float(np.sum(native_active["post_exponent_weight"], dtype=np.float64))
    _require(native_weight_sum > 0.0, "native posterior sum is invalid")
    native_probability = np.asarray(native_active["post_exponent_weight"], dtype=np.float64) / native_weight_sum
    _require("relion_min_diff2" in recovar, "RECOVAR dump misses the raw-score minimum")
    native_min_diff2 = np.asarray(np.uint32(score.header[18])).view(np.float32).item()
    recovar_min_diff2 = np.float32(np.asarray(recovar["relion_min_diff2"]).item())

    rows = []
    for native_rotation, recovar_global_row in mappings:
        selected_rows = np.flatnonzero(global_rows == recovar_global_row)
        _require(selected_rows.size == 1, f"RECOVAR rotation row {recovar_global_row} is absent or duplicated")
        recovar_row = int(selected_rows[0])
        native_rows = np.flatnonzero(
            (native_active["rotation_local"] == native_rotation)
            & (native_active["translation_id"] == translation)
        )
        _require(native_rows.size == 1, f"native tuple ({native_rotation}, {translation}) is absent or duplicated")
        native_row = int(native_rows[0])
        native_candidate = native_active[native_row]
        _require(bool(recovar["candidate_mask"][recovar_row, translation]), "RECOVAR tuple is masked")
        native_matrix = np.asarray(factor.rotations["matrix"][native_rotation], dtype=np.float32).reshape(3, 3)
        recovar_matrix = np.asarray(recovar["rotations"][recovar_row], dtype=np.float32)
        matrix_error = min(
            float(np.max(np.abs(native_matrix - recovar_matrix))),
            float(np.max(np.abs(native_matrix.T - recovar_matrix))),
        )
        _require(matrix_error == 0.0, "rotation matrix differs")
        native_raw = float(np.float32(native_candidate["raw_diff2"]))
        recovar_centered_raw = np.float32(recovar["scores_pre_prior"][recovar_row, translation])
        recovar_raw = float(
            np.subtract(recovar_min_diff2, recovar_centered_raw, dtype=np.float32)
        )
        rows.append(
            {
                "native_rotation_local": native_rotation,
                "recovar_rotation_row_global": recovar_global_row,
                "recovar_rotation_row_in_dump": recovar_row,
                "translation": translation,
                "rotation_matrix_max_abs": matrix_error,
                "native_raw_diff2": native_raw,
                "recovar_raw_diff2": recovar_raw,
                "raw_diff2_delta_recovar_minus_native": recovar_raw - native_raw,
                "native_rotation_log_prior": float(np.float32(native_candidate["orientation_log_prior"])),
                "recovar_rotation_log_prior": float(np.float32(recovar["rotation_log_prior"][recovar_row])),
                "native_translation_log_prior": float(np.float32(native_candidate["translation_log_prior"])),
                "recovar_translation_log_prior": float(np.float32(recovar["translation_log_prior"][translation])),
                "native_combined_preexponent": float(np.float32(native_candidate["combined_preexponent"])),
                "recovar_score_with_prior": float(np.float32(recovar["scores_with_prior"][recovar_row, translation])),
                "native_probability": float(native_probability[native_row]),
                "recovar_probability": float(recovar["probs"][recovar_row, translation]),
            }
        )

    first, second = rows
    native_raw_margin = second["native_raw_diff2"] - first["native_raw_diff2"]
    recovar_raw_margin = second["recovar_raw_diff2"] - first["recovar_raw_diff2"]
    native_log_margin = second["native_combined_preexponent"] - first["native_combined_preexponent"]
    recovar_log_margin = second["recovar_score_with_prior"] - first["recovar_score_with_prior"]
    native_probabilities = np.asarray([row["native_probability"] for row in rows])
    recovar_probabilities = np.asarray([row["recovar_probability"] for row in rows])
    return {
        "schema": "recovar.em.k1_selected_fine_pair.v1",
        "status": "complete",
        "identity": {
            "stack_index_one_based": factor.stack_index,
            "original_index_zero_based": int(np.asarray(recovar["original_index"]).item()),
            "translation": translation,
        },
        "raw_score_minimum": {
            "native": float(native_min_diff2),
            "recovar": float(recovar_min_diff2),
            "delta_recovar_minus_native": float(recovar_min_diff2 - native_min_diff2),
            "native_float32_bits": int(np.asarray(native_min_diff2, dtype=np.float32).view(np.uint32)),
            "recovar_float32_bits": int(np.asarray(recovar_min_diff2, dtype=np.float32).view(np.uint32)),
        },
        "rows": rows,
        "pair_margins": {
            "native_raw_diff2_second_minus_first": native_raw_margin,
            "recovar_raw_diff2_second_minus_first": recovar_raw_margin,
            "raw_diff2_margin_delta": recovar_raw_margin - native_raw_margin,
            "native_log_second_minus_first": native_log_margin,
            "recovar_log_second_minus_first": recovar_log_margin,
            "log_margin_delta": recovar_log_margin - native_log_margin,
        },
        "probability_metric": _metric(native_probabilities, recovar_probabilities),
        "native_posterior_max": float(np.max(native_probability)),
        "recovar_posterior_max": float(np.asarray(recovar["posterior_max"]).item()),
        "artifacts": {
            "factor": str(factor_path.resolve()),
            "factor_sha256": _sha256(factor_path),
            "fine_score": str(fine_score_path.resolve()),
            "fine_score_sha256": _sha256(fine_score_path),
            "recovar": str(recovar_path.resolve()),
            "recovar_sha256": _sha256(recovar_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--rotation-map", type=_parse_mapping, action="append", required=True)
    parser.add_argument("--translation", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        factor_path=args.factor,
        fine_score_path=args.fine_score,
        recovar_path=args.recovar,
        mappings=args.rotation_map,
        translation=args.translation,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
