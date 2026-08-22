#!/usr/bin/env python3
"""Test whether clean RELION coarse operands can reach native raw scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_coarse_operand_boundary_v3 import _atomic_add_log_score_values
from scripts.analyze_k1_native_coarse_boundary import load_native_coarse_capture
from scripts.validate_relion_coarse_operand_capture import (
    load_artifact as load_operands,
    replay_production_lanes,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(operands_path: Path, native_coarse_path: Path, initial_diff2: float) -> dict[str, object]:
    operands = load_operands(operands_path)
    native = load_native_coarse_capture(native_coarse_path)
    if int(native.header[2]) != operands.stack_index or int(native.header[3]) != operands.part_id:
        raise ValueError("operand and native-coarse particle identities differ")
    rotation_count = int(native.header[4])
    translation_count = int(native.header[5])
    if translation_count != int(operands.header[14]):
        raise ValueError("translation topology differs")
    native_raw = native.raw_diff2.reshape(rotation_count, translation_count)
    lane_partials = replay_production_lanes(operands)
    initial = np.float32(initial_diff2)
    rows: list[dict[str, object]] = []
    reachable = 0
    for operand_row, rotation_key in enumerate(operands.rotation_keys):
        for translation in range(translation_count):
            legal_log_scores = _atomic_add_log_score_values(
                lane_partials[operand_row],
                translation_count=translation_count,
                translation=translation,
                initial_diff2=initial,
            )
            native_log_score = np.float32(-native_raw[int(rotation_key), translation])
            exact = bool(np.any(legal_log_scores == native_log_score))
            reachable += int(exact)
            rows.append(
                {
                    "rotation_key": int(rotation_key),
                    "translation": int(translation),
                    "native_log_score": float(native_log_score),
                    "native_log_score_bits": f"0x{int(native_log_score.view(np.uint32)):08x}",
                    "legal_min": float(np.min(legal_log_scores)),
                    "legal_max": float(np.max(legal_log_scores)),
                    "legal_unique_count": int(legal_log_scores.size),
                    "exactly_reachable": exact,
                    "nearest_abs": float(
                        np.min(np.abs(legal_log_scores.astype(np.float64) - float(native_log_score)))
                    ),
                }
            )
    return {
        "schema": "recovar.em.k1_clean_operand_atomic_boundary.v1",
        "artifacts": {
            "operands": str(operands_path.resolve()),
            "operands_sha256": _sha256(operands_path),
            "native_coarse": str(native_coarse_path.resolve()),
            "native_coarse_sha256": _sha256(native_coarse_path),
        },
        "particle": {
            "part_id": operands.part_id,
            "stack_index_one_based": operands.stack_index,
        },
        "initial_diff2": float(initial),
        "fixed_metric": {
            "exactly_reachable": reachable,
            "candidate_count": len(rows),
            "fraction": float(reachable / len(rows)),
        },
        "classification": (
            "clean_operands_and_native_lane_arithmetic_sufficient"
            if reachable == len(rows)
            else "operand_or_per_lane_arithmetic_mismatch_precedes_atomic_order"
        ),
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--native-coarse", type=Path, required=True)
    parser.add_argument("--initial-diff2", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.operands, args.native_coarse, args.initial_diff2)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
