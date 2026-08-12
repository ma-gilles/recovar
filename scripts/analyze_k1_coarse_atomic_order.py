#!/usr/bin/env python3
"""Enumerate RELION coarse-score lane accumulation orders for a captured panel."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.validate_relion_coarse_operand_capture import load_artifact as load_operands
from scripts.validate_relion_coarse_pass1_components import load_artifact as load_components


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fma_f32(left: np.float32, right: np.float32, addend: np.float32) -> np.float32:
    """Correctly round a float32 fused multiply-add using an exact float64 intermediate."""
    return np.float32(np.float64(left) * np.float64(right) + np.float64(addend))


def _lane_partials(operands) -> np.ndarray:
    block_size = int(operands.header[37])
    prefetch_fraction = int(operands.header[38])
    translation_count = int(operands.header[14])
    pixel_count = int(operands.header[13])
    lane_count = block_size // translation_count
    pixels_per_chunk = block_size // prefetch_fraction
    if (block_size, prefetch_fraction, translation_count, lane_count) != (128, 4, 29, 4):
        raise ValueError("this audit is pinned to RELION's block128/prefetch4/translation29 kernel")

    correction_half = np.asarray(operands.correction, dtype=np.float32) * np.float32(0.5)
    partials = np.zeros(
        (operands.rotation_keys.size, translation_count, lane_count),
        dtype=np.float32,
    )
    for rotation in range(operands.rotation_keys.size):
        for translation in range(translation_count):
            for lane in range(lane_count):
                lane_sum = np.float32(0.0)
                for chunk_start in range(0, pixel_count, pixels_per_chunk):
                    for pixel_in_chunk in range(lane, pixels_per_chunk, lane_count):
                        pixel = chunk_start + pixel_in_chunk
                        if pixel >= pixel_count:
                            break
                        diff_real = np.float32(
                            operands.reference_real[rotation, pixel]
                            - operands.shifted_real[translation, pixel]
                        )
                        diff_imag = np.float32(
                            operands.reference_imag[rotation, pixel]
                            - operands.shifted_imag[translation, pixel]
                        )
                        imag_square = np.float32(diff_imag * diff_imag)
                        square_sum = _fma_f32(diff_real, diff_real, imag_square)
                        lane_sum = _fma_f32(square_sum, correction_half[pixel], lane_sum)
                partials[rotation, translation, lane] = lane_sum
    return partials


def _summed_scores(partials: np.ndarray, initial_diff2: np.float32) -> tuple[list[tuple[int, ...]], np.ndarray]:
    permutations = list(itertools.permutations(range(partials.shape[-1])))
    scores = np.empty((len(permutations), *partials.shape[:2]), dtype=np.float32)
    for permutation_index, permutation in enumerate(permutations):
        for rotation in range(partials.shape[0]):
            for translation in range(partials.shape[1]):
                value = initial_diff2
                for lane in permutation:
                    value = np.float32(value + partials[rotation, translation, lane])
                scores[permutation_index, rotation, translation] = value
    return permutations, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")

    components = load_components(args.components)
    operands = load_operands(args.operands)
    if components.part_id != operands.part_id or components.stack_index != operands.stack_index:
        raise ValueError("native component and operand captures identify different particles")
    with np.load(args.recovar, allow_pickle=False) as payload:
        initial_diff2 = np.asarray(payload["coarse_gaussian_initial_diff2"], dtype=np.float32).item()

    lane_partials = _lane_partials(operands)
    permutations, scores = _summed_scores(lane_partials, np.float32(initial_diff2))
    native = np.asarray(components.raw_diff2, dtype=np.float32)[
        np.asarray(operands.rotation_keys, dtype=np.int64)
    ]
    exact = scores.view(np.uint32) == native[None, ...].view(np.uint32)
    attainable = np.any(exact, axis=0)

    fixed_rows = []
    for permutation_index, permutation in enumerate(permutations):
        residual = scores[permutation_index].astype(np.float64) - native.astype(np.float64)
        fixed_rows.append(
            {
                "lane_order": list(permutation),
                "bitwise_equal_count": int(np.count_nonzero(exact[permutation_index])),
                "p95_abs": float(np.percentile(np.abs(residual), 95)),
                "max_abs": float(np.max(np.abs(residual))),
            }
        )
    fixed_rows.sort(key=lambda row: (-row["bitwise_equal_count"], row["p95_abs"], row["lane_order"]))

    compatible_by_translation = []
    for translation in range(native.shape[1]):
        compatible = [
            list(permutations[index])
            for index in range(len(permutations))
            if bool(np.all(exact[index, :, translation]))
        ]
        compatible_by_translation.append(
            {
                "translation_index": translation,
                "both_rotations_bitwise_compatible_orders": compatible,
            }
        )

    report = {
        "schema": "recovar.em.k1_coarse_atomic_order.v1",
        "metric_policy": "exact float32 bits and absolute residuals; no correlation",
        "kernel": {
            "block_size": 128,
            "prefetch_fraction": 4,
            "translation_count": 29,
            "lane_count_per_translation": 4,
        },
        "particle": {
            "part_id": int(components.part_id),
            "stack_index_one_based": int(components.stack_index),
        },
        "candidate_panel": {
            "rotation_keys": np.asarray(operands.rotation_keys, dtype=np.int64).tolist(),
            "score_count": int(native.size),
            "bitwise_attainable_by_some_lane_order": int(np.count_nonzero(attainable)),
            "unattainable_indices": np.argwhere(~attainable).tolist(),
        },
        "best_fixed_lane_orders": fixed_rows[:8],
        "compatible_orders_by_translation": compatible_by_translation,
        "artifacts": {
            "components": str(args.components.resolve()),
            "components_sha256": _sha256(args.components),
            "operands": str(args.operands.resolve()),
            "operands_sha256": _sha256(args.operands),
            "recovar": str(args.recovar.resolve()),
            "recovar_sha256": _sha256(args.recovar),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
