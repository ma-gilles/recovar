#!/usr/bin/env python3
"""Enumerate RELION coarse-score lane accumulation orders for a captured panel."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import _translation_permutation
from scripts.validate_relion_coarse_lane_capture import (
    _float32_from_bits,
)
from scripts.validate_relion_coarse_lane_capture import (
    load_artifact as load_lanes,
)
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


def _captured_lane_partials(lanes) -> np.ndarray:
    """Reshape native thread partials into rotation/translation/lane-group order."""

    translation_count = int(lanes.header[14])
    block_size = int(lanes.header[17])
    active_lanes = block_size // translation_count
    thread_ids = (
        np.arange(translation_count, dtype=np.int64)[:, None]
        + np.arange(active_lanes, dtype=np.int64)[None, :] * translation_count
    )
    return np.asarray(lanes.lane_partials[:, thread_ids], dtype=np.float32)


def _rotation_key_to_recovar(rotation_key: int, n_directions: int, n_psi: int) -> int:
    direction, psi = divmod(int(rotation_key), int(n_psi))
    if direction >= n_directions:
        raise ValueError("native rotation key is out of range")
    return int(psi * n_directions + direction)


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


def _same_bits(left: np.float32, right: np.float32) -> bool:
    left_bits = np.asarray(left, dtype=np.float32).view(np.uint32)
    right_bits = np.asarray(right, dtype=np.float32).view(np.uint32)
    return bool(left_bits == right_bits)


def _pair_order_audit(
    *,
    permutations: list[tuple[int, ...]],
    scores: np.ndarray,
    native: np.ndarray,
    recovar: np.ndarray,
    target: tuple[int, int],
    winner: tuple[int, int],
) -> dict[str, object]:
    """Audit all legal target/winner atomic arrival-order combinations."""

    target_values = scores[:, target[0], target[1]]
    winner_values = scores[:, winner[0], winner[1]]
    native_target = np.float32(native[target])
    native_winner = np.float32(native[winner])
    recovar_target = np.float32(recovar[target])
    recovar_winner = np.float32(recovar[winner])
    native_target_orders = [
        index for index, value in enumerate(target_values) if _same_bits(value, native_target)
    ]
    native_winner_orders = [
        index for index, value in enumerate(winner_values) if _same_bits(value, native_winner)
    ]
    native_relative = np.float32(-native_target + native_winner)
    recovar_relative = np.float32(-recovar_target + recovar_winner)
    native_relative_pairs = []
    recovar_relative_pairs = []
    for target_order, target_value in enumerate(target_values):
        for winner_order, winner_value in enumerate(winner_values):
            relative = np.float32(-target_value + winner_value)
            if _same_bits(relative, native_relative):
                native_relative_pairs.append(
                    [
                        list(permutations[target_order]),
                        list(permutations[winner_order]),
                    ]
                )
            if _same_bits(relative, recovar_relative):
                recovar_relative_pairs.append(
                    [
                        list(permutations[target_order]),
                        list(permutations[winner_order]),
                    ]
                )
    same_order_native = [
        list(permutations[index])
        for index in range(len(permutations))
        if _same_bits(
            np.float32(-target_values[index] + winner_values[index]),
            native_relative,
        )
    ]
    return {
        "native_target_diff2": float(native_target),
        "native_winner_diff2": float(native_winner),
        "recovar_target_diff2": float(recovar_target),
        "recovar_winner_diff2": float(recovar_winner),
        "native_relative_score": float(native_relative),
        "recovar_relative_score": float(recovar_relative),
        "native_target_exact_orders": [list(permutations[index]) for index in native_target_orders],
        "native_winner_exact_orders": [list(permutations[index]) for index in native_winner_orders],
        "native_both_scores_independently_attainable": bool(
            native_target_orders and native_winner_orders
        ),
        "native_relative_independent_order_pair_count": len(native_relative_pairs),
        "native_relative_independent_lane_order_pairs": native_relative_pairs,
        "native_relative_same_order_count": len(same_order_native),
        "native_relative_same_orders": same_order_native,
        "recovar_relative_independent_order_pair_count": len(recovar_relative_pairs),
        "recovar_relative_independent_lane_order_pairs": recovar_relative_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--lanes", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")

    components = load_components(args.components)
    operands = load_operands(args.operands)
    lanes = load_lanes(args.lanes)
    if not (
        components.part_id == operands.part_id == lanes.part_id
        and components.stack_index == operands.stack_index == lanes.stack_index
    ):
        raise ValueError("native component, operand, and lane captures identify different particles")
    if not np.array_equal(operands.rotation_keys, lanes.rotation_keys):
        raise ValueError("native operand and lane rotation keys differ")
    with np.load(args.recovar, allow_pickle=False) as payload:
        recovar_initial_diff2 = np.asarray(
            payload["coarse_gaussian_initial_diff2"], dtype=np.float32
        ).item()
        recovar_scores = np.asarray(
            payload["scores_pre_prior_per_class"][0], dtype=np.float32
        )
        recovar_translations = np.asarray(payload["translations"], dtype=np.float64)
        recovar_significant = np.asarray(payload["significant_mask"], dtype=bool).reshape(
            recovar_scores.shape
        )

    lane_partials = _captured_lane_partials(lanes)
    initial_diff2 = _float32_from_bits(int(lanes.header[20]))
    permutations, scores = _summed_scores(lane_partials, initial_diff2)
    native = np.asarray(components.raw_diff2, dtype=np.float32)[
        np.asarray(operands.rotation_keys, dtype=np.int64)
    ]
    exact = scores.view(np.uint32) == native[None, ...].view(np.uint32)
    attainable = np.any(exact, axis=0)
    n_directions, n_psi = map(int, components.header[10:12])
    recovar_rotation_ids = np.asarray(
        [
            _rotation_key_to_recovar(key, n_directions, n_psi)
            for key in operands.rotation_keys
        ],
        dtype=np.int64,
    )
    translation_permutation, translation_mapping = _translation_permutation(
        components.translations,
        recovar_translations,
    )
    recovar_diff2 = -recovar_scores[recovar_rotation_ids][:, translation_permutation]
    recovar_significant_panel = recovar_significant[recovar_rotation_ids][
        :, translation_permutation
    ]
    native_significant_panel = components.significant_mask[lanes.rotation_keys]
    recovar_exact = scores.view(np.uint32) == recovar_diff2[None, ...].view(np.uint32)
    recovar_attainable = np.any(recovar_exact, axis=0)

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

    anchor = np.unravel_index(np.argmin(native), native.shape)
    native_relative_score = -(native - native[anchor])
    recovar_relative_score = -(recovar_diff2 - recovar_diff2[anchor])
    relative_residual = recovar_relative_score.astype(np.float64) - native_relative_score.astype(np.float64)

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

    native_winner_rotation, native_winner_translation = divmod(
        int(components.header[15]),
        int(components.header[12]),
    )
    panel_by_rotation_key = {
        int(rotation_key): index
        for index, rotation_key in enumerate(lanes.rotation_keys)
    }
    winner_panel_rotation = panel_by_rotation_key.get(native_winner_rotation)
    boundary_rows = []
    for target_rotation, target_translation in np.argwhere(
        native_significant_panel != recovar_significant_panel
    ):
        row = {
            "target_native_rotation_key": int(lanes.rotation_keys[target_rotation]),
            "target_native_translation": int(target_translation),
            "target_recovar_rotation": int(recovar_rotation_ids[target_rotation]),
            "target_recovar_translation": int(translation_permutation[target_translation]),
            "native_significant": bool(
                native_significant_panel[target_rotation, target_translation]
            ),
            "recovar_significant": bool(
                recovar_significant_panel[target_rotation, target_translation]
            ),
        }
        if winner_panel_rotation is None:
            row["winner_captured"] = False
        else:
            row["winner_captured"] = True
            row["winner_native_rotation_key"] = native_winner_rotation
            row["winner_native_translation"] = native_winner_translation
            row["winner_recovar_rotation"] = int(
                recovar_rotation_ids[winner_panel_rotation]
            )
            row["winner_recovar_translation"] = int(
                translation_permutation[native_winner_translation]
            )
            row["order_audit"] = _pair_order_audit(
                permutations=permutations,
                scores=scores,
                native=native,
                recovar=recovar_diff2,
                target=(int(target_rotation), int(target_translation)),
                winner=(int(winner_panel_rotation), native_winner_translation),
            )
        boundary_rows.append(row)

    report = {
        "schema": "recovar.em.k1_coarse_atomic_order.v3",
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
            "recovar_rotation_ids": recovar_rotation_ids.tolist(),
            "score_count": int(native.size),
            "bitwise_attainable_by_some_lane_order": int(np.count_nonzero(attainable)),
            "unattainable_indices": np.argwhere(~attainable).tolist(),
            "recovar_output_bitwise_attainable_by_native_lane_order": int(
                np.count_nonzero(recovar_attainable)
            ),
            "recovar_output_unattainable_indices": np.argwhere(
                ~recovar_attainable
            ).tolist(),
            "native_panel_anchor": [int(anchor[0]), int(anchor[1])],
            "relative_score_residual": {
                "p95_abs": float(np.percentile(np.abs(relative_residual), 95)),
                "max_abs": float(np.max(np.abs(relative_residual))),
            },
        },
        "best_fixed_lane_orders": fixed_rows[:8],
        "compatible_orders_by_translation": compatible_by_translation,
        "support_boundary": {
            "native_winner_rotation_key": native_winner_rotation,
            "native_winner_translation": native_winner_translation,
            "native_winner_captured": winner_panel_rotation is not None,
            "mismatch_count_in_captured_panel": len(boundary_rows),
            "mismatches": boundary_rows,
        },
        "translation_mapping": translation_mapping,
        "artifacts": {
            "components": str(args.components.resolve()),
            "components_sha256": _sha256(args.components),
            "operands": str(args.operands.resolve()),
            "operands_sha256": _sha256(args.operands),
            "lanes": str(args.lanes.resolve()),
            "lanes_sha256": _sha256(args.lanes),
            "recovar": str(args.recovar.resolve()),
            "recovar_sha256": _sha256(args.recovar),
        },
        "initial_diff2": {
            "native": float(initial_diff2),
            "recovar": float(recovar_initial_diff2),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
