#!/usr/bin/env python3
"""Compare a cross-rotation coarse margin to native legal atomic outcomes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _translation_permutation,
)
from scripts.analyze_em_k1_live_reference_counterfactual import (
    relion_reference_on_recovar_window,
    relion_values_on_recovar_window,
)
from scripts.analyze_k1_coarse_operand_boundary_v3 import (
    _atomic_add_log_score_values,
    _component_decomposition,
    _relative_l2,
    _rotation_key_to_recovar,
)
from scripts.validate_relion_coarse_lane_capture import (
    load_artifact as load_lanes,
    validate_capture as validate_lanes,
)
from scripts.validate_relion_coarse_operand_capture import load_artifact as load_operands
from scripts.validate_relion_coarse_pass1_components import load_artifact as load_components


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> np.float32:
    return np.asarray(np.uint32(value)).view(np.float32).item()


def _outcome_report(values: np.ndarray, production: float) -> dict[str, object]:
    outcomes = np.asarray(values, dtype=np.float32).reshape(-1)
    production_f32 = np.float32(production)
    return {
        "unique_count": int(outcomes.size),
        "minimum": float(np.min(outcomes)),
        "maximum": float(np.max(outcomes)),
        "values": outcomes.astype(float).tolist(),
        "production": float(production_f32),
        "production_is_legal": bool(np.any(outcomes == production_f32)),
    }


def _native_atomic_order_fit(
    lane_partials: np.ndarray,
    native_diff2: np.ndarray,
    rotation_keys: np.ndarray,
    *,
    translation_count: int,
    initial_diff2: np.float32,
) -> dict[str, object]:
    """Find lane orders consistent with native production over the panel."""

    lanes = np.asarray(lane_partials, dtype=np.float32)
    keys = np.asarray(rotation_keys, dtype=np.int64)
    if lanes.ndim != 2 or lanes.shape[0] != keys.size:
        raise ValueError("native lane panel topology differs")
    active_lane_count = lanes.shape[1] // int(translation_count)
    if not 0 < active_lane_count <= 8:
        raise ValueError("native lane panel has an unsupported active-lane count")
    orders = list(itertools.permutations(range(active_lane_count)))
    matched_counts = np.zeros(len(orders), dtype=np.int64)
    common = set(range(len(orders)))
    unreachable = []
    for rotation_row, rotation_key in enumerate(keys):
        for translation in range(int(translation_count)):
            terms = lanes[
                rotation_row,
                translation : active_lane_count * int(translation_count) : int(translation_count),
            ]
            production = np.float32(-native_diff2[int(rotation_key), translation])
            matching = set()
            for order_index, order in enumerate(orders):
                accumulator = np.float32(initial_diff2)
                for lane in order:
                    accumulator = np.float32(
                        np.float64(accumulator) + np.float64(terms[lane])
                    )
                if np.float32(-accumulator) == production:
                    matching.add(order_index)
                    matched_counts[order_index] += 1
            common &= matching
            if not matching:
                unreachable.append([int(rotation_key), int(translation)])
    ranking = sorted(
        (
            {
                "lane_order": list(orders[index]),
                "matched_candidate_count": int(matched_counts[index]),
            }
            for index in range(len(orders))
        ),
        key=lambda row: (-row["matched_candidate_count"], row["lane_order"]),
    )
    return {
        "active_lane_count": int(active_lane_count),
        "candidate_count": int(keys.size * int(translation_count)),
        "unreachable_candidate_count": len(unreachable),
        "unreachable_candidates": unreachable,
        "single_order_explains_full_panel": bool(common),
        "full_panel_lane_orders": [list(orders[index]) for index in sorted(common)],
        "lane_order_ranking": ranking,
    }


def analyze(
    *,
    components_path: Path,
    operands_path: Path,
    lanes_path: Path,
    recovar_path: Path,
    winner_native_rotation: int,
    winner_recovar: tuple[int, int],
    target_native_rotation: int,
    target_recovar: tuple[int, int],
    physical_image_size: int,
) -> dict[str, object]:
    components = load_components(components_path)
    operands = load_operands(operands_path)
    lanes = load_lanes(lanes_path)
    if not (
        components.part_id == operands.part_id == lanes.part_id
        and components.stack_index == operands.stack_index == lanes.stack_index
    ):
        raise ValueError("native capture particle identities differ")
    if not np.array_equal(operands.rotation_keys, lanes.rotation_keys):
        raise ValueError("native operand and lane rotations differ")
    lane_validation = validate_lanes(lanes, operands, components)
    if lane_validation["status"] != "pass":
        raise ValueError("native lane capture does not reproduce production scores")

    with np.load(recovar_path, allow_pickle=False) as payload:
        fields = set(payload.files)
        recovar_scores = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float32)[0]
        recovar_translations = np.asarray(payload["translations"], dtype=np.float64)
        original_index = int(payload["original_index"])
        projection_fields = {
            "projected_reference_rotation_ids",
            "projected_reference_per_class",
            "projected_reference_norm_score_per_class",
            "projected_cross_score_per_class",
            "window_indices",
            "shifted_data",
            "ctf2_data",
            "half_weights",
        }
        projection_capture_available = projection_fields <= fields
        projection_payload = (
            {
                "rotation_ids": np.asarray(
                    payload["projected_reference_rotation_ids"], dtype=np.int64
                ),
                "references": np.asarray(
                    payload["projected_reference_per_class"][0], dtype=np.complex128
                ),
                "norms": np.asarray(
                    payload["projected_reference_norm_score_per_class"][0],
                    dtype=np.float64,
                ),
                "crosses": np.asarray(
                    payload["projected_cross_score_per_class"][0], dtype=np.float64
                ),
                "window_indices": np.asarray(payload["window_indices"], dtype=np.int64),
                "shifted": np.asarray(payload["shifted_data"], dtype=np.complex128),
                "ctf2": np.asarray(payload["ctf2_data"][0], dtype=np.float64),
                "half_weights": np.asarray(payload["half_weights"], dtype=np.float64),
            }
            if projection_capture_available
            else None
        )
    if original_index + 1 != components.stack_index:
        raise ValueError("native and RECOVAR particle identities differ")

    n_directions, n_psi, n_trans = (int(value) for value in components.header[10:13])
    mapping = {
        int(key): _rotation_key_to_recovar(int(key), n_directions, n_psi)
        for key in lanes.rotation_keys
    }
    if mapping.get(winner_native_rotation) != winner_recovar[0]:
        raise ValueError("winner rotation mapping differs")
    if mapping.get(target_native_rotation) != target_recovar[0]:
        raise ValueError("target rotation mapping differs")
    lane_rows = {int(key): row for row, key in enumerate(lanes.rotation_keys)}

    relion_to_recovar_translation, translation_report = _translation_permutation(
        components.translations,
        recovar_translations,
    )
    inverse_translation = np.argsort(relion_to_recovar_translation)
    winner_native_translation = int(inverse_translation[winner_recovar[1]])
    target_native_translation = int(inverse_translation[target_recovar[1]])
    initial_diff2 = np.float32(_float32_from_bits(int(lanes.header[20])))

    winner_outcomes = _atomic_add_log_score_values(
        lanes.lane_partials[lane_rows[winner_native_rotation]],
        translation_count=n_trans,
        translation=winner_native_translation,
        initial_diff2=initial_diff2,
    )
    target_outcomes = _atomic_add_log_score_values(
        lanes.lane_partials[lane_rows[target_native_rotation]],
        translation_count=n_trans,
        translation=target_native_translation,
        initial_diff2=initial_diff2,
    )
    native_atomic_order_fit = _native_atomic_order_fit(
        lanes.lane_partials,
        components.raw_diff2,
        lanes.rotation_keys,
        translation_count=n_trans,
        initial_diff2=initial_diff2,
    )
    possible_margins = np.unique(
        target_outcomes.astype(np.float64)[:, None]
        - winner_outcomes.astype(np.float64)[None, :]
    )

    native_winner = float(-components.raw_diff2[winner_native_rotation, winner_native_translation])
    native_target = float(-components.raw_diff2[target_native_rotation, target_native_translation])
    native_margin = native_target - native_winner
    recovar_winner = float(recovar_scores[winner_recovar])
    recovar_target = float(recovar_scores[target_recovar])
    recovar_margin = recovar_target - recovar_winner
    operand_boundary: dict[str, object]
    if projection_payload is None:
        operand_boundary = {
            "status": "not_captured",
            "missing_fields": sorted(projection_fields - fields),
        }
    else:
        if int(physical_image_size) <= 0 or int(physical_image_size) % 2:
            raise ValueError("physical image size must be a positive even integer")
        native_to_recovar_rotation = {
            int(key): _rotation_key_to_recovar(int(key), n_directions, n_psi)
            for key in operands.rotation_keys
        }
        recovar_to_operand_row = {
            recovar_rotation: row
            for row, recovar_rotation in enumerate(native_to_recovar_rotation.values())
        }
        requested_rotations = projection_payload["rotation_ids"]
        if any(int(rotation) not in recovar_to_operand_row for rotation in requested_rotations):
            raise ValueError("RECOVAR projection panel is not covered by native operands")
        operand_order = np.asarray(
            [recovar_to_operand_row[int(rotation)] for rotation in requested_rotations],
            dtype=np.int64,
        )
        window_indices = projection_payload["window_indices"]
        current_size = int(components.header[27])
        native_reference = relion_reference_on_recovar_window(
            (
                operands.reference_real.astype(np.float64)
                + 1j * operands.reference_imag.astype(np.float64)
            )[operand_order],
            window_indices,
            full_image_size=int(physical_image_size),
            current_size=current_size,
        )
        native_shifted = relion_values_on_recovar_window(
            operands.shifted_real.astype(np.float64)
            + 1j * operands.shifted_imag.astype(np.float64),
            window_indices,
            full_image_size=int(physical_image_size),
            current_size=current_size,
        )
        native_shifted_recovar_order = np.empty_like(native_shifted)
        native_shifted_recovar_order[relion_to_recovar_translation] = native_shifted
        native_correction = relion_values_on_recovar_window(
            operands.correction[np.newaxis, :],
            window_indices,
            full_image_size=int(physical_image_size),
            current_size=current_size,
        )[0].real
        image_normalization = float(int(physical_image_size) ** 2)
        native_weighted_shifted = (
            -native_shifted_recovar_order
            * native_correction[np.newaxis, :]
            / (image_normalization * projection_payload["half_weights"][np.newaxis, :])
        )
        native_ctf2 = native_correction / (
            image_normalization**2 * projection_payload["half_weights"]
        )
        mapped_raw = _map_relion_table(
            components.raw_diff2,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=relion_to_recovar_translation,
        )
        mapped_norm = _map_relion_table(
            components.reference_norms,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=relion_to_recovar_translation,
        )
        mapped_cross = _map_relion_table(
            components.cross_terms,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=relion_to_recovar_translation,
        )
        selected_scores = recovar_scores[requested_rotations]
        component_decomposition = _component_decomposition(
            selected_scores.astype(np.float64) + mapped_raw[requested_rotations],
            projection_payload["norms"] + mapped_norm[requested_rotations],
            projection_payload["crosses"] + mapped_cross[requested_rotations],
        )
        operand_boundary = {
            "status": "complete",
            "recovar_rotation_ids": requested_rotations.tolist(),
            "component_decomposition": component_decomposition,
            "operand_relative_l2": {
                "projected_reference": _relative_l2(
                    native_reference,
                    projection_payload["references"],
                ),
                "weighted_shifted_image": _relative_l2(
                    native_weighted_shifted,
                    projection_payload["shifted"],
                ),
                "correction": _relative_l2(
                    native_ctf2,
                    projection_payload["ctf2"],
                ),
            },
        }
    return {
        "schema": "recovar.em.k1_cross_rotation_lane_boundary.v1",
        "status": "complete",
        "identity": {
            "part_id": int(components.part_id),
            "stack_index_one_based": int(components.stack_index),
            "original_index_zero_based": original_index,
        },
        "coordinates": {
            "winner_native": [winner_native_rotation, winner_native_translation],
            "winner_recovar": list(winner_recovar),
            "target_native": [target_native_rotation, target_native_translation],
            "target_recovar": list(target_recovar),
        },
        "candidate_atomic_outcomes": {
            "winner": _outcome_report(winner_outcomes, native_winner),
            "target": _outcome_report(target_outcomes, native_target),
        },
        "native_lane_validation": lane_validation,
        "native_atomic_order_fit": native_atomic_order_fit,
        "operand_boundary": operand_boundary,
        "target_minus_winner_raw_log_score_margin": {
            "native_production": native_margin,
            "recovar_production": recovar_margin,
            "recovar_minus_native": recovar_margin - native_margin,
            "legal_native_atomic_minimum": float(np.min(possible_margins)),
            "legal_native_atomic_maximum": float(np.max(possible_margins)),
            "legal_native_atomic_values": possible_margins.tolist(),
            "native_production_is_legal": bool(np.any(possible_margins == native_margin)),
            "recovar_production_is_legal": bool(np.any(possible_margins == recovar_margin)),
        },
        "translation_mapping": translation_report,
        "artifacts": {
            "components": str(components_path.resolve()),
            "components_sha256": _sha256(components_path),
            "operands": str(operands_path.resolve()),
            "operands_sha256": _sha256(operands_path),
            "lanes": str(lanes_path.resolve()),
            "lanes_sha256": _sha256(lanes_path),
            "recovar": str(recovar_path.resolve()),
            "recovar_sha256": _sha256(recovar_path),
        },
    }


def _pair(value: str) -> tuple[int, int]:
    fields = value.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("coordinate must be ROTATION,TRANSLATION")
    return int(fields[0]), int(fields[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--lanes", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--winner-native-rotation", type=int, required=True)
    parser.add_argument("--winner-recovar", type=_pair, required=True)
    parser.add_argument("--target-native-rotation", type=int, required=True)
    parser.add_argument("--target-recovar", type=_pair, required=True)
    parser.add_argument("--physical-image-size", type=int, default=256)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        components_path=args.components,
        operands_path=args.operands,
        lanes_path=args.lanes,
        recovar_path=args.recovar,
        winner_native_rotation=args.winner_native_rotation,
        winner_recovar=args.winner_recovar,
        target_native_rotation=args.target_native_rotation,
        target_recovar=args.target_recovar,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
