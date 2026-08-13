#!/usr/bin/env python3
"""Compare a native K=1 fine panel when rotation tables only partly overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_fine_score_boundary import (
    _float32_from_bits,
    _translation_map,
)
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def partial_rotation_map(
    factor_rotations: np.ndarray,
    recovar_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return native-to-RECOVAR rows and the unmatched rows on each side."""

    native = np.ascontiguousarray(
        np.asarray(factor_rotations["matrix"], dtype=np.float32)
        .reshape(-1, 3, 3)
        .transpose(0, 2, 1)
    )
    recovar = np.ascontiguousarray(
        np.asarray(recovar_rotations, dtype=np.float32).reshape(-1, 3, 3)
    )
    _require(native.size > 0 and recovar.size > 0, "rotation tables must be non-empty")
    key_dtype = np.dtype((np.void, 9 * np.dtype(np.float32).itemsize))
    native_keys = native.reshape(-1, 9).view(key_dtype).reshape(-1)
    recovar_keys = recovar.reshape(-1, 9).view(key_dtype).reshape(-1)
    _require(
        np.unique(native_keys).size == native_keys.size
        and np.unique(recovar_keys).size == recovar_keys.size,
        "rotation tables contain duplicate exact matrices",
    )
    recovar_by_key = {key.tobytes(): row for row, key in enumerate(recovar_keys)}
    mapping = np.asarray(
        [recovar_by_key.get(key.tobytes(), -1) for key in native_keys],
        dtype=np.int64,
    )
    native_only = np.flatnonzero(mapping < 0).astype(np.int64)
    matched_recovar = mapping[mapping >= 0]
    recovar_only = np.setdiff1d(
        np.arange(recovar.shape[0], dtype=np.int64),
        matched_recovar,
        assume_unique=True,
    )
    _require(
        np.unique(matched_recovar).size == matched_recovar.size,
        "partial rotation mapping is not one-to-one",
    )
    return mapping, native_only, recovar_only


def _records(keys: set[tuple[int, int]], limit: int = 64) -> list[list[int]]:
    return [[int(rotation), int(translation)] for rotation, translation in sorted(keys)[:limit]]


def analyze(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_path: Path,
    physical_image_size: int = 128,
) -> dict[str, object]:
    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(fine_score_path)
    _require(factor.stack_index == score.stack_index, "native capture identities differ")
    with np.load(recovar_path, allow_pickle=False) as archive:
        recovar = {name: np.asarray(archive[name]) for name in archive.files}
    _require(
        int(recovar["original_index"]) == factor.stack_index - 1,
        "cross-engine particle identity differs",
    )

    rotation_map, native_only_rotation_rows, recovar_only_rotation_rows = partial_rotation_map(
        factor.rotations,
        recovar["rotations"],
    )
    translation_map, translation_error = _translation_map(
        factor.translations,
        recovar["fine_translations"],
        physical_image_size=physical_image_size,
    )

    active = (score.candidates["flags"] & ACTIVE) != 0
    candidates = score.candidates[active]
    native_rotation_rows = np.asarray(candidates["rotation_local"], dtype=np.int64)
    native_translation_rows = np.asarray(candidates["translation_id"], dtype=np.int64)
    mapped_rotations = rotation_map[native_rotation_rows]
    mapped_translations = translation_map[native_translation_rows]
    native_has_rotation = mapped_rotations >= 0
    native_common_keys_array = np.column_stack(
        (mapped_rotations[native_has_rotation], mapped_translations[native_has_rotation])
    )
    native_common_keys = {tuple(map(int, row)) for row in native_common_keys_array.tolist()}
    native_unmapped_tuple_count = int(np.count_nonzero(~native_has_rotation))

    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    recovar_keys = {
        tuple(map(int, row)) for row in np.argwhere(candidate_mask).astype(np.int64).tolist()
    }
    common_keys = native_common_keys & recovar_keys
    native_only_common_rotation = native_common_keys - recovar_keys
    recovar_only = recovar_keys - native_common_keys

    native_weights = np.asarray(candidates["post_exponent_weight"], dtype=np.float64)
    native_sum = (
        _float32_from_bits(score.header[32])
        if int(score.header[35]) == 1
        else float(np.sum(native_weights, dtype=np.float64))
    )
    _require(native_sum > 0.0 and np.isfinite(native_sum), "native posterior sum is invalid")
    native_probs = native_weights / native_sum
    recovar_probs = np.asarray(recovar["probs"], dtype=np.float64)

    native_key_to_row: dict[tuple[int, int], int] = {}
    for row, has_rotation in enumerate(native_has_rotation):
        if has_rotation:
            key = (int(mapped_rotations[row]), int(mapped_translations[row]))
            _require(key not in native_key_to_row, "native active tuple keys are duplicated")
            native_key_to_row[key] = row
    ordered_common = sorted(common_keys)
    native_common_rows = np.asarray([native_key_to_row[key] for key in ordered_common], dtype=np.int64)
    recovar_common_values = np.asarray(
        [recovar_probs[rotation, translation] for rotation, translation in ordered_common],
        dtype=np.float64,
    )
    native_common_values = native_probs[native_common_rows]
    native_common_scan_mass = float(np.sum(native_common_values, dtype=np.float64))
    native_full_scan_mass = float(np.sum(native_probs, dtype=np.float64))
    recovar_common_scan_mass = float(np.sum(recovar_common_values, dtype=np.float64))
    recovar_full_scan_mass = float(np.sum(recovar_probs[candidate_mask], dtype=np.float64))
    _require(
        native_common_scan_mass > 0.0
        and native_full_scan_mass > 0.0
        and recovar_common_scan_mass > 0.0
        and recovar_full_scan_mass > 0.0,
        "common posterior is empty",
    )
    native_common_fraction = native_common_scan_mass / native_full_scan_mass
    recovar_common_fraction = recovar_common_scan_mass / recovar_full_scan_mass
    native_common_normalized = native_common_values / native_common_scan_mass
    recovar_common_normalized = recovar_common_values / recovar_common_scan_mass

    return {
        "schema": "recovar.em.k1_partial_fine_topology.v1",
        "status": "complete",
        "identity": {
            "stack_index_one_based": factor.stack_index,
            "original_index_zero_based": int(recovar["original_index"]),
        },
        "rotation_topology": {
            "native_count": int(factor.rotations.size),
            "recovar_count": int(recovar["rotations"].shape[0]),
            "common_count": int(np.count_nonzero(rotation_map >= 0)),
            "native_only_count": int(native_only_rotation_rows.size),
            "recovar_only_count": int(recovar_only_rotation_rows.size),
            "native_only_rows_first": native_only_rotation_rows[:64].tolist(),
            "recovar_only_rows_first": recovar_only_rotation_rows[:64].tolist(),
        },
        "active_tuple_topology": {
            "native_count": int(candidates.size),
            "recovar_count": len(recovar_keys),
            "common_count": len(common_keys),
            "native_only_count": native_unmapped_tuple_count + len(native_only_common_rotation),
            "native_only_unmapped_rotation_count": native_unmapped_tuple_count,
            "native_only_on_common_rotations_count": len(native_only_common_rotation),
            "recovar_only_count": len(recovar_only),
            "native_only_on_common_rotations_first": _records(native_only_common_rotation),
            "recovar_only_first": _records(recovar_only),
        },
        "posterior": {
            "native_full_pmax": float(np.max(native_probs)),
            "recovar_full_pmax": float(np.max(recovar_probs[candidate_mask])),
            "native_full_scan_mass": native_full_scan_mass,
            "native_scan_mass_on_common_tuples": native_common_scan_mass,
            "native_fraction_on_common_tuples": native_common_fraction,
            "native_fraction_missing_from_common_tuples": 1.0 - native_common_fraction,
            "recovar_full_scan_mass": recovar_full_scan_mass,
            "recovar_scan_mass_on_common_tuples": recovar_common_scan_mass,
            "recovar_fraction_on_common_tuples": recovar_common_fraction,
            "native_common_domain_pmax": float(np.max(native_common_normalized)),
            "recovar_common_domain_pmax": float(np.max(recovar_common_normalized)),
            "common_domain_total_variation": float(
                0.5 * np.sum(np.abs(recovar_common_normalized - native_common_normalized))
            ),
        },
        "translation_map_max_abs": translation_error,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        factor_path=args.factor,
        fine_score_path=args.fine_score,
        recovar_path=args.recovar,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
