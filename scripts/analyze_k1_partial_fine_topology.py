#!/usr/bin/env python3
"""Compare a native K=1 fine panel when rotation tables only partly overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.compact_candidate_capture import (
    SCHEMA as PRODUCTION_CAPTURE_SCHEMA,
)
from recovar.em.dense_single_volume.helpers.compact_candidate_capture import (
    validate_raw_capture_shard,
)
from scripts.analyze_k1_fine_score_boundary import (
    _center,
    _float32_from_bits,
    _metric,
    _translation_map,
)
from scripts.analyze_k1_native_coarse_boundary import _raw_residual_structure
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


def _tuple_sequence_report(native: np.ndarray, recovar: np.ndarray) -> dict[str, object]:
    """Compare exact class-pose traversal order without constructing dense pairs."""

    left = np.ascontiguousarray(np.asarray(native, dtype=np.int64).reshape(-1, 2))
    right = np.ascontiguousarray(np.asarray(recovar, dtype=np.int64).reshape(-1, 2))
    common = min(left.shape[0], right.shape[0])
    equal_positions = np.all(left[:common] == right[:common], axis=1)
    mismatch = np.flatnonzero(~equal_positions)
    first = None
    if mismatch.size:
        row = int(mismatch[0])
        first = {
            "position": row,
            "native_key": left[row].tolist(),
            "recovar_key": right[row].tolist(),
        }
    elif left.shape[0] != right.shape[0]:
        first = {
            "position": common,
            "native_key": None if common == left.shape[0] else left[common].tolist(),
            "recovar_key": None if common == right.shape[0] else right[common].tolist(),
        }
    return {
        "native_count": int(left.shape[0]),
        "recovar_count": int(right.shape[0]),
        "exact": bool(left.shape == right.shape and np.array_equal(left, right)),
        "equal_position_count": int(np.count_nonzero(equal_positions)),
        "native_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        "recovar_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
        "first_mismatch": first,
    }


def _native_significant_count(factor, candidate_count: int) -> int:
    """Validate the native fine-support count for full or geometry-only captures."""

    count = int(factor.header[45])
    _require(
        0 < count <= int(candidate_count),
        "native BPref significant count is outside the active fine table",
    )
    return count


def load_recovar_candidate_table(path: Path) -> dict[str, np.ndarray]:
    """Normalize either the legacy pass-2 dump or one production raw shard."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        schema = str(np.asarray(archive["schema"]).item()) if "schema" in archive.files else ""
        if schema == PRODUCTION_CAPTURE_SCHEMA:
            required = {
                "schema", "original_indices", "candidate_offset", "rotation_offset",
                "candidate_local_rotation", "candidate_translation", "raw_combined_score",
                "posterior", "significant", "rotation_log_prior", "translation_log_prior",
                "rotation_matrix", "rotation_global_index", "rotation_parent_global",
                "fine_translations",
            }
        else:
            required = {
                "original_index", "rotations", "fine_translations", "candidate_mask", "probs",
            }
            if "reconstruction_mask" in archive.files:
                required.add("reconstruction_mask")
        missing = required - set(archive.files)
        _require(not missing, f"RECOVAR capture is missing {sorted(missing)}")
        # Full pass-2 diagnostics can contain multi-gigabyte projected-reference
        # and pixel-operand arrays.  Candidate topology analysis must not load
        # fields it never reads.
        recovar = {name: np.asarray(archive[name]) for name in required}
    if schema != PRODUCTION_CAPTURE_SCHEMA:
        candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
        return {
            **recovar,
            "candidate_sequence": np.argwhere(candidate_mask).astype(np.int64, copy=False),
            "capture_schema": np.asarray(schema),
        }

    inventory = validate_raw_capture_shard(path)
    _require(inventory["particle_count"] == 1, "production shard must contain one particle")
    _require(len(inventory["fragments"]) == 1, "production shard must contain one fragment")
    fragment = inventory["fragments"][0]
    _require(fragment["fragment_count"] == 1, "production particle must be complete in one shard")
    candidate_count = int(recovar["candidate_offset"][-1])
    rotation_count = int(recovar["rotation_offset"][-1])
    translation_count = int(recovar["fine_translations"].shape[0])
    rotation = np.asarray(recovar["candidate_local_rotation"], dtype=np.int64)
    translation = np.asarray(recovar["candidate_translation"], dtype=np.int64)
    _require(rotation.shape == translation.shape == (candidate_count,), "candidate keys changed")
    _require(
        np.unique(np.column_stack((rotation, translation)), axis=0).shape[0] == candidate_count,
        "production candidate keys are duplicated",
    )
    shape = (rotation_count, translation_count)
    candidate_mask = np.zeros(shape, dtype=bool)
    candidate_mask[rotation, translation] = True

    def dense(name: str, *, fill, dtype=None) -> np.ndarray:
        values = np.asarray(recovar[name], dtype=dtype)
        _require(values.shape == (candidate_count,), f"{name} candidate topology changed")
        output = np.full(shape, fill, dtype=values.dtype)
        output[rotation, translation] = values
        return output

    return {
        **recovar,
        "original_index": np.asarray(recovar["original_indices"][0], dtype=np.int64),
        "rotations": np.asarray(recovar["rotation_matrix"], dtype=np.float32),
        "candidate_mask": candidate_mask,
        "candidate_sequence": np.column_stack((rotation, translation)),
        "probs": dense("posterior", fill=0.0),
        "production_combined_score": dense("raw_combined_score", fill=-np.inf),
        "production_rotation_log_prior": dense("rotation_log_prior", fill=np.nan),
        "production_translation_log_prior": dense("translation_log_prior", fill=np.nan),
        "production_significant": dense("significant", fill=0, dtype=bool),
        "capture_schema": np.asarray(schema),
    }


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
    recovar = load_recovar_candidate_table(recovar_path)
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
    native_mapped_sequence = np.column_stack(
        (mapped_rotations[native_has_rotation], mapped_translations[native_has_rotation])
    )

    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    recovar_keys = {
        tuple(map(int, row)) for row in np.argwhere(candidate_mask).astype(np.int64).tolist()
    }
    common_keys = native_common_keys & recovar_keys
    native_only_common_rotation = native_common_keys - recovar_keys
    recovar_only = recovar_keys - native_common_keys
    recovar_sequence = np.asarray(recovar["candidate_sequence"], dtype=np.int64).reshape(-1, 2)
    translation_stride = int(
        max(
            native_mapped_sequence[:, 1].max(initial=-1),
            recovar_sequence[:, 1].max(initial=-1),
        )
        + 1
    )
    native_sequence_codes = (
        native_mapped_sequence[:, 0] * translation_stride + native_mapped_sequence[:, 1]
    )
    recovar_sequence_codes = recovar_sequence[:, 0] * translation_stride + recovar_sequence[:, 1]
    common_codes = np.intersect1d(
        native_sequence_codes,
        recovar_sequence_codes,
        assume_unique=True,
    )
    native_common_sequence = native_mapped_sequence[
        np.isin(native_sequence_codes, common_codes, assume_unique=True)
    ]
    recovar_common_sequence = recovar_sequence[
        np.isin(recovar_sequence_codes, common_codes, assume_unique=True)
    ]

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

    production_boundary = None
    if "production_combined_score" in recovar:
        native_common_log = np.asarray(
            candidates["combined_preexponent"][native_common_rows],
            dtype=np.float32,
        )
        native_common_rot_prior = np.asarray(
            candidates["orientation_log_prior"][native_common_rows],
            dtype=np.float32,
        )
        native_common_trans_prior = np.asarray(
            candidates["translation_log_prior"][native_common_rows],
            dtype=np.float32,
        )
        recovar_common_log = np.asarray(
            [recovar["production_combined_score"][key] for key in ordered_common],
            dtype=np.float32,
        )
        recovar_common_rot_prior = np.asarray(
            [recovar["production_rotation_log_prior"][key] for key in ordered_common],
            dtype=np.float32,
        )
        recovar_common_trans_prior = np.asarray(
            [recovar["production_translation_log_prior"][key] for key in ordered_common],
            dtype=np.float32,
        )
        native_common_preprior = -np.asarray(
            candidates["raw_diff2"][native_common_rows],
            dtype=np.float32,
        )
        recovar_common_preprior = (
            recovar_common_log
            - recovar_common_rot_prior
            - recovar_common_trans_prior
        ).astype(np.float32, copy=False)
        preprior_residual = np.zeros(candidate_mask.shape, dtype=np.float64)
        preprior_common = np.zeros(candidate_mask.shape, dtype=bool)
        for row, (rotation, translation) in enumerate(ordered_common):
            preprior_residual[rotation, translation] = (
                float(recovar_common_preprior[row]) - float(native_common_preprior[row])
            )
            preprior_common[rotation, translation] = True
        preprior_global_offset = float(np.median(preprior_residual[preprior_common]))
        preprior_residual[preprior_common] -= preprior_global_offset
        centered_preprior_residual = (
            recovar_common_preprior.astype(np.float64)
            - native_common_preprior.astype(np.float64)
            - preprior_global_offset
        )
        best_centered_preprior_residual = (
            _center(recovar_common_preprior) - _center(native_common_preprior)
        )
        worst_preprior_rows = np.argsort(
            -np.abs(best_centered_preprior_residual),
            kind="stable",
        )[:16]
        worst_preprior_records = []
        for common_row in worst_preprior_rows:
            rotation, translation = ordered_common[int(common_row)]
            native_row = int(native_common_rows[int(common_row)])
            worst_preprior_records.append(
                {
                    "recovar_rotation_local": int(rotation),
                    "recovar_rotation_global": int(recovar["rotation_global_index"][rotation]),
                    "recovar_rotation_parent_global": int(
                        recovar["rotation_parent_global"][rotation]
                    ),
                    "recovar_translation": int(translation),
                    "native_rotation_local": int(native_rotation_rows[native_row]),
                    "native_translation": int(native_translation_rows[native_row]),
                    "native_preprior_score": float(native_common_preprior[common_row]),
                    "recovar_preprior_score": float(recovar_common_preprior[common_row]),
                    "best_centered_residual_recovar_minus_native": float(
                        best_centered_preprior_residual[common_row]
                    ),
                    "global_offset_removed_residual_recovar_minus_native": float(
                        centered_preprior_residual[common_row]
                    ),
                }
            )

        native_significant_count = _native_significant_count(factor, candidates.size)
        native_significant_rows = np.argsort(-native_weights, kind="stable")[
            :native_significant_count
        ]
        native_significant_keys = {
            (int(mapped_rotations[row]), int(mapped_translations[row]))
            for row in native_significant_rows
            if mapped_rotations[row] >= 0
        }
        recovar_significant_keys = {
            tuple(map(int, row))
            for row in np.argwhere(recovar["production_significant"]).tolist()
        }
        recovar_only_significant = recovar_significant_keys - native_significant_keys
        native_only_significant = native_significant_keys - recovar_significant_keys
        recovar_only_support_records = []
        for rotation, translation in sorted(recovar_only_significant)[:64]:
            key = (rotation, translation)
            native_row = native_key_to_row.get(key)
            recovar_only_support_records.append(
                {
                    "rotation": int(rotation),
                    "translation": int(translation),
                    "origin": (
                        "common_active_tuple"
                        if native_row is not None
                        else "recovar_only_active_tuple"
                    ),
                    "native_probability": (
                        None if native_row is None else float(native_probs[native_row])
                    ),
                    "recovar_probability": float(recovar_probs[rotation, translation]),
                }
            )
        production_boundary = {
            "combined_log_weight_centered": _metric(
                _center(native_common_log),
                _center(recovar_common_log),
            ),
            "preprior_score_centered": _metric(
                _center(native_common_preprior),
                _center(recovar_common_preprior),
            ),
            "preprior_score_global_offset_recovar_minus_native": preprior_global_offset,
            "preprior_score_worst_centered_records": worst_preprior_records,
            "preprior_residual_structure": _raw_residual_structure(
                preprior_residual,
                preprior_common,
            ),
            "orientation_log_prior": _metric(
                native_common_rot_prior,
                recovar_common_rot_prior,
            ),
            "translation_log_prior": _metric(
                native_common_trans_prior,
                recovar_common_trans_prior,
            ),
            "posterior_on_common_native_normalization": _metric(
                native_common_values,
                recovar_common_values,
            ),
            "posterior_common_domain_renormalized": _metric(
                native_common_normalized,
                recovar_common_normalized,
            ),
            "fine_significant_support": {
                "native_count": native_significant_count,
                "native_mapped_count": len(native_significant_keys),
                "recovar_count": len(recovar_significant_keys),
                "exact": native_significant_keys == recovar_significant_keys,
                "native_probability_threshold": float(
                    np.min(native_probs[native_significant_rows])
                ),
                "recovar_probability_threshold": float(
                    np.min(recovar_probs[np.asarray(recovar["production_significant"], dtype=bool)])
                ),
                "native_only_count": len(native_only_significant),
                "recovar_only_count": len(recovar_only_significant),
                "recovar_only_common_active_count": sum(
                    key in native_common_keys for key in recovar_only_significant
                ),
                "recovar_only_active_tuple_count": sum(
                    key not in native_common_keys for key in recovar_only_significant
                ),
                "native_only_first": _records(native_only_significant),
                "recovar_only_first": _records(recovar_only_significant),
                "recovar_only_records_first": recovar_only_support_records,
            },
        }

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
        "active_tuple_sequence": _tuple_sequence_report(
            native_mapped_sequence,
            recovar_sequence,
        ),
        "common_active_tuple_sequence": _tuple_sequence_report(
            native_common_sequence,
            recovar_common_sequence,
        ),
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
        "production_boundary": production_boundary,
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
