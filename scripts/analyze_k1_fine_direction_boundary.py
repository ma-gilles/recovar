#!/usr/bin/env python3
"""Compare one native RELION and RECOVAR K=1 fine-posterior boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_k1_bpref_contributor_membership import match_rotations


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat(path: Path, dtype: np.dtype) -> np.ndarray:
    payload = path.read_bytes()
    count = struct.unpack_from("<i", payload)[0]
    values = np.frombuffer(payload, dtype=dtype, offset=4).copy()
    _require(values.size == count, f"flat-array size mismatch: {path}")
    return values


def _real(root: Path, name: str) -> np.ndarray:
    return _flat(root / name, np.dtype("<f8"))


def _integer(root: Path, name: str) -> np.ndarray:
    return _flat(root / name, np.dtype("<i4"))


def _scalar(root: Path, name: str) -> float:
    payload = (root / name).read_bytes()
    _require(len(payload) == 8, f"scalar size mismatch: {root / name}")
    return float(struct.unpack("<d", payload)[0])


def _center_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.max(values)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "exact_equal_count": int(np.count_nonzero(values == 0.0)),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
    }


def _float32_key(x: float, y: float) -> tuple[int, int]:
    return (
        int(np.asarray(x, dtype=np.float32).view(np.uint32).item()),
        int(np.asarray(y, dtype=np.float32).view(np.uint32).item()),
    )


def analyze(native_directory: Path, recovar_path: Path) -> dict[str, Any]:
    native_directory = native_directory.resolve()
    recovar_path = recovar_path.resolve()
    native_rotations = _real(
        native_directory, "pass1_class0_fine_eulers.bin"
    ).reshape(-1, 3, 3)
    # RELION's fine Euler table is the exact transpose of RECOVAR's projection
    # rotation convention for this boundary.
    native_rotations = native_rotations.astype(np.float32).transpose(0, 2, 1)
    native_rotation_rows = _integer(native_directory, "pass1_acc_rot_idx.bin")
    native_parent_rotations = _integer(native_directory, "pass1_acc_rot_id.bin")
    native_translation_x = _real(
        native_directory, "pass1_candidate_translation_x.bin"
    ).astype(np.float32)
    native_translation_y = _real(
        native_directory, "pass1_candidate_translation_y.bin"
    ).astype(np.float32)
    native_raw_diff2 = _real(native_directory, "pass1_exp_Mweight_raw_preprior.bin")
    native_prior = _real(native_directory, "pass1_candidate_combined_log_prior.bin")
    native_probability = _real(native_directory, "pass1_candidate_weight_normalized.bin")
    native_significant = _integer(
        native_directory, "pass1_candidate_in_reconstruction_set.bin"
    ).astype(bool)
    candidate_count = native_raw_diff2.size
    for name, values in (
        ("native rotation rows", native_rotation_rows),
        ("native parent rotations", native_parent_rotations),
        ("native translation x", native_translation_x),
        ("native translation y", native_translation_y),
        ("native prior", native_prior),
        ("native probability", native_probability),
        ("native significant mask", native_significant),
    ):
        _require(values.size == candidate_count, f"{name} topology mismatch")

    with np.load(recovar_path, allow_pickle=False) as recovar:
        recovar_original_index = int(
            np.asarray(recovar["original_indices"]).item()
        )
        _require(
            recovar_original_index + 1
            == int(round(_scalar(native_directory, "pass1_acc_stack_index.bin"))),
            "particle identity mismatch",
        )
        actual_rotation_count = int(np.asarray(recovar["actual_counts"]).item())
        recovar_rotations = np.asarray(
            recovar["active_rotations"], dtype=np.float32
        )[:actual_rotation_count]
        recovar_global_rotations = np.asarray(
            recovar["active_global_rotation_indices"], dtype=np.int64
        )[:actual_rotation_count]
        recovar_translations = np.asarray(recovar["fine_translations"], dtype=np.float32)
        recovar_candidate_mask = np.asarray(
            recovar["candidate_mask"], dtype=bool
        )[0, :actual_rotation_count]

        rotation_matches = match_rotations(
            native_rotations,
            recovar_rotations,
            tolerance=0.0,
        )
        _require(
            rotation_matches.pairs.shape[0] == native_rotations.shape[0]
            == recovar_rotations.shape[0],
            "fine rotation sets are not exactly equal after transpose",
        )
        native_to_recovar_rotation = np.empty(native_rotations.shape[0], dtype=np.int64)
        native_to_recovar_rotation[rotation_matches.pairs[:, 0]] = rotation_matches.pairs[:, 1]

        translation_lookup = {
            _float32_key(x, y): index
            for index, (x, y) in enumerate(recovar_translations.tolist())
        }
        native_to_recovar_translation = np.asarray(
            [
                translation_lookup[_float32_key(x, y)]
                for x, y in zip(
                    native_translation_x,
                    native_translation_y,
                    strict=True,
                )
            ],
            dtype=np.int64,
        )
        recovar_rotation_rows = native_to_recovar_rotation[native_rotation_rows]
        n_translations = int(recovar_translations.shape[0])
        native_keys = (
            recovar_rotation_rows * n_translations + native_to_recovar_translation
        )
        _require(
            np.unique(native_keys).size == native_keys.size,
            "native candidate keys are not unique after mapping",
        )
        recovar_keys = np.flatnonzero(recovar_candidate_mask.reshape(-1))
        native_only = np.setdiff1d(native_keys, recovar_keys)
        recovar_only = np.setdiff1d(recovar_keys, native_keys)
        _require(native_only.size == 0, "RECOVAR omits native candidate keys")

        recovar_preprior = np.asarray(
            recovar["candidate_preprior_scores"], dtype=np.float64
        )[0, recovar_rotation_rows, native_to_recovar_translation]
        recovar_prior = (
            np.asarray(recovar["candidate_rotation_log_prior"], dtype=np.float64)[
                0, recovar_rotation_rows
            ]
            + np.asarray(recovar["candidate_translation_log_prior"], dtype=np.float64)[
                0, native_to_recovar_translation
            ]
        )
        recovar_raw_weights_all = np.asarray(
            recovar["candidate_raw_exp_weights_f32"], dtype=np.float32
        )
        recovar_raw_weights = recovar_raw_weights_all[
            0, recovar_rotation_rows, native_to_recovar_translation
        ].astype(np.float64)
        recovar_sum_weight = float(
            np.asarray(recovar["reconstruction_sum_weight"], dtype=np.float64).item()
        )
        recovar_probability = recovar_raw_weights / recovar_sum_weight
        recovar_significant_all = np.asarray(
            recovar["reconstruction_mask"], dtype=bool
        )
        recovar_significant = recovar_significant_all[
            0, recovar_rotation_rows, native_to_recovar_translation
        ]

        extra_rotation_rows, extra_translation_rows = np.divmod(
            recovar_only,
            n_translations,
        )
        extra_probability = (
            recovar_raw_weights_all[
                0, extra_rotation_rows, extra_translation_rows
            ].astype(np.float64)
            / recovar_sum_weight
        )
        extra_significant = recovar_significant_all[
            0, extra_rotation_rows, extra_translation_rows
        ]

    native_direction = np.zeros(768, dtype=np.float64)
    recovar_direction = np.zeros(768, dtype=np.float64)
    native_direction_ids = native_parent_rotations // 48
    _require(
        np.all((native_direction_ids >= 0) & (native_direction_ids < 768)),
        "native direction id outside HEALPix-order-3 range",
    )
    np.add.at(
        native_direction,
        native_direction_ids,
        native_probability * native_significant,
    )
    np.add.at(
        recovar_direction,
        native_direction_ids,
        recovar_probability * recovar_significant,
    )
    if recovar_only.size:
        extra_coarse_rotation = recovar_global_rotations[extra_rotation_rows] // 8
        extra_direction = extra_coarse_rotation % 768
        np.add.at(
            recovar_direction,
            extra_direction,
            extra_probability * extra_significant,
        )

    raw_preprior_centered_delta = (
        _center_max(recovar_preprior) - _center_max(-native_raw_diff2)
    )
    largest_raw_delta_row = int(
        np.argmax(np.abs(raw_preprior_centered_delta))
    )
    largest_raw_delta_recovar_rotation = int(
        recovar_rotation_rows[largest_raw_delta_row]
    )
    largest_raw_delta_translation = int(
        native_to_recovar_translation[largest_raw_delta_row]
    )

    extra_candidates = []
    for rotation_row, translation_row in zip(
        extra_rotation_rows,
        extra_translation_rows,
        strict=True,
    ):
        extra_candidates.append(
            {
                "recovar_rotation_row": int(rotation_row),
                "recovar_global_fine_rotation": int(
                    recovar_global_rotations[rotation_row]
                ),
                "recovar_translation_row": int(translation_row),
                "translation_pixels": recovar_translations[translation_row]
                .astype(float)
                .tolist(),
            }
        )

    return {
        "schema": "recovar.em.k1_fine_direction_boundary.v1",
        "identity": {
            "recovar_original_index_zero_based": int(
                recovar_original_index
            ),
            "relion_part_id": int(round(_scalar(native_directory, "pass1_acc_part_id.bin"))),
            "relion_stack_index_one_based": int(
                round(_scalar(native_directory, "pass1_acc_stack_index.bin"))
            ),
        },
        "rotation_topology": {
            "native_count": int(native_rotations.shape[0]),
            "recovar_count": int(recovar_rotations.shape[0]),
            "orientation_transform": "transpose",
            "exact_match_count": int(rotation_matches.pairs.shape[0]),
            "matched_max_abs": float(np.max(rotation_matches.matched_max_abs)),
        },
        "candidate_topology": {
            "native_count": int(native_keys.size),
            "recovar_count": int(recovar_keys.size),
            "common_count": int(np.intersect1d(native_keys, recovar_keys).size),
            "native_only_count": int(native_only.size),
            "recovar_only_count": int(recovar_only.size),
            "recovar_only": extra_candidates,
        },
        "common_candidates": {
            "raw_preprior_centered_delta_recovar_minus_relion": _stats(
                raw_preprior_centered_delta
            ),
            "largest_raw_preprior_centered_delta": {
                "native_candidate_row": largest_raw_delta_row,
                "native_fine_rotation_local": int(
                    native_rotation_rows[largest_raw_delta_row]
                ),
                "native_parent_rotation": int(
                    native_parent_rotations[largest_raw_delta_row]
                ),
                "recovar_rotation_row": largest_raw_delta_recovar_rotation,
                "recovar_global_fine_rotation": int(
                    recovar_global_rotations[largest_raw_delta_recovar_rotation]
                ),
                "translation_row": largest_raw_delta_translation,
                "translation_pixels": recovar_translations[
                    largest_raw_delta_translation
                ].astype(float).tolist(),
                "delta_recovar_minus_relion": float(
                    raw_preprior_centered_delta[largest_raw_delta_row]
                ),
                "native_significant": bool(
                    native_significant[largest_raw_delta_row]
                ),
                "recovar_significant": bool(
                    recovar_significant[largest_raw_delta_row]
                ),
                "native_probability": float(
                    native_probability[largest_raw_delta_row]
                ),
                "recovar_probability": float(
                    recovar_probability[largest_raw_delta_row]
                ),
            },
            "combined_prior_centered_delta_recovar_minus_relion": _stats(
                _center_max(recovar_prior) - _center_max(native_prior)
            ),
            "normalized_probability_delta_recovar_minus_relion": _stats(
                recovar_probability - native_probability
            ),
            "posterior_total_variation": float(
                0.5 * np.sum(np.abs(recovar_probability - native_probability))
            ),
            "significant_candidate_mismatch_count": int(
                np.count_nonzero(recovar_significant != native_significant)
            ),
            "native_significant_count": int(np.count_nonzero(native_significant)),
            "recovar_significant_count_on_common": int(
                np.count_nonzero(recovar_significant)
            ),
        },
        "recovar_only_candidates": {
            "normalized_probability_sum": float(np.sum(extra_probability)),
            "normalized_probability_max": float(
                np.max(extra_probability) if extra_probability.size else 0.0
            ),
            "significant_count": int(np.count_nonzero(extra_significant)),
        },
        "direction_sufficient_statistic": {
            "native_mass": float(np.sum(native_direction)),
            "recovar_mass": float(np.sum(recovar_direction)),
            "l1_recovar_minus_relion": float(
                np.sum(np.abs(recovar_direction - native_direction))
            ),
            "max_abs_recovar_minus_relion": float(
                np.max(np.abs(recovar_direction - native_direction))
            ),
            "different_direction_count": int(
                np.count_nonzero(recovar_direction != native_direction)
            ),
        },
        "artifacts": {
            "native_directory": str(native_directory),
            "recovar": str(recovar_path),
            "recovar_sha256": _sha256(recovar_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.native_directory, args.recovar_npz)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
