#!/usr/bin/env python3
"""Compare native RELION fine candidates with a passive RECOVAR capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_em_k1_native_fine_operands import _flat_memmap
from scripts.analyze_k1_native_candidate_score_boundary import _stats
from scripts.analyze_k1_partial_fine_topology import load_recovar_candidate_table
from scripts.compare_relion_recovar_estep_dump import _nearest_rotation_rows_by_matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values) if values.size else values


def _key_records(keys: set[tuple[int, int]], limit: int = 32) -> list[list[int]]:
    return [[int(rotation), int(translation)] for rotation, translation in sorted(keys)[:limit]]


def analyze(native_dump_dir: Path, recovar_capture: Path) -> dict[str, object]:
    """Locate the first unequal native/production fine-posterior boundary."""

    native_dump_dir = Path(native_dump_dir).resolve()
    recovar_capture = Path(recovar_capture).resolve()
    recovar = load_recovar_candidate_table(recovar_capture)

    native_rotations = np.asarray(
        _flat_memmap(native_dump_dir / "pass1_class0_fine_eulers.bin"),
        dtype=np.float64,
    ).reshape(-1, 3, 3)
    nearest, rotation_distance, orientation = _nearest_rotation_rows_by_matrix(
        native_rotations,
        np.asarray(recovar["rotations"], dtype=np.float32),
    )
    native_rotation_local = np.asarray(
        _flat_memmap(native_dump_dir / "pass1_acc_rot_idx.bin", np.int32),
        dtype=np.int64,
    )
    native_translation = np.asarray(
        _flat_memmap(native_dump_dir / "pass1_acc_trans_idx.bin", np.int32),
        dtype=np.int64,
    )
    native_raw = np.asarray(
        _flat_memmap(native_dump_dir / "pass1_exp_Mweight_raw_preprior.bin"),
        dtype=np.float64,
    )
    native_prior = np.asarray(
        _flat_memmap(native_dump_dir / "pass1_candidate_combined_log_prior.bin"),
        dtype=np.float64,
    )
    native_posterior = np.asarray(
        _flat_memmap(native_dump_dir / "pass1_candidate_weight_normalized.bin"),
        dtype=np.float64,
    )
    native_significant = np.asarray(
        _flat_memmap(
            native_dump_dir / "pass1_candidate_in_reconstruction_set.bin",
            np.int32,
        ),
        dtype=bool,
    )
    lengths = {
        native_rotation_local.size,
        native_translation.size,
        native_raw.size,
        native_prior.size,
        native_posterior.size,
        native_significant.size,
    }
    if len(lengths) != 1:
        raise ValueError(f"native candidate arrays have incompatible lengths: {sorted(lengths)}")
    if np.any(native_rotation_local < 0) or np.any(native_rotation_local >= nearest.size):
        raise ValueError("native candidate references an out-of-range rotation")
    mapped_rotation = nearest[native_rotation_local]
    native_keys_ordered = list(
        zip(mapped_rotation.tolist(), native_translation.tolist(), strict=True)
    )
    native_keys = set(native_keys_ordered)
    if len(native_keys) != len(native_keys_ordered):
        raise ValueError("native active candidate keys are duplicated")

    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    recovar_keys = {tuple(map(int, row)) for row in np.argwhere(candidate_mask)}
    common_keys = sorted(native_keys & recovar_keys)
    native_row_by_key = {key: row for row, key in enumerate(native_keys_ordered)}
    native_rows = np.asarray([native_row_by_key[key] for key in common_keys], dtype=np.int64)
    recovar_rotation = np.asarray([key[0] for key in common_keys], dtype=np.int64)
    recovar_translation = np.asarray([key[1] for key in common_keys], dtype=np.int64)

    recovar_combined = np.asarray(recovar["production_combined_score"], dtype=np.float64)[
        recovar_rotation,
        recovar_translation,
    ]
    recovar_rotation_prior = np.asarray(
        recovar["production_rotation_log_prior"], dtype=np.float64
    )[recovar_rotation, recovar_translation]
    recovar_translation_prior = np.asarray(
        recovar["production_translation_log_prior"], dtype=np.float64
    )[recovar_rotation, recovar_translation]
    recovar_prior = recovar_rotation_prior + recovar_translation_prior
    recovar_preprior = recovar_combined - recovar_prior
    recovar_posterior = np.asarray(recovar["probs"], dtype=np.float64)[
        recovar_rotation,
        recovar_translation,
    ]
    recovar_significant = np.asarray(recovar["production_significant"], dtype=bool)[
        recovar_rotation,
        recovar_translation,
    ]

    native_combined = -native_raw[native_rows] + native_prior[native_rows]
    candidate_set_exact = native_keys == recovar_keys
    preprior_residual = _center(recovar_preprior) - _center(-native_raw[native_rows])
    prior_residual = recovar_prior - native_prior[native_rows]
    combined_residual = _center(recovar_combined) - _center(native_combined)
    posterior_residual = recovar_posterior - native_posterior[native_rows]
    support_exact = bool(
        candidate_set_exact
        and np.array_equal(recovar_significant, native_significant[native_rows])
    )

    if not candidate_set_exact:
        first_boundary = "candidate_set"
    elif np.any(preprior_residual != 0):
        first_boundary = "preprior_score"
    elif np.any(prior_residual != 0):
        first_boundary = "log_prior"
    elif np.any(combined_residual != 0):
        first_boundary = "combined_log_weight"
    elif np.any(posterior_residual != 0):
        first_boundary = "posterior_normalization"
    elif not support_exact:
        first_boundary = "significant_support"
    else:
        first_boundary = None

    native_winner = native_keys_ordered[int(np.argmax(native_posterior))]
    recovar_dense_posterior = np.asarray(recovar["probs"], dtype=np.float64)
    recovar_winner = tuple(
        map(int, np.unravel_index(int(np.argmax(recovar_dense_posterior)), recovar_dense_posterior.shape))
    )
    return {
        "schema": "recovar.em.k1_native_production_boundary.v1",
        "status": "complete",
        "first_nonidentical_boundary": first_boundary,
        "identity": {"original_index_zero_based": int(recovar["original_index"])},
        "rotation_mapping": {
            "orientation": orientation,
            "median_frobenius": float(np.median(rotation_distance)),
            "max_frobenius": float(np.max(rotation_distance)),
        },
        "candidate_set": {
            "exact": candidate_set_exact,
            "native_count": len(native_keys),
            "recovar_count": len(recovar_keys),
            "common_count": len(common_keys),
            "native_only_count": len(native_keys - recovar_keys),
            "recovar_only_count": len(recovar_keys - native_keys),
            "native_only_first": _key_records(native_keys - recovar_keys),
            "recovar_only_first": _key_records(recovar_keys - native_keys),
        },
        "preprior_score_centered_residual_recovar_minus_native": _stats(preprior_residual),
        "log_prior_residual_recovar_minus_native": _stats(prior_residual),
        "combined_log_weight_centered_residual_recovar_minus_native": _stats(combined_residual),
        "posterior": {
            "residual_recovar_minus_native": _stats(posterior_residual),
            "native_pmax": float(np.max(native_posterior)),
            "recovar_pmax": float(np.max(recovar_dense_posterior[candidate_mask])),
        },
        "significant_support": {
            "exact": support_exact,
            "native_count": int(np.count_nonzero(native_significant)),
            "recovar_count": int(np.count_nonzero(recovar["production_significant"])),
            "common_mismatch_count": int(
                np.count_nonzero(recovar_significant != native_significant[native_rows])
            ),
        },
        "winner": {"native": list(native_winner), "recovar": list(recovar_winner)},
        "artifacts": {
            "native_dump_dir": str(native_dump_dir),
            "recovar_capture": str(recovar_capture),
            "recovar_capture_sha256": _sha256(recovar_capture),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dump-dir", type=Path, required=True)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.native_dump_dir, args.recovar_capture)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
