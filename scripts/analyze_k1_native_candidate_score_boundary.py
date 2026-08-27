#!/usr/bin/env python3
"""Find the first native RELION/RECOVAR K=1 fine-posterior mismatch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_em_k1_native_fine_operands import _flat_memmap  # noqa: E402
from scripts.compare_relion_recovar_estep_dump import (  # noqa: E402
    _nearest_rotation_rows_by_matrix,
)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "rms": float(np.sqrt(np.mean(values * values))) if values.size else 0.0,
        "p95_abs": float(np.percentile(absolute, 95)) if values.size else 0.0,
        "max_abs": float(np.max(absolute)) if values.size else 0.0,
    }


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values) if values.size else values


def _key_records(keys: set[tuple[int, int]], limit: int = 20) -> list[dict[str, int]]:
    return [
        {"rotation_row": int(rotation), "translation_row": int(translation)}
        for rotation, translation in sorted(keys)[:limit]
    ]


def analyze(dump_dir: Path, recovar_npz: Path) -> dict:
    """Compare candidate identity, raw cost, priors, posterior, then support."""

    dump_dir = Path(dump_dir)
    with np.load(recovar_npz, allow_pickle=False) as payload:
        rec = {name: np.array(payload[name]) for name in payload.files}

    native_eulers = _flat_memmap(dump_dir / "pass1_class0_fine_eulers.bin").reshape(-1, 3, 3)
    nearest, rotation_distance, orientation = _nearest_rotation_rows_by_matrix(
        native_eulers,
        rec["rotations"],
    )
    native_rotation_index = np.asarray(
        _flat_memmap(dump_dir / "pass1_acc_rot_idx.bin", np.int32),
        dtype=np.int64,
    )
    native_translation = np.asarray(
        _flat_memmap(dump_dir / "pass1_acc_trans_idx.bin", np.int32),
        dtype=np.int64,
    )
    native_rotation = nearest[native_rotation_index]
    native_keys_ordered = list(
        zip(native_rotation.tolist(), native_translation.tolist(), strict=True)
    )
    native_keys = set(native_keys_ordered)
    recovar_keys = set(map(tuple, np.argwhere(np.asarray(rec["candidate_mask"], dtype=bool))))
    common_keys = native_keys & recovar_keys
    native_only = native_keys - recovar_keys
    recovar_only = recovar_keys - native_keys

    native_row_by_key = {key: row for row, key in enumerate(native_keys_ordered)}
    common_ordered = sorted(common_keys)
    native_rows = np.asarray([native_row_by_key[key] for key in common_ordered], dtype=np.int64)
    rec_rotation = np.asarray([key[0] for key in common_ordered], dtype=np.int64)
    rec_translation = np.asarray([key[1] for key in common_ordered], dtype=np.int64)

    native_raw_all = np.asarray(
        _flat_memmap(dump_dir / "pass1_exp_Mweight_raw_preprior.bin"),
        dtype=np.float64,
    )
    native_prior_all = np.asarray(
        _flat_memmap(dump_dir / "pass1_candidate_combined_log_prior.bin"),
        dtype=np.float64,
    )
    native_probability_all = np.asarray(
        _flat_memmap(dump_dir / "pass1_candidate_weight_normalized.bin"),
        dtype=np.float64,
    )
    native_significant_all = np.asarray(
        _flat_memmap(
            dump_dir / "pass1_candidate_in_reconstruction_set.bin",
            np.int32,
        ),
        dtype=bool,
    )
    lengths = {
        native_raw_all.size,
        native_prior_all.size,
        native_probability_all.size,
        native_significant_all.size,
        len(native_keys_ordered),
    }
    if len(lengths) != 1:
        raise ValueError(f"native candidate arrays have incompatible lengths: {sorted(lengths)}")

    raw_field = (
        "raw_operand_raw_diff2"
        if "raw_operand_raw_diff2" in rec
        else "relion_raw_diff2"
    )
    # RELION's normalized candidate dump is the full fine posterior before
    # significant-support truncation.  RECOVAR's matching array is ``probs``;
    # ``reconstruction_probs`` has already zeroed the excluded tail.
    probability_field = "probs" if "probs" in rec else "reconstruction_probs"
    native_raw = native_raw_all[native_rows]
    native_prior = native_prior_all[native_rows]
    native_probability = native_probability_all[native_rows]
    native_significant = native_significant_all[native_rows]
    recovar_raw = np.asarray(rec[raw_field], dtype=np.float64)[rec_rotation, rec_translation]
    recovar_prior = (
        np.asarray(rec["rotation_log_prior"], dtype=np.float64)[rec_rotation]
        + np.asarray(rec["translation_log_prior"], dtype=np.float64)[rec_translation]
    )
    recovar_probability = np.asarray(rec[probability_field], dtype=np.float64)[
        rec_rotation,
        rec_translation,
    ]
    recovar_significant = np.asarray(rec["reconstruction_mask"], dtype=bool)[
        rec_rotation,
        rec_translation,
    ]

    raw_residual = native_raw - recovar_raw
    raw_centered_residual = _center(raw_residual)
    prior_residual = native_prior - recovar_prior
    total_residual = _center((-native_raw + native_prior) - (-recovar_raw + recovar_prior))
    probability_residual = native_probability - recovar_probability

    candidate_keys_exact = (
        len(native_keys_ordered) == len(native_keys)
        and native_keys == recovar_keys
    )
    raw_exact = bool(
        candidate_keys_exact
        and np.array_equal(
            native_raw.astype(np.float32),
            recovar_raw.astype(np.float32),
        )
    )
    raw_equal_up_to_constant = bool(
        candidate_keys_exact and np.all(raw_centered_residual == 0)
    )
    priors_exact = bool(
        candidate_keys_exact
        and np.array_equal(
            native_prior.astype(np.float32),
            recovar_prior.astype(np.float32),
        )
    )
    posterior_exact = bool(
        candidate_keys_exact
        and np.array_equal(
            native_probability.astype(np.float32),
            recovar_probability.astype(np.float32),
        )
    )
    support_exact = bool(
        candidate_keys_exact
        and np.array_equal(
            native_significant,
            recovar_significant,
        )
    )
    if not candidate_keys_exact:
        first_nonidentical_boundary = "candidate_set"
    elif not raw_exact:
        first_nonidentical_boundary = "raw_cost"
    elif not priors_exact:
        first_nonidentical_boundary = "log_prior"
    elif not posterior_exact:
        first_nonidentical_boundary = "posterior_normalization"
    elif not support_exact:
        first_nonidentical_boundary = "significant_support"
    else:
        first_nonidentical_boundary = None

    native_winner = int(np.argmax(native_probability_all))
    recovar_probability_full = np.asarray(rec[probability_field], dtype=np.float64)
    recovar_winner_flat = int(np.argmax(recovar_probability_full))
    recovar_winner = np.unravel_index(recovar_winner_flat, recovar_probability_full.shape)

    return {
        "schema": "em-k1-native-candidate-score-boundary-v1",
        "status": "complete",
        "first_nonidentical_boundary": first_nonidentical_boundary,
        "recovar_original_index": int(rec["original_index"]),
        "raw_field": raw_field,
        "probability_field": probability_field,
        "rotation_matrix_orientation": orientation,
        "rotation_matrix_median_frobenius": float(np.median(rotation_distance)),
        "rotation_matrix_max_frobenius": float(np.max(rotation_distance)),
        "candidate_set": {
            "exact": candidate_keys_exact,
            "native_count": len(native_keys_ordered),
            "native_unique_count": len(native_keys),
            "recovar_count": len(recovar_keys),
            "common_count": len(common_keys),
            "native_only_count": len(native_only),
            "recovar_only_count": len(recovar_only),
            "native_only_first20": _key_records(native_only),
            "recovar_only_first20": _key_records(recovar_only),
        },
        "raw_cost": {
            "float32_exact": raw_exact,
            "equal_up_to_exact_common_constant": raw_equal_up_to_constant,
            "residual": _stats(raw_residual),
            "centered_residual": _stats(raw_centered_residual),
            "median_common_addend_native_minus_recovar": (
                float(np.median(raw_residual)) if raw_residual.size else 0.0
            ),
        },
        "log_prior": {
            "float32_exact": priors_exact,
            "residual": _stats(prior_residual),
        },
        "total_log_weight_centered_residual": _stats(total_residual),
        "posterior": {
            "float32_exact": posterior_exact,
            "residual": _stats(probability_residual),
            "native_pmax": float(np.max(native_probability_all)),
            "recovar_pmax": float(np.max(recovar_probability_full)),
            "recovar_reconstruction_probability_mass": (
                float(np.sum(rec["reconstruction_probs"]))
                if "reconstruction_probs" in rec
                else None
            ),
        },
        "significant_support": {
            "exact": support_exact,
            "native_count": int(np.count_nonzero(native_significant_all)),
            "recovar_count": int(np.count_nonzero(rec["reconstruction_mask"])),
            "common_candidate_mismatch_count": int(
                np.count_nonzero(native_significant != recovar_significant)
            ),
        },
        "winner": {
            "native": {
                "rotation_row": int(native_rotation[native_winner]),
                "translation_row": int(native_translation[native_winner]),
            },
            "recovar": {
                "rotation_row": int(recovar_winner[0]),
                "translation_row": int(recovar_winner[1]),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--recovar-pass2-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.relion_dump_dir, args.recovar_pass2_npz)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
