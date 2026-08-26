#!/usr/bin/env python3
"""Join native RELION verbose fine posterior state to a RECOVAR pass-2 dump."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from scripts.parse_relion_dump_dir import parse_dump_dir
else:
    from parse_relion_dump_dir import parse_dump_dir


REQUIRED_NATIVE = {
    "pass1_acc_rot_idx",
    "pass1_acc_trans_idx",
    "pass1_candidate_combined_log_prior",
    "pass1_candidate_in_reconstruction_set",
    "pass1_candidate_offset_log_prior",
    "pass1_candidate_orientation_log_prior",
    "pass1_candidate_translation_x",
    "pass1_candidate_translation_y",
    "pass1_candidate_weight_normalized",
    "pass1_class0_fine_eulers",
    "pass1_exp_Mweight_raw_preprior",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    left = np.ascontiguousarray(reference)
    right = np.ascontiguousarray(candidate)
    _require(left.shape == right.shape and left.size > 0, "metric operands changed shape")
    delta = right.astype(np.float64) - left.astype(np.float64)
    denominator = float(np.linalg.norm(left.astype(np.float64).reshape(-1)))
    mismatch = np.flatnonzero(left.reshape(-1) != right.reshape(-1))
    return {
        "shape": list(left.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "exact_equal": bool(mismatch.size == 0),
        "mismatch_count": int(mismatch.size),
        "relative_l2_over_reference": (
            float(np.linalg.norm(delta.reshape(-1)) / denominator)
            if denominator
            else float(np.linalg.norm(delta.reshape(-1)))
        ),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "first_mismatch_flat": int(mismatch[0]) if mismatch.size else None,
    }


def _center(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.subtract(array, np.max(array), dtype=np.float32)


def _rotation_map(native_eulers: np.ndarray, recovar_rotations: np.ndarray) -> np.ndarray:
    native = np.asarray(native_eulers, dtype=np.float32).reshape(-1, 3, 3)
    native = np.ascontiguousarray(native.transpose(0, 2, 1))
    recovar = np.ascontiguousarray(
        np.asarray(recovar_rotations, dtype=np.float32).reshape(-1, 3, 3)
    )
    key_dtype = np.dtype((np.void, 9 * np.dtype(np.float32).itemsize))
    native_keys = native.reshape(-1, 9).view(key_dtype).reshape(-1)
    recovar_keys = recovar.reshape(-1, 9).view(key_dtype).reshape(-1)
    _require(
        np.unique(native_keys).size == native_keys.size
        and np.unique(recovar_keys).size == recovar_keys.size,
        "native or RECOVAR fine rotations contain duplicate matrices",
    )
    order = np.argsort(recovar_keys)
    sorted_keys = recovar_keys[order]
    positions = np.searchsorted(sorted_keys, native_keys)
    in_bounds = positions < sorted_keys.size
    exact = np.zeros(native_keys.size, dtype=bool)
    exact[in_bounds] = sorted_keys[positions[in_bounds]] == native_keys[in_bounds]
    _require(bool(np.all(exact)), "native fine rotations are absent from RECOVAR")
    mapped = order[positions]
    _require(np.unique(mapped).size == mapped.size, "fine rotation map is not injective")
    return mapped.astype(np.int64)


def _records(
    keys: np.ndarray,
    native_probability: np.ndarray,
    recovar_probability: np.ndarray,
    native_selected: np.ndarray,
    recovar_selected: np.ndarray,
    *,
    count: int = 32,
) -> list[dict[str, object]]:
    priority = np.argsort(-np.maximum(native_probability, recovar_probability), kind="stable")
    output = []
    for index in priority[:count]:
        output.append(
            {
                "recovar_rotation_row": int(keys[index, 0]),
                "recovar_translation_row": int(keys[index, 1]),
                "native_probability": float(native_probability[index]),
                "recovar_probability": float(recovar_probability[index]),
                "native_selected": bool(native_selected[index]),
                "recovar_selected": bool(recovar_selected[index]),
            }
        )
    return output


def analyze(*, native_dump_dir: Path, recovar_capture: Path) -> dict[str, object]:
    native = parse_dump_dir(native_dump_dir, include_names=REQUIRED_NATIVE)
    missing = REQUIRED_NATIVE - set(native)
    _require(not missing, f"native verbose dump misses fields: {sorted(missing)}")

    with np.load(recovar_capture, allow_pickle=False) as archive:
        required_recovar = {
            "candidate_mask",
            "current_size",
            "fine_translations",
            "original_index",
            "probs",
            "reconstruction_mask",
            "rotation_log_prior",
            "rotations",
            "scores_pre_prior",
            "scores_with_prior",
            "translation_log_prior",
        }
        missing_recovar = required_recovar - set(archive.files)
        _require(not missing_recovar, f"RECOVAR pass-2 dump misses fields: {sorted(missing_recovar)}")
        recovar = {name: np.asarray(archive[name]) for name in required_recovar}

    native_rotation_local = np.asarray(native["pass1_acc_rot_idx"], dtype=np.int64)
    native_translation = np.asarray(native["pass1_acc_trans_idx"], dtype=np.int64)
    native_count = native_rotation_local.size
    aligned_native_names = REQUIRED_NATIVE - {"pass1_class0_fine_eulers"}
    _require(
        all(np.asarray(native[name]).size == native_count for name in aligned_native_names),
        "native fine candidate arrays are misaligned",
    )

    rotation_map = _rotation_map(
        native["pass1_class0_fine_eulers"],
        recovar["rotations"],
    )
    _require(
        bool(np.all((native_rotation_local >= 0) & (native_rotation_local < rotation_map.size))),
        "native candidate rotation index is outside the fine table",
    )
    recovar_rotation = rotation_map[native_rotation_local]
    translations = np.asarray(recovar["fine_translations"], dtype=np.float64)
    _require(
        bool(np.all((native_translation >= 0) & (native_translation < translations.shape[0]))),
        "native candidate translation index is outside the RECOVAR fine table",
    )
    native_xy = np.column_stack(
        (
            np.asarray(native["pass1_candidate_translation_x"], dtype=np.float64),
            np.asarray(native["pass1_candidate_translation_y"], dtype=np.float64),
        )
    )
    translation_error = np.max(np.abs(native_xy - translations[native_translation]), axis=1)
    _require(
        bool(np.all(translation_error <= 1.0e-6)),
        "native fine translation ids do not reproduce RECOVAR coordinates",
    )

    native_keys = np.column_stack((recovar_rotation, native_translation))
    _require(
        np.unique(native_keys, axis=0).shape[0] == native_count,
        "native verbose fine candidate keys are not unique",
    )
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    recovar_keys = np.argwhere(candidate_mask).astype(np.int64)
    native_by_key = {tuple(key): i for i, key in enumerate(native_keys.tolist())}
    recovar_key_set = {tuple(key) for key in recovar_keys.tolist()}
    native_key_set = set(native_by_key)
    common_keys = np.asarray(
        [key for key in native_keys.tolist() if tuple(key) in recovar_key_set],
        dtype=np.int64,
    ).reshape(-1, 2)
    _require(common_keys.size > 0, "native and RECOVAR fine candidate sets are disjoint")
    native_rows = np.asarray([native_by_key[tuple(key)] for key in common_keys], dtype=np.int64)
    rr = common_keys[:, 0]
    tt = common_keys[:, 1]

    native_raw_cost = np.asarray(
        native["pass1_exp_Mweight_raw_preprior"], dtype=np.float32
    )[native_rows]
    native_orientation_prior = np.asarray(
        native["pass1_candidate_orientation_log_prior"], dtype=np.float32
    )[native_rows]
    native_translation_prior = np.asarray(
        native["pass1_candidate_offset_log_prior"], dtype=np.float32
    )[native_rows]
    native_total = np.add(
        np.negative(native_raw_cost, dtype=np.float32),
        np.asarray(native["pass1_candidate_combined_log_prior"], dtype=np.float32)[native_rows],
        dtype=np.float32,
    )
    native_probability = np.asarray(
        native["pass1_candidate_weight_normalized"], dtype=np.float32
    )[native_rows]
    native_selected = np.asarray(
        native["pass1_candidate_in_reconstruction_set"], dtype=np.int32
    )[native_rows].astype(bool)

    recovar_raw = np.asarray(recovar["scores_pre_prior"], dtype=np.float32)[rr, tt]
    recovar_orientation_prior = np.asarray(
        recovar["rotation_log_prior"], dtype=np.float32
    )[rr]
    recovar_translation_prior = np.asarray(
        recovar["translation_log_prior"], dtype=np.float32
    )[tt]
    recovar_total = np.asarray(recovar["scores_with_prior"], dtype=np.float32)[rr, tt]
    recovar_probability = np.asarray(recovar["probs"], dtype=np.float32)[rr, tt]
    recovar_selected = np.asarray(recovar["reconstruction_mask"], dtype=bool)[rr, tt]

    comparisons = {
        "preprior_score_centered": _metric(_center(-native_raw_cost), _center(recovar_raw)),
        "orientation_log_prior": _metric(native_orientation_prior, recovar_orientation_prior),
        "translation_log_prior": _metric(native_translation_prior, recovar_translation_prior),
        "combined_log_weight_centered": _metric(_center(native_total), _center(recovar_total)),
        "normalized_posterior_common_candidates": _metric(
            native_probability, recovar_probability
        ),
        "significant_support_common_candidates": _metric(
            native_selected, recovar_selected
        ),
    }
    native_only = sorted(native_key_set - recovar_key_set)
    recovar_only = sorted(recovar_key_set - native_key_set)
    stage_exact = {
        "candidate_tuple_presence": not native_only and not recovar_only,
        **{name: bool(metric["exact_equal"]) for name, metric in comparisons.items()},
    }
    stage_order = [
        "candidate_tuple_presence",
        "preprior_score_centered",
        "orientation_log_prior",
        "translation_log_prior",
        "combined_log_weight_centered",
        "normalized_posterior_common_candidates",
        "significant_support_common_candidates",
    ]
    first_unequal = next((name for name in stage_order if not stage_exact[name]), None)

    native_selected_all = np.asarray(
        native["pass1_candidate_in_reconstruction_set"], dtype=np.int32
    ).astype(bool)
    recovar_selected_all = np.asarray(recovar["reconstruction_mask"], dtype=bool)
    native_probability_all = np.asarray(
        native["pass1_candidate_weight_normalized"], dtype=np.float32
    )
    recovar_probability_all = np.asarray(recovar["probs"], dtype=np.float32)
    native_winner_row = int(np.argmax(native_probability_all))
    native_winner_key = tuple(int(v) for v in native_keys[native_winner_row])
    recovar_winner = np.unravel_index(
        int(np.argmax(recovar_probability_all)), recovar_probability_all.shape
    )

    return {
        "schema": "recovar.em.k1_native_verbose_fine_state.v1",
        "status": "complete",
        "metric_policy": "identity-aligned exact and relative-L2 intermediates; no correlation",
        "identity": {
            "source_row_zero_based": int(np.asarray(recovar["original_index"]).item()),
            "stack_index_one_based": int(np.asarray(recovar["original_index"]).item()) + 1,
            "current_size": int(np.asarray(recovar["current_size"]).item()),
        },
        "candidate_sets": {
            "native_count": native_count,
            "recovar_count": int(recovar_keys.shape[0]),
            "common_count": int(common_keys.shape[0]),
            "native_only_count": len(native_only),
            "recovar_only_count": len(recovar_only),
            "native_only_first": [list(key) for key in native_only[:32]],
            "recovar_only_first": [list(key) for key in recovar_only[:32]],
        },
        "geometry": {
            "native_rotation_count": int(rotation_map.size),
            "rotation_map": rotation_map.tolist(),
            "translation_max_abs_error": float(np.max(translation_error, initial=0.0)),
        },
        "stage_order": stage_order,
        "stage_exact": stage_exact,
        "first_exact_unequal_boundary": first_unequal,
        "comparisons": comparisons,
        "posterior": {
            "native_probability_sum": float(np.sum(native_probability_all, dtype=np.float64)),
            "recovar_probability_sum": float(np.sum(recovar_probability_all, dtype=np.float64)),
            "native_significant_count": int(np.count_nonzero(native_selected_all)),
            "recovar_significant_count": int(np.count_nonzero(recovar_selected_all)),
            "support_native_only_count": len(
                {
                    tuple(key)
                    for key in native_keys[native_selected_all].tolist()
                }
                - {
                    tuple(key)
                    for key in np.argwhere(recovar_selected_all).tolist()
                }
            ),
            "support_recovar_only_count": len(
                {
                    tuple(key)
                    for key in np.argwhere(recovar_selected_all).tolist()
                }
                - {
                    tuple(key)
                    for key in native_keys[native_selected_all].tolist()
                }
            ),
            "native_winner": list(native_winner_key),
            "recovar_winner": [int(recovar_winner[0]), int(recovar_winner[1])],
            "winner_exact": native_winner_key == recovar_winner,
            "top_common_candidates": _records(
                common_keys,
                native_probability,
                recovar_probability,
                native_selected,
                recovar_selected,
            ),
        },
        "artifacts": {
            "native_dump_dir": str(native_dump_dir.resolve()),
            "native_inputs": {
                name: {
                    "path": str((native_dump_dir / f"{name}.bin").resolve()),
                    "sha256": _sha256(native_dump_dir / f"{name}.bin"),
                }
                for name in sorted(REQUIRED_NATIVE)
            },
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dump-dir", type=Path, required=True)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        native_dump_dir=args.native_dump_dir,
        recovar_capture=args.recovar_capture,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
