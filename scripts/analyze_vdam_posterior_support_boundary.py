#!/usr/bin/env python3
"""Audit a VDAM posterior when RELION and RECOVAR rotation supports differ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_vdam_storewavg_boundary import (
    _load_native,
    _posterior_metric,
    _production_score_gradient_rows,
    _require,
)


SCHEMA = "recovar.vdam_posterior_support_boundary.v1"


def _partial_rotation_match(
    native_rotations: np.ndarray,
    recovar_rotations: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return native-to-RECOVAR rows, unmatched RECOVAR rows, and distances."""

    native = np.asarray(native_rotations, dtype=np.float32).reshape(-1, 3, 3)
    recovar = np.asarray(recovar_rotations, dtype=np.float32).reshape(-1, 3, 3)
    _require(native.shape[0] > 0 and recovar.shape[0] > 0, "rotation supports must be non-empty")
    distances = np.max(np.abs(native[:, None] - recovar[None, :]), axis=(2, 3))
    nearest_rows = np.argmin(distances, axis=1).astype(np.int64)
    nearest_distances = distances[np.arange(native.shape[0]), nearest_rows]
    native_to_recovar = np.where(nearest_distances <= tolerance, nearest_rows, -1)
    matched_rows = native_to_recovar[native_to_recovar >= 0]
    _require(
        np.unique(matched_rows).size == matched_rows.size,
        "native-to-RECOVAR partial rotation mapping is not one-to-one",
    )
    unmatched_recovar = np.setdiff1d(
        np.arange(recovar.shape[0], dtype=np.int64),
        matched_rows,
        assume_unique=False,
    )
    return native_to_recovar, unmatched_recovar, nearest_distances


def summarize_support_boundary(
    native_rotations: np.ndarray,
    native_probabilities: np.ndarray,
    recovar_rotations: np.ndarray,
    recovar_probabilities: np.ndarray,
    *,
    rotation_tolerance: float = 1.0e-5,
) -> dict[str, object]:
    native_probabilities = np.asarray(native_probabilities, dtype=np.float32)
    recovar_probabilities = np.asarray(recovar_probabilities, dtype=np.float32)
    _require(native_probabilities.ndim == 2, "native posterior must have shape (R,T)")
    _require(recovar_probabilities.ndim == 2, "RECOVAR posterior must have shape (R,T)")
    _require(
        native_probabilities.shape[1] == recovar_probabilities.shape[1],
        "translation support size differs",
    )
    _require(
        native_probabilities.shape[0] == np.asarray(native_rotations).shape[0]
        and recovar_probabilities.shape[0] == np.asarray(recovar_rotations).shape[0],
        "posterior and rotation counts differ",
    )

    native_to_recovar, recovar_only, nearest_distances = _partial_rotation_match(
        native_rotations,
        recovar_rotations,
        rotation_tolerance,
    )
    native_matched = np.flatnonzero(native_to_recovar >= 0)
    native_only = np.flatnonzero(native_to_recovar < 0)
    recovar_matched = native_to_recovar[native_matched]

    matched_metric = _posterior_metric(
        native_probabilities[native_matched],
        recovar_probabilities[recovar_matched],
    )
    native_rotation_mass = np.sum(native_probabilities, axis=1, dtype=np.float64)
    recovar_rotation_mass = np.sum(recovar_probabilities, axis=1, dtype=np.float64)
    native_flat_winner = int(np.argmax(native_probabilities))
    recovar_flat_winner = int(np.argmax(recovar_probabilities))
    native_winner = np.unravel_index(native_flat_winner, native_probabilities.shape)
    recovar_winner = np.unravel_index(recovar_flat_winner, recovar_probabilities.shape)
    mapped_native_winner = int(native_to_recovar[native_winner[0]])

    top_native_only = sorted(
        (
            {
                "native_rotation_row": int(row),
                "rotation_mass": float(native_rotation_mass[row]),
                "max_probability": float(np.max(native_probabilities[row])),
                "nearest_recovar_rotation_distance": float(nearest_distances[row]),
            }
            for row in native_only
        ),
        key=lambda item: item["rotation_mass"],
        reverse=True,
    )

    return {
        "rotation_support": {
            "native_count": int(native_probabilities.shape[0]),
            "recovar_count": int(recovar_probabilities.shape[0]),
            "matched_count": int(native_matched.size),
            "native_only_count": int(native_only.size),
            "recovar_only_count": int(recovar_only.size),
            "native_only_rows": native_only.tolist(),
            "recovar_only_rows": recovar_only.tolist(),
            "native_only_retained_mass": float(np.sum(native_rotation_mass[native_only])),
            "recovar_only_retained_mass": float(np.sum(recovar_rotation_mass[recovar_only])),
            "top_native_only": top_native_only,
        },
        "matched_posterior": matched_metric,
        "winner": {
            "native_rotation_row": int(native_winner[0]),
            "native_translation_row": int(native_winner[1]),
            "native_probability": float(native_probabilities[native_winner]),
            "native_rotation_recovar_row": mapped_native_winner,
            "recovar_rotation_row": int(recovar_winner[0]),
            "recovar_translation_row": int(recovar_winner[1]),
            "recovar_probability": float(recovar_probabilities[recovar_winner]),
            "same_hypothesis": bool(
                mapped_native_winner == int(recovar_winner[0])
                and int(native_winner[1]) == int(recovar_winner[1])
            ),
        },
    }


def analyze(
    native_directory: Path,
    native_prefix: str,
    recovar_score_dump: Path,
    *,
    rotation_tolerance: float = 1.0e-5,
) -> dict[str, object]:
    native = _load_native(native_directory, native_prefix, load_projector=False)
    with np.load(recovar_score_dump, allow_pickle=False) as archive:
        score = {name: archive[name] for name in archive.files}
    _, _, recovar_probabilities = _production_score_gradient_rows(score)
    selected = np.asarray(score["selected_global_image_indices"], dtype=np.int64)
    _require(selected.size == 1, "score dump must contain exactly one particle")
    report = summarize_support_boundary(
        np.asarray(native["rotations"], dtype=np.float32),
        np.asarray(native["probabilities"], dtype=np.float32),
        np.asarray(score["local_rotation_matrices"], dtype=np.float32),
        np.asarray(recovar_probabilities, dtype=np.float32),
        rotation_tolerance=rotation_tolerance,
    )
    return {
        "schema": SCHEMA,
        "identity": {
            "recovar_original_index": int(selected[0]),
            "current_size": int(np.asarray(score["current_size"]).reshape(-1)[0]),
            "translation_count": int(np.asarray(recovar_probabilities).shape[-1]),
        },
        **report,
        "artifacts": {
            "native_directory": str(native_directory.resolve()),
            "native_prefix": native_prefix,
            "recovar_score_dump": str(recovar_score_dump.resolve()),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-prefix", required=True)
    parser.add_argument("--recovar-score-dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = analyze(
        args.native_directory,
        args.native_prefix,
        args.recovar_score_dump,
        rotation_tolerance=args.rotation_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
