#!/usr/bin/env python3
"""Measure competing serialized orientation modes in one VDAM score capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recovar.data_io.starfile import read_star
from scripts.audit_em_particle_state_distribution import _angular_error_deg

SCHEMA = "recovar.vdam_orientation_mode_boundary.v1"


class OrientationModeError(RuntimeError):
    """Raised when a mode cannot be tied to one captured score hypothesis."""


def _column(table, name: str) -> str:
    matches = [str(column) for column in table.columns if str(column).lstrip("_") == name]
    if len(matches) != 1:
        raise OrientationModeError(f"expected one {name} column, found {matches}")
    return matches[0]


def summarize_orientation_modes(
    *,
    local_eulers: np.ndarray,
    total_scores: np.ndarray,
    posterior: np.ndarray,
    modes: dict[str, np.ndarray],
    translation_index: int,
    angular_tolerance_deg: float = 1.0e-3,
) -> dict[str, Any]:
    """Map serialized modes to local rows and report their score spacing."""

    local_eulers = np.asarray(local_eulers, dtype=np.float64)
    total_scores = np.asarray(total_scores)
    posterior = np.asarray(posterior)
    if local_eulers.ndim != 2 or local_eulers.shape[1] != 3 or local_eulers.shape[0] == 0:
        raise OrientationModeError("local Euler table must have nonempty shape (R, 3)")
    if (
        total_scores.ndim != 2
        or total_scores.shape != posterior.shape
        or total_scores.shape[0] != local_eulers.shape[0]
    ):
        raise OrientationModeError("score, posterior, and local rotation shapes differ")
    if not modes:
        raise OrientationModeError("at least one serialized mode is required")
    if not 0 <= int(translation_index) < total_scores.shape[1]:
        raise OrientationModeError("translation index is outside the score table")
    finite_scores = np.where(np.isfinite(total_scores), total_scores, -np.inf)
    winner_flat = int(np.argmax(finite_scores))
    winner_row, winner_translation = np.unravel_index(winner_flat, total_scores.shape)
    winner_score = float(total_scores[winner_row, winner_translation])
    if not np.isfinite(winner_score):
        raise OrientationModeError("captured score table has no finite hypothesis")
    ulp = abs(float(np.spacing(np.float32(winner_score))))

    rows: list[dict[str, Any]] = []
    for label, raw_eulers in sorted(modes.items()):
        eulers = np.asarray(raw_eulers, dtype=np.float64).reshape(1, 3)
        errors = _angular_error_deg(np.repeat(eulers, local_eulers.shape[0], axis=0), local_eulers)
        nearest = int(np.argmin(errors))
        error = float(errors[nearest])
        if error > float(angular_tolerance_deg):
            raise OrientationModeError(
                f"mode {label} misses local support: {error:.9g} > {angular_tolerance_deg:.9g} deg"
            )
        score = float(total_scores[nearest, int(translation_index)])
        probability = float(posterior[nearest, int(translation_index)])
        gap = winner_score - score
        rows.append(
            {
                "label": label,
                "serialized_eulers_deg": eulers.reshape(3).tolist(),
                "local_rotation_row": nearest,
                "angular_mapping_error_deg": error,
                "translation_index": int(translation_index),
                "total_log_score": score,
                "posterior": probability,
                "winner_minus_mode_score": gap,
                "winner_minus_mode_score_float32_ulps": (gap / ulp if ulp else None),
                "is_global_winner": (
                    nearest == int(winner_row)
                    and int(translation_index) == int(winner_translation)
                ),
                "exact_score_tie_with_global_winner": score == winner_score,
            }
        )
    row_groups: dict[str, list[str]] = {}
    for row in rows:
        row_groups.setdefault(str(row["local_rotation_row"]), []).append(str(row["label"]))
    return {
        "global_winner": {
            "rotation_row": int(winner_row),
            "translation_index": int(winner_translation),
            "total_log_score": winner_score,
            "posterior": float(posterior[winner_row, winner_translation]),
            "float32_ulp": ulp,
        },
        "evaluated_translation_index": int(translation_index),
        "modes": rows,
        "labels_by_local_rotation_row": row_groups,
    }


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("state must use LABEL=/path/to/run_itNNN_data.star")
    return label, Path(raw_path)


def analyze(
    *, score_dump: Path, states: list[tuple[str, Path]], image_name: str
) -> dict[str, Any]:
    modes: dict[str, np.ndarray] = {}
    state_rows = []
    for label, path in states:
        table = read_star(str(path))[0]
        identity = _column(table, "rlnImageName")
        matches = table[table[identity].astype(str) == image_name]
        if len(matches) != 1:
            raise OrientationModeError(f"{label}: target identity occurs {len(matches)} times")
        row = matches.iloc[0]
        eulers = np.asarray(
            [
                float(row[_column(table, "rlnAngleRot")]),
                float(row[_column(table, "rlnAngleTilt")]),
                float(row[_column(table, "rlnAnglePsi")]),
            ],
            dtype=np.float64,
        )
        modes[label] = eulers
        state_rows.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "eulers_deg": eulers.tolist(),
                "translation_angstrom": [
                    float(row[_column(table, "rlnOriginXAngst")]),
                    float(row[_column(table, "rlnOriginYAngst")]),
                ],
                "pmax": float(row[_column(table, "rlnMaxValueProbDistribution")]),
            }
        )
    with np.load(score_dump, allow_pickle=False) as archive:
        required = {
            "selected_global_image_indices",
            "local_rotation_eulers",
            "pass2_scores_total",
            "posterior",
            "best_score_translation_index",
            "debug_iteration",
        }
        missing = required - set(archive.files)
        if missing:
            raise OrientationModeError(f"score dump misses fields: {sorted(missing)}")
        selected = np.asarray(archive["selected_global_image_indices"], dtype=np.int64)
        if selected.size != 1:
            raise OrientationModeError("score dump must contain exactly one selected particle")
        scores = np.asarray(archive["pass2_scores_total"])
        posterior = np.asarray(archive["posterior"])
        if scores.ndim == 3 and scores.shape[0] == 1:
            scores = scores[0]
        if posterior.ndim == 3 and posterior.shape[0] == 1:
            posterior = posterior[0]
        translation_index = int(
            np.asarray(archive["best_score_translation_index"]).reshape(-1)[0]
        )
        summary = summarize_orientation_modes(
            local_eulers=np.asarray(archive["local_rotation_eulers"]),
            total_scores=scores,
            posterior=posterior,
            modes=modes,
            translation_index=translation_index,
        )
        iteration = int(np.asarray(archive["debug_iteration"]).reshape(-1)[0])
    return {
        "schema": SCHEMA,
        "scoring": False,
        "image_name": image_name,
        "original_index_zero_based": int(selected[0]),
        "iteration": iteration,
        "score_dump": str(score_dump.resolve()),
        "serialized_states": state_rows,
        "score_boundary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dump", type=Path, required=True)
    parser.add_argument("--state", type=_parse_labeled_path, action="append", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(score_dump=args.score_dump, states=args.state, image_name=args.image_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
