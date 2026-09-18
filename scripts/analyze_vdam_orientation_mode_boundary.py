#!/usr/bin/env python3
"""Measure competing serialized orientation modes in one VDAM score capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recovar.data_io.starfile import read_star
from scripts.analyze_em_k1_native_fine_operands import _flat_memmap
from scripts.audit_em_particle_state_distribution import _angular_error_deg
from scripts.compare_relion_recovar_estep_dump import _nearest_rotation_rows_by_matrix

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
    raw_scores: np.ndarray,
    total_scores: np.ndarray,
    posterior: np.ndarray,
    rotation_log_prior: np.ndarray,
    translation_log_prior: np.ndarray,
    modes: dict[str, np.ndarray],
    translation_index: int,
    angular_tolerance_deg: float = 1.0e-3,
) -> dict[str, Any]:
    """Map serialized modes to local rows and report their score spacing."""

    local_eulers = np.asarray(local_eulers, dtype=np.float64)
    raw_scores = np.asarray(raw_scores)
    total_scores = np.asarray(total_scores)
    posterior = np.asarray(posterior)
    rotation_log_prior = np.asarray(rotation_log_prior).reshape(-1)
    translation_log_prior = np.asarray(translation_log_prior).reshape(-1)
    if local_eulers.ndim != 2 or local_eulers.shape[1] != 3 or local_eulers.shape[0] == 0:
        raise OrientationModeError("local Euler table must have nonempty shape (R, 3)")
    if (
        total_scores.ndim != 2
        or total_scores.shape != posterior.shape
        or total_scores.shape != raw_scores.shape
        or total_scores.shape[0] != local_eulers.shape[0]
        or rotation_log_prior.shape != (local_eulers.shape[0],)
        or translation_log_prior.shape != (total_scores.shape[1],)
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
    winner_raw = float(raw_scores[winner_row, winner_translation])
    winner_rotation_prior = float(rotation_log_prior[winner_row])
    winner_translation_prior = float(translation_log_prior[winner_translation])

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
        raw_score = float(raw_scores[nearest, int(translation_index)])
        rotation_prior = float(rotation_log_prior[nearest])
        translation_prior = float(translation_log_prior[int(translation_index)])
        probability = float(posterior[nearest, int(translation_index)])
        gap = winner_score - score
        recomposed = raw_score + rotation_prior + translation_prior
        rows.append(
            {
                "label": label,
                "serialized_eulers_deg": eulers.reshape(3).tolist(),
                "local_rotation_row": nearest,
                "angular_mapping_error_deg": error,
                "translation_index": int(translation_index),
                "total_log_score": score,
                "raw_log_score": raw_score,
                "rotation_log_prior": rotation_prior,
                "translation_log_prior": translation_prior,
                "recomposed_total_log_score": recomposed,
                "total_score_minus_recomposed": score - recomposed,
                "posterior": probability,
                "winner_minus_mode_score": gap,
                "winner_minus_mode_components": {
                    "raw_log_score": winner_raw - raw_score,
                    "rotation_log_prior": winner_rotation_prior - rotation_prior,
                    "translation_log_prior": winner_translation_prior - translation_prior,
                },
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
            "raw_log_score": winner_raw,
            "rotation_log_prior": winner_rotation_prior,
            "translation_log_prior": winner_translation_prior,
            "posterior": float(posterior[winner_row, winner_translation]),
            "float32_ulp": ulp,
        },
        "evaluated_translation_index": int(translation_index),
        "modes": rows,
        "labels_by_local_rotation_row": row_groups,
    }


def summarize_native_orientation_modes(
    *,
    native_candidate_rotation_rows: np.ndarray,
    native_candidate_translation_indices: np.ndarray,
    native_raw_diff2: np.ndarray,
    native_rotation_log_prior: np.ndarray,
    native_translation_log_prior: np.ndarray,
    native_posterior: np.ndarray,
    candidate_boundary: dict[str, Any],
) -> dict[str, Any]:
    """Compare observed RECOVAR modes with one aligned native fine search."""

    rotation_rows = np.asarray(native_candidate_rotation_rows, dtype=np.int64).reshape(-1)
    translation_indices = np.asarray(
        native_candidate_translation_indices, dtype=np.int64
    ).reshape(-1)
    raw_diff2 = np.asarray(native_raw_diff2, dtype=np.float64).reshape(-1)
    rotation_prior = np.asarray(native_rotation_log_prior, dtype=np.float64).reshape(-1)
    translation_prior = np.asarray(native_translation_log_prior, dtype=np.float64).reshape(-1)
    posterior = np.asarray(native_posterior, dtype=np.float64).reshape(-1)
    arrays = (
        translation_indices,
        raw_diff2,
        rotation_prior,
        translation_prior,
        posterior,
    )
    if rotation_rows.size == 0 or any(values.shape != rotation_rows.shape for values in arrays):
        raise OrientationModeError("native fine-search candidate arrays differ")
    if (
        not np.all(np.isfinite(raw_diff2))
        or not np.all(np.isfinite(rotation_prior))
        or not np.all(np.isfinite(translation_prior))
        or not np.all(np.isfinite(posterior))
        or np.any(posterior < 0.0)
        or not np.any(posterior > 0.0)
    ):
        raise OrientationModeError("native fine-search candidate values are invalid")

    native_raw_log_score = -raw_diff2
    native_total = native_raw_log_score + rotation_prior + translation_prior
    winner = int(np.argmax(posterior))
    winner_components = {
        "raw_log_score": float(native_raw_log_score[winner]),
        "rotation_log_prior": float(rotation_prior[winner]),
        "translation_log_prior": float(translation_prior[winner]),
    }
    candidate_modes = candidate_boundary.get("modes")
    if not isinstance(candidate_modes, list) or not candidate_modes:
        raise OrientationModeError("candidate score boundary has no modes")

    rows: list[dict[str, Any]] = []
    for candidate_mode in candidate_modes:
        local_rotation_row = int(candidate_mode["local_rotation_row"])
        translation_index = int(candidate_mode["translation_index"])
        matches = np.flatnonzero(
            (rotation_rows == local_rotation_row)
            & (translation_indices == translation_index)
        )
        if matches.size != 1:
            raise OrientationModeError(
                f"native mode {candidate_mode['label']} maps to {matches.size} candidates"
            )
        index = int(matches[0])
        native_components = {
            "raw_log_score": float(native_raw_log_score[index]),
            "rotation_log_prior": float(rotation_prior[index]),
            "translation_log_prior": float(translation_prior[index]),
        }
        native_gap_components = {
            name: winner_components[name] - native_components[name]
            for name in winner_components
        }
        native_gap = float(native_total[winner] - native_total[index])
        candidate_gap_components = candidate_mode["winner_minus_mode_components"]
        candidate_gap = float(candidate_mode["winner_minus_mode_score"])
        rows.append(
            {
                "label": str(candidate_mode["label"]),
                "local_rotation_row": local_rotation_row,
                "translation_index": translation_index,
                "native_candidate_index": index,
                "native_raw_diff2": float(raw_diff2[index]),
                "native_raw_log_score": native_components["raw_log_score"],
                "native_rotation_log_prior": native_components["rotation_log_prior"],
                "native_translation_log_prior": native_components[
                    "translation_log_prior"
                ],
                "native_total_log_score": float(native_total[index]),
                "native_posterior": float(posterior[index]),
                "native_winner_minus_mode_score": native_gap,
                "native_winner_minus_mode_components": native_gap_components,
                "candidate_winner_minus_mode_score": candidate_gap,
                "candidate_minus_native_winner_gap": candidate_gap - native_gap,
                "candidate_minus_native_winner_gap_components": {
                    name: float(candidate_gap_components[name])
                    - native_gap_components[name]
                    for name in native_gap_components
                },
                "native_posterior_log_odds_winner_over_mode": (
                    float(np.log(posterior[winner]) - np.log(posterior[index]))
                    if posterior[index] > 0.0
                    else None
                ),
                "is_native_global_winner": index == winner,
            }
        )
    return {
        "native_global_winner": {
            "native_candidate_index": winner,
            "local_rotation_row": int(rotation_rows[winner]),
            "translation_index": int(translation_indices[winner]),
            "raw_diff2": float(raw_diff2[winner]),
            "raw_log_score": winner_components["raw_log_score"],
            "rotation_log_prior": winner_components["rotation_log_prior"],
            "translation_log_prior": winner_components["translation_log_prior"],
            "total_log_score": float(native_total[winner]),
            "posterior": float(posterior[winner]),
        },
        "modes": rows,
    }


def _native_score_boundary(
    native_directory: Path,
    *,
    local_rotation_matrices: np.ndarray,
    candidate_boundary: dict[str, Any],
) -> dict[str, Any]:
    native_eulers = np.asarray(
        _flat_memmap(native_directory / "pass1_class0_fine_eulers.bin")
    ).reshape(-1, 3, 3)
    nearest, distance, orientation = _nearest_rotation_rows_by_matrix(
        native_eulers, local_rotation_matrices
    )
    native_rotation_indices = np.asarray(
        _flat_memmap(native_directory / "pass1_acc_rot_idx.bin", np.int32),
        dtype=np.int64,
    )
    native_translation_indices = np.asarray(
        _flat_memmap(native_directory / "pass1_acc_trans_idx.bin", np.int32),
        dtype=np.int64,
    )
    report = summarize_native_orientation_modes(
        native_candidate_rotation_rows=nearest[native_rotation_indices],
        native_candidate_translation_indices=native_translation_indices,
        native_raw_diff2=np.asarray(
            _flat_memmap(native_directory / "pass1_exp_Mweight_raw_preprior.bin")
        ),
        native_rotation_log_prior=np.asarray(
            _flat_memmap(native_directory / "pass1_candidate_orientation_log_prior.bin")
        ),
        native_translation_log_prior=np.asarray(
            _flat_memmap(native_directory / "pass1_candidate_offset_log_prior.bin")
        ),
        native_posterior=np.asarray(
            _flat_memmap(native_directory / "pass1_candidate_weight_normalized.bin")
        ),
        candidate_boundary=candidate_boundary,
    )
    report["rotation_mapping"] = {
        "orientation": orientation,
        "max_frobenius_distance": float(np.max(distance)),
    }
    report["native_directory"] = str(native_directory.resolve())
    return report


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("state must use LABEL=/path/to/run_itNNN_data.star")
    return label, Path(raw_path)


def analyze(
    *,
    score_dump: Path,
    states: list[tuple[str, Path]],
    image_name: str,
    native_directory: Path | None = None,
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
            "pass2_scores_raw",
            "posterior",
            "rotation_log_prior",
            "translation_log_prior",
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
            raw_scores=np.asarray(archive["pass2_scores_raw"])[0],
            total_scores=scores,
            posterior=posterior,
            rotation_log_prior=np.asarray(archive["rotation_log_prior"]),
            translation_log_prior=np.asarray(archive["translation_log_prior"]),
            modes=modes,
            translation_index=translation_index,
        )
        native_boundary = (
            _native_score_boundary(
                native_directory,
                local_rotation_matrices=np.asarray(archive["local_rotation_matrices"]),
                candidate_boundary=summary,
            )
            if native_directory is not None
            else None
        )
        iteration = int(np.asarray(archive["debug_iteration"]).reshape(-1)[0])
    report = {
        "schema": SCHEMA,
        "scoring": False,
        "image_name": image_name,
        "original_index_zero_based": int(selected[0]),
        "iteration": iteration,
        "score_dump": str(score_dump.resolve()),
        "serialized_states": state_rows,
        "score_boundary": summary,
    }
    if native_boundary is not None:
        report["native_score_boundary"] = native_boundary
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dump", type=Path, required=True)
    parser.add_argument("--state", type=_parse_labeled_path, action="append", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--native-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(
        score_dump=args.score_dump,
        states=args.state,
        image_name=args.image_name,
        native_directory=args.native_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
