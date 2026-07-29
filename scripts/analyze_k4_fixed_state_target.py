#!/usr/bin/env python3
"""Classify a fixed-state K-class pass-2 target in the correct offset frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import starfile

from recovar import utils

TARGET_ORIGINAL_INDEX = 53722
TARGET_IDENTITY = "53723@particles.256.mrcs"
TARGET_CLASS_ZERO_BASED = 0
ITERATION_KEY = "001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relion_round_away_from_zero(values: np.ndarray) -> np.ndarray:
    """Apply RELION's integer pre-shift rounding to pixel offsets."""

    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values >= 0.0,
        np.floor(values + 0.5),
        -np.floor(-values + 0.5),
    ).astype(np.float32)


def relative_to_absolute_translations(
    relative_translations: np.ndarray,
    previous_absolute_translation: np.ndarray,
) -> np.ndarray:
    """Convert pass-2 search-grid shifts to written RELION metadata offsets."""

    relative = np.asarray(relative_translations, dtype=np.float64)
    previous = np.asarray(previous_absolute_translation, dtype=np.float64)
    if relative.ndim != 2 or relative.shape[1] != 2:
        raise ValueError(
            f"relative_translations must have shape (N, 2), got {relative.shape}"
        )
    if previous.shape != (2,):
        raise ValueError(
            f"previous_absolute_translation must have shape (2,), got {previous.shape}"
        )
    search_base = relion_round_away_from_zero(previous).astype(np.float64)
    return relative + search_base[None, :]


def rotation_error_deg(candidate: np.ndarray, target: np.ndarray) -> float:
    relative = np.asarray(candidate, dtype=np.float64) @ np.asarray(
        target, dtype=np.float64
    ).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def load_autonomous_pose(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        class_index = int(
            archive[f"class_assignments_by_image_iter_{ITERATION_KEY}"][
                TARGET_ORIGINAL_INDEX
            ]
        )
        euler_deg = np.asarray(
            archive[f"best_rotation_eulers_by_image_iter_{ITERATION_KEY}"][
                TARGET_ORIGINAL_INDEX
            ],
            dtype=np.float64,
        )
        translation_pixels = np.asarray(
            archive[f"best_translations_by_image_iter_{ITERATION_KEY}"][
                TARGET_ORIGINAL_INDEX
            ],
            dtype=np.float64,
        )
    return {
        "class_index_zero_based": class_index,
        "euler_deg": euler_deg,
        "rotation": np.asarray(
            utils.R_from_relion(euler_deg, degrees=True)[0],
            dtype=np.float64,
        ),
        "translation_pixels": translation_pixels,
    }


def load_relion_pose(path: Path) -> tuple[dict[str, object], float, int]:
    tables = starfile.read(path, always_dict=True)
    particles = tables["particles"]
    matches = particles[
        particles["rlnImageName"].astype(str) == TARGET_IDENTITY
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {TARGET_IDENTITY!r} row in {path}, found {len(matches)}"
        )
    row_index = int(matches.index[0])
    row = matches.iloc[0]
    optics_group = int(row["rlnOpticsGroup"])
    optics = tables["optics"]
    optics_match = optics[optics["rlnOpticsGroup"].astype(int) == optics_group]
    if len(optics_match) != 1:
        raise ValueError(f"expected one optics-group row for {optics_group}")
    voxel_size = float(optics_match.iloc[0]["rlnImagePixelSize"])
    euler_deg = np.asarray(
        [row["rlnAngleRot"], row["rlnAngleTilt"], row["rlnAnglePsi"]],
        dtype=np.float64,
    )
    translation_angstrom = np.asarray(
        [row["rlnOriginXAngst"], row["rlnOriginYAngst"]],
        dtype=np.float64,
    )
    return (
        {
            "class_index_zero_based": int(row["rlnClassNumber"]) - 1,
            "euler_deg": euler_deg,
            "rotation": np.asarray(
                utils.R_from_relion(euler_deg, degrees=True)[0],
                dtype=np.float64,
            ),
            "translation_pixels": translation_angstrom / voxel_size,
            "translation_angstrom": translation_angstrom,
        },
        voxel_size,
        row_index,
    )


def pose_error(
    *,
    rotation: np.ndarray,
    translation_pixels: np.ndarray,
    target: dict[str, object],
    voxel_size: float,
) -> dict[str, float]:
    translation_delta_pixels = np.asarray(translation_pixels) - np.asarray(
        target["translation_pixels"]
    )
    return {
        "rotation_deg": rotation_error_deg(rotation, target["rotation"]),
        "translation_pixels": float(np.linalg.norm(translation_delta_pixels)),
        "translation_angstrom": float(
            np.linalg.norm(translation_delta_pixels) * voxel_size
        ),
    }


def candidate_record(
    *,
    arrays: dict[str, np.ndarray],
    absolute_translations: np.ndarray,
    rotation_row: int,
    translation_index: int,
    voxel_size: float,
) -> dict[str, object]:
    relative_translation = np.asarray(
        arrays["fine_translations"][translation_index], dtype=np.float64
    )
    absolute_translation = np.asarray(
        absolute_translations[translation_index], dtype=np.float64
    )
    score_pre_prior = float(
        arrays["scores_pre_prior"][rotation_row, translation_index]
    )
    score_with_prior = float(
        arrays["scores_with_prior"][rotation_row, translation_index]
    )
    return {
        "rotation_row": int(rotation_row),
        "oversampled_rotation_index": int(
            arrays["oversampled_rot_indices"][rotation_row]
        ),
        "parent_rotation_row": int(arrays["parent_map"][rotation_row]),
        "translation_index": int(translation_index),
        "fine_translation_parent": int(
            arrays["fine_translation_parent"][translation_index]
        ),
        "candidate_mask": bool(
            arrays["candidate_mask"][rotation_row, translation_index]
        ),
        "translation_relative_pixels": relative_translation.tolist(),
        "translation_absolute_pixels": absolute_translation.tolist(),
        "translation_absolute_angstrom": (
            absolute_translation * voxel_size
        ).tolist(),
        "score_pre_prior": score_pre_prior,
        "score_with_prior": score_with_prior,
        "combined_log_prior": score_with_prior - score_pre_prior,
        "probability": float(arrays["probs"][rotation_row, translation_index]),
    }


def nearest_translation_index(
    absolute_translations: np.ndarray,
    pose: dict[str, object],
) -> tuple[int, float]:
    errors = np.linalg.norm(
        absolute_translations - np.asarray(pose["translation_pixels"]), axis=1
    )
    index = int(np.argmin(errors))
    return index, float(errors[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--relion-prev-data", required=True, type=Path)
    parser.add_argument("--relion-data", required=True, type=Path)
    parser.add_argument("--prior-results", required=True, type=Path)
    parser.add_argument("--phase-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    previous_relion, previous_voxel_size, previous_row_index = load_relion_pose(
        args.relion_prev_data
    )
    relion, voxel_size, relion_row_index = load_relion_pose(args.relion_data)
    if previous_voxel_size != voxel_size:
        raise ValueError(
            "RELION previous/target pixel sizes differ: "
            f"{previous_voxel_size} vs {voxel_size}"
        )
    prior = load_autonomous_pose(args.prior_results)
    phase = load_autonomous_pose(args.phase_results)
    for label, pose in (("relion", relion), ("prior", prior), ("phase", phase)):
        if int(pose["class_index_zero_based"]) != TARGET_CLASS_ZERO_BASED:
            raise ValueError(
                f"{label} target class is {pose['class_index_zero_based']}, "
                f"expected {TARGET_CLASS_ZERO_BASED}"
            )

    with np.load(args.capture, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if int(arrays["original_index"]) != TARGET_ORIGINAL_INDEX:
        raise ValueError("capture original index does not match target")
    if int(arrays["class_index"]) != TARGET_CLASS_ZERO_BASED:
        raise ValueError("capture class does not match target")
    if int(arrays["current_size"]) != 38:
        raise ValueError("capture current size does not match iteration-2 target")

    rotations = np.asarray(arrays["rotations"], dtype=np.float64)
    relative_translations = np.asarray(
        arrays["fine_translations"], dtype=np.float64
    )
    absolute_translations = relative_to_absolute_translations(
        relative_translations,
        np.asarray(previous_relion["translation_pixels"], dtype=np.float64),
    )

    target_indices = {}
    for label, pose in (("relion", relion), ("prior", prior), ("phase", phase)):
        rotation_errors = np.asarray(
            [
                rotation_error_deg(rotation, pose["rotation"])
                for rotation in rotations
            ],
            dtype=np.float64,
        )
        rotation_row = int(np.argmin(rotation_errors))
        translation_index, translation_error = nearest_translation_index(
            absolute_translations, pose
        )
        if float(rotation_errors[rotation_row]) > 1.0e-3:
            raise ValueError(
                f"{label} rotation is absent from capture: "
                f"{rotation_errors[rotation_row]} deg"
            )
        if translation_error > 1.0e-4:
            raise ValueError(
                f"{label} absolute translation is absent from capture: "
                f"{translation_error} pixels"
            )
        target_indices[label] = (rotation_row, translation_index)

    candidate_mask = np.asarray(arrays["candidate_mask"], dtype=bool)
    masked_scores = np.where(
        candidate_mask,
        np.asarray(arrays["scores_with_prior"], dtype=np.float64),
        -np.inf,
    )
    winner_flat = int(np.argmax(masked_scores))
    winner_rotation_row, winner_translation_index = np.unravel_index(
        winner_flat, masked_scores.shape
    )
    winner_rotation = rotations[winner_rotation_row]
    winner_absolute_translation = absolute_translations[winner_translation_index]

    candidate_records = {
        label: candidate_record(
            arrays=arrays,
            absolute_translations=absolute_translations,
            rotation_row=indices[0],
            translation_index=indices[1],
            voxel_size=voxel_size,
        )
        for label, indices in target_indices.items()
    }
    winner = candidate_record(
        arrays=arrays,
        absolute_translations=absolute_translations,
        rotation_row=int(winner_rotation_row),
        translation_index=int(winner_translation_index),
        voxel_size=voxel_size,
    )
    winner["errors"] = {
        label: pose_error(
            rotation=winner_rotation,
            translation_pixels=winner_absolute_translation,
            target=pose,
            voxel_size=voxel_size,
        )
        for label, pose in (("relion", relion), ("prior", prior), ("phase", phase))
    }

    matches_relion = (
        winner["errors"]["relion"]["rotation_deg"] <= 1.0e-3
        and winner["errors"]["relion"]["translation_pixels"] <= 1.0e-4
    )
    matches_phase = (
        winner["errors"]["phase"]["rotation_deg"] <= 1.0e-3
        and winner["errors"]["phase"]["translation_pixels"] <= 1.0e-4
    )
    if matches_relion:
        classification = (
            "fixed_relion_state_phaseffi_selects_relion_winner__"
            "autonomous_regression_is_upstream"
        )
    elif matches_phase:
        classification = (
            "fixed_relion_state_phaseffi_reproduces_away_winner__"
            "phase_score_path_is_causal"
        )
    else:
        classification = "fixed_relion_state_phaseffi_selects_third_winner"

    relion_candidate = candidate_records["relion"]
    phase_candidate = candidate_records["phase"]
    report = {
        "schema": "recovar.k4_it2_orig53722_fixed_state_target.v2",
        "status": "complete",
        "classification": classification,
        "scorecard_change_admissible": False,
        "metric_policy": (
            "candidate support, score margins, and pose errors only; map "
            "acceptance remains FSC/FSC-AUC and correlation is not computed"
        ),
        "translation_frame": {
            "capture": "relative pass-2 search-grid pixels",
            "comparison": "absolute metadata pixels",
            "conversion": "round_away_from_zero(previous_absolute) + relative",
            "previous_relion_row_index_zero_based": previous_row_index,
            "previous_absolute_pixels": np.asarray(
                previous_relion["translation_pixels"]
            ).tolist(),
            "integer_search_base_pixels": relion_round_away_from_zero(
                np.asarray(previous_relion["translation_pixels"])
            ).tolist(),
        },
        "target": {
            "identity": TARGET_IDENTITY,
            "original_index_zero_based": TARGET_ORIGINAL_INDEX,
            "relion_star_row_index_zero_based": relion_row_index,
            "class_index_zero_based": TARGET_CLASS_ZERO_BASED,
            "physical_iteration": 2,
            "current_size": int(arrays["current_size"]),
            "voxel_size_angstrom": voxel_size,
        },
        "poses": {
            label: {
                "euler_deg": np.asarray(pose["euler_deg"]).tolist(),
                "translation_absolute_pixels": np.asarray(
                    pose["translation_pixels"]
                ).tolist(),
                "translation_absolute_angstrom": (
                    np.asarray(pose["translation_pixels"]) * voxel_size
                ).tolist(),
            }
            for label, pose in (("relion", relion), ("prior", prior), ("phase", phase))
        },
        "candidate_records": candidate_records,
        "phase_minus_relion_candidate": {
            "score_pre_prior": (
                phase_candidate["score_pre_prior"]
                - relion_candidate["score_pre_prior"]
            ),
            "score_with_prior": (
                phase_candidate["score_with_prior"]
                - relion_candidate["score_with_prior"]
            ),
            "combined_log_prior": (
                phase_candidate["combined_log_prior"]
                - relion_candidate["combined_log_prior"]
            ),
            "probability": (
                phase_candidate["probability"]
                - relion_candidate["probability"]
            ),
        },
        "winner": winner,
        "candidate_count": int(np.count_nonzero(candidate_mask)),
        "inputs": {
            "capture": {
                "path": str(args.capture.resolve()),
                "sha256": sha256(args.capture),
            },
            "relion_prev_data": {
                "path": str(args.relion_prev_data.resolve()),
                "sha256": sha256(args.relion_prev_data),
            },
            "relion_data": {
                "path": str(args.relion_data.resolve()),
                "sha256": sha256(args.relion_data),
            },
            "prior_results": {
                "path": str(args.prior_results.resolve()),
                "sha256": sha256(args.prior_results),
            },
            "phase_results": {
                "path": str(args.phase_results.resolve()),
                "sha256": sha256(args.phase_results),
            },
        },
        "supersedes": {
            "schema": "recovar.k4_it2_orig53722_fixed_state_target.v1",
            "reason": (
                "v1 compared relative pass-2 translations directly with "
                "absolute metadata offsets"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
