#!/usr/bin/env python3
"""Compare one stopped RECOVAR fine-pass winner with RELION particle state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import starfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _particle_table(path: Path):
    value = starfile.read(path)
    return value["particles"] if isinstance(value, dict) else value


def _relion_euler_to_matrix(eulers_deg: np.ndarray) -> np.ndarray:
    eulers = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    alpha, beta, gamma = np.deg2rad(eulers).T
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    matrices = np.empty((eulers.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = cg * cc - sg * sa
    matrices[:, 0, 1] = cg * cs + sg * ca
    matrices[:, 0, 2] = -cg * sb
    matrices[:, 1, 0] = -sg * cc - cg * sa
    matrices[:, 1, 1] = -sg * cs + cg * ca
    matrices[:, 1, 2] = sg * sb
    matrices[:, 2, 0] = sc
    matrices[:, 2, 1] = ss
    matrices[:, 2, 2] = cb
    return matrices


def _relion_matrix_to_euler(matrix: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
    out = np.empty((matrices.shape[0], 3), dtype=np.float64)
    abs_sb = np.sqrt(matrices[:, 0, 2] ** 2 + matrices[:, 1, 2] ** 2)
    nonsingular = abs_sb > 16.0 * np.finfo(np.float32).eps

    if np.any(nonsingular):
        selected = matrices[nonsingular]
        gamma = np.arctan2(selected[:, 1, 2], -selected[:, 0, 2])
        alpha = np.arctan2(selected[:, 2, 1], selected[:, 2, 0])
        sign_sb = np.empty_like(gamma)
        small = np.abs(np.sin(gamma)) < np.finfo(np.float32).eps
        sign = lambda value: np.where(value >= 0.0, 1.0, -1.0)
        sign_sb[small] = sign(-selected[small, 0, 2] / np.cos(gamma[small]))
        sign_sb[~small] = np.where(
            np.sin(gamma[~small]) > 0.0,
            sign(selected[~small, 1, 2]),
            -sign(selected[~small, 1, 2]),
        )
        beta = np.arctan2(sign_sb * abs_sb[nonsingular], selected[:, 2, 2])
        out[nonsingular] = np.rad2deg(np.stack((alpha, beta, gamma), axis=1))

    if np.any(~nonsingular):
        selected = matrices[~nonsingular]
        positive = selected[:, 2, 2] >= 0.0
        alpha = np.zeros(selected.shape[0], dtype=np.float64)
        beta = np.where(positive, 0.0, np.pi)
        gamma = np.empty(selected.shape[0], dtype=np.float64)
        gamma[positive] = np.arctan2(-selected[positive, 1, 0], selected[positive, 0, 0])
        gamma[~positive] = np.arctan2(selected[~positive, 1, 0], -selected[~positive, 0, 0])
        out[~nonsingular] = np.rad2deg(np.stack((alpha, beta, gamma), axis=1))
    return out


def _angular_error_deg(lhs_matrix: np.ndarray, rhs_eulers: np.ndarray) -> float:
    lhs = np.asarray(lhs_matrix, dtype=np.float64).reshape(3, 3)
    rhs = _relion_euler_to_matrix(rhs_eulers)[0]
    relative = lhs @ rhs.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    skew = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    sine = 0.5 * np.linalg.norm(skew)
    return float(np.degrees(np.arctan2(sine, cosine)))


def _rotation_distances_deg(matrices: np.ndarray, rhs_eulers: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_angular_error_deg(matrix, rhs_eulers) for matrix in np.asarray(matrices)],
        dtype=np.float64,
    )


def analyze(
    *,
    capture_path: Path,
    particle_star: Path,
    relion_data_star: Path,
    voxel_size: float,
    autonomous_results: Path | None = None,
    autonomous_iteration_index: int | None = None,
) -> dict[str, object]:
    with np.load(capture_path, allow_pickle=False) as capture:
        source_row = int(np.asarray(capture["original_index"]).item())
        probabilities = np.asarray(capture["probs"], dtype=np.float64)
        winner = np.unravel_index(int(np.argmax(probabilities)), probabilities.shape)
        rotation_index, translation_index = (int(value) for value in winner)
        winner_matrix = np.asarray(capture["rotations"][rotation_index], dtype=np.float64)
        winner_euler = _relion_matrix_to_euler(winner_matrix)[0]
        integer_pre_shift = np.asarray(capture["relion_integer_pre_shift"], dtype=np.float64)
        fine_translation = np.asarray(capture["fine_translations"][translation_index], dtype=np.float64)
        winner_translation_pixels = integer_pre_shift + fine_translation
        winner_translation_angstrom = winner_translation_pixels * float(voxel_size)
        winner_pmax = float(probabilities[winner])
        support = int(np.asarray(capture["reconstruction_n_significant"]).item())
        physical_iteration = int(np.asarray(capture["iteration"]).item()) + 1
        rotations = np.asarray(capture["rotations"], dtype=np.float64)
        fine_translations = np.asarray(capture["fine_translations"], dtype=np.float64)
        candidate_mask = (
            np.asarray(capture["candidate_mask"], dtype=bool)
            if "candidate_mask" in capture.files
            else np.ones(probabilities.shape, dtype=bool)
        )
        scores_with_prior = (
            np.asarray(capture["scores_with_prior"], dtype=np.float64)
            if "scores_with_prior" in capture.files
            else None
        )
        scores_pre_prior = (
            np.asarray(capture["scores_pre_prior"], dtype=np.float64)
            if "scores_pre_prior" in capture.files
            else None
        )

    particles = _particle_table(particle_star)
    if not 0 <= source_row < len(particles):
        raise ValueError(f"capture source row {source_row} is outside particle STAR")
    identity = str(particles.iloc[source_row]["rlnImageName"])
    relion = _particle_table(relion_data_star)
    matched = relion[relion["rlnImageName"].astype(str) == identity]
    if len(matched) != 1:
        raise ValueError(f"RELION identity {identity!r} matched {len(matched)} rows")
    row = matched.iloc[0]
    relion_euler = np.asarray(
        [row["rlnAngleRot"], row["rlnAngleTilt"], row["rlnAnglePsi"]],
        dtype=np.float64,
    )
    relion_translation = np.asarray(
        [row["rlnOriginXAngst"], row["rlnOriginYAngst"]],
        dtype=np.float64,
    )
    relion_pmax = float(row["rlnMaxValueProbDistribution"])
    relion_support = int(row["rlnNrOfSignificantSamples"])
    rotation_distances = _rotation_distances_deg(rotations, relion_euler)
    closest_rotation = int(np.argmin(rotation_distances))
    translation_candidates_angstrom = (
        fine_translations + integer_pre_shift[None, :]
    ) * float(voxel_size)
    translation_distances = np.linalg.norm(
        translation_candidates_angstrom - relion_translation[None, :],
        axis=1,
    )
    closest_translation = int(np.argmin(translation_distances))
    closest_joint = (closest_rotation, closest_translation)
    active_flat = np.flatnonzero(candidate_mask.reshape(-1))
    ordered_active = active_flat[
        np.argsort(-probabilities.reshape(-1)[active_flat], kind="stable")
    ]
    if active_flat.size < 2:
        raise ValueError("pose-tail capture must contain at least two active candidates")
    winner_flat = int(ordered_active[0])
    runner_up_flat = int(ordered_active[1])
    runner_up = tuple(
        int(value) for value in np.unravel_index(runner_up_flat, probabilities.shape)
    )
    closest_flat = int(np.ravel_multi_index(closest_joint, probabilities.shape))
    closest_rank_positions = np.flatnonzero(ordered_active == closest_flat)
    closest_rank = (
        None if closest_rank_positions.size == 0 else int(closest_rank_positions[0]) + 1
    )

    report: dict[str, object] = {
        "schema": "recovar.em.k1_pose_tail_capture.v1",
        "status": "complete",
        "physical_iteration": physical_iteration,
        "source_row": source_row,
        "rln_image_name": identity,
        "winner_indices": {
            "rotation": rotation_index,
            "translation": translation_index,
        },
        "active_candidate_count": int(active_flat.size),
        "capture_winner": {
            "euler_deg": winner_euler.tolist(),
            "translation_pixels": winner_translation_pixels.tolist(),
            "translation_angstrom": winner_translation_angstrom.tolist(),
            "pmax": winner_pmax,
            "significant_count": support,
        },
        "winner_margin": {
            "runner_up_indices": {
                "rotation": runner_up[0],
                "translation": runner_up[1],
            },
            "runner_up_probability": float(probabilities[runner_up]),
            "probability_gap": float(
                probabilities.reshape(-1)[winner_flat]
                - probabilities.reshape(-1)[runner_up_flat]
            ),
            "score_with_prior_gap": (
                None
                if scores_with_prior is None
                else float(
                    scores_with_prior.reshape(-1)[winner_flat]
                    - scores_with_prior.reshape(-1)[runner_up_flat]
                )
            ),
            "score_pre_prior_gap": (
                None
                if scores_pre_prior is None
                else float(
                    scores_pre_prior.reshape(-1)[winner_flat]
                    - scores_pre_prior.reshape(-1)[runner_up_flat]
                )
            ),
        },
        "relion": {
            "euler_deg": relion_euler.tolist(),
            "translation_angstrom": relion_translation.tolist(),
            "pmax": relion_pmax,
            "significant_count": relion_support,
        },
        "capture_vs_relion": {
            "rotation_geodesic_deg": _angular_error_deg(winner_matrix, relion_euler),
            "translation_l2_angstrom": float(
                np.linalg.norm(winner_translation_angstrom - relion_translation)
            ),
            "pmax_residual": winner_pmax - relion_pmax,
            "significant_count_residual": support - relion_support,
        },
        "relion_pose_in_capture_grid": {
            "closest_rotation_index": closest_rotation,
            "closest_rotation_geodesic_deg": float(rotation_distances[closest_rotation]),
            "closest_translation_index": closest_translation,
            "closest_translation_l2_angstrom": float(translation_distances[closest_translation]),
            "joint_candidate_active": bool(candidate_mask[closest_joint]),
            "joint_candidate_probability": float(probabilities[closest_joint]),
            "joint_candidate_rank": closest_rank,
            "probability_ratio_to_winner": float(
                probabilities[closest_joint] / probabilities.reshape(-1)[winner_flat]
            ),
            "joint_candidate_score_with_prior": (
                None if scores_with_prior is None else float(scores_with_prior[closest_joint])
            ),
            "score_with_prior_delta_from_winner": (
                None
                if scores_with_prior is None
                else float(
                    scores_with_prior[closest_joint]
                    - scores_with_prior.reshape(-1)[winner_flat]
                )
            ),
            "joint_candidate_score_pre_prior": (
                None if scores_pre_prior is None else float(scores_pre_prior[closest_joint])
            ),
            "score_pre_prior_delta_from_winner": (
                None
                if scores_pre_prior is None
                else float(
                    scores_pre_prior[closest_joint]
                    - scores_pre_prior.reshape(-1)[winner_flat]
                )
            ),
        },
        "artifacts": {
            "capture": str(capture_path.resolve()),
            "capture_sha256": _sha256(capture_path),
            "particle_star": str(particle_star.resolve()),
            "particle_star_sha256": _sha256(particle_star),
            "relion_data_star": str(relion_data_star.resolve()),
            "relion_data_star_sha256": _sha256(relion_data_star),
        },
    }

    if (autonomous_results is None) != (autonomous_iteration_index is None):
        raise ValueError("autonomous results and iteration index must be provided together")
    if autonomous_results is not None:
        suffix = f"{int(autonomous_iteration_index):03d}"
        with np.load(autonomous_results, allow_pickle=False) as result:
            autonomous_euler = np.asarray(
                result[f"best_rotation_eulers_by_image_iter_{suffix}"][source_row],
                dtype=np.float64,
            )
            autonomous_translation_pixels = np.asarray(
                result[f"best_translations_by_image_iter_{suffix}"][source_row],
                dtype=np.float64,
            )
            autonomous_pmax = float(result[f"pmax_per_image_by_image_iter_{suffix}"][source_row])
            autonomous_support = int(result[f"sig_counts_by_image_iter_{suffix}"][source_row])
        report["autonomous"] = {
            "iteration_index": int(autonomous_iteration_index),
            "euler_deg": autonomous_euler.tolist(),
            "translation_pixels": autonomous_translation_pixels.tolist(),
            "pmax": autonomous_pmax,
            "significant_count": autonomous_support,
        }
        report["capture_vs_autonomous"] = {
            "rotation_geodesic_deg": _angular_error_deg(winner_matrix, autonomous_euler),
            "translation_l2_pixels": float(
                np.linalg.norm(winner_translation_pixels - autonomous_translation_pixels)
            ),
            "pmax_residual": winner_pmax - autonomous_pmax,
            "significant_count_residual": support - autonomous_support,
        }
        report["artifacts"]["autonomous_results"] = str(autonomous_results.resolve())
        report["artifacts"]["autonomous_results_sha256"] = _sha256(autonomous_results)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--particle-star", required=True, type=Path)
    parser.add_argument("--relion-data-star", required=True, type=Path)
    parser.add_argument("--voxel-size", required=True, type=float)
    parser.add_argument("--autonomous-results", type=Path)
    parser.add_argument("--autonomous-iteration-index", type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        capture_path=args.capture,
        particle_star=args.particle_star,
        relion_data_star=args.relion_data_star,
        voxel_size=args.voxel_size,
        autonomous_results=args.autonomous_results,
        autonomous_iteration_index=args.autonomous_iteration_index,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
