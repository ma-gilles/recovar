#!/usr/bin/env python3
"""Compare native RELION and RECOVAR first-iteration K-class CC surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.validate_relion_firstiter_cc_capture import load_firstiter_cc_capture

SCHEMA = "recovar.em_real_k4_firstiter_cc_boundary.v1"


class AnalysisError(RuntimeError):
    """Raised when a supposedly matched boundary artifact is incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relion_to_recovar_rotation_indices(n_directions: int, n_psi: int) -> np.ndarray:
    """Map RELION direction-major rotations to RECOVAR psi-major rotations."""

    _require(n_directions > 0 and n_psi > 0, "rotation-grid dimensions must be positive")
    relion_indices = np.arange(n_directions * n_psi, dtype=np.int64)
    direction = relion_indices // n_psi
    psi = relion_indices % n_psi
    result = psi * n_directions + direction
    _require(np.unique(result).size == result.size, "rotation-order map is not a permutation")
    return result


def relion_round(values: np.ndarray) -> np.ndarray:
    """Match RELION's ROUND macro (nearest integer, half away from zero)."""

    array = np.asarray(values, dtype=np.float64)
    rounded = np.where(array > 0.0, np.floor(array + 0.5), np.ceil(array - 0.5))
    return rounded.astype(np.int64)


def score_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool]:
    """Measure direct and additive-offset-invariant agreement."""

    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    _require(left.shape == right.shape and left.size > 1, "score vectors have different shapes")
    _require(np.isfinite(left).all() and np.isfinite(right).all(), "score vectors are non-finite")
    delta = right - left
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered))
    _require(denominator > 0.0, "reference score surface is constant")
    return {
        "bitwise_equal": bool(np.array_equal(left, right)),
        "correlation": float(np.corrcoef(left, right)[0, 1]),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(left), 1e-30)),
        "centered_relative_l2": float(
            np.linalg.norm(right_centered - left_centered) / denominator
        ),
        "mean_offset": float(delta.mean()),
    }


def _winner(scores: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.unravel_index(np.argmax(scores), scores.shape))


def _native_coarse_scores(capture) -> tuple[np.ndarray, np.ndarray]:
    _require(capture.pass_index == 0, f"{capture.path} is not a coarse capture")
    n_classes = capture.header[12] - capture.header[11] + 1
    n_directions = capture.header[13]
    n_psi = capture.header[14]
    n_translations = capture.header[15]
    n_rotations = n_directions * n_psi
    native = -np.asarray(capture.candidates["raw_score"], dtype=np.float64).reshape(
        n_classes,
        n_rotations,
        n_translations,
    )
    rotation_map = relion_to_recovar_rotation_indices(n_directions, n_psi)
    reordered = np.empty_like(native)
    reordered[:, rotation_map, :] = native
    return reordered, rotation_map


def _recovar_coarse_scores(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "n_classes",
            "n_rot",
            "n_trans",
            "scores_pre_prior_per_class",
            "score_capture_mode",
        }
        _require(required <= set(payload.files), f"{path} lacks coarse-score fields")
        shape = (
            int(payload["n_classes"].item()),
            int(payload["n_rot"].item()),
            int(payload["n_trans"].item()),
        )
        scores = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64)
        _require(scores.shape == shape, f"{path} has inconsistent coarse-score shape")
        _require(
            str(payload["score_capture_mode"].item()) == "passive_cached_after_support",
            f"{path} is not a passive coarse capture",
        )
    return scores


def _fine_comparison(native_path: Path, recovar_path: Path, rotation_map: np.ndarray) -> dict:
    native = load_firstiter_cc_capture(native_path)
    _require(native.pass_index == 1, f"{native_path} is not a fine capture")
    with np.load(recovar_path, allow_pickle=False) as payload:
        required = {
            "candidate_mask",
            "scores_pre_prior",
            "oversampled_rot_indices",
            "relion_integer_pre_shift",
        }
        _require(required <= set(payload.files), f"{recovar_path} lacks fine-score fields")
        mask = np.asarray(payload["candidate_mask"], dtype=bool)
        recovar_scores = np.asarray(payload["scores_pre_prior"], dtype=np.float64)
        _require(mask.shape == recovar_scores.shape, "RECOVAR fine score/mask shapes differ")
        coordinates = np.argwhere(mask)
        selected_scores = recovar_scores[mask]
        oversampled_rotation_ids = np.asarray(payload["oversampled_rot_indices"], dtype=np.int64)
        integer_pre_shift = np.asarray(payload["relion_integer_pre_shift"], dtype=np.int64)

    native_candidates = native.candidates
    _require(coordinates.shape == (native_candidates.size, 2), "fine candidate counts differ")
    _require(
        np.array_equal(coordinates[:, 0], native_candidates["rotation_local"]),
        "fine child-rotation order differs",
    )
    _require(
        np.array_equal(coordinates[:, 1], native_candidates["translation_id"]),
        "fine child-translation order differs",
    )
    native_parent_ids = np.unique(native_candidates["rotation_id"])
    recovar_parent_ids = np.unique(oversampled_rotation_ids // native.header[16])
    _require(native_parent_ids.size == recovar_parent_ids.size == 1, "fine support has multiple parents")
    native_parent = int(native_parent_ids[0])
    recovar_parent = int(recovar_parent_ids[0])
    _require(native_parent < rotation_map.size, "native fine parent is outside the coarse grid")
    native_scores = -np.asarray(native_candidates["raw_score"], dtype=np.float64)
    return {
        "native_path": str(native_path.resolve()),
        "native_sha256": native.sha256,
        "recovar_path": str(recovar_path.resolve()),
        "recovar_sha256": _sha256(recovar_path),
        "candidate_count": int(native_scores.size),
        "class_one_based": int(native_candidates["class_one_based"][0]),
        "native_parent_rotation": native_parent,
        "native_parent_in_recovar_order": int(rotation_map[native_parent]),
        "recovar_parent_rotation": recovar_parent,
        "parent_rotation_equal": bool(int(rotation_map[native_parent]) == recovar_parent),
        "winner_equal": bool(int(np.argmax(native_scores)) == int(np.argmax(selected_scores))),
        "integer_pre_shift": integer_pre_shift.tolist(),
        "score": score_metrics(native_scores, selected_scores),
    }


def _input_origins(path: Path, voxel_size: float) -> tuple[object, np.ndarray]:
    import starfile

    document = starfile.read(path)
    particles = document["particles"] if isinstance(document, dict) else document
    if {"rlnOriginXAngst", "rlnOriginYAngst"} <= set(particles.columns):
        _require(voxel_size > 0.0, "a positive voxel size is required for Angstrom origins")
        origins = np.stack(
            [particles["rlnOriginXAngst"], particles["rlnOriginYAngst"]],
            axis=1,
        ).astype(np.float64) / float(voxel_size)
    elif {"rlnOriginX", "rlnOriginY"} <= set(particles.columns):
        origins = np.stack([particles["rlnOriginX"], particles["rlnOriginY"]], axis=1).astype(
            np.float64
        )
    else:
        origins = np.zeros((len(particles), 2), dtype=np.float64)
    return particles, origins


def analyze(
    *,
    panel_path: Path,
    native_dir: Path,
    recovar_coarse_dir: Path,
    recovar_fine_dir: Path,
    input_data_star: Path,
    voxel_size: float,
) -> dict:
    panel = json.loads(panel_path.read_text())
    _require(panel.get("schema") == "bpref-factor-stratification-v1", "panel schema changed")
    selected = panel.get("selected")
    _require(isinstance(selected, list) and selected, "panel is empty")
    particles, origins = _input_origins(input_data_star, voxel_size)
    _require(len(particles) == origins.shape[0], "input origin table shape changed")

    rows = []
    for record in selected:
        part_id = int(record["recovar_source_row_zero_based"])
        stack_id = int(record["stack_index_1based"])
        image_name = str(particles.iloc[part_id]["rlnImageName"])
        _require(int(image_name.split("@", 1)[0]) == stack_id, "panel/input image identity differs")
        native_coarse_path = native_dir / f"part{part_id}_stack{stack_id}_pass0.firstiter-cc-v1.bin"
        native_fine_path = native_dir / f"part{part_id}_stack{stack_id}_pass1.firstiter-cc-v1.bin"
        recovar_coarse_path = (
            recovar_coarse_dir / f"significance_orig{part_id:06d}_it001_cs020.npz"
        )
        recovar_fine_path = recovar_fine_dir / f"pass2_orig{part_id:06d}_cs048.npz"
        native_capture = load_firstiter_cc_capture(native_coarse_path)
        native_scores, rotation_map = _native_coarse_scores(native_capture)
        recovar_scores = _recovar_coarse_scores(recovar_coarse_path)
        _require(native_scores.shape == recovar_scores.shape, "coarse score topology differs")
        native_winner = _winner(native_scores)
        recovar_winner = _winner(recovar_scores)
        class_winners_equal = []
        for class_index in range(native_scores.shape[0]):
            class_winners_equal.append(
                _winner(native_scores[class_index]) == _winner(recovar_scores[class_index])
            )
        fine = _fine_comparison(native_fine_path, recovar_fine_path, rotation_map)
        expected_pre_shift = relion_round(origins[part_id])
        fine["expected_integer_pre_shift"] = expected_pre_shift.tolist()
        fine["integer_pre_shift_equal"] = bool(
            np.array_equal(fine["integer_pre_shift"], expected_pre_shift)
        )
        rows.append(
            {
                "stratum": str(record["stratum"]),
                "part_id": part_id,
                "stack_index_1based": stack_id,
                "native_coarse_path": str(native_coarse_path.resolve()),
                "native_coarse_sha256": native_capture.sha256,
                "recovar_coarse_path": str(recovar_coarse_path.resolve()),
                "recovar_coarse_sha256": _sha256(recovar_coarse_path),
                "native_winner": list(native_winner),
                "recovar_winner": list(recovar_winner),
                "winner_class_equal": bool(native_winner[0] == recovar_winner[0]),
                "winner_pose_equal": bool(native_winner == recovar_winner),
                "per_class_winner_pose_equal_count": int(sum(class_winners_equal)),
                "coarse_score": score_metrics(native_scores, recovar_scores),
                "fine": fine,
            }
        )

    correlations = np.asarray([row["coarse_score"]["correlation"] for row in rows])
    centered_errors = np.asarray(
        [row["coarse_score"]["centered_relative_l2"] for row in rows]
    )
    fine_max_abs = np.asarray([row["fine"]["score"]["max_abs"] for row in rows])
    summary = {
        "particle_count": len(rows),
        "coarse_candidate_count": int(sum(np.prod((4, 576, 29)) for _ in rows)),
        "winner_class_equal_count": int(sum(row["winner_class_equal"] for row in rows)),
        "winner_pose_equal_count": int(sum(row["winner_pose_equal"] for row in rows)),
        "per_class_winner_pose_equal_count": int(
            sum(row["per_class_winner_pose_equal_count"] for row in rows)
        ),
        "fine_parent_equal_count": int(sum(row["fine"]["parent_rotation_equal"] for row in rows)),
        "fine_winner_equal_count": int(sum(row["fine"]["winner_equal"] for row in rows)),
        "integer_pre_shift_equal_count": int(
            sum(row["fine"]["integer_pre_shift_equal"] for row in rows)
        ),
        "coarse_correlation_minimum": float(correlations.min()),
        "coarse_correlation_median": float(np.median(correlations)),
        "coarse_centered_relative_l2_maximum": float(centered_errors.max()),
        "fine_score_max_abs_maximum": float(fine_max_abs.max()),
    }
    summary["boundary_closed"] = bool(
        summary["winner_class_equal_count"] == len(rows)
        and summary["winner_pose_equal_count"] == len(rows)
        and summary["per_class_winner_pose_equal_count"] == 4 * len(rows)
        and summary["fine_parent_equal_count"] == len(rows)
        and summary["fine_winner_equal_count"] == len(rows)
        and summary["integer_pre_shift_equal_count"] == len(rows)
        and summary["coarse_correlation_minimum"] > 0.999999
        and summary["fine_score_max_abs_maximum"] < 1e-6
    )
    return {
        "schema": SCHEMA,
        "panel_path": str(panel_path.resolve()),
        "panel_sha256": _sha256(panel_path),
        "input_data_star": str(input_data_star.resolve()),
        "input_data_star_sha256": _sha256(input_data_star),
        "voxel_size_angstrom": float(voxel_size),
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--recovar-coarse-dir", type=Path, required=True)
    parser.add_argument("--recovar-fine-dir", type=Path, required=True)
    parser.add_argument("--input-data-star", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        panel_path=args.panel,
        native_dir=args.native_dir,
        recovar_coarse_dir=args.recovar_coarse_dir,
        recovar_fine_dir=args.recovar_fine_dir,
        input_data_star=args.input_data_star,
        voxel_size=args.voxel_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if not report["summary"]["boundary_closed"]:
        raise SystemExit("K-class firstiter-CC boundary remains open")


if __name__ == "__main__":
    main()
