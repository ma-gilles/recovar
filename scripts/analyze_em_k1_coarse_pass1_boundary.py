#!/usr/bin/env python3
"""Classify the K=1 case-22 coarse pass-1 boundary on a fixed cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.validate_relion_coarse_pass1_capture import (
    CoarsePass1Artifact,
    validate_directory,
)

SCORE_P95_GATE = 1e-4
SCORE_MAX_GATE = 1e-3
POSTERIOR_TV_GATE = 1e-4
RELION_INVALID_DIFF2 = np.finfo(np.float32).min


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stats(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    _require(absolute.size > 0, "cannot summarize an empty array")
    _require(np.all(np.isfinite(absolute)), "cannot summarize non-finite values")
    return {
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def _score_gate(stats: dict[str, float]) -> bool:
    return stats["p95_abs"] <= SCORE_P95_GATE and stats["max_abs"] < SCORE_MAX_GATE


def _center_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.max(values)


def _relion_prior_support(raw_diff2: np.ndarray) -> np.ndarray:
    """Return candidates for which RELION evaluated a finite-prior score."""

    raw_diff2 = np.asarray(raw_diff2, dtype=np.float32)
    return raw_diff2 != RELION_INVALID_DIFF2


def _map_relion_rotations_to_recovar(
    values: np.ndarray,
    *,
    n_directions: int,
    n_psi: int,
) -> np.ndarray:
    """Convert RELION direction-major rows to RECOVAR psi-major rows."""

    values = np.asarray(values)
    _require(
        values.shape[0] == n_directions * n_psi,
        "RELION rotation count does not match direction/psi topology",
    )
    trailing = values.shape[1:]
    shaped = values.reshape(n_directions, n_psi, *trailing)
    axes = (1, 0, *range(2, shaped.ndim))
    return shaped.transpose(axes).reshape(values.shape)


def _relion_parent_to_recovar(
    relion_orientation_class_key: int,
    *,
    n_directions: int,
    n_psi: int,
) -> int:
    """Convert one RELION direction-major parent key to RECOVAR psi-major."""

    n_parents = n_directions * n_psi
    _require(
        0 <= relion_orientation_class_key < n_parents,
        "RELION orientation-class key is out of range",
    )
    direction_id, psi_id = divmod(relion_orientation_class_key, n_psi)
    return psi_id * n_directions + direction_id


def _translation_permutation(
    relion_translations: np.ndarray,
    recovar_translations: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map RELION Å rows to RECOVAR pixel rows with a per-particle offset."""

    relion = np.asarray(relion_translations, dtype=np.float64)
    recovar = np.asarray(recovar_translations, dtype=np.float64)
    _require(relion.shape == recovar.shape, "translation grids have different shapes")
    _require(relion.ndim == 2 and relion.shape[1] == 2, "translations must be 2D")
    relion_centered = relion - np.mean(relion, axis=0, keepdims=True)
    recovar_centered = recovar - np.mean(recovar, axis=0, keepdims=True)
    denominator = float(np.sum(relion_centered**2))
    scale = (
        float(np.sqrt(np.sum(recovar_centered**2) / denominator))
        if denominator > 0
        else 1.0
    )
    offset = np.mean(recovar, axis=0) - scale * np.mean(relion, axis=0)
    scaled = relion * scale + offset
    distances = np.linalg.norm(
        scaled[:, None, :] - recovar[None, :, :],
        axis=2,
    )
    source, target = linear_sum_assignment(distances)
    _require(
        np.array_equal(source, np.arange(relion.shape[0])),
        "translation assignment did not cover RELION rows",
    )
    assigned = distances[source, target]
    max_error = float(np.max(assigned))
    rms_error = float(np.sqrt(np.mean(assigned**2)))
    permutation = target.astype(np.int64)
    grid_scale = max(float(np.max(np.linalg.norm(recovar, axis=1))), 1.0)
    _require(
        max_error <= 1e-5 * grid_scale,
        f"translation grids do not match under scale/offset mapping: {max_error}",
    )
    return permutation, {
        "mapping": "hungarian_after_positive_scale_and_offset_fit",
        "axis_order": "relion_xy_equals_recovar_xy",
        "axis_sign": "relion_xy_equals_recovar_xy",
        "scale_relion_to_recovar": float(scale),
        "offset_relion_to_recovar": np.asarray(offset, dtype=np.float64).tolist(),
        "max_coordinate_error": float(max_error),
        "rms_coordinate_error": float(rms_error),
    }


def _map_relion_table(
    values: np.ndarray,
    *,
    n_directions: int,
    n_psi: int,
    relion_to_recovar_translation: np.ndarray,
) -> np.ndarray:
    rotation_mapped = _map_relion_rotations_to_recovar(
        values,
        n_directions=n_directions,
        n_psi=n_psi,
    )
    output = np.empty_like(rotation_mapped)
    output[:, relion_to_recovar_translation] = rotation_mapped
    return output


def _load_recovar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        scores_pre = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64)
        scores_with = np.asarray(payload["scores_with_prior_per_class"], dtype=np.float64)
        _require(scores_pre.ndim == 3 and scores_pre.shape[0] == 1, "expected K=1 pre-prior scores")
        _require(scores_with.shape == scores_pre.shape, "RECOVAR score shapes differ")
        n_rot, n_trans = scores_pre.shape[1:]
        weights = np.asarray(payload["weights_per_class"], dtype=np.float64)
        _require(weights.size == n_rot * n_trans, "RECOVAR weight topology mismatch")
        significant = np.asarray(payload["significant_mask"], dtype=bool)
        _require(significant.size == n_rot * n_trans, "RECOVAR mask topology mismatch")
        rotations = np.asarray(payload["rotations"], dtype=np.float64)
        translations = np.asarray(payload["translations"], dtype=np.float64)
        _require(rotations.shape == (n_rot, 3, 3), "RECOVAR rotation topology mismatch")
        _require(translations.shape == (n_trans, 2), "RECOVAR translation topology mismatch")
        return {
            "path": str(path.resolve()),
            "original_index": int(np.asarray(payload["original_index"]).item()),
            "debug_iteration": int(np.asarray(payload["debug_iteration"]).item()),
            "current_size": int(np.asarray(payload["current_size"]).item()),
            "scores_pre": scores_pre[0],
            "scores_with": scores_with[0],
            "probabilities": weights.reshape(n_rot, n_trans),
            "significant_mask": significant.reshape(n_rot, n_trans),
            "rotations": rotations,
            "translations": translations,
            "hard_assignment": int(np.asarray(payload["hard_assignment"]).item()),
            "n_significant": int(np.asarray(payload["n_significant"]).item()),
        }


def _recovar_by_original_index(directory: Path) -> dict[int, dict[str, Any]]:
    paths = sorted(Path(directory).glob("*.npz"))
    _require(bool(paths), f"no RECOVAR significance dumps in {directory}")
    result = {}
    for path in paths:
        artifact = _load_recovar(path)
        index = artifact["original_index"]
        _require(index not in result, f"duplicate RECOVAR original index {index}")
        result[index] = artifact
    return result


def _parent_row(
    *,
    relion_orientation_class_key: int,
    canonical_parent_id: int,
    expected_side: str,
    relion_data_scores: np.ndarray,
    recovar_data_scores: np.ndarray,
    relion_with_prior_scores: np.ndarray,
    recovar_with_prior_scores: np.ndarray,
    relion_probabilities: np.ndarray,
    recovar_probabilities: np.ndarray,
    relion_mask: np.ndarray,
    recovar_mask: np.ndarray,
    common_prior_support: np.ndarray,
    prior_support_exact: np.ndarray,
) -> dict[str, Any]:
    parent_id = canonical_parent_id
    _require(0 <= parent_id < relion_mask.shape[0], "target parent is out of range")
    _require(
        np.count_nonzero(common_prior_support[parent_id]) > 0,
        "target parent has no common finite-prior support",
    )
    raw_stats = _stats(
        _center_max(recovar_data_scores[parent_id][common_prior_support[parent_id]])
        - _center_max(relion_data_scores[parent_id][common_prior_support[parent_id]])
    )
    finite_with_prior = np.isfinite(relion_with_prior_scores[parent_id])
    finite_with_prior &= np.isfinite(recovar_with_prior_scores[parent_id])
    _require(np.count_nonzero(finite_with_prior) > 0, "target parent has no finite with-prior scores")
    prior_stats = _stats(
        _center_max(recovar_with_prior_scores[parent_id][finite_with_prior])
        - _center_max(relion_with_prior_scores[parent_id][finite_with_prior])
    )
    relion_present = bool(np.any(relion_mask[parent_id]))
    recovar_present = bool(np.any(recovar_mask[parent_id]))
    side_reproduced = (
        relion_present and not recovar_present
        if expected_side == "relion_only"
        else recovar_present and not relion_present
    )
    return {
        "relion_orientation_class_key": int(relion_orientation_class_key),
        "canonical_parent_id": int(canonical_parent_id),
        "expected_side": expected_side,
        "relion_significant": relion_present,
        "recovar_significant": recovar_present,
        "expected_side_reproduced": side_reproduced,
        "prior_support_exact": bool(np.all(prior_support_exact[parent_id])),
        "common_prior_support_count": int(
            np.count_nonzero(common_prior_support[parent_id])
        ),
        "raw_centered_score_diff": raw_stats,
        "raw_score_arithmetic_gate_passed": _score_gate(raw_stats),
        "with_prior_centered_score_diff": prior_stats,
        "with_prior_score_arithmetic_gate_passed": _score_gate(prior_stats),
        "relion_max_probability": float(np.max(relion_probabilities[parent_id])),
        "recovar_max_probability": float(np.max(recovar_probabilities[parent_id])),
        "relion_significant_translation_count": int(np.count_nonzero(relion_mask[parent_id])),
        "recovar_significant_translation_count": int(np.count_nonzero(recovar_mask[parent_id])),
    }


def _compare_particle(
    *,
    row: dict[str, Any],
    relion: CoarsePass1Artifact,
    recovar: dict[str, Any],
) -> dict[str, Any]:
    n_directions, n_psi, n_trans = relion.header[10:13]
    _require(relion.header[5] == recovar["debug_iteration"] == 2, "iteration mismatch")
    _require(relion.stack_index == row["stack_index_one_based"], "RELION stack mismatch")
    _require(relion.part_id == row["relion_part_id"], "RELION part mismatch")
    _require(recovar["original_index"] == row["original_index_zero_based"], "RECOVAR index mismatch")
    _require(recovar["scores_pre"].shape == (n_directions * n_psi, n_trans), "score topology mismatch")

    translation_permutation, translation_details = _translation_permutation(
        relion.translations,
        recovar["translations"],
    )
    relion_raw = _map_relion_table(
        relion.raw_diff2,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    )
    relion_weights = _map_relion_table(
        relion.weights,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(np.float64)
    relion_mask = _map_relion_table(
        relion.significant_mask,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(bool)
    relion_data_scores = -np.asarray(relion_raw, dtype=np.float64)
    recovar_data_scores = np.asarray(recovar["scores_pre"], dtype=np.float64)
    relion_prior_support = _relion_prior_support(relion_raw)
    recovar_prior_support = np.isfinite(recovar["scores_with"])
    prior_support_exact = relion_prior_support == recovar_prior_support
    common_prior_support = relion_prior_support & recovar_prior_support
    _require(
        np.count_nonzero(common_prior_support) > 0,
        "particle has no common finite-prior support",
    )
    raw_stats = _stats(
        _center_max(recovar_data_scores[common_prior_support])
        - _center_max(relion_data_scores[common_prior_support])
    )

    relion_probabilities = relion_weights / np.sum(relion_weights)
    recovar_probabilities = np.asarray(recovar["probabilities"], dtype=np.float64)
    recovar_probabilities /= np.sum(recovar_probabilities)
    posterior_tv = float(
        0.5 * np.sum(np.abs(recovar_probabilities - relion_probabilities))
    )
    posterior_max_abs = float(
        np.max(np.abs(recovar_probabilities - relion_probabilities))
    )

    relion_with_prior = np.full(relion_weights.shape, -np.inf, dtype=np.float64)
    positive = relion_weights > 0
    relion_with_prior[positive] = np.log(relion_weights[positive])
    recovar_with_prior = np.asarray(recovar["scores_with"], dtype=np.float64)
    common_positive = positive & np.isfinite(recovar_with_prior)
    _require(np.count_nonzero(common_positive) > 0, "no common positive with-prior support")
    with_prior_stats = _stats(
        _center_max(recovar_with_prior[common_positive])
        - _center_max(relion_with_prior[common_positive])
    )
    recovar_mask = np.asarray(recovar["significant_mask"], dtype=bool)
    mask_mismatch_count = int(np.count_nonzero(relion_mask != recovar_mask))
    relion_parent_mask = np.any(relion_mask, axis=1)
    recovar_parent_mask = np.any(recovar_mask, axis=1)
    parent_mask_mismatch_count = int(
        np.count_nonzero(relion_parent_mask != recovar_parent_mask)
    )

    target_parents = []
    for relion_orientation_class_key in row["relion_only_parent_ids"]:
        canonical_parent_id = _relion_parent_to_recovar(
            relion_orientation_class_key,
            n_directions=n_directions,
            n_psi=n_psi,
        )
        target_parents.append(
            _parent_row(
                relion_orientation_class_key=relion_orientation_class_key,
                canonical_parent_id=canonical_parent_id,
                expected_side="relion_only",
                relion_data_scores=relion_data_scores,
                recovar_data_scores=recovar_data_scores,
                relion_with_prior_scores=relion_with_prior,
                recovar_with_prior_scores=recovar_with_prior,
                relion_probabilities=relion_probabilities,
                recovar_probabilities=recovar_probabilities,
                relion_mask=relion_mask,
                recovar_mask=recovar_mask,
                common_prior_support=common_prior_support,
                prior_support_exact=prior_support_exact,
            )
        )
    for relion_orientation_class_key in row["recovar_only_parent_ids"]:
        canonical_parent_id = _relion_parent_to_recovar(
            relion_orientation_class_key,
            n_directions=n_directions,
            n_psi=n_psi,
        )
        target_parents.append(
            _parent_row(
                relion_orientation_class_key=relion_orientation_class_key,
                canonical_parent_id=canonical_parent_id,
                expected_side="recovar_only",
                relion_data_scores=relion_data_scores,
                recovar_data_scores=recovar_data_scores,
                relion_with_prior_scores=relion_with_prior,
                recovar_with_prior_scores=recovar_with_prior,
                relion_probabilities=relion_probabilities,
                recovar_probabilities=recovar_probabilities,
                relion_mask=relion_mask,
                recovar_mask=recovar_mask,
                common_prior_support=common_prior_support,
                prior_support_exact=prior_support_exact,
            )
        )

    raw_pass = _score_gate(raw_stats)
    prior_pass = _score_gate(with_prior_stats)
    posterior_pass = posterior_tv <= POSTERIOR_TV_GATE
    if not np.all(prior_support_exact):
        classification = "coarse_prior_support_difference"
    elif not raw_pass:
        classification = "raw_coarse_score_surface_difference"
    elif not prior_pass:
        classification = "coarse_prior_or_weight_conversion_difference"
    elif not posterior_pass:
        classification = "coarse_posterior_normalization_or_reduction_difference"
    elif mask_mismatch_count:
        classification = "coarse_significance_threshold_difference"
    else:
        classification = "coarse_pass1_exact_under_registered_gates"

    relion_winner = int(np.argmax(relion_probabilities))
    recovar_winner = int(np.argmax(recovar_probabilities))
    return {
        "group": row["group"],
        "stack_index_one_based": row["stack_index_one_based"],
        "original_index_zero_based": row["original_index_zero_based"],
        "relion_part_id": row["relion_part_id"],
        "classification": classification,
        "topology": {
            "n_directions": n_directions,
            "n_psi": n_psi,
            "n_translations": n_trans,
            "candidate_count": int(n_directions * n_psi * n_trans),
        },
        "translation_mapping": translation_details,
        "prior_support": {
            "exact": bool(np.all(prior_support_exact)),
            "mismatch_count": int(np.count_nonzero(~prior_support_exact)),
            "common_count": int(np.count_nonzero(common_prior_support)),
            "relion_count": int(np.count_nonzero(relion_prior_support)),
            "recovar_count": int(np.count_nonzero(recovar_prior_support)),
            "relion_invalid_diff2_sentinel": float(RELION_INVALID_DIFF2),
        },
        "raw_centered_score_diff": raw_stats,
        "raw_score_arithmetic_gate_passed": raw_pass,
        "with_prior_centered_score_diff": with_prior_stats,
        "with_prior_score_arithmetic_gate_passed": prior_pass,
        "common_positive_with_prior_count": int(np.count_nonzero(common_positive)),
        "posterior_total_variation": posterior_tv,
        "posterior_max_abs": posterior_max_abs,
        "posterior_gate_passed": posterior_pass,
        "significant_candidate_mismatch_count": mask_mismatch_count,
        "significant_parent_mismatch_count": parent_mask_mismatch_count,
        "significant_parent_sets_exact": parent_mask_mismatch_count == 0,
        "relion_significant_count": int(np.count_nonzero(relion_mask)),
        "recovar_significant_count": int(np.count_nonzero(recovar_mask)),
        "relion_winner": {
            "rotation_id": relion_winner // n_trans,
            "translation_id": relion_winner % n_trans,
        },
        "recovar_winner": {
            "rotation_id": recovar_winner // n_trans,
            "translation_id": recovar_winner % n_trans,
        },
        "winner_exact": relion_winner == recovar_winner,
        "target_parents": target_parents,
        "artifact_paths": {
            "relion": str(relion.path.resolve()),
            "recovar": recovar["path"],
        },
        "artifact_sha256": {"relion": relion.sha256},
    }


def build_report(
    *,
    cohort_json: Path,
    relion_directory: Path,
    recovar_directory: Path,
) -> dict[str, Any]:
    cohort = json.loads(Path(cohort_json).read_text())
    _require(cohort["selected_particle_count"] == 14, "cohort denominator must be 14")
    _require(cohort["mismatch_particle_count"] == 10, "mismatch denominator must be 10")
    _require(cohort["control_particle_count"] == 4, "control denominator must be 4")
    _require(cohort["mismatch_parent_count"] == 13, "parent denominator must be 13")
    expected_stacks = np.asarray(
        cohort["selected_stack_indices_one_based"], dtype=np.int64
    )
    relion_artifacts, relion_validation = validate_directory(
        relion_directory,
        expected_particles=14,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=1,
    )
    relion_by_stack = {artifact.stack_index: artifact for artifact in relion_artifacts}
    recovar_by_index = _recovar_by_original_index(recovar_directory)
    _require(len(recovar_by_index) == 14, "RECOVAR capture denominator must be 14")

    particles = [
        _compare_particle(
            row=row,
            relion=relion_by_stack[row["stack_index_one_based"]],
            recovar=recovar_by_index[row["original_index_zero_based"]],
        )
        for row in cohort["rows"]
    ]
    mismatch_particles = [row for row in particles if row["group"] == "mismatch"]
    control_particles = [row for row in particles if row["group"] == "control"]
    target_parents = [
        parent for particle in mismatch_particles for parent in particle["target_parents"]
    ]
    _require(len(mismatch_particles) == 10, "evaluated mismatch denominator drift")
    _require(len(control_particles) == 4, "evaluated control denominator drift")
    _require(len(target_parents) == 13, "evaluated parent denominator drift")

    fixed_metric = {
        "mismatch_particles": {
            "prior_support_exact": sum(
                row["prior_support"]["exact"] for row in mismatch_particles
            ),
            "raw_score_arithmetic_passed": sum(
                row["raw_score_arithmetic_gate_passed"] for row in mismatch_particles
            ),
            "with_prior_score_arithmetic_passed": sum(
                row["with_prior_score_arithmetic_gate_passed"]
                for row in mismatch_particles
            ),
            "posterior_gate_passed": sum(
                row["posterior_gate_passed"] for row in mismatch_particles
            ),
            "denominator": 10,
        },
        "target_mismatch_parents": {
            "expected_side_reproduced": sum(
                row["expected_side_reproduced"] for row in target_parents
            ),
            "prior_support_exact": sum(
                row["prior_support_exact"] for row in target_parents
            ),
            "raw_score_arithmetic_passed": sum(
                row["raw_score_arithmetic_gate_passed"] for row in target_parents
            ),
            "with_prior_score_arithmetic_passed": sum(
                row["with_prior_score_arithmetic_gate_passed"]
                for row in target_parents
            ),
            "denominator": 13,
        },
        "controls": {
            "prior_support_exact": sum(
                row["prior_support"]["exact"] for row in control_particles
            ),
            "exact_parent_sets": sum(
                row["significant_parent_sets_exact"] for row in control_particles
            ),
            "raw_score_arithmetic_passed": sum(
                row["raw_score_arithmetic_gate_passed"] for row in control_particles
            ),
            "denominator": 4,
        },
    }
    classifications = {}
    for row in particles:
        classifications[row["classification"]] = (
            classifications.get(row["classification"], 0) + 1
        )
    if fixed_metric["target_mismatch_parents"]["expected_side_reproduced"] < 13:
        classification = "coarse_capture_does_not_reproduce_all_candidate_parent_sides"
    elif fixed_metric["target_mismatch_parents"]["prior_support_exact"] < 13:
        classification = "candidate_parent_difference_originates_in_coarse_prior_support"
    elif fixed_metric["target_mismatch_parents"]["raw_score_arithmetic_passed"] < 13:
        classification = "candidate_parent_difference_originates_in_raw_coarse_scores"
    elif fixed_metric["target_mismatch_parents"]["with_prior_score_arithmetic_passed"] < 13:
        classification = "candidate_parent_difference_originates_in_coarse_priors_or_weight_conversion"
    elif fixed_metric["mismatch_particles"]["posterior_gate_passed"] < 10:
        classification = "candidate_parent_difference_originates_in_coarse_posterior_normalization_or_reduction"
    else:
        classification = "candidate_parent_difference_originates_in_coarse_significance_thresholding"

    return {
        "schema": "recovar-k1-case22-coarse-pass1-boundary-v1",
        "status": "complete",
        "classification": classification,
        "metric_policy": (
            "fixed 10 mismatch particles, 4 controls, and 13 mismatch parents; "
            "raw scores compared only on exact common finite-prior support; "
            "centered score p95 <=1e-4 and max <1e-3; posterior TV <=1e-4; "
            "no correlation"
        ),
        "gates": {
            "centered_score_p95_abs_max": SCORE_P95_GATE,
            "centered_score_max_abs_strictly_below": SCORE_MAX_GATE,
            "posterior_total_variation_max": POSTERIOR_TV_GATE,
        },
        "fixed_metric": fixed_metric,
        "particle_classification_counts": classifications,
        "particles": particles,
        "relion_validation": relion_validation,
        "cohort": str(Path(cohort_json).resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-json", type=Path, required=True)
    parser.add_argument("--relion-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(
        cohort_json=args.cohort_json,
        relion_directory=args.relion_directory,
        recovar_directory=args.recovar_directory,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
