#!/usr/bin/env python3
"""Audit real-data K-class InitialModel trajectories against RELION.

This is a permutation-invariant cross-engine audit.  It deliberately does not
claim gold-standard resolution: native InitialModel writes one map per class,
not independent half maps.  The report therefore records map FSC/FSC-AUC,
hard assignments, and class populations while marking half-map evidence as
unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from recovar.data_io.starfile import read_star

if __package__:
    from scripts.summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc
else:
    from summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc


class AuditError(RuntimeError):
    """Raised when a required trajectory artifact is missing or ambiguous."""


def _pairwise_fsc_auc(
    candidate_maps: list[np.ndarray],
    reference_maps: list[np.ndarray],
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    if not candidate_maps or len(candidate_maps) != len(reference_maps):
        raise ValueError("candidate and reference map lists must have the same non-zero length")
    shape = candidate_maps[0].shape
    if any(volume.shape != shape for volume in (*candidate_maps, *reference_maps)):
        raise ValueError("all class maps must have the same shape")
    scores = np.empty((len(candidate_maps), len(reference_maps)), dtype=np.float64)
    curves: dict[tuple[int, int], np.ndarray] = {}
    for candidate_class, candidate in enumerate(candidate_maps):
        for reference_class, reference in enumerate(reference_maps):
            curve = np.asarray(shell_fsc(candidate, reference), dtype=np.float64)
            curves[(candidate_class, reference_class)] = curve
            scores[candidate_class, reference_class] = normalized_fsc_auc(curve)
    return scores, curves


def _best_class_permutation(score_matrix: np.ndarray) -> tuple[int, ...]:
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"class score matrix must be non-empty and square, got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("class score matrix must be finite")
    rows, columns = linear_sum_assignment(-scores)
    permutation = np.empty(scores.shape[0], dtype=np.int64)
    permutation[rows] = columns
    return tuple(int(value) for value in permutation)


def _class_assignment_accuracy(
    candidate_labels: np.ndarray,
    reference_labels: np.ndarray,
    permutation: tuple[int, ...],
) -> float:
    candidate = np.asarray(candidate_labels, dtype=np.int64).reshape(-1)
    reference = np.asarray(reference_labels, dtype=np.int64).reshape(-1)
    if candidate.shape != reference.shape or candidate.size == 0:
        raise ValueError("candidate and reference assignments must have the same non-zero shape")
    mapping = np.asarray(permutation, dtype=np.int64)
    if np.any(candidate < 0) or np.any(candidate >= mapping.size):
        raise ValueError("candidate class assignments are outside the permutation")
    return float(np.mean(mapping[candidate] == reference))


def _column(table, names: tuple[str, ...], *, path: Path) -> str:
    name = next((candidate for candidate in names if candidate in table.columns), None)
    if name is None:
        raise AuditError(f"{path} has none of the required columns {names}")
    return name


def _assignments_by_image(path: Path, K: int) -> dict[str, int]:
    table, _ = read_star(str(path))
    image_column = _column(table, ("_rlnImageName", "rlnImageName"), path=path)
    class_column = _column(table, ("_rlnClassNumber", "rlnClassNumber"), path=path)
    images = [str(value) for value in table[image_column]]
    if len(images) != len(set(images)):
        raise AuditError(f"{path} contains duplicate image identities")
    labels = np.asarray(table[class_column], dtype=np.int64) - 1
    if np.any(labels < -1) or np.any(labels >= K):
        raise AuditError(f"{path} contains class labels outside 0..{K}")
    return dict(zip(images, (int(value) for value in labels)))


def _population(labels: np.ndarray, K: int, minimum_class_fraction: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    assigned = labels[labels >= 0]
    counts = np.bincount(assigned, minlength=K)[:K]
    total = int(np.sum(counts))
    fractions = counts.astype(np.float64) / total if total else np.zeros(K, dtype=np.float64)
    collapsed = (
        [int(index + 1) for index, value in enumerate(fractions) if value < minimum_class_fraction]
        if total
        else []
    )
    return {
        "particles": int(labels.size),
        "assigned_particles": total,
        "unassigned_particles": int(labels.size - total),
        "evaluable": bool(total),
        "counts": [int(value) for value in counts],
        "fractions": [float(value) for value in fractions],
        "minimum_fraction": float(np.min(fractions)) if total else None,
        "minimum_class_fraction_threshold": float(minimum_class_fraction),
        "collapsed_class_ids_one_based": collapsed,
        "collapse_detected": bool(collapsed),
    }


def _matched_assignments(
    candidate_star: Path,
    reference_star: Path,
    permutation: tuple[int, ...],
    minimum_class_fraction: float,
) -> dict[str, Any]:
    K = len(permutation)
    candidate_by_image = _assignments_by_image(candidate_star, K)
    reference_by_image = _assignments_by_image(reference_star, K)
    candidate_images = set(candidate_by_image)
    reference_images = set(reference_by_image)
    common = sorted(candidate_images.intersection(reference_images))
    if not common:
        raise AuditError("candidate and reference data STAR files share no image identities")
    candidate_labels = np.asarray([candidate_by_image[image] for image in common], dtype=np.int64)
    reference_labels = np.asarray([reference_by_image[image] for image in common], dtype=np.int64)
    candidate_assigned = candidate_labels >= 0
    reference_assigned = reference_labels >= 0
    jointly_assigned = candidate_assigned & reference_assigned
    candidate_all_labels = np.fromiter(candidate_by_image.values(), dtype=np.int64)
    reference_all_labels = np.fromiter(reference_by_image.values(), dtype=np.int64)
    candidate_only = sorted(candidate_images - reference_images)
    reference_only = sorted(reference_images - candidate_images)
    return {
        "common_particles": len(common),
        "candidate_particles": len(candidate_by_image),
        "reference_particles": len(reference_by_image),
        "image_identity_sets_match": not candidate_only and not reference_only,
        "candidate_only_particle_count": len(candidate_only),
        "reference_only_particle_count": len(reference_only),
        "candidate_only_examples": candidate_only[:10],
        "reference_only_examples": reference_only[:10],
        "assignment_status_match": bool(np.array_equal(candidate_assigned, reference_assigned)),
        "common_assigned_particles": int(np.sum(jointly_assigned)),
        "accuracy": (
            _class_assignment_accuracy(
                candidate_labels[jointly_assigned],
                reference_labels[jointly_assigned],
                permutation,
            )
            if np.any(jointly_assigned)
            else None
        ),
        "candidate_population": _population(candidate_all_labels, K, minimum_class_fraction),
        "reference_population": _population(reference_all_labels, K, minimum_class_fraction),
    }


def audit_trajectory(
    *,
    candidate_dir: Path,
    reference_dir: Path,
    K: int,
    checkpoints: tuple[int, ...],
    minimum_fsc_auc: float,
    minimum_assignment_accuracy: float,
    minimum_class_fraction: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if K < 2:
        raise ValueError("K-class audit requires K >= 2")
    if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be sorted and unique")
    if not 0.0 <= minimum_class_fraction < 1.0 / K:
        raise ValueError("minimum_class_fraction must be in [0, 1/K)")

    shellwise: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []
    for iteration in checkpoints:
        candidate_maps = [candidate_dir / f"run_it{iteration:03d}_class{index:03d}.mrc" for index in range(1, K + 1)]
        reference_maps = [reference_dir / f"run_it{iteration:03d}_class{index:03d}.mrc" for index in range(1, K + 1)]
        candidate_star = candidate_dir / f"run_it{iteration:03d}_data.star"
        reference_star = reference_dir / f"run_it{iteration:03d}_data.star"
        required = [*candidate_maps, *reference_maps, candidate_star, reference_star]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise AuditError(f"iteration {iteration} is missing required artifacts: {missing}")

        candidate_volumes = [_load_relion_volume(path) for path in candidate_maps]
        reference_volumes = [_load_relion_volume(path) for path in reference_maps]
        scores, curves = _pairwise_fsc_auc(candidate_volumes, reference_volumes)
        permutation = _best_class_permutation(scores)
        matched_scores = np.asarray([scores[index, match] for index, match in enumerate(permutation)])
        for (candidate_class, reference_class), curve in curves.items():
            key = f"it{iteration:03d}_candidate{candidate_class + 1:03d}_reference{reference_class + 1:03d}"
            shellwise[key] = curve
        assignments = _matched_assignments(
            candidate_star,
            reference_star,
            permutation,
            minimum_class_fraction,
        )
        collapse_free = not (
            assignments["candidate_population"]["collapse_detected"]
            or assignments["reference_population"]["collapse_detected"]
        )
        assignment_accuracy = assignments["accuracy"]
        assignment_pass = bool(
            assignments["image_identity_sets_match"]
            and assignments["assignment_status_match"]
            and (
                assignment_accuracy is None
                or assignment_accuracy >= minimum_assignment_accuracy
            )
        )
        parity_pass = bool(np.min(matched_scores) >= minimum_fsc_auc and assignment_pass)
        results.append(
            {
                "iteration": iteration,
                "permutation_candidate_to_reference": list(permutation),
                "pairwise_fsc_auc": scores.tolist(),
                "matched_fsc_auc": matched_scores.tolist(),
                "minimum_matched_fsc_auc": float(np.min(matched_scores)),
                "mean_matched_fsc_auc": float(np.mean(matched_scores)),
                "class_assignments": assignments,
                "class_assignment_pass": assignment_pass,
                "class_collapse_free": collapse_free,
                "trajectory_parity_pass": parity_pass,
                "pass": bool(parity_pass and collapse_free),
            }
        )

    report = {
        "schema": "recovar.em_real_kclass_initialmodel_trajectory.v2",
        "K": K,
        "checkpoints": list(checkpoints),
        "thresholds": {
            "minimum_per_class_fsc_auc": minimum_fsc_auc,
            "minimum_class_assignment_accuracy": minimum_assignment_accuracy,
            "minimum_class_fraction": minimum_class_fraction,
        },
        "metric_policy": "signed shellwise FSC and normalized non-DC FSC-AUC; no map correlation",
        "correlation_used": False,
        "gold_standard_halfmaps_available": False,
        "gold_standard_halfmaps_missing_reason": (
            "RELION and RECOVAR InitialModel emit one map per class, not independent half maps"
        ),
        "iterations": results,
        "minimum_matched_fsc_auc": min(row["minimum_matched_fsc_auc"] for row in results),
        "minimum_class_assignment_accuracy": (
            min(
                accuracy
                for row in results
                if (accuracy := row["class_assignments"]["accuracy"]) is not None
            )
            if any(row["class_assignments"]["accuracy"] is not None for row in results)
            else None
        ),
        "trajectory_parity_result": "pass" if all(row["trajectory_parity_pass"] for row in results) else "fail",
        "class_stability_result": "pass" if all(row["class_collapse_free"] for row in results) else "fail",
        "result": "pass" if all(row["pass"] for row in results) else "fail",
    }
    return report, shellwise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--checkpoint", type=int, action="append", required=True)
    parser.add_argument("--minimum-fsc-auc", type=float, default=0.999)
    parser.add_argument("--minimum-assignment-accuracy", type=float, default=0.995)
    parser.add_argument("--minimum-class-fraction", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shells-npz", type=Path, required=True)
    args = parser.parse_args(argv)
    report, shellwise = audit_trajectory(
        candidate_dir=args.candidate_dir,
        reference_dir=args.reference_dir,
        K=args.K,
        checkpoints=tuple(args.checkpoint),
        minimum_fsc_auc=args.minimum_fsc_auc,
        minimum_assignment_accuracy=args.minimum_assignment_accuracy,
        minimum_class_fraction=args.minimum_class_fraction,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_shells_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
