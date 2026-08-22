#!/usr/bin/env python3
"""Audit K-class InitialModel trajectories with permutation-invariant FSC gates."""

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
    """Raised when a required K-class parity artifact is missing or invalid."""


def _pairwise_fsc_auc(
    candidate_maps: list[np.ndarray],
    reference_maps: list[np.ndarray],
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    if not candidate_maps or len(candidate_maps) != len(reference_maps):
        raise ValueError("candidate and reference map lists must have the same non-zero length")
    shape = candidate_maps[0].shape
    if any(volume.shape != shape for volume in (*candidate_maps, *reference_maps)):
        raise ValueError("all class maps must have the same shape")
    K = len(candidate_maps)
    scores = np.empty((K, K), dtype=np.float64)
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


def _matched_assignments(
    candidate_star: Path,
    reference_star: Path,
    permutation: tuple[int, ...],
) -> dict[str, Any]:
    candidate, _ = read_star(str(candidate_star))
    reference, _ = read_star(str(reference_star))
    candidate_image = _column(candidate, ("_rlnImageName", "rlnImageName"), path=candidate_star)
    reference_image = _column(reference, ("_rlnImageName", "rlnImageName"), path=reference_star)
    candidate_class = _column(candidate, ("_rlnClassNumber", "rlnClassNumber"), path=candidate_star)
    reference_class = _column(reference, ("_rlnClassNumber", "rlnClassNumber"), path=reference_star)

    candidate_by_image = {
        str(image): int(label) - 1 for image, label in zip(candidate[candidate_image], candidate[candidate_class])
    }
    reference_by_image = {
        str(image): int(label) - 1 for image, label in zip(reference[reference_image], reference[reference_class])
    }
    common = sorted(set(candidate_by_image).intersection(reference_by_image))
    if not common:
        raise AuditError("candidate and reference data STAR files share no image identities")
    candidate_labels = np.asarray([candidate_by_image[image] for image in common], dtype=np.int64)
    reference_labels = np.asarray([reference_by_image[image] for image in common], dtype=np.int64)
    K = len(permutation)
    assigned = (candidate_labels >= 0) & (candidate_labels < K) & (reference_labels >= 0) & (reference_labels < K)
    accuracy = (
        _class_assignment_accuracy(candidate_labels[assigned], reference_labels[assigned], permutation)
        if np.any(assigned)
        else None
    )
    return {
        "common_particles": len(common),
        "common_assigned_particles": int(np.count_nonzero(assigned)),
        "candidate_particles": len(candidate_by_image),
        "reference_particles": len(reference_by_image),
        "accuracy": accuracy,
    }


def audit_trajectory(
    *,
    candidate_dir: Path,
    reference_dir: Path,
    K: int,
    checkpoints: tuple[int, ...],
    minimum_fsc_auc: float,
    minimum_assignment_accuracy: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if K < 2:
        raise ValueError("K-class audit requires K >= 2")
    if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be sorted and unique")

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
            shellwise[f"it{iteration:03d}_candidate{candidate_class + 1:03d}_reference{reference_class + 1:03d}"] = (
                curve
            )
        assignments = _matched_assignments(candidate_star, reference_star, permutation)
        assignment_pass = assignments["accuracy"] is None or assignments["accuracy"] >= minimum_assignment_accuracy
        passed = bool(np.min(matched_scores) >= minimum_fsc_auc and assignment_pass)
        results.append(
            {
                "iteration": iteration,
                "permutation_candidate_to_reference": list(permutation),
                "pairwise_fsc_auc": scores.tolist(),
                "matched_fsc_auc": matched_scores.tolist(),
                "minimum_matched_fsc_auc": float(np.min(matched_scores)),
                "mean_matched_fsc_auc": float(np.mean(matched_scores)),
                "class_assignments": assignments,
                "artifact_topology_exact": True,
                "pass": passed,
            }
        )

    assignment_accuracies = [
        row["class_assignments"]["accuracy"] for row in results if row["class_assignments"]["accuracy"] is not None
    ]
    report = {
        "schema": "recovar.vdam_kclass_trajectory_audit.v1",
        "K": K,
        "checkpoints": list(checkpoints),
        "thresholds": {
            "minimum_per_class_fsc_auc": minimum_fsc_auc,
            "minimum_class_assignment_accuracy": minimum_assignment_accuracy,
        },
        "metric_policy": "signed shellwise FSC and normalized non-DC FSC-AUC; no map correlation",
        "correlation_used": False,
        "iterations": results,
        "minimum_matched_fsc_auc": min(row["minimum_matched_fsc_auc"] for row in results),
        "minimum_class_assignment_accuracy": min(assignment_accuracies) if assignment_accuracies else None,
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
    parser.add_argument("--minimum-assignment-accuracy", type=float, default=0.999)
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
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_shells_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
