#!/usr/bin/env python3
"""Audit a fixed K=1 cohort before BPref scatter using compact membership data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_k1_bpref_contributor_membership import match_rotations
from scripts.validate_relion_bpref_membership import (
    MembershipArtifact,
    validate_directory,
)

COHORT_SCHEMA = "recovar-k1-bpref-support-cohort-v1"
EXPECTED_COHORT_SHA256 = (
    "07901c4f17e9e13d878f9341fe6293a9f2968673c77784ae176d45c017b90c18"
)
EXPECTED_GROUP_COUNTS = {
    "support_delta_le_minus_3": 2,
    "support_delta_minus_2": 16,
    "support_delta_minus_1": 24,
    "exact_support_control": 22,
}
MASS_RELATIVE_L2_TOLERANCE = 1.0e-3
ROTATION_TOLERANCE = 1.0e-6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    _require(left.shape == right.shape, "relative-L2 topology mismatch")
    if not left.size:
        return None
    denominator = float(np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0 if np.array_equal(left, right) else None
    return float(np.linalg.norm(left - right) / denominator)


def compare_particle(
    *,
    relion_rotations: np.ndarray,
    relion_weights: np.ndarray,
    relion_significant_weight: float,
    relion_weight_norm: float,
    recovar_rotations: np.ndarray,
    recovar_posterior: np.ndarray,
    recovar_reconstruction: np.ndarray,
    rotation_tolerance: float = ROTATION_TOLERANCE,
) -> dict[str, Any]:
    """Compare candidate, significance, and mass boundaries for one particle."""

    relion_rotations = np.asarray(relion_rotations, dtype=np.float32).reshape(-1, 3, 3)
    recovar_rotations = np.asarray(recovar_rotations, dtype=np.float32).reshape(-1, 3, 3)
    relion_weights = np.asarray(relion_weights, dtype=np.float64)
    recovar_posterior = np.asarray(recovar_posterior, dtype=np.float64)
    recovar_reconstruction = np.asarray(recovar_reconstruction, dtype=np.float64)
    _require(
        relion_weights.ndim == 2
        and relion_weights.shape[0] == relion_rotations.shape[0],
        "RELION weight topology mismatch",
    )
    _require(
        recovar_posterior.ndim == 2
        and recovar_posterior.shape == recovar_reconstruction.shape
        and recovar_posterior.shape[0] == recovar_rotations.shape[0],
        "RECOVAR posterior topology mismatch",
    )
    _require(
        np.isfinite(relion_significant_weight) and relion_significant_weight >= 0.0,
        "invalid RELION significant threshold",
    )
    _require(
        np.isfinite(relion_weight_norm) and relion_weight_norm > 0.0,
        "invalid RELION weight norm",
    )
    _require(
        np.all(np.isfinite(relion_weights))
        and np.all(np.isfinite(recovar_posterior))
        and np.all(np.isfinite(recovar_reconstruction)),
        "non-finite posterior mass",
    )
    _require(
        np.all(relion_weights >= 0.0)
        and np.all(recovar_posterior >= 0.0)
        and np.all(recovar_reconstruction >= 0.0),
        "negative posterior mass",
    )

    matches = match_rotations(
        relion_rotations, recovar_rotations, tolerance=rotation_tolerance
    )
    relion_positive_samples = relion_weights >= relion_significant_weight
    recovar_positive_samples = recovar_reconstruction > 0.0
    relion_positive_rotations = np.any(relion_positive_samples, axis=1)
    recovar_positive_rotations = np.any(recovar_positive_samples, axis=1)
    relion_rows = matches.pairs[:, 0]
    recovar_rows = matches.pairs[:, 1]
    matched_relion_positive = relion_positive_rotations[relion_rows]
    matched_recovar_positive = recovar_positive_rotations[recovar_rows]
    both_positive = matched_relion_positive & matched_recovar_positive

    relion_posterior_mass = np.sum(relion_weights, axis=1) / relion_weight_norm
    relion_reconstruction_mass = np.sum(
        np.where(relion_positive_samples, relion_weights, 0.0), axis=1
    ) / relion_weight_norm
    recovar_posterior_mass = np.sum(recovar_posterior, axis=1)
    recovar_reconstruction_mass = np.sum(recovar_reconstruction, axis=1)

    candidate_sets_exact = bool(
        matches.relion_unmatched.size == 0
        and matches.recovar_unmatched.size == 0
        and matches.pairs.shape[0] == relion_rotations.shape[0]
        and matches.pairs.shape[0] == recovar_rotations.shape[0]
    )
    relion_only_positive_matched = int(
        np.count_nonzero(matched_relion_positive & ~matched_recovar_positive)
    )
    recovar_only_positive_matched = int(
        np.count_nonzero(~matched_relion_positive & matched_recovar_positive)
    )
    relion_only_positive_unmatched = int(
        np.count_nonzero(relion_positive_rotations[matches.relion_unmatched])
    )
    recovar_only_positive_unmatched = int(
        np.count_nonzero(recovar_positive_rotations[matches.recovar_unmatched])
    )
    positive_sets_exact = bool(
        candidate_sets_exact
        and relion_only_positive_matched == 0
        and recovar_only_positive_matched == 0
    )
    significant_count_relion = int(np.count_nonzero(relion_positive_samples))
    significant_count_recovar = int(np.count_nonzero(recovar_positive_samples))
    posterior_l2 = _relative_l2(
        recovar_posterior_mass[recovar_rows],
        relion_posterior_mass[relion_rows],
    )
    reconstruction_l2 = _relative_l2(
        recovar_reconstruction_mass[recovar_rows[both_positive]],
        relion_reconstruction_mass[relion_rows[both_positive]],
    )
    mass_gate_passed = bool(
        reconstruction_l2 is not None
        and reconstruction_l2 <= MASS_RELATIVE_L2_TOLERANCE
    )
    return {
        "relion_candidate_count": int(relion_rotations.shape[0]),
        "recovar_candidate_count": int(recovar_rotations.shape[0]),
        "matched_candidate_count": int(matches.pairs.shape[0]),
        "relion_unmatched_candidate_count": int(matches.relion_unmatched.size),
        "recovar_unmatched_candidate_count": int(matches.recovar_unmatched.size),
        "candidate_sets_exact": candidate_sets_exact,
        "relion_positive_rotation_count": int(
            np.count_nonzero(relion_positive_rotations)
        ),
        "recovar_positive_rotation_count": int(
            np.count_nonzero(recovar_positive_rotations)
        ),
        "both_positive_matched_rotation_count": int(np.count_nonzero(both_positive)),
        "relion_only_positive_matched_rotation_count": relion_only_positive_matched,
        "recovar_only_positive_matched_rotation_count": recovar_only_positive_matched,
        "relion_only_positive_unmatched_rotation_count": (
            relion_only_positive_unmatched
        ),
        "recovar_only_positive_unmatched_rotation_count": (
            recovar_only_positive_unmatched
        ),
        "positive_rotation_sets_exact": positive_sets_exact,
        "relion_significant_sample_count": significant_count_relion,
        "recovar_significant_sample_count": significant_count_recovar,
        "significant_sample_count_exact": (
            significant_count_relion == significant_count_recovar
        ),
        "matched_candidate_posterior_mass_relative_l2": posterior_l2,
        "matched_positive_reconstruction_mass_relative_l2": reconstruction_l2,
        "reconstruction_mass_gate_tolerance": MASS_RELATIVE_L2_TOLERANCE,
        "reconstruction_mass_gate_passed": mass_gate_passed,
        "strict_particle_passed": bool(
            candidate_sets_exact
            and positive_sets_exact
            and significant_count_relion == significant_count_recovar
            and mass_gate_passed
        ),
    }


def _load_cohort(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    cohort = json.loads(path.read_text())
    _require(cohort.get("schema") == COHORT_SCHEMA, "cohort schema changed")
    _require(
        cohort.get("cohort_rows_sha256") == EXPECTED_COHORT_SHA256,
        "fixed cohort row hash changed",
    )
    _require(
        cohort.get("selected_group_counts") == EXPECTED_GROUP_COUNTS,
        "fixed cohort group counts changed",
    )
    _require(cohort.get("selected_particle_count") == 64, "fixed cohort size changed")
    rows = cohort.get("rows")
    _require(isinstance(rows, list) and len(rows) == 64, "cohort rows changed")
    by_stack = {int(row["stack_index_one_based"]): row for row in rows}
    _require(len(by_stack) == 64, "cohort stack identities are duplicated")
    return cohort, by_stack


def _load_recovar_particles(
    directory: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], list[Path]]:
    paths = sorted(directory.glob("*.npz"))
    _require(bool(paths), "no RECOVAR contribution shards")
    particles: dict[int, dict[str, np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                archive["schema"].item() == "recovar-bpref-contribution-rows-v3",
                f"RECOVAR contribution schema changed: {path}",
            )
            _require(
                not bool(archive["shadow_only_mode"].item()),
                f"shadow-only contribution is inadmissible: {path}",
            )
            stacks = np.asarray(archive["stack_indices_1based"], dtype=np.int64)
            active_particle = np.asarray(
                archive["active_particle_rows"], dtype=np.int64
            )
            active_rotation = np.asarray(
                archive["active_rotation_rows"], dtype=np.int64
            )
            active_rotations = np.asarray(
                archive["active_rotations"], dtype=np.float32
            )
            posterior = np.asarray(archive["posterior_probs"], dtype=np.float64)
            reconstruction = np.asarray(
                archive["reconstruction_probs"], dtype=np.float64
            )
            _require(posterior.shape == reconstruction.shape, "posterior shape changed")
            for particle, stack_value in enumerate(stacks):
                stack = int(stack_value)
                _require(stack not in particles, f"duplicate RECOVAR stack {stack}")
                active = np.flatnonzero(active_particle == particle)
                local_rotation = active_rotation[active]
                _require(active.size > 0, f"missing RECOVAR rotations for stack {stack}")
                _require(
                    np.unique(local_rotation).size == local_rotation.size,
                    f"duplicate RECOVAR rotation row for stack {stack}",
                )
                particles[stack] = {
                    "rotations": active_rotations[active],
                    "posterior": posterior[particle, local_rotation],
                    "reconstruction": reconstruction[particle, local_rotation],
                }
    return particles, paths


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "particle_count": count,
        "candidate_sets_exact_count": int(
            sum(bool(row["candidate_sets_exact"]) for row in rows)
        ),
        "positive_rotation_sets_exact_count": int(
            sum(bool(row["positive_rotation_sets_exact"]) for row in rows)
        ),
        "significant_sample_count_exact_count": int(
            sum(bool(row["significant_sample_count_exact"]) for row in rows)
        ),
        "reconstruction_mass_gate_passed_count": int(
            sum(bool(row["reconstruction_mass_gate_passed"]) for row in rows)
        ),
        "strict_particle_passed_count": int(
            sum(bool(row["strict_particle_passed"]) for row in rows)
        ),
    }


def analyze(
    *,
    cohort_json: Path,
    relion_directory: Path,
    recovar_directory: Path,
    mpi_rank: int,
    rotation_tolerance: float,
) -> dict[str, Any]:
    cohort, cohort_by_stack = _load_cohort(cohort_json)
    expected_stacks = np.asarray(sorted(cohort_by_stack), dtype=np.int64)
    relion_artifacts, relion_validation = validate_directory(
        relion_directory,
        expected_particles=64,
        expected_stack_indices=expected_stacks,
        expected_stack_mpi_rank=mpi_rank,
    )
    relion_by_stack: dict[int, MembershipArtifact] = {
        artifact.stack_index: artifact for artifact in relion_artifacts
    }
    recovar_by_stack, recovar_paths = _load_recovar_particles(recovar_directory)
    _require(
        set(relion_by_stack) == set(recovar_by_stack) == set(cohort_by_stack),
        "RELION/RECOVAR/cohort stack identity sets differ",
    )

    particle_rows = []
    grouped: dict[str, list[dict[str, Any]]] = {
        group: [] for group in EXPECTED_GROUP_COUNTS
    }
    for stack in sorted(cohort_by_stack):
        cohort_row = cohort_by_stack[stack]
        artifact = relion_by_stack[stack]
        recovar = recovar_by_stack[stack]
        row = compare_particle(
            relion_rotations=artifact.rotations["matrix"]
            .reshape(-1, 3, 3)
            .transpose(0, 2, 1),
            relion_weights=artifact.weights,
            relion_significant_weight=artifact.significant_weight,
            relion_weight_norm=artifact.weight_norm,
            recovar_rotations=recovar["rotations"],
            recovar_posterior=recovar["posterior"],
            recovar_reconstruction=recovar["reconstruction"],
            rotation_tolerance=rotation_tolerance,
        )
        row.update(
            {
                "stack_index_one_based": stack,
                "original_index_zero_based": int(
                    cohort_row["original_index_zero_based"]
                ),
                "support_delta": int(cohort_row["support_delta"]),
                "group": str(cohort_row["group"]),
            }
        )
        particle_rows.append(row)
        grouped[row["group"]].append(row)

    groups = {name: _group_summary(rows) for name, rows in grouped.items()}
    _require(
        {name: summary["particle_count"] for name, summary in groups.items()}
        == EXPECTED_GROUP_COUNTS,
        "analyzed group counts changed",
    )
    overall = _group_summary(particle_rows)
    if overall["candidate_sets_exact_count"] < 64:
        classification = "candidate_grid_membership_difference"
    elif overall["positive_rotation_sets_exact_count"] < 64:
        classification = "significance_membership_difference"
    elif overall["significant_sample_count_exact_count"] < 64:
        classification = "translation_significance_membership_difference"
    elif overall["reconstruction_mass_gate_passed_count"] < 64:
        classification = "posterior_reconstruction_mass_difference"
    else:
        classification = "membership_and_posterior_mass_exact_scatter_arithmetic_remains"

    return {
        "schema": "recovar.em.k1_case22_it2_bpref_membership_cohort.v1",
        "status": "complete",
        "classification": classification,
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed 64-particle cohort; exact candidate/significant membership and "
            "normalized posterior mass before scatter; map acceptance remains FSC/FSC-AUC; "
            "correlation is not computed"
        ),
        "fixed_metric": {
            "denominator": 64,
            **overall,
            "groups": groups,
        },
        "scope": {
            "iteration": int(cohort["iteration"]),
            "half": 1,
            "class_one_based": 1,
            "mpi_rank": mpi_rank,
            "rotation_tolerance": rotation_tolerance,
            "cohort_rows_sha256": EXPECTED_COHORT_SHA256,
        },
        "gates": {
            "cohort_schema_and_hash_exact": True,
            "relion_capture_validated": True,
            "stack_identity_sets_exact": True,
            "relion_capture_passive": True,
        },
        "relion_validation": relion_validation,
        "particles": particle_rows,
        "inputs": {
            "cohort_json": {
                "path": str(cohort_json.resolve()),
                "sha256": _sha256(cohort_json),
            },
            "relion_directory": str(relion_directory.resolve()),
            "recovar_contribution_shards": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in recovar_paths
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-json", required=True, type=Path)
    parser.add_argument("--relion-directory", required=True, type=Path)
    parser.add_argument("--recovar-directory", required=True, type=Path)
    parser.add_argument("--mpi-rank", type=int, default=1)
    parser.add_argument(
        "--rotation-tolerance", type=float, default=ROTATION_TOLERANCE
    )
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        cohort_json=args.cohort_json,
        relion_directory=args.relion_directory,
        recovar_directory=args.recovar_directory,
        mpi_rank=args.mpi_rank,
        rotation_tolerance=args.rotation_tolerance,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "fixed_metric": report["fixed_metric"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
