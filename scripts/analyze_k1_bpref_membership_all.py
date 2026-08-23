#!/usr/bin/env python3
"""Compare every K=1 half-1 fine posterior immediately before BPref scatter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_bpref_membership_cohort import (
    MASS_RELATIVE_L2_TOLERANCE,
)
from scripts.analyze_k1_bpref_contributor_membership import match_rotations
from scripts.validate_relion_bpref_rotation_mass import validate_directory


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.all(np.isfinite(array)), "quantile input is empty or nonfinite")
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _load_recovar(directory: Path) -> tuple[dict[int, dict[str, np.ndarray]], list[Path]]:
    paths = sorted(directory.glob("bpref_membership_*.npz"))
    _require(bool(paths), "no RECOVAR membership shards")
    particles: dict[int, dict[str, np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                str(archive["schema"].item()) == "recovar-bpref-rotation-mass-v2",
                f"RECOVAR membership schema changed: {path}",
            )
            _require(int(archive["half"]) == 1, f"unexpected RECOVAR half: {path}")
            stacks = np.asarray(archive["stack_indices_1based"], dtype=np.int64)
            counts = np.asarray(archive["actual_counts"], dtype=np.int64)
            rotations = np.asarray(archive["rotations"], dtype=np.float32)
            candidate_count = np.asarray(
                archive["candidate_translation_count"],
                dtype=np.int64,
            )
            posterior_mass = np.asarray(archive["posterior_rotation_mass"])
            reconstruction_mass = np.asarray(
                archive["reconstruction_rotation_mass"]
            )
            significant_count = np.asarray(
                archive["significant_translation_count"],
                dtype=np.int64,
            )
            _require(
                posterior_mass.shape
                == reconstruction_mass.shape
                == candidate_count.shape
                == significant_count.shape,
                f"RECOVAR rotation-mass topology changed: {path}",
            )
            _require(
                rotations.shape[:2] == posterior_mass.shape,
                f"rotation topology changed: {path}",
            )
            for particle_row, stack_value in enumerate(stacks):
                stack = int(stack_value)
                _require(stack not in particles, f"duplicate RECOVAR stack {stack}")
                count = int(counts[particle_row])
                _require(
                    0 < count <= posterior_mass.shape[1],
                    f"invalid rotation count for stack {stack}",
                )
                _require(
                    not np.any(candidate_count[particle_row, count:]),
                    f"padded candidates are active for stack {stack}",
                )
                particles[stack] = {
                    "rotations": rotations[particle_row, :count],
                    "candidate_count": candidate_count[particle_row, :count],
                    "posterior_mass": posterior_mass[particle_row, :count],
                    "reconstruction_mass": reconstruction_mass[particle_row, :count],
                    "significant_count": significant_count[particle_row, :count],
                }
    return particles, paths


def _count(rows: list[dict[str, Any]], field: str) -> int:
    return int(sum(bool(row[field]) for row in rows))


def _relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    _require(lhs.shape == rhs.shape, "relative-L2 topology mismatch")
    if not lhs.size:
        return None
    denominator = float(np.linalg.norm(rhs))
    return (
        float(np.linalg.norm(lhs - rhs) / denominator)
        if denominator > 0
        else (0.0 if np.array_equal(lhs, rhs) else None)
    )


def _compare_compact_particle(
    *,
    relion_rows: np.ndarray,
    recovar: dict[str, np.ndarray],
    rotation_tolerance: float,
) -> dict[str, Any]:
    relion_rotations = np.asarray(relion_rows["matrix"], dtype=np.float32).reshape(-1, 3, 3)
    relion_rotations = relion_rotations.transpose(0, 2, 1)
    recovar_rotations = np.asarray(recovar["rotations"], dtype=np.float32).reshape(-1, 3, 3)
    matches = match_rotations(
        relion_rotations,
        recovar_rotations,
        tolerance=rotation_tolerance,
    )
    relion_rows_matched = matches.pairs[:, 0]
    recovar_rows_matched = matches.pairs[:, 1]
    relion_posterior = np.asarray(relion_rows["posterior_rotation_mass"], dtype=np.float64)
    relion_reconstruction = np.asarray(
        relion_rows["reconstruction_rotation_mass"],
        dtype=np.float64,
    )
    relion_significant = np.asarray(
        relion_rows["significant_translation_count"],
        dtype=np.int64,
    )
    recovar_posterior = np.asarray(recovar["posterior_mass"], dtype=np.float64)
    recovar_reconstruction = np.asarray(recovar["reconstruction_mass"], dtype=np.float64)
    recovar_significant = np.asarray(recovar["significant_count"], dtype=np.int64)
    relion_positive = relion_significant > 0
    recovar_positive = recovar_significant > 0
    matched_relion_positive = relion_positive[relion_rows_matched]
    matched_recovar_positive = recovar_positive[recovar_rows_matched]
    both_positive = matched_relion_positive & matched_recovar_positive

    def union_metric(relion_mass: np.ndarray, recovar_mass: np.ndarray) -> float | None:
        return _relative_l2(
            np.concatenate(
                (
                    recovar_mass[recovar_rows_matched],
                    np.zeros(matches.relion_unmatched.size, dtype=np.float64),
                    recovar_mass[matches.recovar_unmatched],
                )
            ),
            np.concatenate(
                (
                    relion_mass[relion_rows_matched],
                    relion_mass[matches.relion_unmatched],
                    np.zeros(matches.recovar_unmatched.size, dtype=np.float64),
                )
            ),
        )

    candidate_sets_exact = bool(
        matches.relion_unmatched.size == 0 and matches.recovar_unmatched.size == 0
    )
    relion_only_positive = int(
        np.count_nonzero(matched_relion_positive & ~matched_recovar_positive)
    )
    recovar_only_positive = int(
        np.count_nonzero(~matched_relion_positive & matched_recovar_positive)
    )
    reconstruction_l2 = _relative_l2(
        recovar_reconstruction[recovar_rows_matched[both_positive]],
        relion_reconstruction[relion_rows_matched[both_positive]],
    )
    significant_relion = int(np.sum(relion_significant))
    significant_recovar = int(np.sum(recovar_significant))
    return {
        "candidate_sets_exact": candidate_sets_exact,
        "positive_rotation_sets_exact": bool(
            candidate_sets_exact
            and relion_only_positive == 0
            and recovar_only_positive == 0
        ),
        "significant_sample_count_exact": significant_relion == significant_recovar,
        "reconstruction_mass_gate_passed": bool(
            reconstruction_l2 is not None
            and reconstruction_l2 <= MASS_RELATIVE_L2_TOLERANCE
        ),
        "strict_particle_passed": bool(
            candidate_sets_exact
            and relion_only_positive == 0
            and recovar_only_positive == 0
            and significant_relion == significant_recovar
            and reconstruction_l2 is not None
            and reconstruction_l2 <= MASS_RELATIVE_L2_TOLERANCE
        ),
        "relion_candidate_count": int(relion_rows.size),
        "recovar_candidate_count": int(recovar_rotations.shape[0]),
        "matched_candidate_count": int(matches.pairs.shape[0]),
        "relion_unmatched_candidate_count": int(matches.relion_unmatched.size),
        "recovar_unmatched_candidate_count": int(matches.recovar_unmatched.size),
        "relion_positive_rotation_count": int(np.count_nonzero(relion_positive)),
        "recovar_positive_rotation_count": int(np.count_nonzero(recovar_positive)),
        "relion_only_positive_matched_rotation_count": relion_only_positive,
        "recovar_only_positive_matched_rotation_count": recovar_only_positive,
        "relion_significant_sample_count": significant_relion,
        "recovar_significant_sample_count": significant_recovar,
        "matched_candidate_posterior_mass_relative_l2": _relative_l2(
            recovar_posterior[recovar_rows_matched],
            relion_posterior[relion_rows_matched],
        ),
        "matched_positive_reconstruction_mass_relative_l2": reconstruction_l2,
        "candidate_union_posterior_mass_relative_l2": union_metric(
            relion_posterior,
            recovar_posterior,
        ),
        "candidate_union_reconstruction_mass_relative_l2": union_metric(
            relion_reconstruction,
            recovar_reconstruction,
        ),
    }


def analyze(
    *,
    relion_directory: Path,
    recovar_directory: Path,
    mpi_rank: int,
    expected_relion_particles: int,
    expected_recovar_particles: int,
    rotation_tolerance: float,
) -> dict[str, Any]:
    recovar, recovar_paths = _load_recovar(recovar_directory)
    _require(
        len(recovar) == expected_recovar_particles,
        f"RECOVAR particle count differs: {len(recovar)} != {expected_recovar_particles}",
    )
    artifacts, relion_validation = validate_directory(
        relion_directory,
        expected_particles=expected_relion_particles,
    )
    available_ranks = {artifact.mpi_rank for artifact in artifacts}
    if mpi_rank in available_ranks:
        selected_artifacts = [
            artifact for artifact in artifacts if artifact.mpi_rank == mpi_rank
        ]
        selection_mode = "mpi_rank"
    elif relion_validation["mpi_rank_tracking"] == "unavailable_srun_environment":
        selected_artifacts = [
            artifact for artifact in artifacts if artifact.stack_index in recovar
        ]
        selection_mode = "immutable_stack_identity"
    else:
        raise ValueError(f"requested MPI rank {mpi_rank} is absent")
    relion = {artifact.stack_index: artifact for artifact in selected_artifacts}
    _require(
        set(relion) == set(recovar),
        "RELION and RECOVAR half-1 stack identities differ",
    )

    rows: list[dict[str, Any]] = []
    for stack in sorted(recovar):
        artifact = relion[stack]
        candidate = recovar[stack]
        row = _compare_compact_particle(
            relion_rows=artifact.rows,
            recovar=candidate,
            rotation_tolerance=rotation_tolerance,
        )
        row["stack_index_one_based"] = stack
        rows.append(row)

    particle_count = len(rows)
    posterior_l2 = [
        float(row["matched_candidate_posterior_mass_relative_l2"])
        for row in rows
        if row["matched_candidate_posterior_mass_relative_l2"] is not None
    ]
    reconstruction_l2 = [
        float(row["candidate_union_reconstruction_mass_relative_l2"])
        for row in rows
        if row["candidate_union_reconstruction_mass_relative_l2"] is not None
    ]
    fixed_metric = {
        "denominator": particle_count,
        "candidate_sets_exact_count": _count(rows, "candidate_sets_exact"),
        "positive_rotation_sets_exact_count": _count(rows, "positive_rotation_sets_exact"),
        "significant_sample_count_exact_count": _count(rows, "significant_sample_count_exact"),
        "reconstruction_mass_gate_passed_count": _count(rows, "reconstruction_mass_gate_passed"),
        "strict_particle_passed_count": _count(rows, "strict_particle_passed"),
        "posterior_mass_relative_l2": _quantiles(posterior_l2),
        "candidate_union_reconstruction_mass_relative_l2": _quantiles(
            reconstruction_l2
        ),
    }
    if fixed_metric["candidate_sets_exact_count"] < particle_count:
        classification = "candidate_rotation_membership_differs_before_bpref"
    elif fixed_metric["positive_rotation_sets_exact_count"] < particle_count:
        classification = "positive_rotation_membership_differs_before_bpref"
    elif fixed_metric["significant_sample_count_exact_count"] < particle_count:
        classification = "translation_significance_membership_differs_before_bpref"
    elif fixed_metric["reconstruction_mass_gate_passed_count"] < particle_count:
        classification = "posterior_reconstruction_mass_differs_before_bpref"
    else:
        classification = "all_particle_membership_mass_closes_reduction_remains"

    worst = sorted(
        rows,
        key=lambda row: (
            -1.0
            if row["candidate_union_reconstruction_mass_relative_l2"] is None
            else float(row["candidate_union_reconstruction_mass_relative_l2"])
        ),
        reverse=True,
    )[:32]
    return {
        "schema": "recovar.em.k1_bpref_membership_all.v1",
        "classification": classification,
        "metric_policy": (
            "all half-1 particles; exact candidate/significant membership and normalized "
            "posterior mass before scatter; map acceptance remains signed FSC/FSC-AUC; "
            "correlation is not computed"
        ),
        "physical_iteration": 2,
        "half": 1,
        "mpi_rank": mpi_rank,
        "relion_selection_mode": selection_mode,
        "recovar_particle_count": len(recovar),
        "selected_relion_particle_count": len(relion),
        "fixed_metric": fixed_metric,
        "worst_particles": worst,
        "relion_validation": relion_validation,
        "inputs": {
            "relion_directory": str(relion_directory.resolve()),
            "recovar_shards": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in recovar_paths
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-directory", required=True, type=Path)
    parser.add_argument("--recovar-directory", required=True, type=Path)
    parser.add_argument("--mpi-rank", type=int, default=1)
    parser.add_argument("--expected-relion-particles", type=int, default=1490)
    parser.add_argument("--expected-recovar-particles", type=int, default=1490)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        relion_directory=args.relion_directory,
        recovar_directory=args.recovar_directory,
        mpi_rank=args.mpi_rank,
        expected_relion_particles=args.expected_relion_particles,
        expected_recovar_particles=args.expected_recovar_particles,
        rotation_tolerance=args.rotation_tolerance,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": report["classification"], "fixed_metric": report["fixed_metric"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
