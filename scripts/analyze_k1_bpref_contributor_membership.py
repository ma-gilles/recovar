#!/usr/bin/env python3
"""Separate K=1 pre-scatter candidate-grid and significance differences."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROTATION_TOLERANCE = 1.0e-6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _float32_from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(value) & 0xFFFFFFFF))[0]


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: None for key in ("min", "p05", "p50", "p95", "p99", "max")}
    return {
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def classify_threshold_substitution(
    *,
    significance_gap: bool,
    relion_positive_recovar_nonpositive: int,
    recovar_positive_relion_nonpositive: int,
) -> str:
    """Classify whether RELION's scalar threshold closes common support."""

    if not significance_gap:
        return "not_applicable_no_common_candidate_significance_gap"
    if relion_positive_recovar_nonpositive or recovar_positive_relion_nonpositive:
        return "common_candidate_significance_gap_persists_under_relion_threshold"
    return "relion_threshold_closes_common_candidate_significance_gap"


def same_identity_set(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    """Return whether unique integer identities agree without assuming order."""

    left = np.asarray(lhs, dtype=np.int64).reshape(-1)
    right = np.asarray(rhs, dtype=np.int64).reshape(-1)
    return bool(
        np.unique(left).size == left.size
        and np.unique(right).size == right.size
        and np.array_equal(np.sort(left), np.sort(right))
    )


@dataclass(frozen=True)
class RotationMatches:
    pairs: np.ndarray
    relion_unmatched: np.ndarray
    recovar_unmatched: np.ndarray
    relion_ambiguous: int
    recovar_ambiguous: int
    matched_max_abs: np.ndarray
    relion_nearest_max_abs: np.ndarray
    recovar_nearest_max_abs: np.ndarray


def match_rotations(
    relion_rotations: np.ndarray,
    recovar_rotations: np.ndarray,
    *,
    tolerance: float = ROTATION_TOLERANCE,
) -> RotationMatches:
    """Uniquely match rotation tables without a quadratic distance matrix."""

    relion = np.asarray(relion_rotations, dtype=np.float32).reshape(-1, 3, 3)
    recovar = np.asarray(recovar_rotations, dtype=np.float32).reshape(-1, 3, 3)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("rotation tolerance must be finite and nonnegative")
    if not relion.size or not recovar.size:
        return RotationMatches(
            pairs=np.empty((0, 2), dtype=np.int64),
            relion_unmatched=np.arange(relion.shape[0], dtype=np.int64),
            recovar_unmatched=np.arange(recovar.shape[0], dtype=np.int64),
            relion_ambiguous=0,
            recovar_ambiguous=0,
            matched_max_abs=np.empty(0, dtype=np.float64),
            relion_nearest_max_abs=np.full(relion.shape[0], np.inf),
            recovar_nearest_max_abs=np.full(recovar.shape[0], np.inf),
        )
    relion_flat = relion.astype(np.float64).reshape(relion.shape[0], -1)
    recovar_flat = recovar.astype(np.float64).reshape(recovar.shape[0], -1)
    relion_tree = cKDTree(relion_flat)
    recovar_tree = cKDTree(recovar_flat)
    inclusive_upper_bound = np.nextafter(tolerance, np.inf)

    def bounded_two_nearest(
        source: np.ndarray,
        target_tree: cKDTree,
        target_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query_k = min(2, target_count)
        bounded_distance, bounded_index = target_tree.query(
            source,
            k=query_k,
            p=np.inf,
            distance_upper_bound=inclusive_upper_bound,
            workers=-1,
        )
        bounded_distance = np.asarray(bounded_distance, dtype=np.float64).reshape(
            source.shape[0], query_k
        )
        bounded_index = np.asarray(bounded_index, dtype=np.int64).reshape(
            source.shape[0], query_k
        )
        within_count = np.sum(np.isfinite(bounded_distance), axis=1)
        nearest_distance = np.asarray(
            target_tree.query(source, k=1, p=np.inf, workers=-1)[0],
            dtype=np.float64,
        )
        return (
            bounded_distance[:, 0],
            bounded_index[:, 0],
            within_count,
            nearest_distance,
        )

    (
        relion_bounded_distance,
        relion_bounded_index,
        relion_degree,
        relion_nearest,
    ) = bounded_two_nearest(relion_flat, recovar_tree, recovar.shape[0])
    (
        _,
        recovar_bounded_index,
        recovar_degree,
        recovar_nearest,
    ) = bounded_two_nearest(recovar_flat, relion_tree, relion.shape[0])
    relion_unique = np.flatnonzero(relion_degree == 1)
    candidate_recovar = relion_bounded_index[relion_unique]
    reciprocal_unique = (
        (recovar_degree[candidate_recovar] == 1)
        & (recovar_bounded_index[candidate_recovar] == relion_unique)
    )
    matched_relion = relion_unique[reciprocal_unique]
    matched_recovar = candidate_recovar[reciprocal_unique]
    pair_array = np.stack((matched_relion, matched_recovar), axis=1).astype(
        np.int64,
        copy=False,
    )
    relion_matched_mask = np.zeros(relion.shape[0], dtype=bool)
    recovar_matched_mask = np.zeros(recovar.shape[0], dtype=bool)
    relion_matched_mask[matched_relion] = True
    recovar_matched_mask[matched_recovar] = True
    return RotationMatches(
        pairs=pair_array,
        relion_unmatched=np.flatnonzero(~relion_matched_mask),
        recovar_unmatched=np.flatnonzero(~recovar_matched_mask),
        relion_ambiguous=int(np.count_nonzero(relion_degree > 1)),
        recovar_ambiguous=int(np.count_nonzero(recovar_degree > 1)),
        matched_max_abs=relion_bounded_distance[matched_relion],
        relion_nearest_max_abs=relion_nearest,
        recovar_nearest_max_abs=recovar_nearest,
    )


def compare_particle_membership(
    *,
    relion_rotations: np.ndarray,
    relion_positive: np.ndarray,
    recovar_rotations: np.ndarray,
    recovar_positive: np.ndarray,
    recovar_posterior_mass: np.ndarray,
    recovar_reconstruction_mass: np.ndarray,
    recovar_max_sample_posterior: np.ndarray,
    recovar_reconstruction_threshold: float,
    relion_normalized_reconstruction_threshold: float,
    tolerance: float = ROTATION_TOLERANCE,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compare candidate and positive-contributor membership for one particle."""

    relion_positive = np.asarray(relion_positive, dtype=bool)
    recovar_positive = np.asarray(recovar_positive, dtype=bool)
    recovar_posterior_mass = np.asarray(recovar_posterior_mass, dtype=np.float64)
    recovar_reconstruction_mass = np.asarray(
        recovar_reconstruction_mass, dtype=np.float64
    )
    recovar_max_sample_posterior = np.asarray(
        recovar_max_sample_posterior, dtype=np.float64
    )
    relion_count = np.asarray(relion_rotations).shape[0]
    recovar_count = np.asarray(recovar_rotations).shape[0]
    _require(relion_positive.shape == (relion_count,), "RELION positive mask shape changed")
    for name, array in (
        ("RECOVAR positive mask", recovar_positive),
        ("RECOVAR posterior mass", recovar_posterior_mass),
        ("RECOVAR reconstruction mass", recovar_reconstruction_mass),
        ("RECOVAR max sample posterior", recovar_max_sample_posterior),
    ):
        _require(array.shape == (recovar_count,), f"{name} shape changed")
    _require(
        np.array_equal(recovar_positive, recovar_reconstruction_mass > 0.0),
        "RECOVAR positive rows do not equal positive reconstruction mass",
    )
    _require(np.all(recovar_posterior_mass >= 0.0), "negative RECOVAR posterior mass")
    _require(
        np.all(recovar_reconstruction_mass >= 0.0),
        "negative RECOVAR reconstruction mass",
    )

    matches = match_rotations(
        relion_rotations, recovar_rotations, tolerance=tolerance
    )
    relion_matched = matches.pairs[:, 0]
    recovar_matched = matches.pairs[:, 1]
    relion_matched_positive = relion_positive[relion_matched]
    recovar_matched_positive = recovar_positive[recovar_matched]
    both_positive = relion_matched_positive & recovar_matched_positive
    relion_positive_recovar_nonpositive = (
        relion_matched_positive & ~recovar_matched_positive
    )
    recovar_positive_relion_nonpositive = (
        ~relion_matched_positive & recovar_matched_positive
    )
    relion_positive_unmatched = matches.relion_unmatched[
        relion_positive[matches.relion_unmatched]
    ]
    recovar_positive_unmatched = matches.recovar_unmatched[
        recovar_positive[matches.recovar_unmatched]
    ]
    recovar_rows_relion_positive_only = recovar_matched[
        relion_positive_recovar_nonpositive
    ]
    recovar_rows_recovar_positive_only = recovar_matched[
        recovar_positive_relion_nonpositive
    ]

    threshold = float(recovar_reconstruction_threshold)
    _require(np.isfinite(threshold) and threshold >= 0.0, "invalid threshold")
    relion_threshold = float(relion_normalized_reconstruction_threshold)
    _require(
        np.isfinite(relion_threshold) and relion_threshold >= 0.0,
        "invalid normalized RELION threshold",
    )
    recovar_positive_at_relion_threshold = (
        recovar_max_sample_posterior > 0.0
    ) & (recovar_max_sample_posterior >= relion_threshold)
    recovar_matched_positive_at_relion_threshold = (
        recovar_positive_at_relion_threshold[recovar_matched]
    )
    relion_positive_recovar_at_relion_threshold_nonpositive = (
        relion_matched_positive & ~recovar_matched_positive_at_relion_threshold
    )
    recovar_at_relion_threshold_positive_relion_nonpositive = (
        ~relion_matched_positive & recovar_matched_positive_at_relion_threshold
    )

    def threshold_ratio(rows: np.ndarray) -> np.ndarray:
        if threshold == 0.0:
            return np.full(rows.size, np.nan, dtype=np.float64)
        return recovar_max_sample_posterior[rows] / threshold

    report = {
        "recovar_reconstruction_threshold": threshold,
        "recovar_reconstruction_threshold_positive": threshold > 0.0,
        "relion_normalized_reconstruction_threshold": relion_threshold,
        "relion_normalized_reconstruction_threshold_positive": (
            relion_threshold > 0.0
        ),
        "relion_candidate_count": relion_count,
        "recovar_candidate_count": recovar_count,
        "candidate_unique_match_count": int(matches.pairs.shape[0]),
        "relion_candidate_unmatched_count": int(matches.relion_unmatched.size),
        "recovar_candidate_unmatched_count": int(matches.recovar_unmatched.size),
        "relion_ambiguous_candidate_count": matches.relion_ambiguous,
        "recovar_ambiguous_candidate_count": matches.recovar_ambiguous,
        "relion_positive_contributor_count": int(np.count_nonzero(relion_positive)),
        "recovar_positive_contributor_count": int(np.count_nonzero(recovar_positive)),
        "both_positive_matched_count": int(np.count_nonzero(both_positive)),
        "relion_positive_recovar_nonpositive_matched_count": int(
            np.count_nonzero(relion_positive_recovar_nonpositive)
        ),
        "recovar_positive_relion_nonpositive_matched_count": int(
            np.count_nonzero(recovar_positive_relion_nonpositive)
        ),
        "relion_positive_recovar_at_relion_threshold_nonpositive_matched_count": int(
            np.count_nonzero(
                relion_positive_recovar_at_relion_threshold_nonpositive
            )
        ),
        "recovar_at_relion_threshold_positive_relion_nonpositive_matched_count": int(
            np.count_nonzero(
                recovar_at_relion_threshold_positive_relion_nonpositive
            )
        ),
        "relion_positive_unmatched_candidate_count": int(
            relion_positive_unmatched.size
        ),
        "recovar_positive_unmatched_candidate_count": int(
            recovar_positive_unmatched.size
        ),
        "candidate_sets_exact_at_tolerance": bool(
            matches.pairs.shape[0] == relion_count == recovar_count
            and matches.relion_ambiguous == matches.recovar_ambiguous == 0
        ),
        "positive_contributor_sets_exact_at_tolerance": bool(
            np.count_nonzero(both_positive)
            == np.count_nonzero(relion_positive)
            == np.count_nonzero(recovar_positive)
            and matches.relion_ambiguous == matches.recovar_ambiguous == 0
        ),
        "matched_positive_sets_exact_using_relion_threshold_on_recovar_posterior": bool(
            not np.any(
                relion_positive_recovar_at_relion_threshold_nonpositive
            )
            and not np.any(
                recovar_at_relion_threshold_positive_relion_nonpositive
            )
            and matches.relion_ambiguous == matches.recovar_ambiguous == 0
        ),
    }
    arrays = {
        "matched_rotation_max_abs": matches.matched_max_abs,
        "relion_candidate_nearest_recovar_max_abs": matches.relion_nearest_max_abs,
        "recovar_candidate_nearest_relion_max_abs": matches.recovar_nearest_max_abs,
        "recovar_reconstruction_threshold": np.asarray([threshold], dtype=np.float64),
        "relion_normalized_reconstruction_threshold": np.asarray(
            [relion_threshold], dtype=np.float64
        ),
        "recovar_over_relion_reconstruction_threshold": np.asarray(
            [
                threshold / relion_threshold
                if relion_threshold > 0.0
                else np.nan
            ],
            dtype=np.float64,
        ),
        "recovar_preprune_mass_relion_positive_recovar_nonpositive": (
            recovar_posterior_mass[recovar_rows_relion_positive_only]
        ),
        "recovar_max_sample_relion_positive_recovar_nonpositive": (
            recovar_max_sample_posterior[recovar_rows_relion_positive_only]
        ),
        "recovar_max_over_threshold_relion_positive_recovar_nonpositive": (
            threshold_ratio(recovar_rows_relion_positive_only)
        ),
        "recovar_preprune_mass_recovar_positive_relion_nonpositive": (
            recovar_posterior_mass[recovar_rows_recovar_positive_only]
        ),
        "recovar_reconstruction_mass_recovar_positive_relion_nonpositive": (
            recovar_reconstruction_mass[recovar_rows_recovar_positive_only]
        ),
        "recovar_max_over_threshold_recovar_positive_relion_nonpositive": (
            threshold_ratio(recovar_rows_recovar_positive_only)
        ),
        "recovar_preprune_mass_recovar_positive_unmatched": (
            recovar_posterior_mass[recovar_positive_unmatched]
        ),
        "recovar_reconstruction_mass_recovar_positive_unmatched": (
            recovar_reconstruction_mass[recovar_positive_unmatched]
        ),
    }
    return report, arrays


def _load_original_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _require(payload.get("status") == "complete", "original operand audit incomplete")
    _require(
        payload.get("classification") == "pre_scatter_contributor_membership_difference",
        "original audit did not select contributor membership",
    )
    gates = payload.get("gates", {})
    _require(gates.get("capture_inertness") is True, "capture inertness was not accepted")
    _require(
        gates.get("fresh_capture_validation") is True,
        "fresh RELION capture validation was not accepted",
    )
    return payload


def _summed_counts(rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, int]:
    return {name: int(sum(int(row[name]) for row in rows)) for name in names}


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    original = _load_original_gate(args.original_audit_json)
    sys.path.insert(0, str(args.scripts_directory))
    from validate_relion_bpref_prescatter import FILE_NAME, load_artifact

    artifact_paths: dict[int, Path] = {}
    for path in sorted(args.capture_directory.glob("*.bpre-v1.bin")):
        match = FILE_NAME.fullmatch(path.name)
        _require(match is not None, f"unexpected artifact name: {path}")
        stack = int(match["stack"])
        _require(stack not in artifact_paths, f"duplicate RELION stack {stack}")
        artifact_paths[stack] = path

    particle_rows: list[dict[str, Any]] = []
    aggregate_arrays: dict[str, list[np.ndarray]] = {}
    recovar_stacks: list[int] = []
    processed_relion_stacks: list[int] = []
    per_particle_arrays: dict[str, list[int]] = {}
    positive_ctf_reconstruction_mismatch = 0
    contribution_paths = sorted(args.contribution_directory.glob("*.npz"))
    _require(bool(contribution_paths), "no RECOVAR contribution shards")

    for contribution_path in contribution_paths:
        with np.load(contribution_path, allow_pickle=False) as archive:
            _require(
                not bool(np.asarray(archive["shadow_only_mode"]).item()),
                f"shadow-only contribution is inadmissible: {contribution_path}",
            )
            stacks = np.asarray(archive["stack_indices_1based"], dtype=np.int64)
            active_particle = np.asarray(
                archive["active_particle_rows"], dtype=np.int64
            )
            active_rotation = np.asarray(
                archive["active_rotation_rows"], dtype=np.int64
            )
            rotations = np.asarray(archive["active_rotations"], dtype=np.float32)
            ctf_probs = np.asarray(archive["active_ctf_probs"])
            posterior = np.asarray(archive["posterior_probs"], dtype=np.float64)
            reconstruction = np.asarray(
                archive["reconstruction_probs"], dtype=np.float64
            )
            threshold = np.asarray(
                archive["reconstruction_threshold"], dtype=np.float64
            )
            _require(
                posterior.shape == reconstruction.shape,
                f"posterior topology changed: {contribution_path}",
            )
            _require(
                threshold.shape == (stacks.size,),
                f"threshold topology changed: {contribution_path}",
            )
            for particle, stack_value in enumerate(stacks):
                stack = int(stack_value)
                recovar_stacks.append(stack)
                active = np.flatnonzero(active_particle == particle)
                _require(active.size > 0, f"missing RECOVAR candidate rows: {stack}")
                local_rotation = active_rotation[active]
                _require(
                    np.unique(local_rotation).size == local_rotation.size,
                    f"duplicate RECOVAR local rotation rows: {stack}",
                )
                _require(
                    np.all((0 <= local_rotation) & (local_rotation < posterior.shape[1])),
                    f"RECOVAR local rotation row out of range: {stack}",
                )
                recovar_posterior_mass = np.sum(
                    posterior[particle, local_rotation], axis=1
                )
                recovar_reconstruction_mass = np.sum(
                    reconstruction[particle, local_rotation], axis=1
                )
                recovar_max_sample = np.max(
                    posterior[particle, local_rotation], axis=1
                )
                recovar_positive = recovar_reconstruction_mass > 0.0
                ctf_positive = np.any(ctf_probs[active] > 0.0, axis=1)
                positive_ctf_reconstruction_mismatch += int(
                    np.count_nonzero(recovar_positive != ctf_positive)
                )

                artifact_path = artifact_paths.get(stack)
                _require(artifact_path is not None, f"missing RELION stack {stack}")
                artifact = load_artifact(artifact_path)
                _require(artifact.mpi_rank == args.mpi_rank, f"wrong MPI rank: {stack}")
                processed_relion_stacks.append(int(artifact.stack_index))
                relion_significant_weight = _float32_from_bits(
                    artifact.header[21]
                )
                relion_weight_norm = _float32_from_bits(artifact.header[22])
                _require(
                    np.isfinite(relion_significant_weight)
                    and relion_significant_weight >= 0.0,
                    f"invalid RELION significant weight: {stack}",
                )
                _require(
                    np.isfinite(relion_weight_norm) and relion_weight_norm > 0.0,
                    f"invalid RELION weight norm: {stack}",
                )
                relion_positive_rows = np.unique(
                    artifact.rows["orientation_local"]
                ).astype(np.int64)
                relion_positive = np.zeros(artifact.rotations.size, dtype=bool)
                relion_positive[relion_positive_rows] = True
                relion_rotations = (
                    artifact.rotations["matrix"]
                    .reshape(-1, 3, 3)
                    .transpose(0, 2, 1)
                )
                row, arrays = compare_particle_membership(
                    relion_rotations=relion_rotations,
                    relion_positive=relion_positive,
                    recovar_rotations=rotations[active],
                    recovar_positive=recovar_positive,
                    recovar_posterior_mass=recovar_posterior_mass,
                    recovar_reconstruction_mass=recovar_reconstruction_mass,
                    recovar_max_sample_posterior=recovar_max_sample,
                    recovar_reconstruction_threshold=float(threshold[particle]),
                    relion_normalized_reconstruction_threshold=(
                        relion_significant_weight / relion_weight_norm
                    ),
                    tolerance=args.rotation_tolerance,
                )
                row["stack_index_1based"] = stack
                particle_rows.append(row)
                for name, values in arrays.items():
                    aggregate_arrays.setdefault(name, []).append(values)
                for name, value in row.items():
                    if name.endswith("_count"):
                        per_particle_arrays.setdefault(name, []).append(int(value))

    _require(
        positive_ctf_reconstruction_mismatch == 0,
        "positive CTF rows do not equal positive reconstruction mass",
    )
    stack_identities_exact = same_identity_set(
        np.asarray(recovar_stacks), np.asarray(processed_relion_stacks)
    )
    _require(
        len(processed_relion_stacks) == int(original["scope"]["particle_count"]),
        "processed RELION rank count does not replay original audit",
    )
    _require(stack_identities_exact, "RELION/RECOVAR stack identity sets differ")

    count_names = (
        "relion_candidate_count",
        "recovar_candidate_count",
        "candidate_unique_match_count",
        "relion_candidate_unmatched_count",
        "recovar_candidate_unmatched_count",
        "relion_ambiguous_candidate_count",
        "recovar_ambiguous_candidate_count",
        "relion_positive_contributor_count",
        "recovar_positive_contributor_count",
        "both_positive_matched_count",
        "relion_positive_recovar_nonpositive_matched_count",
        "recovar_positive_relion_nonpositive_matched_count",
        "relion_positive_recovar_at_relion_threshold_nonpositive_matched_count",
        "recovar_at_relion_threshold_positive_relion_nonpositive_matched_count",
        "relion_positive_unmatched_candidate_count",
        "recovar_positive_unmatched_candidate_count",
    )
    totals = _summed_counts(particle_rows, count_names)
    positive_threshold_particle_count = int(
        sum(
            bool(row["recovar_reconstruction_threshold_positive"])
            for row in particle_rows
        )
    )
    candidate_gap = bool(
        totals["relion_candidate_unmatched_count"]
        or totals["recovar_candidate_unmatched_count"]
    )
    significance_gap = bool(
        totals["relion_positive_recovar_nonpositive_matched_count"]
        or totals["recovar_positive_relion_nonpositive_matched_count"]
    )
    threshold_substitution_classification = classify_threshold_substitution(
        significance_gap=significance_gap,
        relion_positive_recovar_nonpositive=totals[
            "relion_positive_recovar_at_relion_threshold_nonpositive_matched_count"
        ],
        recovar_positive_relion_nonpositive=totals[
            "recovar_at_relion_threshold_positive_relion_nonpositive_matched_count"
        ],
    )
    if candidate_gap and significance_gap:
        classification = "candidate_grid_and_significance_membership_differences"
    elif candidate_gap:
        classification = "candidate_grid_membership_difference"
    elif significance_gap:
        classification = "significance_membership_difference_on_common_candidate_grids"
    else:
        classification = "candidate_and_positive_contributor_membership_exact"

    joined_arrays = {
        name: np.concatenate(values) if values else np.empty(0, dtype=np.float64)
        for name, values in aggregate_arrays.items()
    }
    _require(
        totals["relion_positive_contributor_count"]
        == int(
            original["identity_and_source_support"][
                "relion_emitted_contributor_count"
            ]
        ),
        "RELION positive count does not replay original audit",
    )
    _require(
        totals["recovar_positive_contributor_count"]
        == int(
            original["identity_and_source_support"][
                "recovar_positive_contributor_count"
            ]
        ),
        "RECOVAR positive count does not replay original audit",
    )
    _require(
        totals["both_positive_matched_count"]
        == int(original["identity_and_source_support"]["unique_rotation_match_count"]),
        "common positive count does not replay original audit",
    )

    diagnostic_arrays = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in per_particle_arrays.items()
    }
    diagnostic_arrays["stack_indices_1based"] = np.asarray(
        [row["stack_index_1based"] for row in particle_rows], dtype=np.int64
    )
    for name, values in joined_arrays.items():
        diagnostic_arrays[name] = values
    np.savez(args.output_arrays, **diagnostic_arrays)

    interesting = [
        row
        for row in particle_rows
        if not row["candidate_sets_exact_at_tolerance"]
        or not row["positive_contributor_sets_exact_at_tolerance"]
    ]
    report = {
        "schema": "k1_bpref_contributor_membership_diagnostic_v2",
        "status": "complete",
        "classification": classification,
        "threshold_substitution_classification": (
            threshold_substitution_classification
        ),
        "metric_policy": (
            "rotation matrices and posterior/support membership only; "
            "FSC/FSC-AUC is inherited solely from the sealed capture-inertness gate; "
            "correlation is not computed"
        ),
        "scorecard_change_admissible": False,
        "gates": {
            "original_membership_audit_complete": True,
            "original_capture_inertness": True,
            "original_fresh_capture_validation": True,
            "stack_identity_sets_exact": stack_identities_exact,
            "original_positive_counts_replayed": True,
            "recovar_positive_ctf_equals_positive_reconstruction_mass": True,
        },
        "scope": {
            "iteration": int(original["scope"]["iteration"]),
            "half": int(original["scope"]["half"]),
            "class_one_based": 1,
            "mpi_rank": args.mpi_rank,
            "particle_count": len(particle_rows),
            "rotation_tolerance": args.rotation_tolerance,
            "recovar_shard_count": len(contribution_paths),
            "positive_reconstruction_threshold_particle_count": (
                positive_threshold_particle_count
            ),
            "zero_reconstruction_threshold_particle_count": (
                len(particle_rows) - positive_threshold_particle_count
            ),
        },
        "counts": {
            **totals,
            "candidate_sets_exact_particle_count": int(
                sum(bool(row["candidate_sets_exact_at_tolerance"]) for row in particle_rows)
            ),
            "positive_contributor_sets_exact_particle_count": int(
                sum(
                    bool(row["positive_contributor_sets_exact_at_tolerance"])
                    for row in particle_rows
                )
            ),
        },
        "distributions": {
            "recovar_reconstruction_threshold": _quantiles(
                joined_arrays["recovar_reconstruction_threshold"]
            ),
            "relion_normalized_reconstruction_threshold": _quantiles(
                joined_arrays["relion_normalized_reconstruction_threshold"]
            ),
            "recovar_over_relion_reconstruction_threshold": _quantiles(
                joined_arrays["recovar_over_relion_reconstruction_threshold"]
            ),
            "matched_candidate_rotation_max_abs": _quantiles(
                joined_arrays["matched_rotation_max_abs"]
            ),
            "relion_candidate_nearest_recovar_max_abs": _quantiles(
                joined_arrays["relion_candidate_nearest_recovar_max_abs"]
            ),
            "recovar_candidate_nearest_relion_max_abs": _quantiles(
                joined_arrays["recovar_candidate_nearest_relion_max_abs"]
            ),
            "recovar_preprune_mass_relion_positive_recovar_nonpositive": _quantiles(
                joined_arrays[
                    "recovar_preprune_mass_relion_positive_recovar_nonpositive"
                ]
            ),
            "recovar_max_over_threshold_relion_positive_recovar_nonpositive": _quantiles(
                joined_arrays[
                    "recovar_max_over_threshold_relion_positive_recovar_nonpositive"
                ]
            ),
            "recovar_preprune_mass_recovar_positive_relion_nonpositive": _quantiles(
                joined_arrays[
                    "recovar_preprune_mass_recovar_positive_relion_nonpositive"
                ]
            ),
            "recovar_reconstruction_mass_recovar_positive_relion_nonpositive": _quantiles(
                joined_arrays[
                    "recovar_reconstruction_mass_recovar_positive_relion_nonpositive"
                ]
            ),
            "recovar_preprune_mass_recovar_positive_unmatched": _quantiles(
                joined_arrays["recovar_preprune_mass_recovar_positive_unmatched"]
            ),
            "recovar_reconstruction_mass_recovar_positive_unmatched": _quantiles(
                joined_arrays[
                    "recovar_reconstruction_mass_recovar_positive_unmatched"
                ]
            ),
        },
        "examples_first40": interesting[:40],
        "qualification": (
            "The original order-sensitive stack-identity report is superseded by "
            "sorted unique-set equality. The original RELION oversampled child versus "
            "RECOVAR global-index modulo comparison is rejected: RECOVAR indices address "
            "the global fine grid and are not parent*8+child identities. Candidate and "
            "contributor membership are therefore aligned only by captured rotation "
            "matrices. A matched candidate that is positive in only one engine localizes "
            "a significance-membership difference; a positive rotation with no candidate "
            "matrix match localizes an earlier candidate-grid difference. Positive "
            "membership is read from the captured reconstruction mass; max-over-threshold "
            "ratios are undefined and reported as null when the saved per-particle "
            "threshold is zero. The normalized RELION threshold is its captured "
            "float32 significant_weight divided by its captured float32 weight_norm. "
            "Applying it to RECOVAR's saved pre-pruning posterior isolates threshold "
            "selection from posterior/candidate arithmetic on common rotation matrices. "
            "A residual after substitution does not by itself distinguish score, "
            "normalization, or candidate-grid causes."
        ),
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", required=True, type=Path)
    parser.add_argument("--contribution-directory", required=True, type=Path)
    parser.add_argument("--scripts-directory", required=True, type=Path)
    parser.add_argument("--original-audit-json", required=True, type=Path)
    parser.add_argument("--mpi-rank", type=int, default=1)
    parser.add_argument(
        "--rotation-tolerance", type=float, default=ROTATION_TOLERANCE
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-arrays", required=True, type=Path)
    return parser


def main() -> None:
    report = analyze(_parser().parse_args())
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "threshold_substitution_classification": report[
                    "threshold_substitution_classification"
                ],
                "counts": report["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
