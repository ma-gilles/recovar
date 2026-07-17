#!/usr/bin/env python3
"""Classify RECOVAR/RELION hidden-variable change disagreements.

This diagnostic compares one numbered-iteration boundary in original particle
identity order.  It uses RELION's mean-of-matrix-row-angles orientation-change
metric, retains the exact per-particle arrays, and reports tail concentration
so a small unstable subgroup cannot be hidden by a p95 summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import audit_em_particle_state_distribution as particle_audit


SCHEMA = "em_hidden_change_distribution_v1"
ARRAY_SCHEMA = "em_hidden_change_distribution_arrays_v1"
DEFAULT_THRESHOLDS_DEG = (0.01, 0.1, 1.0, 5.0, 20.0, 80.0)
DEFAULT_TOP_FRACTIONS = (0.001, 0.005, 0.01, 0.05, 0.1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relion_change_per_particle(previous_eulers: np.ndarray, current_eulers: np.ndarray) -> np.ndarray:
    """Return ``HealpixSampling::calculateAngularDistance`` for C1 particles."""
    previous = particle_audit._relion_euler_matrices(previous_eulers)
    current = particle_audit._relion_euler_matrices(current_eulers)
    row_cosines = np.clip(np.einsum("nij,nij->ni", previous, current), -1.0, 1.0)
    result = np.degrees(np.arccos(row_cosines)).mean(axis=1)
    exact = np.all(np.asarray(previous_eulers) == np.asarray(current_eulers), axis=1)
    result[exact] = 0.0
    return result


def _threshold_counts(values: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"gt_{threshold:g}_deg": {
            "count": int(np.count_nonzero(values > threshold)),
            "fraction": float(np.mean(values > threshold)),
        }
        for threshold in thresholds
    }


def _cohort_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    return {
        "subgroup": particle_audit._summary(values[mask]),
        "complement": particle_audit._summary(values[~mask]),
    }


def _load_relion_eulers_and_pmax(path: Path, identities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table = particle_audit._particle_table(path)
    order = particle_audit._aligned_order(
        identities,
        particle_audit._identity_array(table, source=path),
        source=path,
    )
    euler_columns = [
        particle_audit._numeric_column(table, name, order, source=path, required=True)
        for name in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    ]
    pmax = particle_audit._numeric_column(
        table,
        "rlnMaxValueProbDistribution",
        order,
        source=path,
        required=True,
    )
    return np.column_stack(euler_columns), np.asarray(pmax, dtype=np.float64)


def analyze(
    *,
    recovar_results: Path,
    recovar_particles_star: Path,
    relion_previous_star: Path,
    relion_current_star: Path,
    previous_iteration: int,
    current_iteration: int,
    subgroup_threshold_deg: float = 0.1,
    thresholds_deg: tuple[float, ...] = DEFAULT_THRESHOLDS_DEG,
    top_fractions: tuple[float, ...] = DEFAULT_TOP_FRACTIONS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if current_iteration != previous_iteration + 1:
        raise particle_audit.AuditError("iterations must describe one consecutive boundary")
    source_table = particle_audit._particle_table(recovar_particles_star)
    identities = particle_audit._identity_array(source_table, source=recovar_particles_star)
    n_images = int(identities.size)

    with np.load(recovar_results, allow_pickle=True) as recovar:
        rec_previous = particle_audit._optional_recovar_matrix(
            recovar, "best_rotation_eulers", previous_iteration - 1, n_images, 3
        )
        rec_current = particle_audit._optional_recovar_matrix(
            recovar, "best_rotation_eulers", current_iteration - 1, n_images, 3
        )
        if rec_previous is None or rec_current is None:
            raise particle_audit.AuditError("RECOVAR results are missing by-image Euler arrays")
        rec_previous_pmax = particle_audit._load_recovar_array(
            recovar, "pmax_per_image", previous_iteration - 1, n_images
        )
        rec_current_pmax = particle_audit._load_recovar_array(
            recovar, "pmax_per_image", current_iteration - 1, n_images
        )
        halves = particle_audit._half_labels(recovar, source_table, n_images)

    rel_previous, rel_previous_pmax = _load_relion_eulers_and_pmax(relion_previous_star, identities)
    rel_current, rel_current_pmax = _load_relion_eulers_and_pmax(relion_current_star, identities)

    rec_change = _relion_change_per_particle(rec_previous, rec_current)
    rel_change = _relion_change_per_particle(rel_previous, rel_current)
    signed_difference = rec_change - rel_change
    absolute_difference = np.abs(signed_difference)
    cross_previous = particle_audit._angular_error_deg(rec_previous, rel_previous)
    cross_current = particle_audit._angular_error_deg(rec_current, rel_current)
    subgroup = absolute_difference > float(subgroup_threshold_deg)
    cross_pose_union = (cross_previous > float(subgroup_threshold_deg)) | (
        cross_current > float(subgroup_threshold_deg)
    )

    descending = np.argsort(absolute_difference)[::-1]
    total_absolute = float(np.sum(absolute_difference))
    concentration = []
    for fraction in top_fractions:
        count = max(1, int(np.ceil(n_images * fraction)))
        selected = descending[:count]
        concentration.append(
            {
                "top_fraction": float(fraction),
                "count": count,
                "absolute_difference_share": (
                    float(np.sum(absolute_difference[selected]) / total_absolute)
                    if total_absolute > 0.0
                    else 0.0
                ),
                "signed_mean_contribution_deg": float(np.sum(signed_difference[selected]) / n_images),
            }
        )
    one_percent_concentration = next(
        (row for row in concentration if np.isclose(row["top_fraction"], 0.01)),
        None,
    )

    low_pmax_enrichment = []
    for percentile in (1, 5, 10, 25, 50):
        rec_previous_low = rec_previous_pmax <= np.percentile(rec_previous_pmax, percentile)
        rec_current_low = rec_current_pmax <= np.percentile(rec_current_pmax, percentile)
        low_pmax_enrichment.append(
            {
                "percentile": percentile,
                "expected_fraction": percentile / 100.0,
                "subgroup_fraction_previous": float(np.mean(rec_previous_low[subgroup])) if subgroup.any() else None,
                "subgroup_fraction_current": float(np.mean(rec_current_low[subgroup])) if subgroup.any() else None,
            }
        )

    report = {
        "schema": SCHEMA,
        "quality_metric_policy": (
            "Exact/array distribution metrics for intermediate state; no correlation is computed. "
            "FSC/FSC-AUC remain the map-quality metrics."
        ),
        "inputs": {
            "recovar_results": str(recovar_results.resolve()),
            "recovar_particles_star": str(recovar_particles_star.resolve()),
            "relion_previous_star": str(relion_previous_star.resolve()),
            "relion_current_star": str(relion_current_star.resolve()),
            "previous_iteration": int(previous_iteration),
            "current_iteration": int(current_iteration),
            "n_images": n_images,
        },
        "metric": "RELION C1 mean angle between corresponding Euler-matrix rows",
        "recovar_change_deg": particle_audit._summary(rec_change),
        "relion_change_deg": particle_audit._summary(rel_change),
        "signed_change_difference_deg": particle_audit._summary(signed_difference),
        "absolute_change_difference_deg": {
            **particle_audit._summary(absolute_difference),
            "thresholds": _threshold_counts(absolute_difference, thresholds_deg),
        },
        "cross_engine_pose_error_deg": {
            "previous": particle_audit._summary(cross_previous),
            "current": particle_audit._summary(cross_current),
        },
        "tail_concentration": concentration,
        "subgroup": {
            "definition": f"absolute change difference > {subgroup_threshold_deg:g} deg",
            "threshold_deg": float(subgroup_threshold_deg),
            "count": int(np.count_nonzero(subgroup)),
            "fraction": float(np.mean(subgroup)),
            "half1_fraction": float(np.mean(halves[subgroup] == 1)) if subgroup.any() else None,
            "cross_pose_union_contingency": {
                "subgroup_and_cross_pose": int(np.count_nonzero(subgroup & cross_pose_union)),
                "subgroup_only": int(np.count_nonzero(subgroup & ~cross_pose_union)),
                "cross_pose_only": int(np.count_nonzero(~subgroup & cross_pose_union)),
                "neither": int(np.count_nonzero(~subgroup & ~cross_pose_union)),
            },
            "recovar_change_deg": _cohort_summary(rec_change, subgroup),
            "relion_change_deg": _cohort_summary(rel_change, subgroup),
            "recovar_pmax_previous": _cohort_summary(rec_previous_pmax, subgroup),
            "recovar_pmax_current": _cohort_summary(rec_current_pmax, subgroup),
            "relion_pmax_previous": _cohort_summary(rel_previous_pmax, subgroup),
            "relion_pmax_current": _cohort_summary(rel_current_pmax, subgroup),
            "pmax_absolute_difference_previous": _cohort_summary(
                np.abs(rec_previous_pmax - rel_previous_pmax), subgroup
            ),
            "pmax_absolute_difference_current": _cohort_summary(
                np.abs(rec_current_pmax - rel_current_pmax), subgroup
            ),
            "low_recovar_pmax_enrichment": low_pmax_enrichment,
        },
        "classification": {
            "localized_to_cross_engine_pose_mismatch": bool(np.all(~subgroup | cross_pose_union)),
            "subgroup_tail_dominates_absolute_difference": bool(
                one_percent_concentration is not None
                and one_percent_concentration["absolute_difference_share"] > 0.5
            ),
            "p95_can_hide_subgroup": bool(
                np.mean(subgroup) > 0.0
                and np.mean(subgroup) < 0.06
                and particle_audit._summary(cross_previous)["p95"] < subgroup_threshold_deg
                and particle_audit._summary(cross_current)["p95"] < subgroup_threshold_deg
            ),
        },
    }
    arrays = {
        "schema": np.asarray(ARRAY_SCHEMA),
        "image_identities": identities,
        "half": halves,
        "recovar_change_deg": rec_change,
        "relion_change_deg": rel_change,
        "signed_change_difference_deg": signed_difference,
        "absolute_change_difference_deg": absolute_difference,
        "cross_engine_pose_error_previous_deg": cross_previous,
        "cross_engine_pose_error_current_deg": cross_current,
        "subgroup_mask": subgroup,
    }
    return report, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-results", type=Path, required=True)
    parser.add_argument("--recovar-particles-star", type=Path, required=True)
    parser.add_argument("--relion-previous-star", type=Path, required=True)
    parser.add_argument("--relion-current-star", type=Path, required=True)
    parser.add_argument("--previous-iteration", type=int, required=True)
    parser.add_argument("--current-iteration", type=int, required=True)
    parser.add_argument("--subgroup-threshold-deg", type=float, default=0.1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-arrays", type=Path, required=True)
    args = parser.parse_args()

    report, arrays = analyze(
        recovar_results=args.recovar_results,
        recovar_particles_star=args.recovar_particles_star,
        relion_previous_star=args.relion_previous_star,
        relion_current_star=args.relion_current_star,
        previous_iteration=args.previous_iteration,
        current_iteration=args.current_iteration,
        subgroup_threshold_deg=args.subgroup_threshold_deg,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_arrays, **arrays)
    report["array_artifact"] = {
        "path": str(args.output_arrays.resolve()),
        "schema": ARRAY_SCHEMA,
        "sha256": _sha256(args.output_arrays),
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["classification"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
