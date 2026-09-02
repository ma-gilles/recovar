#!/usr/bin/env python3
"""Aggregate matched real-data K-class half-map audits across frozen seeds.

This report is diagnostic.  It retains each audit's prospective science-gate
result and compares same-seed cross-engine assignments with each engine's own
cross-seed assignment stability on the identical particles.  It never promotes
a failed per-seed audit to an accepted benchmark result.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA = "recovar.em.real_kclass_halfmap_multiseed_stability.v1"
AUDIT_SCHEMA = "recovar.em_real_kclass_independent_halfmap_audit.v3"
DEFAULT_EXPECTED_SEEDS = (42001, 42002, 42003)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expected_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"expected seeds must be comma-separated integers: {value!r}") from error
    _require(len(seeds) == 3 and len(set(seeds)) == 3, "exactly three distinct seeds are required")
    return seeds


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "cannot summarize an empty sequence")
    finite = [float(value) for value in values]
    _require(all(np.isfinite(finite)), "summary values must be finite")
    return {
        "min": min(finite),
        "median": float(statistics.median(finite)),
        "mean": float(statistics.fmean(finite)),
        "max": max(finite),
    }


def _best_agreement(left: np.ndarray, right: np.ndarray, n_classes: int) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    _require(left.ndim == right.ndim == 1 and left.shape == right.shape, "assignment shapes differ")
    _require(left.size > 0, "assignment arrays are empty")
    _require(
        np.all((left >= 0) & (left < n_classes))
        and np.all((right >= 0) & (right < n_classes)),
        "assignment labels are outside the declared class range",
    )
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (left, right), 1)
    rows, columns = linear_sum_assignment(-confusion)
    right_to_left = np.full(n_classes, -1, dtype=np.int64)
    right_to_left[columns] = rows
    _require(np.all(right_to_left >= 0), "Hungarian assignment did not cover every class")
    return {
        "best_agreement": float(np.mean(left == right_to_left[right])),
        "permutation_right_to_left": right_to_left.tolist(),
        "confusion_left_rows_right_columns": confusion.tolist(),
    }


def _canonicalize(labels: np.ndarray, source_class_for_anchor: Sequence[int], n_classes: int) -> np.ndarray:
    sources = [int(value) - 1 for value in source_class_for_anchor]
    _require(sorted(sources) == list(range(n_classes)), "class mapping is not a K-class permutation")
    source_to_anchor = np.full(n_classes, -1, dtype=np.int64)
    for anchor, source in enumerate(sources):
        source_to_anchor[source] = anchor
    labels = np.asarray(labels, dtype=np.int64)
    _require(np.all((labels >= 0) & (labels < n_classes)), "native class labels are out of range")
    return source_to_anchor[labels]


def _particle_signature(audit: Mapping[str, Any]) -> dict[str, Any]:
    split = audit["particle_split"]
    return {
        "selected_count": int(split["selected_count"]),
        "half_counts": [int(value) for value in split["half_counts"]],
        "selected_order_sha256": str(split["selected_order_sha256"]),
        "half_order_sha256": [str(value) for value in split["half_order_sha256"]],
        "selected_source_indices_sha256": str(split["selected_source_indices_sha256"]),
        "particle_stack_sha256": str(split["particle_stack_sha256"]),
        "origin_particles_star_sha256": str(split["origin_particles_star_sha256"]),
        "source_indices_sha256": str(split["source_indices_sha256"]),
    }


def _config_without_seed(audit: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(audit["config"])
    config.pop("seed", None)
    return config


def _load_run(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"missing half-map audit: {path}")
    audit = json.loads(path.read_text())
    _require(audit.get("schema") == AUDIT_SCHEMA, f"unexpected audit schema: {path}")
    _require(audit.get("status") == "complete", f"audit is not complete: {path}")
    n_classes = int(audit["config"]["K"])
    _require(n_classes >= 2, "multiseed K-class audit requires K >= 2")
    max_iter = int(audit["config"]["max_iter"])
    seed = int(audit["config"]["seed"])
    audit_root = path.parent
    assignments: dict[int, dict[str, np.ndarray]] = {}
    array_artifacts: list[dict[str, Any]] = []
    mappings = audit["class_matching"]["source_class_for_anchor"]

    for half in (1, 2):
        arrays_path = audit_root / f"half{half}_particle_state_arrays.npz"
        _require(arrays_path.is_file(), f"missing particle-state arrays: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            identity = str(arrays["identity_sha256"].item())
            prefix = f"it{max_iter:03d}"
            relion_native = np.asarray(arrays[f"{prefix}_class_relion_one_based"], dtype=np.int64) - 1
            recovar_native = np.asarray(arrays[f"{prefix}_class_recovar_zero_based"], dtype=np.int64)
        _require(
            relion_native.size == int(audit["particle_split"]["half_counts"][half - 1]),
            f"half-{half} particle-state length differs from audit",
        )
        relion = _canonicalize(relion_native, mappings[f"relion_half{half}"], n_classes)
        recovar = _canonicalize(recovar_native, mappings[f"recovar_half{half}"], n_classes)
        reported = float(audit["assignments_and_support"][half - 1]["agreement"])
        observed = float(np.mean(relion == recovar))
        _require(
            np.isclose(observed, reported, atol=1e-12, rtol=0.0),
            f"half-{half} assignment agreement does not reproduce the audit",
        )
        assignments[half] = {"relion": relion, "recovar": recovar}
        array_artifacts.append(
            {
                "half": half,
                "path": str(arrays_path),
                "sha256": sha256_file(arrays_path),
                "identity_sha256": identity,
            }
        )

    class_rows = []
    for row in audit["classes"]:
        metrics = row["prospective_science_metrics"]["common_masked"]
        class_rows.append(
            {
                "class": int(row["canonical_class"]),
                "relion_halfmap_band_fsc_auc": float(metrics["relion_halfmap_band_fsc_auc"]),
                "recovar_halfmap_band_fsc_auc": float(metrics["recovar_halfmap_band_fsc_auc"]),
                "recovar_minus_relion_halfmap_band_fsc_auc": float(
                    metrics["recovar_minus_relion_halfmap_band_fsc_auc"]
                ),
                "cross_merged_band_fsc_auc": float(metrics["cross_merged_band_fsc_auc"]),
            }
        )
    _require([row["class"] for row in class_rows] == list(range(1, n_classes + 1)), "class rows changed")

    performance = audit["performance"]
    performance_rows = {
        engine: [
            {
                "half": index,
                "wall_s": float(row["wall_s"]),
                "peak_hbm_mib": int(row["peak_hbm_mib"]),
                "max_rss_kib": int(row["max_rss_kib"]),
                "slurm_job_id": str(row["slurm_job_id"]),
                "physical_gpu_uuid": str(row["physical_gpu_uuid"]),
            }
            for index, row in enumerate(performance[engine], start=1)
        ]
        for engine in ("relion", "recovar")
    }
    return {
        "seed": seed,
        "path": path,
        "audit": audit,
        "audit_sha256": sha256_file(path),
        "assignments": assignments,
        "array_artifacts": array_artifacts,
        "class_rows": class_rows,
        "performance": performance_rows,
    }


def aggregate(audit_paths: Sequence[Path], *, expected_seeds: tuple[int, ...]) -> dict[str, Any]:
    _require(len(audit_paths) == len(expected_seeds), "one audit is required for every expected seed")
    runs = sorted((_load_run(Path(path)) for path in audit_paths), key=lambda row: row["seed"])
    seeds = tuple(int(row["seed"]) for row in runs)
    _require(seeds == tuple(sorted(expected_seeds)), f"expected seeds {expected_seeds}, got {seeds}")

    anchor = runs[0]["audit"]
    contract = {
        "dataset": anchor["dataset"],
        "profile": anchor["profile"],
        "config_without_seed": _config_without_seed(anchor),
        "particle_signature": _particle_signature(anchor),
        "analysis_policy": anchor["analysis_policy"],
        "thresholds": anchor["thresholds"],
        "source_commit": anchor["source"]["commit"],
        "source_tree": anchor["source"]["tree"],
    }
    for run in runs[1:]:
        audit = run["audit"]
        candidate = {
            "dataset": audit["dataset"],
            "profile": audit["profile"],
            "config_without_seed": _config_without_seed(audit),
            "particle_signature": _particle_signature(audit),
            "analysis_policy": audit["analysis_policy"],
            "thresholds": audit["thresholds"],
            "source_commit": audit["source"]["commit"],
            "source_tree": audit["source"]["tree"],
        }
        _require(candidate == contract, f"frozen run contract differs for seed {run['seed']}")

    half_rows: list[dict[str, Any]] = []
    for half in (1, 2):
        particle_state_identities = {
            str(run["array_artifacts"][half - 1]["identity_sha256"]) for run in runs
        }
        _require(
            len(particle_state_identities) == 1,
            f"half-{half} particle-state identity hash differs across seeds",
        )
        within_pairs = []
        within_values: list[float] = []
        for left, right in itertools.combinations(runs, 2):
            row: dict[str, Any] = {"seeds": [left["seed"], right["seed"]]}
            for engine in ("relion", "recovar"):
                comparison = _best_agreement(
                    left["assignments"][half][engine],
                    right["assignments"][half][engine],
                    int(contract["config_without_seed"]["K"]),
                )
                row[engine] = comparison
                within_values.append(float(comparison["best_agreement"]))
            within_pairs.append(row)

        cross_rows = []
        cross_values: list[float] = []
        for run in runs:
            comparison = _best_agreement(
                run["assignments"][half]["relion"],
                run["assignments"][half]["recovar"],
                int(contract["config_without_seed"]["K"]),
            )
            cross_values.append(float(comparison["best_agreement"]))
            cross_rows.append({"seed": run["seed"], **comparison})
        half_rows.append(
            {
                "half": half,
                "particle_order_sha256": contract["particle_signature"]["half_order_sha256"][
                    half - 1
                ],
                "particle_state_identity_sha256": next(iter(particle_state_identities)),
                "within_engine_cross_seed": within_pairs,
                "same_seed_cross_engine": cross_rows,
                "within_engine_cross_seed_summary": _numeric_summary(within_values),
                "same_seed_cross_engine_summary": _numeric_summary(cross_values),
                "same_seed_cross_engine_min_exceeds_within_engine_cross_seed_max": min(cross_values)
                > max(within_values),
            }
        )

    quality_rows = [
        {"seed": run["seed"], **class_row}
        for run in runs
        for class_row in run["class_rows"]
    ]
    deltas = [row["recovar_minus_relion_halfmap_band_fsc_auc"] for row in quality_rows]
    per_class = []
    for class_id in range(1, int(contract["config_without_seed"]["K"]) + 1):
        selected = [row for row in quality_rows if row["class"] == class_id]
        per_class.append(
            {
                "class": class_id,
                "relion_halfmap_band_fsc_auc": _numeric_summary(
                    [row["relion_halfmap_band_fsc_auc"] for row in selected]
                ),
                "recovar_halfmap_band_fsc_auc": _numeric_summary(
                    [row["recovar_halfmap_band_fsc_auc"] for row in selected]
                ),
                "paired_recovar_minus_relion": _numeric_summary(
                    [row["recovar_minus_relion_halfmap_band_fsc_auc"] for row in selected]
                ),
            }
        )

    performance_summary = {}
    for engine in ("relion", "recovar"):
        rows = [item for run in runs for item in run["performance"][engine]]
        performance_summary[engine] = {
            "wall_s": _numeric_summary([row["wall_s"] for row in rows]),
            "peak_hbm_mib": _numeric_summary([row["peak_hbm_mib"] for row in rows]),
            "max_rss_kib": _numeric_summary([row["max_rss_kib"] for row in rows]),
        }

    output_runs = []
    for run in runs:
        audit = run["audit"]
        output_runs.append(
            {
                "seed": run["seed"],
                "audit": {"path": str(run["path"]), "sha256": run["audit_sha256"]},
                "particle_state_arrays": run["array_artifacts"],
                "prospective_science_gate": audit["prospective_science_gate"],
                "same_seed_assignment_agreement": [
                    float(row["agreement"]) for row in audit["assignments_and_support"]
                ],
                "classes": run["class_rows"],
                "performance": run["performance"],
            }
        )

    return {
        "schema": SCHEMA,
        "status": "complete",
        "admission_status": "DIAGNOSTIC_ONLY_PER_SEED_GATES_RETAINED",
        "expected_seeds": list(sorted(expected_seeds)),
        "contract": contract,
        "runs": output_runs,
        "assignment_stability": {"halves": half_rows},
        "quality": {
            "rows": quality_rows,
            "paired_recovar_minus_relion_halfmap_band_fsc_auc": _numeric_summary(deltas),
            "per_class_across_seeds": per_class,
            "cross_merged_band_fsc_auc": _numeric_summary(
                [row["cross_merged_band_fsc_auc"] for row in quality_rows]
            ),
        },
        "performance_across_six_half_runs": performance_summary,
        "observations": {
            "all_per_seed_prospective_science_gates_rejected": all(
                run["audit"]["prospective_science_gate"]["accepted"] is False for run in runs
            ),
            "same_seed_cross_engine_assignment_exceeds_within_engine_seed_stability_each_half": all(
                row["same_seed_cross_engine_min_exceeds_within_engine_cross_seed_max"]
                for row in half_rows
            ),
            "interpretation": (
                "The assignment comparison is evidence of seed-sensitive K-class local optima, not a "
                "replacement for any failed map-quality or prospective per-seed gate."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", required=True, type=Path)
    parser.add_argument("--expected-seeds", default="42001,42002,42003")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = aggregate(args.audit, expected_seeds=parse_expected_seeds(args.expected_seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"status={payload['status']} output={args.output}")


if __name__ == "__main__":
    main()
