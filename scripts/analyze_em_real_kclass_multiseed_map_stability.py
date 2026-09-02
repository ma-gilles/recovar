#!/usr/bin/env python3
"""Compare final real-data K-class maps across engines and frozen seeds.

The independent-half auditor aligns every map set to that seed's RELION-half-1
frame.  This analyzer reuses those sealed transforms, merges the two genuine
particle halves, then applies one additional proper-rigid transform per seed
and engine before measuring cross-seed FSC.  Same-seed cross-engine and
within-engine cross-seed comparisons therefore use the same full, unmasked,
non-DC FSC-AUC metric.  The result is diagnostic and cannot rescue a rejected
per-seed science gate.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import aggregate_em_real_kclass_halfmap_seeds as assignment_aggregate
from scripts.audit_em_real_kclass_halfmaps import (
    N_CLASSES,
    _align_set_to_anchor,
    _hungarian_to_anchor,
    _load_maps,
    _pairwise_auc,
)
from scripts.collect_em_k1_science_diagnostics import (
    ShellFscCalculator,
    apply_proper_rigid_transform,
)

SCHEMA = "recovar.em.real_kclass_multiseed_map_stability.v1"
REPRODUCTION_SCHEMA = "recovar.em.real_kclass_multiseed_map_stability_reproduction.v1"
DEFAULT_MAX_FSC_AUC_ABS_DELTA = 5e-3
DEFAULT_MIN_PER_CLASS_SEPARATION_MARGIN = 5e-2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _alignment_transform(
    maps: Sequence[np.ndarray],
    fit: Mapping[str, Any],
    *,
    interpolation_order: int,
) -> list[np.ndarray]:
    if fit.get("identity_anchor") is True:
        return [np.asarray(volume, dtype=np.float32) for volume in maps]
    rotation = np.asarray(fit["rotation_matrix_recovar_to_relion"], dtype=np.float64)
    translation = np.asarray(fit["translation_recovar_to_relion_zyx"], dtype=np.float64)
    _require(rotation.shape == (3, 3), "stored alignment rotation is not 3x3")
    _require(translation.shape == (3,), "stored alignment translation is not length 3")
    _require(np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6), "stored alignment is improper")
    return [
        apply_proper_rigid_transform(
            volume,
            rotation,
            translation,
            interpolation_order=interpolation_order,
        )
        for volume in maps
    ]


def _canonical_order(audit: Mapping[str, Any], label: str) -> list[int]:
    one_based = [int(value) for value in audit["class_matching"]["source_class_for_anchor"][label]]
    _require(
        sorted(one_based) == list(range(1, N_CLASSES + 1)),
        f"{label} class map is not a K-class permutation",
    )
    return [value - 1 for value in one_based]


def _relion_paths(run_root: Path, *, half: int, max_iter: int) -> list[Path]:
    paths = [
        run_root / f"half{half}" / "relion" / f"run_it{max_iter:03d}_class{class_id:03d}.mrc"
        for class_id in range(1, N_CLASSES + 1)
    ]
    _require(all(path.is_file() for path in paths), f"missing RELION half-{half} final maps")
    return paths


def _recovar_paths(audit: Mapping[str, Any], *, half: int) -> list[Path]:
    rows = sorted(
        audit["recovar_internal_half_selection_audit"][half - 1],
        key=lambda row: int(row["class"]),
    )
    _require([int(row["class"]) for row in rows] == list(range(1, N_CLASSES + 1)), "RECOVAR map rows changed")
    paths = [Path(row["selected_map"]).resolve() for row in rows]
    for row, path in zip(rows, paths, strict=True):
        _require(path.is_file(), f"missing RECOVAR selected map: {path}")
        _require(
            assignment_aggregate.sha256_file(path) == str(row["selected_sha256"]),
            f"RECOVAR selected-map hash changed: {path}",
        )
    return paths


def _load_seed_maps(path: Path) -> dict[str, Any]:
    path = path.resolve()
    audit = json.loads(path.read_text())
    _require(
        audit.get("schema") == assignment_aggregate.AUDIT_SCHEMA,
        f"unexpected audit schema: {path}",
    )
    seed = int(audit["config"]["seed"])
    max_iter = int(audit["config"]["max_iter"])
    run_root = path.parent.parent
    interpolation_order = int(audit["analysis_policy"]["alignment"]["interpolation_order"])
    aligned: dict[str, list[np.ndarray]] = {}
    artifacts: list[dict[str, Any]] = []
    voxel_sizes: list[float] = []

    for engine in ("relion", "recovar"):
        for half in (1, 2):
            label = f"{engine}_half{half}"
            paths = (
                _relion_paths(run_root, half=half, max_iter=max_iter)
                if engine == "relion"
                else _recovar_paths(audit, half=half)
            )
            maps, voxel_size = _load_maps(paths, frame=engine)
            voxel_sizes.append(float(voxel_size))
            maps = _alignment_transform(
                maps,
                audit["alignment"]["sets"][label],
                interpolation_order=interpolation_order,
            )
            order = _canonical_order(audit, label)
            aligned[label] = [maps[source] for source in order]
            artifacts.extend(
                {
                    "engine": engine,
                    "half": half,
                    "source_class": source_class + 1,
                    "canonical_class": canonical_class + 1,
                    "path": str(paths[source_class]),
                    "sha256": assignment_aggregate.sha256_file(paths[source_class]),
                }
                for canonical_class, source_class in enumerate(order)
            )

    _require(max(voxel_sizes) - min(voxel_sizes) <= 1e-5, f"seed {seed} map voxel sizes differ")
    merged = {
        engine: [
            np.float32(0.5) * (half1 + half2)
            for half1, half2 in zip(
                aligned[f"{engine}_half1"],
                aligned[f"{engine}_half2"],
                strict=True,
            )
        ]
        for engine in ("relion", "recovar")
    }
    return {
        "seed": seed,
        "audit": audit,
        "audit_path": path,
        "merged": merged,
        "artifacts": artifacts,
        "voxel_size": voxel_sizes[0],
    }


def _match_sets(
    calculator: ShellFscCalculator,
    source: Sequence[np.ndarray],
    anchor: Sequence[np.ndarray],
    *,
    label: str,
) -> dict[str, Any]:
    scores = _pairwise_auc(calculator, source, anchor)
    source_for_anchor, optimum = _hungarian_to_anchor(
        scores,
        label=label,
        min_absolute_margin=0.0,
        min_relative_margin=0.0,
    )
    matched = scores[np.asarray(source_for_anchor, dtype=np.int64), np.arange(N_CLASSES)]
    return {
        "source_class_for_anchor": [value + 1 for value in source_for_anchor],
        "matched_per_class_fsc_auc": matched.tolist(),
        "matched_mean_fsc_auc": float(np.mean(matched)),
        "pairwise_fsc_auc": scores.tolist(),
        "objective_margin": float(optimum["objective_margin"]),
        "relative_objective_margin": float(optimum["relative_objective_margin"]),
        "exact_optimum_count": int(optimum["exact_optimum_count"]),
    }


def summarize_separation(
    same_seed_rows: Sequence[Mapping[str, Any]],
    within_engine_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_class = []
    for class_index in range(N_CLASSES):
        same_values = [float(row["matched_per_class_fsc_auc"][class_index]) for row in same_seed_rows]
        within_values = [
            float(row["matched_per_class_fsc_auc"][class_index]) for row in within_engine_rows
        ]
        per_class.append(
            {
                "class": class_index + 1,
                "same_seed_cross_engine": assignment_aggregate._numeric_summary(same_values),
                "within_engine_cross_seed": assignment_aggregate._numeric_summary(within_values),
                "same_seed_min_exceeds_within_engine_cross_seed_max": min(same_values)
                > max(within_values),
            }
        )
    same_all = [
        float(value) for row in same_seed_rows for value in row["matched_per_class_fsc_auc"]
    ]
    within_all = [
        float(value) for row in within_engine_rows for value in row["matched_per_class_fsc_auc"]
    ]
    return {
        "per_class": per_class,
        "all_class_cells": {
            "same_seed_cross_engine": assignment_aggregate._numeric_summary(same_all),
            "within_engine_cross_seed": assignment_aggregate._numeric_summary(within_all),
        },
        "separated_for_every_class": all(
            row["same_seed_min_exceeds_within_engine_cross_seed_max"] for row in per_class
        ),
    }


def _keyed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: tuple[str, ...],
    label: str,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    keyed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        identity = tuple(row[key] for key in keys)
        _require(identity not in keyed, f"duplicate {label} row: {identity}")
        keyed[identity] = row
    return keyed


def _row_fsc_auc_values(row: Mapping[str, Any], *, label: str) -> np.ndarray:
    matched = np.asarray(row["matched_per_class_fsc_auc"], dtype=np.float64)
    pairwise = np.asarray(row["pairwise_fsc_auc"], dtype=np.float64)
    _require(matched.shape == (N_CLASSES,), f"{label} matched FSC-AUC shape changed")
    _require(
        pairwise.shape == (N_CLASSES, N_CLASSES),
        f"{label} pairwise FSC-AUC shape changed",
    )
    values = np.concatenate(
        [
            matched,
            pairwise.reshape(-1),
            np.asarray([row["matched_mean_fsc_auc"]], dtype=np.float64),
        ]
    )
    _require(np.all(np.isfinite(values)), f"{label} contains a non-finite FSC-AUC")
    return values


def _seed_alignment_contract(report: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    stable_fit_fields = (
        "seed",
        "fit_box_size",
        "fit_max_shell_full_box",
        "interpolation_order",
        "optimizer_success",
        "optimizer_status",
        "optimizer_message",
        "refine_healpix_orders",
        "seed_healpix_order",
        "seed_rotation_matrix",
        "seed_source",
        "phase_correlation_seed_shift_fit_zyx",
    )
    contract: dict[str, list[dict[str, Any]]] = {}
    alignments = report["seed_alignment"]
    _require(set(alignments) == {"relion", "recovar"}, "seed-alignment engines changed")
    for engine in ("relion", "recovar"):
        rows = []
        for row in alignments[engine]:
            seed = int(row["seed"])
            if row.get("identity_anchor") is True:
                rows.append({"seed": seed, "identity_anchor": True})
                continue
            rotation = np.asarray(row["rotation_matrix_recovar_to_relion"], dtype=np.float64)
            _require(rotation.shape == (3, 3), f"{engine} seed {seed} fit rotation is not 3x3")
            _require(
                np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6),
                f"{engine} seed {seed} fit rotation is improper",
            )
            _require(
                bool(row["optimizer_success"]),
                f"{engine} seed {seed} continuous fit did not converge",
            )
            rows.append({field: row[field] for field in stable_fit_fields})
        contract[engine] = rows
    return contract


def verify_reproduction(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_fsc_auc_abs_delta: float = DEFAULT_MAX_FSC_AUC_ABS_DELTA,
    min_per_class_separation_margin: float = DEFAULT_MIN_PER_CLASS_SEPARATION_MARGIN,
) -> dict[str, Any]:
    """Verify stable science invariants while tolerating continuous-fit drift.

    The proper-rigid fit uses a continuous optimizer whose final parameters can
    differ slightly across CPU implementations.  Reproduction therefore freezes
    all inputs, discrete class assignments, and decision outcomes exactly, while
    allowing a small absolute drift in reported FSC-AUC values.
    """

    _require(max_fsc_auc_abs_delta >= 0.0, "FSC-AUC tolerance must be nonnegative")
    _require(
        min_per_class_separation_margin >= 0.0,
        "minimum per-class separation margin must be nonnegative",
    )
    violations: list[str] = []
    checks: dict[str, bool] = {}

    exact_top_level_fields = (
        "schema",
        "status",
        "admission_status",
        "accepted_result",
        "metric",
        "expected_seeds",
        "anchor_seed",
        "contract",
        "source_assignment_observations",
        "audits",
        "map_artifacts",
        "observations",
    )
    changed_fields = [
        field for field in exact_top_level_fields if reference.get(field) != candidate.get(field)
    ]
    checks["frozen_inputs_and_decisions_exact"] = not changed_fields
    if changed_fields:
        violations.append(f"frozen top-level fields changed: {', '.join(changed_fields)}")

    try:
        reference_alignment = _seed_alignment_contract(reference)
        candidate_alignment = _seed_alignment_contract(candidate)
        checks["seed_alignment_discrete_contract_exact"] = reference_alignment == candidate_alignment
        if reference_alignment != candidate_alignment:
            violations.append("seed-alignment discrete contract changed")
    except (KeyError, TypeError, ValueError) as error:
        checks["seed_alignment_discrete_contract_exact"] = False
        violations.append(f"invalid seed-alignment contract: {error}")

    row_specs = (
        ("same_seed_cross_engine", ("seed",)),
        ("within_engine_cross_seed", ("engine", "anchor_seed", "source_seed")),
    )
    max_deltas: dict[str, float | None] = {}
    for label, identity_fields in row_specs:
        try:
            reference_rows = _keyed_rows(
                reference[label],
                keys=identity_fields,
                label=f"reference {label}",
            )
            candidate_rows = _keyed_rows(
                candidate[label],
                keys=identity_fields,
                label=f"candidate {label}",
            )
            identities_match = reference_rows.keys() == candidate_rows.keys()
            checks[f"{label}_row_identities_exact"] = identities_match
            if not identities_match:
                violations.append(f"{label} row identities changed")
                max_deltas[label] = None
                continue

            discrete_match = True
            row_max_deltas = []
            for identity in reference_rows:
                reference_row = reference_rows[identity]
                candidate_row = candidate_rows[identity]
                for field in ("source_class_for_anchor", "exact_optimum_count"):
                    if reference_row[field] != candidate_row[field]:
                        discrete_match = False
                reference_values = _row_fsc_auc_values(
                    reference_row,
                    label=f"reference {label} {identity}",
                )
                candidate_values = _row_fsc_auc_values(
                    candidate_row,
                    label=f"candidate {label} {identity}",
                )
                row_max_deltas.append(float(np.max(np.abs(reference_values - candidate_values))))
            checks[f"{label}_class_assignments_exact"] = discrete_match
            if not discrete_match:
                violations.append(f"{label} class assignment or optimum count changed")
            max_delta = max(row_max_deltas, default=0.0)
            max_deltas[label] = max_delta
            within_tolerance = max_delta <= max_fsc_auc_abs_delta
            checks[f"{label}_fsc_auc_within_tolerance"] = within_tolerance
            if not within_tolerance:
                violations.append(
                    f"{label} FSC-AUC drift {max_delta:.9g} exceeds "
                    f"{max_fsc_auc_abs_delta:.9g}"
                )
        except (KeyError, TypeError, ValueError) as error:
            checks[f"{label}_report_valid"] = False
            max_deltas[label] = None
            violations.append(f"invalid {label} report: {error}")

    try:
        recomputed_separation = summarize_separation(
            candidate["same_seed_cross_engine"],
            candidate["within_engine_cross_seed"],
        )
        checks["candidate_separation_summary_self_consistent"] = (
            recomputed_separation == candidate["separation"]
        )
        if recomputed_separation != candidate["separation"]:
            violations.append("candidate separation summary is not self-consistent")
        per_class_margins = [
            float(row["same_seed_cross_engine"]["min"])
            - float(row["within_engine_cross_seed"]["max"])
            for row in recomputed_separation["per_class"]
        ]
        minimum_margin = min(per_class_margins)
        separation_retained = bool(recomputed_separation["separated_for_every_class"])
        checks["all_classes_remain_separated"] = separation_retained
        checks["minimum_per_class_separation_margin_retained"] = (
            minimum_margin >= min_per_class_separation_margin
        )
        if not separation_retained:
            violations.append("same-seed and cross-seed FSC-AUC no longer separate every class")
        if minimum_margin < min_per_class_separation_margin:
            violations.append(
                f"minimum per-class separation margin {minimum_margin:.9g} is below "
                f"{min_per_class_separation_margin:.9g}"
            )
    except (KeyError, TypeError, ValueError) as error:
        checks["candidate_separation_report_valid"] = False
        per_class_margins = []
        minimum_margin = None
        violations.append(f"invalid candidate separation report: {error}")

    passed = all(checks.values()) and not violations
    return {
        "schema": REPRODUCTION_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "policy": {
            "max_fsc_auc_absolute_delta": max_fsc_auc_abs_delta,
            "minimum_per_class_separation_margin": min_per_class_separation_margin,
            "continuous_proper_rigid_fit_parameters_byte_exact": False,
        },
        "checks": checks,
        "measurements": {
            "max_same_seed_cross_engine_fsc_auc_absolute_delta": max_deltas.get(
                "same_seed_cross_engine"
            ),
            "max_within_engine_cross_seed_fsc_auc_absolute_delta": max_deltas.get(
                "within_engine_cross_seed"
            ),
            "candidate_per_class_separation_margins": per_class_margins,
            "candidate_minimum_per_class_separation_margin": minimum_margin,
        },
        "violations": violations,
    }


def analyze(audit_paths: Sequence[Path], *, expected_seeds: tuple[int, ...]) -> dict[str, Any]:
    assignment_report = assignment_aggregate.aggregate(
        audit_paths,
        expected_seeds=expected_seeds,
    )
    runs = sorted((_load_seed_maps(Path(path)) for path in audit_paths), key=lambda row: row["seed"])
    seeds = tuple(int(run["seed"]) for run in runs)
    _require(seeds == tuple(sorted(expected_seeds)), f"expected seeds {expected_seeds}, got {seeds}")
    box_sizes = {int(run["merged"]["relion"][0].shape[0]) for run in runs}
    _require(len(box_sizes) == 1, "map box sizes differ across seeds")
    calculator = ShellFscCalculator(next(iter(box_sizes)))
    alignment_policy = assignment_report["contract"]["analysis_policy"]["alignment"]

    same_seed_rows = []
    for run in runs:
        same_seed_rows.append(
            {
                "seed": run["seed"],
                **_match_sets(
                    calculator,
                    run["merged"]["recovar"],
                    run["merged"]["relion"],
                    label=f"seed-{run['seed']}-recovar-to-relion",
                ),
            }
        )

    aligned: dict[str, dict[int, list[np.ndarray]]] = {"relion": {}, "recovar": {}}
    seed_alignment: dict[str, list[dict[str, Any]]] = {"relion": [], "recovar": []}
    anchor_seed = seeds[0]
    by_seed = {int(run["seed"]): run for run in runs}
    for engine in ("relion", "recovar"):
        aligned[engine][anchor_seed] = by_seed[anchor_seed]["merged"][engine]
        seed_alignment[engine].append({"seed": anchor_seed, "identity_anchor": True})
        for seed in seeds[1:]:
            aligned_maps, fit = _align_set_to_anchor(
                by_seed[seed]["merged"][engine],
                aligned[engine][anchor_seed],
                policy=alignment_policy,
            )
            aligned[engine][seed] = aligned_maps
            seed_alignment[engine].append({"seed": seed, **fit})

    within_engine_rows = []
    for engine in ("relion", "recovar"):
        for left_seed, right_seed in itertools.combinations(seeds, 2):
            within_engine_rows.append(
                {
                    "engine": engine,
                    "anchor_seed": left_seed,
                    "source_seed": right_seed,
                    **_match_sets(
                        calculator,
                        aligned[engine][right_seed],
                        aligned[engine][left_seed],
                        label=f"{engine}-{right_seed}-to-{left_seed}",
                    ),
                }
            )

    separation = summarize_separation(same_seed_rows, within_engine_rows)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "admission_status": "DIAGNOSTIC_ONLY_PER_SEED_GATES_RETAINED",
        "accepted_result": False,
        "metric": {
            "name": "full_unmasked_non_dc_fsc_auc",
            "proper_rigid_alignment": True,
            "one_transform_shared_by_all_four_classes": True,
            "reflection_sign_and_scale_fit": False,
            "absolute_resolution_claim": False,
        },
        "expected_seeds": list(seeds),
        "anchor_seed": anchor_seed,
        "contract": assignment_report["contract"],
        "source_assignment_observations": assignment_report["observations"],
        "audits": [
            {
                "seed": run["seed"],
                "path": str(run["audit_path"]),
                "sha256": assignment_aggregate.sha256_file(run["audit_path"]),
                "prospective_science_gate": run["audit"]["prospective_science_gate"],
            }
            for run in runs
        ],
        "map_artifacts": [row for run in runs for row in run["artifacts"]],
        "seed_alignment": seed_alignment,
        "same_seed_cross_engine": same_seed_rows,
        "within_engine_cross_seed": within_engine_rows,
        "separation": separation,
        "observations": {
            "all_per_seed_prospective_science_gates_rejected": all(
                run["audit"]["prospective_science_gate"]["accepted"] is False for run in runs
            ),
            "same_seed_cross_engine_min_exceeds_cross_seed_max_for_every_class": separation[
                "separated_for_every_class"
            ],
            "interpretation": (
                "Seed-matched RECOVAR and RELION maps are closer than either engine's maps across "
                "seeds for every class. This supports a shared seed-sensitive local-optimum "
                "boundary; it does not replace or rescue the rejected per-seed half-map gates."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", required=True, type=Path)
    parser.add_argument("--expected-seeds", default="42001,42002,42003")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--verification-output", type=Path)
    parser.add_argument(
        "--max-fsc-auc-abs-delta",
        default=DEFAULT_MAX_FSC_AUC_ABS_DELTA,
        type=float,
    )
    parser.add_argument(
        "--min-per-class-separation-margin",
        default=DEFAULT_MIN_PER_CLASS_SEPARATION_MARGIN,
        type=float,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (args.reference_report is None) != (args.verification_output is None):
        raise SystemExit("--reference-report and --verification-output must be provided together")
    if args.reference_report is not None and args.reference_report.resolve() == args.output.resolve():
        raise SystemExit("--reference-report and --output must be different files")
    reference_payload = None
    if args.reference_report is not None:
        reference_payload = json.loads(args.reference_report.read_text())
    payload = analyze(
        args.audit,
        expected_seeds=assignment_aggregate.parse_expected_seeds(args.expected_seeds),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"status={payload['status']} output={args.output}")
    if reference_payload is not None:
        verification = verify_reproduction(
            reference_payload,
            payload,
            max_fsc_auc_abs_delta=args.max_fsc_auc_abs_delta,
            min_per_class_separation_margin=args.min_per_class_separation_margin,
        )
        verification["reference_report"] = {
            "path": str(args.reference_report.resolve()),
            "sha256": assignment_aggregate.sha256_file(args.reference_report),
        }
        verification["candidate_report"] = {
            "path": str(args.output.resolve()),
            "sha256": assignment_aggregate.sha256_file(args.output),
        }
        args.verification_output.parent.mkdir(parents=True, exist_ok=True)
        args.verification_output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
        print(
            f"reproduction_status={verification['status']} "
            f"verification_output={args.verification_output}"
        )
        if verification["status"] != "PASS":
            raise SystemExit(3)


if __name__ == "__main__":
    main()
