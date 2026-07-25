#!/usr/bin/env python3
"""Audit K=4 RELION/RECOVAR fine candidates and their coarse parents.

The RELION BPref pre-scatter capture stores every fine candidate considered
for a particle in ``artifact.rotations`` and identifies positive class
contributors through ``artifact.rows["orientation_local"]``.  RECOVAR's
validated contribution companions store the corresponding fine-candidate
global indices and reconstruction mask.  Integer identities are compared only
after each engine's indices have been qualified against its captured rotation
matrices and converted into one canonical index order.  A sampling-
perturbation mismatch invalidates the cross-engine support classification.
The default comparison is exact because RELION's rounded STAR value can differ
enough from its live seed-derived value to change outer-shell support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _stack_index(identity: object) -> int:
    return int(str(identity).split("@", 1)[0])


def _quantiles(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {key: None for key in ("min", "p05", "p50", "p95", "max")}
    _require(bool(np.all(np.isfinite(array))), "non-finite metric input")
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _load_panel(path: Path) -> tuple[dict[int, str], dict[str, object]]:
    payload = json.loads(path.read_text())
    _require(payload.get("status") == "complete", "panel is incomplete")
    _require(payload.get("diagnostic_only") is True, "panel is not diagnostic-only")
    categories: dict[int, str] = {}
    for row in payload["records"]:
        stack = int(row["stack_index_one_based"])
        _require(stack not in categories, f"duplicate panel stack {stack}")
        categories[stack] = str(row["category"])
    _require(
        len(categories) == int(payload["n_selected"]),
        "panel count differs from its manifest",
    )
    return categories, payload


def _load_recovar_candidates(
    geometry_directory: Path,
    scripts_directory: Path,
    *,
    class_index: int,
    expected_fine_rotations: np.ndarray,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    sys.path.insert(0, str(scripts_directory))
    from validate_bpref_device_signature import _validate_signature  # type: ignore

    records: dict[int, dict[str, np.ndarray]] = {}
    signature_paths: list[str] = []
    current_sizes: set[int] = set()
    geometry_max_abs = 0.0
    for path in sorted(geometry_directory.glob("*.device.npz")):
        try:
            result = _validate_signature(path)
        except (KeyError, ValueError):
            # The directory may also contain the aggregate native-panel NPZ,
            # which intentionally is not a per-call device signature.
            continue
        signature = result["signature"]
        contribution = result["contribution"]
        if int(np.asarray(signature["class_index"]).item()) != class_index:
            continue

        identities = np.asarray(contribution["image_identities"]).astype(str)
        identity_to_particle = {identity: index for index, identity in enumerate(identities.tolist())}
        _require(
            len(identity_to_particle) == identities.size,
            f"duplicate contribution identity: {path}",
        )
        active_particles = np.asarray(contribution["active_particle_rows"], dtype=np.int64)
        active_rows = np.asarray(contribution["active_rotation_rows"], dtype=np.int64)
        active_global = np.asarray(contribution["active_global_rotation_indices"], dtype=np.int64)
        _require(
            bool(np.all((0 <= active_global) & (active_global < expected_fine_rotations.shape[0]))),
            f"RECOVAR active rotation index outside generated grid: {path}",
        )
        active_rotations = np.asarray(contribution["active_rotations"], dtype=np.float64)
        geometry_max_abs = max(
            geometry_max_abs,
            float(
                np.max(
                    np.abs(active_rotations - expected_fine_rotations[active_global]),
                    initial=0.0,
                )
            ),
        )
        reconstruction_mask = np.asarray(contribution["reconstruction_mask"], dtype=bool)

        for identity in np.asarray(signature["particle_image_identities"]).astype(str):
            particle = identity_to_particle.get(identity)
            _require(
                particle is not None,
                f"signature identity absent from contribution: {identity}",
            )
            stack = _stack_index(identity)
            _require(stack not in records, f"duplicate RECOVAR stack {stack}")
            selected = np.flatnonzero(active_particles == particle)
            candidate_globals = np.unique(active_global[selected])
            contributor_globals = []
            for active in selected:
                local_row = int(active_rows[active])
                if np.any(reconstruction_mask[particle, local_row]):
                    contributor_globals.append(int(active_global[active]))
            records[stack] = {
                "candidate_globals": candidate_globals,
                "contributor_globals": np.unique(np.asarray(contributor_globals, dtype=np.int64)),
            }
        current_sizes.add(int(np.asarray(signature["current_size"]).item()))
        signature_paths.append(str(path.resolve()))

    _require(records, f"no class-{class_index + 1} signatures found")
    _require(len(current_sizes) == 1, "RECOVAR current size changed")
    return records, {
        "signature_count": len(signature_paths),
        "signature_paths": signature_paths,
        "current_size": next(iter(current_sizes)),
        "geometry_max_abs": geometry_max_abs,
    }


def _coarse_grid_dimensions(healpix_order: int, coarse_rotation_count: int) -> tuple[int, int]:
    pixel_count = 12 * (2 ** int(healpix_order)) ** 2
    _require(
        coarse_rotation_count % pixel_count == 0,
        "coarse rotation count is incompatible with the HEALPix order",
    )
    return pixel_count, coarse_rotation_count // pixel_count


def _relion_to_canonical_coarse_indices(
    relion_indices: np.ndarray,
    *,
    healpix_order: int,
    coarse_rotation_count: int,
) -> np.ndarray:
    """Convert RELION direction-major indices to RECOVAR's psi-major order."""
    pixel_count, psi_count = _coarse_grid_dimensions(healpix_order, coarse_rotation_count)
    relion_indices = np.asarray(relion_indices, dtype=np.int64)
    pixel = relion_indices // psi_count
    psi = relion_indices % psi_count
    return psi * pixel_count + pixel


def _relion_global_indices(
    rotations: np.ndarray,
    *,
    class_index: int,
    coarse_rotation_count: int,
    oversampling_factor: int,
    healpix_order: int,
) -> np.ndarray:
    class_keys = np.asarray(rotations["orientation_class_key"], dtype=np.int64)
    oversampled = np.asarray(rotations["oversampled_rotation"], dtype=np.int64)
    coarse = class_keys - class_index * coarse_rotation_count
    _require(
        bool(np.all((0 <= coarse) & (coarse < coarse_rotation_count))),
        "RELION orientation-class key is outside the requested class",
    )
    _require(
        bool(np.all((0 <= oversampled) & (oversampled < oversampling_factor))),
        "RELION oversampled-rotation index is outside the configured factor",
    )
    canonical_coarse = _relion_to_canonical_coarse_indices(
        coarse,
        healpix_order=healpix_order,
        coarse_rotation_count=coarse_rotation_count,
    )
    return np.unique(canonical_coarse * oversampling_factor + oversampled)


def _relion_geometry_max_abs(
    artifacts: list[object],
    *,
    class_index: int,
    coarse_rotation_count: int,
    oversampling_factor: int,
    expected_relion_fine_rotations: np.ndarray,
) -> float:
    max_abs = 0.0
    for artifact in artifacts:
        rotations = artifact.rotations
        class_keys = np.asarray(rotations["orientation_class_key"], dtype=np.int64)
        oversampled = np.asarray(rotations["oversampled_rotation"], dtype=np.int64)
        coarse = class_keys - class_index * coarse_rotation_count
        _require(
            bool(np.all((0 <= coarse) & (coarse < coarse_rotation_count))),
            "RELION orientation-class key is outside the requested class",
        )
        _require(
            bool(np.all((0 <= oversampled) & (oversampled < oversampling_factor))),
            "RELION oversampled-rotation index is outside the configured factor",
        )
        expected = expected_relion_fine_rotations[coarse * oversampling_factor + oversampled]
        # RELION's captured Projector matrix is the transpose of RECOVAR's
        # host-side rotation convention.
        captured = np.asarray(rotations["matrix"], dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
        max_abs = max(
            max_abs,
            float(np.max(np.abs(captured - expected), initial=0.0)),
        )
    return max_abs


def _particle_report(
    artifact: object,
    recovar: dict[str, np.ndarray],
    category: str,
    *,
    class_index: int,
    coarse_rotation_count: int,
    oversampling_factor: int,
    healpix_order: int,
) -> dict[str, object]:
    relion_candidates = _relion_global_indices(
        artifact.rotations,
        class_index=class_index,
        coarse_rotation_count=coarse_rotation_count,
        oversampling_factor=oversampling_factor,
        healpix_order=healpix_order,
    )
    contributor_rows = np.unique(np.asarray(artifact.rows["orientation_local"], dtype=np.int64))
    relion_contributors = _relion_global_indices(
        artifact.rotations[contributor_rows],
        class_index=class_index,
        coarse_rotation_count=coarse_rotation_count,
        oversampling_factor=oversampling_factor,
        healpix_order=healpix_order,
    )
    recovar_candidates = np.asarray(recovar["candidate_globals"], dtype=np.int64)
    recovar_contributors = np.asarray(recovar["contributor_globals"], dtype=np.int64)

    relion_parents = np.unique(relion_candidates // oversampling_factor)
    recovar_parents = np.unique(recovar_candidates // oversampling_factor)
    parent_overlap = np.intersect1d(relion_parents, recovar_parents)
    fine_overlap = np.intersect1d(relion_candidates, recovar_candidates)
    contributor_overlap = np.intersect1d(relion_contributors, recovar_contributors)
    return {
        "stack_index_one_based": int(artifact.stack_index),
        "category": category,
        "relion_fine_candidate_count": int(relion_candidates.size),
        "recovar_fine_candidate_count": int(recovar_candidates.size),
        "fine_candidate_overlap_count": int(fine_overlap.size),
        "fine_candidate_sets_exact": bool(np.array_equal(relion_candidates, recovar_candidates)),
        "relion_coarse_parent_count": int(relion_parents.size),
        "recovar_coarse_parent_count": int(recovar_parents.size),
        "coarse_parent_overlap_count": int(parent_overlap.size),
        "coarse_parent_sets_exact": bool(np.array_equal(relion_parents, recovar_parents)),
        "relion_is_complete_oversampled_expansion": bool(
            relion_candidates.size == oversampling_factor * relion_parents.size
        ),
        "recovar_is_complete_oversampled_expansion": bool(
            recovar_candidates.size == oversampling_factor * recovar_parents.size
        ),
        "relion_contributor_count": int(relion_contributors.size),
        "recovar_contributor_count": int(recovar_contributors.size),
        "contributor_overlap_count": int(contributor_overlap.size),
        "contributor_sets_exact": bool(
            np.array_equal(relion_contributors, recovar_contributors)
        ),
        "all_relion_contributors_in_recovar_candidates": bool(np.all(np.isin(relion_contributors, recovar_candidates))),
        "all_recovar_contributors_in_relion_candidates": bool(np.all(np.isin(recovar_contributors, relion_candidates))),
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    relion_parents = sum(int(row["relion_coarse_parent_count"]) for row in rows)
    recovar_parents = sum(int(row["recovar_coarse_parent_count"]) for row in rows)
    parent_overlap = sum(int(row["coarse_parent_overlap_count"]) for row in rows)
    relion_fine = sum(int(row["relion_fine_candidate_count"]) for row in rows)
    recovar_fine = sum(int(row["recovar_fine_candidate_count"]) for row in rows)
    fine_overlap = sum(int(row["fine_candidate_overlap_count"]) for row in rows)
    return {
        "particle_count": len(rows),
        "relion_fine_candidate_count": relion_fine,
        "recovar_fine_candidate_count": recovar_fine,
        "fine_candidate_overlap_count": fine_overlap,
        "fine_candidate_overlap_fraction_of_relion": (fine_overlap / relion_fine if relion_fine else None),
        "fine_candidate_overlap_fraction_of_recovar": (fine_overlap / recovar_fine if recovar_fine else None),
        "fine_candidate_sets_exact_count": sum(bool(row["fine_candidate_sets_exact"]) for row in rows),
        "relion_coarse_parent_count": relion_parents,
        "recovar_coarse_parent_count": recovar_parents,
        "coarse_parent_overlap_count": parent_overlap,
        "coarse_parent_overlap_fraction_of_relion": (parent_overlap / relion_parents if relion_parents else None),
        "coarse_parent_overlap_fraction_of_recovar": (parent_overlap / recovar_parents if recovar_parents else None),
        "coarse_parent_sets_exact_count": sum(bool(row["coarse_parent_sets_exact"]) for row in rows),
        "complete_relion_oversampled_expansion_count": sum(
            bool(row["relion_is_complete_oversampled_expansion"]) for row in rows
        ),
        "complete_recovar_oversampled_expansion_count": sum(
            bool(row["recovar_is_complete_oversampled_expansion"]) for row in rows
        ),
        "relion_contributor_count": sum(int(row["relion_contributor_count"]) for row in rows),
        "recovar_contributor_count": sum(int(row["recovar_contributor_count"]) for row in rows),
        "contributor_overlap_count": sum(int(row["contributor_overlap_count"]) for row in rows),
        "contributor_sets_exact_count": sum(bool(row["contributor_sets_exact"]) for row in rows),
        "all_relion_contributors_in_recovar_candidates_count": sum(
            bool(row["all_relion_contributors_in_recovar_candidates"]) for row in rows
        ),
        "all_recovar_contributors_in_relion_candidates_count": sum(
            bool(row["all_recovar_contributors_in_relion_candidates"]) for row in rows
        ),
        "coarse_parent_overlap_fraction_per_particle": _quantiles(
            [int(row["coarse_parent_overlap_count"]) / max(int(row["relion_coarse_parent_count"]), 1) for row in rows]
        ),
    }


def _classify_support(
    *,
    all_complete: bool,
    any_parent_difference: bool,
    all_fine_candidate_sets_exact: bool = False,
    any_contributor_difference: bool = False,
    relion_random_perturbation: float,
    recovar_random_perturbation: float,
    perturbation_tolerance: float,
) -> tuple[str, str, bool]:
    perturbations_match = bool(
        np.isclose(
            relion_random_perturbation,
            recovar_random_perturbation,
            rtol=0.0,
            atol=perturbation_tolerance,
        )
    )
    if not perturbations_match:
        return (
            "invalid_comparison",
            "incomparable_sampling_perturbation_precludes_cross_engine_support_claim",
            False,
        )
    if all_complete and any_parent_difference:
        return (
            "complete",
            "coarse_parent_support_difference_precedes_fine_scoring",
            True,
        )
    if all_fine_candidate_sets_exact and any_contributor_difference:
        return (
            "complete",
            "fine_rotation_contributor_support_difference_after_candidate_generation",
            True,
        )
    if all_fine_candidate_sets_exact:
        return (
            "complete",
            "candidate_and_rotation_contributor_support_exact",
            True,
        )
    return (
        "complete",
        "candidate_support_difference_not_localized_to_coarse_parents",
        True,
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    categories, panel = _load_panel(args.panel_json)
    class_index = args.class_one_based - 1
    sys.path.insert(0, str(args.scripts_directory))
    from validate_relion_bpref_prescatter import validate_directory  # type: ignore

    expected = np.asarray(sorted(categories), dtype=np.uint64)
    artifacts, relion_validation = validate_directory(
        args.capture_directory,
        expected_particles=expected.size,
        expected_stack_indices=expected,
    )
    _require(
        int(relion_validation["class_one_based"]) == args.class_one_based,
        "RELION capture class differs from the requested class",
    )
    _require(
        bool(relion_validation["classification_ready"]),
        "RELION capture validation is not classification-ready",
    )
    from recovar.em.sampling import get_oversampled_rotation_grid_from_samples

    coarse_indices = np.arange(args.coarse_rotation_count, dtype=np.int64)
    expected_relion_fine_rotations, _ = get_oversampled_rotation_grid_from_samples(
        coarse_indices,
        args.healpix_order,
        oversampling_order=args.oversampling_order,
        random_perturbation=args.relion_random_perturbation,
        rotation_index_order="relion",
    )
    expected_recovar_fine_rotations, _ = get_oversampled_rotation_grid_from_samples(
        coarse_indices,
        args.healpix_order,
        oversampling_order=args.oversampling_order,
        random_perturbation=args.recovar_random_perturbation,
    )
    expected_relion_fine_rotations = np.asarray(expected_relion_fine_rotations, dtype=np.float64)
    expected_recovar_fine_rotations = np.asarray(expected_recovar_fine_rotations, dtype=np.float64)
    _require(
        expected_relion_fine_rotations.shape[0] == args.coarse_rotation_count * args.oversampling_factor,
        "generated RELION fine-grid size differs from requested topology",
    )
    _require(
        expected_recovar_fine_rotations.shape == expected_relion_fine_rotations.shape,
        "generated engine fine-grid shapes differ",
    )
    relion_geometry_max_abs = _relion_geometry_max_abs(
        artifacts,
        class_index=class_index,
        coarse_rotation_count=args.coarse_rotation_count,
        oversampling_factor=args.oversampling_factor,
        expected_relion_fine_rotations=expected_relion_fine_rotations,
    )
    recovar, recovar_validation = _load_recovar_candidates(
        args.geometry_directory,
        args.scripts_directory,
        class_index=class_index,
        expected_fine_rotations=expected_recovar_fine_rotations,
    )
    geometry_tolerance = float(args.geometry_tolerance)
    _require(
        relion_geometry_max_abs <= geometry_tolerance,
        "RELION captured matrices do not match the generated RELION grid: "
        f"{relion_geometry_max_abs:.9g} > {geometry_tolerance:.9g}",
    )
    _require(
        recovar_validation["geometry_max_abs"] <= geometry_tolerance,
        "RECOVAR captured matrices do not match the generated RECOVAR grid: "
        f"{recovar_validation['geometry_max_abs']:.9g} > "
        f"{geometry_tolerance:.9g}",
    )
    _require(set(recovar) == set(categories), "RECOVAR panel identities differ")
    artifact_by_stack = {int(artifact.stack_index): artifact for artifact in artifacts}
    rows = [
        _particle_report(
            artifact_by_stack[stack],
            recovar[stack],
            categories[stack],
            class_index=class_index,
            coarse_rotation_count=args.coarse_rotation_count,
            oversampling_factor=args.oversampling_factor,
            healpix_order=args.healpix_order,
        )
        for stack in sorted(categories)
    ]
    summary = _summarize(rows)
    all_complete = summary["complete_relion_oversampled_expansion_count"] == len(rows) and summary[
        "complete_recovar_oversampled_expansion_count"
    ] == len(rows)
    any_parent_difference = summary["coarse_parent_sets_exact_count"] != len(rows)
    all_fine_candidate_sets_exact = summary["fine_candidate_sets_exact_count"] == len(rows)
    any_contributor_difference = summary["contributor_sets_exact_count"] != len(rows)
    status, classification, perturbations_match = _classify_support(
        all_complete=all_complete,
        any_parent_difference=any_parent_difference,
        all_fine_candidate_sets_exact=all_fine_candidate_sets_exact,
        any_contributor_difference=any_contributor_difference,
        relion_random_perturbation=args.relion_random_perturbation,
        recovar_random_perturbation=args.recovar_random_perturbation,
        perturbation_tolerance=args.perturbation_tolerance,
    )
    return {
        "schema": "em_k4_relion_recovar_candidate_support_audit_v3",
        "status": status,
        "metric_policy": ("matrix-qualified canonical candidate identities; no correlation"),
        "classification": classification,
        "gates": {
            "fresh_relion_capture_validation": True,
            "panel_identity_exact": True,
            "recovar_device_signatures_validated": True,
            "relion_rotation_geometry_validated": True,
            "recovar_rotation_geometry_validated": True,
            "sampling_perturbations_match": perturbations_match,
            "complete_oversampled_expansion_for_every_particle": all_complete,
            "coarse_parent_sets_exact_for_every_particle": not any_parent_difference,
            "fine_candidate_sets_exact_for_every_particle": all_fine_candidate_sets_exact,
            "rotation_contributor_sets_exact_for_every_particle": not any_contributor_difference,
        },
        "scope": {
            "iteration": args.iteration,
            "half": args.half,
            "class_one_based": args.class_one_based,
            "particle_count": len(rows),
            "healpix_order": args.healpix_order,
            "oversampling_order": args.oversampling_order,
            "coarse_rotation_count": args.coarse_rotation_count,
            "oversampling_factor": args.oversampling_factor,
            "relion_random_perturbation": args.relion_random_perturbation,
            "recovar_random_perturbation": args.recovar_random_perturbation,
            "sampling_perturbation_delta": (args.recovar_random_perturbation - args.relion_random_perturbation),
            "geometry_tolerance": geometry_tolerance,
            "relion_geometry_max_abs": relion_geometry_max_abs,
            "recovar_geometry_max_abs": recovar_validation["geometry_max_abs"],
            "recovar_current_size": recovar_validation["current_size"],
            "recovar_signature_count": recovar_validation["signature_count"],
            "panel_identity_sha256": panel.get("identity_sha256", panel.get("panel_identity_sha256")),
        },
        "descriptive_index_overlap_unqualified": {
            "all_particles": summary,
            "by_category": {
                category: _summarize([row for row in rows if row["category"] == category])
                for category in sorted(set(categories.values()))
            },
        },
        "particles": rows,
        "qualification": (
            "RELION direction-major orientation_class_key and RECOVAR "
            "psi-major active_global_rotation_indices were each validated "
            "against captured matrices, then converted to canonical psi-major "
            "identities. Cross-engine overlap remains descriptive and cannot "
            "localize parity unless the sampling perturbations match."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", required=True, type=Path)
    parser.add_argument("--geometry-directory", required=True, type=Path)
    parser.add_argument("--panel-json", required=True, type=Path)
    parser.add_argument("--scripts-directory", required=True, type=Path)
    parser.add_argument("--class-one-based", required=True, type=int)
    parser.add_argument("--healpix-order", required=True, type=int)
    parser.add_argument("--oversampling-order", required=True, type=int)
    parser.add_argument("--coarse-rotation-count", required=True, type=int)
    parser.add_argument("--oversampling-factor", required=True, type=int)
    parser.add_argument("--relion-random-perturbation", required=True, type=float)
    parser.add_argument("--recovar-random-perturbation", required=True, type=float)
    parser.add_argument(
        "--perturbation-tolerance",
        default=0.0,
        type=float,
        help=(
            "Absolute tolerance for comparing live sampling perturbations "
            "(default: exact equality; STAR-rounded values are not equivalent)"
        ),
    )
    parser.add_argument("--geometry-tolerance", default=1e-6, type=float)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--half", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = audit(args)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
