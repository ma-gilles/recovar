#!/usr/bin/env python3
"""Audit K=4 RELION/RECOVAR fine candidates and their coarse parents.

The RELION BPref pre-scatter capture stores every fine candidate considered
for a particle in ``artifact.rotations`` and identifies positive class
contributors through ``artifact.rows["orientation_local"]``.  RECOVAR's
validated contribution companions store the corresponding fine-candidate
global indices and reconstruction mask.  This auditor compares those exact
integer identities; it does not use matrix proximity or correlation.
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
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    sys.path.insert(0, str(scripts_directory))
    from validate_bpref_device_signature import _validate_signature  # type: ignore

    records: dict[int, dict[str, np.ndarray]] = {}
    signature_paths: list[str] = []
    current_sizes: set[int] = set()
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
        identity_to_particle = {
            identity: index for index, identity in enumerate(identities.tolist())
        }
        _require(
            len(identity_to_particle) == identities.size,
            f"duplicate contribution identity: {path}",
        )
        active_particles = np.asarray(
            contribution["active_particle_rows"], dtype=np.int64
        )
        active_rows = np.asarray(
            contribution["active_rotation_rows"], dtype=np.int64
        )
        active_global = np.asarray(
            contribution["active_global_rotation_indices"], dtype=np.int64
        )
        reconstruction_mask = np.asarray(
            contribution["reconstruction_mask"], dtype=bool
        )

        for identity in np.asarray(signature["particle_image_identities"]).astype(
            str
        ):
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
                "contributor_globals": np.unique(
                    np.asarray(contributor_globals, dtype=np.int64)
                ),
            }
        current_sizes.add(int(np.asarray(signature["current_size"]).item()))
        signature_paths.append(str(path.resolve()))

    _require(records, f"no class-{class_index + 1} signatures found")
    _require(len(current_sizes) == 1, "RECOVAR current size changed")
    return records, {
        "signature_count": len(signature_paths),
        "signature_paths": signature_paths,
        "current_size": next(iter(current_sizes)),
    }


def _relion_global_indices(
    rotations: np.ndarray,
    *,
    class_index: int,
    coarse_rotation_count: int,
    oversampling_factor: int,
) -> np.ndarray:
    class_keys = np.asarray(
        rotations["orientation_class_key"], dtype=np.int64
    )
    oversampled = np.asarray(
        rotations["oversampled_rotation"], dtype=np.int64
    )
    coarse = class_keys - class_index * coarse_rotation_count
    _require(
        bool(np.all((0 <= coarse) & (coarse < coarse_rotation_count))),
        "RELION orientation-class key is outside the requested class",
    )
    _require(
        bool(np.all((0 <= oversampled) & (oversampled < oversampling_factor))),
        "RELION oversampled-rotation index is outside the configured factor",
    )
    return np.unique(coarse * oversampling_factor + oversampled)


def _particle_report(
    artifact: object,
    recovar: dict[str, np.ndarray],
    category: str,
    *,
    class_index: int,
    coarse_rotation_count: int,
    oversampling_factor: int,
) -> dict[str, object]:
    relion_candidates = _relion_global_indices(
        artifact.rotations,
        class_index=class_index,
        coarse_rotation_count=coarse_rotation_count,
        oversampling_factor=oversampling_factor,
    )
    contributor_rows = np.unique(
        np.asarray(artifact.rows["orientation_local"], dtype=np.int64)
    )
    relion_contributors = _relion_global_indices(
        artifact.rotations[contributor_rows],
        class_index=class_index,
        coarse_rotation_count=coarse_rotation_count,
        oversampling_factor=oversampling_factor,
    )
    recovar_candidates = np.asarray(
        recovar["candidate_globals"], dtype=np.int64
    )
    recovar_contributors = np.asarray(
        recovar["contributor_globals"], dtype=np.int64
    )

    relion_parents = np.unique(relion_candidates // oversampling_factor)
    recovar_parents = np.unique(recovar_candidates // oversampling_factor)
    parent_overlap = np.intersect1d(relion_parents, recovar_parents)
    fine_overlap = np.intersect1d(relion_candidates, recovar_candidates)
    return {
        "stack_index_one_based": int(artifact.stack_index),
        "category": category,
        "relion_fine_candidate_count": int(relion_candidates.size),
        "recovar_fine_candidate_count": int(recovar_candidates.size),
        "fine_candidate_overlap_count": int(fine_overlap.size),
        "fine_candidate_sets_exact": bool(
            np.array_equal(relion_candidates, recovar_candidates)
        ),
        "relion_coarse_parent_count": int(relion_parents.size),
        "recovar_coarse_parent_count": int(recovar_parents.size),
        "coarse_parent_overlap_count": int(parent_overlap.size),
        "coarse_parent_sets_exact": bool(
            np.array_equal(relion_parents, recovar_parents)
        ),
        "relion_is_complete_oversampled_expansion": bool(
            relion_candidates.size
            == oversampling_factor * relion_parents.size
        ),
        "recovar_is_complete_oversampled_expansion": bool(
            recovar_candidates.size
            == oversampling_factor * recovar_parents.size
        ),
        "relion_contributor_count": int(relion_contributors.size),
        "recovar_contributor_count": int(recovar_contributors.size),
        "all_relion_contributors_in_recovar_candidates": bool(
            np.all(np.isin(relion_contributors, recovar_candidates))
        ),
        "all_recovar_contributors_in_relion_candidates": bool(
            np.all(np.isin(recovar_contributors, relion_candidates))
        ),
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
        "fine_candidate_overlap_fraction_of_relion": (
            fine_overlap / relion_fine if relion_fine else None
        ),
        "fine_candidate_overlap_fraction_of_recovar": (
            fine_overlap / recovar_fine if recovar_fine else None
        ),
        "fine_candidate_sets_exact_count": sum(
            bool(row["fine_candidate_sets_exact"]) for row in rows
        ),
        "relion_coarse_parent_count": relion_parents,
        "recovar_coarse_parent_count": recovar_parents,
        "coarse_parent_overlap_count": parent_overlap,
        "coarse_parent_overlap_fraction_of_relion": (
            parent_overlap / relion_parents if relion_parents else None
        ),
        "coarse_parent_overlap_fraction_of_recovar": (
            parent_overlap / recovar_parents if recovar_parents else None
        ),
        "coarse_parent_sets_exact_count": sum(
            bool(row["coarse_parent_sets_exact"]) for row in rows
        ),
        "complete_relion_oversampled_expansion_count": sum(
            bool(row["relion_is_complete_oversampled_expansion"]) for row in rows
        ),
        "complete_recovar_oversampled_expansion_count": sum(
            bool(row["recovar_is_complete_oversampled_expansion"]) for row in rows
        ),
        "relion_contributor_count": sum(
            int(row["relion_contributor_count"]) for row in rows
        ),
        "recovar_contributor_count": sum(
            int(row["recovar_contributor_count"]) for row in rows
        ),
        "all_relion_contributors_in_recovar_candidates_count": sum(
            bool(row["all_relion_contributors_in_recovar_candidates"])
            for row in rows
        ),
        "all_recovar_contributors_in_relion_candidates_count": sum(
            bool(row["all_recovar_contributors_in_relion_candidates"])
            for row in rows
        ),
        "coarse_parent_overlap_fraction_per_particle": _quantiles(
            [
                int(row["coarse_parent_overlap_count"])
                / max(int(row["relion_coarse_parent_count"]), 1)
                for row in rows
            ]
        ),
    }


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
    recovar, recovar_validation = _load_recovar_candidates(
        args.geometry_directory,
        args.scripts_directory,
        class_index=class_index,
    )
    _require(set(recovar) == set(categories), "RECOVAR panel identities differ")
    artifact_by_stack = {
        int(artifact.stack_index): artifact for artifact in artifacts
    }
    rows = [
        _particle_report(
            artifact_by_stack[stack],
            recovar[stack],
            categories[stack],
            class_index=class_index,
            coarse_rotation_count=args.coarse_rotation_count,
            oversampling_factor=args.oversampling_factor,
        )
        for stack in sorted(categories)
    ]
    summary = _summarize(rows)
    all_complete = (
        summary["complete_relion_oversampled_expansion_count"] == len(rows)
        and summary["complete_recovar_oversampled_expansion_count"] == len(rows)
    )
    any_parent_difference = summary["coarse_parent_sets_exact_count"] != len(rows)
    classification = (
        "coarse_parent_support_difference_precedes_fine_scoring"
        if all_complete and any_parent_difference
        else "candidate_support_difference_not_localized_to_coarse_parents"
    )
    return {
        "schema": "em_k4_relion_recovar_candidate_support_audit_v1",
        "status": "complete",
        "metric_policy": "exact integer candidate identities; no correlation",
        "classification": classification,
        "gates": {
            "fresh_relion_capture_validation": True,
            "panel_identity_exact": True,
            "recovar_device_signatures_validated": True,
            "complete_oversampled_expansion_for_every_particle": all_complete,
            "coarse_parent_sets_exact_for_every_particle": not any_parent_difference,
        },
        "scope": {
            "iteration": args.iteration,
            "half": args.half,
            "class_one_based": args.class_one_based,
            "particle_count": len(rows),
            "coarse_rotation_count": args.coarse_rotation_count,
            "oversampling_factor": args.oversampling_factor,
            "recovar_current_size": recovar_validation["current_size"],
            "recovar_signature_count": recovar_validation["signature_count"],
            "panel_identity_sha256": panel.get(
                "identity_sha256", panel.get("panel_identity_sha256")
            ),
        },
        "all_particles": summary,
        "by_category": {
            category: _summarize(
                [row for row in rows if row["category"] == category]
            )
            for category in sorted(set(categories.values()))
        },
        "particles": rows,
        "qualification": (
            "RELION orientation_class_key is converted to a class-local coarse "
            "index, then combined with oversampled_rotation. RECOVAR uses its "
            "validated active_global_rotation_indices. Because every particle "
            "has a complete oversampling-factor expansion in both engines, "
            "fine-candidate differences are inherited exactly from coarse-parent "
            "support selection and precede fine scoring and reconstruction."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", required=True, type=Path)
    parser.add_argument("--geometry-directory", required=True, type=Path)
    parser.add_argument("--panel-json", required=True, type=Path)
    parser.add_argument("--scripts-directory", required=True, type=Path)
    parser.add_argument("--class-one-based", required=True, type=int)
    parser.add_argument("--coarse-rotation-count", required=True, type=int)
    parser.add_argument("--oversampling-factor", required=True, type=int)
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
