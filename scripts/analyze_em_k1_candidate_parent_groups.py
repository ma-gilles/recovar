#!/usr/bin/env python3
"""Classify K=1 candidate mismatches as complete oversampled parent groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_k1_bpref_contributor_membership import match_rotations
from scripts.validate_relion_bpref_membership import (
    INVALID_WEIGHT_SENTINEL,
    load_artifact,
)

MEMBERSHIP_SCHEMA = "recovar.em.k1_case22_it2_bpref_membership_cohort.v1"
MEMBERSHIP_SHA256 = (
    "09b4cb69e585d2d0907541e407e386b3fe695d69206331d4714a3e79a46bbdc1"
)
INERTNESS_SCHEMA = "em-k1-membership-capture-inertness-v1"
INERTNESS_SHA256 = (
    "d5fea1c8e795ff5739efaf5db03b502d4d75cc857895471f98daf6609ba93e1a"
)
ROTATION_TOLERANCE = 1.0e-6
EXPECTED_FINE_CHILDREN = tuple(range(8))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_keys(matrices: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(
        np.asarray(matrices, dtype=np.float32).reshape(-1, 9)
    )
    return values.view(np.dtype((np.void, values.dtype.itemsize * 9))).reshape(-1)


def summarize_parent_groups(
    orientation_class_keys: np.ndarray,
    oversampled_rotations: np.ndarray,
) -> list[dict[str, Any]]:
    parents = np.asarray(orientation_class_keys, dtype=np.uint64).reshape(-1)
    children = np.asarray(oversampled_rotations, dtype=np.uint64).reshape(-1)
    _require(parents.shape == children.shape, "parent/child identity shape mismatch")
    groups = []
    for parent in np.unique(parents):
        selected = np.sort(children[parents == parent])
        unique = np.unique(selected)
        groups.append(
            {
                "orientation_class_key": int(parent),
                "oversampled_rotations": [int(value) for value in selected],
                "child_count": int(selected.size),
                "children_unique": bool(unique.size == selected.size),
                "complete_expected_children": bool(
                    np.array_equal(
                        selected,
                        np.asarray(EXPECTED_FINE_CHILDREN, dtype=np.uint64),
                    )
                ),
            }
        )
    return groups


def _load_reports(
    membership_path: Path,
    inertness_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(_sha256(membership_path) == MEMBERSHIP_SHA256, "membership bytes changed")
    _require(_sha256(inertness_path) == INERTNESS_SHA256, "inertness bytes changed")
    membership = json.loads(membership_path.read_text())
    inertness = json.loads(inertness_path.read_text())
    _require(membership.get("schema") == MEMBERSHIP_SCHEMA, "membership schema changed")
    _require(membership.get("status") == "complete", "membership report incomplete")
    _require(
        membership.get("classification") == "candidate_grid_membership_difference",
        "candidate-grid classification changed",
    )
    fixed_metric = membership.get("fixed_metric", {})
    _require(fixed_metric.get("denominator") == 64, "fixed denominator changed")
    _require(
        fixed_metric.get("candidate_sets_exact_count") == 54,
        "candidate mismatch count changed",
    )
    _require(inertness.get("schema") == INERTNESS_SCHEMA, "inertness schema changed")
    _require(
        inertness.get("status") == "pass"
        and inertness.get("capture_inertness_qualified") is True,
        "capture inertness did not qualify",
    )
    return membership, inertness


def _load_recovar_target_rotations(
    membership: dict[str, Any],
    recovar_directory: Path,
    target_stacks: set[int],
) -> tuple[dict[int, np.ndarray], dict[str, str]]:
    descriptors = membership["inputs"]["recovar_contribution_shards"]
    paths = [Path(item["path"]) for item in descriptors]
    expected_hashes = {Path(item["path"]): str(item["sha256"]) for item in descriptors}
    observed_paths = set(recovar_directory.glob("*.npz"))
    _require(set(paths) == observed_paths, "RECOVAR contribution shard set changed")
    target_rotations: dict[int, np.ndarray] = {}
    target_hashes: dict[str, str] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            stacks = np.asarray(archive["stack_indices_1based"], dtype=np.int64)
            wanted = [
                (particle, int(stack))
                for particle, stack in enumerate(stacks)
                if int(stack) in target_stacks
            ]
            if not wanted:
                continue
            _require(
                archive["schema"].item() == "recovar-bpref-contribution-rows-v3",
                f"RECOVAR contribution schema changed: {path}",
            )
            active_particle = np.asarray(
                archive["active_particle_rows"], dtype=np.int64
            )
            active_rotation = np.asarray(
                archive["active_rotation_rows"], dtype=np.int64
            )
            rotations = np.asarray(archive["active_rotations"], dtype=np.float32)
            for particle, stack in wanted:
                _require(stack not in target_rotations, f"duplicate stack {stack}")
                active = np.flatnonzero(active_particle == particle)
                local_rotation = active_rotation[active]
                _require(active.size > 0, f"missing rotations for stack {stack}")
                _require(
                    np.unique(local_rotation).size == local_rotation.size,
                    f"duplicate rotation row for stack {stack}",
                )
                target_rotations[stack] = rotations[active]
        current_hash = _sha256(path)
        _require(
            current_hash == expected_hashes[path],
            f"RECOVAR contribution shard bytes changed: {path}",
        )
        target_hashes[str(path.resolve())] = current_hash
    _require(
        set(target_rotations) == target_stacks,
        "RECOVAR target stack coverage changed",
    )
    return target_rotations, target_hashes


def _relion_paths_by_stack(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(directory.glob("*.bpm-v1.bin")):
        marker = path.name.split("_stack", maxsplit=1)
        _require(len(marker) == 2, f"unexpected RELION artifact name: {path}")
        stack = int(marker[1].split("_", maxsplit=1)[0])
        _require(stack not in result, f"duplicate RELION stack {stack}")
        result[stack] = path
    return result


def analyze(
    *,
    membership_report: Path,
    inertness_report: Path,
    relion_directory: Path,
    recovar_directory: Path,
) -> dict[str, Any]:
    membership, inertness = _load_reports(membership_report, inertness_report)
    mismatch_rows = [
        row
        for row in membership["particles"]
        if not bool(row["candidate_sets_exact"])
    ]
    _require(len(mismatch_rows) == 10, "candidate mismatch denominator changed")
    target_stacks = {
        int(row["stack_index_one_based"]) for row in mismatch_rows
    }
    recovar_rotations, target_shard_hashes = _load_recovar_target_rotations(
        membership,
        recovar_directory,
        target_stacks,
    )
    relion_paths = _relion_paths_by_stack(relion_directory)
    expected_relion_hashes = membership["relion_validation"]["artifact_sha256"]
    _require(
        set(relion_paths) == target_stacks
        | {
            int(row["stack_index_one_based"])
            for row in membership["particles"]
            if bool(row["candidate_sets_exact"])
        },
        "RELION artifact stack coverage changed",
    )

    target_relion_rotations: dict[int, np.ndarray] = {}
    particle_matches: dict[int, Any] = {}
    recovar_only_matrices = []
    for row in mismatch_rows:
        stack = int(row["stack_index_one_based"])
        artifact = load_artifact(relion_paths[stack])
        _require(
            artifact.sha256 == expected_relion_hashes[artifact.path.name],
            f"RELION artifact bytes changed: {artifact.path}",
        )
        candidate = np.any(
            artifact.weights != float(INVALID_WEIGHT_SENTINEL),
            axis=1,
        )
        rotations = artifact.rotations[candidate]
        target_relion_rotations[stack] = rotations
        matches = match_rotations(
            rotations["matrix"].reshape(-1, 3, 3).transpose(0, 2, 1),
            recovar_rotations[stack],
            tolerance=ROTATION_TOLERANCE,
        )
        _require(
            matches.relion_unmatched.size
            == int(row["relion_unmatched_candidate_count"]),
            f"RELION unmatched count changed for stack {stack}",
        )
        _require(
            matches.recovar_unmatched.size
            == int(row["recovar_unmatched_candidate_count"]),
            f"RECOVAR unmatched count changed for stack {stack}",
        )
        particle_matches[stack] = matches
        for matrix in recovar_rotations[stack][matches.recovar_unmatched]:
            recovar_only_matrices.append(matrix)

    recovar_only = np.asarray(recovar_only_matrices, dtype=np.float32).reshape(
        -1, 3, 3
    )
    _require(recovar_only.shape[0] == 16, "RECOVAR-only rotation count changed")
    recovar_only_keys = _matrix_keys(recovar_only)
    _require(
        np.unique(recovar_only_keys).size == recovar_only_keys.size,
        "RECOVAR-only rotations are duplicated",
    )
    resolved: dict[bytes, set[tuple[int, int]]] = {
        bytes(key): set() for key in recovar_only_keys
    }
    all_relion_hashes: dict[str, str] = {}
    for path in relion_paths.values():
        artifact = load_artifact(path)
        _require(
            artifact.sha256 == expected_relion_hashes[artifact.path.name],
            f"RELION artifact bytes changed: {artifact.path}",
        )
        all_relion_hashes[str(artifact.path.resolve())] = artifact.sha256
        matrices = artifact.rotations["matrix"].reshape(-1, 3, 3).transpose(
            0, 2, 1
        )
        keys = _matrix_keys(matrices)
        selected = np.flatnonzero(np.isin(keys, recovar_only_keys))
        for index in selected:
            resolved[bytes(keys[index])].add(
                (
                    int(artifact.rotations["orientation_class_key"][index]),
                    int(artifact.rotations["oversampled_rotation"][index]),
                )
            )
    _require(
        all(len(identities) == 1 for identities in resolved.values()),
        "RECOVAR-only rotation identity did not resolve uniquely",
    )

    per_particle = []
    for source_row in mismatch_rows:
        stack = int(source_row["stack_index_one_based"])
        matches = particle_matches[stack]
        relion_rotations = target_relion_rotations[stack]
        relion_only_rows = relion_rotations[matches.relion_unmatched]
        relion_groups = summarize_parent_groups(
            relion_only_rows["orientation_class_key"],
            relion_only_rows["oversampled_rotation"],
        )
        recovar_keys = _matrix_keys(
            recovar_rotations[stack][matches.recovar_unmatched]
        )
        recovar_identities = [
            next(iter(resolved[bytes(key)])) for key in recovar_keys
        ]
        recovar_groups = summarize_parent_groups(
            np.asarray([value[0] for value in recovar_identities]),
            np.asarray([value[1] for value in recovar_identities]),
        )
        groups = [*relion_groups, *recovar_groups]
        complete = all(bool(group["complete_expected_children"]) for group in groups)
        per_particle.append(
            {
                "stack_index_one_based": stack,
                "original_index_zero_based": int(
                    source_row["original_index_zero_based"]
                ),
                "group": str(source_row["group"]),
                "support_delta": int(source_row["support_delta"]),
                "relion_only_rotation_count": int(
                    matches.relion_unmatched.size
                ),
                "recovar_only_rotation_count": int(
                    matches.recovar_unmatched.size
                ),
                "relion_only_parent_groups": relion_groups,
                "recovar_only_parent_groups": recovar_groups,
                "all_unmatched_rotations_form_complete_parent_groups": complete,
            }
        )

    relion_group_count = sum(
        len(row["relion_only_parent_groups"]) for row in per_particle
    )
    recovar_group_count = sum(
        len(row["recovar_only_parent_groups"]) for row in per_particle
    )
    complete_group_count = sum(
        bool(group["complete_expected_children"])
        for row in per_particle
        for group in (
            row["relion_only_parent_groups"]
            + row["recovar_only_parent_groups"]
        )
    )
    group_count = relion_group_count + recovar_group_count
    all_complete = complete_group_count == group_count
    _require(group_count == 13, "candidate parent-group count changed")
    classification = (
        "candidate_mismatches_are_complete_adaptive_oversampling_parent_groups"
        if all_complete
        else "candidate_mismatches_include_partial_oversampling_parent_groups"
    )
    return {
        "schema": "recovar.em.k1_case22_it2_candidate_parent_groups.v1",
        "status": "complete",
        "classification": classification,
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed 64-particle cohort and 10 candidate-mismatch particles; "
            "exact parent/child identities; map acceptance is inherited from "
            "the hash-pinned six-map FSC/FSC-AUC inertness report; no correlation"
        ),
        "fixed_metric": {
            "cohort_particle_denominator": 64,
            "candidate_mismatch_particle_denominator": 10,
            "candidate_mismatch_particle_count": len(per_particle),
            "relion_only_rotation_count": int(
                sum(row["relion_only_rotation_count"] for row in per_particle)
            ),
            "recovar_only_rotation_count": int(
                sum(row["recovar_only_rotation_count"] for row in per_particle)
            ),
            "relion_only_parent_group_count": relion_group_count,
            "recovar_only_parent_group_count": recovar_group_count,
            "parent_group_count": group_count,
            "complete_parent_group_count": complete_group_count,
            "all_candidate_mismatches_are_complete_parent_groups": all_complete,
            "expected_children_per_parent": list(EXPECTED_FINE_CHILDREN),
        },
        "particles": per_particle,
        "gates": {
            "membership_report_hash_exact": True,
            "inertness_report_hash_exact_and_qualified": True,
            "relion_artifact_hashes_exact": True,
            "target_recovar_shard_hashes_exact": True,
            "rotation_identity_resolution_unique": True,
        },
        "inputs": {
            "membership_report": {
                "path": str(membership_report.resolve()),
                "sha256": MEMBERSHIP_SHA256,
            },
            "inertness_report": {
                "path": str(inertness_report.resolve()),
                "sha256": INERTNESS_SHA256,
                "minimum_fsc_auc_non_dc": min(
                    float(row["fsc_auc_non_dc"])
                    for row in inertness["comparisons"].values()
                ),
            },
            "target_recovar_contribution_shards": target_shard_hashes,
            "relion_artifacts": all_relion_hashes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership-report", type=Path, required=True)
    parser.add_argument("--inertness-report", type=Path, required=True)
    parser.add_argument("--relion-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = analyze(
        membership_report=args.membership_report,
        inertness_report=args.inertness_report,
        relion_directory=args.relion_directory,
        recovar_directory=args.recovar_directory,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "fixed_metric": report["fixed_metric"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
