#!/usr/bin/env python3
"""Audit genuine K=4 half maps from four independent Class3D processes.

RELION rejects ``--split_random_halves`` when ``K > 1``.  The matched
gold-standard construction is therefore two independent K=4 Class3D runs per
engine, one on each frozen particle half, all starting from the same four
references.  A RECOVAR K-class process currently writes its combined numbered
map twice (``half1`` and ``half2``); those replicas are checked but only one is
used.  They are never reported as independent half maps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from recovar.core.mask import make_mask
from recovar.utils import helpers
from scripts.collect_em_k1_science_diagnostics import (
    ShellFscCalculator,
    apply_proper_rigid_transform,
    fit_proper_rigid_transform,
)

SCHEMA = "recovar.em_real_kclass_independent_halfmap_audit.v2"
MANIFEST_SCHEMA = "recovar.em_real_kclass_independent_halfmap_submission.v2"
ANALYSIS_POLICY_SCHEMA = "recovar.em_real_kclass_halfmap_analysis_policy.v1"
N_CLASSES = 4
EXPECTED_THRESHOLDS = {
    "fsc_threshold": 1.0 / 7.0,
    "crossing_consecutive_shells": 3,
    "masked_resolution_ratio_max": 1.05,
    "masked_resolution_shell_lag_max": 1,
    "half_band_auc_drop_max": 0.01,
    "cross_merged_band_auc_min": 0.99,
    "cross_each_half_band_auc_min": 0.90,
    "assignment_agreement_min": 0.99,
    "minimum_class_count": 1,
    # The assignment objective is the sum of four normalized FSC-AUC values.
    # Requiring both margins rejects numerical tie-breaking and map sets whose
    # class identities are not scientifically distinguishable.
    "permutation_objective_margin_abs_min": 0.01,
    "permutation_objective_margin_rel_min": 0.0025,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RELION_MAP_RE = re.compile(r"^run_it(?P<iteration>\d{3})_class(?P<class_id>\d{3})\.mrc$")
RECOVAR_MAP_RE = re.compile(
    r"^it(?P<iteration>\d{3})_half(?P<replica>[12])_class(?P<class_id>\d+)_reg\.mrc$"
)


class AuditError(RuntimeError):
    """Raised when an input cannot support a half-map claim."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_ints(values: Sequence[int]) -> str:
    return sha256_strings([str(int(value)) for value in values])


def expected_analysis_policy(grid_size: int) -> dict[str, Any]:
    """Return the only accepted, manifest-frozen analysis policy."""

    _require(grid_size // 2 - 1 >= 32, "analysis policy requires Fourier shell 32")
    return {
        "schema": ANALYSIS_POLICY_SCHEMA,
        "alignment": {
            "fit_max_shell": 32,
            "coarse_healpix_order": 1,
            "refine_healpix_orders": [2],
            "interpolation_order": 1,
        },
        "fsc": {
            "crossing_consecutive_shells": 3,
            "phase_randomization_corrected": False,
            "absolute_resolution_claim": False,
        },
        "common_mask": {
            "construction": "nonnegative_voxelwise_rms_envelope",
            "threshold": "auto",
            "lowpass_sigma": max(2, int(math.ceil(grid_size / 128))),
            "extend": max(1, int(math.ceil(grid_size / 32))),
            "soft_edge": max(1, int(math.ceil(grid_size / 32))),
            "cleanup": True,
        },
    }


def validate_analysis_policy(
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Reject post-hoc analysis choices and bind the CLI to the manifest."""

    expected = expected_analysis_policy(int(manifest["config"]["grid_size"]))
    _require(manifest.get("analysis_policy") == expected, "manifest analysis_policy changed")
    observed = {
        "schema": ANALYSIS_POLICY_SCHEMA,
        "alignment": {
            "fit_max_shell": int(args.fit_max_shell),
            "coarse_healpix_order": int(args.coarse_healpix_order),
            "refine_healpix_orders": [int(value) for value in args.refine_healpix_order],
            "interpolation_order": int(args.interpolation_order),
        },
        "fsc": {
            "crossing_consecutive_shells": int(args.crossing_consecutive_shells),
            "phase_randomization_corrected": (
                str(args.phase_randomization_corrected).lower() == "true"
            ),
            "absolute_resolution_claim": str(args.absolute_resolution_claim).lower() == "true",
        },
        "common_mask": {
            "construction": "nonnegative_voxelwise_rms_envelope",
            "threshold": str(args.mask_threshold),
            "lowpass_sigma": int(args.mask_lowpass_sigma),
            "extend": int(args.mask_extend),
            "soft_edge": int(args.mask_soft_edge),
            "cleanup": str(args.mask_cleanup).lower() == "true",
        },
    }
    _require(observed == expected, "audit CLI analysis policy differs from frozen manifest policy")
    return expected


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _particle_table(path: Path):
    import starfile

    payload = starfile.read(path)
    if isinstance(payload, dict):
        for key in ("particles", "data_particles"):
            if key in payload:
                return payload[key]
        tables = [table for table in payload.values() if hasattr(table, "columns")]
        _require(len(tables) == 1, f"cannot identify particle table in {path}")
        return tables[0]
    return payload


def _column(table, name: str) -> np.ndarray:
    for candidate in (name, f"_{name}"):
        if candidate in table.columns:
            return np.asarray(table[candidate])
    raise AuditError(f"particle table is missing {name}")


def _image_stack_identity(value: str) -> tuple[int, Path]:
    fields = str(value).split("@", 1)
    _require(
        len(fields) == 2 and fields[0].isdigit() and fields[1],
        f"invalid RELION image identity: {value}",
    )
    one_based = int(fields[0])
    _require(one_based >= 1, f"RELION image identity is not one-based: {value}")
    return one_based - 1, Path(fields[1])


def _image_stack_index(value: str) -> int:
    return _image_stack_identity(value)[0]


def validate_particle_split(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate generated half STARs against immutable identities and labels."""

    selection = manifest["particle_selection"]
    selected = [str(value) for value in selection["selected_image_names"]]
    _require(len(selected) == len(set(selected)), "selected particle identities are not unique")
    _require(
        sha256_strings(selected) == selection["ordered_image_names_sha256"],
        "selected particle-order hash mismatch",
    )
    particle_stack_path = Path(selection["particle_stack_path"])
    _require(particle_stack_path.is_absolute(), "runtime particle-stack path is not absolute")
    particle_stack_path = particle_stack_path.resolve()
    _require(particle_stack_path.is_file(), f"missing runtime particle stack: {particle_stack_path}")
    _require(
        SHA256_RE.fullmatch(str(selection["particle_stack_sha256"])) is not None,
        "runtime particle-stack SHA-256 is invalid",
    )
    stack_records = [
        row
        for row in manifest.get("input_artifacts", [])
        if str(row.get("role", "")).startswith("particle_stack_grid")
    ]
    _require(len(stack_records) == 1, "manifest must seal exactly one runtime particle stack")
    stack_record = stack_records[0]
    _require(
        Path(stack_record["path"]).resolve() == particle_stack_path,
        "runtime particle-stack path differs from the sealed input artifact",
    )
    _require(
        str(stack_record["sha256"]) == str(selection["particle_stack_sha256"]),
        "runtime particle-stack SHA-256 differs from the sealed input artifact",
    )
    selected_stack_paths = [path for _index, path in map(_image_stack_identity, selected)]
    _require(
        all(path.is_absolute() and path.resolve() == particle_stack_path for path in selected_stack_paths),
        "selected identities do not reference the sealed absolute particle stack",
    )

    origin_star = Path(selection["origin_particles_star"])
    source_indices_path = Path(selection["source_indices_npy"])
    _require(origin_star.is_file(), f"missing immutable origin particle STAR: {origin_star}")
    _require(source_indices_path.is_file(), f"missing immutable source indices: {source_indices_path}")
    _require(
        sha256_file(origin_star) == selection["origin_particles_star_sha256"],
        "immutable origin particle STAR hash mismatch",
    )
    _require(
        sha256_file(source_indices_path) == selection["source_indices_sha256"],
        "immutable source-index hash mismatch",
    )
    origin_table = _particle_table(origin_star)
    origin_names = [str(value) for value in _column(origin_table, "rlnImageName")]
    origin_subsets = np.asarray(_column(origin_table, "rlnRandomSubset"), dtype=np.int64)
    source_indices_raw = np.load(source_indices_path, allow_pickle=False)
    _require(
        source_indices_raw.ndim == 1 and np.issubdtype(source_indices_raw.dtype, np.integer),
        "immutable source indices must be a one-dimensional integer array",
    )
    source_indices = np.asarray(source_indices_raw, dtype=np.int64)
    _require(len(origin_names) == source_indices.size, "origin STAR/source-index lengths differ")
    _require(len(origin_names) == len(set(origin_names)), "origin particle identities are not unique")
    _require(len(set(source_indices.tolist())) == source_indices.size, "immutable source indices are not unique")
    _require(np.all(source_indices >= 0), "immutable source indices contain negative values")
    _require(np.all(np.isin(origin_subsets, (1, 2))), "origin STAR has invalid random-subset labels")
    origin_stack_indices = [_image_stack_index(name) for name in origin_names]
    _require(
        len(origin_stack_indices) == len(set(origin_stack_indices)),
        "origin STAR image-stack indices are not unique",
    )
    _require(
        origin_stack_indices == source_indices.tolist(),
        "origin STAR image-stack indices differ from immutable source-index order",
    )
    origin_position = {image_index: position for position, image_index in enumerate(origin_stack_indices)}

    selected_stack_indices = [_image_stack_index(name) for name in selected]
    _require(
        len(selected_stack_indices) == len(set(selected_stack_indices)),
        "selected image-stack indices are not unique",
    )
    _require(
        set(selected_stack_indices).issubset(origin_position),
        "selected identities are absent from immutable origin STAR",
    )
    selected_positions = [origin_position[index] for index in selected_stack_indices]
    _require(selected_positions == sorted(selected_positions), "selected identities changed immutable origin order")
    selection_mode = str(selection["mode"])
    if selection_mode == "full10k":
        _require(
            selected_positions == list(range(len(origin_names))),
            "full10k selection does not contain every immutable origin row",
        )
    elif selection_mode == "shared200":
        selection_path = Path(selection["selection_source_json"])
        _require(selection_path.is_file(), f"missing immutable selection JSON: {selection_path}")
        _require(
            sha256_file(selection_path) == selection["selection_source_sha256"],
            "immutable selection JSON hash mismatch",
        )
        payload = json.loads(selection_path.read_text())
        _require(payload.get("same_visited_particle_ids") is True, "shared200 selection was not admitted")
        requested = {_image_stack_index(str(value)) for value in payload["visited_particle_ids"]}
        expected_selected = [value for value in origin_stack_indices if value in requested]
        _require(len(requested) == len(expected_selected), "shared200 identities are absent from origin STAR")
        _require(
            selected_stack_indices == expected_selected,
            "selected identities differ from immutable shared200 selection",
        )
    else:
        raise AuditError(f"unknown particle-selection mode: {selection_mode}")
    expected_selected_source_indices = source_indices[selected_positions].tolist()
    declared_selected_source_indices = [int(value) for value in selection["selected_source_indices"]]
    _require(
        declared_selected_source_indices == expected_selected_source_indices,
        "selected source-index sequence changed",
    )
    _require(
        sha256_ints(declared_selected_source_indices) == selection["ordered_source_indices_sha256"],
        "selected source-index order hash mismatch",
    )
    expected_subsets = origin_subsets[selected_positions]

    source_star = Path(selection["source_particles_star"])
    _require(source_star.is_file(), f"missing generated selected particle STAR: {source_star}")
    source_table = _particle_table(source_star)
    source_names = [str(value) for value in _column(source_table, "rlnImageName")]
    source_subsets = np.asarray(_column(source_table, "rlnRandomSubset"), dtype=np.int64)
    _require(source_names == selected, "generated selected STAR changed selected order or identities")
    _require(
        [_image_stack_index(name) for name in source_names] == selected_stack_indices,
        "generated selected STAR changed image-stack indices",
    )
    _require(
        np.array_equal(source_subsets, expected_subsets),
        "generated selected STAR changed immutable random-subset labels",
    )

    halves: list[list[str]] = []
    for expected_half, row in enumerate(manifest["halves"], start=1):
        _require(int(row["half"]) == expected_half, "manifest halves are not ordered 1,2")
        star_path = Path(row["particles_star"])
        _require(star_path.is_file(), f"missing half-{expected_half} particle STAR: {star_path}")
        table = _particle_table(star_path)
        names = [str(value) for value in _column(table, "rlnImageName")]
        subsets = np.asarray(_column(table, "rlnRandomSubset"), dtype=np.int64)
        _require(len(names) == int(row["particle_count"]), f"half-{expected_half} count changed")
        _require(len(names) == len(set(names)), f"half-{expected_half} identities are not unique")
        _require(
            sha256_strings(names) == row["ordered_image_names_sha256"],
            f"half-{expected_half} particle-order hash mismatch",
        )
        _require(
            all(
                stack_path.is_absolute() and stack_path.resolve() == particle_stack_path
                for _index, stack_path in map(_image_stack_identity, names)
            ),
            f"half-{expected_half} identities do not reference the sealed absolute particle stack",
        )
        _require(
            np.array_equal(subsets, np.full(len(names), expected_half, dtype=np.int64)),
            f"half-{expected_half} STAR has wrong rlnRandomSubset labels",
        )
        expected_names = [
            name
            for name, immutable_subset in zip(selected, expected_subsets, strict=True)
            if int(immutable_subset) == expected_half
        ]
        _require(
            names == expected_names,
            f"half-{expected_half} STAR does not preserve immutable selected-source order and labels",
        )
        halves.append(names)
    _require(set(halves[0]).isdisjoint(halves[1]), "particle halves overlap")
    combined = [name for name in selected if name in set(halves[0]) or name in set(halves[1])]
    _require(len(combined) == len(selected), "particle halves do not cover the selected identities")
    _require(set(halves[0]) | set(halves[1]) == set(selected), "particle-half union changed")
    return {
        "selected_count": len(selected),
        "half_counts": [len(values) for values in halves],
        "selected_order_sha256": sha256_strings(selected),
        "half_order_sha256": [sha256_strings(values) for values in halves],
        "disjoint": True,
        "complete_union": True,
        "generated_selected_particles_star": str(source_star.resolve()),
        "origin_particles_star": str(origin_star.resolve()),
        "origin_particles_star_sha256": sha256_file(origin_star),
        "source_indices_npy": str(source_indices_path.resolve()),
        "source_indices_sha256": sha256_file(source_indices_path),
        "selected_source_indices_sha256": sha256_ints(declared_selected_source_indices),
        "particle_stack_path": str(particle_stack_path),
        "particle_stack_sha256": str(selection["particle_stack_sha256"]),
        "particle_stack_binding": "absolute rlnImageName path in every selected and half STAR",
        "selection_mode": selection_mode,
        "random_subset_label_source": "immutable origin particle STAR",
    }


def validate_input_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate sealed inputs, directly for small files and via the job log for all files."""

    records = manifest.get("input_artifacts")
    _require(isinstance(records, list) and records, "input_artifacts must be a non-empty list")
    seen: set[Path] = set()
    expected_log_lines: set[str] = set()
    checked_directly: list[dict[str, Any]] = []
    for row in records:
        path = Path(row["path"]).resolve()
        expected_hash = str(row["sha256"])
        _require(path not in seen, f"duplicate input artifact declaration: {path}")
        seen.add(path)
        _require(SHA256_RE.fullmatch(expected_hash) is not None, f"invalid SHA-256 for {path}")
        _require(path.is_file(), f"missing sealed input artifact: {path}")
        _require(path.stat().st_size == int(row["size_bytes"]), f"input size changed: {path}")
        expected_log_lines.add(f"{path}: OK")
        if path.stat().st_size <= int(manifest["provenance"]["direct_rehash_max_bytes"]):
            observed = sha256_file(path)
            _require(observed == expected_hash, f"input SHA-256 changed: {path}")
            checked_directly.append({"path": str(path), "sha256": observed})

    sha_manifest = Path(manifest["provenance"]["input_sha256_manifest"])
    _require(sha_manifest.is_file(), f"missing input SHA-256 manifest: {sha_manifest}")
    declared_lines = {
        f"{row['sha256']}  {Path(row['path']).resolve()}" for row in records
    }
    observed_lines = {line.rstrip() for line in sha_manifest.read_text().splitlines() if line.strip()}
    _require(observed_lines == declared_lines, "input SHA-256 manifest does not match submission manifest")

    check_log = Path(manifest["provenance"]["input_sha256_check_log"])
    _require(check_log.is_file(), f"missing in-job SHA-256 verification log: {check_log}")
    log_lines = {line.strip() for line in check_log.read_text(errors="replace").splitlines() if line.strip()}
    missing = sorted(expected_log_lines - log_lines)
    _require(not missing, f"in-job SHA-256 verification is incomplete: {missing}")
    _require(not any("FAILED" in line for line in log_lines), "in-job SHA-256 verification contains failures")
    return {
        "entry_count": len(records),
        "directly_rehashed_count": len(checked_directly),
        "directly_rehashed": checked_directly,
        "all_entries_verified_at_job_start": True,
        "verification_log": str(check_log.resolve()),
        "verification_log_sha256": sha256_file(check_log),
        "sha256_manifest": str(sha_manifest.resolve()),
        "sha256_manifest_sha256": sha256_file(sha_manifest),
    }


def validate_launcher_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label in ("setup", "run"):
        path = Path(manifest["provenance"][f"{label}_script"])
        _require(path.is_file(), f"missing {label} Slurm script: {path}")
        observed = sha256_file(path)
        _require(
            observed == manifest["provenance"][f"{label}_script_sha256"],
            f"{label} Slurm script changed",
        )
        text = path.read_text(errors="replace")
        _require("#SBATCH --exclusive" not in text, f"{label} Slurm script requests --exclusive")
        rows.append({"role": label, "path": str(path.resolve()), "sha256": observed})
    return {"scripts": rows, "nonexclusive": True}


def validate_commands(manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for half in manifest["halves"]:
        half_id = int(half["half"])
        for engine in ("relion", "recovar"):
            command_path = Path(half[f"{engine}_command_path"])
            _require(command_path.is_file(), f"missing {engine} half-{half_id} command record")
            observed = json.loads(command_path.read_text())
            expected = half[f"{engine}_command"]
            _require(observed == expected, f"{engine} half-{half_id} command changed")
            rows.append(
                {
                    "half": half_id,
                    "engine": engine,
                    "path": str(command_path.resolve()),
                    "sha256": sha256_file(command_path),
                    "argv": observed,
                }
            )
    return {"exact_match": True, "commands": rows}


def _validate_slurm_allocation_record(path: Path, *, role: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {role} Slurm allocation record: {path}")
    row = json.loads(path.read_text())
    _require(row.get("under_slurm") is True, f"{role} was not executed under Slurm")
    _require(row.get("ReqTRES") == row.get("AllocTRES"), f"{role} Slurm ReqTRES != AllocTRES")
    _require(int(row.get("requested_gpus", -1)) == 1, f"{role} did not request exactly one GPU")
    _require(int(row.get("allocated_gpus", -1)) == 1, f"{role} did not allocate exactly one GPU")
    _require(row.get("OverSubscribe") == "OK", f"{role} was exclusive")
    _require(str(row.get("job_id", "")), f"{role} Slurm job ID is missing")
    return {**row, "path": str(path.resolve()), "sha256": sha256_file(path), "valid": True}


def validate_slurm_allocation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    provenance = manifest["provenance"]
    return {
        "setup": _validate_slurm_allocation_record(
            Path(provenance["setup_slurm_allocation_json"]), role="setup job"
        ),
        "qualification": _validate_slurm_allocation_record(
            Path(provenance["slurm_allocation_json"]), role="qualification job"
        ),
        "both_valid": True,
    }


def _latest_relion_maps(directory: Path, expected_iteration: int) -> list[Path]:
    grouped: dict[int, dict[int, Path]] = {}
    for path in directory.glob("run_it*_class*.mrc"):
        match = RELION_MAP_RE.fullmatch(path.name)
        if match is None:
            continue
        iteration = int(match.group("iteration"))
        class_id = int(match.group("class_id"))
        grouped.setdefault(iteration, {})[class_id] = path
    _require(grouped, f"no numbered RELION Class3D maps found in {directory}")
    _require(max(grouped) == expected_iteration, f"RELION stopped at iteration {max(grouped)}, expected {expected_iteration}")
    paths = grouped[expected_iteration]
    _require(set(paths) == set(range(1, N_CLASSES + 1)), "RELION final numbered class topology is incomplete")
    return [paths[class_id] for class_id in range(1, N_CLASSES + 1)]


def _latest_recovar_combined_maps(
    directory: Path,
    expected_iteration: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Return one representative per class after proving internal duplication."""

    grouped: dict[tuple[int, int, int], Path] = {}
    for path in directory.glob("it*_half*_class*_reg.mrc"):
        match = RECOVAR_MAP_RE.fullmatch(path.name)
        if match is None:
            continue
        key = tuple(int(match.group(name)) for name in ("iteration", "replica", "class_id"))
        _require(key not in grouped, f"duplicate RECOVAR map identity {key}")
        grouped[key] = path
    _require(grouped, f"no numbered RECOVAR K-class maps found in {directory}")
    observed_iterations = sorted({key[0] for key in grouped})
    _require(
        observed_iterations[-1] == expected_iteration,
        f"RECOVAR stopped at iteration {observed_iterations[-1]}, expected {expected_iteration}",
    )
    expected_final_keys = {
        (expected_iteration, replica, class_id)
        for replica in (1, 2)
        for class_id in range(1, N_CLASSES + 1)
    }
    observed_final_keys = {key for key in grouped if key[0] == expected_iteration}
    _require(
        observed_final_keys == expected_final_keys,
        "RECOVAR final numbered class topology has missing or extra class IDs",
    )
    representatives: list[Path] = []
    duplicates: list[dict[str, Any]] = []
    for class_id in range(1, N_CLASSES + 1):
        first = grouped.get((expected_iteration, 1, class_id))
        second = grouped.get((expected_iteration, 2, class_id))
        _require(first is not None and second is not None, f"RECOVAR class {class_id} replica topology is incomplete")
        first_hash = sha256_file(first)
        second_hash = sha256_file(second)
        _require(
            first_hash == second_hash,
            "RECOVAR K-class numbered half labels are not proven combined-map replicas; "
            f"refusing ambiguous class {class_id} products",
        )
        representatives.append(first)
        duplicates.append(
            {
                "class": class_id,
                "representative": str(first.resolve()),
                "discarded_replica": str(second.resolve()),
                "sha256": first_hash,
                "byte_identical": True,
                "semantic_role": "combined_Class3D_map_replica_not_independent_halfmap",
            }
        )
    return representatives, duplicates


def _reject_cross_process_duplicates(paths_by_half: Sequence[Sequence[Path]], *, engine: str) -> None:
    hashes = [{sha256_file(path) for path in paths} for paths in paths_by_half]
    overlap = sorted(hashes[0] & hashes[1])
    _require(
        not overlap,
        f"{engine} selected half processes share byte-identical maps; refusing false independent-half claim: {overlap}",
    )


def _reject_within_process_duplicates(paths: Sequence[Path], *, engine: str, half: int) -> None:
    by_hash: dict[str, list[str]] = {}
    for path in paths:
        by_hash.setdefault(sha256_file(path), []).append(path.name)
    duplicates = {digest: names for digest, names in by_hash.items() if len(names) > 1}
    _require(
        not duplicates,
        f"{engine} half {half} contains byte-identical class maps: {duplicates}",
    )


def _load_maps(paths: Iterable[Path], *, frame: str) -> tuple[list[np.ndarray], float]:
    volumes: list[np.ndarray] = []
    voxel_sizes: list[float] = []
    for path in paths:
        if frame == "recovar":
            volume, voxel = helpers.load_mrc(str(path), return_voxel_size=True)
        elif frame == "relion":
            volume, voxel = helpers.load_relion_volume(str(path), return_voxel_size=True)
        else:
            raise AuditError(f"unknown frame {frame}")
        array = np.asarray(volume, dtype=np.float32)
        _require(array.ndim == 3 and len(set(array.shape)) == 1, f"non-cubic map: {path} {array.shape}")
        _require(np.all(np.isfinite(array)), f"non-finite map: {path}")
        voxel_array = np.asarray(voxel)
        if voxel_array.dtype.names:
            values = [float(voxel_array[name]) for name in ("x", "y", "z") if name in voxel_array.dtype.names]
        else:
            values = [float(value) for value in voxel_array.reshape(-1)]
        _require(values and max(values) - min(values) <= 1.0e-5, f"invalid/anisotropic voxel size: {path}")
        volumes.append(array)
        voxel_sizes.append(values[0])
    _require(len({volume.shape for volume in volumes}) == 1, "map shapes differ")
    _require(max(voxel_sizes) - min(voxel_sizes) <= 1.0e-5, "map voxel sizes differ")
    return volumes, voxel_sizes[0]


def _unit_rms(volume: np.ndarray) -> np.ndarray:
    centered = np.asarray(volume, dtype=np.float32) - np.float32(np.mean(volume, dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.asarray(centered, dtype=np.float64) ** 2)))
    _require(rms > 0.0 and math.isfinite(rms), "map has zero or invalid RMS")
    return centered / np.float32(rms)


def _ensemble(maps: Sequence[np.ndarray]) -> np.ndarray:
    _require(len(maps) == N_CLASSES, f"expected {N_CLASSES} class maps")
    return np.mean(np.stack([_unit_rms(volume) for volume in maps], axis=0), axis=0, dtype=np.float32)


def _normalized_auc(curve: np.ndarray) -> float:
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    _require(values.size >= 3 and np.all(np.isfinite(values[1:])), "FSC curve has invalid non-DC shells")
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(values[1:]) / max(1, values.size - 2))


def _crossing_resolution(
    curve: np.ndarray,
    *,
    threshold: float,
    consecutive: int,
    box_size: int,
    voxel_size: float,
) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    crossing = None
    for shell in range(1, values.size - consecutive + 1):
        window = values[shell : shell + consecutive]
        if np.all(np.isfinite(window)) and np.all(window < threshold):
            crossing = shell
            break
    return {
        "threshold": float(threshold),
        "consecutive_shells": int(consecutive),
        "crossing_shell": crossing,
        "resolution_angstrom": None if crossing is None else float(box_size * voxel_size / crossing),
        "beyond_measured_range": crossing is None,
    }


def _pairwise_auc(calculator: ShellFscCalculator, lhs: Sequence[np.ndarray], rhs: Sequence[np.ndarray]) -> np.ndarray:
    scores = np.empty((N_CLASSES, N_CLASSES), dtype=np.float64)
    lhs_ft = [calculator.fourier(volume) for volume in lhs]
    rhs_ft = [calculator.fourier(volume) for volume in rhs]
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            scores[i, j] = _normalized_auc(calculator.curve_from_fourier(lhs_ft[i], rhs_ft[j]))
    return scores


def _hungarian_to_anchor(
    scores: np.ndarray,
    *,
    label: str,
    min_absolute_margin: float,
    min_relative_margin: float,
) -> tuple[list[int], dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    _require(values.shape == (N_CLASSES, N_CLASSES) and np.all(np.isfinite(values)), f"invalid {label} matrix")
    source_rows, anchor_cols = linear_sum_assignment(-values)
    _require(np.array_equal(np.sort(anchor_cols), np.arange(N_CLASSES)), f"{label} does not cover anchors")
    scipy_source_for_anchor = np.empty(N_CLASSES, dtype=np.int64)
    for source, anchor in zip(source_rows, anchor_cols, strict=True):
        scipy_source_for_anchor[anchor] = source

    candidates: list[tuple[float, tuple[int, ...]]] = []
    anchors = np.arange(N_CLASSES, dtype=np.int64)
    for source_for_anchor in itertools.permutations(range(N_CLASSES)):
        objective = float(np.sum(values[np.asarray(source_for_anchor, dtype=np.int64), anchors]))
        candidates.append((objective, source_for_anchor))
    best_objective = max(objective for objective, _ in candidates)
    exact_winners = [permutation for objective, permutation in candidates if objective == best_objective]
    _require(
        len(exact_winners) == 1,
        f"{label} class assignment optimum is not unique: {len(exact_winners)} exact optima",
    )
    winner = exact_winners[0]
    _require(
        np.array_equal(scipy_source_for_anchor, np.asarray(winner, dtype=np.int64)),
        f"{label} Hungarian assignment disagrees with exhaustive unique optimum",
    )
    second_objective = max(
        objective for objective, permutation in candidates if permutation != winner
    )
    margin = best_objective - second_objective
    objective_scale = max(abs(best_objective), abs(second_objective), np.finfo(np.float64).eps)
    relative_margin = margin / objective_scale
    _require(
        margin >= float(min_absolute_margin),
        f"{label} class assignment absolute objective margin {margin:.9g} is below "
        f"the frozen minimum {float(min_absolute_margin):.9g}",
    )
    _require(
        relative_margin >= float(min_relative_margin),
        f"{label} class assignment relative objective margin {relative_margin:.9g} is below "
        f"the frozen minimum {float(min_relative_margin):.9g}",
    )
    return list(winner), {
        "objective": best_objective,
        "second_best_objective": second_objective,
        "objective_margin": margin,
        "relative_objective_margin": relative_margin,
        "minimum_absolute_objective_margin": float(min_absolute_margin),
        "minimum_relative_objective_margin": float(min_relative_margin),
        "exact_optimum_count": 1,
        "permutations_exhaustively_checked": math.factorial(N_CLASSES),
    }


def _align_set_to_anchor(
    maps: Sequence[np.ndarray],
    anchor: Sequence[np.ndarray],
    *,
    policy: Mapping[str, Any],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    fit = fit_proper_rigid_transform(
        _ensemble(maps),
        _ensemble(anchor),
        fit_max_shell=int(policy["fit_max_shell"]),
        coarse_healpix_order=int(policy["coarse_healpix_order"]),
        refine_healpix_orders=tuple(int(value) for value in policy["refine_healpix_orders"]),
        interpolation_order=int(policy["interpolation_order"]),
    )
    rotation = np.asarray(fit["rotation_matrix_recovar_to_relion"], dtype=np.float64)
    translation = np.asarray(fit["translation_recovar_to_relion_zyx"], dtype=np.float64)
    aligned = [
        apply_proper_rigid_transform(
            volume,
            rotation,
            translation,
            interpolation_order=int(policy["interpolation_order"]),
        )
        for volume in maps
    ]
    return aligned, _jsonable(fit)


def _common_mask(
    all_sets: Sequence[Sequence[np.ndarray]],
    *,
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = np.stack([_unit_rms(volume) for maps in all_sets for volume in maps], axis=0)
    # A signed mean can cancel when heterogeneous classes contain opposite
    # contrast excursions.  The RMS envelope is engine/half/class symmetric,
    # nonnegative, and cannot erase support through sign cancellation.
    envelope = np.sqrt(np.mean(np.asarray(normalized, dtype=np.float64) ** 2, axis=0)).astype(np.float32)
    params = {
        "threshold": str(policy["threshold"]),
        "lowpass_sigma": int(policy["lowpass_sigma"]),
        "extend": int(policy["extend"]),
        "soft_edge": int(policy["soft_edge"]),
        "cleanup": bool(policy["cleanup"]),
    }
    mask = np.asarray(make_mask(envelope, **params), dtype=np.float32)
    _require(np.all(np.isfinite(mask)), "common mask contains non-finite values")
    _require(np.any(mask > 0.5) and np.any(mask < 0.5), "common mask is empty or all ones")
    return mask, {
        **params,
        "construction": (
            "voxelwise RMS envelope of equal-weight unit-RMS maps from all 16 aligned "
            "engine/half/class products"
        ),
        "nonnegative_envelope": True,
        "sign_cancellation_possible": False,
        "engine_half_class_symmetric": True,
        "support_fraction_gt_0p5": float(np.mean(mask > 0.5)),
    }


def _curve_metric(
    calculator: ShellFscCalculator,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    key: str,
    curves: dict[str, np.ndarray],
    voxel_size: float,
    consecutive: int,
) -> dict[str, Any]:
    curve = calculator.curve_from_fourier(calculator.fourier(lhs), calculator.fourier(rhs))
    curves[key] = curve
    return {
        "shellwise_key": key,
        "fsc_auc": _normalized_auc(curve),
        "resolution_0p143": _crossing_resolution(
            curve,
            threshold=1.0 / 7.0,
            consecutive=consecutive,
            box_size=calculator.box_size,
            voxel_size=voxel_size,
        ),
        "resolution_0p5": _crossing_resolution(
            curve,
            threshold=0.5,
            consecutive=consecutive,
            box_size=calculator.box_size,
            voxel_size=voxel_size,
        ),
    }


def _first_sustained_crossing(curve: np.ndarray, threshold: float, consecutive: int) -> int | None:
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    for shell in range(1, values.size - consecutive + 1):
        if np.all(np.isfinite(values[shell : shell + consecutive])) and np.all(
            values[shell : shell + consecutive] < threshold
        ):
            return shell
    return None


def _band_auc(curve: np.ndarray, last_shell: int) -> float:
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    _require(2 <= last_shell < values.size, "FSC comparison band is too short")
    band = values[1 : last_shell + 1]
    _require(np.all(np.isfinite(band)), "FSC band contains non-finite values")
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(band) / max(1, band.size - 1))


def _science_metrics_for_class(
    class_id: int,
    curves: Mapping[str, np.ndarray],
    *,
    box_size: int,
    voxel_size: float,
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    metrics: dict[str, Any] = {}
    failures: list[str] = []
    threshold = float(thresholds["fsc_threshold"])
    consecutive = int(thresholds["crossing_consecutive_shells"])
    prefix = f"class{class_id:03d}_"
    unmasked_relion_curve = curves[prefix + "relion_halfmap_unmasked"]
    unmasked_relion_crossing = _first_sustained_crossing(
        unmasked_relion_curve,
        threshold,
        consecutive,
    )
    unmasked_relion_effective_crossing = (
        len(unmasked_relion_curve)
        if unmasked_relion_crossing is None
        else unmasked_relion_crossing
    )
    comparison_last_shell = unmasked_relion_effective_crossing - 1
    _require(
        comparison_last_shell >= 2,
        f"class {class_id} frozen RELION unmasked comparison band is too short",
    )
    for route in ("unmasked", "common_masked"):
        relion_curve = curves[prefix + f"relion_halfmap_{route}"]
        recovar_curve = curves[prefix + f"recovar_halfmap_{route}"]
        relion_crossing = _first_sustained_crossing(relion_curve, threshold, consecutive)
        recovar_crossing = _first_sustained_crossing(recovar_curve, threshold, consecutive)
        # The unmasked RELION half map freezes one comparison band for both
        # routes.  Neither RECOVAR's earlier crossing nor masking may shorten
        # or extend the range over which an acceptance AUC is integrated.
        last_shell = comparison_last_shell
        relion_auc = _band_auc(relion_curve, last_shell)
        recovar_auc = _band_auc(recovar_curve, last_shell)
        selection = slice(1, last_shell + 1)
        delta = np.asarray(recovar_curve[selection] - relion_curve[selection], dtype=np.float64)
        route_metrics = {
            "comparison_band_policy": "frozen RELION unmasked-resolved non-DC shells",
            "relion_unmasked_resolved_last_shell": int(last_shell),
            "relion_crossing_shell": None if relion_crossing is None else int(relion_crossing),
            "recovar_crossing_shell": None if recovar_crossing is None else int(recovar_crossing),
            "relion_crossing_beyond_measured_range": relion_crossing is None,
            "recovar_crossing_beyond_measured_range": recovar_crossing is None,
            "relion_resolution_angstrom": (
                None if relion_crossing is None else float(box_size * voxel_size / relion_crossing)
            ),
            "recovar_resolution_angstrom": (
                None if recovar_crossing is None else float(box_size * voxel_size / recovar_crossing)
            ),
            "relion_resolution_better_than_angstrom": (
                None
                if relion_crossing is not None
                else float(box_size * voxel_size / (len(relion_curve) - 1))
            ),
            "recovar_resolution_better_than_angstrom": (
                None
                if recovar_crossing is not None
                else float(box_size * voxel_size / (len(recovar_curve) - 1))
            ),
            "relion_halfmap_band_fsc_auc": relion_auc,
            "recovar_halfmap_band_fsc_auc": recovar_auc,
            "recovar_minus_relion_halfmap_band_fsc_auc": recovar_auc - relion_auc,
            "halfmap_curve_rmse": float(np.sqrt(np.mean(delta * delta))),
            "halfmap_curve_p95_abs_delta": float(np.quantile(np.abs(delta), 0.95)),
            "cross_merged_band_fsc_auc": _band_auc(
                curves[prefix + f"cross_merged_{route}"], last_shell
            ),
            "cross_half1_band_fsc_auc": _band_auc(
                curves[prefix + f"cross_half1_{route}"], last_shell
            ),
            "cross_half2_band_fsc_auc": _band_auc(
                curves[prefix + f"cross_half2_{route}"], last_shell
            ),
        }
        metrics[route] = route_metrics
        if recovar_auc < relion_auc - float(thresholds["half_band_auc_drop_max"]):
            failures.append(f"class{class_id:03d}:{route}:half_band_auc_drop")

    masked = metrics["common_masked"]
    relion_beyond = bool(masked["relion_crossing_beyond_measured_range"])
    recovar_beyond = bool(masked["recovar_crossing_beyond_measured_range"])
    resolution_ratio = None
    shell_lag = None
    if relion_beyond and recovar_beyond:
        resolution_pass = True
        comparison_status = "both_beyond_measured_range"
    elif recovar_beyond:
        resolution_pass = True
        comparison_status = "recovar_beyond_relion_observed_crossing"
    elif relion_beyond:
        # RELION may cross arbitrarily beyond the measured band, so a finite
        # RECOVAR crossing cannot prove the one-shell/5% condition.
        resolution_pass = False
        comparison_status = "relion_beyond_recovar_observed_fail_closed"
    else:
        resolution_ratio = masked["recovar_resolution_angstrom"] / masked["relion_resolution_angstrom"]
        shell_lag = masked["relion_crossing_shell"] - masked["recovar_crossing_shell"]
        resolution_pass = (
            resolution_ratio <= float(thresholds["masked_resolution_ratio_max"])
            or shell_lag <= int(thresholds["masked_resolution_shell_lag_max"])
        )
        comparison_status = "both_crossings_observed"
    masked["recovar_to_relion_resolution_ratio"] = (
        None if resolution_ratio is None else float(resolution_ratio)
    )
    masked["recovar_crossing_shell_lag"] = None if shell_lag is None else int(shell_lag)
    masked["resolution_comparison_status"] = comparison_status
    masked["resolution_not_worse_by_more_than_one_shell_or_five_percent"] = resolution_pass
    if not resolution_pass:
        failures.append(f"class{class_id:03d}:common_masked:resolution")

    unmasked = metrics["unmasked"]
    if unmasked["cross_merged_band_fsc_auc"] < float(thresholds["cross_merged_band_auc_min"]):
        failures.append(f"class{class_id:03d}:unmasked:cross_merged_band_fsc_auc")
    if min(unmasked["cross_half1_band_fsc_auc"], unmasked["cross_half2_band_fsc_auc"]) < float(
        thresholds["cross_each_half_band_auc_min"]
    ):
        failures.append(f"class{class_id:03d}:unmasked:cross_each_half_band_fsc_auc")
    return metrics, failures


def _read_assignments(
    relion_dir: Path,
    recovar_dir: Path,
    iteration: int,
    input_particles_star: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    relion_star = relion_dir / f"run_it{iteration:03d}_data.star"
    recovar_results = recovar_dir / "refinement_results.npz"
    _require(relion_star.is_file(), f"missing RELION assignment STAR: {relion_star}")
    _require(recovar_results.is_file(), f"missing RECOVAR refinement results: {recovar_results}")
    input_names = [str(value) for value in _column(_particle_table(input_particles_star), "rlnImageName")]
    _require(len(input_names) == len(set(input_names)), "input particle identities are not unique")
    relion_table = _particle_table(relion_star)
    relion_names = [str(value) for value in _column(relion_table, "rlnImageName")]
    relion_classes = np.asarray(_column(relion_table, "rlnClassNumber"), dtype=np.int64) - 1
    _require(len(relion_names) == len(set(relion_names)), "RELION assignment identities are not unique")
    _require(set(relion_names) == set(input_names), "RELION assignment identities changed")
    relion_by_name = dict(zip(relion_names, relion_classes, strict=True))
    relion = np.asarray([relion_by_name[name] for name in input_names], dtype=np.int64)
    key = f"class_assignments_by_image_iter_{iteration - 1:03d}"
    support_key = f"sig_counts_by_image_iter_{iteration - 1:03d}"
    with np.load(recovar_results, allow_pickle=False) as payload:
        _require(key in payload.files, f"missing RECOVAR assignment field {key}")
        recovar = np.asarray(payload[key], dtype=np.int64).reshape(-1)
        support = np.asarray(payload[support_key], dtype=np.float64).reshape(-1) if support_key in payload.files else None
        final_all_data = bool(np.asarray(payload["final_all_data_ran"]).item())
        commit = str(np.asarray(payload["git_commit"]).item()) if "git_commit" in payload.files else None
        symmetry = str(np.asarray(payload["symmetry_label"]).item()) if "symmetry_label" in payload.files else None
    _require(relion.size == recovar.size, "RELION/RECOVAR assignment lengths differ")
    _require(np.all((relion >= 0) & (relion < N_CLASSES)), "RELION assignments are out of range")
    _require(np.all((recovar >= 0) & (recovar < N_CLASSES)), "RECOVAR assignments are out of range")
    support_summary = None
    if support is not None:
        _require(support.size == recovar.size and np.all(np.isfinite(support)), "invalid RECOVAR support counts")
        support_summary = {
            "field": support_key,
            "mean": float(np.mean(support)),
            "median": float(np.median(support)),
            "p95": float(np.quantile(support, 0.95)),
            "max": float(np.max(support)),
        }
    return relion, recovar, {
        "relion_data_star": str(relion_star.resolve()),
        "recovar_results": str(recovar_results.resolve()),
        "recovar_assignment_field": key,
        "recovar_support": support_summary,
        "recovar_final_all_data_ran": final_all_data,
        "recovar_git_commit": commit,
        "recovar_symmetry": symmetry,
        "identity_order": "input half STAR rlnImageName order",
        "ordered_image_names_sha256": sha256_strings(input_names),
    }


def _parse_peak_hbm(path: Path) -> tuple[int, str]:
    _require(path.is_file(), f"missing GPU monitor: {path}")
    with path.open(newline="", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    _require(rows, f"GPU monitor contains no samples: {path}")
    required = {"epoch", "gpu_uuid", "memory_used_mib"}
    _require(required.issubset(rows[0]), f"GPU monitor schema mismatch: {path}")
    uuids = {str(row["gpu_uuid"]).strip() for row in rows}
    _require(len(uuids) == 1 and next(iter(uuids)).startswith("GPU-"), f"GPU identity changed: {path}")
    values = [int(float(row["memory_used_mib"])) for row in rows]
    _require(all(0 <= value <= 200000 for value in values), f"invalid HBM sample: {path}")
    return max(values), next(iter(uuids))


def _parse_max_rss(path: Path) -> int:
    _require(path.is_file(), f"missing GNU time report: {path}")
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(errors="replace"))
    _require(match is not None, f"GNU time report has no MaxRSS: {path}")
    return int(match.group(1))


def _resource_row(directory: Path) -> dict[str, Any]:
    wall_path = directory / "slurm_walltime.json"
    monitor_path = directory / "gpu_monitor.csv"
    time_path = directory / "time.txt"
    _require(wall_path.is_file(), f"missing wall-time record: {wall_path}")
    wall = json.loads(wall_path.read_text())
    _require(int(wall.get("exit_status", -1)) == 0, f"engine did not exit successfully: {directory}")
    slurm_job_id = str(wall.get("slurm_job_id", ""))
    _require(slurm_job_id, f"engine wall-time record has no Slurm job ID: {directory}")
    peak_hbm, gpu_uuid = _parse_peak_hbm(monitor_path)
    return {
        "wall_s": float(wall["external_wall_s"]),
        "slurm_job_id": slurm_job_id,
        "peak_hbm_mib": peak_hbm,
        "physical_gpu_uuid": gpu_uuid,
        "max_rss_kib": _parse_max_rss(time_path),
        "wall_record": str(wall_path.resolve()),
        "gpu_monitor": str(monitor_path.resolve()),
        "time_report": str(time_path.resolve()),
    }


def validate_engine_job_binding(
    performance: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    qualification_job_id: str,
) -> dict[str, Any]:
    """Bind all four serial engine measurements to the audited Slurm job."""

    _require(bool(qualification_job_id), "qualification Slurm job ID is missing")
    observed: dict[str, list[str]] = {}
    for engine in ("relion", "recovar"):
        rows = list(performance.get(engine, []))
        _require(len(rows) == 2, f"expected two {engine} engine wall records")
        observed[engine] = [str(row.get("slurm_job_id", "")) for row in rows]
        _require(
            observed[engine] == [qualification_job_id, qualification_job_id],
            f"{engine} engine wall records are not bound to qualification job {qualification_job_id}",
        )
    return {
        "qualification_job_id": qualification_job_id,
        "engine_wall_job_ids": observed,
        "record_count": 4,
        "all_bound": True,
    }


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    manifest_path = args.manifest.resolve()
    _require(manifest_path.is_file(), f"missing submission manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "submission manifest schema mismatch")
    _require(int(manifest["config"]["K"]) == N_CLASSES, "audit requires exactly K=4")
    _require(manifest["config"]["symmetry"] == "C1", "frozen EMPIAR-10076 contract requires C1")
    _require(manifest.get("thresholds") == EXPECTED_THRESHOLDS, "prospective K=4 science thresholds changed")
    analysis_policy = validate_analysis_policy(manifest, args)
    _require(bool(manifest["source"]["clean"]), "source worktree was not clean at launch")
    _require(GIT_SHA_RE.fullmatch(str(manifest["source"]["commit"])) is not None, "invalid source commit")
    input_audit = validate_input_artifacts(manifest)
    launcher_audit = validate_launcher_artifacts(manifest)
    command_audit = validate_commands(manifest)
    slurm_audit = validate_slurm_allocation(manifest)
    split = validate_particle_split(manifest)
    expected_iteration = int(manifest["config"]["max_iter"])

    relion_paths: list[list[Path]] = []
    recovar_paths: list[list[Path]] = []
    replica_audits: list[list[dict[str, Any]]] = []
    assignment_raw: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    performance: dict[str, Any] = {"relion": [], "recovar": []}
    for row in manifest["halves"]:
        relion_dir = Path(row["relion_dir"])
        recovar_dir = Path(row["recovar_dir"])
        relion_paths.append(_latest_relion_maps(relion_dir, expected_iteration))
        representatives, replica_audit = _latest_recovar_combined_maps(
            Path(row["recovar_intermediates_dir"]), expected_iteration - 1
        )
        recovar_paths.append(representatives)
        replica_audits.append(replica_audit)
        _reject_within_process_duplicates(relion_paths[-1], engine="RELION", half=int(row["half"]))
        _reject_within_process_duplicates(representatives, engine="RECOVAR", half=int(row["half"]))
        assignment_raw.append(
            _read_assignments(
                relion_dir,
                recovar_dir,
                expected_iteration,
                Path(row["particles_star"]),
            )
        )
        performance["relion"].append(_resource_row(relion_dir))
        performance["recovar"].append(_resource_row(recovar_dir))
    engine_job_binding = validate_engine_job_binding(
        performance,
        qualification_job_id=str(slurm_audit["qualification"]["job_id"]),
    )
    _reject_cross_process_duplicates(relion_paths, engine="RELION")
    _reject_cross_process_duplicates(recovar_paths, engine="RECOVAR")
    physical_gpu_path = Path(manifest_path.parent / "provenance" / "physical_gpu_uuid.txt")
    _require(physical_gpu_path.is_file(), f"missing physical GPU identity: {physical_gpu_path}")
    physical_gpu_uuid = physical_gpu_path.read_text().strip()
    observed_gpu_uuids = {
        row["physical_gpu_uuid"] for engine_rows in performance.values() for row in engine_rows
    }
    _require(observed_gpu_uuids == {physical_gpu_uuid}, "engine GPU identity changed")

    relion_sets = [_load_maps(paths, frame="relion") for paths in relion_paths]
    recovar_sets = [_load_maps(paths, frame="recovar") for paths in recovar_paths]
    voxel_sizes = [row[1] for row in (*relion_sets, *recovar_sets)]
    _require(max(voxel_sizes) - min(voxel_sizes) <= 1.0e-5, "engine/half voxel sizes differ")
    voxel_size = voxel_sizes[0]
    anchor = relion_sets[0][0]
    box_size = int(anchor[0].shape[0])
    _require(box_size == int(manifest["config"]["grid_size"]), "map grid differs from frozen analysis grid")
    _require(all(volume.shape == anchor[0].shape for maps, _ in (*relion_sets, *recovar_sets) for volume in maps), "map shapes differ")
    alignment_policy = analysis_policy["alignment"]
    fit_max_shell = int(alignment_policy["fit_max_shell"])
    _require(fit_max_shell <= box_size // 2 - 1, "frozen fit_max_shell exceeds measured Fourier band")
    _require(fit_max_shell >= 4, "map is too small for rigid alignment")

    raw_sets: dict[str, list[np.ndarray]] = {
        "relion_half1": relion_sets[0][0],
        "relion_half2": relion_sets[1][0],
        "recovar_half1": recovar_sets[0][0],
        "recovar_half2": recovar_sets[1][0],
    }
    aligned: dict[str, list[np.ndarray]] = {"relion_half1": anchor}
    alignment: dict[str, Any] = {
        "relion_half1": {
            "identity_anchor": True,
            "rotation_matrix": np.eye(3).tolist(),
            "translation_zyx": [0.0, 0.0, 0.0],
        }
    }
    for label, maps in (
        ("relion_half2", relion_sets[1][0]),
        ("recovar_half1", recovar_sets[0][0]),
        ("recovar_half2", recovar_sets[1][0]),
    ):
        aligned[label], alignment[label] = _align_set_to_anchor(
            maps,
            anchor,
            policy=alignment_policy,
        )
    calculator = ShellFscCalculator(box_size)
    mask, mask_metadata = _common_mask(
        list(aligned.values()),
        policy=analysis_policy["common_mask"],
    )
    permutations: dict[str, list[int]] = {"relion_half1": list(range(N_CLASSES))}
    pairwise: dict[str, Any] = {}
    permutation_optima: dict[str, Any] = {
        "relion_half1": {
            "identity_anchor": True,
            "objective": None,
            "second_best_objective": None,
            "objective_margin": None,
            "exact_optimum_count": 1,
        }
    }
    for label in ("relion_half2", "recovar_half1", "recovar_half2"):
        matrix = _pairwise_auc(
            calculator,
            [volume * mask for volume in aligned[label]],
            [volume * mask for volume in anchor],
        )
        pairwise[label + "_to_relion_half1"] = matrix.tolist()
        permutations[label], permutation_optima[label] = _hungarian_to_anchor(
            matrix,
            label=label,
            min_absolute_margin=float(
                manifest["thresholds"]["permutation_objective_margin_abs_min"]
            ),
            min_relative_margin=float(
                manifest["thresholds"]["permutation_objective_margin_rel_min"]
            ),
        )

    canonical: dict[str, list[np.ndarray]] = {}
    raw_canonical: dict[str, list[np.ndarray]] = {}
    for label, maps in aligned.items():
        canonical[label] = [maps[source_id] for source_id in permutations[label]]
        raw_canonical[label] = [raw_sets[label][source_id] for source_id in permutations[label]]
    curves: dict[str, np.ndarray] = {}
    classes: list[dict[str, Any]] = []
    for class_id in range(N_CLASSES):
        rh1 = raw_canonical["relion_half1"][class_id]
        rh2 = raw_canonical["relion_half2"][class_id]
        ch1 = raw_canonical["recovar_half1"][class_id]
        ch2 = raw_canonical["recovar_half2"][class_id]
        rh1_registered = canonical["relion_half1"][class_id]
        rh2_registered = canonical["relion_half2"][class_id]
        ch1_registered = canonical["recovar_half1"][class_id]
        ch2_registered = canonical["recovar_half2"][class_id]
        rmerged = np.float32(0.5) * (rh1_registered + rh2_registered)
        cmerged = np.float32(0.5) * (ch1_registered + ch2_registered)
        row: dict[str, Any] = {
            "canonical_class": class_id + 1,
            "source_class_ids": {
                label: permutations[label][class_id] + 1 for label in permutations
            },
            "unmasked": {},
            "common_masked": {},
        }
        pairs = {
            "relion_halfmap": (rh1_registered, rh2_registered),
            "recovar_halfmap": (ch1_registered, ch2_registered),
            "cross_half1": (ch1_registered, rh1_registered),
            "cross_half2": (ch2_registered, rh2_registered),
            "cross_merged": (cmerged, rmerged),
            "diagnostic_relion_halfmap_raw": (rh1, rh2),
            "diagnostic_recovar_halfmap_raw": (ch1, ch2),
            "diagnostic_cross_half1_raw": (ch1, rh1),
            "diagnostic_cross_half2_raw": (ch2, rh2),
            "diagnostic_cross_merged_raw": (
                np.float32(0.5) * (ch1 + ch2),
                np.float32(0.5) * (rh1 + rh2),
            ),
        }
        for metric_name, (lhs, rhs) in pairs.items():
            key = f"class{class_id + 1:03d}_{metric_name}_unmasked"
            row["unmasked"][metric_name] = _curve_metric(
                calculator,
                lhs,
                rhs,
                key=key,
                curves=curves,
                voxel_size=voxel_size,
                consecutive=int(analysis_policy["fsc"]["crossing_consecutive_shells"]),
            )
            if not metric_name.endswith("_raw"):
                masked_key = f"class{class_id + 1:03d}_{metric_name}_common_masked"
                row["common_masked"][metric_name] = _curve_metric(
                    calculator,
                    lhs * mask,
                    rhs * mask,
                    key=masked_key,
                    curves=curves,
                    voxel_size=voxel_size,
                    consecutive=int(analysis_policy["fsc"]["crossing_consecutive_shells"]),
                )
        classes.append(row)

    science_failures: list[str] = []
    for row in classes:
        metrics, failures = _science_metrics_for_class(
            int(row["canonical_class"]),
            curves,
            box_size=box_size,
            voxel_size=voxel_size,
            thresholds=manifest["thresholds"],
        )
        row["prospective_science_metrics"] = metrics
        science_failures.extend(failures)

    assignment_rows: list[dict[str, Any]] = []
    for half_index, (relion, recovar, metadata) in enumerate(assignment_raw, start=1):
        relion_label = f"relion_half{half_index}"
        recovar_label = f"recovar_half{half_index}"
        relion_inverse = np.empty(N_CLASSES, dtype=np.int64)
        recovar_inverse = np.empty(N_CLASSES, dtype=np.int64)
        for canonical_id, source_id in enumerate(permutations[relion_label]):
            relion_inverse[source_id] = canonical_id
        for canonical_id, source_id in enumerate(permutations[recovar_label]):
            recovar_inverse[source_id] = canonical_id
        relion_canonical = relion_inverse[relion]
        recovar_canonical = recovar_inverse[recovar]
        _require(metadata["recovar_git_commit"] == manifest["source"]["commit"], "RECOVAR result commit mismatch")
        _require(metadata["recovar_symmetry"] == manifest["config"]["symmetry"], "RECOVAR result symmetry mismatch")
        assignment_rows.append(
            {
                "half": half_index,
                "particle_count": int(relion.size),
                "agreement": float(np.mean(relion_canonical == recovar_canonical)),
                "relion_counts": np.bincount(relion_canonical, minlength=N_CLASSES).tolist(),
                "recovar_counts": np.bincount(recovar_canonical, minlength=N_CLASSES).tolist(),
                **metadata,
            }
        )
        if assignment_rows[-1]["agreement"] < float(manifest["thresholds"]["assignment_agreement_min"]):
            science_failures.append(f"half{half_index}:class_assignment_agreement")
        for engine in ("relion", "recovar"):
            counts = assignment_rows[-1][f"{engine}_counts"]
            if min(counts) < int(manifest["thresholds"]["minimum_class_count"]):
                science_failures.append(f"half{half_index}:{engine}:collapsed_class")

    mask_path = args.output_dir / "common_soft_mask.mrc"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    helpers.write_mrc(str(mask_path), mask, voxel_size=voxel_size)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "claim_scope": (
            "two independent K=4 Class3D processes per engine; one process per frozen particle half; "
            "last numbered maps only"
        ),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "dataset": manifest["dataset"],
        "profile": manifest["profile"],
        "source": manifest["source"],
        "config": manifest["config"],
        "thresholds": manifest["thresholds"],
        "analysis_policy": analysis_policy,
        "analysis_policy_binding": {
            "manifest_exact": True,
            "cli_flags_exact": True,
            "run_script_hashed": True,
        },
        "provenance_audit": {
            "input_artifacts": input_audit,
            "launcher": launcher_audit,
            "commands": command_audit,
            "slurm": slurm_audit,
            "physical_gpu_uuid": physical_gpu_uuid,
        },
        "particle_split": split,
        "box_size": box_size,
        "voxel_size_angstrom": voxel_size,
        "map_selection_policy": {
            "relion": "run_itNNN_classXXX.mrc from each independent half process",
            "recovar": "itNNN_half1_classX_reg.mrc representative from each independent half process",
            "recovar_internal_replica_policy": (
                "half1/half2 files inside one K-class process must be byte-identical combined-map replicas; "
                "the second is discarded and never called an independent half map"
            ),
            "final_all_data_maps_used": False,
            "within_process_duplicate_class_maps_rejected": True,
            "cross_process_duplicate_maps_rejected": True,
            "extra_final_recovar_class_ids_rejected": True,
        },
        "recovar_internal_replica_audit": replica_audits,
        "alignment": {
            "policy": (
                "one proper rigid transform per four-class map set, fitted on the label-invariant equal-weight "
                "unit-RMS ensemble and applied unchanged to all four classes; no reflection/sign/scale fit"
            ),
            "anchor": "relion_half1",
            "sets": alignment,
            "acceptance_halfmap_fsc_uses_alignment": True,
            "cross_engine_fsc_uses_alignment": True,
        },
        "class_matching": {
            "anchor": "relion_half1",
            "metric": "common-mask normalized non-DC FSC-AUC after shared-set proper-rigid alignment",
            "source_class_for_anchor": {label: [value + 1 for value in values] for label, values in permutations.items()},
            "pairwise_fsc_auc": pairwise,
            "unique_exact_permutation_required": True,
            "minimum_absolute_objective_margin": manifest["thresholds"][
                "permutation_objective_margin_abs_min"
            ],
            "minimum_relative_objective_margin": manifest["thresholds"][
                "permutation_objective_margin_rel_min"
            ],
            "permutation_optima": permutation_optima,
        },
        "common_mask": {
            **mask_metadata,
            "path": str(mask_path.resolve()),
            "sha256": sha256_file(mask_path),
            "applied_identically_to_all_engines_halves_classes": True,
            "construction_frame": "RELION-half1 anchor after one shared transform per four-class set",
            "fsc_correction": "none",
            "phase_randomization_corrected": False,
            "scientific_role": "relative uncorrected common-mask diagnostic",
            "absolute_resolution_claim": False,
        },
        "metric_policy": {
            "within_engine_halfmap": (
                "mandatory proper-rigid registered curves; one transform is shared by all four classes "
                "and global coordinate drift is accepted"
            ),
            "cross_engine_primary": "proper-rigid registered curves after shared-set alignment",
            "diagnostics": "raw frozen-frame halfmap/cross-engine routes are retained unmasked and cannot rescue",
            "masked_can_rescue_unmasked": False,
            "comparison_band": (
                "every masked and unmasked RECOVAR and cross-engine curve is integrated over one "
                "band frozen from the corresponding RELION unmasked half-map's resolved non-DC shells"
            ),
            "common_masked_fsc": "uncorrected relative diagnostic; no phase-randomization correction",
            "absolute_resolution_claim": False,
        },
        "classes": classes,
        "assignments_and_support": assignment_rows,
        "performance": {
            **performance,
            "same_job_serial": True,
            "job_binding": engine_job_binding,
            "hbm_sampling": "one-second nvidia-smi lower bound",
        },
        "prospective_science_gate": {
            "accepted": not science_failures,
            "failures": science_failures,
            "policy": (
                "frozen before launch: per-class masked resolution no worse than RELION by more than "
                "one Fourier shell or 5%, whichever is larger; masked and unmasked half-map band "
                "FSC-AUC drop <=0.01 over the frozen RELION unmasked-resolved band; registered "
                "unmasked merged cross-engine band FSC-AUC >=0.99; "
                "each-half cross-engine band FSC-AUC >=0.90; assignment agreement >=0.99; no class collapse"
                "; every class permutation has frozen absolute and relative objective margins"
            ),
        },
        "limitations": [
            "This audit compares scientific half-map quality; it does not require the two independent halves to follow identical class trajectories.",
            "Class labels are matched independently to RELION half 1; a split/merge remains visible in per-class FSC and populations.",
            "The common-mask FSC is uncorrected and is used only for matched relative diagnostics, not an absolute-resolution claim.",
            "This bounded single-seed harness does not yet report Pmax, pose/translation agreement, or the required multi-seed consensus aggregate.",
            "A completed result must be admitted to the benchmark registry separately after Slurm/accounting and artifact sealing.",
        ],
    }
    return report, curves, mask


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-max-shell", type=int, required=True)
    parser.add_argument("--crossing-consecutive-shells", type=int, required=True)
    parser.add_argument(
        "--phase-randomization-corrected",
        choices=("true", "false"),
        required=True,
    )
    parser.add_argument(
        "--absolute-resolution-claim",
        choices=("true", "false"),
        required=True,
    )
    parser.add_argument("--coarse-healpix-order", type=int, required=True)
    parser.add_argument("--refine-healpix-order", type=int, action="append", required=True)
    parser.add_argument("--interpolation-order", type=int, required=True)
    parser.add_argument("--mask-threshold", required=True)
    parser.add_argument("--mask-lowpass-sigma", type=int, required=True)
    parser.add_argument("--mask-extend", type=int, required=True)
    parser.add_argument("--mask-soft-edge", type=int, required=True)
    parser.add_argument("--mask-cleanup", choices=("true", "false"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report, curves, _mask = audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "halfmap_fsc_curves.npz"
    np.savez_compressed(curve_path, **curves)
    report["curve_archive"] = {
        "path": str(curve_path.resolve()),
        "sha256": sha256_file(curve_path),
        "fields": sorted(curves),
    }
    report_path = args.output_dir / "halfmap_audit.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0 if report["prospective_science_gate"]["accepted"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from exc
