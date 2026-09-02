#!/usr/bin/env python3
"""Seal a completed positive K-class comparison as benchmark evidence.

The robustness launcher intentionally writes bulky maps outside the repository.
This sealer converts one completed three-seed slice of such a run into the
compact ``campaign_schema_v1.json`` ledger format.  It fails closed when jobs,
trajectory audits, endpoint evaluations, assignments, or source provenance are
missing.  A deterministic checksum manifest pins every numbered map consumed
by the trajectory audit without copying those maps into git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.audit_k4_fsc_trajectory import _column, _particle_table  # noqa: E402
from scripts.seal_em_kclass_negative_diagnostic import (  # noqa: E402
    _file_ref,
    _git_source,
    _gpu_identity,
    _parse_env,
    _peak_hbm_mib,
    _sacct_job,
)

DEFAULT_SEEDS = (41001, 41002, 41003)
COLLAPSE_THRESHOLD = 0.01


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rows_for_k(run_root: Path, n_classes: int, seeds: tuple[int, ...]) -> list[dict[str, str]]:
    with (run_root / "case_table.tsv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="|") if int(row["n_classes"]) == n_classes]
    rows.sort(key=lambda row: int(row["seed"]))
    actual_seeds = tuple(int(row["seed"]) for row in rows)
    if actual_seeds != seeds:
        raise ValueError(f"K={n_classes} seeds are {actual_seeds}, expected exactly {seeds}")
    if len({row["index"] for row in rows}) != 1:
        raise ValueError(f"K={n_classes} rows do not share one frozen matrix case")
    return rows


def _audit_jobs(run_root: Path, n_classes: int, seeds: tuple[int, ...]) -> dict[int, str]:
    with (run_root / "audit_jobs.tsv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="\t") if int(row["n_classes"]) == n_classes]
    jobs = {int(row["seed"]): row["audit_job_id"] for row in rows}
    if tuple(sorted(jobs)) != seeds:
        raise ValueError(f"K={n_classes} audit-job seeds are incomplete: {sorted(jobs)}")
    return jobs


def _job_logs(run_root: Path, job_id: str, suffix: str) -> Path:
    matches = sorted(run_root.glob(f"*_{job_id}.{suffix}"))
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} log ending in job {job_id}, found {matches}")
    return matches[0].resolve()


def _job_reference(
    job_id: str,
    *,
    role: str,
    logs: list[Path],
    gpu_model: str | None,
    gpu_uuid: str | None,
) -> dict[str, Any]:
    job = _sacct_job(job_id)
    if job["status"] not in {"COMPLETED", "FAILED"}:
        raise ValueError(f"job {job_id} is not terminal: {job['status']}")
    if (job["status"] == "COMPLETED") != (job["exit_code"] == "0:0"):
        raise ValueError(f"job {job_id} state conflicts with exit code: {job}")
    if job["req_tres"] != job["alloc_tres"]:
        raise ValueError(f"job {job_id} requested and allocated resources differ")
    for path in logs:
        if not path.is_file():
            raise ValueError(f"job {job_id} is missing log {path}")
    return {
        "role": role,
        **job,
        "gpu_model": gpu_model,
        "gpu_uuid": gpu_uuid,
        "logs": [str(path.resolve()) for path in logs],
    }


def _trajectory_map_paths(case_root: Path, *, iterations: int, n_classes: int) -> list[Path]:
    paths: list[Path] = []
    for iteration in range(iterations):
        for half in (1, 2):
            for class_id in range(1, n_classes + 1):
                paths.append(
                    case_root / "recovar" / "intermediates" / f"it{iteration:03d}_half{half}_class{class_id}_reg.mrc"
                )
    for iteration in range(1, iterations + 1):
        for class_id in range(1, n_classes + 1):
            paths.append(case_root / "relion_ref" / f"run_it{iteration:03d}_class{class_id:03d}.mrc")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"trajectory source-map topology is incomplete: {missing[:8]}")
    return paths


def _write_trajectory_manifest(case_root: Path, *, iterations: int, n_classes: int) -> Path:
    path = case_root / "trajectory_analysis" / "trajectory_source_checksums.json"
    payload = {
        "schema": "recovar.em.kclass_trajectory_source_checksums.v1",
        "case_root": str(case_root),
        "n_classes": n_classes,
        "numbered_iterations": iterations,
        "files": [
            _file_ref(source) for source in _trajectory_map_paths(case_root, iterations=iterations, n_classes=n_classes)
        ],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if path.exists() and path.read_text() != serialized:
        raise ValueError(f"existing trajectory checksum manifest differs: {path}")
    path.write_text(serialized)
    return path


def _passing_class_cells(audit: dict[str, Any]) -> int:
    direct_min = float(audit["thresholds"]["per_class_direct_fsc_auc_min"])
    gt_min = float(audit["thresholds"]["per_class_recovar_minus_relion_gt_fsc_auc_min"])
    return sum(
        float(row["cross_engine"]["fsc_auc"]) >= direct_min and float(row["gt_fsc_auc_delta"]) >= gt_min
        for iteration in audit["numbered_iterations"]
        for row in iteration["classes"]
    )


def _science_passes(audit: dict[str, Any]) -> bool:
    gt_min = float(audit["thresholds"]["per_class_recovar_minus_relion_gt_fsc_auc_min"])
    cells = [row for iteration in audit["numbered_iterations"] for row in iteration["classes"]] + list(
        audit["final"]["classes"]
    )
    return all(float(row["gt_fsc_auc_delta"]) >= gt_min for row in cells)


def _endpoint_by_class(payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    primary = payload["primary"]
    return {(int(row["class"]) + 1, int(row["matched_gt_class"]) + 1): row for row in primary["per_class"]}


def _final_class_rows(
    audit: dict[str, Any], recovar_endpoint: dict[str, Any], relion_endpoint: dict[str, Any]
) -> list[dict[str, Any]]:
    recovar_by_class = _endpoint_by_class(recovar_endpoint)
    relion_by_class = _endpoint_by_class(relion_endpoint)
    rows: list[dict[str, Any]] = []
    for row in audit["final"]["classes"]:
        recovar_class = int(row["recovar_class"])
        relion_class = int(row["relion_class"])
        gt_class = int(row["gt_class"])
        rec_key = (recovar_class, gt_class)
        rel_key = (relion_class, gt_class)
        if rec_key not in recovar_by_class or rel_key not in relion_by_class:
            raise ValueError(
                "independent endpoint evaluator disagrees with trajectory class matching: "
                f"RECOVAR {rec_key}, RELION {rel_key}"
            )
        recovar_gt = float(row["vs_gt"]["recovar"]["fsc_auc"])
        relion_gt = float(row["vs_gt"]["relion"]["fsc_auc"])
        delta = float(row["gt_fsc_auc_delta"])
        if not np.isclose(delta, recovar_gt - relion_gt, rtol=0.0, atol=1e-12):
            raise ValueError("trajectory signed GT FSC-AUC delta is inconsistent")
        rows.append(
            {
                "recovar_class": recovar_class,
                "relion_class": relion_class,
                "gt_class": gt_class,
                "cross_engine_fsc_auc": float(row["cross_engine"]["fsc_auc"]),
                "recovar_gt_fsc_auc": recovar_gt,
                "relion_gt_fsc_auc": relion_gt,
                "gt_fsc_auc_delta": delta,
                "recovar_gt_resolution_0_143_angstrom": float(recovar_by_class[rec_key]["resolution_0143_A"]),
                "relion_gt_resolution_0_143_angstrom": float(relion_by_class[rel_key]["resolution_0143_A"]),
            }
        )
    return rows


def _class_counts(case_root: Path, *, n_classes: int, iterations: int) -> tuple[list[int], list[int]]:
    result_path = case_root / "recovar" / "refinement_results.npz"
    key = f"class_assignments_by_image_iter_{iterations - 1:03d}"
    with np.load(result_path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise ValueError(f"missing {key} in {result_path}")
        recovar = np.asarray(payload[key], dtype=np.int64).reshape(-1)
    relion_table = _particle_table(case_root / "relion_ref" / f"run_it{iterations:03d}_data.star")
    relion = np.asarray(_column(relion_table, "rlnClassNumber"), dtype=np.int64) - 1
    for label, values in (("RECOVAR", recovar), ("RELION", relion)):
        if values.size == 0 or values.min() < 0 or values.max() >= n_classes:
            raise ValueError(f"{label} final class assignments are missing or out of range")
    return (
        np.bincount(recovar, minlength=n_classes).astype(int).tolist(),
        np.bincount(relion, minlength=n_classes).astype(int).tolist(),
    )


def _occupancy(case_root: Path, *, n_classes: int, iterations: int, particles: int) -> dict[str, Any]:
    recovar_counts, relion_counts = _class_counts(case_root, n_classes=n_classes, iterations=iterations)
    if sum(recovar_counts) != particles or sum(relion_counts) != particles:
        raise ValueError("final class counts do not sum to the frozen particle count")
    recovar_flags = [
        class_id for class_id, count in enumerate(recovar_counts, start=1) if count / particles < COLLAPSE_THRESHOLD
    ]
    relion_flags = [
        class_id for class_id, count in enumerate(relion_counts, start=1) if count / particles < COLLAPSE_THRESHOLD
    ]
    flagged = recovar_flags or relion_flags
    zero = any(count == 0 for count in recovar_counts + relion_counts)
    status = "ZERO_CLASS" if zero else "NEAR_COLLAPSE" if flagged else "CLEAR"
    return {
        "source_iteration": iterations,
        "recovar_metric": "hard_assignments",
        "recovar_counts": recovar_counts,
        "recovar_fractions": None,
        "relion_counts": relion_counts,
        "collapse_threshold_fraction": COLLAPSE_THRESHOLD,
        "recovar_flagged_classes": recovar_flags,
        "relion_flagged_classes": relion_flags,
        "status": status,
        "note": (
            f"Hard assignments come from RECOVAR iteration {iterations - 1} and RELION "
            f"run_it{iterations:03d}_data.star; flags are exact counts below 1% of "
            f"{particles} particles."
        ),
    }


def _outcome(audit: dict[str, Any], occupancy: dict[str, Any], final_rows: list[dict[str, Any]]) -> dict[str, str]:
    formal_pass = audit["status"] == "pass"
    science_pass = _science_passes(audit)
    occupancy_status = occupancy["status"]
    if occupancy_status == "ZERO_CLASS":
        raise ValueError("a positive campaign cannot contain a zero-population final class")
    if formal_pass and occupancy_status == "CLEAR":
        formal_status = "PASS"
        science_status = "PASS"
        classification = "TRAJECTORY_EXACT"
    elif formal_pass and occupancy_status == "NEAR_COLLAPSE":
        formal_status = "PASS"
        science_status = "BOUNDARY"
        classification = "TRAJECTORY_EXACT_NEAR_COLLAPSE"
    elif not formal_pass and science_pass:
        formal_status = "FAIL"
        science_status = "PASS"
        classification = "SCIENCE_EQUIVALENT"
    else:
        formal_status = "FAIL"
        science_status = "UNRESOLVED"
        classification = "UNRESOLVED_TRAJECTORY_FAILURE"
    min_direct = min(row["cross_engine_fsc_auc"] for row in final_rows)
    min_delta = min(row["gt_fsc_auc_delta"] for row in final_rows)
    reason = (
        f"The frozen trajectory audit is {formal_status}; the signed GT science gate is "
        f"{science_status}. Final minimum direct FSC-AUC is {min_direct:.12g}, final "
        f"minimum RECOVAR-minus-RELION GT FSC-AUC is {min_delta:.12g}, and final "
        f"occupancy is {occupancy_status}."
    )
    return {
        "endpoint_status": "COMPLETE",
        "formal_status": formal_status,
        "science_status": science_status,
        "classification": classification,
        "reason": reason,
    }


def _existing_refs(paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    missing = [str(path) for _, path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"required evidence files are missing: {missing}")
    return [_file_ref(path, role=role) for role, path in paths]


def _case_inputs(case_root: Path, *, n_classes: int) -> list[dict[str, Any]]:
    data = case_root / "data"
    particle_paths = sorted(data.glob("particles.*.mrcs"))
    if len(particle_paths) != 1:
        raise ValueError(f"expected one particle stack under {data}, found {particle_paths}")
    paths: list[tuple[str, Path]] = [
        ("particles", particle_paths[0]),
        ("poses", data / "poses.pkl"),
        ("ctf", data / "ctf.pkl"),
        ("particles_star", data / "particles.star"),
        ("generation_config", data / "generation_config.json"),
        ("class_manifest", data / "class_manifest.json"),
        ("simulation_info", data / "simulation_info.pkl"),
        ("relion_initial_reference_star", data / "reference_init_classes_relion.star"),
    ]
    for class_id in range(1, n_classes + 1):
        paths.extend(
            [
                (f"gt_class_{class_id:03d}", data / f"reference_gt_class{class_id:03d}.mrc"),
                (
                    f"recovar_initial_class_{class_id:03d}",
                    data / f"reference_init_class{class_id:03d}.mrc",
                ),
                (
                    f"relion_initial_class_{class_id:03d}",
                    data / f"reference_init_class{class_id:03d}_relion.mrc",
                ),
            ]
        )
    return _existing_refs(paths)


def _case_artifacts(
    run_root: Path,
    row: dict[str, str],
    *,
    n_classes: int,
    iterations: int,
    trajectory_manifest: Path,
) -> list[dict[str, Any]]:
    case_root = Path(row["case_root"])
    paths: list[tuple[str, Path]] = [
        ("paired_gpu_uuid", case_root / "paired_gpu_uuid.json"),
        ("physical_gpu_inventory", case_root / "physical_gpu_inventory.csv"),
        ("recovar_gpu_monitor", case_root / "recovar_gpu_monitor.csv"),
        ("relion_gpu_monitor", case_root / "relion_gpu_monitor.csv"),
        ("recovar_walltime", case_root / "recovar" / "slurm_walltime.json"),
        ("relion_walltime", case_root / "relion_ref" / "slurm_walltime.json"),
        ("recovar_log", case_root / "recovar" / "run_full_refinement.log"),
        ("relion_log", case_root / "relion_class3d.log"),
        ("recovar_refinement_results", case_root / "recovar" / "refinement_results.npz"),
        ("recovar_benchmark_ledger", case_root / "recovar" / "benchmark_ledger.json"),
        ("recovar_endpoint_gt_fsc", case_root / "kclass_gt_fsc.json"),
        ("relion_endpoint_gt_fsc", case_root / "relion_kclass_gt_fsc.json"),
        (
            "relion_class_population_audit",
            case_root / "relion_ref" / "class_population_audit.json",
        ),
        ("relion_dispatch", case_root / "relion_ref" / "dispatch.tsv"),
        ("relion_dispatch_schedule", case_root / "relion_ref" / "dispatch_schedule.npz"),
        (
            "trajectory_report",
            case_root / "trajectory_analysis" / f"k{n_classes}_fsc_trajectory.json",
        ),
        (
            "trajectory_markdown",
            case_root / "trajectory_analysis" / f"k{n_classes}_fsc_trajectory.md",
        ),
        (
            "trajectory_shellwise",
            case_root / "trajectory_analysis" / f"k{n_classes}_fsc_trajectory_shellwise.npz",
        ),
        ("trajectory_source_checksums", trajectory_manifest),
        (
            "job_stdout",
            run_root / f"em_kclass_matrix_{row['index']}_{row['name']}.out",
        ),
        (
            "job_stderr",
            run_root / f"em_kclass_matrix_{row['index']}_{row['name']}.err",
        ),
        (
            "relion_final_assignments",
            case_root / "relion_ref" / f"run_it{iterations:03d}_data.star",
        ),
    ]
    for class_id in range(1, n_classes + 1):
        paths.extend(
            [
                (
                    f"recovar_final_class_{class_id:03d}",
                    case_root / "recovar" / f"final_class{class_id:03d}.mrc",
                ),
                (
                    f"relion_iteration{iterations}_class_{class_id:03d}",
                    case_root / "relion_ref" / f"run_it{iterations:03d}_class{class_id:03d}.mrc",
                ),
            ]
        )
    return _existing_refs(paths)


def _source_structures(case_root: Path, *, n_classes: int) -> list[dict[str, Any]]:
    manifest = json.loads((case_root / "data" / "class_manifest.json").read_text())
    if [int(row["class_number"]) for row in manifest] != list(range(1, n_classes + 1)):
        raise ValueError("class manifest is not ordered 1..K")
    return [
        _file_ref(Path(row["pdb_path"]), role=f"source_pdb_class_{class_id:03d}")
        for class_id, row in enumerate(manifest, start=1)
    ]


def _read_wall_s(path: Path) -> float:
    value = json.loads(path.read_text())["external_wall_s"]
    return float(value)


def _build_case(
    run_root: Path,
    row: dict[str, str],
    audit_job_id: str,
    *,
    n_classes: int,
) -> dict[str, Any]:
    case_root = Path(row["case_root"]).resolve()
    config_path = case_root / "case_config.json"
    config = json.loads(config_path.read_text())
    iterations = int(config["max_iter"])
    audit_path = case_root / "trajectory_analysis" / f"k{n_classes}_fsc_trajectory.json"
    audit = json.loads(audit_path.read_text())
    if int(audit["n_classes"]) != n_classes:
        raise ValueError(f"trajectory audit K mismatch: {audit_path}")
    if int(audit["numbered_iteration_count"]) != iterations:
        raise ValueError(f"trajectory audit iteration mismatch: {audit_path}")
    recovar_endpoint_path = case_root / "kclass_gt_fsc.json"
    relion_endpoint_path = case_root / "relion_kclass_gt_fsc.json"
    recovar_endpoint = json.loads(recovar_endpoint_path.read_text())
    relion_endpoint = json.loads(relion_endpoint_path.read_text())
    final_rows = _final_class_rows(audit, recovar_endpoint, relion_endpoint)
    particles = int(config["n_images"])
    occupancy = _occupancy(case_root, n_classes=n_classes, iterations=iterations, particles=particles)
    outcome = _outcome(audit, occupancy, final_rows)
    trajectory_manifest = _write_trajectory_manifest(case_root, iterations=iterations, n_classes=n_classes)
    gpu_model, gpu_uuid = _gpu_identity(case_root)
    execution = _job_reference(
        row["job_id"],
        role=f"paired_k{n_classes}_seed_{row['seed']}",
        logs=[
            run_root / f"em_kclass_matrix_{row['index']}_{row['name']}.out",
            run_root / f"em_kclass_matrix_{row['index']}_{row['name']}.err",
        ],
        gpu_model=gpu_model,
        gpu_uuid=gpu_uuid,
    )
    if execution["status"] != "COMPLETED":
        raise ValueError(f"paired campaign job did not complete: {execution}")
    recovar_wall_s = _read_wall_s(case_root / "recovar" / "slurm_walltime.json")
    relion_wall_s = _read_wall_s(case_root / "relion_ref" / "slurm_walltime.json")
    thresholds = audit["thresholds"]
    final_agreement = float(audit["numbered_iterations"][-1]["class_agreement"]["agreement"])
    return {
        "case_id": int(row["seed"]),
        "name": row["name"],
        "particles": particles,
        "box_size": int(config["grid_size"]),
        "configuration": config,
        "case_config": _file_ref(config_path),
        "launcher": _file_ref(Path(row["script"])),
        "reproduction_command": f"sbatch {Path(row['script']).resolve()}",
        "inputs": _case_inputs(case_root, n_classes=n_classes),
        "input_equivalence": {
            "group": f"k{n_classes}-independent-seed-replicates",
            "admissible_for_execution_invariance": False,
            "reason": (
                "Seeds 41001, 41002, and 41003 deliberately generate different particle "
                "stacks; this group tests stochastic robustness, not byte-identical execution."
            ),
        },
        "execution": execution,
        "artifacts": _case_artifacts(
            run_root,
            row,
            n_classes=n_classes,
            iterations=iterations,
            trajectory_manifest=trajectory_manifest,
        ),
        "quality": {
            "matching_method": "hungarian_max_fsc_auc",
            "final_classes": final_rows,
            "mean_recovar_gt_fsc_auc": float(np.mean([item["recovar_gt_fsc_auc"] for item in final_rows])),
            "mean_relion_gt_fsc_auc": float(np.mean([item["relion_gt_fsc_auc"] for item in final_rows])),
            "mean_gt_fsc_auc_delta": float(np.mean([item["gt_fsc_auc_delta"] for item in final_rows])),
            "class_assignment_agreement": final_agreement,
            "occupancy": occupancy,
            "trajectory": {
                "report": _file_ref(audit_path),
                "formal_thresholds": {
                    "direct_fsc_auc_min": float(thresholds["per_class_direct_fsc_auc_min"]),
                    "gt_fsc_auc_delta_min": float(thresholds["per_class_recovar_minus_relion_gt_fsc_auc_min"]),
                    "class_assignment_agreement_min": float(
                        thresholds["class_assignment_agreement_min_when_available"]
                    ),
                },
                "status": outcome["formal_status"],
                "evaluated_iterations": iterations,
                "evaluated_class_cells": iterations * n_classes,
                "passing_class_cells": _passing_class_cells(audit),
                "earliest_failure": audit["earliest_failure"],
            },
        },
        "performance": {
            "recovar_wall_s": recovar_wall_s,
            "relion_wall_s": relion_wall_s,
            "recovar_peak_hbm_mib": _peak_hbm_mib(case_root / "recovar_gpu_monitor.csv"),
            "relion_peak_hbm_mib": _peak_hbm_mib(case_root / "relion_gpu_monitor.csv"),
            "hardware_comparable": True,
            "recovar_over_relion_wall_ratio": recovar_wall_s / relion_wall_s,
        },
        "outcome": outcome,
        "_audit_job_id": audit_job_id,
    }


def _reproduction_environment(env: dict[str, str]) -> dict[str, Any]:
    names = (
        "REPO_ROOT",
        "EXPECTED_GIT_HEAD",
        "EM_KCLASS_MATRIX_PIXI_PY",
        "RECOVAR_CUDA_LIB",
        "RECOVAR_RELION_BIND_BUILD_DIR",
        "RELION_SRC_DIR",
        "EM_KCLASS_MATRIX_RELION_REFINE_MPI",
        "RELION_MODULE",
        "RELION_MPI_RANKS",
        "KCLASS_IMAGE_BATCH_SIZE",
        "KCLASS_ROTATION_BLOCK_SIZE",
        "EM_KCLASS_MATRIX_GT_ALIGN_REFINE_ORDERS",
        "SBATCH_ACCOUNT",
        "SBATCH_PARTITION",
        "SBATCH_CONSTRAINT",
    )
    values = {
        "PYTHONNOUSERSITE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PIXI_FROZEN": "true",
    }
    values.update({name: env[name] for name in names})
    return {
        "set": values,
        "confirmed_unset": [
            "PYTHONPATH",
            "PYTHONHOME",
            "CONDA_PREFIX",
            "VIRTUAL_ENV",
            "TF_GPU_ALLOCATOR",
            "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT",
            "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER",
            "EM_KCLASS_MATRIX_EXCLUSIVE",
        ],
    }


def _build_campaign(
    run_root: Path,
    *,
    n_classes: int,
    seeds: tuple[int, ...],
    campaign_id: str,
    recorded_at: str,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    rows = _rows_for_k(run_root, n_classes, seeds)
    audit_jobs = _audit_jobs(run_root, n_classes, seeds)
    env = _parse_env(run_root / "submission.env")
    recovar_source = _git_source(Path(env["REPO_ROOT"]))
    if recovar_source["commit"] != env["EXPECTED_GIT_HEAD"]:
        raise ValueError("RECOVAR checkout HEAD conflicts with submission provenance")
    relion_checkout = Path(env["RELION_SRC_DIR"]).resolve().parent
    relion_diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=relion_checkout)
    source = {
        "recovar": recovar_source,
        "relion": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=relion_checkout, text=True).strip(),
            "tree": subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=relion_checkout, text=True
            ).strip(),
            "checkout": str(relion_checkout),
            "source_diff_sha256": _sha256_bytes(relion_diff),
            "executable": _file_ref(Path(env["EM_KCLASS_MATRIX_RELION_REFINE_MPI"])),
        },
        "custom_cuda": _file_ref(Path(env["RECOVAR_CUDA_LIB"])),
    }
    cases = [
        _build_case(
            run_root,
            row,
            audit_jobs[int(row["seed"])],
            n_classes=n_classes,
        )
        for row in rows
    ]
    iterations = int(cases[0]["configuration"]["max_iter"])
    if any(int(case["configuration"]["max_iter"]) != iterations for case in cases):
        raise ValueError("campaign replicates do not share one iteration cap")
    audit_job_refs = []
    for case in cases:
        seed = int(case["configuration"]["seed"])
        audit_job_id = case.pop("_audit_job_id")
        audit_job_refs.append(
            _job_reference(
                audit_job_id,
                role=f"k{n_classes}_seed_{seed}_trajectory_audit",
                logs=[
                    _job_logs(run_root, audit_job_id, "out"),
                    _job_logs(run_root, audit_job_id, "err"),
                ],
                gpu_model=None,
                gpu_uuid=None,
            )
        )
    setup_job_id = env["SETUP_JOB_ID"]
    shared_jobs = [
        _job_reference(
            setup_job_id,
            role="environment_setup",
            logs=[run_root / "em_kclass_matrix_setup.out", run_root / "em_kclass_matrix_setup.err"],
            gpu_model=None,
            gpu_uuid=None,
        ),
        *audit_job_refs,
    ]
    setup_launcher = run_root / "jobs" / "em_kclass_matrix_setup.sh"
    summary_launcher = run_root / "jobs" / "em_kclass_matrix_summary.sh"
    audit_launcher = run_root / "jobs" / "audit_kclass_trajectory.sbatch"
    shared_artifacts = _existing_refs(
        [
            ("safe_to_delete_marker", run_root / "SAFE_TO_DELETE"),
            ("case_table", run_root / "case_table.tsv"),
            ("submission_environment", run_root / "submission.env"),
            ("audit_jobs", run_root / "audit_jobs.tsv"),
            ("setup_launcher", setup_launcher),
            ("summary_launcher", summary_launcher),
            ("trajectory_audit_launcher", audit_launcher),
            ("setup_stdout", run_root / "em_kclass_matrix_setup.out"),
            ("setup_stderr", run_root / "em_kclass_matrix_setup.err"),
        ]
    )
    shared_artifacts.extend(_source_structures(Path(rows[0]["case_root"]), n_classes=n_classes))
    base_name = str(cases[0]["configuration"]["base_name"])
    case_template = str(Path(rows[0]["script"]).resolve()).replace(str(seeds[0]), "<SEED>")
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "recorded_at": recorded_at,
        "record_type": "synthetic_kclass_campaign",
        "run_root": str(run_root),
        "subject": {
            "kind": "synthetic",
            "molecular_family": "CryoBench2 Ribosembly 50S assembly intermediates",
            "k": n_classes,
            "iterations": iterations,
            "symmetry": str(cases[0]["configuration"]["symmetry"]),
            "trajectory_scope": (
                f"Three independent {base_name} seeds; every numbered RECOVAR half-map "
                f"average and RELION Class3D full map at iterations 1-{iterations}, plus "
                "last-numbered final class maps."
            ),
        },
        "source": source,
        "reproduction": {
            "setup_command": f"sbatch {setup_launcher.resolve()}",
            "case_command_template": f"sbatch {case_template}",
            "summary_command": f"sbatch {summary_launcher.resolve()}",
            "trajectory_audit_command": (
                f"sbatch --export=ALL,CASE_ROOT=<CASE_ROOT>,N_CLASSES={n_classes} {audit_launcher.resolve()}"
            ),
            "environment": _reproduction_environment(env),
        },
        "shared_jobs": shared_jobs,
        "shared_artifacts": shared_artifacts,
        "cases": cases,
        "limitations": [
            (
                f"This is a three-seed synthetic C1 K={n_classes}, box-128, "
                f"{iterations}-iteration campaign; it does not replace real-data half-map "
                "validation or a convergence-to-stopping-criterion study."
            ),
            (
                "RECOVAR did not converge before the cap and final-all-data reconstruction "
                "was not forced; final_class maps equal the last numbered half-map averages."
            ),
            (
                "RELION Class3D emits full class maps rather than independent half maps, so "
                "masked half-map resolution is outside this synthetic campaign's scope."
            ),
            (
                "The trajectory gate uses direct FSC-AUC, signed per-class GT FSC-AUC delta, "
                "and assignment agreement. SCIENCE_EQUIVALENT means every GT-delta cell "
                "passes even though a stricter trajectory criterion fails."
            ),
            (
                "Per-class 0.143 GT resolutions come from the independently rigid-aligned "
                "endpoint evaluator; final FSC-AUC values come from the frozen trajectory audit."
            ),
            "Peak HBM is a lower bound sampled every five seconds, not an allocator-exact maximum.",
            (
                "External run roots are disposable and carry SAFE_TO_DELETE markers; all "
                "source maps consumed by the audit are pinned in per-case checksum manifests."
            ),
        ],
        "verification": {
            "evidence_checked_at": recorded_at,
            "commands": [
                (
                    "pixi run python scripts/seal_em_kclass_campaign.py "
                    f"--run-root {run_root} --n-classes {n_classes} "
                    f"--campaign-id {campaign_id} --output <OUTPUT>"
                ),
                "pixi run python scripts/validate_em_benchmark_registry.py",
                "pixi run python scripts/validate_em_benchmark_registry.py --verify-files",
                "pixi run pytest tests/unit/test_validate_em_benchmark_registry.py",
            ],
        },
    }


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item)
    if not seeds or len(seeds) != len(set(seeds)) or tuple(sorted(seeds)) != seeds:
        raise argparse.ArgumentTypeError("seeds must be a nonempty sorted unique CSV")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--n-classes", required=True, type=int)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--recorded-at",
        default=None,
        help="UTC ISO timestamp; defaults to the current time.",
    )
    args = parser.parse_args()
    if args.n_classes < 2:
        parser.error("--n-classes must be at least 2")
    recorded_at = args.recorded_at or datetime.now(timezone.utc).isoformat()
    campaign = _build_campaign(
        args.run_root,
        n_classes=args.n_classes,
        seeds=args.expected_seeds,
        campaign_id=args.campaign_id,
        recorded_at=recorded_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(campaign, indent=2, sort_keys=False) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
