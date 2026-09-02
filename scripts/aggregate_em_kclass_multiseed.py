#!/usr/bin/env python3
"""Validate and aggregate a frozen three-seed K-class robustness suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "recovar.em.kclass_multiseed_summary.v1"
TRAJECTORY_SCHEMA = "recovar.em.kclass_multiseed_summary.v2"
K4_TRAJECTORY_AUDIT_SCHEMA = "em_k4_fsc_trajectory_audit_v2"
KCLASS_TRAJECTORY_AUDIT_SCHEMA = "em_kclass_fsc_trajectory_audit_v1"
TRAJECTORY_METRIC_POLICY = "shellwise FSC and normalized FSC-AUC only; correlation is not computed"
DEFAULT_EXPECTED_SEEDS = (41001, 41002, 41003)
SCIENCE_COLUMNS = (
    "index",
    "pdb_dir",
    "n_classes",
    "n_images",
    "grid",
    "noise_level",
    "noise_model",
    "dataset_params_option",
    "pdb_bfactor",
    "init_radius",
    "noise_scale_std",
    "contrast_std",
    "volume_radius",
    "image_offset_n_std",
    "percent_outliers",
    "max_iter",
    "class_distribution",
    "image_batch_size_override",
    "rotation_block_size_override",
    "symmetry",
    "base_seed",
    "shared_input_group",
    "shared_input_role",
)
REQUIRED_COLUMNS = {
    *SCIENCE_COLUMNS,
    "name",
    "seed",
    "base_name",
    "seed_replicate",
    "case_root",
    "script",
    "job_id",
}
COLLAPSE_MARKERS = (
    "class-collapse gate failed",
    "class direction prior row must have positive mass",
    "zero direction-prior row",
    "collapsed class",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expected_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"expected seeds must be comma-separated integers: {value!r}") from error
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(f"exactly three distinct expected seeds are required, got {seeds}")
    return seeds


def read_case_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    if not rows:
        raise ValueError(f"case table is empty: {path}")
    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    if missing:
        raise ValueError(f"case table is missing required columns {missing}: {path}")
    return rows


def read_matrix_cases(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "em_robustness_matrix_summary_v1":
        raise ValueError(f"unexpected matrix summary schema in {path}: {payload.get('schema')!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"matrix summary cases must be a list: {path}")
    by_root: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not case.get("case_root"):
            raise ValueError(f"matrix summary contains a malformed case row: {case!r}")
        root = str(Path(str(case["case_root"])).resolve())
        if root in by_root:
            raise ValueError(f"matrix summary contains duplicate case_root {root}")
        by_root[root] = case
    return by_root


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def reduced_metric(rows: list[dict[str, Any]], key: str, reduction: str) -> float | None:
    values = [value for row in rows if (value := finite_number(row.get(key))) is not None]
    if not values:
        return None
    if reduction == "min":
        return min(values)
    if reduction == "max":
        return max(values)
    if reduction == "median":
        return float(statistics.median(values))
    raise ValueError(f"unknown reduction {reduction!r}")


def read_population_audit(case_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = case_root / "relion_ref" / "class_population_audit.json"
    if not path.is_file():
        return None, None
    payload = json.loads(path.read_text())
    if payload.get("schema") != "recovar.em.relion_class_population_audit.v1":
        raise ValueError(f"unexpected class-population audit schema: {path}")
    return payload, sha256_file(path)


def read_trajectory_audit(
    case_root: Path,
    *,
    n_classes: int,
    expected_iterations: int,
) -> dict[str, Any]:
    """Load and independently replay one strict per-class trajectory audit."""

    path = case_root / "trajectory_analysis" / f"k{n_classes}_fsc_trajectory.json"
    if not path.is_file():
        raise ValueError(f"required K={n_classes} trajectory audit is missing: {path}")
    payload = json.loads(path.read_text())
    expected_schema = K4_TRAJECTORY_AUDIT_SCHEMA if n_classes == 4 else KCLASS_TRAJECTORY_AUDIT_SCHEMA
    if payload.get("schema") != expected_schema:
        raise ValueError(
            f"unexpected K={n_classes} trajectory audit schema in {path}: "
            f"{payload.get('schema')!r}"
        )
    if payload.get("n_classes") != n_classes:
        raise ValueError(f"trajectory audit K disagrees with the case configuration: {path}")
    if payload.get("numbered_iteration_count") != expected_iterations:
        raise ValueError(
            f"trajectory audit must contain {expected_iterations} numbered iterations: {path}"
        )
    numbered = payload.get("numbered_iterations")
    if not isinstance(numbered, list) or len(numbered) != expected_iterations:
        raise ValueError(f"malformed numbered trajectory in {path}")
    if [row.get("relion_iteration") for row in numbered] != list(range(1, expected_iterations + 1)):
        raise ValueError(f"trajectory audit iteration identity/order changed: {path}")
    if payload.get("quality_metric_policy") != TRAJECTORY_METRIC_POLICY:
        raise ValueError(f"trajectory audit metric policy changed: {path}")

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"trajectory audit thresholds are missing: {path}")
    direct_min = finite_number(thresholds.get("per_class_direct_fsc_auc_min"))
    gt_delta_min = finite_number(thresholds.get("per_class_recovar_minus_relion_gt_fsc_auc_min"))
    agreement_min = finite_number(thresholds.get("class_assignment_agreement_min_when_available"))
    if None in (direct_min, gt_delta_min, agreement_min):
        raise ValueError(f"trajectory audit thresholds are malformed: {path}")
    if not (-1 <= direct_min <= 1 and -2 <= gt_delta_min <= 2 and 0 <= agreement_min <= 1):
        raise ValueError(f"trajectory audit thresholds are outside their valid ranges: {path}")

    direct_values: list[float] = []
    gt_delta_values: list[float] = []
    class_gate_results: list[bool] = []
    agreement_values: list[float] = []

    def validate_class_permutation(
        classes: list[dict[str, Any]],
        key: str,
        *,
        label: str,
    ) -> None:
        values = [row.get(key) for row in classes]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"trajectory audit {label} has invalid {key} values: {path}")
        if sorted(values) != list(range(1, n_classes + 1)):
            raise ValueError(f"trajectory audit {label} {key} topology changed: {path}")

    def replay_class_metrics(class_row: dict[str, Any], *, label: str) -> tuple[float, float]:
        direct = finite_number(class_row.get("cross_engine", {}).get("fsc_auc"))
        delta = finite_number(class_row.get("gt_fsc_auc_delta"))
        recovar_gt = finite_number(class_row.get("vs_gt", {}).get("recovar", {}).get("fsc_auc"))
        relion_gt = finite_number(class_row.get("vs_gt", {}).get("relion", {}).get("fsc_auc"))
        if direct is None or not -1 <= direct <= 1:
            raise ValueError(f"trajectory audit {label} has invalid direct FSC-AUC: {path}")
        if recovar_gt is None or relion_gt is None or not -1 <= recovar_gt <= 1 or not -1 <= relion_gt <= 1:
            raise ValueError(f"trajectory audit {label} has invalid GT FSC-AUC: {path}")
        if delta is None or not -2 <= delta <= 2:
            raise ValueError(f"trajectory audit {label} has invalid GT FSC-AUC delta: {path}")
        if not math.isclose(delta, recovar_gt - relion_gt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"trajectory audit {label} signed GT FSC-AUC delta is inconsistent: {path}")
        direct_values.append(direct)
        gt_delta_values.append(delta)
        class_gate_results.append(direct >= direct_min and delta >= gt_delta_min)
        return direct, delta

    for expected_iteration, row in enumerate(numbered, start=1):
        classes = row.get("classes")
        if not isinstance(classes, list) or len(classes) != n_classes:
            raise ValueError(
                f"trajectory audit iteration {expected_iteration} must contain K class rows: {path}"
            )
        label = f"iteration {expected_iteration}"
        for key in ("recovar_class", "relion_class", "gt_class"):
            validate_class_permutation(classes, key, label=label)
        for class_row in classes:
            replay_class_metrics(class_row, label=label)
        agreement = row.get("class_agreement", {})
        if agreement.get("status") == "available":
            value = finite_number(agreement.get("agreement"))
            if value is None:
                raise ValueError(
                    f"trajectory audit iteration {expected_iteration} has invalid class agreement: {path}"
                )
            if not 0 <= value <= 1:
                raise ValueError(
                    f"trajectory audit iteration {expected_iteration} class agreement is out of range: {path}"
                )
            agreement_values.append(value)
        elif agreement.get("status") != "unavailable":
            raise ValueError(
                f"trajectory audit iteration {expected_iteration} has invalid agreement status: {path}"
            )

    final = payload.get("final")
    final_classes = final.get("classes") if isinstance(final, dict) else None
    if not isinstance(final_classes, list) or len(final_classes) != n_classes:
        raise ValueError(f"trajectory audit final state must contain K class rows: {path}")
    for key in ("recovar_class", "relion_class", "gt_class"):
        validate_class_permutation(final_classes, key, label="final state")
    final_gt_deltas: list[float] = []
    for row in final_classes:
        _, delta = replay_class_metrics(row, label="final state")
        final_gt_deltas.append(delta)

    failures = payload.get("failures")
    if not isinstance(failures, list):
        raise ValueError(f"trajectory audit failures must be a list: {path}")
    replay_pass = (
        all(value >= direct_min for value in direct_values)
        and all(value >= gt_delta_min for value in gt_delta_values)
        and all(value >= agreement_min for value in agreement_values)
    )
    expected_status = "pass" if replay_pass else "fail"
    if payload.get("status") != expected_status:
        raise ValueError(f"trajectory audit status does not replay from its recorded metrics: {path}")
    if (not failures) != replay_pass:
        raise ValueError(f"trajectory audit failures conflict with its recorded status: {path}")
    if payload.get("earliest_failure") != (None if replay_pass else failures[0]):
        raise ValueError(f"trajectory audit earliest failure conflicts with its failures: {path}")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": payload["schema"],
        "status": expected_status.upper(),
        "thresholds": {
            "direct_fsc_auc_min": direct_min,
            "final_gt_fsc_auc_delta_min": gt_delta_min,
            "class_assignment_agreement_min_when_available": agreement_min,
        },
        "evaluated_iterations": len(numbered),
        "evaluated_class_cells": len(direct_values),
        "passing_class_cells": sum(class_gate_results),
        "minimum_direct_fsc_auc": min(direct_values),
        "minimum_gt_fsc_auc_delta": min(gt_delta_values),
        "minimum_final_gt_fsc_auc_delta": min(final_gt_deltas),
        "minimum_class_assignment_agreement": min(agreement_values) if agreement_values else None,
        "class_agreement_unavailable_iterations": [
            row["relion_iteration"]
            for row in numbered
            if row.get("class_agreement", {}).get("status") == "unavailable"
        ],
        "earliest_failure": payload.get("earliest_failure"),
    }


def read_and_validate_case_config(
    row: dict[str, str],
    case_root: Path,
    summary: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Bind a table row to the runtime-written immutable case identity."""

    path = case_root / "case_config.json"
    if not path.is_file():
        return None, None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"case_config.json must contain an object: {path}")
    expected: dict[str, Any] = {
        "index": int(row["index"]),
        "name": row["name"],
        "pdb_dir": row["pdb_dir"],
        "n_classes": int(row["n_classes"]),
        "n_images": int(row["n_images"]),
        "grid_size": int(row["grid"]),
        "noise_level": float(row["noise_level"]),
        "noise_model": row["noise_model"],
        "dataset_params_option": row["dataset_params_option"],
        "class_distribution": row["class_distribution"],
        "seed": int(row["seed"]),
        "pdb_bfactor": float(row["pdb_bfactor"]),
        "init_radius": int(row["init_radius"]),
        "noise_scale_std": float(row["noise_scale_std"]),
        "contrast_std": float(row["contrast_std"]),
        "volume_radius": float(row["volume_radius"]),
        "image_offset_n_std": float(row["image_offset_n_std"]),
        "percent_outliers": float(row["percent_outliers"]),
        "max_iter": int(row["max_iter"]),
        "symmetry": row["symmetry"],
        "base_name": row["base_name"],
        "base_seed": int(row["base_seed"]),
        "seed_replicate": int(row["seed_replicate"]),
        "shared_input_group": row["shared_input_group"] or None,
        "shared_input_role": row["shared_input_role"] or None,
        "case_root": str(case_root),
        "slurm_job_id": row["job_id"],
    }
    if row["image_batch_size_override"]:
        expected["image_batch_size"] = int(row["image_batch_size_override"])
    if row["rotation_block_size_override"]:
        expected["rotation_block_size"] = int(row["rotation_block_size_override"])
    mismatches = {
        key: {"case_table": value, "case_config": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    summary_name = summary.get("case_name")
    if summary_name is not None and str(summary_name) != row["name"]:
        mismatches["summary.case_name"] = {
            "case_table": row["name"],
            "matrix_summary": summary_name,
        }
    summary_job_id = summary.get("job_id")
    if summary_job_id is not None and str(summary_job_id) != row["job_id"]:
        mismatches["summary.job_id"] = {
            "case_table": row["job_id"],
            "matrix_summary": summary_job_id,
        }
    if mismatches:
        raise ValueError(f"case identity mismatch for {case_root}: {mismatches}")
    return payload, sha256_file(path)


def classify_replicate(
    summary: dict[str, Any],
    population_audit: dict[str, Any] | None,
) -> tuple[str, str | None]:
    failure_text = " ".join(
        str(summary.get(key) or "") for key in ("failure_reason", "log_excerpt", "recovar_latest_stage")
    ).lower()
    audit_collapsed = bool(population_audit and population_audit.get("collapsed"))
    if audit_collapsed or any(marker in failure_text for marker in COLLAPSE_MARKERS):
        return "FAIL_CLASS_COLLAPSE", "one or more RELION classes have non-positive population"
    status = str(summary.get("status") or "pending").lower()
    if status == "ok":
        if population_audit is None:
            return "INCOMPLETE", "completed row lacks the required class-population audit"
        if population_audit.get("passed") is not True:
            return "FAILED", "class-population audit did not explicitly pass"
        return "COMPLETE", None
    if status == "failed":
        return "FAILED", str(summary.get("failure_reason") or "matrix case failed")
    return "INCOMPLETE", f"matrix case status is {status!r}"


def group_outcome(replicates: list[dict[str, Any]]) -> str:
    outcomes = {str(row["outcome"]) for row in replicates}
    if outcomes == {"COMPLETE"}:
        return "COMPLETE_ALL_SEEDS"
    if "FAIL_CLASS_COLLAPSE" in outcomes:
        return "FAIL_CLASS_COLLAPSE"
    if "FAILED" in outcomes:
        return "FAILED"
    return "INCOMPLETE"


def overall_outcome(groups: list[dict[str, Any]]) -> str:
    outcomes = {str(group["outcome"]) for group in groups}
    if outcomes == {"COMPLETE_ALL_SEEDS"}:
        return "COMPLETE_ALL_CASES_ALL_SEEDS"
    if "FAIL_CLASS_COLLAPSE" in outcomes:
        return "FAIL_CLASS_COLLAPSE"
    if "FAILED" in outcomes:
        return "FAILED"
    return "INCOMPLETE"


def aggregate(
    scratch_root: Path,
    *,
    matrix_summary: Path,
    case_table: Path,
    expected_seeds: tuple[int, ...] = DEFAULT_EXPECTED_SEEDS,
    require_trajectory_audits: bool = False,
) -> dict[str, Any]:
    scratch_root = scratch_root.resolve()
    matrix_summary = matrix_summary.resolve()
    case_table = case_table.resolve()
    table_rows = read_case_table(case_table)
    summary_by_root = read_matrix_cases(matrix_summary)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in table_rows:
        if not row["seed_replicate"].strip():
            raise ValueError("multi-seed aggregation requires every case-table row to have seed_replicate")
        groups[row["base_name"]].append(row)

    output_groups: list[dict[str, Any]] = []
    expected_seed_set = set(expected_seeds)
    for base_name in sorted(groups):
        rows = sorted(groups[base_name], key=lambda row: int(row["seed_replicate"]))
        observed_seeds = [int(row["seed"]) for row in rows]
        observed_replicates = [int(row["seed_replicate"]) for row in rows]
        if len(rows) != len(expected_seeds) or set(observed_seeds) != expected_seed_set:
            raise ValueError(f"{base_name} must contain seeds {expected_seeds}, got {tuple(observed_seeds)}")
        if observed_replicates != list(range(1, len(expected_seeds) + 1)):
            raise ValueError(
                f"{base_name} must contain seed_replicates 1..{len(expected_seeds)}, got {observed_replicates}"
            )
        if tuple(observed_seeds) != expected_seeds:
            raise ValueError(
                f"{base_name} must map seed_replicates 1..{len(expected_seeds)} to "
                f"seeds {expected_seeds} in order, got {tuple(observed_seeds)}"
            )
        signature = tuple(rows[0][column] for column in SCIENCE_COLUMNS)
        for row in rows[1:]:
            candidate = tuple(row[column] for column in SCIENCE_COLUMNS)
            if candidate != signature:
                changed = [
                    column
                    for column, left, right in zip(SCIENCE_COLUMNS, signature, candidate, strict=True)
                    if left != right
                ]
                raise ValueError(f"{base_name} changes frozen scientific columns across seeds: {changed}")

        replicates: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for row in rows:
            case_root = Path(row["case_root"]).resolve()
            summary = summary_by_root.get(str(case_root))
            if summary is None:
                raise ValueError(f"matrix summary is missing case root {case_root}")
            case_config, case_config_sha256 = read_and_validate_case_config(row, case_root, summary)
            population_audit, population_audit_sha256 = read_population_audit(case_root)
            outcome, reason = classify_replicate(summary, population_audit)
            if str(summary.get("status") or "pending").lower() == "ok" and case_config is None:
                outcome = "INCOMPLETE"
                reason = "completed row lacks the runtime-written case_config.json"
            summaries.append(summary)
            replicates.append(
                {
                    "seed": int(row["seed"]),
                    "seed_replicate": int(row["seed_replicate"]),
                    "case_name": row["name"],
                    "case_root": str(case_root),
                    "job_id": row["job_id"],
                    "outcome": outcome,
                    "reason": reason,
                    "matrix_status": summary.get("status"),
                    "slurm_state": summary.get("slurm_state"),
                    "slurm_exit_code": summary.get("slurm_exit_code"),
                    "case_config": str(case_root / "case_config.json") if case_config is not None else None,
                    "case_config_sha256": case_config_sha256,
                    "recovar_fsc_auc_vs_gt": finite_number(summary.get("fsc_auc_vs_gt")),
                    "relion_fsc_auc_vs_gt": finite_number(summary.get("relion_fsc_auc_vs_gt")),
                    "recovar_minus_relion_gt_fsc_auc": finite_number(summary.get("fsc_auc_delta_vs_relion")),
                    "recovar_wall_s": finite_number(summary.get("wall_s")),
                    "relion_wall_s": finite_number(summary.get("relion_wall_s")),
                    "recovar_peak_hbm_mib": finite_number(summary.get("recovar_peak_gpu_memory_mib")),
                    "relion_peak_hbm_mib": finite_number(summary.get("relion_peak_gpu_memory_mib")),
                    "slurm_max_rss_mib": finite_number(summary.get("slurm_max_rss_mib")),
                    "class_population_audit": (
                        str(case_root / "relion_ref" / "class_population_audit.json")
                        if population_audit is not None
                        else None
                    ),
                    "class_population_audit_sha256": population_audit_sha256,
                    **(
                        {
                            "trajectory_audit": read_trajectory_audit(
                                case_root,
                                n_classes=int(row["n_classes"]),
                                expected_iterations=int(row["max_iter"]),
                            )
                        }
                        if require_trajectory_audits
                        else {}
                    ),
                }
            )

        trajectory_audits = (
            [replicate["trajectory_audit"] for replicate in replicates]
            if require_trajectory_audits
            else []
        )

        output_groups.append(
            {
                "base_name": base_name,
                "base_index": int(rows[0]["index"]),
                "base_seed": int(rows[0]["base_seed"]),
                "symmetry": rows[0]["symmetry"],
                "scientific_contract": {column: rows[0][column] for column in SCIENCE_COLUMNS},
                "outcome": group_outcome(replicates),
                "metrics_across_seeds": {
                    "min_recovar_fsc_auc_vs_gt": reduced_metric(summaries, "fsc_auc_vs_gt", "min"),
                    "min_relion_fsc_auc_vs_gt": reduced_metric(summaries, "relion_fsc_auc_vs_gt", "min"),
                    "worst_recovar_minus_relion_gt_fsc_auc": reduced_metric(
                        summaries, "fsc_auc_delta_vs_relion", "min"
                    ),
                    "median_recovar_wall_s": reduced_metric(summaries, "wall_s", "median"),
                    "median_relion_wall_s": reduced_metric(summaries, "relion_wall_s", "median"),
                    "max_recovar_peak_hbm_mib": reduced_metric(summaries, "recovar_peak_gpu_memory_mib", "max"),
                    "max_relion_peak_hbm_mib": reduced_metric(summaries, "relion_peak_gpu_memory_mib", "max"),
                    "max_slurm_rss_mib": reduced_metric(summaries, "slurm_max_rss_mib", "max"),
                    **(
                        {
                            "minimum_direct_fsc_auc": min(
                                row["minimum_direct_fsc_auc"] for row in trajectory_audits
                            ),
                            "minimum_final_gt_fsc_auc_delta": min(
                                row["minimum_final_gt_fsc_auc_delta"]
                                for row in trajectory_audits
                            ),
                            "minimum_gt_fsc_auc_delta": min(
                                row["minimum_gt_fsc_auc_delta"] for row in trajectory_audits
                            ),
                            "minimum_class_assignment_agreement": (
                                min(agreement_values)
                                if (
                                    agreement_values := [
                                        value
                                        for row in trajectory_audits
                                        if (value := row["minimum_class_assignment_agreement"])
                                        is not None
                                    ]
                                )
                                else None
                            ),
                            "passing_trajectory_class_cells": sum(
                                row["passing_class_cells"] for row in trajectory_audits
                            ),
                            "evaluated_trajectory_class_cells": sum(
                                row["evaluated_class_cells"] for row in trajectory_audits
                            ),
                        }
                        if trajectory_audits
                        else {}
                    ),
                },
                **(
                    {
                        "formal_trajectory_status": (
                            "PASS"
                            if all(row["status"] == "PASS" for row in trajectory_audits)
                            else "FAIL"
                        )
                    }
                    if trajectory_audits
                    else {}
                ),
                "replicates": replicates,
            }
        )

    formal_gate_claim = None
    formal_gate_claim_reason = (
        "This suite index preserves completion, collapse, GT-FSC, timing, and memory "
        "evidence; per-trajectory Hungarian/direct-FSC records remain separate registry entries."
    )
    if require_trajectory_audits:
        all_audits = [
            replicate["trajectory_audit"]
            for group in output_groups
            for replicate in group["replicates"]
        ]
        formal_gate_claim = {
            "status": "PASS" if all(row["status"] == "PASS" for row in all_audits) else "FAIL",
            "audited_replicates": len(all_audits),
            "passing_replicates": sum(row["status"] == "PASS" for row in all_audits),
            "evaluated_class_cells": sum(row["evaluated_class_cells"] for row in all_audits),
            "passing_class_cells": sum(row["passing_class_cells"] for row in all_audits),
            "audit_sha256": [row["sha256"] for row in all_audits],
        }
        formal_gate_claim_reason = (
            "Every expected seed is bound to a fail-closed permutation-aware trajectory audit; "
            "the formal status is replayed from every numbered and final direct FSC-AUC and signed "
            "GT FSC-AUC cell, plus every available class-assignment agreement."
        )

    return {
        "schema": TRAJECTORY_SCHEMA if require_trajectory_audits else SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scratch_root": str(scratch_root),
        "expected_seeds": list(expected_seeds),
        "base_case_count": len(output_groups),
        "replicate_count": sum(len(group["replicates"]) for group in output_groups),
        "outcome": overall_outcome(output_groups),
        "formal_gate_claim": formal_gate_claim,
        "formal_gate_claim_reason": formal_gate_claim_reason,
        "artifacts": {
            "matrix_summary": str(matrix_summary),
            "matrix_summary_sha256": sha256_file(matrix_summary),
            "case_table": str(case_table),
            "case_table_sha256": sha256_file(case_table),
        },
        "cases": output_groups,
    }


def render_markdown(payload: dict[str, Any], output_json: Path) -> str:
    formal_claim = payload["formal_gate_claim"]
    formal_summary = (
        "not made by this aggregate"
        if formal_claim is None
        else (
            f"{formal_claim['status']} "
            f"({formal_claim['passing_replicates']}/{formal_claim['audited_replicates']} replicates; "
            f"{formal_claim['passing_class_cells']}/{formal_claim['evaluated_class_cells']} class cells)"
        )
    )
    lines = [
        "# K-class three-seed robustness aggregate",
        "",
        f"- JSON: `{output_json}`",
        f"- outcome: **{payload['outcome']}**",
        f"- frozen seeds: `{','.join(str(seed) for seed in payload['expected_seeds'])}`",
        f"- base cases: **{payload['base_case_count']}**",
        f"- replicate rows: **{payload['replicate_count']}**",
        f"- formal trajectory claim: **{formal_summary}**",
        "",
        "| Case | Sym | Outcome | Seeds | Worst REC-REL GT FSC-AUC | Median RECOVAR s | Median RELION s | Max RECOVAR HBM MiB | Max RELION HBM MiB |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        metrics = case["metrics_across_seeds"]
        seeds = ",".join(str(row["seed"]) for row in case["replicates"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["base_name"]),
                    str(case["symmetry"]),
                    str(case["outcome"]),
                    seeds,
                    str(metrics["worst_recovar_minus_relion_gt_fsc_auc"]),
                    str(metrics["median_recovar_wall_s"]),
                    str(metrics["median_relion_wall_s"]),
                    str(metrics["max_recovar_peak_hbm_mib"]),
                    str(metrics["max_relion_peak_hbm_mib"]),
                ]
            )
            + " |"
        )
    if formal_claim is not None:
        lines.extend(
            [
                "",
                "## Strict trajectory gates",
                "",
                "| Case | K | Formal | Class cells | Min direct FSC-AUC | Min signed GT FSC-AUC delta | Min assignment |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for case in payload["cases"]:
            metrics = case["metrics_across_seeds"]
            agreement = metrics["minimum_class_assignment_agreement"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(case["base_name"]),
                        str(case["scientific_contract"]["n_classes"]),
                        str(case["formal_trajectory_status"]),
                        (
                            f"{metrics['passing_trajectory_class_cells']}/"
                            f"{metrics['evaluated_trajectory_class_cells']}"
                        ),
                        f"{metrics['minimum_direct_fsc_auc']:.9f}",
                        f"{metrics['minimum_gt_fsc_auc_delta']:+.9f}",
                        "unavailable" if agreement is None else f"{agreement:.6f}",
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Replicates", ""])
    for case in payload["cases"]:
        for row in case["replicates"]:
            lines.append(
                f"- `{case['base_name']}` seed `{row['seed']}`: **{row['outcome']}**; "
                f"job `{row['job_id']}`; root `{row['case_root']}`"
                + (
                    f"; trajectory: {row['trajectory_audit']['status']} "
                    f"(min direct FSC-AUC {row['trajectory_audit']['minimum_direct_fsc_auc']:.9f})"
                    if "trajectory_audit" in row
                    else ""
                )
                + (f"; reason: {row['reason']}" if row["reason"] else "")
            )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch_root", type=Path)
    parser.add_argument("--matrix-summary", type=Path, default=None)
    parser.add_argument("--case-table", type=Path, default=None)
    parser.add_argument(
        "--expected-seeds",
        default=",".join(str(seed) for seed in DEFAULT_EXPECTED_SEEDS),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--require-trajectory-audits",
        action="store_true",
        help=(
            "require and replay one strict permutation-aware trajectory audit per seed; "
            "emits the v2 aggregate schema and a formal trajectory result"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.scratch_root.expanduser().resolve()
    matrix_summary = (
        args.matrix_summary.expanduser().resolve()
        if args.matrix_summary is not None
        else root / "em_kclass_robustness_summary.json"
    )
    case_table = args.case_table.expanduser().resolve() if args.case_table is not None else root / "case_table.tsv"
    payload = aggregate(
        root,
        matrix_summary=matrix_summary,
        case_table=case_table,
        expected_seeds=parse_expected_seeds(args.expected_seeds),
        require_trajectory_audits=args.require_trajectory_audits,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(render_markdown(payload, args.output_json))
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_markdown}")
    print(f"Outcome: {payload['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
