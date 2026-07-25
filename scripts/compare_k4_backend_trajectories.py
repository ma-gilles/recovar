#!/usr/bin/env python3
"""Compare two complete K=4 FSC trajectory audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FSC_SCHEMA = "em_k4_fsc_trajectory_audit_v2"
TOPOLOGY_SCHEMA = "em_k4_control_topology_audit_v1"
OUTPUT_SCHEMA = "em_k4_backend_trajectory_comparison_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"{path} does not contain a JSON object")
    return value


def summarize_backend(
    fsc_audit: dict[str, Any],
    topology_audit: dict[str, Any],
    walltime: dict[str, Any],
    *,
    direct_fsc_auc_gate: float,
) -> dict[str, Any]:
    _require(fsc_audit.get("schema") == FSC_SCHEMA, "unexpected FSC audit schema")
    _require(
        topology_audit.get("schema") == TOPOLOGY_SCHEMA,
        "unexpected topology audit schema",
    )
    iterations = fsc_audit.get("numbered_iterations")
    _require(isinstance(iterations, list) and iterations, "FSC audit has no iterations")
    _require(
        fsc_audit.get("numbered_iteration_count") == len(iterations),
        "FSC iteration count is inconsistent",
    )

    trajectory: list[dict[str, Any]] = []
    total_gate_checks = 0
    passed_gate_checks = 0
    for expected_iteration, iteration in enumerate(iterations, start=1):
        _require(
            iteration.get("relion_iteration") == expected_iteration,
            "FSC iterations are not contiguous",
        )
        classes = iteration.get("classes")
        _require(isinstance(classes, list) and classes, "FSC iteration has no classes")
        cross_fsc_auc = [
            float(class_row["cross_engine"]["fsc_auc"]) for class_row in classes
        ]
        gt_fsc_auc_delta = [
            float(class_row["gt_fsc_auc_delta"]) for class_row in classes
        ]
        gate_passes = [value >= direct_fsc_auc_gate for value in cross_fsc_auc]
        total_gate_checks += len(gate_passes)
        passed_gate_checks += sum(gate_passes)
        agreement = iteration["class_agreement"]
        class_agreement = (
            float(agreement["agreement"])
            if agreement.get("status") == "available"
            else None
        )
        trajectory.append(
            {
                "relion_iteration": expected_iteration,
                "cross_engine_fsc_auc": cross_fsc_auc,
                "direct_fsc_auc_gate_pass": gate_passes,
                "all_classes_pass_direct_fsc_auc_gate": all(gate_passes),
                "min_cross_engine_fsc_auc": min(cross_fsc_auc),
                "mean_cross_engine_fsc_auc": sum(cross_fsc_auc)
                / len(cross_fsc_auc),
                "min_gt_fsc_auc_delta": min(gt_fsc_auc_delta),
                "class_agreement": class_agreement,
            }
        )

    measured_agreements = [
        row["class_agreement"]
        for row in trajectory
        if row["class_agreement"] is not None
    ]
    return {
        "fsc_audit_status": fsc_audit["status"],
        "earliest_fsc_failure": fsc_audit.get("earliest_failure"),
        "topology_audit_status": topology_audit["status"],
        "exact_control_topology": bool(
            topology_audit.get("combined_control_pass", False)
        ),
        "wall_s": int(walltime["wall_s"]),
        "gpu_uuid": walltime.get("gpu_uuid"),
        "numbered_iterations": len(trajectory),
        "direct_fsc_auc_gate": direct_fsc_auc_gate,
        "direct_fsc_auc_checks_passed": passed_gate_checks,
        "direct_fsc_auc_checks_total": total_gate_checks,
        "iterations_all_classes_passed": sum(
            row["all_classes_pass_direct_fsc_auc_gate"] for row in trajectory
        ),
        "min_cross_engine_fsc_auc": min(
            row["min_cross_engine_fsc_auc"] for row in trajectory
        ),
        "min_gt_fsc_auc_delta": min(
            row["min_gt_fsc_auc_delta"] for row in trajectory
        ),
        "min_class_agreement": (
            min(measured_agreements) if measured_agreements else None
        ),
        "trajectory": trajectory,
    }


def compare(
    baseline_fsc: dict[str, Any],
    candidate_fsc: dict[str, Any],
    baseline_topology: dict[str, Any],
    candidate_topology: dict[str, Any],
    baseline_walltime: dict[str, Any],
    candidate_walltime: dict[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
    direct_fsc_auc_gate: float = 0.995,
) -> dict[str, Any]:
    _require(baseline_label != candidate_label, "backend labels must be distinct")
    baseline = summarize_backend(
        baseline_fsc,
        baseline_topology,
        baseline_walltime,
        direct_fsc_auc_gate=direct_fsc_auc_gate,
    )
    candidate = summarize_backend(
        candidate_fsc,
        candidate_topology,
        candidate_walltime,
        direct_fsc_auc_gate=direct_fsc_auc_gate,
    )
    _require(
        baseline["numbered_iterations"] == candidate["numbered_iterations"],
        "backend trajectories have different iteration counts",
    )
    _require(
        baseline["direct_fsc_auc_checks_total"]
        == candidate["direct_fsc_auc_checks_total"],
        "backend trajectories have different class counts",
    )
    baseline_uuid = baseline["gpu_uuid"]
    candidate_uuid = candidate["gpu_uuid"]
    _require(
        baseline_uuid is not None and baseline_uuid == candidate_uuid,
        "backend trajectories did not use the same physical GPU",
    )

    per_iteration = []
    for baseline_row, candidate_row in zip(
        baseline["trajectory"],
        candidate["trajectory"],
        strict=True,
    ):
        _require(
            baseline_row["relion_iteration"] == candidate_row["relion_iteration"],
            "backend iteration identities differ",
        )
        _require(
            len(baseline_row["cross_engine_fsc_auc"])
            == len(candidate_row["cross_engine_fsc_auc"]),
            "backend iteration class counts differ",
        )
        per_iteration.append(
            {
                "relion_iteration": baseline_row["relion_iteration"],
                "candidate_minus_baseline_min_cross_engine_fsc_auc": (
                    candidate_row["min_cross_engine_fsc_auc"]
                    - baseline_row["min_cross_engine_fsc_auc"]
                ),
                "candidate_minus_baseline_mean_cross_engine_fsc_auc": (
                    candidate_row["mean_cross_engine_fsc_auc"]
                    - baseline_row["mean_cross_engine_fsc_auc"]
                ),
                "candidate_minus_baseline_min_gt_fsc_auc_delta": (
                    candidate_row["min_gt_fsc_auc_delta"]
                    - baseline_row["min_gt_fsc_auc_delta"]
                ),
                "candidate_minus_baseline_class_agreement": (
                    candidate_row["class_agreement"]
                    - baseline_row["class_agreement"]
                    if candidate_row["class_agreement"] is not None
                    and baseline_row["class_agreement"] is not None
                    else None
                ),
            }
        )

    gate_delta = (
        candidate["direct_fsc_auc_checks_passed"]
        - baseline["direct_fsc_auc_checks_passed"]
    )
    if gate_delta > 0:
        classification = "candidate_improves_fixed_direct_fsc_auc_gate_count"
    elif gate_delta < 0:
        classification = "candidate_regresses_fixed_direct_fsc_auc_gate_count"
    else:
        classification = "candidate_preserves_fixed_direct_fsc_auc_gate_count"
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "classification": classification,
        "quality_metric_policy": "shellwise FSC/FSC-AUC; correlation is not used",
        "same_physical_gpu": True,
        "backends": {
            baseline_label: baseline,
            candidate_label: candidate,
        },
        "candidate_minus_baseline": {
            "direct_fsc_auc_checks_passed": gate_delta,
            "iterations_all_classes_passed": (
                candidate["iterations_all_classes_passed"]
                - baseline["iterations_all_classes_passed"]
            ),
            "wall_s": candidate["wall_s"] - baseline["wall_s"],
            "per_iteration": per_iteration,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-fsc", type=Path, required=True)
    parser.add_argument("--candidate-fsc", type=Path, required=True)
    parser.add_argument("--baseline-topology", type=Path, required=True)
    parser.add_argument("--candidate-topology", type=Path, required=True)
    parser.add_argument("--baseline-walltime", type=Path, required=True)
    parser.add_argument("--candidate-walltime", type=Path, required=True)
    parser.add_argument("--baseline-label", default="host_numpy")
    parser.add_argument("--candidate-label", default="relion_cuda")
    parser.add_argument("--direct-fsc-auc-gate", type=float, default=0.995)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare(
        _load_json(args.baseline_fsc),
        _load_json(args.candidate_fsc),
        _load_json(args.baseline_topology),
        _load_json(args.candidate_topology),
        _load_json(args.baseline_walltime),
        _load_json(args.candidate_walltime),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        direct_fsc_auc_gate=args.direct_fsc_auc_gate,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
