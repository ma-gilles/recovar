#!/usr/bin/env python3
"""Bind the fixed case-26 map, support, and winner-divergence timeline.

This diagnostic combines five independently sealed reports:

* the shellwise FSC/FSC-AUC trajectory,
* the aligned particle-state distribution,
* the physical-iteration-3 fine top-pair operand factorial,
* the map-to-PPref-to-texture source-boundary replay, and
* the matched x-half precision factorial.

The result is a compact, hash-pinned temporal metric.  It establishes whether
map divergence precedes significant-support drift, which in turn precedes a
hard pose/winner split.  It does not use correlation and does not promote the
fixed scorecard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "em-k1-case26-causal-chain-v1"
TRAJECTORY_SCHEMA = "em_k1_fsc_trajectory_audit_v2"
PARTICLE_STATE_SCHEMA = "em_particle_state_distribution_audit_v1"
OPERAND_SCHEMA = "em-k1-fine-top-pair-operands-v1"
SOURCE_BOUNDARY_SCHEMA = "em-k1-fine-ppref-source-boundary-v1"
PRECISION_FACTORIAL_SCHEMA = "em-k1-case26-xhalf-precision-factorial-v1"

EXPECTED_OPERAND_CLASSIFICATION = "fine_winner_flip_is_projected_reference_determined"
EXPECTED_SOURCE_CLASSIFICATION = "fine_projection_difference_is_iteration_start_map_state"
EXPECTED_PRECISION_CLASSIFICATION = (
    "double_xhalf_mstep_introduces_numbered_failures_and_worsens_case26_final_parity_on_matched_head"
)
CHAIN_CLASSIFICATION = (
    "iteration_map_divergence_precedes_support_then_hard_pose_divergence"
    "__fine_path_inherits_iteration_start_map_state"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _count_above_threshold(summary: dict[str, Any], threshold_key: str) -> int:
    n = int(summary["n"])
    fraction = float(summary["threshold_fractions"][threshold_key])
    _require(n > 0, "distribution count must be positive")
    _require(0.0 <= fraction <= 1.0, "threshold fraction must be in [0, 1]")
    return n - int(round(fraction * n))


def _first_iteration(rows: list[dict[str, Any]], predicate: Any) -> int | None:
    for row in rows:
        if predicate(row):
            return int(row["physical_iteration"])
    return None


def build_report(
    *,
    trajectory_json: Path,
    particle_state_json: Path,
    operand_report_json: Path,
    source_boundary_json: Path,
    precision_factorial_json: Path,
    expected_original_index: int,
) -> dict[str, Any]:
    paths = {
        "trajectory": trajectory_json.resolve(),
        "particle_state": particle_state_json.resolve(),
        "operand_report": operand_report_json.resolve(),
        "source_boundary": source_boundary_json.resolve(),
        "precision_factorial": precision_factorial_json.resolve(),
    }
    for label, path in paths.items():
        _require(path.is_file(), f"missing {label} input: {path}")

    trajectory = _load_json(paths["trajectory"])
    particles = _load_json(paths["particle_state"])
    operands = _load_json(paths["operand_report"])
    source = _load_json(paths["source_boundary"])
    precision = _load_json(paths["precision_factorial"])

    _require(trajectory.get("schema") == TRAJECTORY_SCHEMA, "wrong trajectory schema")
    _require(particles.get("schema") == PARTICLE_STATE_SCHEMA, "wrong particle-state schema")
    _require(operands.get("schema") == OPERAND_SCHEMA, "wrong operand-report schema")
    _require(source.get("schema") == SOURCE_BOUNDARY_SCHEMA, "wrong source-boundary schema")
    _require(precision.get("schema") == PRECISION_FACTORIAL_SCHEMA, "wrong precision-factorial schema")
    _require(particles.get("status") == "pass", "particle-state report did not pass")
    _require(operands.get("status") == "pass", "operand report did not pass")
    _require(source.get("status") == "pass", "source-boundary report did not pass")
    _require(precision.get("status") == "complete", "precision factorial is incomplete")
    _require(
        operands.get("classification") == EXPECTED_OPERAND_CLASSIFICATION,
        "fine operand classification changed",
    )
    _require(
        source.get("classification") == EXPECTED_SOURCE_CLASSIFICATION,
        "fine source-boundary classification changed",
    )
    _require(
        precision.get("classification") == EXPECTED_PRECISION_CLASSIFICATION,
        "x-half precision classification changed",
    )

    operand_identity = operands["identity"]
    source_identity = source["identity"]
    for identity_name, identity in (("operand", operand_identity), ("source", source_identity)):
        _require(
            int(identity["recovar_original_index_zero_based"]) == expected_original_index,
            f"{identity_name} original-index mismatch",
        )
        _require(int(identity["relion_stack_index_one_based"]) == expected_original_index + 1, f"{identity_name} stack-index mismatch")
    _require(
        operand_identity["ordered_top_keys"] == source_identity["ordered_top_keys"],
        "fine-report top-pair identity mismatch",
    )
    _require(
        int(operand_identity["current_size"]) == int(source_identity["current_size"]),
        "fine-report current-size mismatch",
    )

    precision_control_audit = Path(precision["paths"]["control"]["audit"]).resolve()
    _require(
        precision_control_audit == paths["trajectory"],
        "precision factorial does not bind the supplied control trajectory",
    )
    precision_hashes = precision.get("artifact_sha256", {})
    _require(
        precision_hashes.get(str(precision_control_audit)) == _sha256(paths["trajectory"]),
        "precision factorial trajectory hash does not match the supplied artifact",
    )
    _require(
        int(precision["fixed_metric"]["control_numbered_failures"]) == 0,
        "control acquired a numbered-iteration failure",
    )
    _require(
        int(precision["fixed_metric"]["double_numbered_failures"]) > 0,
        "double-precision arm no longer introduces a numbered-iteration failure",
    )
    _require(
        float(precision["fixed_metric"]["double_minus_control_final_cross_engine_fsc_auc"]) < 0.0,
        "double-precision arm no longer worsens final cross-engine FSC-AUC",
    )

    trajectory_by_iteration = {
        int(item["relion_iteration"]): item for item in trajectory["numbered_iterations"]
    }
    particles_by_iteration = {
        int(item["relion_iteration"]): item for item in particles["iterations"]
    }
    _require(len(trajectory_by_iteration) == len(trajectory["numbered_iterations"]), "duplicate trajectory iteration")
    _require(len(particles_by_iteration) == len(particles["iterations"]), "duplicate particle-state iteration")
    _require(trajectory_by_iteration.keys() == particles_by_iteration.keys(), "iteration topology mismatch")
    _require({1, 2, 3} <= trajectory_by_iteration.keys(), "physical iterations 1-3 are required")

    timeline: list[dict[str, Any]] = []
    for physical_iteration in sorted(trajectory_by_iteration):
        trajectory_row = trajectory_by_iteration[physical_iteration]
        particle_row = particles_by_iteration[physical_iteration]
        state = particle_row["recovar_vs_relion"]
        merged_fsc_auc = float(trajectory_row["cross_engine"]["merged"]["fsc_auc"])
        timeline.append(
            {
                "physical_iteration": physical_iteration,
                "merged_cross_engine_fsc_auc": merged_fsc_auc,
                "merged_cross_engine_fsc_loss_from_one": 1.0 - merged_fsc_auc,
                "merged_gt_fsc_auc_delta": float(trajectory_row["merged_gt_fsc_auc_delta"]),
                "significant_support_different_count": int(state["significant_support"]["different_count"]),
                "significant_support_max_abs_candidate_count": float(state["significant_support"]["absolute"]["max"]),
                "pmax_max_abs": float(state["pmax"]["absolute"]["max"]),
                "angular_error_gt_0p01_deg_count": _count_above_threshold(
                    state["angular_error_deg"], "le_0.01"
                ),
                "translation_error_gt_0p01_angstrom_count": _count_above_threshold(
                    state["translation_error"], "le_0.01"
                ),
                "angular_error_max_deg": float(state["angular_error_deg"]["max"]),
                "translation_error_max_angstrom": float(state["translation_error"]["max"]),
            }
        )

    map_divergence_onset = _first_iteration(
        timeline, lambda row: row["merged_cross_engine_fsc_loss_from_one"] > 0.0
    )
    support_divergence_onset = _first_iteration(
        timeline, lambda row: row["significant_support_different_count"] > 0
    )
    hard_pose_divergence_onset = _first_iteration(
        timeline,
        lambda row: row["angular_error_gt_0p01_deg_count"] > 0
        or row["translation_error_gt_0p01_angstrom_count"] > 0,
    )
    _require(map_divergence_onset == 1, "map divergence onset changed")
    _require(support_divergence_onset == 2, "support divergence onset changed")
    _require(hard_pose_divergence_onset == 3, "hard pose divergence onset changed")

    first_three = timeline[:3]
    losses = [float(row["merged_cross_engine_fsc_loss_from_one"]) for row in first_three]
    _require(0.0 < losses[0] < losses[1] < losses[2], "early map FSC losses are not strictly increasing")
    _require(first_three[0]["significant_support_different_count"] == 0, "iteration-1 support changed")
    _require(first_three[1]["significant_support_different_count"] > 0, "iteration-2 support did not change")
    _require(
        first_three[0]["angular_error_gt_0p01_deg_count"] == 0
        and first_three[0]["translation_error_gt_0p01_angstrom_count"] == 0
        and first_three[1]["angular_error_gt_0p01_deg_count"] == 0
        and first_three[1]["translation_error_gt_0p01_angstrom_count"] == 0,
        "hard pose divergence appeared before physical iteration 3",
    )
    _require(
        first_three[2]["angular_error_gt_0p01_deg_count"] > 0
        or first_three[2]["translation_error_gt_0p01_angstrom_count"] > 0,
        "physical iteration 3 has no hard pose divergence",
    )

    return {
        "schema": SCHEMA,
        "status": "pass",
        "classification": CHAIN_CLASSIFICATION,
        "metric_policy": (
            "signed shellwise normalized non-DC FSC-AUC plus exact aligned state-distribution "
            "counts; no correlation; diagnostic only and not a fixed-scorecard promotion"
        ),
        "identity": {
            "case_id": 26,
            "n_images": int(particles["n_images"]),
            "target_original_index_zero_based": expected_original_index,
            "target_relion_stack_index_one_based": expected_original_index + 1,
            "target_top_keys": operand_identity["ordered_top_keys"],
            "target_current_size": int(operand_identity["current_size"]),
        },
        "onsets": {
            "cross_engine_map_fsc_nonidentity_physical_iteration": map_divergence_onset,
            "significant_support_count_divergence_physical_iteration": support_divergence_onset,
            "hard_pose_divergence_physical_iteration": hard_pose_divergence_onset,
        },
        "early_timeline": first_three,
        "fine_boundary_closure": {
            "winner_flip_operand_classification": operands["classification"],
            "projection_source_classification": source["classification"],
            "first_open_boundary": source["first_open_boundary"],
            "closed_boundaries": source["closed_boundaries"],
        },
        "rejected_precision_intervention": {
            "classification": precision["classification"],
            "control_numbered_failures": int(precision["fixed_metric"]["control_numbered_failures"]),
            "double_numbered_failures": int(precision["fixed_metric"]["double_numbered_failures"]),
            "double_minus_control_final_cross_engine_fsc_auc": float(
                precision["fixed_metric"]["double_minus_control_final_cross_engine_fsc_auc"]
            ),
        },
        "fixed_gates": {
            "expected_onset_order": [1, 2, 3],
            "early_map_fsc_loss_strictly_increasing": True,
            "fine_winner_operand_must_be_projection": EXPECTED_OPERAND_CLASSIFICATION,
            "fine_projection_source_must_be_iteration_start_map": EXPECTED_SOURCE_CLASSIFICATION,
            "double_xhalf_precision_must_remain_rejected": EXPECTED_PRECISION_CLASSIFICATION,
        },
        "input_artifacts": {
            label: {"path": str(path), "sha256": _sha256(path)} for label, path in paths.items()
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-json", required=True, type=Path)
    parser.add_argument("--particle-state-json", required=True, type=Path)
    parser.add_argument("--operand-report-json", required=True, type=Path)
    parser.add_argument("--source-boundary-json", required=True, type=Path)
    parser.add_argument("--precision-factorial-json", required=True, type=Path)
    parser.add_argument("--expected-original-index", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_report(
        trajectory_json=args.trajectory_json,
        particle_state_json=args.particle_state_json,
        operand_report_json=args.operand_report_json,
        source_boundary_json=args.source_boundary_json,
        precision_factorial_json=args.precision_factorial_json,
        expected_original_index=args.expected_original_index,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
