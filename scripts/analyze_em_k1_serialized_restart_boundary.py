#!/usr/bin/env python3
"""Test whether a serialized RELION restart closes the K=1 case-22 score gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _translation_permutation,
)
from scripts.analyze_em_k1_corr_img_conditioning import (
    SHELL_PARTITION_CLASSIFICATION,
)
from scripts.analyze_em_k1_corr_img_factorial import _sha256
from scripts.analyze_em_k1_live_reference_counterfactual import (
    _load_recovar,
    reference_swap_counterfactual,
)
from scripts.analyze_em_k1_postoptics_score_transfer import (
    EXPECTED_PARTICLES,
    _require,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
    load_artifact,
)

CLASSIFICATION = (
    "serialized_restart_closes_case22_raw_coarse_residual_under_fixed_gates"
)
SCORE_P95_MAX_ABS = 1.0e-4
SCORE_MAX_ABS_STRICTLY_BELOW = 1.0e-3


def classify_serialized_restart(
    *,
    qualified: bool,
    dominated: int,
    absolute_gate_passed: int,
    expected_particles: int,
) -> str:
    """Classify the fixed restart intervention without fitted parameters."""

    if not qualified:
        return "serialized_restart_inputs_not_qualified"
    if (
        dominated == expected_particles
        and absolute_gate_passed == expected_particles
    ):
        return CLASSIFICATION
    if dominated == expected_particles:
        return (
            "serialized_restart_removes_majority_residual_but_not_"
            "absolute_score_gates"
        )
    if dominated == 0:
        return "serialized_restart_does_not_remove_fresh_process_residual"
    return "serialized_restart_has_mixed_case22_raw_coarse_effect"


def _validate_parent(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    _require(report.get("status") == "complete", "parent report is incomplete")
    _require(
        report.get("classification_ready") is True,
        "parent report is not classification-ready",
    )
    shell_metric = report.get("shell_partition_metric", {})
    _require(
        shell_metric.get("classification") == SHELL_PARTITION_CLASSIFICATION,
        "parent shell-partition classification changed",
    )
    _require(
        shell_metric.get("evaluated_particles") == EXPECTED_PARTICLES
        and shell_metric.get("expected_particles") == EXPECTED_PARTICLES,
        "parent shell-partition denominator changed",
    )
    return report


def _validate_component_report(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    _require(
        report.get("particle_count") == EXPECTED_PARTICLES,
        "restart component validation denominator changed",
    )
    fixed = report.get("fixed_metric", {})
    _require(
        fixed.get("evaluated_particles") == EXPECTED_PARTICLES,
        "restart component validator did not evaluate every particle",
    )
    return report


def _validate_operand_report(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    _require(
        report.get("status") == "pass"
        and report.get("classification_ready") is True,
        "restart direct operand validation did not pass",
    )
    fixed = report.get("fixed_metric", {})
    expected = {
        "evaluated_particles": EXPECTED_PARTICLES,
        "expected_particles": EXPECTED_PARTICLES,
        "reference_replay_passed": EXPECTED_PARTICLES,
        "cross_replay_p95_passed": EXPECTED_PARTICLES,
        "cross_replay_max_passed": EXPECTED_PARTICLES,
        "production_diff2_centered_replay_p95_passed": EXPECTED_PARTICLES,
        "production_diff2_centered_replay_max_passed": EXPECTED_PARTICLES,
    }
    _require(
        fixed == expected,
        "restart direct operand fixed gates did not pass 14/14",
    )
    return report


def _component_by_stack(directory: Path) -> dict[int, Any]:
    paths = sorted(Path(directory).glob("*.p1-v2.bin"))
    _require(
        len(paths) == EXPECTED_PARTICLES,
        "restart component capture denominator changed",
    )
    result = {}
    for path in paths:
        artifact = load_artifact(path)
        _require(
            artifact.stack_index not in result,
            f"duplicate restart stack index {artifact.stack_index}",
        )
        result[artifact.stack_index] = artifact
    return result


def _mapped_selected_raw(
    *,
    component: Any,
    recovar: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    translation_permutation, translation_mapping = _translation_permutation(
        component.translations,
        recovar["translations"],
    )
    mapped = _map_relion_table(
        component.raw_diff2,
        n_directions=component.header[10],
        n_psi=component.header[11],
        relion_to_recovar_translation=translation_permutation,
    )
    selected = mapped[recovar["rotation_ids"]]
    _require(
        np.all(selected != RELION_INVALID_DIFF2),
        "selected restart panel contains inactive RELION scores",
    )
    return selected, translation_mapping


def _absolute_gate(counterfactual: dict[str, Any]) -> bool:
    return bool(
        counterfactual["swapped_centered_p95_abs"] <= SCORE_P95_MAX_ABS
        and counterfactual["swapped_centered_max_abs"]
        < SCORE_MAX_ABS_STRICTLY_BELOW
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def build_report(
    *,
    parent_analysis_json: Path,
    restart_capture_directory: Path,
    restart_component_validation_json: Path,
    restart_operand_validation_json: Path,
    source_optimiser_star: Path,
) -> dict[str, Any]:
    """Build the fixed fresh-process versus serialized-restart comparison."""

    parent = _validate_parent(parent_analysis_json)
    component_validation = _validate_component_report(
        restart_component_validation_json
    )
    operand_validation = _validate_operand_report(
        restart_operand_validation_json
    )
    restart_by_stack = _component_by_stack(restart_capture_directory)
    particles = []
    for parent_row in parent["particles"]:
        stack_index = int(parent_row["stack_index_one_based"])
        original_index = int(parent_row["original_index_zero_based"])
        fresh_path = Path(parent_row["artifact_paths"]["component"])
        recovar_path = Path(parent_row["artifact_paths"]["recovar"])
        _require(
            _sha256(fresh_path)
            == parent_row["artifact_sha256"]["component"],
            "fresh component artifact hash changed",
        )
        _require(
            _sha256(recovar_path)
            == parent_row["artifact_sha256"]["recovar"],
            "RECOVAR artifact hash changed",
        )
        fresh = load_artifact(fresh_path)
        restart = restart_by_stack.get(stack_index)
        _require(restart is not None, "restart component stack identity missing")
        recovar = _load_recovar(recovar_path)
        _require(
            fresh.stack_index == restart.stack_index == stack_index,
            "fresh/restart stack identity mismatch",
        )
        _require(
            fresh.part_id == restart.part_id,
            "fresh/restart particle identity mismatch",
        )
        _require(
            int(recovar["original_index"]) == original_index
            and stack_index == original_index + 1,
            "RECOVAR/RELION particle identity mismatch",
        )
        _require(
            fresh.header[5] == restart.header[5] == 2,
            "fresh/restart capture iteration mismatch",
        )
        _require(
            fresh.header[10:14] == restart.header[10:14],
            "fresh/restart coarse topology mismatch",
        )

        fresh_raw, fresh_translation = _mapped_selected_raw(
            component=fresh,
            recovar=recovar,
        )
        restart_raw, restart_translation = _mapped_selected_raw(
            component=restart,
            recovar=recovar,
        )
        _require(
            fresh_translation == restart_translation,
            "fresh/restart translation mapping changed",
        )
        selected_recovar = recovar["scores"][recovar["rotation_ids"]]
        fresh_residual = selected_recovar + fresh_raw
        restart_residual = selected_recovar + restart_raw
        counterfactual = reference_swap_counterfactual(
            fresh_residual,
            restart_residual,
        )
        absolute_gate_passed = _absolute_gate(counterfactual)
        particles.append(
            {
                "group": parent_row["group"],
                "stack_index_one_based": stack_index,
                "original_index_zero_based": original_index,
                "counterfactual": counterfactual,
                "absolute_score_gate_passed": absolute_gate_passed,
                "translation_mapping": restart_translation,
                "artifact_paths": {
                    "fresh_component": str(fresh_path.resolve()),
                    "restart_component": str(restart.path.resolve()),
                    "recovar": str(recovar_path.resolve()),
                },
                "artifact_sha256": {
                    "fresh_component": fresh.sha256,
                    "restart_component": restart.sha256,
                    "recovar": recovar["sha256"],
                },
            }
        )

    _require(
        len(particles) == EXPECTED_PARTICLES,
        "serialized-restart particle denominator changed",
    )
    dominated = sum(
        row["counterfactual"]["live_reference_dominated"]
        for row in particles
    )
    absolute_gate_passed = sum(
        row["absolute_score_gate_passed"] for row in particles
    )
    classification = classify_serialized_restart(
        qualified=True,
        dominated=dominated,
        absolute_gate_passed=absolute_gate_passed,
        expected_particles=EXPECTED_PARTICLES,
    )
    return {
        "schema": "em-k1-serialized-restart-boundary-v1",
        "status": "complete",
        "classification_ready": True,
        "classification": classification,
        "metric_policy": (
            "fixed 14-particle fresh-process versus serialized-it000 "
            "restart intervention on the same RECOVAR coarse score panels; "
            "centered residual-energy removal strictly above 0.5; "
            "centered p95 <=1e-4 and max <1e-3; no fitted scale/sign; "
            "no correlation"
        ),
        "fixed_gates": {
            "expected_particles": EXPECTED_PARTICLES,
            "component_dominance_fraction_strictly_greater_than": 0.5,
            "centered_score_p95_abs_max": SCORE_P95_MAX_ABS,
            "centered_score_max_abs_strictly_below": (
                SCORE_MAX_ABS_STRICTLY_BELOW
            ),
        },
        "fixed_metric": {
            "evaluated_particles": len(particles),
            "expected_particles": EXPECTED_PARTICLES,
            "serialized_restart_dominated": dominated,
            "absolute_score_gate_passed": absolute_gate_passed,
            "counterfactual_energy_removal_fraction": _summary(
                [
                    row["counterfactual"][
                        "counterfactual_energy_removal_fraction"
                    ]
                    for row in particles
                ]
            ),
            "restart_centered_p95_abs": _summary(
                [
                    row["counterfactual"]["swapped_centered_p95_abs"]
                    for row in particles
                ]
            ),
            "restart_centered_max_abs": _summary(
                [
                    row["counterfactual"]["swapped_centered_max_abs"]
                    for row in particles
                ]
            ),
        },
        "parent_analysis": {
            "path": str(Path(parent_analysis_json).resolve()),
            "sha256": _sha256(parent_analysis_json),
            "shell_partition_classification": (
                parent["shell_partition_metric"]["classification"]
            ),
        },
        "restart_component_validation": {
            "path": str(Path(restart_component_validation_json).resolve()),
            "sha256": _sha256(restart_component_validation_json),
            "status": component_validation.get("status"),
            "fixed_metric": component_validation["fixed_metric"],
        },
        "restart_operand_validation": {
            "path": str(Path(restart_operand_validation_json).resolve()),
            "sha256": _sha256(restart_operand_validation_json),
            "status": operand_validation["status"],
            "fixed_metric": operand_validation["fixed_metric"],
        },
        "source_optimiser_star": {
            "path": str(Path(source_optimiser_star).resolve()),
            "sha256": _sha256(source_optimiser_star),
        },
        "particles": particles,
        "notes": [
            (
                "The fresh and restart RELION score panels use the same "
                "captured topology, fixed rotations, and registered "
                "translations against the same RECOVAR score artifact."
            ),
            (
                "The restart begins from RELION's serialized iteration-0 "
                "state; the fresh run created that state and continued in "
                "the same process."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-analysis-json", type=Path, required=True)
    parser.add_argument(
        "--restart-capture-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--restart-component-validation-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--restart-operand-validation-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-optimiser-star", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(
        not args.output_json.exists(),
        f"refusing to overwrite report: {args.output_json}",
    )
    report = build_report(
        parent_analysis_json=args.parent_analysis_json,
        restart_capture_directory=args.restart_capture_directory,
        restart_component_validation_json=(
            args.restart_component_validation_json
        ),
        restart_operand_validation_json=(
            args.restart_operand_validation_json
        ),
        source_optimiser_star=args.source_optimiser_star,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["fixed_metric"], indent=2, sort_keys=True))
    print(report["classification"])


if __name__ == "__main__":
    main()
