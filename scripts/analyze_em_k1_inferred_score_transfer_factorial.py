#!/usr/bin/env python3
"""Split a passive K=1 base-image capture into pixel and corr_img factors."""

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
from scripts.analyze_em_k1_live_reference_counterfactual import (
    RECOVAR_REPLAY_MAX_GATE,
    RECOVAR_REPLAY_P95_GATE,
    _load_recovar,
    _relative_l2,
    _sha256,
    _validate_operand_capture,
    recovar_score_components,
    reference_swap_counterfactual,
    relion_values_on_recovar_window,
)
from scripts.analyze_em_k1_postoptics_score_transfer import (
    _recover_base_and_phase,
    relion_live_score_base,
)
from scripts.analyze_em_k1_score_transfer_factorial import (
    classify_score_transfer_factorial,
    inferred_score_transfer_factorial_bases,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
)

PARENT_CLASSIFICATION = (
    "raw_coarse_residual_is_live_base_corrected_image_dominated_"
    "not_reference_correction_or_translation_phase"
)
DIAGONAL_REPLAY_RELATIVE_L2_GATE = 1.0e-14


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_parent(path: Path, *, expected_particles: int) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    _require(report.get("status") == "complete", "parent report is incomplete")
    _require(
        report.get("classification_ready") is True,
        "parent report is not classification-ready",
    )
    _require(
        report.get("classification") == PARENT_CLASSIFICATION,
        "parent report does not localize the base-corrected image",
    )
    fixed = report.get("fixed_metric", {})
    _require(
        fixed.get("evaluated_particles") == expected_particles
        and fixed.get("expected_particles") == expected_particles
        and fixed.get("live_operand_dominated", {}).get(
            "base_corrected_image"
        )
        == expected_particles,
        "parent report denominator or base-image result differs",
    )
    return report


def build_report(
    *,
    cohort_json: Path,
    capture_directory: Path,
    recovar_directory: Path,
    parent_analysis_json: Path,
    full_image_size: int,
    expected_particles: int,
) -> dict[str, Any]:
    """Build the fixed pixel/corr_img factorial from passive captures."""

    _require(expected_particles > 0, "expected particle count must be positive")
    parent = _validate_parent(
        parent_analysis_json,
        expected_particles=expected_particles,
    )
    cohort = json.loads(Path(cohort_json).read_text())
    _require(
        cohort.get("selected_particle_count") == expected_particles,
        "cohort denominator differs from the expected particle count",
    )
    expected_stacks = np.asarray(
        cohort["selected_stack_indices_one_based"],
        dtype=np.int64,
    )
    operands, components, operand_validation = _validate_operand_capture(
        capture_directory,
        expected_particles=expected_particles,
        expected_stacks=expected_stacks,
    )
    operands_by_stack = {item.stack_index: item for item in operands}
    components_by_stack = {item.stack_index: item for item in components}
    recovar_items = [
        _load_recovar(path)
        for path in sorted(Path(recovar_directory).glob("*.npz"))
    ]
    _require(
        len(recovar_items) == expected_particles,
        "RECOVAR denominator differs from the expected particle count",
    )
    recovar_by_index = {item["original_index"]: item for item in recovar_items}
    _require(
        len(recovar_by_index) == expected_particles,
        "duplicate RECOVAR original index",
    )

    particles = []
    for cohort_row in cohort["rows"]:
        stack_index = int(cohort_row["stack_index_one_based"])
        original_index = int(cohort_row["original_index_zero_based"])
        operand = operands_by_stack[stack_index]
        component = components_by_stack[stack_index]
        recovar = recovar_by_index[original_index]

        live_base, live_shifted = relion_live_score_base(
            operand,
            recovar,
            full_image_size=full_image_size,
        )
        recovar_base, recovar_phase = _recover_base_and_phase(
            live_base=live_base,
            live_shifted=live_shifted,
            recovar_shifted=recovar["shifted_data"],
        )
        window_indices = recovar["window_indices"]
        relion_pixel = relion_values_on_recovar_window(
            (
                operand.image_real.astype(np.float64)
                + np.complex128(1j) * operand.image_imag.astype(np.float64)
            ).reshape(1, -1),
            window_indices,
            full_image_size=full_image_size,
            current_size=recovar["current_size"],
        )[0]
        relion_corr = relion_values_on_recovar_window(
            operand.correction.reshape(1, -1),
            window_indices,
            full_image_size=full_image_size,
            current_size=recovar["current_size"],
        )[0].real
        bases = inferred_score_transfer_factorial_bases(
            relion_pixel_corrected_native=relion_pixel,
            relion_corr_img=relion_corr,
            recovar_base_corrected=recovar_base,
            recovar_ctf2_data=recovar["ctf2_data"],
            half_weights=recovar["half_weights"],
            full_image_size=full_image_size,
        )
        diagonal_replay = {
            "actual_relion_relative_l2": _relative_l2(
                bases["actual_relion"],
                live_base,
            ),
            "recovar_both_relative_l2": _relative_l2(
                bases["recovar_pixel_and_corr_img"],
                recovar_base,
            ),
        }
        diagonal_replay_passed = all(
            value <= DIAGONAL_REPLAY_RELATIVE_L2_GATE
            for value in diagonal_replay.values()
        )

        replay_norm, replay_cross = recovar_score_components(
            recovar["references"],
            recovar["shifted_data"],
            recovar["ctf2_data"],
            recovar["half_weights"],
        )
        replay_error = (
            replay_norm
            + replay_cross
            - recovar["scores"][recovar["rotation_ids"]]
        )
        replay_absolute = np.abs(replay_error)
        recovar_replay = {
            "p95_abs": float(np.percentile(replay_absolute, 95)),
            "max_abs": float(np.max(replay_absolute)),
        }
        recovar_replay_passed = (
            recovar_replay["p95_abs"] <= RECOVAR_REPLAY_P95_GATE
            and recovar_replay["max_abs"] <= RECOVAR_REPLAY_MAX_GATE
        )

        translation_permutation, translation_mapping = _translation_permutation(
            component.translations,
            recovar["translations"],
        )
        mapped_raw = _map_relion_table(
            component.raw_diff2,
            n_directions=component.header[10],
            n_psi=component.header[11],
            relion_to_recovar_translation=translation_permutation,
        )
        selected_raw = mapped_raw[recovar["rotation_ids"]]
        _require(
            np.all(selected_raw != RELION_INVALID_DIFF2),
            "captured rotation has inactive RELION score",
        )
        baseline_residual = (
            recovar["scores"][recovar["rotation_ids"]] + selected_raw
        )
        counterfactuals = {}
        base_relative_l2 = {}
        for label, base in bases.items():
            shifted = base[np.newaxis, :] * recovar_phase
            swapped_norm, swapped_cross = recovar_score_components(
                recovar["references"],
                shifted,
                recovar["ctf2_data"],
                recovar["half_weights"],
            )
            counterfactuals[label] = reference_swap_counterfactual(
                baseline_residual,
                swapped_norm + swapped_cross + selected_raw,
            )
            base_relative_l2[label] = _relative_l2(base, recovar_base)

        particles.append(
            {
                "group": cohort_row["group"],
                "stack_index_one_based": stack_index,
                "original_index_zero_based": original_index,
                "base_relative_l2": base_relative_l2,
                "counterfactuals": counterfactuals,
                "diagonal_replay": diagonal_replay,
                "diagonal_replay_passed": diagonal_replay_passed,
                "recovar_replay": recovar_replay,
                "recovar_replay_passed": recovar_replay_passed,
                "translation_mapping": translation_mapping,
                "artifact_paths": {
                    "component": str(component.path.resolve()),
                    "operand": str(operand.path.resolve()),
                    "recovar": recovar["path"],
                },
                "artifact_sha256": {
                    "component": component.sha256,
                    "operand": operand.sha256,
                    "recovar": recovar["sha256"],
                },
            }
        )

    arm_labels = tuple(particles[0]["counterfactuals"])
    dominated = {
        label: sum(
            row["counterfactuals"][label]["live_reference_dominated"]
            for row in particles
        )
        for label in arm_labels
    }
    recovar_replay_passed = sum(row["recovar_replay_passed"] for row in particles)
    diagonal_replay_passed = sum(
        row["diagonal_replay_passed"] for row in particles
    )
    capture_qualified = bool(operand_validation["classification_ready"])
    qualified = (
        capture_qualified
        and recovar_replay_passed == expected_particles
        and diagonal_replay_passed == expected_particles
    )
    classification = classify_score_transfer_factorial(
        qualified=qualified,
        dominated=dominated,
        expected_particles=expected_particles,
    )
    return {
        "schema": "em-k1-inferred-score-transfer-factorial-v1",
        "status": "complete",
        "classification_ready": qualified,
        "classification": classification,
        "metric_policy": (
            "fixed 2x2 pixel-correction/corr_img intervention inferred from "
            "passive operands; centered residual-energy removal; no fitted "
            "scale/sign; no correlation"
        ),
        "fixed_gates": {
            "expected_particles": expected_particles,
            "component_dominance_fraction_strictly_greater_than": 0.5,
            "diagonal_replay_relative_l2_max": DIAGONAL_REPLAY_RELATIVE_L2_GATE,
            "recovar_replay_p95_abs_max": RECOVAR_REPLAY_P95_GATE,
            "recovar_replay_max_abs_max": RECOVAR_REPLAY_MAX_GATE,
        },
        "fixed_metric": {
            "evaluated_particles": len(particles),
            "expected_particles": expected_particles,
            "live_reference_dominated": dominated,
            "operand_capture_qualified": (
                expected_particles if capture_qualified else 0
            ),
            "recovar_replay_passed": recovar_replay_passed,
            "diagonal_replay_passed": diagonal_replay_passed,
        },
        "full_image_size": full_image_size,
        "operand_validation": operand_validation,
        "parent_analysis": {
            "path": str(Path(parent_analysis_json).resolve()),
            "sha256": _sha256(parent_analysis_json),
            "classification": parent["classification"],
        },
        "cohort": {
            "path": str(Path(cohort_json).resolve()),
            "sha256": _sha256(cohort_json),
        },
        "particles": particles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-json", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--parent-analysis-json", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, required=True)
    parser.add_argument("--expected-particles", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(
        not args.output_json.exists(),
        f"refusing to overwrite report: {args.output_json}",
    )
    report = build_report(
        cohort_json=args.cohort_json,
        capture_directory=args.capture_directory,
        recovar_directory=args.recovar_directory,
        parent_analysis_json=args.parent_analysis_json,
        full_image_size=args.full_image_size,
        expected_particles=args.expected_particles,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["fixed_metric"], indent=2, sort_keys=True))
    print(report["classification"])
    if not report["classification_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
