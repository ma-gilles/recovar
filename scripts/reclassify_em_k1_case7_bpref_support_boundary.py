#!/usr/bin/env python3
"""Correct a case-7 BPref report by gating operands on exact support identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "recovar.em_k1_case7_bpref_factor_boundary.v1"
SCHEMA = "recovar.em_k1_case7_bpref_support_gated_boundary.v1"
EXPECTED_PARTICLE_COUNT = 10
EXACT_COMPARISONS = (
    "same_posterior_numerator_terms",
    "same_posterior_denominator_terms",
    "relion_summary_to_recovar_sequential_numerator",
    "relion_summary_to_recovar_sequential_denominator",
    "relion_summary_to_recovar_highest_numerator",
    "relion_summary_to_recovar_highest_denominator",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_selected_comparisons_exact(particle: dict[str, Any]) -> bool:
    comparisons = particle.get("comparisons")
    _require(isinstance(comparisons, dict), "particle comparisons are missing")
    for name in EXACT_COMPARISONS:
        comparison = comparisons.get(name)
        _require(isinstance(comparison, dict), f"comparison {name} is missing")
        if comparison.get("exact_equal") is not True:
            return False
    return True


def classify_particle(particle: dict[str, Any]) -> str:
    """Return the first measured boundary, with support identity evaluated first."""

    if particle.get("capture_self_closes") is not True:
        return "relion_factor_capture_does_not_reproduce_relion_summary"
    if particle.get("support_exact") is not True:
        return "significant_support_identity_mismatch_before_bpref_operands"
    if particle.get("same_posterior_operands_close") is not True:
        return "bpref_operand_mismatch_before_translation_reduction"

    sequential = particle.get("sequential_summary_closes") is True
    highest = particle.get("highest_summary_closes") is True
    if sequential and not highest:
        return "recovar_translation_reduction_order_mismatch"
    if not sequential:
        return "translation_reduction_or_unmeasured_operand_mismatch"
    if _all_selected_comparisons_exact(particle):
        return "particle_prescatter_boundary_exactly_closes"
    return "particle_prescatter_boundary_closes_within_relative_l2_bound"


def _aggregate(classifications: list[str]) -> tuple[str, str]:
    if any(value == "relion_factor_capture_does_not_reproduce_relion_summary" for value in classifications):
        return "fixed_panel_relion_capture_self_mismatch", "repair_or_requalify_native_capture"
    if any(value == "significant_support_identity_mismatch_before_bpref_operands" for value in classifications):
        return "fixed_panel_significant_support_identity_mismatch", "posterior_normalization_and_significance"
    if any(value == "bpref_operand_mismatch_before_translation_reduction" for value in classifications):
        return "fixed_panel_bpref_operand_mismatch", "per_candidate_bpref_operands"
    if any(value == "recovar_translation_reduction_order_mismatch" for value in classifications):
        return "fixed_panel_translation_reduction_order_mismatch", "translation_reduction_order"
    if any(value == "translation_reduction_or_unmeasured_operand_mismatch" for value in classifications):
        return "fixed_panel_translation_reduction_or_unmeasured_operand_mismatch", "translation_reduction_operands"
    if all(value == "particle_prescatter_boundary_exactly_closes" for value in classifications):
        return "fixed_panel_particle_prescatter_boundary_exactly_closes", "accumulator_destination_and_inter_particle_reduction"
    return (
        "fixed_panel_particle_prescatter_boundary_closes_within_relative_l2_bound",
        "exactness_residual_then_accumulator_destination_and_inter_particle_reduction",
    )


def reclassify(report: dict[str, Any]) -> dict[str, Any]:
    """Build a non-destructive support-gated classification from the original report."""

    _require(report.get("schema") == INPUT_SCHEMA, "input report schema changed")
    _require(report.get("status") == "complete", "input report is incomplete")
    _require(report.get("production_authorized") is False, "input unexpectedly authorizes production")
    _require(report.get("fixed_scorecard_changed") is False, "input unexpectedly changes a scorecard")
    particles = report.get("particles")
    _require(
        isinstance(particles, list) and len(particles) == EXPECTED_PARTICLE_COUNT,
        "fixed ten-particle panel changed",
    )

    corrected_particles = []
    classifications = []
    for particle in particles:
        classification = classify_particle(particle)
        classifications.append(classification)
        corrected_particles.append(
            {
                "original_index_zero_based": int(particle["original_index_zero_based"]),
                "stack_index_one_based": int(particle["stack_index_one_based"]),
                "image_identity": particle["image_identity"],
                "support_exact": particle.get("support_exact") is True,
                "support_intersection_count": int(particle["support_intersection_count"]),
                "support_union_count": int(particle["support_union_count"]),
                "native_support_count": int(particle["accepted_hypothesis_count_native"]),
                "recovar_support_count": int(particle["accepted_hypothesis_count_recovar"]),
                "capture_self_closes_within_relative_l2_bound": particle.get("capture_self_closes") is True,
                "same_posterior_operands_close_within_relative_l2_bound": particle.get("same_posterior_operands_close") is True,
                "sequential_summary_closes_within_relative_l2_bound": particle.get("sequential_summary_closes") is True,
                "highest_summary_closes_within_relative_l2_bound": particle.get("highest_summary_closes") is True,
                "selected_comparisons_exact": _all_selected_comparisons_exact(particle),
                "original_classification": particle.get("classification"),
                "corrected_classification": classification,
            }
        )

    classification, next_boundary = _aggregate(classifications)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": classification,
        "next_boundary": next_boundary,
        "particle_count": len(corrected_particles),
        "support_exact_count": sum(row["support_exact"] for row in corrected_particles),
        "particle_classification_counts": dict(sorted(Counter(classifications).items())),
        "relative_l2_bound": report.get("relative_l2_bound"),
        "metric_policy": (
            "exact significant-support identity precedes BPref operand/reduction closure; "
            "exact_equal and relative-L2 closure are reported separately; no correlation"
        ),
        "original_aggregate_classification": report.get("classification"),
        "production_authorized": False,
        "fixed_scorecard_changed": False,
        "particles": corrected_particles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    original = json.loads(args.input_json.read_text())
    corrected = reclassify(original)
    corrected["input"] = {
        "path": str(args.input_json.resolve()),
        "sha256": _sha256(args.input_json),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(corrected, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": corrected["classification"], "next_boundary": corrected["next_boundary"]}))


if __name__ == "__main__":
    main()
