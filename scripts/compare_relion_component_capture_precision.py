#!/usr/bin/env python3
"""Compare float32 and FP64 RELION coarse-component diagnostic captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
    CoarsePass1Components,
    validate_directory,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _centered_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.median(values)
    absolute = np.abs(centered)
    return {
        "median": float(np.median(values)),
        "centered_p95_abs": float(np.percentile(absolute, 95)),
        "centered_max_abs": float(np.max(absolute)),
    }


def _replay_metrics(artifact: CoarsePass1Components) -> dict[str, Any]:
    active = artifact.raw_diff2 != RELION_INVALID_DIFF2
    replay = (artifact.raw_diff2[active] - artifact.reference_norms[active] - artifact.cross_terms[active]).astype(
        np.float64
    )
    metrics = _centered_metrics(replay)
    image_constant = np.float32(abs(metrics["median"]))
    image_constant_ulp = float(np.spacing(image_constant))
    _require(image_constant_ulp > 0.0, "image-constant ULP must be positive")
    p95_ulps = metrics["centered_p95_abs"] / image_constant_ulp
    max_ulps = metrics["centered_max_abs"] / image_constant_ulp
    return {
        "image_constant_median": metrics["median"],
        "image_constant_float32_ulp": image_constant_ulp,
        "centered_replay_p95_abs": metrics["centered_p95_abs"],
        "centered_replay_max_abs": metrics["centered_max_abs"],
        "centered_replay_p95_image_constant_ulps": float(p95_ulps),
        "centered_replay_max_image_constant_ulps": float(max_ulps),
        "p95_is_integral_image_constant_ulps": bool(np.isclose(p95_ulps, np.rint(p95_ulps), rtol=0.0, atol=1.0e-9)),
        "max_is_integral_image_constant_ulps": bool(np.isclose(max_ulps, np.rint(max_ulps), rtol=0.0, atol=1.0e-9)),
    }


def compare_artifacts(
    baseline: CoarsePass1Components,
    fp64: CoarsePass1Components,
) -> dict[str, Any]:
    """Compare one identity-matched pair without correlation."""

    _require(baseline.part_id == fp64.part_id, "part identities differ")
    _require(baseline.stack_index == fp64.stack_index, "stack identities differ")
    _require(baseline.raw_diff2.shape == fp64.raw_diff2.shape, "topologies differ")
    baseline_active = baseline.raw_diff2 != RELION_INVALID_DIFF2
    fp64_active = fp64.raw_diff2 != RELION_INVALID_DIFF2
    _require(np.array_equal(baseline_active, fp64_active), "active supports differ")
    active = baseline_active

    raw_baseline = baseline.raw_diff2[active]
    raw_fp64 = fp64.raw_diff2[active]
    raw_delta = raw_fp64.astype(np.float64) - raw_baseline.astype(np.float64)
    norm_delta = fp64.reference_norms[active].astype(np.float64) - baseline.reference_norms[active].astype(np.float64)
    cross_delta = fp64.cross_terms[active].astype(np.float64) - baseline.cross_terms[active].astype(np.float64)
    return {
        "part_id": fp64.part_id,
        "stack_index_one_based": fp64.stack_index,
        "active_candidate_count": int(np.count_nonzero(active)),
        "raw_score_bitwise_equal_fraction": float(np.mean(raw_baseline == raw_fp64)),
        "raw_score_delta": _centered_metrics(raw_delta),
        "reference_norm_delta": _centered_metrics(norm_delta),
        "cross_term_delta": _centered_metrics(cross_delta),
        "baseline_replay": _replay_metrics(baseline),
        "fp64_replay": _replay_metrics(fp64),
        "artifact_paths": {
            "baseline": str(baseline.path.resolve()),
            "fp64": str(fp64.path.resolve()),
        },
        "artifact_sha256": {
            "baseline": baseline.sha256,
            "fp64": fp64.sha256,
        },
    }


def build_report(
    baseline_directory: Path,
    fp64_directory: Path,
    *,
    expected_particles: int = 14,
) -> dict[str, Any]:
    baseline, baseline_validation = validate_directory(
        baseline_directory,
        expected_particles=expected_particles,
    )
    fp64, fp64_validation = validate_directory(
        fp64_directory,
        expected_particles=expected_particles,
    )
    baseline_by_stack = {item.stack_index: item for item in baseline}
    fp64_by_stack = {item.stack_index: item for item in fp64}
    _require(
        baseline_by_stack.keys() == fp64_by_stack.keys(),
        "capture stack-identity sets differ",
    )
    particles = [
        compare_artifacts(baseline_by_stack[stack], fp64_by_stack[stack]) for stack in sorted(baseline_by_stack)
    ]
    fixed_metric = {
        "evaluated_particles": len(particles),
        "expected_particles": expected_particles,
        "baseline_replay_p95_passed": baseline_validation["fixed_metric"]["replay_p95_passed"],
        "baseline_reference_translation_invariance_passed": baseline_validation["fixed_metric"][
            "reference_translation_invariance_passed"
        ],
        "fp64_replay_p95_passed": fp64_validation["fixed_metric"]["replay_p95_passed"],
        "fp64_reference_translation_invariance_passed": fp64_validation["fixed_metric"][
            "reference_translation_invariance_passed"
        ],
        "raw_score_bitwise_equal_particles": sum(row["raw_score_bitwise_equal_fraction"] == 1.0 for row in particles),
        "raw_score_centered_delta_p95_at_most_one_float32_ulp": sum(
            row["raw_score_delta"]["centered_p95_abs"] <= row["fp64_replay"]["image_constant_float32_ulp"]
            for row in particles
        ),
        "fp64_replay_p95_is_integral_image_constant_ulps": sum(
            row["fp64_replay"]["p95_is_integral_image_constant_ulps"] for row in particles
        ),
    }
    return {
        "schema": "relion-component-capture-precision-comparison-v1",
        "status": "complete",
        "classification_ready": False,
        "classification": (
            "fp64_capture_rejected_expanded_component_arithmetic_does_not_replay_production_float32_diff2"
        ),
        "fixed_metric": fixed_metric,
        "fixed_gates": fp64_validation["fixed_gates"],
        "baseline_validation": baseline_validation,
        "fp64_validation": fp64_validation,
        "particles": particles,
        "notes": [
            "No correlation is computed.",
            "The historical component-replay gates are unchanged.",
            "The FP64 diagnostic expands norm and cross terms separately while "
            "production diff2 retains the original float32 squared-difference path.",
            "This report is diagnostic and cannot promote component attribution.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_directory", type=Path)
    parser.add_argument("fp64_directory", type=Path)
    parser.add_argument("--expected-particles", type=int, default=14)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.baseline_directory,
        args.fp64_directory,
        expected_particles=args.expected_particles,
    )
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
