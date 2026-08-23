#!/usr/bin/env python3
"""Summarize a fixed panel of native/RECOVAR K=1 fine-score captures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_k1_fine_score_stages import STAGES, analyze
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import load_fine_score_capture


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_expected_stacks(text: str) -> set[int]:
    values = [int(value) for value in text.replace(":", ",").split(",")]
    _require(bool(values) and all(value > 0 for value in values), "expected stacks must be positive")
    _require(len(values) == len(set(values)), "expected stacks must be unique")
    return set(values)


def _unique_native_by_stack(paths: list[Path], loader, label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in paths:
        stack_index = int(loader(path).stack_index)
        _require(stack_index not in result, f"duplicate {label} capture for stack {stack_index}")
        result[stack_index] = path
    return result


def _unique_recovar_by_stack(paths: list[Path]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            stack_index = int(archive["original_index"]) + 1
        _require(stack_index not in result, f"duplicate RECOVAR capture for stack {stack_index}")
        result[stack_index] = path
    return result


def summarize_reports(reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Return exact fixed-denominator stage gates and first-unequal counts."""

    _require(bool(reports), "fine-score stage panel is empty")
    pass_counts: Counter[str] = Counter()
    first_unequal_counts: Counter[str] = Counter()
    comparison_metrics: dict[str, list[dict[str, Any]]] = {}
    particles: dict[str, Any] = {}
    for stack_index in sorted(reports):
        report = reports[stack_index]
        stage_exact = {stage: bool(report["stage_exact"][stage]) for stage in STAGES}
        for stage, exact in stage_exact.items():
            pass_counts[stage] += int(exact)
        first_unequal = next((stage for stage in STAGES if not stage_exact[stage]), "all_stages_exact")
        _require(
            report["first_exact_unequal_boundary"] == first_unequal,
            f"stage classification is inconsistent for stack {stack_index}",
        )
        first_unequal_counts[first_unequal] += 1
        for stage, metrics in report.get("comparisons", {}).items():
            comparison_metrics.setdefault(stage, []).append(metrics)
        particles[str(stack_index)] = {
            "stage_exact": stage_exact,
            "first_exact_unequal_boundary": first_unequal,
            "native_active_count": int(report["native_active_count"]),
            "recovar_active_candidate_count": int(report["recovar_active_candidate_count"]),
            "native_significant_count": int(report["native_significant_count"]),
            "recovar_significant_count": int(report["recovar_significant_count"]),
            "support_intersection_count": int(report["support_intersection_count"]),
        }
    total = len(reports)
    stage_error_envelopes = {
        stage: {
            "particle_count": len(metrics),
            "worst_max_abs": max(float(item["max_abs"]) for item in metrics),
            "worst_relative_l2_over_reference": max(
                float(item["relative_l2_over_reference"]) for item in metrics
            ),
            "total_mismatch_count": sum(int(item["mismatch_count"]) for item in metrics),
        }
        for stage, metrics in sorted(comparison_metrics.items())
    }
    return {
        "particle_count": total,
        "stage_pass_counts": {
            stage: {"passed": int(pass_counts[stage]), "total": total}
            for stage in STAGES
        },
        "first_unequal_boundary_counts": dict(sorted(first_unequal_counts.items())),
        "stage_error_envelopes": stage_error_envelopes,
        "particles": particles,
    }


def analyze_panel(
    *,
    native_capture_dir: Path,
    recovar_capture_dir: Path,
    expected_stacks: set[int],
    physical_image_size: int,
    top_count: int,
) -> dict[str, Any]:
    factors = _unique_native_by_stack(
        sorted(native_capture_dir.glob("*.bpre-v2.bin")),
        load_factor_capture,
        "native BPref",
    )
    fine_scores = _unique_native_by_stack(
        sorted(native_capture_dir.glob("*.fine-score-v1.bin")),
        load_fine_score_capture,
        "native fine-score",
    )
    recovar = _unique_recovar_by_stack(sorted(recovar_capture_dir.glob("pass2_orig*.npz")))
    _require(set(factors) == expected_stacks, "native BPref stack set differs from expected")
    _require(set(fine_scores) == expected_stacks, "native fine-score stack set differs from expected")
    _require(set(recovar) == expected_stacks, "RECOVAR pass-2 stack set differs from expected")
    reports = {
        stack_index: analyze(
            native_factor=factors[stack_index],
            native_fine_score=fine_scores[stack_index],
            recovar_capture=recovar[stack_index],
            physical_image_size=physical_image_size,
            top_count=top_count,
        )
        for stack_index in sorted(expected_stacks)
    }
    return {
        "schema": "recovar.em.k1_fine_score_stage_panel.v1",
        "status": "complete",
        "expected_stacks_one_based": sorted(expected_stacks),
        "summary": summarize_reports(reports),
        "reports": {str(stack_index): reports[stack_index] for stack_index in sorted(reports)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-capture-dir", type=Path, required=True)
    parser.add_argument("--recovar-capture-dir", type=Path, required=True)
    parser.add_argument("--expected-stacks", required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--top-count", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_panel(
        native_capture_dir=args.native_capture_dir,
        recovar_capture_dir=args.recovar_capture_dir,
        expected_stacks=_parse_expected_stacks(args.expected_stacks),
        physical_image_size=args.physical_image_size,
        top_count=args.top_count,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
