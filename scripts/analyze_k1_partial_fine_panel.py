#!/usr/bin/env python3
"""Analyze a fixed panel of native/RECOVAR K=1 fine-boundary captures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.analyze_k1_partial_fine_topology import (
    analyze,
    load_recovar_candidate_table,
)
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import load_fine_score_capture

STAGES = (
    "rotation_topology",
    "active_tuple_topology",
    "preprior_score_centered",
    "orientation_log_prior",
    "translation_log_prior",
    "posterior",
    "fine_significant_support",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_expected_stacks(text: str | None) -> set[int] | None:
    if text is None:
        return None
    values = [int(value) for value in text.replace(":", ",").split(",")]
    _require(bool(values) and all(value > 0 for value in values), "expected stacks must be positive")
    _require(len(values) == len(set(values)), "expected stacks must be unique")
    return set(values)


def stage_outcomes(report: dict[str, object]) -> dict[str, bool]:
    """Return ordered exactness gates for one partial fine-boundary report."""

    rotations = report["rotation_topology"]
    tuples = report["active_tuple_topology"]
    production = report["production_boundary"]
    _require(isinstance(rotations, dict), "rotation topology is absent")
    _require(isinstance(tuples, dict), "active tuple topology is absent")
    _require(isinstance(production, dict), "production boundary is absent")
    rotation_exact = (
        rotations["native_count"]
        == rotations["recovar_count"]
        == rotations["common_count"]
    )
    tuple_exact = (
        tuples["native_count"]
        == tuples["recovar_count"]
        == tuples["common_count"]
    )
    return {
        "rotation_topology": bool(rotation_exact),
        "active_tuple_topology": bool(tuple_exact),
        "preprior_score_centered": bool(production["preprior_score_centered"]["exact_equal"]),
        "orientation_log_prior": bool(production["orientation_log_prior"]["exact_equal"]),
        "translation_log_prior": bool(production["translation_log_prior"]["exact_equal"]),
        "posterior": bool(production["posterior_on_common_native_normalization"]["exact_equal"]),
        "fine_significant_support": bool(production["fine_significant_support"]["exact"]),
    }


def summarize_reports(reports: dict[int, dict[str, object]]) -> dict[str, object]:
    """Build fixed-denominator stage counts and first-unequal classifications."""

    _require(bool(reports), "fine-boundary panel is empty")
    outcomes_by_stack: dict[str, dict[str, object]] = {}
    first_unequal_counts: Counter[str] = Counter()
    pass_counts = Counter()
    for stack_index in sorted(reports):
        outcomes = stage_outcomes(reports[stack_index])
        for stage, passed in outcomes.items():
            pass_counts[stage] += int(passed)
        first_unequal = next((stage for stage in STAGES if not outcomes[stage]), "closed")
        first_unequal_counts[first_unequal] += 1
        outcomes_by_stack[str(stack_index)] = {
            "stage_exact": outcomes,
            "first_unequal_boundary": first_unequal,
        }
    total = len(reports)
    return {
        "particle_count": total,
        "stage_pass_counts": {
            stage: {"passed": int(pass_counts[stage]), "total": total}
            for stage in STAGES
        },
        "first_unequal_boundary_counts": dict(sorted(first_unequal_counts.items())),
        "particles": outcomes_by_stack,
    }


def _unique_by_stack(paths: list[Path], loader, label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in paths:
        stack_index = int(loader(path).stack_index)
        _require(stack_index not in result, f"duplicate {label} capture for stack {stack_index}")
        result[stack_index] = path
    return result


def analyze_panel(
    *,
    native_capture_dir: Path,
    recovar_capture_dir: Path,
    physical_image_size: int,
    expected_stacks: set[int] | None,
) -> dict[str, object]:
    factors = _unique_by_stack(
        sorted(native_capture_dir.glob("*.bpre-v2.bin")),
        load_factor_capture,
        "native BPref",
    )
    fine_scores = _unique_by_stack(
        sorted(native_capture_dir.glob("*.fine-score-v1.bin")),
        load_fine_score_capture,
        "native fine-score",
    )
    recovar: dict[int, Path] = {}
    for path in sorted(recovar_capture_dir.glob("raw_k1_*.npz")):
        stack_index = int(load_recovar_candidate_table(path)["original_index"]) + 1
        _require(stack_index not in recovar, f"duplicate RECOVAR capture for stack {stack_index}")
        recovar[stack_index] = path
    observed = set(factors)
    _require(observed == set(fine_scores) == set(recovar), "capture stack sets differ")
    if expected_stacks is not None:
        _require(observed == expected_stacks, "capture stack set differs from expected stacks")
    reports = {
        stack_index: analyze(
            factor_path=factors[stack_index],
            fine_score_path=fine_scores[stack_index],
            recovar_path=recovar[stack_index],
            physical_image_size=physical_image_size,
        )
        for stack_index in sorted(observed)
    }
    return {
        "schema": "recovar.em.k1_partial_fine_panel.v1",
        "status": "complete",
        "summary": summarize_reports(reports),
        "reports": {str(stack_index): reports[stack_index] for stack_index in sorted(reports)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-capture-dir", type=Path, required=True)
    parser.add_argument("--recovar-capture-dir", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--expected-stacks")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_panel(
        native_capture_dir=args.native_capture_dir,
        recovar_capture_dir=args.recovar_capture_dir,
        physical_image_size=args.physical_image_size,
        expected_stacks=_parse_expected_stacks(args.expected_stacks),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
