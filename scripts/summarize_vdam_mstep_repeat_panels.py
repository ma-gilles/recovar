#!/usr/bin/env python3
"""Aggregate independent VDAM M-step repeat panels against native variability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_vdam_mstep_repeat_panel import SCHEMA as PANEL_SCHEMA

SCHEMA = "recovar.vdam_mstep_repeat_panel_summary.v1"
RAW_ACCUMULATOR_STAGES = (
    "raw_accumulator_data_half0",
    "raw_accumulator_data_half1",
    "raw_accumulator_weight_half0",
    "raw_accumulator_weight_half1",
)


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _load_panel(path: Path, *, iteration: int) -> tuple[dict, str]:
    path = path.resolve()
    payload = path.read_bytes()
    report = json.loads(payload)
    expected = {
        "schema": PANEL_SCHEMA,
        "status": "complete",
        "iteration": int(iteration),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, got {report.get(key)!r}"
            )
    return report, hashlib.sha256(payload).hexdigest()


def summarize_repeat_panels(report_paths: list[Path], *, iteration: int) -> dict:
    paths = [Path(path).resolve() for path in report_paths]
    if len(paths) < 2:
        raise ValueError("at least two independent repeat-panel reports are required")
    if len(set(paths)) != len(paths):
        raise ValueError("repeat-panel report paths must be unique")

    loaded = [_load_panel(path, iteration=iteration) for path in paths]
    reports = [item[0] for item in loaded]
    directory_sets = [tuple(sorted(report["directories"].values())) for report in reports]
    if len(set(directory_sets)) != len(directory_sets):
        raise ValueError("repeat-panel reports must describe unique capture directories")

    stages = tuple(reports[0]["native_floor_ratios"])
    for path, report in zip(paths[1:], reports[1:], strict=True):
        if tuple(report["native_floor_ratios"]) != stages:
            raise ValueError(f"{path}: stage names/order do not match the first panel")

    stage_summaries = {}
    for stage in stages:
        native_repeat = [
            float(report["native_repeat"][stage]["relative_l2"])
            for report in reports
        ]
        ratios = [
            float(value)
            for report in reports
            for value in report["native_floor_ratios"][stage].values()
            if value is not None
        ]
        stage_summaries[stage] = {
            "native_repeat_relative_l2": _distribution(native_repeat),
            "cross_over_native_repeat": {
                **_distribution(ratios),
                "within_native_floor_count": sum(value <= 1.0 for value in ratios),
                "comparison_count": len(ratios),
            },
        }

    missing_raw = [stage for stage in RAW_ACCUMULATOR_STAGES if stage not in stages]
    if missing_raw:
        raise ValueError(f"repeat panels are missing raw accumulator stages: {missing_raw}")
    raw_ratios = [
        float(value)
        for report in reports
        for stage in RAW_ACCUMULATOR_STAGES
        for value in report["native_floor_ratios"][stage].values()
        if value is not None
    ]

    return {
        "schema": SCHEMA,
        "status": "complete",
        "iteration": int(iteration),
        "panel_count": len(reports),
        "source_reports": [
            {"path": str(path), "sha256": digest}
            for path, (_, digest) in zip(paths, loaded, strict=True)
        ],
        "stages": stage_summaries,
        "raw_accumulator_gate": {
            "criterion": "cross_over_native_repeat <= 1.0",
            **_distribution(raw_ratios),
            "within_native_floor_count": sum(value <= 1.0 for value in raw_ratios),
            "comparison_count": len(raw_ratios),
            "all_within_native_floor": all(value <= 1.0 for value in raw_ratios),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_repeat_panels(args.report, iteration=args.iteration)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
