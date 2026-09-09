#!/usr/bin/env python3
"""Audit paired RELION/RECOVAR VDAM M-step captures against native repeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_vdam_mstep_boundary import (
    SCHEMA as BOUNDARY_SCHEMA,
)
from scripts.analyze_vdam_mstep_boundary import (
    _metric,
    _read_relion_array,
    _stages_for_iteration,
)

SCHEMA = "recovar.vdam_mstep_repeat_panel.v1"


def _load_cross_report(arm_root: Path, *, iteration: int) -> dict:
    arm_root = arm_root.resolve()
    path = arm_root / "analysis" / "mstep_boundary.json"
    report = json.loads(path.read_text())
    expected = {
        "schema": BOUNDARY_SCHEMA,
        "status": "complete",
        "iteration": int(iteration),
        "native_directory": str((arm_root / "native_mstep").resolve()),
        "recovar_directory": str((arm_root / "recovar_mstep").resolve()),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}, got {report.get(key)!r}")
    return report


def analyze_repeat_panel(arm_a: Path, arm_b: Path, *, iteration: int = 1) -> dict:
    arm_a = Path(arm_a).resolve()
    arm_b = Path(arm_b).resolve()
    cross_a = _load_cross_report(arm_a, iteration=iteration)
    cross_b = _load_cross_report(arm_b, iteration=iteration)
    native_repeat = {}
    recovar_repeat = {}

    for name, native_name, recovar_name, complex_values in _stages_for_iteration(iteration):
        native_repeat[name] = _metric(
            _read_relion_array(arm_b / "native_mstep" / native_name, complex_values=complex_values),
            _read_relion_array(arm_a / "native_mstep" / native_name, complex_values=complex_values),
        )
        recovar_repeat[name] = _metric(
            np.load(arm_b / "recovar_mstep" / recovar_name, allow_pickle=False),
            np.load(arm_a / "recovar_mstep" / recovar_name, allow_pickle=False),
        )

    ratios = {}
    for name, native_metric in native_repeat.items():
        native_floor = float(native_metric["relative_l2"])
        ratios[name] = {
            "cross_arm_a_over_native_repeat": (
                float(cross_a["comparisons"][name]["relative_l2"]) / native_floor
                if native_floor > 0.0
                else None
            ),
            "cross_arm_b_over_native_repeat": (
                float(cross_b["comparisons"][name]["relative_l2"]) / native_floor
                if native_floor > 0.0
                else None
            ),
        }

    return {
        "schema": SCHEMA,
        "status": "complete",
        "iteration": int(iteration),
        "directories": {
            "native_a": str(arm_a / "native_mstep"),
            "native_b": str(arm_b / "native_mstep"),
            "recovar_a": str(arm_a / "recovar_mstep"),
            "recovar_b": str(arm_b / "recovar_mstep"),
        },
        "cross_arm_a": cross_a,
        "cross_arm_b": cross_b,
        "native_repeat": native_repeat,
        "recovar_repeat": recovar_repeat,
        "native_floor_ratios": ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, required=True)
    parser.add_argument("--arm-b", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_repeat_panel(args.arm_a, args.arm_b, iteration=args.iteration)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
