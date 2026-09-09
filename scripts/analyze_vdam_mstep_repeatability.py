#!/usr/bin/env python3
"""Compare repeated RELION and RECOVAR InitialModel M-step captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_vdam_mstep_boundary import (
        _STAGES,
        _metric,
        _read_relion_array,
        analyze,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from analyze_vdam_mstep_boundary import (  # type: ignore[no-redef]
        _STAGES,
        _metric,
        _read_relion_array,
        analyze,
    )


SCHEMA = "recovar.vdam_mstep_repeatability.v1"


def compare(
    native_a: Path,
    native_b: Path,
    recovar_a: Path,
    recovar_b: Path,
) -> dict[str, object]:
    native_a = Path(native_a).resolve()
    native_b = Path(native_b).resolve()
    recovar_a = Path(recovar_a).resolve()
    recovar_b = Path(recovar_b).resolve()

    native_repeat = {}
    recovar_repeat = {}
    for name, native_name, recovar_name, complex_values in _STAGES:
        native_repeat[name] = _metric(
            _read_relion_array(native_b / native_name, complex_values=complex_values),
            _read_relion_array(native_a / native_name, complex_values=complex_values),
        )
        recovar_repeat[name] = _metric(
            np.load(recovar_b / recovar_name, allow_pickle=False),
            np.load(recovar_a / recovar_name, allow_pickle=False),
        )
    cross_arm_a = analyze(native_a, recovar_a)
    cross_arm_b = analyze(native_b, recovar_b)
    native_floor_ratios = {}
    for name, *_ in _STAGES:
        native_floor = float(native_repeat[name]["relative_l2"])
        native_floor_ratios[name] = {
            "cross_arm_a_over_native_repeat": (
                float(cross_arm_a["comparisons"][name]["relative_l2"])
                / native_floor
                if native_floor > 0.0
                else None
            ),
            "cross_arm_b_over_native_repeat": (
                float(cross_arm_b["comparisons"][name]["relative_l2"])
                / native_floor
                if native_floor > 0.0
                else None
            ),
        }

    return {
        "schema": SCHEMA,
        "status": "complete",
        "directories": {
            "native_a": str(native_a),
            "native_b": str(native_b),
            "recovar_a": str(recovar_a),
            "recovar_b": str(recovar_b),
        },
        "native_repeat": native_repeat,
        "recovar_repeat": recovar_repeat,
        "native_floor_ratios": native_floor_ratios,
        "cross_arm_a": cross_arm_a,
        "cross_arm_b": cross_arm_b,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-a", type=Path, required=True)
    parser.add_argument("--native-b", type=Path, required=True)
    parser.add_argument("--recovar-a", type=Path, required=True)
    parser.add_argument("--recovar-b", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.native_a, args.native_b, args.recovar_a, args.recovar_b)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
