#!/usr/bin/env python3
"""Measure VDAM compact selection across fixed source-16 capacities."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from scripts.run_vdam_schedule_representation_panel import (
    _add_common_arguments,
    _expected_arm,
    _prepare_run,
    _run_arm,
)

SCHEMA = "recovar.vdam_schedule_capacity_sweep.v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_arguments(parser)
    parser.add_argument("--capacity", action="append", type=int, required=True)
    args = parser.parse_args(argv)
    if any(capacity <= 0 for capacity in args.capacity):
        parser.error("every --capacity must be positive")
    if len(set(args.capacity)) != len(args.capacity):
        parser.error("--capacity values must be unique")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    expected_environment = _prepare_run(args)
    prewarm = _expected_arm(args, "prewarm_dynamic", 64)
    arms = [
        _run_arm(args, f"capacity_{capacity:03d}", capacity)
        for capacity in args.capacity
    ]
    report = {
        "schema": SCHEMA,
        "classification": "diagnostic_schedule_capacity_sweep",
        "checkpoint_iteration": int(args.checkpoint_iteration),
        "profiled_iteration": int(args.checkpoint_iteration) + 1,
        "environment_without_block_capacity": {
            name: value
            for name, value in sorted(expected_environment.items())
            if name != "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY"
        },
        "prewarm": prewarm,
        "capacity_order": list(args.capacity),
        "arms": {arm["label"]: arm for arm in arms},
    }
    report_path = args.output_root / "capacity_sweep.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
