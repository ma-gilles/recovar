#!/usr/bin/env python3
"""Compare direct VDAM candidate-repeat map spread with native repeats.

This diagnostic accepts immutable map-prefix directories directly.  It does
not infer trajectory provenance or promote a frozen score; callers must supply
one already-qualified, common-frame native panel.  Keeping that admission
explicit prevents a distant native basin from silently making the diameter
large enough to accept unrelated candidate maps.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.analyze_vdam_map_relative_l2_envelope import symmetric_relative_l2
    from scripts.summarize_em_completion_bench import _load_relion_volume
else:
    from analyze_vdam_map_relative_l2_envelope import symmetric_relative_l2
    from summarize_em_completion_bench import _load_relion_volume


SCHEMA = "recovar.vdam_direct_repeat_map_spread.v1"


class DirectRepeatMapSpreadError(RuntimeError):
    """Raised when a direct repeat panel is incomplete or inconsistent."""


def _validated_maps(maps: list[np.ndarray], *, label: str) -> list[np.ndarray]:
    if len(maps) < 2:
        raise DirectRepeatMapSpreadError(f"{label} panel requires at least two maps")
    arrays = [np.asarray(volume, dtype=np.float64) for volume in maps]
    shape = arrays[0].shape
    if len(shape) != 3 or len(set(shape)) != 1:
        raise DirectRepeatMapSpreadError(f"{label} maps must be cubic 3D arrays, got {shape}")
    for index, array in enumerate(arrays, start=1):
        if array.shape != shape:
            raise DirectRepeatMapSpreadError(
                f"{label} map {index} shape differs: {array.shape} vs {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise DirectRepeatMapSpreadError(f"{label} map {index} contains non-finite values")
    return arrays


def summarize_repeat_map_spread(
    candidate_maps: list[np.ndarray],
    native_maps: list[np.ndarray],
) -> dict[str, Any]:
    """Return one iteration's direct, scale-sensitive repeat geometry."""

    candidate = _validated_maps(candidate_maps, label="candidate")
    native = _validated_maps(native_maps, label="native")
    if candidate[0].shape != native[0].shape:
        raise DirectRepeatMapSpreadError(
            f"candidate/native map shapes differ: {candidate[0].shape} vs {native[0].shape}"
        )

    candidate_pairs = [
        symmetric_relative_l2(candidate[lhs], candidate[rhs])
        for lhs, rhs in itertools.combinations(range(len(candidate)), 2)
    ]
    native_pairs = [
        symmetric_relative_l2(native[lhs], native[rhs])
        for lhs, rhs in itertools.combinations(range(len(native)), 2)
    ]
    native_diameter = float(max(native_pairs))
    candidate_diameter = float(max(candidate_pairs))
    cross = np.asarray(
        [
            [symmetric_relative_l2(candidate_map, native_map) for native_map in native]
            for candidate_map in candidate
        ],
        dtype=np.float64,
    )
    nearest_indices = np.argmin(cross, axis=1)
    nearest = cross[np.arange(cross.shape[0]), nearest_indices]
    within = nearest <= native_diameter

    def over_native(value: float) -> float | None:
        if native_diameter > 0.0:
            return float(value / native_diameter)
        return 0.0 if value == 0.0 else None

    return {
        "pass": bool(np.all(within)),
        "candidate_repeat_count": len(candidate),
        "native_repeat_count": len(native),
        "candidate_pair_relative_l2": candidate_pairs,
        "candidate_diameter_relative_l2": candidate_diameter,
        "native_pair_relative_l2": native_pairs,
        "native_diameter_relative_l2": native_diameter,
        "candidate_diameter_over_native_diameter": over_native(candidate_diameter),
        "candidate_native_relative_l2": cross.tolist(),
        "candidate_nearest_native_index": (nearest_indices + 1).tolist(),
        "candidate_nearest_native_relative_l2": nearest.tolist(),
        "candidate_nearest_over_native_diameter": [over_native(float(value)) for value in nearest],
        "candidate_within_native_diameter": within.tolist(),
        "candidate_indices_outside_native_diameter": (np.flatnonzero(~within) + 1).tolist(),
    }


def analyze_direct_repeat_map_spread(
    *,
    candidate_prefixes: list[Path],
    native_prefixes: list[Path],
    iterations: tuple[int, ...],
) -> dict[str, Any]:
    """Load direct map prefixes and summarize every requested iteration."""

    if len(candidate_prefixes) < 2 or len(native_prefixes) < 2:
        raise DirectRepeatMapSpreadError("candidate and native panels each require at least two prefixes")
    if not iterations or iterations != tuple(sorted(set(iterations))) or iterations[0] < 0:
        raise DirectRepeatMapSpreadError("iterations must be sorted, unique, and non-negative")
    if len(set(candidate_prefixes)) != len(candidate_prefixes):
        raise DirectRepeatMapSpreadError("candidate prefixes must be unique")
    if len(set(native_prefixes)) != len(native_prefixes):
        raise DirectRepeatMapSpreadError("native prefixes must be unique")

    rows = []
    for iteration in iterations:
        filename = f"run_it{iteration:03d}_class001.mrc"
        candidate_paths = [prefix / filename for prefix in candidate_prefixes]
        native_paths = [prefix / filename for prefix in native_prefixes]
        missing = [path for path in candidate_paths + native_paths if not path.is_file()]
        if missing:
            raise DirectRepeatMapSpreadError(
                f"iteration {iteration} is missing map files: {[str(path) for path in missing]}"
            )
        row = summarize_repeat_map_spread(
            [_load_relion_volume(path) for path in candidate_paths],
            [_load_relion_volume(path) for path in native_paths],
        )
        rows.append({"iteration": int(iteration), **row})

    failures = [row for row in rows if not row["pass"]]
    return {
        "schema": SCHEMA,
        "result": "fail" if failures else "pass",
        "scope": "diagnostic direct map repeatability; no frozen-score acceptance effect",
        "definition": "norm(lhs-rhs) / max(norm(lhs), norm(rhs)); no alignment or scale fitting",
        "native_panel_contract": (
            "callers must supply one provenance-qualified common-frame native panel; "
            "the analyzer does not merge distant native basins"
        ),
        "candidate_prefixes": [str(path) for path in candidate_prefixes],
        "native_prefixes": [str(path) for path in native_prefixes],
        "iterations": list(iterations),
        "first_failure_iteration": None if not failures else failures[0]["iteration"],
        "failure_count": len(failures),
        "checkpoints": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-prefix", type=Path, action="append", required=True)
    parser.add_argument("--native-prefix", type=Path, action="append", required=True)
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_direct_repeat_map_spread(
        candidate_prefixes=[path.resolve() for path in args.candidate_prefix],
        native_prefixes=[path.resolve() for path in args.native_prefix],
        iterations=tuple(args.iterations),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
