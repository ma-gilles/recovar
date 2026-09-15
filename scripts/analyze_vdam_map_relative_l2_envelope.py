#!/usr/bin/env python3
"""Measure candidate map drift against a native RELION repeat diameter."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.summarize_em_completion_bench import _load_relion_volume
else:
    from summarize_em_completion_bench import _load_relion_volume


SCHEMA = "recovar.vdam_map_relative_l2_envelope.v1"
TRAJECTORY_SCHEMA = "recovar.vdam_relion_fsc_trajectory_audit.v1"


class MapRelativeL2EnvelopeError(RuntimeError):
    """Raised when the candidate/native map panel is incomplete or mixed."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MapRelativeL2EnvelopeError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MapRelativeL2EnvelopeError(f"{label} must contain a JSON object: {path}")
    return value


def symmetric_relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return direct map disagreement without alignment or scale fitting."""

    lhs_array = np.asarray(lhs, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    if lhs_array.shape != rhs_array.shape:
        raise MapRelativeL2EnvelopeError(
            f"relative-L2 map shapes differ: {lhs_array.shape} vs {rhs_array.shape}"
        )
    if not np.all(np.isfinite(lhs_array)) or not np.all(np.isfinite(rhs_array)):
        raise MapRelativeL2EnvelopeError("relative-L2 maps must contain only finite values")
    denominator = max(float(np.linalg.norm(lhs_array)), float(np.linalg.norm(rhs_array)))
    if denominator == 0.0:
        return 0.0
    value = float(np.linalg.norm(lhs_array - rhs_array) / denominator)
    if not np.isfinite(value):
        raise MapRelativeL2EnvelopeError("relative-L2 map disagreement is not finite")
    return value


def _map_path(root: Path, engine: str, iteration: int) -> Path:
    return root / engine / f"run_it{iteration:03d}_class001.mrc"


def _trajectory_identity(root: Path, *, label: str) -> tuple[str, str, tuple[int, ...]]:
    report = _load_json(root / "trajectory_audit.json", label=f"{label} trajectory audit")
    if report.get("schema") != TRAJECTORY_SCHEMA:
        raise MapRelativeL2EnvelopeError(f"{label} trajectory schema differs")
    if report.get("artifact_topology_exact") is not True:
        raise MapRelativeL2EnvelopeError(f"{label} artifact topology is incomplete")
    suite_id = str(report.get("suite_id", ""))
    case_id = str(report.get("case_id", ""))
    iterations = tuple(int(row["iteration"]) for row in report.get("checkpoints", ()))
    if not suite_id or not case_id:
        raise MapRelativeL2EnvelopeError(f"{label} suite or case identity is missing")
    if not iterations or iterations != tuple(sorted(set(iterations))):
        raise MapRelativeL2EnvelopeError(f"{label} checkpoint topology is invalid")
    return suite_id, case_id, iterations


def analyze_map_relative_l2_envelope(
    *, candidate_root: Path, native_roots: list[Path]
) -> dict[str, Any]:
    if len(native_roots) < 2:
        raise MapRelativeL2EnvelopeError("native envelope requires at least two repeats")
    suite_id, case_id, iterations = _trajectory_identity(candidate_root, label="candidate")
    for index, root in enumerate(native_roots, start=1):
        identity = _trajectory_identity(root, label=f"native repeat {index}")
        if identity != (suite_id, case_id, iterations):
            raise MapRelativeL2EnvelopeError(
                f"native repeat {index} suite, case, or checkpoint topology differs"
            )

    rows: list[dict[str, Any]] = []
    for iteration in iterations:
        candidate = _load_relion_volume(_map_path(candidate_root, "recovar", iteration))
        native = [
            _load_relion_volume(_map_path(root, "relion", iteration))
            for root in native_roots
        ]
        candidate_native = [symmetric_relative_l2(candidate, volume) for volume in native]
        native_repeat = [
            symmetric_relative_l2(native[lhs], native[rhs])
            for lhs, rhs in itertools.combinations(range(len(native)), 2)
        ]
        candidate_best = float(min(candidate_native))
        native_max = float(max(native_repeat))
        within = candidate_best <= native_max
        if native_max > 0.0:
            ratio: float | None = float(candidate_best / native_max)
        elif candidate_best == 0.0:
            ratio = 0.0
        else:
            ratio = None
        rows.append(
            {
                "iteration": iteration,
                "candidate_native_relative_l2": candidate_native,
                "candidate_best_native_relative_l2": candidate_best,
                "native_repeat_relative_l2": native_repeat,
                "native_repeat_max_relative_l2": native_max,
                "candidate_best_over_native_repeat_max_relative_l2": ratio,
                "candidate_within_native_repeat_relative_l2_envelope": within,
            }
        )

    outside = [
        row
        for row in rows
        if not row["candidate_within_native_repeat_relative_l2_envelope"]
    ]
    finite_ratios = [
        row["candidate_best_over_native_repeat_max_relative_l2"]
        for row in rows
        if row["candidate_best_over_native_repeat_max_relative_l2"] is not None
    ]
    return {
        "schema": SCHEMA,
        "suite_id": suite_id,
        "case_id": case_id,
        "scope": "direct scale-sensitive map repeatability diagnostic only; no acceptance effect",
        "definition": "norm(candidate-native) / max(norm(candidate), norm(native)); no alignment or scale fitting",
        "candidate_root": str(candidate_root),
        "native_roots": [str(root) for root in native_roots],
        "native_repeat_count": len(native_roots),
        "checkpoint_count": len(rows),
        "first_iteration_outside_native_repeat_envelope": (
            outside[0]["iteration"] if outside else None
        ),
        "outside_native_repeat_envelope_count": len(outside),
        "maximum_finite_candidate_over_native_repeat_envelope_ratio": (
            max(finite_ratios) if finite_ratios else None
        ),
        "checkpoints": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_map_relative_l2_envelope(
        candidate_root=args.candidate_root.resolve(),
        native_roots=[path.resolve() for path in args.native_root],
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
