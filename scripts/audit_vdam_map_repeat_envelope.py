#!/usr/bin/env python3
"""Compare a VDAM map trajectory with the native RELION repeat envelope."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.summarize_em_completion_bench import (
    _load_relion_volume,
    normalized_fsc_auc,
    shell_fsc,
)


SCHEMA = "recovar.vdam_map_repeat_envelope.v1"


def _fsc_metric(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    key: str,
    shellwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    if lhs.shape != rhs.shape:
        raise ValueError(f"map shapes differ for {key}: {lhs.shape} != {rhs.shape}")
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    if curve.size <= 1 or not np.isfinite(curve[1:]).any():
        raise ValueError(f"{key} produced no finite non-DC FSC shells")
    auc = float(normalized_fsc_auc(curve))
    shellwise[key] = curve
    return {
        "fsc_auc": auc,
        "fsc_distance": float(1.0 - auc),
        "n_shells": int(curve.size),
        "shellwise_key": key,
    }


def compare_repeat_envelope(
    candidate: np.ndarray,
    natives: list[np.ndarray],
    *,
    iteration: int,
    shellwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Measure candidate distance relative to the native-repeat diameter."""

    if len(natives) < 2:
        raise ValueError("at least two native repeats are required")
    candidate_metrics = [
        _fsc_metric(
            candidate,
            native,
            key=f"it{iteration:03d}_candidate_native{index:02d}",
            shellwise=shellwise,
        )
        for index, native in enumerate(natives, start=1)
    ]
    native_metrics = [
        _fsc_metric(
            natives[first],
            natives[second],
            key=f"it{iteration:03d}_native{first + 1:02d}_native{second + 1:02d}",
            shellwise=shellwise,
        )
        for first, second in itertools.combinations(range(len(natives)), 2)
    ]
    nearest_candidate_distance = min(row["fsc_distance"] for row in candidate_metrics)
    native_diameter = max(row["fsc_distance"] for row in native_metrics)
    if native_diameter > np.finfo(np.float64).eps:
        diameter_ratio: float | None = float(nearest_candidate_distance / native_diameter)
    else:
        diameter_ratio = 0.0 if nearest_candidate_distance <= 0.0 else None
    return {
        "iteration": int(iteration),
        "candidate_vs_native": candidate_metrics,
        "native_pairwise": native_metrics,
        "nearest_candidate_fsc_distance": float(nearest_candidate_distance),
        "native_repeat_fsc_diameter": float(native_diameter),
        "candidate_to_native_diameter_ratio": diameter_ratio,
        "inside_native_repeat_fsc_diameter": bool(
            nearest_candidate_distance <= native_diameter + 1.0e-12
        ),
    }


def audit(
    candidate_dir: Path,
    native_dirs: list[Path],
    iterations: list[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(native_dirs) < 2:
        raise ValueError("at least two --native-dir values are required")
    if not iterations or iterations != sorted(set(iterations)):
        raise ValueError("--iterations must be non-empty, sorted, and unique")
    shellwise: dict[str, np.ndarray] = {}
    checkpoints = []
    for iteration in iterations:
        name = f"run_it{iteration:03d}_class001.mrc"
        candidate_path = candidate_dir / name
        native_paths = [directory / name for directory in native_dirs]
        missing = [str(path) for path in (candidate_path, *native_paths) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"iteration {iteration} is missing maps: {missing}")
        candidate = _load_relion_volume(candidate_path)
        natives = [_load_relion_volume(path) for path in native_paths]
        checkpoints.append(
            compare_repeat_envelope(
                candidate,
                natives,
                iteration=iteration,
                shellwise=shellwise,
            )
        )
    first_outside = next(
        (row["iteration"] for row in checkpoints if not row["inside_native_repeat_fsc_diameter"]),
        None,
    )
    return (
        {
            "schema": SCHEMA,
            "scope": "diagnostic signed FSC envelope; not a frozen-suite promotion",
            "correlation_used": False,
            "candidate_dir": str(candidate_dir.resolve()),
            "native_dirs": [str(path.resolve()) for path in native_dirs],
            "native_repeat_count": len(native_dirs),
            "first_outside_iteration": first_outside,
            "result": "pass" if first_outside is None else "fail",
            "checkpoints": checkpoints,
        },
        shellwise,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, action="append", required=True)
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shells-npz", type=Path, required=True)
    args = parser.parse_args(argv)
    report, shellwise = audit(args.candidate_dir, args.native_dir, args.iterations)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_shells_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(args.output_shells_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
