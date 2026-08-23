#!/usr/bin/env python3
"""Measure loss from serializing a live K=1 direction prior through STAR."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import starfile

REPORT_SCHEMA = "recovar.em.k1_direction_prior_serialization.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    if reference64.shape != candidate64.shape:
        raise ValueError(f"shape mismatch: {reference64.shape} != {candidate64.shape}")
    delta = candidate64 - reference64
    denominator = float(np.linalg.norm(reference64))
    return {
        "shape": list(reference64.shape),
        "exact_equal": bool(np.array_equal(reference64, candidate64)),
        "mismatch_count": int(np.count_nonzero(reference64 != candidate64)),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "relative_l2_over_live": (
            float(np.linalg.norm(delta) / denominator) if denominator > 0.0 else None
        ),
    }


def compare_direction_priors(live: np.ndarray, serialized: np.ndarray) -> dict[str, Any]:
    live = np.asarray(live, dtype=np.float32).reshape(-1)
    serialized = np.asarray(serialized, dtype=np.float32).reshape(-1)
    if live.shape != serialized.shape:
        raise ValueError(f"direction-prior shape mismatch: {live.shape} != {serialized.shape}")
    if np.any(live < 0) or np.any(serialized < 0):
        raise ValueError("direction priors must be nonnegative")
    finite = (live > 0) & (serialized > 0)
    if not np.any(finite):
        raise ValueError("direction priors have no common positive entry")
    live_log = np.log(live[finite], dtype=np.float32)
    serialized_log = np.log(serialized[finite], dtype=np.float32)
    log_delta = serialized_log - live_log
    max_index_finite = int(np.argmax(np.abs(log_delta)))
    direction_index = int(np.flatnonzero(finite)[max_index_finite])
    return {
        "n_directions": int(live.size),
        "live_mass": float(np.sum(live, dtype=np.float64)),
        "serialized_mass": float(np.sum(serialized, dtype=np.float64)),
        "zero_mask_exact": bool(np.array_equal(live == 0, serialized == 0)),
        "probability": _metric(live, serialized),
        "finite_log_probability": _metric(live_log, serialized_log),
        "maximum_log_delta": {
            "direction_index": direction_index,
            "live_probability": float(live[direction_index]),
            "serialized_probability": float(serialized[direction_index]),
            "live_log_probability": float(live_log[max_index_finite]),
            "serialized_log_probability": float(serialized_log[max_index_finite]),
            "absolute_delta": float(abs(log_delta[max_index_finite])),
        },
    }


def _read_star_prior(path: Path) -> np.ndarray:
    model = starfile.read(path)
    key = "model_pdf_orient_class_1"
    if key not in model:
        raise ValueError(f"{path} does not contain {key}")
    return np.asarray(model[key]["rlnOrientationDistribution"], dtype=np.float32)


def analyze(
    *,
    refinement_results: Path,
    relion_half_models: tuple[Path, Path],
    iteration_index: int,
    staged_fine_report: Path | None,
) -> dict[str, Any]:
    with np.load(refinement_results, allow_pickle=True) as payload:
        live_trajectory = np.asarray(payload["direction_prior_trajectory_per_half"], dtype=np.float32)
    if live_trajectory.ndim != 3 or live_trajectory.shape[1] != 2:
        raise ValueError(f"unexpected live direction-prior shape {live_trajectory.shape}")
    if not 0 <= iteration_index < live_trajectory.shape[0]:
        raise ValueError(f"iteration index {iteration_index} outside {live_trajectory.shape[0]} states")

    halves = []
    for half_index, model_path in enumerate(relion_half_models):
        halves.append(
            compare_direction_priors(
                live_trajectory[iteration_index, half_index],
                _read_star_prior(model_path),
            )
        )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "metric_policy": "exact, mass, maximum absolute, and relative L2; no correlation",
        "iteration_index": iteration_index,
        "sources": {
            "refinement_results": str(refinement_results.resolve()),
            "refinement_results_sha256": _sha256(refinement_results),
            "relion_half_models": [str(path.resolve()) for path in relion_half_models],
            "relion_half_model_sha256": [_sha256(path) for path in relion_half_models],
        },
        "halves": halves,
    }
    if staged_fine_report is not None:
        staged = json.loads(staged_fine_report.read_text())
        observed = float(staged["comparisons"]["orientation_log_prior"]["max_abs"])
        predicted = [float(half["finite_log_probability"]["max_abs"]) for half in halves]
        report["staged_fine_score_crosscheck"] = {
            "path": str(staged_fine_report.resolve()),
            "sha256": _sha256(staged_fine_report),
            "observed_orientation_log_prior_max_abs": observed,
            "predicted_half_log_prior_max_abs": predicted,
            "exact_match_by_half": [observed == value for value in predicted],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-results", type=Path, required=True)
    parser.add_argument("--relion-half1-model", type=Path, required=True)
    parser.add_argument("--relion-half2-model", type=Path, required=True)
    parser.add_argument("--iteration-index", type=int, default=0)
    parser.add_argument("--staged-fine-report", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    report = analyze(
        refinement_results=args.refinement_results,
        relion_half_models=(args.relion_half1_model, args.relion_half2_model),
        iteration_index=args.iteration_index,
        staged_fine_report=args.staged_fine_report,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
