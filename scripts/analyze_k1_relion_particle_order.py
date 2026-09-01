#!/usr/bin/env python3
"""Compare a live RELION post-randomization order with RECOVAR's order plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import starfile

from scripts.run_full_refinement import (
    _particle_identity_rows,
    _relion_halfset_and_accuracy_layout,
)


def _sha256_int64(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8").reshape(-1))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.int64).reshape(-1)
    expected = np.asarray(expected, dtype=np.int64).reshape(-1)
    if actual.shape != expected.shape:
        raise ValueError(f"order shapes differ: {actual.shape} versus {expected.shape}")
    unequal = np.flatnonzero(actual != expected)
    return {
        "exact_equal": bool(unequal.size == 0),
        "n": int(actual.size),
        "mismatch_count": int(unequal.size),
        "fixed_position_count": int(actual.size - unequal.size),
        "first_mismatch_position": None if unequal.size == 0 else int(unequal[0]),
        "actual_sha256_int64_le": _sha256_int64(actual),
        "expected_sha256_int64_le": _sha256_int64(expected),
        "actual_head": actual[:12].tolist(),
        "expected_head": expected[:12].tolist(),
    }


def _particles(path: Path):
    value = starfile.read(path)
    return value["particles"] if isinstance(value, dict) else value


def analyze(
    *,
    order_dump: Path,
    input_star: Path,
    relion_data_star: Path,
    random_seed: int,
    first_iteration: int,
    historical_results: Path | None,
) -> dict[str, object]:
    dump = np.loadtxt(order_dump, comments="#", dtype=np.int64, ndmin=2)
    if dump.shape[1] != 3:
        raise ValueError(f"order dump must have three columns, got {dump.shape}")
    iterations, positions, native_order = dump.T
    if not np.all(iterations == int(first_iteration)):
        raise ValueError("order dump iteration does not match the requested iteration")
    if not np.array_equal(positions, np.arange(positions.size, dtype=np.int64)):
        raise ValueError("order dump positions are not contiguous")
    if np.unique(native_order).size != native_order.size:
        raise ValueError("native order is not bijective")

    input_particles = _particles(input_star)
    relion_particles = _particles(relion_data_star)
    half1, half2, _, _, _ = _relion_halfset_and_accuracy_layout(
        input_particles,
        relion_particles,
        random_seed=int(random_seed),
        first_iteration=int(first_iteration),
    )
    expected = np.concatenate([half1, half2]).astype(np.int64, copy=False)
    if not np.array_equal(np.sort(expected), np.arange(expected.size, dtype=np.int64)):
        raise ValueError("RECOVAR order plan is not bijective")

    input_row_by_identity = _particle_identity_rows(
        input_particles,
        label="fresh RELION input STAR",
    )
    relion_row_by_identity = _particle_identity_rows(
        relion_particles,
        label="RELION numbered data STAR",
    )
    relion_identities = list(relion_row_by_identity)
    live_native_source_rows = np.asarray(
        [input_row_by_identity[relion_identities[row]] for row in native_order],
        dtype=np.int64,
    )
    input_identities = list(input_row_by_identity)
    input_to_relion_row = np.asarray(
        [relion_row_by_identity[identity] for identity in input_identities],
        dtype=np.int64,
    )
    expected_internal_rows = input_to_relion_row[expected]

    result: dict[str, object] = {
        "schema": "recovar.em.k1_relion_particle_order_audit.v1",
        "status": "complete",
        "metric_policy": "exact particle-index order; no floating-point metric",
        "order_dump": str(order_dump.resolve()),
        "input_star": str(input_star.resolve()),
        "relion_data_star": str(relion_data_star.resolve()),
        "random_seed": int(random_seed),
        "first_iteration": int(first_iteration),
        "half_sizes": [int(half1.size), int(half2.size)],
        "recovar_plan_vs_live_native_source_rows": _comparison(
            expected,
            live_native_source_rows,
        ),
        "recovar_plan_vs_live_native_internal_rows": _comparison(
            expected_internal_rows,
            native_order,
        ),
    }
    if historical_results is not None:
        with np.load(historical_results, allow_pickle=True) as saved:
            historical = np.concatenate(
                [
                    np.asarray(saved["half1_indices"], dtype=np.int64),
                    np.asarray(saved["half2_indices"], dtype=np.int64),
                ]
            )
        result["historical_plan_vs_live_native_source_rows"] = _comparison(
            historical,
            live_native_source_rows,
        )
        result["active_plan_vs_historical_plan"] = _comparison(
            expected,
            historical,
        )
        result["historical_results"] = str(historical_results.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-dump", type=Path, required=True)
    parser.add_argument("--input-star", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--first-iteration", type=int, default=1)
    parser.add_argument("--historical-results", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        order_dump=args.order_dump,
        input_star=args.input_star,
        relion_data_star=args.relion_data_star,
        random_seed=args.random_seed,
        first_iteration=args.first_iteration,
        historical_results=args.historical_results,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
