#!/usr/bin/env python
"""Validate a RELION VDAM OpenMP worker trace and seal its replay schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_MARKER = "# RELION_VDAM_WORKER_LOG_SCHEMA_V1"


def load_worker_trace(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != SCHEMA_MARKER:
        raise ValueError(f"missing exact schema marker {SCHEMA_MARKER!r}")
    rows = np.loadtxt(path, dtype=np.int64, comments="#", ndmin=2)
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError(f"worker trace must contain seven integer columns; got {rows.shape}")
    if not np.all(rows[:, 0] == 1):
        raise ValueError("worker trace contains a non-v1 record")
    return rows


def validate_worker_trace(
    rows: np.ndarray,
    *,
    iteration: int,
    n_particles: int,
    n_threads: int,
    pool_size: int,
) -> dict[str, np.ndarray | int | list[int]]:
    selected = rows[rows[:, 1] == iteration]
    if selected.shape[0] != n_particles:
        raise ValueError(
            f"iteration {iteration} must contain {n_particles} rows; got {selected.shape[0]}"
        )
    order = np.argsort(selected[:, 4], kind="stable")
    selected = selected[order]
    positions = selected[:, 4]
    expected_positions = np.arange(n_particles, dtype=np.int64)
    if not np.array_equal(positions, expected_positions):
        raise ValueError("sorted positions are not an exact zero-based particle bijection")

    original_ids = selected[:, 5]
    if not np.array_equal(np.sort(original_ids), expected_positions):
        raise ValueError("original particle IDs are not an exact zero-based particle bijection")
    owners = selected[:, 6]
    if np.any(owners < 0) or np.any(owners >= n_threads):
        raise ValueError(f"worker IDs must be in [0, {n_threads})")

    expected_pool_first = (positions // pool_size) * pool_size
    expected_pool_last = np.minimum(expected_pool_first + pool_size - 1, n_particles - 1)
    if not np.array_equal(selected[:, 2], expected_pool_first):
        raise ValueError("pool_first_sorted does not match fixed physical pool boundaries")
    if not np.array_equal(selected[:, 3], expected_pool_last):
        raise ValueError("pool_last_sorted does not match fixed physical pool boundaries")

    thread_counts = np.bincount(owners, minlength=n_threads).astype(np.int64)
    return {
        "iteration": int(iteration),
        "owner_by_sorted_position": owners,
        "original_particle_id_by_sorted_position": original_ids,
        "pool_first_by_sorted_position": expected_pool_first,
        "thread_counts": thread_counts.tolist(),
        "pool_count": int(np.unique(expected_pool_first).size),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--n-particles", required=True, type=int)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=24)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    trace = args.trace.expanduser().resolve(strict=True)
    rows = load_worker_trace(trace)
    schedule = validate_worker_trace(
        rows,
        iteration=args.iteration,
        n_particles=args.n_particles,
        n_threads=args.n_threads,
        pool_size=args.pool_size,
    )
    trace_sha256 = hashlib.sha256(trace.read_bytes()).hexdigest()

    output_npz = args.output_npz.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        schema_version=np.int64(1),
        iteration=np.int64(args.iteration),
        n_particles=np.int64(args.n_particles),
        n_threads=np.int64(args.n_threads),
        pool_size=np.int64(args.pool_size),
        owner_by_sorted_position=schedule["owner_by_sorted_position"],
        original_particle_id_by_sorted_position=(
            schedule["original_particle_id_by_sorted_position"]
        ),
        pool_first_by_sorted_position=schedule["pool_first_by_sorted_position"],
        source_trace_sha256=np.asarray(trace_sha256),
    )
    report = {
        "schema": "recovar.vdam_worker_schedule.v1",
        "result": "pass",
        "source_trace": str(trace),
        "source_trace_sha256": trace_sha256,
        "iteration": schedule["iteration"],
        "n_particles": args.n_particles,
        "n_threads": args.n_threads,
        "pool_size": args.pool_size,
        "pool_count": schedule["pool_count"],
        "thread_counts": schedule["thread_counts"],
        "output_npz": str(output_npz),
    }
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
