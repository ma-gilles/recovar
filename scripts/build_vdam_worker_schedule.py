#!/usr/bin/env python
"""Validate a RELION VDAM OpenMP worker trace and seal its replay schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_MARKERS = {
    "# RELION_VDAM_WORKER_LOG_SCHEMA_V1": 1,
    "# RELION_VDAM_WORKER_LOG_SCHEMA_V2": 2,
}


def load_worker_trace(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    if not lines or lines[0] not in SCHEMA_MARKERS:
        raise ValueError(
            "missing exact schema marker; expected one of "
            f"{sorted(SCHEMA_MARKERS)!r}"
        )
    rows = np.loadtxt(path, dtype=np.int64, comments="#", ndmin=2)
    schema_version = SCHEMA_MARKERS[lines[0]]
    expected_columns = 7 if schema_version == 1 else 8
    if rows.ndim != 2 or rows.shape[1] != expected_columns:
        raise ValueError(
            f"v{schema_version} worker trace must contain {expected_columns} "
            f"integer columns; got {rows.shape}"
        )
    if not np.all(rows[:, 0] == schema_version):
        raise ValueError(f"worker trace contains a non-v{schema_version} record")
    return rows


def validate_worker_trace(
    rows: np.ndarray,
    *,
    iteration: int,
    n_particles: int,
    dataset_particles: int,
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

    schema_version = int(selected[0, 0])
    internal_ids = selected[:, 5]
    if (
        np.unique(internal_ids).size != n_particles
        or np.any(internal_ids < 0)
        or np.any(internal_ids >= dataset_particles)
    ):
        raise ValueError("internal particle IDs are not unique rows in the full dataset")
    if schema_version == 2:
        stack_indices = selected[:, 6]
        if (
            np.unique(stack_indices).size != n_particles
            or np.any(stack_indices < 0)
            or np.any(stack_indices >= dataset_particles)
        ):
            raise ValueError("stack indices are not unique rows in the full dataset")
        owners = selected[:, 7]
    else:
        stack_indices = None
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
        "schema_version": schema_version,
        "owner_by_sorted_position": owners,
        "internal_particle_id_by_sorted_position": internal_ids,
        "stack_index_by_sorted_position": stack_indices,
        "pool_first_by_sorted_position": expected_pool_first,
        "thread_counts": thread_counts.tolist(),
        "pool_count": int(np.unique(expected_pool_first).size),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--n-particles", required=True, type=int)
    parser.add_argument("--dataset-particles", required=True, type=int)
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
        dataset_particles=args.dataset_particles,
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
        schema_version=np.int64(schedule["schema_version"]),
        iteration=np.int64(args.iteration),
        n_particles=np.int64(args.n_particles),
        dataset_particles=np.int64(args.dataset_particles),
        n_threads=np.int64(args.n_threads),
        pool_size=np.int64(args.pool_size),
        owner_by_sorted_position=schedule["owner_by_sorted_position"],
        internal_particle_id_by_sorted_position=(
            schedule["internal_particle_id_by_sorted_position"]
        ),
        stack_index_by_sorted_position=(
            np.asarray([], dtype=np.int64)
            if schedule["stack_index_by_sorted_position"] is None
            else schedule["stack_index_by_sorted_position"]
        ),
        pool_first_by_sorted_position=schedule["pool_first_by_sorted_position"],
        source_trace_sha256=np.asarray(trace_sha256),
    )
    report = {
        "schema": "recovar.vdam_worker_schedule.v1",
        "result": "pass",
        "source_trace": str(trace),
        "source_trace_sha256": trace_sha256,
        "trace_schema_version": schedule["schema_version"],
        "iteration": schedule["iteration"],
        "n_particles": args.n_particles,
        "dataset_particles": args.dataset_particles,
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
