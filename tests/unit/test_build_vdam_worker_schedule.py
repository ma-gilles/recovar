from pathlib import Path

import numpy as np
import pytest

from scripts.build_vdam_worker_schedule import load_worker_trace, validate_worker_trace


def _write_trace(path: Path, *, n_particles: int = 30) -> None:
    rows = []
    for position in range(n_particles):
        pool_first = (position // 24) * 24
        pool_last = min(pool_first + 23, n_particles - 1)
        rows.append(
            f"1 1 {pool_first} {pool_last} {position} {n_particles - 1 - position} "
            f"{position % 8}"
        )
    path.write_text(
        "# RELION_VDAM_WORKER_LOG_SCHEMA_V1\n"
        "# schema iteration pool_first_sorted pool_last_sorted sorted_position "
        "original_part_id thread_id\n"
        + "\n".join(rows)
        + "\n"
    )


def test_valid_worker_trace_is_sealed_by_sorted_position(tmp_path):
    trace = tmp_path / "workers.tsv"
    _write_trace(trace)
    schedule = validate_worker_trace(
        load_worker_trace(trace),
        iteration=1,
        n_particles=30,
        n_threads=8,
        pool_size=24,
    )
    np.testing.assert_array_equal(schedule["owner_by_sorted_position"], np.arange(30) % 8)
    np.testing.assert_array_equal(
        schedule["original_particle_id_by_sorted_position"], np.arange(29, -1, -1)
    )
    assert schedule["pool_count"] == 2
    assert schedule["thread_counts"] == [4, 4, 4, 4, 4, 4, 3, 3]


def test_worker_trace_rejects_missing_schema_marker(tmp_path):
    trace = tmp_path / "workers.tsv"
    trace.write_text("1 1 0 0 0 0 0\n")
    with pytest.raises(ValueError, match="missing exact schema marker"):
        load_worker_trace(trace)


def test_worker_trace_rejects_duplicate_sorted_position(tmp_path):
    trace = tmp_path / "workers.tsv"
    _write_trace(trace)
    rows = load_worker_trace(trace)
    rows[1, 4] = rows[0, 4]
    with pytest.raises(ValueError, match="sorted positions"):
        validate_worker_trace(
            rows, iteration=1, n_particles=30, n_threads=8, pool_size=24
        )


@pytest.mark.parametrize("column,value,match", [(2, 1, "pool_first"), (6, 8, "worker IDs")])
def test_worker_trace_rejects_invalid_topology(tmp_path, column, value, match):
    trace = tmp_path / "workers.tsv"
    _write_trace(trace)
    rows = load_worker_trace(trace)
    rows[0, column] = value
    with pytest.raises(ValueError, match=match):
        validate_worker_trace(
            rows, iteration=1, n_particles=30, n_threads=8, pool_size=24
        )
