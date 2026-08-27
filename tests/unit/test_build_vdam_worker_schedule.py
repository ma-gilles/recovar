from pathlib import Path

import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_engine
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


def _write_v2_trace(path: Path, *, n_particles: int = 30) -> None:
    rows = []
    for position in range(n_particles):
        pool_first = (position // 24) * 24
        pool_last = min(pool_first + 23, n_particles - 1)
        rows.append(
            f"2 1 {pool_first} {pool_last} {position} "
            f"{n_particles - 1 - position} {100 - n_particles + position} "
            f"{position % 8}"
        )
    path.write_text(
        "# RELION_VDAM_WORKER_LOG_SCHEMA_V2\n"
        "# schema iteration pool_first_sorted pool_last_sorted sorted_position "
        "internal_part_id stack_index_zero_based thread_id\n"
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
        dataset_particles=100,
        n_threads=8,
        pool_size=24,
    )
    np.testing.assert_array_equal(schedule["owner_by_sorted_position"], np.arange(30) % 8)
    np.testing.assert_array_equal(
        schedule["internal_particle_id_by_sorted_position"], np.arange(29, -1, -1)
    )
    assert schedule["stack_index_by_sorted_position"] is None
    assert schedule["pool_count"] == 2
    assert schedule["thread_counts"] == [4, 4, 4, 4, 4, 4, 3, 3]


def test_worker_trace_rejects_missing_schema_marker(tmp_path):
    trace = tmp_path / "workers.tsv"
    trace.write_text("1 1 0 0 0 0 0\n")
    with pytest.raises(ValueError, match="missing exact schema marker"):
        load_worker_trace(trace)


def test_v2_worker_trace_preserves_stack_index_join_key(tmp_path):
    trace = tmp_path / "workers-v2.tsv"
    _write_v2_trace(trace)
    schedule = validate_worker_trace(
        load_worker_trace(trace),
        iteration=1,
        n_particles=30,
        dataset_particles=100,
        n_threads=8,
        pool_size=24,
    )
    assert schedule["schema_version"] == 2
    np.testing.assert_array_equal(
        schedule["stack_index_by_sorted_position"], np.arange(70, 100)
    )
    np.testing.assert_array_equal(
        schedule["owner_by_sorted_position"], np.arange(30) % 8
    )


def test_v2_schedule_resolves_worker_lanes_by_stack_index(tmp_path, monkeypatch):
    schedule = tmp_path / "schedule.npz"
    np.savez_compressed(
        schedule,
        schema_version=np.int64(2),
        dataset_particles=np.int64(100),
        n_threads=np.int64(8),
        stack_index_by_sorted_position=np.asarray([71, 14, 82], dtype=np.int64),
        owner_by_sorted_position=np.asarray([6, 2, 5], dtype=np.int64),
    )

    class Dataset:
        @staticmethod
        def original_image_indices_from_local(image_indices):
            lookup = np.asarray([82, 71, 14], dtype=np.int64)
            return lookup[np.asarray(image_indices)]

    monkeypatch.setenv(local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV, str(schedule))
    local_em_engine._load_relion_vdam_worker_schedule.cache_clear()
    lanes = local_em_engine._relion_vdam_worker_lanes_for_images(
        Dataset(), np.asarray([1, 2, 0], dtype=np.int64)
    )
    np.testing.assert_array_equal(lanes, np.asarray([6, 2, 5], dtype=np.int32))


def test_v2_schedule_rejects_an_untraced_selected_stack_index(tmp_path, monkeypatch):
    schedule = tmp_path / "schedule.npz"
    np.savez_compressed(
        schedule,
        schema_version=np.int64(2),
        dataset_particles=np.int64(100),
        n_threads=np.int64(8),
        stack_index_by_sorted_position=np.asarray([71], dtype=np.int64),
        owner_by_sorted_position=np.asarray([6], dtype=np.int64),
    )

    class Dataset:
        @staticmethod
        def original_image_indices_from_local(image_indices):
            return np.asarray([72], dtype=np.int64)

    monkeypatch.setenv(local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV, str(schedule))
    local_em_engine._load_relion_vdam_worker_schedule.cache_clear()
    with pytest.raises(ValueError, match=r"missing selected stack indices \[72\]"):
        local_em_engine._relion_vdam_worker_lanes_for_images(
            Dataset(), np.asarray([0], dtype=np.int64)
        )


def test_worker_trace_rejects_duplicate_sorted_position(tmp_path):
    trace = tmp_path / "workers.tsv"
    _write_trace(trace)
    rows = load_worker_trace(trace)
    rows[1, 4] = rows[0, 4]
    with pytest.raises(ValueError, match="sorted positions"):
        validate_worker_trace(
            rows,
            iteration=1,
            n_particles=30,
            dataset_particles=100,
            n_threads=8,
            pool_size=24,
        )


@pytest.mark.parametrize("column,value,match", [(2, 1, "pool_first"), (6, 8, "worker IDs")])
def test_worker_trace_rejects_invalid_topology(tmp_path, column, value, match):
    trace = tmp_path / "workers.tsv"
    _write_trace(trace)
    rows = load_worker_trace(trace)
    rows[0, column] = value
    with pytest.raises(ValueError, match=match):
        validate_worker_trace(
            rows,
            iteration=1,
            n_particles=30,
            dataset_particles=100,
            n_threads=8,
            pool_size=24,
        )
