from pathlib import Path

import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_engine
from scripts.build_vdam_block_chronology import RECORD_DTYPE


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    schedule = tmp_path / "worker_schedule.npz"
    np.savez_compressed(
        schedule,
        schema_version=np.int64(2),
        iteration=np.int64(1),
        dataset_particles=np.int64(8),
        n_threads=np.int64(8),
        internal_particle_id_by_sorted_position=np.asarray([17, 23], dtype=np.int64),
        stack_index_by_sorted_position=np.asarray([5, 2], dtype=np.int64),
        owner_by_sorted_position=np.asarray([6, 1], dtype=np.int64),
    )
    records = np.zeros(6, dtype=RECORD_DTYPE)
    records["particle_id"] = np.asarray([17, 17, 17, 23, 23, 23])
    records["orientation_row"] = np.asarray([0, 1, 2, 0, 1, 2])
    records["image_count"] = 3
    records["block_start_globaltimer"] = np.asarray([30, 10, 20, 12, 32, 22])
    chronology = tmp_path / "chronology.npz"
    np.savez_compressed(
        chronology,
        schema_version=np.int64(1),
        iteration=np.int64(1),
        n_particles=np.int64(2),
        records=records,
    )
    return schedule, chronology


class _Dataset:
    @staticmethod
    def original_image_indices_from_local(image_indices):
        lookup = np.asarray([5, 2], dtype=np.int64)
        return lookup[np.asarray(image_indices)]


def _configure(monkeypatch, schedule: Path, chronology: Path) -> None:
    monkeypatch.setenv(local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV, str(schedule))
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_BLOCK_CHRONOLOGY_ENV,
        str(chronology),
    )
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured_block_start",
    )
    local_em_engine._load_relion_vdam_worker_schedule.cache_clear()
    local_em_engine._load_relion_vdam_block_start_orders.cache_clear()


def test_block_start_replay_joins_native_particle_ids_to_stack_indices(
    tmp_path, monkeypatch
):
    schedule, chronology = _write_inputs(tmp_path)
    _configure(monkeypatch, schedule, chronology)

    orders = local_em_engine._relion_vdam_block_start_orders_for_images(
        _Dataset(),
        np.asarray([1, 0], dtype=np.int64),
        rotation_count=3,
        debug_iteration=1,
    )
    np.testing.assert_array_equal(
        orders,
        np.asarray([[0, 2, 1], [1, 2, 0]], dtype=np.int32),
    )
    lanes = local_em_engine._relion_vdam_worker_lanes_for_images(
        _Dataset(),
        np.asarray([1, 0], dtype=np.int64),
        debug_iteration=1,
    )
    np.testing.assert_array_equal(lanes, np.asarray([1, 6], dtype=np.int32))


def test_block_start_replay_is_confined_to_the_traced_iteration(tmp_path, monkeypatch):
    schedule, chronology = _write_inputs(tmp_path)
    _configure(monkeypatch, schedule, chronology)

    assert (
        local_em_engine._relion_vdam_block_start_orders_for_images(
            _Dataset(),
            np.asarray([0, 1], dtype=np.int64),
            rotation_count=3,
            debug_iteration=2,
        )
        is None
    )
    assert (
        local_em_engine._relion_vdam_worker_lanes_for_images(
            _Dataset(),
            np.asarray([0, 1], dtype=np.int64),
            debug_iteration=2,
        )
        is None
    )


def test_block_start_replay_rejects_a_different_rotation_bucket(tmp_path, monkeypatch):
    schedule, chronology = _write_inputs(tmp_path)
    _configure(monkeypatch, schedule, chronology)

    with pytest.raises(ValueError, match="rotation count differs"):
        local_em_engine._relion_vdam_block_start_orders_for_images(
            _Dataset(),
            np.asarray([0], dtype=np.int64),
            rotation_count=2,
            debug_iteration=1,
        )


def test_block_start_replay_rejects_nonbijective_orientations(tmp_path, monkeypatch):
    schedule, chronology = _write_inputs(tmp_path)
    with np.load(chronology, allow_pickle=False) as sealed:
        records = np.asarray(sealed["records"]).copy()
    records["orientation_row"][1] = 0
    np.savez_compressed(
        chronology,
        schema_version=np.int64(1),
        iteration=np.int64(1),
        n_particles=np.int64(2),
        records=records,
    )
    _configure(monkeypatch, schedule, chronology)

    with pytest.raises(ValueError, match="orientation rows are not a bijection"):
        local_em_engine._relion_vdam_block_start_orders_for_images(
            _Dataset(),
            np.asarray([0], dtype=np.int64),
            rotation_count=3,
            debug_iteration=1,
        )
