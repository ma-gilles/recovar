from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume import local_em_engine
from scripts.build_vdam_candidate_block_map import INVALID_ROW, load_map


def test_candidate_block_map_writer_appends_exact_physical_rows(tmp_path, monkeypatch):
    output = tmp_path / "candidate-map.bin"
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_TRACE_ENV, str(tmp_path / "trace.bin"))
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_MAP_ENV, str(output))
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_MAP_ITER_ENV, "1")
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_MAP_CAPACITY_ENV, "16")

    local_em_engine._maybe_write_vdam_candidate_block_map(
        particle_ids=np.array([10, 20], dtype=np.int32),
        reconstruction_take_indices=np.array([[2, 0, 0], [1, 3, 0]], dtype=np.int32),
        reconstruction_pack_mask=np.array([[True, False, False], [True, True, False]]),
        reconstruction_contributing_mask=np.array(
            [[True, False, False], [True, False, False]]
        ),
        local_rotation_ids=np.array([[100, 101, 102, 103], [200, 201, 202, 203]], dtype=np.int32),
        reconstruction_group_ids=np.array([0, 1], dtype=np.int32),
        debug_iteration=1,
    )
    local_em_engine._maybe_write_vdam_candidate_block_map(
        particle_ids=np.array([30], dtype=np.int32),
        reconstruction_take_indices=np.array([[0, 1]], dtype=np.int32),
        reconstruction_pack_mask=np.array([[True, False]]),
        reconstruction_contributing_mask=np.array([[True, False]]),
        local_rotation_ids=np.array([[300, 301]], dtype=np.int32),
        reconstruction_group_ids=None,
        debug_iteration=1,
    )

    header, records = load_map(output)
    assert header["schema_version"] == 2
    assert header["record_count"] == 8
    assert records["particle_id"].tolist() == [10, 10, 10, 20, 20, 20, 30, 30]
    assert records["candidate_orientation_row"].tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert records["native_orientation_row"].tolist() == [2, INVALID_ROW, INVALID_ROW, 1, 3, INVALID_ROW, 0, INVALID_ROW]
    assert records["global_rotation_id"].tolist() == [102, -1, -1, 201, 203, -1, 300, -1]
    assert records["reconstruction_group_id"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    assert (records["flags"] != 0).tolist() == [True, False, False, True, True, False, True, False]
    assert records["flags"].tolist() == [3, 0, 0, 3, 1, 0, 3, 0]


def test_candidate_block_map_writer_ignores_non_target_iteration(tmp_path, monkeypatch):
    output = tmp_path / "candidate-map.bin"
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_TRACE_ENV, str(tmp_path / "trace.bin"))
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_MAP_ENV, str(output))
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_MAP_ITER_ENV, "2")
    local_em_engine._maybe_write_vdam_candidate_block_map(
        particle_ids=np.array([10], dtype=np.int32),
        reconstruction_take_indices=np.array([[0]], dtype=np.int32),
        reconstruction_pack_mask=np.array([[True]]),
        reconstruction_contributing_mask=np.array([[True]]),
        local_rotation_ids=np.array([[100]], dtype=np.int32),
        reconstruction_group_ids=None,
        debug_iteration=1,
    )
    assert not output.exists()


def test_candidate_block_map_writer_keeps_only_launched_native_grid(tmp_path, monkeypatch):
    output = tmp_path / "candidate-map.bin"
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_TRACE_ENV, str(tmp_path / "trace.bin"))
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_MAP_ENV, str(output))
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_MAP_ITER_ENV, "1")
    monkeypatch.setenv(local_em_engine.VDAM_CANDIDATE_BLOCK_MAP_CAPACITY_ENV, "16")

    local_em_engine._maybe_write_vdam_candidate_block_map(
        particle_ids=np.array([10, 20], dtype=np.int32),
        reconstruction_take_indices=np.array([[2, 0, 0], [1, 3, 0]], dtype=np.int32),
        reconstruction_pack_mask=np.array([[True, False, False], [True, True, False]]),
        reconstruction_contributing_mask=np.array(
            [[True, False, False], [True, False, False]]
        ),
        local_rotation_ids=np.array([[100, 101, 102, 103], [200, 201, 202, 203]], dtype=np.int32),
        reconstruction_group_ids=np.array([0, 1], dtype=np.int32),
        candidate_launch_counts=np.array([1, 2], dtype=np.int32),
        debug_iteration=1,
    )

    header, records = load_map(output)
    assert header["record_count"] == 3
    assert records["particle_id"].tolist() == [10, 20, 20]
    assert records["candidate_orientation_row"].tolist() == [0, 0, 1]
    assert records["native_orientation_row"].tolist() == [2, 1, 3]
