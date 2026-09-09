from __future__ import annotations

import struct

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import vdam_replay
from scripts.build_vdam_candidate_block_map import INVALID_ROW, load_map


def test_candidate_block_map_writer_appends_exact_physical_rows(tmp_path, monkeypatch):
    output = tmp_path / "candidate-map.bin"
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_TRACE_ENV, str(tmp_path / "trace.bin"))
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ENV, str(output))
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ITER_ENV, "1")
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_CAPACITY_ENV, "16")

    vdam_replay._maybe_write_vdam_candidate_block_map(
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
    vdam_replay._maybe_write_vdam_candidate_block_map(
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
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_TRACE_ENV, str(tmp_path / "trace.bin"))
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ENV, str(output))
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ITER_ENV, "2")
    vdam_replay._maybe_write_vdam_candidate_block_map(
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
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_TRACE_ENV, str(tmp_path / "trace.bin"))
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ENV, str(output))
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ITER_ENV, "1")
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_CAPACITY_ENV, "16")

    vdam_replay._maybe_write_vdam_candidate_block_map(
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



def test_candidate_map_writer_preserves_wire_bytes(tmp_path, monkeypatch):
    """Independent struct layout protects the shared producer/reader schema."""
    output = tmp_path / "map.bin"
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_TRACE_ENV, "trace")
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ENV, str(output))
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ITER_ENV, "7")
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_CAPACITY_ENV, "5")
    vdam_replay._maybe_write_vdam_candidate_block_map(
        particle_ids=[27], reconstruction_take_indices=[[1, -1]],
        reconstruction_pack_mask=[[True, False]],
        reconstruction_contributing_mask=[[True, False]],
        local_rotation_ids=[[100, 101]], reconstruction_group_ids=[3],
        debug_iteration=7,
    )
    header = struct.pack("<16sIIIIQQQQ", b"RECOVAR_VDAMBM1\0", 2, 64, 40, 7, 2, 5, 0, 0)
    valid = struct.pack("<qIIiiiIII", 27, 0, 1, 101, 0, 3, 7, 3, 0)
    padding = struct.pack("<qIIiiiIII", 27, 1, 0xFFFFFFFF, -1, 0, 3, 7, 0, 0)
    assert output.read_bytes() == header + valid + padding


@pytest.mark.parametrize("version", [1, 2])
def test_candidate_map_reader_accepts_existing_versions(tmp_path, version):
    output = tmp_path / "old-map.bin"
    output.write_bytes(
        struct.pack("<16sIIIIQQQQ", b"RECOVAR_VDAMBM1\0", version, 64, 40, 7, 1, 5, 0, 0)
        + struct.pack("<qIIiiiIII", 27, 0, 1, 101, 0, 3, 7, 3, 0)
    )
    header, records = load_map(output)
    assert header["schema_version"] == version
    assert records.tolist() == [(27, 0, 1, 101, 0, 3, 7, 3, 0)]


def test_candidate_map_capacity_error_preserves_created_header(tmp_path, monkeypatch):
    """Keep the existing failure-side file state; this move is not an IO repair."""
    output = tmp_path / "map.bin"
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_TRACE_ENV, "trace")
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ENV, str(output))
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_ITER_ENV, "7")
    monkeypatch.setenv(vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_CAPACITY_ENV, "1")
    with pytest.raises(ValueError, match="exceeds its declared capacity"):
        vdam_replay._maybe_write_vdam_candidate_block_map(
            particle_ids=[27], reconstruction_take_indices=[[0, 1]],
            reconstruction_pack_mask=[[True, True]],
            reconstruction_contributing_mask=[[True, False]],
            local_rotation_ids=[[100, 101]], reconstruction_group_ids=None,
            debug_iteration=7,
        )
    assert output.read_bytes() == struct.pack(
        "<16sIIIIQQQQ", b"RECOVAR_VDAMBM1\0", 2, 64, 40, 7, 0, 1, 0, 0,
    )
