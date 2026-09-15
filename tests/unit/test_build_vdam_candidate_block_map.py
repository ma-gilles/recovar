from __future__ import annotations

import numpy as np
import pytest

from scripts.build_vdam_block_chronology import (
    FLAG_NO_ATOMIC,
    FLAG_SGD,
)
from scripts.build_vdam_block_chronology import (
    RECORD_DTYPE as BLOCK_DTYPE,
)
from scripts.build_vdam_candidate_block_map import (
    INVALID_ROW,
    MAP_CONTRIBUTING,
    MAP_VALID,
    RECORD_DTYPE,
    validate_map,
)


def _fixture():
    map_records = np.zeros(8, dtype=RECORD_DTYPE)
    candidate = np.zeros(8, dtype=BLOCK_DTYPE)
    native = np.zeros(6, dtype=BLOCK_DTYPE)
    for particle_offset, (stack_id, internal_id) in enumerate(((10, 101), (20, 202))):
        candidate_slice = slice(4 * particle_offset, 4 * (particle_offset + 1))
        candidate[candidate_slice]["particle_id"] = stack_id
        candidate[candidate_slice]["orientation_row"] = np.arange(4, dtype=np.uint32)
        candidate[candidate_slice]["flags"] = np.array(
            [FLAG_SGD, FLAG_SGD, FLAG_SGD | FLAG_NO_ATOMIC, FLAG_SGD | FLAG_NO_ATOMIC],
            dtype=np.uint32,
        )
        native_slice = slice(3 * particle_offset, 3 * (particle_offset + 1))
        native[native_slice]["particle_id"] = internal_id
        native[native_slice]["orientation_row"] = np.arange(3, dtype=np.uint32)
        native[native_slice]["flags"] = np.array(
            [FLAG_SGD, FLAG_SGD | FLAG_NO_ATOMIC, FLAG_SGD],
            dtype=np.uint32,
        )
        rows = map_records[candidate_slice]
        rows["particle_id"] = stack_id
        rows["candidate_orientation_row"] = np.arange(4, dtype=np.uint32)
        rows["native_orientation_row"] = np.array([0, 2, INVALID_ROW, INVALID_ROW], dtype=np.uint32)
        rows["global_rotation_id"] = np.array(
            [1000 + particle_offset * 10, 1002 + particle_offset * 10, -1, -1],
            dtype=np.int32,
        )
        rows["iteration"] = 1
        rows["flags"] = np.array([MAP_VALID, MAP_VALID, 0, 0], dtype=np.uint32)
    header = {"schema_version": 1, "iteration": 1, "record_count": 8}
    return header, map_records, candidate, native


def test_candidate_block_map_bijects_all_atomic_rows():
    header, map_records, candidate, native = _fixture()
    result = validate_map(
        header,
        map_records,
        candidate_records=candidate,
        native_records=native,
        native_internal_to_stack={101: 10, 202: 20},
        iteration=1,
        n_particles=2,
    )
    assert result["mapped_atomic_count"] == 4
    assert result["mapped_logical_count"] == 4
    assert result["mapped_atomic_free_count"] == 0
    assert result["candidate_padding_count"] == 4
    assert result["native_atomic_free_count"] == 2
    assert result["candidate_atomic_free_count"] == 4
    assert result["candidate_to_native"].tolist() == [0, 2, -1, -1, 3, 5, -1, -1]
    assert result["native_to_candidate"].tolist() == [0, -1, 1, 4, -1, 5]


def test_candidate_block_map_rejects_atomic_flag_disagreement():
    header, map_records, candidate, native = _fixture()
    candidate[0]["flags"] |= FLAG_NO_ATOMIC
    with pytest.raises(ValueError, match="atomic flags disagree"):
        validate_map(
            header,
            map_records,
            candidate_records=candidate,
            native_records=native,
            native_internal_to_stack={101: 10, 202: 20},
            iteration=1,
            n_particles=2,
        )


def test_v2_candidate_block_map_retains_native_atomic_free_grid_rows():
    header, map_records, candidate, native = _fixture()
    header["schema_version"] = 2
    for particle_offset in range(2):
        rows = map_records[4 * particle_offset : 4 * (particle_offset + 1)]
        rows["native_orientation_row"] = np.array(
            [0, 2, 1, INVALID_ROW],
            dtype=np.uint32,
        )
        rows["global_rotation_id"] = np.array(
            [
                1000 + particle_offset * 10,
                1002 + particle_offset * 10,
                1001 + particle_offset * 10,
                -1,
            ],
            dtype=np.int32,
        )
        rows["flags"] = np.array(
            [
                MAP_VALID | MAP_CONTRIBUTING,
                MAP_VALID | MAP_CONTRIBUTING,
                MAP_VALID,
                0,
            ],
            dtype=np.uint32,
        )
    result = validate_map(
        header,
        map_records,
        candidate_records=candidate,
        native_records=native,
        native_internal_to_stack={101: 10, 202: 20},
        iteration=1,
        n_particles=2,
    )
    assert result["mapped_logical_count"] == 6
    assert result["mapped_atomic_count"] == 4
    assert result["mapped_atomic_free_count"] == 2
    assert result["candidate_padding_count"] == 2
    assert result["candidate_to_native"].tolist() == [0, 2, 1, -1, 3, 5, 4, -1]
    assert result["native_to_candidate"].tolist() == [0, 2, 1, 4, 6, 5]
