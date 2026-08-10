from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from scripts import validate_relion_bpref_rotation_mass as validator
from scripts.analyze_k1_bpref_membership_all import _compare_compact_particle


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def test_load_rotation_mass_artifact(tmp_path: Path) -> None:
    rows = np.zeros(2, dtype=validator.ROW_DTYPE)
    rows["orientation_class_key"] = [10, 11]
    rows["oversampled_rotation"] = [2, 3]
    rows["matrix"] = np.stack(
        [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
    ).reshape(2, 9)
    rows["orientation_local"] = [4, 9]
    rows["candidate_translation_count"] = [3, 2]
    rows["significant_translation_count"] = [2, 1]
    rows["posterior_rotation_mass"] = [0.6, 0.4]
    rows["reconstruction_rotation_mass"] = [0.55, 0.35]
    header = [0] * 40
    header[0] = 2
    header[1] = validator.HEADER_STRUCT.size
    header[2] = validator.ROW_DTYPE.itemsize
    header[3] = validator.FOOTER_STRUCT.size
    header[4] = 2
    header[5] = 1
    header[6] = 7
    header[7] = 42
    header[8] = 0
    header[9] = 1
    header[10] = 0
    header[11] = 3
    header[12] = 2
    header[13] = _float_bits(0.1)
    header[14] = _float_bits(1.0)
    header[15] = 1
    header[16] = 1
    header[17] = 1
    header[18] = 10_000
    header[19] = (
        validator.HEADER_STRUCT.size
        + rows.nbytes
        + validator.FOOTER_STRUCT.size
    )
    header[20] = 123
    header[21] = 456
    header[22] = 60
    header[23] = 3
    header[24] = 30
    header[25] = 10
    path = tmp_path / "part7_stack42_img0_class1.bpm-v2.bin"
    path.write_bytes(
        validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header)
        + rows.tobytes()
        + validator.FOOTER_STRUCT.pack(validator.FOOTER_MAGIC, 3, 2)
    )

    artifact = validator.load_artifact(path)
    assert artifact.part_id == 7
    assert artifact.stack_index == 42
    assert artifact.mpi_rank == 1
    assert artifact.rows.size == 2
    assert np.array_equal(
        artifact.rows["significant_translation_count"],
        [2, 1],
    )

    header[9] = int(validator.UNKNOWN_MPI_RANK)
    path.write_bytes(
        validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header)
        + rows.tobytes()
        + validator.FOOTER_STRUCT.pack(validator.FOOTER_MAGIC, 3, 2)
    )
    _, summary = validator.validate_directory(tmp_path, expected_particles=1)
    assert summary["mpi_rank_tracking"] == "unavailable_srun_environment"
    assert summary["mpi_rank_counts"] == {"unknown": 1}


def test_compact_rotation_mass_comparison_is_union_sensitive() -> None:
    rows = np.zeros(2, dtype=validator.ROW_DTYPE)
    rotations = np.stack(
        [np.eye(3, dtype=np.float32), np.diag([1.0, -1.0, -1.0]).astype(np.float32)]
    )
    rows["matrix"] = rotations.reshape(2, 9)
    rows["candidate_translation_count"] = [2, 1]
    rows["significant_translation_count"] = [1, 1]
    rows["posterior_rotation_mass"] = [0.8, 0.2]
    rows["reconstruction_rotation_mass"] = [0.75, 0.15]
    recovar = {
        "rotations": rotations[:1],
        "candidate_count": np.asarray([2]),
        "posterior_mass": np.asarray([0.8]),
        "reconstruction_mass": np.asarray([0.75]),
        "significant_count": np.asarray([1]),
    }
    report = _compare_compact_particle(
        relion_rows=rows,
        recovar=recovar,
        rotation_tolerance=1.0e-6,
    )
    assert not report["candidate_sets_exact"]
    assert report["relion_unmatched_candidate_count"] == 1
    assert report["candidate_union_reconstruction_mass_relative_l2"] > 0.0
