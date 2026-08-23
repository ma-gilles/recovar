from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import validate_relion_bpref_membership as validator

PATCH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "patches"
    / "relion_bpref_membership_chunked_bc319d0.patch"
)
PATCH_SHA256 = "30c2d2f7d7bdd34312ed792b86cdc1aaf3976b4ffe8cd64828def0add1f79a76"


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_artifact(
    directory: Path,
    *,
    part: int,
    stack: int,
    mpi_rank: int = 1,
    weights: np.ndarray | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rotations = np.zeros(2, dtype=validator.ROTATION_DTYPE)
    rotations["orientation_class_key"] = [10, 11]
    rotations["oversampled_rotation"] = [2, 3]
    rotations["matrix"] = np.arange(18, dtype=np.float32).reshape(2, 9)
    rotations["orientation_local"] = np.arange(2, dtype=np.uint32)
    if weights is None:
        weights = np.asarray([[0.7, 0.2, 0.1], [0.4, 0.3, 0.0]], dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    header = [0] * 40
    header[0] = 1
    header[1] = validator.HEADER_STRUCT.size
    header[2] = validator.ROTATION_DTYPE.itemsize
    header[3] = validator.WEIGHT_DTYPE.itemsize
    header[4] = validator.FOOTER_STRUCT.size
    header[5] = 2
    header[6] = 1
    header[7] = part
    header[8] = stack
    header[9] = part
    header[10] = mpi_rank
    header[12] = rotations.size
    header[13] = weights.shape[1]
    header[14] = weights.size
    header[15] = _bits(0.25)
    header[16] = _bits(1.0)
    header[17] = 1
    header[18] = 1
    header[19] = 1
    artifact_bytes = (
        validator.HEADER_STRUCT.size
        + rotations.nbytes
        + weights.nbytes
        + validator.FOOTER_STRUCT.size
    )
    header[20] = artifact_bytes
    header[21] = artifact_bytes
    header[22] = 123
    header[23] = 456
    header[24] = 1
    header[25] = 60
    payload = bytearray(validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header))
    payload.extend(rotations.tobytes())
    payload.extend(weights.tobytes())
    payload.extend(
        validator.FOOTER_STRUCT.pack(
            validator.FOOTER_MAGIC, weights.size, rotations.size
        )
    )
    path = directory / f"part{part}_stack{stack}_img{part}_class1.bpm-v1.bin"
    path.write_bytes(payload)
    return path


def test_load_membership_artifact(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, part=4, stack=101)
    artifact = validator.load_artifact(path)
    assert artifact.weights.shape == (2, 3)
    assert artifact.significant_weight == np.float32(0.25)
    assert artifact.weight_norm == 1.0
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_directory_reports_fixed_counts(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, part=4, stack=101)
    validator.load_artifact(path)
    _, report = validator.validate_directory(
        tmp_path,
        expected_particles=1,
        expected_stack_indices=np.asarray([101]),
        expected_stack_mpi_rank=1,
    )
    assert report["particle_count"] == 1
    assert report["total_rotation_count"] == 2
    assert report["total_weight_count"] == 6
    assert report["total_significant_sample_count"] == 3


def test_rejects_truncated_artifact(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, part=4, stack=101)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_artifact(path)


def test_accepts_exact_invalid_weight_sentinel(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path,
        part=4,
        stack=101,
        weights=np.asarray(
            [[0.7, validator.INVALID_WEIGHT_SENTINEL, 0.1], [0.4, 0.3, 0.0]],
            dtype=np.float32,
        ),
    )
    artifact = validator.load_artifact(path)
    assert np.count_nonzero(
        artifact.weights == validator.INVALID_WEIGHT_SENTINEL
    ) == 1


def test_rejects_unexpected_negative_weight(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path,
        part=4,
        stack=101,
        weights=np.asarray([[0.7, -0.2, 0.1], [0.4, 0.3, 0.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="unexpected negative posterior"):
        validator.load_artifact(path)


def test_rejects_incomplete_temporary_file(tmp_path: Path) -> None:
    _write_artifact(tmp_path, part=4, stack=101)
    (tmp_path / "orphan.tmp.1").write_bytes(b"x")
    with pytest.raises(ValueError, match="incomplete"):
        validator.validate_directory(tmp_path)


def test_membership_patch_bytes_and_passive_guards_are_frozen() -> None:
    payload = PATCH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PATCH_SHA256
    text = payload.decode()
    assert 'std::getenv("RELION_BPM_CAPTURE_DIR")' in text
    assert "cudaMemcpyDeviceToHost" in text
    assert "relion_capture_bpref_membership_v1(" in text
    assert not any(
        line.startswith("+") and "runBackProjectKernel(" in line
        for line in text.splitlines()
    )
