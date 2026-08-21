from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import validate_relion_coarse_pass1_capture as validator

PATCH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "patches"
    / "relion_coarse_pass1_capture_after_membership_bc319d0.patch"
)
PATCH_SHA256 = "dec649ced8ff8d0facd14203199b37fe01835820d0e8485783d0d7b0d15885e9"


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_artifact(
    directory: Path,
    *,
    part: int,
    stack: int,
    mpi_rank: int = 1,
    weights: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    significant_weight: float = 0.3,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    raw_diff2 = np.arange(12, dtype=np.float32).reshape(4, 3)
    if weights is None:
        weights = np.asarray(
            [[0.8, 0.1, 0.0], [0.3, 0.2, 0.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=np.float32,
        )
    weights = np.asarray(weights, dtype=np.float32)
    if mask is None:
        mask = weights >= np.float32(0.3)
    mask = np.asarray(mask, dtype=np.uint8)
    translations = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    header = [0] * 40
    header[0] = 1
    header[1] = validator.HEADER_STRUCT.size
    header[2] = validator.FLOAT_DTYPE.itemsize
    header[3] = validator.MASK_DTYPE.itemsize
    header[4] = validator.FOOTER_STRUCT.size
    header[5] = 2
    header[6] = part
    header[7] = stack
    header[8] = mpi_rank
    header[10] = 2
    header[11] = 2
    header[12] = 3
    header[13] = weights.size
    header[14] = int(np.count_nonzero(mask))
    header[15] = int(np.argmax(weights))
    header[16] = _bits(significant_weight)
    header[17] = _bits(float(np.sum(weights)))
    header[18] = _bits(float(np.max(weights)))
    header[19] = _bits(float(np.min(raw_diff2)))
    header[20] = 1
    header[21] = 1
    header[22] = 1
    artifact_bytes = (
        validator.HEADER_STRUCT.size
        + raw_diff2.nbytes
        + weights.nbytes
        + mask.nbytes
        + translations.nbytes
        + validator.FOOTER_STRUCT.size
    )
    header[23] = artifact_bytes
    header[24] = artifact_bytes
    header[25] = 123
    header[26] = 456
    header[27] = 60
    header[28] = _bits(0.999)
    header[29] = np.iinfo(np.uint64).max
    header[30] = translations.size
    header[31:34] = [1, 1, 1]
    payload = bytearray(validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header))
    payload.extend(raw_diff2.tobytes())
    payload.extend(weights.tobytes())
    payload.extend(mask.tobytes())
    payload.extend(translations.tobytes())
    payload.extend(
        validator.FOOTER_STRUCT.pack(
            validator.FOOTER_MAGIC, weights.size, int(np.count_nonzero(mask))
        )
    )
    path = directory / f"part{part}_stack{stack}.p1-v1.bin"
    path.write_bytes(payload)
    return path


def test_load_coarse_pass1_artifact(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, part=4, stack=101)
    artifact = validator.load_artifact(path)
    assert artifact.raw_diff2.shape == (4, 3)
    assert artifact.weights.shape == (4, 3)
    assert artifact.significant_mask.sum() == 3
    assert artifact.translations.shape == (3, 2)
    assert artifact.inferred_significant_weight == pytest.approx(0.3)
    assert artifact.significant_weight_semantics == "recorded_threshold_reproduces_mask"
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_accepts_source_proven_cuda_coarse_zero_sentinel(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path,
        part=4,
        stack=101,
        significant_weight=0.0,
    )
    artifact = validator.load_artifact(path)
    assert artifact.significant_weight == 0.0
    assert artifact.inferred_significant_weight == pytest.approx(0.3)
    assert (
        artifact.significant_weight_semantics
        == "relion_cuda_coarse_op_sentinel_zero__threshold_inferred_from_exact_production_mask"
    )


def test_validate_directory_reports_fixed_counts(tmp_path: Path) -> None:
    _write_artifact(tmp_path, part=4, stack=101)
    _, report = validator.validate_directory(
        tmp_path,
        expected_particles=1,
        expected_stack_indices=np.asarray([101]),
        expected_mpi_rank=1,
    )
    assert report["particle_count"] == 1
    assert report["total_candidate_count"] == 12
    assert report["total_significant_count"] == 3


def test_rejects_truncated_artifact(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, part=4, stack=101)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_artifact(path)


def test_rejects_nonbinary_mask(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path,
        part=4,
        stack=101,
        mask=np.asarray(
            [[2, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.uint8
        ),
    )
    with pytest.raises(ValueError, match="non-binary"):
        validator.load_artifact(path)


def test_rejects_threshold_mask_mismatch(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path,
        part=4,
        stack=101,
        mask=np.asarray(
            [[1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.uint8
        ),
    )
    with pytest.raises(ValueError, match="recorded threshold/mask"):
        validator.load_artifact(path)


def test_zero_sentinel_still_rejects_nonmonotone_production_mask(
    tmp_path: Path,
) -> None:
    path = _write_artifact(
        tmp_path,
        part=4,
        stack=101,
        mask=np.asarray(
            [[1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8
        ),
        significant_weight=0.0,
    )
    with pytest.raises(ValueError, match="weight rank/mask"):
        validator.load_artifact(path)


def test_rejects_incomplete_temporary_file(tmp_path: Path) -> None:
    _write_artifact(tmp_path, part=4, stack=101)
    (tmp_path / "orphan.tmp.1").write_bytes(b"x")
    with pytest.raises(ValueError, match="incomplete"):
        validator.validate_directory(tmp_path)


def test_coarse_pass1_patch_bytes_and_passive_guards_are_frozen() -> None:
    payload = PATCH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PATCH_SHA256
    text = payload.decode()
    assert 'std::getenv("RELION_P1_CAPTURE_DIR")' in text
    assert "raw_coarse_diff2.assign" in text
    assert "relion_capture_coarse_pass1_v1(" in text
    assert "refuses to overwrite an existing artifact" in text
    assert not any(
        line.startswith("+") and "runDiff2KernelCoarse(" in line
        for line in text.splitlines()
    )
