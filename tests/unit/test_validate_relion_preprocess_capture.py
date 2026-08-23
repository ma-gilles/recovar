from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import validate_relion_preprocess_capture as validator

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "docs" / "patches" / "0006-RELION-preprocessing-boundary-capture.patch"
PATCH_SHA256 = "a655a40e561167d1b39f1157d3ac3754751ac87e06448b3b5133bbca799517b4"


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_artifact(
    directory: Path,
    *,
    part_id: int = 4,
    stack_index: int = 101,
    mpi_rank: int = 1,
    iteration: int = 2,
    expected_particles: int = 1,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    real_shape = (1, 4, 4)
    fourier_shape = (1, 4, 3)
    real_count = int(np.prod(real_shape))
    fourier_count = int(np.prod(fourier_shape))
    raw = np.arange(real_count, dtype=np.float32)
    normalized = raw + np.float32(20)
    masked = raw + np.float32(40)
    fourier_stages = [
        (
            np.arange(fourier_count, dtype=np.float32) + np.float32(100 * stage),
            np.arange(fourier_count, dtype=np.float32) + np.float32(50 + 100 * stage),
        )
        for stage in range(4)
    ]

    header = [0] * 40
    header[0] = 1
    header[1] = validator.HEADER_STRUCT.size
    header[2] = validator.FLOAT_DTYPE.itemsize
    header[3] = 8
    header[4] = validator.FOOTER_STRUCT.size
    header[5] = iteration
    header[6] = part_id
    header[7] = stack_index
    header[8] = mpi_rank
    header[9] = 0
    header[10] = 0
    header[11] = 0
    header[12:16] = [real_shape[2], real_shape[1], real_shape[0], real_count]
    header[16:20] = [
        fourier_shape[2],
        fourier_shape[1],
        fourier_shape[0],
        fourier_count,
    ]
    header[20] = _bits(0.75)
    header[21] = _bits(-2.0)
    header[22] = _bits(1.0)
    header[23] = _bits(0.0)
    header[24] = _bits(2.0)
    header[25] = _bits(3.0)
    header[26] = _bits(1.0)
    header[27] = _bits(0.25)
    header[28] = 1
    header[29] = expected_particles
    header[30] = expected_particles
    header[31] = 1
    artifact_bytes = (
        validator.HEADER_STRUCT.size
        + 3 * real_count * validator.FLOAT_DTYPE.itemsize
        + 8 * fourier_count * validator.FLOAT_DTYPE.itemsize
        + validator.FOOTER_STRUCT.size
    )
    header[32] = artifact_bytes * expected_particles
    header[33] = artifact_bytes * expected_particles
    header[34] = 123
    header[35] = 1000 + part_id
    header[36:39] = [127, 1, 1]
    header[39] = 128

    payload = bytearray(validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header))
    payload.extend(raw.tobytes())
    payload.extend(normalized.tobytes())
    for real_values, imag_values in fourier_stages[:2]:
        payload.extend(real_values.tobytes())
        payload.extend(imag_values.tobytes())
    payload.extend(masked.tobytes())
    for real_values, imag_values in fourier_stages[2:]:
        payload.extend(real_values.tobytes())
        payload.extend(imag_values.tobytes())
    payload.extend(
        validator.FOOTER_STRUCT.pack(
            validator.FOOTER_MAGIC,
            real_count,
            fourier_count,
        )
    )
    path = directory / f"part{part_id}_stack{stack_index}.preprocess-v1.bin"
    path.write_bytes(payload)
    return path


def test_load_preprocess_artifact(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path)
    artifact = validator.load_artifact(path)
    assert artifact.part_id == 4
    assert artifact.stack_index == 101
    assert artifact.iteration == 2
    assert artifact.raw_input_real.shape == (1, 4, 4)
    assert artifact.masked_fourier_post_optics.shape == (1, 4, 3)
    assert artifact.norm_correction == np.float32(0.75)
    assert np.array_equal(
        artifact.old_offset,
        np.asarray([-2.0, 1.0, 0.0], dtype=np.float32),
    )
    assert artifact.mask_parameters == {
        "radius": 2.0,
        "radius_p": 3.0,
        "cosine_width": 1.0,
        "background": 0.25,
    }
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_directory_reports_fixed_structural_metric(tmp_path: Path) -> None:
    _write_artifact(tmp_path)
    artifacts, report = validator.validate_directory(
        tmp_path,
        expected_particles=1,
        expected_part_ids=np.asarray([4]),
        expected_stack_indices=np.asarray([101]),
        expected_mpi_rank=1,
        expected_iteration=2,
    )
    assert len(artifacts) == 1
    assert report["status"] == "pass"
    assert report["fixed_metric"] == {
        "evaluated_particles": 1,
        "expected_particles": 1,
        "structurally_qualified": 1,
        "complete_seven_stage_payload": 1,
    }


def test_rejects_truncated_artifact(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_artifact(path)


def test_rejects_unrounded_old_offset(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path)
    payload = bytearray(path.read_bytes())
    header = list(validator.HEADER_STRUCT.unpack_from(payload))
    header[1 + 21] = _bits(0.5)
    payload[: validator.HEADER_STRUCT.size] = validator.HEADER_STRUCT.pack(*header)
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="not rounded"):
        validator.load_artifact(path)


def test_rejects_missing_passive_stage_flag(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path)
    payload = bytearray(path.read_bytes())
    header = list(validator.HEADER_STRUCT.unpack_from(payload))
    header[1 + 37] = 0
    payload[: validator.HEADER_STRUCT.size] = validator.HEADER_STRUCT.pack(*header)
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="passive stage flags"):
        validator.load_artifact(path)


def test_rejects_incomplete_temporary_file(tmp_path: Path) -> None:
    _write_artifact(tmp_path)
    (tmp_path / "orphan.tmp.1").write_bytes(b"x")
    with pytest.raises(ValueError, match="incomplete"):
        validator.validate_directory(tmp_path)


def test_preprocess_patch_is_frozen_bounded_and_passive() -> None:
    payload = PATCH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PATCH_SHA256
    text = payload.decode()
    assert 'std::getenv("RELION_P1_PREPROCESS_CAPTURE_DIR")' in text
    assert "PART_IDS must be an explicit non-empty list" in text
    assert "runtime dimensions exceed RELION_P1_PREPROCESS_CAPTURE_MAX_BYTES" in text
    assert "refuses to overwrite an existing artifact" in text
    assert "normalized_shifted_real" in text
    assert "masked_fourier_post_optics_real" in text
    assert "g_diff2s" not in text
    assert "Mweight" not in text
    assert "BPref" not in text
    assert not any(
        line.startswith("-") and not line.startswith("---")
        for line in text.splitlines()
    )
