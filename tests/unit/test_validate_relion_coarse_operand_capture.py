from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import validate_relion_coarse_operand_capture as validator
from scripts import validate_relion_coarse_pass1_components as component_validator

PATCH = Path(__file__).resolve().parents[2] / "docs" / "patches" / "0003-RELION-coarse-live-operand-capture.patch"
PATCH_SHA256 = "a00ad73ac496be4b2cc0513ee7aa2fd0dd8de137927db66b4d80420d0b06ad1e"
SHIFTED_PATCH = (
    Path(__file__).resolve().parents[2] / "docs" / "patches" / "0004-RELION-coarse-shifted-image-capture.patch"
)
SHIFTED_PATCH_SHA256 = "3d090744381306bdccc3be641834909286355f2bc15abc707053ad48d95f3b21"


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_operand(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rotation_keys = np.asarray([1, 3], dtype=np.uint64)
    local_indices = np.asarray([0, 2], dtype=np.uint64)
    matrices = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    reference_real = np.asarray(
        [[1.0, 0.5, -0.25, 0.75], [0.25, -0.5, 1.0, 0.125]],
        dtype=np.float32,
    )
    reference_imag = np.asarray(
        [[0.0, 0.25, 0.5, -0.25], [0.75, 0.0, -0.125, 0.5]],
        dtype=np.float32,
    )
    image_real = np.asarray([0.5, 1.0, -0.5, 0.25], dtype=np.float32)
    image_imag = np.asarray([0.25, -0.5, 0.75, 1.0], dtype=np.float32)
    correction = np.asarray([2.0, 1.0, 0.5, 3.0], dtype=np.float32)
    translations = np.asarray(
        [[0.0, 0.2, -0.3], [0.0, -0.1, 0.25], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    pixels = np.arange(4)
    x = (pixels % 2).astype(np.float32)
    y = (pixels // 2).astype(np.int64)
    y = np.where(y > 1, y - 2, y).astype(np.float32)
    phase = (translations[0, :, None] * x[None] + translations[1, :, None] * y[None]).astype(np.float32)
    sine = np.sin(phase).astype(np.float32)
    cosine = np.cos(phase).astype(np.float32)
    shifted_real = (cosine * image_real[None] - sine * image_imag[None]).astype(np.float32)
    shifted_imag = (cosine * image_imag[None] + sine * image_real[None]).astype(np.float32)
    header = [0] * 40
    header[0:5] = [
        2,
        validator.HEADER_STRUCT.size,
        validator.FLOAT_DTYPE.itemsize,
        validator.UINT64_DTYPE.itemsize,
        validator.FOOTER_STRUCT.size,
    ]
    header[5:17] = [2, 4, 101, 1, 0, 1, 0, 2, 4, 3, 2, 4]
    header[17:24] = [2, 2, 1, 1, 8, 8, 8]
    header[24] = _bits(1.0)
    header[25:28] = [1, 1, 1]
    artifact_bytes = (
        validator.HEADER_STRUCT.size
        + rotation_keys.nbytes
        + local_indices.nbytes
        + matrices.nbytes
        + reference_real.nbytes
        + reference_imag.nbytes
        + image_real.nbytes
        + image_imag.nbytes
        + correction.nbytes
        + translations.nbytes
        + shifted_real.nbytes
        + shifted_imag.nbytes
        + validator.FOOTER_STRUCT.size
    )
    header[28:32] = [artifact_bytes, artifact_bytes, 123, 456]
    header[32:40] = [1, 1, 1, 1, 1, 8, 2, 2]
    payload = bytearray(validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header))
    for values in (
        rotation_keys,
        local_indices,
        matrices,
        reference_real,
        reference_imag,
        image_real,
        image_imag,
        correction,
        translations,
        shifted_real,
        shifted_imag,
    ):
        payload.extend(values.tobytes())
    payload.extend(validator.FOOTER_STRUCT.pack(validator.FOOTER_MAGIC, 2, 4))
    path = directory / "part4_stack101.p1-op-v2.bin"
    path.write_bytes(payload)
    return path


def _write_components(
    directory: Path,
    operand_path: Path,
    *,
    target_perturbation: float = 0.0,
) -> Path:
    operand = validator.load_artifact(operand_path)
    replay_reference, replay_cross = validator.replay_components(operand)
    replay_diff2 = validator.replay_production_diff2(operand)
    reference = np.repeat(np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), 3, axis=1)
    cross = np.zeros((4, 3), dtype=np.float32)
    reference[operand.rotation_keys] = replay_reference[:, None].astype(np.float32)
    cross[operand.rotation_keys] = replay_cross.astype(np.float32)
    reference[operand.rotation_keys[0]] += np.float32(target_perturbation)
    raw = reference + cross + np.float32(9.0)
    raw[operand.rotation_keys] = replay_diff2
    weights = np.asarray(
        [[0.8, 0.1, 0], [0.3, 0.2, 0], [0, 0, 0], [0.4, 0, 0]],
        dtype=np.float32,
    )
    mask = (weights >= np.float32(0.3)).astype(np.uint8)
    physical_translations = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    header = [0] * 40
    header[0:5] = [
        2,
        component_validator.HEADER_STRUCT.size,
        component_validator.FLOAT_DTYPE.itemsize,
        component_validator.MASK_DTYPE.itemsize,
        component_validator.FOOTER_STRUCT.size,
    ]
    header[5:16] = [2, 4, 101, 1, 0, 2, 2, 3, 12, 3, 0]
    header[16:20] = [
        _bits(0.3),
        _bits(float(np.sum(weights))),
        _bits(float(np.max(weights))),
        _bits(float(np.min(raw))),
    ]
    header[20:23] = [1, 1, 1]
    artifact_bytes = (
        component_validator.HEADER_STRUCT.size
        + raw.nbytes
        + weights.nbytes
        + reference.nbytes
        + cross.nbytes
        + mask.nbytes
        + physical_translations.nbytes
        + component_validator.FOOTER_STRUCT.size
    )
    header[23:25] = [artifact_bytes, artifact_bytes]
    header[25:28] = [123, 456, 2]
    header[28] = _bits(0.999)
    header[29] = np.iinfo(np.uint64).max
    header[30] = physical_translations.size
    header[31:37] = [1, 1, 1, 1, 1, 2]
    payload = bytearray(component_validator.HEADER_STRUCT.pack(component_validator.HEADER_MAGIC, *header))
    for values in (
        raw,
        weights,
        reference,
        cross,
        mask,
        physical_translations,
    ):
        payload.extend(values.tobytes())
    payload.extend(component_validator.FOOTER_STRUCT.pack(component_validator.FOOTER_MAGIC, 12, 3))
    path = directory / "part4_stack101.p1-v2.bin"
    path.write_bytes(payload)
    return path


def test_validates_live_operand_replay_against_fp64_components(tmp_path: Path) -> None:
    operand_path = _write_operand(tmp_path)
    _write_components(tmp_path, operand_path)
    artifacts, report = validator.validate_directory(
        tmp_path,
        expected_particles=1,
        expected_stack_indices=np.asarray([101]),
        expected_mpi_rank=1,
    )
    assert artifacts[0].reference_real.shape == (2, 4)
    assert report["status"] == "pass"
    assert report["paired_component_status"] == "rejected"
    assert report["fixed_metric"] == {
        "evaluated_particles": 1,
        "expected_particles": 1,
        "reference_replay_passed": 1,
        "cross_replay_p95_passed": 1,
        "cross_replay_max_passed": 1,
        "production_diff2_replay_p95_passed": 1,
        "production_diff2_replay_max_passed": 1,
    }


def test_rejects_operand_that_does_not_replay_component_target(
    tmp_path: Path,
) -> None:
    operand_path = _write_operand(tmp_path)
    _write_components(tmp_path, operand_path, target_perturbation=0.1)
    _, report = validator.validate_directory(tmp_path)
    assert report["status"] == "rejected"
    assert report["fixed_metric"]["reference_replay_passed"] == 0


def test_rejects_truncated_operand_artifact(tmp_path: Path) -> None:
    path = _write_operand(tmp_path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_artifact(path)


def test_rejects_duplicate_rotation_key(tmp_path: Path) -> None:
    path = _write_operand(tmp_path)
    payload = bytearray(path.read_bytes())
    offset = validator.HEADER_STRUCT.size
    payload[offset : offset + 16] = np.asarray([1, 1], dtype=np.uint64).tobytes()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="duplicate rotation key"):
        validator.load_artifact(path)


def test_operand_patch_is_frozen_and_only_adds_passive_capture() -> None:
    payload = PATCH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PATCH_SHA256
    text = payload.decode()
    assert "RELION_P1_CAPTURE_OPERAND_ROTATION_KEYS" in text
    assert "cuda_kernel_capture_diff2_coarse_reference" in text
    assert "relion_capture_coarse_operands_v1" in text
    assert "y -= projector.imgY;" in text
    removed_lines = [line for line in text.splitlines() if line.startswith("-") and not line.startswith("---")]
    assert removed_lines == []


def test_shifted_image_delta_is_frozen_and_keeps_score_buffers_passive() -> None:
    payload = SHIFTED_PATCH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SHIFTED_PATCH_SHA256
    text = payload.decode()
    assert "cuda_kernel_capture_diff2_coarse_shifted_images" in text
    assert "exact GPU translatePixel outputs" in text
    assert "production CUDA REF3D coarse prefetch factor" in text
    assert "g_diff2s" not in text
    assert "g_shifted_real[index]" in text
    assert "g_shifted_imag[index]" in text
