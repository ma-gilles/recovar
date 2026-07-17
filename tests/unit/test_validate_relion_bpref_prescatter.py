from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import compare_relion_recovar_bpref_prescatter as comparator
from scripts import validate_relion_bpref_prescatter as validator

pytestmark = pytest.mark.unit


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_artifact(
    directory: Path,
    *,
    part: int,
    stack: int,
    rank: int,
    expected_particles: int = 2,
    excluded: int = 1,
) -> Path:
    rotations = np.zeros(1, dtype=validator.ROTATION_DTYPE)
    rotations["orientation_class_key"] = 17
    rotations["oversampled_rotation"] = 0
    rotations["matrix"][0] = np.eye(3, dtype=np.float32).reshape(-1)
    rotations["orientation_local"] = 0

    rows = np.zeros(1, dtype=validator.ROW_DTYPE)
    rows["state"] = 1
    rows["orientation_local"] = 0
    rows["pixel"] = 47 * 25 + 1
    rows["flags"] = validator.ROW_FLAG_FWEIGHT_POSITIVE | validator.ROW_FLAG_RADIUS_SUPPORT
    rows["x"] = 1
    rows["y"] = -1
    rows["source_re"] = 2.0
    rows["source_im"] = -3.0
    rows["source_weight"] = 4.0

    values = [0] * 40
    values[0:5] = [1, 336, 40, 64, 32]
    values[5:20] = [1, 1, part, stack, part, rank, 0, 25, 48, 1, 1200, 1, 1, 2, 4]
    values[20:25] = [_float_bits(2.0), _float_bits(0.1), _float_bits(1.0), 0, 0]
    values[25:30] = [expected_particles, 1, 2, 10_000_000, 1_000_000]
    values[30:38] = [1234, 2000 + stack, 9, 9, 9, 0, 0, 1]
    values[38:40] = [rows.size + excluded, excluded]
    header = validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *values)
    footer = validator.FOOTER_STRUCT.pack(validator.FOOTER_MAGIC, rows.size, rotations.size)
    path = directory / f"part{part}_stack{stack}_img{part}_class1.bpre-v1.bin"
    path.write_bytes(header + rotations.tobytes() + rows.tobytes() + footer)
    return path


def test_validate_complete_directory_and_recovar_identities(tmp_path: Path):
    _write_artifact(tmp_path, part=10, stack=101, rank=1)
    _write_artifact(tmp_path, part=11, stack=202, rank=2)
    shard = tmp_path / "recovar.npz"
    np.savez(shard, stack_indices_1based=np.asarray([202], dtype=np.int64))

    expected = validator.load_recovar_stack_indices((tmp_path,))
    artifacts, summary = validator.validate_directory(
        tmp_path,
        expected_particles=2,
        expected_stack_indices=expected,
        expected_stack_mpi_rank=2,
    )

    assert len(artifacts) == 2
    assert summary["particle_count"] == 2
    assert summary["emitted_supported_row_count"] == 2
    assert summary["positive_fweight_candidate_count"] == 4
    assert summary["radius_excluded_positive_fweight_count"] == 2
    assert summary["classification_ready"] is True


def test_load_artifact_rejects_truncation(tmp_path: Path):
    path = _write_artifact(tmp_path, part=10, stack=101, rank=1)
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="artifact byte count mismatch"):
        validator.load_artifact(path)


def test_validate_directory_rejects_support_accounting_mismatch(tmp_path: Path):
    path = _write_artifact(tmp_path, part=10, stack=101, rank=1, expected_particles=1)
    payload = bytearray(path.read_bytes())
    struct.pack_into("<Q", payload, 16 + 38 * 8, 99)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="candidate/support accounting mismatch"):
        validator.validate_directory(tmp_path)


def test_validate_directory_rejects_unsealed_temporary_file(tmp_path: Path):
    _write_artifact(tmp_path, part=10, stack=101, rank=1, expected_particles=1)
    (tmp_path / "part10.tmp.1").write_bytes(b"partial")

    with pytest.raises(ValueError, match="incomplete temporary artifacts"):
        validator.validate_directory(tmp_path)


def test_compare_complete_aligned_prescatter_operands(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_artifact(capture, part=10, stack=101, rank=1)
    _write_artifact(capture, part=11, stack=202, rank=2)

    contributions = tmp_path / "contributions"
    contributions.mkdir()
    physical_pixel = 255 * 129 + 1
    summed = np.zeros((8, 2), dtype=np.complex64)
    ctf = np.zeros((8, 2), dtype=np.float32)
    summed[0, 0] = np.complex64(complex(-2.0, 3.0) * (2.0**-16))
    ctf[0, 0] = np.float32(4.0 * (2.0**-32))
    # Pixel 2 is in the RECOVAR source window but did not reach the indexed
    # device scatter. It is outside the aligned pre-scatter comparison support.
    summed[0, 1] = np.complex64(99 + 100j)
    ctf[0, 1] = np.float32(101)
    contribution_path = contributions / "rows.npz"
    np.savez(
        contribution_path,
        shadow_only_mode=np.asarray(False),
        stack_indices_1based=np.asarray([202], dtype=np.int64),
        active_particle_rows=np.zeros(8, dtype=np.int32),
        active_summed=summed,
        active_ctf_probs=ctf,
        active_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (8, 3, 3)),
        active_global_rotation_indices=np.arange(136, 144, dtype=np.int64),
        window_indices=np.asarray([physical_pixel, 2], dtype=np.int32),
        image_shape=np.asarray([256, 256], dtype=np.int32),
        current_size=np.asarray(48, dtype=np.int32),
    )
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    np.savez(
        geometry / "geometry.npz",
        companion_contribution_path=np.asarray(str(contribution_path)),
        signature_particle_rows=np.asarray([0], dtype=np.int32),
        signature_pixel_indices=np.asarray([[physical_pixel, 2]], dtype=np.int32),
        signature_row_flags=np.asarray([[64, 0]], dtype=np.uint32),
    )
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"classification_ready": True, "particle_count": 2}))
    inertness = tmp_path / "inertness.json"
    inertness.write_text(json.dumps({"capture_inertness_qualified": True}))

    report, arrays = comparator.compare(
        capture,
        contributions,
        geometry,
        validation_json=validation,
        inertness_json=inertness,
        mpi_rank=2,
    )

    assert report["gates"]["comparison_ready"] is True
    assert report["scope"]["relion_current_size"] == 48
    assert report["scope"]["physical_image_box_size"] == 256
    assert report["classification"] == "pre_scatter_operand_generation_difference"
    assert report["operands"]["data_numerator_recovar_vs_scaled_negative_relion"][
        "exact_equal"
    ] is True
    assert report["operands"]["real_weight_recovar_vs_scaled_relion"]["exact_equal"] is True
    assert np.array_equal(arrays["stack_indices_1based"], np.asarray([202]))
    assert np.array_equal(arrays["recovar_device_support_mask"], np.asarray([[True, False]]))
