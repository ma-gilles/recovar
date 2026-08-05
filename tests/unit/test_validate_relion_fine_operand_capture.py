import struct

import numpy as np
import pytest

from scripts.validate_relion_fine_operand_capture import (
    CANDIDATE_DTYPE,
    CANDIDATE_SIZE,
    FOOTER_SIZE,
    HEADER_SIZE,
    LANE_COUNT,
    PIXEL_DTYPE,
    PIXEL_SIZE,
    _cuda_fine_contribution,
    _cuda_fine_production_lanes,
    _reduce_lanes,
    _replay_lanes,
    load_fine_operand_capture,
    validate_capture,
)


def _write_capture(
    path,
    *,
    corrupt_lane=False,
    production_offset=0,
    production_sass=False,
):
    image_size = 513 if production_sass else 5
    candidates = np.zeros(1, dtype=CANDIDATE_DTYPE)
    pixels = np.zeros(image_size, dtype=PIXEL_DTYPE)
    pixels["target_index"] = 0
    pixels["pixel"] = np.arange(image_size, dtype=np.uint32)
    pixels["x"] = np.arange(image_size, dtype=np.int32)
    pixels["flags"] = 1
    if production_sass:
        rng = np.random.default_rng(20)
        pixels["reference_real"] = rng.normal(0, 0.02, image_size).astype(np.float32)
        pixels["reference_imag"] = rng.normal(0, 0.02, image_size).astype(np.float32)
        pixels["corr"] = rng.uniform(0, 150_000, image_size).astype(np.float32)
        pixels["corr"][rng.random(image_size) < 0.2] = 0
    else:
        pixels["reference_real"] = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
        pixels["reference_imag"] = np.asarray([0, 1, 0, 1, 0], dtype=np.float32)
        pixels["shifted_real"] = np.asarray([0, 0.5, 1, 1.5, 2], dtype=np.float32)
        pixels["shifted_imag"] = np.asarray([0, 0, 0.5, 0.5, 1], dtype=np.float32)
        pixels["corr"] = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
    pixels["diff_real"] = np.subtract(
        pixels["reference_real"], pixels["shifted_real"], dtype=np.float32
    )
    pixels["diff_imag"] = np.subtract(
        pixels["reference_imag"], pixels["shifted_imag"], dtype=np.float32
    )
    pixels["contribution"] = _cuda_fine_contribution(
        pixels["diff_real"],
        pixels["diff_imag"],
        pixels["corr"],
    )
    lanes = _replay_lanes(pixels["contribution"])
    if corrupt_lane:
        lanes[0] = np.float32(lanes[0] + 1)
    sum_init = np.float32(7.25)
    replay = np.float32(_reduce_lanes(lanes) + sum_init)
    candidates["sparse_index"] = 9
    candidates["rotation_id"] = 101
    candidates["rotation_local"] = 124
    candidates["translation_id"] = 56
    candidates["matrix"][0] = np.eye(3, dtype=np.float32).reshape(-1)
    candidates["translation"][0] = np.asarray([0.1, 0.2, 0], dtype=np.float32)
    candidates["sum_init"] = sum_init
    if production_sass:
        production_lanes = _cuda_fine_production_lanes(
            pixels["diff_real"],
            pixels["diff_imag"],
            pixels["corr"],
        )
        production = np.float32(_reduce_lanes(production_lanes) + sum_init)
    else:
        production = np.float32(replay + production_offset)
    candidates["production_raw_diff2"] = production
    candidates["replay_raw_diff2"] = replay
    production_exact = bool(
        np.asarray([production], dtype=np.float32).view(np.uint32)[0]
        == np.asarray([replay], dtype=np.float32).view(np.uint32)[0]
    )
    candidates["flags"] = 3 if production_exact else 1
    candidates["lane_partials"][0] = lanes

    header = np.zeros(64, dtype="<u8")
    header[:5] = [1, HEADER_SIZE, CANDIDATE_SIZE, PIXEL_SIZE, FOOTER_SIZE]
    header[5:9] = [10, 2, 36655, 42988]
    header[11:20] = [124, 1, image_size, image_size, 256, 116, 12, LANE_COUNT, 7]
    header[20] = struct.unpack("<I", struct.pack("<f", sum_init))[0]
    header[21] = int(production_exact)
    header[28:31] = 1
    footer = np.asarray([1, image_size], dtype="<u8")
    path.write_bytes(
        b"RLNFNOP1HEADER\0\0"
        + header.tobytes()
        + candidates.tobytes()
        + pixels.tobytes()
        + b"RLNFNOP1FOOTER\0\0"
        + footer.tobytes()
    )


def test_fine_operand_capture_validates_bitwise_replay(tmp_path):
    path = tmp_path / "capture.bin"
    _write_capture(path)

    report = validate_capture(
        load_fine_operand_capture(path),
        expected_stack=42988,
        expected_class=2,
        expected_rotation_local=124,
        expected_translations=(56,),
    )

    assert report["status"] == "accepted"
    assert report["exact_production_replay_count"] == 1
    assert report["candidates"][0]["production_replay_exact"] is True


def test_fine_operand_capture_rejects_corrupt_lane(tmp_path):
    path = tmp_path / "capture.bin"
    _write_capture(path, corrupt_lane=True)

    with pytest.raises(ValueError, match="pre-tree lanes do not replay bitwise"):
        validate_capture(load_fine_operand_capture(path))


def test_fine_operand_capture_reports_nonexact_production_replay(tmp_path):
    path = tmp_path / "capture.bin"
    _write_capture(path, production_offset=np.float32(0.25))

    report = validate_capture(load_fine_operand_capture(path))

    assert report["status"] == "accepted"
    assert report["exact_production_replay_count"] == 0
    assert report["production_replay_mismatch_count"] == 1


def test_fine_operand_capture_replays_production_sass_fma_order(tmp_path):
    path = tmp_path / "capture.bin"
    _write_capture(path, production_sass=True)

    report = validate_capture(load_fine_operand_capture(path))

    assert report["exact_production_replay_count"] == 0
    assert report["exact_production_sass_replay_count"] == 1
    assert report["production_sass_replay_mismatch_count"] == 0
    assert report["candidates"][0]["production_sass_replay_exact"] is True
