from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_k1_coarse_live_operands as analyzer


def _write_live_artifact(path: Path) -> tuple[np.ndarray, ...]:
    pixel_count = 4
    translation_count = 2
    reference_real = np.asarray([1, 2, 3, 4], dtype="<f4")
    reference_imag = np.asarray([5, 6, 7, 8], dtype="<f4")
    shifted_real = np.arange(8, dtype="<f4").reshape(2, 4)
    shifted_imag = (np.arange(8, dtype="<f4") + 10).reshape(2, 4)
    correction_half = np.asarray([0.5, 1.0, 1.5, 2.0], dtype="<f4")
    values = np.concatenate(
        [
            reference_real,
            reference_imag,
            shifted_real.reshape(-1),
            shifted_imag.reshape(-1),
            correction_half,
        ]
    )
    header = [0] * 32
    header[0] = 1
    header[1] = analyzer.HEADER_STRUCT.size
    header[2] = 4
    header[3] = analyzer.FOOTER_STRUCT.size
    header[4] = 2
    header[5] = 22883
    header[6] = 30592
    header[9] = 1
    header[10] = 0
    header[12] = pixel_count
    header[13] = translation_count
    header[14] = 20197
    header[15] = 5
    header[16] = 16
    header[17] = 128
    header[18] = 4
    header[19] = 16
    header[20] = values.size
    header[21] = 4096
    header[22] = 1
    header[23] = 2
    header[24:27] = [1, 1, 1]
    payload = analyzer.HEADER_STRUCT.pack(analyzer.HEADER_MAGIC, *header)
    payload += values.tobytes()
    payload += analyzer.FOOTER_STRUCT.pack(
        analyzer.FOOTER_MAGIC, translation_count, pixel_count
    )
    path.write_bytes(payload)
    return (
        reference_real,
        reference_imag,
        shifted_real,
        shifted_imag,
        correction_half,
    )


def test_load_live_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "part22883_stack30592.p1-live-v1.bin"
    expected = _write_live_artifact(path)

    actual = analyzer.load_live_artifact(path)

    assert actual.part_id == 22883
    assert actual.stack_index == 30592
    for actual_values, expected_values in zip(
        (
            actual.reference_real,
            actual.reference_imag,
            actual.shifted_real,
            actual.shifted_imag,
            actual.correction_half,
        ),
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(actual_values, expected_values)


def test_comparison_reports_first_bitwise_mismatch() -> None:
    left = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    right = left.copy()
    right.view(np.uint32)[1] += 1

    report = analyzer._comparison(left, right)

    assert report["bitwise_equal_count"] == 2
    assert report["first_mismatch_index"] == [1]
    assert report["first_left"] == 2.0
    assert report["first_right"] != 2.0


def test_replay_live_lanes_uses_relion_thread_mapping() -> None:
    header = [0] * 32
    header[12] = 32
    header[13] = 29
    header[17] = 128
    header[18] = 4
    live = analyzer.LiveCoarseOperands(
        path=Path("capture.bin"),
        sha256="0" * 64,
        header=tuple(header),
        reference_real=np.ones(32, dtype=np.float32),
        reference_imag=np.zeros(32, dtype=np.float32),
        shifted_real=np.zeros((29, 32), dtype=np.float32),
        shifted_imag=np.zeros((29, 32), dtype=np.float32),
        correction_half=np.ones(32, dtype=np.float32),
    )

    lanes = analyzer.replay_live_lanes(live)

    np.testing.assert_array_equal(lanes[:4], np.full(4, 8, dtype=np.float32))
    np.testing.assert_array_equal(lanes[29:33], np.full(4, 8, dtype=np.float32))
    np.testing.assert_array_equal(lanes[58:62], np.full(4, 8, dtype=np.float32))
    np.testing.assert_array_equal(lanes[87:91], np.full(4, 8, dtype=np.float32))
    np.testing.assert_array_equal(lanes[116:], np.zeros(12, dtype=np.float32))


def test_optional_control_captures_are_all_or_none() -> None:
    paths = (Path("operands.bin"), Path("lanes.bin"), Path("components.bin"))

    assert analyzer._has_complete_controls(paths)
    assert not analyzer._has_complete_controls((None, None, None))
    with pytest.raises(ValueError, match="supplied together"):
        analyzer._has_complete_controls((paths[0], None, paths[2]))
