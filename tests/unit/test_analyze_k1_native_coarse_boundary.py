from __future__ import annotations

import struct

import numpy as np

from scripts.analyze_k1_native_coarse_boundary import load_native_coarse_capture


def test_load_native_coarse_capture(tmp_path):
    path = tmp_path / "particle.coarse-v1.bin"
    header = np.zeros(32, dtype="<u8")
    header[:18] = [1, 2, 1204, 229, 2, 2, 4, 4, 1, 3, 0, 0, 0, 0, 0, 4, 2, 2]
    arrays = [
        np.arange(4, dtype="<f4"),
        np.array([0.1, 0.2], dtype="<f4"),
        np.array([0.3, 0.4], dtype="<f4"),
        np.array([0, 1], dtype="u1"),
        np.array([1, 0], dtype="u1"),
        np.arange(4, dtype="<f4") + 10,
        np.arange(4, dtype="<f4") + 20,
        np.arange(4, dtype="<f4") + 30,
        np.arange(4, dtype="<f4") + 40,
    ]
    path.write_bytes(
        b"RLNCOARSEV1".ljust(16, b"\0")
        + header.tobytes()
        + b"".join(array.tobytes() for array in arrays)
    )
    capture = load_native_coarse_capture(path)
    assert int(capture.header[2]) == 1204
    np.testing.assert_array_equal(capture.raw_diff2, np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(capture.orientation_zero, [False, True])
    np.testing.assert_array_equal(capture.translation_zero, [True, False])
    np.testing.assert_array_equal(capture.cumulative_weights, np.arange(4) + 40)


def test_float_header_layout_is_little_endian():
    assert struct.unpack("<f", struct.pack("<I", 0x3F800000))[0] == 1.0
