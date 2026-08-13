import struct

import numpy as np

from scripts.analyze_k1_projected_power_boundary import _raw_f32, _shell_features


def test_raw_f32_round_trip(tmp_path):
    values = np.asarray([1.25, -2.5, 3.75], dtype="<f4")
    path = tmp_path / "values.f32"
    path.write_bytes(struct.pack("<Q", values.size) + values.tobytes())

    np.testing.assert_array_equal(_raw_f32(path), values)


def test_shell_features_matches_explicit_float32_arithmetic():
    real = np.asarray([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]], dtype=np.float32)
    imag = np.asarray([[2.0, 1.0, 0.0], [1.5, 0.5, -2.0]], dtype=np.float32)
    ctf = np.asarray([0.5, -2.0, 1.25], dtype=np.float32)
    shells = np.asarray([0, 1, 1], dtype=np.int32)

    feature_f32, feature_f64 = _shell_features(
        real,
        imag,
        ctf,
        shells,
        np.asarray([0, 1], dtype=np.int32),
    )

    scaled_real = (real * ctf[None, :]).astype(np.float32)
    scaled_imag = (imag * ctf[None, :]).astype(np.float32)
    power = (scaled_real * scaled_real + scaled_imag * scaled_imag).astype(np.float32)
    expected = np.stack((power[:, 0], power[:, 1] + power[:, 2]), axis=1)
    np.testing.assert_array_equal(feature_f32, expected)
    np.testing.assert_allclose(feature_f64, expected, rtol=0.0, atol=0.0)
