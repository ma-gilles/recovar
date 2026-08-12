from pathlib import Path

import numpy as np

from scripts.analyze_k1_scoring_noise_boundary import (
    _load_capture,
    _ulp_distance,
    _ulp_summary,
)


def test_load_capture_round_trips_runtime_noise(tmp_path: Path):
    sigma2 = np.asarray([2.0, 4.0, 8.0], dtype="<f8")
    minvsigma2 = np.asarray([0.5, 0.25, 0.125], dtype="<f4")
    header = np.asarray(
        [1, 2, 1, 0, sigma2.size, 8, 8, 4, 0] + [0] * 7,
        dtype="<u8",
    )
    path = tmp_path / "scoring-noise.bin"
    path.write_bytes(
        b"RLNSIGMAV1" + b"\0" * 6 + header.tobytes() + sigma2.tobytes() + minvsigma2.tobytes()
    )

    actual_header, actual_sigma2, actual_minvsigma2 = _load_capture(path)

    assert np.array_equal(actual_header, header)
    assert np.array_equal(actual_sigma2, sigma2)
    assert np.array_equal(actual_minvsigma2, minvsigma2)


def test_ulp_distance_handles_adjacent_positive_float32_values():
    reference = np.asarray([1.0, 2.0], dtype=np.float32)
    candidate = np.nextafter(reference, np.float32(np.inf), dtype=np.float32)

    assert np.array_equal(_ulp_distance(reference, candidate), np.ones(2, dtype=np.int64))


def test_ulp_summary_partitions_exact_adjacent_and_larger_distances():
    assert _ulp_summary(np.asarray([0, 0, 1, 2, 9], dtype=np.int64)) == {
        "exact_shell_count": 2,
        "one_ulp_shell_count": 1,
        "greater_than_one_ulp_shell_count": 2,
        "max_ulp": 9,
    }
