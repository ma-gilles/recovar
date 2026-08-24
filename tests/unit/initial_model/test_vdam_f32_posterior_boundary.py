import numpy as np
import pytest

from scripts.analyze_vdam_f32_posterior_boundary import (
    _candidate_rotation_matrices,
    _metric,
    _scalar_metric,
    _target_row,
)


def test_metric_reports_float32_bit_mismatch_without_tolerance():
    reference = np.array([1.0, 2.0], dtype=np.float32)
    candidate = reference.copy()
    candidate[1] = np.nextafter(candidate[1], np.float32(np.inf))

    report = _metric(reference, candidate)

    assert not report["bitwise_equal"]
    assert report["exact_count"] == 1
    assert report["value_count"] == 2
    assert report["max_abs"] > 0.0


def test_metric_accepts_exact_centered_log_weight_spacing():
    native = np.array([3.0, 2.5, -1.0], dtype=np.float32)
    candidate = native + np.float32(7.25)
    native_centered = native.astype(np.float64) - np.mean(native, dtype=np.float64)
    candidate_centered = candidate.astype(np.float64) - np.mean(
        candidate, dtype=np.float64
    )

    report = _metric(native_centered, candidate_centered)

    assert report["bitwise_equal"]
    assert report["max_abs"] == 0.0


def test_scalar_metric_records_float32_bits():
    reference = np.float32(3.0)
    candidate = np.nextafter(reference, np.float32(np.inf))

    report = _scalar_metric(reference, candidate)

    assert not report["bitwise_equal"]
    assert report["candidate_bits"] == report["reference_bits"] + 1
    assert report["signed_error"] > 0.0


def test_target_row_requires_exactly_one_original_identity():
    payload = {"original_indices": np.array([12, 99, 101], dtype=np.int64)}
    assert _target_row(payload, 99) == 1
    with pytest.raises(ValueError, match="expected one contribution row"):
        _target_row(payload, 7)


def test_candidate_rotation_matrices_restores_local_row_order():
    matrices = np.stack(
        [
            np.eye(3, dtype=np.float32),
            np.full((3, 3), 2.0, dtype=np.float32),
        ]
    )
    payload = {
        "actual_counts": np.array([2], dtype=np.int64),
        "active_particle_rows": np.array([0, 0], dtype=np.int32),
        "active_rotation_rows": np.array([1, 0], dtype=np.int32),
        "active_rotations": matrices,
    }

    restored = _candidate_rotation_matrices(payload, particle_row=0)

    np.testing.assert_array_equal(restored[0], matrices[1])
    np.testing.assert_array_equal(restored[1], matrices[0])
