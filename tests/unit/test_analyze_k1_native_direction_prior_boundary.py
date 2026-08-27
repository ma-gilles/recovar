import numpy as np
import pytest

from scripts.analyze_k1_native_direction_prior_boundary import (
    compare_rotation_log_priors,
    native_direction_major_to_recovar_psi_major,
)


@pytest.mark.unit
def test_native_direction_major_to_recovar_psi_major():
    native = np.arange(12, dtype=np.float32)
    actual = native_direction_major_to_recovar_psi_major(
        native,
        n_directions=4,
        n_psi=3,
    )
    expected = np.asarray([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.unit
def test_compare_rotation_log_priors_classifies_candidate_closer_to_native():
    native = np.asarray([-4.0, -3.0, -2.0, -1.0], dtype=np.float32)
    native_recovar_order = native_direction_major_to_recovar_psi_major(
        native,
        n_directions=2,
        n_psi=2,
    )
    candidate = native_recovar_order.copy()
    candidate[1] += np.float32(0.125)
    replay = native_recovar_order.copy()
    replay[1] += np.float32(0.5)
    report = compare_rotation_log_priors(
        native_direction_major=native,
        candidate_psi_major=candidate,
        replay_psi_major=replay,
        n_directions=2,
        n_psi=2,
        target_rotations=(1,),
    )

    assert report["classification"]["candidate_closer_to_native_by_support_then_rms"] is True
    assert report["classification"]["candidate_native_rms_over_replay_native_rms"] == 0.25
    assert report["targets"] == [
        {
            "rotation": 1,
            "native_log_prior": -2.0,
            "candidate_log_prior": -1.875,
            "candidate_minus_native": 0.125,
            "replay_log_prior": -1.5,
            "replay_minus_native": 0.5,
            "candidate_minus_replay": -0.375,
        }
    ]


@pytest.mark.unit
def test_compare_rotation_log_priors_handles_zero_probability_directions():
    # Native RELION's GPU buffer uses 0 plus a separate zero-mask sentinel;
    # RECOVAR represents the same log-probability as -inf.
    native = np.asarray([0.0, -3.0, -2.0, -1.0], dtype=np.float32)
    native_recovar_order = native_direction_major_to_recovar_psi_major(
        native,
        n_directions=2,
        n_psi=2,
    )
    candidate = native_recovar_order.copy()
    candidate[candidate == 0.0] = -np.inf
    report = compare_rotation_log_priors(
        native_direction_major=native,
        candidate_psi_major=candidate,
        n_directions=2,
        n_psi=2,
        target_rotations=(0,),
    )

    assert report["candidate_vs_native"] == {
        "float32_exact": True,
        "finite_support_mismatch_count": 0,
        "common_finite_count": 3,
        "residual": {
            "median_abs": 0.0,
            "p95_abs": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
        },
    }
