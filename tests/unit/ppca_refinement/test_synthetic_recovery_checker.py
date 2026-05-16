import numpy as np
import pytest

from scripts.check_ppca_synthetic_recovery import _assignment_and_contrast_checks, _linear_r2, _pose_checks
from scripts.compute_ppca_best_pose_embedding import _candidate_rotations_from_source


pytestmark = pytest.mark.unit


def test_linear_r2_recovers_exact_linear_targets():
    features = np.asarray(
        [
            [-1.0, 0.5],
            [0.0, -0.25],
            [1.0, 1.5],
            [2.0, -1.0],
        ],
        dtype=np.float64,
    )
    targets = np.stack(
        [
            2.0 * features[:, 0] - 0.5 * features[:, 1],
            -features[:, 0] + 3.0 * features[:, 1],
        ],
        axis=1,
    )

    np.testing.assert_allclose(_linear_r2(features, targets), np.ones((2,)), rtol=1e-12, atol=1e-12)


def test_assignment_and_contrast_checks_separate_class_signal_from_contrast():
    assignments = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    centers = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float64)
    jitter = np.asarray(
        [
            [0.05, 0.01],
            [-0.04, 0.03],
            [0.02, -0.02],
            [-0.01, -0.04],
        ],
        dtype=np.float64,
    )
    embedding = np.asarray(
        np.concatenate([centers[class_idx][None, :] + jitter for class_idx in range(3)], axis=0),
        dtype=np.float64,
    )
    simulation_info = {
        "image_assignment": assignments,
        "per_image_contrast": np.tile(np.asarray([0.8, 1.2, 1.2, 0.8]), 3),
    }

    checks = _assignment_and_contrast_checks(embedding, simulation_info)

    assert checks["assignment_onehot_mean_r2"] > 0.95
    assert checks["contrast_r2"] < 0.2
    np.testing.assert_array_equal(checks["assignment_counts"], np.asarray([4, 4, 4]))


def test_embedding_candidate_rotations_support_exact_plus_healpix():
    rots = np.tile(np.eye(3, dtype=np.float32)[None, :, :], (3, 1, 1))
    rots[1, 0, 0] = 0.5
    rots[2, 1, 1] = 0.25
    candidates = _candidate_rotations_from_source(
        rotation_source="simulation-info-plus-healpix",
        simulation_info={"rots": rots},
        healpix_order=0,
        n_total=2,
    )

    assert candidates.shape[0] > 2
    np.testing.assert_allclose(candidates[:2], rots[:2])


def test_pose_checks_accept_local_result_pose_matrices():
    rots = np.tile(np.eye(3, dtype=np.float32)[None, :, :], (2, 1, 1))
    trans = np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)
    checks = _pose_checks(
        {"rots": rots, "trans": trans},
        {
            "best_rotation_matrix": rots,
            "best_translation": trans,
            "best_translation_idx": np.asarray([0, 1], dtype=np.int64),
        },
        rotation_source="result-matrices",
        healpix_order=0,
        offset_range_px=1.0,
        offset_step_px=1.0,
        translation_source="result-vectors",
    )

    assert checks["rotation_error_deg_median"] == pytest.approx(0.0)
    assert checks["translation_error_px_median"] == pytest.approx(0.0)
