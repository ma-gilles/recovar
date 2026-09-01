from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile

from scripts.analyze_k1_pose_tail_capture import _relion_euler_to_matrix, analyze


def test_pose_tail_capture_recovers_winner_and_compares_relion_state(tmp_path: Path):
    eulers = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    rotations = _relion_euler_to_matrix(eulers).astype(np.float32)
    capture = tmp_path / "capture.npz"
    np.savez(
        capture,
        original_index=np.asarray(1),
        iteration=np.asarray(6),
        probs=np.asarray([[0.1, 0.2], [0.3, 0.9]]),
        rotations=rotations,
        fine_translations=np.asarray([[0.0, 0.0], [0.5, -0.25]], dtype=np.float32),
        relion_integer_pre_shift=np.asarray([1, -2], dtype=np.int32),
        reconstruction_n_significant=np.asarray(7),
        candidate_mask=np.ones((2, 2), dtype=bool),
        scores_with_prior=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        scores_pre_prior=np.asarray([[0.5, 1.5], [2.5, 3.5]]),
    )
    particle_star = tmp_path / "particles.star"
    starfile.write(
        {"particles": pd.DataFrame({"rlnImageName": ["1@stack.mrcs", "2@stack.mrcs"]})},
        particle_star,
    )
    relion_star = tmp_path / "run_it007_data.star"
    starfile.write(
        {
            "particles": pd.DataFrame(
                {
                    "rlnImageName": ["2@stack.mrcs"],
                    "rlnAngleRot": [40.0],
                    "rlnAngleTilt": [50.0],
                    "rlnAnglePsi": [60.0],
                    "rlnOriginXAngst": [3.0],
                    "rlnOriginYAngst": [-4.5],
                    "rlnMaxValueProbDistribution": [0.85],
                    "rlnNrOfSignificantSamples": [8],
                }
            )
        },
        relion_star,
    )

    report = analyze(
        capture_path=capture,
        particle_star=particle_star,
        relion_data_star=relion_star,
        voxel_size=2.0,
    )

    assert report["physical_iteration"] == 7
    assert report["rln_image_name"] == "2@stack.mrcs"
    assert report["winner_indices"] == {"rotation": 1, "translation": 1}
    assert report["active_candidate_count"] == 4
    assert report["capture_winner"]["translation_angstrom"] == [3.0, -4.5]
    assert report["winner_margin"]["runner_up_indices"] == {
        "rotation": 1,
        "translation": 0,
    }
    assert report["winner_margin"]["probability_gap"] == pytest.approx(0.6)
    assert report["winner_margin"]["score_with_prior_gap"] == pytest.approx(1.0)
    assert report["capture_vs_relion"]["rotation_geodesic_deg"] == pytest.approx(0.0, abs=2e-5)
    assert report["capture_vs_relion"]["translation_l2_angstrom"] == 0.0
    assert report["capture_vs_relion"]["pmax_residual"] == pytest.approx(0.05)
    assert report["capture_vs_relion"]["significant_count_residual"] == -1
    assert report["relion_pose_in_capture_grid"]["closest_rotation_index"] == 1
    assert report["relion_pose_in_capture_grid"]["closest_translation_index"] == 1
    assert report["relion_pose_in_capture_grid"]["joint_candidate_rank"] == 1
    assert report["relion_pose_in_capture_grid"]["joint_candidate_probability"] == 0.9
    assert report["relion_pose_in_capture_grid"]["probability_ratio_to_winner"] == 1.0
    assert report["relion_pose_in_capture_grid"]["score_with_prior_delta_from_winner"] == 0.0
