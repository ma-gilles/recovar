from pathlib import Path

import numpy as np

from scripts.analyze_k1_target_factor_matrix import summarize_arm


def test_summarize_arm_reports_fixed_coarse_to_fine_amplification(tmp_path: Path) -> None:
    root = tmp_path / "arm"
    (root / "pass1").mkdir(parents=True)
    (root / "pass2").mkdir()
    weights = np.asarray([0.6, 0.25, 0.149, 0.001], dtype=np.float64)
    significant = np.asarray([True, True, True, False])
    np.savez(
        root / "pass1" / "significance_orig082009_it002_cs100.npz",
        original_index=np.int64(82009),
        weights_full=weights,
        significant_mask=significant,
        adaptive_fraction=np.float64(0.999),
        relion_f32_sum_weight=np.float32(10.0),
        relion_f32_significant_weight=np.float32(1.0),
    )
    candidate_mask = np.asarray([[True, True], [True, False]])
    probabilities = np.asarray([[0.4, 0.3], [0.3, 0.0]], dtype=np.float64)
    np.savez(
        root / "pass2" / "pass2_orig082009_cs100.npz",
        original_index=np.int64(82009),
        candidate_mask=candidate_mask,
        probs=probabilities,
        oversampled_rot_indices=np.asarray([10, 20], dtype=np.int64),
        reconstruction_n_significant=np.int64(2),
    )

    result = summarize_arm(
        name="candidate",
        root=root,
        expected_native_coarse_count=2,
        extra_coarse_index=2,
        extra_fine_rotation_first=20,
        extra_fine_rotation_last=20,
    )

    assert result["coarse"]["selected_count"] == 3
    assert result["coarse"]["extra_rank_one_based"] == 3
    assert result["coarse"]["extra_selected"] is True
    assert result["fine"]["active_candidate_count"] == 3
    assert result["fine"]["extra_active_candidate_count"] == 1
    assert result["fine"]["extra_probability_mass"] == 0.3
    assert result["fine"]["winner_global_rotation"] == 10
