from __future__ import annotations

import numpy as np

from scripts.compare_relion_recovar_estep_dump import compare_dumps


def _write_flat_real(path, values):
    arr = np.asarray(values, dtype=np.float64)
    with open(path, "wb") as f:
        np.asarray([arr.size], dtype=np.int32).tofile(f)
        arr.tofile(f)


def _write_flat_int(path, values):
    arr = np.asarray(values, dtype=np.int32)
    with open(path, "wb") as f:
        np.asarray([arr.size], dtype=np.int32).tofile(f)
        arr.tofile(f)


def _write_scalar(path, value):
    np.asarray([float(value)], dtype=np.float64).tofile(path)


def test_compare_relion_matrix_match_uses_compact_rotation_rows(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass1_acc_rot_id.bin", [1000, 1001])
    _write_flat_int(relion_dir / "pass1_acc_rot_idx.bin", [0, 1])
    _write_flat_int(relion_dir / "pass1_acc_trans_idx.bin", [0, 0])
    _write_flat_real(relion_dir / "pass1_candidate_weight_normalized.bin", [0.25, 0.75])
    _write_flat_real(relion_dir / "pass1_exp_Mweight_raw_preprior.bin", [-4.0, -8.0])

    relion_rots = np.stack(
        [
            np.eye(3),
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ],
        axis=0,
    )
    _write_flat_real(relion_dir / "pass1_class0_fine_eulers.bin", relion_rots.reshape(-1))

    recovar_npz = tmp_path / "recovar_pass2.npz"
    recovar_rots = np.broadcast_to(np.eye(3) * 3.0, (32, 3, 3)).copy()
    recovar_rots[10] = relion_rots[0]
    recovar_rots[20] = relion_rots[1]
    scores_pre = np.full((32, 1), -np.inf, dtype=np.float64)
    scores_pre[10, 0] = 4.0
    scores_pre[20, 0] = 8.0
    probs = np.zeros_like(scores_pre)
    probs[10, 0] = 0.25
    probs[20, 0] = 0.75
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        current_size=np.int64(56),
        n_fine_trans=np.int64(1),
        fine_translations=np.zeros((1, 2), dtype=np.float32),
        rotations=recovar_rots.astype(np.float32),
        oversampled_rot_indices=np.arange(32, dtype=np.int64),
        parent_map=np.arange(32, dtype=np.int32),
        candidate_mask=np.isfinite(scores_pre),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.zeros(32, dtype=np.float64),
        translation_log_prior=np.zeros(1, dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((1, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(relion_dir, recovar_npz, match_mode="matrix")

    assert result["match_mode"] == "matrix"
    assert result["match_details"]["rotation_matrix_matcher"] == "chunked_unique_relion_rows"
    assert result["common_candidate_count"] == 2
    assert result["relion_top_key"] == [20, 0]
    assert result["recovar_top_key"] == [20, 0]
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_matrix_match_selects_dumped_kclass_fine_eulers(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass1_acc_rot_id.bin", [11])
    _write_flat_int(relion_dir / "pass1_acc_rot_idx.bin", [2])
    _write_flat_int(relion_dir / "pass1_acc_trans_idx.bin", [0])
    _write_flat_real(relion_dir / "pass1_candidate_weight_normalized.bin", [1.0])
    _write_flat_real(relion_dir / "pass1_exp_Mweight_raw_preprior.bin", [-9.0])

    class0_rots = np.stack([np.eye(3), np.eye(3) * 2.0], axis=0)
    class1_rots = np.stack(
        [
            np.eye(3),
            np.eye(3),
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ],
        axis=0,
    )
    _write_flat_real(relion_dir / "pass1_class0_fine_eulers.bin", class0_rots.reshape(-1))
    _write_flat_real(relion_dir / "pass1_class1_fine_eulers.bin", class1_rots.reshape(-1))

    recovar_npz = tmp_path / "recovar_pass2.npz"
    scores_pre = np.array([[9.0]], dtype=np.float64)
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        current_size=np.int64(56),
        n_fine_trans=np.int64(1),
        fine_translations=np.zeros((1, 2), dtype=np.float32),
        rotations=class1_rots[2:3].astype(np.float32),
        oversampled_rot_indices=np.array([99], dtype=np.int64),
        parent_map=np.array([0], dtype=np.int32),
        candidate_mask=np.ones((1, 1), dtype=bool),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=np.ones((1, 1), dtype=np.float64),
        rotation_log_prior=np.zeros(1, dtype=np.float64),
        translation_log_prior=np.zeros(1, dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((1, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(relion_dir, recovar_npz, match_mode="matrix")

    assert result["match_mode"] == "matrix"
    assert result["common_candidate_count"] == 1
    assert result["relion_top_key"] == [0, 0]
    assert result["recovar_top_key"] == [0, 0]
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_matrix_match_selects_requested_kclass_fine_eulers(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass1_acc_rot_id.bin", [11, 22, 23])
    _write_flat_int(relion_dir / "pass1_acc_rot_idx.bin", [0, 0, 1])
    _write_flat_int(relion_dir / "pass1_acc_trans_idx.bin", [0, 0, 0])
    _write_flat_int(relion_dir / "pass1_candidate_class_idx.bin", [0, 1, 1])
    _write_flat_real(relion_dir / "pass1_candidate_weight_normalized.bin", [1.0, 0.4, 0.6])
    _write_flat_real(relion_dir / "pass1_exp_Mweight_raw_preprior.bin", [-9.0, -2.0, -3.0])

    class0_rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    class1_rots = np.stack(
        [
            np.eye(3),
            np.diag([-1.0, -1.0, 1.0]),
        ],
        axis=0,
    )
    _write_flat_real(relion_dir / "pass1_class0_fine_eulers.bin", class0_rot.reshape(-1))
    _write_flat_real(relion_dir / "pass1_class1_fine_eulers.bin", class1_rots.reshape(-1))

    recovar_npz = tmp_path / "recovar_pass2.npz"
    scores_pre = np.array([[9.0]], dtype=np.float64)
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        class_index=np.int64(0),
        current_size=np.int64(56),
        n_fine_trans=np.int64(1),
        fine_translations=np.zeros((1, 2), dtype=np.float32),
        rotations=class0_rot[None].astype(np.float32),
        oversampled_rot_indices=np.array([0], dtype=np.int64),
        parent_map=np.array([0], dtype=np.int32),
        candidate_mask=np.ones((1, 1), dtype=bool),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=np.ones((1, 1), dtype=np.float64),
        rotation_log_prior=np.zeros(1, dtype=np.float64),
        translation_log_prior=np.zeros(1, dtype=np.float64),
    )

    result = compare_dumps(relion_dir, recovar_npz, match_mode="matrix")

    assert result["match_mode"] == "matrix"
    assert result["common_candidate_count"] == 1
    assert result["match_details"]["rotation_matrix_unique_relion_rows"] == 1
    assert result["match_details"]["rotation_matrix_match_max_frobenius"] == 0.0
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_recovar_estep_dump_matches_candidate_keys(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass0_acc_rot_idx.bin", [5, 7, 9])
    _write_flat_int(relion_dir / "pass0_acc_trans_idx.bin", [0, 2, 1])
    _write_flat_real(relion_dir / "pass0_candidate_weight_normalized.bin", [0.2, 0.7, 0.1])
    _write_flat_real(relion_dir / "pass0_exp_Mweight_raw_preprior.bin", [-10.0, -12.0, -8.0])
    _write_flat_real(relion_dir / "pass0_coarse_log_weight_preexp.bin", [9.0, 11.0, 7.0])
    _write_flat_real(relion_dir / "pass0_candidate_orientation_log_prior.bin", [-1.0, -0.5, -2.0])
    _write_flat_real(relion_dir / "pass0_candidate_offset_log_prior.bin", [-0.1, -0.2, -0.3])

    recovar_npz = tmp_path / "recovar_pass2.npz"
    probs = np.array([[0.2, 0.0, 0.0], [0.0, 0.05, 0.75]], dtype=np.float64)
    scores_pre = np.array([[10.0, -np.inf, -np.inf], [-np.inf, 1.0, 12.0]], dtype=np.float64)
    scores_with = np.array([[9.0, -np.inf, -np.inf], [-np.inf, 0.5, 11.0]], dtype=np.float64)
    mask = np.array([[True, False, False], [False, True, True]], dtype=bool)
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(123),
        local_index=np.int64(45),
        current_size=np.int64(56),
        n_fine_trans=np.int64(3),
        fine_translations=np.zeros((3, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([5, 7], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        candidate_mask=mask,
        scores_with_prior=scores_with,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.array([-1.0, -0.5], dtype=np.float64),
        translation_log_prior=np.array([-0.1, -0.4, -0.2], dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((2, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(relion_dir, recovar_npz, reconstruction_only=True)

    assert result["common_candidate_count"] == 2
    assert result["relion_only_count"] == 1
    assert result["recovar_only_count"] == 1
    assert result["relion_top_key"] == [7, 2]
    assert result["recovar_top_key"] == [7, 2]
    assert result["common_prob_l1_after_common_renorm"] < 0.05
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0
    assert result["common_combined_log_prior_diff"]["max_abs"] == 0.0
    assert result["common_combined_log_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_recovar_estep_dump_reports_both_engines_top_candidate_terms(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    # A real adaptive dump contains both passes. These deliberately
    # incompatible coarse values must not be mixed into the fine comparison.
    _write_flat_int(relion_dir / "pass0_acc_rot_id.bin", [50])
    _write_flat_int(relion_dir / "pass0_acc_rot_idx.bin", [0])
    _write_flat_int(relion_dir / "pass0_acc_trans_idx.bin", [0])
    _write_flat_real(relion_dir / "pass0_candidate_weight_normalized.bin", [1.0])
    _write_flat_real(relion_dir / "pass0_exp_Mweight_raw_preprior.bin", [-100.0])
    _write_flat_real(relion_dir / "pass0_candidate_orientation_log_prior.bin", [-10.0])
    _write_flat_real(relion_dir / "pass0_candidate_offset_log_prior.bin", [-20.0])
    _write_flat_real(relion_dir / "pass0_candidate_combined_log_prior.bin", [-99.0])
    _write_flat_int(relion_dir / "pass1_acc_rot_id.bin", [5, 7])
    _write_flat_int(relion_dir / "pass1_acc_rot_idx.bin", [0, 1])
    _write_flat_int(relion_dir / "pass1_acc_trans_idx.bin", [0, 1])
    _write_flat_real(relion_dir / "pass1_candidate_weight_normalized.bin", [0.7, 0.3])
    _write_flat_real(relion_dir / "pass1_exp_Mweight_raw_preprior.bin", [-12.0, -10.0])
    _write_flat_real(relion_dir / "pass1_candidate_orientation_log_prior.bin", [-1.0, -2.0])
    _write_flat_real(relion_dir / "pass1_candidate_offset_log_prior.bin", [-0.1, -0.2])

    recovar_npz = tmp_path / "recovar_pass2.npz"
    scores_pre = np.array([[11.5, -np.inf], [-np.inf, 10.5]], dtype=np.float64)
    scores_with = np.array([[10.4, -np.inf], [-np.inf, 8.3]], dtype=np.float64)
    probs = np.array([[0.2, 0.0], [0.0, 0.8]], dtype=np.float64)
    mask = np.isfinite(scores_pre)
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(123),
        local_index=np.int64(45),
        current_size=np.int64(56),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([5, 7], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        candidate_mask=mask,
        scores_with_prior=scores_with,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.array([-1.0, -2.0], dtype=np.float64),
        translation_log_prior=np.array([-0.1, -0.2], dtype=np.float64),
    )

    result = compare_dumps(relion_dir, recovar_npz)

    assert result["relion_generic_candidate_prefix"] == "pass1"
    assert result["relion_top_key"] == [5, 0]
    assert result["recovar_top_key"] == [7, 1]
    assert result["cross_top_candidate_details"] == [
        {
            "key": [5, 0],
            "relion": {
                "prob": 0.7,
                "score_pre_prior": 12.0,
                "score_with_prior": 10.9,
                "rotation_log_prior": -1.0,
                "translation_log_prior": -0.1,
                "combined_log_prior": -1.1,
            },
            "recovar": {
                "prob": 0.2,
                "score_pre_prior": 11.5,
                "score_with_prior": 10.4,
                "rotation_log_prior": -1.0,
                "translation_log_prior": -0.1,
                "combined_log_prior": -1.1,
            },
        },
        {
            "key": [7, 1],
            "relion": {
                "prob": 0.3,
                "score_pre_prior": 10.0,
                "score_with_prior": 7.8,
                "rotation_log_prior": -2.0,
                "translation_log_prior": -0.2,
                "combined_log_prior": -2.2,
            },
            "recovar": {
                "prob": 0.8,
                "score_pre_prior": 10.5,
                "score_with_prior": 8.3,
                "rotation_log_prior": -2.0,
                "translation_log_prior": -0.2,
                "combined_log_prior": -2.2,
            },
        },
    ]


def test_compare_relion_recovar_estep_dump_uses_reconstruction_masks(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass1_acc_rot_idx.bin", [5, 7, 9])
    _write_flat_int(relion_dir / "pass1_acc_trans_idx.bin", [0, 2, 1])
    _write_flat_real(relion_dir / "pass1_candidate_weight_normalized.bin", [0.2, 0.7, 0.1])
    _write_flat_real(relion_dir / "pass1_exp_Mweight_raw_preprior.bin", [-10.0, -12.0, -8.0])
    _write_flat_real(relion_dir / "pass1_candidate_combined_log_prior.bin", [-1.1, -1.0, -2.3])
    _write_flat_int(relion_dir / "pass1_candidate_in_reconstruction_set.bin", [0, 1, 0])

    recovar_npz = tmp_path / "recovar_pass2.npz"
    probs = np.array([[0.2, 0.0, 0.0], [0.0, 0.05, 0.75]], dtype=np.float64)
    scores_pre = np.array([[10.0, -np.inf, -np.inf], [-np.inf, 1.0, 12.0]], dtype=np.float64)
    scores_with = np.array([[8.9, -np.inf, -np.inf], [-np.inf, -0.5, 11.0]], dtype=np.float64)
    mask = np.array([[True, False, False], [False, True, True]], dtype=bool)
    reconstruction_mask = np.array([[False, False, False], [False, False, True]], dtype=bool)
    reconstruction_probs = np.where(reconstruction_mask, probs, 0.0)
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(123),
        local_index=np.int64(45),
        current_size=np.int64(56),
        n_fine_trans=np.int64(3),
        fine_translations=np.zeros((3, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([5, 7], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        candidate_mask=mask,
        reconstruction_mask=reconstruction_mask,
        reconstruction_probs=reconstruction_probs,
        reconstruction_n_significant=np.int64(1),
        scores_with_prior=scores_with,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.array([-1.0, -0.5], dtype=np.float64),
        translation_log_prior=np.array([-0.1, -0.4, -0.2], dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((2, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(relion_dir, recovar_npz, reconstruction_only=True)

    assert result["relion_selected_field"] == "candidate_in_reconstruction_set"
    assert result["recovar_selected_field"] == "reconstruction_mask"
    assert result["recovar_reconstruction_n_significant"] == 1
    assert result["relion_candidate_count"] == 1
    assert result["recovar_candidate_count"] == 1
    assert result["common_candidate_count"] == 1
    assert result["relion_top_key"] == [7, 2]
    assert result["recovar_top_key"] == [7, 2]
    assert result["common_combined_log_prior_diff"]["max_abs"] == 0.30000000000000004


def test_compare_relion_recovar_estep_dump_reads_local_score_schema(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass1_acc_rot_id.bin", [5, 7, 7])
    _write_flat_int(relion_dir / "pass1_acc_rot_idx.bin", [0, 1, 1])
    _write_flat_int(relion_dir / "pass1_acc_trans_idx.bin", [0, 1, 2])
    _write_flat_real(relion_dir / "pass1_candidate_weight_normalized.bin", [0.2, 0.7, 0.1])
    _write_flat_real(relion_dir / "pass1_exp_Mweight_raw_preprior.bin", [-10.0, -12.0, -8.0])
    _write_flat_int(relion_dir / "pass1_candidate_in_reconstruction_set.bin", [0, 1, 0])

    recovar_npz = tmp_path / "local_score_it011_image_000007_final.npz"
    scores = np.array([[10.0, -np.inf, -np.inf], [-np.inf, 12.0, 8.0]], dtype=np.float64)
    posterior = np.array([[0.2, 0.0, 0.0], [0.0, 0.7, 0.1]], dtype=np.float64)
    reconstruction_mask = np.array([[False, False, False], [False, True, False]], dtype=bool)
    np.savez_compressed(
        recovar_npz,
        selected_global_image_indices=np.array([7], dtype=np.int64),
        selected_local_image_indices=np.array([7], dtype=np.int64),
        current_size=np.array([-1], dtype=np.int32),
        posterior=posterior[None, :, :],
        pass2_scores_raw=scores[None, :, :],
        pass2_scores_total=scores[None, :, :],
        rotation_log_prior=np.zeros((1, 2), dtype=np.float64),
        translation_log_prior=np.zeros((1, 3), dtype=np.float64),
        local_rotation_indices=np.array([5, 7], dtype=np.int64),
        local_rotation_parent_indices=np.array([100, 101], dtype=np.int64),
        translation_parent_indices=np.array([0, 1, 1], dtype=np.int64),
        local_rotation_matrices=np.zeros((2, 3, 3), dtype=np.float32),
        reconstruction_sample_mask=reconstruction_mask[None, :, :],
        n_significant_samples=np.array([1], dtype=np.int32),
    )

    all_result = compare_dumps(relion_dir, recovar_npz)

    assert all_result["recovar_selected_field"] == "finite_pass2_scores_total"
    assert all_result["common_candidate_count"] == 3
    assert all_result["relion_top_key"] == [7, 1]
    assert all_result["recovar_top_key"] == [7, 1]
    assert all_result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0

    reconstruction_result = compare_dumps(relion_dir, recovar_npz, reconstruction_only=True)

    assert reconstruction_result["recovar_selected_field"] == "reconstruction_sample_mask"
    assert reconstruction_result["recovar_reconstruction_n_significant"] == 1
    assert reconstruction_result["common_candidate_count"] == 1
    assert reconstruction_result["relion_top_key"] == [7, 1]
    assert reconstruction_result["recovar_top_key"] == [7, 1]


def test_compare_relion_recovar_estep_dump_prefers_global_acc_rot_id(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass1_acc_rot_id.bin", [5, 7, 9])
    _write_flat_int(relion_dir / "pass1_acc_rot_idx.bin", [0, 1, 2])
    _write_flat_int(relion_dir / "pass1_acc_trans_idx.bin", [0, 2, 1])
    _write_flat_real(relion_dir / "pass1_candidate_weight_normalized.bin", [0.2, 0.7, 0.1])
    _write_flat_real(relion_dir / "pass1_exp_Mweight_raw_preprior.bin", [-10.0, -12.0, -8.0])

    recovar_npz = tmp_path / "recovar_pass2.npz"
    scores_pre = np.array([[10.0, -np.inf, -np.inf], [-np.inf, 1.0, 12.0]], dtype=np.float64)
    probs = np.array([[0.2, 0.0, 0.0], [0.0, 0.05, 0.75]], dtype=np.float64)
    mask = np.array([[True, False, False], [False, True, True]], dtype=bool)
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(123),
        local_index=np.int64(45),
        current_size=np.int64(56),
        n_fine_trans=np.int64(3),
        fine_translations=np.zeros((3, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([5, 7], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        candidate_mask=mask,
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.zeros(2, dtype=np.float64),
        translation_log_prior=np.zeros(3, dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((2, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(relion_dir, recovar_npz)

    assert result["common_candidate_count"] == 2
    assert result["relion_top_key"] == [7, 2]
    assert result["recovar_top_key"] == [7, 2]


def test_compare_relion_recovar_estep_dump_reads_firstiter_cc_schema(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass0_firstiter_cc_raw_rot_idx.bin", [1, 2, 3, 4])
    _write_flat_int(relion_dir / "pass0_firstiter_cc_raw_trans_idx.bin", [0, 1, 2, 3])
    _write_flat_real(relion_dir / "pass0_firstiter_cc_exp_Mweight_raw_preonehot.bin", [-1.0, -2.0, -3.0, -4.0])
    _write_flat_int(relion_dir / "pass1_firstiter_cc_raw_rot_idx.bin", [5, 7, 9])
    _write_flat_int(relion_dir / "pass1_firstiter_cc_raw_trans_idx.bin", [0, 2, 1])
    _write_flat_real(relion_dir / "pass1_firstiter_cc_exp_Mweight_raw_preonehot.bin", [-10.0, -12.0, -8.0])

    recovar_npz = tmp_path / "recovar_pass2.npz"
    probs = np.array([[0.2, 0.0, 0.0], [0.0, 0.05, 0.75]], dtype=np.float64)
    scores_pre = np.array([[10.0, -np.inf, -np.inf], [-np.inf, 1.0, 12.0]], dtype=np.float64)
    mask = np.array([[True, False, False], [False, True, True]], dtype=bool)
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(123),
        local_index=np.int64(45),
        current_size=np.int64(56),
        n_fine_trans=np.int64(3),
        fine_translations=np.zeros((3, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([5, 7], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        candidate_mask=mask,
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.zeros(2, dtype=np.float64),
        translation_log_prior=np.zeros(3, dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((2, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(relion_dir, recovar_npz)

    assert result["relion_candidate_count"] == 3
    assert result["common_candidate_count"] == 2
    assert result["relion_top_key"] == [7, 2]
    assert result["recovar_top_key"] == [7, 2]
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_recovar_estep_dump_uses_part_specific_acc_table(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    # Stale generic firstiter files point at a different particle and would pick
    # the wrong top key if the comparator did not use the explicit ACC prefix.
    _write_flat_int(relion_dir / "pass1_firstiter_cc_raw_rot_idx.bin", [0, 1])
    _write_flat_int(relion_dir / "pass1_firstiter_cc_raw_trans_idx.bin", [0, 1])
    _write_flat_real(relion_dir / "pass1_firstiter_cc_exp_Mweight_raw_preonehot.bin", [-100.0, -1.0])

    prefix = "img0_part7778_pass1_class0_pass1"
    _write_flat_real(relion_dir / f"{prefix}_diff2_weights.bin", [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    # Some firstiter StoreWeightedSums debug dumps report orientation_num=0
    # even though the dense table is present; infer it from table length.
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 0)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)

    recovar_npz = tmp_path / "recovar_significance.npz"
    scores_pre = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float64,
    )
    weights = np.zeros_like(scores_pre)
    weights[2, 1] = 1.0
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        current_size=np.int64(56),
        n_classes=np.int64(1),
        n_rot=np.int64(3),
        n_trans=np.int64(2),
        weights_full=weights.reshape(-1),
        weights_per_class=weights[None, :, :],
        scores_pre_prior_per_class=scores_pre[None, :, :],
        scores_with_prior_per_class=scores_pre[None, :, :],
        rotations=np.zeros((3, 3, 3), dtype=np.float32),
        translations=np.zeros((2, 2), dtype=np.float32),
        rotation_log_prior=np.zeros(3, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
    )

    assert result["relion_selected_field"] == f"acc_full_grid:{prefix}"
    assert result["recovar_selected_field"] == "finite_scores_pre_prior_per_class"
    assert result["relion_candidate_count"] == 6
    assert result["recovar_candidate_count"] == 6
    assert result["common_candidate_count"] == 6
    assert result["relion_top_key"] == [2, 1]
    assert result["recovar_top_key"] == [2, 1]
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_recovar_estep_dump_ranks_part_specific_acc_tables(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    requested_prefix = "img0_part6756_pass1_class0_pass1"
    matching_prefix = "img0_part6398_pass1_class0_pass1"
    recovar_scores = np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float64)
    wrong_scores = np.array([[4.0, 1.0], [3.0, 2.0]], dtype=np.float64)
    for prefix, scores in ((requested_prefix, wrong_scores), (matching_prefix, recovar_scores)):
        _write_flat_real(relion_dir / f"{prefix}_diff2_weights.bin", -scores.reshape(-1))
        _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 2)
        _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)

    weights = np.zeros_like(recovar_scores)
    weights[0, 1] = 1.0
    recovar_npz = tmp_path / "recovar_significance.npz"
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(6756),
        local_index=np.int64(3407),
        current_size=np.int64(56),
        n_classes=np.int64(1),
        n_rot=np.int64(2),
        n_trans=np.int64(2),
        weights_full=weights.reshape(-1),
        weights_per_class=weights[None, :, :],
        scores_pre_prior_per_class=recovar_scores[None, :, :],
        scores_with_prior_per_class=recovar_scores[None, :, :],
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        translations=np.zeros((2, 2), dtype=np.float32),
        rotation_log_prior=np.zeros(2, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=requested_prefix,
    )

    assert result["requested_relion_acc_table_prefix"] == requested_prefix
    assert result["best_relion_acc_table_prefix"] == matching_prefix
    assert result["requested_relion_acc_table_prefix_rank"] == 2
    assert [item["prefix"] for item in result["relion_acc_table_prefix_rankings"]] == [
        matching_prefix,
        requested_prefix,
    ]
    assert result["relion_acc_table_prefix_rankings"][0]["score_pre_prior_centered_corr"] == 1.0
    assert result["relion_acc_table_prefix_rankings"][0]["score_pre_prior_centered_diff"]["max_abs"] == 0.0
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] > 0.0


def test_compare_relion_recovar_estep_dump_selects_significance_class(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part6334_pass1_class1_pass1"
    class0_scores = np.array([[9.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    class1_scores = np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float64)
    _write_flat_real(relion_dir / f"{prefix}_diff2_weights.bin", -class1_scores.reshape(-1))
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)

    weights = np.zeros((2, 2, 2), dtype=np.float64)
    weights[0, 0, 0] = 1.0
    weights[1, 0, 1] = 1.0
    recovar_npz = tmp_path / "recovar_significance.npz"
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(156),
        local_index=np.int64(156),
        current_size=np.int64(56),
        n_classes=np.int64(2),
        n_rot=np.int64(2),
        n_trans=np.int64(2),
        weights_full=weights.reshape(-1),
        weights_per_class=weights,
        scores_pre_prior_per_class=np.stack([class0_scores, class1_scores], axis=0),
        scores_with_prior_per_class=np.stack([class0_scores, class1_scores], axis=0),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        translations=np.zeros((2, 2), dtype=np.float32),
        rotation_log_prior=np.zeros((2, 2), dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
        recovar_class_index=1,
    )

    assert result["recovar_class_index"] == 1
    assert result["relion_top_key"] == [0, 1]
    assert result["recovar_top_key"] == [0, 1]
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_recovar_estep_dump_auto_selects_best_acc_table(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    wrong_prefix = "img0_part38821_pass1_class1_pass1"
    matching_prefix = "img0_part6334_pass1_class1_pass1"
    class1_scores = np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float64)
    wrong_scores = np.array([[4.0, 1.0], [3.0, 2.0]], dtype=np.float64)
    for prefix, scores in ((wrong_prefix, wrong_scores), (matching_prefix, class1_scores)):
        _write_flat_real(relion_dir / f"{prefix}_diff2_weights.bin", -scores.reshape(-1))
        _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 2)
        _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)

    weights = np.zeros((2, 2, 2), dtype=np.float64)
    weights[1, 0, 1] = 1.0
    recovar_npz = tmp_path / "recovar_significance.npz"
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(156),
        local_index=np.int64(156),
        current_size=np.int64(56),
        n_classes=np.int64(2),
        n_rot=np.int64(2),
        n_trans=np.int64(2),
        weights_full=weights.reshape(-1),
        weights_per_class=weights,
        scores_pre_prior_per_class=np.stack([wrong_scores, class1_scores], axis=0),
        scores_with_prior_per_class=np.stack([wrong_scores, class1_scores], axis=0),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        translations=np.zeros((2, 2), dtype=np.float32),
        rotation_log_prior=np.zeros((2, 2), dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix="auto",
        recovar_class_index=1,
    )

    assert result["requested_relion_acc_table_prefix"] == "auto"
    assert result["selected_relion_acc_table_prefix"] == matching_prefix
    assert result["best_relion_acc_table_prefix"] == matching_prefix
    assert result["recovar_class_index"] == 1
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_recovar_estep_dump_reads_storewavg_sorted_weights(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_storeWavg"
    weights = np.array(
        [
            -np.finfo(np.float32).max,
            0.0,
            2.0,
            0.0,
            4.0,
            1.0,
        ],
        dtype=np.float64,
    )
    _write_flat_real(relion_dir / f"{prefix}_sorted_weights.bin", weights)
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 3)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)

    recovar_npz = tmp_path / "recovar_significance.npz"
    scores_pre = np.full((3, 2), -np.inf, dtype=np.float64)
    scores_pre[1, 0] = np.log(2.0)
    scores_pre[2, 0] = np.log(4.0)
    scores_pre[2, 1] = np.log(1.0)
    recovar_weights = np.zeros_like(scores_pre)
    recovar_weights[1, 0] = 2.0 / 7.0
    recovar_weights[2, 0] = 4.0 / 7.0
    recovar_weights[2, 1] = 1.0 / 7.0
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        current_size=np.int64(56),
        n_classes=np.int64(1),
        n_rot=np.int64(3),
        n_trans=np.int64(2),
        weights_full=recovar_weights.reshape(-1),
        weights_per_class=recovar_weights[None, :, :],
        scores_pre_prior_per_class=scores_pre[None, :, :],
        scores_with_prior_per_class=scores_pre[None, :, :],
        rotations=np.zeros((3, 3, 3), dtype=np.float32),
        translations=np.zeros((2, 2), dtype=np.float32),
        rotation_log_prior=np.zeros(3, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
    )

    assert result["relion_selected_field"] == f"acc_storewavg_positive_weights:{prefix}"
    assert result["relion_candidate_count"] == 3
    assert result["recovar_candidate_count"] == 3
    assert result["common_candidate_count"] == 3
    assert result["relion_top_key"] == [2, 0]
    assert result["recovar_top_key"] == [2, 0]
    assert result["common_prob_l1_after_common_renorm"] == 0.0
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0
    assert result["common_score_with_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_k1_storewavg_maps_compact_rows_to_global_rotations(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part1965_storeWavg"
    _write_flat_real(relion_dir / f"{prefix}_sorted_weights.bin", [2.0, 0.0, 0.0, 4.0])
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_nr_classes.bin", 1)
    _write_scalar(relion_dir / f"{prefix}_iclass_min.bin", 0)
    _write_flat_int(relion_dir / "pass1_class0_fine_class_entries.bin", [2])
    _write_flat_int(relion_dir / "pass1_class0_fine_class_idx.bin", [0])
    _write_flat_int(relion_dir / "pass1_class0_fine_iorientclasses.bin", [3, 5])
    _write_flat_int(relion_dir / "pass1_class0_fine_iover_rots.bin", [0, 0])
    _write_flat_real(relion_dir / "pass1_pdf_orientation.bin", np.ones(8))

    recovar_npz = tmp_path / "recovar_pass2.npz"
    scores = np.array([[np.log(2.0), -np.inf], [-np.inf, np.log(4.0)]])
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(2767),
        local_index=np.int64(2767),
        current_size=np.int64(56),
        class_index=np.int64(0),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([3, 5], dtype=np.int64),
        candidate_mask=np.isfinite(scores),
        scores_with_prior=scores,
        scores_pre_prior=scores,
        probs=np.array([[1.0 / 3.0, 0.0], [0.0, 2.0 / 3.0]]),
        rotation_log_prior=np.zeros(2),
        translation_log_prior=np.zeros(2),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix="auto",
        relion_n_psi=48,
    )

    assert result["selected_relion_acc_table_prefix"] == prefix
    assert result["match_mode"] == "global"
    assert result["relion_rotation_key_mode"] == (
        "fine_iorientclasses_mod_pdf_orientation_times_iover_rots"
    )
    assert result["relion_top_key"] == [5, 1]
    assert result["recovar_top_key"] == [5, 1]
    assert result["common_candidate_count"] == 2


def test_compare_relion_recovar_estep_dump_slices_kclass_storewavg_sorted_weights(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_storeWavg"
    class0 = np.array([8.0, 0.0, 0.0, 0.0], dtype=np.float64)
    class1 = np.array([0.0, 2.0, 0.0, 4.0], dtype=np.float64)
    _write_flat_real(relion_dir / f"{prefix}_sorted_weights.bin", np.concatenate([class0, class1]))
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_nr_classes.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_iclass_min.bin", 0)

    scores_pre = np.full((2, 2), -np.inf, dtype=np.float64)
    scores_pre[0, 1] = np.log(2.0)
    scores_pre[1, 1] = np.log(4.0)
    recovar_weights = np.zeros_like(scores_pre)
    recovar_weights[0, 1] = 2.0 / 6.0
    recovar_weights[1, 1] = 4.0 / 6.0
    recovar_npz = tmp_path / "recovar_class1_pass2.npz"
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        class_index=np.int64(1),
        current_size=np.int64(56),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([0, 1], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        candidate_mask=np.isfinite(scores_pre),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=recovar_weights,
        rotation_log_prior=np.zeros(2, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
    )

    assert result["relion_selected_field"] == f"acc_storewavg_positive_weights:{prefix}:class1"
    assert result["relion_candidate_count"] == 2
    assert result["recovar_candidate_count"] == 2
    assert result["common_candidate_count"] == 2
    assert result["relion_top_key"] == [1, 1]
    assert result["recovar_top_key"] == [1, 1]
    assert result["common_prob_l1_after_common_renorm"] == 0.0
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0
    assert result["common_score_with_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_storewavg_reconstruction_uses_global_significant_weight(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_storeWavg"
    class0 = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)
    class1 = np.array([0.0, 3.0, 0.0, 10.0], dtype=np.float64)
    _write_flat_real(relion_dir / f"{prefix}_sorted_weights.bin", np.concatenate([class0, class1]))
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_nr_classes.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_iclass_min.bin", 0)
    _write_scalar(relion_dir / f"{prefix}_significant_weight.bin", 10.0)

    common_fields = dict(
        original_index=np.int64(7),
        local_index=np.int64(7),
        current_size=np.int64(56),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([0, 1], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        rotation_log_prior=np.zeros(2, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    class0_npz = tmp_path / "recovar_class0_pass2.npz"
    class0_scores = np.full((2, 2), -np.inf, dtype=np.float64)
    class0_scores[0, 0] = np.log(2.0)
    np.savez_compressed(
        class0_npz,
        **common_fields,
        class_index=np.int64(0),
        candidate_mask=np.isfinite(class0_scores),
        scores_with_prior=class0_scores,
        scores_pre_prior=class0_scores,
        probs=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        reconstruction_mask=np.zeros((2, 2), dtype=bool),
        reconstruction_probs=np.zeros((2, 2), dtype=np.float64),
        reconstruction_n_significant=np.int64(0),
    )
    class0_result = compare_dumps(
        relion_dir,
        class0_npz,
        reconstruction_only=True,
        relion_acc_table_prefix=prefix,
    )
    assert class0_result["relion_selected_field"] == f"acc_storewavg_significant_weights:{prefix}:class0"
    assert class0_result["relion_candidate_count"] == 0
    assert class0_result["recovar_candidate_count"] == 0

    class1_npz = tmp_path / "recovar_class1_pass2.npz"
    class1_scores = np.full((2, 2), -np.inf, dtype=np.float64)
    class1_scores[0, 1] = np.log(3.0)
    class1_scores[1, 1] = np.log(10.0)
    class1_recon_mask = np.zeros((2, 2), dtype=bool)
    class1_recon_mask[1, 1] = True
    class1_recon_probs = np.zeros((2, 2), dtype=np.float64)
    class1_recon_probs[1, 1] = 1.0
    np.savez_compressed(
        class1_npz,
        **common_fields,
        class_index=np.int64(1),
        candidate_mask=np.isfinite(class1_scores),
        scores_with_prior=class1_scores,
        scores_pre_prior=class1_scores,
        probs=np.array([[0.0, 3.0 / 13.0], [0.0, 10.0 / 13.0]], dtype=np.float64),
        reconstruction_mask=class1_recon_mask,
        reconstruction_probs=class1_recon_probs,
        reconstruction_n_significant=np.int64(1),
    )
    class1_result = compare_dumps(
        relion_dir,
        class1_npz,
        reconstruction_only=True,
        relion_acc_table_prefix=prefix,
    )
    assert class1_result["relion_selected_field"] == f"acc_storewavg_significant_weights:{prefix}:class1"
    assert class1_result["relion_candidate_count"] == 1
    assert class1_result["recovar_candidate_count"] == 1
    assert class1_result["common_candidate_count"] == 1
    assert class1_result["relion_top_key"] == [1, 1]
    assert class1_result["recovar_top_key"] == [1, 1]
    assert class1_result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_recovar_estep_dump_slices_variable_class_storewavg(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_storeWavg"
    class0 = np.array([9.0, 0.0], dtype=np.float64)
    class1 = np.array([0.0, 2.0, 0.0, 4.0, 1.0, 0.0], dtype=np.float64)
    _write_flat_real(relion_dir / f"{prefix}_sorted_weights.bin", np.concatenate([class0, class1]))
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 1)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_nr_classes.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_iclass_min.bin", 0)
    _write_flat_int(relion_dir / "pass1_class0_fine_class_entries.bin", [1, 3])

    scores_pre = np.full((3, 2), -np.inf, dtype=np.float64)
    scores_pre[0, 1] = np.log(2.0)
    scores_pre[1, 1] = np.log(4.0)
    scores_pre[2, 0] = np.log(1.0)
    recovar_weights = np.zeros_like(scores_pre)
    recovar_weights[0, 1] = 2.0 / 7.0
    recovar_weights[1, 1] = 4.0 / 7.0
    recovar_weights[2, 0] = 1.0 / 7.0
    recovar_npz = tmp_path / "recovar_class1_pass2.npz"
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        class_index=np.int64(1),
        current_size=np.int64(56),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=np.zeros((3, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([0, 1, 2], dtype=np.int64),
        parent_map=np.array([0, 1, 2], dtype=np.int32),
        candidate_mask=np.isfinite(scores_pre),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=recovar_weights,
        rotation_log_prior=np.zeros(3, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
    )

    assert result["relion_selected_field"] == f"acc_storewavg_positive_weights:{prefix}:class1"
    assert result["relion_candidate_count"] == 3
    assert result["recovar_candidate_count"] == 3
    assert result["common_candidate_count"] == 3
    assert result["relion_top_key"] == [1, 1]
    assert result["recovar_top_key"] == [1, 1]
    assert result["common_prob_l1_after_common_renorm"] == 0.0
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0
    assert result["common_score_with_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_storewavg_maps_fine_rows_to_global_rotation_ids(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_storeWavg"
    class0 = np.array([9.0, 0.0], dtype=np.float64)
    class1 = np.array([0.0, 2.0, 0.0, 4.0, 1.0, 0.0], dtype=np.float64)
    _write_flat_real(relion_dir / f"{prefix}_sorted_weights.bin", np.concatenate([class0, class1]))
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 1)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_nr_classes.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_iclass_min.bin", 0)
    _write_flat_int(relion_dir / "pass1_class0_fine_class_entries.bin", [1, 3])
    _write_flat_int(relion_dir / "pass1_class0_fine_class_idx.bin", [0, 1])
    _write_flat_int(relion_dir / "pass1_class0_fine_iorientclasses.bin", [0, 11, 12, 13])
    _write_flat_int(relion_dir / "pass1_class0_fine_iover_rots.bin", [0, 1, 2, 3])
    _write_flat_real(relion_dir / "pass1_pdf_orientation.bin", np.ones(10, dtype=np.float64))

    scores_pre = np.full((3, 2), -np.inf, dtype=np.float64)
    scores_pre[0, 1] = np.log(2.0)
    scores_pre[1, 1] = np.log(4.0)
    scores_pre[2, 0] = np.log(1.0)
    recovar_weights = np.zeros_like(scores_pre)
    recovar_weights[0, 1] = 2.0 / 7.0
    recovar_weights[1, 1] = 4.0 / 7.0
    recovar_weights[2, 0] = 1.0 / 7.0
    recovar_npz = tmp_path / "recovar_class1_pass2.npz"
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        class_index=np.int64(1),
        current_size=np.int64(56),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=np.zeros((3, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([5, 10, 15], dtype=np.int64),
        parent_map=np.array([1, 2, 3], dtype=np.int32),
        candidate_mask=np.isfinite(scores_pre),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=recovar_weights,
        rotation_log_prior=np.zeros(3, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
    )

    assert result["relion_rotation_key_mode"] == "fine_iorientclasses_mod_pdf_orientation_times_iover_rots"
    assert result["relion_selected_field"] == f"acc_storewavg_positive_weights:{prefix}:class1"
    assert result["relion_candidate_count"] == 3
    assert result["recovar_candidate_count"] == 3
    assert result["common_candidate_count"] == 3
    assert result["relion_top_key"] == [10, 1]
    assert result["recovar_top_key"] == [10, 1]
    assert result["common_prob_l1_after_common_renorm"] == 0.0
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0
    assert result["common_score_with_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_storewavg_zero_orientation_num_slices_before_matrix_match(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_storeWavg"
    _write_flat_real(relion_dir / f"{prefix}_sorted_weights.bin", np.ones(6, dtype=np.float64))
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 0)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_nr_classes.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_iclass_min.bin", 0)
    _write_flat_int(relion_dir / "pass1_class1_fine_class_entries.bin", [1, 2])

    class1_rots = np.stack(
        [
            np.eye(3),
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ],
        axis=0,
    )
    _write_flat_real(relion_dir / "pass1_class1_fine_eulers.bin", class1_rots.reshape(-1))

    recovar_npz = tmp_path / "recovar_class1_pass2.npz"
    scores_pre = np.zeros((2, 2), dtype=np.float64)
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        class_index=np.int64(1),
        current_size=np.int64(56),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=class1_rots.astype(np.float32),
        oversampled_rot_indices=np.array([0, 1], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        candidate_mask=np.ones((2, 2), dtype=bool),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=np.full((2, 2), 0.25, dtype=np.float64),
        rotation_log_prior=np.zeros(2, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
        match_mode="matrix",
    )

    assert result["match_mode"] == "matrix"
    assert result["relion_selected_field"] == f"acc_storewavg_positive_weights:{prefix}:class1"
    assert result["relion_candidate_count"] == 4
    assert result["common_candidate_count"] == 4
    assert result["match_details"]["rotation_matrix_unique_relion_rows"] == 2


def test_compare_relion_storewavg_zero_class_entries_are_empty(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_storeWavg"
    _write_flat_real(relion_dir / f"{prefix}_sorted_weights.bin", np.array([0.0, 10.0, 0.0, 3.0]))
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 0)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_nr_classes.bin", 2)
    _write_scalar(relion_dir / f"{prefix}_iclass_min.bin", 0)
    _write_scalar(relion_dir / f"{prefix}_significant_weight.bin", 10.0)
    _write_flat_int(relion_dir / "pass1_class1_fine_class_entries.bin", [0, 2])

    recovar_npz = tmp_path / "recovar_class0_pass2.npz"
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        class_index=np.int64(0),
        current_size=np.int64(56),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=np.zeros((0, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([], dtype=np.int64),
        parent_map=np.array([], dtype=np.int32),
        candidate_mask=np.zeros((0, 2), dtype=bool),
        scores_with_prior=np.zeros((0, 2), dtype=np.float64),
        scores_pre_prior=np.zeros((0, 2), dtype=np.float64),
        probs=np.zeros((0, 2), dtype=np.float64),
        rotation_log_prior=np.zeros(0, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
        reconstruction_mask=np.zeros((0, 2), dtype=bool),
        reconstruction_probs=np.zeros((0, 2), dtype=np.float64),
        reconstruction_n_significant=np.int64(0),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        reconstruction_only=True,
        relion_acc_table_prefix=prefix,
    )

    assert result["match_mode"] == "global"
    assert result["relion_selected_field"] == f"acc_storewavg_significant_weights:{prefix}:class0"
    assert result["relion_candidate_count"] == 0
    assert result["recovar_candidate_count"] == 0
    assert result["common_candidate_count"] == 0


def test_compare_relion_recovar_estep_dump_maps_relion_grid_rotation_order(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_pass1_class0_pass1"
    # npsi=2, npixels=3. RELION rot 3 means pixel=1, psi=1; RECOVAR
    # psi-major order maps that to rot 4.
    relion_scores = np.array([0.0, 1.0, 2.0, 6.0, 4.0, 5.0], dtype=np.float64)
    _write_flat_real(relion_dir / f"{prefix}_diff2_weights.bin", -relion_scores)
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 6)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 1)

    recovar_scores = np.zeros((6, 1), dtype=np.float64)
    for rel_rot, score in enumerate(relion_scores):
        pixel = rel_rot // 2
        psi = rel_rot % 2
        recovar_scores[psi * 3 + pixel, 0] = score
    weights = np.zeros_like(recovar_scores)
    weights[4, 0] = 1.0
    recovar_npz = tmp_path / "recovar_significance.npz"
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        current_size=np.int64(56),
        n_classes=np.int64(1),
        n_rot=np.int64(6),
        n_trans=np.int64(1),
        weights_full=weights.reshape(-1),
        weights_per_class=weights[None, :, :],
        scores_pre_prior_per_class=recovar_scores[None, :, :],
        scores_with_prior_per_class=recovar_scores[None, :, :],
        rotations=np.zeros((6, 3, 3), dtype=np.float32),
        translations=np.zeros((1, 2), dtype=np.float32),
        rotation_log_prior=np.zeros(6, dtype=np.float64),
        translation_log_prior=np.zeros(1, dtype=np.float64),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
        match_mode="relion_grid",
        relion_n_psi=2,
    )

    assert result["match_mode"] == "relion_grid"
    assert result["match_details"]["rotation_index_mapping"] == "relion_pixel_major_to_recovar_psi_major"
    assert result["common_candidate_count"] == 6
    assert result["relion_top_key"] == [4, 0]
    assert result["recovar_top_key"] == [4, 0]
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_grid_mapping_uses_acc_rotation_count_for_local_pass2(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_pass1_class0_pass1"
    # npsi=2, npixels=3. RELION raw rot 3 maps to RECOVAR global rot 4.
    relion_scores = np.array([0.0, 1.0, 2.0, 6.0, 4.0, 5.0], dtype=np.float64)
    _write_flat_real(relion_dir / f"{prefix}_diff2_weights.bin", -relion_scores)
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 6)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 1)

    recovar_npz = tmp_path / "recovar_pass2_local.npz"
    scores_pre = np.array([[6.0], [5.0]], dtype=np.float64)
    probs = np.array([[1.0], [0.0]], dtype=np.float64)
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        current_size=np.int64(56),
        n_fine_trans=np.int64(1),
        fine_translations=np.zeros((1, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([4, 5], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        candidate_mask=np.ones((2, 1), dtype=bool),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.zeros(2, dtype=np.float64),
        translation_log_prior=np.zeros(1, dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((2, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
        match_mode="relion_grid",
        relion_n_psi=2,
    )

    assert result["match_mode"] == "relion_grid"
    assert result["match_details"]["relion_rotation_count"] == 6
    assert result["common_candidate_count"] == 2
    assert result["relion_raw_top_key"] == [3, 0]
    assert result["relion_matched_top_key"] == [4, 0]
    assert result["relion_top_key"] == [4, 0]
    assert result["recovar_top_key"] == [4, 0]
    assert result["common_score_pre_prior_centered_diff"]["max_abs"] == 0.0


def test_compare_relion_grid_parent_mapping_collapses_fine_pass2_children(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    prefix = "img0_part7778_pass1_class0_pass1"
    # npsi=2, npixels=3. RELION raw rot 3 maps to RECOVAR coarse rot 4.
    relion_scores = np.zeros((6, 4), dtype=np.float64)
    relion_scores[3, 2] = 9.0
    _write_flat_real(relion_dir / f"{prefix}_diff2_weights.bin", -relion_scores.reshape(-1))
    _write_scalar(relion_dir / f"{prefix}_orientation_num.bin", 6)
    _write_scalar(relion_dir / f"{prefix}_translation_num.bin", 4)

    recovar_npz = tmp_path / "recovar_pass2_local.npz"
    scores_pre = np.full((1, 10), -np.inf, dtype=np.float64)
    scores_pre[0, 9] = 8.0
    scores_pre[0, 8] = 9.0
    probs = np.zeros_like(scores_pre)
    probs[0, 8] = 0.4
    probs[0, 9] = 0.6
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        current_size=np.int64(56),
        n_fine_trans=np.int64(10),
        fine_translations=np.zeros((10, 2), dtype=np.float32),
        rotations=np.zeros((1, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([35], dtype=np.int64),
        parent_map=np.array([0], dtype=np.int32),
        candidate_mask=np.isfinite(scores_pre),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.zeros(1, dtype=np.float64),
        translation_log_prior=np.zeros(10, dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((1, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        relion_acc_table_prefix=prefix,
        match_mode="relion_grid_parent",
        relion_n_psi=2,
        recovar_parent_rot_divisor=8,
        recovar_parent_trans_divisor=4,
    )

    assert result["match_mode"] == "relion_grid_parent"
    assert result["match_details"]["recovar_key_mapping"] == "fine_pass2_candidate_to_coarse_parent"
    assert result["match_details"]["recovar_parent_rotation_mapping"] == "oversampled_rot_indices_divisor"
    assert result["common_candidate_count"] == 1
    assert result["recovar_duplicate_keys_collapsed"] == 1
    assert result["common_recovar_prob_mass"] == 1.0
    assert result["relion_raw_top_key"] == [3, 2]
    assert result["relion_matched_top_key"] == [4, 2]
    assert result["relion_top_key"] == [4, 2]
    assert result["recovar_top_key"] == [4, 2]


def test_compare_relion_grid_parent_mapping_uses_acc_parent_keys(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass1_acc_rot_id.bin", [100, 101, 108])
    _write_flat_int(relion_dir / "pass1_acc_rot_idx.bin", [8, 9, 16])
    _write_flat_int(relion_dir / "pass1_acc_trans_idx.bin", [8, 9, 12])
    _write_flat_int(relion_dir / "pass1_candidate_coarse_trans_idx.bin", [2, 2, 3])
    _write_flat_real(relion_dir / "pass1_candidate_weight_normalized.bin", [0.1, 0.8, 0.1])
    _write_flat_real(relion_dir / "pass1_exp_Mweight_raw_preprior.bin", [-1.0, -9.0, -2.0])

    recovar_npz = tmp_path / "recovar_pass2_local.npz"
    scores_pre = np.full((3, 16), -np.inf, dtype=np.float64)
    scores_pre[1, 8] = 8.0
    scores_pre[1, 9] = 9.0
    scores_pre[2, 12] = 2.0
    probs = np.zeros_like(scores_pre)
    probs[1, 8] = 0.1
    probs[1, 9] = 0.8
    probs[2, 12] = 0.1
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        current_size=np.int64(56),
        n_fine_trans=np.int64(16),
        fine_translations=np.zeros((16, 2), dtype=np.float32),
        rotations=np.zeros((3, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([200, 201, 202], dtype=np.int64),
        parent_map=np.array([0, 1, 2], dtype=np.int32),
        candidate_mask=np.isfinite(scores_pre),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.zeros(3, dtype=np.float64),
        translation_log_prior=np.zeros(16, dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((1, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(
        relion_dir,
        recovar_npz,
        match_mode="relion_grid_parent",
        relion_n_psi=2,
        relion_parent_rot_divisor=8,
        recovar_parent_trans_divisor=4,
    )

    assert result["match_mode"] == "relion_grid_parent"
    assert result["match_details"]["relion_parent_rotation_mapping"] == "acc_rot_idx_divisor"
    assert result["match_details"]["relion_parent_translation_mapping"] == "candidate_coarse_trans_idx"
    assert result["match_details"]["recovar_parent_translation_mapping"] == "fine_trans_indices_divisor"
    assert result["common_candidate_count"] == 2
    assert result["relion_duplicate_keys_collapsed"] == 1
    assert result["recovar_duplicate_keys_collapsed"] == 1
    assert result["relion_top_key"] == [1, 2]
    assert result["recovar_top_key"] == [1, 2]


def test_compare_relion_generic_candidate_table_filters_to_recovar_class(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    _write_flat_int(relion_dir / "pass1_acc_rot_id.bin", [0, 0, 1])
    _write_flat_int(relion_dir / "pass1_acc_rot_idx.bin", [0, 0, 1])
    _write_flat_int(relion_dir / "pass1_acc_trans_idx.bin", [0, 1, 0])
    _write_flat_int(relion_dir / "pass1_candidate_class_idx.bin", [0, 1, 1])
    _write_flat_real(relion_dir / "pass1_candidate_weight_normalized.bin", [0.9, 0.3, 0.7])
    _write_flat_real(relion_dir / "pass1_exp_Mweight_raw_preprior.bin", [-9.0, -3.0, -7.0])

    recovar_npz = tmp_path / "recovar_pass2_class1.npz"
    scores_pre = np.full((2, 2), -np.inf, dtype=np.float64)
    scores_pre[0, 1] = 3.0
    scores_pre[1, 0] = 7.0
    probs = np.zeros_like(scores_pre)
    probs[0, 1] = 0.3
    probs[1, 0] = 0.7
    np.savez_compressed(
        recovar_npz,
        original_index=np.int64(7),
        local_index=np.int64(7),
        class_index=np.int64(1),
        current_size=np.int64(56),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        oversampled_rot_indices=np.array([0, 1], dtype=np.int64),
        parent_map=np.array([0, 1], dtype=np.int32),
        candidate_mask=np.isfinite(scores_pre),
        scores_with_prior=scores_pre,
        scores_pre_prior=scores_pre,
        probs=probs,
        rotation_log_prior=np.zeros(2, dtype=np.float64),
        translation_log_prior=np.zeros(2, dtype=np.float64),
        shifted_corrected=np.empty((0,), dtype=np.complex64),
        ctf2_over_nv_score=np.ones(1, dtype=np.float64),
        proj_half=np.empty((1, 1), dtype=np.complex64),
        half_weights=np.ones(1, dtype=np.float64),
        window_indices=np.array([0], dtype=np.int32),
    )

    result = compare_dumps(relion_dir, recovar_npz, match_mode="global")

    assert result["relion_selected_field"] == "all_candidates:class1"
    assert result["relion_candidate_count"] == 2
    assert result["common_candidate_count"] == 2
    assert result["relion_top_key"] == [1, 0]
    assert result["recovar_top_key"] == [1, 0]
