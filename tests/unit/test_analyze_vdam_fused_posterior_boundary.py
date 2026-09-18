import numpy as np

from scripts.analyze_vdam_fused_posterior_boundary import (
    compare_posteriors,
    compare_score_spacing,
)


def test_compare_fused_posterior_maps_rotation_rows_and_support():
    identity = np.eye(3, dtype=np.float32)
    quarter_turn = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    posterior = np.zeros((1, 2, 3), dtype=np.float32)
    posterior[0, 1, 2] = 0.75
    posterior[0, 0, 1] = 0.25
    reconstruction_mask = np.zeros_like(posterior, dtype=bool)
    reconstruction_mask[0, 1, 2] = True

    report = compare_posteriors(
        native_rotation_ids=np.array([0, 1], dtype=np.int32),
        native_translation_ids=np.array([2, 1], dtype=np.int32),
        native_rotation_matrices=np.stack([identity, quarter_turn]),
        native_unnormalized_weights=np.array([3.0, 1.0]),
        native_sum_weight=4.0,
        native_reconstruction_mask=np.array([True, False]),
        live={
            "local_rotation_matrices": np.stack([quarter_turn, identity]),
            "posterior": posterior,
            "reconstruction_sample_mask": reconstruction_mask,
        },
    )

    assert report["argmax_equal"]
    assert report["native_best_mapped_key"] == [1, 2]
    assert report["recovar_best_key"] == [1, 2]
    assert report["probability_l1"] == 0.0
    assert report["probability_relative_l2"] == 0.0
    assert report["reconstruction_mask_on_native_equal"]


def test_compare_fused_posterior_rejects_unmatched_rotation_support():
    identity = np.eye(3, dtype=np.float32)
    half_turn = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    report = compare_posteriors(
        native_rotation_ids=np.array([0], dtype=np.int32),
        native_translation_ids=np.array([0], dtype=np.int32),
        native_rotation_matrices=identity[None],
        native_unnormalized_weights=np.array([1.0]),
        native_sum_weight=1.0,
        native_reconstruction_mask=np.array([True]),
        live={
            "local_rotation_matrices": half_turn[None],
            "posterior": np.ones((1, 1, 1), dtype=np.float32),
            "reconstruction_sample_mask": np.ones((1, 1, 1), dtype=bool),
        },
    )

    assert report["status"] == "rotation_support_mismatch"
    assert report["native_candidate_unmatched_count"] == 1
    assert report["native_rotation_unmatched_ids"] == [0]


def test_compare_score_spacing_reports_nonfinite_mismatch_without_nan():
    identity = np.eye(3, dtype=np.float32)
    report = compare_score_spacing(
        native_rotation_ids=np.array([0, 0], dtype=np.int32),
        native_translation_ids=np.array([0, 1], dtype=np.int32),
        native_rotation_matrices=identity[None],
        native_log_weights=np.array([1.0, 0.0]),
        native_combined_log_prior=np.zeros(2),
        live={
            "local_rotation_matrices": identity[None],
            "pass2_scores_total": np.array([[[1.0, -np.inf]]], dtype=np.float32),
            "rotation_log_prior": np.zeros((1, 1), dtype=np.float32),
            "translation_log_prior": np.zeros((1, 2), dtype=np.float32),
        },
    )

    metric = report["total_log_weight_centered"]
    assert metric["status"] == "nonfinite_mismatch"
    assert metric["finite_pair_count"] == 1
    assert metric["nonfinite_mismatch_count"] == 1
    assert np.isfinite(metric["candidate_minus_native_common_offset"])


def test_compare_score_spacing_splits_prior_from_preprior_winner_flip():
    identity = np.eye(3, dtype=np.float32)
    quarter_turn = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    total_scores = np.full((1, 2, 2), -np.inf, dtype=np.float32)
    # Native rotation row 0 maps to RECOVAR row 1 and native row 1 maps to row 0.
    total_scores[0, 1, 1] = 15.0
    total_scores[0, 0, 0] = 15.1
    report = compare_score_spacing(
        native_rotation_ids=np.array([0, 1], dtype=np.int32),
        native_translation_ids=np.array([1, 0], dtype=np.int32),
        native_rotation_matrices=np.stack([identity, quarter_turn]),
        native_log_weights=np.array([10.0, 9.9]),
        native_combined_log_prior=np.array([0.3, -0.2]),
        live={
            "local_rotation_matrices": np.stack([quarter_turn, identity]),
            "pass2_scores_total": total_scores,
            "rotation_log_prior": np.array([[-0.2, 0.3]], dtype=np.float32),
            "translation_log_prior": np.zeros((1, 2), dtype=np.float32),
        },
    )

    assert report["status"] == "captured"
    assert report["combined_log_prior_centered"]["max_abs"] < 1e-8
    assert np.isclose(report["total_log_weight_centered"]["max_abs"], 0.1)
    assert np.isclose(report["preprior_score_centered"]["max_abs"], 0.1)
    assert report["winner_pair"]["native_winner_mapped_key"] == [1, 1]
    assert report["winner_pair"]["candidate_winner_mapped_key"] == [0, 0]
    assert report["winner_pair"]["native_margin_native_minus_candidate"] > 0.0
    assert report["winner_pair"]["candidate_margin_native_minus_candidate"] < 0.0


def test_compare_score_spacing_reports_missing_optional_capture():
    report = compare_score_spacing(
        native_rotation_ids=np.array([0], dtype=np.int32),
        native_translation_ids=np.array([0], dtype=np.int32),
        native_rotation_matrices=np.eye(3, dtype=np.float32)[None],
        native_log_weights=np.array([1.0]),
        native_combined_log_prior=np.array([0.0]),
        live={"local_rotation_matrices": np.eye(3, dtype=np.float32)[None]},
    )

    assert report["status"] == "not_captured"
    assert "pass2_scores_total" in report["missing_fields"]
