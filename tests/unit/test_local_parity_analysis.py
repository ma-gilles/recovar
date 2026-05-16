"""Unit tests for local parity dump analysis helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "recovar" / "utils" / "local_parity_analysis.py"
SPEC = importlib.util.spec_from_file_location("local_parity_analysis", MODULE_PATH)
parity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = parity
SPEC.loader.exec_module(parity)


pytestmark = pytest.mark.unit


def test_summarize_relion_operands_checks_normalized_pmax():
    relion = {
        "exp_Mweight_posterior": np.array([0.0, 2.0, 1.0, 0.0], dtype=np.float64),
        "exp_sum_weight": np.array(3.0, dtype=np.float64),
        "Pmax": np.array(2.0 / 3.0, dtype=np.float64),
        "header_nr_dir": np.array(2, dtype=np.int32),
        "header_nr_psi": np.array(2, dtype=np.int32),
        "header_nr_trans": np.array(1, dtype=np.int32),
        "header_nr_oversampled_rot": np.array(1, dtype=np.int32),
        "header_nr_oversampled_trans": np.array(1, dtype=np.int32),
        "header_current_size": np.array(16, dtype=np.int32),
        "header_ori_size": np.array(32, dtype=np.int32),
        "candidate_in_denominator_set": np.array([1, 1, 1, 1], dtype=np.int32),
        "candidate_in_fine_threshold_set": np.array([0, 1, 1, 0], dtype=np.int32),
        "candidate_in_reconstruction_set": np.array([0, 1, 1, 0], dtype=np.int32),
        "candidate_weight_normalized": np.array([0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=np.float64),
    }

    summary = parity.summarize_relion_operands(relion)

    assert summary["stored_pmax"] == pytest.approx(2.0 / 3.0)
    assert summary["pmax_from_raw"] == pytest.approx(2.0 / 3.0)
    assert summary["pmax_semantics_gap"] == pytest.approx(0.0)
    assert summary["support_size"] == 2
    assert summary["top2_mass"] == pytest.approx(1.0)
    assert summary["denominator_count"] == 4
    assert summary["fine_threshold_count"] == 2


def test_summarize_recovar_score_dump_reconstructs_pmax_and_terms():
    raw_scores = np.array(
        [[[0.0, -1.0], [-2.0, -3.0], [-4.0, -5.0]]],
        dtype=np.float32,
    )
    rotation_prior = np.array([[0.0, -0.5, -1e30]], dtype=np.float32)
    translation_prior = np.array([[0.0, -0.25]], dtype=np.float32)
    total_scores = raw_scores + rotation_prior[:, :, None] + translation_prior[:, None, :]
    total_scores[:, 2, :] = -np.inf
    cross = np.array(
        [[[-0.0, 2.0], [4.0, 6.0], [8.0, 10.0]]],
        dtype=np.float32,
    )
    norms = np.array(
        [[[0.0], [0.0], [0.0]]],
        dtype=np.float32,
    )
    probs = np.exp(total_scores[0] - np.max(total_scores[0][np.isfinite(total_scores[0])]))
    probs = probs / np.sum(probs[np.isfinite(probs)])
    recovar = {
        "selected_global_image_indices": np.array([17], dtype=np.int64),
        "pass1_scores_raw": raw_scores.copy(),
        "pass2_scores_raw": raw_scores,
        "pass2_scores_total": total_scores,
        "pass2_cross_term": cross,
        "pass2_norm_term": norms,
        "rotation_log_prior": rotation_prior,
        "translation_log_prior": translation_prior,
        "rotation_candidate_mask": np.array([[True, True, False]], dtype=bool),
        "max_posterior": np.array([np.max(probs)], dtype=np.float32),
        "best_score": np.array([np.max(total_scores[np.isfinite(total_scores)])], dtype=np.float32),
        "log_Z": np.array([np.log(np.sum(np.exp(total_scores[np.isfinite(total_scores)])))], dtype=np.float32),
    }

    summary = parity.summarize_recovar_score_dump(recovar, image_position=0)

    assert summary["selected_global_image_index"] == 17
    assert summary["saved_pmax"] == pytest.approx(summary["recomputed_pmax"], rel=1e-6)
    assert summary["pass_raw_max_abs_diff"] == pytest.approx(0.0)
    assert summary["cross_norm_max_abs_diff"] == pytest.approx(0.0)
    assert summary["support_rotations"] == 2
    assert summary["full"]["pmax"] > summary["raw_only"]["pmax"]


def test_masked_score_mass_summary_distinguishes_denominator_and_support():
    scores = np.array([[0.0, -1.0], [-3.0, -4.0]], dtype=np.float64)
    denom = np.array([[True, True], [True, False]])
    support = np.array([[True, False], [True, False]])

    summary = parity.masked_score_mass_summary(scores, denom, support)

    assert summary["normalization_support_size"] == 3
    assert summary["support_size"] == 2
    assert 0.0 < summary["support_mass"] < 1.0
    assert summary["renormalized"]["pmax"] > summary["normalized_pmax"]


def test_summarize_recovar_mask_ladder_reports_full_to_subset_mass():
    relion = {
        "exp_Mweight_posterior": np.array([3.0, 2.0], dtype=np.float64),
        "exp_sum_weight": np.array(5.0, dtype=np.float64),
        "Pmax": np.array(3.0 / 5.0, dtype=np.float64),
        "pointer_dir_nonzeroprior": np.array([10, 11], dtype=np.int32),
        "pointer_psi_nonzeroprior": np.array([20, 21], dtype=np.int32),
        "acc_rot_id": np.array([0, 3], dtype=np.int32),
        "acc_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_coarse_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_in_denominator_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_fine_threshold_set": np.array([1, 0], dtype=np.int32),
        "candidate_in_reconstruction_set": np.array([1, 0], dtype=np.int32),
        "translations_x": np.array([0.0, 1.5], dtype=np.float64),
        "translations_y": np.array([0.0, -0.5], dtype=np.float64),
    }
    recovar = {
        "selected_global_image_indices": np.array([4], dtype=np.int64),
        "local_rotation_pixel_indices": np.array([10, 10, 11, 11], dtype=np.int64),
        "local_rotation_psi_indices": np.array([20, 21, 20, 21], dtype=np.int64),
        "grid_n_psi": np.array(2, dtype=np.int32),
        "rotation_candidate_mask": np.array([[True, True, True, True]], dtype=bool),
        "pass2_scores_total": np.array([[[2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "pass2_scores_raw": np.array([[[2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "translations": np.array([[0.0, 0.0], [1.5, -0.5]], dtype=np.float32),
    }

    mapping = parity.build_relion_recovar_candidate_mapping(relion, recovar, image_position=0)
    ladder = parity.summarize_recovar_mask_ladder(recovar, mapping, image_position=0)

    assert ladder["denominator_under_full"]["support_mass"] < 1.0
    assert ladder["threshold_under_full"]["support_mass"] <= ladder["denominator_under_full"]["support_mass"]


def test_build_relion_recovar_candidate_mapping_maps_factorized_subset():
    relion = {
        "exp_Mweight_posterior": np.array([3.0, 2.0], dtype=np.float64),
        "exp_sum_weight": np.array(5.0, dtype=np.float64),
        "Pmax": np.array(3.0 / 5.0, dtype=np.float64),
        "pointer_dir_nonzeroprior": np.array([10, 11], dtype=np.int32),
        "pointer_psi_nonzeroprior": np.array([20, 21], dtype=np.int32),
        "acc_rot_id": np.array([0, 3], dtype=np.int32),
        "acc_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_coarse_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_in_denominator_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_fine_threshold_set": np.array([1, 0], dtype=np.int32),
        "candidate_in_reconstruction_set": np.array([1, 0], dtype=np.int32),
        "translations_x": np.array([0.0, 1.5], dtype=np.float64),
        "translations_y": np.array([0.0, -0.5], dtype=np.float64),
    }
    recovar = {
        "selected_global_image_indices": np.array([4], dtype=np.int64),
        "local_rotation_pixel_indices": np.array([10, 10, 11, 11], dtype=np.int64),
        "local_rotation_psi_indices": np.array([20, 21, 20, 21], dtype=np.int64),
        "grid_n_psi": np.array(2, dtype=np.int32),
        "rotation_candidate_mask": np.array([[True, True, True, True]], dtype=bool),
        "pass2_scores_total": np.zeros((1, 4, 2), dtype=np.float32),
        "translations": np.array([[0.0, 0.0], [1.5, -0.5]], dtype=np.float32),
    }

    mapping = parity.build_relion_recovar_candidate_mapping(relion, recovar, image_position=0)

    assert mapping.pixel_support_equal
    assert mapping.psi_support_equal
    np.testing.assert_array_equal(mapping.recovar_rot_slot, np.array([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(mapping.recovar_trans_idx, np.array([0, 1], dtype=np.int64))
    assert mapping.denominator_mask[0, 0]
    assert mapping.denominator_mask[3, 1]
    assert mapping.fine_threshold_mask[0, 0]
    assert not mapping.fine_threshold_mask[3, 1]


def test_build_relion_recovar_candidate_mapping_maps_translation_order_by_coordinates():
    relion = {
        "exp_Mweight_posterior": np.array([3.0, 2.0], dtype=np.float64),
        "exp_sum_weight": np.array(5.0, dtype=np.float64),
        "Pmax": np.array(3.0 / 5.0, dtype=np.float64),
        "pointer_dir_nonzeroprior": np.array([10, 11], dtype=np.int32),
        "pointer_psi_nonzeroprior": np.array([20, 21], dtype=np.int32),
        "acc_rot_id": np.array([0, 3], dtype=np.int32),
        "acc_trans_idx": np.array([1, 2], dtype=np.int32),
        "candidate_coarse_trans_idx": np.array([1, 2], dtype=np.int32),
        "candidate_in_denominator_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_fine_threshold_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_reconstruction_set": np.array([1, 1], dtype=np.int32),
        # RELION coarse grid order: x-major, y-minor.
        "translations_x": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
        "translations_y": np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float64),
    }
    recovar = {
        "selected_global_image_indices": np.array([4], dtype=np.int64),
        "local_rotation_pixel_indices": np.array([10, 10, 11, 11], dtype=np.int64),
        "local_rotation_psi_indices": np.array([20, 21, 20, 21], dtype=np.int64),
        "grid_n_psi": np.array(2, dtype=np.int32),
        "rotation_candidate_mask": np.array([[True, True, True, True]], dtype=bool),
        "pass2_scores_total": np.zeros((1, 4, 4), dtype=np.float32),
        # RECOVAR grid order: x-fast, y-major.
        "translations": np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    mapping = parity.build_relion_recovar_candidate_mapping(relion, recovar, image_position=0)

    np.testing.assert_array_equal(mapping.relion_trans_idx, np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(mapping.recovar_trans_idx, np.array([2, 1], dtype=np.int64))
    assert mapping.denominator_mask[0, 2]
    assert mapping.denominator_mask[3, 1]


def test_build_relion_recovar_candidate_mapping_prefers_actual_candidate_translations():
    relion = {
        "exp_Mweight_posterior": np.array([3.0, 2.0], dtype=np.float64),
        "exp_sum_weight": np.array(5.0, dtype=np.float64),
        "Pmax": np.array(3.0 / 5.0, dtype=np.float64),
        "pointer_dir_nonzeroprior": np.array([10, 11], dtype=np.int32),
        "pointer_psi_nonzeroprior": np.array([20, 21], dtype=np.int32),
        "acc_rot_id": np.array([0, 3], dtype=np.int32),
        "acc_trans_idx": np.array([0, 0], dtype=np.int32),
        "candidate_coarse_trans_idx": np.array([0, 0], dtype=np.int32),
        "candidate_in_denominator_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_fine_threshold_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_reconstruction_set": np.array([1, 1], dtype=np.int32),
        "translations_x": np.array([0.0, 1.5], dtype=np.float64),
        "translations_y": np.array([0.0, -0.5], dtype=np.float64),
        "candidate_translation_x": np.array([0.0, 1.5], dtype=np.float64),
        "candidate_translation_y": np.array([0.0, -0.5], dtype=np.float64),
    }
    recovar = {
        "selected_global_image_indices": np.array([4], dtype=np.int64),
        "local_rotation_pixel_indices": np.array([10, 10, 11, 11], dtype=np.int64),
        "local_rotation_psi_indices": np.array([20, 21, 20, 21], dtype=np.int64),
        "grid_n_psi": np.array(2, dtype=np.int32),
        "rotation_candidate_mask": np.array([[True, True, True, True]], dtype=bool),
        "pass2_scores_total": np.zeros((1, 4, 2), dtype=np.float32),
        "translations": np.array([[0.0, 0.0], [1.5, -0.5]], dtype=np.float32),
    }

    mapping = parity.build_relion_recovar_candidate_mapping(relion, recovar, image_position=0)

    np.testing.assert_array_equal(mapping.relion_trans_idx, np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(mapping.recovar_trans_idx, np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(mapping.relion_translation_x, np.array([0.0, 1.5], dtype=np.float64))
    np.testing.assert_allclose(mapping.relion_translation_y, np.array([0.0, -0.5], dtype=np.float64))
    assert mapping.denominator_mask[0, 0]
    assert mapping.denominator_mask[3, 1]


def test_build_relion_recovar_candidate_mapping_uses_candidate_translation_pixel_units_directly():
    relion = {
        "exp_Mweight_posterior": np.array([3.0], dtype=np.float64),
        "exp_sum_weight": np.array(3.0, dtype=np.float64),
        "Pmax": np.array(1.0, dtype=np.float64),
        "pointer_dir_nonzeroprior": np.array([10], dtype=np.int32),
        "pointer_psi_nonzeroprior": np.array([20], dtype=np.int32),
        "acc_rot_id": np.array([0], dtype=np.int32),
        "acc_trans_idx": np.array([5], dtype=np.int32),
        "candidate_coarse_trans_idx": np.array([5], dtype=np.int32),
        "candidate_in_denominator_set": np.array([1], dtype=np.int32),
        "candidate_in_fine_threshold_set": np.array([1], dtype=np.int32),
        "candidate_in_reconstruction_set": np.array([1], dtype=np.int32),
        # Coarse translation metadata is in Angstrom-like units.
        "translations_x": np.array([0.0, 4.25], dtype=np.float64),
        "translations_y": np.array([0.0, 4.25], dtype=np.float64),
        # Actual candidate translation dump is already in pixel units.
        "candidate_translation_x": np.array([0.25], dtype=np.float64),
        "candidate_translation_y": np.array([-0.25], dtype=np.float64),
    }
    recovar = {
        "selected_global_image_indices": np.array([4], dtype=np.int64),
        "local_rotation_pixel_indices": np.array([10], dtype=np.int64),
        "local_rotation_psi_indices": np.array([20], dtype=np.int64),
        "grid_n_psi": np.array(1, dtype=np.int32),
        "rotation_candidate_mask": np.array([[True]], dtype=bool),
        "pass2_scores_total": np.zeros((1, 1, 2), dtype=np.float32),
        "translations": np.array([[0.0, 0.0], [0.25, -0.25]], dtype=np.float32),
    }

    mapping = parity.build_relion_recovar_candidate_mapping(relion, recovar, image_position=0)

    np.testing.assert_array_equal(mapping.recovar_trans_idx, np.array([1], dtype=np.int64))
    np.testing.assert_allclose(mapping.relion_translation_x, np.array([0.25], dtype=np.float64))
    np.testing.assert_allclose(mapping.relion_translation_y, np.array([-0.25], dtype=np.float64))
    assert mapping.denominator_mask[0, 1]


def test_align_mask_to_score_shape_pads_trailing_rotation_rows():
    mask = np.array([[True, False], [False, True]], dtype=bool)

    aligned = parity._align_mask_to_score_shape(mask, (4, 2), name="demo_mask")

    expected = np.array(
        [
            [True, False],
            [False, True],
            [False, False],
            [False, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(aligned, expected)


def test_align_mask_to_score_shape_trims_all_false_trailing_rotation_rows():
    mask = np.array(
        [
            [True, False],
            [False, True],
            [False, False],
            [False, False],
        ],
        dtype=bool,
    )

    aligned = parity._align_mask_to_score_shape(mask, (2, 2), name="demo_mask")

    expected = np.array([[True, False], [False, True]], dtype=bool)
    np.testing.assert_array_equal(aligned, expected)


def test_mask_ladder_accepts_mapping_from_unpadded_support_dump():
    relion = {
        "exp_Mweight_posterior": np.array([3.0, 2.0], dtype=np.float64),
        "exp_sum_weight": np.array(5.0, dtype=np.float64),
        "Pmax": np.array(3.0 / 5.0, dtype=np.float64),
        "pointer_dir_nonzeroprior": np.array([10, 11], dtype=np.int32),
        "pointer_psi_nonzeroprior": np.array([20, 21], dtype=np.int32),
        "acc_rot_id": np.array([0, 3], dtype=np.int32),
        "acc_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_coarse_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_in_denominator_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_fine_threshold_set": np.array([1, 0], dtype=np.int32),
        "candidate_in_reconstruction_set": np.array([1, 0], dtype=np.int32),
        "translations_x": np.array([0.0, 1.5], dtype=np.float64),
        "translations_y": np.array([0.0, -0.5], dtype=np.float64),
    }
    recovar_support = {
        "selected_global_image_indices": np.array([4], dtype=np.int64),
        "local_rotation_pixel_indices": np.array([10, 10, 11, 11], dtype=np.int64),
        "local_rotation_psi_indices": np.array([20, 21, 20, 21], dtype=np.int64),
        "grid_n_psi": np.array(2, dtype=np.int32),
        "rotation_candidate_mask": np.array([[True, True, True, True]], dtype=bool),
        "pass2_scores_total": np.array([[[2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "pass2_scores_raw": np.array([[[2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "translations": np.array([[0.0, 0.0], [1.5, -0.5]], dtype=np.float32),
    }
    recovar_padded = {
        **recovar_support,
        "pass2_scores_total": np.pad(
            recovar_support["pass2_scores_total"],
            ((0, 0), (0, 2), (0, 0)),
            constant_values=-np.inf,
        ),
        "pass2_scores_raw": np.pad(
            recovar_support["pass2_scores_raw"],
            ((0, 0), (0, 2), (0, 0)),
            constant_values=-np.inf,
        ),
    }

    mapping = parity.build_relion_recovar_candidate_mapping(relion, recovar_support, image_position=0)
    ladder = parity.summarize_recovar_mask_ladder(recovar_padded, mapping, image_position=0)
    table = parity.build_shared_subset_candidate_table(relion, recovar_padded, mapping, image_position=0)

    assert ladder["denominator_under_full"]["support_size"] == 2
    np.testing.assert_array_equal(table["recovar_rot_slot"], np.array([0, 3], dtype=np.int64))
    expected = np.exp(np.array([2.0, 1.0], dtype=np.float64))
    expected = expected / np.sum(expected)
    np.testing.assert_allclose(table["recovar_denominator_probability"], expected)


def test_compare_shared_subset_score_deltas_returns_linear_fit():
    relion = {
        "exp_Mweight_posterior": np.array([np.exp(0.0), np.exp(-1.0)], dtype=np.float64),
        "exp_sum_weight": np.array(np.exp(0.0) + np.exp(-1.0), dtype=np.float64),
        "Pmax": np.array(np.exp(0.0) / (np.exp(0.0) + np.exp(-1.0)), dtype=np.float64),
        "pointer_dir_nonzeroprior": np.array([10, 11], dtype=np.int32),
        "pointer_psi_nonzeroprior": np.array([20, 21], dtype=np.int32),
        "acc_rot_id": np.array([0, 3], dtype=np.int32),
        "acc_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_coarse_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_in_denominator_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_fine_threshold_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_reconstruction_set": np.array([1, 1], dtype=np.int32),
        "translations_x": np.array([0.0, 1.5], dtype=np.float64),
        "translations_y": np.array([0.0, -0.5], dtype=np.float64),
    }
    recovar = {
        "selected_global_image_indices": np.array([4], dtype=np.int64),
        "local_rotation_pixel_indices": np.array([10, 10, 11, 11], dtype=np.int64),
        "local_rotation_psi_indices": np.array([20, 21, 20, 21], dtype=np.int64),
        "grid_n_psi": np.array(2, dtype=np.int32),
        "rotation_candidate_mask": np.array([[True, True, True, True]], dtype=bool),
        "pass2_scores_total": np.array([[[2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "translations": np.array([[0.0, 0.0], [1.5, -0.5]], dtype=np.float32),
    }

    mapping = parity.build_relion_recovar_candidate_mapping(relion, recovar, image_position=0)
    fit = parity.compare_shared_subset_score_deltas(relion, recovar, mapping, image_position=0)

    assert fit["best_relion_rot_id"] == 0
    assert fit["best_relion_trans_idx"] == 0
    assert fit["corr"] == pytest.approx(1.0)


def test_build_shared_subset_candidate_table_contains_common_scores():
    relion = {
        "exp_Mweight_posterior": np.array([np.exp(0.0), np.exp(-1.0)], dtype=np.float64),
        "exp_sum_weight": np.array(np.exp(0.0) + np.exp(-1.0), dtype=np.float64),
        "Pmax": np.array(np.exp(0.0) / (np.exp(0.0) + np.exp(-1.0)), dtype=np.float64),
        "pointer_dir_nonzeroprior": np.array([10, 11], dtype=np.int32),
        "pointer_psi_nonzeroprior": np.array([20, 21], dtype=np.int32),
        "acc_rot_id": np.array([0, 3], dtype=np.int32),
        "acc_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_coarse_trans_idx": np.array([0, 1], dtype=np.int32),
        "candidate_in_denominator_set": np.array([1, 1], dtype=np.int32),
        "candidate_in_fine_threshold_set": np.array([1, 0], dtype=np.int32),
        "candidate_in_reconstruction_set": np.array([1, 0], dtype=np.int32),
        "translations_x": np.array([0.0, 1.5], dtype=np.float64),
        "translations_y": np.array([0.0, -0.5], dtype=np.float64),
    }
    recovar = {
        "selected_global_image_indices": np.array([4], dtype=np.int64),
        "local_rotation_pixel_indices": np.array([10, 10, 11, 11], dtype=np.int64),
        "local_rotation_psi_indices": np.array([20, 21, 20, 21], dtype=np.int64),
        "local_rotation_psi_deg": np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32),
        "local_rotation_dir_vecs": np.array(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1]],
            dtype=np.float32,
        ),
        "local_rotation_eulers": np.array(
            [[0.0, 90.0, 0.0], [10.0, 80.0, 0.0], [20.0, 70.0, 0.0], [30.0, 60.0, 0.0]],
            dtype=np.float32,
        ),
        "grid_n_psi": np.array(2, dtype=np.int32),
        "rotation_candidate_mask": np.array([[True, True, True, True]], dtype=bool),
        "pass2_scores_raw": np.array([[[2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "pass2_scores_total": np.array([[[2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "pass2_cross_term": np.array([[[-4.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [0.0, -2.0]]], dtype=np.float32),
        "pass2_norm_term": np.zeros((1, 4, 1), dtype=np.float32),
        "rotation_log_prior": np.zeros((1, 4), dtype=np.float32),
        "translation_log_prior": np.zeros((1, 2), dtype=np.float32),
        "translations": np.array([[0.0, 0.0], [1.5, -0.5]], dtype=np.float32),
    }

    mapping = parity.build_relion_recovar_candidate_mapping(relion, recovar, image_position=0)
    table = parity.build_shared_subset_candidate_table(relion, recovar, mapping, image_position=0)

    np.testing.assert_array_equal(table["relion_rot_id"], np.array([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(table["recovar_rot_slot"], np.array([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(table["relion_pixel_index"], np.array([10, 11], dtype=np.int64))
    np.testing.assert_array_equal(table["recovar_pixel_index"], np.array([10, 11], dtype=np.int64))
    np.testing.assert_array_equal(table["relion_psi_index"], np.array([20, 21], dtype=np.int64))
    np.testing.assert_array_equal(table["recovar_psi_index"], np.array([20, 21], dtype=np.int64))
    np.testing.assert_allclose(table["recovar_total_score"], np.array([2.0, 1.0]))
    np.testing.assert_array_equal(table["recovar_trans_idx"], np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(table["recovar_translation_x"], np.array([0.0, 1.5]))
    np.testing.assert_allclose(table["recovar_translation_y"], np.array([0.0, -0.5]))
    np.testing.assert_allclose(table["recovar_cross_term"], np.array([-4.0, -2.0]))
    np.testing.assert_allclose(table["recovar_norm_term"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(table["recovar_cross_score_delta_to_relion_best"], np.array([0.0, -1.0]))
    np.testing.assert_allclose(table["recovar_norm_score_delta_to_relion_best"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        table["relion_normalized_weight"],
        np.array([np.exp(0.0), np.exp(-1.0)]) / (np.exp(0.0) + np.exp(-1.0)),
    )
    assert int(table["best_relion_candidate_index"]) == 0
    np.testing.assert_allclose(table["recovar_combined_log_prior"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(table["prior_error"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(table["data_delta_error_to_relion_best"], np.array([0.0, 0.0]))


def test_summarize_candidate_table_components_splits_prior_and_data_errors():
    candidate_table = {
        "relion_delta_to_best": np.array([0.0, -1.0, -2.0], dtype=np.float64),
        "recovar_delta_to_relion_best": np.array([0.0, -0.8, -2.5], dtype=np.float64),
        "relion_data_delta_to_relion_best": np.array([0.0, -0.6, -1.5], dtype=np.float64),
        "recovar_data_delta_to_relion_best": np.array([0.0, -0.4, -2.2], dtype=np.float64),
        "relion_prior_delta_to_relion_best": np.array([0.0, -0.4, -0.5], dtype=np.float64),
        "recovar_prior_delta_to_relion_best": np.array([0.0, -0.4, -0.3], dtype=np.float64),
        "prior_error": np.array([0.1, -0.1, 0.2], dtype=np.float64),
        "best_relion_candidate_index": np.int64(0),
        "best_recovar_candidate_index": np.int64(0),
    }

    summary = parity.summarize_candidate_table_components(candidate_table)

    assert summary["candidate_count"] == 3
    assert summary["same_best_candidate"]
    assert summary["total_delta_error"]["mean_abs"] == pytest.approx((0.0 + 0.2 + 0.5) / 3.0)
    assert summary["data_delta_error"]["max_abs"] == pytest.approx(0.7)
    assert summary["prior_delta_error"]["max_abs"] == pytest.approx(0.2)
    assert summary["prior_level_error"]["rms"] == pytest.approx(np.sqrt((0.1**2 + 0.1**2 + 0.2**2) / 3.0))


def test_solve_scale_for_target_pmax_hits_requested_value():
    scores = np.array([0.0, -1.0, -2.0], dtype=np.float64)
    target = 0.7
    alpha = parity.solve_scale_for_target_pmax(scores, target)
    summary = parity.score_summary_from_log_scores(alpha * scores)

    assert summary["pmax"] == pytest.approx(target, rel=1e-6, abs=1e-6)
