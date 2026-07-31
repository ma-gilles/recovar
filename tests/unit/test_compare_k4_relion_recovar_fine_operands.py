import numpy as np

from scripts.compare_k4_relion_recovar_fine_operands import (
    _center,
    _component_counterfactual,
    _direct_score_image_factor,
    _infer_current_size,
    _metric,
    _metric_up_to_global_sign,
    _relion_cuda_normalization_factors,
    _select_component_classification,
    _translation_alignment,
    _tree_raw_diff2,
    _zero_dc_compact_score_weight,
)


def test_fine_operand_current_size_and_translation_alignment():
    assert _infer_current_size(74 * 38) == 74
    fine = np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    relion = np.asarray(-2 * np.pi * fine[1] / 256, dtype=np.float32)

    index, error = _translation_alignment(relion, fine, 256)

    assert index == 1
    assert error < 1e-7


def test_fine_operand_tree_replay_preserves_float32_topology():
    reference = np.asarray([1 + 2j, 2 + 0j, 3 - 1j], dtype=np.complex64)
    shifted = np.asarray([0 + 1j, 1 + 0j, 2 - 2j], dtype=np.complex64)
    corr = np.asarray([1, 2, 3], dtype=np.float32)

    raw, contribution, lanes = _tree_raw_diff2(
        reference, shifted, corr, np.float32(7)
    )

    np.testing.assert_array_equal(contribution, np.asarray([1, 1, 3], dtype=np.float32))
    np.testing.assert_array_equal(lanes[:3], contribution)
    assert raw == np.float32(12)


def test_fine_operand_counterfactual_identifies_reference_component():
    relion = np.asarray([10, 20, 30], dtype=np.float32)
    all_recovar = np.asarray([11, 18, 33], dtype=np.float32)
    substitutions = {
        "reference": all_recovar.copy(),
        "shifted_image": np.asarray([10.5, 20, 30], dtype=np.float32),
        "corr": relion.copy(),
    }

    report = _component_counterfactual(relion, all_recovar, substitutions)

    assert report["strongest_single_component"] == "reference"
    assert report["strongest_target_delta_energy_removed_fraction"] == 1.0


def test_fine_operand_counterfactual_can_remove_common_score_offsets():
    relion = np.asarray([10, 20, 30], dtype=np.float32)
    all_recovar = np.asarray([111, 118, 133], dtype=np.float32)
    substitutions = {
        "reference": np.asarray([110, 120, 130], dtype=np.float32),
        "shifted_image": all_recovar.copy(),
        "corr": relion.copy(),
    }

    report = _component_counterfactual(
        relion,
        all_recovar,
        substitutions,
        center_deltas=True,
    )

    assert report["deltas_centered"] is True
    assert report["strongest_single_component"] == "shifted_image"
    assert report["strongest_target_delta_energy_removed_fraction"] == 1.0


def test_fine_operand_center_removes_only_common_offset():
    centered = _center(np.asarray([101, 103, 108], dtype=np.float32))

    np.testing.assert_allclose(centered, np.asarray([-3, -1, 4], dtype=np.float64))
    assert np.sum(centered) == 0.0


def test_single_candidate_classification_uses_raw_residual():
    relion = np.asarray([10], dtype=np.float32)
    all_recovar = np.asarray([12], dtype=np.float32)
    substitutions = {
        "reference": np.asarray([10], dtype=np.float32),
        "shifted_image": np.asarray([11], dtype=np.float32),
        "corr": np.asarray([12], dtype=np.float32),
    }
    raw = _component_counterfactual(relion, all_recovar, substitutions)
    centered = _component_counterfactual(
        relion,
        all_recovar,
        substitutions,
        center_deltas=True,
    )

    classification, basis = _select_component_classification(
        raw,
        centered,
        candidate_count=1,
    )

    assert raw["informative"] is True
    assert centered["informative"] is False
    assert (
        classification
        == "corr_has_largest_raw_fine_operand_single_substitution_effect"
    )
    assert basis == "raw_diff2"


def test_multi_candidate_classification_can_use_centered_residual():
    relion = np.asarray([10, 20, 30], dtype=np.float32)
    all_recovar = np.asarray([111, 118, 133], dtype=np.float32)
    substitutions = {
        "reference": np.asarray([110, 120, 130], dtype=np.float32),
        "shifted_image": all_recovar.copy(),
        "corr": relion.copy(),
    }
    raw = _component_counterfactual(relion, all_recovar, substitutions)
    centered = _component_counterfactual(
        relion,
        all_recovar,
        substitutions,
        center_deltas=True,
    )

    classification, basis = _select_component_classification(
        raw,
        centered,
        candidate_count=3,
    )

    assert (
        classification
        == (
            "shifted_image_has_largest_centered_fine_operand_"
            "single_substitution_effect"
        )
    )
    assert basis == "centered_raw_diff2"


def test_zero_residual_classification_fails_closed():
    relion = np.asarray([10], dtype=np.float32)
    substitutions = {
        name: relion.copy()
        for name in ("reference", "shifted_image", "corr")
    }
    raw = _component_counterfactual(relion, relion, substitutions)
    centered = _component_counterfactual(
        relion,
        relion,
        substitutions,
        center_deltas=True,
    )

    classification, basis = _select_component_classification(
        raw,
        centered,
        candidate_count=1,
    )

    assert classification == "no_nonzero_fine_operand_residual"
    assert basis == "none"


def test_fine_operand_metric_reports_directional_delta():
    report = _metric(
        np.asarray([1 + 1j, 2 + 0j], dtype=np.complex64),
        np.asarray([1 + 1j, 2.5 + 0j], dtype=np.complex64),
    )

    assert report["exact_equal"] is False
    assert report["mismatch_count"] == 1
    assert report["max_abs"] == 0.5


def test_fine_operand_metric_separates_global_fourier_sign():
    relion = np.asarray([1 + 2j, 3 - 4j], dtype=np.complex64)
    recovar = -relion

    report = _metric_up_to_global_sign(relion, recovar)

    assert report["raw"]["relative_l2_over_relion"] == 2.0
    assert report["recovar_alignment_multiplier"] == -1
    assert report["sign_aligned"]["exact_equal"] is True


def test_fine_operand_score_weight_applies_production_dc_zero():
    image_shape = (8, 8)
    compact_indices = np.asarray([20, 1, 5, 8], dtype=np.int32)
    score_weight = np.asarray([7, 2, 3, 4], dtype=np.float32)

    result, dc_mask = _zero_dc_compact_score_weight(
        score_weight,
        compact_indices,
        image_shape,
    )

    np.testing.assert_array_equal(dc_mask, np.asarray([True, False, False, False]))
    np.testing.assert_array_equal(result, np.asarray([0, 2, 3, 4], dtype=np.float32))


def test_fine_operand_direct_score_factor_tracks_preprocess_backend():
    image_correction = np.float32(0.99644595)
    scale_correction = np.float32(0.993949)

    relion_cuda = _direct_score_image_factor(
        relion_cuda_preprocess=True,
        image_correction=image_correction,
        scale_correction=scale_correction,
    )
    dataset_native = _direct_score_image_factor(
        relion_cuda_preprocess=False,
        image_correction=image_correction,
        scale_correction=scale_correction,
    )

    assert relion_cuda == np.float32(1.0) / scale_correction
    assert dataset_native == np.float32(
        relion_cuda * np.float32(image_correction / scale_correction)
    )


def test_fine_operand_relion_cuda_counterfactual_derives_normalization():
    values = {
        "relion_preprocess_normalization_factors": np.asarray(
            [7.0, 8.0], dtype=np.float32
        ),
        "image_corrections": np.asarray([0.9, 1.1], dtype=np.float32),
        "scale_corrections": np.asarray([0.6, 2.2], dtype=np.float32),
    }

    captured = _relion_cuda_normalization_factors(
        values,
        captured_backend_is_relion_cuda=True,
    )
    derived = _relion_cuda_normalization_factors(
        values,
        captured_backend_is_relion_cuda=False,
    )

    np.testing.assert_array_equal(captured, np.asarray([7.0, 8.0], dtype=np.float32))
    np.testing.assert_allclose(
        derived,
        np.asarray([1.5, 0.5], dtype=np.float32),
        rtol=np.finfo(np.float32).eps,
        atol=0.0,
    )
