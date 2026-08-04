import numpy as np
import pytest

import scripts.compare_k4_relion_recovar_fine_operands as comparator
from scripts.compare_k4_relion_recovar_fine_operands import (
    _center,
    _component_counterfactual,
    _direct_score_image_factor,
    _infer_current_size,
    _is_relion_cuda_replay_mode,
    _jax_tree_raw_diff2,
    _metric,
    _metric_up_to_global_sign,
    _preprocess_normalization_source,
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


def test_fine_operand_jax_tree_accepts_batched_native_operands():
    reference = np.asarray(
        [[1 + 2j, 2 + 0j, 3 - 1j], [2 + 1j, 0 + 0j, 1 - 2j]],
        dtype=np.complex64,
    )
    shifted = np.asarray(
        [[0 + 1j, 1 + 0j, 2 - 2j], [1 + 0j, 0 + 0j, 2 - 1j]],
        dtype=np.complex64,
    )
    corr = np.asarray([[1, 2, 3], [2, 1, 4]], dtype=np.float32)
    sum_init = np.asarray([7, 3], dtype=np.float32)

    raw = _jax_tree_raw_diff2(reference, shifted, corr, sum_init)
    expected = np.asarray(
        [
            _tree_raw_diff2(reference[index], shifted[index], corr[index], sum_init[index])[0]
            for index in range(2)
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_max_ulp(raw, expected, maxulp=1)
    assert raw.shape == (2,)
    assert raw.dtype == np.float32


def test_fine_operand_jax_tree_preserves_full_grid_gap_topology():
    full_to_compact = np.asarray([0, -1, -1, 1, -1, 2], dtype=np.int32)
    reference = np.asarray([[1 + 2j, 2 + 0j, 3 - 1j]], dtype=np.complex64)
    shifted = np.asarray([[0 + 1j, 1 + 0j, 2 - 2j]], dtype=np.complex64)
    corr = np.asarray([[1, 2, 3]], dtype=np.float32)
    sum_init = np.asarray([7], dtype=np.float32)
    full_reference = np.zeros((1, full_to_compact.size), dtype=np.complex64)
    full_shifted = np.zeros_like(full_reference)
    full_corr = np.zeros((1, full_to_compact.size), dtype=np.float32)
    valid = full_to_compact >= 0
    full_reference[:, valid] = reference[:, full_to_compact[valid]]
    full_shifted[:, valid] = shifted[:, full_to_compact[valid]]
    full_corr[:, valid] = corr[:, full_to_compact[valid]]

    compact = _jax_tree_raw_diff2(
        reference,
        shifted,
        corr,
        sum_init,
        full_to_compact,
    )
    full = _jax_tree_raw_diff2(
        full_reference,
        full_shifted,
        full_corr,
        sum_init,
    )

    np.testing.assert_array_equal(compact, full)


def test_fine_operand_counterfactual_can_identify_jax_arithmetic():
    relion = np.asarray([10, 20, 30], dtype=np.float32)
    all_recovar = np.asarray([11, 18, 33], dtype=np.float32)
    substitutions = {
        "reference": np.asarray([10.5, 20, 30], dtype=np.float32),
        "shifted_image": np.asarray([10, 19.5, 30], dtype=np.float32),
        "corr": np.asarray([10, 20, 30.5], dtype=np.float32),
        "jax_arithmetic_on_native_operands": all_recovar.copy(),
    }

    report = _component_counterfactual(
        relion,
        all_recovar,
        substitutions,
        center_deltas=True,
    )

    assert report["strongest_single_component"] == (
        "jax_arithmetic_on_native_operands"
    )
    assert report["strongest_target_delta_energy_removed_fraction"] == 1.0


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


def test_single_candidate_exact_component_tie_is_unresolved():
    relion = np.asarray([10], dtype=np.float32)
    all_recovar = np.asarray([12], dtype=np.float32)
    substitutions = {
        "reference": all_recovar.copy(),
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
        candidate_count=1,
    )

    assert raw["strongest_components"] == ["reference", "shifted_image"]
    assert raw["strongest_is_unique"] is False
    assert (
        classification
        == "multiple_fine_operand_components_tie_for_largest_raw_"
        "single_substitution_effect"
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


def test_multi_candidate_exact_centered_component_tie_is_unresolved():
    relion = np.asarray([10, 20], dtype=np.float32)
    all_recovar = np.asarray([11, 19], dtype=np.float32)
    substitutions = {
        "reference": all_recovar.copy(),
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
        candidate_count=2,
    )

    assert centered["strongest_components"] == ["reference", "shifted_image"]
    assert centered["strongest_is_unique"] is False
    assert (
        classification
        == "multiple_fine_operand_components_tie_for_largest_centered_"
        "single_substitution_effect"
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


@pytest.mark.parametrize(
    ("mode", "captured_backend_is_relion_cuda", "expected"),
    [
        ("dataset_native_jax_fft", True, "not_applied"),
        ("dataset_native_jax_fft", False, "not_applied"),
        ("relion_cuda", True, "captured"),
        ("relion_cuda_native_lane", True, "captured"),
        ("relion_cuda", False, "derived_image_correction_over_scale"),
    ],
)
def test_fine_operand_reports_normalization_source(
    mode, captured_backend_is_relion_cuda, expected
):
    assert (
        _preprocess_normalization_source(
            mode,
            captured_backend_is_relion_cuda=captured_backend_is_relion_cuda,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("mode", "expected_native_lane"),
    [("relion_cuda", False), ("relion_cuda_native_lane", True)],
)
def test_fine_operand_relion_cuda_counterfactual_routes_reduction_tree(
    monkeypatch,
    mode,
    expected_native_lane,
):
    captured = {}

    def fake_preprocess(
        images,
        normalization_factors,
        integer_shifts,
        radius,
        cosine_width,
        apply_mask,
        *,
        native_lane_reduction=False,
    ):
        captured["normalization_factors"] = np.asarray(normalization_factors)
        captured["native_lane_reduction"] = native_lane_reduction
        return images, images

    monkeypatch.setattr(comparator, "relion_preprocess_real_f32", fake_preprocess)
    values = {
        "raw_real_images": np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        "relion_preprocess_normalization_factors": np.ones(1, dtype=np.float32),
        "integer_pre_shifts": np.zeros((1, 2), dtype=np.int32),
        "relion_cuda_preprocess": np.bool_(False),
        "preprocess_backend": np.asarray("dataset_native"),
        "score_with_masked_images": np.bool_(True),
        "voxel_size": np.float32(1.5),
        "image_corrections": np.asarray([0.9], dtype=np.float32),
        "scale_corrections": np.asarray([0.6], dtype=np.float32),
    }

    processed, replay_mode = comparator._reconstruct_processed_score_half(
        values,
        particle_diameter_angstrom=3.0,
        mask_edge_pixels=2.0,
        mode_override=mode,
    )

    assert replay_mode == mode
    assert processed.shape == (1, 12)
    assert captured["native_lane_reduction"] is expected_native_lane
    np.testing.assert_array_equal(
        captured["normalization_factors"],
        np.asarray([np.float32(0.9) / np.float32(0.6)], dtype=np.float32),
    )
    assert _is_relion_cuda_replay_mode(mode)


def test_fine_operand_dataset_native_counterfactual_accepts_relion_capture():
    values = {
        "raw_real_images": np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        "relion_preprocess_normalization_factors": np.asarray(
            [0.75], dtype=np.float32
        ),
        "integer_pre_shifts": np.zeros((1, 2), dtype=np.int32),
        "relion_cuda_preprocess": np.bool_(True),
        "preprocess_backend": np.asarray("relion_cuda"),
        "score_with_masked_images": np.bool_(False),
    }

    processed, replay_mode = comparator._reconstruct_processed_score_half(
        values,
        particle_diameter_angstrom=3.0,
        mask_edge_pixels=2.0,
        mode_override="dataset_native_jax_fft",
    )

    assert replay_mode == "dataset_native_jax_fft"
    np.testing.assert_array_equal(
        np.asarray(processed),
        np.asarray(
            comparator._centered_rfft2_jax(
                values["raw_real_images"]
            ).reshape(1, -1)
        ),
    )


def test_fine_operand_dataset_native_capture_rejects_active_normalization():
    values = {
        "raw_real_images": np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        "relion_preprocess_normalization_factors": np.asarray(
            [0.75], dtype=np.float32
        ),
        "integer_pre_shifts": np.zeros((1, 2), dtype=np.int32),
        "relion_cuda_preprocess": np.bool_(False),
        "preprocess_backend": np.asarray("dataset_native"),
        "score_with_masked_images": np.bool_(False),
    }

    with pytest.raises(
        ValueError,
        match="dataset-native capture unexpectedly stored active RELION normalization",
    ):
        comparator._reconstruct_processed_score_half(
            values,
            particle_diameter_angstrom=3.0,
            mask_edge_pixels=2.0,
        )


def test_fine_operand_replays_captured_native_lane_mode(monkeypatch):
    captured = {}

    def fake_preprocess(
        images,
        normalization_factors,
        integer_shifts,
        radius,
        cosine_width,
        apply_mask,
        *,
        native_lane_reduction=False,
    ):
        captured["native_lane_reduction"] = native_lane_reduction
        return images, images

    monkeypatch.setattr(comparator, "relion_preprocess_real_f32", fake_preprocess)
    values = {
        "raw_real_images": np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        "relion_preprocess_normalization_factors": np.ones(1, dtype=np.float32),
        "integer_pre_shifts": np.zeros((1, 2), dtype=np.int32),
        "relion_cuda_preprocess": np.bool_(True),
        "relion_native_lane_reduction": np.bool_(True),
        "preprocess_backend": np.asarray("relion_cuda"),
        "score_with_masked_images": np.bool_(True),
        "voxel_size": np.float32(1.5),
    }

    processed, replay_mode = comparator._reconstruct_processed_score_half(
        values,
        particle_diameter_angstrom=3.0,
        mask_edge_pixels=2.0,
    )

    assert replay_mode == "relion_cuda_native_lane"
    assert processed.shape == (1, 12)
    assert captured["native_lane_reduction"] is True
