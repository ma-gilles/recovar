import numpy as np

from scripts.compare_k4_relion_recovar_fine_operands import (
    _component_counterfactual,
    _infer_current_size,
    _metric,
    _metric_up_to_global_sign,
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
