from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import analyze_em_real_k4_native_coarse_operands as analysis


def _support(exact_records: int, jaccard: float) -> dict[str, float | int]:
    return {"exact_records": exact_records, "jaccard": jaccard}


def test_rotation_order_maps_class_local_direction_major_to_psi_major() -> None:
    artifact = SimpleNamespace(
        class_one_based=2,
        rotation_keys=np.arange(6, 12, dtype=np.uint64),
    )
    order = analysis._native_rotation_order(
        artifact, n_directions=2, n_psi=3
    )
    np.testing.assert_array_equal(order, np.asarray([0, 3, 1, 4, 2, 5]))


def test_support_factorial_tracks_exact_top_count_and_centered_energy() -> None:
    native_raw = np.asarray([[[4.0, 3.0, 0.0, -1.0]]])
    native_prior = np.zeros_like(native_raw)
    native = {
        "raw_score": native_raw,
        "total_prior": native_prior,
        "combined_score": native_raw,
        "significant": np.asarray([[[True, True, False, False]]]),
    }
    recovar_raw = np.asarray([[[0.0, 3.0, 4.0, -1.0]]])
    recovar = {"combined_score": recovar_raw}
    support, energy = analysis._support_factorial(
        native=native,
        recovar=recovar,
        raw_surfaces={
            "recovar_operands": recovar_raw,
            "native_projected_reference": native_raw,
            "native_shifted_image": recovar_raw,
            "native_correction": recovar_raw,
            "all_native_operands": native_raw,
        },
    )
    assert support["recovar_operands"]["exact"] is False
    assert support["native_projected_reference"]["exact"] is True
    assert energy["native_projected_reference_fraction_baseline_energy_remaining"] == 0
    assert energy["native_shifted_image_fraction_baseline_energy_remaining"] == 1


def test_classifies_projected_reference_boundary() -> None:
    baseline = _support(5, 0.9517896274653032)
    summary = {
        "support_counterfactuals": {
            "recovar_operands": baseline,
            "native_projected_reference": _support(16, 1.0),
            "native_shifted_image": baseline.copy(),
            "native_correction": baseline.copy(),
            "all_native_operands": _support(16, 1.0),
        },
        "score_residual_energy": {
            "native_projected_reference_fraction_baseline_energy_remaining": 1.2e-9,
            "native_shifted_image_fraction_baseline_energy_remaining": 1.0000001,
            "native_correction_fraction_baseline_energy_remaining": 1.0000004,
            "all_native_operands_fraction_baseline_energy_remaining": 1.1e-9,
        },
    }
    assert analysis.classify(summary, 16) == (
        "projected_reference_is_first_material_k4_coarse_likelihood_operand_difference"
    )


def test_classifies_closed_projected_reference_and_support_boundary() -> None:
    exact = _support(16, 1.0)
    summary = {
        "support_counterfactuals": {
            "recovar_operands": exact,
            "recovar_captured_combined": exact.copy(),
        },
        "score_residual_energy": {},
        "operand_relative_l2": {
            "projected_reference": {"maximum": 0.0},
        },
        "mapping": {"euler_transpose_max_abs": 0.0},
    }

    assert analysis.classify(summary, 16) == (
        "k4_coarse_projected_reference_and_significant_support_match_native"
    )


def test_rejects_projected_reference_classification_when_shifted_image_also_helps() -> None:
    baseline = _support(5, 0.95)
    summary = {
        "support_counterfactuals": {
            "recovar_operands": baseline,
            "native_projected_reference": _support(16, 1.0),
            "native_shifted_image": _support(12, 0.99),
            "native_correction": baseline.copy(),
            "all_native_operands": _support(16, 1.0),
        },
        "score_residual_energy": {
            "native_projected_reference_fraction_baseline_energy_remaining": 1e-9,
            "native_shifted_image_fraction_baseline_energy_remaining": 0.5,
            "native_correction_fraction_baseline_energy_remaining": 1.0,
            "all_native_operands_fraction_baseline_energy_remaining": 1e-9,
        },
    }
    assert analysis.classify(summary, 16) == (
        "k4_coarse_likelihood_operand_difference_is_mixed_or_unresolved"
    )
