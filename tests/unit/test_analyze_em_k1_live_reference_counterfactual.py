from __future__ import annotations

import numpy as np

from scripts.analyze_em_k1_live_reference_counterfactual import (
    classify_live_operands,
    classify_live_reference,
    recovar_score_components,
    reference_swap_counterfactual,
    relion_reference_on_recovar_window,
)


def test_maps_relion_fftw_rows_to_recovar_centered_window() -> None:
    current_size = 6
    full_size = 8
    current_half = 4
    reference = np.arange(current_size * current_half).reshape(1, -1)
    # Centered full rows ky=-2, 0, +3 and columns 1, 2, 0.
    window = np.asarray(
        [
            (full_size // 2 - 2) * (full_size // 2 + 1) + 1,
            (full_size // 2) * (full_size // 2 + 1) + 2,
            (full_size // 2 + 3) * (full_size // 2 + 1),
        ]
    )
    selected = relion_reference_on_recovar_window(
        reference,
        window,
        full_image_size=full_size,
        current_size=current_size,
    )
    expected_indices = np.asarray([4 * current_half + 1, 2, 3 * current_half])
    assert np.array_equal(
        selected[0],
        -(full_size**2) * reference[0, expected_indices],
    )


def test_recomputes_recovar_norm_and_cross_components() -> None:
    references = np.asarray([[1 + 2j, 3 - 1j], [2 - 1j, -1 + 0.5j]])
    shifted = np.asarray([[0.5 + 1j, 2 - 0.5j], [-1 + 0j, 0.25 + 2j]])
    ctf2 = np.asarray([2.0, 0.5])
    half_weights = np.asarray([1.0, 2.0])
    norm, cross = recovar_score_components(
        references,
        shifted,
        ctf2,
        half_weights,
    )
    expected_norm = -0.5 * np.sum(
        ctf2[None] * np.abs(references) ** 2 * half_weights[None],
        axis=1,
    )
    expected_cross = np.real(np.einsum("tp,rp,p->rt", np.conj(shifted), references, half_weights))
    assert np.allclose(norm, expected_norm[:, None])
    assert np.allclose(cross, expected_cross)


def test_reference_swap_reports_causal_energy_removal() -> None:
    rows = np.arange(4, dtype=np.float64)[:, None]
    columns = np.arange(3, dtype=np.float64)[None, :]
    baseline = 4.0 * rows + columns
    swapped = 0.1 * rows + 0.05 * columns
    report = reference_swap_counterfactual(baseline, swapped)
    assert report["live_reference_dominated"]
    assert report["counterfactual_energy_removal_fraction"] > 0.99


def test_classifies_fixed_cohort_live_reference_outcomes() -> None:
    assert (
        classify_live_reference(capture_qualified=False, dominated=14, expected=14)
        == "operand_capture_not_qualified"
    )
    assert (
        classify_live_reference(capture_qualified=True, dominated=14, expected=14)
        == "raw_coarse_residual_is_live_projected_reference_dominated"
    )
    assert (
        classify_live_reference(capture_qualified=True, dominated=0, expected=14)
        == "live_projected_reference_rejected_as_raw_coarse_residual_cause"
    )
    assert (
        classify_live_reference(capture_qualified=True, dominated=3, expected=14)
        == "raw_coarse_residual_has_mixed_live_projected_reference_effect"
    )


def test_classifies_live_base_image_factorial() -> None:
    dominated = {
        "reference": 0,
        "shifted_image": 14,
        "correction": 0,
        "reference_and_shifted_image": 14,
        "reference_and_correction": 0,
        "shifted_image_and_correction": 14,
        "all_live": 14,
        "base_corrected_image": 14,
        "translation_phase": 0,
    }
    assert (
        classify_live_operands(
            capture_qualified=True,
            dominated=dominated,
            expected=14,
        )
        == "raw_coarse_residual_is_live_base_corrected_image_dominated_"
        "not_reference_correction_or_translation_phase"
    )
    assert (
        classify_live_operands(
            capture_qualified=False,
            dominated=dominated,
            expected=14,
        )
        == "operand_capture_not_qualified"
    )
