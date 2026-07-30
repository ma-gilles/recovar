from __future__ import annotations

import numpy as np

from scripts.analyze_em_k1_live_reference_counterfactual import (
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
    assert np.array_equal(selected[0], reference[0, expected_indices])


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
