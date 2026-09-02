from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_em_real_k4_native_coarse_components as analyzer
from scripts import validate_relion_coarse_component_capture as validator


def test_native_component_surface_converts_direction_major_to_score_order():
    candidates = np.zeros(4, dtype=validator.CANDIDATE_DTYPE)
    candidates["reference_norm"] = [10.0, 11.0, 12.0, 13.0]
    header = [0] * 64
    # One class, two directions, two psi values, and one translation.
    header[9:14] = [0, 1, 2, 2, 1]
    capture = SimpleNamespace(header=tuple(header), candidates=candidates)

    surface = analyzer._native_component_surface(capture, "reference_norm")

    # RELION: (d0,p0), (d0,p1), (d1,p0), (d1,p1).
    # RECOVAR: (p0,d0), (p0,d1), (p1,d0), (p1,d1).
    assert surface.reshape(-1).tolist() == [-10.0, -12.0, -11.0, -13.0]


def test_component_swaps_localize_support_difference_to_cross_term():
    native_norm = np.asarray([[[4.0, 3.0, 2.0, 1.0]]])
    native_cross = np.asarray([[[5.0, 5.0, 1.0, 1.0]]])
    recovar_norm = np.asarray([[[4.0, 3.0, 2.1, 0.9]]])
    recovar_cross = np.asarray([[[1.0, 1.0, 5.0, 5.0]]])
    prior = np.zeros_like(native_norm)
    native = {
        "combined_score": native_norm + native_cross,
        "total_prior": prior,
        "significant": np.asarray([[[True, True, False, False]]]),
    }
    recovar = {
        "combined_score": recovar_norm + recovar_cross,
        "total_prior": prior,
    }

    report = analyzer._component_support_counterfactuals(
        native=native,
        recovar=recovar,
        native_norm=native_norm,
        native_cross=native_cross,
        recovar_norm=recovar_norm,
        recovar_cross=recovar_cross,
    )

    assert report["recovar_captured_combined"]["exact"] is False
    assert report["recovar_norm_native_cross_native_prior"]["exact"] is True
    assert report["native_norm_recovar_cross_native_prior"]["exact"] is False
    assert report["set_identities"]["cross_swap_is_prior_invariant"] is True


def test_residual_decomposition_closes_and_reports_remaining_energy():
    native_norm = np.asarray([1.0, 2.0, 3.0, 4.0])
    native_cross = np.asarray([4.0, 3.0, 2.0, 1.0])
    norm_residual = np.asarray([0.0, 1.0, 0.0, -1.0])
    cross_residual = np.asarray([1.0, 0.0, -1.0, 0.0])
    recovar_norm = native_norm + norm_residual
    recovar_cross = native_cross + cross_residual
    native_raw = native_norm + native_cross + 100.0
    recovar_raw = recovar_norm + recovar_cross + 200.0

    report = analyzer._residual_decomposition(
        recovar_raw=recovar_raw,
        native_raw=native_raw,
        recovar_norm=recovar_norm,
        native_norm=native_norm,
        recovar_cross=recovar_cross,
        native_cross=native_cross,
    )

    assert report["component_residual_closure_centered_rms"] == pytest.approx(0.0)
    assert report["recovar_component_closure_centered_rms"] == pytest.approx(0.0)
    assert report["native_component_closure_centered_rms"] == pytest.approx(0.0)
    assert report[
        "fraction_raw_residual_energy_remaining_after_native_cross"
    ] == pytest.approx(0.5)
    assert report[
        "fraction_raw_residual_energy_remaining_after_native_norm"
    ] == pytest.approx(0.5)
