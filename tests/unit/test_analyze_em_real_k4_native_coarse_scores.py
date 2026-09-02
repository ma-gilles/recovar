from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_em_real_k4_native_coarse_scores as analyzer
from scripts import validate_relion_coarse_score_capture as validator


def test_centered_comparison_removes_only_additive_constant():
    relion = np.asarray([1.0, 2.0, 4.0, 8.0])
    exact = analyzer._centered_comparison(relion + 13.0, relion)
    deformed = analyzer._centered_comparison(relion + [13.0, 13.0, 13.0, 14.0], relion)

    assert exact["centered_max_abs"] == 0
    assert exact["winner_exact"] is True
    assert deformed["centered_max_abs"] == pytest.approx(0.75)
    assert deformed["centered_relative_l2"] > 0


def test_native_surface_converts_direction_major_to_psi_major():
    n_classes, n_directions, n_psi, n_translations = 1, 2, 2, 1
    candidates = np.zeros(4, dtype=validator.CANDIDATE_DTYPE)
    # Native flattened rotation order: (d0,p0), (d0,p1), (d1,p0), (d1,p1).
    candidates["raw_diff2"] = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
    candidates["orientation_log_prior"] = -1
    candidates["translation_log_prior"] = -2
    candidates["combined_preexponent"] = -candidates["raw_diff2"] - 3
    candidates["flags"] = validator.ACTIVE
    candidates[1]["flags"] |= validator.SIGNIFICANT
    header = [0] * 64
    header[9:14] = [0, n_classes, n_directions, n_psi, n_translations]
    capture = SimpleNamespace(header=tuple(header), candidates=candidates)

    surface = analyzer._native_surface(capture)

    # RECOVAR rotation order: (p0,d0), (p0,d1), (p1,d0), (p1,d1).
    assert surface["raw_score"].reshape(-1).tolist() == [-10.0, -12.0, -11.0, -13.0]
    assert surface["significant"].reshape(-1).tolist() == [False, False, True, False]


def test_score_swaps_localize_support_difference_to_raw_likelihood():
    native_raw = np.asarray([[[9.0, 8.0, 7.0, 6.0]]])
    recovar_raw = np.asarray([[[9.0, 7.0, 8.0, 6.0]]])
    prior = np.asarray([[[0.0, 0.0, 0.0, 0.0]]])
    native = {
        "raw_score": native_raw,
        "total_prior": prior,
        "combined_score": native_raw + prior,
        "significant": np.asarray([[[True, True, False, False]]]),
    }
    recovar = {
        "raw_score": recovar_raw,
        "total_prior": prior,
        "combined_score": recovar_raw + prior,
        "significant": np.asarray([[[True, False, True, False]]]),
    }

    report, _ = analyzer._support_counterfactuals(native=native, recovar=recovar)

    assert report["native_combined"]["exact"] is True
    assert report["recovar_combined"]["jaccard"] == pytest.approx(1 / 3)
    assert report["native_raw_recovar_prior"]["exact"] is True
    assert report["set_identities"]["recovar_raw_native_prior_equals_recovar_combined"] is True


def test_raw_difference_axis_decomposition_separates_rotation_and_translation_terms():
    native = np.zeros((1, 2, 3))
    rotation_only = np.asarray([[[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]])
    translation_only = np.asarray([[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]])

    rotation_report = analyzer._raw_difference_axis_decomposition(rotation_only, native)
    translation_report = analyzer._raw_difference_axis_decomposition(translation_only, native)

    assert rotation_report[
        "class_rotation_mean_fraction_of_centered_squared_difference"
    ] == pytest.approx(1.0)
    assert rotation_report[
        "within_class_rotation_fraction_of_centered_squared_difference"
    ] == pytest.approx(0.0)
    assert translation_report[
        "class_rotation_mean_fraction_of_centered_squared_difference"
    ] == pytest.approx(0.0)
    assert translation_report[
        "translation_main_effect_fraction_of_centered_squared_difference"
    ] == pytest.approx(1.0)


def test_classification_requires_both_score_swap_identities():
    base = {
        "native_combined": {"exact": True},
        "native_raw_recovar_prior": {"exact": True},
        "set_identities": {"recovar_raw_native_prior_equals_recovar_combined": True},
    }
    records = [{"support_counterfactuals": base}]

    assert analyzer._classification(records) == (
        "raw_likelihood_surface_is_first_material_support_difference"
    )

    base["native_raw_recovar_prior"]["exact"] = False
    assert analyzer._classification(records) == "likelihood_and_prior_contributions_both_affect_support"
