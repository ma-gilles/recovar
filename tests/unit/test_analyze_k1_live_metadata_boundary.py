import numpy as np
import pytest

from scripts.analyze_k1_live_metadata_boundary import (
    align_random_subsets_by_stack,
    analyze_arrays,
    angular_error_deg,
    source_rows_for_live_order,
)


def test_random_subsets_are_joined_by_stack_not_half_star_row():
    np.testing.assert_array_equal(
        align_random_subsets_by_stack(
            np.array([10, 20, 30, 40]),
            np.array([30, 10, 40, 20]),
            np.array([1, 1, 2, 2]),
        ),
        [1, 2, 1, 2],
    )


def test_source_rows_join_physical_order_and_validate_halves():
    rows = source_rows_for_live_order(
        np.array([11, 22, 33, 44]),
        np.array([1, 2, 1, 2]),
        np.array([33, 11, 44, 22]),
        np.array([1, 1, 2, 2]),
    )
    np.testing.assert_array_equal(rows, [2, 0, 3, 1])


def test_source_rows_reject_half_mismatch():
    with pytest.raises(ValueError, match="follower rank disagrees"):
        source_rows_for_live_order(
            np.array([11, 22]),
            np.array([1, 2]),
            np.array([22, 11]),
            np.array([1, 1]),
        )


def test_relion_euler_geodesic_handles_wrap_and_known_angle():
    np.testing.assert_allclose(
        angular_error_deg(np.array([[0.0, 0.0, 0.0]]), np.array([[360.0, 0.0, 0.0]])),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        angular_error_deg(np.array([[0.0, 0.0, 0.0]]), np.array([[90.0, 0.0, 0.0]])),
        90.0,
        atol=1e-12,
    )


def test_analysis_aligns_recovar_by_stack_and_reports_panel():
    source_stacks = np.array([10, 20, 30, 40])
    halves = np.array([1, 2, 1, 2])
    live_stacks = np.array([30, 10, 40, 20])
    live_ranks = np.array([1, 1, 2, 2])
    metadata_input = np.zeros((4, 31), dtype=np.float64)
    metadata_output = np.zeros((4, 31), dtype=np.float64)
    metadata_output[:, 8] = [0.3, 0.1, 0.4, 0.2]
    metadata_output[:, 9] = [3, 1, 4, 2]
    metadata_output[:, 3:5] = [[3, 0], [1, 0], [4, 0], [2, 0]]
    recovar_pmax = np.array([0.11, 0.22, 0.33, 0.44])
    recovar_eulers = np.zeros((4, 3), dtype=np.float64)
    recovar_translations = np.array([[1.1, 0], [2.2, 0], [3.3, 0], [4.4, 0]])

    report, arrays = analyze_arrays(
        source_stack_indices=source_stacks,
        source_random_subsets=halves,
        live_stack_indices=live_stacks,
        live_follower_ranks=live_ranks,
        metadata_input=metadata_input,
        metadata_output=metadata_output,
        recovar_pmax_by_image=recovar_pmax,
        recovar_eulers_by_image=recovar_eulers,
        recovar_translations_by_image=recovar_translations,
        panel_stack_indices=np.array([10, 40]),
    )

    np.testing.assert_allclose(arrays["recovar_pmax"], [0.33, 0.11, 0.44, 0.22])
    np.testing.assert_allclose(
        arrays["pmax_delta_recovar_minus_relion"], [0.03, 0.01, 0.04, 0.02]
    )
    np.testing.assert_allclose(arrays["translation_error_pixels"], [0.3, 0.1, 0.4, 0.2])
    assert report["cohorts"]["declared_panel"]["particle_count"] == 2
    assert report["cohorts"]["all"]["pmax_delta_recovar_minus_relion"]["nonzero_count"] == 4
    assert report["largest_discrepancies"]["translation_norm_pixels"][0][
        "stack_index_one_based"
    ] == 40
