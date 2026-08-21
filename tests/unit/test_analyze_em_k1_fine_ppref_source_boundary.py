from __future__ import annotations

import numpy as np

from scripts.analyze_em_k1_fine_ppref_source_boundary import (
    _array_metrics,
    classify_source_boundary,
)


def test_array_metrics_report_exact_and_direct_errors() -> None:
    target = np.asarray([1.0 + 2.0j, -3.0 + 0.5j])
    exact = _array_metrics(target.copy(), target)
    assert exact["array_equal"]
    assert exact["relative_l2_lhs_minus_rhs_over_rhs"] == 0.0
    assert exact["max_abs"] == 0.0

    changed = _array_metrics(target + np.asarray([0.0, 1.0]), target)
    assert not changed["array_equal"]
    assert changed["relative_l2_lhs_minus_rhs_over_rhs"] > 0.0
    assert changed["max_abs"] == 1.0


def test_classifies_iteration_start_map_state_boundary() -> None:
    assert (
        classify_source_boundary(
            frozen_relion_texture_exact=True,
            relion_map_rebuild_relative_l2=1.1e-8,
            recovar_map_replay_relative_l2=4.2e-9,
            cross_engine_projection_relative_l2=2.5e-3,
            map_states_equal=False,
        )
        == "fine_projection_difference_is_iteration_start_map_state"
    )


def test_classification_keeps_unclosed_texture_or_rebuild_boundaries_open() -> None:
    assert (
        classify_source_boundary(
            frozen_relion_texture_exact=False,
            relion_map_rebuild_relative_l2=1.0e-9,
            recovar_map_replay_relative_l2=1.0e-9,
            cross_engine_projection_relative_l2=1.0e-3,
            map_states_equal=False,
        )
        == "texture_projection_boundary_remains_open"
    )
    assert (
        classify_source_boundary(
            frozen_relion_texture_exact=True,
            relion_map_rebuild_relative_l2=2.0e-7,
            recovar_map_replay_relative_l2=1.0e-9,
            cross_engine_projection_relative_l2=1.0e-3,
            map_states_equal=False,
        )
        == "map_to_ppref_or_replay_boundary_remains_open"
    )
