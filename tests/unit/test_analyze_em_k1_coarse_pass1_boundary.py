from __future__ import annotations

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_rotations_to_recovar,
    _map_relion_table,
    _relion_prior_support,
    _score_gate,
    _stats,
    _translation_permutation,
)


def test_direction_major_relion_grid_maps_to_psi_major_recovar_grid() -> None:
    relion = np.asarray(
        [
            [0, 1],
            [10, 11],
            [20, 21],
            [30, 31],
            [40, 41],
            [50, 51],
        ]
    )
    mapped = _map_relion_rotations_to_recovar(
        relion,
        n_directions=3,
        n_psi=2,
    )
    np.testing.assert_array_equal(
        mapped,
        relion[[0, 2, 4, 1, 3, 5]],
    )


def test_translation_mapping_recovers_offset_scale_and_permutation() -> None:
    recovar = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]]
    )
    source_order = np.asarray([3, 1, 4, 0, 2])
    relion = (recovar[source_order] - np.asarray([0.25, -0.5])) * 2.0
    permutation, details = _translation_permutation(relion, recovar)
    relion_values = np.arange(10).reshape(2, 5)
    mapped = _map_relion_table(
        relion_values,
        n_directions=1,
        n_psi=2,
        relion_to_recovar_translation=permutation,
    )
    expected = np.empty_like(relion_values)
    expected[:, permutation] = relion_values
    np.testing.assert_array_equal(mapped, expected)
    assert details["max_coordinate_error"] <= 1e-12


def test_registered_score_gate_uses_fixed_p95_and_strict_max() -> None:
    passing = _stats(np.asarray([0.0] * 99 + [9e-4]))
    failing_p95 = _stats(np.asarray([0.0] * 90 + [2e-4] * 10))
    failing_max = _stats(np.asarray([0.0] * 99 + [1e-3]))
    assert _score_gate(passing)
    assert not _score_gate(failing_p95)
    assert not _score_gate(failing_max)


def test_relion_prior_support_excludes_only_lowest_float32_sentinel() -> None:
    values = np.asarray(
        [np.finfo(np.float32).min, -100.0, 0.0, np.inf],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        _relion_prior_support(values),
        np.asarray([False, True, True, True]),
    )
