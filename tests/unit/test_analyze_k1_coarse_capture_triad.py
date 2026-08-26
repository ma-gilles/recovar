import numpy as np

from scripts.analyze_k1_coarse_capture_triad import (
    _arm_summary,
    _coordinate,
    _score_components,
    _subtract_components,
)
from scripts.analyze_k1_coarse_operand_swap import _restore_square_references, _shapley


def _payload():
    raw = np.asarray([[[4.0, 3.0], [2.0, 1.0]]], dtype=np.float32)
    prior = np.asarray([[[0.0, 0.25], [0.5, 0.75]]], dtype=np.float32)
    return {
        "weights_full": np.asarray([0.6, 0.25, 0.1, 0.05], dtype=np.float64),
        "significant_mask": np.asarray([True, True, False, False]),
        "n_significant": np.asarray(2),
        "max_posterior": np.asarray(0.6),
        "hard_assignment": np.asarray(0),
        "scores_pre_prior_per_class": raw,
        "scores_with_prior_per_class": raw + prior,
    }


def test_arm_summary_tracks_stable_ranks_and_masses():
    result = _arm_summary(
        _payload(),
        reference_top_count=2,
        tracked_flat_indices={"winner": 0, "excluded": 2},
    )

    assert result["own_significant_mass"] == 0.85
    assert result["top_reference_count_mass"] == 0.85
    assert result["tracked"]["winner"] == {
        "flat_index": 0,
        "rank": 1,
        "posterior": 0.6,
        "selected": True,
    }
    assert result["tracked"]["excluded"]["rank"] == 3


def test_score_components_separate_raw_and_prior_margins():
    result = _score_components(_payload(), flat_index=2, anchor_flat_index=0)

    assert result == {
        "raw": 2.0,
        "prior": 0.5,
        "total": 2.5,
        "raw_margin_to_exact_winner": -2.0,
        "prior_margin_to_exact_winner": 0.5,
        "total_margin_to_exact_winner": -1.5,
    }
    assert _subtract_components(result, {name: value + 0.25 for name, value in result.items()}) == {
        name: 0.25 for name in result
    }


def test_coordinate_uses_rotation_major_flattening():
    assert _coordinate(738414, 29) == (25462, 16)


def test_shapley_exactly_attributes_additive_three_factor_change():
    coefficients = {"image": 1.0, "weight": 2.0, "initial_diff2": 4.0}
    values = {
        frozenset(subset): 10.0 + sum(coefficients[factor] for factor in subset)
        for size in range(4)
        for subset in __import__("itertools").combinations(coefficients, size)
    }

    assert _shapley(values) == coefficients


def test_restore_square_references_matches_score_pixel_topology():
    image_shape = (8, 8)
    current_size = 6
    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_indices_np,
    )
    from recovar.em.dense_single_volume.helpers.significance import (
        _compact_projection_window_positions,
    )

    score_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        square=True,
        include_dc=False,
    )
    active_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        square=False,
        include_dc=False,
    )
    active_positions = _compact_projection_window_positions(score_indices, active_indices)
    compact = np.arange(active_positions.size, dtype=np.float32).astype(np.complex64)[None, :]

    restored = _restore_square_references(
        compact,
        image_shape=image_shape,
        current_size=current_size,
        score_indices=score_indices,
    )

    assert restored.shape == (1, score_indices.size)
    np.testing.assert_array_equal(restored[:, active_positions], compact)
    np.testing.assert_array_equal(
        restored[:, np.setdiff1d(np.arange(score_indices.size), active_positions)],
        0,
    )
