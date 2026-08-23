"""Regression tests for the K=4 fixed-state target translation frame."""

import numpy as np
import pytest

from scripts.analyze_k4_fixed_state_target import (
    relative_to_absolute_translations,
    relion_round_away_from_zero,
)

pytestmark = pytest.mark.unit


def test_relion_round_away_from_zero_preserves_tie_policy():
    values = np.asarray([-3.5, -3.49, -0.5, 0.5, 3.49, 3.5])
    expected = np.asarray([-4.0, -3.0, -1.0, 1.0, 3.0, 4.0], dtype=np.float32)

    np.testing.assert_array_equal(relion_round_away_from_zero(values), expected)


def test_relative_pass2_winner_maps_to_absolute_phase_winner():
    previous_absolute = np.asarray(
        [-7.66883 / 2.125, -1.29383 / 2.125],
        dtype=np.float64,
    )
    relative_candidates = np.asarray(
        [
            [2.0410656929016113, 0.04106569290161133],
            [3.0410656929016113, 0.04106569290161133],
        ],
        dtype=np.float64,
    )

    absolute = relative_to_absolute_translations(
        relative_candidates,
        previous_absolute,
    )

    np.testing.assert_array_equal(
        absolute,
        np.asarray(
            [
                [-1.9589343070983887, -0.9589343070983887],
                [-0.9589343070983887, -0.9589343070983887],
            ],
            dtype=np.float64,
        ),
    )
