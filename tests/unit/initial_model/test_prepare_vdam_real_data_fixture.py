from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_vdam_real_data_fixture import select_balanced_half_indices


def _particles(halves):
    return pd.DataFrame(
        {
            "rlnImageName": [f"{index + 1}@particles.mrcs" for index in range(len(halves))],
            "rlnRandomSubset": halves,
        }
    )


def test_balanced_half_selection_is_deterministic_sorted_and_exactly_balanced():
    particles = _particles([1] * 8 + [2] * 8)

    first = select_balanced_half_indices(particles, particles_per_half=4, seed=23)
    second = select_balanced_half_indices(particles, particles_per_half=4, seed=23)

    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first) > 0)
    selected_halves = np.asarray(particles.iloc[first]["rlnRandomSubset"])
    assert np.count_nonzero(selected_halves == 1) == 4
    assert np.count_nonzero(selected_halves == 2) == 4


def test_balanced_half_selection_rejects_insufficient_half():
    particles = _particles([1] * 5 + [2] * 2)

    with pytest.raises(ValueError, match="half 2 contains 2 particles"):
        select_balanced_half_indices(particles, particles_per_half=3, seed=0)


def test_balanced_half_selection_rejects_invalid_subset_identifiers():
    particles = _particles([1, 1, 2, 3])

    with pytest.raises(ValueError, match="only RELION half identifiers 1 and 2"):
        select_balanced_half_indices(particles, particles_per_half=1, seed=0)
