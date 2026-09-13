"""Convergence inputs have owners: combined assignment stacks and the optimizer-Pmax mass.

``helpers.convergence`` joins both half-sets' assignment indices for
convergence tracking; ``mean_helpers`` selects the per-half normalization mass
that ``_relion_optimizer_average_pmax`` divides by.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.helpers.convergence import concatenate_assignments, concatenate_assignments_or_none
from recovar.em.refinement import mean_helpers

pytestmark = pytest.mark.unit


def test_concatenate_assignments_joins_halves_in_order_as_int32():
    half1 = np.asarray([3, 1], dtype=np.int64)
    half2 = np.asarray([7.0, 0.0, 2.0], dtype=np.float32)
    result = concatenate_assignments([half1, half2])
    assert result.dtype == np.int32
    assert result.tolist() == [3, 1, 7, 0, 2]


def test_concatenate_assignments_requires_every_half():
    with pytest.raises(TypeError):
        concatenate_assignments([np.zeros(2, dtype=np.int32), None])


@pytest.mark.parametrize("missing", [[None, np.zeros(2, dtype=np.int32)], [np.zeros(2, dtype=np.int32), None], [None, None]])
def test_concatenate_assignments_or_none_returns_none_for_missing_half(missing):
    assert concatenate_assignments_or_none(missing) is None


def test_concatenate_assignments_or_none_matches_the_strict_join():
    halves = [np.asarray([1, 2], dtype=np.int32), np.asarray([4], dtype=np.int32)]
    result = concatenate_assignments_or_none(halves)
    assert result.tobytes() == concatenate_assignments(halves).tobytes()
    assert result.dtype == np.int32 and result.tolist() == [1, 2, 4]


def test_kclass_pmax_mass_is_the_retained_posterior_mass_per_half():
    class_posterior_per_half = [np.asarray([0.25, 0.5, 0.125], dtype=np.float32), np.asarray([1.0, 2.0], dtype=np.float64)]
    result = mean_helpers._relion_pmax_normalization_mass_per_half(
        k_class_enabled=True, class_posterior_per_half=class_posterior_per_half, noise_stats_per_half=[None, None]
    )
    assert result == [0.875, 3.0]
    assert all(type(value) is float for value in result)


def test_k1_pmax_mass_is_the_noise_sumw_or_none():
    noise_stats_per_half = [SimpleNamespace(sumw=np.float32(12.5)), None]
    result = mean_helpers._relion_pmax_normalization_mass_per_half(
        k_class_enabled=False, class_posterior_per_half=[None, None], noise_stats_per_half=noise_stats_per_half
    )
    assert result == [12.5, None]
    assert type(result[0]) is float


def test_pmax_mass_feeds_the_optimizer_average():
    max_posterior_per_half = [np.asarray([0.5, 0.25], dtype=np.float32), np.asarray([1.0], dtype=np.float32)]
    mass = mean_helpers._relion_pmax_normalization_mass_per_half(
        k_class_enabled=False,
        class_posterior_per_half=[None, None],
        noise_stats_per_half=[SimpleNamespace(sumw=3.0), SimpleNamespace(sumw=1.0)],
    )
    combined, average, denominator = mean_helpers._relion_optimizer_average_pmax(max_posterior_per_half, mass)
    assert denominator == 3.0 and average == pytest.approx(0.25)
    assert combined.tolist() == [0.5, 0.25, 1.0]
