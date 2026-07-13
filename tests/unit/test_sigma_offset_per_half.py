import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.types import make_noise_stats
from recovar.em.dense_single_volume.mean_helpers import update_c1_sigma_offset_from_posterior
from recovar.em.dense_single_volume.relion_replay import _as_sigma_offset_half_pair

pytestmark = pytest.mark.unit


def _noise_stats(wsum_sigma2_offset, sumw, n_shells=4):
    return make_noise_stats(
        wsum_sigma2_noise=np.ones(n_shells),
        wsum_img_power=np.ones(n_shells),
        wsum_sigma2_offset=wsum_sigma2_offset,
        sumw=sumw,
    )


def test_non_kclass_sigma_offset_is_independent_per_half():
    """Each gold-standard half must use only its own posterior moment."""
    noise_stats_per_half = [_noise_stats(12.0, 2.0), _noise_stats(40.0, 2.0)]

    result = update_c1_sigma_offset_from_posterior(
        noise_stats_per_half=noise_stats_per_half,
        noise_stats_per_half_per_class=[None, None],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        n_classes=1,
        k_class_enabled=False,
        state_fallback_offsets_angstrom=float("nan"),
    )

    expected_h1 = np.sqrt(max(12.0 / (2.0 * 2.0), 2.0))
    expected_h2 = np.sqrt(max(40.0 / (2.0 * 2.0), 2.0))
    assert result.current_sigma_offset_angstrom_per_half[0] == pytest.approx(expected_h1)
    assert result.current_sigma_offset_angstrom_per_half[1] == pytest.approx(expected_h2)
    assert result.current_sigma_offset_angstrom_per_half[0] != pytest.approx(
        result.current_sigma_offset_angstrom_per_half[1]
    )


def test_kclass_sigma_offset_is_shared_across_halves():
    """K-class classification has no gold-standard split; both halves pool."""
    noise_stats_per_half = [_noise_stats(12.0, 2.0), _noise_stats(40.0, 2.0)]

    result = update_c1_sigma_offset_from_posterior(
        noise_stats_per_half=noise_stats_per_half,
        noise_stats_per_half_per_class=[None, None],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        n_classes=1,
        k_class_enabled=True,
        state_fallback_offsets_angstrom=float("nan"),
    )

    expected_shared = np.sqrt(max((12.0 + 40.0) / (2.0 * (2.0 + 2.0)), 2.0))
    assert result.current_sigma_offset_angstrom_per_half[0] == pytest.approx(expected_shared)
    assert result.current_sigma_offset_angstrom_per_half[1] == pytest.approx(expected_shared)


def test_sigma_offset_falls_back_per_half_when_moment_missing():
    """A half without a posterior moment falls back independently, not via the other half's data."""
    noise_stats_per_half = [_noise_stats(0.0, 0.0), _noise_stats(40.0, 2.0)]

    result = update_c1_sigma_offset_from_posterior(
        noise_stats_per_half=noise_stats_per_half,
        noise_stats_per_half_per_class=[None, None],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        n_classes=1,
        k_class_enabled=False,
        state_fallback_offsets_angstrom=3.0,
    )

    assert result.current_sigma_offset_angstrom_per_half[0] == pytest.approx(3.0)
    assert result.current_sigma_offset_angstrom_per_half[1] == pytest.approx(np.sqrt(10.0))


def test_sigma_offset_half_pair_normalizes_scalar_and_pair():
    assert _as_sigma_offset_half_pair(1.5) == [1.5, 1.5]
    assert _as_sigma_offset_half_pair([1.5, 2.5]) == [1.5, 2.5]
    assert _as_sigma_offset_half_pair((1.5, 2.5)) == [1.5, 2.5]
