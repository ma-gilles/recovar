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
    result = update_c1_sigma_offset_from_posterior(
        noise_stats_per_half=[_noise_stats(12.0, 2.0), _noise_stats(40.0, 2.0)],
        noise_stats_per_half_per_class=[None, None],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        n_classes=1,
        k_class_enabled=False,
        state_fallback_offsets_angstrom=float("nan"),
    )
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([np.sqrt(3.0), np.sqrt(10.0)])
    assert result.current_sigma_offset_angstrom == pytest.approx((np.sqrt(3.0) + np.sqrt(10.0)) / 2.0)


def test_kclass_sigma_offset_is_shared_across_halves():
    result = update_c1_sigma_offset_from_posterior(
        noise_stats_per_half=[_noise_stats(12.0, 2.0), _noise_stats(40.0, 2.0)],
        noise_stats_per_half_per_class=[None, None],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        n_classes=1,
        k_class_enabled=True,
        state_fallback_offsets_angstrom=float("nan"),
    )
    expected = np.sqrt(52.0 / 8.0)
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([expected, expected])


def test_missing_half_uses_hard_assignment_fallback_independently():
    result = update_c1_sigma_offset_from_posterior(
        noise_stats_per_half=[_noise_stats(0.0, 0.0), _noise_stats(40.0, 2.0)],
        noise_stats_per_half_per_class=[None, None],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        n_classes=1,
        k_class_enabled=False,
        state_fallback_offsets_angstrom=3.0,
    )
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([3.0, np.sqrt(10.0)])


def test_sigma_offset_half_pair_normalizes_scalar_and_pair():
    assert _as_sigma_offset_half_pair(1.5) == [1.5, 1.5]
    assert _as_sigma_offset_half_pair([1.5, 2.5]) == [1.5, 2.5]
    with pytest.raises(ValueError, match="exactly two"):
        _as_sigma_offset_half_pair([1.0, 2.0, 3.0])
