"""M9: Noise update formula parity — RELION vs recovar.

Tests that update_noise_estimate produces exactly the same sigma2_noise
as recovar's normalize_wsum_to_sigma2_noise.

Exact parity required: rel_err < 1e-12.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import update_noise_estimate


def _reference_noise_update(wsum_sigma2, npix_per_shell, sum_weight):
    """Pure numpy reference matching RELION ml_optimiser.cpp:5246-5286."""
    n = len(wsum_sigma2)
    out = np.zeros(n)
    for i in range(n):
        denom = 2.0 * sum_weight * npix_per_shell[i]
        if denom > 0:
            out[i] = wsum_sigma2[i] / denom
        else:
            out[i] = 0.0
        if out[i] < 1e-15:
            out[i] = 1e-15
        if out[i] < 1e-14 and i > 0:
            out[i] = out[i - 1]
    return out


class TestM9Parity:
    """Exact parity between RELION binding and numpy reference."""

    def test_basic(self):
        rng = np.random.default_rng(42)
        n_shells = 65
        wsum = rng.uniform(1e3, 1e6, n_shells)
        npix = np.arange(1, n_shells + 1, dtype=np.float64) * 6.0
        sumw = 5000.0

        relion = update_noise_estimate(wsum, npix, sumw)
        reference = _reference_noise_update(wsum, npix, sumw)

        max_diff = np.max(np.abs(relion - reference))
        assert max_diff < 1e-15, f"max_diff={max_diff}"

    def test_zero_shells(self):
        """Shells with zero npix should get clamped to 1e-15."""
        n_shells = 10
        wsum = np.ones(n_shells) * 100.0
        npix = np.zeros(n_shells)
        npix[0:5] = 10.0
        sumw = 100.0

        relion = update_noise_estimate(wsum, npix, sumw)
        reference = _reference_noise_update(wsum, npix, sumw)

        np.testing.assert_array_equal(relion, reference)

    def test_hole_filling(self):
        """Very small values should be filled from previous shell."""
        n_shells = 10
        wsum = np.array([1e6, 1e6, 1e-20, 1e-20, 1e6, 1e6, 1e-20, 1e6, 1e6, 1e6])
        npix = np.ones(n_shells) * 100.0
        sumw = 1000.0

        relion = update_noise_estimate(wsum, npix, sumw)
        reference = _reference_noise_update(wsum, npix, sumw)

        np.testing.assert_array_equal(relion, reference)

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed):
        rng = np.random.default_rng(seed)
        n_shells = rng.integers(10, 65)
        wsum = rng.exponential(1e4, n_shells)
        npix = np.arange(1, n_shells + 1, dtype=np.float64) * rng.uniform(2, 10)
        sumw = rng.uniform(100, 10000)

        relion = update_noise_estimate(wsum, npix, sumw)
        reference = _reference_noise_update(wsum, npix, sumw)

        max_diff = np.max(np.abs(relion - reference))
        assert max_diff < 1e-15, f"seed={seed}: max_diff={max_diff}"
