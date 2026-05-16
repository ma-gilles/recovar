"""Phase 1: M1 gridding correction — build smoke test + parity check.

Tests that RELION's Projector::griddingCorrect (radial sinc²) produces
the expected correction pattern.  Also compares against recovar's
implementation in relion_functions.py.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import (
    NEAREST_NEIGHBOUR,
    TRILINEAR,
    gridding_correct,
)


class TestGriddingCorrectSmoke:
    """Build smoke tests — verifies the C++ binding loads and runs."""

    def test_returns_correct_shape(self):
        vol = np.ones((16, 16, 16))
        result = gridding_correct(vol, ori_size=16, padding_factor=1, interpolator=TRILINEAR)
        assert result.shape == (16, 16, 16)

    def test_does_not_modify_input(self):
        vol = np.ones((16, 16, 16))
        vol_copy = vol.copy()
        gridding_correct(vol, ori_size=16, padding_factor=1, interpolator=TRILINEAR)
        np.testing.assert_array_equal(vol, vol_copy)

    def test_dc_unchanged(self):
        """The center voxel (r=0) should be unchanged (sinc(0)=1)."""
        vol = np.zeros((16, 16, 16))
        vol[8, 8, 8] = 1.0
        result = gridding_correct(vol, ori_size=16, padding_factor=1, interpolator=TRILINEAR)
        np.testing.assert_allclose(result[8, 8, 8], 1.0, atol=1e-14)


class TestGriddingCorrectFormula:
    """Verify the formula: divide by sinc²(r / (N * pf)) for trilinear."""

    @pytest.mark.parametrize("N", [16, 32, 64])
    @pytest.mark.parametrize("pf", [1, 2])
    def test_trilinear_formula(self, N, pf):
        vol = np.ones((N, N, N))
        result = gridding_correct(vol, ori_size=N, padding_factor=pf, interpolator=TRILINEAR)

        idx = np.arange(N) - N // 2
        kz, ky, kx = np.meshgrid(idx, idx, idx, indexing="ij")
        r = np.sqrt(kz**2.0 + ky**2.0 + kx**2.0)
        rval = r / (N * pf)
        sinc = np.where(r > 0, np.sin(np.pi * rval) / (np.pi * rval), 1.0)
        expected = np.where(r > 0, 1.0 / (sinc * sinc), 1.0)

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.parametrize("N", [16, 32])
    def test_nearest_neighbour_with_r_min_nn_gt0(self, N):
        """When r_min_nn > 0 (default=10), NN also uses sinc² correction."""
        vol = np.ones((N, N, N))
        result = gridding_correct(vol, ori_size=N, padding_factor=1, interpolator=NEAREST_NEIGHBOUR)

        idx = np.arange(N) - N // 2
        kz, ky, kx = np.meshgrid(idx, idx, idx, indexing="ij")
        r = np.sqrt(kz**2.0 + ky**2.0 + kx**2.0)
        rval = r / N
        sinc = np.where(r > 0, np.sin(np.pi * rval) / (np.pi * rval), 1.0)
        expected = np.where(r > 0, 1.0 / (sinc * sinc), 1.0)

        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestGriddingCorrectVsRecovar:
    """Compare RELION's radial sinc² against recovar's implementation."""

    @pytest.mark.parametrize("N", [16, 32, 64])
    def test_radial_matches_recovar(self, N):
        """recovar's griddingCorrect (radial mode) should match RELION exactly."""
        from recovar.reconstruction.relion_functions import griddingCorrect as recovar_gc

        vol = np.ones((N, N, N))
        relion_result = gridding_correct(vol, ori_size=N, padding_factor=1, interpolator=TRILINEAR)
        recovar_result, _ = recovar_gc(vol.copy(), N, padding_factor=1, order=1)

        np.testing.assert_allclose(recovar_result, relion_result, rtol=1e-12)
