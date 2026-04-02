"""Unit tests for RELION-style resolution criterion and current_size growth.

Tests the functions added in Tasks C4 and C5:
- compute_data_vs_prior
- resolution_from_data_vs_prior
- compute_current_size_relion
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.reconstruction.regularization import (
    compute_data_vs_prior,
    resolution_from_data_vs_prior,
    compute_current_size_relion,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# compute_data_vs_prior
# ---------------------------------------------------------------------------


class TestComputeDataVsPrior:
    """Test compute_data_vs_prior shell averaging and scaling."""

    def test_shape_matches_n_shells(self):
        """Output should have n_shells = volume_shape[0] // 2 - 1 entries."""
        volume_shape = (16, 16, 16)
        n_voxels = 16 ** 3
        Ft_ctf = jnp.ones(n_voxels, dtype=jnp.float32) * 10.0
        n_shells = volume_shape[0] // 2 - 1
        tau2 = jnp.ones(n_shells, dtype=jnp.float32)
        dvp = compute_data_vs_prior(Ft_ctf, tau2, volume_shape)
        assert dvp.shape == (n_shells,)

    def test_zero_tau_gives_zero(self):
        """If tau2 = 0 everywhere, data_vs_prior should be 0."""
        volume_shape = (16, 16, 16)
        n_voxels = 16 ** 3
        Ft_ctf = jnp.ones(n_voxels, dtype=jnp.float32) * 5.0
        n_shells = volume_shape[0] // 2 - 1
        tau2 = jnp.zeros(n_shells, dtype=jnp.float32)
        dvp = compute_data_vs_prior(Ft_ctf, tau2, volume_shape)
        np.testing.assert_allclose(np.array(dvp), 0.0, atol=1e-7)

    def test_zero_weight_gives_zero(self):
        """If Ft_ctf = 0, data_vs_prior should be 0 regardless of tau2."""
        volume_shape = (16, 16, 16)
        n_voxels = 16 ** 3
        Ft_ctf = jnp.zeros(n_voxels, dtype=jnp.float32)
        n_shells = volume_shape[0] // 2 - 1
        tau2 = jnp.ones(n_shells, dtype=jnp.float32) * 100.0
        dvp = compute_data_vs_prior(Ft_ctf, tau2, volume_shape)
        np.testing.assert_allclose(np.array(dvp), 0.0, atol=1e-7)

    def test_padding_factor_scaling(self):
        """padding_factor should scale result by padding_factor^3."""
        volume_shape = (16, 16, 16)
        n_voxels = 16 ** 3
        Ft_ctf = jnp.ones(n_voxels, dtype=jnp.float32) * 2.0
        n_shells = volume_shape[0] // 2 - 1
        tau2 = jnp.ones(n_shells, dtype=jnp.float32) * 3.0

        dvp_1 = compute_data_vs_prior(Ft_ctf, tau2, volume_shape, padding_factor=1)
        dvp_2 = compute_data_vs_prior(Ft_ctf, tau2, volume_shape, padding_factor=2)
        # dvp_2 should be 8x dvp_1
        np.testing.assert_allclose(
            np.array(dvp_2), np.array(dvp_1) * 8.0, rtol=1e-5,
        )

    def test_positive_inputs_give_positive_output(self):
        """Positive weight and tau2 should give positive data_vs_prior."""
        volume_shape = (16, 16, 16)
        n_voxels = 16 ** 3
        Ft_ctf = jnp.ones(n_voxels, dtype=jnp.float32) * 5.0
        n_shells = volume_shape[0] // 2 - 1
        tau2 = jnp.ones(n_shells, dtype=jnp.float32) * 2.0
        dvp = compute_data_vs_prior(Ft_ctf, tau2, volume_shape)
        assert jnp.all(dvp > 0), "All shells should have positive data_vs_prior"


# ---------------------------------------------------------------------------
# resolution_from_data_vs_prior
# ---------------------------------------------------------------------------


class TestResolutionFromDataVsPrior:
    """Test resolution_from_data_vs_prior shell finding."""

    def test_all_above_one_returns_last_shell(self):
        """If data_vs_prior > 1 everywhere, return the last shell."""
        dvp = np.array([10.0, 5.0, 3.0, 2.0, 1.5])
        assert resolution_from_data_vs_prior(dvp) == 4

    def test_drops_at_shell_3(self):
        """data_vs_prior drops below 1 at shell 3 -> return shell 2."""
        dvp = np.array([10.0, 5.0, 2.0, 0.5, 0.1])
        assert resolution_from_data_vs_prior(dvp) == 2

    def test_drops_at_shell_1(self):
        """data_vs_prior drops below 1 at shell 1 -> return shell 0."""
        dvp = np.array([10.0, 0.5, 0.1, 0.01])
        assert resolution_from_data_vs_prior(dvp) == 0

    def test_all_below_one_returns_zero(self):
        """If data_vs_prior < 1 from shell 1 onward, return 0."""
        dvp = np.array([0.5, 0.3, 0.1])
        assert resolution_from_data_vs_prior(dvp) == 0

    def test_single_shell(self):
        """Single-element array: return shell 0."""
        dvp = np.array([5.0])
        assert resolution_from_data_vs_prior(dvp) == 0

    def test_jax_array_input(self):
        """Should work with JAX arrays too."""
        dvp = jnp.array([10.0, 5.0, 0.5, 0.1])
        assert resolution_from_data_vs_prior(dvp) == 1


# ---------------------------------------------------------------------------
# compute_current_size_relion
# ---------------------------------------------------------------------------


class TestComputeCurrentSizeRelion:
    """Test RELION-style current_size growth logic."""

    def test_conservative_growth(self):
        """Low ave_Pmax -> conservative growth by incr_size shells."""
        # resolution_shell=10, ori_size=128, incr_size=10
        # maxres = 10 + 10 = 20, current_size = min(40, 128) = 40
        cs = compute_current_size_relion(10, 128, ave_Pmax=0.05)
        assert cs == 40

    def test_aggressive_growth(self):
        """High ave_Pmax + high FSC -> aggressive jump of 25% of ori_size/2."""
        # resolution_shell=10, ori_size=128
        # maxres = 10 + round(0.25 * 64) = 10 + 16 = 26
        # current_size = min(52, 128) = 52
        cs = compute_current_size_relion(
            10, 128, ave_Pmax=0.5, has_high_fsc_at_limit=True
        )
        assert cs == 52

    def test_clamp_to_ori_size(self):
        """Result should never exceed ori_size."""
        cs = compute_current_size_relion(100, 128, ave_Pmax=0.0)
        assert cs == 128

    def test_aggressive_clamp_to_ori_size(self):
        """Even aggressive growth should clamp to ori_size."""
        cs = compute_current_size_relion(
            60, 128, ave_Pmax=0.9, has_high_fsc_at_limit=True
        )
        # maxres = 60 + round(0.25 * 64) = 60 + 16 = 76
        # 2 * 76 = 152 -> clamped to 128
        assert cs == 128

    def test_ave_pmax_threshold(self):
        """ave_Pmax exactly at threshold (0.1) should NOT trigger aggressive growth."""
        cs = compute_current_size_relion(
            10, 128, ave_Pmax=0.1, has_high_fsc_at_limit=True
        )
        # 0.1 is NOT > 0.1, so conservative: 10 + 10 = 20, cs = 40
        assert cs == 40

    def test_high_pmax_no_fsc(self):
        """High ave_Pmax but no high FSC at limit -> conservative growth."""
        cs = compute_current_size_relion(
            10, 128, ave_Pmax=0.9, has_high_fsc_at_limit=False
        )
        assert cs == 40

    def test_custom_incr_size(self):
        """Custom incr_size should be used for conservative growth."""
        cs = compute_current_size_relion(
            10, 128, ave_Pmax=0.0, incr_size=5
        )
        # maxres = 10 + 5 = 15, cs = min(30, 128) = 30
        assert cs == 30

    def test_resolution_shell_zero(self):
        """Starting from shell 0 with default incr_size."""
        cs = compute_current_size_relion(0, 128)
        # maxres = 0 + 10 = 10, cs = min(20, 128) = 20
        assert cs == 20

    def test_small_ori_size(self):
        """Works correctly for small images (e.g. 32px)."""
        cs = compute_current_size_relion(5, 32, ave_Pmax=0.0)
        # maxres = 5 + 10 = 15, cs = min(30, 32) = 30
        assert cs == 30

    def test_aggressive_small_ori_size(self):
        """Aggressive growth for small images."""
        cs = compute_current_size_relion(
            5, 32, ave_Pmax=0.5, has_high_fsc_at_limit=True
        )
        # maxres = 5 + round(0.25 * 16) = 5 + 4 = 9
        # cs = min(18, 32) = 18
        assert cs == 18
