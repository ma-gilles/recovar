"""Pure RELION current-size and first-iteration scheduling contracts."""

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import resolution as resolution_helpers
from recovar.reconstruction import regularization as regularization_module

pytestmark = pytest.mark.unit


class TestResolutionScheduling:
    """Resolution policies independent of refinement dispatch and datasets."""

    def test_firstiter_cc_ini_high_tau2_taper_matches_relion_squared_cosine(self):
        taper = resolution_helpers._firstiter_cc_ini_high_tau2_taper(
            65,
            128,
            4.25,
            30.0,
            filter_edgewidth=2,
        )

        radius = 128 * 4.25 / 30.0 - 1.0
        radius_p = radius + 2.0
        expected18 = (0.5 - 0.5 * np.cos(np.pi * (radius_p - 18.0) / 2.0)) ** 2
        expected19 = (0.5 - 0.5 * np.cos(np.pi * (radius_p - 19.0) / 2.0)) ** 2
        assert taper[17] == 1.0
        np.testing.assert_allclose(taper[18], expected18, rtol=0, atol=1e-15)
        np.testing.assert_allclose(taper[19], expected19, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(taper[20:], 0.0)

    def test_k1_current_size_scheduling_raw_fsc_matches_gui_default(self):
        """GUI-default K=1 scheduling uses raw FSC-derived DVP."""
        raw_fsc = np.ones(129, dtype=np.float32) * 0.9
        raw_fsc[0] = 1.0
        raw_fsc[28:] = 0.0

        dvp = resolution_helpers._k1_data_vs_prior_for_scheduling(
            raw_fsc=raw_fsc,
            corrected_data_vs_prior=None,
            current_size=56,
            grid_size=256,
            tau2_fudge=1.0,
        )
        shell = regularization_module.resolution_from_data_vs_prior(
            dvp,
            allow_high_res_recovery=True,
        )
        current_size = regularization_module.compute_current_size_relion(
            shell,
            256,
            ave_Pmax=1.0,
            has_high_fsc_at_limit=True,
        )

        assert shell == 27
        assert current_size == 118

    def test_k1_current_size_scheduling_keeps_boundary_shell(self):
        """Raw-FSC and corrected-DVP scheduling must agree at current_size//2."""
        current_size = 56
        boundary_shell = current_size // 2
        raw_fsc = np.zeros(129, dtype=np.float32)
        corrected_dvp = np.zeros_like(raw_fsc)
        raw_fsc[:boundary_shell] = 0.05
        corrected_dvp[:boundary_shell] = 0.05
        raw_fsc[boundary_shell] = 0.9
        corrected_dvp[boundary_shell] = 10.0

        raw_dvp = resolution_helpers._k1_data_vs_prior_for_scheduling(
            raw_fsc=raw_fsc,
            corrected_data_vs_prior=None,
            current_size=current_size,
            grid_size=256,
            tau2_fudge=1.0,
        )
        corrected = resolution_helpers._k1_data_vs_prior_for_scheduling(
            raw_fsc=raw_fsc,
            corrected_data_vs_prior=corrected_dvp,
            current_size=current_size,
            grid_size=256,
            tau2_fudge=1.0,
        )

        assert raw_dvp[boundary_shell] > 1.0
        assert corrected[boundary_shell] > 1.0
        assert raw_dvp[boundary_shell + 1] < 1.0
        assert corrected[boundary_shell + 1] == 0.0
        assert (
            regularization_module.resolution_from_data_vs_prior(
                raw_dvp,
                allow_high_res_recovery=True,
            )
            == boundary_shell
        )
        assert (
            regularization_module.resolution_from_data_vs_prior(
                corrected,
                allow_high_res_recovery=True,
            )
            == boundary_shell
        )

    def test_current_resolution_state_keeps_boundary_shell(self):
        """The post-M-step DVP state must retain RELION's inclusive boundary shell."""
        current_size = 68
        boundary_shell = current_size // 2
        data_vs_prior = np.zeros((2, 65), dtype=np.float32)
        data_vs_prior[:, : boundary_shell + 2] = 0.25
        data_vs_prior[:, boundary_shell] = 10.0

        truncated = resolution_helpers._truncate_data_vs_prior_for_current_size(
            data_vs_prior,
            current_size=current_size,
            grid_size=128,
        )

        np.testing.assert_array_equal(
            truncated[:, : boundary_shell + 1],
            data_vs_prior[:, : boundary_shell + 1],
        )
        assert np.all(truncated[:, boundary_shell + 1 :] == 0.0)
        assert all(
            regularization_module.resolution_from_data_vs_prior(
                half_dvp,
                allow_high_res_recovery=True,
            )
            == boundary_shell
            for half_dvp in truncated
        )

    @pytest.mark.parametrize(
        ("current_size", "grid_size", "boundary_fsc", "expected_next_size"),
        [
            (68, 128, 0.9487659, 100),
            (100, 256, 0.5574918, 164),
        ],
    )
    def test_current_resolution_boundary_reproduces_matrix_growth(
        self,
        current_size,
        grid_size,
        boundary_fsc,
        expected_next_size,
    ):
        """Cases 2, 3, and 33 grow from RELION's supported boundary shell."""
        boundary_shell = current_size // 2
        fsc = np.zeros(grid_size // 2 + 1, dtype=np.float32)
        fsc[: boundary_shell + 1] = boundary_fsc
        fsc[0] = 1.0
        data_vs_prior = regularization_module.fsc_to_relion_ssnr(fsc, tau2_fudge=1.0)

        truncated = resolution_helpers._truncate_data_vs_prior_for_current_size(
            data_vs_prior,
            current_size=current_size,
            grid_size=grid_size,
        )
        resolution_shell = regularization_module.resolution_from_data_vs_prior(
            truncated,
            allow_high_res_recovery=True,
        )
        next_size = regularization_module.compute_current_size_relion(
            resolution_shell,
            grid_size,
            ave_Pmax=1.0,
            has_high_fsc_at_limit=True,
            incr_size=10,
        )

        assert resolution_shell == boundary_shell
        assert next_size == expected_next_size

    def test_k1_growth_fsc_keeps_case15_size68_boundary_shell(self):
        """Case 15 iter 8 must scan shell 34 and grow 68 -> 78 like RELION."""
        fsc = np.zeros(65, dtype=np.float32)
        fsc[1:25] = 0.6
        fsc[25:34] = 0.18
        fsc[33] = 0.184406
        fsc[34] = 0.159366

        growth_fsc = resolution_helpers._truncate_fsc_for_current_size_growth(
            fsc,
            current_size=68,
            grid_size=128,
        )
        incr_size, has_high_fsc = regularization_module.update_relion_growth_state_from_fsc(
            growth_fsc,
            68,
            incr_size=10,
            has_high_fsc_at_limit=False,
        )
        next_size = regularization_module.compute_current_size_relion(
            24,
            128,
            ave_Pmax=0.521225,
            has_high_fsc_at_limit=has_high_fsc,
            incr_size=incr_size,
        )

        assert growth_fsc[34] == pytest.approx(0.159366)
        assert growth_fsc[35] == 0.0
        assert regularization_module.first_shell_below_threshold(growth_fsc, 0.5) == 25
        assert regularization_module.first_shell_below_threshold(growth_fsc, 0.143) == 35
        assert incr_size == 15
        assert has_high_fsc is False
        assert next_size == 78

    def test_firstiter_cc_scheduling_uses_ini_high_shell(self):
        """RELION iter-1 firstiter_cc grows from ini_high, not DVP."""
        shell = resolution_helpers._firstiter_cc_ini_high_resolution_shell(256, 2.125, 30.0)
        current_size = regularization_module.compute_current_size_relion(
            shell,
            256,
            ave_Pmax=1.0,
            has_high_fsc_at_limit=True,
        )
        assert shell == 18
        assert current_size == 100

        raw_fsc = np.ones(129, dtype=np.float32) * 0.9
        raw_fsc[0] = 1.0
        raw_fsc[28:] = 0.0
        dvp = resolution_helpers._k1_data_vs_prior_for_scheduling(
            raw_fsc=raw_fsc,
            corrected_data_vs_prior=None,
            current_size=56,
            grid_size=256,
            tau2_fudge=1.0,
        )
        assert regularization_module.resolution_from_data_vs_prior(dvp, allow_high_res_recovery=True) == 27
        assert dvp[29] < 1.0

    def test_firstiter_cc_scheduling_override_is_class_count_independent(self):
        """The ini_high rule also applies to Class3D/K-class iteration 1."""
        shell = resolution_helpers._firstiter_cc_scheduling_resolution_shell(
            10,
            emulate_relion_firstiter_cc=True,
            ini_high_angstrom=60.0,
            relion_iteration=1,
            grid_size=256,
            voxel_size=2.125,
        )
        current_size = regularization_module.compute_current_size_relion(
            shell,
            256,
            ave_Pmax=1.0,
            has_high_fsc_at_limit=False,
            incr_size=10,
        )

        assert shell == 9
        assert current_size == 38
        assert resolution_helpers.shell_index_to_resolution_angstrom(
            shell,
            256,
            2.125,
        ) == pytest.approx(60.4444444444)

    def test_firstiter_cc_scheduling_override_is_only_physical_iteration_one(self):
        common = {
            "resolution_shell": 10,
            "grid_size": 256,
            "voxel_size": 2.125,
        }
        assert (
            resolution_helpers._firstiter_cc_scheduling_resolution_shell(
                **common,
                emulate_relion_firstiter_cc=True,
                ini_high_angstrom=60.0,
                relion_iteration=2,
            )
            == 10
        )
        assert (
            resolution_helpers._firstiter_cc_scheduling_resolution_shell(
                **common,
                emulate_relion_firstiter_cc=False,
                ini_high_angstrom=60.0,
                relion_iteration=1,
            )
            == 10
        )
        assert (
            resolution_helpers._firstiter_cc_scheduling_resolution_shell(
                **common,
                emulate_relion_firstiter_cc=True,
                ini_high_angstrom=None,
                relion_iteration=1,
            )
            == 10
        )
