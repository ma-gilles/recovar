"""Unit tests for convergence detection, angular step refinement, and local angular search.

Tests cover:
- RefinementState construction and properties
- Assignment change tracking
- Translation change tracking
- Average Pmax computation
- Convergence detection logic
- Angular step refinement triggers and parameter updates
- Local search activation
- Full update_refinement_state workflow
- get_rotation_grid_at_order from sampling.py
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Import targets
# ---------------------------------------------------------------------------

from recovar.em.dense_single_volume.helpers.convergence import (
    MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
    MAX_NR_ITER_WO_RESOL_GAIN,
    RefinementState,
    check_convergence,
    compute_assignment_changes,
    compute_ave_Pmax,
    compute_translation_changes,
    effective_angular_step,
    healpix_angular_step,
    refine_angular_sampling,
    resolution_required_angular_sampling,
    resolution_triggers_angular_refinement,
    should_refine_angular_sampling,
    update_refinement_state,
)
from recovar.em.sampling import (
    get_rotation_grid,
    get_rotation_grid_at_order,
)

# =========================================================================
# RefinementState construction
# =========================================================================


class TestRefinementStateConstruction:
    """Tests for RefinementState dataclass creation and defaults."""

    def test_default_construction(self):
        state = RefinementState()
        assert state.iteration == 0
        assert state.healpix_order == 2
        assert state.has_converged is False
        assert state.do_local_search is False
        assert state.best_rotations is None
        assert state.best_translations is None

    def test_angular_step_auto_computed(self):
        """angular_step is auto-computed from healpix_order when not specified."""
        state = RefinementState(healpix_order=3)
        expected = 360.0 / (6.0 * 2**3)
        assert abs(state.angular_step - expected) < 1e-10

    def test_angular_step_explicit(self):
        """Explicit angular_step overrides auto-computation."""
        state = RefinementState(healpix_order=3, angular_step=5.0)
        assert state.angular_step == 5.0

    def test_effective_step_property(self):
        state = RefinementState(healpix_order=3, adaptive_oversampling=1)
        expected = healpix_angular_step(3) / 2.0
        assert abs(state.effective_step - expected) < 1e-10

    def test_max_healpix_order_is_not_fine_enough_by_itself(self):
        state = RefinementState(healpix_order=7, max_healpix_order=7)
        assert state.has_fine_enough_angular_sampling is False

    def test_has_fine_enough_below_max(self):
        state = RefinementState(healpix_order=3, max_healpix_order=7)
        assert state.has_fine_enough_angular_sampling is False

    def test_has_fine_enough_from_acc_rot(self):
        state = RefinementState(healpix_order=7, max_healpix_order=7, acc_rot=1.0)
        assert state.has_fine_enough_angular_sampling is True

    def test_resolution_required_sampling_caps_loose_acc_rot(self):
        """Do not stop at order 2 when high resolution requires a finer grid."""
        state = RefinementState(
            healpix_order=2,
            adaptive_oversampling=1,
            acc_rot=12.247,
            current_resolution=4.86,
            particle_diameter_angstrom=544.0,
        )

        assert resolution_required_angular_sampling(4.86, 544.0) == pytest.approx(1.0227, rel=1e-3)
        assert state.has_fine_enough_angular_sampling is False

    def test_loose_acc_rot_still_applies_without_particle_diameter(self):
        state = RefinementState(
            healpix_order=2,
            adaptive_oversampling=1,
            acc_rot=12.247,
            current_resolution=4.86,
            particle_diameter_angstrom=0.0,
        )

        assert state.has_fine_enough_angular_sampling is True

    def test_should_do_local_search_at_order_4(self):
        state = RefinementState(healpix_order=4)
        assert state.should_do_local_search is True
        assert state.do_local_search is True

    def test_should_not_do_local_search_below_order_4(self):
        state = RefinementState(healpix_order=3)
        assert state.should_do_local_search is False
        assert state.do_local_search is False


# =========================================================================
# healpix_angular_step / effective_angular_step
# =========================================================================


class TestAngularStepFunctions:
    def test_healpix_angular_step_known_values(self):
        """Check against the known table in the RELION reference doc."""
        # Order 0: ~58.6 deg (360 / 6 = 60, close enough)
        assert abs(healpix_angular_step(0) - 60.0) < 1e-10
        # Order 3: ~7.5 deg
        assert abs(healpix_angular_step(3) - 7.5) < 1e-10
        # Order 4: ~3.75 deg
        assert abs(healpix_angular_step(4) - 3.75) < 1e-10

    def test_effective_angular_step_no_oversampling(self):
        assert effective_angular_step(3, 0) == healpix_angular_step(3)

    def test_effective_angular_step_with_oversampling(self):
        """Oversampling 1 halves the step, oversampling 2 quarters it."""
        step3 = healpix_angular_step(3)
        assert abs(effective_angular_step(3, 1) - step3 / 2) < 1e-10
        assert abs(effective_angular_step(3, 2) - step3 / 4) < 1e-10


# =========================================================================
# Assignment change tracking
# =========================================================================


class TestAssignmentChanges:
    def test_identical_assignments_zero_change(self):
        n_rot, n_trans = 100, 5
        assignments = np.arange(50) * n_trans + 2  # 50 images
        frac = compute_assignment_changes(assignments, assignments, n_rot, n_trans, 3)
        assert frac == 0.0

    def test_all_different_assignments(self):
        n_rot, n_trans = 100, 5
        current = np.arange(50) * n_trans
        previous = (np.arange(50) + 1) * n_trans
        frac = compute_assignment_changes(current, previous, n_rot, n_trans, 3)
        assert frac == 1.0

    def test_half_changed(self):
        n_rot, n_trans = 100, 5
        n_images = 100
        current = np.arange(n_images) * n_trans
        previous = current.copy()
        # Change first 50
        previous[:50] = (np.arange(50) + 50) * n_trans
        frac = compute_assignment_changes(current, previous, n_rot, n_trans, 3)
        assert abs(frac - 0.5) < 1e-10

    def test_translation_only_change_not_counted(self):
        """If only translation changed but rotation is same, fraction = 0."""
        n_rot, n_trans = 100, 5
        current = np.array([0, 5, 10, 15])  # rot indices 0, 1, 2, 3
        previous = np.array([1, 6, 11, 16])  # same rot indices, different trans
        frac = compute_assignment_changes(current, previous, n_rot, n_trans, 3)
        assert frac == 0.0

    def test_none_assignments_return_one(self):
        frac = compute_assignment_changes(None, np.array([1, 2, 3]), 10, 5, 3)
        assert frac == 1.0

    def test_mismatched_shapes_return_one(self):
        frac = compute_assignment_changes(np.array([1, 2]), np.array([1, 2, 3]), 10, 5, 3)
        assert frac == 1.0

    def test_empty_assignments_return_zero(self):
        frac = compute_assignment_changes(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            10,
            5,
            3,
        )
        assert frac == 0.0


# =========================================================================
# Translation change tracking
# =========================================================================


class TestTranslationChanges:
    def test_identical_assignments_zero_change(self):
        translations = np.array([[0, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        n_trans = len(translations)
        assignments = np.array([0, 1, 2, 3])
        rms = compute_translation_changes(assignments, assignments, translations, n_trans)
        assert rms == 0.0

    def test_known_shift(self):
        """All images shift by (1, 0) -> RMS = 1.0."""
        translations = np.array([[0, 0], [1, 0]], dtype=np.float32)
        n_trans = 2
        # 4 images, all at trans_idx=0 -> trans_idx=1 (shift of (1,0))
        current = np.array([0, 0, 0, 0])  # rot_idx=0, trans_idx=0
        previous = np.array([1, 1, 1, 1])  # rot_idx=0, trans_idx=1
        rms = compute_translation_changes(current, previous, translations, n_trans)
        assert abs(rms - 1.0) < 1e-6

    def test_none_returns_inf(self):
        translations = np.array([[0, 0]], dtype=np.float32)
        rms = compute_translation_changes(None, np.array([0]), translations, 1)
        assert rms == float("inf")


# =========================================================================
# Average Pmax
# =========================================================================


class TestAvePmax:
    def test_uniform_pmax(self):
        pmax = np.ones(100) * 0.5
        assert abs(compute_ave_Pmax(pmax) - 0.5) < 1e-10

    def test_varied_pmax(self):
        pmax = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        expected = np.mean(pmax)
        assert abs(compute_ave_Pmax(pmax) - expected) < 1e-10

    def test_empty_returns_zero(self):
        assert compute_ave_Pmax(np.array([])) == 0.0


# =========================================================================
# Convergence detection
# =========================================================================


class TestCheckConvergence:
    def test_not_converged_when_sampling_coarse(self):
        """Not converged when healpix_order < max."""
        state = RefinementState(
            healpix_order=3,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=5,
            nr_iter_wo_assignment_changes=5,
        )
        assert check_convergence(state) is False

    def test_not_converged_when_resolution_improving(self):
        state = RefinementState(
            healpix_order=7,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=5,
        )
        assert check_convergence(state) is False

    def test_not_converged_when_assignments_unstable(self):
        state = RefinementState(
            healpix_order=7,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=5,
            nr_iter_wo_assignment_changes=0,
        )
        assert check_convergence(state) is False

    def test_converged_when_all_criteria_met(self):
        state = RefinementState(
            healpix_order=7,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=MAX_NR_ITER_WO_RESOL_GAIN,
            nr_iter_wo_assignment_changes=MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
            acc_rot=1.0,
        )
        assert check_convergence(state) is True

    def test_converged_with_excess_stalls(self):
        """Extra stall iterations still converge."""
        state = RefinementState(
            healpix_order=7,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=10,
            nr_iter_wo_assignment_changes=10,
            acc_rot=1.0,
        )
        assert check_convergence(state) is True

    def test_not_converged_when_only_runtime_cap_reached(self):
        state = RefinementState(
            healpix_order=7,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=MAX_NR_ITER_WO_RESOL_GAIN,
            nr_iter_wo_assignment_changes=MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
        )
        assert check_convergence(state) is False


# =========================================================================
# Angular step refinement
# =========================================================================


class TestShouldRefineAngularSampling:
    def test_not_refine_when_at_max(self):
        state = RefinementState(
            healpix_order=7,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=5,
            nr_iter_wo_assignment_changes=5,
        )
        assert should_refine_angular_sampling(state) is False

    def test_not_refine_when_resolution_improving(self):
        state = RefinementState(
            healpix_order=3,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=5,
        )
        assert should_refine_angular_sampling(state) is False

    def test_not_refine_when_assignments_unstable(self):
        state = RefinementState(
            healpix_order=3,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=4,
            nr_iter_wo_assignment_changes=0,
        )
        assert should_refine_angular_sampling(state) is False

    def test_refine_when_stalled_and_stable(self):
        state = RefinementState(
            healpix_order=3,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=MAX_NR_ITER_WO_RESOL_GAIN,
            nr_iter_wo_assignment_changes=MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
        )
        assert should_refine_angular_sampling(state) is True

    def test_not_refine_beyond_75pct_acc_rot(self):
        """Don't refine if current step < 75% of estimated accuracy."""
        state = RefinementState(
            healpix_order=5,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=5,
            nr_iter_wo_assignment_changes=5,
            acc_rot=1.0,  # 1 degree accuracy
        )
        # effective_step at order 5 = 360 / (6 * 32) = 1.875 deg
        # 75% of 1.0 = 0.75; 1.875 > 0.75, so should refine
        assert should_refine_angular_sampling(state) is True

        # Now set acc_rot large enough that step < 0.75 * acc_rot
        state2 = RefinementState(
            healpix_order=6,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=5,
            nr_iter_wo_assignment_changes=5,
            acc_rot=1.0,  # 1 degree accuracy
        )
        # effective_step at order 6 = 360 / (6 * 64) = 0.9375 deg
        # 0.9375 > 0.75 so should still refine
        assert should_refine_angular_sampling(state2) is True

        # Make acc_rot so that step is below threshold
        state3 = RefinementState(
            healpix_order=6,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=5,
            nr_iter_wo_assignment_changes=5,
            acc_rot=0.5,  # 0.5 degree accuracy
        )
        # effective_step at order 6 = 0.9375 deg; 0.75 * 0.5 = 0.375
        # 0.9375 > 0.375, so should still refine
        assert should_refine_angular_sampling(state3) is True

        state4 = RefinementState(
            healpix_order=6,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=5,
            nr_iter_wo_assignment_changes=5,
            acc_rot=2.0,
        )
        # effective_step at order 6 = 0.9375 deg; 0.75 * 2.0 = 1.5
        # 0.9375 < 1.5, so RELION considers angular sampling fine enough.
        assert should_refine_angular_sampling(state4) is False

    def test_refines_when_resolution_requires_finer_sampling_despite_loose_acc_rot(self):
        state = RefinementState(
            healpix_order=2,
            adaptive_oversampling=1,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=MAX_NR_ITER_WO_RESOL_GAIN,
            nr_iter_wo_assignment_changes=MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
            acc_rot=12.247,
            current_resolution=4.86,
            particle_diameter_angstrom=544.0,
        )

        assert should_refine_angular_sampling(state) is True

    def test_resolution_based_trigger_requires_relion_auto_resol_angles_flag(self):
        state = RefinementState(
            healpix_order=2,
            adaptive_oversampling=1,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
            acc_rot=12.247,
            current_resolution=4.86,
            particle_diameter_angstrom=544.0,
            auto_resolution_based_angles=False,
        )

        assert resolution_triggers_angular_refinement(state) is False
        assert should_refine_angular_sampling(state) is False

        state.auto_resolution_based_angles = True
        assert resolution_triggers_angular_refinement(state) is True
        assert should_refine_angular_sampling(state) is True

    def test_resolution_based_trigger_does_not_enter_local_search_directly(self):
        state = RefinementState(
            healpix_order=3,
            adaptive_oversampling=1,
            max_healpix_order=7,
            auto_local_healpix_order=4,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
            acc_rot=12.247,
            current_resolution=4.86,
            particle_diameter_angstrom=544.0,
            auto_resolution_based_angles=True,
        )

        assert resolution_triggers_angular_refinement(state) is False
        assert should_refine_angular_sampling(state) is False

        state.nr_iter_wo_resol_gain = MAX_NR_ITER_WO_RESOL_GAIN
        assert should_refine_angular_sampling(state) is True


class TestRefineAngularSampling:
    def test_order_increments_by_one(self):
        state = RefinementState(healpix_order=3)
        new_state = refine_angular_sampling(state)
        assert new_state.healpix_order == 4

    def test_angular_step_updated(self):
        state = RefinementState(healpix_order=3)
        new_state = refine_angular_sampling(state)
        expected = healpix_angular_step(4)
        assert abs(new_state.angular_step - expected) < 1e-10

    def test_counters_reset(self):
        state = RefinementState(
            healpix_order=3,
            nr_iter_wo_resol_gain=5,
            nr_iter_wo_assignment_changes=5,
        )
        new_state = refine_angular_sampling(state)
        assert new_state.nr_iter_wo_resol_gain == 0
        assert new_state.nr_iter_wo_assignment_changes == 0

    def test_local_search_activated_at_order_4(self):
        state = RefinementState(healpix_order=3)
        new_state = refine_angular_sampling(state)
        assert new_state.healpix_order == 4
        assert new_state.do_local_search is True
        assert new_state.sigma_rot > 0.0
        assert new_state.sigma_psi > 0.0

    def test_local_search_not_activated_below_order_4(self):
        state = RefinementState(healpix_order=2)
        new_state = refine_angular_sampling(state)
        assert new_state.healpix_order == 3
        assert new_state.do_local_search is False
        assert new_state.sigma_rot == 0.0

    def test_translation_step_from_acc_trans(self):
        state = RefinementState(
            healpix_order=3,
            acc_trans=2.0,
            adaptive_oversampling=1,
            current_changes_optimal_offsets_angstrom=2.0,
            voxel_size_angstrom=4.25,
        )
        new_state = refine_angular_sampling(state)
        expected = min(1.5, 0.75 * 2.0) * (2**1) / 4.25
        assert abs(new_state.translation_step - expected) < 1e-10

    def test_translation_range_from_offset_changes(self):
        state = RefinementState(
            healpix_order=3,
            current_changes_optimal_offsets_angstrom=1.5,
            translation_range=10.0,
            voxel_size_angstrom=4.25,
        )
        new_state = refine_angular_sampling(state)
        expected = min(5.0 * 1.5, 1.3 * 10.0 * 4.25) / 4.25
        assert abs(new_state.translation_range - expected) < 1e-10

    def test_translation_range_capped_at_1_3x(self):
        """Range is capped at 1.3x previous when 5 * changes is larger."""
        state = RefinementState(
            healpix_order=3,
            current_changes_optimal_offsets_angstrom=100.0,
            translation_range=5.0,
            voxel_size_angstrom=4.25,
        )
        new_state = refine_angular_sampling(state)
        expected = 1.3 * 5.0
        assert abs(new_state.translation_range - expected) < 1e-10

    def test_relion_width_guard_coarsens_missing_acc_trans_fallback(self):
        state = RefinementState(
            healpix_order=2,
            adaptive_oversampling=1,
            translation_range=3.0,
            translation_step=1.0,
            current_changes_optimal_offsets_angstrom=5.0,
            voxel_size_angstrom=4.25,
        )

        new_state = refine_angular_sampling(state)

        assert new_state.translation_range == pytest.approx(3.0 * 1.3)
        assert new_state.translation_step == pytest.approx(new_state.translation_range / 4.0)

    def test_sigma_rot_formula(self):
        """sigma2_rot = 2 * 2 * step^2 (RELION convention)."""
        state = RefinementState(healpix_order=3, adaptive_oversampling=0)
        new_state = refine_angular_sampling(state)
        step_deg = healpix_angular_step(4)
        step_rad = np.deg2rad(step_deg)
        expected_sigma = np.sqrt(4.0) * step_rad
        assert abs(new_state.sigma_rot - expected_sigma) < 1e-10


# =========================================================================
# Full update_refinement_state workflow
# =========================================================================


class TestUpdateRefinementState:
    def _make_base_state(self, **kwargs):
        defaults = dict(
            iteration=0,
            healpix_order=3,
            max_healpix_order=7,
            current_resolution=5.0,
            translation_range=10.0,
            translation_step=2.0,
        )
        defaults.update(kwargs)
        return RefinementState(**defaults)

    def test_iteration_increments(self):
        state = self._make_base_state()
        n_rot, n_trans = 100, 5
        assignments = np.zeros(50, dtype=np.int32)
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            None,
            n_rot,
            n_trans,
            translations,
            new_resolution=4.5,
        )
        assert updated.iteration == 1

    def test_resolution_improvement_resets_stall(self):
        state = self._make_base_state(
            current_resolution=5.0,
            nr_iter_wo_resol_gain=3,
        )
        n_rot, n_trans = 100, 5
        assignments = np.zeros(50, dtype=np.int32)
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            None,
            n_rot,
            n_trans,
            translations,
            new_resolution=4.0,  # better than 5.0
        )
        assert updated.nr_iter_wo_resol_gain == 0

    def test_resolution_stall_increments(self):
        state = self._make_base_state(
            current_resolution=5.0,
            nr_iter_wo_resol_gain=0,
        )
        n_rot, n_trans = 100, 5
        assignments = np.zeros(50, dtype=np.int32)
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            None,
            n_rot,
            n_trans,
            translations,
            new_resolution=5.5,  # worse than 5.0
        )
        assert updated.nr_iter_wo_resol_gain == 1

    def test_stable_assignments_increment_counter(self):
        # Use improving resolution so angular refinement is NOT triggered
        # (refinement requires both stalls to be >= 1)
        state = self._make_base_state(current_resolution=5.0)
        n_rot, n_trans = 100, 5
        # All assignments identical -> fraction_changed = 0
        assignments = np.arange(50) * n_trans
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            assignments,
            n_rot,
            n_trans,
            translations,
            new_resolution=4.0,  # improving -> no resol stall -> no refinement
        )
        assert updated.fraction_changed == 0.0
        assert updated.nr_iter_wo_assignment_changes == 1
        assert updated.nr_iter_wo_resol_gain == 0  # resolution improved

    def test_unstable_assignments_reset_counter(self):
        state = self._make_base_state(nr_iter_wo_assignment_changes=5)
        n_rot, n_trans = 100, 5
        current = np.arange(50) * n_trans
        previous = (np.arange(50) + 50) * n_trans  # all different
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            current,
            previous,
            n_rot,
            n_trans,
            translations,
            new_resolution=5.0,
        )
        assert updated.fraction_changed == 1.0
        assert updated.nr_iter_wo_assignment_changes == 0

    def test_angular_refinement_triggered(self):
        """When both stalls are met and not at max order, order should increase."""
        state = self._make_base_state(
            healpix_order=3,
            nr_iter_wo_resol_gain=0,  # will become 1 after this iter
            nr_iter_wo_assignment_changes=0,  # will become 1
        )
        n_rot, n_trans = 100, 5
        assignments = np.arange(50) * n_trans
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            assignments,
            n_rot,
            n_trans,
            translations,
            new_resolution=5.5,  # stall
        )
        # After update: resol_gain=1, assignment_changes=1 -> should refine
        assert updated.healpix_order == 4
        # Counters should be reset after refinement
        assert updated.nr_iter_wo_resol_gain == 0
        assert updated.nr_iter_wo_assignment_changes == 0

    def test_runtime_cap_alone_does_not_converge(self):
        """The RECOVAR max order cap is not RELION's convergence criterion."""
        state = self._make_base_state(
            healpix_order=7,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=0,
        )
        n_rot, n_trans = 100, 5
        assignments = np.arange(50) * n_trans
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            assignments,
            n_rot,
            n_trans,
            translations,
            new_resolution=5.5,
        )
        assert updated.has_converged is False

    def test_convergence_when_fine_enough_at_max_order(self):
        """At max order with RELION fine-enough sampling, stalls converge."""
        state = self._make_base_state(
            healpix_order=7,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=0,
            acc_rot=1.0,
        )
        n_rot, n_trans = 100, 5
        assignments = np.arange(50) * n_trans
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            assignments,
            n_rot,
            n_trans,
            translations,
            new_resolution=5.5,
        )
        assert updated.has_converged is True

    def test_pmax_tracking(self):
        state = self._make_base_state()
        n_rot, n_trans = 100, 5
        assignments = np.zeros(50, dtype=np.int32)
        translations = np.zeros((n_trans, 2), dtype=np.float32)
        pmax = np.ones(50) * 0.42

        updated = update_refinement_state(
            state,
            assignments,
            None,
            n_rot,
            n_trans,
            translations,
            new_resolution=4.0,
            max_posterior_per_image=pmax,
        )
        assert abs(updated.ave_Pmax - 0.42) < 1e-6

    def test_k_class_change_tracking_counts_hard_class_changes(self):
        state = self._make_base_state()
        n_rot, n_trans = 100, 5
        assignments = np.zeros(5, dtype=np.int32)
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            assignments,
            n_rot,
            n_trans,
            translations,
            new_resolution=4.0,
            current_classes=np.array([0, 0, 1, 1, 2], dtype=np.int32),
            previous_classes=np.array([0, 1, 1, 0, 2], dtype=np.int32),
        )
        assert updated.current_changes_optimal_classes == 2.0
        assert updated.smallest_changes_optimal_classes == 2

    def test_single_class_change_tracking_remains_zero_when_classes_omitted(self):
        state = self._make_base_state()
        n_rot, n_trans = 100, 5
        assignments = np.zeros(5, dtype=np.int32)
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            assignments,
            n_rot,
            n_trans,
            translations,
            new_resolution=4.0,
        )
        assert updated.current_changes_optimal_classes == 0.0

    def test_hidden_variable_translation_ratio_uses_effective_oversampled_step(self):
        state = self._make_base_state(
            healpix_order=2,
            adaptive_oversampling=1,
            current_resolution=10.0,
            translation_step=1.0,
            smallest_changes_optimal_classes=0.0,
            smallest_changes_optimal_orientations=999.0,
            smallest_changes_optimal_offsets_angstrom=999.0,
            nr_iter_wo_large_hidden_variable_changes=0,
        )
        n_rot, n_trans = 100, 1
        assignments = np.zeros(5, dtype=np.int32)
        translations = np.zeros((n_trans, 2), dtype=np.float32)
        rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 5, axis=0)
        previous_trans = np.zeros((5, 2), dtype=np.float32)
        current_trans = np.column_stack(
            [
                np.full(5, 0.55, dtype=np.float32),
                np.zeros(5, dtype=np.float32),
            ]
        )

        updated = update_refinement_state(
            state,
            assignments,
            assignments,
            n_rot,
            n_trans,
            translations,
            new_resolution=9.0,
            current_rotation_matrices=rotations,
            previous_rotation_matrices=rotations,
            current_translations_pixel=current_trans,
            previous_translations_pixel=previous_trans,
            voxel_size_angstrom=1.0,
        )

        assert updated.current_changes_optimal_offsets_angstrom == pytest.approx(0.55 / np.sqrt(2.0))
        assert updated.nr_iter_wo_large_hidden_variable_changes == 0


# =========================================================================
# get_rotation_grid_at_order (sampling.py)
# =========================================================================


class TestGetRotationGridAtOrder:
    def test_returns_matrices(self):
        rots = get_rotation_grid_at_order(2, matrices=True)
        assert rots.ndim == 3
        assert rots.shape[1:] == (3, 3)

    def test_returns_euler_angles(self):
        angles = get_rotation_grid_at_order(2, matrices=False)
        assert angles.ndim == 2
        assert angles.shape[1] == 3

    def test_matches_get_rotation_grid(self):
        """get_rotation_grid_at_order should produce identical output."""
        for order in [1, 2, 3]:
            expected = get_rotation_grid(order, matrices=True)
            actual = get_rotation_grid_at_order(order, matrices=True)
            np.testing.assert_array_equal(actual, expected)

    def test_count_increases_with_order(self):
        n2 = get_rotation_grid_at_order(2, matrices=True).shape[0]
        n3 = get_rotation_grid_at_order(3, matrices=True).shape[0]
        assert n3 > n2


# =========================================================================
# Integration: RefinementState + update across multiple iterations
# =========================================================================


class TestMultiIterationWorkflow:
    """Simulate several iterations and verify the state machine behavior."""

    def test_three_iteration_convergence(self):
        """
        Iter 0: improving resolution, assignments unstable
        Iter 1: resolution stalls, assignments stabilize -> refine order 2->3
        Iter 2: at max order but without RELION fine-enough acc_rot -> no convergence
        Iter 3: at max order with fine-enough acc_rot, stalls -> converge
        """
        n_rot, n_trans = 100, 5
        n_images = 50
        translations = np.zeros((n_trans, 2), dtype=np.float32)

        # Start at order 2
        state = RefinementState(
            healpix_order=2,
            max_healpix_order=3,  # small max for fast test
            current_resolution=10.0,
        )

        # Iter 0: resolution improves, assignments change
        ha0 = np.arange(n_images) * n_trans
        state = update_refinement_state(
            state,
            ha0,
            None,
            n_rot,
            n_trans,
            translations,
            new_resolution=8.0,
        )
        assert state.iteration == 1
        assert state.has_converged is False
        assert state.healpix_order == 2  # not refined yet

        # Iter 1: resolution stalls, assignments stable -> triggers refinement
        state = update_refinement_state(
            state,
            ha0,
            ha0,
            n_rot,
            n_trans,
            translations,
            new_resolution=9.0,  # worse
        )
        # update_refinement_state increments iteration to 2, then
        # refine_angular_sampling preserves that iteration count
        assert state.iteration == 2
        assert state.healpix_order == 3  # refined!
        assert state.nr_iter_wo_resol_gain == 0  # reset
        assert state.has_converged is False

        # Iter 2: at max order, resolution stalls, assignments stable. The
        # RECOVAR runtime cap prevents further refinement but does not imply
        # RELION convergence by itself.
        state2 = RefinementState(
            iteration=state.iteration,
            healpix_order=3,
            max_healpix_order=3,
            current_resolution=9.0,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=0,
        )
        state2 = update_refinement_state(
            state2,
            ha0,
            ha0,
            n_rot,
            n_trans,
            translations,
            new_resolution=9.5,
        )
        assert state2.has_converged is False

        # Iter 3: once angular accuracy says the current sampling is fine
        # enough, the same stall counters can trigger convergence.
        state3 = RefinementState(
            iteration=state2.iteration,
            healpix_order=3,
            max_healpix_order=3,
            current_resolution=9.5,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=0,
            acc_rot=100.0,
        )
        state3 = update_refinement_state(
            state3,
            ha0,
            ha0,
            n_rot,
            n_trans,
            translations,
            new_resolution=10.0,
        )
        assert state3.has_converged is True
