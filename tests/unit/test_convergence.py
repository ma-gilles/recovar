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
    relion_mpi_hidden_variable_change_is_small,
    resolution_required_angular_sampling,
    resolution_triggers_angular_refinement,
    should_refine_angular_sampling,
    update_angular_sampling,
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

    def test_has_fine_enough_is_latched_at_sampling_boundary(self):
        state = RefinementState(
            healpix_order=7,
            max_healpix_order=7,
            acc_rot=1.0,
            nr_iter_wo_resol_gain=1,
            nr_iter_wo_assignment_changes=1,
        )
        assert state.has_fine_enough_angular_sampling is False
        assert update_angular_sampling(state).has_fine_enough_angular_sampling is True

    def test_resolution_required_sampling_does_not_replace_measured_acc_rot(self):
        """RELION's fine-enough decision uses acc_rot, not resolution."""
        state = RefinementState(
            healpix_order=2,
            adaptive_oversampling=1,
            acc_rot=12.247,
            current_resolution=4.86,
            particle_diameter_angstrom=544.0,
        )

        assert resolution_required_angular_sampling(4.86, 544.0) == pytest.approx(1.0227, rel=1e-3)
        state.nr_iter_wo_resol_gain = 1
        state.nr_iter_wo_assignment_changes = 1
        assert update_angular_sampling(state).has_fine_enough_angular_sampling is True

    def test_loose_acc_rot_still_applies_without_particle_diameter(self):
        state = RefinementState(
            healpix_order=2,
            adaptive_oversampling=1,
            acc_rot=12.247,
            current_resolution=4.86,
            particle_diameter_angstrom=0.0,
        )

        state.nr_iter_wo_resol_gain = 1
        state.nr_iter_wo_assignment_changes = 1
        assert update_angular_sampling(state).has_fine_enough_angular_sampling is True

    def test_resolution_required_sampling_cannot_replace_missing_acc_rot(self):
        """A resolution-derived step must not terminate without acc_rot."""
        state = RefinementState(
            healpix_order=7,
            adaptive_oversampling=1,
            acc_rot=float("inf"),
            current_resolution=15.11,
            particle_diameter_angstrom=200.0,
        )

        assert resolution_required_angular_sampling(15.11, 200.0) == pytest.approx(8.5714, rel=1e-3)
        assert state.has_fine_enough_angular_sampling is False

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
            has_fine_enough_angular_sampling=True,
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
            has_fine_enough_angular_sampling=True,
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

    def test_not_converged_when_resolution_fine_enough_without_acc_rot(self):
        state = RefinementState(
            healpix_order=7,
            adaptive_oversampling=1,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=MAX_NR_ITER_WO_RESOL_GAIN,
            nr_iter_wo_assignment_changes=MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
            acc_rot=float("inf"),
            current_resolution=15.11,
            particle_diameter_angstrom=200.0,
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

    def test_does_not_refine_when_measured_acc_rot_is_already_fine_enough(self):
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

        assert resolution_triggers_angular_refinement(state) is False
        assert should_refine_angular_sampling(state) is False

    def test_resolution_based_trigger_requires_relion_auto_resol_angles_flag(self):
        state = RefinementState(
            healpix_order=2,
            adaptive_oversampling=1,
            max_healpix_order=7,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
            acc_rot=1.0,
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
            acc_rot=1.0,
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

    def test_resolution_gain_uses_relion_reciprocal_tolerance(self):
        state = self._make_base_state(
            current_resolution=5.0,
            nr_iter_wo_resol_gain=0,
        )
        assignments = np.zeros(50, dtype=np.int32)
        translations = np.zeros((5, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            None,
            100,
            5,
            translations,
            new_resolution=4.999,
        )

        assert 1.0 / 4.999 > 1.0 / 5.0
        assert 1.0 / 4.999 <= 1.0 / 5.0 + 0.0001
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

    def test_post_m_update_defers_sampling_and_convergence_to_next_boundary(self):
        """Native AutoRefine records M-step observations without acting on them."""
        state = self._make_base_state(
            healpix_order=3,
            nr_iter_wo_resol_gain=0,
            nr_iter_wo_assignment_changes=0,
            acc_rot=999.0,
        )
        assignments = np.arange(50, dtype=np.int32) * 5
        translations = np.zeros((5, 2), dtype=np.float32)

        updated = update_refinement_state(
            state,
            assignments,
            assignments,
            100,
            5,
            translations,
            new_resolution=5.5,
            update_sampling=False,
            check_convergence_now=False,
        )

        assert updated.healpix_order == 3
        assert updated.nr_iter_wo_resol_gain == 1
        assert updated.nr_iter_wo_assignment_changes == 1
        assert updated.has_converged is False
        boundary_state = update_angular_sampling(updated)
        assert boundary_state.has_fine_enough_angular_sampling is True
        assert check_convergence(boundary_state) is True

    def test_refined_step_does_not_dynamically_become_fine_enough(self):
        """RELION latches against the old step, then runs the refined grid once."""
        state = self._make_base_state(
            healpix_order=3,
            nr_iter_wo_resol_gain=1,
            nr_iter_wo_assignment_changes=1,
            acc_rot=6.0,
        )

        refined = update_angular_sampling(state)

        assert refined.healpix_order == 4
        assert refined.effective_step < 0.75 * refined.acc_rot
        assert refined.has_fine_enough_angular_sampling is False
        refined.nr_iter_wo_resol_gain = 1
        refined.nr_iter_wo_assignment_changes = 1
        assert check_convergence(refined) is False
        assert update_angular_sampling(refined).has_fine_enough_angular_sampling is True

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
        assert updated.current_changes_optimal_classes == pytest.approx(0.4)
        assert updated.smallest_changes_optimal_classes == 0

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

    def test_hidden_variable_translation_ratio_uses_initial_mpi_leader_effective_step(self):
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
                np.full(5, 0.25, dtype=np.float32),
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

        assert updated.current_changes_optimal_offsets_angstrom == pytest.approx(0.25 / np.sqrt(2.0))
        assert updated.nr_iter_wo_large_hidden_variable_changes == 1

    def test_relion_mpi_case13_it8_star_values_reset_hidden_change_counter(self):
        """Case 13 STAR values require the stale MPI-leader translation step."""
        # run_it000_sampling.star: order=3, offset_step=4.25 A, oversampling=1.
        # run_it007/008_optimiser.star provide the smallest/current changes.
        assert not relion_mpi_hidden_variable_change_is_small(
            current_classes=0.0,
            current_offsets_angstrom=0.984143,
            current_orientations_deg=1.054518,
            smallest_classes=0.0,
            smallest_offsets_angstrom=1.035820,
            smallest_orientations_deg=1.136805,
            mpi_leader_angular_step_deg=7.5 / 2.0,
            mpi_leader_translation_step_angstrom=4.25 / 2.0,
        )

    def test_relion_mpi_case15_it7_star_values_increment_hidden_change_counter(self):
        """Case 15 STAR values also match the fixed process-initial sampling."""
        # run_it000_sampling.star: order=3, offset_step=4.25 A, oversampling=1.
        # run_it006/007_optimiser.star provide the smallest/current changes.
        assert relion_mpi_hidden_variable_change_is_small(
            current_classes=0.0,
            current_offsets_angstrom=0.575090,
            current_orientations_deg=1.266198,
            smallest_classes=0.0,
            smallest_offsets_angstrom=1.112159,
            smallest_orientations_deg=2.265981,
            mpi_leader_angular_step_deg=7.5 / 2.0,
            mpi_leader_translation_step_angstrom=4.25 / 2.0,
        )

    @staticmethod
    def _rotation_delta(n_images, relion_mean_angle_deg):
        # A z rotation changes two of three matrix rows by theta, so RELION's
        # row-mean angular distance is 2*theta/3.
        theta = np.deg2rad(1.5 * relion_mean_angle_deg)
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return np.repeat(rotation[None, :, :], n_images, axis=0)

    @staticmethod
    def _translation_delta(n_images, relion_rms_angstrom, voxel_size):
        # RELION divides the summed x/y squared displacement by 2*N.
        delta_x_pixel = np.sqrt(2.0) * relion_rms_angstrom / voxel_size
        return np.column_stack(
            [
                np.full(n_images, delta_x_pixel, dtype=np.float64),
                np.zeros(n_images, dtype=np.float64),
            ]
        )

    def _record_relion_hidden_change(self, state, angle_deg, offset_angstrom):
        n_images = 8
        identity = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n_images, axis=0)
        zeros = np.zeros((n_images, 2), dtype=np.float64)
        assignments = np.zeros(n_images, dtype=np.int32)
        return update_refinement_state(
            state,
            assignments,
            assignments,
            n_rotations=1,
            n_translations=1,
            translations=zeros[:1],
            new_resolution=22.6667,
            current_rotation_matrices=self._rotation_delta(n_images, angle_deg),
            previous_rotation_matrices=identity,
            current_translations_pixel=self._translation_delta(
                n_images,
                offset_angstrom,
                state.voxel_size_angstrom,
            ),
            previous_translations_pixel=zeros,
            voxel_size_angstrom=state.voxel_size_angstrom,
            update_sampling=False,
            check_convergence_now=False,
        )

    def test_relion_order4_to_order5_hidden_change_boundary(self):
        state = RefinementState(
            healpix_order=4,
            adaptive_oversampling=1,
            translation_step=1.87425 / 4.25,
            current_resolution=22.6667,
            voxel_size_angstrom=4.25,
            acc_rot=1.065,
            smallest_changes_optimal_classes=0.0,
            smallest_changes_optimal_orientations=1.996281,
            smallest_changes_optimal_offsets_angstrom=0.810118,
            mpi_leader_hidden_variable_angular_step_deg=7.5 / 2.0,
            mpi_leader_hidden_variable_translation_step_angstrom=4.25 / 2.0,
        )

        state = self._record_relion_hidden_change(state, 1.309840, 0.730098)

        assert state.nr_iter_wo_resol_gain == 1
        assert state.nr_iter_wo_large_hidden_variable_changes == 1
        assert update_angular_sampling(state).healpix_order == 5
        refined = update_angular_sampling(state)
        assert refined.mpi_leader_hidden_variable_angular_step_deg == pytest.approx(7.5 / 2.0)
        assert refined.mpi_leader_hidden_variable_translation_step_angstrom == pytest.approx(4.25 / 2.0)

    def test_real10076_relion_hidden_change_sequence_refines_at_iteration_10(self):
        state = RefinementState(
            healpix_order=3,
            adaptive_oversampling=1,
            translation_range=3.0,
            translation_step=1.0,
            current_resolution=14.4552,
            voxel_size_angstrom=1.6375,
            acc_rot=1.397,
            max_healpix_order=7,
            auto_local_healpix_order=4,
            nr_iter_wo_resol_gain=3,
            smallest_changes_optimal_classes=0.0,
            smallest_changes_optimal_orientations=4.901576,
            smallest_changes_optimal_offsets_angstrom=0.932432,
            mpi_leader_hidden_variable_angular_step_deg=3.75,
            mpi_leader_hidden_variable_translation_step_angstrom=0.81875,
        )

        for iteration, angle, offset in (
            (7, 4.743451, 0.928163),
            (8, 4.595795, 0.896570),
        ):
            state = self._record_relion_hidden_change(state, angle, offset)
            assert state.nr_iter_wo_large_hidden_variable_changes == 0
            state = update_angular_sampling(state)
            assert state.healpix_order == 3, iteration

        state = self._record_relion_hidden_change(state, 4.503409, 0.896290)
        assert state.nr_iter_wo_large_hidden_variable_changes == 1
        state = update_angular_sampling(state)
        assert state.healpix_order == 4

    def test_real10076_recovar_hidden_change_sequence_refines_at_iteration_8(self):
        state = RefinementState(
            healpix_order=3,
            adaptive_oversampling=1,
            translation_range=3.0,
            translation_step=1.0,
            current_resolution=14.4552,
            voxel_size_angstrom=1.6375,
            acc_rot=1.393,
            max_healpix_order=7,
            auto_local_healpix_order=4,
            nr_iter_wo_resol_gain=3,
            smallest_changes_optimal_classes=0.0,
            smallest_changes_optimal_orientations=4.838344,
            smallest_changes_optimal_offsets_angstrom=0.929794,
            mpi_leader_hidden_variable_angular_step_deg=3.75,
            mpi_leader_hidden_variable_translation_step_angstrom=0.81875,
        )

        state = self._record_relion_hidden_change(state, 4.836995, 0.933869)
        assert state.nr_iter_wo_large_hidden_variable_changes == 1
        state = update_angular_sampling(state)
        assert state.healpix_order == 4

    def test_relion_order5_to_order6_hidden_change_boundary(self):
        state = RefinementState(
            healpix_order=4,
            adaptive_oversampling=1,
            translation_step=1.87425 / 4.25,
            current_resolution=22.6667,
            voxel_size_angstrom=4.25,
            acc_rot=1.030,
            acc_trans=1.122,
            mpi_leader_hidden_variable_angular_step_deg=7.5 / 2.0,
            mpi_leader_hidden_variable_translation_step_angstrom=4.25 / 2.0,
        )
        state = refine_angular_sampling(state)

        assert state.healpix_order == 5
        assert state.suppress_hidden_variable_increment_once is True

        state = self._record_relion_hidden_change(state, 0.908481, 0.722427)
        assert state.nr_iter_wo_large_hidden_variable_changes == 0
        assert state.suppress_hidden_variable_increment_once is False
        state = self._record_relion_hidden_change(state, 0.600638, 0.606800)

        assert state.nr_iter_wo_resol_gain >= 1
        assert state.nr_iter_wo_large_hidden_variable_changes == 1
        assert update_angular_sampling(state).healpix_order == 6


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
