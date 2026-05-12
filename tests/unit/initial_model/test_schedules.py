"""Phase-1 schedule tests.

These tests pin the recovar schedule functions against hand-derived values
read directly from `relion/src/ml_optimiser.cpp`. They do NOT require a
RELION binding. The Phase-2 binding tests at
`tests/unit/test_relion_bind/test_initialmodel_bind.py` will re-run the
same inputs through a C++ bind and assert byte-identical outputs.

The reference fixture dimensions used throughout:
  - nr_iter = 200 (GUI default)
  - grad_ini_frac = 0.3, grad_fin_frac = 0.2
  - So grad_ini_iter = 60, grad_fin_iter = 40, grad_inbetween_iter = 100.
"""

from __future__ import annotations

import pytest

from recovar.em.initial_model import (
    GuiInitialModelDefaults,
    compute_phase_lengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
    default_step_size_for_3d_initial_model,
    default_subset_sizes_for_3d_initial_model,
    default_tau2_fudge_for_3d_initial_model,
)
from recovar.em.initial_model.schedules import _relion_round, _step_sigmoid_value

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Phase lengths (ml_optimiser.cpp:411-420 + 994-998)
# ---------------------------------------------------------------------------


class TestPhaseLengths:
    def test_gui_defaults(self):
        # nr_iter=200, 0.3/0.2  -> 60 / 100 / 40
        p = compute_phase_lengths(200, 0.3, 0.2)
        assert p.grad_ini_iter == 60
        assert p.grad_inbetween_iter == 100
        assert p.grad_fin_iter == 40

    def test_phases_sum_to_nr_iter(self):
        for nr_iter in [50, 123, 200, 500]:
            p = compute_phase_lengths(nr_iter, 0.3, 0.2)
            assert p.grad_ini_iter + p.grad_inbetween_iter + p.grad_fin_iter == nr_iter

    def test_renorm_branch_triggers_above_0p9(self):
        # 0.5 + 0.5 = 1.0 > 0.9 triggers renorm by (sum+0.1)=1.1
        # new_ini = 0.5/1.1, new_fin = 0.5/1.1 -> sum = 10/11, inbetween = 1/11
        p = compute_phase_lengths(1100, 0.5, 0.5)
        assert p.grad_ini_iter == int(1100 * (0.5 / 1.1))
        assert p.grad_fin_iter == int(1100 * (0.5 / 1.1))
        # nr_iter - ini - fin
        assert p.grad_inbetween_iter == 1100 - p.grad_ini_iter - p.grad_fin_iter

    def test_c_integer_truncation_not_banker_rounding(self):
        # nr_iter=10, frac=0.15 -> int(10*0.15) = int(1.5) = 1 (trunc), NOT 2
        p = compute_phase_lengths(10, 0.15, 0.15)
        assert p.grad_ini_iter == 1
        assert p.grad_fin_iter == 1
        assert p.grad_inbetween_iter == 8

    def test_invalid_fractions(self):
        with pytest.raises(ValueError):
            compute_phase_lengths(100, 0.0, 0.2)
        with pytest.raises(ValueError):
            compute_phase_lengths(100, 1.0, 0.2)
        with pytest.raises(ValueError):
            compute_phase_lengths(100, 0.3, 0.0)


# ---------------------------------------------------------------------------
# Default subset sizes (ml_optimiser.cpp:2672-2696, is_3d_model branch)
# ---------------------------------------------------------------------------


class TestDefaultSubsetSizes:
    def test_small_dataset_clamps_at_min(self):
        # N=500: ini = clamp(round(2.5), 200, 5000) = 200
        #        fin = clamp(round(50.0), 1000, 50000) = 1000
        ini, fin = default_subset_sizes_for_3d_initial_model(500)
        assert ini == 200
        assert fin == 1000

    def test_mid_dataset(self):
        # N=100000: ini = clamp(500, 200, 5000) = 500
        #           fin = clamp(10000, 1000, 50000) = 10000
        ini, fin = default_subset_sizes_for_3d_initial_model(100000)
        assert ini == 500
        assert fin == 10000

    def test_large_dataset_clamps_at_max(self):
        # N=10_000_000: ini = clamp(50000, 200, 5000) = 5000
        #               fin = clamp(1_000_000, 1000, 50000) = 50000
        ini, fin = default_subset_sizes_for_3d_initial_model(10_000_000)
        assert ini == 5000
        assert fin == 50000


# ---------------------------------------------------------------------------
# updateSubsetSize (ml_optimiser.cpp:10238-10271)
# ---------------------------------------------------------------------------


class TestSubsetSizeSchedule:
    def setup_method(self):
        # Use a large synthetic dataset so subset_size doesn't collapse to -1
        # via the `subset_size >= nr_particles` branch at line 10268.
        self.phases = compute_phase_lengths(200, 0.3, 0.2)
        self.nr_particles = 1_000_000
        self.ini = 500  # grad_ini_subset_size
        self.fin = 10_000  # grad_fin_subset_size

    def test_ini_phase_iter_equals_ini_subset(self):
        for it in [0, 1, 30, 59]:
            size = compute_subset_size(
                iter=it,
                phase_lengths=self.phases,
                grad_ini_subset_size=self.ini,
                grad_fin_subset_size=self.fin,
                nr_particles=self.nr_particles,
                nr_iter=200,
            )
            assert size == self.ini, f"iter={it}: got {size}, expected {self.ini}"

    def test_final_phase_equals_fin_subset(self):
        # After grad_ini_iter + grad_inbetween_iter = 60 + 100 = 160
        # For iters 160..(200 - grad_em_iters - 1) = 160..199 with em_iters=0
        for it in [160, 170, 180, 198]:
            size = compute_subset_size(
                iter=it,
                phase_lengths=self.phases,
                grad_ini_subset_size=self.ini,
                grad_fin_subset_size=self.fin,
                nr_particles=self.nr_particles,
                nr_iter=200,
            )
            assert size == self.fin, f"iter={it}: got {size}, expected {self.fin}"

    def test_final_iter_with_grad_em_iters_zero_collapses_to_all(self):
        # With grad_em_iters=0, `nr_iter - iter < 0` is false at iter=nr_iter.
        # The standard path keeps subset=grad_fin_subset_size at iter=nr_iter-1.
        # But at iter == nr_iter (200), `nr_iter - iter = 0 < 0` is false,
        # so it stays at fin. Not a collapse. Tested to document the edge.
        size = compute_subset_size(
            iter=200,
            phase_lengths=self.phases,
            grad_ini_subset_size=self.ini,
            grad_fin_subset_size=self.fin,
            nr_particles=self.nr_particles,
            nr_iter=200,
            grad_em_iters=0,
        )
        assert size == self.fin

    def test_inbetween_linear_ramp_endpoints(self):
        # iter = grad_ini_iter = 60: frac = 0, subset = ini
        size_lo = compute_subset_size(
            iter=60,
            phase_lengths=self.phases,
            grad_ini_subset_size=self.ini,
            grad_fin_subset_size=self.fin,
            nr_particles=self.nr_particles,
            nr_iter=200,
        )
        assert size_lo == self.ini

        # iter = grad_ini_iter + grad_inbetween_iter - 1 = 159
        # frac = 99/100 = 0.99
        # increment = round(0.99 * (10000 - 500)) = round(9405) = 9405
        # subset = 500 + 9405 = 9905
        size_hi = compute_subset_size(
            iter=159,
            phase_lengths=self.phases,
            grad_ini_subset_size=self.ini,
            grad_fin_subset_size=self.fin,
            nr_particles=self.nr_particles,
            nr_iter=200,
        )
        assert size_hi == 9905

    def test_inbetween_midpoint(self):
        # iter = 60 + 50 = 110, frac = 50/100 = 0.5
        # increment = round(0.5 * 9500) = 4750
        # subset = 500 + 4750 = 5250
        size = compute_subset_size(
            iter=110,
            phase_lengths=self.phases,
            grad_ini_subset_size=self.ini,
            grad_fin_subset_size=self.fin,
            nr_particles=self.nr_particles,
            nr_iter=200,
        )
        assert size == 5250

    def test_do_grad_false_collapses(self):
        size = compute_subset_size(
            iter=30,
            phase_lengths=self.phases,
            grad_ini_subset_size=self.ini,
            grad_fin_subset_size=self.fin,
            nr_particles=self.nr_particles,
            nr_iter=200,
            do_grad=False,
        )
        assert size == -1

    def test_grad_em_tail_collapses(self):
        # Last 5 iters with grad_em_iters=5: at iter=196, nr_iter-iter = 4 < 5
        size = compute_subset_size(
            iter=196,
            phase_lengths=self.phases,
            grad_ini_subset_size=self.ini,
            grad_fin_subset_size=self.fin,
            nr_particles=self.nr_particles,
            nr_iter=200,
            grad_em_iters=5,
        )
        assert size == -1

    def test_subset_exceeds_particles_collapses(self):
        # Fin subset > nr_particles --> collapses to -1
        size = compute_subset_size(
            iter=180,
            phase_lengths=self.phases,
            grad_ini_subset_size=200,
            grad_fin_subset_size=5000,
            nr_particles=500,  # 500-particle fixture
            nr_iter=200,
        )
        assert size == -1

    def test_fixture_500_particles(self):
        """Concrete case for the /scratch/.../relion_initialmodel_64_... fixture.

        N=500: default ini=200, fin=1000. fin > N so the second branch collapses
        to -1. But at iters 0..59 the ini=200 path still applies since
        subset_size=200 < 500.
        """
        ini, fin = default_subset_sizes_for_3d_initial_model(500)
        assert (ini, fin) == (200, 1000)
        phases = compute_phase_lengths(200, 0.3, 0.2)

        # Iter 0-59: ini phase, subset=200, does not collapse
        assert (
            compute_subset_size(
                iter=30,
                phase_lengths=phases,
                grad_ini_subset_size=ini,
                grad_fin_subset_size=fin,
                nr_particles=500,
                nr_iter=200,
            )
            == 200
        )

        # Iter >= 60 with fin=1000 > N=500 --> collapses to -1 at iter=60
        # because at iter=60 the inbetween branch gives subset=ini=200,
        # then as iter grows into the ramp the subset crosses 500 and collapses.
        # Specifically: subset>=500 first happens when
        # 200 + round(frac * 800) >= 500 --> frac >= 300/800 = 0.375
        # --> iter - 60 >= 37.5 --> iter >= 98.
        size_98 = compute_subset_size(
            iter=98,
            phase_lengths=phases,
            grad_ini_subset_size=ini,
            grad_fin_subset_size=fin,
            nr_particles=500,
            nr_iter=200,
        )
        assert size_98 == -1

        # Just below 98 still in-range
        size_97 = compute_subset_size(
            iter=97,
            phase_lengths=phases,
            grad_ini_subset_size=ini,
            grad_fin_subset_size=fin,
            nr_particles=500,
            nr_iter=200,
        )
        expected_97 = 200 + _relion_round((97 - 60) / 100 * 800)
        assert size_97 == expected_97


# ---------------------------------------------------------------------------
# updateStepSize (ml_optimiser.cpp:10278-10325)
# ---------------------------------------------------------------------------


class TestStepSizeSchedule:
    def test_default_3d_initial_model_stepsize(self):
        assert default_step_size_for_3d_initial_model() == 0.5

    def test_3d_initial_model_default_scheme_decays_0p9_to_0p5(self):
        """For is_3d_model=True, ref_dim=3, default stepsize=0.5, default scheme=1.8-step.

        scale = 1 / (10 ** ((x - b - a/2) / (a/4)) + 1) with b=grad_ini_iter,
        a=grad_inbetween_iter/2. Stepsize = 0.9*scale + 0.5*(1-scale).
        """
        p = compute_phase_lengths(200, 0.3, 0.2)
        # Early iter 0: x - b - a/2 = -60 - 25 = -85; exponent = -85/12.5 = -6.8
        # scale ~ 1 / (10^-6.8 + 1) ~ 1 --> stepsize ~ 0.9
        v0 = compute_stepsize(iter=0, phase_lengths=p, is_3d_model=True, ref_dim=3)
        assert abs(v0 - 0.9) < 1e-6

        # Very late iter 200: scale -> 0, stepsize -> 0.5
        v200 = compute_stepsize(iter=200, phase_lengths=p, is_3d_model=True, ref_dim=3)
        assert abs(v200 - 0.5) < 1e-6

        # Sigmoid midpoint uses the inclusive in-between span:
        # b + (grad_inbetween_iter - 1) / 4 = 60 + 24.75 = 84.75.
        # stepsize = 0.9*0.5 + 0.5*0.5 = 0.45 + 0.25 = 0.7
        v_mid = compute_stepsize(iter=85, phase_lengths=p, is_3d_model=True, ref_dim=3)
        assert abs(v_mid - 0.695349139902437) < 1e-12

    def test_short_8_iter_reference_schedule_matches_relion_initialmodel(self):
        """Pinned from the 50k/256 RELION InitialModel iter-8 reference."""
        p = compute_phase_lengths(8, 0.3, 0.2)
        assert p.grad_ini_iter == 2
        assert p.grad_inbetween_iter == 5
        assert p.grad_fin_iter == 1

        step_values = [
            compute_stepsize(iter=it, phase_lengths=p, is_3d_model=True, ref_dim=3) for it in range(1, 9)
        ]
        assert step_values[0] == pytest.approx(0.89996, abs=5e-6)
        assert step_values[1] == pytest.approx(0.89604, abs=5e-6)
        assert step_values[2] == pytest.approx(0.7, abs=5e-6)
        assert step_values[3] == pytest.approx(0.50396, abs=5e-6)

        tau_values = [
            compute_tau2_fudge(
                iter=it,
                phase_lengths=p,
                is_3d_model=True,
                ref_dim=3,
                tau2_fudge_arg=4.0,
            )
            for it in range(1, 9)
        ]
        assert tau_values[0] == pytest.approx(1.000003, abs=5e-6)
        assert tau_values[1] == pytest.approx(1.029703, abs=5e-6)
        assert tau_values[2] == pytest.approx(3.970297, abs=5e-6)
        assert tau_values[3] == pytest.approx(3.999997, abs=5e-6)

    def test_3d_class_default_scheme_is_plain(self):
        p = compute_phase_lengths(200, 0.3, 0.2)
        v = compute_stepsize(iter=50, phase_lengths=p, is_3d_model=False, ref_dim=3)
        assert v == 0.3  # 3D classification default, plain

    def test_user_provided_stepsize_overrides_default(self):
        p = compute_phase_lengths(200, 0.3, 0.2)
        v = compute_stepsize(
            iter=0,
            phase_lengths=p,
            is_3d_model=True,
            ref_dim=3,
            grad_stepsize=0.7,
        )
        # Default scheme = "1.285714-step" for stepsize=0.7 (= 0.9/0.7)
        # at iter 0 (scale ~ 1): value = 0.9 * 1 + 0.7 * 0 = 0.9
        assert abs(v - 0.9) < 1e-6

    def test_plain_scheme_returns_stepsize(self):
        p = compute_phase_lengths(200, 0.3, 0.2)
        v = compute_stepsize(
            iter=50,
            phase_lengths=p,
            is_3d_model=True,
            ref_dim=3,
            grad_stepsize=0.42,
            grad_stepsize_scheme="plain",
        )
        assert v == 0.42

    def test_invalid_ref_dim(self):
        p = compute_phase_lengths(200, 0.3, 0.2)
        with pytest.raises(ValueError):
            compute_stepsize(iter=0, phase_lengths=p, is_3d_model=False, ref_dim=4)


# ---------------------------------------------------------------------------
# updateTau2Fudge (ml_optimiser.cpp:10327-10379) -- v1 had this inverted
# ---------------------------------------------------------------------------


class TestTau2FudgeSchedule:
    def test_default_3d_initial_model_fudge(self):
        assert default_tau2_fudge_for_3d_initial_model() == 4.0

    def test_3d_initial_model_default_scheme_grows_1_to_4(self):
        """Default scheme = "4-step" -> deflate=4.

        tau = (4/4)*scale + 4*(1-scale) = scale + 4*(1-scale) = 4 - 3*scale.
        scale has sigmoid_length = grad_inbetween_iter/4 = 25.
        """
        p = compute_phase_lengths(200, 0.3, 0.2)

        # Very early iter: scale -> 1, tau -> 1
        v0 = compute_tau2_fudge(
            iter=0,
            phase_lengths=p,
            is_3d_model=True,
            ref_dim=3,
            tau2_fudge_arg=4.0,
        )
        assert abs(v0 - 1.0) < 1e-3

        # Very late iter: scale -> 0, tau -> 4
        v_late = compute_tau2_fudge(
            iter=200,
            phase_lengths=p,
            is_3d_model=True,
            ref_dim=3,
            tau2_fudge_arg=4.0,
        )
        assert abs(v_late - 4.0) < 1e-3

        # Midpoint iter = b + a/2 = grad_ini_iter + (grad_inbetween_iter - 1) / 8
        # = 60 + 12.375 --> use 72 or 73 integer
        # scale = 0.5 --> tau = 4 - 3*0.5 = 2.5
        v_mid = compute_tau2_fudge(
            iter=72,
            phase_lengths=p,
            is_3d_model=True,
            ref_dim=3,
            tau2_fudge_arg=4.0,
        )
        # Accept within a small tolerance since iter=72 is not exactly 72.5
        assert 2.3 < v_mid < 2.7

    def test_trajectory_is_monotone_increasing_1_to_4(self):
        """The v1 plan had this inverted (16 -> 4). Pin the correct direction."""
        p = compute_phase_lengths(200, 0.3, 0.2)
        values = [
            compute_tau2_fudge(
                iter=it,
                phase_lengths=p,
                is_3d_model=True,
                ref_dim=3,
                tau2_fudge_arg=4.0,
            )
            for it in range(0, 201, 10)
        ]
        # Every step (after the flat initial tail) is non-decreasing
        for a, b in zip(values, values[1:]):
            assert b >= a - 1e-12, f"trajectory not monotone: {a} -> {b}"
        assert values[0] < 1.5  # starts near 1
        assert values[-1] > 3.5  # ends near 4

    def test_auto_refine_fudge_default_is_1(self):
        p = compute_phase_lengths(200, 0.3, 0.2)
        v = compute_tau2_fudge(
            iter=30,
            phase_lengths=p,
            is_3d_model=False,
            ref_dim=3,
            do_auto_refine=True,
            # tau2_fudge_arg=None triggers default -> 1 for auto_refine
        )
        # Scheme for 3D classification is "plain" --> returns _fudge=1
        assert v == 1.0

    def test_3d_class_default_scheme_is_plain(self):
        p = compute_phase_lengths(200, 0.3, 0.2)
        v = compute_tau2_fudge(
            iter=30,
            phase_lengths=p,
            is_3d_model=False,  # 3D classification, not initial model
            ref_dim=3,
            tau2_fudge_arg=4.0,
        )
        assert v == 4.0  # plain scheme returns fudge directly

    def test_user_supplied_scheme_plain(self):
        p = compute_phase_lengths(200, 0.3, 0.2)
        v = compute_tau2_fudge(
            iter=100,
            phase_lengths=p,
            is_3d_model=True,
            ref_dim=3,
            tau2_fudge_arg=2.5,
            tau2_fudge_scheme="plain",
        )
        assert v == 2.5


# ---------------------------------------------------------------------------
# Sigmoid / rounding helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_relion_round_matches_floor_plus_0p5(self):
        # RELION's ROUND macro (xmipp_macros.h):
        #   (x > 0) ? (int)(x + 0.5) : (int)(x - 0.5)
        # which rounds ties away from zero, not banker's-style.
        assert _relion_round(0.5) == 1  # Python round(0.5) == 0 (bankers)
        assert _relion_round(1.5) == 2
        assert _relion_round(2.5) == 3  # Python round(2.5) == 2 (bankers)
        assert _relion_round(0.49) == 0
        assert _relion_round(0.51) == 1
        # Negative ties round away from zero
        assert _relion_round(-0.5) == -1  # (int)(-0.5 - 0.5) = (int)(-1.0) = -1
        assert _relion_round(-1.5) == -2
        assert _relion_round(-2.5) == -3
        assert _relion_round(-0.49) == 0
        assert _relion_round(-0.51) == -1

    def test_step_sigmoid_extremes(self):
        # Far past the sigmoid midpoint --> scale ~ 0, value ~ base
        v = _step_sigmoid_value(
            iter=1_000_000,
            grad_ini_iter=60,
            grad_inbetween_iter=100,
            base=0.5,
            inflated=0.9,
            sigmoid_length=50.0,
        )
        assert abs(v - 0.5) < 1e-9

        # Far before --> scale ~ 1, value ~ inflated
        v2 = _step_sigmoid_value(
            iter=-1_000_000,
            grad_ini_iter=60,
            grad_inbetween_iter=100,
            base=0.5,
            inflated=0.9,
            sigmoid_length=50.0,
        )
        assert abs(v2 - 0.9) < 1e-9

    def test_step_sigmoid_degenerate_a_zero(self):
        # Inbetween phase is 0 --> fall through to base
        v = _step_sigmoid_value(
            iter=100,
            grad_ini_iter=60,
            grad_inbetween_iter=0,
            base=0.5,
            inflated=0.9,
            sigmoid_length=0.0,
        )
        assert v == 0.5


# ---------------------------------------------------------------------------
# Collective sanity check: GUI defaults produce the trajectory the plan claims
# ---------------------------------------------------------------------------


def test_gui_initial_model_defaults_produce_plan_trajectory():
    """Quick visual check: at GUI defaults, step 0.9->0.5 and tau 1->4."""
    defaults = GuiInitialModelDefaults()
    p = compute_phase_lengths(
        defaults.nr_iter,
        defaults.grad_ini_frac,
        defaults.grad_fin_frac,
    )
    step0 = compute_stepsize(0, p, is_3d_model=True, ref_dim=3)
    step_end = compute_stepsize(defaults.nr_iter, p, is_3d_model=True, ref_dim=3)
    assert step0 > step_end
    assert abs(step0 - 0.9) < 1e-3
    assert abs(step_end - 0.5) < 1e-3

    tau0 = compute_tau2_fudge(0, p, is_3d_model=True, ref_dim=3, tau2_fudge_arg=defaults.tau2_fudge)
    tau_end = compute_tau2_fudge(
        defaults.nr_iter,
        p,
        is_3d_model=True,
        ref_dim=3,
        tau2_fudge_arg=defaults.tau2_fudge,
    )
    assert tau0 < tau_end
    assert abs(tau0 - 1.0) < 1e-2
    assert abs(tau_end - 4.0) < 1e-2
