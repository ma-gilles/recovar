"""Phase 4 tests: VDAM M-step pipeline.

Full RELION-fixture per-iter parity (comparing against run_it001_*.star
after one VDAM iteration) is Phase 5's gate. Here we verify that the
Phase-4 M-step pipeline:

  - Correctly routes the 5 moment-primitive calls in the order RELION
    uses (reweightGrad -> first moment -> second moment -> applyMomenta
    -> reconstructGrad).
  - Updates Iref, Igrad1, Igrad2 in place with finite values.
  - Respects the pseudo_halfsets flag: 2K accumulators required when on,
    K when off.
  - Leaves the input state untouched (returns a new state).
  - Produces the same Igrad2 imaginary part as RELION: exactly zero.
"""

from __future__ import annotations

import numpy as np
import pytest

import recovar.em.initial_model.m_step as m_step
from recovar.em.initial_model import (
    initialise_denovo_state,
)
from recovar.em.initial_model.init import initialise_data_vs_prior_from_references, seed_noise_from_mavg
from recovar.em.initial_model.m_step import (
    VdamAccumulator,
    _grad_min_resol_shell_from_state,
    vdam_m_step,
    vdam_m_step_single_class,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def bind():
    try:
        from recovar.relion_bind import _relion_bind_core as m
    except ImportError:
        pytest.skip("relion_bind not built")
    if not hasattr(m, "vdam_reweight_grad"):
        pytest.skip("relion_bind built without InitialModel VDAM primitives; rebuild recovar/relion_bind")
    return m


def _make_accumulator(k: int, h: int, ori_size: int, seed: int, min_weight: float = 10.0) -> VdamAccumulator:
    """Build a deterministic accumulator with the RELION pad=1 shape."""
    pf = 1
    Nz_pad = ori_size * pf
    Ny_pad = ori_size * pf
    Nx_pad_half = (ori_size * pf) // 2 + 1
    rng = np.random.default_rng(seed)
    data = (
        rng.standard_normal((Nz_pad, Ny_pad, Nx_pad_half)) + 1j * rng.standard_normal((Nz_pad, Ny_pad, Nx_pad_half))
    ).astype(np.complex128)
    # Weights need to be well above zero so reconstruct's Wiener solve
    # doesn't divide by tiny numbers and produce NaN. Real RELION
    # accumulators have weights ~ N_particles * CTF^2/sigma^2 >> 1.
    weight = rng.uniform(min_weight, min_weight * 10.0, size=(Nz_pad, Ny_pad, Nx_pad_half)).astype(np.float64)
    return VdamAccumulator(data=data, weight=weight, class_idx=k, halfset_idx=h)


def test_default_grad_min_resol_shell_matches_relion_initialmodel_default():
    state = initialise_denovo_state(
        ori_size=256,
        pixel_size=2.125,
        K=1,
        nr_iter=4,
        n_directions=48,
        pseudo_halfsets=True,
    )

    assert _grad_min_resol_shell_from_state(state, None) == 27.0
    assert _grad_min_resol_shell_from_state(state, 0.0) == 0.0


def test_m_step_matches_relion_fsc_routing_for_ssnr_and_reconstruct(monkeypatch):
    ori = 16
    state = initialise_denovo_state(
        ori_size=ori,
        pixel_size=1.0,
        K=2,
        nr_iter=10,
        n_directions=12,
        pseudo_halfsets=False,
    )
    state.fsc_halves_class[0] = np.linspace(0.0, 0.25, ori // 2 + 1, dtype=np.float64)
    state.fsc_halves_class[1] = np.linspace(0.5, 0.75, ori // 2 + 1, dtype=np.float64)
    state.Iref[1] = np.random.default_rng(0).standard_normal((ori, ori, ori))
    captured = {}

    class FakeBindings:
        @staticmethod
        def vdam_reweight_grad(data, weight, *_args):
            return np.asarray(data)

        @staticmethod
        def vdam_first_moment(data, old, *_args, **_kwargs):
            return np.asarray(old)

        @staticmethod
        def vdam_apply_momenta(data_h0, *_args):
            return np.asarray(data_h0), np.zeros(ori // 2 + 1, dtype=np.float64)

        @staticmethod
        def vdam_update_ssnr_arrays_from_bpref(weight, fsc, tau2, *_args):
            captured["ssnr_fsc"] = np.asarray(fsc, dtype=np.float64).copy()
            shells = ori // 2 + 1
            return (
                np.asarray(tau2, dtype=np.float64),
                np.ones(shells, dtype=np.float64),
                np.ones(shells, dtype=np.float64),
                np.ones(shells, dtype=np.float64),
            )

        @staticmethod
        def vdam_reconstruct_grad(iref_relion, _post_data, _weight, fsc, *_args):
            captured["reconstruct_fsc"] = np.asarray(fsc, dtype=np.float64).copy()
            return np.asarray(iref_relion)

    monkeypatch.setattr(m_step, "_get_bindings", lambda: FakeBindings)
    accum = _make_accumulator(k=1, h=0, ori_size=ori, seed=4)

    vdam_m_step_single_class(
        state,
        k=1,
        accum_h0=accum,
        accum_h1=None,
        grad_current_stepsize=0.5,
        tau2_fudge_factor=4.0,
    )

    np.testing.assert_allclose(captured["ssnr_fsc"], state.fsc_halves_class[0])
    np.testing.assert_allclose(captured["reconstruct_fsc"], state.fsc_halves_class[1])


class TestMstepSingleClass:
    def test_seeded_tau2_prevents_zero_tau_current_window_runaway(self, bind):
        ori = 64
        state = initialise_denovo_state(
            ori_size=ori,
            pixel_size=2.0,
            K=1,
            nr_iter=10,
            n_directions=12,
            pseudo_halfsets=True,
        )
        state.Iref[0, ori // 2, ori // 2, ori // 2] = 1.0
        state = seed_noise_from_mavg(state, np.ones_like(state.sigma2_noise))
        seeded = initialise_data_vs_prior_from_references(state, nr_particles=1_000_000)
        r_max = seeded.current_size // 2
        half_ps = r_max + 1
        weight = np.full((2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1), 1e5, dtype=np.float64)
        fsc = np.zeros(ori // 2 + 1, dtype=np.float64)
        zero_tau = np.zeros(ori // 2 + 1, dtype=np.float64)

        def cutoff(dvp):
            i = 1
            while i < ori // 2 and i < len(dvp):
                if float(dvp[i]) < 1.0:
                    break
                i += 1
            return i - 1

        _tau2, _sigma2, zero_dvp, _coverage = bind.vdam_update_ssnr_arrays_from_bpref(
            weight,
            fsc,
            zero_tau,
            1.0,
            ori,
            1,
            1,
            r_max,
            False,
            False,
            False,
        )
        _tau2, _sigma2, seeded_dvp, _coverage = bind.vdam_update_ssnr_arrays_from_bpref(
            weight,
            fsc,
            seeded.tau2_class[0],
            1.0,
            ori,
            1,
            1,
            r_max,
            False,
            False,
            False,
        )

        assert cutoff(np.asarray(zero_dvp)) == r_max
        assert cutoff(np.asarray(seeded_dvp)) < r_max

    def test_pseudo_halfsets_on_updates_iref_and_moments(self, bind):
        ori = 16
        state = initialise_denovo_state(
            ori_size=ori,
            pixel_size=1.0,
            K=1,
            nr_iter=10,
            n_directions=12,
            pseudo_halfsets=True,
        )
        # Give reconstructGrad something meaningful: seed Iref nonzero so
        # the gradient update has a reference shape.
        state.Iref[0] = np.random.default_rng(42).standard_normal((ori, ori, ori))

        a0 = _make_accumulator(k=0, h=0, ori_size=ori, seed=1)
        a1 = _make_accumulator(k=0, h=1, ori_size=ori, seed=2)

        new_state = vdam_m_step_single_class(
            state,
            k=0,
            accum_h0=a0,
            accum_h1=a1,
            grad_current_stepsize=0.5,
            tau2_fudge_factor=1.0,
        )
        # Iref updated (and finite)
        assert new_state.Iref.shape == state.Iref.shape
        assert np.all(np.isfinite(new_state.Iref))
        assert not np.array_equal(new_state.Iref, state.Iref)

        # Igrad1 slots populated (non-zero within r_max)
        assert new_state.Igrad1.shape == state.Igrad1.shape
        assert not np.all(new_state.Igrad1 == 0)

        # Igrad2 updated within r_max: real part >= 0 there, imag part == 0.
        # Cells outside r_max keep the initial constant (1+1j); that's
        # RELION's behaviour since getSecondMoment only touches r2 < max_r2.
        assert new_state.Igrad2.shape == state.Igrad2.shape
        # At least some cells were updated (differ from 1+1j)
        mask_updated = new_state.Igrad2[0] != (1.0 + 1.0j)
        assert mask_updated.any(), "getSecondMoment did not update any cells"
        np.testing.assert_array_equal(new_state.Igrad2[0].imag[mask_updated], 0.0)
        assert np.all(new_state.Igrad2[0].real[mask_updated] >= 0.0)
        assert np.any(new_state.data_vs_prior_class[0] > 1.0)

    def test_inactive_class_with_tiny_weight_is_preserved(self, bind):
        ori = 16
        state = initialise_denovo_state(
            ori_size=ori,
            pixel_size=1.0,
            K=1,
            nr_iter=10,
            n_directions=12,
            pseudo_halfsets=True,
        )
        rng = np.random.default_rng(42)
        state.Iref[0] = rng.standard_normal((ori, ori, ori))
        state.Igrad1[:] = rng.standard_normal(state.Igrad1.shape) + 1j * rng.standard_normal(state.Igrad1.shape)

        a0 = _make_accumulator(k=0, h=0, ori_size=ori, seed=1, min_weight=1e-12)
        a1 = _make_accumulator(k=0, h=1, ori_size=ori, seed=2, min_weight=1e-12)

        iref_before = state.Iref.copy()
        igrad1_before = state.Igrad1.copy()
        igrad2_before = state.Igrad2.copy()
        tau2_before = state.tau2_class.copy()
        sigma2_before = state.sigma2_class.copy()
        data_vs_prior_before = state.data_vs_prior_class.copy()

        new_state = vdam_m_step_single_class(
            state,
            k=0,
            accum_h0=a0,
            accum_h1=a1,
            grad_current_stepsize=0.5,
            tau2_fudge_factor=1.0,
        )

        np.testing.assert_array_equal(new_state.Iref, iref_before)
        np.testing.assert_array_equal(new_state.Igrad1, igrad1_before)
        np.testing.assert_array_equal(new_state.Igrad2, igrad2_before)
        np.testing.assert_array_equal(new_state.tau2_class, tau2_before)
        np.testing.assert_array_equal(new_state.sigma2_class, sigma2_before)
        np.testing.assert_array_equal(new_state.data_vs_prior_class, data_vs_prior_before)

    def test_input_state_unchanged(self, bind):
        ori = 16
        state = initialise_denovo_state(ori_size=ori, pixel_size=1.0, K=1, nr_iter=10, n_directions=12)
        a0 = _make_accumulator(k=0, h=0, ori_size=ori, seed=1)
        a1 = _make_accumulator(k=0, h=1, ori_size=ori, seed=2)

        iref_before = state.Iref.copy()
        igrad1_before = state.Igrad1.copy()
        igrad2_before = state.Igrad2.copy()
        tau2_before = state.tau2_class.copy()
        data_vs_prior_before = state.data_vs_prior_class.copy()

        vdam_m_step_single_class(
            state,
            k=0,
            accum_h0=a0,
            accum_h1=a1,
            grad_current_stepsize=0.5,
            tau2_fudge_factor=1.0,
        )
        np.testing.assert_array_equal(state.Iref, iref_before)
        np.testing.assert_array_equal(state.Igrad1, igrad1_before)
        np.testing.assert_array_equal(state.Igrad2, igrad2_before)
        np.testing.assert_array_equal(state.tau2_class, tau2_before)
        np.testing.assert_array_equal(state.data_vs_prior_class, data_vs_prior_before)

    def test_pseudo_halfsets_mismatch_raises(self, bind):
        state = initialise_denovo_state(
            ori_size=16,
            pixel_size=1.0,
            K=1,
            nr_iter=10,
            n_directions=12,
            pseudo_halfsets=True,
        )
        a0 = _make_accumulator(0, 0, 16, seed=1)
        with pytest.raises(ValueError):
            vdam_m_step_single_class(
                state,
                k=0,
                accum_h0=a0,
                accum_h1=None,
                grad_current_stepsize=0.5,
                tau2_fudge_factor=1.0,
            )

    def test_pseudo_halfsets_off_without_halfsets(self, bind):
        ori = 16
        state = initialise_denovo_state(
            ori_size=ori,
            pixel_size=1.0,
            K=1,
            nr_iter=10,
            n_directions=12,
            pseudo_halfsets=False,
        )
        a0 = _make_accumulator(0, 0, ori, seed=1)
        new_state = vdam_m_step_single_class(
            state,
            k=0,
            accum_h0=a0,
            accum_h1=None,
            grad_current_stepsize=0.5,
            tau2_fudge_factor=1.0,
        )
        # Igrad1 has K slots, not 2K
        assert new_state.Igrad1.shape == (1, 16, 16, 9)

    def test_invalid_class_index(self, bind):
        state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=2, nr_iter=10, n_directions=12)
        a0 = _make_accumulator(0, 0, 16, seed=1)
        a1 = _make_accumulator(0, 1, 16, seed=2)
        with pytest.raises(ValueError):
            vdam_m_step_single_class(
                state,
                k=5,
                accum_h0=a0,
                accum_h1=a1,
                grad_current_stepsize=0.5,
                tau2_fudge_factor=1.0,
            )


class TestMstepFull:
    def test_multi_class_pseudo_halfsets(self, bind):
        ori = 16
        K = 2
        state = initialise_denovo_state(
            ori_size=ori,
            pixel_size=1.0,
            K=K,
            nr_iter=10,
            n_directions=12,
            pseudo_halfsets=True,
        )
        # Seed Iref nonzero to give reconstructGrad a meaningful reference
        rng = np.random.default_rng(0)
        state.Iref = rng.standard_normal((K, ori, ori, ori))

        # 2K accumulators: [k0_h0, k1_h0, k0_h1, k1_h1]
        accumulators = []
        for h in (0, 1):
            for k in range(K):
                accumulators.append(_make_accumulator(k, h, ori, seed=10 * h + k))

        new_state = vdam_m_step(
            state,
            accumulators=accumulators,
            grad_current_stepsize=0.5,
            tau2_fudge_factor=1.0,
        )
        assert new_state.Iref.shape == (K, ori, ori, ori)
        assert np.all(np.isfinite(new_state.Iref))
        # Each class reference changed
        for k in range(K):
            assert not np.array_equal(new_state.Iref[k], state.Iref[k])

    def test_accumulator_count_mismatch_raises(self, bind):
        state = initialise_denovo_state(
            ori_size=16,
            pixel_size=1.0,
            K=2,
            nr_iter=10,
            n_directions=12,
            pseudo_halfsets=True,
        )
        # Only 3 accumulators when 4 are needed
        accumulators = [_make_accumulator(k, h, 16, seed=k + h) for k, h in [(0, 0), (1, 0), (0, 1)]]
        with pytest.raises(ValueError):
            vdam_m_step(
                state,
                accumulators=accumulators,
                grad_current_stepsize=0.5,
                tau2_fudge_factor=1.0,
            )
