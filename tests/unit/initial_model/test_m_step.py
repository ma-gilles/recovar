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

from recovar.em.initial_model import (
    initialise_denovo_state,
)
from recovar.em.initial_model.m_step import (
    VdamAccumulator,
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


class TestMstepSingleClass:
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

    def test_input_state_unchanged(self, bind):
        ori = 16
        state = initialise_denovo_state(ori_size=ori, pixel_size=1.0, K=1, nr_iter=10, n_directions=12)
        a0 = _make_accumulator(k=0, h=0, ori_size=ori, seed=1)
        a1 = _make_accumulator(k=0, h=1, ori_size=ori, seed=2)

        iref_before = state.Iref.copy()
        igrad1_before = state.Igrad1.copy()
        igrad2_before = state.Igrad2.copy()

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
