"""Regression tests for the K=1 halfset-averaging SsnrMap fix.

Pinned at commit ``db4f49c4`` after the unconditional version
(``58092af8``) regressed K>1 trajectories. Behavior locked in:

* ``state.K == 1`` AND ``accum_h1 is not None`` → SsnrMap weight is
  ``0.5 * (accum_h0.weight + accum_h1.weight)``. RELION at iter-1
  passes a single unified-halfset BPref weight to ``updateSSNRarrays``
  (``do_split_random_halves=0``); recovar's pseudo-halfset architecture
  produces two per-class accumulators, each ~equivalent to a full pass
  via Hermitian doubling, so averaging recovers the unified weight.
  Aligns autosampling timing with RELION (HEALPix 1→2 fires at iter-10
  in the K=1 nr_iter=10 PDB walkthrough).

* ``state.K > 1`` (any pseudo_halfsets setting) → SsnrMap weight is
  ``accum_h0.weight``. Per-class softmax assignment makes h0/h1 not
  symmetric across particles; averaging there inflates resolution and
  regressed K=4 nr_iter=10 mean CC vs RELION from 0.997 → 0.90 in one
  test before this conditional was added.

* ``accum_h1 is None`` (pseudo_halfsets=False) → SsnrMap weight is
  ``accum_h0.weight``. Falls back to single-set behavior cleanly.

The tests intercept ``vdam_update_ssnr_arrays_from_bpref`` to inspect
the weight argument exactly as ``vdam_m_step_single_class`` calls it,
without relying on downstream tau2/sigma2 values that are sensitive to
the binding's numerical details.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.initial_model import initialise_denovo_state
from recovar.em.initial_model.m_step import (
    VdamAccumulator,
    vdam_m_step_single_class,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def _bind_module():
    try:
        from recovar.relion_bind import _relion_bind_core as m
    except ImportError:
        pytest.skip("relion_bind not built")
    if not hasattr(m, "vdam_update_ssnr_arrays_from_bpref"):
        pytest.skip("relion_bind built without VDAM SsnrMap primitive; rebuild recovar/relion_bind")
    return m


def _make_accumulator(k: int, h: int, ori_size: int, seed: int, weight_scale: float) -> VdamAccumulator:
    """Build a deterministic accumulator with the RELION pad=1 shape."""
    Nz = ori_size
    Ny = ori_size
    Nx_half = ori_size // 2 + 1
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal((Nz, Ny, Nx_half)) + 1j * rng.standard_normal((Nz, Ny, Nx_half))).astype(np.complex128)
    weight = (rng.uniform(1.0, 10.0, size=(Nz, Ny, Nx_half)) * weight_scale).astype(np.float64)
    return VdamAccumulator(data=data, weight=weight, class_idx=k, halfset_idx=h)


class _SsnrCallRecorder:
    """Capture the first positional argument passed to vdam_update_ssnr_arrays_from_bpref."""

    def __init__(self, real_fn):
        self._real = real_fn
        self.captured_weight: np.ndarray | None = None
        self.call_count = 0

    def __call__(self, weight, *args, **kwargs):
        # Copy because the binding may overwrite/free the input.
        self.captured_weight = np.asarray(weight, dtype=np.float64).copy()
        self.call_count += 1
        return self._real(weight, *args, **kwargs)


def _patch_ssnr(monkeypatch, bind):
    rec = _SsnrCallRecorder(bind.vdam_update_ssnr_arrays_from_bpref)
    monkeypatch.setattr(bind, "vdam_update_ssnr_arrays_from_bpref", rec)
    return rec


def test_k1_pseudo_halfsets_uses_averaged_weight(_bind_module, monkeypatch):
    """K=1 with both halfsets present → SsnrMap input is 0.5*(h0+h1)."""
    rec = _patch_ssnr(monkeypatch, _bind_module)

    ori = 16
    state = initialise_denovo_state(ori_size=ori, pixel_size=1.0, K=1, nr_iter=10, n_directions=12, pseudo_halfsets=True)
    state.Iref[0] = np.random.default_rng(0).standard_normal((ori, ori, ori))
    a0 = _make_accumulator(k=0, h=0, ori_size=ori, seed=1, weight_scale=1.0)
    a1 = _make_accumulator(k=0, h=1, ori_size=ori, seed=2, weight_scale=2.0)

    vdam_m_step_single_class(
        state,
        k=0,
        accum_h0=a0,
        accum_h1=a1,
        grad_current_stepsize=0.5,
        tau2_fudge_factor=1.0,
    )

    assert rec.call_count == 1
    expected = 0.5 * (a0.weight + a1.weight)
    np.testing.assert_array_equal(rec.captured_weight, expected)
    # Sanity: the average is genuinely different from h0 alone (so the
    # condition is non-trivial — different scales above guarantee this).
    assert not np.array_equal(rec.captured_weight, a0.weight)


def test_k1_no_halfsets_uses_h0_weight(_bind_module, monkeypatch):
    """K=1 with pseudo_halfsets=False → SsnrMap input is accum_h0.weight unchanged."""
    rec = _patch_ssnr(monkeypatch, _bind_module)

    ori = 16
    state = initialise_denovo_state(ori_size=ori, pixel_size=1.0, K=1, nr_iter=10, n_directions=12, pseudo_halfsets=False)
    state.Iref[0] = np.random.default_rng(0).standard_normal((ori, ori, ori))
    a0 = _make_accumulator(k=0, h=0, ori_size=ori, seed=1, weight_scale=1.0)

    vdam_m_step_single_class(
        state,
        k=0,
        accum_h0=a0,
        accum_h1=None,
        grad_current_stepsize=0.5,
        tau2_fudge_factor=1.0,
    )

    assert rec.call_count == 1
    np.testing.assert_array_equal(rec.captured_weight, a0.weight)


@pytest.mark.parametrize("K", [2, 4])
def test_k_class_pseudo_halfsets_uses_h0_weight(_bind_module, monkeypatch, K):
    """K>1 with pseudo_halfsets=True → SsnrMap input is accum_h0.weight, NOT averaged.

    Locks down the K=4 regression fix: averaging here drove K=4 nr_iter=10
    mean CC from 0.997 → 0.90 because per-class halfset accumulators are
    not symmetric under softmax class assignment. Even when accum_h1 is
    available, K>1 must keep the h0-only path until the K-class structural
    rewrite (Class3D single-set + prev-Iref tau2) lands.
    """
    rec = _patch_ssnr(monkeypatch, _bind_module)

    ori = 16
    state = initialise_denovo_state(ori_size=ori, pixel_size=1.0, K=K, nr_iter=10, n_directions=12, pseudo_halfsets=True)
    rng = np.random.default_rng(0)
    state.Iref[0] = rng.standard_normal((ori, ori, ori))
    a0 = _make_accumulator(k=0, h=0, ori_size=ori, seed=1, weight_scale=1.0)
    a1 = _make_accumulator(k=0, h=1, ori_size=ori, seed=2, weight_scale=2.0)

    vdam_m_step_single_class(
        state,
        k=0,
        accum_h0=a0,
        accum_h1=a1,
        grad_current_stepsize=0.5,
        tau2_fudge_factor=1.0,
    )

    assert rec.call_count == 1
    np.testing.assert_array_equal(rec.captured_weight, a0.weight)
    # And critically NOT the averaged weight.
    averaged = 0.5 * (a0.weight + a1.weight)
    assert not np.array_equal(rec.captured_weight, averaged), (
        f"K={K} M-step is averaging halfset weights — this regressed K>1 in 58092af8 "
        f"and must be guarded against in the upcoming EM/VDAM/PPCA merge."
    )
