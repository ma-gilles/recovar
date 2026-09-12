"""The composed native update must equal the existing primitive sequence."""

from copy import deepcopy
from dataclasses import fields

import numpy as np
import pytest

from recovar.em.vdam import initialise_denovo_state
from recovar.em.vdam.mstep_accumulator import VdamAccumulator
from recovar.em.vdam.mstep_single_class import vdam_m_step_single_class

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def transaction_bind():
    from recovar.relion_bind import _relion_bind_core as bind

    assert hasattr(bind, "vdam_m_step_transaction"), "build the transaction binding"
    return bind


def _case(K, pseudo, current_size, populated_moments, padding_factor=1):
    n = 16
    state = initialise_denovo_state(
        ori_size=n,
        pixel_size=1.5,
        K=K,
        nr_iter=200,
        n_directions=12,
        pseudo_halfsets=pseudo,
        padding_factor=padding_factor,
    )
    state.current_size = current_size
    state.iter = 48
    rng = np.random.default_rng(29)
    state.Iref[:] = rng.normal(size=state.Iref.shape)
    state.tau2_class[:] = rng.uniform(0.1, 2.0, state.tau2_class.shape)
    state.fsc_halves_class[:] = rng.uniform(0.1, 0.9, state.fsc_halves_class.shape)
    if populated_moments:
        state.Igrad1[:] = rng.normal(size=state.Igrad1.shape) + 1j * rng.normal(size=state.Igrad1.shape)
        state.Igrad2[:] = rng.uniform(0.5, 2, state.Igrad2.shape) + 1j
    pad = (current_size or n) * padding_factor + 3
    shape = (pad, pad, pad // 2 + 1)
    accumulators = [
        VdamAccumulator(
            data=rng.normal(size=shape) + 1j * rng.normal(size=shape),
            weight=rng.uniform(10, 100, shape),
            class_idx=K - 1,
            halfset_idx=h,
        )
        for h in range(2 if pseudo else 1)
    ]
    return state, accumulators


def _assert_state_exact(actual, expected):
    for field in fields(actual):
        a, e = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, e, err_msg=field.name)
        else:
            assert a == e, field.name


@pytest.mark.parametrize("K,pseudo", [(1, False), (1, True), (4, False), (4, True)])
@pytest.mark.parametrize("current_size", [0, 8, 16])
@pytest.mark.parametrize("padding_factor", [1, 2])
@pytest.mark.parametrize("populated_moments", [False, True])
def test_transaction_matches_primitives_exactly(
    transaction_bind,
    K,
    pseudo,
    current_size,
    populated_moments,
    padding_factor,
):
    state, accumulators = _case(K, pseudo, current_size, populated_moments, padding_factor)
    original = deepcopy(state)
    original_accum = deepcopy(accumulators)
    args = dict(
        k=K - 1,
        accum_h0=accumulators[0],
        accum_h1=accumulators[1] if pseudo else None,
        grad_current_stepsize=0.3,
        tau2_fudge_factor=4.0,
        padding_factor=padding_factor,
    )
    expected = vdam_m_step_single_class(state, **args, use_native_transaction=False)
    actual = vdam_m_step_single_class(state, **args, use_native_transaction=True)
    _assert_state_exact(actual, expected)
    _assert_state_exact(state, original)
    for a, e in zip(accumulators, original_accum):
        np.testing.assert_array_equal(a.data, e.data)
        np.testing.assert_array_equal(a.weight, e.weight)


def test_dump_keeps_primitive_boundaries(transaction_bind, monkeypatch, tmp_path):
    state, accumulators = _case(1, True, 8, False)
    monkeypatch.setenv("RECOVAR_MSTEP_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_MSTEP_DUMP_ITER", "48")

    def forbidden(*args, **kwargs):
        raise AssertionError("dump bypassed primitive boundaries")

    monkeypatch.setattr(transaction_bind, "vdam_m_step_transaction", forbidden)
    vdam_m_step_single_class(
        state,
        k=0,
        accum_h0=accumulators[0],
        accum_h1=accumulators[1],
        grad_current_stepsize=0.3,
        tau2_fudge_factor=4.0,
    )
    assert (tmp_path / "data_h0_post_reweight.npy").is_file()
    assert (tmp_path / "iref_out_relion_frame.npy").is_file()


@pytest.mark.parametrize("K,pseudo", [(1, False), (1, True), (4, False), (4, True)])
@pytest.mark.parametrize("current_size,padding", [(0, 1), (8, 1), (16, 2)])
def test_device_backend_preserves_complete_state_and_inputs(transaction_bind, K, pseudo, current_size, padding, monkeypatch):
    from recovar.em.relion import relion_vdam_mstep

    state, accumulators = _case(K, pseudo, current_size, True, padding)
    original = deepcopy(state)
    original_accum = deepcopy(accumulators)
    args = dict(k=K-1, accum_h0=accumulators[0],
                accum_h1=accumulators[1] if pseudo else None,
                grad_current_stepsize=0.3, tau2_fudge_factor=4.0, padding_factor=padding)
    expected = vdam_m_step_single_class(state, **args)
    calls = []
    actual_host = relion_vdam_mstep.relion_vdam_m_step_host

    def counted(*a, **kw):
        calls.append(1)
        return actual_host(*a, **kw)

    monkeypatch.setattr(relion_vdam_mstep, "relion_vdam_m_step_host", counted)
    actual = vdam_m_step_single_class(state, **args, mstep_backend="jax")
    assert calls == [1]
    changed_arrays = {"Iref", "Igrad1", "Igrad2", "tau2_class", "sigma2_class",
                      "data_vs_prior_class", "fourier_coverage_class"}
    for field in fields(actual):
        a, e = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, np.ndarray):
            assert a.shape == e.shape and a.dtype == e.dtype, field.name
            if field.name in changed_arrays and np.any(e):
                difference = np.abs(a-e)
                metrics = (np.linalg.norm(difference.ravel()) / np.linalg.norm(e.ravel()),
                           difference.max() / np.abs(e).max(), difference.mean() / np.abs(e).mean())
                assert max(metrics) < 1e-12, (field.name, metrics)
            else:
                np.testing.assert_array_equal(a, e, err_msg=field.name)
        else:
            assert a == e, field.name
    _assert_state_exact(state, original)
    for a, e in zip(accumulators, original_accum):
        np.testing.assert_array_equal(a.data, e.data)
        np.testing.assert_array_equal(a.weight, e.weight)


def test_device_request_keeps_native_dump_boundaries(transaction_bind, monkeypatch, tmp_path):
    from recovar.em.relion import relion_vdam_mstep

    state, accumulators = _case(1, True, 8, False)
    monkeypatch.setenv("RECOVAR_MSTEP_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_MSTEP_DUMP_ITER", "48")

    def forbidden(*args, **kwargs):
        raise AssertionError("device request bypassed native diagnostic boundaries")

    monkeypatch.setattr(relion_vdam_mstep, "relion_vdam_m_step_host", forbidden)
    monkeypatch.setattr(transaction_bind, "vdam_m_step_transaction", forbidden)
    vdam_m_step_single_class(state, k=0, accum_h0=accumulators[0], accum_h1=accumulators[1],
                            grad_current_stepsize=0.3, tau2_fudge_factor=4.0, mstep_backend="jax")
    assert (tmp_path / "data_h0_post_reweight.npy").is_file()
