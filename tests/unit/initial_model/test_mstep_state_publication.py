"""Exact state and ownership checks for avoiding overwritten M-step copies."""

from dataclasses import fields

import numpy as np
import pytest

from recovar.em.vdam import m_step, mstep_single_class
from recovar.em.vdam.state import InitialModelState, half_slot_index

pytestmark = pytest.mark.unit
SELECTOR = "RECOVAR_VDAM_MSTEP_COPY_UNTOUCHED"
CHANGED = {"Iref", "Igrad1", "Igrad2", "tau2_class", "sigma2_class", "data_vs_prior_class", "fourier_coverage_class"}


def _case(K, pseudo, order):
    rng = np.random.default_rng(21)

    def array(shape, complex_values=False):
        values = rng.normal(size=shape)
        if complex_values:
            values = values + 1j * rng.normal(size=shape)
        # Untouched slots must retain nonfinite payloads and signed zeros too.
        words = values.view(np.uint64).reshape(-1)
        patterns = [0, 0x8000000000000000, 0x7FF8000000000021, 0x7FF0000000000000]
        words[: min(4, words.size)] = patterns[: min(4, words.size)]
        if order == "F":
            values = np.asfortranarray(values)
        elif order == "strided":
            values = values[..., ::-1]
        return values

    state = InitialModelState(
        K=K,
        ori_size=4,
        pseudo_halfsets=pseudo,
        Iref=array((K, 4, 4, 4)),
        Igrad1=array(((2 if pseudo else 1) * K, 4, 4, 3), True),
        Igrad2=array((K, 4, 4, 3), True),
        pdf_class=np.full(K, 1 / K),
        fsc_halves_class=np.ones((K, 3)),
    )
    for name in CHANGED - {"Iref", "Igrad1", "Igrad2"}:
        setattr(state, name, array((K, 3)))
    return state


def _call(state, k, transaction):
    accum = m_step.VdamAccumulator(np.zeros((4, 4, 3), dtype=np.complex128), np.ones((4, 4, 3)), k, 0)
    return mstep_single_class._run_m_step_transaction(
        transaction,
        state,
        k,
        accum,
        accum if state.pseudo_halfsets else None,
        grad_current_stepsize=0.3,
        tau2_fudge_factor=4.0,
        padding_factor=1,
        r_max=2,
        min_resol_shell=1.0,
    )


@pytest.mark.parametrize("K,k", [(1, 0), (4, 0), (4, 1), (4, 3)])
@pytest.mark.parametrize("pseudo", [False, True])
@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_publication_matches_full_copy_and_preserves_ownership(monkeypatch, K, k, pseudo, order):
    state = _case(K, pseudo, order)
    originals = {
        f.name: getattr(state, f.name).tobytes()
        for f in fields(state)
        if isinstance(getattr(state, f.name), np.ndarray)
    }
    # Transaction outputs deliberately alias input state: publication must
    # still return independent storage, including for every updated slot.
    result = {
        "iref": state.Iref[k],
        "mom1_h0": state.Igrad1[k],
        "mom1_h1": state.Igrad1[-1],
        "mom2": state.Igrad2[k],
        "tau2": state.tau2_class[k],
        "sigma2": state.sigma2_class[k],
        "data_vs_prior": state.data_vs_prior_class[k],
        "fourier_coverage": state.fourier_coverage_class[k],
    }
    for name in originals:
        getattr(state, name).flags.writeable = False
    calls = []

    def transaction(*args):
        calls.append(tuple((a.shape, a.dtype, a.tobytes()) if isinstance(a, np.ndarray) else a for a in args))
        return result

    monkeypatch.setenv(SELECTOR, "0")
    expected = _call(state, k, transaction)
    monkeypatch.setenv(SELECTOR, "1")
    slots = []
    original = mstep_single_class._copy_mstep_untouched_slots

    def observe(values, updated):
        slots.append(updated)
        return original(values, updated)

    monkeypatch.setattr(mstep_single_class, "_copy_mstep_untouched_slots", observe)
    actual = _call(state, k, transaction)
    assert calls[0] == calls[1]
    h1 = half_slot_index(k, 1, K, True) if pseudo else None
    assert slots == [(k,), (k, h1) if pseudo else (k,), (k,)]
    assert actual is not state
    for f in fields(state):
        a, e, before = getattr(actual, f.name), getattr(expected, f.name), getattr(state, f.name)
        if isinstance(a, np.ndarray):
            assert a.shape == e.shape and a.dtype == e.dtype
            assert a.tobytes() == e.tobytes(), f.name
            assert before.tobytes() == originals[f.name], f.name
            if f.name in CHANGED:
                assert a.flags.c_contiguous and a.flags.writeable
                assert not np.shares_memory(a, before), f.name
                assert all(not np.shares_memory(a, v) for v in result.values()), f.name
            else:
                assert a is before
        else:
            assert a == e == before


@pytest.mark.parametrize("token", ["", "true", "2", " 1", "1 "])
def test_invalid_selector_rejected_before_transaction(monkeypatch, token):
    monkeypatch.setenv(SELECTOR, token)

    def forbidden(*args):
        raise AssertionError("invalid selector invoked scientific transaction")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        _call(_case(1, True, "C"), 0, forbidden)


def test_selector_defaults_to_existing_copy_path(monkeypatch):
    monkeypatch.delenv(SELECTOR, raising=False)

    def forbidden(*args):
        raise AssertionError("default enabled new publication helper")

    monkeypatch.setattr(mstep_single_class, "_copy_mstep_untouched_slots", forbidden)
    state = _case(1, False, "C")
    result = {
        "iref": state.Iref[0],
        "mom1_h0": state.Igrad1[0],
        "mom2": state.Igrad2[0],
        "tau2": state.tau2_class[0],
        "sigma2": state.sigma2_class[0],
        "data_vs_prior": state.data_vs_prior_class[0],
        "fourier_coverage": state.fourier_coverage_class[0],
    }
    actual = _call(state, 0, lambda *args: result)
    assert actual.Igrad1.tobytes() == state.Igrad1.tobytes()
    assert not np.shares_memory(actual.Igrad1, state.Igrad1)
