"""The single-class VDAM M-step and its accumulator record have their own owners; ``m_step`` keeps the multi-class driver."""

import inspect

from recovar.em.initial_model import m_step, mstep_accumulator, mstep_single_class


def test_owners_and_routing():
    src = inspect.getsource(m_step)
    for name in ("vdam_m_step_single_class", "_run_m_step_transaction", "_validate_mstep_precision_route", "_maybe_replay_native_bpref_accumulators"):
        assert inspect.getmodule(getattr(mstep_single_class, name)) is mstep_single_class and f"\ndef {name}(" not in src
    assert inspect.getmodule(mstep_accumulator.VdamAccumulator) is mstep_accumulator and "\nclass VdamAccumulator" not in src
    assert m_step.vdam_m_step_single_class is mstep_single_class.vdam_m_step_single_class
    assert m_step.VdamAccumulator is mstep_accumulator.VdamAccumulator
    assert "initial_model.m_step import" not in inspect.getsource(mstep_single_class)
