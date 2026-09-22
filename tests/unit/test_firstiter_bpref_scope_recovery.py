"""Final-Q native first-iteration scope guards at the current owner."""
import pytest
from recovar.em.diagnostics.bpref_diagnostics import _bpref_contribution_context
from recovar.em.sparse_pass2.firstiter_bpref import _relion_firstiter_fused_bpref_enabled

def test_firstiter_fused_bpref_defaults_only_inside_complete_fresh_k1_guard(monkeypatch):
    monkeypatch.delenv("RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF", raising=False)
    monkeypatch.setitem(_bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(_bpref_contribution_context, "half", 2)
    kwargs = dict(
        fresh_k1_guard=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        relion_exact_bpref_operands=True,
        use_relion_x_half_mstep=True,
        score_only=False,
    )

    assert _relion_firstiter_fused_bpref_enabled(**kwargs)
    assert not _relion_firstiter_fused_bpref_enabled(
        **{**kwargs, "winner_take_all": False}
    )

def test_firstiter_fused_bpref_override_can_disable_but_not_expand_scope(monkeypatch):
    monkeypatch.setitem(_bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(_bpref_contribution_context, "half", 1)
    kwargs = dict(
        fresh_k1_guard=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        relion_exact_bpref_operands=True,
        use_relion_x_half_mstep=True,
        score_only=False,
    )
    monkeypatch.setenv("RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF", "0")
    assert not _relion_firstiter_fused_bpref_enabled(**kwargs)

    monkeypatch.setenv("RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF", "1")
    with pytest.raises(ValueError, match="requires the fresh K=1"):
        _relion_firstiter_fused_bpref_enabled(
            **{**kwargs, "fresh_k1_guard": False}
        )
