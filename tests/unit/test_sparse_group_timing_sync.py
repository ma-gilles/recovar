"""Diagnostic timing is inert by default and accounts for an enabled fence."""
from recovar.em.diagnostics import sparse_pass2_dump as timing


def test_disabled_timing_does_not_fence(monkeypatch):
    monkeypatch.setattr(timing, '_group_timing_device_barrier_s', lambda: (_ for _ in ()).throw(AssertionError()))
    timing._add_sparse_group_timing(None, 'score', 2.)


def test_elapsed_includes_fence_wait(monkeypatch):
    monkeypatch.setattr(timing, '_group_timing_device_barrier_s', lambda: 0.25)
    values = {'score': 1.}
    timing._add_sparse_group_timing(values, 'score', 2.)
    assert values == {'score': 3.25}


def test_sync_default_off_does_not_create_device_token(monkeypatch):
    monkeypatch.delenv('RECOVAR_SPARSE_KCLASS_GROUP_TIMING_SYNC', raising=False)
    monkeypatch.setattr(timing, '_GROUP_TIMING_SYNC_STATE', {})
    assert timing._group_timing_device_barrier_s() == 0.
    assert timing._GROUP_TIMING_SYNC_STATE == {'enabled': False}
