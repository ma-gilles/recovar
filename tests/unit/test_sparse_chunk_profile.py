"""Selector and cleanup contracts for the optional pass-2 profiler."""
import itertools
import pytest
from recovar.em.diagnostics import chunk_profile


@pytest.fixture
def events(monkeypatch):
    calls = []
    monkeypatch.setattr(chunk_profile, "_CALLS", itertools.count())
    monkeypatch.setattr(chunk_profile.jax, "block_until_ready", lambda x: calls.append(("wait", x)))
    monkeypatch.setattr(chunk_profile.jax.profiler, "start_trace", lambda path, **kw: calls.append(("start", path)))
    monkeypatch.setattr(chunk_profile.jax.profiler, "stop_trace", lambda: calls.append(("stop",)))
    return calls


@pytest.mark.parametrize("spec", ["", "1:0", "0:3"])
def test_unselected_call_has_no_effect(monkeypatch, events, spec):
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_PROFILE_CHUNK", spec)
    with chunk_profile.SparseChunkProfile(3) as profile:
        for i in range(3):
            profile.begin_bucket(i, "before")
            profile.end_bucket("after")
    assert events == []


@pytest.mark.parametrize("fail", [False, True])
def test_selected_bucket_stops_once_even_on_error(monkeypatch, events, fail):
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_PROFILE_CHUNK", "0:1:trace")
    try:
        with chunk_profile.SparseChunkProfile(3) as profile:
            profile.begin_bucket(0, "ignored")
            profile.begin_bucket(1, "before")
            if fail:
                raise RuntimeError("bucket failure")
            profile.end_bucket("after")
    except RuntimeError as error:
        assert fail and str(error) == "bucket failure"
    assert events == [("wait", "before"), ("start", "trace")] + ([] if fail else [("wait", "after")]) + [("stop",)]


def test_wildcard_matches_each_eligible_call(monkeypatch, events):
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_PROFILE_CHUNK", "*:1:trace")
    for size in [1, 2, 3]:
        with chunk_profile.SparseChunkProfile(size) as profile:
            for i in range(size):
                profile.begin_bucket(i, "before")
                profile.end_bucket("after")
    assert sum(e[0] == "start" for e in events) == 2
    assert sum(e[0] == "stop" for e in events) == 2


def test_cleanup_when_completion_wait_fails(monkeypatch, events):
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_PROFILE_CHUNK", "0:0:trace")
    with pytest.raises(RuntimeError, match="wait failed"):
        with chunk_profile.SparseChunkProfile(1) as profile:
            profile.begin_bucket(0, "before")
            def broken_wait(_):
                raise RuntimeError("wait failed")
            monkeypatch.setattr(chunk_profile.jax, "block_until_ready", broken_wait)
            profile.end_bucket("after")
    assert events[-1] == ("stop",)
    assert sum(e[0] == "stop" for e in events) == 1
