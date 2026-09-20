"""P4-J: the helper-thread compile-ahead pool.

Ticket: ``em_parity_tickets_20260918/P4J_compile_ahead_and_autotune_default.md``.

The three properties the design rests on are measured here rather than assumed,
because the ticket asks for exactly that:

* a program the helper compiled is a program the main thread's call does NOT
  recompile;
* when both threads start the same program at once, one backend compile happens
  and both resolve to it;
* a job that raises is recorded and dropped, and the run continues.

Plus the option's own contract: off by default, the cap is honoured, duplicate
submissions are collapsed, and the helper is only ever handed avals.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.sparse_pass2 import compile_ahead as ca

pytestmark = pytest.mark.unit


class _CompileCounter:
    """Count XLA backend compiles, and the thread each ran on."""

    def __enter__(self):
        from jax._src import dispatch

        self._dispatch = dispatch
        self._base = dispatch.LogElapsedTimeContextManager
        self._original = dispatch.log_elapsed_time
        self.events = []
        counter = self

        class Timed(self._base):
            def __exit__(self, *exc):
                if self.event == dispatch.BACKEND_COMPILE_EVENT:
                    counter.events.append(
                        (str(self.fun_name), threading.current_thread().name)
                    )
                return counter._base.__exit__(self, *exc)

        dispatch.log_elapsed_time = Timed
        return self

    def __exit__(self, *exc):
        self._dispatch.log_elapsed_time = self._original
        return False

    @property
    def count(self) -> int:
        return len(self.events)


def _program(depth: int):
    """A function whose compile is slow enough to be worth overlapping."""

    @jax.jit
    def run(x):
        y = x
        for i in range(depth):
            y = jnp.sin(y) * (i + 1) + jnp.cos(y[::-1])
        return y.sum()

    return run


def _aval(n=256):
    return jax.ShapeDtypeStruct((n,), jnp.float32)


# ------------------------------------------------------------------ option ---


def test_default_is_off(monkeypatch):
    monkeypatch.delenv(ca.COMPILE_AHEAD_ENV, raising=False)
    assert ca.resolve_compile_ahead_config().enabled is False


@pytest.mark.parametrize("token,expected", [
    ("1", True), ("on", True), ("true", True), ("YES", True),
    ("0", False), ("off", False), ("", False), ("no", False),
])
def test_env_tokens(monkeypatch, token, expected):
    monkeypatch.setenv(ca.COMPILE_AHEAD_ENV, token)
    assert ca.resolve_compile_ahead_config().enabled is expected


def test_explicit_beats_the_environment(monkeypatch):
    monkeypatch.setenv(ca.COMPILE_AHEAD_ENV, "1")
    assert ca.resolve_compile_ahead_config(False).enabled is False
    monkeypatch.setenv(ca.COMPILE_AHEAD_ENV, "0")
    assert ca.resolve_compile_ahead_config(True).enabled is True


def test_a_bad_token_is_refused(monkeypatch):
    monkeypatch.setenv(ca.COMPILE_AHEAD_ENV, "sometimes")
    with pytest.raises(ValueError):
        ca.resolve_compile_ahead_config()


def test_disabled_pool_accepts_nothing():
    with ca.CompileAheadPool(ca.CompileAheadConfig(enabled=False)) as pool:
        assert pool.submit("p", _program(2), (_aval(),)) is False
    assert pool.summary.submitted == 0
    assert pool.summary.compiled == 0


# ------------------------------------------------------------ the mechanism ---


def test_the_helper_warms_the_main_threads_call():
    program = _program(40)
    aval = _aval()
    x = jnp.arange(aval.shape[0], dtype=jnp.float32)
    with _CompileCounter() as counter:
        with ca.CompileAheadPool(ca.CompileAheadConfig(enabled=True)) as pool:
            assert pool.submit("warm", program, (aval,)) is True
        assert pool.summary.compiled == 1, pool.summary.errors
        after_warm = counter.count
        program(x).block_until_ready()
        assert counter.count == after_warm, (
            "the main thread recompiled a program the helper had already compiled: "
            f"{counter.events[after_warm:]}"
        )


def test_a_race_resolves_to_one_compile():
    program = _program(40)
    aval = _aval()
    x = jnp.arange(aval.shape[0], dtype=jnp.float32)
    with _CompileCounter() as counter:
        with ca.CompileAheadPool(ca.CompileAheadConfig(enabled=True)) as pool:
            pool.submit("race", program, (aval,))
            program(x).block_until_ready()
    backend = [e for e in counter.events if "run" in e[0]]
    assert len(backend) <= 1, f"the program was compiled {len(backend)} times: {backend}"


def test_a_failing_job_is_recorded_and_the_run_continues():
    good = _program(3)
    with ca.CompileAheadPool(ca.CompileAheadConfig(enabled=True)) as pool:
        # a one-argument program handed two avals: lowering raises TypeError.
        # (An int or higher-rank aval is NOT a failure here -- this program is
        # rank-agnostic and int32 promotes, which the first version of this
        # test got wrong.)
        pool.submit("bad", good, (_aval(), _aval()))
        pool.submit("good", good, (_aval(),))
    assert pool.summary.failed == 1
    assert pool.summary.compiled == 1
    assert pool.summary.errors and "bad" in pool.summary.errors[0]


def test_the_helper_is_only_ever_handed_avals():
    program = _program(3)
    seen = []

    class Recorder:
        def lower(self, *args, **kwargs):
            seen.extend(args)
            return program.lower(*args, **kwargs)

    with ca.CompileAheadPool(ca.CompileAheadConfig(enabled=True)) as pool:
        pool.submit("aval only", Recorder(), (_aval(),))
    assert seen and all(isinstance(a, jax.ShapeDtypeStruct) for a in seen)


# ------------------------------------------------------------------- limits ---


def test_the_cap_is_honoured():
    config = ca.CompileAheadConfig(enabled=True, max_programs=2)
    with ca.CompileAheadPool(config) as pool:
        accepted = [pool.submit(f"p{i}", _program(2), (_aval(8 + i),)) for i in range(5)]
    assert accepted.count(True) == 2
    assert pool.summary.submitted == 2
    assert pool.summary.refused == 3


def test_duplicate_submissions_are_collapsed():
    program = _program(2)
    with ca.CompileAheadPool(ca.CompileAheadConfig(enabled=True)) as pool:
        first = pool.submit("same", program, (_aval(),))
        second = pool.submit("same", program, (_aval(),))
    assert first is True and second is False
    assert pool.summary.submitted == 1


def test_a_negative_cap_is_refused():
    with pytest.raises(ValueError):
        ca.CompileAheadConfig(enabled=True, max_programs=-1)
