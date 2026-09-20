"""Compile predictable programs on a helper thread while the main thread works.

Why this can work at all, measured rather than assumed (P4-J, 2026-09-20):

* XLA's backend compile releases the GIL (a spinner thread keeps 97 % of its
  idle rate on an A100 while the main thread compiles), and on GPU that phase is
  95 % of lower-plus-compile. Lowering holds the GIL, so a helper pays about a
  twentieth of its work in contention with the main thread.
* ``f.lower(*avals).compile()`` on a helper thread warms the *call*: a later
  ``f(x)`` on the main thread issues no further backend compile and returns in
  0.03 s against 0.47 s cold.
* If both threads start the same program at once, exactly one backend compile
  happens and both resolve to it. JAX's own compilation cache deduplicates, so
  a helper that loses the race costs nothing but its own wait.

The pool is deliberately narrow:

* jobs carry ``jax.ShapeDtypeStruct`` avals, never real arrays, so the helper
  cannot touch device buffers and cannot have a side effect on program state;
* a job that raises is recorded and dropped, never re-raised into the main
  thread, because a failed warm-up must not change what the run computes;
* the pool is capped, so a mis-specified caller cannot queue unbounded work;
* it is off unless asked for.

Nothing here changes what any program computes. A warmed program is the same
program the main thread would have compiled, with the same signature, so the
option is a performance switch and its correctness claim is bitwise equality.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

COMPILE_AHEAD_ENV = "RECOVAR_EM_COMPILE_AHEAD"
COMPILE_AHEAD_MAX_ENV = "RECOVAR_EM_COMPILE_AHEAD_MAX_PROGRAMS"
DEFAULT_MAX_PROGRAMS = 64


@dataclass(frozen=True)
class CompileAheadConfig:
    """Typed configuration for the helper-thread warm-up. Off by default."""

    enabled: bool = False
    max_programs: int = DEFAULT_MAX_PROGRAMS

    def __post_init__(self) -> None:
        if self.max_programs < 0:
            raise ValueError("max_programs must not be negative")


def resolve_compile_ahead_config(explicit=None) -> CompileAheadConfig:
    """Resolve the option; ``explicit`` beats the environment.

    Accepted: ``None`` (consult the environment), a ``CompileAheadConfig``,
    ``False``/``"0"``/``"off"``/``""`` (disabled), ``True``/``"1"``/``"on"``
    (enabled with the default cap). Only the EM entry points call this with
    ``None``; every other caller passes an explicit value, so the variable
    cannot re-configure a pipeline that has its own validation.
    """

    if isinstance(explicit, CompileAheadConfig):
        return explicit
    raw = explicit
    if raw is None:
        raw = os.environ.get(COMPILE_AHEAD_ENV, "")
    if raw is False:
        return CompileAheadConfig(enabled=False)
    if raw is True:
        return CompileAheadConfig(enabled=True, max_programs=_max_programs_from_env())
    token = str(raw).strip().lower()
    if token in {"", "0", "off", "false", "no", "none"}:
        return CompileAheadConfig(enabled=False)
    if token in {"1", "on", "true", "yes"}:
        return CompileAheadConfig(enabled=True, max_programs=_max_programs_from_env())
    raise ValueError(
        f"{COMPILE_AHEAD_ENV} must be one of 0/off/false/no or 1/on/true/yes, got {raw!r}"
    )


def _max_programs_from_env() -> int:
    raw = os.environ.get(COMPILE_AHEAD_MAX_ENV, "")
    if not raw.strip():
        return DEFAULT_MAX_PROGRAMS
    value = int(raw)
    if value < 0:
        raise ValueError(f"{COMPILE_AHEAD_MAX_ENV} must not be negative")
    return value


@dataclass
class CompileAheadSummary:
    """What the pool did, for the run's log and for a test to assert on."""

    submitted: int = 0
    compiled: int = 0
    refused: int = 0
    failed: int = 0
    seconds: float = 0.0
    errors: list = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"compile-ahead: {self.compiled} programs from {self.submitted} jobs in "
            f"{self.seconds:.2f}s ({self.refused} over the cap, {self.failed} failed)"
        )


class CompileAheadPool:
    """One helper thread that lowers and compiles jobs submitted to it.

    Use as a context manager around the work the compiles should overlap::

        with CompileAheadPool(config) as pool:
            pool.submit("chunk posterior", program, avals, static)
            ...run pass 1 on the main thread...
        summary = pool.summary
    """

    def __init__(self, config: CompileAheadConfig):
        self._config = config
        self._queue: queue.Queue = queue.Queue()
        self._summary = CompileAheadSummary()
        self._lock = threading.Lock()
        self._thread = None
        self._seen: set = set()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "CompileAheadPool":
        if self._config.enabled:
            self._thread = threading.Thread(
                target=self._run, name="recovar-compile-ahead", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def close(self) -> CompileAheadSummary:
        """Stop accepting work and wait for what is queued."""

        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
        return self._summary

    @property
    def summary(self) -> CompileAheadSummary:
        return self._summary

    # -- submission --------------------------------------------------------

    def submit(self, label, program, avals, static_kwargs=None) -> bool:
        """Queue one program for warm-up. Returns whether it was accepted.

        ``avals`` must be ``jax.ShapeDtypeStruct`` (or anything else with a
        shape and a dtype and no device buffer): the helper never receives a
        real array, so it cannot read or free one.
        """

        if not self._config.enabled:
            return False
        key = (label, _aval_key(avals), _static_key(static_kwargs))
        with self._lock:
            if key in self._seen:
                return False
            if self._summary.submitted >= self._config.max_programs:
                self._summary.refused += 1
                return False
            self._seen.add(key)
            self._summary.submitted += 1
        self._queue.put((label, _direct(program, tuple(avals), dict(static_kwargs or {}))))
        return True

    def submit_thunk(self, label, thunk) -> bool:
        """Queue work whose avals are themselves derived on the helper thread.

        Some signatures are only reachable by tracing a real function, which
        costs host time the submitting thread is trying to spend elsewhere.
        ``thunk()`` runs on the helper and returns ``(program, avals,
        static_kwargs)``, or a list of such triples when one derivation feeds
        several programs -- the per-stage chunk path is three programs over one
        set of operand avals. It must close over shape/dtype stand-ins only,
        never a device buffer, and a raise inside it is recorded and dropped
        exactly like a failed compile.

        Deduplication is by ``label`` alone here, because the avals do not
        exist yet when the job is queued: give each distinct signature a label
        that names it.
        """

        if not self._config.enabled:
            return False
        key = ("thunk", label)
        with self._lock:
            if key in self._seen:
                return False
            if self._summary.submitted >= self._config.max_programs:
                self._summary.refused += 1
                return False
            self._seen.add(key)
            self._summary.submitted += 1
        self._queue.put((label, thunk))
        return True

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            label, thunk = job
            start = time.perf_counter()
            done = 0
            try:
                work = thunk()
                if not isinstance(work, list):
                    work = [work]
                for program, avals, static_kwargs in work:
                    program.lower(*avals, **static_kwargs).compile()
                    done += 1
            except Exception as exc:  # noqa: BLE001 - a warm-up must never fail a run
                with self._lock:
                    self._summary.failed += 1
                    self._summary.errors.append(f"{label}: {type(exc).__name__}: {exc}")
                logger.debug("compile-ahead skipped %s: %r", label, exc)
            finally:
                with self._lock:
                    self._summary.seconds += time.perf_counter() - start
                    self._summary.compiled += done


def _aval_key(avals) -> tuple:
    out = []
    for value in avals:
        shape = tuple(int(d) for d in getattr(value, "shape", ()))
        dtype = getattr(getattr(value, "dtype", None), "name", str(getattr(value, "dtype", None)))
        out.append((shape, dtype))
    return tuple(out)


def _static_key(static_kwargs) -> tuple:
    if not static_kwargs:
        return ()
    return tuple(sorted((str(k), repr(v)) for k, v in static_kwargs.items()))


def _direct(program, avals, static_kwargs):
    """Wrap an already-derived signature as the thunk the worker runs."""

    def thunk():
        return program, avals, static_kwargs

    return thunk
