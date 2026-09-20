"""P4-L: the half-overlap guard and the two-thread dispatcher.

These cover the decision to overlap and the dispatcher's contract. They do not
cover the E-step itself, which the GPU pair checks by comparing every discrete
field between a serial and an overlapped run.
"""

from __future__ import annotations

import logging
import threading

import pytest

from recovar.em.refinement.iteration_loop import (
    _half_overlap_active,
    _run_halves_overlapped,
)
from recovar.em.refinement.refinement_options import (
    HalfOverlapOptions,
    RefinementOptions,
)

LOG = logging.getLogger(__name__)


def test_overlap_is_off_by_default():
    """The option is a performance experiment, so it must not arm itself."""
    assert RefinementOptions().overlap.overlap_halves is False
    assert HalfOverlapOptions().overlap_halves is False


def test_overlap_requires_the_request():
    assert _half_overlap_active(False, diagnostic_half_indices=(0, 1), log=LOG) is False


def test_overlap_runs_when_both_halves_are_scored():
    assert _half_overlap_active(True, diagnostic_half_indices=(0, 1), log=LOG) is True


@pytest.mark.parametrize("halves", [(0,), (1,), ()])
def test_overlap_refuses_a_half_subset(halves):
    """Nothing to overlap when only one half is being scored."""
    assert _half_overlap_active(True, diagnostic_half_indices=halves, log=LOG) is False


@pytest.mark.parametrize(
    "env_var",
    [
        "RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR",
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR",
        "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
    ],
)
def test_overlap_refuses_while_a_bpref_dump_is_armed(monkeypatch, tmp_path, env_var):
    """The dump context is process-global and set per half, so it wins."""
    monkeypatch.setenv(env_var, str(tmp_path))
    assert _half_overlap_active(True, diagnostic_half_indices=(0, 1), log=LOG) is False


def test_overlap_ignores_a_blank_dump_variable(monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", "   ")
    assert _half_overlap_active(True, diagnostic_half_indices=(0, 1), log=LOG) is True


def test_dispatcher_runs_every_half_once():
    seen = []
    lock = threading.Lock()

    def run_half(k):
        with lock:
            seen.append(k)

    _run_halves_overlapped(run_half, (0, 1))
    assert sorted(seen) == [0, 1]


def test_dispatcher_actually_overlaps():
    """Both halves must be inside their work at the same moment.

    A barrier that needs two participants deadlocks under serial dispatch, so
    passing it is evidence of real concurrency rather than of two sequential
    calls.
    """
    barrier = threading.Barrier(2, timeout=30)

    def run_half(k):
        barrier.wait()

    _run_halves_overlapped(run_half, (0, 1))
    assert barrier.n_waiting == 0


def test_dispatcher_reraises_in_half_order():
    """A failure must surface the way it would have when halves ran in order."""

    class HalfError(RuntimeError):
        pass

    def run_half(k):
        raise HalfError(f"half {k}")

    with pytest.raises(HalfError, match="half 0"):
        _run_halves_overlapped(run_half, (0, 1))


def test_dispatcher_reraises_the_only_failure():
    def run_half(k):
        if k == 1:
            raise ValueError("half 1 only")

    with pytest.raises(ValueError, match="half 1 only"):
        _run_halves_overlapped(run_half, (0, 1))


def test_dispatcher_waits_for_both_before_raising():
    """The surviving half must finish rather than be abandoned mid-flight."""
    finished = []

    def run_half(k):
        if k == 0:
            raise RuntimeError("half 0 fails")
        finished.append(k)

    with pytest.raises(RuntimeError):
        _run_halves_overlapped(run_half, (0, 1))
    assert finished == [1]
