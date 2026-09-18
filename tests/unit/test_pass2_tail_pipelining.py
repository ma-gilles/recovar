"""Bucket-tail pipelining: ordered worker, error propagation, opt-in knob."""

import threading
import time

import pytest

from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed


def test_pipeline_tail_knob_is_default_off(monkeypatch):
    monkeypatch.delenv(bucketed._PIPELINE_TAIL_ENV, raising=False)
    assert bucketed._pipeline_tail_enabled() is False
    monkeypatch.setenv(bucketed._PIPELINE_TAIL_ENV, "1")
    assert bucketed._pipeline_tail_enabled() is True
    assert bucketed._PIPELINE_TAIL_ENV == "RECOVAR_SPARSE_PASS2_PIPELINE_TAIL"


def test_runner_preserves_submission_order_and_runs_off_main_thread():
    seen = []
    threads = set()

    def tail(snap):
        threads.add(threading.current_thread().name)
        time.sleep(0.005 * (3 - snap["i"] % 3))  # uneven durations must not reorder
        seen.append(snap["i"])

    runner = bucketed._BucketTailRunner(tail)
    for i in range(12):
        runner.submit({"i": i})
    runner.drain()
    assert seen == list(range(12))
    assert threads == {"pass2-bucket-tail"}


def test_runner_bounds_lookahead_to_one_pending_tail():
    started = threading.Event()
    release = threading.Event()
    order = []

    def tail(snap):
        order.append(("start", snap["i"]))
        if snap["i"] == 0:
            started.set()
            release.wait(5)
        order.append(("end", snap["i"]))

    runner = bucketed._BucketTailRunner(tail)
    runner.submit({"i": 0})
    assert started.wait(5)
    runner.submit({"i": 1})  # fits in the one-slot queue
    blocked = threading.Event()

    def third():
        runner.submit({"i": 2})  # must block until tail 0 finishes and tail 1 is taken
        blocked.set()

    t = threading.Thread(target=third)
    t.start()
    time.sleep(0.05)
    assert not blocked.is_set(), "submit did not block with a full lookahead slot"
    release.set()
    t.join(5)
    runner.drain()
    assert [i for kind, i in order if kind == "end"] == [0, 1, 2]


def test_runner_reraises_worker_failure_on_main_thread():
    def tail(snap):
        if snap["i"] == 1:
            raise RuntimeError("tail failed on bucket 1")

    runner = bucketed._BucketTailRunner(tail)
    runner.submit({"i": 0})
    runner.submit({"i": 1})
    with pytest.raises(RuntimeError, match="bucket 1"):
        for i in range(2, 6):
            runner.submit({"i": i})
        runner.drain()


def test_snapshot_names_are_the_tail_inputs():
    names = bucketed._BUCKET_TAIL_SNAPSHOT_NAMES
    assert len(names) == len(set(names)) == 35
    for required in ("image_indices", "probs", "best_argmax", "log_Z", "proj_for_noise", "bucket_size"):
        assert required in names
