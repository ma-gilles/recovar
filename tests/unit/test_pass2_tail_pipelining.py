"""Bucket-tail pipelining: ordered worker, error propagation, opt-in knob."""

import gc
import inspect
import re
import threading
import time

import numpy as np
import pytest

from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed


def test_pipeline_tail_knob_defaults_on_and_is_disableable(monkeypatch):
    """Default on after the matched hp3 pair (job 14100290); 0 runs the tail inline."""
    monkeypatch.delenv(bucketed._PIPELINE_TAIL_ENV, raising=False)
    assert bucketed._pipeline_tail_enabled() is True
    monkeypatch.setenv(bucketed._PIPELINE_TAIL_ENV, "0")
    assert bucketed._pipeline_tail_enabled() is False
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
    tail_reads = set(re.findall(r'snap\.get\("(\w+)"\)', inspect.getsource(bucketed.compute_pass2_stats_sparse_bucketed)))
    assert len(names) == len(set(names)) == 32
    assert set(names) == tail_reads
    for required in (
        "image_indices",
        "probs",
        "best_argmax",
        "log_Z",
        "proj_for_noise",
        "processed_score_half_for_noise",
    ):
        assert required in names


def test_bucket_loop_leaves_no_namespace_cycle(monkeypatch):
    """After a multi-bucket pass-2 call, its local namespace is not cyclic garbage.

    A ``locals()`` dict that stored itself kept each call's namespace
    (projection cache, bucket arrays) until a full cyclic GC. The K1 100k/256
    completion run (job 14282511) ran out of memory with ~41 GiB of earlier
    calls' arrays still allocated (census 14284210, referrer probe 14284301).
    """

    from test_sparse_pass2_bucketed_parity import TestSparsePass2Bucketed

    from recovar.em.sparse_pass2.dispatch import compute_pass2_stats_sparse

    submitted = []
    submit = bucketed._BucketTailRunner.submit

    def counting_submit(self, snapshot):
        submitted.append(len(snapshot))
        return submit(self, snapshot)

    monkeypatch.setattr(bucketed._BucketTailRunner, "submit", counting_submit)
    # Keep one bucket per rotation-count size, as for a production-size dataset.
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES", "0")
    sig_indices = [
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
        np.array([5], dtype=np.int32),
        np.arange(20, dtype=np.int32),
        np.array([0, 4, 8, 12], dtype=np.int32),
    ]
    ds, volume, mean_variance, noise_variance, translations, nside = TestSparsePass2Bucketed()._common_args(sig_indices)

    gc.collect()
    debug_flags = gc.get_debug()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        compute_pass2_stats_sparse(
            ds,
            volume,
            mean_variance,
            noise_variance,
            translations,
            sig_indices,
            nside_level=nside,
            disc_type="linear_interp",
            return_stats=True,
        )
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()
        # Sizes of unreachable dicts holding the bucket loop's names.
        leaked = [len(obj) for obj in gc.garbage if isinstance(obj, dict) and "_bucket_tail" in obj]
    finally:
        gc.set_debug(debug_flags)
        gc.garbage.clear()
        if was_enabled:
            gc.enable()

    assert len(submitted) >= 2, "the call must run several buckets for the cycle to form"
    assert leaked == [], "pass-2 namespace left in a reference cycle"
