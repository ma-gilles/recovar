"""Progress reporting preserves clock, count and log cadence independently of EM."""

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume import local_timing as timing

pytestmark = pytest.mark.unit


def bucket(size):
    return SimpleNamespace(image_indices=np.arange(size))


@pytest.fixture
def clock(monkeypatch, caplog):
    clock = SimpleNamespace(now=100.0, calls=0)

    def read():
        clock.calls += 1
        return clock.now

    monkeypatch.setattr(timing, "time", SimpleNamespace(time=read))
    monkeypatch.delenv(timing.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, raising=False)
    monkeypatch.delenv(timing.EXACT_LOCAL_PROGRESS_SECONDS_ENV, raising=False)
    caplog.set_level("INFO", logger=timing.logger.name)
    return clock


def test_default_start_and_forced_completion(clock, caplog):
    progress = timing.LocalBucketProgress([bucket(2), bucket(3)], total_local_rotations=7, n_trans=4)
    assert caplog.records[-1].args == (2, 5, 7, 4, 1000, 300)
    assert clock.calls == 1
    progress.mark_bucket_done(bucket(2))
    progress.mark_bucket_done(bucket(3))
    assert len(caplog.records) == 1
    clock.now = 110
    progress.log(force=True, done=True)
    assert caplog.records[-1].args == ("done", 2, 2, 5, 5, 10.0, 0.5)
    assert clock.calls == 4


def test_chunk_cadence_and_forced_final_message(monkeypatch, clock, caplog):
    monkeypatch.setenv(timing.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, "2")
    monkeypatch.setenv(timing.EXACT_LOCAL_PROGRESS_SECONDS_ENV, "0")
    progress = timing.LocalBucketProgress([bucket(3)] * 3, total_local_rotations=9, n_trans=2)
    clock.now = 105
    progress.mark_bucket_done(bucket(3))
    assert len(caplog.records) == 1
    clock.now = 110
    progress.mark_bucket_done(bucket(3))
    assert caplog.records[-1].args == ("progress", 2, 3, 6, 9, 10.0, 0.6)
    clock.now = 120
    progress.mark_bucket_done(bucket(3))
    assert len(caplog.records) == 2
    progress.log(force=True, done=True)
    assert caplog.records[-1].args == ("done", 3, 3, 9, 9, 20.0, 0.45)


def test_time_cadence_uses_last_emitted_message(monkeypatch, clock, caplog):
    monkeypatch.setenv(timing.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, "0")
    monkeypatch.setenv(timing.EXACT_LOCAL_PROGRESS_SECONDS_ENV, "10")
    progress = timing.LocalBucketProgress([bucket(2)] * 2, total_local_rotations=9, n_trans=2)
    clock.now = 109
    progress.mark_bucket_done(bucket(2))
    assert len(caplog.records) == 1
    clock.now = 110
    progress.log()
    assert caplog.records[-1].args == ("progress", 1, 2, 2, 4, 10.0, 0.2)
    clock.now = 119
    progress.mark_bucket_done(bucket(2))
    assert len(caplog.records) == 2
    clock.now = 120
    progress.log()
    assert caplog.records[-1].args == ("progress", 2, 2, 4, 4, 20.0, 0.2)
    assert progress.last_log_at == 120


def test_empty_work_logs_nothing_and_does_not_poll_on_completion(clock, caplog):
    progress = timing.LocalBucketProgress([], total_local_rotations=0, n_trans=4)
    progress.log(force=True, done=True)
    assert not caplog.records
    assert clock.calls == 1
    assert progress.completed_chunks == progress.completed_images == 0


def test_padded_bucket_count_and_clock_rollback_are_preserved(monkeypatch, clock, caplog):
    monkeypatch.setenv(timing.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, "1")
    progress = timing.LocalBucketProgress([bucket(2)], total_local_rotations=2, n_trans=4)
    clock.now = 90
    progress.mark_bucket_done(bucket(4))
    assert caplog.records[-1].args == ("progress", 1, 1, 4, 2, 0.0, 0.0)
    assert progress.completed_images == 4


@pytest.mark.parametrize("name", [timing.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, timing.EXACT_LOCAL_PROGRESS_SECONDS_ENV])
@pytest.mark.parametrize("value", ["-1", "invalid"])
def test_invalid_policy_precedes_clock_and_logging(monkeypatch, clock, caplog, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="non-negative integer"):
        timing.LocalBucketProgress([bucket(2)], total_local_rotations=2, n_trans=4)
    assert clock.calls == 0
    assert not caplog.records


def test_both_cadences_disabled_still_allow_forced_completion(monkeypatch, clock, caplog):
    monkeypatch.setenv(timing.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, "0")
    monkeypatch.setenv(timing.EXACT_LOCAL_PROGRESS_SECONDS_ENV, "0")
    progress = timing.LocalBucketProgress([bucket(2)], total_local_rotations=2, n_trans=4)
    clock.now = 1000
    progress.mark_bucket_done(bucket(2))
    assert len(caplog.records) == 1
    progress.log(force=True, done=True)
    assert caplog.records[-1].args == ("done", 1, 1, 2, 2, 900.0, 2 / 900)
