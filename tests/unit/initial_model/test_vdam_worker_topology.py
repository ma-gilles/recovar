from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_engine


pytestmark = pytest.mark.unit


def _clear_worker_replay(monkeypatch):
    monkeypatch.delenv(local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV, raising=False)
    monkeypatch.delenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_ITER_ENV,
        raising=False,
    )


def test_default_vdam_worker_topology_keeps_single_controller(monkeypatch):
    _clear_worker_replay(monkeypatch)

    owners = local_em_engine._relion_vdam_worker_lanes_for_images(
        object(),
        np.arange(10, dtype=np.int64),
    )

    assert owners is None


def test_round_robin_vdam_worker_topology_selects_eight_host_workers(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )

    owners = local_em_engine._relion_vdam_worker_lanes_for_images(
        object(),
        np.arange(10, dtype=np.int64),
    )

    np.testing.assert_array_equal(
        owners,
        np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=np.int32),
    )


def test_round_robin_vdam_worker_topology_preserves_bucket_shape(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )

    owners = local_em_engine._relion_vdam_worker_lanes_for_images(
        object(),
        np.arange(12, dtype=np.int64).reshape(3, 4),
    )

    assert owners.shape == (3, 4)
    np.testing.assert_array_equal(owners.ravel(), np.arange(12, dtype=np.int32) % 8)


def test_round_robin_vdam_worker_topology_can_target_one_iteration(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )
    monkeypatch.setenv(local_em_engine.RELION_VDAM_WORKER_REPLAY_ITER_ENV, "58")

    assert (
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
            debug_iteration=57,
        )
        is None
    )
    np.testing.assert_array_equal(
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
            debug_iteration=58,
        ),
        np.arange(8, dtype=np.int32),
    )


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_round_robin_vdam_worker_topology_rejects_invalid_iteration(
    monkeypatch,
    value,
):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )
    monkeypatch.setenv(local_em_engine.RELION_VDAM_WORKER_REPLAY_ITER_ENV, value)

    with pytest.raises(ValueError, match="must be a positive integer"):
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
            debug_iteration=58,
        )


def test_round_robin_vdam_worker_topology_rejects_captured_schedule(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV,
        "/tmp/must-not-be-read.npz",
    )

    with pytest.raises(ValueError, match="cannot also use a captured schedule"):
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
        )


def test_captured_vdam_worker_topology_can_skip_non_target_iteration(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    )
    monkeypatch.setenv(local_em_engine.RELION_VDAM_WORKER_REPLAY_ITER_ENV, "58")
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV,
        "/tmp/must-not-be-read-before-target.npz",
    )

    assert (
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
            debug_iteration=57,
        )
        is None
    )
