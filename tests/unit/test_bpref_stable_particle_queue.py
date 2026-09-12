"""Stable physical particle shapes must preserve prefix bytes and byte bounds."""

import numpy as np
import pytest
from test_bpref_transaction import operands, setup

from recovar.em.helpers.bpref_transaction import _PARTICLE, BprefTransactionQueue
from recovar.em.helpers.env_flags import parse_env_binary_flag
from recovar.em.local import local_em_engine as engine

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [None, 0, 1, "1"])
def test_capacity_selector_requires_bool(value):
    with pytest.raises(TypeError, match="Python bool"):
        BprefTransactionQueue(stable_particle_capacity=value)


@pytest.mark.parametrize("sizes", [(2,), (2, 3), (3, 2, 1)])
def test_capacity_packs_single_and_multiple_batches_without_rebasing(sizes):
    common, calls, callback = setup()
    queue = BprefTransactionQueue(max_images=8, stable_particle_capacity=True)
    data = np.zeros((2, 1), np.complex64)
    weight = np.zeros((2, 1), np.float32)
    expected = []
    offset = 0
    for n in sizes:
        args, options = operands(n, offset, common, data, weight)
        expected.append((args, options))
        data, weight, _ = queue.accumulate(callback, *args, **options)
        offset += n
    data, weight, _ = queue.flush(data, weight)
    assert len(calls) == 1
    actual = calls[0]
    assert actual["particle_tail_mask"] is True and actual["parallel_worker_replay"] is False
    assert actual["return_denominator"] is False
    for name, index in [
        ("images", 2),
        ("ctf", 3),
        ("minvsigma2", 4),
        ("posterior_over_weight_norm", 5),
        ("rotation_matrices", 9),
    ]:
        prefix = np.concatenate([args[index] for args, _ in expected])
        values = np.asarray(actual[name])
        assert values.shape[0] == 8 and values.dtype == prefix.dtype
        assert values[:offset].tobytes() == prefix.tobytes()
        assert not np.any(values[offset:])
    for name, prefix, fill in (
        ("reconstruction_group_ids", np.arange(offset) % 2, -1),
        ("worker_lane_ids", np.concatenate([np.arange(n) % 8 for n in sizes]), 0),
        ("particle_trace_ids", np.concatenate([np.arange(n) for n in sizes]), 0),
    ):
        np.testing.assert_array_equal(np.asarray(actual[name]), np.concatenate([prefix, np.full(8 - offset, fill)]))
    assert actual["projector_full"] is common["projector"]
    np.testing.assert_array_equal(data, sum(np.sum(args[2]).real for args, _ in expected))
    np.testing.assert_array_equal(weight, 4 * offset)


def test_byte_bound_includes_generated_metadata_and_limits_capacity():
    common, calls, callback = setup()
    # Fixture row: image24 + ctf12 + noise12 + posterior16 + rotations72 +
    # groups4 + generated worker/trace8 = 148 bytes. 739 allows exactly four.
    queue = BprefTransactionQueue(max_images=256, max_input_bytes=739, stable_particle_capacity=True)
    data = np.zeros((2, 1), np.complex64)
    weight = np.zeros((2, 1), np.float32)
    for n, first in [(3, 0), (2, 3)]:
        args, options = operands(n, first, common, data, weight)
        data, weight, _ = queue.accumulate(callback, *args, **options)
    data, weight, _ = queue.flush(data, weight)
    assert len(calls) == 2
    for values, active in zip(calls, (3, 2), strict=True):
        assert values["images"].shape[0] == 4
        assert sum(values[n].size * values[n].dtype.itemsize for n in _PARTICLE) == 592
        assert np.sum(np.asarray(values["reconstruction_group_ids"]) >= 0) == active
    np.testing.assert_array_equal(data, 495)
    np.testing.assert_array_equal(weight, 20)


@pytest.mark.parametrize("limit", [1, 147])
def test_oversized_call_keeps_original_unpadded_route(limit):
    common, calls, callback = setup()
    queue = BprefTransactionQueue(max_input_bytes=limit, stable_particle_capacity=True)
    args, options = operands(1, 0, common, np.zeros((2, 1), np.complex64), np.zeros((2, 1), np.float32))
    result = queue.accumulate(callback, *args, **options)
    assert len(calls) == 1 and "particle_tail_mask" not in calls[0]
    assert calls[0]["images"] is args[2] and result[2] is not None


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_engine_capacity_selector(monkeypatch, token, expected):
    monkeypatch.delenv(engine.EXACT_LOCAL_BPREF_PARTICLE_CAPACITY_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_PARTICLE_CAPACITY_ENV, token)
    assert parse_env_binary_flag(engine.EXACT_LOCAL_BPREF_PARTICLE_CAPACITY_ENV) is expected


@pytest.mark.parametrize("token", ["", "yes", "2", "-1"])
def test_invalid_engine_capacity_selector(monkeypatch, token):
    monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_PARTICLE_CAPACITY_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_env_binary_flag(engine.EXACT_LOCAL_BPREF_PARTICLE_CAPACITY_ENV)


def test_capacity_requires_queue_before_dataset_access(monkeypatch):
    monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_PARTICLE_CAPACITY_ENV, "1")
    monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_TRANSACTION_ENV, "0")
    with pytest.raises(ValueError, match="requires transactions"):
        engine.run_local_em_exact(
            None, None, None, None, None, "linear_interp", image_batch_size=1, rotation_block_size=1, current_size=8
        )
