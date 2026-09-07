"""Queue chronology, ownership and memory limits without CUDA execution."""

import numpy as np
import pytest

from recovar.em.dense_single_volume.bpref_transaction import BprefTransactionQueue

pytestmark = pytest.mark.unit


def test_cuda_packing_queue_dispatch_preserves_bucket_boundaries(monkeypatch):
    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.bpref_transaction import _pad_particle_fields

    packed_calls = []

    def observed(columns, capacity):
        packed_calls.append((columns, capacity))
        # The existing JAX implementation is the CPU semantic reference.
        return _pad_particle_fields(columns, capacity, 5)

    monkeypatch.setattr(cuda_backproject, "pack_bpref_particle_fields", observed)
    common, calls, callback = setup()
    queue = BprefTransactionQueue(stable_particle_capacity=True, cuda_packing=True)
    data = np.zeros((2, 1), np.complex64)
    weight = np.zeros((2, 1), np.float32)
    offset = 0
    for n in (42, 42, 42, 42, 32):
        args, options = operands(n, offset, common, data, weight)
        data, weight, _ = queue.accumulate(callback, *args, **options)
        offset += n
    queue.flush(data, weight)
    assert len(packed_calls) == len(calls) == 1
    columns, capacity = packed_calls[0]
    assert capacity == 256 and len(columns) == 6
    assert all(tuple(a.shape[0] for a in column) == (42, 42, 42, 42, 32) for column in columns)
    assert calls[0]["particle_tail_mask"] is True
    np.testing.assert_array_equal(np.asarray(calls[0]["reconstruction_group_ids"])[200:], -1)
    assert calls[0]["projector_full"] is common["projector"]


def test_deferred_scorer_donation_cannot_replace_pending_accumulators():
    from functools import partial
    import jax
    import jax.numpy as jnp

    options = dict(return_deferred_mstep_inputs=True, disable_adjoint_y=True, disable_adjoint_ctf=True)
    shapes = []

    @partial(jax.jit, donate_argnums=(7, 8), static_argnames=tuple(options))
    def scorer(a, b, c, d, e, f, g, data, weight, **options):
        shapes.append((data.shape, weight.shape))
        return data, weight, jnp.asarray(19, jnp.int32)

    common, calls, callback = setup()
    queue = BprefTransactionQueue()
    data = jnp.zeros(1, jnp.complex64)
    weight = jnp.zeros(1, jnp.float32)
    for first in (0, 2):
        args, kwargs = operands(2, first, common, data, weight)
        data, weight, _ = queue.accumulate(callback, *args, **kwargs)
        result = queue.run_deferred_scorer(scorer, (None,) * 7 + (data, weight), options)
        assert result[0] is data and result[1] is weight and int(result[2]) == 19
        assert not data.is_deleted() and not weight.is_deleted()
    assert shapes == [((0,), (0,))]
    assert not calls
    queue.flush(data, weight)
    assert len(calls) == 1 and calls[0]["images"].shape[0] == 4


@pytest.mark.parametrize("invalid", ["return_deferred_mstep_inputs", "disable_adjoint_y", "disable_adjoint_ctf"])
def test_scorer_with_active_accumulation_rejected(invalid):
    options = dict(return_deferred_mstep_inputs=True, disable_adjoint_y=True, disable_adjoint_ctf=True)
    options[invalid] = False
    with pytest.raises(ValueError, match="both adjoints disabled"):
        BprefTransactionQueue().run_deferred_scorer(None, (), options)


class Shared:
    shape = (9, 9, 9)
    dtype = np.dtype("complex64")

    def __array__(self, *args, **kwargs):
        raise AssertionError("Shared device operand downloaded")


def setup():
    common = {
        "projector": Shared(),
        "translations": np.zeros((2, 2), np.float32),
        "pixels": np.arange(3, dtype=np.int32),
    }
    calls = []

    def callback(**values):
        calls.append(values)
        data = values["data_volume"] + np.sum(np.asarray(values["images"]).real)
        weight = values["weight_volume"] + np.sum(np.asarray(values["posterior_over_weight_norm"]))
        return data, weight, None if values.get("return_denominator", True) is False else np.ones(1, np.float32)

    return common, calls, callback


def operands(n, first, common, data, weight, *, rotations=2):
    ids = np.arange(first, first + n, dtype=np.float32)
    args = (
        data,
        weight,
        (ids[:, None] * 16 + np.arange(3, dtype=np.float32)).astype(np.complex64),
        np.ones((n, 3), np.float32),
        np.ones((n, 3), np.float32),
        np.ones((n, rotations, 2), np.float32),
        common["translations"],
        common["pixels"],
        common["projector"],
        np.broadcast_to(np.eye(3, dtype=np.float32), (n, rotations, 3, 3)).copy(),
        (32, 32),
        (11, 11, 11),
        4.0,
        3,
        1,
    )
    return args, {
        "reconstruction_group_ids": np.arange(first, first + n, dtype=np.int32) % 2,
        "worker_lane_ids": None,
        "particle_trace_ids": None,
        "parallel_worker_replay": None,
    }


@pytest.mark.parametrize("bounds", [(0, 1), (1, 0), (-1, 1), (True, 1), (1, 2.5)])
def test_invalid_bounds_rejected(bounds):
    with pytest.raises(ValueError):
        BprefTransactionQueue(max_images=bounds[0], max_input_bytes=bounds[1])


def test_consecutive_batches_preserve_all_rows_and_local_worker_ids():
    common, calls, callback = setup()
    queue = BprefTransactionQueue()
    data = np.zeros(1, np.complex64)
    weight = np.zeros(1, np.float32)
    original = []
    offset = 0
    for n in (42, 42, 42, 42, 32):
        args, options = operands(n, offset, common, data, weight)
        original.append(args)
        data, weight, _ = queue.accumulate(callback, *args, **options)
        offset += n
    assert not calls
    data, weight, third = queue.flush(data, weight)
    assert third is None and len(calls) == 1
    merged = calls[0]
    for name, index in [
        ("images", 2),
        ("ctf", 3),
        ("minvsigma2", 4),
        ("posterior_over_weight_norm", 5),
        ("rotation_matrices", 9),
    ]:
        expected = np.concatenate([a[index] for a in original], axis=0)
        assert np.asarray(merged[name]).dtype == expected.dtype
        assert np.asarray(merged[name]).tobytes() == expected.tobytes()
    np.testing.assert_array_equal(
        np.asarray(merged["worker_lane_ids"]), np.concatenate([np.arange(n) % 8 for n in (42, 42, 42, 42, 32)])
    )
    np.testing.assert_array_equal(
        np.asarray(merged["particle_trace_ids"]), np.concatenate([np.arange(n) for n in (42, 42, 42, 42, 32)])
    )
    np.testing.assert_array_equal(np.asarray(merged["reconstruction_group_ids"]), np.arange(200) % 2)
    assert merged["parallel_worker_replay"] is False and merged["return_denominator"] is False
    assert merged["projector_full"] is common["projector"]
    assert merged["translation_angles"] is common["translations"]
    assert data[0] == sum(np.sum(a[2]).real for a in original) and weight[0] == 800
    assert queue.flush(data, weight) == (data, weight, None)


@pytest.mark.parametrize("limit", ["images", "bytes"])
def test_limit_flushes_previous_work_and_preserves_carry(limit):
    common, calls, callback = setup()
    queue = BprefTransactionQueue(
        max_images=4 if limit == "images" else 256, max_input_bytes=600 if limit == "bytes" else 128 * 1024**2
    )
    data = np.zeros(1, np.complex64)
    weight = np.zeros(1, np.float32)
    for n, first in ((3, 0), (2, 3)):
        args, options = operands(n, first, common, data, weight)
        data, weight, _ = queue.accumulate(callback, *args, **options)
    assert len(calls) == 1 and calls[0]["images"].shape[0] == 3
    data, weight, _ = queue.flush(data, weight)
    assert [c["images"].shape[0] for c in calls] == [3, 2]
    assert weight[0] == 20 and data[0] == 495


def test_single_oversized_call_executes_unchanged():
    common, calls, callback = setup()
    queue = BprefTransactionQueue(max_images=1)
    args, options = operands(3, 0, common, np.zeros(1, np.complex64), np.zeros(1, np.float32))
    result = queue.accumulate(callback, *args, **options)
    assert len(calls) == 1 and calls[0]["worker_lane_ids"] is None
    assert "return_denominator" not in calls[0] and result[2] is not None


@pytest.mark.parametrize("change", ["projector_identity", "rotation_shape", "replay"])
def test_incompatible_context_flushes_without_shared_array_reads(change):
    common, calls, callback = setup()
    queue = BprefTransactionQueue()
    data = np.zeros(1, np.complex64)
    weight = np.zeros(1, np.float32)
    args, options = operands(1, 0, common, data, weight)
    data, weight, _ = queue.accumulate(callback, *args, **options)
    if change == "projector_identity":
        common = {**common, "projector": Shared()}
    args, options = operands(1, 1, common, data, weight, rotations=3 if change == "rotation_shape" else 2)
    if change == "replay":
        options["worker_lane_ids"] = np.zeros(1, np.int32)
    data, weight, _ = queue.accumulate(callback, *args, **options)
    queue.flush(data, weight)
    assert len(calls) == 2
    if change == "replay":
        assert calls[1]["worker_lane_ids"] is options["worker_lane_ids"]
        assert calls[1]["parallel_worker_replay"] is None


def test_changed_carry_fails_before_callback():
    common, calls, callback = setup()
    queue = BprefTransactionQueue()
    args, options = operands(1, 0, common, np.zeros(1, np.complex64), np.zeros(1, np.float32))
    data, weight, _ = queue.accumulate(callback, *args, **options)
    with pytest.raises(RuntimeError, match="carry changed"):
        queue.flush(data.copy(), weight)
    assert not calls


def test_empty_queue_is_identity():
    data = np.zeros(1, np.complex64)
    weight = np.zeros(1, np.float32)
    result = BprefTransactionQueue().flush(data, weight)
    assert result[0] is data and result[1] is weight and result[2] is None


def test_zero_image_call_is_not_hidden_by_merging():
    common, calls, callback = setup()
    args, options = operands(0, 0, common, np.zeros(1, np.complex64), np.zeros(1, np.float32))
    with pytest.raises(ValueError, match="at least one image"):
        BprefTransactionQueue().accumulate(callback, *args, **options)
    assert not calls


def test_callback_change_flushes_original_callback_first():
    common, first_calls, first_callback = setup()
    _, second_calls, second_callback = setup()
    queue = BprefTransactionQueue()
    data = np.zeros(1, np.complex64)
    weight = np.zeros(1, np.float32)
    for callback, first in ((first_callback, 0), (second_callback, 1)):
        args, options = operands(1, first, common, data, weight)
        data, weight, _ = queue.accumulate(callback, *args, **options)
    assert len(first_calls) == 1 and not second_calls
    queue.flush(data, weight)
    assert len(second_calls) == 1
    assert second_calls[0]["weight_volume"][0] == 4
