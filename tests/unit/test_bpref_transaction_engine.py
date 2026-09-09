"""Opt-in selection and actual engine helper forwarding into the BPref queue."""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject
from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.bpref_transaction import BprefTransactionQueue
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_binary_flag

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_transaction_selector(monkeypatch, token, expected):
    if token is None:
        monkeypatch.delenv(engine.EXACT_LOCAL_BPREF_TRANSACTION_ENV, raising=False)
    else:
        monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_TRANSACTION_ENV, token)
    assert parse_env_binary_flag(engine.EXACT_LOCAL_BPREF_TRANSACTION_ENV) is expected


@pytest.mark.parametrize("token", ["", "yes", "2", "-1"])
def test_invalid_transaction_selector(monkeypatch, token):
    monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_TRANSACTION_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_env_binary_flag(engine.EXACT_LOCAL_BPREF_TRANSACTION_ENV)


def test_unsupported_route_rejected_before_dataset_access(monkeypatch):
    monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_TRANSACTION_ENV, "1")
    with pytest.raises(ValueError, match="require deferred packed final-noise"):
        engine.run_local_em_exact(
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=8,
        )


@pytest.mark.parametrize("queued", [False, True], ids=["immediate", "queued"])
def test_helper_masks_rows_before_projector_dispatch(monkeypatch, queued):
    signature = inspect.signature(cuda_backproject.relion_vdam_mstep_fused_projector_x_half)
    calls = []

    def callback(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        calls.append(values)
        assert values["return_denominator"] is (not queued)
        return (
            values["data_volume"] + jnp.sum(values["posterior_over_weight_norm"]),
            values["weight_volume"] + values["images"].shape[0],
            None,
        )

    monkeypatch.setattr(cuda_backproject, "relion_vdam_mstep_fused_projector_x_half", callback)
    queue = BprefTransactionQueue() if queued else None
    data = jnp.zeros(1, jnp.complex64)
    weight = jnp.zeros(1, jnp.float32)
    shared = dict(
        projector_full=jnp.zeros((9, 9, 9), jnp.complex64),
        projector_r_max=3,
        pixel_indices=jnp.arange(3, dtype=jnp.int32),
        image_shape=(32, 32),
        volume_shape=(11, 11, 11),
        max_r=4.0,
        stable_dense_positions=jnp.arange(3, dtype=jnp.int32),
        logical_current_size=8,
    )
    translations = jnp.zeros((2, 2), jnp.float32)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 2, 3, 3)).copy()
    for first in (0, 2):
        data, weight = engine._accumulate_relion_vdam_physical_particle_grid(
            jnp.ones((2, 3), jnp.complex64),
            jnp.ones((2, 3)),
            jnp.ones((2, 3)),
            jnp.ones((2, 2, 2)),
            translations,
            None,
            rotations,
            jnp.asarray([[True, False], [True, True]]),
            data,
            weight,
            scoring_rotations=rotations,
            reconstruction_group_ids=np.arange(first, first + 2, dtype=np.int32) % 2,
            transaction_queue=queue,
            **shared,
        )
    if queued:
        assert not calls
        data, weight, _ = queue.flush(data, weight)
        assert len(calls) == 1
        np.testing.assert_array_equal(np.asarray(calls[0]["worker_lane_ids"]), [0, 1, 0, 1])
        assert calls[0]["parallel_worker_replay"] is False
    else:
        assert len(calls) == 2
        for call in calls:
            assert call["worker_lane_ids"] is None
            assert call["parallel_worker_replay"] is None
    expected = np.tile(np.asarray([[[1, 1], [0, 0]], [[1, 1], [1, 1]]], np.float32), (2, 1, 1))
    actual = np.concatenate([np.asarray(call["posterior_over_weight_norm"]) for call in calls])
    assert actual.tobytes() == expected.tobytes()
    for call in calls:
        assert call["image_shape"] == shared["image_shape"]
        assert call["volume_shape"] == shared["volume_shape"]
        assert call["max_r"] == shared["max_r"]
        assert call["logical_current_size"] == shared["logical_current_size"]
        assert call["projector_max_r"] == shared["projector_r_max"]
        np.testing.assert_array_equal(call["projector_full"], shared["projector_full"])
        np.testing.assert_array_equal(call["pixel_indices"], shared["pixel_indices"])
        np.testing.assert_array_equal(call["stable_dense_positions"], shared["stable_dense_positions"])
    assert data[0] == 12 and weight[0] == 4
