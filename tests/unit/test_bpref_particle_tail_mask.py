"""Physical BPref capacity must exclude padded rows without changing active work."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from test_bpref_optional_denominator import arguments, device

pytestmark = pytest.mark.unit

PARTICLE_FIELDS = (
    "images",
    "ctf",
    "minvsigma2",
    "posterior_over_weight_norm",
    "rotation_matrices",
    "reconstruction_group_ids",
)


def padded_arguments(stable, active=2, capacity=4):
    values = arguments(True, stable)
    prefix = dict(values)
    for key in PARTICLE_FIELDS:
        prefix[key] = values[key][:active].copy()
    padded = dict(prefix)
    for key in PARTICLE_FIELDS:
        value = prefix[key]
        tail = np.full((capacity - active, *value.shape[1:]), 7, value.dtype)
        if key == "reconstruction_group_ids":
            tail.fill(-1)
        padded[key] = np.concatenate((value, tail))
    # Invalid worker IDs in the masked suffix also prove that only the active
    # prefix reaches the original metadata validation and scheduling loop.
    prefix["worker_lane_ids"] = np.arange(active, dtype=np.int32)
    padded["worker_lane_ids"] = np.concatenate((prefix["worker_lane_ids"], np.full(capacity - active, 99, np.int32)))
    for value in (prefix, padded):
        value.update(return_denominator=False, parallel_worker_replay=False)
    padded["particle_tail_mask"] = True
    return prefix, padded


@pytest.mark.parametrize("value", [None, 0, 1, "false", np.bool_(False)])
def test_mask_requires_python_bool_before_cuda(monkeypatch, value):
    monkeypatch.setattr(cb, "_ensure_ffi", lambda: pytest.fail("loaded CUDA before validation"))
    with pytest.raises(TypeError, match="particle_tail_mask must be a Python bool"):
        cb.relion_vdam_mstep_fused_projector_x_half.__wrapped__(**arguments(True, True), particle_tail_mask=value)


@pytest.mark.parametrize(
    "override",
    [
        {"return_denominator": True},
        {"reconstruction_group_ids": None},
        {"parallel_worker_replay": None},
        {"parallel_worker_replay": True},
        {"runtime_projector_radius": np.asarray(3, np.int32)},
        {"rotation_replay_order": np.zeros((4, 1), np.int32)},
        {"rotation_replay_counts": np.ones(4, np.int32)},
        {"particle_start_offsets_ns": np.zeros(4, np.int32)},
        {"serial_rotation_replay": True},
        {"persistent_serial_rotation_replay": True},
        {"float64_accumulator_replay": True},
        {"reverse_rotation_replay": True},
        {"rotation_replay_stride": 1},
        {"native_trace_shape_replay": True},
        {"candidate_trace_active": True},
    ],
)
def test_mask_rejects_unsupported_modes_before_cuda(monkeypatch, override):
    monkeypatch.setattr(cb, "_ensure_ffi", lambda: pytest.fail("loaded CUDA before validation"))
    _, values = padded_arguments(True)
    values.update(override)
    with pytest.raises(ValueError, match="particle_tail_mask requires"):
        cb.relion_vdam_mstep_fused_projector_x_half.__wrapped__(**values)


def test_mask_rejects_single_group_before_cuda(monkeypatch):
    monkeypatch.setattr(cb, "_ensure_ffi", lambda: pytest.fail("loaded CUDA before validation"))
    _, values = padded_arguments(True)
    for name in ("data_volume", "weight_volume"):
        values[name] = values[name][:1]
    with pytest.raises(ValueError, match="at least two accumulator groups"):
        cb.relion_vdam_mstep_fused_projector_x_half.__wrapped__(**values)


@pytest.mark.parametrize("stable", [False, True])
def test_mask_is_default_off_and_changes_only_ffi_attribute(monkeypatch, stable):
    monkeypatch.setattr(cb, "_ensure_ffi", lambda: None)
    records = []

    def ffi(target, outputs, **options):
        def call(*operands, **attrs):
            records.append((target, outputs, options, operands, attrs))
            return (*operands[14:17], jnp.empty((0,), jnp.float32))

        return call

    monkeypatch.setattr(jax.ffi, "ffi_call", ffi)
    fn = cb.relion_vdam_mstep_fused_projector_x_half.__wrapped__
    assert inspect.signature(fn).parameters["particle_tail_mask"].default is False
    values, _ = padded_arguments(stable)
    for mask in (False, True):
        fn(**device(values), particle_tail_mask=mask)
    left, right = records
    assert left[:3] == right[:3]
    assert left[4]["particle_tail_mask"] == 0
    assert right[4]["particle_tail_mask"] == 1
    assert {k: v for k, v in left[4].items() if k != "particle_tail_mask"} == {
        k: v for k, v in right[4].items() if k != "particle_tail_mask"
    }
    for a, b in zip(left[3], right[3], strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize("active,capacity", [(1, 4), (2, 4), (2, 2)])
def test_gpu_nonzero_tail_is_excluded_bitwise(stable, active, capacity):
    assert jax.default_backend() == "gpu"
    fn = cb.relion_vdam_mstep_fused_projector_x_half
    prefix, padded = padded_arguments(stable, active, capacity)
    original = jax.block_until_ready(fn(**device(prefix)))
    actual = jax.block_until_ready(fn(**device(padded)))
    assert actual[2] is None
    # Ensure this fixture actually scatters, rather than accepting a no-op.
    assert np.asarray(original[0]).tobytes() != prefix["data_volume"].tobytes()
    for a, b in zip(original[:2], actual[:2], strict=True):
        assert np.isfinite(np.asarray(b)).all()
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "groups,mask",
    [
        ([0, -1, 1, -1], True),
        ([-1, -1, -1, -1], True),
        ([0, 1, -2, -1], True),
        ([0, 2, -1, -1], True),
        ([0, 1, -1, -1], False),
    ],
)
def test_gpu_invalid_prefix_and_legacy_negative_groups_fail(groups, mask):
    assert jax.default_backend() == "gpu"
    _, values = padded_arguments(True)
    values.update(reconstruction_group_ids=np.asarray(groups, np.int32), particle_tail_mask=mask)
    with pytest.raises(RuntimeError, match="invalid argument"):
        jax.block_until_ready(cb.relion_vdam_mstep_fused_projector_x_half(**device(values)))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "override",
    [
        {"particle_tail_mask": 2},
        {"parallel_worker_replay": 1},
        {"candidate_trace_active": 1},
        {"serial_rotation_replay": 1},
    ],
)
def test_gpu_raw_ffi_rejects_invalid_mask_attributes(monkeypatch, override):
    assert jax.default_backend() == "gpu"
    original = jax.ffi.ffi_call

    def ffi(target, outputs, **options):
        call = original(target, outputs, **options)

        def invoke(*operands, **attrs):
            attrs.update({k: np.int64(v) for k, v in override.items()})
            return call(*operands, **attrs)

        return invoke

    monkeypatch.setattr(jax.ffi, "ffi_call", ffi)
    fn = cb.relion_vdam_mstep_fused_projector_x_half
    fn.clear_cache()
    try:
        _, values = padded_arguments(True)
        with pytest.raises(RuntimeError, match="particle tail mask"):
            jax.block_until_ready(fn(**device(values)))
    finally:
        fn.clear_cache()
