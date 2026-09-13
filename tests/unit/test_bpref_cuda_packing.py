"""Byte-preserving bounded BPref packing, including per-bucket worker restarts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject
from recovar.em.helpers.bpref_transaction import BprefTransactionQueue
from recovar.em.helpers.env_flags import parse_env_binary_flag
from recovar.em.local import local_em_engine

pytestmark = pytest.mark.unit


def test_old_library_needs_new_symbol_only_for_requested_packing(monkeypatch):
    from types import SimpleNamespace

    assert all(symbol != "BprefParticlePack" for _, symbol in cuda_backproject._FFI_REGISTRATIONS)
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject, "_get_lib", lambda: SimpleNamespace())
    monkeypatch.setattr(cuda_backproject, "_bpref_particle_pack_ffi_registered", False)
    with pytest.raises(RuntimeError, match="explicit build with BprefParticlePack"):
        cuda_backproject._ensure_bpref_particle_pack_ffi()


def columns_for(counts, pixels=7, rotations=3, translations=5):
    rng = np.random.default_rng(491)
    columns = [[] for _ in range(6)]
    for count in counts:
        shapes = (
            (count, pixels),
            (count, pixels),
            (count, pixels),
            (count, rotations, translations),
            (count, rotations, 3, 3),
            (count,),
        )
        for field, shape in enumerate(shapes):
            if field == 5:
                value = rng.integers(0, 2, shape, dtype=np.int32)
            else:
                words = int(np.prod(shape)) * (2 if field == 0 else 1)
                bits = rng.integers(0, 2**32, words, dtype=np.uint32)
                # Include signed zero, infinities and distinct quiet NaN payloads.
                special = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7FC12345], np.uint32)
                bits[: min(words, len(special))] = special[: min(words, len(special))]
                value = bits.view(np.complex64 if field == 0 else np.float32).reshape(shape)
            columns[field].append(value)
    return tuple(tuple(column) for column in columns)


def numpy_expected(columns, capacity):
    result = []
    for field, column in enumerate(columns):
        output = np.full((capacity, *column[0].shape[1:]), -1 if field == 5 else 0, dtype=column[0].dtype)
        active = np.concatenate(column)
        output[: len(active)] = active
        result.append(output)
    local_ids = np.concatenate([np.arange(len(v), dtype=np.int32) for v in columns[0]])
    for active in (local_ids % 8, local_ids):
        output = np.zeros(capacity, np.int32)
        output[: len(active)] = active
        result.append(output)
    return tuple(result)


@pytest.mark.parametrize(
    "counts,capacity", [((1,), 1), ((1,), 256), ((42, 42, 42, 42, 32), 256), ((63, 11), 128), ((1,) * 256, 256)]
)
def test_pack_shape_contract(counts, capacity):
    columns = columns_for(counts)
    shapes = cuda_backproject._bpref_particle_pack_shapes(columns, capacity)
    expected = numpy_expected(columns, capacity)
    assert [(s.shape, s.dtype) for s in shapes] == [(a.shape, a.dtype) for a in expected]


@pytest.mark.parametrize("capacity", [0, -1, 257, True, 1.0, "2", None])
def test_bad_capacity_rejected(capacity):
    with pytest.raises(ValueError, match="capacity"):
        cuda_backproject._bpref_particle_pack_shapes(columns_for((1,)), capacity)


@pytest.mark.parametrize(
    "case", ["truncation", "empty", "missing", "buckets", "dtype", "axes", "pixels", "rotation", "zero"]
)
def test_bad_input_contract_rejected(case):
    columns = [list(x) for x in columns_for((2, 3))]
    capacity = 8
    if case == "truncation":
        capacity = 4
    elif case == "empty":
        columns = [[] for _ in range(6)]
    elif case == "missing":
        columns.pop()
    elif case == "buckets":
        columns[2].pop()
    elif case == "dtype":
        columns[1][0] = columns[1][0].astype(np.float64)
    elif case == "axes":
        columns[5][0] = np.zeros(1, np.int32)
    elif case == "pixels":
        columns[1] = [np.zeros((n, 8), np.float32) for n in (2, 3)]
    elif case == "rotation":
        columns[4] = [np.zeros((n, 4, 3, 3), np.float32) for n in (2, 3)]
    elif case == "zero":
        columns[0][0] = np.zeros((0, 7), np.complex64)
    with pytest.raises((TypeError, ValueError)):
        cuda_backproject._bpref_particle_pack_shapes(tuple(map(tuple, columns)), capacity)


@pytest.mark.parametrize(
    "options",
    [
        {"cuda_packing": 1},
        {"cuda_packing": True},
        {"cuda_packing": True, "stable_particle_capacity": True, "max_images": 257},
    ],
)
def test_queue_rejects_unsupported_cuda_configuration(options):
    with pytest.raises((TypeError, ValueError), match="CUDA|cuda_packing"):
        BprefTransactionQueue(**options)


@pytest.mark.parametrize(
    "value,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True), ("", None), ("true", None), ("2", None)]
)
def test_cuda_packing_selector(monkeypatch, value, expected):
    name = "RECOVAR_EXACT_LOCAL_BPREF_CUDA_PACKING"
    monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(name, value)
    if expected is None:
        with pytest.raises(ValueError, match="must be 0 or 1"):
            parse_env_binary_flag(local_em_engine.EXACT_LOCAL_BPREF_CUDA_PACKING_ENV)
    else:
        assert parse_env_binary_flag(local_em_engine.EXACT_LOCAL_BPREF_CUDA_PACKING_ENV) is expected


@pytest.mark.gpu
@pytest.mark.parametrize(
    "counts,capacity",
    [((1,), 1), ((1,), 256), ((42, 42, 42, 42, 32), 256), ((63, 11), 128), ((1,) * 256, 256), ((32,), 32)],
)
def test_cuda_pack_preserves_every_bit_and_bucket_worker_ids(counts, capacity):
    from recovar.em.helpers.bpref_transaction import _pad_particle_fields

    assert jax.default_backend() == "gpu"
    assert cuda_backproject.cuda_available()
    columns = columns_for(counts)
    originals = tuple(tuple(x.tobytes() for x in column) for column in columns)
    device = jax.tree.map(jnp.asarray, columns)
    result = cuda_backproject.pack_bpref_particle_fields(device, capacity)
    control = _pad_particle_fields(device, capacity, 5)
    expected = numpy_expected(columns, capacity)
    for actual, wanted in zip(result, expected, strict=True):
        actual.block_until_ready()
        actual = np.asarray(actual)
        assert actual.shape == wanted.shape and actual.dtype == wanted.dtype
        assert actual.tobytes() == wanted.tobytes()
    for column, before in zip(device, originals, strict=True):
        assert tuple(np.asarray(x).tobytes() for x in column) == before
    for old, new in zip(control, result, strict=True):
        assert np.asarray(old).tobytes() == np.asarray(new).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case", ["input_count", "output_count", "dtype", "capacity", "output_axis", "truncation", "input_axis", "row_shape"]
)
def test_raw_cuda_pack_rejects_malformed_buffers(case):
    assert jax.default_backend() == "gpu"
    cuda_backproject._ensure_bpref_particle_pack_ffi()
    columns = columns_for((2, 3))
    outputs = list(cuda_backproject._bpref_particle_pack_shapes(columns, 8))
    args = [jnp.asarray(x) for column in columns for x in column]
    if case == "input_count":
        args.pop()
    elif case == "output_count":
        outputs.pop()
    elif case == "dtype":
        args[2] = jnp.zeros(args[2].shape, jnp.int32)
    elif case == "capacity":
        outputs = [jax.ShapeDtypeStruct((257, *x.shape[1:]), x.dtype) for x in outputs]
    elif case == "output_axis":
        outputs[6] = jax.ShapeDtypeStruct((7,), jnp.int32)
    elif case == "truncation":
        outputs = [jax.ShapeDtypeStruct((4, *x.shape[1:]), x.dtype) for x in outputs]
    elif case == "input_axis":
        args[-1] = jnp.zeros((2,), jnp.int32)
    elif case == "row_shape":
        args[0] = jnp.zeros((2, 8), jnp.complex64)
    with pytest.raises((ValueError, RuntimeError), match="BprefParticlePack:"):
        result = jax.ffi.ffi_call(
            cuda_backproject._TARGET_BPREF_PARTICLE_PACK,
            tuple(outputs),
            vmap_method="sequential",
        )(*args)
        jax.block_until_ready(result)
