"""The complete deferred host-plan CUDA transaction preserves seven outputs."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cuda
from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.helpers import deferred_vdam_host_pack as helper

pytestmark = pytest.mark.unit


def operands(batch=2, case="valid"):
    rng = np.random.default_rng(37)
    source, dense, packed, translations, pixels = 3, 5, 3, 4, 7
    posterior = rng.random((source, dense, translations), dtype=np.float32)
    sum_t = posterior.sum(axis=2)
    images = (
        rng.random((source, pixels), dtype=np.float32) + 1j * rng.random((source, pixels), dtype=np.float32)
    ).astype(np.complex64)
    ctf = rng.random((source, pixels), dtype=np.float32)
    inv = rng.random((source, pixels), dtype=np.float32)
    flat = (
        rng.random((source * dense, pixels), dtype=np.float32)
        + 1j * rng.random((source * dense, pixels), dtype=np.float32)
    ).astype(np.complex64)
    take = np.array([[4, 1, 0], [2, 0, 3], [1, 4, 2]], np.int32)[:batch].copy()
    flat_take = np.array([[14, 2, 0], [7, 1, 8], [6, 5, 13]], np.int32)[:batch].copy()
    mask = np.ones((batch, packed), bool)
    mask[0, 1] = False
    if case == "masked":
        mask[:] = False
        take[:] = np.iinfo(np.int32).min
        flat_take[:] = np.iinfo(np.int32).max
    elif case == "negative":
        take -= dense
        flat_take -= source * dense
    elif case == "invalid":
        take[0] = [-dense - 1, dense, np.iinfo(np.int32).max]
        flat_take[0] = [-source * dense - 1, source * dense, np.iinfo(np.int32).min]
        mask[0] = True
    elif case == "bits":
        words = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7FC12345], np.uint32)
        for a in (images, flat, ctf, inv, posterior, sum_t):
            a.view(np.uint32).reshape(-1)[: len(words)] = words
    return posterior, sum_t, images, ctf, inv, flat, take, mask, flat_take


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_shape_contract(batch):
    args = operands(batch)
    outputs = cuda._deferred_vdam_host_pack_shapes(*args)
    assert [x.shape for x in outputs] == [
        (batch, 3, 4),
        (batch, 3),
        (batch, 7),
        (batch, 7),
        (batch, 7),
        (batch, 3, 7),
        (batch, 3, 7),
    ]
    assert [x.dtype for x in outputs] == [
        np.float32,
        np.float32,
        np.complex64,
        np.float32,
        np.float32,
        np.complex64,
        np.float32,
    ]


@pytest.mark.parametrize("case", ["dtype", "rank", "empty", "batch", "sum", "ctf", "pixels", "mask", "flat_take"])
def test_invalid_abi_rejected(case):
    args = list(operands())
    if case == "dtype":
        args[0] = args[0].astype(np.float64)
    elif case == "rank":
        args[0] = args[0].reshape(-1)
    elif case == "empty":
        args[5] = args[5][:0]
    elif case == "batch":
        args[0] = args[0][:1]
    elif case == "sum":
        args[1] = args[1][:1]
    elif case == "ctf":
        args[3] = args[3][:1]
    elif case == "pixels":
        args[5] = args[5][:, :6]
    elif case == "mask":
        args[7] = args[7][:1]
    elif case == "flat_take":
        args[8] = args[8][:1]
    with pytest.raises((TypeError, ValueError)):
        cuda._deferred_vdam_host_pack_shapes(*args)


@pytest.mark.parametrize(
    "token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True), ("", None), ("true", None), ("2", None)]
)
def test_selector(monkeypatch, token, expected):
    monkeypatch.delenv(engine.EXACT_LOCAL_HOST_PLAN_CUDA_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(engine.EXACT_LOCAL_HOST_PLAN_CUDA_ENV, token)
    if expected is None:
        with pytest.raises(ValueError, match="must be 0 or 1"):
            engine._local_host_plan_cuda_requested()
    else:
        assert engine._local_host_plan_cuda_requested() is expected


def test_requires_host_plan_before_dataset_access(monkeypatch):
    monkeypatch.setenv(engine.EXACT_LOCAL_HOST_PLAN_CUDA_ENV, "1")
    monkeypatch.setenv(engine.EXACT_LOCAL_HOST_PLAN_PACK_ENV, "0")
    with pytest.raises(ValueError, match="requires host-plan packing"):
        engine.run_local_em_exact(
            None, None, None, None, None, "linear_interp", image_batch_size=1, rotation_block_size=1, current_size=8
        )


def test_older_library_supported_until_new_transaction_requested(monkeypatch):
    assert all(symbol != "DeferredVdamHostPack" for _, symbol in cuda._FFI_REGISTRATIONS)
    monkeypatch.setattr(cuda, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda, "_get_lib", lambda: SimpleNamespace())
    monkeypatch.setattr(cuda, "_deferred_vdam_host_pack_ffi_registered", False)
    with pytest.raises(RuntimeError, match="explicit build with DeferredVdamHostPack"):
        cuda._ensure_deferred_vdam_host_pack_ffi()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch,case",
    [(1, "valid"), (2, "valid"), (3, "valid"), (2, "negative"), (2, "masked"), (2, "invalid"), (3, "bits")],
)
def test_cuda_preserves_all_seven_outputs_and_input_bytes(batch, case):
    assert jax.default_backend() == "gpu"
    assert cuda.cuda_available()
    arrays = operands(batch, case)
    inputs = tuple(jnp.asarray(x) for x in arrays)
    original = helper.pack_deferred_vdam_host_plan(*inputs)
    actual = helper.pack_deferred_vdam_host_plan_cuda(*inputs)
    jax.block_until_ready((original, actual))
    for a, b in zip(actual, original, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()
    for before, after in zip(arrays, inputs, strict=True):
        assert before.tobytes() == np.asarray(after).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["dtype", "rank", "shape", "output_dtype", "output_shape"])
def test_raw_ffi_rejects_bad_buffers(case):
    assert jax.default_backend() == "gpu"
    cuda._ensure_deferred_vdam_host_pack_ffi()
    arrays = operands()
    outputs = list(cuda._deferred_vdam_host_pack_shapes(*arrays))
    args = [jnp.asarray(x) for x in arrays]
    if case == "dtype":
        args[7] = jnp.ones(args[7].shape, jnp.int32)
    elif case == "rank":
        args[0] = args[0].reshape(-1)
    elif case == "shape":
        args[3] = args[3][:1]
    elif case == "output_dtype":
        outputs[5] = jax.ShapeDtypeStruct(outputs[5].shape, jnp.float32)
    elif case == "output_shape":
        outputs[2] = jax.ShapeDtypeStruct((1, 7), jnp.complex64)
    with pytest.raises((ValueError, RuntimeError), match="DeferredVdamHostPack:"):
        result = jax.ffi.ffi_call(cuda._TARGET_DEFERRED_VDAM_HOST_PACK, tuple(outputs), vmap_method="sequential")(*args)
        jax.block_until_ready(result)
