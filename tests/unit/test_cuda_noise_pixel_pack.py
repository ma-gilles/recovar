"""Noise padding remains an exact byte copy with a device-resident tail index."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cuda
from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.deferred_noise_pack import _pad_noise_pixels, pack_noise_pixel_capacity

pytestmark = pytest.mark.unit


def operands(batch=11, wide=False):
    rng = np.random.default_rng(45)
    real, complex_type, integer = (
        (np.float64, np.complex128, np.int64) if wide else (np.float32, np.complex64, np.int32)
    )
    probs = rng.random((batch, 3, 7)).astype(real)
    projection = (rng.random((batch, 3, 5)) + 1j * rng.random((batch, 3, 5))).astype(complex_type)
    ctf = rng.random((batch, 3, 5)).astype(real)
    indices = np.arange(batch, dtype=integer)
    spare = np.array(2**34 + 7 if wide else 3000, integer)
    words = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7FC12345], np.uint32)
    for value in (probs, projection, ctf):
        value.view(np.uint32).reshape(-1)[: len(words)] = words
    return probs, projection, ctf, indices, spare


@pytest.mark.parametrize("wide", [False, True])
def test_shape_contract(wide):
    args = operands(wide=wide)
    out = cuda._noise_pixel_pack_shapes(*args, target_batch=42)
    assert [x.shape for x in out] == [(42, 3, 7), (42, 3, 5), (42, 3, 5), (42,)]
    assert [x.dtype for x in out] == [x.dtype for x in args[:4]]


@pytest.mark.parametrize(
    "case",
    [
        "rank",
        "dtype",
        "empty",
        "rotation",
        "pixels",
        "indices",
        "spare",
        "spare_dtype",
        "truncate",
        "bool_target",
        "float_target",
    ],
)
def test_bad_abi(case):
    args = list(operands())
    target = 42
    if case == "rank":
        args[0] = args[0].reshape(-1)
    elif case == "dtype":
        args[1] = args[1].real
    elif case == "empty":
        args[0] = args[0][:0]
    elif case == "rotation":
        args[1] = args[1][:, :2]
    elif case == "pixels":
        args[2] = args[2][:, :, :4]
    elif case == "indices":
        args[3] = args[3][:1]
    elif case == "spare":
        args[4] = args[4].reshape(1)
    elif case == "spare_dtype":
        args[4] = args[4].astype(np.int64)
    elif case == "truncate":
        target = 1
    elif case == "bool_target":
        target = True
    elif case == "float_target":
        target = 42.0
    with pytest.raises((TypeError, ValueError)):
        cuda._noise_pixel_pack_shapes(*args, target_batch=target)


@pytest.mark.parametrize("no_spare", [False, True])
def test_noop_returns_original_objects_before_cuda(no_spare, monkeypatch):
    args = operands()

    def forbidden(*args, **kwargs):
        raise AssertionError("No-op entered CUDA")

    monkeypatch.setattr(cuda, "pad_noise_pixels_cuda", forbidden)
    result = pack_noise_pixel_capacity(
        *args[:4],
        target_batch=42 if no_spare else 11,
        n_images=3000,
        norm_capacity=3000 if no_spare else 3072,
        cuda_packing=True,
    )
    assert all(a is b for a, b in zip(result, args[:4], strict=True))


@pytest.mark.parametrize(
    "token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True), ("", None), ("true", None), ("2", None)]
)
def test_selector(monkeypatch, token, expected):
    name = "RECOVAR_EXACT_LOCAL_NOISE_PIXEL_CUDA"
    monkeypatch.delenv(name, raising=False)
    if token is not None:
        monkeypatch.setenv(name, token)
    if expected is None:
        with pytest.raises(ValueError, match="must be 0 or 1"):
            engine._local_noise_pixel_cuda_requested()
    else:
        assert engine._local_noise_pixel_cuda_requested() is expected


def test_requires_pixel_capacity_before_dataset_access(monkeypatch):
    monkeypatch.setenv("RECOVAR_EXACT_LOCAL_NOISE_PIXEL_CUDA", "1")
    monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_PIXEL_CAPACITY_ENV, "0")
    with pytest.raises(ValueError, match="CUDA packing requires noise pixel capacity"):
        engine.run_local_em_exact(
            None, None, None, None, None, "linear_interp", image_batch_size=1, rotation_block_size=1, current_size=8
        )


def test_optional_symbol(monkeypatch):
    assert all(symbol != "NoisePixelPack" for _, symbol in cuda._FFI_REGISTRATIONS)
    monkeypatch.setattr(cuda, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda, "_get_lib", lambda: SimpleNamespace())
    monkeypatch.setattr(cuda, "_noise_pixel_pack_ffi_registered", False)
    with pytest.raises(RuntimeError, match="explicit build with NoisePixelPack"):
        cuda._ensure_noise_pixel_pack_ffi()


def test_public_helper_forwards_original_buffers_and_dynamic_spare(monkeypatch):
    args = operands()
    sentinel = object()

    def record(*values, target_batch):
        assert target_batch == 42
        assert all(a is b for a, b in zip(values[:4], args[:4], strict=True))
        assert values[4].shape == () and values[4].dtype == np.int32
        assert int(values[4]) == 3000
        return sentinel

    monkeypatch.setattr(cuda, "pad_noise_pixels_cuda", record)
    assert (
        pack_noise_pixel_capacity(*args[:4], target_batch=42, n_images=3000, norm_capacity=3072, cuda_packing=True)
        is sentinel
    )
    with pytest.raises(TypeError, match="must be a boolean"):
        pack_noise_pixel_capacity(*args[:4], target_batch=42, n_images=3000, norm_capacity=3072, cuda_packing="1")


@pytest.mark.gpu
@pytest.mark.parametrize("batch,target", [(1, 42), (11, 63), (32, 42), (41, 42), (42, 42)])
@pytest.mark.parametrize("wide", [False, True])
def test_gpu_all_prefix_tail_and_input_bytes(batch, target, wide):
    assert jax.default_backend() == "gpu" and cuda.cuda_available()
    arrays = operands(batch, wide)
    inputs = tuple(jnp.asarray(x) for x in arrays)
    reference = _pad_noise_pixels(*inputs, target_batch=target)
    actual = cuda.pad_noise_pixels_cuda(*inputs, target_batch=target)
    jax.block_until_ready((reference, actual))
    for a, b in zip(actual, reference, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    for a, b in zip(arrays, inputs, strict=True):
        assert a.tobytes() == np.asarray(b).tobytes()
    for a in actual[:3]:
        assert np.asarray(a)[batch:].tobytes() == np.zeros(a.shape[0:1] + a.shape[1:], a.dtype)[batch:].tobytes()
    assert np.asarray(actual[3])[batch:].tobytes() == np.full(target - batch, arrays[4], arrays[3].dtype).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["rank", "dtype", "shape", "output_dtype", "output_shape", "target"])
def test_raw_ffi_rejects_before_copy(case):
    cuda._ensure_noise_pixel_pack_ffi()
    args = [jnp.asarray(a) for a in operands()]
    outputs = list(cuda._noise_pixel_pack_shapes(*args, target_batch=42))
    target = 42
    if case == "rank":
        args[0] = args[0].reshape(-1)
    elif case == "dtype":
        args[4] = args[4].astype(jnp.float32)
    elif case == "shape":
        args[2] = args[2][:1]
    elif case == "output_dtype":
        outputs[0] = jax.ShapeDtypeStruct(outputs[0].shape, jnp.int32)
    elif case == "output_shape":
        outputs[1] = jax.ShapeDtypeStruct((41, 3, 5), jnp.complex64)
    elif case == "target":
        target = 10
    with pytest.raises((ValueError, RuntimeError), match="NoisePixelPack:"):
        result = jax.ffi.ffi_call(cuda._TARGET_NOISE_PIXEL_PACK, tuple(outputs), vmap_method="sequential")(
            *args, target_batch=target
        )
        jax.block_until_ready(result)
