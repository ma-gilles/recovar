"""The GPU powerClass fold changes dispatch, never the addition order."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cuda
from recovar.em.sparse_pass2 import sparse_pass2_scoring as scoring

pytestmark = pytest.mark.unit


def _left_fold(values):
    total = np.zeros(values.shape[0], dtype=values.dtype)
    for column in values.T:
        total = np.add(total, column, dtype=values.dtype)
    return total


@pytest.mark.parametrize(
    "dtype,backend,custom,expected_calls",
    [
        (np.complex64, "gpu", True, 1),
        (np.complex64, "gpu", False, 0),
        (np.complex64, "cpu", True, 0),
        (np.complex128, "gpu", True, 0),
    ],
)
def test_powerclass_fold_dispatch_preserves_cpu_double_and_non_cuda(
    monkeypatch,
    dtype,
    backend,
    custom,
    expected_calls,
):
    rng = np.random.default_rng(813)
    images = (rng.normal(size=(2, 32 * 17)) + 1j * rng.normal(size=(2, 32 * 17))).astype(dtype)
    fn = scoring._relion_cuda_powerclass_highres_xi2_half.__wrapped__
    monkeypatch.setattr(cuda, "custom_cuda_requested", lambda: False)
    expected = fn(jnp.asarray(images), image_shape=(32, 32), current_size=20)
    calls = []

    def folded(values):
        calls.append(np.asarray(values))
        return jnp.asarray(_left_fold(np.asarray(values)))

    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(cuda, "custom_cuda_requested", lambda: custom)
    monkeypatch.setattr(cuda, "relion_ordered_sum_f32", folded)
    actual = fn(jnp.asarray(images), image_shape=(32, 32), current_size=20)
    assert len(calls) == expected_calls
    assert actual.dtype == expected.dtype
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize(
    "shape,dtype,error",
    [
        ((2, 5), jnp.float64, TypeError),
        ((5,), jnp.float32, ValueError),
        ((0, 5), jnp.float32, ValueError),
        ((2, 0), jnp.float32, ValueError),
    ],
)
def test_ordered_sum_validates_before_loading_cuda(shape, dtype, error):
    with pytest.raises(error):
        cuda.relion_ordered_sum_f32.__wrapped__(jnp.zeros(shape, dtype))


def test_ordered_sum_cpu_fails_closed(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="GPU backend"):
        cuda.relion_ordered_sum_f32.__wrapped__(jnp.ones((2, 3), jnp.float32))


def test_ordered_sum_custom_cuda_disabled_fails_closed(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda, "custom_cuda_requested", lambda: False)
    with pytest.raises(RuntimeError, match="custom CUDA"):
        cuda.relion_ordered_sum_f32.__wrapped__(jnp.ones((2, 3), jnp.float32))


@pytest.mark.gpu
@pytest.mark.parametrize("rows,columns", [(1, 1), (4, 7), (16, 568), (18, 568), (250, 129), (3, 2504)])
def test_ordered_sum_gpu_is_bitwise_left_fold(monkeypatch, custom_cuda_lib, gpu_device, rows, columns):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda, "_cuda_ok", None)
    rng = np.random.default_rng(81)
    values = (rng.normal(size=(rows, columns)) * np.exp(rng.uniform(-20, 20, size=(rows, columns)))).astype(np.float32)
    if columns >= 3:
        values[0, :3] = [2**24, 1, -(2**24)]
    expected = _left_fold(values)
    with jax.default_device(gpu_device):
        actual = cuda.relion_ordered_sum_f32(jnp.asarray(values))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), expected.view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "size,current_size,rows,runtime", [(32, 20, 4, False), (128, 70, 2, True), (380, 202, 18, False)]
)
def test_powerclass_full_gpu_matches_unchanged_jax_recurrence(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
    size,
    current_size,
    rows,
    runtime,
):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda, "_cuda_ok", None)
    rng = np.random.default_rng(903)
    shape = (rows, size * (size // 2 + 1))
    images = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    fn = scoring._relion_cuda_powerclass_highres_xi2_half.__wrapped__
    request = cuda.custom_cuda_requested
    with jax.default_device(gpu_device):
        inputs = jnp.asarray(images)
        limit = jnp.asarray(current_size, dtype=jnp.int32) if runtime else None
        # Independent trace with CUDA dispatch disabled preserves the original
        # barriered pixel tree and ascending-block JAX recurrence on this GPU.
        monkeypatch.setattr(cuda, "custom_cuda_requested", lambda: False)
        reference = jax.jit(
            lambda x, n: fn(x, image_shape=(size, size), current_size=current_size, runtime_current_size=n)
        )
        expected = np.asarray(reference(inputs, limit))
        monkeypatch.setattr(cuda, "custom_cuda_requested", request)
        candidate = jax.jit(
            lambda x, n: fn(x, image_shape=(size, size), current_size=current_size, runtime_current_size=n)
        )
        actual = np.asarray(candidate(inputs, limit))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
