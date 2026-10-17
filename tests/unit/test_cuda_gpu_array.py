"""``cuda_backproject.GpuArray``: the dev ctypes device buffer stays public."""

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import cuda_backproject

pytestmark = pytest.mark.unit


def test_gpu_array_is_public():
    assert callable(cuda_backproject.GpuArray)
    for method in ("as_float_ptr", "to_numpy", "free"):
        assert callable(getattr(cuda_backproject.GpuArray, method))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.int32])
def test_gpu_array_round_trips_host_data(gpu_device, dtype):
    rng = np.random.default_rng(0)
    host = (rng.standard_normal((3, 5, 7)) * 100).astype(dtype)

    array = cuda_backproject.GpuArray(host)
    try:
        assert array.shape == host.shape and array.dtype == host.dtype and array.nbytes == host.nbytes
        assert array.as_float_ptr()
        np.testing.assert_array_equal(array.to_numpy(), host)
    finally:
        array.free()
    assert not array._ptr
    array.free()
