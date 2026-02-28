import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.heterogeneity import ppca

pytestmark = pytest.mark.unit


def test_batch_vec_and_unvec_roundtrip_real():
    x = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    v = ppca.batch_vec(x)
    x_rt = ppca.batch_unvec(v)
    assert v.shape == (2, 9)
    assert x_rt.shape == x.shape
    assert np.allclose(x_rt, x)


def test_batch_vec_and_unvec_roundtrip_complex():
    rng = np.random.default_rng(0)
    xr = rng.normal(size=(4, 2, 2)).astype(np.float32)
    xi = rng.normal(size=(4, 2, 2)).astype(np.float32)
    x = xr + 1j * xi
    v = ppca.batch_vec(x)
    x_rt = ppca.batch_unvec(v)
    assert v.shape == (4, 4)
    assert x_rt.shape == x.shape
    assert np.allclose(x_rt, x)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_batch_vec_unvec_roundtrip_gpu(gpu_device):
    x = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)

    cpu_v = np.asarray(ppca.batch_vec(x))
    cpu_rt = np.asarray(ppca.batch_unvec(cpu_v))

    with jax.default_device(gpu_device):
        x_g = jax.device_put(jnp.array(x), gpu_device)
        gpu_v = np.asarray(ppca.batch_vec(x_g))
        gpu_rt = np.asarray(ppca.batch_unvec(gpu_v))

    np.testing.assert_allclose(cpu_v, gpu_v, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_rt, gpu_rt, atol=1e-5, rtol=1e-5)
