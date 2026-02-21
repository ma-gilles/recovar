import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import ppca

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
