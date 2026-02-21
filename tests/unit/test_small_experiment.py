import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp
from jax.experimental import sparse

from recovar import small_experiment as se


def test_vec_unvec_roundtrip():
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    v = se.vec(x)
    x2 = se.unvec(v)
    assert x2.shape == x.shape
    assert np.allclose(x2, x)


def test_indices_to_coo_shape_and_values():
    indices = np.array([[0, 2], [1, 3]], dtype=np.int32)
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    coo = se.indices_to_coo(indices, n=5, data=data)
    dense = np.array(coo.todense())
    assert dense.shape == (2, 5)
    assert np.allclose(dense[0, [0, 2]], [1.0, 2.0])
    assert np.allclose(dense[1, [1, 3]], [3.0, 4.0])


def test_subsample_coo_columns_remaps_columns():
    rows = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    cols = jnp.array([0, 2, 1, 3], dtype=jnp.int32)
    vals = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
    sp = sparse.BCOO((vals, jnp.stack([rows, cols], axis=1)), shape=(2, 4))
    sub = se.subsample_coo_columns(sp, right_indices=jnp.array([1, 3], dtype=jnp.int32))
    dense = np.array(sub.todense())
    assert dense.shape == (2, 2)
    assert np.allclose(dense[0], [0.0, 0.0])
    assert np.allclose(dense[1], [30.0, 40.0])


def test_make_random_sampling_scheme_shape_and_range():
    indices = se.make_random_sampling_scheme(grid_size=6, m=3, seed=1)
    n = 6 * 6 * 6
    assert indices.shape[0] == 3
    assert int(indices.min()) >= 0
    assert int(indices.max()) < n

