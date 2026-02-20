import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.linalg as linalg

pytestmark = pytest.mark.unit


def test_blockwise_xtx_matches_direct():
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(12, 5)) + 1j * rng.normal(size=(12, 5))).astype(np.complex64)
    out = linalg.blockwise_X_T_X(x, batch_size=4)
    ref = np.conj(x).T @ x
    np.testing.assert_allclose(out, ref, atol=5e-3, rtol=5e-3)


def test_broadcast_dot_and_outer():
    x = np.array([[1 + 1j, 2 + 0j], [0 + 1j, 3 + 2j]], dtype=np.complex64)
    y = np.array([[2 + 0j, 1 + 1j], [1 - 1j, 0 + 2j]], dtype=np.complex64)

    dot = np.asarray(linalg.broadcast_dot(x, y))
    ref_dot = np.sum(np.conj(x) * y, axis=-1)
    np.testing.assert_allclose(dot, ref_dot, atol=1e-6, rtol=1e-6)

    outer = np.asarray(linalg.broadcast_outer(x, y))
    ref_outer = np.stack([np.outer(xi, np.conj(yi)) for xi, yi in zip(x, y)], axis=0)
    np.testing.assert_allclose(outer, ref_outer, atol=1e-6, rtol=1e-6)


def test_multiply_along_axis_and_l2_distance():
    a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.asarray(linalg.multiply_along_axis(a, b, axis=1))
    np.testing.assert_allclose(out[:, 0, :], a[:, 0, :] * 1.0)
    np.testing.assert_allclose(out[:, 1, :], a[:, 1, :] * 2.0)
    np.testing.assert_allclose(out[:, 2, :], a[:, 2, :] * 3.0)

    x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    d = np.asarray(linalg.l2_distance(x, y))
    ref = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(d, ref, atol=1e-6, rtol=1e-6)
