"""Unit tests for recovar.core.linalg."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.linalg as linalg

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# batched Fourier helpers
# ---------------------------------------------------------------------------


class TestBatchedFourierHelpers:
    def test_batch_idft3_overwrite_reuses_input(self, monkeypatch):
        x = np.arange(12, dtype=np.complex64).reshape(4, 3)
        original = x.copy()

        def fake_idft3(block, vec_shape):
            assert vec_shape == (2, 2, 1)
            return np.asarray(block) + 10

        monkeypatch.setattr(linalg, "idft3", fake_idft3)

        out = linalg.batch_idft3(x, (2, 2, 1), batch_size=2, overwrite=True)

        assert out is x
        np.testing.assert_allclose(out, original + 10)

    def test_batch_idft3_default_preserves_input(self, monkeypatch):
        x = np.arange(12, dtype=np.complex64).reshape(4, 3)
        original = x.copy()

        monkeypatch.setattr(linalg, "idft3", lambda block, vec_shape: np.asarray(block) + 5)

        out = linalg.batch_idft3(x, (2, 2, 1), batch_size=2)

        assert out is not x
        np.testing.assert_allclose(x, original)
        np.testing.assert_allclose(out, original + 5)


# ---------------------------------------------------------------------------
# batch_st_end
# ---------------------------------------------------------------------------


class TestBatchStEnd:
    def test_first_batch(self):
        st, end = linalg.batch_st_end(0, 10, 50)
        assert st == 0
        assert end == 10

    def test_middle_batch(self):
        st, end = linalg.batch_st_end(2, 10, 50)
        assert st == 20
        assert end == 30

    def test_last_batch_clipped(self):
        st, end = linalg.batch_st_end(4, 10, 45)
        assert st == 40
        assert end == 45

    def test_single_element_batches(self):
        st, end = linalg.batch_st_end(3, 1, 5)
        assert st == 3
        assert end == 4


# ---------------------------------------------------------------------------
# inner_product
# ---------------------------------------------------------------------------


class TestInnerProduct:
    def test_real_vectors(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        result = linalg.inner_product(x, y)
        np.testing.assert_allclose(float(result), 32.0, atol=1e-5)

    def test_complex_conjugate(self):
        x = jnp.array([1 + 1j, 2 + 0j])
        y = jnp.array([1 - 1j, 0 + 2j])
        result = linalg.inner_product(x, y)
        expected = np.vdot(np.array(x), np.array(y))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_self_inner_product_is_norm_squared(self):
        x = jnp.array([3.0, 4.0])
        result = linalg.inner_product(x, x)
        np.testing.assert_allclose(float(result), 25.0, atol=1e-5)

    def test_shape_mismatch_raises(self):
        x = jnp.array([1.0, 2.0])
        y = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="same shape"):
            linalg.inner_product(x, y)

    def test_2d_arrays(self):
        x = jnp.ones((3, 4))
        y = jnp.ones((3, 4)) * 2
        result = linalg.inner_product(x, y)
        np.testing.assert_allclose(float(result), 24.0, atol=1e-5)


# ---------------------------------------------------------------------------
# batch_inner_product
# ---------------------------------------------------------------------------


class TestBatchInnerProduct:
    def test_basic(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = linalg.batch_inner_product(x, y)
        expected = np.array([1 * 5 + 2 * 6, 3 * 7 + 4 * 8], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_complex(self):
        x = jnp.array([[1 + 1j, 2 + 0j]])
        y = jnp.array([[1 - 1j, 0 + 2j]])
        result = linalg.batch_inner_product(x, y)
        expected = np.conj(1 + 1j) * (1 - 1j) + np.conj(2 + 0j) * (0 + 2j)
        np.testing.assert_allclose(result, [expected], atol=1e-5)

    def test_shape_mismatch_raises(self):
        x = jnp.ones((3, 4))
        y = jnp.ones((3, 5))
        with pytest.raises(ValueError, match="same shape"):
            linalg.batch_inner_product(x, y)

    def test_1d_raises(self):
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        with pytest.raises(ValueError, match="ndim>=2"):
            linalg.batch_inner_product(x, y)

    def test_3d_input(self):
        x = jnp.ones((2, 3, 4))
        y = jnp.ones((2, 3, 4)) * 2
        result = linalg.batch_inner_product(x, y)
        np.testing.assert_allclose(result, [24.0, 24.0], atol=1e-5)


# ---------------------------------------------------------------------------
# half_spectrum_last_axis_weights
# ---------------------------------------------------------------------------


class TestHalfSpectrumWeights:
    def test_even_n(self):
        w = linalg.half_spectrum_last_axis_weights(8)
        # For even n=8: m=5 bins, edges=[0,4], interior=[1,2,3]
        assert w.shape == (5,)
        np.testing.assert_allclose(w[0], 1.0)
        np.testing.assert_allclose(w[-1], 1.0)
        np.testing.assert_allclose(w[1:-1], 2.0)

    def test_odd_n(self):
        w = linalg.half_spectrum_last_axis_weights(7)
        # For odd n=7: m=4 bins, edge=[0], interior=[1,2,3]
        assert w.shape == (4,)
        np.testing.assert_allclose(w[0], 1.0)
        np.testing.assert_allclose(w[1:], 2.0)

    def test_n_1(self):
        w = linalg.half_spectrum_last_axis_weights(1)
        assert w.shape == (1,)
        np.testing.assert_allclose(w[0], 1.0)

    def test_n_2(self):
        w = linalg.half_spectrum_last_axis_weights(2)
        assert w.shape == (2,)
        np.testing.assert_allclose(w, [1.0, 1.0])

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            linalg.half_spectrum_last_axis_weights(0)


# ---------------------------------------------------------------------------
# broadcast_dot / broadcast_outer
# ---------------------------------------------------------------------------


class TestBroadcastOps:
    def test_broadcast_dot_real(self):
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = linalg.broadcast_dot(x, y)
        np.testing.assert_allclose(result, [1.0, 5.0], atol=1e-5)

    def test_broadcast_dot_complex(self):
        x = jnp.array([[1 + 1j, 2 + 0j]])
        y = jnp.array([[1 + 1j, 2 + 0j]])
        result = linalg.broadcast_dot(x, y)
        expected = np.conj(1 + 1j) * (1 + 1j) + np.conj(2) * 2
        np.testing.assert_allclose(result, [expected], atol=1e-5)

    def test_broadcast_outer_shape(self):
        x = jnp.ones((3, 4))
        y = jnp.ones((3, 5))
        result = linalg.broadcast_outer(x, y)
        assert result.shape == (3, 4, 5)


# ---------------------------------------------------------------------------
# multiply_along_axis
# ---------------------------------------------------------------------------


class TestMultiplyAlongAxis:
    def test_axis_0(self):
        A = jnp.ones((3, 4, 5))
        B = jnp.array([1.0, 2.0, 3.0])
        result = linalg.multiply_along_axis(A, B, axis=0)
        assert result.shape == (3, 4, 5)
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1], 2.0)
        np.testing.assert_allclose(result[2], 3.0)

    def test_axis_last(self):
        A = jnp.ones((2, 3))
        B = jnp.array([10.0, 20.0, 30.0])
        result = linalg.multiply_along_axis(A, B, axis=-1)
        expected = jnp.array([[10, 20, 30], [10, 20, 30]], dtype=jnp.float32)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# batch_hermitian_linear_solver / batch_linear_solver
# ---------------------------------------------------------------------------


class TestLinearSolvers:
    def test_hermitian_solver_identity(self):
        A = jnp.eye(3)
        b = jnp.array([1.0, 2.0, 3.0])
        x = linalg.batch_hermitian_linear_solver(A, b)
        np.testing.assert_allclose(x, b, atol=1e-5)

    def test_hermitian_solver_spd(self):
        rng = np.random.RandomState(42)
        M = rng.randn(4, 4).astype(np.float32)
        A = jnp.array(M.T @ M + np.eye(4) * 0.1)
        b_true = jnp.array([1.0, -1.0, 0.5, -0.5])
        b = A @ b_true
        x = linalg.batch_hermitian_linear_solver(A, b)
        np.testing.assert_allclose(x, b_true, atol=1e-4)

    def test_batch_linear_solver(self):
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        b = jnp.array([5.0, 7.0])
        x = linalg.batch_linear_solver(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-5)


# ---------------------------------------------------------------------------
# solve_by_SVD
# ---------------------------------------------------------------------------


class TestSolveBySVD:
    def test_identity_system(self):
        A = jnp.eye(3)[None, ...]  # batch of 1
        b = jnp.array([[1.0, 2.0, 3.0]])
        x = linalg.solve_by_SVD(A, b)
        np.testing.assert_allclose(x, b, atol=1e-4)

    def test_simple_system(self):
        A = jnp.array([[[2.0, 0.0], [0.0, 3.0]]])
        b = jnp.array([[6.0, 9.0]])
        x = linalg.solve_by_SVD(A, b)
        np.testing.assert_allclose(x, [[3.0, 3.0]], atol=1e-4)

    def test_batch_dimension(self):
        A = jnp.stack([jnp.eye(2), 2 * jnp.eye(2)])  # (2, 2, 2)
        b = jnp.array([[1.0, 1.0], [4.0, 4.0]])
        x = linalg.solve_by_SVD(A, b)
        np.testing.assert_allclose(x, [[1.0, 1.0], [2.0, 2.0]], atol=1e-4)


# ---------------------------------------------------------------------------
# l2_distance
# ---------------------------------------------------------------------------


class TestL2Distance:
    def test_identical_points(self):
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        D = linalg.l2_distance(X, X)
        np.testing.assert_allclose(jnp.diag(D), 0.0, atol=1e-5)

    def test_known_distances(self):
        X = jnp.array([[0.0, 0.0]])
        Y = jnp.array([[3.0, 4.0]])
        D = linalg.l2_distance(X, Y)
        np.testing.assert_allclose(D[0, 0], 25.0, atol=1e-4)

    def test_symmetry(self):
        rng = np.random.RandomState(42)
        X = jnp.array(rng.randn(5, 3).astype(np.float32))
        D = linalg.l2_distance(X, X)
        np.testing.assert_allclose(D, D.T, atol=1e-4)

    def test_shape(self):
        X = jnp.ones((3, 4))
        Y = jnp.ones((5, 4))
        D = linalg.l2_distance(X, Y)
        assert D.shape == (3, 5)


# ---------------------------------------------------------------------------
# thin_svd
# ---------------------------------------------------------------------------


class TestThinSVD:
    def test_identity_matrix(self):
        A = np.eye(4, dtype=np.float32)
        U, S, V = linalg.thin_svd(A)
        np.testing.assert_allclose(S, np.ones(4), atol=1e-5)

    def test_rank1(self):
        A = np.array([[1, 2], [2, 4], [3, 6]], dtype=np.float32)
        U, S, V = linalg.thin_svd(A)
        # One large singular value, one near-zero
        assert S[0] > 1.0
        np.testing.assert_allclose(S[1], 0.0, atol=1e-5)

    def test_reconstruction(self):
        rng = np.random.RandomState(42)
        A = rng.randn(10, 5).astype(np.float32)
        U, S, V = linalg.thin_svd(A)
        # thin_svd returns V as column vectors, so A ≈ U @ diag(S) @ V.T
        reconstructed = (U * S) @ V.T
        np.testing.assert_allclose(reconstructed, A, atol=1e-4)


# ---------------------------------------------------------------------------
# half_spectrum_inner_product
# ---------------------------------------------------------------------------


class TestHalfSpectrumInnerProduct:
    def test_matches_full_spectrum_2d(self):
        """Half-spectrum inner product should equal full-spectrum inner product."""
        rng = np.random.RandomState(42)
        full_shape = (8, 8)
        # Create two random real-valued images
        img1 = rng.randn(*full_shape).astype(np.float32)
        img2 = rng.randn(*full_shape).astype(np.float32)
        # Full DFT
        ft1_full = np.fft.fft2(img1)
        ft2_full = np.fft.fft2(img2)
        full_ip = np.vdot(ft1_full, ft2_full)
        # Half-spectrum (real FFT)
        ft1_half = jnp.array(np.fft.rfft2(img1))
        ft2_half = jnp.array(np.fft.rfft2(img2))
        half_ip = linalg.half_spectrum_inner_product(ft1_half, ft2_half, full_shape)
        np.testing.assert_allclose(float(half_ip.real), float(full_ip.real), rtol=1e-4)

    def test_self_product_positive(self):
        rng = np.random.RandomState(42)
        full_shape = (6, 6)
        img = rng.randn(*full_shape).astype(np.float32)
        ft_half = jnp.array(np.fft.rfft2(img))
        ip = linalg.half_spectrum_inner_product(ft_half, ft_half, full_shape)
        assert float(ip.real) > 0

    def test_shape_mismatch_raises(self):
        x = jnp.ones((4, 3))
        y = jnp.ones((4, 4))
        with pytest.raises(ValueError):
            linalg.half_spectrum_inner_product(x, y, (4, 4))


class TestBatchHalfSpectrumInnerProduct:
    def test_batch_matches_loop(self):
        rng = np.random.RandomState(42)
        full_shape = (6, 6)
        batch = 4
        imgs = rng.randn(batch, *full_shape).astype(np.float32)
        fts_half = jnp.array(np.fft.rfft2(imgs, axes=(-2, -1)))
        result = linalg.batch_half_spectrum_inner_product(fts_half, fts_half, full_shape)
        assert result.shape == (batch,)
        for i in range(batch):
            single = linalg.half_spectrum_inner_product(fts_half[i], fts_half[i], full_shape)
            np.testing.assert_allclose(float(result[i].real), float(single.real), rtol=1e-4)


# ---------------------------------------------------------------------------
# _coerce_half_grid
# ---------------------------------------------------------------------------


class TestCoerceHalfGrid:
    def test_grid_shape_passthrough(self):
        full_shape = (4, 6)
        half_shape = (4, 4)  # 6//2+1 = 4
        arr = jnp.ones(half_shape)
        out, was_flat = linalg._coerce_half_grid(arr, full_shape, "test")
        assert out.shape == half_shape
        assert not was_flat

    def test_flat_input_reshaped(self):
        full_shape = (4, 6)
        half_shape = (4, 4)  # 6//2+1 = 4
        flat_size = 16
        arr = jnp.ones((flat_size,))
        out, was_flat = linalg._coerce_half_grid(arr, full_shape, "test")
        assert out.shape == half_shape
        assert was_flat

    def test_batched_flat(self):
        full_shape = (4, 6)
        half_shape = (4, 4)
        flat_size = 16
        arr = jnp.ones((3, flat_size))
        out, was_flat = linalg._coerce_half_grid(arr, full_shape, "test")
        assert out.shape == (3, 4, 4)
        assert was_flat

    def test_invalid_shape_raises(self):
        full_shape = (4, 6)
        arr = jnp.ones((7, 7))
        with pytest.raises(ValueError, match="half-spectrum"):
            linalg._coerce_half_grid(arr, full_shape, "test")


# ---------------------------------------------------------------------------
# blockwise operations (small-scale tests)
# ---------------------------------------------------------------------------


class TestBlockwiseOps:
    def test_blockwise_X_T_X(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5).astype(np.float32)
        result = linalg.blockwise_X_T_X(X, batch_size=10)
        expected = X.T @ X
        np.testing.assert_allclose(result, expected, atol=1e-2)

    def test_blockwise_Y_T_X(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5).astype(np.float32)
        Y = rng.randn(20, 4).astype(np.float32)
        result = linalg.blockwise_Y_T_X(Y, X, batch_size=10)
        expected = Y.T @ X
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_blockwise_A_X(self):
        rng = np.random.RandomState(42)
        A = rng.randn(20, 10).astype(np.float32)
        X = rng.randn(10, 3).astype(np.float32)
        result = linalg.blockwise_A_X(A, X, batch_size=5)
        expected = A @ X
        np.testing.assert_allclose(result, expected, atol=1e-3)
