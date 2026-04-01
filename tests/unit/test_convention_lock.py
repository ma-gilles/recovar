"""Convention lock tests for RECOVAR Fourier representations.

Freezes the following internal conventions (Phase 0A of the RELION-parity plan):

1. Packed-half column ordering: packed col 0 = DC, packed col -1 = Nyquist (even W).
2. Half-image roundtrip: full_image_to_half_image -> half_image_to_full_image is lossless
   for Hermitian data.
3. Full-vs-half inner product: weighted half-spectrum inner product matches full-spectrum.
4. Forward projection full-vs-half: converting full projection to half matches direct
   half projection.
5. FFT normalization roundtrip: forward FFT then inverse FFT recovers original signal.
6. Default max_r semantics: Nyquist edge is excluded (image_shape[0]//2 - 1).
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.core.slicing import slice_volume, _default_max_r

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (64, 64)
VOLUME_SHAPE = (64, 64, 64)
H, W = IMAGE_SHAPE
N_HALF = H * (W // 2 + 1)  # 64 * 33 = 2112
N_FULL = H * W              # 64 * 64 = 4096


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hermitian_image_2d(image_shape, seed=42):
    """Generate a Hermitian-symmetric 2D spectrum (DFT of real data), centered."""
    rng = np.random.default_rng(seed)
    real_img = rng.standard_normal(image_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fft2(real_img))
    return jnp.array(ft, dtype=jnp.complex64)


def _hermitian_volume(volume_shape, seed=42):
    """Generate a Hermitian-symmetric 3D volume (DFT of real data), centered."""
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _random_rotations(n, seed=42):
    """Generate n random rotation matrices via QR decomposition."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return jnp.array(q, dtype=jnp.float32)


# ===========================================================================
# Test 1: Packed-half column ordering
# ===========================================================================


class TestPackedHalfColumnOrdering:
    """Verify packed column 0 is DC and packed column -1 is Nyquist for even W."""

    def test_dc_at_packed_column_zero(self):
        """A constant image (pure DC) should have all energy at packed column 0."""
        # Constant image -> DFT has energy only at DC
        const_img = jnp.ones(IMAGE_SHAPE, dtype=jnp.float32)
        ft_full = ftu.get_dft2(const_img)  # centered full spectrum

        # Pack to half
        half = ftu.full_image_to_half_image(
            ft_full.reshape(1, H * W), IMAGE_SHAPE
        ).reshape(H, W // 2 + 1)

        # DC is the (0, 0) frequency in the centered spectrum, which is at
        # index (H//2, W//2) = (32, 32). After packing, this should be at
        # column 0, row H//2.
        dc_value = half[H // 2, 0]
        assert abs(float(dc_value.real)) > 0, "DC component should be nonzero"

        # All other elements should be zero (or nearly so)
        mask = jnp.ones((H, W // 2 + 1), dtype=bool).at[H // 2, 0].set(False)
        non_dc_max = float(jnp.max(jnp.abs(half[mask])))
        assert non_dc_max < 1e-5, (
            f"Non-DC elements should be ~zero for constant image, got max={non_dc_max}"
        )

    def test_nyquist_at_packed_column_last(self):
        """A Nyquist-frequency signal along the W axis should appear at packed column -1."""
        # Create an image that alternates +1/-1 along columns (Nyquist in W)
        # f(x, y) = (-1)^y = cos(pi * y), which has all energy at ky = W/2
        row = np.array([(-1) ** j for j in range(W)], dtype=np.float32)
        nyquist_img = jnp.broadcast_to(jnp.array(row), IMAGE_SHAPE)

        ft_full = ftu.get_dft2(nyquist_img)
        half = ftu.full_image_to_half_image(
            ft_full.reshape(1, H * W), IMAGE_SHAPE
        ).reshape(H, W // 2 + 1)

        # Nyquist column is packed column -1
        nyquist_col = half[:, -1]
        # Should have nonzero energy at row H//2 (DC in the H dimension)
        assert float(jnp.abs(nyquist_col[H // 2])) > 0, (
            "Nyquist component should be nonzero at packed column -1"
        )

        # Interior columns (not DC col 0, not Nyquist col -1) should be zero
        interior = half[:, 1:-1]
        interior_max = float(jnp.max(jnp.abs(interior)))
        assert interior_max < 1e-5, (
            f"Interior columns should be ~zero for Nyquist signal, got max={interior_max}"
        )

    def test_packed_indices_structure(self):
        """Verify get_real_fft_packed_last_axis_indices maps col 0 -> DC, col -1 -> Nyquist."""
        idx = ftu.get_real_fft_packed_last_axis_indices(W)
        idx_np = np.array(idx)

        # In a shifted full axis of length W=64, DC is at position W//2=32
        # and Nyquist is at position 0
        assert int(idx_np[0]) == W // 2, (
            f"Packed col 0 should map to shifted DC index {W // 2}, got {int(idx_np[0])}"
        )
        assert int(idx_np[-1]) == 0, (
            f"Packed col -1 should map to shifted Nyquist index 0, got {int(idx_np[-1])}"
        )

        # Length should be W//2 + 1
        assert len(idx_np) == W // 2 + 1


# ===========================================================================
# Test 2: Half-image roundtrip
# ===========================================================================


class TestHalfImageRoundtrip:
    """Random Hermitian data -> full_to_half -> half_to_full must be lossless."""

    def test_roundtrip_single_image(self):
        """Single Hermitian image roundtrips through half representation."""
        full_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=42)
        full_flat = full_2d.reshape(1, H * W)

        half = ftu.full_image_to_half_image(full_flat, IMAGE_SHAPE)
        recovered = ftu.half_image_to_full_image(half, IMAGE_SHAPE)

        # Machine precision for complex64
        np.testing.assert_allclose(
            np.array(recovered),
            np.array(full_flat),
            atol=1e-5,
            rtol=1e-5,
            err_msg="Half-image roundtrip failed for single Hermitian image",
        )

    def test_roundtrip_batch(self):
        """Batch of Hermitian images roundtrips correctly."""
        n_images = 5
        full_batch = jnp.stack(
            [_hermitian_image_2d(IMAGE_SHAPE, seed=i).reshape(H * W) for i in range(n_images)]
        )

        half = ftu.full_image_to_half_image(full_batch, IMAGE_SHAPE)
        recovered = ftu.half_image_to_full_image(half, IMAGE_SHAPE)

        np.testing.assert_allclose(
            np.array(recovered),
            np.array(full_batch),
            atol=1e-5,
            rtol=1e-5,
            err_msg="Half-image roundtrip failed for batch",
        )

    def test_half_shape(self):
        """Verify the packed half shape is (H, W//2+1)."""
        full_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=0)
        half = ftu.full_image_to_half_image(
            full_2d.reshape(1, H * W), IMAGE_SHAPE
        )
        assert half.shape == (1, N_HALF), (
            f"Expected half shape (1, {N_HALF}), got {half.shape}"
        )


# ===========================================================================
# Test 3: Full-vs-half inner product
# ===========================================================================


class TestFullVsHalfInnerProduct:
    """Weighted half-spectrum inner product must match full-spectrum inner product."""

    def _make_half_weights(self):
        """Weights for the half-spectrum inner product: 1 for DC/Nyquist cols, 2 for interior."""
        w = 2.0 * jnp.ones((H, W // 2 + 1), dtype=jnp.float32)
        w = w.at[:, 0].set(1.0)    # DC column
        w = w.at[:, -1].set(1.0)   # Nyquist column
        return w.reshape(-1)

    def test_inner_product_agreement(self):
        """Re<a, b>_full == Re[sum(conj(a_half) * w * b_half)] for Hermitian data."""
        a_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=10)
        b_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=20)

        a_flat = a_2d.reshape(H * W)
        b_flat = b_2d.reshape(H * W)

        # Full inner product: Re[sum_k conj(a(k)) * b(k)] over all N pixels
        ip_full = jnp.sum(jnp.conj(a_flat) * b_flat).real

        # Half inner product with weights
        a_half = ftu.full_image_to_half_image(a_flat[None, :], IMAGE_SHAPE).reshape(-1)
        b_half = ftu.full_image_to_half_image(b_flat[None, :], IMAGE_SHAPE).reshape(-1)
        w = self._make_half_weights()
        ip_half = jnp.sum(jnp.conj(a_half) * w * b_half).real

        # Use rtol because float32 summation over ~4K terms yields absolute
        # errors proportional to magnitude.  Relative error should be < 1e-5.
        np.testing.assert_allclose(
            float(ip_full),
            float(ip_half),
            rtol=1e-5,
            err_msg="Full-vs-half inner product mismatch",
        )

    def test_inner_product_multiple_pairs(self):
        """Test inner product agreement across multiple random pairs."""
        w = self._make_half_weights()
        for seed_a, seed_b in [(0, 1), (3, 7), (42, 99), (100, 200)]:
            a_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=seed_a)
            b_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=seed_b)

            a_flat = a_2d.reshape(H * W)
            b_flat = b_2d.reshape(H * W)

            ip_full = jnp.sum(jnp.conj(a_flat) * b_flat).real
            a_half = ftu.full_image_to_half_image(a_flat[None, :], IMAGE_SHAPE).reshape(-1)
            b_half = ftu.full_image_to_half_image(b_flat[None, :], IMAGE_SHAPE).reshape(-1)
            ip_half = jnp.sum(jnp.conj(a_half) * w * b_half).real

            np.testing.assert_allclose(
                float(ip_full),
                float(ip_half),
                rtol=1e-5,
                err_msg=f"Inner product mismatch for seeds ({seed_a}, {seed_b})",
            )

    def test_self_inner_product(self):
        """Self inner product (norm squared) must agree between full and half."""
        a_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=55)
        a_flat = a_2d.reshape(H * W)

        norm_sq_full = jnp.sum(jnp.abs(a_flat) ** 2).real

        a_half = ftu.full_image_to_half_image(a_flat[None, :], IMAGE_SHAPE).reshape(-1)
        w = self._make_half_weights()
        norm_sq_half = jnp.sum(w * jnp.abs(a_half) ** 2).real

        np.testing.assert_allclose(
            float(norm_sq_full),
            float(norm_sq_half),
            rtol=1e-5,
            err_msg="Self-inner-product (norm squared) mismatch between full and half",
        )


# ===========================================================================
# Test 4: Forward projection full-vs-half
# ===========================================================================


class TestForwardProjectionFullVsHalf:
    """full_image_to_half_image(slice_volume(..., half_image=False)) must match
    slice_volume(..., half_image=True)."""

    def test_identity_rotation(self):
        """Forward projection matches for identity rotation."""
        vol = _hermitian_volume(VOLUME_SHAPE, seed=42)
        rot = jnp.eye(3, dtype=jnp.float32)[None, :, :]

        proj_full = slice_volume(
            vol, rot, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=False
        )
        proj_half = slice_volume(
            vol, rot, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=True
        )

        proj_full_to_half = ftu.full_image_to_half_image(proj_full, IMAGE_SHAPE)
        np.testing.assert_allclose(
            np.array(proj_full_to_half),
            np.array(proj_half),
            atol=1e-6,
            err_msg="Full-to-half vs direct half projection mismatch (identity rotation)",
        )

    def test_random_rotations(self):
        """Forward projection matches for several random rotations."""
        vol = _hermitian_volume(VOLUME_SHAPE, seed=7)
        rots = _random_rotations(5, seed=12)

        proj_full = slice_volume(
            vol, rots, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=False
        )
        proj_half = slice_volume(
            vol, rots, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=True
        )

        proj_full_to_half = ftu.full_image_to_half_image(proj_full, IMAGE_SHAPE)
        max_diff = float(jnp.max(jnp.abs(proj_full_to_half - proj_half)))
        scale = float(jnp.max(jnp.abs(proj_half)))

        np.testing.assert_allclose(
            np.array(proj_full_to_half),
            np.array(proj_half),
            atol=1e-6,
            err_msg=(
                f"Full-to-half vs direct half projection mismatch "
                f"(random rotations): max_diff={max_diff:.2e}, scale={scale:.2e}"
            ),
        )


# ===========================================================================
# Test 5: FFT normalization roundtrip
# ===========================================================================


class TestFFTNormalizationRoundtrip:
    """Forward FFT then inverse FFT must recover the original signal."""

    def test_fft2_roundtrip(self):
        """2D FFT -> iFFT roundtrip preserves signal to machine precision."""
        rng = np.random.default_rng(42)
        real_img = jnp.array(rng.standard_normal(IMAGE_SHAPE).astype(np.float32))

        ft = ftu.get_dft2(real_img)
        recovered = ftu.get_idft2(ft)

        np.testing.assert_allclose(
            np.array(recovered.real),
            np.array(real_img),
            atol=1e-5,
            err_msg="2D FFT roundtrip failed",
        )
        # Imaginary part should be negligible
        assert float(jnp.max(jnp.abs(recovered.imag))) < 1e-5

    def test_fft3_roundtrip(self):
        """3D FFT -> iFFT roundtrip preserves signal to machine precision."""
        rng = np.random.default_rng(42)
        real_vol = jnp.array(rng.standard_normal(VOLUME_SHAPE).astype(np.float32))

        ft = ftu.get_dft3(real_vol)
        recovered = ftu.get_idft3(ft)

        np.testing.assert_allclose(
            np.array(recovered.real),
            np.array(real_vol),
            atol=1e-4,
            err_msg="3D FFT roundtrip failed",
        )
        assert float(jnp.max(jnp.abs(recovered.imag))) < 1e-4

    def test_real_fft2_roundtrip(self):
        """Real 2D FFT (rfft2-based) -> inverse roundtrip preserves signal."""
        rng = np.random.default_rng(42)
        real_img = jnp.array(rng.standard_normal(IMAGE_SHAPE).astype(np.float32))

        ft_half = ftu.get_dft2_real(real_img)
        recovered = ftu.get_idft2_real(ft_half, image_shape=IMAGE_SHAPE)

        np.testing.assert_allclose(
            np.array(recovered),
            np.array(real_img),
            atol=1e-5,
            err_msg="Real 2D FFT roundtrip failed",
        )

    def test_real_fft3_roundtrip(self):
        """Real 3D FFT (rfftn-based) -> inverse roundtrip preserves signal."""
        rng = np.random.default_rng(42)
        real_vol = jnp.array(rng.standard_normal(VOLUME_SHAPE).astype(np.float32))

        ft_half = ftu.get_dft3_real(real_vol)
        recovered = ftu.get_idft3_real(ft_half, volume_shape=VOLUME_SHAPE)

        np.testing.assert_allclose(
            np.array(recovered),
            np.array(real_vol),
            atol=1e-4,
            err_msg="Real 3D FFT roundtrip failed",
        )

    def test_default_fft_norm_is_backward(self):
        """Verify DEFAULT_FFT_NORM is 'backward' (unnormalized forward, 1/N inverse)."""
        assert ftu.DEFAULT_FFT_NORM == "backward"


# ===========================================================================
# Test 6: max_r semantics
# ===========================================================================


class TestMaxRSemantics:
    """Default max_r for slicing must exclude the Nyquist edge."""

    def test_default_max_r_value(self):
        """Default max_r = image_shape[0] // 2 - 1 for standard sizes."""
        assert _default_max_r(IMAGE_SHAPE) == IMAGE_SHAPE[0] // 2 - 1
        assert _default_max_r((64, 64)) == 31
        assert _default_max_r((128, 128)) == 63
        assert _default_max_r((256, 256)) == 127

    def test_max_r_excludes_nyquist(self):
        """Default max_r is strictly less than N//2 (Nyquist excluded).

        RELION clips at N//2 (Nyquist included). RECOVAR uses N//2 - 1 to
        ensure half-image and full-image results are exactly equivalent and
        the Nyquist edge (which has no Hermitian conjugate partner in the
        full representation) is excluded.
        """
        for N in [32, 64, 128, 256]:
            max_r = _default_max_r((N, N))
            nyquist = N // 2
            assert max_r == nyquist - 1, (
                f"For N={N}: max_r={max_r} should be {nyquist - 1} (Nyquist-1)"
            )
            assert max_r < nyquist, (
                f"For N={N}: max_r={max_r} must be < Nyquist={nyquist}"
            )
