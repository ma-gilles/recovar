"""Tests for periodic cubic B-spline coefficients and half-volume roundtrip.

Verifies:
- Periodic coefficients preserve Hermitian symmetry
- Half-volume roundtrip is lossless
- Half-image slicing matches full-image slicing
- Periodic coefficients same shape as input (no boundary padding)
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.core.cubic_interpolation as cubic_interp
import recovar.core.fourier_transform_utils as ftu
import recovar.core.slicing as slicing

pytestmark = pytest.mark.unit


def _make_hermitian_volume(N, rng, dtype=np.complex128):
    """Create a Hermitian-symmetric volume in centered (fftshift) convention."""
    real_vol = rng.standard_normal((N, N, N)).astype(np.float64)
    # fft → fftshift gives centered Hermitian volume
    vol_ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(vol_ft, dtype=dtype)


def _hermitian_error(vol, volume_shape):
    """Measure Hermitian symmetry violation: max|V[k] - conj(V[-k])| / max|V|."""
    N0, N1, N2 = volume_shape
    v = np.asarray(vol).reshape(volume_shape)
    # In centered convention, partner(j) = (N - (N&1) - j) % N
    partner = np.zeros_like(v)
    for i0 in range(N0):
        for i1 in range(N1):
            for i2 in range(N2):
                p0 = (N0 - (N0 & 1) - i0) % N0
                p1 = (N1 - (N1 & 1) - i1) % N1
                p2 = (N2 - (N2 & 1) - i2) % N2
                partner[i0, i1, i2] = np.conj(v[p0, p1, p2])
    return np.max(np.abs(v - partner)) / np.max(np.abs(v))


class TestPeriodicCubicCoefficients:
    """Test that periodic cubic coefficients preserve Hermitian symmetry."""

    def test_output_shape_matches_input(self):
        """Periodic coefficients have same shape as input (no boundary padding)."""
        rng = np.random.default_rng(42)
        for N in [8, 16, 32]:
            vol = _make_hermitian_volume(N, rng)
            coeffs = cubic_interp.calculate_spline_coefficients(vol)
            assert coeffs.shape == vol.shape, f"N={N}: expected {vol.shape}, got {coeffs.shape}"

    def test_hermitian_symmetry_preserved(self):
        """Periodic coefficients of a Hermitian volume are Hermitian."""
        rng = np.random.default_rng(123)
        N = 16
        vol = _make_hermitian_volume(N, rng, dtype=np.complex128)
        volume_shape = (N, N, N)

        # Input should be Hermitian
        input_err = _hermitian_error(vol, volume_shape)
        assert input_err < 1e-14, f"Input not Hermitian: {input_err}"

        # Coefficients should also be Hermitian
        coeffs = cubic_interp.calculate_spline_coefficients(vol)
        coeff_err = _hermitian_error(np.asarray(coeffs), volume_shape)
        assert coeff_err < 1e-12, f"Coefficients not Hermitian: {coeff_err}"

    def test_1d_eigenvalues_positive(self):
        """All eigenvalues of [1,4,1] circulant are positive → stable division."""
        for N in [4, 8, 16, 64, 128]:
            k = np.arange(N)
            eigenvalues = 4.0 + 2.0 * np.cos(2.0 * np.pi * k / N)
            assert np.all(eigenvalues > 0), f"N={N}: negative eigenvalue"


class TestHalfVolumeRoundtrip:
    """Test that half-volume storage is lossless for periodic coefficients."""

    def test_roundtrip_lossless(self):
        """full → half → full roundtrip is exact for Hermitian coefficients."""
        rng = np.random.default_rng(456)
        N = 16
        volume_shape = (N, N, N)
        vol = _make_hermitian_volume(N, rng, dtype=np.complex128)

        coeffs = cubic_interp.calculate_spline_coefficients(vol)
        half_coeffs = ftu.full_volume_to_half_volume(coeffs, volume_shape)
        recovered = ftu.half_volume_to_full_volume(half_coeffs, volume_shape)

        np.testing.assert_allclose(
            np.asarray(recovered), np.asarray(coeffs), atol=1e-12, rtol=1e-12,
            err_msg="Half-volume roundtrip not lossless"
        )

    def test_precompute_half_matches_full(self):
        """precompute_cubic_coefficients_half matches full → half."""
        rng = np.random.default_rng(789)
        N = 16
        volume_shape = (N, N, N)
        vol = _make_hermitian_volume(N, rng, dtype=np.complex128)

        full_coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)
        half_from_full = ftu.full_volume_to_half_volume(full_coeffs, volume_shape).reshape(-1)
        half_direct = slicing.precompute_cubic_coefficients_half(vol, volume_shape)

        np.testing.assert_allclose(
            np.asarray(half_direct), np.asarray(half_from_full),
            atol=1e-12, rtol=1e-12,
        )


class TestCubicSlicingEquivalence:
    """Test that periodic cubic slicing gives consistent results."""

    def _random_rotations(self, n, rng):
        """Generate random rotation matrices."""
        from scipy.spatial.transform import Rotation
        return Rotation.random(n, random_state=rng).as_matrix().astype(np.float64)

    def test_half_image_matches_full_image(self):
        """Cubic half-image slicing matches full → extract half."""
        rng = np.random.default_rng(101)
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        vol = _make_hermitian_volume(N, rng, dtype=np.complex128)

        rots = self._random_rotations(3, rng)

        # Precompute coefficients (callers always do this for cubic)
        coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)

        full_slices = slicing.slice_volume(
            coeffs, rots, image_shape, volume_shape, "cubic",
            max_r=None,
        )
        half_slices = slicing.slice_volume(
            coeffs, rots, image_shape, volume_shape, "cubic",
            half_image=True, max_r=None,
        )

        # Extract half from full
        full_half = ftu.full_image_to_half_image(np.asarray(full_slices), image_shape)
        np.testing.assert_allclose(
            np.asarray(half_slices), full_half,
            atol=1e-10, rtol=1e-10,
            err_msg="Half-image cubic slicing doesn't match full",
        )

    def test_precomputed_matches_direct(self):
        """Precomputed path via slice_from_cubic_coefficients matches slice_volume."""
        rng = np.random.default_rng(202)
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        vol = _make_hermitian_volume(N, rng, dtype=np.complex128)

        rots = self._random_rotations(2, rng)

        # Precompute coefficients (callers always do this for cubic)
        coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)

        # slice_volume with pre-computed coefficients
        direct = slicing.slice_volume(
            coeffs, rots, image_shape, volume_shape, "cubic",
            max_r=None,
        )

        # Explicit precomputed slicer
        precomp = slicing.slice_from_cubic_coefficients(
            coeffs, rots, image_shape, volume_shape,
        )

        np.testing.assert_allclose(
            np.asarray(precomp), np.asarray(direct),
            atol=1e-10, rtol=1e-10,
        )

    def test_half_coeffs_matches_full_coeffs(self):
        """Slicing from half-volume coefficients matches full-volume coefficients."""
        rng = np.random.default_rng(303)
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        vol = _make_hermitian_volume(N, rng, dtype=np.complex128)

        rots = self._random_rotations(2, rng)

        full_coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)
        half_coeffs = slicing.precompute_cubic_coefficients_half(vol, volume_shape)

        result_full = slicing.slice_from_cubic_coefficients(
            full_coeffs, rots, image_shape, volume_shape,
        )
        result_half = slicing.slice_from_cubic_coefficients(
            half_coeffs, rots, image_shape, volume_shape,
        )

        np.testing.assert_allclose(
            np.asarray(result_half), np.asarray(result_full),
            atol=1e-10, rtol=1e-10,
            err_msg="Half-coefficients slicing doesn't match full",
        )

    def test_half_volume_slicing_matches_full_volume(self):
        """slice_volume with half_volume=True matches full_volume for cubic."""
        rng = np.random.default_rng(404)
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        vol = _make_hermitian_volume(N, rng, dtype=np.complex128)

        rots = self._random_rotations(2, rng)

        result_full = slicing.slice_volume(
            vol.ravel(), rots, image_shape, volume_shape, "cubic",
            half_volume=False, max_r=None,
        )
        half_vol = ftu.full_volume_to_half_volume(vol, volume_shape).ravel()
        result_half = slicing.slice_volume(
            half_vol, rots, image_shape, volume_shape, "cubic",
            half_volume=True, max_r=None,
        )

        np.testing.assert_allclose(
            np.asarray(result_half), np.asarray(result_full),
            atol=1e-10, rtol=1e-10,
        )
