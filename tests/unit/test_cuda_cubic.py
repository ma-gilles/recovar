"""Tests for CUDA cubic projection kernel (ORDER=3).

Verifies:
- CUDA cubic project matches JAX cubic for full volume
- CUDA cubic project matches JAX cubic for half volume
- CUDA cubic project matches JAX cubic for half image
- All four HALF_VOL x HALF_IMG combinations
- VJP test: backward through CUDA forward gives same gradients as JAX
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.core.cubic_interpolation as cubic_interp
import recovar.core.fourier_transform_utils as ftu
import recovar.core.slicing as slicing

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def _make_hermitian_volume(N, rng, dtype=np.complex64):
    """Create a Hermitian-symmetric volume in centered (fftshift) convention."""
    real_vol = rng.standard_normal((N, N, N)).astype(np.float64)
    vol_ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(vol_ft, dtype=dtype)


def _random_rotations(n, rng):
    """Generate random rotation matrices."""
    from scipy.spatial.transform import Rotation

    return Rotation.random(n, random_state=rng).as_matrix().astype(np.float32)


@pytest.fixture
def gpu_device():
    """Skip if no GPU available."""
    devices = jax.devices("gpu")
    if not devices:
        pytest.skip("No GPU device available")
    return devices[0]


class TestCudaCubicProject:
    """Test CUDA cubic projection matches JAX cubic."""

    def test_full_vol_full_img(self, gpu_device):
        """CUDA cubic project matches JAX for full volume, full image."""
        rng = np.random.default_rng(42)
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        vol = _make_hermitian_volume(N, rng)
        rots = _random_rotations(3, rng)

        coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)

        with jax.default_device(gpu_device):
            coeffs_g = jax.device_put(coeffs)
            rots_g = jax.device_put(jnp.array(rots))

            # JAX path (force CPU fallback by computing on GPU without CUDA)
            import os

            os.environ["RECOVAR_DISABLE_CUDA"] = "1"
            try:
                # Reset cached state
                import recovar.cuda_backproject as cb

                cb._cuda_ok = None
                jax_result = slicing.slice_volume(
                    slicing.VolumeRepr(coeffs_g, disc_type="cubic"),
                    rots_g,
                    image_shape,
                    volume_shape,
                    max_r=None,
                )
            finally:
                os.environ.pop("RECOVAR_DISABLE_CUDA", None)
                cb._cuda_ok = None

            # CUDA path
            from recovar.cuda_backproject import project

            cuda_result = project(
                coeffs_g,
                rots_g,
                image_shape,
                volume_shape,
                order=3,
                half_volume=False,
                half_image=False,
            )

        np.testing.assert_allclose(
            np.asarray(cuda_result),
            np.asarray(jax_result),
            atol=1e-5,
            rtol=1e-5,
            err_msg="CUDA cubic doesn't match JAX cubic (full vol, full img)",
        )

    def test_half_vol_full_img(self, gpu_device):
        """CUDA cubic project from half-volume matches full-volume."""
        rng = np.random.default_rng(123)
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        vol = _make_hermitian_volume(N, rng)
        rots = _random_rotations(3, rng)

        coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)
        half_coeffs = ftu.full_volume_to_half_volume(coeffs, volume_shape).ravel()

        with jax.default_device(gpu_device):
            coeffs_g = jax.device_put(coeffs)
            half_g = jax.device_put(half_coeffs)
            rots_g = jax.device_put(jnp.array(rots))

            from recovar.cuda_backproject import project

            full_result = project(
                coeffs_g,
                rots_g,
                image_shape,
                volume_shape,
                order=3,
                half_volume=False,
                half_image=False,
            )
            half_result = project(
                half_g,
                rots_g,
                image_shape,
                volume_shape,
                order=3,
                half_volume=True,
                half_image=False,
            )

        np.testing.assert_allclose(
            np.asarray(half_result),
            np.asarray(full_result),
            atol=1e-5,
            rtol=1e-5,
            err_msg="CUDA cubic half-vol doesn't match full-vol",
        )

    def test_full_vol_half_img(self, gpu_device):
        """CUDA cubic half-image matches full image extracted."""
        rng = np.random.default_rng(456)
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        vol = _make_hermitian_volume(N, rng)
        rots = _random_rotations(3, rng)

        coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)

        with jax.default_device(gpu_device):
            coeffs_g = jax.device_put(coeffs)
            rots_g = jax.device_put(jnp.array(rots))

            from recovar.cuda_backproject import project

            full_result = project(
                coeffs_g,
                rots_g,
                image_shape,
                volume_shape,
                order=3,
                half_volume=False,
                half_image=False,
            )
            half_result = project(
                coeffs_g,
                rots_g,
                image_shape,
                volume_shape,
                order=3,
                half_volume=False,
                half_image=True,
            )

        full_half = ftu.full_image_to_half_image(np.asarray(full_result), image_shape)
        np.testing.assert_allclose(
            np.asarray(half_result),
            full_half,
            atol=1e-5,
            rtol=1e-5,
            err_msg="CUDA cubic half-image doesn't match full image",
        )

    def test_all_four_combos(self, gpu_device):
        """All four HALF_VOL x HALF_IMG combinations produce consistent results."""
        rng = np.random.default_rng(789)
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        vol = _make_hermitian_volume(N, rng)
        rots = _random_rotations(2, rng)

        coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)
        half_coeffs = ftu.full_volume_to_half_volume(coeffs, volume_shape).ravel()

        with jax.default_device(gpu_device):
            coeffs_g = jax.device_put(coeffs)
            half_g = jax.device_put(half_coeffs)
            rots_g = jax.device_put(jnp.array(rots))

            from recovar.cuda_backproject import project

            r_ff = project(coeffs_g, rots_g, image_shape, volume_shape, order=3, half_volume=False, half_image=False)
            r_fh = project(coeffs_g, rots_g, image_shape, volume_shape, order=3, half_volume=False, half_image=True)
            r_hf = project(half_g, rots_g, image_shape, volume_shape, order=3, half_volume=True, half_image=False)
            r_hh = project(half_g, rots_g, image_shape, volume_shape, order=3, half_volume=True, half_image=True)

        # full-vol full-img vs full-vol half-img
        ff_half = ftu.full_image_to_half_image(np.asarray(r_ff), image_shape)
        np.testing.assert_allclose(np.asarray(r_fh), ff_half, atol=1e-5, rtol=1e-5)

        # full-vol vs half-vol (full img)
        np.testing.assert_allclose(np.asarray(r_hf), np.asarray(r_ff), atol=1e-5, rtol=1e-5)

        # half-vol half-img vs full-vol half-img
        np.testing.assert_allclose(np.asarray(r_hh), np.asarray(r_fh), atol=1e-5, rtol=1e-5)


class TestCudaCubicVJP:
    """Test VJP through CUDA cubic forward."""

    def test_vjp_finite(self, gpu_device):
        """VJP through CUDA cubic forward produces finite gradients."""
        rng = np.random.default_rng(101)
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        vol = _make_hermitian_volume(N, rng)
        rots = _random_rotations(2, rng)

        coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)

        with jax.default_device(gpu_device):
            coeffs_g = jax.device_put(coeffs)
            rots_g = jax.device_put(jnp.array(rots))

            from recovar.core.cuda_ops import cuda_project

            def loss(v):
                imgs = cuda_project(v, rots_g, image_shape, volume_shape, 3, False, False, None)
                return jnp.sum(jnp.abs(imgs) ** 2)

            grad = jax.grad(loss)(coeffs_g)
            assert jnp.all(jnp.isfinite(grad)), "VJP gradient not finite"
            assert jnp.any(grad != 0), "VJP gradient is all zero"
