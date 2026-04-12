"""Tests for RELION-style interpolation module.

Verifies that the RELION-style projection/backprojection produces results
consistent with the existing recovar implementation, and that the forward
and adjoint are consistent (dot-product test).

NOTE: The comparison against slicing.slice_volume uses the JAX fallback path
(not CUDA) because relion_interp uses JAX's meshgrid("xy") pixel ordering
while the CUDA kernel uses a different centered row-major ordering.  Both
produce mathematically equivalent results but in different pixel orders.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from recovar.core import fourier_transform_utils as ftu
from recovar.core import relion_interp, slicing

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _random_rotation(key):
    """Generate a random 3×3 rotation matrix via QR decomposition."""
    mat = random.normal(key, (3, 3))
    q, r = jnp.linalg.qr(mat)
    # Ensure proper rotation (det = +1)
    q = q * jnp.sign(jnp.linalg.det(q))
    return q


def _random_volume(key, shape, complex=True):
    """Generate a random complex volume."""
    if complex:
        re = random.normal(key, shape)
        im = random.normal(random.fold_in(key, 1), shape)
        return (re + 1j * im).astype(jnp.complex64)
    return random.normal(key, shape).astype(jnp.float32)


def _make_hermitian_volume(vol, volume_shape):
    """Make a volume satisfy Hermitian symmetry (so half-vol ↔ full-vol is consistent)."""
    N0, N1, N2 = volume_shape
    vol = vol.reshape(volume_shape)
    # Make it by: full = (vol + hermitian_conj(vol)) / 2
    i0 = (N0 - (N0 % 2) - jnp.arange(N0)) % N0
    i1 = (N1 - (N1 % 2) - jnp.arange(N1)) % N1
    i2 = (N2 - (N2 % 2) - jnp.arange(N2)) % N2
    partner = jnp.conj(vol[i0][:, i1][:, :, i2])
    return ((vol + partner) / 2).ravel()


def _force_jax_path():
    """Context manager to force slicing to use JAX fallback (same pixel ordering)."""
    return patch.object(slicing, "_on_gpu", return_value=False)


def _slice_volume(volume, rotation_matrices, image_shape, volume_shape, disc_type, half_volume=False, **kwargs):
    wrapped = volume
    if not isinstance(wrapped, (slicing.Volume, slicing.CubicVolume)):
        if disc_type == "cubic":
            wrapped = slicing.to_cubic(wrapped, volume_shape, half_volume=half_volume)
        else:
            wrapped = slicing.Volume(wrapped, disc_type=disc_type, half_volume=half_volume)
    return slicing.slice_volume(wrapped, rotation_matrices, image_shape, volume_shape, **kwargs)


# ---------------------------------------------------------------------------
# Test: projection agrees with existing implementation
# ---------------------------------------------------------------------------


class TestProjectionAgreement:
    """Verify RELION-style projection matches existing recovar projection."""

    @pytest.mark.parametrize("N", [16, 32])
    @pytest.mark.parametrize("order", [0, 1])
    def test_full_vol_full_img(self, N, order):
        """Full volume, full image — should agree closely."""
        key = random.PRNGKey(42)
        volume_shape = (N, N, N)
        image_shape = (N, N)
        disc_type = "nearest" if order == 0 else "linear_interp"

        k1, k2, k3 = random.split(key, 3)
        vol = _random_volume(k1, (N * N * N,))
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        with _force_jax_path():
            existing = _slice_volume(
                vol,
                rots,
                image_shape,
                volume_shape,
                disc_type,
                half_volume=False,
                half_image=False,
                max_r=None,
            )

        relion = relion_interp.project(
            vol,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=False,
            half_image=False,
        )

        np.testing.assert_allclose(
            np.array(relion),
            np.array(existing),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Full-vol full-img order={order} N={N}",
        )

    @pytest.mark.parametrize("N", [16, 32])
    @pytest.mark.parametrize("order", [0, 1])
    def test_half_vol_full_img(self, N, order):
        """Half volume, full image."""
        key = random.PRNGKey(123)
        volume_shape = (N, N, N)
        image_shape = (N, N)
        disc_type = "nearest" if order == 0 else "linear_interp"

        k1, k2 = random.split(key)
        vol_full = _make_hermitian_volume(_random_volume(k1, (N * N * N,)), volume_shape)
        vol_half = ftu.full_volume_to_half_volume(vol_full, volume_shape)
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        with _force_jax_path():
            existing = _slice_volume(
                vol_half,
                rots,
                image_shape,
                volume_shape,
                disc_type,
                half_volume=True,
                half_image=False,
                max_r=None,
            )

        relion = relion_interp.project(
            vol_half,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=True,
            half_image=False,
        )

        np.testing.assert_allclose(
            np.array(relion),
            np.array(existing),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Half-vol full-img order={order} N={N}",
        )

    @pytest.mark.parametrize("N", [16, 32])
    @pytest.mark.parametrize("order", [0, 1])
    def test_full_vol_half_img(self, N, order):
        """Full volume, half image."""
        key = random.PRNGKey(456)
        volume_shape = (N, N, N)
        image_shape = (N, N)
        disc_type = "nearest" if order == 0 else "linear_interp"

        k1, k2 = random.split(key)
        vol = _random_volume(k1, (N * N * N,))
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        with _force_jax_path():
            existing = _slice_volume(
                vol,
                rots,
                image_shape,
                volume_shape,
                disc_type,
                half_volume=False,
                half_image=True,
                max_r=None,
            )

        relion = relion_interp.project(
            vol,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=False,
            half_image=True,
        )

        np.testing.assert_allclose(
            np.array(relion),
            np.array(existing),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Full-vol half-img order={order} N={N}",
        )

    @pytest.mark.parametrize("N", [16, 32])
    @pytest.mark.parametrize("order", [0, 1])
    def test_half_vol_half_img(self, N, order):
        """Half volume, half image."""
        key = random.PRNGKey(789)
        volume_shape = (N, N, N)
        image_shape = (N, N)
        disc_type = "nearest" if order == 0 else "linear_interp"

        k1, k2 = random.split(key)
        vol_full = _make_hermitian_volume(_random_volume(k1, (N * N * N,)), volume_shape)
        vol_half = ftu.full_volume_to_half_volume(vol_full, volume_shape)
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        with _force_jax_path():
            existing = _slice_volume(
                vol_half,
                rots,
                image_shape,
                volume_shape,
                disc_type,
                half_volume=True,
                half_image=True,
                max_r=None,
            )

        relion = relion_interp.project(
            vol_half,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=True,
            half_image=True,
        )

        np.testing.assert_allclose(
            np.array(relion),
            np.array(existing),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Half-vol half-img order={order} N={N}",
        )


# ---------------------------------------------------------------------------
# Test: backprojection agrees with existing
# ---------------------------------------------------------------------------


class TestBackprojectionAgreement:
    """Verify RELION-style backprojection matches existing."""

    @pytest.mark.parametrize("N", [16])
    @pytest.mark.parametrize("order", [0, 1])
    def test_full_vol_full_img(self, N, order):
        key = random.PRNGKey(111)
        volume_shape = (N, N, N)
        image_shape = (N, N)
        disc_type = "nearest" if order == 0 else "linear_interp"

        k1, k2 = random.split(key)
        imgs = _random_volume(k1, (3, N * N))
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        with _force_jax_path():
            existing = slicing.adjoint_slice_volume(
                imgs,
                rots,
                image_shape,
                volume_shape,
                disc_type,
                half_image=False,
                half_volume=False,
                max_r=None,
            )

        relion = relion_interp.backproject(
            imgs,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=False,
            half_image=False,
        )

        np.testing.assert_allclose(
            np.array(relion),
            np.array(existing),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"BP full-vol full-img order={order} N={N}",
        )

    @pytest.mark.parametrize("N", [16])
    @pytest.mark.parametrize("order", [0, 1])
    def test_half_vol_full_img(self, N, order):
        key = random.PRNGKey(222)
        volume_shape = (N, N, N)
        image_shape = (N, N)
        disc_type = "nearest" if order == 0 else "linear_interp"

        k1, k2 = random.split(key)
        imgs = _random_volume(k1, (3, N * N))
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        with _force_jax_path():
            existing = slicing.adjoint_slice_volume(
                imgs,
                rots,
                image_shape,
                volume_shape,
                disc_type,
                half_image=False,
                half_volume=True,
                max_r=None,
            )

        relion = relion_interp.backproject(
            imgs,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=True,
            half_image=False,
        )

        np.testing.assert_allclose(
            np.array(relion),
            np.array(existing),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"BP half-vol full-img order={order} N={N}",
        )


# ---------------------------------------------------------------------------
# Test: adjoint consistency (dot-product test)
# ---------------------------------------------------------------------------


class TestAdjointConsistency:
    """Verify ⟨Ax, y⟩ = ⟨x, A†y⟩ for the RELION-style forward/adjoint pair."""

    @pytest.mark.parametrize("N", [16])
    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("half_volume", [False, True])
    def test_dot_product(self, N, order, half_volume):
        key = random.PRNGKey(333)
        volume_shape = (N, N, N)
        image_shape = (N, N)
        half_image = False

        k1, k2, k3, k4 = random.split(key, 4)
        rots = jnp.stack([_random_rotation(random.fold_in(k1, i)) for i in range(2)])

        if half_volume:
            vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        else:
            vol_shape = volume_shape
        n_voxels = int(np.prod(vol_shape))
        n_pixels = N * N

        x = _random_volume(k2, (n_voxels,))
        if half_volume:
            x_full = ftu.half_volume_to_full_volume(x, volume_shape)
            x_hermitian = ftu.full_volume_to_half_volume(_make_hermitian_volume(x_full, volume_shape), volume_shape)
            x = x_hermitian
        y = _random_volume(k3, (2, n_pixels))

        # Forward: Ax
        Ax = relion_interp.project(
            x,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
        )

        # Adjoint: A†y
        Ady = relion_interp.backproject(
            y,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
        )

        # ⟨Ax, y⟩
        lhs = jnp.sum(jnp.conj(Ax) * y).real

        # ⟨x, A†y⟩
        rhs = jnp.sum(jnp.conj(x) * Ady).real

        # These should match to reasonable tolerance
        # (explicit scatter has some order-of-operations differences from VJP)
        rel_err = abs(float(lhs) - float(rhs)) / max(abs(float(lhs)), abs(float(rhs)), 1e-10)
        assert rel_err < 0.05, (
            f"Adjoint test failed: ⟨Ax,y⟩={float(lhs):.6f}, ⟨x,A†y⟩={float(rhs):.6f}, "
            f"rel_err={rel_err:.2e} (order={order}, half_vol={half_volume})"
        )


# ---------------------------------------------------------------------------
# Test: max_r clipping
# ---------------------------------------------------------------------------


class TestMaxRClipping:
    """Verify that max_r clips high-frequency pixels to zero."""

    def test_small_max_r_zeros_high_freq(self):
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        key = random.PRNGKey(444)

        k1, k2 = random.split(key)
        vol = _random_volume(k1, (N * N * N,))
        rots = jnp.eye(3)[None, :, :]  # identity rotation

        # With very small max_r, most pixels should be zero
        result = relion_interp.project(
            vol,
            rots,
            image_shape,
            volume_shape,
            order=1,
            half_volume=False,
            half_image=False,
            max_r=2.0,
        )

        n_nonzero = jnp.count_nonzero(jnp.abs(result) > 1e-10)
        n_total = result.size
        # Only a small sphere (r <= 2) should be nonzero
        assert n_nonzero < n_total * 0.1, f"Expected < 10% nonzero with max_r=2, got {n_nonzero}/{n_total}"

    def test_no_max_r_all_nonzero(self):
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)
        key = random.PRNGKey(555)

        k1, k2 = random.split(key)
        vol = _random_volume(k1, (N * N * N,))
        # Use identity rotation — all pixels map to z=0 plane
        rots = jnp.eye(3)[None, :, :]

        result = relion_interp.project(
            vol,
            rots,
            image_shape,
            volume_shape,
            order=0,
            half_volume=False,
            half_image=False,
            max_r=None,
        )

        # Most interior pixels should be nonzero
        n_nonzero = jnp.count_nonzero(jnp.abs(result) > 1e-10)
        assert n_nonzero > result.size * 0.5


# ---------------------------------------------------------------------------
# Test: identity rotation roundtrip
# ---------------------------------------------------------------------------


class TestIdentityRotation:
    """With identity rotation, projection extracts the z=0 central slice."""

    def test_identity_nearest(self):
        N = 16
        volume_shape = (N, N, N)
        image_shape = (N, N)

        key = random.PRNGKey(666)
        vol = _random_volume(key, volume_shape)

        rots = jnp.eye(3)[None, :, :]
        result = relion_interp.project(
            vol.ravel(),
            rots,
            image_shape,
            volume_shape,
            order=0,
            half_volume=False,
            half_image=False,
        )

        # Compare with existing implementation using JAX path
        with _force_jax_path():
            existing = _slice_volume(
                vol.ravel(),
                rots,
                image_shape,
                volume_shape,
                "nearest",
                half_volume=False,
                half_image=False,
                max_r=None,
            )

        np.testing.assert_allclose(
            np.array(result),
            np.array(existing),
            rtol=1e-5,
            atol=1e-6,
            err_msg="Identity rotation nearest should match existing code",
        )


# ---------------------------------------------------------------------------
# Test: CUDA matches relion_interp (and JAX fallback) — verifies dtype fix
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test: backprojection with half_image input
# ---------------------------------------------------------------------------


def _make_hermitian_images(imgs_random, image_shape):
    """Make images Hermitian-symmetric (as would come from FFT of real-space images)."""
    H, W = image_shape
    imgs = imgs_random.reshape(-1, H, W)
    # Hermitian: img[k0, k1] = conj(img[-k0, -k1])
    i0 = (H - (H % 2) - jnp.arange(H)) % H
    i1 = (W - (W % 2) - jnp.arange(W)) % W
    partner = jnp.conj(imgs[:, i0][:, :, i1])
    return ((imgs + partner) / 2).reshape(imgs_random.shape)


class TestHalfImageBackprojection:
    """Verify half-image backprojection matches full-image backprojection.

    Half-image backprojection is only equivalent to full-image when images are
    Hermitian-symmetric (as from FFT of real-space data).
    """

    @pytest.mark.parametrize("N", [16])
    @pytest.mark.parametrize("order", [0, 1])
    def test_half_img_full_vol(self, N, order):
        """half_image backproject into full volume should match full-image backproject."""
        key = random.PRNGKey(777)
        volume_shape = (N, N, N)
        image_shape = (N, N)

        k1, k2 = random.split(key)
        # Hermitian-symmetric images (as from FFT of real data)
        imgs_full = _make_hermitian_images(_random_volume(k1, (3, N * N)), image_shape)
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        # Reference: backproject from full images
        bp_full = relion_interp.backproject(
            imgs_full,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=False,
            half_image=False,
        )

        # Extract half-images (rfft convention)
        imgs_half = ftu.full_image_to_half_image(imgs_full, image_shape)

        # Backproject from half images
        bp_half = relion_interp.backproject(
            imgs_half,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=False,
            half_image=True,
        )

        np.testing.assert_allclose(
            np.array(bp_half),
            np.array(bp_full),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"half-img vs full-img BP into full vol, order={order}",
        )

    @pytest.mark.parametrize("N", [16])
    @pytest.mark.parametrize("order", [0, 1])
    def test_half_img_half_vol(self, N, order):
        """half_image backproject into half volume should match full-image into half volume."""
        key = random.PRNGKey(888)
        volume_shape = (N, N, N)
        image_shape = (N, N)

        k1, k2 = random.split(key)
        imgs_full = _make_hermitian_images(_random_volume(k1, (3, N * N)), image_shape)
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        # Reference: backproject from full images into half volume
        bp_full = relion_interp.backproject(
            imgs_full,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=True,
            half_image=False,
        )

        # Extract half-images
        imgs_half = ftu.full_image_to_half_image(imgs_full, image_shape)

        # Backproject from half images into half volume
        bp_half = relion_interp.backproject(
            imgs_half,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=True,
            half_image=True,
        )

        np.testing.assert_allclose(
            np.array(bp_half),
            np.array(bp_full),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"half-img vs full-img BP into half vol, order={order}",
        )


# ---------------------------------------------------------------------------
# Test: adjoint with half_image
# ---------------------------------------------------------------------------


class TestAdjointHalfImage:
    """Verify adjoint consistency for the half_image backprojection path.

    The half_image backprojection is designed as the adjoint of the full-image
    forward projection restricted to Hermitian-symmetric inputs.  The correct
    adjoint identity is:

        ⟨P_full x, y_full⟩ = ⟨x, BP_half(half(y_full))⟩

    where P_full projects to full images, half() extracts the rfft half, and
    BP_half is backproject with half_image=True.
    """

    @pytest.mark.parametrize("N", [16])
    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("half_volume", [False, True])
    def test_dot_product_half_image(self, N, order, half_volume):
        key = random.PRNGKey(999)
        volume_shape = (N, N, N)
        image_shape = (N, N)

        k1, k2, k3 = random.split(key, 3)
        rots = jnp.stack([_random_rotation(random.fold_in(k1, i)) for i in range(2)])

        if half_volume:
            vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        else:
            vol_shape = volume_shape
        n_voxels = int(np.prod(vol_shape))
        n_full_pixels = N * N

        x = _random_volume(k2, (n_voxels,))
        if half_volume:
            x_full = ftu.half_volume_to_full_volume(x, volume_shape)
            x = ftu.full_volume_to_half_volume(_make_hermitian_volume(x_full, volume_shape), volume_shape)
        # Use Hermitian-symmetric images (from real-space data)
        y_full = _make_hermitian_images(_random_volume(k3, (2, n_full_pixels)), image_shape)
        y_half = ftu.full_image_to_half_image(y_full, image_shape)

        # Forward: Ax (full image output)
        Ax = relion_interp.project(
            x,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=False,
        )

        # Adjoint: A†y via half_image backprojection
        Ady = relion_interp.backproject(
            y_half,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=True,
        )

        lhs = jnp.sum(jnp.conj(Ax) * y_full).real
        rhs = jnp.sum(jnp.conj(x) * Ady).real

        rel_err = abs(float(lhs) - float(rhs)) / max(abs(float(lhs)), abs(float(rhs)), 1e-10)
        assert rel_err < 0.05, (
            f"Half-image adjoint test failed: ⟨Ax,y⟩={float(lhs):.6f}, ⟨x,A†y⟩={float(rhs):.6f}, "
            f"rel_err={rel_err:.2e} (order={order}, half_vol={half_volume})"
        )


# ---------------------------------------------------------------------------
# Test: CUDA matches relion_interp (forward projection)
# ---------------------------------------------------------------------------


class TestCUDAMatch:
    """Verify CUDA projection matches relion_interp (requires GPU)."""

    @pytest.fixture(autouse=True)
    def _enable_custom_cuda(self, monkeypatch, custom_cuda_lib):
        import recovar.cuda_backproject as cuda_backproject

        monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
        monkeypatch.setenv("RECOVAR_ENABLE_CUSTOM_CUDA", "1")
        monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
        monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    @pytest.mark.parametrize("N", [16])
    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("half_volume", [False, True])
    @pytest.mark.parametrize("half_image", [False, True])
    def test_cuda_project_vs_relion(self, N, order, half_volume, half_image):
        """CUDA and relion_interp forward projection should agree to FP precision."""
        if jax.default_backend() != "gpu":
            pytest.skip("No GPU available")

        key = random.PRNGKey(42)
        volume_shape = (N, N, N)
        image_shape = (N, N)
        disc_type = "nearest" if order == 0 else "linear_interp"

        k1, k2 = random.split(key)
        if half_volume:
            vol_full = _make_hermitian_volume(_random_volume(k1, (N * N * N,)), volume_shape)
            vol = ftu.full_volume_to_half_volume(vol_full, volume_shape)
        else:
            vol = _random_volume(k1, (N * N * N,))

        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        # CUDA path (through slicing which calls cuda_project)
        cuda_result = _slice_volume(
            vol,
            rots,
            image_shape,
            volume_shape,
            disc_type,
            half_volume=half_volume,
            half_image=half_image,
            max_r=None,
        )

        # relion_interp
        relion_result = relion_interp.project(
            vol,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
        )

        np.testing.assert_allclose(
            np.array(cuda_result),
            np.array(relion_result),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"CUDA vs relion project: order={order} hv={half_volume} hi={half_image} N={N}",
        )

    @pytest.mark.parametrize("N", [16])
    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("half_volume", [False, True])
    @pytest.mark.parametrize("half_image", [False, True])
    def test_cuda_backproject_vs_relion(self, N, order, half_volume, half_image):
        """CUDA and relion_interp backprojection should agree to FP precision."""
        if jax.default_backend() != "gpu":
            pytest.skip("No GPU available")

        from recovar.cuda_backproject import backproject as cuda_bp

        key = random.PRNGKey(99)
        volume_shape = (N, N, N)
        image_shape = (N, N)

        k1, k2 = random.split(key)
        # For half_image: both CUDA and JAX assume Hermitian-symmetric input,
        # so we must use Hermitian images to get consistent results.
        if half_image:
            imgs_full = _make_hermitian_images(_random_volume(k1, (3, N * N)), image_shape)
            imgs = ftu.full_image_to_half_image(imgs_full, image_shape)
        else:
            imgs = _random_volume(k1, (3, N * N))
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(3)])

        vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape) if half_volume else volume_shape
        vol_flat = int(np.prod(vol_shape))

        # CUDA backproject
        cuda_vol = jnp.zeros(vol_flat, dtype=imgs.dtype)
        cuda_result = cuda_bp(
            cuda_vol,
            imgs,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
        )

        # relion_interp backproject
        relion_result = relion_interp.backproject(
            imgs,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
        )

        np.testing.assert_allclose(
            np.array(cuda_result),
            np.array(relion_result),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"CUDA vs relion BP: order={order} hv={half_volume} hi={half_image} N={N}",
        )


# ---------------------------------------------------------------------------
# Test: larger volume size
# ---------------------------------------------------------------------------


class TestLargerVolume:
    """Stress test with N=64 to catch boundary / off-by-one issues."""

    @pytest.mark.parametrize("order", [0, 1])
    def test_project_N64(self, order):
        N = 64
        key = random.PRNGKey(1234)
        volume_shape = (N, N, N)
        image_shape = (N, N)

        k1, k2 = random.split(key)
        vol = _random_volume(k1, (N * N * N,))
        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(2)])

        with _force_jax_path():
            existing = _slice_volume(
                vol,
                rots,
                image_shape,
                volume_shape,
                "nearest" if order == 0 else "linear_interp",
                half_volume=False,
                half_image=False,
                max_r=None,
            )

        relion = relion_interp.project(
            vol,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=False,
            half_image=False,
        )

        np.testing.assert_allclose(
            np.array(relion),
            np.array(existing),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"N=64 full-vol full-img order={order}",
        )


# ---------------------------------------------------------------------------
# Test: dispatch through slicing.py matches relion_interp directly
# ---------------------------------------------------------------------------


class TestSlicingDispatch:
    """Verify slicing.py JAX fallback routes correctly through relion_interp."""

    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("half_volume", [False, True])
    @pytest.mark.parametrize("half_image", [False, True])
    def test_slicing_jax_matches_relion_interp(self, order, half_volume, half_image):
        """On CPU, slicing.py should produce identical results to relion_interp."""
        N = 16
        key = random.PRNGKey(5678)
        volume_shape = (N, N, N)
        image_shape = (N, N)
        disc_type = "nearest" if order == 0 else "linear_interp"

        k1, k2 = random.split(key)
        if half_volume:
            vol_full = _make_hermitian_volume(_random_volume(k1, (N * N * N,)), volume_shape)
            vol = ftu.full_volume_to_half_volume(vol_full, volume_shape)
        else:
            vol = _random_volume(k1, (N * N * N,))

        rots = jnp.stack([_random_rotation(random.fold_in(k2, i)) for i in range(2)])

        with _force_jax_path():
            dispatch_result = _slice_volume(
                vol,
                rots,
                image_shape,
                volume_shape,
                disc_type,
                half_volume=half_volume,
                half_image=half_image,
                max_r=None,
            )

        relion_result = relion_interp.project(
            vol,
            rots,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
        )

        np.testing.assert_allclose(
            np.array(dispatch_result),
            np.array(relion_result),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Dispatch vs direct: order={order} hv={half_volume} hi={half_image}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
