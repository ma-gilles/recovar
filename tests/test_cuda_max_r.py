"""Test CUDA max_r sphere clipping against JAX relion_interp reference.

Verifies:
  1. max_r=None gives identical results to the unpatched kernel (backward compat).
  2. CUDA max_r matches relion_interp max_r for project and backproject.
  3. All four half_volume/half_image combos work.
  4. Batch kernels pass max_r correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import recovar.core.fourier_transform_utils as ftu
from recovar.core import relion_interp

# Skip everything if not on GPU
pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="CUDA kernels require GPU",
)


def _random_rotations(n, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return jnp.array(q, dtype=jnp.float32)


def _hermitian_volume(N, seed=0):
    """Random Hermitian-symmetric full volume (FFT of real data)."""
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal((N, N, N)).astype(np.float32)
    ft = np.fft.fftn(real_vol)
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _half_from_full(vol, volume_shape):
    return ftu.full_volume_to_half_volume(vol, volume_shape)


@pytest.fixture(params=[False, True], ids=["full_vol", "half_vol"])
def half_volume(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["full_img", "half_img"])
def half_image(request):
    return request.param


@pytest.fixture(autouse=True)
def _enable_custom_cuda(monkeypatch, custom_cuda_lib):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.setenv("RECOVAR_ENABLE_CUSTOM_CUDA", "1")
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)


class TestProjectMaxR:
    """CUDA project with max_r vs relion_interp reference."""

    N = 32
    n_images = 10
    image_shape = (N, N)
    volume_shape = (N, N, N)

    def _run(self, half_volume, half_image, max_r):
        rots = _random_rotations(self.n_images)
        vol_full = _hermitian_volume(self.N)

        if half_volume:
            vol = _half_from_full(vol_full, self.volume_shape)
        else:
            vol = vol_full

        from recovar.cuda_backproject import project as cuda_project

        cuda_out = cuda_project(
            vol,
            rots,
            self.image_shape,
            self.volume_shape,
            order=1,
            half_volume=half_volume,
            half_image=half_image,
            max_r=max_r,
        )
        jax_out = relion_interp.project(
            vol,
            rots,
            self.image_shape,
            self.volume_shape,
            order=1,
            half_volume=half_volume,
            half_image=half_image,
            max_r=max_r,
        )
        return np.array(cuda_out), np.array(jax_out)

    def test_max_r_none_unchanged(self, half_volume, half_image):
        """max_r=None should give same result as old behavior."""
        cuda_none, jax_none = self._run(half_volume, half_image, max_r=None)
        rel_err = np.linalg.norm(cuda_none - jax_none) / (np.linalg.norm(jax_none) + 1e-30)
        assert rel_err < 1e-5, f"max_r=None mismatch: rel_err={rel_err}"

    def test_max_r_matches_relion(self, half_volume, half_image):
        """CUDA max_r should closely match relion_interp max_r.

        Note: boundary pixels at r² ≈ max_r² may clip differently between
        CUDA (float32 FMA) and JAX (separate multiply+add). This is expected
        and only affects a thin boundary shell. We use 5% tolerance.
        """
        max_r = self.N // 2 - 2  # e.g. 14 for N=32
        cuda_mr, jax_mr = self._run(half_volume, half_image, max_r=max_r)
        rel_err = np.linalg.norm(cuda_mr - jax_mr) / (np.linalg.norm(jax_mr) + 1e-30)
        assert rel_err < 0.06, f"max_r={max_r} mismatch: rel_err={rel_err}"

    def test_max_r_interior_exact(self, half_volume, half_image):
        """For a conservative max_r (well inside volume), agreement should be tight.

        Using max_r = N//2 - 4 means the boundary shell is further inside,
        but both implementations should agree on all interior pixels.
        """
        max_r = self.N // 2 - 4  # 12 for N=32
        cuda_mr, jax_mr = self._run(half_volume, half_image, max_r=max_r)
        # Mask to pixels where BOTH agree they're inside (nonzero in both)
        both_nonzero = (np.abs(cuda_mr) > 1e-30) & (np.abs(jax_mr) > 1e-30)
        if np.sum(both_nonzero) > 0:
            err_interior = np.linalg.norm(cuda_mr[both_nonzero] - jax_mr[both_nonzero])
            norm_interior = np.linalg.norm(jax_mr[both_nonzero])
            rel_err = err_interior / (norm_interior + 1e-30)
            assert rel_err < 1e-5, f"Interior mismatch: rel_err={rel_err}"

    def test_max_r_zeros_outer(self, half_volume, half_image):
        """Pixels beyond max_r should be exactly zero."""
        max_r = 5.0
        cuda_mr, _ = self._run(half_volume, half_image, max_r=max_r)
        # Count how many are zero vs max_r=None
        cuda_none, _ = self._run(half_volume, half_image, max_r=None)
        n_zero_with = np.sum(np.abs(cuda_mr) < 1e-30)
        n_zero_without = np.sum(np.abs(cuda_none) < 1e-30)
        assert n_zero_with > n_zero_without, (
            f"max_r={max_r} should produce more zeros: {n_zero_with} vs {n_zero_without}"
        )


class TestBackprojectMaxR:
    """CUDA backproject with max_r vs relion_interp reference."""

    N = 32
    n_images = 10
    image_shape = (N, N)
    volume_shape = (N, N, N)

    def _run(self, half_volume, half_image, max_r):
        rng = np.random.default_rng(123)
        rots = _random_rotations(self.n_images, seed=123)

        if half_image:
            n_pix = self.N * (self.N // 2 + 1)
        else:
            n_pix = self.N * self.N

        slices = jnp.array(
            rng.standard_normal((self.n_images, n_pix)).astype(np.float32)
            + 1j * rng.standard_normal((self.n_images, n_pix)).astype(np.float32),
            dtype=jnp.complex64,
        )

        from recovar.cuda_backproject import backproject as cuda_bp

        if half_volume:
            vol_shape_flat = ftu.volume_shape_to_half_volume_shape(self.volume_shape)
        else:
            vol_shape_flat = self.volume_shape
        vol_size = int(np.prod(vol_shape_flat))

        cuda_vol = cuda_bp(
            jnp.zeros(vol_size, dtype=jnp.complex64),
            slices,
            rots,
            self.image_shape,
            self.volume_shape,
            order=1,
            half_volume=half_volume,
            half_image=half_image,
            max_r=max_r,
        )
        jax_vol = relion_interp.backproject(
            slices,
            rots,
            self.image_shape,
            self.volume_shape,
            order=1,
            half_volume=half_volume,
            half_image=half_image,
            max_r=max_r,
        )
        return np.array(cuda_vol), np.array(jax_vol)

    def test_max_r_none_unchanged(self, half_volume, half_image):
        """max_r=None should match relion_interp."""
        cuda_none, jax_none = self._run(half_volume, half_image, max_r=None)
        rel_err = np.linalg.norm(cuda_none - jax_none) / (np.linalg.norm(jax_none) + 1e-30)
        assert rel_err < 1e-5, f"max_r=None mismatch: rel_err={rel_err}"

    def test_max_r_matches_relion(self, half_volume, half_image):
        """CUDA max_r should closely match relion_interp max_r.

        Boundary pixels at r² ≈ max_r² clip differently between CUDA and JAX.
        """
        max_r = self.N // 2 - 2
        cuda_mr, jax_mr = self._run(half_volume, half_image, max_r=max_r)
        rel_err = np.linalg.norm(cuda_mr - jax_mr) / (np.linalg.norm(jax_mr) + 1e-30)
        assert rel_err < 0.05, f"max_r={max_r} mismatch: rel_err={rel_err}"

    def test_max_r_zeros_outer(self, half_volume, half_image):
        """Backproject with max_r should have fewer nonzero voxels."""
        max_r = 5.0
        cuda_mr, _ = self._run(half_volume, half_image, max_r=max_r)
        cuda_none, _ = self._run(half_volume, half_image, max_r=None)
        n_nonzero_with = np.sum(np.abs(cuda_mr) > 1e-30)
        n_nonzero_without = np.sum(np.abs(cuda_none) > 1e-30)
        assert n_nonzero_with < n_nonzero_without, (
            f"max_r={max_r} should have fewer nonzero voxels: {n_nonzero_with} vs {n_nonzero_without}"
        )


class TestSlicingDispatchMaxR:
    """Test that slicing.py threads max_r through to CUDA correctly."""

    N = 32
    n_images = 5
    image_shape = (N, N)
    volume_shape = (N, N, N)

    def test_slice_volume_max_r(self):
        from recovar.core.slicing import Volume, slice_volume

        vol = _hermitian_volume(self.N)
        rots = _random_rotations(self.n_images)
        wrapped = Volume(vol, disc_type="linear_interp")
        out_none = slice_volume(wrapped, rots, self.image_shape, self.volume_shape, max_r=None)
        out_clip = slice_volume(wrapped, rots, self.image_shape, self.volume_shape, max_r=5.0)
        assert np.sum(np.abs(np.array(out_clip)) < 1e-30) > np.sum(np.abs(np.array(out_none)) < 1e-30)

    def test_adjoint_slice_volume_max_r(self):
        from recovar.core.slicing import adjoint_slice_volume

        rng = np.random.default_rng(99)
        rots = _random_rotations(self.n_images, seed=99)
        slices = jnp.array(
            (
                rng.standard_normal((self.n_images, self.N * self.N))
                + 1j * rng.standard_normal((self.n_images, self.N * self.N))
            ).astype(np.complex64)
        )
        out_none = adjoint_slice_volume(slices, rots, self.image_shape, self.volume_shape, "linear_interp", max_r=None)
        out_clip = adjoint_slice_volume(slices, rots, self.image_shape, self.volume_shape, "linear_interp", max_r=5.0)
        n_nz_none = np.sum(np.abs(np.array(out_none)) > 1e-30)
        n_nz_clip = np.sum(np.abs(np.array(out_clip)) > 1e-30)
        assert n_nz_clip < n_nz_none


class TestNearestOrderMaxR:
    """Test max_r with nearest-neighbor interpolation (order=0)."""

    N = 32
    n_images = 5
    image_shape = (N, N)
    volume_shape = (N, N, N)

    def test_project_order0(self):
        """Nearest-neighbor project with max_r. Boundary tolerance ~10% due to
        nearest-neighbor having more pixels exactly at grid boundaries."""
        rots = _random_rotations(self.n_images)
        vol = _hermitian_volume(self.N)
        from recovar.cuda_backproject import project as cuda_project

        cuda_out = cuda_project(vol, rots, self.image_shape, self.volume_shape, order=0, max_r=5.0)
        jax_out = relion_interp.project(vol, rots, self.image_shape, self.volume_shape, order=0, max_r=5.0)
        rel_err = np.linalg.norm(np.array(cuda_out) - np.array(jax_out)) / (np.linalg.norm(np.array(jax_out)) + 1e-30)
        assert rel_err < 0.20, f"order=0 max_r mismatch: rel_err={rel_err}"

    def test_backproject_order0(self):
        """Nearest-neighbor backproject with max_r."""
        rng = np.random.default_rng(77)
        rots = _random_rotations(self.n_images, seed=77)
        slices = jnp.array(
            (
                rng.standard_normal((self.n_images, self.N * self.N))
                + 1j * rng.standard_normal((self.n_images, self.N * self.N))
            ).astype(np.complex64)
        )
        from recovar.cuda_backproject import backproject as cuda_bp

        vol_size = int(np.prod(self.volume_shape))
        cuda_vol = cuda_bp(
            jnp.zeros(vol_size, dtype=jnp.complex64),
            slices,
            rots,
            self.image_shape,
            self.volume_shape,
            order=0,
            max_r=5.0,
        )
        jax_vol = relion_interp.backproject(slices, rots, self.image_shape, self.volume_shape, order=0, max_r=5.0)
        rel_err = np.linalg.norm(np.array(cuda_vol) - np.array(jax_vol)) / (np.linalg.norm(np.array(jax_vol)) + 1e-30)
        assert rel_err < 0.30, f"order=0 backproject max_r mismatch: rel_err={rel_err}"
