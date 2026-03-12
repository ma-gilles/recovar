"""Test that half-volume backprojection preserves Hermitian symmetry on the
kz=0 and kz=Nyquist planes.

For a volume representing the 3D DFT of a real object, the rfft3 half-volume
(kz >= 0) must satisfy on the kz=0 plane:

    vol[j0, j1, 0] == conj(vol[(N0-j0)%N0, (N1-j1)%N1, 0])

and on the kz=N2//2 (Nyquist) plane:

    vol[j0, j1, N2//2] == conj(vol[(N0-j0)%N0, (N1-j1)%N1, N2//2])

We verify this directly on the raw half-volume array — no ``irfft3`` or
``half_volume_to_full_volume`` conversions.

Tested implementations:
  - CUDA backproject (GPU, order 0 and 1)
  - JAX relion_interp.backproject (CPU fallback, order 0 and 1)

Notes on max_r and boundary clipping:
  Without max_r, high-frequency pixels can have rotated 3D positions near
  the volume boundary.  The bounds check ``g ∈ [-1, N)`` is asymmetric under
  the Hermitian map ``g → N - g`` for even N, so one of a conjugate pair may
  be clipped while the other is not.  This breaks Hermitian symmetry.
  Using ``max_r = N//2 - 1`` keeps ``|rk| ≤ max_r``, so grid positions stay
  in ``[1, N-1]`` — safely inside bounds for both the pixel and its partner.
  In production, max_r is always set, so this is the relevant test case.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation

from recovar.core import relion_interp


pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="CUDA kernels require GPU",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_rotations(n, seed=42):
    return jnp.array(
        Rotation.random(n, random_state=np.random.RandomState(seed)).as_matrix(),
        dtype=jnp.float32,
    )


def _real_signal_images(n_images, image_shape, seed=0):
    """Generate full images that are 2D DFTs of real signals, in centered convention.

    The CUDA kernel uses centered layout: pixel (H//2, W//2) = DC.
    ``np.fft.fft2`` returns DC at (0, 0), so we apply ``fftshift``.
    """
    H, W = image_shape
    rng = np.random.RandomState(seed)
    real_imgs = rng.randn(n_images, H, W).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fft2(real_imgs), axes=(-2, -1))
    return ft.reshape(n_images, -1).astype(np.complex64)


def _real_signal_half_images(n_images, image_shape, seed=0):
    """Generate half-images (rfft2) from real signals, in centered ky convention.

    The CUDA half-image kernel uses: ky centered (DC at index H//2),
    kx = 0..W//2 (rfft order). ``np.fft.rfft2`` has ky DC at index 0,
    so we ``fftshift`` only the ky axis.
    """
    H, W = image_shape
    rng = np.random.RandomState(seed)
    real_imgs = rng.randn(n_images, H, W).astype(np.float32)
    ft = np.fft.fftshift(np.fft.rfft2(real_imgs), axes=-2)
    return ft.reshape(n_images, -1).astype(np.complex64)


def _single_freq_full_image(fy, fx, image_shape, amp=1.0 + 0.5j):
    """Full image with a single real-signal frequency: F(fy,fx) = amp, F(-fy,-fx) = conj(amp).

    Uses the CUDA kernel convention: pixel (row, col) has freq
    k0 = row - H//2, k1 = col - W//2.
    """
    H, W = image_shape
    img = np.zeros((1, H, W), dtype=np.complex64)
    img[0, (fy + H // 2) % H, (fx + W // 2) % W] = amp
    img[0, (-fy + H // 2) % H, (-fx + W // 2) % W] = np.conj(amp)
    return img.reshape(1, -1)


def _single_freq_half_image(fy, fx, image_shape, amp=1.0 + 0.5j):
    """Half image with a single real-signal frequency.

    Half-image layout: row = k0_idx (0..H-1), col = k1_idx (0..W//2).
    Kernel freq: k0 = k0_idx - H//2, k1 = k1_idx.

    For boundary kx (fx=0 or fx=W//2), both (fy,fx) and (-fy,fx) must
    be placed since CONJ_MODE skips boundary pixels.
    """
    H, W = image_shape
    W_half = W // 2 + 1
    img = np.zeros((1, H, W_half), dtype=np.complex64)
    if fx < 0:
        # Store via conjugate: F(-fy,-fx) = conj(amp) at positive kx
        img[0, (-fy + H // 2) % H, -fx] = np.conj(amp)
        if -fx == 0 or -fx == W // 2:
            img[0, (fy + H // 2) % H, -fx] = amp
    else:
        img[0, (fy + H // 2) % H, fx] = amp
        if fx == 0 or fx == W // 2:
            img[0, (-fy + H // 2) % H, fx] = np.conj(amp)
    return img.reshape(1, -1)


def _hermitian_error_on_plane(half_vol_flat, volume_shape, kz_idx):
    """Compute relative Hermitian error on a fixed-kz plane of the half-volume.

    Hermitian condition: vol[j0, j1, kz] == conj(vol[(N0-j0)%N0, (N1-j1)%N1, kz])
    Returns (relative_error, max_absolute_error, n_violations at > 1e-6 threshold).
    """
    N0, N1, N2 = volume_shape
    half_N2 = N2 // 2 + 1
    vol = np.asarray(half_vol_flat).reshape(N0, N1, half_N2)
    plane = vol[:, :, kz_idx]

    # Build Hermitian partner
    partner = np.empty_like(plane)
    for j0 in range(N0):
        for j1 in range(N1):
            partner[j0, j1] = np.conj(plane[(N0 - j0) % N0, (N1 - j1) % N1])

    diff = plane - partner
    norm_plane = np.linalg.norm(plane.ravel())
    rel_err = np.linalg.norm(diff.ravel()) / norm_plane if norm_plane > 0 else 0.0
    max_abs = float(np.max(np.abs(diff)))
    n_violations = int(np.count_nonzero(np.abs(diff) > 1e-6))
    return rel_err, max_abs, n_violations


def _backproject_cuda(imgs, rots, image_shape, volume_shape, order, half_image, max_r):
    """CUDA half-volume backprojection."""
    from recovar.cuda_backproject import backproject
    N0, N1, N2 = volume_shape
    vol_size = N0 * N1 * (N2 // 2 + 1)
    vol = jnp.zeros(vol_size, dtype=jnp.complex64)
    return np.asarray(backproject(
        vol, jnp.asarray(imgs), jnp.asarray(rots),
        image_shape=image_shape, volume_shape=volume_shape,
        order=order, half_volume=True, half_image=half_image,
        max_r=max_r,
    ))


def _backproject_jax(imgs, rots, image_shape, volume_shape, order, half_image, max_r):
    """JAX relion_interp half-volume backprojection."""
    return np.asarray(relion_interp.backproject(
        jnp.asarray(imgs), jnp.asarray(rots),
        image_shape=image_shape, volume_shape=volume_shape,
        order=order, half_volume=True, half_image=half_image,
        max_r=max_r,
    ))


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------

BACKENDS = [
    pytest.param("cuda", id="cuda"),
    pytest.param("jax", id="jax"),
]

ORDERS = [
    pytest.param(0, id="nearest"),
    pytest.param(1, id="trilinear"),
]

IMAGE_MODES = [
    pytest.param(False, id="full_img"),
    pytest.param(True, id="half_img"),
]

# Float32 tolerance: trilinear weights computed with FMA can differ by ~1 ULP
HERMITIAN_RTOL = 5e-6


# ---------------------------------------------------------------------------
# Tests: single-frequency images
# ---------------------------------------------------------------------------

class TestHermitianSingleFreq:
    """Hermitian symmetry for single-frequency real-signal images.

    Single frequencies have small |k|, so they never hit the volume boundary.
    No max_r needed.
    """

    N = 16
    image_shape = (N, N)
    volume_shape = (N, N, N)

    # Non-Nyquist frequencies + one Nyquist-row frequency
    FREQS = [
        pytest.param((1, 0), id="freq_1_0"),
        pytest.param((1, 1), id="freq_1_1"),
        pytest.param((3, 2), id="freq_3_2"),
        pytest.param((2, 5), id="freq_2_5"),
        pytest.param((0, 1), id="freq_0_1_nyquist_row"),
    ]

    ROTATIONS = [
        pytest.param(np.eye(3, dtype=np.float32), id="R=I"),
        pytest.param(
            np.array([[1, 0, 0], [0, np.cos(0.1), -np.sin(0.1)],
                       [0, np.sin(0.1), np.cos(0.1)]], dtype=np.float32),
            id="Rx_0.1",
        ),
        pytest.param(
            np.array([[np.cos(0.1), 0, np.sin(0.1)], [0, 1, 0],
                       [-np.sin(0.1), 0, np.cos(0.1)]], dtype=np.float32),
            id="Ry_0.1",
        ),
        pytest.param(
            np.array([[np.cos(0.1), -np.sin(0.1), 0],
                       [np.sin(0.1), np.cos(0.1), 0], [0, 0, 1]], dtype=np.float32),
            id="Rz_0.1",
        ),
    ]

    @pytest.mark.parametrize("freq", FREQS)
    @pytest.mark.parametrize("rotation", ROTATIONS)
    @pytest.mark.parametrize("order", ORDERS)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_kz0_hermitian_full_image(self, freq, rotation, order, backend):
        fy, fx = freq
        imgs = _single_freq_full_image(fy, fx, self.image_shape)
        rots = rotation.reshape(1, 3, 3)
        bp = (_backproject_cuda if backend == "cuda" else _backproject_jax)
        vol = bp(imgs, rots, self.image_shape, self.volume_shape,
                 order=order, half_image=False, max_r=None)
        rel_err, max_abs, _ = _hermitian_error_on_plane(vol, self.volume_shape, 0)
        assert max_abs < HERMITIAN_RTOL, (
            f"kz=0 Hermitian violation: freq={freq}, max_abs={max_abs:.2e}"
        )

    @pytest.mark.parametrize("freq", FREQS)
    @pytest.mark.parametrize("rotation", ROTATIONS)
    @pytest.mark.parametrize("order", ORDERS)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_kz0_hermitian_half_image(self, freq, rotation, order, backend):
        fy, fx = freq
        imgs = _single_freq_half_image(fy, fx, self.image_shape)
        rots = rotation.reshape(1, 3, 3)
        bp = (_backproject_cuda if backend == "cuda" else _backproject_jax)
        vol = bp(imgs, rots, self.image_shape, self.volume_shape,
                 order=order, half_image=True, max_r=None)
        rel_err, max_abs, _ = _hermitian_error_on_plane(vol, self.volume_shape, 0)
        assert max_abs < HERMITIAN_RTOL, (
            f"kz=0 Hermitian violation: freq={freq}, max_abs={max_abs:.2e}"
        )


# ---------------------------------------------------------------------------
# Tests: random real-signal images (many frequencies, with max_r)
# ---------------------------------------------------------------------------

class TestHermitianRandomImages:
    """Hermitian symmetry with many random real-signal images and max_r.

    max_r = N//2 - 1 ensures all scattered positions stay within volume
    bounds for both a pixel and its Hermitian partner.  This is the
    production-relevant configuration.
    """

    N = 32
    image_shape = (N, N)
    volume_shape = (N, N, N)
    n_images = 20
    max_r = N // 2 - 1  # = 15

    @pytest.mark.parametrize("order", ORDERS)
    @pytest.mark.parametrize("half_image", IMAGE_MODES)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_kz0_hermitian(self, order, half_image, backend):
        rots = _random_rotations(self.n_images, seed=7)
        if half_image:
            imgs = _real_signal_half_images(self.n_images, self.image_shape, seed=0)
        else:
            imgs = _real_signal_images(self.n_images, self.image_shape, seed=0)
        bp = (_backproject_cuda if backend == "cuda" else _backproject_jax)
        vol = bp(imgs, rots, self.image_shape, self.volume_shape,
                 order=order, half_image=half_image, max_r=self.max_r)
        rel_err, max_abs, n_viol = _hermitian_error_on_plane(vol, self.volume_shape, 0)
        assert rel_err < 1e-4, (
            f"kz=0 rel_err={rel_err:.2e}, max_abs={max_abs:.2e}, violations={n_viol}"
        )

    @pytest.mark.parametrize("order", ORDERS)
    @pytest.mark.parametrize("half_image", IMAGE_MODES)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_kz_nyquist_hermitian(self, order, half_image, backend):
        rots = _random_rotations(self.n_images, seed=7)
        if half_image:
            imgs = _real_signal_half_images(self.n_images, self.image_shape, seed=0)
        else:
            imgs = _real_signal_images(self.n_images, self.image_shape, seed=0)
        bp = (_backproject_cuda if backend == "cuda" else _backproject_jax)
        vol = bp(imgs, rots, self.image_shape, self.volume_shape,
                 order=order, half_image=half_image, max_r=self.max_r)
        kz_nyq = self.N // 2
        rel_err, max_abs, n_viol = _hermitian_error_on_plane(vol, self.volume_shape, kz_nyq)
        assert rel_err < 1e-4, (
            f"kz=Nyquist rel_err={rel_err:.2e}, max_abs={max_abs:.2e}, violations={n_viol}"
        )


# ---------------------------------------------------------------------------
# Tests: full-image vs half-image agreement
# ---------------------------------------------------------------------------

class TestFullHalfImageAgreement:
    """Full-image and half-image backprojection should produce identical half-volumes."""

    N = 32
    image_shape = (N, N)
    volume_shape = (N, N, N)
    n_images = 10

    @pytest.mark.parametrize("order", ORDERS)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_full_vs_half_image_match(self, order, backend):
        rots = _random_rotations(self.n_images, seed=3)
        imgs_full = _real_signal_images(self.n_images, self.image_shape, seed=1)
        imgs_half = _real_signal_half_images(self.n_images, self.image_shape, seed=1)
        bp = (_backproject_cuda if backend == "cuda" else _backproject_jax)
        vol_full = bp(imgs_full, rots, self.image_shape, self.volume_shape,
                      order=order, half_image=False, max_r=None)
        vol_half = bp(imgs_half, rots, self.image_shape, self.volume_shape,
                      order=order, half_image=True, max_r=None)
        rel_err = np.linalg.norm(vol_full - vol_half) / (np.linalg.norm(vol_full) + 1e-30)
        assert rel_err < 1e-5, f"full vs half image mismatch: rel_err={rel_err:.2e}"
