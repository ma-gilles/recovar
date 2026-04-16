"""E7: Weighted noise accumulation parity — RELION vs numpy reference.

Tests that compute_weighted_noise produces exactly the same per-shell
noise as a numpy reference implementation.

Exact parity required: rel_err < 1e-12.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import compute_weighted_noise


def _make_shell_indices(ny, nx):
    """FFTW half-complex shell indices: shell = round(sqrt(ky² + kx²))."""
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    kx = np.arange(nx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    r = np.sqrt(KY**2 + KX**2)
    return np.round(r).astype(np.int32)


def _reference_noise(images, projections, ctf, pixel_weights, shell_indices, n_shells):
    """Pure numpy reference: sigma2[shell] += w * |ctf*proj - img|²."""
    n_images, ny, nx = images.shape
    sigma2 = np.zeros(n_shells)
    for i in range(n_images):
        for y in range(ny):
            for x in range(nx):
                s = shell_indices[y, x]
                if s < 0 or s >= n_shells:
                    continue
                diff = ctf[i, y, x] * projections[i, y, x] - images[i, y, x]
                sigma2[s] += pixel_weights[i, y, x] * (diff.real**2 + diff.imag**2)
    return sigma2


class TestE7Parity:
    """Exact parity between RELION binding and numpy reference."""

    def test_basic(self):
        rng = np.random.default_rng(42)
        N = 32
        n_img = 5
        nx = N // 2 + 1
        n_shells = N // 2 + 1

        images = rng.standard_normal((n_img, N, nx)) + 1j * rng.standard_normal((n_img, N, nx))
        projs = rng.standard_normal((n_img, N, nx)) + 1j * rng.standard_normal((n_img, N, nx))
        ctf = rng.uniform(0.5, 1.5, (n_img, N, nx))
        pw = rng.uniform(0, 1, (n_img, N, nx))
        shells = _make_shell_indices(N, nx)

        relion = compute_weighted_noise(images, projs, ctf, pw, shells, n_shells)
        reference = _reference_noise(images, projs, ctf, pw, shells, n_shells)

        max_diff = np.max(np.abs(relion - reference))
        scale = np.max(np.abs(reference)) + 1e-30
        rel_err = max_diff / scale
        assert rel_err < 1e-12, f"rel_err={rel_err}, max_diff={max_diff}"

    def test_single_image(self):
        rng = np.random.default_rng(99)
        N = 16
        nx = N // 2 + 1
        n_shells = N // 2 + 1

        images = rng.standard_normal((1, N, nx)) + 1j * rng.standard_normal((1, N, nx))
        projs = rng.standard_normal((1, N, nx)) + 1j * rng.standard_normal((1, N, nx))
        ctf = np.ones((1, N, nx))
        pw = np.ones((1, N, nx))
        shells = _make_shell_indices(N, nx)

        relion = compute_weighted_noise(images, projs, ctf, pw, shells, n_shells)
        reference = _reference_noise(images, projs, ctf, pw, shells, n_shells)

        max_diff = np.max(np.abs(relion - reference))
        assert max_diff < 1e-12, f"max_diff={max_diff}"

    def test_zero_ctf(self):
        """CTF=0 means proj contribution is zero; noise = |img|²."""
        rng = np.random.default_rng(77)
        N = 16
        nx = N // 2 + 1
        n_shells = N // 2 + 1

        images = rng.standard_normal((2, N, nx)) + 1j * rng.standard_normal((2, N, nx))
        projs = rng.standard_normal((2, N, nx)) + 1j * rng.standard_normal((2, N, nx))
        ctf = np.zeros((2, N, nx))
        pw = np.ones((2, N, nx))
        shells = _make_shell_indices(N, nx)

        relion = compute_weighted_noise(images, projs, ctf, pw, shells, n_shells)
        reference = _reference_noise(images, projs, ctf, pw, shells, n_shells)

        max_diff = np.max(np.abs(relion - reference))
        assert max_diff < 1e-12, f"max_diff={max_diff}"

    @pytest.mark.parametrize("N", [16, 32, 64])
    def test_various_sizes(self, N):
        rng = np.random.default_rng(55)
        n_img = 3
        nx = N // 2 + 1
        n_shells = N // 2 + 1

        images = rng.standard_normal((n_img, N, nx)) + 1j * rng.standard_normal((n_img, N, nx))
        projs = rng.standard_normal((n_img, N, nx)) + 1j * rng.standard_normal((n_img, N, nx))
        ctf = rng.uniform(-1, 1, (n_img, N, nx))
        pw = rng.uniform(0, 0.01, (n_img, N, nx))
        shells = _make_shell_indices(N, nx)

        relion = compute_weighted_noise(images, projs, ctf, pw, shells, n_shells)
        reference = _reference_noise(images, projs, ctf, pw, shells, n_shells)

        max_diff = np.max(np.abs(relion - reference))
        scale = np.max(np.abs(reference)) + 1e-30
        rel_err = max_diff / scale
        assert rel_err < 1e-12, f"N={N}: rel_err={rel_err}"
