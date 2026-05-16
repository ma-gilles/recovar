"""Phase 2 (P3): Compare RELION's windowFourierTransform against recovar's fourier_window.

Tests:
1. 2D crop: windowFourierTransform_2d downsample matches recovar's index-based approach
2. 2D pad: windowFourierTransform_2d upsample (zero-padding in Fourier domain)
3. 3D crop/pad: windowFourierTransform_3d
4. Round-trip: crop then pad back, verify identity at original frequencies
"""

import numpy as np
from recovar.relion_bind._relion_bind_core import (
    window_fourier_transform_2d,
    window_fourier_transform_3d,
)


def _make_random_fftw_half_2d(n, rng=None):
    """Random complex half-transform of shape (n, n//2+1)."""
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.standard_normal((n, n // 2 + 1)) + 1j * rng.standard_normal((n, n // 2 + 1))


def _make_random_fftw_half_3d(n, rng=None):
    """Random complex half-transform of shape (n, n, n//2+1)."""
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.standard_normal((n, n, n // 2 + 1)) + 1j * rng.standard_normal((n, n, n // 2 + 1))


def _numpy_window_fourier_2d(ft_in, newdim):
    """Pure-numpy reference for RELION's windowFourierTransform (2D).

    RELION convention: FFTW half-complex layout.
    Crop: keep low-frequency components, discard high.
    Pad: zero-fill high-frequency components.
    """
    ny_in, nx_in = ft_in.shape
    nx_out = newdim // 2 + 1

    if newdim == ny_in:
        return ft_in.copy()

    out = np.zeros((newdim, nx_out), dtype=ft_in.dtype)

    if newdim < ny_in:
        # Crop: keep [0..newdim//2] and [-(newdim//2-1)..] in y
        nx_copy = min(nx_in, nx_out)
        half_new = newdim // 2
        # Positive y frequencies: rows 0..half_new
        out[: half_new + 1, :nx_copy] = ft_in[: half_new + 1, :nx_copy]
        # Negative y frequencies: last (half_new-1) rows if newdim is even
        if newdim > 1:
            n_neg = newdim - half_new - 1
            if n_neg > 0:
                out[half_new + 1 :, :nx_copy] = ft_in[ny_in - n_neg :, :nx_copy]
    else:
        # Pad: RELION applies a radial mask (kp^2+ip^2+jp^2 <= max_r2)
        # where max_r2 = (XSIZE(in)-1)^2 to avoid asymmetric corners.
        max_r2 = (nx_in - 1) ** 2
        for iy_idx in range(ny_in):
            ip = iy_idx if iy_idx <= ny_in // 2 else iy_idx - ny_in
            oy = ip if ip >= 0 else newdim + ip
            for ix_idx in range(nx_in):
                jp = ix_idx
                if ip * ip + jp * jp <= max_r2:
                    out[oy, ix_idx] = ft_in[iy_idx, ix_idx]

    return out


class TestWindowFourier2D:
    """2D windowFourierTransform: RELION C++ vs numpy reference."""

    def test_crop_128_to_64(self):
        ft_in = _make_random_fftw_half_2d(128)
        relion_out = window_fourier_transform_2d(ft_in, 64)
        ref_out = _numpy_window_fourier_2d(ft_in, 64)

        assert relion_out.shape == (64, 33), f"Shape mismatch: {relion_out.shape}"
        max_diff = np.max(np.abs(relion_out - ref_out))
        assert max_diff < 1e-14, f"Crop 128→64 diverges: max diff = {max_diff:.2e}"

    def test_crop_128_to_96(self):
        ft_in = _make_random_fftw_half_2d(128)
        relion_out = window_fourier_transform_2d(ft_in, 96)
        ref_out = _numpy_window_fourier_2d(ft_in, 96)

        assert relion_out.shape == (96, 49)
        max_diff = np.max(np.abs(relion_out - ref_out))
        assert max_diff < 1e-14, f"Crop 128→96 diverges: max diff = {max_diff:.2e}"

    def test_pad_64_to_128(self):
        ft_in = _make_random_fftw_half_2d(64)
        relion_out = window_fourier_transform_2d(ft_in, 128)
        ref_out = _numpy_window_fourier_2d(ft_in, 128)

        assert relion_out.shape == (128, 65)
        max_diff = np.max(np.abs(relion_out - ref_out))
        assert max_diff < 1e-14, f"Pad 64→128 diverges: max diff = {max_diff:.2e}"

    def test_identity(self):
        ft_in = _make_random_fftw_half_2d(128)
        relion_out = window_fourier_transform_2d(ft_in, 128)

        max_diff = np.max(np.abs(relion_out - ft_in))
        assert max_diff < 1e-14, f"Identity diverges: max diff = {max_diff:.2e}"

    def test_roundtrip_crop_pad(self):
        """Crop then pad back: low frequencies should be preserved."""
        ft_in = _make_random_fftw_half_2d(128)
        cropped = window_fourier_transform_2d(ft_in, 64)
        padded_back = window_fourier_transform_2d(cropped, 128)

        # The low-frequency part should match exactly
        ref_cropped = _numpy_window_fourier_2d(ft_in, 64)
        recovered = _numpy_window_fourier_2d(ref_cropped, 128)

        max_diff = np.max(np.abs(padded_back - recovered))
        assert max_diff < 1e-14, f"Round-trip crop→pad diverges: max diff = {max_diff:.2e}"


class TestWindowFourier3D:
    """3D windowFourierTransform: basic shape and consistency checks."""

    def test_crop_64_to_32(self):
        ft_in = _make_random_fftw_half_3d(64)
        relion_out = window_fourier_transform_3d(ft_in, 32)

        assert relion_out.shape == (32, 32, 17), f"Shape mismatch: {relion_out.shape}"

    def test_pad_32_to_64(self):
        ft_in = _make_random_fftw_half_3d(32)
        relion_out = window_fourier_transform_3d(ft_in, 64)

        assert relion_out.shape == (64, 64, 33), f"Shape mismatch: {relion_out.shape}"

    def test_identity_3d(self):
        ft_in = _make_random_fftw_half_3d(32)
        relion_out = window_fourier_transform_3d(ft_in, 32)

        max_diff = np.max(np.abs(relion_out - ft_in))
        assert max_diff < 1e-14, f"3D identity diverges: max diff = {max_diff:.2e}"


class TestWindowVsRecovarFourierWindow:
    """Compare RELION's windowFourierTransform against recovar's index-based approach.

    recovar uses make_fourier_window_indices_np to select pixels within
    a radius, while RELION does array crop/pad. These are different operations:
    - RELION: rectangular crop in (ky, kx) space
    - recovar: radial mask (round(radius) <= current_size//2)

    This test documents where they agree and disagree.
    """

    def test_compare_selected_pixels(self):
        """Compare which pixels survive: RELION rectangular vs recovar radial."""
        from recovar.em.dense_single_volume.helpers.fourier_window import (
            make_fourier_window_indices_np,
        )

        image_shape = (128, 128)
        current_size = 64

        # recovar: radial index mask on half-spectrum
        indices_recovar, n_wind = make_fourier_window_indices_np(image_shape, current_size)

        # RELION: rectangular crop 128→64 in FFTW layout
        # After crop, shape is (64, 33). In the original (128, 65) layout,
        # identify which pixels survive.
        ny_in, nx_in = 128, 65
        ny_out, nx_out = 64, 33

        # Build RELION's surviving pixel map in original layout
        relion_mask = np.zeros((ny_in, nx_in), dtype=bool)
        # Positive y: rows 0..32
        relion_mask[: ny_out // 2 + 1, :nx_out] = True
        # Negative y: rows 128-31..127 → maps to rows 97..127
        n_neg = ny_out - ny_out // 2 - 1
        if n_neg > 0:
            relion_mask[ny_in - n_neg :, :nx_out] = True
        relion_indices = np.where(relion_mask.ravel())[0]

        # recovar indices are into half-spectrum which may have different layout
        # Both should select the low-frequency region but with different boundaries
        n_relion = len(relion_indices)
        n_recovar = n_wind

        print("\n=== Pixel Selection Comparison (128→64) ===")
        print(f"  RELION rectangular: {n_relion} pixels")
        print(f"  recovar radial:     {n_recovar} pixels")
        print(
            f"  Difference: {abs(n_relion - n_recovar)} "
            f"({100 * abs(n_relion - n_recovar) / max(n_relion, n_recovar):.1f}%)"
        )

        # The rectangular vs radial selection should differ at corners
        # but agree in the bulk. Document this.
        overlap = len(set(relion_indices) & set(indices_recovar))
        print(f"  Overlap: {overlap} pixels")
        print(f"  RELION-only: {n_relion - overlap}")
        print(f"  recovar-only: {n_recovar - overlap}")
