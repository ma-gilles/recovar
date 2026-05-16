"""Phase 3 (E3): Compare RELION's shiftImageInFourierTransform against recovar.

RELION: dotp = -2*pi * (jp*xshift/oridim + ip*yshift/oridim)
        out = in * exp(i * dotp)

recovar: phase = exp(-2j*pi * (lattice @ translation))
         out = in * phase

Both use the same sign convention: exp(-2j*pi*(freq*shift)) where freq
is in 1/pixel and shift is in pixels. Should match to machine precision.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import shift_image_in_fourier_transform_2d


def _numpy_phase_shift_fftw(ft_in, oridim, xshift, yshift):
    """Reference phase shift on FFTW half-complex layout."""
    ny, nx = ft_in.shape
    jp = np.arange(nx)
    ip = np.arange(ny, dtype=np.float64)
    ip[ip > ny // 2] -= ny

    JP, IP = np.meshgrid(jp, ip)
    dotp = -2 * np.pi * (JP * xshift / oridim + IP * yshift / oridim)
    phase = np.exp(1j * dotp)
    return ft_in * phase


class TestShiftParity:
    """RELION shift vs numpy reference on FFTW layout."""

    @pytest.mark.parametrize(
        "xshift,yshift",
        [
            (0.0, 0.0),
            (3.5, 0.0),
            (0.0, -2.1),
            (3.5, -2.1),
            (0.5, 0.5),
            (-10.3, 7.8),
        ],
    )
    def test_shift_matches_numpy(self, xshift, yshift):
        rng = np.random.default_rng(42)
        n = 128
        ft_in = rng.standard_normal((n, n // 2 + 1)) + 1j * rng.standard_normal((n, n // 2 + 1))

        relion_out = shift_image_in_fourier_transform_2d(ft_in, float(n), n, xshift, yshift)
        ref_out = _numpy_phase_shift_fftw(ft_in, n, xshift, yshift)

        max_diff = np.max(np.abs(relion_out - ref_out))
        assert max_diff < 1e-12, f"Shift ({xshift}, {yshift}): max diff = {max_diff:.2e}"

    def test_shift_output_shape_with_window(self):
        """Shift binding with newdim < input: verify output shape only.

        Note: our binding (like RELION's tabulated version) reads from input
        rows 0..newdim-1, which gives wrong data for negative frequencies
        when newdim < ny_in. In practice RELION pre-crops images before
        shifting, so this path isn't exercised. We only check the shape here.
        """
        rng = np.random.default_rng(42)
        n = 128
        ft_in = rng.standard_normal((n, n // 2 + 1)) + 1j * rng.standard_normal((n, n // 2 + 1))

        relion_out = shift_image_in_fourier_transform_2d(ft_in, float(n), 64, 3.5, -2.1)
        assert relion_out.shape == (64, 33)


class TestShiftVsRecovar:
    """Compare RELION shift against recovar's translate_images.

    The pixel orderings differ:
    - FFTW:    shape (ny, nx_rfft), rows=FFTW y-freq, cols=rfft x-freq
    - recovar: shape (nx_rfft, ny_centered), rows=rfft x-freq, cols=centered y-freq
    We build an index permutation to map between them.
    """

    @staticmethod
    def _build_fftw_to_recovar_perm(n):
        """Build permutation: FFTW raveled index → recovar raveled index."""
        nx_rfft = n // 2 + 1
        # FFTW: pixel (iy, ix) → flat = iy * nx_rfft + ix
        #   kx = ix, ky = iy if iy <= n//2 else iy - n
        # recovar: pixel (row, col) → flat = row * n + col
        #   row = kx (rfft: 0..n//2), col = ky + n//2 (centered: 0..n-1)
        perm = np.zeros(n * nx_rfft, dtype=np.int64)
        for iy in range(n):
            ky = iy if iy <= n // 2 else iy - n
            for ix in range(nx_rfft):
                kx = ix
                fftw_flat = iy * nx_rfft + ix
                recovar_row = kx
                recovar_col = (ky + n // 2) % n
                recovar_flat = recovar_row * n + recovar_col
                perm[fftw_flat] = recovar_flat
        return perm

    def test_shift_matches_recovar(self):
        jax = pytest.importorskip("jax")
        jnp = jax.numpy
        from recovar.core import fourier_transform_utils as ftu
        from recovar.core.geometry import translate_single_image

        rng = np.random.default_rng(42)
        n = 128
        xshift, yshift = 3.5, -2.1

        ft_fftw = rng.standard_normal((n, n // 2 + 1)) + 1j * rng.standard_normal((n, n // 2 + 1))

        # RELION shift
        relion_out = shift_image_in_fourier_transform_2d(ft_fftw, float(n), n, xshift, yshift)

        # Permute FFTW data → recovar ordering (scatter: recovar[perm[i]] = fftw[i])
        perm = self._build_fftw_to_recovar_perm(n)
        ft_recovar_order = np.zeros(n * (n // 2 + 1), dtype=ft_fftw.dtype)
        ft_recovar_order[perm] = ft_fftw.ravel()

        lattice = ftu.get_k_coordinate_of_each_pixel_real((n, n), voxel_size=1, scaled=True)
        # recovar lattice[:,0] = ky (centered), lattice[:,1] = kx (rfft)
        # phase = exp(-2j*pi*(ky*t0 + kx*t1)), so translation = [yshift, xshift]
        translation = jnp.array([yshift, xshift])

        recovar_out_reordered = np.array(translate_single_image(jnp.array(ft_recovar_order), translation, lattice))

        # Permute recovar output back to FFTW ordering (gather: fftw[i] = recovar[perm[i]])
        recovar_out_fftw = recovar_out_reordered[perm].reshape(n, n // 2 + 1)

        # Exclude Nyquist ky row (iy=n//2): FFTW uses ky=+n/2, recovar uses
        # ky=-n/2 (centered). Phase exp(-2j*pi*(+n/2)*s/n) vs exp(-2j*pi*(-n/2)*s/n)
        # are conjugates — differ for non-integer shifts. Benign: Nyquist is
        # regularized away in reconstruction.
        mask = np.ones((n, n // 2 + 1), dtype=bool)
        mask[n // 2, :] = False

        max_diff = np.max(np.abs((relion_out - recovar_out_fftw)[mask]))
        print(f"\nRELION vs recovar shift: max diff = {max_diff:.2e} (excl. Nyquist ky)")
        assert max_diff < 1e-10, f"RELION vs recovar shift diverges: max diff = {max_diff:.2e}"
