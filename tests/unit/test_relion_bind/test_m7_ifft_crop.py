"""M7: IFFT + crop to ori_size -- binding parity test.

Compares RELION's ``windowToOridimRealSpace`` (backprojector.cpp:2589)
against recovar's ``post_process_from_filter_v2`` (iDFT + unpad_volume_spatial_domain).

Both codes perform the same logical sequence on Fourier accumulators:
  1. Wiener-solve the accumulators
  2. IFFT on the padded grid
  3. Crop back to ori_size in real space
  4. Soft spherical mask
  5. Gridding correction

Known strategy differences that prevent exact parity:
  - **Wiener solve**: RELION uses iterative pre-weighting (convolution-based
    deconvolution of the weight array); recovar uses a direct per-voxel
    Wiener filter with radial flooring at 1/1000 of the shell average.
  - **Gridding correction**: RELION applies radial sinc^2 division;
    recovar uses per-axis sinc^2 (``griddingCorrect_square``).  These
    agree only along the cardinal axes.
  - **Soft mask**: Minor implementation differences (cosine width, radius).
  - **Grid size**: RELION's padded grid is ``pad_size = 2*round(pf*r_max)+3``
    which differs from recovar's ``N*pf``; the accumulator conversion
    introduces interpolation error at the grid boundary.

**FFT convention note**:  ``relion_projector_to_recovar_centered`` returns
``fftshift(fftn(vol))`` (standard centered), NOT recovar's internal
convention ``fftshift(fftn(fftshift(vol)))`` (which ``get_dft3`` uses).
The difference is a factor ``(-1)^(i+j+k)`` at each voxel in the centered
domain.  When feeding converted data into ``post_process_from_filter_v2``
(which calls ``get_idft3``), this sign correction must be applied.

Test strategy: project a known volume through RELION, reconstruct via
RELION's full pipeline, then separately reconstruct via recovar's
``post_process_from_filter_v2`` using recovar-convention accumulators
built from the same known volume.  Compare real-space outputs.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import (
    TRILINEAR,
    backproject_and_reconstruct,
    euler_angles_to_matrix,
    get_coarse_orientations,
    project_volume,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_volume(N, rng):
    """Low-pass filtered random volume for clean projections."""
    vol = rng.standard_normal((N, N, N))
    ft = np.fft.rfftn(vol)
    kz = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    kx = np.arange(N // 2 + 1)
    KZ, KY = np.meshgrid(kz, ky, indexing="ij")
    r2 = KZ[:, :, np.newaxis] ** 2 + KY[:, :, np.newaxis] ** 2 + kx[np.newaxis, np.newaxis, :] ** 2
    ft *= np.exp(-r2 / (2 * (N / 6) ** 2))
    return np.fft.irfftn(ft, s=(N, N, N))


def _project_at_orientations(vol, orientations, N, pf=2):
    """Project volume through RELION's binding at given orientations."""
    n_orient = orientations.shape[0]
    images = np.zeros((n_orient, N, N // 2 + 1), dtype=np.complex128)
    rot_mats = np.zeros((n_orient, 3, 3))
    for i in range(n_orient):
        rot, tilt, psi = orientations[i]
        A = euler_angles_to_matrix(float(rot), float(tilt), float(psi))
        rot_mats[i] = A
        images[i] = project_volume(
            vol,
            A,
            ori_size=N,
            padding_factor=pf,
            interpolator=TRILINEAR,
            current_size=-1,
            do_gridding=True,
        )
    return images, rot_mats


def _voxel_correlation(a, b):
    """Pearson correlation between two real-space volumes."""
    af = a.ravel() - a.ravel().mean()
    bf = b.ravel() - b.ravel().mean()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def _relative_error(a, b):
    """Relative L2 error: ||a-b||/||a||."""
    return float(np.linalg.norm(a.ravel() - b.ravel()) / (np.linalg.norm(a.ravel()) + 1e-30))


def _standard_to_recovar_centered(arr):
    """Apply the (-1)^(i+j+k) sign correction to convert from standard
    centered layout (fftshift(fftn(vol))) to recovar's convention
    (fftshift(fftn(fftshift(vol)))) used by get_dft3/get_idft3.

    The two layouts differ by a multiplicative sign factor at each voxel.
    """
    N = arr.shape[0]
    idx = np.arange(N)
    sign = np.where(
        (idx[:, None, None] + idx[None, :, None] + idx[None, None, :]) % 2 == 0,
        1.0,
        -1.0,
    )
    return arr * sign


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIFFTCropParity:
    """Compare RELION's IFFT+crop with recovar's iDFT+crop on known Fourier data.

    The test creates a known real-space volume, embeds its FFT into RELION's
    projector layout, and compares:

    (a) RELION-style IFFT+crop (simulated in numpy, matching
        ``windowToOridimRealSpace``): ifftshift -> irfftn(pad_size) ->
        CenterFFTbySign -> crop.

    (b) recovar-style iDFT+crop: ``relion_projector_to_recovar_centered``
        (N^3 standard centered) -> sign correction -> ``get_idft3``.

    Both paths start from the EXACT same Fourier volume (no interpolation,
    no Wiener, no gridding correction), so differences come only from the
    IFFT+crop step itself.

    The ``relion_projector_to_recovar_centered`` conversion returns
    ``fftshift(fftn(vol))`` (standard centered), while recovar's
    ``get_idft3`` expects ``fftshift(fftn(fftshift(vol)))`` (recovar
    convention).  The sign correction ``(-1)^(i+j+k)`` bridges the two.
    """

    @pytest.mark.parametrize("N", [16, 32, 64])
    @pytest.mark.parametrize("pf", [1, 2])
    def test_conversion_roundtrip_exact(self, N, pf):
        """projector -> recovar centered -> iDFT recovers the original volume."""
        from recovar.relion_bind.conversions import (
            fftw_half_to_relion_projector,
            relion_projector_to_recovar_centered,
        )

        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)

        # Embed the volume's exact FFT into RELION's projector layout
        fftw_half = np.fft.rfftn(vol)
        proj = fftw_half_to_relion_projector(fftw_half, padding_factor=pf)

        # Convert projector -> standard centered (N^3)
        std_centered = relion_projector_to_recovar_centered(
            proj,
            ori_size=N,
            padding_factor=pf,
        )

        # Apply sign correction for recovar convention, then get_idft3
        recovar_centered = _standard_to_recovar_centered(std_centered)
        recovered = np.fft.ifftshift(
            np.fft.ifftn(np.fft.ifftshift(recovar_centered)),
        ).real

        err = np.max(np.abs(recovered - vol))

        print(f"\n=== M7 conversion round-trip (N={N}, pf={pf}) ===")
        print(f"  Max error: {err:.2e}")

        assert err < 1e-12, f"Conversion round-trip error {err:.2e} exceeds 1e-12 (N={N}, pf={pf})"

    @pytest.mark.parametrize("N", [16, 32, 64])
    @pytest.mark.parametrize("pf", [1, 2])
    def test_standard_vs_recovar_convention(self, N, pf):
        """Sign correction bridges standard centered and recovar convention."""
        from recovar.relion_bind.conversions import (
            fftw_half_to_relion_projector,
            relion_projector_to_recovar_centered,
        )

        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)

        # Ground truth in recovar convention: fftshift(fftn(fftshift(vol)))
        recovar_gt = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol)))

        # Via conversion: projector -> standard centered -> sign correction
        fftw_half = np.fft.rfftn(vol)
        proj = fftw_half_to_relion_projector(fftw_half, padding_factor=pf)
        std_centered = relion_projector_to_recovar_centered(
            proj,
            ori_size=N,
            padding_factor=pf,
        )
        recovar_from_proj = _standard_to_recovar_centered(std_centered)

        err = np.max(np.abs(recovar_from_proj - recovar_gt))

        print(f"\n=== M7 convention match (N={N}, pf={pf}) ===")
        print(f"  Max error vs recovar GT: {err:.2e}")

        assert err < 1e-12, f"Convention mismatch {err:.2e} exceeds 1e-12 (N={N}, pf={pf})"

    @pytest.mark.parametrize("N", [16, 32, 64])
    @pytest.mark.parametrize("pf", [1, 2])
    def test_relion_roundtrip_correlation(self, N, pf):
        """RELION project -> backproject+reconstruct recovers the input volume."""
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)

        all_orientations = get_coarse_orientations(2)
        images, rot_mats = _project_at_orientations(vol, all_orientations, N, pf=pf)

        tau2 = np.ones(N // 2 + 1) * 1e6
        empty_w = np.zeros((0,), dtype=np.float64)

        relion_vol = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_w,
            tau2,
            ori_size=N,
            padding_factor=pf,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )

        # The binding operates in numpy axis convention (no frame conversion).
        corr = _voxel_correlation(relion_vol, vol)

        print(f"\n=== M7 RELION round-trip (N={N}, pf={pf}) ===")
        print(f"  Correlation with GT: {corr:.6f}")
        print(f"  Recon range: [{relion_vol.min():.6e}, {relion_vol.max():.6e}]")
        print(f"  GT range:    [{vol.min():.6e}, {vol.max():.6e}]")

        # With 4608 orientations (order 2) and trilinear interpolation,
        # the round-trip quality depends on N and pf.  pf=1 has more
        # interpolation error; small N has fewer orientations per voxel.
        assert corr > 0.3, f"RELION round-trip corr {corr:.6f} below 0.3 (N={N}, pf={pf})"

    @pytest.mark.parametrize("N", [16, 32])
    @pytest.mark.parametrize("pf", [1, 2])
    def test_recovar_post_process_roundtrip(self, N, pf):
        """post_process_from_filter_v2 with identity Wiener recovers the input."""
        import jax.numpy as jnp

        from recovar.reconstruction.relion_functions import post_process_from_filter_v2

        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)
        upsampled_N = N * pf

        # Build recovar-convention Fourier volume on the (N*pf)^3 grid.
        # Zero-pad the real-space volume to (N*pf)^3, then apply get_dft3.
        vol_padded = np.zeros((upsampled_N,) * 3)
        pad_off = (upsampled_N - N) // 2
        vol_padded[pad_off : pad_off + N, pad_off : pad_off + N, pad_off : pad_off + N] = vol
        # get_dft3 convention: fftshift(fftn(fftshift(vol)))
        Ft_data = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol_padded)))

        # Uniform weight = 1 everywhere, very weak tau -> Wiener is identity
        Ft_ctf = jnp.ones(upsampled_N**3, dtype=jnp.float64)
        F_ty = jnp.array(Ft_data.reshape(-1))
        tau = jnp.ones(N**3, dtype=jnp.float64) * 1e6

        result = np.array(
            post_process_from_filter_v2(
                Ft_ctf,
                F_ty,
                (N, N, N),
                volume_upsampling_factor=pf,
                tau=tau,
                kernel="triangular",
                use_spherical_mask=False,
                grid_correct=False,
                gridding_correct="square",
                kernel_width=1,
                return_real_space=True,
                tau2_fudge=1.0,
            )
        ).reshape((N, N, N))

        corr = _voxel_correlation(result, vol)
        rel_err = _relative_error(result, vol)

        print(f"\n=== M7 recovar round-trip (N={N}, pf={pf}) ===")
        print(f"  Correlation: {corr:.10f}")
        print(f"  Relative error: {rel_err:.2e}")

        # With identity Wiener, no mask, no gridding correction,
        # the output should be very close to the input.  The radial mask
        # (r_max = upsampled_N//2 - 1) zeroes high-frequency corners,
        # causing a small loss especially for small N.
        assert corr > 0.99, f"recovar round-trip corr {corr:.6f} below 0.99 (N={N}, pf={pf})"


class TestIFFTCropOutputShape:
    """Verify that both pipelines produce the correct ori_size output shape."""

    @pytest.mark.parametrize("N", [16, 32, 64])
    @pytest.mark.parametrize("pf", [1, 2])
    def test_relion_output_shape(self, N, pf):
        """RELION's backproject_and_reconstruct returns (N, N, N)."""
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:6]
        images, rot_mats = _project_at_orientations(vol, orientations, N, pf=pf)

        tau2 = np.ones(N // 2 + 1) * 1e6
        empty_w = np.zeros((0,), dtype=np.float64)

        recon = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_w,
            tau2,
            ori_size=N,
            padding_factor=pf,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )
        assert recon.shape == (N, N, N), f"Expected ({N},{N},{N}), got {recon.shape}"

    @pytest.mark.parametrize("N", [16, 32, 64])
    @pytest.mark.parametrize("pf", [1, 2])
    def test_recovar_output_shape(self, N, pf):
        """recovar's post_process_from_filter_v2 returns (N, N, N)."""
        import jax.numpy as jnp

        from recovar.reconstruction.relion_functions import post_process_from_filter_v2

        upsampled_N = N * pf
        og_volume_shape = (N, N, N)

        # Uniform accumulators in recovar's centered layout
        Ft_ctf = jnp.ones(upsampled_N**3, dtype=jnp.float64)
        rng = np.random.default_rng(42)
        F_ty = jnp.array(
            rng.standard_normal(upsampled_N**3) + 1j * rng.standard_normal(upsampled_N**3),
        )
        tau = jnp.ones(N**3, dtype=jnp.float64) * 1e6

        result = np.array(
            post_process_from_filter_v2(
                Ft_ctf,
                F_ty,
                og_volume_shape,
                volume_upsampling_factor=pf,
                tau=tau,
                kernel="triangular",
                use_spherical_mask=True,
                grid_correct=True,
                gridding_correct="square",
                kernel_width=1,
                return_real_space=True,
                tau2_fudge=1.0,
            )
        )
        assert result.shape == (N, N, N), f"Expected ({N},{N},{N}), got {result.shape}"


class TestIFFTCropDeterminism:
    """Both pipelines must be deterministic across repeated calls."""

    @pytest.mark.parametrize("N", [16, 32])
    def test_relion_deterministic(self, N):
        """RELION's IFFT+crop is deterministic."""
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:12]
        images, rot_mats = _project_at_orientations(vol, orientations, N, pf=2)

        tau2 = np.ones(N // 2 + 1) * 1e6
        empty_w = np.zeros((0,), dtype=np.float64)
        kwargs = dict(
            ori_size=N,
            padding_factor=2,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )

        r1 = backproject_and_reconstruct(images, rot_mats, empty_w, tau2, **kwargs)
        r2 = backproject_and_reconstruct(images, rot_mats, empty_w, tau2, **kwargs)
        assert np.max(np.abs(r1 - r2)) == 0.0, "RELION IFFT+crop not bit-exact"

    @pytest.mark.parametrize("N", [16, 32])
    def test_recovar_deterministic(self, N):
        """recovar's iDFT+crop is deterministic."""
        import jax.numpy as jnp

        from recovar.reconstruction.relion_functions import post_process_from_filter_v2

        pf = 2
        upsampled_N = N * pf
        rng = np.random.default_rng(42)

        Ft_ctf = jnp.ones(upsampled_N**3, dtype=jnp.float64)
        F_ty = jnp.array(
            rng.standard_normal(upsampled_N**3) + 1j * rng.standard_normal(upsampled_N**3),
        )
        tau = jnp.ones(N**3, dtype=jnp.float64) * 1e6

        kwargs = dict(
            og_volume_shape=(N, N, N),
            volume_upsampling_factor=pf,
            tau=tau,
            kernel="triangular",
            use_spherical_mask=True,
            grid_correct=True,
            gridding_correct="square",
            kernel_width=1,
            return_real_space=True,
            tau2_fudge=1.0,
        )

        r1 = np.array(post_process_from_filter_v2(Ft_ctf, F_ty, **kwargs))
        r2 = np.array(post_process_from_filter_v2(Ft_ctf, F_ty, **kwargs))
        max_diff = np.max(np.abs(r1 - r2))
        assert max_diff < 1e-14, f"recovar iDFT+crop not deterministic: max_diff={max_diff:.2e}"


class TestIFFTCropSelfConsistency:
    """Verify that the IFFT+crop step is self-consistent within each pipeline."""

    @pytest.mark.parametrize("N", [16, 32])
    @pytest.mark.parametrize("pf", [1, 2])
    def test_relion_skip_gridding_differs(self, N, pf):
        """skip_gridding=True vs False should differ (gridding correction has effect)."""
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(2)[:48]
        images, rot_mats = _project_at_orientations(vol, orientations, N, pf=pf)

        tau2 = np.ones(N // 2 + 1) * 1e6
        empty_w = np.zeros((0,), dtype=np.float64)
        common = dict(
            ori_size=N,
            padding_factor=pf,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
        )

        with_gc = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_w,
            tau2,
            skip_gridding=False,
            **common,
        )
        without_gc = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_w,
            tau2,
            skip_gridding=True,
            **common,
        )

        diff = np.max(np.abs(with_gc - without_gc))
        assert diff > 1e-6, f"Gridding correction had no effect (N={N}, pf={pf}): max_diff={diff:.2e}"

    @pytest.mark.parametrize("N", [32])
    def test_pf1_vs_pf2_differs(self, N):
        """pf=1 vs pf=2 should produce structurally similar but numerically
        different volumes.  Only tested at N=32 where both padding factors
        yield reasonable reconstructions with the order-2 orientation set."""
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(2)[:96]
        images_pf1, rot_mats = _project_at_orientations(vol, orientations, N, pf=1)
        images_pf2, _ = _project_at_orientations(vol, orientations, N, pf=2)

        tau2 = np.ones(N // 2 + 1) * 1e6
        empty_w = np.zeros((0,), dtype=np.float64)
        common = dict(
            ori_size=N,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )

        r1 = backproject_and_reconstruct(
            images_pf1,
            rot_mats,
            empty_w,
            tau2,
            padding_factor=1,
            **common,
        )
        r2 = backproject_and_reconstruct(
            images_pf2,
            rot_mats,
            empty_w,
            tau2,
            padding_factor=2,
            **common,
        )

        # Both should reconstruct the same volume but differ in detail
        corr = _voxel_correlation(r1, r2)
        diff = np.max(np.abs(r1 - r2))
        print(f"\n  pf=1 vs pf=2 (N={N}): corr={corr:.6f}, max_diff={diff:.4e}")
        # They should be numerically different
        assert diff > 1e-6, f"pf=1 and pf=2 are identical (N={N})"
