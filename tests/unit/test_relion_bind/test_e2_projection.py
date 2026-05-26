"""Phase 2b: E2 projection parity — RELION's Projector::project vs recovar.

Tests that RELION's 3D→2D Fourier slice via trilinear interpolation
matches recovar's slice_volume on identical inputs.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import (
    TRILINEAR,
    compute_fourier_transform_map,
    project_volume,
)


def _make_test_volume(N, rng):
    """Create a smooth test volume (low-pass filtered random)."""
    vol = rng.standard_normal((N, N, N))
    ft = np.fft.rfftn(vol, axes=(0, 1, 2))
    idx = np.arange(N) - N // 2
    kz, ky = np.meshgrid(idx, idx, indexing="ij")
    kx = np.arange(N // 2 + 1)
    r2 = kz[:, :, np.newaxis] ** 2 + ky[:, :, np.newaxis] ** 2 + kx[np.newaxis, np.newaxis, :] ** 2
    mask = np.exp(-r2 / (2 * (N / 4) ** 2))
    ft *= mask
    return np.fft.irfftn(ft, s=(N, N, N), axes=(0, 1, 2))


def _relion_center_fft_by_sign(arr):
    """Replicate RELION's CenterFFTbySign: negate where (k+i+j) is odd.

    RELION uses ``((k ^ i ^ j) & 1) != 0`` on physical indices (fftw.h:400).
    """
    result = arr.copy()
    ndim = arr.ndim
    if ndim == 3:
        nz, ny, nx = arr.shape
        k_idx = np.arange(nz)[:, None, None]
        i_idx = np.arange(ny)[None, :, None]
        j_idx = np.arange(nx)[None, None, :]
        sign = np.where((k_idx + i_idx + j_idx) % 2 == 1, -1.0, 1.0)
    elif ndim == 2:
        ny, nx = arr.shape
        i_idx = np.arange(ny)[:, None]
        j_idx = np.arange(nx)[None, :]
        sign = np.where((i_idx + j_idx) % 2 == 1, -1.0, 1.0)
    else:
        sign = np.where(np.arange(arr.shape[0]) % 2 == 1, -1.0, 1.0)
    result *= sign
    return result


class TestProjectVolumeSmoke:
    """Basic smoke tests for the project binding."""

    def test_identity_rotation(self):
        rng = np.random.default_rng(42)
        N = 32
        vol = _make_test_volume(N, rng)
        A = np.eye(3)
        result = project_volume(
            vol, A, ori_size=N, padding_factor=2, interpolator=TRILINEAR, current_size=-1, do_gridding=True
        )
        assert result.shape == (N, N // 2 + 1)
        assert result.dtype == np.complex128
        assert np.max(np.abs(result)) > 0

    def test_different_rotations_give_different_results(self):
        rng = np.random.default_rng(42)
        N = 32
        vol = _make_test_volume(N, rng)
        A1 = np.eye(3)
        A2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        p1 = project_volume(vol, A1, ori_size=N, padding_factor=2)
        p2 = project_volume(vol, A2, ori_size=N, padding_factor=2)
        assert not np.allclose(p1, p2)


class TestComputeFourierTransformMap:
    """Test the projector storage (E1)."""

    def test_returns_correct_projector_info(self):
        rng = np.random.default_rng(42)
        N = 32
        vol = _make_test_volume(N, rng)
        proj_data, ps, osz, pf, r_max, r_min_nn, interp = compute_fourier_transform_map(
            vol, ori_size=N, padding_factor=2
        )
        assert osz == N
        assert pf == 2
        assert r_max == N // 2
        assert interp == TRILINEAR
        assert proj_data.ndim == 3
        assert proj_data.dtype == np.complex128

    def test_projector_data_nonzero(self):
        rng = np.random.default_rng(42)
        N = 32
        vol = _make_test_volume(N, rng)
        proj_data, *_ = compute_fourier_transform_map(vol, ori_size=N, padding_factor=1)
        assert np.max(np.abs(proj_data)) > 0

    def test_data_dim_argument_is_exposed_for_initialmodel(self):
        rng = np.random.default_rng(42)
        N = 16
        vol = _make_test_volume(N, rng)
        proj_data, *_ = compute_fourier_transform_map(
            vol,
            ori_size=N,
            padding_factor=1,
            current_size=8,
            data_dim=2,
        )
        assert proj_data.ndim == 3
        assert proj_data.dtype == np.complex128

    def test_projector_defaults_preserve_dense_em_data_dim(self):
        rng = np.random.default_rng(42)
        N = 16
        vol = _make_test_volume(N, rng)
        A = np.eye(3)

        default_map, *_ = compute_fourier_transform_map(
            vol,
            ori_size=N,
            padding_factor=1,
            current_size=8,
            do_gridding=False,
        )
        dense_em_map, *_ = compute_fourier_transform_map(
            vol,
            ori_size=N,
            padding_factor=1,
            current_size=8,
            do_gridding=False,
            data_dim=3,
        )
        np.testing.assert_array_equal(default_map, dense_em_map)

        default_projection = project_volume(
            vol,
            A,
            ori_size=N,
            padding_factor=1,
            current_size=8,
            do_gridding=False,
        )
        dense_em_projection = project_volume(
            vol,
            A,
            ori_size=N,
            padding_factor=1,
            current_size=8,
            do_gridding=False,
            data_dim=3,
        )
        np.testing.assert_array_equal(default_projection, dense_em_projection)

    @pytest.mark.parametrize("N", [16, 32])
    def test_projector_data_matches_numpy(self, N):
        """Verify projector storage matches a numpy reference.

        RELION's computeFourierTransformMap:
        1. No padding for pf=1 (padoridim = ori_size)
        2. rFFT with 1/N^3 normalization (fftw.cpp:358)
        3. CenterFFTbySign on the FFT output (linear-index parity)
        4. Copy into yz-centered projector storage with normfft=1 (pf=1)
        """
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)

        proj_data, *_ = compute_fourier_transform_map(
            vol,
            ori_size=N,
            padding_factor=1,
            do_gridding=False,
            data_dim=3,
        )
        pad_size = proj_data.shape[0]
        half_x = N // 2 + 1

        # Numpy reference
        vol_rfft = np.fft.rfftn(vol, axes=(0, 1, 2)) / N**3
        vol_rfft_csign = _relion_center_fft_by_sign(vol_rfft)

        # Map FFTW physical → projector logical (yz-centered, x left-anchored)
        # Only within r_max (projector.cpp:514: r2 <= max_r2)
        c = pad_size // 2
        r_max = N // 2
        max_r2 = r_max * r_max
        ref_proj = np.zeros_like(proj_data)
        for k in range(N):
            kp = k if k <= N // 2 else k - N
            for i in range(N):
                ip = i if i <= N // 2 else i - N
                for j in range(half_x):
                    if kp * kp + ip * ip + j * j <= max_r2:
                        ref_proj[kp + c, ip + c, j] = vol_rfft_csign[k, i, j]

        mask = np.abs(proj_data) + np.abs(ref_proj) > 0
        np.testing.assert_allclose(proj_data[mask], ref_proj[mask], rtol=1e-12)


class TestProjectionParity:
    """Compare RELION's projection against numpy reference."""

    @pytest.mark.parametrize("N", [16, 32])
    def test_identity_vs_projector_data(self, N):
        """Identity projection should extract the kz=0 plane from projector data."""
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)

        proj_data, *_ = compute_fourier_transform_map(vol, ori_size=N, padding_factor=1, do_gridding=False)
        pad_size = proj_data.shape[0]
        c = pad_size // 2

        A = np.eye(3)
        relion_slice = project_volume(vol, A, ori_size=N, padding_factor=1, do_gridding=False, current_size=-1)

        # Identity projection reads (x, y, kz=0) from projector data
        ref_output = np.zeros((N, N // 2 + 1), dtype=complex)
        for i in range(N):
            y = i if i <= N // 2 else i - N
            ref_output[i, :] = proj_data[c, c + y, : N // 2 + 1]

        assert relion_slice.shape == (N, N // 2 + 1)
        np.testing.assert_allclose(relion_slice, ref_output, atol=1e-15)

    @pytest.mark.parametrize("N", [32, 64])
    def test_relion_self_consistency(self, N):
        """180 deg rotation should preserve power spectrum.

        Uses N>=32: RELION's trilinear interpolation boundary handling
        breaks exact Hermitian symmetry at small N with pf=2."""
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)

        A_id = np.eye(3)
        A_180z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

        p_id = project_volume(vol, A_id, ori_size=N, padding_factor=2)
        p_180 = project_volume(vol, A_180z, ori_size=N, padding_factor=2)

        power_id = np.sum(np.abs(p_id) ** 2)
        power_180 = np.sum(np.abs(p_180) ** 2)
        np.testing.assert_allclose(power_id, power_180, rtol=0.005)

    def test_current_size_output_allocates_cropped_image(self):
        rng = np.random.default_rng(42)
        N = 16
        vol = _make_test_volume(N, rng)
        projected = project_volume(
            vol,
            np.eye(3),
            ori_size=N,
            padding_factor=1,
            current_size=8,
            data_dim=2,
            current_size_output=True,
        )
        assert projected.shape == (8, 5)
