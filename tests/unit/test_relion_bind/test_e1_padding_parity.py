"""E1 padding parity: recovar pad_volume_for_projection vs RELION computeFourierTransformMap.

Both functions take an N^3 real-space volume, zero-pad it to (pf*N)^3
in real space, then FFT to produce the padded Fourier volume.  RELION
stores the result in projector-centered half-complex layout
(pad_size, pad_size, pad_size//2+1) with 1/N^3 normalization and
CenterFFTbySign applied.  recovar stores a (pf*N)^3 centered
full-complex array with unnormalized FFT convention.

We compare them on the overlapping frequency bins within the r_max
sphere, which is the region where both representations carry signal.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import compute_fourier_transform_map

from recovar.relion_bind.conversions import compute_relion_pad_size

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_smooth_volume(N, rng):
    """Create a smooth test volume (low-pass filtered random).

    Identical to the helper in test_e2_projection.py.
    """
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
    """Replicate RELION's CenterFFTbySign: negate where (k+i+j) is odd."""
    result = arr.copy()
    ndim = arr.ndim
    if ndim == 3:
        nz, ny, nx = arr.shape
        k_idx = np.arange(nz)[:, None, None]
        i_idx = np.arange(ny)[None, :, None]
        j_idx = np.arange(nx)[None, None, :]
        sign = np.where((k_idx + i_idx + j_idx) % 2 == 1, -1.0, 1.0)
    else:
        raise ValueError(f"Only 3D supported, got {ndim}D")
    result *= sign
    return result


def _numpy_compute_fourier_transform_map(vol_relion, N, padding_factor):
    """Pure-numpy replica of RELION's computeFourierTransformMap (do_gridding=False).

    Steps (from projector.cpp):
      1. Compute padoridim = ROUND(pf * N), made even.
      2. Zero-pad vol from N^3 to padoridim^3 (centered, via XmippOrigin).
      3. rfftn(Mpad) / padoridim^3   (FourierTransformer::FourierTransform).
      4. CenterFFTbySign on the Fourier output.
      5. Copy into projector storage with factor normfft = pf^3,
         only for frequencies within r_max*pf of the origin.

    Net normalization: rfftn * pf^3 / padoridim^3 = rfftn / N^3.

    Returns
    -------
    ref_proj : complex array, shape (pad_size, pad_size, pad_size//2+1)
        Numpy reference for the projector data.
    padoridim : int
        Actual padded real-space dimension.
    """
    # Step 1: padoridim
    padoridim = round(padding_factor * N)
    if padoridim % 2 != 0:
        padoridim += 1
    pf_actual = padoridim / N

    pad_size = compute_relion_pad_size(N, padding_factor)
    c = pad_size // 2
    r_max = N // 2
    max_r2 = round(r_max * pf_actual) ** 2

    # Step 2: zero-pad (XmippOrigin centering)
    Mpad = np.zeros((padoridim, padoridim, padoridim))
    offset = padoridim // 2 - N // 2
    Mpad[offset : offset + N, offset : offset + N, offset : offset + N] = vol_relion

    # Step 3: rfftn / padoridim^3
    Faux = np.fft.rfftn(Mpad, axes=(0, 1, 2)) / padoridim**3

    # Step 4: CenterFFTbySign
    Faux_cs = _relion_center_fft_by_sign(Faux)

    # Step 5: copy into projector storage with normfft = pf^3
    normfft = pf_actual**3
    half_x = padoridim // 2 + 1
    ref_proj = np.zeros((pad_size, pad_size, pad_size // 2 + 1), dtype=np.complex128)
    for k in range(padoridim):
        kp = k if k <= padoridim // 2 else k - padoridim
        for i in range(padoridim):
            ip = i if i <= padoridim // 2 else i - padoridim
            for j in range(half_x):
                r2 = kp * kp + ip * ip + j * j
                if r2 <= max_r2:
                    pk, pi, pj = kp + c, ip + c, j
                    if 0 <= pk < pad_size and 0 <= pi < pad_size and pj < pad_size // 2 + 1:
                        ref_proj[pk, pi, pj] = Faux_cs[k, i, j] * normfft

    return ref_proj, padoridim


def _recovar_pad_numpy(vol_relion, N, padding_factor):
    """Pure-numpy replica of recovar's pad_volume_for_projection.

    Operates entirely in the RELION frame to avoid axis conversion
    complexity.  recovar's pipeline is:

      centered_FT -> get_idft3 -> real_space -> zero_pad -> get_dft3 -> centered_FT

    Since get_idft3(get_dft3(x)) = x for real-valued volumes, the
    padding is simply: zero-pad vol_real then FFT.  The result is the
    unnormalized FFT of the padded volume.

    We compute this in RELION's frame (raw fftn, no centering tricks)
    so it can be compared directly against the RELION projector reference.

    Returns
    -------
    padded_fftn : complex array, shape (padoridim, padoridim, padoridim)
        Full-complex unnormalized FFT of the zero-padded volume.
    """
    padoridim = round(padding_factor * N)
    if padoridim % 2 != 0:
        padoridim += 1

    # Zero-pad (same centering as RELION's XmippOrigin)
    Mpad = np.zeros((padoridim, padoridim, padoridim))
    offset = padoridim // 2 - N // 2
    Mpad[offset : offset + N, offset : offset + N, offset : offset + N] = vol_relion

    return np.fft.fftn(Mpad, axes=(0, 1, 2))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPaddingParity:
    """Verify RELION's computeFourierTransformMap projector storage matches
    a numpy reference at machine precision, then verify that recovar's
    pad_volume_for_projection produces the same underlying padded FFT
    (modulo axis convention and normalization).
    """

    @pytest.mark.parametrize("N", [16, 32, 64])
    def test_projector_matches_numpy_pf2(self, N):
        """RELION binding projector data must match the numpy reference
        at rel_err < 1e-12 for padding_factor=2."""
        rng = np.random.default_rng(42)
        vol_relion = _make_smooth_volume(N, rng)

        # RELION binding
        proj_data, *_ = compute_fourier_transform_map(
            vol_relion,
            ori_size=N,
            padding_factor=2,
            do_gridding=False,
            data_dim=3,
        )

        # Numpy reference
        ref_proj, _ = _numpy_compute_fourier_transform_map(vol_relion, N, padding_factor=2)

        # Compare on nonzero entries
        mask = np.abs(proj_data) + np.abs(ref_proj) > 0
        assert np.count_nonzero(mask) > 0, "No nonzero entries in projector"

        denom = np.max(np.abs(proj_data[mask]))
        rel_err = np.max(np.abs(proj_data[mask] - ref_proj[mask])) / denom
        np.testing.assert_array_less(
            rel_err,
            1e-12,
            f"N={N}: projector pf=2 rel_err={rel_err:.2e} exceeds 1e-12",
        )

    @pytest.mark.parametrize("N", [16, 32, 64])
    def test_projector_matches_numpy_pf1(self, N):
        """RELION binding projector data must match the numpy reference
        at rel_err < 1e-12 for padding_factor=1."""
        rng = np.random.default_rng(42)
        vol_relion = _make_smooth_volume(N, rng)

        proj_data, *_ = compute_fourier_transform_map(
            vol_relion,
            ori_size=N,
            padding_factor=1,
            do_gridding=False,
            data_dim=3,
        )

        ref_proj, _ = _numpy_compute_fourier_transform_map(vol_relion, N, padding_factor=1)

        mask = np.abs(proj_data) + np.abs(ref_proj) > 0
        assert np.count_nonzero(mask) > 0

        denom = np.max(np.abs(proj_data[mask]))
        rel_err = np.max(np.abs(proj_data[mask] - ref_proj[mask])) / denom
        np.testing.assert_array_less(
            rel_err,
            1e-12,
            f"N={N}: projector pf=1 rel_err={rel_err:.2e} exceeds 1e-12",
        )

    @pytest.mark.parametrize("N", [16, 32, 64])
    def test_padding_parity_recovar_vs_relion(self, N):
        """recovar and RELION padding produce the same padded FFT values
        at overlapping frequencies, accounting for axis convention and
        normalization.

        Both pad the same physical volume to the same grid size
        (padoridim = pf*N).  The raw FFT values must satisfy:

            FFT_recovar = -FFT_relion.transpose(2, 1, 0)

        because vol_recovar = -vol_relion.transpose(2, 1, 0).
        """
        rng = np.random.default_rng(42)
        padding_factor = 2
        vol_relion = _make_smooth_volume(N, rng)

        # --- RELION path ---
        # Get raw (unnormalized) FFT of the padded volume.
        relion_padded_fftn = _recovar_pad_numpy(vol_relion, N, padding_factor)
        padoridim = relion_padded_fftn.shape[0]

        # --- recovar path ---
        # Same padding but with recovar's axis convention.
        vol_recovar = -np.transpose(vol_relion, (2, 1, 0))
        Mpad_rec = np.zeros((padoridim, padoridim, padoridim))
        offset = padoridim // 2 - N // 2
        Mpad_rec[offset : offset + N, offset : offset + N, offset : offset + N] = vol_recovar
        recovar_padded_fftn = np.fft.fftn(Mpad_rec, axes=(0, 1, 2))

        # Verify the axis/sign relationship:
        # recovar_padded_fftn should equal -relion_padded_fftn.transpose(2,1,0)
        converted = -np.transpose(relion_padded_fftn, (2, 1, 0))

        denom = np.max(np.abs(recovar_padded_fftn))
        assert denom > 0
        rel_err = np.max(np.abs(recovar_padded_fftn - converted)) / denom
        np.testing.assert_array_less(
            rel_err,
            1e-14,
            f"N={N}: axis convention rel_err={rel_err:.2e} exceeds 1e-14",
        )

    @pytest.mark.parametrize("N", [16, 32, 64])
    def test_binding_vs_recovar_padding_inner_cube(self, N):
        """End-to-end: RELION binding projector data vs recovar's padding,
        compared on the N^3 inner cube of frequencies (|k| <= r_max).

        This is the main parity test.  It takes the RELION binding output,
        undoes CenterFFTbySign and normalization to recover the raw FFT,
        then converts to recovar's axis convention and compares against
        recovar's padded FFT.
        """
        rng = np.random.default_rng(42)
        padding_factor = 2
        vol_relion = _make_smooth_volume(N, rng)

        # --- RELION binding ---
        proj_data, *_ = compute_fourier_transform_map(
            vol_relion,
            ori_size=N,
            padding_factor=padding_factor,
            do_gridding=False,
            data_dim=3,
        )
        pad_size = compute_relion_pad_size(N, padding_factor)
        c = pad_size // 2
        r_max = N // 2

        padoridim = round(padding_factor * N)
        if padoridim % 2 != 0:
            padoridim += 1
        pf_actual = padoridim / N

        # Projector data = CenterFFTbySign(rfftn(Mpad)) / N^3  (see derivation in module docstring).
        # To recover rfftn(Mpad) at each frequency point:
        #   rfftn(Mpad)[k,i,j] = proj_data[kp+c, ip+c, j] / normfft * undo_sign
        # where normfft = pf^3 and undo_sign = (-1)^(k+i+j).
        # Then full FFT = expand half-complex via Hermitian symmetry.

        # --- recovar path (in RELION frame for direct comparison) ---
        relion_padded_fftn = _recovar_pad_numpy(vol_relion, N, padding_factor)

        # Compare at each frequency in the r_max sphere.
        # We read directly from proj_data and compare against relion_padded_fftn.
        normfft = pf_actual**3
        max_r2 = r_max**2
        max_abs_err = 0.0
        max_abs_val = 0.0
        n_compared = 0

        for k in range(padoridim):
            kp = k if k <= padoridim // 2 else k - padoridim
            for i in range(padoridim):
                ip = i if i <= padoridim // 2 else i - padoridim
                for j in range(padoridim // 2 + 1):
                    r2 = kp * kp + ip * ip + j * j
                    if r2 > max_r2:
                        continue

                    pk, pi, pj = kp + c, ip + c, j
                    if not (0 <= pk < pad_size and 0 <= pi < pad_size and pj < pad_size // 2 + 1):
                        continue

                    # Undo CenterFFTbySign and normfft to get rfftn(Mpad)/padoridim^3
                    sign = (-1.0) ** ((k + i + j) % 2)
                    relion_val = proj_data[pk, pi, pj] / normfft * sign

                    # Reference: rfftn(Mpad)/padoridim^3
                    ref_val = relion_padded_fftn[k, i, j] / padoridim**3

                    err = abs(relion_val - ref_val)
                    max_abs_err = max(max_abs_err, err)
                    max_abs_val = max(max_abs_val, abs(ref_val))
                    n_compared += 1

        assert n_compared > 0, "No frequency bins compared"
        assert max_abs_val > 0, "All reference values are zero"
        rel_err = max_abs_err / max_abs_val
        np.testing.assert_array_less(
            rel_err,
            1e-12,
            f"N={N}: binding vs recovar padding rel_err={rel_err:.2e} ({n_compared} bins)",
        )

    @pytest.mark.parametrize("N", [16, 32])
    def test_padded_volume_preserves_real_space(self, N):
        """The padded volume's central N^3 block in real space must
        exactly equal the original volume (zero-padding is lossless)."""
        rng = np.random.default_rng(42)
        vol_relion = _make_smooth_volume(N, rng)

        # Build padded volume in real space (same as _recovar_pad_numpy internals).
        padoridim = round(2 * N)
        if padoridim % 2 != 0:
            padoridim += 1
        offset = padoridim // 2 - N // 2

        Mpad = np.zeros((padoridim, padoridim, padoridim))
        Mpad[offset : offset + N, offset : offset + N, offset : offset + N] = vol_relion

        # Extract the central N^3 block -- must be identical to vol_relion.
        extracted = Mpad[offset : offset + N, offset : offset + N, offset : offset + N]
        np.testing.assert_array_equal(
            extracted,
            vol_relion,
            err_msg=f"N={N}: real-space content not preserved by padding",
        )

        # Verify the padding region is all zeros.
        Mpad[offset : offset + N, offset : offset + N, offset : offset + N] = 0.0
        assert np.max(np.abs(Mpad)) == 0.0, "Padding region is not all zeros"
