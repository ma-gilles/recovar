"""Phase 0: Layout validation — round-trip all Fourier layout conversions.

These tests MUST pass before any functional binding test is meaningful.
They verify that the three Fourier representations (FFTW half-complex,
RELION projector-centered, recovar centered full-complex) convert losslessly.
"""

import numpy as np
import pytest

from recovar.relion_bind.conversions import (
    compute_relion_pad_size,
    fftw_half_to_recovar_centered,
    fftw_half_to_relion_projector,
    recovar_centered_to_fftw_half,
    recovar_centered_to_relion_projector,
    recovar_real_to_relion_real,
    relion_projector_to_fftw_half,
    relion_projector_to_recovar_centered,
    relion_real_to_recovar_real,
)

# -----------------------------------------------------------------------
# Pad size formula
# -----------------------------------------------------------------------


class TestPadSize:
    def test_pf1_even(self):
        # N=128, pf=1: r_max=64, pad_size = 2*(64+1)+1 = 131
        assert compute_relion_pad_size(128, 1) == 131

    def test_pf2_even(self):
        # N=128, pf=2: r_max=64, pad_size = 2*(128+1)+1 = 259
        assert compute_relion_pad_size(128, 2) == 259

    def test_pf1_small(self):
        # N=16, pf=1: r_max=8, pad_size = 2*(8+1)+1 = 19
        assert compute_relion_pad_size(16, 1) == 19

    def test_pf1_odd_sizes(self):
        for N in [15, 17, 31, 33, 63, 65]:
            ps = compute_relion_pad_size(N, 1)
            assert ps % 2 == 1, f"pad_size must be odd, got {ps} for N={N}"


# -----------------------------------------------------------------------
# Round-trip: FFTW half <-> RELION projector
# -----------------------------------------------------------------------


class TestFFTWToRelionRoundTrip:
    @pytest.mark.parametrize("pf", [1, 2])
    def test_round_trip(self, random_fftw_half, box_size, pf):
        proj = fftw_half_to_relion_projector(random_fftw_half, padding_factor=pf)
        recovered = relion_projector_to_fftw_half(proj, box_size, padding_factor=pf)
        np.testing.assert_allclose(recovered, random_fftw_half, atol=1e-10)

    def test_projector_shape_pf1(self, random_fftw_half, box_size):
        proj = fftw_half_to_relion_projector(random_fftw_half, padding_factor=1)
        pad_size = compute_relion_pad_size(box_size, 1)
        assert proj.shape == (pad_size, pad_size, pad_size // 2 + 1)

    def test_projector_shape_pf2(self, random_fftw_half, box_size):
        proj = fftw_half_to_relion_projector(random_fftw_half, padding_factor=2)
        pad_size = compute_relion_pad_size(box_size, 2)
        assert proj.shape == (pad_size, pad_size, pad_size // 2 + 1)

    def test_logical_origin_pf1(self, random_fftw_half, box_size):
        """RELION projector logical (0,0,0) at array [pad_size//2, pad_size//2, 0]
        holds F_csign(0,0,0) = sum(vol * (-1)^(i+j+k)), NOT the true DC."""
        from recovar.relion_bind.conversions import _centerfftbysign

        proj = fftw_half_to_relion_projector(random_fftw_half, padding_factor=1)
        pad_size = compute_relion_pad_size(box_size, 1)
        logical_origin = proj[pad_size // 2, pad_size // 2, 0]
        vol = np.fft.irfftn(random_fftw_half, s=(box_size,) * 3)
        sign = _centerfftbysign(box_size)
        expected = np.sum(vol * sign)  # F_csign(0,0,0) = sum(vol * sign)
        np.testing.assert_allclose(logical_origin, expected, atol=1e-8)


# -----------------------------------------------------------------------
# Round-trip: FFTW half <-> recovar centered
# -----------------------------------------------------------------------


class TestFFTWToRecovarRoundTrip:
    def test_round_trip(self, random_fftw_half):
        centered = fftw_half_to_recovar_centered(random_fftw_half)
        recovered = recovar_centered_to_fftw_half(centered)
        np.testing.assert_allclose(recovered, random_fftw_half, atol=1e-10)

    def test_centered_shape(self, random_fftw_half, box_size):
        centered = fftw_half_to_recovar_centered(random_fftw_half)
        assert centered.shape == (box_size, box_size, box_size)

    def test_dc_location(self, random_fftw_half, box_size):
        """DC in recovar centered is at [N//2, N//2, N//2]."""
        centered = fftw_half_to_recovar_centered(random_fftw_half)
        dc_recovar = centered[box_size // 2, box_size // 2, box_size // 2]
        dc_fftw = random_fftw_half[0, 0, 0]
        np.testing.assert_allclose(dc_recovar, dc_fftw, atol=1e-10)


# -----------------------------------------------------------------------
# Round-trip: RELION projector <-> recovar centered (composite)
# -----------------------------------------------------------------------


class TestRelionToRecovarRoundTrip:
    @pytest.mark.parametrize("pf", [1, 2])
    def test_round_trip_via_fftw(self, random_fftw_half, box_size, pf):
        """FFTW -> RELION -> recovar -> RELION -> FFTW should round-trip."""
        proj1 = fftw_half_to_relion_projector(random_fftw_half, padding_factor=pf)
        centered = relion_projector_to_recovar_centered(proj1, box_size, padding_factor=pf)
        proj2 = recovar_centered_to_relion_projector(centered, padding_factor=pf)
        recovered = relion_projector_to_fftw_half(proj2, box_size, padding_factor=pf)
        np.testing.assert_allclose(recovered, random_fftw_half, atol=1e-10)

    def test_dc_in_recovar_centered(self, random_fftw_half, box_size):
        """True DC in recovar centered is at [N//2, N//2, N//2]."""
        dc_fftw = random_fftw_half[0, 0, 0]
        centered = fftw_half_to_recovar_centered(random_fftw_half)
        dc_recovar = centered[box_size // 2, box_size // 2, box_size // 2]
        np.testing.assert_allclose(dc_recovar, dc_fftw, atol=1e-10)

    def test_relion_logical_origin_is_not_true_dc(self, random_fftw_half, box_size):
        """RELION projector logical (0,0,0) holds F_csign(0,0,0), not sum(vol).
        This is by design — CenterFFTbySign shifts all frequencies by N/2."""
        from recovar.relion_bind.conversions import _centerfftbysign

        proj = fftw_half_to_relion_projector(random_fftw_half, padding_factor=1)
        pad_size = compute_relion_pad_size(box_size, 1)
        logical_origin = proj[pad_size // 2, pad_size // 2, 0]

        vol = np.fft.irfftn(random_fftw_half, s=(box_size,) * 3)
        sign = _centerfftbysign(box_size)
        csign_dc = np.sum(vol * sign)
        np.testing.assert_allclose(logical_origin, csign_dc, atol=1e-8)


# -----------------------------------------------------------------------
# Real-space convention
# -----------------------------------------------------------------------


class TestRealSpaceConvention:
    def test_round_trip(self, random_real_volume):
        """relion -> recovar -> relion should round-trip."""
        recovar_vol = relion_real_to_recovar_real(random_real_volume)
        recovered = recovar_real_to_relion_real(recovar_vol)
        np.testing.assert_allclose(recovered, random_real_volume, atol=1e-15)

    def test_involution(self, random_real_volume):
        """The transform is its own inverse."""
        once = relion_real_to_recovar_real(random_real_volume)
        twice = relion_real_to_recovar_real(once)
        np.testing.assert_allclose(twice, random_real_volume, atol=1e-15)

    def test_transpose_and_negate(self, random_real_volume):
        """Verify the explicit formula."""
        expected = -np.transpose(random_real_volume, (2, 1, 0))
        result = relion_real_to_recovar_real(random_real_volume)
        np.testing.assert_array_equal(result, expected)


# -----------------------------------------------------------------------
# Nyquist shell handling
# -----------------------------------------------------------------------


class TestNyquistShell:
    def test_nyquist_preserved_pf1(self, random_fftw_half, box_size):
        """RELION includes r_max = ori_size//2; verify Nyquist shell survives round-trip."""
        proj = fftw_half_to_relion_projector(random_fftw_half, padding_factor=1)
        recovered = relion_projector_to_fftw_half(proj, box_size, padding_factor=1)
        # The Nyquist plane (kx = N//2) should survive
        np.testing.assert_allclose(
            recovered[:, :, box_size // 2],
            random_fftw_half[:, :, box_size // 2],
            atol=1e-10,
        )

    def test_signal_energy_preserved(self, random_fftw_half, box_size):
        """Total signal energy should be preserved through all conversions."""
        # Energy in FFTW half: need to account for Hermitian doubling
        energy_fftw = np.sum(np.abs(random_fftw_half[:, :, 0]) ** 2)
        energy_fftw += 2 * np.sum(np.abs(random_fftw_half[:, :, 1 : box_size // 2]) ** 2)
        if box_size % 2 == 0:
            energy_fftw += np.sum(np.abs(random_fftw_half[:, :, box_size // 2]) ** 2)

        centered = fftw_half_to_recovar_centered(random_fftw_half)
        energy_centered = np.sum(np.abs(centered) ** 2)

        np.testing.assert_allclose(energy_centered, energy_fftw, rtol=1e-10)


# -----------------------------------------------------------------------
# CenterFFTbySign verification
# -----------------------------------------------------------------------


class TestCenterFFTbySign:
    def test_sign_pattern_matches_relion(self, box_size):
        """Verify our CenterFFTbySign matches RELION's ((k^i^j)&1) pattern."""
        from recovar.relion_bind.conversions import _centerfftbysign

        sign = _centerfftbysign(box_size)
        for k in range(box_size):
            for i in range(min(box_size, 4)):  # spot-check
                for j in range(min(box_size, 4)):
                    expected = -1.0 if ((k + i + j) % 2 != 0) else 1.0
                    assert sign[k, i, j] == expected, f"Mismatch at ({k},{i},{j})"

    def test_fftw_to_relion_logical_origin(self, box_size, rng):
        """Logical (0,0,0) in projector holds F_csign(0,0,0) = sum(vol*sign)."""
        from recovar.relion_bind.conversions import _centerfftbysign

        vol = rng.standard_normal((box_size, box_size, box_size))
        fftw_half = np.fft.rfftn(vol)
        proj = fftw_half_to_relion_projector(fftw_half, padding_factor=1)
        pad_size = compute_relion_pad_size(box_size, 1)

        logical_origin = proj[pad_size // 2, pad_size // 2, 0]
        sign = _centerfftbysign(box_size)
        expected = np.sum(vol * sign)
        np.testing.assert_allclose(logical_origin, expected, atol=1e-8)
