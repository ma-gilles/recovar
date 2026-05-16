"""Phase 2 (P1): Compare RELION's CTF::getFftwImage against recovar's evaluate_ctf.

Sign convention: CTF_relion = -CTF_recovar.
RELION uses CTF = -sin(gamma) where amplitude contrast is folded into gamma
via K3 = atan(Q0/sqrt(1-Q0^2)).  recovar uses CTF = sqrt(1-w^2)*sin(gamma) -
w*cos(gamma) with no K3 in gamma.  Combined with the volume sign flip
(vol_recovar = -transpose(vol_relion)), products CTF*proj are identical.

Tests:
1. Unpadded CTF: verify CTF_relion = -CTF_recovar (up to lambda constant diff ~1e-6)
2. Padded CTF: quantify the 2x-padding divergence
3. Single-frequency spot checks
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import get_ctf_image, get_ctf_value

jax = pytest.importorskip("jax")
jnp = jax.numpy

from recovar.core.ctf import evaluate_ctf


def _build_fftw_half_freq_grid(orixdim, oriydim, angpix):
    """Build RELION-convention FFTW half-complex frequency grid.

    Returns shape (oriydim * (orixdim//2+1), 2) with (kx, ky) in 1/A.
    """
    nx = orixdim // 2 + 1
    ny = oriydim

    ix = np.arange(nx, dtype=np.float64)
    iy = np.arange(ny, dtype=np.float64)
    iy[iy > ny // 2] -= ny

    kx = ix / (orixdim * angpix)
    ky = iy / (oriydim * angpix)

    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    freqs = np.stack([KX.ravel(), KY.ravel()], axis=-1)
    return freqs


def _recovar_ctf_2d(params, orixdim, oriydim, angpix):
    """Evaluate recovar CTF on FFTW half-complex grid, return (oriydim, orixdim//2+1)."""
    freqs = _build_fftw_half_freq_grid(orixdim, oriydim, angpix)
    ctf_params = np.array(
        [
            [
                params["defU"],
                params["defV"],
                params["defAng"],
                params["voltage"],
                params["Cs"],
                params["Q0"],
                params.get("phase_shift", 0.0),
                params.get("Bfac", 0.0),
                params.get("contrast", 1.0),
            ]
        ],
        dtype=np.float64,
    )
    ctf_flat = np.array(evaluate_ctf(jnp.array(freqs), jnp.array(ctf_params)))
    return ctf_flat.reshape(oriydim, orixdim // 2 + 1)


PARAMS = dict(
    defU=10000.0,
    defV=12000.0,
    defAng=45.0,
    voltage=300.0,
    Cs=2.7,
    Q0=0.1,
    angpix=1.5,
    orixdim=128,
    oriydim=128,
)

# Electron wavelength constants differ: RELION 12.2643247 vs recovar 12.2642598.
# At high frequencies the ~5e-6 relative lambda error amplifies through the
# rapidly oscillating sin(gamma) → max |diff| ~ 7e-4 at shell 64 for N=128.
SIGN_CONVENTION_TOL = 1e-3


class TestCTFSignConvention:
    """Verify CTF_relion = -CTF_recovar (sign convention documented above)."""

    def test_ctf_sign_flip_unpadded(self):
        p = PARAMS
        relion_ctf = get_ctf_image(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            angpix=p["angpix"],
            orixdim=p["orixdim"],
            oriydim=p["oriydim"],
            do_ctf_padding=False,
            do_abs=False,
            do_damping=False,
        )
        recovar_ctf = _recovar_ctf_2d(p, p["orixdim"], p["oriydim"], p["angpix"])

        # CTF_relion should equal -CTF_recovar
        max_diff = np.max(np.abs(relion_ctf - (-recovar_ctf)))
        print(f"\nSign-flip parity: max |relion - (-recovar)| = {max_diff:.2e}")
        assert max_diff < SIGN_CONVENTION_TOL, f"CTF sign convention violated: max |relion + recovar| = {max_diff:.2e}"

    def test_ctf_sign_flip_with_bfactor(self):
        p = {**PARAMS, "Bfac": 50.0}
        relion_ctf = get_ctf_image(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            Bfac=p["Bfac"],
            angpix=p["angpix"],
            orixdim=p["orixdim"],
            oriydim=p["oriydim"],
            do_ctf_padding=False,
            do_abs=False,
            do_damping=True,
        )
        recovar_ctf = _recovar_ctf_2d(p, p["orixdim"], p["oriydim"], p["angpix"])

        max_diff = np.max(np.abs(relion_ctf - (-recovar_ctf)))
        print(f"\nB-factor sign-flip parity: max diff = {max_diff:.2e}")
        assert max_diff < SIGN_CONVENTION_TOL

    def test_ctf_sign_flip_strong_astigmatism(self):
        p = {**PARAMS, "defU": 8000.0, "defV": 15000.0, "defAng": 30.0}
        relion_ctf = get_ctf_image(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            angpix=p["angpix"],
            orixdim=p["orixdim"],
            oriydim=p["oriydim"],
            do_ctf_padding=False,
            do_abs=False,
            do_damping=False,
        )
        recovar_ctf = _recovar_ctf_2d(p, p["orixdim"], p["oriydim"], p["angpix"])

        max_diff = np.max(np.abs(relion_ctf - (-recovar_ctf)))
        print(f"\nAstigmatic sign-flip parity: max diff = {max_diff:.2e}")
        assert max_diff < SIGN_CONVENTION_TOL

    def test_abs_ctf_matches(self):
        """With do_abs=True, |CTF| should match regardless of sign convention."""
        p = PARAMS
        relion_abs = get_ctf_image(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            angpix=p["angpix"],
            orixdim=p["orixdim"],
            oriydim=p["oriydim"],
            do_ctf_padding=False,
            do_abs=True,
            do_damping=False,
        )
        recovar_ctf = _recovar_ctf_2d(p, p["orixdim"], p["oriydim"], p["angpix"])

        max_diff = np.max(np.abs(relion_abs - np.abs(recovar_ctf)))
        print(f"\n|CTF| parity: max diff = {max_diff:.2e}")
        assert max_diff < SIGN_CONVENTION_TOL


class TestCTFSingleValue:
    """Spot-check get_ctf_value vs -recovar at individual frequencies."""

    @pytest.mark.parametrize(
        "freq_x,freq_y",
        [
            (0.01, 0.0),
            (0.0, 0.02),
            (0.015, 0.015),
            (0.03, -0.01),
        ],
    )
    def test_single_frequency_sign_flip(self, freq_x, freq_y):
        p = PARAMS
        relion_val = get_ctf_value(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            angpix=p["angpix"],
            freq_x=freq_x,
            freq_y=freq_y,
        )

        ctf_params = np.array(
            [
                [
                    p["defU"],
                    p["defV"],
                    p["defAng"],
                    p["voltage"],
                    p["Cs"],
                    p["Q0"],
                    0.0,
                    0.0,
                    1.0,
                ]
            ],
            dtype=np.float64,
        )
        freqs = np.array([[freq_x, freq_y]], dtype=np.float64)
        recovar_val = float(evaluate_ctf(jnp.array(freqs), jnp.array(ctf_params))[0, 0])

        # RELION clamps |CTF| >= 1e-8, so skip DC
        diff = abs(relion_val - (-recovar_val))
        print(f"  ({freq_x}, {freq_y}): relion={relion_val:+.10f}, -recovar={-recovar_val:+.10f}, diff={diff:.2e}")
        assert diff < SIGN_CONVENTION_TOL, f"Sign-flip mismatch at ({freq_x}, {freq_y}): diff={diff:.2e}"

    def test_dc_clamped(self):
        """At DC, RELION clamps |CTF| >= 1e-8 while recovar returns -Q0."""
        p = PARAMS
        relion_dc = get_ctf_value(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            freq_x=0.0,
            freq_y=0.0,
        )
        ctf_params = np.array(
            [
                [
                    p["defU"],
                    p["defV"],
                    p["defAng"],
                    p["voltage"],
                    p["Cs"],
                    p["Q0"],
                    0.0,
                    0.0,
                    1.0,
                ]
            ],
            dtype=np.float64,
        )
        recovar_dc = float(evaluate_ctf(jnp.array([[0.0, 0.0]]), jnp.array(ctf_params))[0, 0])

        print(f"\n  DC: relion={relion_dc:.10f}, recovar={recovar_dc:.10f}")
        # RELION: -sin(-K3) = sin(K3) = Q0 = 0.1, but clamped to max(|val|, 1e-8)*sign
        # recovar: sqrt(1-w^2)*sin(0) - w*cos(0) = -w = -0.1
        # Sign flip: relion ≈ +0.1, recovar = -0.1
        assert abs(relion_dc - (-recovar_dc)) < SIGN_CONVENTION_TOL


class TestCTFPadding:
    """Quantify RELION's 2x CTF padding divergence (known gap source)."""

    def test_padding_divergence_magnitude(self):
        """CTF padding produces measurable divergence from unpadded."""
        p = PARAMS
        ctf_nopad = get_ctf_image(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            angpix=p["angpix"],
            orixdim=p["orixdim"],
            oriydim=p["oriydim"],
            do_ctf_padding=False,
        )
        ctf_pad = get_ctf_image(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            angpix=p["angpix"],
            orixdim=p["orixdim"],
            oriydim=p["oriydim"],
            do_ctf_padding=True,
        )

        diff = np.abs(ctf_pad - ctf_nopad)
        max_diff = diff.max()
        mean_diff = diff.mean()
        rms_diff = np.sqrt(np.mean(diff**2))

        print(f"\n=== CTF Padding Divergence (box={p['orixdim']}, angpix={p['angpix']}) ===")
        print(f"  Max  |pad - nopad|: {max_diff:.6f}")
        print(f"  Mean |pad - nopad|: {mean_diff:.6f}")
        print(f"  RMS  |pad - nopad|: {rms_diff:.6f}")
        print(f"  Padded range: [{ctf_pad.min():.4f}, {ctf_pad.max():.4f}]")
        print(f"  Unpadded range: [{ctf_nopad.min():.4f}, {ctf_nopad.max():.4f}]")

        assert max_diff > 0.01, "Expected measurable CTF padding divergence"

    def test_padding_divergence_per_shell(self):
        """Shell-by-shell divergence between padded and unpadded CTF."""
        p = PARAMS
        ctf_nopad = get_ctf_image(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            angpix=p["angpix"],
            orixdim=p["orixdim"],
            oriydim=p["oriydim"],
            do_ctf_padding=False,
        )
        ctf_pad = get_ctf_image(
            p["defU"],
            p["defV"],
            p["defAng"],
            p["voltage"],
            p["Cs"],
            p["Q0"],
            angpix=p["angpix"],
            orixdim=p["orixdim"],
            oriydim=p["oriydim"],
            do_ctf_padding=True,
        )

        ny, nx = ctf_nopad.shape
        orixdim = p["orixdim"]

        ix = np.arange(nx)
        iy = np.arange(ny, dtype=np.float64)
        iy[iy > ny // 2] -= ny
        IX, IY = np.meshgrid(ix, iy)
        radii = np.sqrt(IX**2 + IY**2)
        shell = np.round(radii).astype(int)

        diff = np.abs(ctf_pad - ctf_nopad)
        r_max = orixdim // 2
        shell_rms = np.zeros(r_max + 1)
        shell_max = np.zeros(r_max + 1)
        for r in range(r_max + 1):
            mask = shell == r
            if mask.any():
                shell_rms[r] = np.sqrt(np.mean(diff[mask] ** 2))
                shell_max[r] = diff[mask].max()

        print("\n=== Per-Shell CTF Padding Divergence ===")
        print(f"{'Shell':>6} {'RMS diff':>10} {'Max diff':>10} {'Res (A)':>10}")
        for r in range(min(r_max + 1, 20)):
            res = orixdim * p["angpix"] / max(r, 1) if r > 0 else float("inf")
            print(f"{r:6d} {shell_rms[r]:10.6f} {shell_max[r]:10.6f} {res:10.2f}")
        print("  ... (showing first 20 shells)")

        high_shell_rms = shell_rms[5:r_max].mean()
        print(f"\nMean RMS diff shells 5-{r_max}: {high_shell_rms:.6f}")

    @pytest.mark.parametrize("box_size", [64, 128, 256])
    def test_padding_divergence_vs_box_size(self, box_size):
        """Padding divergence at different box sizes."""
        ctf_nopad = get_ctf_image(
            PARAMS["defU"],
            PARAMS["defV"],
            PARAMS["defAng"],
            PARAMS["voltage"],
            PARAMS["Cs"],
            PARAMS["Q0"],
            angpix=PARAMS["angpix"],
            orixdim=box_size,
            oriydim=box_size,
            do_ctf_padding=False,
        )
        ctf_pad = get_ctf_image(
            PARAMS["defU"],
            PARAMS["defV"],
            PARAMS["defAng"],
            PARAMS["voltage"],
            PARAMS["Cs"],
            PARAMS["Q0"],
            angpix=PARAMS["angpix"],
            orixdim=box_size,
            oriydim=box_size,
            do_ctf_padding=True,
        )

        diff = np.abs(ctf_pad - ctf_nopad)
        print(f"\nBox {box_size}: max={diff.max():.6f}, rms={np.sqrt(np.mean(diff**2)):.6f}")
