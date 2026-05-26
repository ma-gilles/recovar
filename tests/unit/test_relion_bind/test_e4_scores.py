"""Phase 3 (E4): Composite E-step scoring parity.

Validate that RELION's diff2 formula and recovar's cross-term + norm
formula produce identical scores for the same (image, projection, CTF,
translation, noise) inputs.

RELION diff2:
    d2[i,r,t] = Σ_k (1/σ²_k) |y_k·e^{-2πi k·t} − CTF_k · proj_k|²
              = ||y/√σ²||² − 2 Re⟨CTF·y/σ², shifted_proj⟩ + ||CTF·proj/√σ²||²

recovar cross-term:
    score[i,r,t] = −2 Re⟨conj(S_t(CTF·y/σ²)), proj⟩ + ||y/√σ²||²

So:  diff2 = score + ||CTF·proj/√σ²||²  (the proj-norm term)

Tests verify this algebraic identity for multiple rotations × translations.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import (
    TRILINEAR,
    euler_angles_to_matrix,
    get_coarse_orientations,
    get_ctf_image,
    project_volume,
    shift_image_in_fourier_transform_2d,
)


def _make_test_volume(N, rng):
    """Low-pass filtered random volume."""
    vol = rng.standard_normal((N, N, N))
    ft = np.fft.rfftn(vol, axes=(0, 1, 2))
    idx = np.arange(N) - N // 2
    kz, ky = np.meshgrid(idx, idx, indexing="ij")
    kx = np.arange(N // 2 + 1)
    r2 = kz[:, :, np.newaxis] ** 2 + ky[:, :, np.newaxis] ** 2 + kx[np.newaxis, np.newaxis, :] ** 2
    ft *= np.exp(-r2 / (2 * (N / 4) ** 2))
    return np.fft.irfftn(ft, s=(N, N, N), axes=(0, 1, 2))


def _shell_indices_fftw(N):
    """Return per-pixel shell index in FFTW half-complex layout (N, N//2+1)."""
    ky = np.arange(N)
    ky = np.where(ky <= N // 2, ky, ky - N)
    kx = np.arange(N // 2 + 1)
    ky2d, kx2d = np.meshgrid(ky, kx, indexing="ij")
    return np.round(np.sqrt(ky2d**2 + kx2d**2)).astype(int)


def _compute_diff2_relion(shifted_img, ctf, proj, sigma2, shell_idx, max_shell):
    """Compute RELION's ||shifted_img - CTF*proj||²/σ² per pixel, summed."""
    residual = shifted_img - ctf * proj
    weights = np.zeros_like(ctf)
    for s in range(max_shell + 1):
        mask = shell_idx == s
        if sigma2[s] > 0:
            weights[mask] = 1.0 / sigma2[s]
    return np.sum(weights * np.abs(residual) ** 2)


class TestDiff2Formula:
    """Verify that RELION diff2 = recovar score + CTFed proj norm."""

    def _make_scenario(self, N, rng):
        vol = _make_test_volume(N, rng)
        ctf = get_ctf_image(
            defU=15000.0,
            defV=14000.0,
            defAng=30.0,
            voltage=300.0,
            Cs=2.7,
            Q0=0.07,
            Bfac=0.0,
            angpix=1.5,
            orixdim=N,
            oriydim=N,
            do_ctf_padding=False,
            do_abs=False,
            do_damping=False,
            phase_shift=0.0,
            scale=1.0,
        )
        shell_idx = _shell_indices_fftw(N)
        max_shell = N // 2
        sigma2 = np.ones(max_shell + 1) * 0.01
        sigma2[0] = 1e10
        return vol, ctf, shell_idx, max_shell, sigma2

    @pytest.mark.parametrize("N", [32, 64])
    def test_diff2_decomposition(self, N):
        """diff2 = cross_term + batch_norm + proj_norm."""
        rng = np.random.default_rng(42)
        vol, ctf, shell_idx, max_shell, sigma2 = self._make_scenario(N, rng)

        A = euler_angles_to_matrix(45.0, 30.0, 60.0)
        proj = project_volume(
            vol, A, ori_size=N, padding_factor=2, interpolator=TRILINEAR, current_size=-1, do_gridding=True
        )

        img = ctf * proj + rng.standard_normal(proj.shape) * 0.1
        xshift, yshift = 1.3, -0.7

        shifted_img = shift_image_in_fourier_transform_2d(
            img,
            float(N),
            N,
            xshift,
            yshift,
        )

        diff2 = _compute_diff2_relion(shifted_img, ctf, proj, sigma2, shell_idx, max_shell)

        # Decompose: diff2 = batch_norm + cross_term + proj_norm
        weights = np.zeros_like(ctf)
        for s in range(max_shell + 1):
            mask = shell_idx == s
            if sigma2[s] > 0:
                weights[mask] = 1.0 / sigma2[s]

        batch_norm = np.sum(weights * np.abs(shifted_img) ** 2)
        cross_term = -2.0 * np.sum(weights * np.real(np.conj(shifted_img) * ctf * proj))
        proj_norm = np.sum(weights * ctf**2 * np.abs(proj) ** 2)

        reconstructed = batch_norm + cross_term + proj_norm
        rel_err = abs(diff2 - reconstructed) / (abs(diff2) + 1e-30)
        print(f"\nN={N}: diff2={diff2:.6e}, reconstructed={reconstructed:.6e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-12, f"Decomposition error: {rel_err:.2e}"


class TestScoreMatrixParity:
    """Compare full score matrices: RELION pipeline vs recovar-style formula."""

    def test_multi_rotation_multi_translation(self):
        """Score matrix over 12 rotations × 5 translations matches."""
        N = 32
        rng = np.random.default_rng(123)
        vol = _make_test_volume(N, rng)

        ctf = get_ctf_image(
            defU=12000.0,
            defV=12000.0,
            defAng=0.0,
            voltage=300.0,
            Cs=2.7,
            Q0=0.07,
            Bfac=0.0,
            angpix=1.5,
            orixdim=N,
            oriydim=N,
            do_ctf_padding=False,
            do_abs=False,
            do_damping=False,
            phase_shift=0.0,
            scale=1.0,
        )

        shell_idx = _shell_indices_fftw(N)
        max_shell = N // 2
        sigma2 = np.ones(max_shell + 1) * 0.05
        sigma2[0] = 1e10

        orientations = get_coarse_orientations(1)
        n_rot = min(12, orientations.shape[0])
        true_trans = np.array([1.0, 0.0])
        translations = np.array([[0, 0], true_trans, [0, 1], [-1.5, 0.5], [2.0, -1.0]])
        n_trans = translations.shape[0]

        # Build projections for each rotation
        projections = []
        for irot in range(n_rot):
            rot, tilt, psi = orientations[irot]
            A = euler_angles_to_matrix(float(rot), float(tilt), float(psi))
            p = project_volume(
                vol, A, ori_size=N, padding_factor=2, interpolator=TRILINEAR, current_size=-1, do_gridding=True
            )
            projections.append(p)

        # Create synthetic image from rotation 3 with a grid-aligned shift
        true_rot = 3
        true_proj = projections[true_rot]
        img_clean = ctf * true_proj
        img = shift_image_in_fourier_transform_2d(
            img_clean,
            float(N),
            N,
            true_trans[0],
            true_trans[1],
        )
        # Noiseless — decomposition test is exact; rotation-finding is a sanity check

        weights = np.zeros_like(ctf)
        for s in range(max_shell + 1):
            mask = shell_idx == s
            if sigma2[s] > 0:
                weights[mask] = 1.0 / sigma2[s]

        # Score matrix: diff2[rot, trans]
        diff2_direct = np.zeros((n_rot, n_trans))
        batch_norm = np.sum(weights * np.abs(img) ** 2)
        cross_term = np.zeros((n_rot, n_trans))
        proj_norm = np.zeros(n_rot)

        for irot in range(n_rot):
            proj_norm[irot] = np.sum(weights * ctf**2 * np.abs(projections[irot]) ** 2)
            for itrans in range(n_trans):
                shifted = shift_image_in_fourier_transform_2d(
                    img,
                    float(N),
                    N,
                    translations[itrans, 0],
                    translations[itrans, 1],
                )
                residual = shifted - ctf * projections[irot]
                diff2_direct[irot, itrans] = np.sum(weights * np.abs(residual) ** 2)
                cross_term[irot, itrans] = -2.0 * np.sum(weights * np.real(np.conj(shifted) * ctf * projections[irot]))

        # Verify decomposition: diff2 = batch_norm + cross + proj_norm
        diff2_decomposed = batch_norm + cross_term + proj_norm[:, np.newaxis]
        max_err = np.max(np.abs(diff2_direct - diff2_decomposed))
        rel_err = max_err / (np.max(np.abs(diff2_direct)) + 1e-30)
        print(f"\nScore matrix: {n_rot}×{n_trans}, max_err={max_err:.2e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-12

        # Sanity: true rotation should be among the top-3 scorers
        # (order-1 HEALPix is ~30° coarse, so neighbors can win)
        best_per_rot = np.min(diff2_direct, axis=1)
        rank = np.argsort(best_per_rot)
        true_rank = np.where(rank == true_rot)[0][0]
        print(f"  True rot rank: {true_rank}/{n_rot} (0=best)")
        assert true_rank < 3, f"True rotation ranked {true_rank} — too far from best"

    def test_batch_norm_is_translation_invariant(self):
        """||shifted_img||²/σ² = ||img||²/σ² (Parseval's theorem)."""
        N = 32
        rng = np.random.default_rng(77)
        img = rng.standard_normal((N, N // 2 + 1)) + 1j * rng.standard_normal((N, N // 2 + 1))

        shell_idx = _shell_indices_fftw(N)
        sigma2 = np.ones(N // 2 + 1) * 0.1
        weights = np.zeros_like(shell_idx, dtype=float)
        for s in range(N // 2 + 1):
            mask = shell_idx == s
            if sigma2[s] > 0:
                weights[mask] = 1.0 / sigma2[s]

        norm_orig = np.sum(weights * np.abs(img) ** 2)

        for xshift, yshift in [(2.0, 0.0), (0.0, -3.0), (1.5, 2.5)]:
            shifted = shift_image_in_fourier_transform_2d(
                img,
                float(N),
                N,
                xshift,
                yshift,
            )
            norm_shifted = np.sum(weights * np.abs(shifted) ** 2)
            rel = abs(norm_orig - norm_shifted) / (abs(norm_orig) + 1e-30)
            assert rel < 1e-12, f"Shift ({xshift},{yshift}): Parseval violation {rel:.2e}"


class TestCrossTermVsRecovarFormula:
    """Verify that the cross-term computed from RELION components matches
    recovar's GEMM-based formula: -2 Re⟨conj(CTF·y/σ² shifted), proj⟩.

    recovar pre-multiplies the image by CTF/σ², then shifts, then does
    a single dot product against the projection. This is algebraically
    equivalent to the per-pixel formula but operationally different.
    """

    def test_cross_term_equivalence(self):
        N = 32
        rng = np.random.default_rng(55)
        vol = _make_test_volume(N, rng)

        ctf = get_ctf_image(
            defU=10000.0,
            defV=10000.0,
            defAng=0.0,
            voltage=300.0,
            Cs=2.7,
            Q0=0.07,
            Bfac=0.0,
            angpix=1.5,
            orixdim=N,
            oriydim=N,
            do_ctf_padding=False,
            do_abs=False,
            do_damping=False,
            phase_shift=0.0,
            scale=1.0,
        )

        shell_idx = _shell_indices_fftw(N)
        max_shell = N // 2
        sigma2 = np.ones(max_shell + 1) * 0.02
        sigma2[0] = 1e10

        weights = np.zeros_like(ctf)
        for s in range(max_shell + 1):
            mask = shell_idx == s
            if sigma2[s] > 0:
                weights[mask] = 1.0 / sigma2[s]

        A = euler_angles_to_matrix(90.0, 45.0, -30.0)
        proj = project_volume(
            vol, A, ori_size=N, padding_factor=2, interpolator=TRILINEAR, current_size=-1, do_gridding=True
        )
        img = (rng.standard_normal(proj.shape) + 1j * rng.standard_normal(proj.shape)) * 0.5
        xshift, yshift = 1.2, -0.8

        # Method 1: shift image, then compute cross-term per-pixel
        shifted = shift_image_in_fourier_transform_2d(
            img,
            float(N),
            N,
            xshift,
            yshift,
        )
        cross_perpixel = -2.0 * np.sum(weights * np.real(np.conj(shifted) * ctf * proj))

        # Method 2: recovar-style pre-multiply then dot product
        img_premult = ctf * img * weights
        shifted_premult = shift_image_in_fourier_transform_2d(
            img_premult,
            float(N),
            N,
            xshift,
            yshift,
        )
        cross_gemm = -2.0 * np.sum(np.real(np.conj(shifted_premult) * proj))

        rel_err = abs(cross_perpixel - cross_gemm) / (abs(cross_perpixel) + 1e-30)
        print(f"\nCross-term: per-pixel={cross_perpixel:.6e}, GEMM={cross_gemm:.6e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-12, f"Cross-term formula mismatch: {rel_err:.2e}"
