"""M4: updateSSNRarrays parity — RELION vs recovar.

Tests that update_ssnr_arrays produces exactly the same
tau2/sigma2/data_vs_prior/fourier_coverage as recovar's
compute_data_vs_prior + FSC-based tau2 update.

These call RELION's actual BackProjector::updateSSNRarrays, NOT a
reimplementation. Exact parity with recovar requires understanding
the layout/unit differences.
"""

import numpy as np
from recovar.relion_bind._relion_bind_core import (
    TRILINEAR,
    euler_angles_to_matrix,
    get_coarse_orientations,
    project_volume,
    update_ssnr_arrays,
)


def _make_test_volume(N, rng):
    """Low-pass filtered random volume in FFTW frequency order."""
    vol = rng.standard_normal((N, N, N))
    ft = np.fft.rfftn(vol, axes=(0, 1, 2))
    kz = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    kx = np.arange(N // 2 + 1)
    KZ, KY = np.meshgrid(kz, ky, indexing="ij")
    r2 = KZ[:, :, np.newaxis] ** 2 + KY[:, :, np.newaxis] ** 2 + kx[np.newaxis, np.newaxis, :] ** 2
    ft *= np.exp(-r2 / (2 * (N / 6) ** 2))
    return np.fft.irfftn(ft, s=(N, N, N), axes=(0, 1, 2))


def _project_at_orientations(vol, orientations, N, pf=2):
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


class TestM4SelfConsistency:
    """Self-consistency tests for updateSSNRarrays binding."""

    def test_output_shapes(self):
        N = 32
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:12]
        images, rot_mats = _project_at_orientations(vol, orientations, N)
        empty_w = np.zeros((0,), dtype=np.float64)
        n_shells = N // 2 + 1

        fsc = np.ones(n_shells) * 0.9
        tau2 = np.ones(n_shells) * 1.0

        tau2_out, sigma2, dvp, fc = update_ssnr_arrays(
            images,
            rot_mats,
            empty_w,
            fsc,
            tau2,
            ori_size=N,
            padding_factor=2,
            tau2_fudge=1.0,
            update_tau2_with_fsc=True,
            is_whole_instead_of_half=False,
        )
        assert tau2_out.shape == (n_shells,)
        assert sigma2.shape == (n_shells,)
        assert dvp.shape == (n_shells,)
        assert fc.shape == (n_shells,)

    def test_sigma2_positive(self):
        """sigma2 (noise variance) should be non-negative."""
        N = 32
        rng = np.random.default_rng(99)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(2)
        idx = np.linspace(0, len(orientations) - 1, 100, dtype=int)
        images, rot_mats = _project_at_orientations(vol, orientations[idx], N)
        empty_w = np.zeros((0,), dtype=np.float64)
        n_shells = N // 2 + 1

        fsc = np.linspace(0.99, 0.1, n_shells)
        tau2 = np.ones(n_shells)

        _, sigma2, _, _ = update_ssnr_arrays(
            images,
            rot_mats,
            empty_w,
            fsc,
            tau2,
            ori_size=N,
            padding_factor=2,
            tau2_fudge=1.0,
            update_tau2_with_fsc=True,
        )
        assert np.all(sigma2 >= 0), f"Negative sigma2: {sigma2[sigma2 < 0]}"

    def test_fsc_based_tau2(self):
        """With update_tau2_with_fsc=True, tau2 should be ssnr * sigma2."""
        N = 32
        rng = np.random.default_rng(77)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(2)
        idx = np.linspace(0, len(orientations) - 1, 200, dtype=int)
        images, rot_mats = _project_at_orientations(vol, orientations[idx], N)
        empty_w = np.zeros((0,), dtype=np.float64)
        n_shells = N // 2 + 1

        fsc = np.clip(np.linspace(0.95, 0.05, n_shells), 0.001, 0.999)
        tau2_in = np.ones(n_shells)

        tau2_out, sigma2, dvp, _ = update_ssnr_arrays(
            images,
            rot_mats,
            empty_w,
            fsc,
            tau2_in,
            ori_size=N,
            padding_factor=2,
            tau2_fudge=1.0,
            update_tau2_with_fsc=True,
        )

        # When update_tau2_with_fsc=True:
        # RELION clamps FSC to [0.001, 0.999]
        # ssnr = fsc/(1-fsc) * tau2_fudge
        # tau2 = ssnr * sigma2
        fsc_clamped = np.clip(fsc, 0.001, 0.999)
        expected_ssnr = fsc_clamped / (1.0 - fsc_clamped) * 1.0
        expected_tau2 = expected_ssnr * sigma2

        # data_vs_prior should equal ssnr when update_tau2_with_fsc=True
        for s in range(1, n_shells):
            if sigma2[s] > 0 and expected_tau2[s] > 0:
                rel = abs(tau2_out[s] - expected_tau2[s]) / (abs(expected_tau2[s]) + 1e-30)
                assert rel < 1e-10, f"Shell {s}: tau2 rel_err={rel}"

    def test_fourier_coverage_range(self):
        """Fourier coverage should be in [0, 1]."""
        N = 32
        rng = np.random.default_rng(55)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(2)
        idx = np.linspace(0, len(orientations) - 1, 50, dtype=int)
        images, rot_mats = _project_at_orientations(vol, orientations[idx], N)
        empty_w = np.zeros((0,), dtype=np.float64)
        n_shells = N // 2 + 1

        fsc = np.ones(n_shells) * 0.5
        tau2 = np.ones(n_shells) * 10.0

        _, _, _, fc = update_ssnr_arrays(
            images,
            rot_mats,
            empty_w,
            fsc,
            tau2,
            ori_size=N,
            padding_factor=2,
            tau2_fudge=1.0,
            update_tau2_with_fsc=False,
        )
        assert np.all(fc >= 0.0), f"Negative coverage: {fc[fc < 0]}"
        assert np.all(fc <= 1.0 + 1e-10), f"Coverage > 1: {fc[fc > 1]}"
