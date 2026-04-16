"""Phase 4: BackProjector parity — project → backproject → reconstruct.

Tests:
1. Smoke: backproject + reconstruct produces non-zero output
2. Round-trip: project many orientations → backproject → reconstruct ≈ original
3. Unweighted vs weighted backprojection
4. FSC between half-sets from known volume
5. Accumulator data/weight shapes match expectations
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import (
    TRILINEAR,
    backproject_and_reconstruct,
    compute_fsc_from_halfsets,
    euler_angles_to_matrix,
    get_backprojector_data,
    get_coarse_orientations,
    project_volume,
)


def _make_test_volume(N, rng):
    """Low-pass filtered random volume."""
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
    """Project volume at multiple orientations, return (images, rot_matrices)."""
    n_orient = orientations.shape[0]
    images = np.zeros((n_orient, N, N // 2 + 1), dtype=np.complex128)
    rot_mats = np.zeros((n_orient, 3, 3))
    for i in range(n_orient):
        rot, tilt, psi = orientations[i]
        A = euler_angles_to_matrix(float(rot), float(tilt), float(psi))
        rot_mats[i] = A
        images[i] = project_volume(
            vol, A, ori_size=N, padding_factor=pf, interpolator=TRILINEAR, current_size=-1, do_gridding=True
        )
    return images, rot_mats


def _relion_pad_size(ori_size, padding_factor):
    """RELION BackProjector pad_size: 2*ROUND(pf*r_max)+3."""
    r_max = ori_size // 2
    return 2 * round(padding_factor * r_max) + 3


class TestBackprojectSmoke:
    """Basic smoke tests for backprojection binding."""

    def test_reconstruct_nonzero(self):
        N = 32
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)

        orientations = get_coarse_orientations(1)[:12]
        images, rot_mats = _project_at_orientations(vol, orientations, N)

        tau2 = np.ones(N // 2 + 1)
        empty_weights = np.zeros((0,), dtype=np.float64)

        recon = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_weights,
            tau2,
            ori_size=N,
            padding_factor=2,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )
        assert recon.shape == (N, N, N)
        assert np.max(np.abs(recon)) > 0
        print(f"\nRecon range: [{recon.min():.4f}, {recon.max():.4f}]")

    def test_backprojector_data_shape(self):
        """RELION pad_size = 2*ROUND(pf*r_max)+3, NOT pf*N."""
        N = 32
        pf = 2
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)

        orientations = get_coarse_orientations(1)[:6]
        images, rot_mats = _project_at_orientations(vol, orientations, N, pf=pf)

        empty_weights = np.zeros((0,), dtype=np.float64)
        data, weight = get_backprojector_data(
            images,
            rot_mats,
            empty_weights,
            ori_size=N,
            padding_factor=pf,
        )
        expected_pad = _relion_pad_size(N, pf)
        assert data.shape == (expected_pad, expected_pad, expected_pad // 2 + 1)
        assert weight.shape == data.shape
        assert np.sum(np.abs(data)) > 0
        assert np.sum(weight) > 0


class TestRoundTrip:
    """Project → backproject → reconstruct should recover the original volume."""

    @pytest.mark.parametrize("N", [32])
    def test_round_trip_correlation(self, N):
        """With enough projections, reconstruction correlates with input."""
        rng = np.random.default_rng(99)
        vol = _make_test_volume(N, rng)

        all_orientations = get_coarse_orientations(2)
        n_use = 192
        idx = np.linspace(0, len(all_orientations) - 1, n_use, dtype=int)
        orientations = all_orientations[idx]
        images, rot_mats = _project_at_orientations(vol, orientations, N)

        tau2 = np.ones(N // 2 + 1)
        empty_weights = np.zeros((0,), dtype=np.float64)

        recon = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_weights,
            tau2,
            ori_size=N,
            padding_factor=2,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )

        vol_f = vol.ravel()
        recon_f = recon.ravel()
        vol_f -= vol_f.mean()
        recon_f -= recon_f.mean()
        corr = np.dot(vol_f, recon_f) / (np.linalg.norm(vol_f) * np.linalg.norm(recon_f) + 1e-30)
        print(f"\nRound-trip correlation (N={N}, {n_use} projections): {corr:.4f}")
        assert corr > 0.8, f"Round-trip correlation too low: {corr:.4f}"


class TestWeightedBackprojection:
    """Verify that per-pixel CTF² weights change the result."""

    def test_weighted_vs_unweighted_differs(self):
        N = 32
        rng = np.random.default_rng(55)
        vol = _make_test_volume(N, rng)

        orientations = get_coarse_orientations(1)[:12]
        images, rot_mats = _project_at_orientations(vol, orientations, N)

        tau2 = np.ones(N // 2 + 1)
        empty_weights = np.zeros((0,), dtype=np.float64)

        recon_unweighted = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_weights,
            tau2,
            ori_size=N,
            padding_factor=2,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )

        n_img = images.shape[0]
        weights = np.ones((n_img, N, N // 2 + 1), dtype=np.float64) * 0.5
        recon_weighted = backproject_and_reconstruct(
            images,
            rot_mats,
            weights,
            tau2,
            ori_size=N,
            padding_factor=2,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )

        diff = np.max(np.abs(recon_unweighted - recon_weighted))
        print(f"\nWeighted vs unweighted max diff: {diff:.4e}")
        assert diff > 1e-6, "Weights had no effect"


class TestFSC:
    """Verify FSC computation from half-set backprojections."""

    def test_fsc_from_same_volume(self):
        """FSC between two halves of projections from same volume should be high."""
        N = 32
        rng = np.random.default_rng(77)
        vol = _make_test_volume(N, rng)

        all_orientations = get_coarse_orientations(2)
        n_total = 384
        idx = np.linspace(0, len(all_orientations) - 1, n_total, dtype=int)
        orientations = all_orientations[idx]
        images, rot_mats = _project_at_orientations(vol, orientations, N)

        n_half = n_total // 2
        fsc = compute_fsc_from_halfsets(
            images[:n_half],
            rot_mats[:n_half],
            images[n_half:n_total],
            rot_mats[n_half:n_total],
            ori_size=N,
            padding_factor=2,
        )

        print(f"\nFSC shells: {len(fsc)}, FSC[0]={fsc[0]:.4f}, FSC[1]={fsc[1]:.4f}, min={fsc.min():.4f}")
        assert fsc[1] > 0.5, f"Low-frequency FSC too low: {fsc[1]:.4f}"

    def test_fsc_different_volumes_is_low(self):
        """FSC between projections of different volumes should be low."""
        N = 32
        rng1 = np.random.default_rng(11)
        rng2 = np.random.default_rng(22)
        vol1 = _make_test_volume(N, rng1)
        vol2 = _make_test_volume(N, rng2)

        orientations = get_coarse_orientations(2)[:48]
        images1, rot_mats1 = _project_at_orientations(vol1, orientations, N)
        images2, rot_mats2 = _project_at_orientations(vol2, orientations, N)

        fsc = compute_fsc_from_halfsets(
            images1,
            rot_mats1,
            images2,
            rot_mats2,
            ori_size=N,
            padding_factor=2,
        )
        mid_shell = len(fsc) // 2
        print(f"\nDifferent-volume FSC at shell {mid_shell}: {fsc[mid_shell]:.4f}")
        assert fsc[mid_shell] < 0.5, f"Mid-frequency FSC too high for different volumes: {fsc[mid_shell]:.4f}"
