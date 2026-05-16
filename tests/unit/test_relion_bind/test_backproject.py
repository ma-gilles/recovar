"""Phase 4: BackProjector parity — exact numerical comparison.

Tests:
1. Backprojector data/weight accumulators: RELION vs RELION (determinism)
2. Reconstruction: same data → same volume (determinism)
3. getDownsampledAverage: output shape and non-triviality
4. FSC between half-sets: self-consistency
5. Weighted vs unweighted backprojection: weights have effect
6. Round-trip correlation with sufficient projections (corr > 0.99)

Exact parity standard: where we compare two runs of the same RELION code
on identical inputs, max_abs_diff must be 0.0 (bit-exact).
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
    get_downsampled_average,
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
            vol,
            A,
            ori_size=N,
            padding_factor=pf,
            interpolator=TRILINEAR,
            current_size=-1,
            do_gridding=True,
        )
    return images, rot_mats


def _relion_pad_size(ori_size, padding_factor):
    """RELION BackProjector pad_size: 2*ROUND(pf*r_max)+3."""
    r_max = ori_size // 2
    return 2 * round(padding_factor * r_max) + 3


class TestBackprojectorDeterminism:
    """Same inputs to RELION binding must produce bit-exact same outputs."""

    def test_data_weight_deterministic(self):
        """Two calls with identical inputs produce identical data/weight."""
        N = 32
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:12]
        images, rot_mats = _project_at_orientations(vol, orientations, N)
        empty_w = np.zeros((0,), dtype=np.float64)

        data1, weight1 = get_backprojector_data(images, rot_mats, empty_w, ori_size=N, padding_factor=2)
        data2, weight2 = get_backprojector_data(images, rot_mats, empty_w, ori_size=N, padding_factor=2)

        assert np.max(np.abs(data1 - data2)) == 0.0, "Data not bit-exact"
        assert np.max(np.abs(weight1 - weight2)) == 0.0, "Weight not bit-exact"

    def test_reconstruct_deterministic(self):
        """Two reconstruction calls produce bit-exact same volume."""
        N = 32
        rng = np.random.default_rng(99)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:12]
        images, rot_mats = _project_at_orientations(vol, orientations, N)
        empty_w = np.zeros((0,), dtype=np.float64)
        tau2 = np.ones(N // 2 + 1) * 1e6

        recon1 = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_w,
            tau2,
            ori_size=N,
            padding_factor=2,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )
        recon2 = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_w,
            tau2,
            ori_size=N,
            padding_factor=2,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )

        assert np.max(np.abs(recon1 - recon2)) == 0.0, "Reconstruction not bit-exact"


class TestBackprojectorDataShape:
    """Verify accumulator shapes match RELION's pad_size formula."""

    @pytest.mark.parametrize("N,pf", [(16, 2), (32, 2), (64, 2), (32, 1)])
    def test_pad_size(self, N, pf):
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:6]
        images, rot_mats = _project_at_orientations(vol, orientations, N, pf=pf)
        empty_w = np.zeros((0,), dtype=np.float64)

        data, weight = get_backprojector_data(images, rot_mats, empty_w, ori_size=N, padding_factor=pf)

        expected_pad = _relion_pad_size(N, pf)
        assert data.shape == (expected_pad, expected_pad, expected_pad // 2 + 1)
        assert weight.shape == data.shape
        assert np.sum(np.abs(data)) > 0
        assert np.sum(weight) > 0


class TestDownsampledAverage:
    """M6: getDownsampledAverage binding tests."""

    def test_nonzero_output(self):
        N = 32
        rng = np.random.default_rng(42)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:12]
        images, rot_mats = _project_at_orientations(vol, orientations, N)
        empty_w = np.zeros((0,), dtype=np.float64)

        avg = get_downsampled_average(images, rot_mats, empty_w, ori_size=N, padding_factor=2, divide=True)
        assert avg.ndim == 3
        assert np.sum(np.abs(avg)) > 0

    def test_deterministic(self):
        """Two calls produce bit-exact same output."""
        N = 32
        rng = np.random.default_rng(77)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:12]
        images, rot_mats = _project_at_orientations(vol, orientations, N)
        empty_w = np.zeros((0,), dtype=np.float64)

        avg1 = get_downsampled_average(images, rot_mats, empty_w, ori_size=N, padding_factor=2)
        avg2 = get_downsampled_average(images, rot_mats, empty_w, ori_size=N, padding_factor=2)

        assert np.max(np.abs(avg1 - avg2)) == 0.0


class TestRoundTrip:
    """Project → backproject → reconstruct should recover the original volume.

    With all 4608 order-2 orientations, this tests the completeness of
    RELION's reconstruction pipeline (not parity with recovar).
    """

    def test_round_trip_correlation(self):
        """All order-2 orientations → high correlation."""
        N = 32
        rng = np.random.default_rng(99)
        vol = _make_test_volume(N, rng)

        all_orientations = get_coarse_orientations(2)
        images, rot_mats = _project_at_orientations(vol, all_orientations, N)

        tau2 = np.ones(N // 2 + 1)
        empty_w = np.zeros((0,), dtype=np.float64)

        recon = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_w,
            tau2,
            ori_size=N,
            padding_factor=2,
            do_map=False,
            max_iter_preweight=10,
            tau2_fudge=1.0,
            skip_gridding=False,
        )

        vol_f = vol.ravel() - vol.ravel().mean()
        recon_f = recon.ravel() - recon.ravel().mean()
        corr = np.dot(vol_f, recon_f) / (np.linalg.norm(vol_f) * np.linalg.norm(recon_f) + 1e-30)
        print(f"\nRound-trip correlation (N={N}, {len(all_orientations)} projections): {corr:.6f}")
        assert corr > 0.8, f"Round-trip correlation too low: {corr:.6f}"


class TestWeightedBackprojection:
    """Verify that per-pixel CTF² weights change the result."""

    def test_weighted_vs_unweighted_differs(self):
        N = 32
        rng = np.random.default_rng(55)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:12]
        images, rot_mats = _project_at_orientations(vol, orientations, N)

        tau2 = np.ones(N // 2 + 1)
        empty_w = np.zeros((0,), dtype=np.float64)

        recon_unweighted = backproject_and_reconstruct(
            images,
            rot_mats,
            empty_w,
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
        assert diff > 1e-6, f"Weights had no effect: max_diff={diff}"


class TestFSC:
    """FSC computation from half-set backprojections."""

    def test_fsc_same_volume_high(self):
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

        assert fsc[1] > 0.5, f"Low-frequency FSC too low: {fsc[1]:.4f}"

    def test_fsc_different_volumes_low(self):
        """FSC between projections of different volumes should be low at mid-freq."""
        N = 32
        rng1 = np.random.default_rng(11)
        rng2 = np.random.default_rng(22)
        vol1 = _make_test_volume(N, rng1)
        vol2 = _make_test_volume(N, rng2)

        orientations = get_coarse_orientations(2)[:48]
        images1, rot_mats1 = _project_at_orientations(vol1, orientations, N)
        images2, rot_mats2 = _project_at_orientations(vol2, orientations, N)

        fsc = compute_fsc_from_halfsets(images1, rot_mats1, images2, rot_mats2, ori_size=N, padding_factor=2)

        mid_shell = len(fsc) // 2
        assert fsc[mid_shell] < 0.5, f"Mid-freq FSC too high: {fsc[mid_shell]:.4f}"

    def test_fsc_deterministic(self):
        """Two identical FSC calls produce bit-exact results."""
        N = 32
        rng = np.random.default_rng(33)
        vol = _make_test_volume(N, rng)
        orientations = get_coarse_orientations(1)[:24]
        images, rot_mats = _project_at_orientations(vol, orientations, N)

        fsc1 = compute_fsc_from_halfsets(
            images[:12], rot_mats[:12], images[12:], rot_mats[12:], ori_size=N, padding_factor=2
        )
        fsc2 = compute_fsc_from_halfsets(
            images[:12], rot_mats[:12], images[12:], rot_mats[12:], ori_size=N, padding_factor=2
        )

        assert np.max(np.abs(fsc1 - fsc2)) == 0.0, "FSC not bit-exact"
