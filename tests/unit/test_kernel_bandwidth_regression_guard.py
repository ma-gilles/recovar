"""Merge-regression guard for the kernel-bandwidth-student-clean branch.

This file exists for ONE purpose: when this branch is merged with the
EM / VDAM / PPCA-refinement branches, any change to the code paths
touched here must either (a) leave these pinned numbers exactly intact
or (b) be a deliberate, reviewed update.

The tests below are intentionally redundant with structural unit tests
elsewhere in `tests/unit/`. The structural tests describe *what* the
functions should do; these tests pin *exactly which bits* they produce
on a fixed deterministic input. If a merge silently rewires e.g. the
mask cleanup pipeline or a path's pivot/rotation convention, the
structural tests may still pass while the numerical output drifts; this
file catches that drift.

Conventions used by this guard
------------------------------

* Every test uses a hard-coded seed and shape so it is bit-exact
  reproducible offline.
* SHA-16 fingerprints (sha256, truncated to 16 hex chars) of
  ``np.round(arr, 8).tobytes()`` are pinned. The round-to-8 sandwich
  absorbs trivial fused-op reorderings while still catching any
  algorithmically meaningful change.
* For human-readable floats the test also pins a few representative
  scalar values, so a failure tells you immediately *which direction*
  the drift went.

If a merge legitimately changes these numbers, update **both** the SHA
and the representative scalars in the same PR, with the change-of-behavior
documented in the PR description.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import recovar.core.mask as core_mask
import recovar.heterogeneity.kernel_bandwidth_benchmark as kbb
import recovar.simulation.trajectory_generation as trajgen

pytestmark = pytest.mark.unit


def _sha16(arr: np.ndarray, decimals: int = 8) -> str:
    """Stable 16-hex-char fingerprint of ``np.round(arr, decimals).tobytes()``."""
    rounded = np.round(np.asarray(arr, dtype=np.float64), decimals).tobytes()
    return hashlib.sha256(rounded).hexdigest()[:16]


# ---------------------------------------------------------------------------
# make_state_distribution
# ---------------------------------------------------------------------------


class TestMakeStateDistribution_Pinned:
    """The uniform state distribution is the only kind on this branch; pin it."""

    @pytest.mark.parametrize("n_states,expected_value", [(5, 0.2), (10, 0.1), (50, 0.02)])
    def test_uniform_is_exactly_1_over_n(self, n_states, expected_value):
        d = kbb.make_state_distribution(n_states, kind="uniform")
        assert d.dtype == np.float32
        assert d.shape == (n_states,)
        assert d.sum() == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(d, expected_value, atol=1e-7)


# ---------------------------------------------------------------------------
# shell_labels_for_volume + shell_sums
# ---------------------------------------------------------------------------


class TestShellLabelsAndSums_Pinned:
    def test_shell_labels_8x8x8(self):
        lab = kbb.shell_labels_for_volume((8, 8, 8))
        # shell_labels_for_volume returns a 3-D array, NOT flat.
        assert lab.shape == (8, 8, 8)
        assert lab.dtype == np.int32
        vals, cnts = np.unique(lab, return_counts=True)
        assert len(vals) == 8
        assert int(vals.max()) == 7
        assert int(cnts[0]) == 1  # only DC voxel at shell 0
        assert int(cnts.max()) == 171
        assert _sha16(lab) == "92aae1ee2fd10ad1"

    def test_shell_labels_16x16x16(self):
        lab = kbb.shell_labels_for_volume((16, 16, 16))
        vals = np.unique(lab)
        assert len(vals) == 15
        assert _sha16(lab) == "5e9ade0c490a8083"

    def test_shell_sums_seed0(self):
        """Pin shell_sums on a fixed-seed input."""
        np.random.seed(0)
        shape = (8, 8, 8)
        lab = kbb.shell_labels_for_volume(shape)
        n_shells = int(lab.max()) + 1
        vals = np.random.standard_normal(int(np.prod(shape))).astype(np.float32)
        sums = kbb.shell_sums(vals, lab, n_shells)
        assert sums.shape == (n_shells,)
        # Representative scalars
        np.testing.assert_allclose(sums[0], 2.30391669, rtol=1e-6)
        np.testing.assert_allclose(sums[-1], 1.76405239, rtol=1e-6)
        assert _sha16(sums) == "458b040e46935294"


# ---------------------------------------------------------------------------
# fit_volume_pca
# ---------------------------------------------------------------------------


def _det_volumes(seed: int = 42, n_vol: int = 8, shape=(6, 6, 6)) -> np.ndarray:
    """Deterministic 4-D stack used by multiple tests below."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_vol,) + shape).astype(np.float32)


class TestFitVolumePca_Pinned:
    def test_singular_values_and_shapes(self):
        vols = _det_volumes()
        res = kbb.fit_volume_pca(vols)

        assert res.singular_values.shape == (8,)
        assert res.components.shape == (8, 6, 6, 6)
        assert res.mean.shape == (6, 6, 6)
        assert res.scores.shape == (8, 8)
        assert res.explained_energy.shape == (8,)

        # Representative singular values (decreasing).
        np.testing.assert_allclose(
            res.singular_values[:4],
            np.array([16.36187558, 15.90464865, 15.62427156, 15.29648858]),
            rtol=1e-5,
        )
        # Explained energy sums to 1.0 (or very close).
        np.testing.assert_allclose(res.explained_energy.sum(), 1.0, atol=1e-5)

    def test_fingerprints(self):
        """Bit-exact fingerprint pinning."""
        vols = _det_volumes()
        res = kbb.fit_volume_pca(vols)
        # Use modest decimals for the rounding sandwich; this is robust
        # against trivial fused-op reorderings but catches genuine drift.
        assert _sha16(res.singular_values) == "a110cd12b22d7f12"
        assert _sha16(res.scores) == "84fe66223d40273a"
        assert _sha16(res.mean) == "ad143def6b151772"
        assert _sha16(res.explained_energy) == "d04e672495d41a4d"


# ---------------------------------------------------------------------------
# project_volume_trajectory
# ---------------------------------------------------------------------------


class TestProjectVolumeTrajectory_Pinned:
    def test_coords_and_info(self):
        vols = _det_volumes()
        coords, info = kbb.project_volume_trajectory(vols, n_pcs=3)

        assert coords.shape == (8, 6, 6, 6)
        # info contains everything from VolumePcaResult + n_pcs.
        assert set(info.keys()) == {"components", "explained_energy", "mean", "n_pcs", "scores", "singular_values"}
        assert info["n_pcs"] == 3

        # Representative scalar: coords[0, 0, 0, 0].
        np.testing.assert_allclose(coords[0, 0, 0, 0], 0.48754054, rtol=1e-6)

        assert _sha16(coords) == "f85a27a7cbfe3213"
        assert _sha16(info["mean"]) == "c95630189a04b1fd"
        assert _sha16(info["scores"]) == "ce3e717a8f98e837"
        assert _sha16(info["components"]) == "ff1b8331696f9977"
        assert _sha16(info["singular_values"]) == "6f5ae43f63963847"
        assert _sha16(info["explained_energy"]) == "3304e30689bad1ab"


# ---------------------------------------------------------------------------
# choose_bandwidth_bins
# ---------------------------------------------------------------------------


class TestChooseBandwidthBins_Pinned:
    def test_uniform_quantile_grid_seed0(self):
        rng = np.random.default_rng(0)
        d0 = np.sort(rng.uniform(0, 10, 100)).astype(np.float32)
        d1 = np.sort(rng.uniform(0, 10, 100)).astype(np.float32)
        bins = kbb.choose_bandwidth_bins([d0, d1], n_bandwidths=10, n_min_particles=5)
        assert bins.shape == (10,)
        assert bins.dtype == np.float32
        # Pin the exact bin values (these are quantiles of the merged
        # distance distribution).
        expected = np.array(
            [0.4306689, 0.8602544, 1.436988, 2.1608696, 3.0318995, 4.050077, 5.215403, 6.527877, 7.9874988, 9.594269],
            dtype=np.float32,
        )
        np.testing.assert_allclose(bins, expected, rtol=1e-5)
        assert _sha16(bins) == "f1ef8c8edba6209c"

    def test_monotonic_and_respects_n_bandwidths(self):
        """Soft invariants: bins are non-decreasing and the requested length."""
        rng = np.random.default_rng(1)
        d0 = np.sort(rng.uniform(0, 5, 200)).astype(np.float32)
        d1 = np.sort(rng.uniform(0, 5, 200)).astype(np.float32)
        for n in (3, 7, 20):
            bins = kbb.choose_bandwidth_bins([d0, d1], n_bandwidths=n, n_min_particles=5)
            assert bins.shape == (n,)
            assert np.all(np.diff(bins) >= -1e-6), f"non-monotonic at n={n}"


# ---------------------------------------------------------------------------
# Trajectory paths
# ---------------------------------------------------------------------------


def _det_groups():
    """Three 4-atom synthetic chains; deterministic, plus pivot at origin."""
    return [
        np.array([[1.0, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]),
        np.array([[5.0, 1, 1], [6, 1, 1], [7, 1, 1], [8, 1, 1]]),
        np.array([[9.0, 2, 2], [10, 2, 2], [11, 2, 2], [12, 2, 2]]),
    ], np.zeros(3)


def _path_sha(r):
    return _sha16(np.concatenate([np.asarray(rr).ravel() for rr in r]))


class TestTrajectoryPaths_Pinned:
    def test_path_identity_is_t_invariant(self):
        """path_identity must produce the SAME output at every t.

        This is the *whole point* of path_identity (no-rotation 5nrl
        trajectory). If a merge silently introduces any t-dependence
        — e.g. via the rot_around_x(-10) base offset leaking in —
        this test catches it.
        """
        groups, pivot = _det_groups()
        shas = [_path_sha(trajgen.path_identity(t, groups, pivot)) for t in (0, 5, 10, 20)]
        assert shas[0] == "6c3d8b3f15e602c2"
        # All four must equal the first.
        for i, s in enumerate(shas[1:], start=1):
            assert s == shas[0], f"path_identity t-drift at index {i}: {s} != {shas[0]}"

    def test_path_arm_only_pinned(self):
        groups, pivot = _det_groups()
        expected = {
            0: "8f333f7b7d95bf11",
            5: "60362bb4fc14f241",
            10: "42e7b3cae65e69a9",
            20: "c26f6daac3317df9",
        }
        for t, sha in expected.items():
            got = _path_sha(trajgen.path_arm_only(t, groups, pivot))
            assert got == sha, f"path_arm_only t={t}: got {got}, expected {sha}"

    def test_path_head_only_pinned(self):
        groups, pivot = _det_groups()
        expected = {
            0: "8f333f7b7d95bf11",
            5: "6ffd5ec348ac15a9",
            10: "53040e967cac49f2",
            20: "c0408ffc1fd49be5",
        }
        for t, sha in expected.items():
            got = _path_sha(trajgen.path_head_only(t, groups, pivot))
            assert got == sha, f"path_head_only t={t}: got {got}, expected {sha}"

    def test_arm_only_and_head_only_agree_at_t0(self):
        """At t=0 both arm-only and head-only paths produce the same
        configuration (only the static base rotation -10° is applied,
        which is identical for both)."""
        groups, pivot = _det_groups()
        sha_arm = _path_sha(trajgen.path_arm_only(0, groups, pivot))
        sha_head = _path_sha(trajgen.path_head_only(0, groups, pivot))
        assert sha_arm == sha_head == "8f333f7b7d95bf11"


# ---------------------------------------------------------------------------
# Mask functions
# ---------------------------------------------------------------------------


def _det_mask_volumes(seed: int = 11, N: int = 24, n_vols: int = 6):
    """Fixed-shape volumes with a static blob + moving blob, deterministic."""
    rng = np.random.default_rng(seed)
    shape = (N, N, N)
    vols = rng.standard_normal((n_vols, *shape), dtype=np.float32) * 0.01
    # Static blob at (8:12, 8:12, 8:12)
    vols[:, 8:12, 8:12, 8:12] += 5.0
    # Moving blob at (16:20, 16:20, 16:20), varying sign across the stack
    scale = np.linspace(-1.0, 1.0, n_vols, dtype=np.float32)
    vols[:, 16:20, 16:20, 16:20] += scale[:, None, None, None] * 10.0
    return vols, shape


class TestMaskFunctions_Pinned:
    def test_make_union_gt_mask(self):
        vols, shape = _det_mask_volumes()
        soft, binary = core_mask.make_union_gt_mask(vols, shape)
        assert soft.shape == shape
        assert binary.shape == shape
        # Voxel-count pin: 9273 out of 24^3 = 13824.
        assert int(binary.sum()) == 9273
        np.testing.assert_allclose(soft.sum(), 12243.5, rtol=1e-5)
        assert _sha16(binary.astype(np.uint8)) == "34481f92de87cf9d"
        assert _sha16(soft) == "64da20bdb08418fd"

    def test_make_moving_gt_mask(self):
        vols, shape = _det_mask_volumes()
        soft, binary = core_mask.make_moving_gt_mask(vols, shape)
        assert int(binary.sum()) == 4046
        np.testing.assert_allclose(soft.sum(), 8174.663574, rtol=1e-5)
        assert _sha16(binary.astype(np.uint8)) == "6373e5b6198c24da"
        assert _sha16(soft) == "5d30e7068ddf7e47"

    def test_make_localized_moving_gt_mask_p90(self):
        vols, shape = _det_mask_volumes()
        soft, binary = core_mask.make_localized_moving_gt_mask(vols, shape, percentile=90.0)
        assert int(binary.sum()) == 4288
        np.testing.assert_allclose(soft.sum(), 4400.762207, rtol=1e-5)
        assert _sha16(binary.astype(np.uint8)) == "e4bf0f215f27cc31"
        assert _sha16(soft) == "6d35251dbf77afdc"

    def test_localized_mask_invariants_vs_legacy(self):
        """Localized@p90 picks the moving blob, NOT the static one.

        On this 24^3 fixture the static blob is at (8:12)^3 and the
        moving blob is at (16:20)^3. They are 4 voxels apart along the
        diagonal, and the standard mask-cleanup pipeline applies an
        ``extend=2`` dilation by default, so a handful of voxels at the
        corner of the static cube nearest the moving cube unavoidably
        creep in. We pin the invariant we actually care about: the
        localized mask is overwhelmingly on the moving blob, and the
        static blob remains essentially uncovered."""
        vols, shape = _det_mask_volumes()
        _, bin_legacy = core_mask.make_moving_gt_mask(vols, shape)
        _, bin_p90 = core_mask.make_localized_moving_gt_mask(vols, shape, percentile=90.0)
        moving_voxels = (slice(16, 20), slice(16, 20), slice(16, 20))
        static_voxels = (slice(8, 12), slice(8, 12), slice(8, 12))

        moving_count = int(bin_p90[moving_voxels].sum())
        static_count = int(bin_p90[static_voxels].sum())
        moving_volume = 4 * 4 * 4  # 64

        # The moving cube is fully covered.
        assert moving_count == moving_volume, (
            f"localized@p90 missed part of the moving blob: {moving_count}/{moving_volume}"
        )
        # The static cube is at most marginally touched (the corner
        # voxels nearest the moving blob, from the dilation step).
        assert static_count < moving_volume / 8, (
            f"localized@p90 leaked too much into the static region: {static_count}/{moving_volume} voxels covered"
        )
        # Legacy moving mask must also cover the moving blob.
        assert bin_legacy[moving_voxels].sum() > 0


# ---------------------------------------------------------------------------
# CLI default contract (cross-checks the values pinned in tests above)
# ---------------------------------------------------------------------------


class TestCliDefaults_Contract:
    """The bandwidth walkthrough CLI exposes a small contract: the
    student-friendly defaults that this branch deliberately tuned. If
    a merge silently flips these (e.g., reverts compute_state_maskrad_fraction
    to the 0.5 historical default), this catches it."""

    def test_walkthrough_argparse_defaults(self):
        import argparse

        from recovar.commands import benchmark_kernel_bandwidth_1d as walkthrough

        parser = argparse.ArgumentParser()
        walkthrough.add_args(parser)
        args = parser.parse_args(["--output-dir", "/tmp/out"])

        # Top-line student defaults.
        assert args.grid_size == 64
        assert args.n_states == 50
        assert args.n_images == 10000
        assert args.noise_level == 1.0
        # The 20.0 default mirrors production compute_state.
        assert args.compute_state_maskrad_fraction == 20.0
        assert args.compute_state_n_min_particles == 100
        # Default focus mask = localized @ p95 (legacy = 0).
        assert args.focus_mask_percentile == 95.0
        # voxel-size autodetect (None = "auto from MRC header or 5nrl default").
        assert args.voxel_size is None
        # Default path is symmetric (other paths require explicit choice).
        assert args.path == "symmetric"

    def test_walkthrough_path_choices_contract(self):
        """The four paths that this branch ships must remain available
        as --path choices. If a merge dropped one, the choices list would
        shrink and this test would catch it."""
        import argparse

        from recovar.commands import benchmark_kernel_bandwidth_1d as walkthrough

        parser = argparse.ArgumentParser()
        walkthrough.add_args(parser)
        # Find the --path argument and inspect its choices.
        path_action = next(a for a in parser._actions if a.dest == "path")
        assert set(path_action.choices) >= {"symmetric", "arm_only", "head_only", "identity"}


# ---------------------------------------------------------------------------
# Performance smoke
# ---------------------------------------------------------------------------


class TestPerformanceSmoke:
    """Tight CPU wall-time bounds on the deterministic fixtures pinned
    above. The numbers below are generous (10× headroom over local
    measurements on a quiet box) so we catch ONLY catastrophic
    regressions — for example, a merge that accidentally drops a JIT
    cache, removes a vectorization, or makes an O(N^2) version of a
    function ship.

    These tests are deliberately CPU-only and do NOT require a GPU, so
    they can run in `pixi run test-fast`."""

    def test_fit_volume_pca_runs_under_2s_cpu(self):
        import time

        vols = _det_volumes()  # 8 volumes, 6^3 grid, deterministic
        # Warm any one-time numpy/scipy import overhead.
        kbb.fit_volume_pca(vols)
        t0 = time.perf_counter()
        for _ in range(5):
            kbb.fit_volume_pca(vols)
        elapsed = time.perf_counter() - t0
        # 5 calls in <2 s = 400 ms/call CPU budget. Local measurement
        # is ~20 ms/call; this is generous.
        assert elapsed < 2.0, (
            f"fit_volume_pca CPU slowdown: 5 calls on (8, 6, 6, 6) took {elapsed:.3f}s "
            f"(budget 2.0s). Suspect: lost vectorization or accidental float64 promotion."
        )

    def test_make_localized_moving_gt_mask_runs_under_5s_cpu(self):
        import time

        vols, shape = _det_mask_volumes()  # 6 × 24^3, deterministic
        # Warm.
        core_mask.make_localized_moving_gt_mask(vols, shape, percentile=90.0)
        t0 = time.perf_counter()
        for _ in range(3):
            core_mask.make_localized_moving_gt_mask(vols, shape, percentile=90.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, (
            f"make_localized_moving_gt_mask CPU slowdown: 3 calls on 6 × 24^3 "
            f"took {elapsed:.3f}s (budget 5.0s). Suspect: scipy.ndimage path change."
        )

    def test_choose_bandwidth_bins_runs_under_1s_cpu(self):
        import time

        rng = np.random.default_rng(0)
        d0 = np.sort(rng.uniform(0, 10, 10_000)).astype(np.float32)
        d1 = np.sort(rng.uniform(0, 10, 10_000)).astype(np.float32)
        # Warm.
        kbb.choose_bandwidth_bins([d0, d1], n_bandwidths=50, n_min_particles=200)
        t0 = time.perf_counter()
        for _ in range(20):
            kbb.choose_bandwidth_bins([d0, d1], n_bandwidths=50, n_min_particles=200)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, (
            f"choose_bandwidth_bins CPU slowdown: 20 calls on 10k+10k distances "
            f"took {elapsed:.3f}s (budget 1.0s). Suspect: accidental Python loop."
        )
