"""Smoke tests for refine_single_volume with mode="relion".

Verifies:
1. RELION mode runs without error on a tiny dataset (4 images, 8px, 2 iters)
2. Returns the expected dict keys (including RELION-specific ones)
3. Legacy mode is unchanged by the new mode parameter
4. Invalid mode raises ValueError
5. Convergence state is a RefinementState instance
6. data_vs_prior_trajectory and ave_Pmax_trajectory are populated
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.engine_v2 import run_em_v2
from recovar.em.dense_single_volume.refine import (
    _bootstrap_current_size_relion,
    _compute_significance_batched,
    collapse_rotation_posterior_to_direction_prior,
    compute_coarse_image_size,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    should_skip_adaptive_pass2,
    refine_single_volume,
)
from recovar.em.dense_single_volume.convergence import RefinementState
from recovar.em.dense_single_volume.types import RelionStats

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Test constants -- 8x8 images for fast unit tests
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
H, W = IMAGE_SHAPE
N_ROTATIONS = 5
N_TRANSLATIONS = 3
N_IMAGES = 4  # tiny: 2 per half-set
SEED = 42


# ---------------------------------------------------------------------------
# Helpers (same as test_fsc_resolution_loop.py)
# ---------------------------------------------------------------------------

def _hermitian_image_2d(image_shape, seed=42):
    rng = np.random.default_rng(seed)
    real_img = rng.standard_normal(image_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fft2(real_img))
    return jnp.array(ft, dtype=jnp.complex64)


def _hermitian_volume(volume_shape, seed=42):
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _make_rotations(n, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return q.astype(np.float32)


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _identity_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    return batch


class MockDataset:
    """Minimal mock of CryoEMDataset for unit testing."""

    def __init__(self, n_images, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = n_images
        self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_identity_process)
        self.premultiplied_ctf = False

        self._images = np.zeros((n_images, IMAGE_SIZE), dtype=np.complex64)
        for i in range(n_images):
            self._images[i] = _hermitian_image_2d(
                IMAGE_SHAPE, seed=rng.integers(10000)
            ).reshape(-1)

        self.rotation_matrices = np.tile(
            np.eye(3, dtype=np.float32), (n_images, 1, 1)
        )
        self.translations = np.zeros((n_images, 2), dtype=np.float32)

        class _ImageSource:
            process_images = staticmethod(_identity_process)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        _ = kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                self.rotation_matrices[idx],
                self.translations[idx],
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def update_poses(self, rots, trans):
        self.rotation_matrices = np.asarray(rots)
        self.translations = np.asarray(trans)

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def half_datasets(rng):
    ds0 = MockDataset(N_IMAGES // 2, rng)
    ds1 = MockDataset(N_IMAGES // 2, rng)
    return [ds0, ds1]


@pytest.fixture
def init_volume():
    return _hermitian_volume(VOLUME_SHAPE, seed=42)


@pytest.fixture
def rotations():
    return _make_rotations(N_ROTATIONS, seed=12)


@pytest.fixture
def translations():
    return jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
    )


# ===========================================================================
# Test 1: RELION mode smoke test -- runs without error
# ===========================================================================

class TestRelionModeSmokeTest:
    """Call refine_single_volume with mode='relion' and verify it runs."""

    def test_relion_bootstrap_current_size_matches_benchmark_case(self):
        """128px, 4.25A/px, ini_high=30A should bootstrap from 36 -> 56."""
        assert _bootstrap_current_size_relion(36, 128) == 56

    def test_compute_coarse_image_size_uses_particle_diameter(self):
        """RELION coarse_size should depend on particle diameter, not box size."""
        coarse_from_particle = compute_coarse_image_size(
            14.7, 4.25, 128, particle_diameter=200.0,
        )
        coarse_from_box = compute_coarse_image_size(
            14.7, 4.25, 128,
        )
        assert coarse_from_particle == 52
        assert coarse_from_box == 20
        assert coarse_from_particle > coarse_from_box

    def test_relion_mode_runs_2_iterations(
        self, half_datasets, init_volume, rotations, translations,
    ):
        """mode='relion' completes 2 iterations on a tiny dataset."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        # Basic return dict structure
        assert "mean" in result
        assert "means" in result
        assert "fsc" in result
        assert "hard_assignments" in result
        assert "current_sizes" in result
        assert "fsc_history" in result
        assert "pixel_resolutions" in result
        assert "wall_times" in result

        # RELION-specific keys
        assert "convergence_state" in result
        assert "data_vs_prior_trajectory" in result
        assert "healpix_order_trajectory" in result
        assert "ave_Pmax_trajectory" in result

    def test_relion_mode_finite_outputs(
        self, half_datasets, init_volume, rotations, translations,
    ):
        """RELION mode produces finite volumes and valid assignments."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        # Final mean should be finite
        assert np.all(np.isfinite(np.array(result["mean"]))), "Mean not finite"
        # FSC should be computed
        assert result["fsc"] is not None
        # Hard assignments valid
        for k in range(2):
            ha = result["hard_assignments"][k]
            assert ha is not None
            assert np.all(ha >= 0)

    def test_relion_mode_uses_engine_pmax(
        self, half_datasets, init_volume, rotations, translations,
    ):
        """ave_Pmax should aggregate the engine's posterior maxima across both half-sets."""
        init_noise = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
        init_tau = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0

        expected_per_half = []
        for dataset in half_datasets:
            _, _, _, _, stats = run_em_v2(
                dataset,
                init_volume,
                init_tau,
                init_noise,
                rotations,
                translations,
                "linear_interp",
                image_batch_size=N_IMAGES,
                rotation_block_size=N_ROTATIONS,
                return_stats=True,
            )
            expected_per_half.append(np.asarray(stats.max_posterior_per_image))
        expected_ave_pmax = float(np.mean(np.concatenate(expected_per_half)))

        result = refine_single_volume(
            half_datasets,
            init_volume,
            init_noise,
            init_tau,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            adaptive_oversampling=0,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        assert result["ave_Pmax_trajectory"] == pytest.approx(
            [expected_ave_pmax], abs=1e-6,
        )
        assert result["convergence_state"].ave_Pmax == pytest.approx(
            expected_ave_pmax, abs=1e-6,
        )

    def test_relion_mode_forwards_particle_diameter_to_coarse_size(
        self, half_datasets, init_volume, translations, monkeypatch,
    ):
        """Adaptive RELION mode should pass the explicit particle diameter through."""
        import recovar.em.dense_single_volume.refine as refine_mod

        recorded = {"particle_diameter": None}
        original_compute_coarse_image_size = refine_mod.compute_coarse_image_size

        def wrap_compute_coarse_image_size(*args, **kwargs):
            recorded["particle_diameter"] = kwargs.get("particle_diameter")
            return original_compute_coarse_image_size(*args, **kwargs)

        monkeypatch.setattr(
            refine_mod,
            "compute_coarse_image_size",
            wrap_compute_coarse_image_size,
        )

        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            _make_rotations(20, seed=123),
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=20,
            init_current_size=16,
            adaptive_oversampling=1,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
            particle_diameter_ang=200.0,
        )

        assert recorded["particle_diameter"] == pytest.approx(200.0)

    def test_relion_translation_log_prior_is_mean_normalized(self, translations):
        log_prior = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
        )
        probs = np.exp(np.asarray(log_prior))
        assert np.isclose(np.mean(probs), 1.0, atol=1e-6)
        assert int(np.argmax(probs)) == 0

    def test_relion_translation_log_prior_uses_offset_range_when_active(self, translations):
        log_prior_sigma = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
        )
        log_prior_range = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            offset_range_pixels=3.0,
        )
        probs_sigma = np.exp(np.asarray(log_prior_sigma))
        probs_range = np.exp(np.asarray(log_prior_range))
        # RELION uses sigma = offset_range / 3 while a finite search range is active.
        assert np.isclose(np.mean(probs_range), 1.0, atol=1e-6)
        assert probs_range[0] > probs_sigma[0]

    def test_direction_prior_round_trip_to_rotation_log_prior(self):
        healpix_order = 1
        n_dirs = 48
        direction_prior = np.zeros(n_dirs, dtype=np.float32)
        direction_prior[:3] = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        direction_prior[3:] = np.finfo(np.float32).tiny
        direction_prior /= direction_prior.sum()

        rotation_log_prior = make_relion_direction_log_prior(direction_prior, healpix_order)
        collapsed = collapse_rotation_posterior_to_direction_prior(
            np.exp(rotation_log_prior),
            healpix_order,
        )

        np.testing.assert_allclose(collapsed, direction_prior, rtol=1e-6, atol=1e-8)

    def test_engine_translation_log_prior_changes_pmax(self, half_datasets, init_volume):
        rotations = _make_rotations(1, seed=17)
        translations = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32,
        )
        half_datasets[0]._images = np.zeros_like(half_datasets[0]._images)
        uniform_stats = run_em_v2(
            half_datasets[0],
            jnp.zeros_like(init_volume),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=1,
            return_stats=True,
        )[4]
        biased_prior = np.log(np.array([100.0, 1.0, 1.0], dtype=np.float32))
        biased_stats = run_em_v2(
            half_datasets[0],
            jnp.zeros_like(init_volume),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=1,
            translation_log_prior=biased_prior,
            return_stats=True,
        )[4]
        assert np.allclose(
            np.asarray(uniform_stats.max_posterior_per_image),
            1.0 / 3.0,
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(biased_stats.max_posterior_per_image),
            100.0 / 102.0,
            atol=1e-6,
        )

    def test_significance_batched_supports_padded_rotation_log_prior(
        self, half_datasets, init_volume, translations,
    ):
        """Rotation priors should work even when the last block is padded."""
        rotations = _make_rotations(5, seed=19)
        sig_rot_any, n_sig, ha = _compute_significance_batched(
            half_datasets[0],
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            0.999,
            -1,
            image_batch_size=N_IMAGES,
            rotation_block_size=rotations.shape[0] + 1,
            current_size=None,
            rotation_log_prior=np.zeros(rotations.shape[0], dtype=np.float32),
        )

        assert sig_rot_any.shape == (rotations.shape[0],)
        assert n_sig.shape == (half_datasets[0].n_units,)
        assert ha.shape == (half_datasets[0].n_units,)

    def test_relion_mode_convergence_state(
        self, half_datasets, init_volume, rotations, translations,
    ):
        """Convergence state is a RefinementState with correct fields."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        state = result["convergence_state"]
        assert isinstance(state, RefinementState)
        # After 2 iterations, iteration counter should be at least 1
        assert state.iteration >= 1
        # ave_Pmax should be in [0, 1]
        assert 0.0 <= state.ave_Pmax <= 1.0

    def test_relion_mode_uses_reconstruction_stats_for_prior(
        self, half_datasets, init_volume, rotations, translations, monkeypatch,
    ):
        """RELION mode should not fall back to the pose-based prior helper."""
        from recovar.reconstruction import regularization

        called = {"new": 0}

        def fail_old_prior(*args, **kwargs):
            raise AssertionError("RELION mode should not call compute_relion_prior")

        def wrap_new_prior(*args, **kwargs):
            called["new"] += 1
            return original_new_prior(*args, **kwargs)

        original_new_prior = regularization.compute_relion_prior_from_reconstruction_stats
        monkeypatch.setattr(regularization, "compute_relion_prior", fail_old_prior)
        monkeypatch.setattr(
            regularization,
            "compute_relion_prior_from_reconstruction_stats",
            wrap_new_prior,
        )

        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            adaptive_oversampling=0,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        assert called["new"] == 1

    def test_relion_mode_current_size_no_longer_uses_weight_based_data_vs_prior(
        self, half_datasets, init_volume, rotations, translations, monkeypatch,
    ):
        """RELION mode should derive current_size from FSC-derived SSNR logic."""
        import recovar.em.dense_single_volume.refine as refine_mod

        def fail_old_dvp(*args, **kwargs):
            raise AssertionError("RELION mode should not call compute_data_vs_prior")

        monkeypatch.setattr(refine_mod, "compute_data_vs_prior", fail_old_dvp, raising=False)

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            adaptive_oversampling=0,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        assert len(result["current_sizes"]) == 2

    def test_relion_mode_trajectories_populated(
        self, half_datasets, init_volume, rotations, translations,
    ):
        """RELION-specific trajectories have correct lengths."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        n_iters = len(result["current_sizes"])
        assert n_iters <= 2
        assert len(result["healpix_order_trajectory"]) == n_iters
        assert len(result["ave_Pmax_trajectory"]) == n_iters
        # data_vs_prior is populated starting from iteration 1
        assert len(result["data_vs_prior_trajectory"]) <= n_iters

    def test_should_skip_adaptive_pass2_threshold(self):
        """Adaptive pass 2 should be skipped when mean significant fraction >= 0.5."""
        skip, frac = should_skip_adaptive_pass2(
            np.array([60, 60], dtype=np.int32),
            n_rotations=20,
            n_translations=3,
        )
        assert skip is True
        assert frac == pytest.approx(1.0)

        skip, frac = should_skip_adaptive_pass2(
            np.array([12, 18], dtype=np.int32),
            n_rotations=20,
            n_translations=3,
        )
        assert skip is False
        assert frac == pytest.approx(0.25)

        skip, frac = should_skip_adaptive_pass2(
            np.array([60, 60], dtype=np.int32),
            n_rotations=20,
            n_translations=3,
            threshold=-1.0,
        )
        assert skip is False
        assert frac == pytest.approx(0.0)

    def test_relion_mode_skips_pass2_when_significance_fraction_is_high(
        self, half_datasets, init_volume, translations, monkeypatch,
    ):
        """When pass-1 significance is dense, RELION mode should bypass sparse pass 2."""
        import recovar.em.dense_single_volume.refine as refine_mod

        rotations_many = _make_rotations(20, seed=123)
        captured = {"n_rot": None}

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_rot = np.asarray(args[3]).shape[0]
            n_trans = len(np.asarray(translations))
            captured["n_rot"] = n_rot
            return (
                np.ones(n_rot, dtype=bool),
                np.full(n_images, n_rot * n_trans, dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [np.array([0], dtype=np.int32) for _ in range(n_images)],
            )

        def fail_pass2(*args, **kwargs):
            raise AssertionError("compute_pass2_stats_sparse should be skipped")

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)
        monkeypatch.setattr(refine_mod, "compute_pass2_stats_sparse", fail_pass2)

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=16,
            adaptive_oversampling=1,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        assert result["significant_counts"][0] is not None
        counts = np.asarray(result["significant_counts"][0])
        assert counts.shape == (
            half_datasets[0].n_units + half_datasets[1].n_units,
        )
        np.testing.assert_array_equal(
            counts,
            np.full_like(
                counts, captured["n_rot"] * len(np.asarray(translations))
            ),
        )

    def test_relion_mode_uses_dense_exact_pass2_when_all_samples_significant(
        self, half_datasets, init_volume, translations, monkeypatch,
    ):
        """Exact mode should batch dense pass 2 instead of calling sparse per image."""
        import recovar.em.dense_single_volume.refine as refine_mod

        rotations_many = _make_rotations(20, seed=456)
        dense_calls = {"count": 0}

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_rot = np.asarray(args[3]).shape[0]
            n_trans = len(np.asarray(translations))
            return (
                np.ones(n_rot, dtype=bool),
                np.full(n_images, n_rot * n_trans, dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [None] * n_images,
            )

        def fake_dense_pass2(*args, **kwargs):
            from recovar.em.dense_single_volume.types import NoiseStats
            dataset = args[0]
            n_images = dataset.n_units
            rotations_arg = np.asarray(args[4], dtype=np.float32)
            dense_calls["count"] += 1
            n_shells = dataset.image_shape[0] // 2 + 1
            result = (
                jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
                jnp.ones(VOLUME_SIZE, dtype=jnp.complex64),
                np.zeros(n_images, dtype=np.int32),
                rotations_arg,
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(rotations_arg.shape[0], dtype=jnp.float32),
                ),
            )
            if kwargs.get("accumulate_noise", False):
                result = result + (NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    sumw=float(n_images),
                ),)
            return result

        def fail_sparse(*args, **kwargs):
            raise AssertionError("compute_pass2_stats_sparse should not run for dense exact pass 2")

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)
        monkeypatch.setattr(refine_mod, "compute_pass2_stats", fake_dense_pass2)
        monkeypatch.setattr(refine_mod, "compute_pass2_stats_sparse", fail_sparse)

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=16,
            adaptive_oversampling=1,
            adaptive_pass2_skip_threshold=-1.0,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        assert dense_calls["count"] == 2
        assert result["significant_counts"][0] is not None

    def test_relion_mode_passes_adaptive_pruning_parameters(
        self, half_datasets, init_volume, translations, monkeypatch,
    ):
        """RELION mode should forward adaptive pruning kwargs to pass 1."""
        import recovar.em.dense_single_volume.refine as refine_mod

        rotations_many = _make_rotations(20, seed=321)
        captured = {}

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_rot = np.asarray(args[3]).shape[0]
            n_trans = len(np.asarray(translations))
            captured["adaptive_fraction"] = kwargs["adaptive_fraction"]
            captured["max_significants"] = kwargs["max_significants"]
            return (
                np.ones(n_rot, dtype=bool),
                np.full(n_images, n_rot * n_trans, dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [np.array([0], dtype=np.int32) for _ in range(n_images)],
            )

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)

        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=16,
            adaptive_oversampling=1,
            adaptive_fraction=0.97,
            max_significants=123,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        assert captured["adaptive_fraction"] == pytest.approx(0.97)
        assert captured["max_significants"] == 123

    def test_relion_mode_regenerates_initial_coarse_grid_from_healpix_state(
        self, half_datasets, init_volume, translations, monkeypatch,
    ):
        """Pass 1 should start from the coarse RELION HEALPix order, not a fine caller grid."""
        import recovar.em.dense_single_volume.refine as refine_mod

        captured = {"grid_order": None}

        def fake_grid(order, matrices=True):
            _ = matrices
            captured["grid_order"] = int(order)
            n_rot = 4 if order == 1 else 9
            return np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            rotations_arg = np.asarray(args[3], dtype=np.float32)
            n_trans = len(np.asarray(translations))
            return (
                np.ones(rotations_arg.shape[0], dtype=bool),
                np.full(n_images, rotations_arg.shape[0] * n_trans, dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [None] * n_images,
            )

        monkeypatch.setattr(refine_mod, "get_rotation_grid_at_order", fake_grid)
        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)

        fine_rotations = _make_rotations(9, seed=999)
        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            fine_rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(fine_rotations),
            init_current_size=16,
            adaptive_oversampling=1,
            adaptive_pass2_skip_threshold=-1.0,
            nside_level=3,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        assert captured["grid_order"] == 1


# ===========================================================================
# Test 2: Legacy mode unchanged
# ===========================================================================

class TestLegacyModeUnchanged:
    """Verify that mode='legacy' (default) produces the same result."""

    def test_legacy_mode_explicit(
        self, half_datasets, init_volume, rotations, translations,
    ):
        """Explicit mode='legacy' produces standard output keys."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            relion_current_sizes=[32],
            mode="legacy",
        )

        # Standard keys
        assert "mean" in result
        assert "means" in result
        assert "fsc" in result
        assert "hard_assignments" in result
        assert "current_sizes" in result
        assert result["current_sizes"] == [8]

        # Should NOT have RELION-specific keys
        assert "convergence_state" not in result
        assert "data_vs_prior_trajectory" not in result

    def test_default_mode_is_legacy(
        self, half_datasets, init_volume, rotations, translations,
    ):
        """Calling without mode= uses legacy (no RELION keys)."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            relion_current_sizes=[32],
        )

        assert "convergence_state" not in result


# ===========================================================================
# Test 3: Invalid mode
# ===========================================================================

class TestInvalidMode:

    def test_invalid_mode_raises(
        self, half_datasets, init_volume, rotations, translations,
    ):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            refine_single_volume(
                half_datasets,
                init_volume,
                jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
                jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
                rotations,
                translations,
                mode="bogus",
                max_iter=1,
            )
