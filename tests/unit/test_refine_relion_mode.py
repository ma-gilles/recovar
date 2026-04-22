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
import healpy as hp

from recovar.em.dense_single_volume.convergence import RefinementState
from recovar.em.dense_single_volume.engine_v2 import run_em_v2
from recovar.em.dense_single_volume.refine import (
    _bootstrap_current_size_relion,
    _compute_significance_batched,
    _local_search_chunk_size,
    _local_search_engine_rotation_block_size,
    _local_search_max_union_rotations,
    _pad_local_search_rotations,
    _partition_local_search_groups,
    _local_search_rotation_block_size,
    clamp_relion_coarse_image_size,
    collapse_rotation_posterior_to_direction_prior,
    compute_coarse_image_size,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    refine_single_volume,
    should_skip_adaptive_pass2,
)
from recovar.em.dense_single_volume.types import NoiseStats, RelionStats
from recovar.em.sampling import (
    apply_relion_rotation_perturbation,
    get_relion_rotation_grid,
    rotation_grid_n_in_planes,
    rotation_grid_size,
)

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


def test_local_search_chunk_size_caps_seed_groups_at_64_images():
    assert _local_search_chunk_size(1) == 1
    assert _local_search_chunk_size(64) == 64
    assert _local_search_chunk_size(512) == 64


def test_local_search_rotation_block_size_uses_power_of_two_buckets():
    assert _local_search_rotation_block_size(0, 5000) == 1
    assert _local_search_rotation_block_size(12, 5000) == 16
    assert _local_search_rotation_block_size(17, 5000) == 32
    assert _local_search_rotation_block_size(3000, 5000) == 4096
    assert _local_search_rotation_block_size(5001, 5000) == 5000


def test_local_search_engine_rotation_block_size_caps_dense_tiles():
    assert _local_search_engine_rotation_block_size(64) == 64
    assert _local_search_engine_rotation_block_size(1024) == 1024
    assert _local_search_engine_rotation_block_size(5000) == 1024


def test_local_search_max_union_rotations_tracks_engine_cap():
    assert _local_search_max_union_rotations(64) == 256
    assert _local_search_max_union_rotations(1024) == 4096
    assert _local_search_max_union_rotations(5000) == 4096


def test_pad_local_search_rotations_masks_padding_with_large_negative_prior():
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 12, axis=0)
    log_prior = np.zeros((1, 12), dtype=np.float32)
    padded_rotations, padded_log_prior, actual_count, block_size = _pad_local_search_rotations(
        rotations,
        log_prior,
        5000,
    )

    assert actual_count == 12
    assert block_size == 16
    assert padded_rotations.shape == (16, 3, 3)
    assert padded_log_prior.shape == (1, 16)
    np.testing.assert_allclose(padded_log_prior[0, :12], 0.0)
    np.testing.assert_allclose(padded_log_prior[0, 12:], -1e30)


def test_pad_local_search_rotations_caps_large_neighborhoods_without_recompiling_exact_shape():
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 6000, axis=0)
    log_prior = np.zeros((1, 6000), dtype=np.float32)
    padded_rotations, padded_log_prior, actual_count, block_size = _pad_local_search_rotations(
        rotations,
        log_prior,
        5000,
    )

    assert actual_count == 6000
    assert block_size == 5000
    assert padded_rotations.shape == (6000, 3, 3)
    assert padded_log_prior.shape == (1, 6000)
    np.testing.assert_allclose(padded_log_prior, 0.0)


def test_partition_local_search_groups_keeps_small_exact_unions_together(monkeypatch):
    import recovar.em.dense_single_volume.refine as refine_mod

    def fake_selector(
        prior_rotation_indices,
        sigma_rot,
        sigma_psi,
        healpix_order,
        sigma_cutoff=3.0,
        *,
        per_image=False,
        grid_metadata=None,
    ):
        _ = (sigma_rot, sigma_psi, healpix_order, sigma_cutoff, grid_metadata)
        n = np.asarray(prior_rotation_indices).shape[0]
        assert per_image
        return np.arange(12, dtype=np.int64), np.zeros((n, 12), dtype=np.float32)

    monkeypatch.setattr(refine_mod, "get_local_rotation_grid_fast", fake_selector)

    groups = _partition_local_search_groups(
        np.zeros((4, 3), dtype=np.float32),
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=4,
        image_batch_size=4,
        rotation_block_size=5000,
        grid_metadata={"mode": "factorized", "n_pixels": np.int64(192), "n_psi": np.int64(1536)},
    )

    assert len(groups) == 1
    group_indices, local_indices, local_log_prior = groups[0]
    np.testing.assert_array_equal(np.sort(group_indices), np.array([0, 1, 2, 3], dtype=np.int64))
    assert local_indices.shape == (12,)
    assert local_log_prior.shape == (4, 12)


def test_partition_local_search_groups_splits_large_exact_unions(monkeypatch):
    import recovar.em.dense_single_volume.refine as refine_mod

    def fake_selector(
        prior_rotation_indices,
        sigma_rot,
        sigma_psi,
        healpix_order,
        sigma_cutoff=3.0,
        *,
        per_image=False,
        grid_metadata=None,
    ):
        _ = (sigma_rot, sigma_psi, healpix_order, sigma_cutoff, grid_metadata)
        n = np.asarray(prior_rotation_indices).shape[0]
        assert per_image
        n_union = 5000 if n > 1 else 1200
        return np.arange(n_union, dtype=np.int64), np.zeros((n, n_union), dtype=np.float32)

    monkeypatch.setattr(refine_mod, "get_local_rotation_grid_fast", fake_selector)

    groups = _partition_local_search_groups(
        np.zeros((4, 3), dtype=np.float32),
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=4,
        image_batch_size=4,
        rotation_block_size=5000,
        grid_metadata={"mode": "factorized", "n_pixels": np.int64(192), "n_psi": np.int64(1536)},
    )

    assert len(groups) == 4
    for group_indices, local_indices, local_log_prior in groups:
        assert group_indices.shape == (1,)
        assert local_indices.shape == (1200,)
        assert local_log_prior.shape == (1, 1200)


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
            self._images[i] = _hermitian_image_2d(IMAGE_SHAPE, seed=rng.integers(10000)).reshape(-1)

        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)

        class _Backend:
            image_mask = None

        class _ImageSource:
            process_images = staticmethod(_identity_process)
            backend = _Backend()

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
    return jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)


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
            14.7,
            4.25,
            128,
            particle_diameter=200.0,
        )
        coarse_from_box = compute_coarse_image_size(
            14.7,
            4.25,
            128,
        )
        assert coarse_from_particle == 52
        assert coarse_from_box == 20
        assert coarse_from_particle > coarse_from_box

    def test_clamp_relion_coarse_image_size_caps_at_current_size(self):
        """RELION clamps coarse_size to current_size, not current_size/2."""
        coarse_size = compute_coarse_image_size(
            7.5,
            4.25,
            128,
            particle_diameter=200.0,
        )
        assert coarse_size == 100
        assert clamp_relion_coarse_image_size(coarse_size, current_size=60, ori_size=128) == 60

    def test_make_relion_direction_log_prior_matches_canonical_grid_indices(self):
        order = 2
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.linspace(1.0, float(n_pixels), n_pixels, dtype=np.float32)
        direction_prior /= direction_prior.sum()
        rotations = np.asarray(get_relion_rotation_grid(order), dtype=np.float32)
        view_dirs = rotations[:, 2, :].astype(np.float64)
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
        expected_pixels = hp.vec2pix(
            2**order,
            view_dirs[:, 0],
            view_dirs[:, 1],
            view_dirs[:, 2],
        )

        got = make_relion_direction_log_prior(
            direction_prior,
            order,
            rotations=rotations,
        )
        expected = np.log(direction_prior[expected_pixels]).astype(np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_make_relion_direction_log_prior_tracks_perturbed_view_directions(self):
        order = 3
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.linspace(1.0, float(n_pixels), n_pixels, dtype=np.float32)
        direction_prior /= direction_prior.sum()

        perturbed_rotations = apply_relion_rotation_perturbation(
            np.asarray(get_relion_rotation_grid(order), dtype=np.float32),
            random_perturbation=0.3,
            angular_sampling_deg=360.0 / (6 * 2**order),
        ).astype(np.float32)
        view_dirs = perturbed_rotations[:, 2, :].astype(np.float64)
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
        expected_pixels = hp.vec2pix(
            2**order,
            view_dirs[:, 0],
            view_dirs[:, 1],
            view_dirs[:, 2],
        )

        got = make_relion_direction_log_prior(
            direction_prior,
            order,
            rotations=perturbed_rotations,
        )
        expected = np.log(direction_prior[expected_pixels]).astype(np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_make_relion_direction_log_prior_default_keeps_sample_index_prior(self):
        order = 3
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.linspace(1.0, float(n_pixels), n_pixels, dtype=np.float32)
        direction_prior /= direction_prior.sum()

        got = make_relion_direction_log_prior(direction_prior, order)
        expected = np.log(np.repeat(direction_prior[None, :], rotation_grid_n_in_planes(order), axis=0).reshape(-1))
        np.testing.assert_allclose(got, expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    def test_relion_mode_runs_2_iterations(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
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
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
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
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
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
            [expected_ave_pmax],
            abs=1e-6,
        )
        assert result["convergence_state"].ave_Pmax == pytest.approx(
            expected_ave_pmax,
            abs=1e-6,
        )

    def test_relion_mode_forwards_particle_diameter_to_coarse_size(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
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

    def test_adaptive_pass1_receives_relion_corrections_and_preshifts(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Adaptive pass 1 must receive RELION image corrections and pre-shifts."""
        import recovar.em.dense_single_volume.refine as refine_mod

        recorded = {
            "image_corrections": [],
            "scale_corrections": [],
            "image_pre_shifts": [],
        }

        def fake_significance(*args, **kwargs):
            recorded["image_corrections"].append(kwargs.get("image_corrections"))
            recorded["scale_corrections"].append(kwargs.get("scale_corrections"))
            recorded["image_pre_shifts"].append(kwargs.get("image_pre_shifts"))
            n_rot = args[3].shape[0]
            n_img = args[0].n_units
            return (
                np.ones(n_rot, dtype=bool),
                np.full(n_img, n_rot * translations.shape[0], dtype=np.int32),
                np.zeros(n_img, dtype=np.int32),
                [None] * n_img,
            )

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)
        monkeypatch.setattr(refine_mod, "should_skip_adaptive_pass2", lambda *a, **k: (True, 1.0))

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
            init_image_corrections=[
                np.array([0.8, 1.2], dtype=np.float32),
                np.array([1.1, 0.9], dtype=np.float32),
            ],
            init_scale_corrections=[
                np.array([1.05, 0.95], dtype=np.float32),
                np.array([0.9, 1.1], dtype=np.float32),
            ],
            init_previous_best_translations=[
                np.array([[0.5, -0.25], [0.0, 0.75]], dtype=np.float32),
                np.array([[-0.4, 0.3], [0.6, -0.2]], dtype=np.float32),
            ],
        )

        assert len(recorded["image_corrections"]) == 2
        assert len(recorded["scale_corrections"]) == 2
        assert len(recorded["image_pre_shifts"]) == 2
        np.testing.assert_allclose(
            recorded["image_pre_shifts"][0],
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            recorded["image_pre_shifts"][1],
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

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
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=jnp.float32,
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
        self,
        half_datasets,
        init_volume,
        translations,
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

    def test_significance_batched_matches_run_em_with_pre_shifts_scales_and_projection_padding(
        self,
        half_datasets,
        init_volume,
    ):
        """Adaptive pass 1 must score with the same corrections as the dense engine."""
        dataset = half_datasets[0]
        rotations = _make_rotations(5, seed=23)
        translations = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=jnp.float32,
        )
        init_noise = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
        init_tau = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0
        image_corrections = np.array([0.8, 1.2], dtype=np.float32)
        scale_corrections = np.array([1.1, 0.9], dtype=np.float32)
        image_pre_shifts = np.array([[1.5, -0.5], [-1.0, 1.25]], dtype=np.float32)
        current_size = 6

        _, expected_ha, _, _ = run_em_v2(
            dataset,
            init_volume,
            init_tau,
            init_noise,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=dataset.n_units,
            rotation_block_size=rotations.shape[0],
            current_size=current_size,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            score_with_masked_images=True,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
        )

        _, _, actual_ha = _compute_significance_batched(
            dataset,
            init_volume,
            init_noise,
            rotations,
            translations,
            "linear_interp",
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=rotations.shape[0],
            current_size=current_size,
            score_with_masked_images=True,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
        )

        np.testing.assert_array_equal(np.asarray(actual_ha), np.asarray(expected_ha))

    def test_relion_mode_convergence_state(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
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

    def test_relion_mode_uses_tau2_from_weights_for_prior(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """RELION mode should compute tau2 from Ft_ctf weights + FSC (RELION order)."""
        from recovar.reconstruction import regularization

        called = {"tau2": 0}

        original_tau2 = regularization.compute_relion_tau2_from_weights

        def wrap_tau2(*args, **kwargs):
            called["tau2"] += 1
            return original_tau2(*args, **kwargs)

        monkeypatch.setattr(regularization, "compute_relion_tau2_from_weights", wrap_tau2)

        grid_size = int(np.sqrt(IMAGE_SIZE))
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
            init_fsc=np.ones(grid_size // 2),
        )

        assert called["tau2"] >= 1

    def test_relion_mode_current_size_no_longer_uses_weight_based_data_vs_prior(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
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
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
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
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
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
        assert counts.shape == (half_datasets[0].n_units + half_datasets[1].n_units,)
        np.testing.assert_array_equal(
            counts,
            np.full_like(counts, captured["n_rot"] * len(np.asarray(translations))),
        )

    def test_relion_mode_uses_dense_exact_pass2_when_all_samples_significant(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
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
            rpf = kwargs.get("reconstruction_padding_factor", 1)
            recon_vol_size = VOLUME_SIZE * rpf**3
            result = (
                jnp.zeros(recon_vol_size, dtype=jnp.complex64),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
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
                result = result + (
                    NoiseStats(
                        wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                        wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                        sumw=float(n_images),
                    ),
                )
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
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
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
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Pass 1 should start from the coarse RELION HEALPix order, not a fine caller grid."""
        import recovar.em.dense_single_volume.refine as refine_mod

        captured = {"grid_order": None}

        def fake_grid(order):
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

        monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_grid)
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
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
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
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
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
# Test 3: Local search oversampling regression
# ===========================================================================


def test_local_search_uses_fine_rotation_grid_when_oversampling_is_enabled(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Local search must use the fine-order grid when oversampling is enabled."""
    import recovar.em.dense_single_volume.refine as refine_mod

    order_sizes = {4: 4, 5: 9}
    grid_calls = []
    local_calls = []

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        grid_calls.append(("rot", order))
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        grid_calls.append(("euler", order))
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "n_rot": int(np.asarray(rotation_grid_rotations).shape[0]),
                "n_euler": int(np.asarray(rotation_grid_eulers).shape[0]),
            }
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "_run_grouped_local_search_em", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: np.ones(
            max(1, fake_rotation_grid_size(healpix_order)),
            dtype=np.float64,
        )
        / max(1, fake_rotation_grid_size(healpix_order)),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=99),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
    )

    assert any(kind == "rot" and order == 5 for kind, order in grid_calls)
    assert any(kind == "euler" and order == 5 for kind, order in grid_calls)
    assert local_calls
    for call in local_calls:
        assert call["healpix_order"] == 5
        assert call["n_rot"] == order_sizes[5]
        assert call["n_euler"] == order_sizes[5]


def test_local_search_uses_negative_rounded_previous_offsets_for_translation_prior(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Local-search translation priors must be centered at -ROUND(old_offset)."""
    import recovar.em.dense_single_volume.refine as refine_mod

    order_sizes = {4: 4, 5: 9}
    prev_h1 = np.array([[0.5, -0.25], [1.0, 0.75]], dtype=np.float32)
    prev_h2 = np.array([[-0.75, 0.25], [0.25, -1.25]], dtype=np.float32)
    local_prior_translations = []

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_run_em_v2(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            mean,
            mean_variance,
            noise_variance,
            prior_rotations,
            rotation_grid_rotations,
            rotation_grid_eulers,
            healpix_order,
            sigma_rot,
            sigma_psi,
            translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            kwargs,
        )
        local_prior_translations.append(np.asarray(prior_translations, dtype=np.float32).copy())
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em_v2", fake_run_em_v2)
    monkeypatch.setattr(refine_mod, "_run_grouped_local_search_em", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "_compute_significance_batched",
        lambda *args, **kwargs: (
            np.ones(args[3].shape[0], dtype=bool),
            np.full(args[0].n_units, args[3].shape[0] * np.asarray(translations).shape[0], dtype=np.int32),
            np.zeros(args[0].n_units, dtype=np.int32),
            [None] * args[0].n_units,
        ),
    )
    monkeypatch.setattr(refine_mod, "should_skip_adaptive_pass2", lambda *a, **k: (True, 1.0))
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: np.ones(
            max(1, fake_rotation_grid_size(healpix_order)),
            dtype=np.float64,
        )
        / max(1, fake_rotation_grid_size(healpix_order)),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=123),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
    )

    assert len(local_prior_translations) == 2
    np.testing.assert_allclose(local_prior_translations[0], -np.rint(prev_h1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(local_prior_translations[1], -np.rint(prev_h2), rtol=1e-6, atol=1e-6)


def test_first_local_iteration_uses_previous_best_rotations_without_dense_bootstrap(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """hp4 should enter local search immediately when previous best rotations exist."""
    import recovar.em.dense_single_volume.refine as refine_mod

    dense_calls = []
    local_calls = []
    prev_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    def fake_run_em_v2(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        dense_calls.append(int(np.asarray(rotations).shape[0]))
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            mean,
            mean_variance,
            noise_variance,
            sigma_rot,
            sigma_psi,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            kwargs,
        )
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "prior_shape": np.asarray(prior_rotations).shape,
                "grid_shape": np.asarray(rotation_grid_rotations).shape,
            }
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotation_grid_rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "run_em_v2", fake_run_em_v2)
    monkeypatch.setattr(refine_mod, "_run_grouped_local_search_em", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: np.ones(
            max(1, rotation_grid_size(healpix_order)),
            dtype=np.float64,
        )
        / max(1, rotation_grid_size(healpix_order)),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(rotation_grid_size(4), seed=7),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=512,
        init_current_size=16,
        adaptive_oversampling=0,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        replay_iteration_overrides=[
            {
                "local_search": True,
                "healpix_order": 4,
                "previous_best_rotation_eulers": [prev_h1, prev_h2],
            }
        ],
        skip_final_iteration=True,
    )

    assert local_calls
    assert not dense_calls
    for call in local_calls:
        assert call["healpix_order"] == 4
        assert call["prior_shape"][0] == half_datasets[0].n_units


def test_init_previous_best_rotation_eulers_seed_first_local_iteration(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Initial previous-best eulers should skip the dense hp4 bootstrap."""
    import recovar.em.dense_single_volume.refine as refine_mod

    dense_calls = []
    local_calls = []
    prev_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    def fake_run_em_v2(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        dense_calls.append(int(np.asarray(rotations).shape[0]))
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            mean,
            mean_variance,
            noise_variance,
            sigma_rot,
            sigma_psi,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            kwargs,
        )
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "prior_shape": np.asarray(prior_rotations).shape,
                "grid_shape": np.asarray(rotation_grid_rotations).shape,
            }
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotation_grid_rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "run_em_v2", fake_run_em_v2)
    monkeypatch.setattr(refine_mod, "_run_grouped_local_search_em", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: np.ones(
            max(1, rotation_grid_size(healpix_order)),
            dtype=np.float64,
        )
        / max(1, rotation_grid_size(healpix_order)),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(rotation_grid_size(4), seed=11),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=512,
        init_current_size=16,
        adaptive_oversampling=0,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_rotation_eulers=[prev_h1, prev_h2],
        skip_final_iteration=True,
    )

    assert local_calls
    assert not dense_calls
    for call in local_calls:
        assert call["healpix_order"] == 4
        assert call["prior_shape"][0] == half_datasets[0].n_units


def test_relion_mode_writes_absolute_translations_from_rounded_previous_offset(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """RELION-mode writeback should use ROUND(old_offset) + delta."""
    import recovar.em.dense_single_volume.refine as refine_mod

    prev_h1 = np.array([[0.5, -0.25], [0.0, 0.75]], dtype=np.float32)
    prev_h2 = np.array([[-0.4, 0.3], [0.6, -0.2]], dtype=np.float32)
    chosen_trans = np.asarray(translations[1], dtype=np.float32)

    def fake_run_em_v2(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, rotations, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        hard_assignment = np.full(experiment_dataset.n_units, 1, dtype=np.int32)
        return (
            None,
            hard_assignment,
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "run_em_v2", fake_run_em_v2)

    result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(1, seed=123),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=1,
        init_current_size=16,
        adaptive_oversampling=0,
        nside_level=1,
        mode="relion",
        init_healpix_order=1,
        max_healpix_order=1,
        init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
        skip_final_iteration=True,
    )

    expected_h1 = np.rint(prev_h1) + chosen_trans[None, :]
    expected_h2 = np.rint(prev_h2) + chosen_trans[None, :]
    np.testing.assert_allclose(half_datasets[0].translations, expected_h1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(half_datasets[1].translations, expected_h2, rtol=1e-6, atol=1e-6)

    best_hist = result["best_translations_history"]
    assert len(best_hist) == 1
    np.testing.assert_allclose(
        best_hist[0],
        np.concatenate([expected_h1, expected_h2], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_local_search_decodes_hard_assignments_on_fine_grid(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Oversampled local-search assignments must be decoded on the fine grid."""
    import recovar.em.dense_single_volume.refine as refine_mod

    order_sizes = {4: 4, 5: 9}
    fine_idx = order_sizes[5] - 1
    trans_idx = 1

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        mats = np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))
        for i in range(order_sizes[order]):
            mats[i, 0, 0] = 1.0 + i
        return mats

    def fake_get_grid_eulers(order):
        order = int(order)
        vals = np.arange(order_sizes[order], dtype=np.float32)
        return np.stack([vals, vals + 100.0, vals + 200.0], axis=1)

    def fake_run_em_v2(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, rotations, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            mean,
            mean_variance,
            noise_variance,
            prior_rotations,
            rotation_grid_rotations,
            rotation_grid_eulers,
            healpix_order,
            sigma_rot,
            sigma_psi,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            kwargs,
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        assignment = np.full(
            experiment_dataset.n_units,
            fine_idx * np.asarray(translations).shape[0] + trans_idx,
            dtype=np.int32,
        )
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            assignment,
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em_v2", fake_run_em_v2)
    monkeypatch.setattr(refine_mod, "_run_grouped_local_search_em", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: np.ones(
            max(1, fake_rotation_grid_size(healpix_order)),
            dtype=np.float64,
        )
        / max(1, fake_rotation_grid_size(healpix_order)),
    )

    result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=321),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        perturb_factor=0.0,
    )

    expected_euler = fake_get_grid_eulers(5)[fine_idx]
    observed = np.asarray(result["best_rotation_eulers_history"][1], dtype=np.float32)
    assert observed.shape[0] == N_IMAGES
    np.testing.assert_allclose(
        observed,
        np.repeat(expected_euler[None, :], N_IMAGES, axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


# ===========================================================================
# Test 3: Invalid mode
# ===========================================================================


class TestInvalidMode:
    def test_invalid_mode_raises(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
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
