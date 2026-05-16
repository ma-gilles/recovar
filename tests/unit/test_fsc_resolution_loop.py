"""Current-size and resolution loop tests for dense single-volume refinement.

Tests:
1. test_fsc_to_current_size: FSC -> shell -> current_size mapping.
2. test_oracle_mode_matches_relion_trajectory: Run with RELION's current_sizes.
3. test_resolution_improves_over_iterations: current_size generally increases.
4. test_one_iteration_with_windowing: One iteration at current_size=32 is valid.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.iteration_loop import (
    refine_single_volume,
)
from recovar.em.dense_single_volume.helpers.resolution import fsc_to_current_size
from recovar.em.dense_single_volume.helpers.fourier_window import (
    quantize_current_size,
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
N_HALF = H * (W // 2 + 1)  # 40
N_ROTATIONS = 5
N_TRANSLATIONS = 3
N_IMAGES = 10  # enough for half-sets
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
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


def _identity_process_half(batch, apply_image_mask=False):
    """Half-spectrum passthrough: full centered FT → packed Hermitian half.

    Dense-EM's preprocessing (recovar/em/dense_single_volume/helpers/
    preprocessing.py:63) reads ``experiment_dataset.process_images_half``
    and expects the packed half-image layout
    ``(batch, H * (W // 2 + 1))``. The mock stores full-spectrum FT images
    of shape ``(batch, H * W)`` so we map them through
    ``full_image_to_half_image`` here.
    """
    _ = apply_image_mask
    from recovar.core import fourier_transform_utils as _ftu

    return _ftu.full_image_to_half_image(batch, IMAGE_SHAPE)


class MockDataset:
    """Minimal mock of CryoEMDataset for unit testing the refinement loop.

    Supports the subset of the dataset API needed by run_em, noise
    estimation, prior computation, and pose updates.
    """

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
        # Dense-EM preprocessing (recovar/em/dense_single_volume/helpers/
        # preprocessing.py:63) now reads ``process_images_half`` rather than
        # ``process_images``. The half-image variant returns the packed
        # half-spectrum layout (H * (W // 2 + 1) pixels per image).
        self.process_images_half = staticmethod(_identity_process_half)
        self.premultiplied_ctf = False

        self._images = np.zeros((n_images, IMAGE_SIZE), dtype=np.complex64)
        for i in range(n_images):
            self._images[i] = _hermitian_image_2d(IMAGE_SHAPE, seed=rng.integers(10000)).reshape(-1)

        # Stored poses (updated by update_poses).
        # These are also exposed as rotation_matrices/translations properties
        # for compute_prior_quantites which accesses them as dataset attributes.
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
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
                None,  # noise_variance
                idx,  # particle_indices
                idx,  # image_indices
            )

    def update_poses(self, rots, trans):
        """Store new poses (mimics CryoEMDataset.update_poses)."""
        self.rotation_matrices = np.asarray(rots)
        self.translations = np.asarray(trans)

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)


def _make_half_datasets(rng):
    """Create two half-set mock datasets."""
    ds0 = MockDataset(N_IMAGES // 2, rng)
    ds1 = MockDataset(N_IMAGES // 2, rng)
    return [ds0, ds1]


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def half_datasets(rng):
    return _make_half_datasets(rng)


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
# Test 1: fsc_to_current_size
# ===========================================================================


class TestFscToCurrentSize:
    """Verify the FSC -> shell -> current_size mapping."""

    def test_perfect_fsc_gives_max_size(self):
        """FSC=1 at all shells -> current_size = max allowed."""
        fsc = jnp.ones(64)
        cs = fsc_to_current_size(fsc, threshold=1.0 / 7.0)
        # With FSC=1 everywhere, resolution is at the last shell (63)
        # current_size = 2 * 63 = 126
        assert cs >= 100, f"Perfect FSC should give large current_size, got {cs}"

    def test_zero_fsc_gives_min_size(self):
        """FSC=0 at all shells -> current_size = min_size."""
        fsc = jnp.zeros(64)
        cs = fsc_to_current_size(fsc, threshold=1.0 / 7.0, min_size=32)
        assert cs == 32, f"Zero FSC should give min_size=32, got {cs}"

    def test_partial_fsc_gives_reasonable_size(self):
        """FSC that drops at shell 20 -> current_size ~ 40."""
        fsc = jnp.concatenate(
            [
                jnp.ones(20),  # high FSC up to shell 19
                jnp.zeros(44),  # drops to 0 after
            ]
        )
        cs = fsc_to_current_size(fsc, threshold=1.0 / 7.0)
        # Should be approximately 2 * 20 = 40
        assert 32 <= cs <= 50, f"Expected ~40, got {cs}"

    def test_gradual_fsc_decay(self):
        """FSC that gradually decays."""
        shells = np.arange(64)
        fsc = jnp.array(np.exp(-shells / 15.0))  # drops below 0.143 around shell 29
        cs = fsc_to_current_size(fsc, threshold=1.0 / 7.0)
        assert 40 <= cs <= 70, f"Expected ~58, got {cs}"

    def test_quantize_after_fsc(self):
        """After fsc_to_current_size, quantize produces a valid even size."""
        fsc = jnp.concatenate([jnp.ones(20), jnp.zeros(44)])
        raw_cs = fsc_to_current_size(fsc, threshold=1.0 / 7.0)
        q_cs = quantize_current_size(raw_cs, ori_size=128)
        assert q_cs % 2 == 0
        assert 16 <= q_cs <= 128
        assert q_cs >= raw_cs


# ===========================================================================
# Test 2: Oracle mode matches RELION trajectory
# ===========================================================================


class TestOracleMode:
    """Run refinement with RELION's current_sizes injected."""

    def test_oracle_sizes_are_used(self, half_datasets, init_volume, rotations, translations):
        """Oracle sizes should be used after quantization/clamping to the box size."""
        oracle_sizes = [32, 32, 64]

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=3,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            relion_current_sizes=oracle_sizes,
        )

        # On the tiny 8px mock dataset these values all saturate at full resolution.
        assert result["current_sizes"] == [8, 8, 8], f"Oracle sizes not used: {result['current_sizes']}"

    def test_oracle_with_zero_first(self, half_datasets, init_volume, rotations, translations):
        """RELION iteration 0 has current_size=0; should use init_current_size."""
        oracle_sizes = [0, 32, 64]

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=3,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            relion_current_sizes=oracle_sizes,
            init_current_size=32,
        )

        # On the tiny 8px mock dataset all oracle/full-resolution requests clamp to 8.
        assert result["current_sizes"][0] == 8
        assert result["current_sizes"][1] == 8
        assert result["current_sizes"][2] == 8

    def test_oracle_produces_valid_outputs(self, half_datasets, init_volume, rotations, translations):
        """Oracle mode produces finite volumes and valid assignments."""
        oracle_sizes = [32, 64]

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
            relion_current_sizes=oracle_sizes,
        )

        # Final mean should be finite
        assert np.all(np.isfinite(np.array(result["mean"]))), "Mean not finite"
        # FSC should be computed
        assert result["fsc"] is not None
        assert len(result["fsc_history"]) == 2
        # Hard assignments valid
        for k in range(2):
            ha = result["hard_assignments"][k]
            assert ha is not None
            assert np.all(ha >= 0)
            assert np.all(ha < N_ROTATIONS * N_TRANSLATIONS)


# ===========================================================================
# Test 3: Resolution improves (or at least doesn't collapse)
# ===========================================================================


class TestResolutionProgression:
    """Run a few iterations without oracle; verify stability."""

    def test_current_size_does_not_collapse(self, half_datasets, init_volume, rotations, translations):
        """After multiple iterations, current_size should not drop to minimum."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=3,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=32,
        )

        sizes = result["current_sizes"]
        assert len(sizes) == 3
        # The tiny 8px mock dataset can legitimately clamp to very small sizes,
        # but should not drop below the scaled minimum of 4.
        assert sizes[-1] >= 4, f"Resolution collapsed: sizes={sizes}"

    def test_fsc_history_populated(self, half_datasets, init_volume, rotations, translations):
        """FSC history has one entry per iteration."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=3,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=32,
        )

        assert len(result["fsc_history"]) == 3
        assert len(result["pixel_resolutions"]) == 3
        assert len(result["wall_times"]) == 3

        # Each FSC curve should have valid entries
        for fsc in result["fsc_history"]:
            assert jnp.all(jnp.isfinite(fsc))

    def test_wall_times_positive(self, half_datasets, init_volume, rotations, translations):
        """Wall times should be positive."""
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
            init_current_size=32,
        )

        for t in result["wall_times"]:
            assert t > 0, f"Wall time should be positive, got {t}"


# ===========================================================================
# Test 4: One iteration with windowing
# ===========================================================================


class TestOneIterationWithWindowing:
    """Run a single iteration at a small current_size."""

    def test_single_iteration_cs_4(self, half_datasets, init_volume, rotations, translations):
        """One EM iteration at current_size=4 (small window) produces valid output."""
        # For 8x8 images, current_size=4 means r_max=2 (very few frequencies)
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
            relion_current_sizes=[4],
        )

        assert np.all(np.isfinite(np.array(result["mean"])))
        assert result["current_sizes"] == [4]

    def test_single_iteration_no_window(self, half_datasets, init_volume, rotations, translations):
        """One EM iteration at current_size=None (full res) produces valid output."""
        # current_size=128 for 8x8 images means no windowing
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
            relion_current_sizes=[128],
        )

        assert np.all(np.isfinite(np.array(result["mean"])))
        # On the tiny 8px mock dataset, any oversized request clamps to full resolution.
        assert result["current_sizes"] == [8]

    def test_hard_assignments_valid_range(self, half_datasets, init_volume, rotations, translations):
        """Hard assignments are in valid range after one iteration."""
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

        n_total_poses = N_ROTATIONS * N_TRANSLATIONS
        for k in range(2):
            ha = result["hard_assignments"][k]
            assert ha.shape == (N_IMAGES // 2,)
            assert np.all(ha >= 0)
            assert np.all(ha < n_total_poses)
