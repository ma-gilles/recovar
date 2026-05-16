import jax.numpy as jnp
import numpy as np
import pytest

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout
from recovar.em.ppca_refinement.config import GeometryConfig, PoseSelectionConfig, ScheduleConfig
from recovar.em.ppca_refinement.local_dataset import (
    run_local_ppca_fused_em_iteration,
    run_local_ppca_pose_scoring_iteration,
)

pytestmark = pytest.mark.unit

IMAGE_SHAPE = (4, 4)
VOLUME_SHAPE = (4, 4, 4)
N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
HALF_VOL = int(np.prod(ftu.volume_shape_to_half_volume_shape(VOLUME_SHAPE)))


def _identity_ctf(params, image_shape, voxel_size, *, half_image=False):
    del voxel_size
    n_pix = image_shape[0] * (image_shape[1] // 2 + 1) if half_image else image_shape[0] * image_shape[1]
    return jnp.ones((params.shape[0], n_pix), dtype=jnp.float32)


class _TinyPPCAData:
    def __init__(self):
        rng = np.random.default_rng(14)
        self.image_shape = IMAGE_SHAPE
        self.volume_shape = VOLUME_SHAPE
        self.grid_size = IMAGE_SHAPE[0]
        self.voxel_size = 1.0
        self.n_images = 2
        self.n_units = 2
        self.dtype = jnp.complex64
        self.ctf_evaluator = _identity_ctf
        self.CTF_params = np.zeros((2, 9), dtype=np.float32)
        self._images = (
            rng.standard_normal((2, N_HALF)) + 1j * rng.standard_normal((2, N_HALF))
        ).astype(np.complex64)
        self.image_source = self

    def process_images(self, images, apply_image_mask=False):
        del apply_image_mask
        return images

    def process_images_half(self, images, apply_image_mask=False):
        del apply_image_mask
        return images

    @property
    def already_prefetches(self):
        return True

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        del by_image, kwargs
        indices = np.arange(self.n_images) if indices is None else np.asarray(indices, dtype=np.int64)
        for start in range(0, indices.size, int(batch_size)):
            idx = indices[start : start + int(batch_size)]
            yield (
                jnp.asarray(self._images[idx]),
                None,
                None,
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )


def _half_volume(seed):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
    full = np.fft.fftshift(np.fft.fftn(real)).astype(np.complex64)
    return np.asarray(ftu.full_volume_to_half_volume(full, VOLUME_SHAPE), dtype=np.complex64).reshape(-1)


def test_local_top_p_uses_global_rotation_ids():
    dataset = _TinyPPCAData()
    rotations = np.asarray(
        [
            np.eye(3),
            np.diag([1.0, -1.0, -1.0]),
            np.diag([-1.0, 1.0, -1.0]),
            np.diag([-1.0, -1.0, 1.0]),
        ],
        dtype=np.float32,
    )
    layout = LocalHypothesisLayout(
        n_global_rotations=16,
        n_pixels=4,
        n_psi=4,
        rotation_offsets=np.asarray([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.asarray([3, 7, 4, 8], dtype=np.int32),
        rotations_flat=rotations,
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.asarray([2, 2], dtype=np.int32),
        translation_grid=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )
    result = run_local_ppca_fused_em_iteration(
        dataset,
        _half_volume(1),
        _half_volume(2)[:, None] * np.asarray(0.05, dtype=np.float32),
        mean_prior=jnp.ones((HALF_VOL,), dtype=jnp.float32),
        W_prior=jnp.ones((HALF_VOL, 1), dtype=jnp.float32),
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        local_layout=layout,
        geometry=GeometryConfig(current_size=4, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=2, rotation_block_size=4),
        enforce_x0=False,
        pose_selection=PoseSelectionConfig(top_p_poses=2, top_pose_max_log_score_gap=np.inf),
    )
    top_rotation_idx = np.asarray(result.diagnostics["top_rotation_idx"])
    top_rotation_id = np.asarray(result.diagnostics["top_rotation_id"])
    assert top_rotation_idx.shape == top_rotation_id.shape == (2, 2)
    assert np.all((top_rotation_idx == -1) | ((0 <= top_rotation_idx) & (top_rotation_idx < 2)))
    valid_global = top_rotation_id[top_rotation_id >= 0]
    assert valid_global.size
    assert set(valid_global.tolist()).issubset({3, 7, 4, 8})
    assert not set(valid_global.tolist()).issubset({0, 1})


def test_local_pose_score_only_uses_global_rotation_ids():
    dataset = _TinyPPCAData()
    rotations = np.asarray(
        [
            np.eye(3),
            np.diag([1.0, -1.0, -1.0]),
            np.diag([-1.0, 1.0, -1.0]),
            np.diag([-1.0, -1.0, 1.0]),
        ],
        dtype=np.float32,
    )
    layout = LocalHypothesisLayout(
        n_global_rotations=16,
        n_pixels=4,
        n_psi=4,
        rotation_offsets=np.asarray([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.asarray([3, 7, 4, 8], dtype=np.int32),
        rotations_flat=rotations,
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.asarray([2, 2], dtype=np.int32),
        translation_grid=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )
    result = run_local_ppca_pose_scoring_iteration(
        dataset,
        _half_volume(1),
        _half_volume(2)[:, None] * np.asarray(0.05, dtype=np.float32),
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        local_layout=layout,
        geometry=GeometryConfig(current_size=4, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=2, rotation_block_size=4),
        pose_selection=PoseSelectionConfig(top_p_poses=2, top_pose_max_log_score_gap=np.inf),
    )
    assert result.diagnostics["local_pose_score_only"] is True
    top_rotation_idx = np.asarray(result.diagnostics["top_rotation_idx"])
    top_rotation_id = np.asarray(result.diagnostics["top_rotation_id"])
    assert top_rotation_idx.shape == top_rotation_id.shape == (2, 2)
    assert np.all((top_rotation_idx == -1) | ((0 <= top_rotation_idx) & (top_rotation_idx < 2)))
    valid_global = top_rotation_id[top_rotation_id >= 0]
    assert valid_global.size
    assert set(valid_global.tolist()).issubset({3, 7, 4, 8})
    assert not set(valid_global.tolist()).issubset({0, 1})
