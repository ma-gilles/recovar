"""End-to-end smoke test for paper-faithful per-image-ragged BnB engine.

Drives the full pipeline on a tiny synthetic dataset (2 images, 8x8 box):
- per-image state initialisation,
- per-image scoring,
- per-image bound + prune + subdivide for n_subdivisions=2,
- final layout build + run_local_em_exact,
- output tuple shape matches what _score_half_bnb_k1 expects.

This does NOT verify quality vs RELION (that's the integration test) — it
just confirms the engine runs end-to-end without crashing and produces
sensible candidate counts.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.bnb import BranchBoundOptions
from recovar.em.dense_single_volume.bnb.per_image_engine import (
    run_paper_faithful_bnb_em_k1,
)

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
N_HALF = 8 * 5
N_IMAGES = 2


def _raw_real_image_2d(image_shape, seed):
    return np.random.default_rng(seed).standard_normal(image_shape).astype(np.float32)


def _hermitian_volume(volume_shape, seed):
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _identity_ctf_full(params, image_shape=None, voxel_size=None, *, half_image=False):
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _raw_real_process(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


class _MockDataset:
    def __init__(self, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = N_IMAGES
        self.n_units = N_IMAGES
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((N_IMAGES, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf_full)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)
        self._images = np.stack(
            [_raw_real_image_2d(IMAGE_SHAPE, seed=rng.integers(10000)) for _ in range(N_IMAGES)],
            axis=0,
        )

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                None,
                None,
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)


def test_paper_faithful_engine_runs_end_to_end():
    rng = np.random.default_rng(31337)
    ds = _MockDataset(rng)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=42)
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0

    options = BranchBoundOptions(
        enabled=True,
        subdivision_mode="paper_faithful",
        n_subdivisions=2,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=2.0,
        max_shift_px=2.0,
        # Loose pruning so the test exercises the engine without surprises.
        posterior_tail_tol=0.5,
        max_orientation_fraction=1.0,
        max_shift_fraction=1.0,
        min_orientations_per_image=8,
        min_shifts_per_image=4,
        min_joint_candidates_per_image=8,
        max_joint_candidates_per_image=10**12,
    )

    out = run_paper_faithful_bnb_em_k1(
        ds, volume, mean_variance, noise_variance,
        current_size=8, options=options, disc_type="linear_interp",
        image_batch_size=N_IMAGES, rotation_block_size=64,
        return_best_pose_details=True, accumulate_noise=True,
        score_with_masked_images=False,
    )

    # run_local_em_exact return tuple with return_best_pose_details=True,
    # accumulate_noise=True:
    #   0=Ft_y, 1=Ft_ctf, 2=hard_assignment, 3=best_pose_rotations,
    #   4=best_pose_translations, 5=best_pose_rotation_ids,
    #   6=relion_stats, 7=noise_stats
    assert len(out) >= 8, f"Unexpected return tuple length {len(out)}"
    Ft_y = np.asarray(out[0])
    Ft_ctf = np.asarray(out[1])
    hard_assignment = np.asarray(out[2])
    best_rots = np.asarray(out[3])
    best_trans = np.asarray(out[4])

    assert Ft_y.size > 0
    assert Ft_ctf.size > 0
    assert hard_assignment.shape == (N_IMAGES,)
    assert best_rots.shape[0] == N_IMAGES
    assert best_rots.shape[-2:] == (3, 3)
    assert best_trans.shape[0] == N_IMAGES
    # All values finite.
    assert np.all(np.isfinite(Ft_y.real))
    assert np.all(np.isfinite(Ft_ctf.real))
