"""Hierarchical BnB selector test: verifies the axis-angle/shift subdivision.

This test does NOT run a full EM iteration; it just exercises the
``select_bnb_support_hierarchical_k1`` driver on the MockDataset fixture and
checks that:
  - the final rotation/shift grids are at the paper-stated spacing,
  - the per-image survivor mask has a sane shape,
  - the diagnostic spacing schedule matches the paper.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.bnb import BranchBoundOptions
from recovar.em.dense_single_volume.bnb.hierarchical_support import (
    select_bnb_support_hierarchical_k1,
)

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
H, W = IMAGE_SHAPE
N_HALF = H * (W // 2 + 1)
N_IMAGES = 2
SEED = 1729


def _raw_real_image_2d(image_shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(image_shape).astype(np.float32)


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

        self._images = np.zeros((N_IMAGES, *IMAGE_SHAPE), dtype=np.float32)
        for i in range(N_IMAGES):
            self._images[i] = _raw_real_image_2d(IMAGE_SHAPE, seed=rng.integers(10000))

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


def test_hierarchical_selector_final_spacing_matches_paper():
    """After options.n_subdivisions subdivisions the spacing == initial/2^N."""
    rng = np.random.default_rng(SEED)
    ds = _MockDataset(rng)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=42)
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)

    # Tiny n_subdivisions=2 so the test stays fast: 24->12->6 deg, 5->2.5->1.25 px.
    options = BranchBoundOptions(
        enabled=True,
        subdivision_mode="axis_angle_hierarchical",
        n_subdivisions=2,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=5.0,
        max_shift_px=5.0,
        posterior_tail_tol=1.0,
        max_orientation_fraction=1.0,
        max_shift_fraction=1.0,
        min_orientations_per_image=10**6,
        min_shifts_per_image=10**6,
        min_joint_candidates_per_image=10**6,
        max_joint_candidates_per_image=10**12,
    )

    support, rotations_final, translations_final = select_bnb_support_hierarchical_k1(
        ds, volume, noise_variance,
        max_shift_px=5.0,
        current_size=8,
        options=options,
        disc_type="linear_interp",
        image_batch_size=N_IMAGES,
        rotation_block_size=64,
    )

    # Per-image sample mask should have shape (n_images, n_rot_final, n_trans_final).
    assert support.sample_mask_per_image.shape[0] == N_IMAGES
    assert support.sample_mask_per_image.shape[1] == rotations_final.shape[0]
    assert support.sample_mask_per_image.shape[2] == translations_final.shape[0]

    # Final spacing reported in diagnostics matches paper schedule.
    final_stage = support.diagnostics.stages[-1]
    np.testing.assert_allclose(
        final_stage.angular_spacing_deg, 24.0 / 4.0, atol=1e-6,
    )
    np.testing.assert_allclose(
        final_stage.shift_spacing_px, 5.0 / 4.0, atol=1e-6,
    )


def test_hierarchical_selector_n_subdivisions_zero_returns_initial_grid():
    """n_subdivisions=0 means just evaluate the initial grid once."""
    rng = np.random.default_rng(SEED + 7)
    ds = _MockDataset(rng)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=42)
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)

    options = BranchBoundOptions(
        enabled=True,
        subdivision_mode="axis_angle_hierarchical",
        n_subdivisions=0,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=5.0,
        max_shift_px=5.0,
        posterior_tail_tol=1.0,
        max_orientation_fraction=1.0,
        max_shift_fraction=1.0,
        min_orientations_per_image=10**6,
        min_shifts_per_image=10**6,
        min_joint_candidates_per_image=10**6,
        max_joint_candidates_per_image=10**12,
    )
    support, rotations_final, translations_final = select_bnb_support_hierarchical_k1(
        ds, volume, noise_variance,
        max_shift_px=5.0,
        current_size=8,
        options=options,
        disc_type="linear_interp",
        image_batch_size=N_IMAGES,
        rotation_block_size=64,
    )

    # The "final" stage equals the initial one. Spacing == 24 deg / 5 px.
    final_stage = support.diagnostics.stages[-1]
    np.testing.assert_allclose(final_stage.angular_spacing_deg, 24.0, atol=1e-6)
    np.testing.assert_allclose(final_stage.shift_spacing_px, 5.0, atol=1e-6)
