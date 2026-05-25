"""Equivalence test: bucketed scorer must agree with per-image-loop scorer.

If the bucketed kernel produces the same per-image scores (modulo
floating-point noise) on a tiny dataset with per-image-ragged candidate
sets, then it's safe to use as a drop-in replacement at scale.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.bnb.per_image_score import (
    score_per_image_at_low_freq,
)
from recovar.em.dense_single_volume.bnb.per_image_score_bucketed import (
    score_per_image_at_low_freq_bucketed,
)
from recovar.em.dense_single_volume.bnb.per_image_state import (
    PerImageBnBPoseState,
    initialize_per_image_state,
    subdivide_per_image_state,
)

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
N_IMAGES = 3


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
        self._images = rng.standard_normal((N_IMAGES, *IMAGE_SHAPE)).astype(np.float32)

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


def _hermitian_volume(seed):
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _assert_per_image_scores_close(a: list[np.ndarray], b: list[np.ndarray], atol: float, rtol: float):
    assert len(a) == len(b)
    for i in range(len(a)):
        ai = np.asarray(a[i])
        bi = np.asarray(b[i])
        assert ai.shape == bi.shape, f"image {i}: {ai.shape} vs {bi.shape}"
        # Compare only finite entries (both should be -inf at masked positions).
        finite = np.isfinite(ai) & np.isfinite(bi)
        np.testing.assert_allclose(
            ai[finite], bi[finite],
            atol=atol, rtol=rtol,
            err_msg=f"per-image scores differ for image {i}",
        )
        # -inf positions should agree.
        np.testing.assert_array_equal(
            ~np.isfinite(ai), ~np.isfinite(bi),
            err_msg=f"image {i}: -inf positions differ between bucketed and loop",
        )


def test_bucketed_matches_per_image_loop_full_state():
    """Initial all-active per-image state — both scorers must agree."""
    rng = np.random.default_rng(2026)
    ds = _MockDataset(rng)
    mean = _hermitian_volume(seed=99)
    nv = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)

    state = initialize_per_image_state(
        n_images=N_IMAGES,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=2.0,
        max_shift_px=2.0,
    )
    image_indices = np.arange(N_IMAGES, dtype=np.int32)

    s_loop = score_per_image_at_low_freq(
        ds, mean, nv, state, image_indices,
        L=4, disc_type="linear_interp", image_batch_size=N_IMAGES,
    )
    s_bucket = score_per_image_at_low_freq_bucketed(
        ds, mean, nv, state, image_indices,
        L=4, disc_type="linear_interp", image_batch_size=N_IMAGES,
        axis_quantum=64, shift_quantum=4,
    )
    _assert_per_image_scores_close(s_loop, s_bucket, atol=1e-4, rtol=5e-4)


def test_bucketed_matches_per_image_loop_after_subdivision():
    """After subdivision, per-image grids differ — bucketed must still agree."""
    rng = np.random.default_rng(2026)
    ds = _MockDataset(rng)
    mean = _hermitian_volume(seed=42)
    nv = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)

    state = initialize_per_image_state(
        n_images=N_IMAGES,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=2.0,
        max_shift_px=2.0,
    )
    # Force divergence: image 0 keeps only one parent, images 1+ keep all.
    state.sample_mask[0][:] = False
    state.sample_mask[0][0, 0] = True
    state = subdivide_per_image_state(state)

    image_indices = np.arange(N_IMAGES, dtype=np.int32)
    s_loop = score_per_image_at_low_freq(
        ds, mean, nv, state, image_indices,
        L=4, disc_type="linear_interp", image_batch_size=N_IMAGES,
    )
    s_bucket = score_per_image_at_low_freq_bucketed(
        ds, mean, nv, state, image_indices,
        L=4, disc_type="linear_interp", image_batch_size=N_IMAGES,
        axis_quantum=8, shift_quantum=4,
    )
    _assert_per_image_scores_close(s_loop, s_bucket, atol=1e-4, rtol=5e-4)


def test_bucketed_handles_empty_image_state():
    """An image with no surviving candidates still gets a valid (zero-shape) score array."""
    rng = np.random.default_rng(2026)
    ds = _MockDataset(rng)
    mean = _hermitian_volume(seed=42)
    nv = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)

    state = initialize_per_image_state(
        n_images=N_IMAGES,
        initial_angular_spacing_deg=24.0,
        initial_shift_spacing_px=2.0,
        max_shift_px=2.0,
    )
    # Kill everything for image 0 by zeroing its mask + cells.
    state.axis_cells[0] = np.zeros((0, 3), dtype=np.float32)
    state.axis_rotations[0] = np.zeros((0, 3, 3), dtype=np.float32)
    state.shift_cells[0] = np.zeros((0, 2), dtype=np.float32)
    state.sample_mask[0] = np.zeros((0, 0), dtype=bool)

    image_indices = np.arange(N_IMAGES, dtype=np.int32)
    # Per-image-loop must skip image 0 internally; bucketed must too.
    s_loop = score_per_image_at_low_freq(
        ds, mean, nv, state, image_indices,
        L=4, disc_type="linear_interp", image_batch_size=N_IMAGES,
    )
    assert s_loop[0].shape == (0, 0)
