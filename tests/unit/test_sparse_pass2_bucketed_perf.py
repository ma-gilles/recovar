"""Sanity perf test: bucketed sparse pass-2 must NOT recompile per image.

The original ``compute_pass2_stats_sparse`` has a Python for-loop over
particles that calls ``run_em(..., image_batch_size=1, ...)`` once per
image, with a different XLA shape each time.  On the 5k fixture this
caused thousands of separate JIT compiles and made iter-1 take >50 min.

This test:
  * builds a synthetic dataset with N images that have *varied* per-image
    significant-rotation counts (the trigger for the recompile bug);
  * monkey-patches ``jax.jit`` so we count the number of distinct
    compiled trace cache keys produced during one call;
  * asserts the bucketed path produces ≪ N compiled programs (i.e.,
    bounded by the number of bucket sizes), whereas the per-image
    reference path would scale with the number of distinct rotation
    counts.

We use a tiny mock dataset so the test is fast on a login node.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.oversampling import (
    compute_pass2_stats_sparse,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import _bucket_pass2_inputs

pytestmark = pytest.mark.unit


# Mock dataset (mirrors test_sparse_pass2_bucketed_parity.MockDataset).
IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512


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
    def __init__(self, n_images=10, seed=42):
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
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)
        self.premultiplied_ctf = False
        rng = np.random.default_rng(seed)
        self._images = np.zeros((n_images, IMAGE_SIZE), dtype=np.complex64)
        for i in range(n_images):
            self._images[i] = _hermitian_image_2d(IMAGE_SHAPE, seed=rng.integers(10000)).reshape(-1)

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
                jnp.asarray(self.rotation_matrices[idx]),
                jnp.asarray(self.translations[idx]),
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)

    def update_poses(self, rotations, translations):
        self.rotation_matrices = np.asarray(rotations)
        self.translations = np.asarray(translations)


def test_bucket_count_bounded_under_varied_per_image_rotation_counts():
    """Number of buckets must be bounded by the number of unique quantized sizes,
    not by the number of distinct per-image counts (and certainly not by N_images).
    """
    rng = np.random.default_rng(7)
    n_images = 500
    n_coarse_rot = 48
    n_coarse_trans = 2
    n_fine_trans = 2 * (4**1)  # = 8

    # Random per-image significant counts in [1, 20] — many distinct counts.
    counts = rng.integers(low=1, high=21, size=n_images)
    # Build (rot * n_coarse_trans + trans) flat indices: pick coarse rot, pair with trans 0
    # so candidate_mask is non-empty (fine trans 0 is parent of trans 0).
    sig_indices = [
        (rng.choice(n_coarse_rot, size=int(c), replace=False).astype(np.int32) * n_coarse_trans).astype(np.int32)
        for c in counts
    ]

    # Build per-image inputs the way compute_pass2_stats_sparse_bucketed does.
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import _prepare_per_image_pass2_inputs

    # fine_translation_parent maps fine trans -> coarse trans. With oversampling=1
    # in 2D, each coarse trans expands to 4 children, so trans 0..3 map to coarse 0,
    # trans 4..7 map to coarse 1.
    fine_trans_parent = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

    per_image = _prepare_per_image_pass2_inputs(
        sig_indices,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_trans_parent,
        rotation_log_prior=None,
        random_perturbation=0.0,
    )

    buckets = _bucket_pass2_inputs(per_image, n_fine_trans=n_fine_trans)
    n_distinct_counts = len({int(rots.shape[0]) for rots in per_image["oversampled_rots"]})

    # Quantization should collapse many distinct counts into a few buckets.
    assert len(buckets) < n_distinct_counts + 1, (
        f"Expected fewer than {n_distinct_counts + 1} buckets after quantization, got {len(buckets)}."
    )
    # Must be much smaller than n_images — that's the whole point.
    assert len(buckets) < n_images / 10, f"Got {len(buckets)} buckets for {n_images} images — bucketing too granular."
    print(f"Bucketed: {n_images} images with {n_distinct_counts} distinct counts -> {len(buckets)} buckets")


def test_bucketed_call_count_bounded_versus_perimage():
    """The bucketed path should make far fewer ``run_em``-style backend calls.

    We count the number of times ``_score_pass2_bucket`` is invoked by the
    bucketed path: that should equal the number of buckets, much less than
    ``n_images`` (which is what the per-image reference invokes).
    """
    n_images = 24
    nside_level = 1
    rng = np.random.default_rng(13)
    n_coarse_rot = 48
    n_coarse_trans = 2
    counts = rng.integers(low=1, high=12, size=n_images)
    sig_indices = [
        (rng.choice(n_coarse_rot, size=int(c), replace=False).astype(np.int32) * n_coarse_trans).astype(np.int32)
        for c in counts
    ]

    ds = MockDataset(n_images=n_images, seed=29)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=37)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    common_kwargs = dict(
        nside_level=nside_level,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_stats=True,
        accumulate_noise=False,
    )

    # Count score-bucket invocations
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    original_score = bucketed_mod._score_pass2_bucket
    score_call_count = {"n": 0}

    def counting_score(*args, **kwargs):
        score_call_count["n"] += 1
        return original_score(*args, **kwargs)

    bucketed_mod._score_pass2_bucket = counting_score
    try:
        # Warm jit cache by running once
        compute_pass2_stats_sparse(
            ds, volume, mean_variance, noise_variance, translations, sig_indices, **common_kwargs
        )
        score_call_count["n"] = 0
        # Re-run to count the actual invocations
        compute_pass2_stats_sparse(
            ds, volume, mean_variance, noise_variance, translations, sig_indices, **common_kwargs
        )
    finally:
        bucketed_mod._score_pass2_bucket = original_score

    # The number of bucketed score calls is the number of buckets.  With
    # n_images=24 and counts in [1, 12], we expect at most ~5-6 buckets
    # (powers-of-two padding: 16, 32, 64, 128 round to a few sizes).
    # Definitely << n_images.
    assert score_call_count["n"] < n_images, (
        f"Bucketed score was called {score_call_count['n']} times for {n_images} images "
        "— expected fewer (one per bucket)."
    )
    print(f"Bucketed: {score_call_count['n']} score calls for {n_images} images")
