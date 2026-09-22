"""Recovered exact-local score-only memory bounds and physical-GPU fallback."""
import numpy as np
import jax.numpy as jnp
from recovar.em.local import local_batch_planning as planning
from recovar.em.sparse_pass2 import sparse_pass2_budget as budget
from recovar.em.local.local_layout import LocalHypothesisLayout, bucket_local_hypothesis_layout

def test_exact_local_score_only_preprocess_cap_covers_real_k4_box256_oom():
    """Job 13300875 requested 14.27 GiB before applying its current-size window."""

    image_shape = (256, 256)
    n_trans = 116
    requested_images = 500
    full_half_pixels = image_shape[0] * (image_shape[1] // 2 + 1)
    uncapped_bytes = requested_images * n_trans * full_half_pixels * np.dtype(np.complex64).itemsize
    assert uncapped_bytes == 15_323_136_000

    runtime_free_bytes = 14_778 * 1024**2
    capped_images = planning._exact_local_score_only_preprocess_image_batch_size(
        requested_images,
        image_shape=image_shape,
        n_trans=n_trans,
        score_complex_dtype=jnp.complex64,
        runtime_free_memory_bytes=runtime_free_bytes,
    )
    expected = int(
        runtime_free_bytes
        * planning.EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION
        // (
            n_trans
            * full_half_pixels
            * np.dtype(np.complex64).itemsize
            * planning.EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR
        )
    )

    assert capped_images == expected == 80
    capped_live_bytes = (
        capped_images
        * n_trans
        * full_half_pixels
        * np.dtype(np.complex64).itemsize
        * planning.EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR
    )
    assert capped_live_bytes <= runtime_free_bytes * planning.EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION

    n_images = requested_images
    rotation_counts = np.ones(n_images, dtype=np.int32)
    layout = LocalHypothesisLayout(
        n_global_rotations=1,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.arange(n_images + 1, dtype=np.int64),
        rotation_ids_flat=np.zeros(n_images, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (n_images, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(n_images, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((n_trans, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, n_trans), dtype=np.float32),
    )
    uncapped = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=requested_images,
        rotation_block_size=1,
        max_hypotheses_per_microbatch=10_000,
    )
    capped = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=capped_images,
        rotation_block_size=1,
        max_hypotheses_per_microbatch=10_000,
    )
    assert max(bucket.image_indices.size for bucket in capped) == capped_images
    np.testing.assert_array_equal(
        np.concatenate([bucket.image_indices for bucket in capped]),
        np.concatenate([bucket.image_indices for bucket in uncapped]),
    )

def test_exact_local_runtime_free_memory_uses_physical_h100_fallback(monkeypatch):

    physical_free_bytes = 14_778 * 1024**2

    class FakeDevice:
        @staticmethod
        def memory_stats():
            return {}

    monkeypatch.setattr(planning.jax, "local_devices", lambda: [FakeDevice()])
    monkeypatch.setattr(
        budget,
        "_device_free_memory_bytes",
        lambda: physical_free_bytes,
    )
    assert planning._exact_local_runtime_free_memory_bytes() == physical_free_bytes

    allocator_free_bytes = 13 * 1024**3
    monkeypatch.setattr(
        FakeDevice,
        "memory_stats",
        staticmethod(
            lambda: {
                "bytes_limit": 20 * 1024**3,
                "bytes_in_use": 7 * 1024**3,
            }
        ),
    )
    assert planning._exact_local_runtime_free_memory_bytes() == allocator_free_bytes
