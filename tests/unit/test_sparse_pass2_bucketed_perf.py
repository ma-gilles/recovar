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

import recovar.core as core
import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.helpers.preprocessing import (
    apply_half_translation_phases,
    half_translation_phase_table,
)
from recovar.em.dense_single_volume.helpers.oversampling import (
    compute_pass2_stats_sparse,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _bucket_sparse_k_class_pass2_inputs,
    _bucket_pass2_inputs,
    _logsumexp_pass2_bucket_score_only,
    _max_hypotheses_per_microbatch_for_pass,
    _max_images_for_translation_tile,
    _max_translation_tile_bytes_for_pass,
    _normalize_pass2_bucket,
    _normalize_pass2_bucket_score_only,
    _nvidia_smi_visible_device_memory_bytes,
    _prepare_per_image_pass2_inputs,
    _projection_cache_max_bytes_for_pass,
)
from recovar.em.dense_single_volume.k_class import _run_sparse_k_class_adaptive_pass2

pytestmark = pytest.mark.unit


# Mock dataset (mirrors test_sparse_pass2_bucketed_parity.MockDataset).
IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512


def _raw_real_image_2d(image_shape, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(image_shape).astype(np.float32)


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


def _raw_real_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


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
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)
        self.premultiplied_ctf = False
        rng = np.random.default_rng(seed)
        self._images = np.zeros((n_images, *IMAGE_SHAPE), dtype=np.float32)
        for i in range(n_images):
            self._images[i] = _raw_real_image_2d(IMAGE_SHAPE, seed=rng.integers(10000))

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)

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


def test_default_sparse_pass2_budget_keeps_broad_support_batched():
    """Broad soft K-class supports must not fall back to one image per launch."""

    n_images = 26
    n_fine_trans = 116
    n_rot = 1024
    per_image = {
        "oversampled_rots": [np.zeros((n_rot, 3, 3), dtype=np.float32) for _ in range(n_images)],
    }

    default_buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_images_per_microbatch=13,
    )
    old_cap_buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=100_000,
        max_images_per_microbatch=13,
    )

    assert [len(bucket["image_indices"]) for bucket in default_buckets] == [8, 8, 8, 2]
    assert len(old_cap_buckets) == n_images


def test_score_only_sparse_pass_uses_larger_default_bucket_budget(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES", raising=False)

    device_memory = 80 * 1024**3
    n_score_pixels = 652
    assert (
        _max_hypotheses_per_microbatch_for_pass(
            score_only=True,
            use_window=True,
            has_external_normalization=False,
            dump_pass2_operands=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
        > _max_hypotheses_per_microbatch_for_pass(
            score_only=False,
            use_window=True,
            has_external_normalization=False,
            dump_pass2_operands=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
    )
    assert (
        _max_hypotheses_per_microbatch_for_pass(
            score_only=True,
            use_window=True,
            has_external_normalization=False,
            dump_pass2_operands=False,
            n_score_pixels=n_score_pixels * 2,
            device_memory_bytes=device_memory,
        )
        < _max_hypotheses_per_microbatch_for_pass(
            score_only=True,
            use_window=True,
            has_external_normalization=False,
            dump_pass2_operands=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES", "12345")
    assert (
        _max_hypotheses_per_microbatch_for_pass(
            score_only=True,
            use_window=True,
            has_external_normalization=False,
            dump_pass2_operands=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
        == 12345
    )


def test_sparse_pass2_auto_hypothesis_cap_matches_80gb_probe_scale(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES", raising=False)

    device_memory = 80 * 1024**3
    n_score_pixels = 652
    cap = _max_hypotheses_per_microbatch_for_pass(
        score_only=True,
        use_window=True,
        has_external_normalization=False,
        dump_pass2_operands=False,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory,
    )

    assert 9_500_000 <= cap <= 10_900_000


def test_fused_k_class_sparse_pass2_uses_larger_joint_hypothesis_cap(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", raising=False)

    device_memory = 80 * 1024**3
    n_score_pixels = 1103
    single_class_cap = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=True,
        has_external_normalization=False,
        dump_pass2_operands=False,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory,
    )
    fused_cap = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=True,
        has_external_normalization=False,
        dump_pass2_operands=False,
        fused_k_class=True,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory,
    )

    assert fused_cap > single_class_cap
    assert 5_500_000 <= fused_cap <= 6_500_000

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", "12345")
    assert (
        _max_hypotheses_per_microbatch_for_pass(
            score_only=False,
            use_window=True,
            has_external_normalization=False,
            dump_pass2_operands=False,
            fused_k_class=True,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
        == 12345
    )


def test_sparse_pass2_memory_budgets_auto_scale_with_device_memory(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", raising=False)

    small_gpu = 20 * 1024**3
    large_gpu = 80 * 1024**3

    assert _max_translation_tile_bytes_for_pass(large_gpu) == pytest.approx(
        4 * _max_translation_tile_bytes_for_pass(small_gpu),
        rel=1e-6,
    )
    assert _max_translation_tile_bytes_for_pass(
        large_gpu,
        has_external_normalization=True,
    ) < _max_translation_tile_bytes_for_pass(large_gpu)
    assert _max_translation_tile_bytes_for_pass(
        large_gpu,
        fused_k_class=True,
    ) < _max_translation_tile_bytes_for_pass(
        large_gpu,
        has_external_normalization=True,
    )
    assert _projection_cache_max_bytes_for_pass(large_gpu) == pytest.approx(
        4 * _projection_cache_max_bytes_for_pass(small_gpu),
        rel=1e-6,
    )
    assert (
        _max_images_for_translation_tile(
            (256, 256),
            116,
            max_tile_bytes=_max_translation_tile_bytes_for_pass(large_gpu),
        )
        >= 50
    )
    assert (
        _max_images_for_translation_tile(
            (256, 256),
            116,
            max_tile_bytes=_max_translation_tile_bytes_for_pass(
                large_gpu,
                has_external_normalization=True,
            ),
        )
        >= 35
    )
    assert (
        _max_images_for_translation_tile(
            (256, 256),
            116,
            max_tile_bytes=_max_translation_tile_bytes_for_pass(small_gpu),
        )
        >= 13
    )
    assert 17 <= _max_images_for_translation_tile(
        (256, 256),
        116,
        max_tile_bytes=_max_translation_tile_bytes_for_pass(large_gpu, fused_k_class=True),
    ) <= 22

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES", "123456")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", "654321")
    assert _max_translation_tile_bytes_for_pass(large_gpu) == 123456
    assert _projection_cache_max_bytes_for_pass(large_gpu) == 654321


def test_fused_k_class_sparse_pass2_uses_coarse_tail_bucket_quantum(monkeypatch):
    """Pin the automatic q512-style tail bucketing used by 100k/256 K=4."""

    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    counts = [1025, 1152, 1537, 2049]

    def per_class_inputs():
        return {
            "oversampled_rots": [
                np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
                for count in counts
            ],
        }

    buckets = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(4)],
        n_fine_trans=116,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    default_sizes = sorted({int(bucket["bucket_size"]) for bucket in buckets})
    assert default_sizes == [1536, 2048, 2560]

    monkeypatch.setenv("RECOVAR_LOCAL_BUCKET_QUANTUM", "128")
    finer_buckets = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(4)],
        n_fine_trans=116,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    finer_sizes = sorted({int(bucket["bucket_size"]) for bucket in finer_buckets})
    assert finer_sizes == [1152, 1664, 2176]
    assert default_sizes[0] > finer_sizes[0]


def test_sparse_pass2_device_memory_probe_honors_visible_device():
    smi_output = "\n".join(
        [
            "0, GPU-a100, 40960",
            "1, GPU-h100, 81559",
        ],
    )

    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "1") == 81559 * 1024**2
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "GPU-h100") == 81559 * 1024**2
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "h100") == 81559 * 1024**2
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "-1") is None
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, None) == 40960 * 1024**2


def test_half_translation_phase_table_matches_generic_translate_images():
    rng = np.random.default_rng(13)
    image_shape = (16, 16)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    weighted_half = jnp.asarray(
        rng.normal(size=(3, n_half)).astype(np.float32)
        + 1j * rng.normal(size=(3, n_half)).astype(np.float32),
        dtype=jnp.complex64,
    )
    translations = jnp.asarray(rng.normal(size=(5, 2)).astype(np.float32))

    tiled_images = jnp.repeat(weighted_half[:, None, :], translations.shape[0], axis=1).reshape(
        weighted_half.shape[0] * translations.shape[0],
        -1,
    )
    tiled_translations = jnp.repeat(translations[None], weighted_half.shape[0], axis=0).reshape(
        weighted_half.shape[0] * translations.shape[0],
        -1,
    )
    generic = core.translate_images(tiled_images, tiled_translations, image_shape, half_image=True)
    phase_table = apply_half_translation_phases(
        weighted_half,
        half_translation_phase_table(translations, image_shape),
    )

    np.testing.assert_array_equal(np.asarray(phase_table), np.asarray(generic))


def test_score_only_sparse_normalizer_matches_full_normalizer_stats():
    scores = jnp.asarray(
        [
            [[1.0, -2.0, -jnp.inf], [0.25, -0.5, -3.0]],
            [[-jnp.inf, -jnp.inf, -jnp.inf], [-jnp.inf, -jnp.inf, -jnp.inf]],
        ],
        dtype=jnp.float32,
    )

    full_log_z, _probs, full_best, full_argmax, full_pmax = _normalize_pass2_bucket(scores)
    score_log_z, score_best, score_argmax, score_pmax = _normalize_pass2_bucket_score_only(scores)

    np.testing.assert_allclose(np.asarray(score_log_z), np.asarray(full_log_z), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(score_best), np.asarray(full_best), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(score_argmax), np.asarray(full_argmax))
    logz_only = np.asarray(_logsumexp_pass2_bucket_score_only(scores))
    np.testing.assert_allclose(logz_only[0], np.asarray(full_log_z)[0], rtol=0, atol=0)
    assert np.isneginf(logz_only[1])
    np.testing.assert_allclose(np.asarray(score_pmax), np.asarray(full_pmax), rtol=1e-7, atol=1e-7)


def test_fine_rotation_override_preserves_fine_grid_order_and_parent_map():
    fine_rotations = np.arange(6 * 9, dtype=np.float32).reshape(6, 3, 3)
    fine_parent = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)

    per_image = _prepare_per_image_pass2_inputs(
        [np.array([0, 2], dtype=np.int32)],
        n_coarse_rot=3,
        n_coarse_trans=1,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=1,
        fine_translation_parent=np.array([0], dtype=np.int32),
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
    )

    np.testing.assert_array_equal(per_image["oversampled_rot_indices"][0], np.array([0, 2, 3, 5]))
    np.testing.assert_array_equal(per_image["parent_map"][0], np.array([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(per_image["oversampled_rots"][0], fine_rotations[[0, 2, 3, 5]])


def test_fine_grid_candidate_mask_uses_parented_translation_support():
    fine_rotations = np.arange(5 * 9, dtype=np.float32).reshape(5, 3, 3)
    fine_parent = np.array([0, 0, 1, 3, 3], dtype=np.int64)
    fine_translation_parent = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

    per_image = _prepare_per_image_pass2_inputs(
        [np.array([0, 2, 10], dtype=np.int32)],
        n_coarse_rot=4,
        n_coarse_trans=3,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=fine_translation_parent.size,
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
    )

    expected = np.array(
        [
            [True, False, True, True, False, True],
            [True, False, True, True, False, True],
            [False, True, False, False, True, False],
            [False, True, False, False, True, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(per_image["oversampled_rot_indices"][0], np.array([0, 1, 3, 4]))
    np.testing.assert_array_equal(per_image["candidate_mask"][0], expected)


def test_sparse_pass2_projection_cache_projects_fine_grid_once(monkeypatch):
    """Fine-grid override projections are shared across sparse buckets."""

    n_images = 8
    n_coarse_rot = 48
    n_coarse_trans = 2
    fine_rotations = np.tile(np.eye(3, dtype=np.float32), (10, 1, 1))
    fine_parent = np.array([0, 0, 1, 1, 2, 3, 3, 4, 5, 7], dtype=np.int64)
    fine_translations = np.array(
        [[0.0, 0.0], [0.25, 0.0], [1.0, 0.0], [1.25, 0.0]],
        dtype=np.float32,
    )
    fine_translation_parent = np.array([0, 0, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([parent * n_coarse_trans for parent in parents], dtype=np.int32)
        for parents in ([0], [0, 1], [0, 1, 2], [3], [3, 4], [5], [7], [0, 3, 5])
    ]

    ds = MockDataset(n_images=n_images, seed=41)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=43)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", "16")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", str(1024**3))

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    projection_call_sizes = []

    def fake_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type
        projection_call_sizes.append(int(rotations_block.shape[0]))
        n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
        proj = jnp.zeros((rotations_block.shape[0], n_half), dtype=jnp.complex64)
        return_abs2 = kwargs.get("return_abs2", None)
        return proj, None if return_abs2 is False else jnp.zeros((rotations_block.shape[0], n_half), dtype=jnp.float32)

    monkeypatch.setattr(bucketed_mod, "_compute_projections_block", fake_project)

    compute_pass2_stats_sparse(
        ds,
        volume,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=False,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )

    assert projection_call_sizes == [fine_rotations.shape[0]]


def test_score_log_z_only_matches_full_score_probe(monkeypatch):
    n_images = 6
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]], dtype=np.float32)
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([0, 1, 4, 7], dtype=np.int32),
        np.asarray([2, 3, 6], dtype=np.int32),
        np.asarray([8, 9], dtype=np.int32),
        np.asarray([0, 5, 10], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([2, 11], dtype=np.int32),
    ]

    ds = MockDataset(n_images=n_images, seed=51)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=53)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    full = compute_pass2_stats_sparse(
        ds,
        volume,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=False,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )
    log_evidence, score_log_z = compute_pass2_stats_sparse(
        ds,
        volume,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_score_log_z_only=True,
        accumulate_noise=False,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )

    np.testing.assert_allclose(np.asarray(log_evidence), np.asarray(full[6].log_evidence_per_image), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(score_log_z), np.asarray(full[7]), rtol=0, atol=0)


def test_fused_other_class_log_z_matches_two_pass_normalization(monkeypatch):
    n_images = 5
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]], dtype=np.float32)
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([0, 1, 4, 7], dtype=np.int32),
        np.asarray([2, 3, 6], dtype=np.int32),
        np.asarray([8, 9], dtype=np.int32),
        np.asarray([0, 5, 10], dtype=np.int32),
        np.asarray([1, 2, 11], dtype=np.int32),
    ]

    ds = MockDataset(n_images=n_images, seed=71)
    volume_a = _hermitian_volume(VOLUME_SHAPE, seed=73)
    volume_b = _hermitian_volume(VOLUME_SHAPE, seed=79)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    common = dict(
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_stats=True,
        accumulate_noise=False,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )

    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    _, score_a = compute_pass2_stats_sparse(
        ds,
        volume_a,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        return_score_log_z_only=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        **common,
    )
    log_evidence_b, score_b = compute_pass2_stats_sparse(
        ds,
        volume_b,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        return_score_log_z_only=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        **common,
    )
    global_score_log_z = np.logaddexp(np.asarray(score_a, dtype=np.float64), np.asarray(score_b, dtype=np.float64))

    two_pass = compute_pass2_stats_sparse(
        ds,
        volume_b,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        normalization_log_z=global_score_log_z,
        **common,
    )
    fused = compute_pass2_stats_sparse(
        ds,
        volume_b,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        normalization_other_score_log_z=score_a,
        return_score_log_z=True,
        **common,
    )

    np.testing.assert_allclose(np.asarray(fused[0]), np.asarray(two_pass[0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fused[1]), np.asarray(two_pass[1]), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(fused[2]), np.asarray(two_pass[2]))
    np.testing.assert_array_equal(np.asarray(fused[5]), np.asarray(two_pass[5]))
    np.testing.assert_allclose(
        np.asarray(fused[6].best_log_score_per_image),
        np.asarray(two_pass[6].best_log_score_per_image),
    )
    np.testing.assert_allclose(
        np.asarray(fused[6].max_posterior_per_image),
        np.asarray(two_pass[6].max_posterior_per_image),
    )
    np.testing.assert_allclose(
        np.asarray(fused[6].rotation_posterior_sums),
        np.asarray(two_pass[6].rotation_posterior_sums),
    )
    np.testing.assert_allclose(np.asarray(fused[6].log_evidence_per_image), np.asarray(log_evidence_b), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(fused[7]), np.asarray(score_b), rtol=0, atol=0)


def test_fused_sparse_k_class_pass2_matches_existing_two_pass_path(monkeypatch):
    """Opt-in fused Class3D pass-2 must preserve the existing sparse semantics."""

    from recovar.em.sampling import rotation_grid_size

    n_images = 5
    n_classes = 2
    n_coarse_rot = rotation_grid_size(1)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    coarse_translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    significant_by_class = [
        [
            np.asarray([0, 1, 4, 7], dtype=np.int32),
            np.asarray([2, 3, 6], dtype=np.int32),
            np.asarray([8, 9], dtype=np.int32),
            np.asarray([0, 5, 10], dtype=np.int32),
            np.asarray([1, 2, 11], dtype=np.int32),
        ],
        [
            np.asarray([0, 2, 5], dtype=np.int32),
            np.asarray([1, 3, 7], dtype=np.int32),
            np.asarray([4, 8], dtype=np.int32),
            np.asarray([3, 9, 11], dtype=np.int32),
            np.asarray([0, 6], dtype=np.int32),
        ],
    ]
    ds = MockDataset(n_images=n_images, seed=91)
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=101),
            _hermitian_volume(VOLUME_SHAPE, seed=103),
        ],
    )
    kwargs = dict(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.asarray([0.45, 0.55], dtype=np.float64)),
        accumulate_noise=True,
        return_best_pose_details=True,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={"current_size": None, "relion_half_volume_mstep": False},
    )

    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_FUSED", raising=False)
    existing = _run_sparse_k_class_adaptive_pass2(**kwargs)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    fused = _run_sparse_k_class_adaptive_pass2(**kwargs)

    np.testing.assert_allclose(np.asarray(fused.Ft_y), np.asarray(existing.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(fused.Ft_ctf), np.asarray(existing.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(fused.per_class_hard_assignments),
        np.asarray(existing.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(fused.class_assignments), np.asarray(existing.class_assignments))
    np.testing.assert_array_equal(np.asarray(fused.pose_assignments), np.asarray(existing.pose_assignments))
    np.testing.assert_allclose(
        np.asarray(fused.class_responsibilities),
        np.asarray(existing.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused.class_posterior_sums),
        np.asarray(existing.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    for fused_noise, existing_noise in zip(fused.noise_stats, existing.noise_stats, strict=True):
        np.testing.assert_allclose(
            np.asarray(fused_noise.wsum_sigma2_noise),
            np.asarray(existing_noise.wsum_sigma2_noise),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(fused_noise.wsum_img_power),
            np.asarray(existing_noise.wsum_img_power),
            rtol=1e-5,
            atol=1e-5,
        )
        assert fused_noise.wsum_sigma2_offset == pytest.approx(existing_noise.wsum_sigma2_offset, abs=1e-6)
        assert fused_noise.sumw == pytest.approx(existing_noise.sumw, abs=1e-6)


def test_bucketed_call_count_bounded_versus_perimage():
    """The bucketed path should make far fewer ``run_em``-style backend calls.

    We count the number of times ``_score_pass2_bucket_relion_gpu_diff2`` is invoked by the
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

    original_score = bucketed_mod._score_pass2_bucket_relion_gpu_diff2
    score_call_count = {"n": 0}

    def counting_score(*args, **kwargs):
        score_call_count["n"] += 1
        return original_score(*args, **kwargs)

    bucketed_mod._score_pass2_bucket_relion_gpu_diff2 = counting_score
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
        bucketed_mod._score_pass2_bucket_relion_gpu_diff2 = original_score

    # The number of bucketed score calls is the number of buckets.  With
    # n_images=24 and counts in [1, 12], we expect at most ~5-6 buckets
    # (powers-of-two padding: 16, 32, 64, 128 round to a few sizes).
    # Definitely << n_images.
    assert score_call_count["n"] < n_images, (
        f"Bucketed score was called {score_call_count['n']} times for {n_images} images "
        "— expected fewer (one per bucket)."
    )
    print(f"Bucketed: {score_call_count['n']} score calls for {n_images} images")
