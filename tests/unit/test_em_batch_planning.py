"""EM batch sizing and microbatch memory-policy checks."""

import inspect
import numpy as np
import pytest
import recovar.em.refinement.iteration_loop as iteration_loop_module
from recovar.em.dense import half_scoring
from recovar.em.local.local_batch_planning import (
    EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV,
    EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV,
    EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION,
    EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR,
    EXACT_LOCAL_TARGET_ROW_PIXELS_ENV,
    EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV,
    _exact_local_effective_max_hypotheses_per_microbatch,
    _exact_local_max_hypotheses_per_microbatch,
    _exact_local_xhalf_projection_microbatch_cap,
    _exact_local_xhalf_tail_microbatch_cap,
)
from recovar.em.local.local_em_engine import run_local_em_exact
from recovar.em.local.local_layout import (
    LocalHypothesisLayout,
    _local_search_engine_rotation_block_size,
    bucket_local_hypothesis_layout,
)
from recovar.em.refinement.iteration_loop import _estimate_relion_em_batch_sizes

IMAGE_SHAPE = (8, 8)
VOLUME_SHAPE = (8, 8, 8)


def _force_exact_local_standard_gpu_default(monkeypatch):
    from recovar.em.local import local_batch_planning

    monkeypatch.setattr(local_batch_planning, "_visible_gpu_memory_bytes", lambda: None)


def _identity_layout(rotation_counts, *, n_trans):
    """Build identity poses for the requested per-image neighborhood sizes."""
    rotation_counts = np.asarray(rotation_counts, dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    total = int(offsets[-1])
    return LocalHypothesisLayout(
        n_global_rotations=total,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=offsets,
        rotation_ids_flat=np.arange(total, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (total, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(total, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((n_trans, 2), dtype=np.float32),
        translation_log_priors=np.zeros((len(rotation_counts), n_trans), dtype=np.float32),
    )


def test_exact_local_microbatch_default_matches_profiled_256_window(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)

    assert _exact_local_max_hypotheses_per_microbatch(None, 8018) == 23696


def test_exact_local_microbatch_high_memory_gpu_default(monkeypatch):
    from recovar.em.local import local_batch_planning

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.setattr(local_batch_planning, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    assert (
        _exact_local_max_hypotheses_per_microbatch(
            None,
            12861,
            n_trans=36,
            n_recon_windowed=12723,
        )
        == 19905
    )


def test_exact_local_score_only_cap_covers_100k_parent_tile(monkeypatch):
    from recovar.em.local import local_batch_planning

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)
    monkeypatch.setattr(local_batch_planning, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    layout = _identity_layout(np.asarray([198], dtype=np.int32), n_trans=9)
    runtime_free_bytes = int(46.4 * 1024**3)
    cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        12861,
        n_trans=9,
        n_recon_windowed=12723,
        local_layout=layout,
        image_batch_size=168,
        rotation_block_size=198,
        score_only=True,
        runtime_free_memory_bytes=runtime_free_bytes,
    )
    expected_tile_cap = int(
        runtime_free_bytes
        * EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION
        // (9 * 12861 * np.dtype(np.float32).itemsize * EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR)
    )

    assert cap == expected_tile_cap
    assert cap < 168 * 198
    assert cap // 198 == 86


def test_exact_local_score_only_cap_preserves_smaller_bucket_shape(monkeypatch):
    from recovar.em.local import local_batch_planning

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)
    monkeypatch.setattr(local_batch_planning, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    layout = _identity_layout(np.asarray([198], dtype=np.int32), n_trans=9)
    cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        4003,
        n_trans=9,
        n_recon_windowed=3923,
        local_layout=layout,
        image_batch_size=168,
        rotation_block_size=198,
        score_only=True,
        runtime_free_memory_bytes=int(46.4 * 1024**3),
    )
    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=168,
        rotation_block_size=198,
        max_hypotheses_per_microbatch=cap,
    )

    assert cap >= 168 * 198
    assert len(buckets) == 1


def test_exact_local_xhalf_full_bpref_uses_conservative_high_memory_cap(monkeypatch):
    from recovar.em.local import local_batch_planning

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.setattr(local_batch_planning, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    high_memory_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        8320,
        n_trans=36,
        n_recon_windowed=8320,
    )
    full_bpref_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        8320,
        n_trans=36,
        n_recon_windowed=8320,
        allow_high_memory_default=False,
    )

    assert high_memory_cap == 30769
    assert full_bpref_cap == 17127
    assert full_bpref_cap < high_memory_cap


def test_exact_local_xhalf_current_bpref_uses_conservative_high_memory_cap(monkeypatch):
    from recovar.em.local import local_batch_planning

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.setattr(local_batch_planning, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    high_memory_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        4003,
        n_trans=52,
        n_recon_windowed=3923,
    )
    current_bpref_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        4003,
        n_trans=52,
        n_recon_windowed=3923,
        allow_high_memory_default=False,
    )

    assert high_memory_cap == 63952
    assert current_bpref_cap == 35933
    assert current_bpref_cap < high_memory_cap


def test_exact_local_microbatch_target_row_pixels_override(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.setenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "380000000")

    assert _exact_local_max_hypotheses_per_microbatch(None, 8018) == 47393


def test_exact_local_microbatch_caps_fused_mstep_matmul(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)

    assert (
        _exact_local_max_hypotheses_per_microbatch(
            None,
            1091,
            n_trans=21,
            n_recon_windowed=1134,
        )
        == 65536
    )


def test_exact_local_microbatch_batches_full_parent_256_pass2(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)

    cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=84,
        n_recon_windowed=33024,
    )
    assert cap >= 3072

    n_images = 12
    local_rotations = 1416
    layout = _identity_layout(np.full(n_images, local_rotations, dtype=np.int32), n_trans=84)

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=17,
        rotation_block_size=26,
        max_hypotheses_per_microbatch=cap,
    )

    assert max(int(bucket.image_indices.shape[0]) for bucket in buckets) >= 2
    assert len(buckets) <= (n_images + 1) // 2


def test_exact_local_microbatch_boosts_high_res_local_batch_without_full_floor(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_BUCKET_QUANTUM", raising=False)

    n_images = 13
    local_rotations = 1536
    layout = _identity_layout(np.full(n_images, local_rotations, dtype=np.int32), n_trans=116)

    base_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
    )
    assert base_cap < n_images * local_rotations

    cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
    )

    assert cap == base_cap * 2
    assert cap < n_images * local_rotations
    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=n_images,
        rotation_block_size=19,
        max_hypotheses_per_microbatch=cap,
    )
    assert {int(bucket.image_indices.shape[0]) for bucket in buckets[:-1]} == {5}
    assert len(buckets) == 3
    assert max(int(bucket.image_indices.shape[0]) for bucket in buckets) == 5


def test_exact_local_microbatch_boost_can_be_disabled_for_mstep_pass2(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_BUCKET_QUANTUM", raising=False)

    n_images = 13
    local_rotations = 1536
    layout = _identity_layout(np.full(n_images, local_rotations, dtype=np.int32), n_trans=116)

    base_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
    )
    boosted_cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
    )
    capped_pass2_cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
        allow_auto_boost=False,
    )
    bounded_xhalf_cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
        auto_boost_factor=1.25,
    )

    assert boosted_cap > base_cap
    assert capped_pass2_cap == base_cap
    assert bounded_xhalf_cap == int(np.floor(base_cap * 1.25))


def test_exact_local_xhalf_mstep_uses_explicit_microbatch_boost_hook():
    source = inspect.getsource(run_local_em_exact)

    assert "allow_microbatch_auto_boost = True" in source
    assert "auto_boost_factor=xhalf_auto_microbatch_boost" in source
    assert "allow_high_memory_default=not xhalf_bpref_mstep" in source


def test_exact_local_microbatch_env_override_keeps_lower_cap(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.setenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "190000000")
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)

    n_images = 13
    local_rotations = 1536
    layout = _identity_layout(np.full(n_images, local_rotations, dtype=np.int32), n_trans=116)

    cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
    )

    assert cap == _exact_local_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
    )
    assert cap < n_images * local_rotations


def test_exact_local_xhalf_tail_microbatch_respects_outer_planner_tile():
    layout = _identity_layout(np.asarray([128, 766], dtype=np.int32), n_trans=116)

    cap = _exact_local_xhalf_tail_microbatch_cap(
        15_500,
        layout,
        image_batch_size=37,
        rotation_block_size=136,
    )

    assert cap == 37 * 136


def test_exact_local_xhalf_tail_microbatch_keeps_cap_for_planned_neighborhoods():
    layout = _identity_layout(np.asarray([64, 136], dtype=np.int32), n_trans=116)

    assert (
        _exact_local_xhalf_tail_microbatch_cap(
            15_500,
            layout,
            image_batch_size=37,
            rotation_block_size=136,
        )
        == 15_500
    )


def test_exact_local_xhalf_projection_cap_matches_case10_proven_bucket_boundary(monkeypatch):
    monkeypatch.delenv(EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV, raising=False)
    layout = _identity_layout(np.asarray([128, 256, 520], dtype=np.int32), n_trans=116)

    cap = _exact_local_xhalf_projection_microbatch_cap(
        14_171,
        layout,
        n_projection_pixels=8_258,
        rotation_block_size=383,
    )
    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=37,
        rotation_block_size=383,
        max_hypotheses_per_microbatch=cap,
    )

    assert cap == 40_000_000 // 8_258
    assert (
        max(int(bucket.bucket_image_count) * int(bucket.bucket_rotation_count) * 8_258 for bucket in buckets)
        <= 40_000_000
    )
    assert {int(bucket.bucket_rotation_count) for bucket in buckets} == {128, 256, 766}


def test_exact_local_xhalf_projection_cap_preserves_one_exact_neighborhood(monkeypatch):
    monkeypatch.setenv(EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV, "1000")
    layout = _identity_layout(np.asarray([520], dtype=np.int32), n_trans=116)

    assert (
        _exact_local_xhalf_projection_microbatch_cap(
            14_171,
            layout,
            n_projection_pixels=8_258,
            rotation_block_size=383,
        )
        == 766
    )


def test_local_search_outer_batch_sizing_uses_current_size_window():
    source = inspect.getsource(half_scoring._score_half_local)

    assert "current_size_for_batch=cs_for_engine" in source


def test_exact_local_microbatch_matmul_cap_can_be_disabled(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.setenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, "0")

    assert (
        _exact_local_max_hypotheses_per_microbatch(
            None,
            1091,
            n_trans=21,
            n_recon_windowed=1134,
        )
        == 65536
    )


def test_exact_local_microbatch_target_row_pixels_rejects_invalid(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.setenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "0")

    with pytest.raises(ValueError, match=EXACT_LOCAL_TARGET_ROW_PIXELS_ENV):
        _exact_local_max_hypotheses_per_microbatch(None, 8018)


# ---------------------------------------------------------------------------
# Helpers (same as test_fsc_resolution_loop.py)
# ---------------------------------------------------------------------------


def test_local_search_engine_rotation_block_size_caps_dense_tiles():
    assert _local_search_engine_rotation_block_size(64) == 64
    assert _local_search_engine_rotation_block_size(1024) == 1024
    assert _local_search_engine_rotation_block_size(5000) == 1024


def test_relion_em_batch_sizing_preserves_small_safe_requests():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=4,
        requested_rotation_block_size=8,
        n_rot=8,
        n_trans=5,
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 4
    assert plan.rotation_block_size == 8


def test_relion_em_batch_sizing_clamps_highres_projection_tiles():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=294912,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 4
    assert plan.rotation_block_size == 13
    assert plan.projection_block_gb <= plan.projection_budget_gb
    assert plan.pose_pixel_tile_gb < 3.0
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb
    assert plan.translation_tile_gb <= plan.translation_tile_budget_gb


def test_relion_em_batch_sizing_does_not_pad_beyond_actual_rotation_grid():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=4608,
        n_trans=29,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 46
    assert plan.rotation_block_size == 65
    assert plan.projection_block_gb <= plan.projection_budget_gb
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb


def test_relion_em_batch_sizing_uses_runtime_gpu_occupancy(monkeypatch):
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 60.0)

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=4608,
        n_trans=29,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )

    assert plan.gpu_used_estimate_gb == pytest.approx(60.0)
    assert plan.rotation_block_size < 4608
    assert plan.projection_block_gb <= plan.projection_budget_gb


def test_relion_em_batch_sizing_caps_runtime_highres_local_translation_tile(monkeypatch):
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 24.0)

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=4608,
        n_rot=4608,
        n_trans=116,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )

    assert plan.image_batch_size <= 56
    assert plan.translation_tile_gb <= plan.translation_tile_budget_gb
    assert plan.gpu_used_estimate_gb == pytest.approx(24.0)


def test_relion_em_batch_sizing_caps_dense_big_jit_score_workspace(monkeypatch):
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 41.0)

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )

    assert plan.image_batch_size == 50
    assert plan.rotation_block_size == 146
    assert plan.projection_block_gb <= plan.projection_budget_gb
    assert plan.pose_pixel_tile_gb < 3.0
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb
    assert plan.gpu_used_estimate_gb == pytest.approx(41.0)


def test_relion_em_batch_sizing_uses_active_window_for_dense_score_workspace(monkeypatch):
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 0.0)

    common = dict(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )
    low_res = _estimate_relion_em_batch_sizes(**common, current_size=56)
    high_res = _estimate_relion_em_batch_sizes(**common, current_size=248)

    assert low_res.rotation_block_size == 8192
    assert high_res.rotation_block_size < 8192
    assert low_res.score_pixel_count < high_res.score_pixel_count


def test_relion_em_batch_sizing_allows_larger_adaptive_pass1_blocks():
    common = dict(
        requested_image_batch_size=500,
        requested_rotation_block_size=5000,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    pass1_coarse = _estimate_relion_em_batch_sizes(**common, current_size=100)
    pass2_fine = _estimate_relion_em_batch_sizes(**common, current_size=154)

    assert pass1_coarse.score_pixel_count < pass2_fine.score_pixel_count
    assert pass1_coarse.rotation_block_size > pass2_fine.rotation_block_size
    assert pass1_coarse.pose_pixel_tile_gb <= pass1_coarse.projection_budget_gb
    assert pass2_fine.pose_pixel_tile_gb <= pass2_fine.projection_budget_gb


def test_relion_em_batch_sizing_projection_budget_override_expands_pass1_blocks(monkeypatch):
    common = dict(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
        current_size=100,
    )

    monkeypatch.delenv("RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION", raising=False)
    default = _estimate_relion_em_batch_sizes(**common)

    monkeypatch.setenv("RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION", "0.40")
    expanded = _estimate_relion_em_batch_sizes(**common)

    assert expanded.rotation_block_size > default.rotation_block_size
    assert expanded.rotation_block_size <= common["requested_rotation_block_size"]
    assert expanded.pose_pixel_tile_gb <= expanded.projection_budget_gb


def test_relion_em_batch_sizing_projection_budget_override_rejects_invalid(monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION", "0")

    with pytest.raises(ValueError, match="RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION"):
        _estimate_relion_em_batch_sizes(
            requested_image_batch_size=64,
            requested_rotation_block_size=8192,
            n_rot=36864,
            n_trans=29,
            image_shape=(256, 256),
            volume_shape=(256, 256, 256),
            padding_factor=2,
            n_classes=1,
            gpu_memory_gb=80.0,
            current_size=100,
        )


def test_relion_em_batch_sizing_clamps_highres_translation_tiles():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=1024,
        n_rot=1024,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 9
    assert plan.rotation_block_size == 13
    assert plan.pose_pixel_tile_gb < 3.0
    assert plan.translation_tile_gb <= plan.translation_tile_budget_gb


def test_relion_em_batch_sizing_accounts_for_k_classes():
    single = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=294912,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )
    k4 = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=294912,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=4,
        gpu_memory_gb=80.0,
    )

    assert k4.image_batch_size <= single.image_batch_size
    assert k4.rotation_block_size <= single.rotation_block_size
