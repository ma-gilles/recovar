from types import SimpleNamespace

import numpy as np

from recovar.em.dense_single_volume import iteration_loop
from recovar.em.dense_single_volume.batch_planning import _estimate_relion_em_batch_sizes
from recovar.em.dense_single_volume.firstiter_cc import _safe_firstiter_cc_image_batch_size


def test_firstiter_cc_budget_preserves_256_k4_completion_batch_size():
    # K=4 completion benchmarks use 256^2 images and 116 fine translations
    # at adaptive_oversampling=1. The cap must not collapse the requested
    # batch size 50 back to single digits on A100/H100 runs.
    assert _safe_firstiter_cc_image_batch_size(116, (256, 256)) >= 50


def test_firstiter_cc_budget_still_caps_larger_tiles():
    assert 1 <= _safe_firstiter_cc_image_batch_size(137, (384, 384)) < 250


def test_firstiter_cc_adaptive_dispatch_clamps_against_fine_translation_grid(monkeypatch):
    captured = {}
    fine_trans = np.zeros((116, 2), dtype=np.float32)

    def fake_grids(*args, **kwargs):
        coarse_rot = np.zeros((576, 3, 3), dtype=np.float32)
        coarse_trans = np.zeros((29, 2), dtype=np.float32)
        fine_rot = np.zeros((4608, 3, 3), dtype=np.float32)
        rot_parent = np.zeros(fine_rot.shape[0], dtype=np.int64)
        trans_parent = np.zeros(fine_trans.shape[0], dtype=np.int64)
        return coarse_rot, coarse_trans, fine_rot, fine_trans, rot_parent, trans_parent

    def fake_adaptive(*args, **kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(iteration_loop, "_build_firstiter_cc_pass2_grids", fake_grids)
    monkeypatch.setattr(iteration_loop, "run_dense_k_class_em_adaptive", fake_adaptive)

    result, _rot_parent, _trans_parent, n_trans_fine, _adaptive_os = iteration_loop._score_kclass_firstiter_cc_pass2(
        experiment_dataset=object(),
        mean=None,
        mean_variance=None,
        noise_variance_k=None,
        effective_rotations=np.zeros((576, 3, 3), dtype=np.float32),
        current_translations=np.zeros((29, 2), dtype=np.float32),
        base_translations=np.zeros((29, 2), dtype=np.float32),
        current_healpix_order=1,
        state=SimpleNamespace(adaptive_oversampling=1, translation_step=2.0),
        random_perturbation=0.0,
        disc_type="linear_interp",
        class_log_priors=None,
        image_batch_size=200,
        image_shape_k=(256, 256),
        em_kwargs={"image_batch_size": 88, "rotation_block_size": 576},
        update_em_kwargs_image_batch_size=True,
    )

    assert result == "result"
    assert n_trans_fine == 116
    assert captured["image_batch_size"] == _safe_firstiter_cc_image_batch_size(116, (256, 256))
    assert captured["image_batch_size"] < 88


def test_dense_global_k1_batch_plan_accounts_for_pose_pixel_tile():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=500,
        requested_rotation_block_size=40000,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=42,
        current_size=56,
    )

    assert plan.image_batch_size == 187
    assert plan.rotation_block_size < 9000
    assert plan.pose_pixel_tile_gb <= plan.projection_budget_gb * 1.01


def test_dense_global_k1_high_current_size_keeps_pose_pixel_tile_below_large_allocations():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=500,
        requested_rotation_block_size=40000,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=42,
        current_size=184,
    )

    assert 100 <= plan.image_batch_size < 150
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb * 1.01
    assert plan.rotation_block_size < 250
    assert plan.pose_pixel_tile_gb < 1.7
