from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import iteration_loop, k_class
from recovar.em.dense_single_volume.batch_planning import _estimate_relion_em_batch_sizes
from recovar.em.dense_single_volume.firstiter_cc import (
    _safe_dense_k_class_rotation_block_size,
    _safe_firstiter_cc_image_batch_size,
)
from recovar.em.dense_single_volume.helpers.types import NoiseStats, make_relion_stats
from recovar.em.dense_single_volume.k_class import KClassEMResult


def test_firstiter_winner_take_all_assembly_reports_unit_pmax_across_score_normalizations():
    """RELION reports Pmax=1 after firstiter-CC binarizes the winning weight."""
    per_class_stats = (
        make_relion_stats(
            log_evidence_per_image=np.array([1_000.0], dtype=np.float32),
            best_log_score_per_image=np.array([-1_000.0], dtype=np.float32),
            max_posterior_per_image=np.ones(1, dtype=np.float32),
            rotation_posterior_sums=np.ones(1, dtype=np.float32),
        ),
    )

    result = k_class._assemble_result(
        class_log_evidence=np.array([[1_000.0]], dtype=np.float64),
        new_means=None,
        Ft_y=[jnp.zeros(1, dtype=jnp.complex64)],
        Ft_ctf=[jnp.zeros(1, dtype=jnp.float32)],
        per_class_hard_assignments=np.zeros((1, 1), dtype=np.int32),
        per_class_stats=per_class_stats,
        noise_stats=None,
        firstiter_winner_take_all=True,
    )

    np.testing.assert_array_equal(np.asarray(result.stats.max_posterior_per_image), np.ones(1))


def test_firstiter_cc_budget_preserves_256_k4_completion_batch_size():
    # K=4 completion benchmarks use 256^2 images and 116 fine translations
    # at adaptive_oversampling=1. The cap must not collapse the requested
    # batch size 50 back to single digits on A100/H100 runs.
    assert _safe_firstiter_cc_image_batch_size(116, (256, 256)) >= 50


def test_firstiter_cc_budget_still_caps_larger_tiles():
    assert 1 <= _safe_firstiter_cc_image_batch_size(137, (384, 384)) < 250


def test_firstiter_cc_budget_env_override_lifts_debug_cap(monkeypatch):
    default_batch = _safe_firstiter_cc_image_batch_size(116, (256, 256))
    assert default_batch == 70

    monkeypatch.setenv("RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET", str(3 * 268_435_456))

    assert _safe_firstiter_cc_image_batch_size(116, (256, 256)) >= 187


def test_firstiter_cc_budget_env_override_rejects_invalid(monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET", "0")

    try:
        _safe_firstiter_cc_image_batch_size(116, (256, 256))
    except ValueError as exc:
        assert "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET" in str(exc)
    else:
        raise AssertionError("invalid firstiter budget override did not raise")


def test_kclass_adaptive_grid_batch_plan_uses_fine_grid_for_pass2():
    calls = []

    def fake_safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        calls.append((int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch))
        if int(n_rot) == 4608:
            return 44, 275
        if int(n_rot) == 576:
            return 50, 576
        raise AssertionError((n_rot, n_trans))

    plan = iteration_loop._plan_kclass_adaptive_grid_batch_sizes(
        coarse_rotations=np.zeros((576, 3, 3), dtype=np.float32),
        coarse_translations=np.zeros((29, 2), dtype=np.float32),
        fine_rotations=np.zeros((4608, 3, 3), dtype=np.float32),
        fine_translations=np.zeros((116, 2), dtype=np.float32),
        n_classes=4,
        image_shape=(256, 256),
        coarse_current_size=40,
        fine_current_size=90,
        safe_batch_sizes=fake_safe_batch_sizes,
    )

    assert calls == [
        (4608, 116, 4, (256, 256), 90),
        (576, 29, 4, (256, 256), 40),
    ]
    assert plan.pass2_image_batch_size == 44
    assert plan.pass2_rotation_block_size == 275
    assert plan.significance_image_batch_size == 50
    assert plan.significance_rotation_block_size == 576


def test_firstiter_cc_adaptive_dispatch_clamps_against_fine_translation_grid(monkeypatch):
    captured = {}
    fine_trans = np.zeros((116, 2), dtype=np.float32)

    def fake_grids(*args, **kwargs):
        coarse_rot = np.zeros((576, 3, 3), dtype=np.float32)
        coarse_trans = np.zeros((29, 2), dtype=np.float32)
        fine_rot = np.zeros((4608, 3, 3), dtype=np.float32)
        rot_parent = np.zeros(fine_rot.shape[0], dtype=np.int64)
        trans_parent = np.zeros(fine_trans.shape[0], dtype=np.int64)
        outputs = (coarse_rot, coarse_trans, fine_rot, fine_trans, rot_parent, trans_parent)
        if kwargs.get("return_mstep_rotations", False):
            return (*outputs, np.full_like(fine_rot, 0.25))
        return outputs

    def fake_adaptive(*args, **kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(iteration_loop, "_build_firstiter_cc_pass2_grids", fake_grids)
    monkeypatch.setattr(iteration_loop, "run_dense_k_class_em_adaptive", fake_adaptive)

    def fake_safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        assert classes == 2
        assert image_shape_for_batch == (256, 256)
        if (int(n_rot), int(n_trans), current_size_for_batch) == (4608, 116, 90):
            return 5, 999
        if (int(n_rot), int(n_trans), current_size_for_batch) == (576, 29, 40):
            return 120, 700
        raise AssertionError((n_rot, n_trans, current_size_for_batch))

    result, _rot_parent, _trans_parent, n_trans_fine, _adaptive_os = iteration_loop._score_kclass_firstiter_cc_pass2(
        experiment_dataset=object(),
        mean=np.zeros((2, 4), dtype=np.complex64),
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
        safe_batch_sizes=fake_safe_batch_sizes,
        coarse_current_size=40,
        fine_current_size=90,
        update_em_kwargs_image_batch_size=True,
    )

    assert result == "result"
    assert n_trans_fine == 116
    expected_fine_ibs = min(88, _safe_firstiter_cc_image_batch_size(116, (256, 256)))
    expected_coarse_ibs = min(120, _safe_firstiter_cc_image_batch_size(29, (256, 256)))
    assert captured["image_batch_size"] == expected_fine_ibs
    assert captured["rotation_block_size"] == min(576, _safe_dense_k_class_rotation_block_size(116, expected_fine_ibs))
    assert captured["significance_image_batch_size"] == expected_coarse_ibs
    assert captured["significance_rotation_block_size"] == min(
        700,
        _safe_dense_k_class_rotation_block_size(29, expected_coarse_ibs),
    )
    assert captured["firstiter_cc_pass2_only_best_coarse"] is True
    assert captured["skip_significance_pruning"] is False
    assert captured["relion_fine_mstep_prune"] is True
    assert np.all(captured["fine_mstep_rotations_override"] == 0.25)


def test_k1_firstiter_cc_dispatch_uses_coarse_batch_for_significance(monkeypatch):
    captured = {}
    calls = []

    class TinyDataset:
        image_shape = (256, 256)

    def fake_grids(*args, **kwargs):
        coarse_rot = np.zeros((576, 3, 3), dtype=np.float32)
        coarse_trans = np.zeros((29, 2), dtype=np.float32)
        fine_rot = np.zeros((4608, 3, 3), dtype=np.float32)
        fine_trans = np.zeros((116, 2), dtype=np.float32)
        rot_parent = np.arange(fine_rot.shape[0], dtype=np.int64) % coarse_rot.shape[0]
        trans_parent = np.arange(fine_trans.shape[0], dtype=np.int64) % coarse_trans.shape[0]
        outputs = (coarse_rot, coarse_trans, fine_rot, fine_trans, rot_parent, trans_parent)
        if kwargs.get("return_mstep_rotations", False):
            return (*outputs, np.full_like(fine_rot, 0.25))
        return outputs

    def fake_safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        calls.append((int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch))
        if (int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch) == (
            576,
            29,
            None,
            None,
            90,
        ):
            return 187, 700
        if (int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch) == (
            4608,
            116,
            1,
            (256, 256),
            90,
        ):
            return 5, 999
        if (int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch) == (
            576,
            29,
            1,
            (256, 256),
            40,
        ):
            return 187, 700
        raise AssertionError((n_rot, n_trans, classes, image_shape_for_batch, current_size_for_batch))

    def fake_adaptive(*args, **kwargs):
        captured.update(kwargs)
        n_images = 3
        n_classes = 1
        n_fine_rot = 4608
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=np.zeros(n_images, dtype=np.float32),
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_fine_rot, dtype=np.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(2, dtype=jnp.float32),
            wsum_img_power=jnp.ones(2, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(n_images),
        )
        return KClassEMResult(
            new_means=jnp.zeros((n_classes, 4), dtype=jnp.complex64),
            Ft_y=jnp.zeros((n_classes, 4), dtype=jnp.complex64),
            Ft_ctf=jnp.ones((n_classes, 4), dtype=jnp.float32),
            per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
            class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            class_responsibilities=jnp.ones((n_classes, n_images), dtype=jnp.float32),
            class_posterior_sums=jnp.ones(n_classes, dtype=jnp.float32),
            stats=stats,
            per_class_stats=(stats,),
            noise_stats=(noise_stats,),
            aggregate_noise_stats=noise_stats,
            best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
            best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
            best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
        )

    monkeypatch.setattr(iteration_loop, "_build_firstiter_cc_pass2_grids", fake_grids)
    monkeypatch.setattr(iteration_loop, "run_dense_k_class_em_adaptive", fake_adaptive)

    result = iteration_loop._score_half_dense(
        k=0,
        experiment_dataset=TinyDataset(),
        means_k=jnp.zeros(4, dtype=jnp.complex64),
        mean_variance=jnp.ones(4, dtype=jnp.float32),
        noise_variance_k=jnp.ones(4, dtype=jnp.float32),
        effective_rotations=np.zeros((576, 3, 3), dtype=np.float32),
        current_translations=np.zeros((29, 2), dtype=np.float32),
        base_translations=np.zeros((29, 2), dtype=np.float32),
        current_healpix_order=1,
        state=SimpleNamespace(adaptive_oversampling=1, translation_step=2.0),
        random_perturbation=0.0,
        disc_type="linear_interp",
        image_batch_size=187,
        rotation_log_prior_k=None,
        class_rotation_log_prior_k=None,
        translation_log_prior=None,
        translation_search_base=None,
        trans_prior_center_for_engine=None,
        image_corrections_k=None,
        scale_corrections_k=None,
        firstiter_score_mode_this_iter="normalized_cc",
        firstiter_winner_take_all_this_iter=True,
        cs_for_engine=90,
        class_log_priors=None,
        k_class_enabled=False,
        relion_firstiter_cc_this_iter=True,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        safe_batch_sizes=fake_safe_batch_sizes,
        max_significants=None,
        noise_stats_per_half_per_class=[None, None],
        class_assignments=[None, None],
        class_posterior_per_half=[None, None],
        class_full_posterior_per_half=[None, None],
        class_rotation_posterior_per_half=[None, None],
        best_pose_rotations=[None, None],
        best_pose_rotation_eulers=[None, None],
        best_pose_translations=[None, None],
        firstiter_coarse_current_size=40,
        firstiter_fine_current_size=90,
        bpref_device_signature_active=True,
        debug_iteration=7,
    )

    assert calls == [
        (576, 29, None, None, 90),
        (4608, 116, 1, (256, 256), 90),
        (576, 29, 1, (256, 256), 40),
    ]
    assert captured["image_batch_size"] == _safe_firstiter_cc_image_batch_size(116, (256, 256))
    assert captured["significance_image_batch_size"] == 187
    assert captured["rotation_block_size"] == min(700, _safe_dense_k_class_rotation_block_size(116, captured["image_batch_size"]))
    assert captured["significance_rotation_block_size"] == 700
    assert captured["bpref_device_signature_active"] is True
    assert captured["debug_iteration"] == 7
    assert np.all(captured["fine_mstep_rotations_override"] == 0.25)
    assert result.ha.shape == (3,)
    assert result.coarse_ha.shape == (3,)


def test_kclass_nonfirstiter_adaptive_dispatch_sizes_actual_fine_grid(monkeypatch):
    captured = {}

    class TinyDataset:
        image_shape = (256, 256)

    def fake_grids(*args, **kwargs):
        coarse_rot = np.zeros((576, 3, 3), dtype=np.float32)
        coarse_trans = np.zeros((29, 2), dtype=np.float32)
        fine_rot = np.zeros((4608, 3, 3), dtype=np.float32)
        fine_trans = np.zeros((116, 2), dtype=np.float32)
        rot_parent = np.arange(fine_rot.shape[0], dtype=np.int64) % coarse_rot.shape[0]
        trans_parent = np.arange(fine_trans.shape[0], dtype=np.int64) % coarse_trans.shape[0]
        outputs = (coarse_rot, coarse_trans, fine_rot, fine_trans, rot_parent, trans_parent)
        if kwargs.get("return_mstep_rotations", False):
            return (*outputs, np.full_like(fine_rot, 0.25))
        return outputs

    def fake_safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        assert classes in {None, 4}
        assert image_shape_for_batch in {None, (256, 256)}
        if (int(n_rot), int(n_trans), current_size_for_batch) == (4608, 116, 90):
            return 44, 275
        if (int(n_rot), int(n_trans), current_size_for_batch) == (576, 29, 40):
            return 50, 576
        if (int(n_rot), int(n_trans), current_size_for_batch) == (576, 29, 90):
            return 50, 2000
        raise AssertionError((n_rot, n_trans, current_size_for_batch))

    def fake_adaptive(*args, **kwargs):
        captured.update(kwargs)
        n_images = 3
        n_classes = 4
        n_fine_rot = 4608
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=np.zeros(n_images, dtype=np.float32),
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_fine_rot, dtype=np.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(2, dtype=jnp.float32),
            wsum_img_power=jnp.ones(2, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(n_images),
        )
        return KClassEMResult(
            new_means=jnp.zeros((n_classes, 4), dtype=jnp.complex64),
            Ft_y=jnp.zeros((n_classes, 4), dtype=jnp.complex64),
            Ft_ctf=jnp.ones((n_classes, 4), dtype=jnp.float32),
            per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
            class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            class_responsibilities=jnp.ones((n_classes, n_images), dtype=jnp.float32) / n_classes,
            class_posterior_sums=jnp.ones(n_classes, dtype=jnp.float32),
            stats=stats,
            per_class_stats=tuple(stats for _ in range(n_classes)),
            noise_stats=tuple(noise_stats for _ in range(n_classes)),
            aggregate_noise_stats=noise_stats,
            best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
            best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
            best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
        )

    monkeypatch.setattr(iteration_loop, "_build_firstiter_cc_pass2_grids", fake_grids)
    monkeypatch.setattr(iteration_loop, "run_dense_k_class_em_adaptive", fake_adaptive)

    result = iteration_loop._score_half_dense(
        k=0,
        experiment_dataset=TinyDataset(),
        means_k=jnp.zeros((4, 4), dtype=jnp.complex64),
        mean_variance=jnp.ones(4, dtype=jnp.float32),
        noise_variance_k=jnp.ones(4, dtype=jnp.float32),
        effective_rotations=np.zeros((576, 3, 3), dtype=np.float32),
        current_translations=np.zeros((29, 2), dtype=np.float32),
        base_translations=np.zeros((29, 2), dtype=np.float32),
        current_healpix_order=1,
        state=SimpleNamespace(adaptive_oversampling=1, translation_step=2.0),
        random_perturbation=0.0,
        disc_type="linear_interp",
        image_batch_size=50,
        rotation_log_prior_k=None,
        class_rotation_log_prior_k=None,
        translation_log_prior=None,
        translation_search_base=None,
        trans_prior_center_for_engine=None,
        image_corrections_k=None,
        scale_corrections_k=None,
        firstiter_score_mode_this_iter="gaussian",
        firstiter_winner_take_all_this_iter=False,
        cs_for_engine=90,
        class_log_priors=np.zeros(4, dtype=np.float32),
        k_class_enabled=True,
        relion_firstiter_cc_this_iter=False,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        safe_batch_sizes=fake_safe_batch_sizes,
        max_significants=None,
        noise_stats_per_half_per_class=[None, None],
        class_assignments=[None, None],
        class_posterior_per_half=[None, None],
        class_full_posterior_per_half=[None, None],
        class_rotation_posterior_per_half=[None, None],
        best_pose_rotations=[None, None],
        best_pose_rotation_eulers=[None, None],
        best_pose_translations=[None, None],
        k_class_image_batch_size_override=50,
        k_class_rotation_block_size_override=2000,
        firstiter_coarse_current_size=40,
        firstiter_fine_current_size=90,
    )

    assert captured["image_batch_size"] == 44
    assert captured["rotation_block_size"] == 275
    assert captured["significance_image_batch_size"] == 50
    assert captured["significance_rotation_block_size"] == 576
    assert captured["sparse_pass2"] is True
    assert np.all(captured["fine_mstep_rotations_override"] == 0.25)
    assert result.ha.shape == (3,)


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


def _box800_plan(**overrides):
    kwargs = dict(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=4608,
        n_trans=84,
        image_shape=(800, 800),
        volume_shape=(800, 800, 800),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80,
        current_size=62,
    )
    kwargs.update(overrides)
    return _estimate_relion_em_batch_sizes(**kwargs)


def test_box800_k1_score_tile_plan_uses_resolved_precision():
    historical = _box800_plan()
    float64_plan = _box800_plan(use_float64_scoring=True)
    float32_plan = _box800_plan(use_float64_scoring=False)

    assert historical == float64_plan
    assert float64_plan.rotation_block_size == 194
    assert float64_plan.pose_pixel_tile_gb == pytest.approx(0.49930944)
    assert float32_plan.rotation_block_size == 388
    assert float32_plan.pose_pixel_tile_gb == pytest.approx(float64_plan.pose_pixel_tile_gb)
    assert float32_plan.active_score_tile_gb == pytest.approx(
        float64_plan.active_score_tile_gb / 2.0
    )
    assert float32_plan.image_batch_size == float64_plan.image_batch_size == 1
    assert float32_plan.persistent_estimate_gb == float64_plan.persistent_estimate_gb == 81.92


def test_box800_compact_k1_persistent_estimate_uses_model_current_size_phase_max():
    compact = dict(
        gpu_memory_gb=85,
        use_float64_scoring=False,
        compact_k1_relion_layout=True,
    )
    current_size_62 = _box800_plan(**compact, model_current_size=62)
    full_box_model = _box800_plan(**compact, model_current_size=800)
    small_voxels = 127 * 127 * 64
    full_voxels = 1603 * 1603 * 802
    assert current_size_62.persistent_estimate_mode == "compact_k1_relion_phase_max"
    assert current_size_62.persistent_estimate_gb == pytest.approx(
        4.0 + small_voxels * 12 / 1e9
    )
    assert current_size_62.pending_score_persistent_gb == pytest.approx(
        small_voxels * 8 / 1e9
    )
    assert full_box_model.persistent_estimate_gb == pytest.approx(
        4.0 + full_voxels * 12 / 1e9
    )
    assert full_box_model.pending_score_persistent_gb == pytest.approx(
        full_voxels * 8 / 1e9
    )
    assert current_size_62.score_pixel_count == full_box_model.score_pixel_count == 1532


@pytest.mark.parametrize(
    ("score_size", "model_size", "n_rot", "n_trans", "live_free_gb", "image_batch", "rotation_block"),
    [
        (62, 62, 576, 21, 79.0, 64, 576),
        (62, 62, 4608, 84, 75.0, 23, 4608),
        (78, 800, 576, 21, 47_724 * 1024**2 / 1e9, 42, 576),
        (None, 800, 4608, 84, 47_724 * 1024**2 / 1e9, 3, 11),
    ],
)
def test_box800_compact_k1_observed_live_plans(
    score_size,
    model_size,
    n_rot,
    n_trans,
    live_free_gb,
    image_batch,
    rotation_block,
):
    plan = _box800_plan(
        n_rot=n_rot,
        n_trans=n_trans,
        gpu_memory_gb=85,
        current_size=score_size,
        use_float64_scoring=False,
        compact_k1_relion_layout=True,
        model_current_size=model_size,
        runtime_free_memory_gb=live_free_gb,
    )
    expected_usable = (live_free_gb - plan.pending_score_persistent_gb) * 0.8
    assert plan.usable_estimate_gb == pytest.approx(expected_usable)
    assert plan.image_batch_size == image_batch
    assert plan.rotation_block_size == rotation_block


def test_compact_k1_plan_rejects_genuine_float64_and_kclass_routes():
    compact = dict(
        compact_k1_relion_layout=True,
        model_current_size=62,
    )
    with pytest.raises(ValueError, match="float32/complex64"):
        _box800_plan(
            **compact,
            n_classes=1,
            use_float64_scoring=True,
        )
    with pytest.raises(ValueError, match="K=1-only"):
        _box800_plan(
            **compact,
            n_classes=2,
            use_float64_scoring=False,
        )


def _compact_route_decision(*, projector_half, score_complex_dtype=np.complex64):
    sparse = iteration_loop._sparse_pass2_diagnostics
    return sparse._relion_firstiter_compact_batch_planning_decision(
        source_faithful_spectrum_norm=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        use_relion_x_half_mstep=True,
        projector_half=projector_half,
        score_complex_dtype=score_complex_dtype,
        recon_volume_size=1603 * 1603 * 802,
        bpref_device_signature_active=False,
        fixed_base_bytes=4_000_000_000,
    )


def test_compact_k1_route_gate_requires_host_c64_and_no_diagnostics(monkeypatch):
    sparse = iteration_loop._sparse_pass2_diagnostics
    host_projector = SimpleNamespace(
        shape=(1603, 1603, 802),
        dtype=np.dtype(np.complex64),
    )
    monkeypatch.setattr(sparse, "_device_memory_limit_bytes", lambda: 85 * 1024**3)
    monkeypatch.setattr(sparse, "_jax_allocator_free_memory_bytes", lambda: 40 * 1024**3)
    for name in (
        "RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS",
        "RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF",
        "RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF",
        "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR",
        "RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR",
        "RECOVAR_BPREF_ACCUMULATOR_DELTA_DUMP_DIR",
        "RECOVAR_PASS2_DUMP_DIR",
        "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
        "RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS",
        "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
    ):
        monkeypatch.delenv(name, raising=False)

    sparse.set_bpref_contribution_dump_context(iteration=1, half=1)
    try:
        enabled = _compact_route_decision(projector_half=host_projector)
        assert enabled.enabled is True
        assert enabled.deferred_firstiter_bpref is True

        float64 = _compact_route_decision(
            projector_half=host_projector,
            score_complex_dtype=np.complex128,
        )
        assert float64.enabled is False

        class DeviceProjector:
            shape = (2, 2, 2)
            dtype = np.dtype(np.complex64)

        monkeypatch.setattr(sparse.jax, "Array", DeviceProjector)
        device_owned = _compact_route_decision(projector_half=DeviceProjector())
        assert device_owned.enabled is False

        monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", "/tmp/diagnostic")
        diagnostic = _compact_route_decision(projector_half=host_projector)
        assert diagnostic.enabled is False
        assert diagnostic.deferred_firstiter_bpref is False
    finally:
        sparse.clear_bpref_contribution_dump_context()


def test_compact_k1_route_gate_honors_explicit_route_env(monkeypatch):
    sparse = iteration_loop._sparse_pass2_diagnostics
    host_projector = SimpleNamespace(
        shape=(1603, 1603, 802),
        dtype=np.dtype(np.complex64),
    )
    monkeypatch.setattr(sparse, "_device_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(sparse, "_jax_allocator_free_memory_bytes", lambda: None)
    sparse.set_bpref_contribution_dump_context(iteration=1, half=1)
    try:
        monkeypatch.setenv("RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF", "1")
        forced_deferred = _compact_route_decision(projector_half=host_projector)
        assert forced_deferred.enabled is True
        assert forced_deferred.deferred_firstiter_bpref is True

        monkeypatch.setenv("RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF", "0")
        disabled_deferred = _compact_route_decision(projector_half=host_projector)
        assert disabled_deferred.enabled is False
        assert disabled_deferred.deferred_firstiter_bpref is False

        monkeypatch.delenv("RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF")
        monkeypatch.setenv("RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF", "0")
        disabled_fused = _compact_route_decision(projector_half=host_projector)
        assert disabled_fused.enabled is False
    finally:
        sparse.clear_bpref_contribution_dump_context()
