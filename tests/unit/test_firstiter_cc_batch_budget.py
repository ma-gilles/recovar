from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

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
