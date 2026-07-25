import inspect
from types import SimpleNamespace

import numpy as np
import pytest


def test_k_class_replay_batch_plan_applies_estimator_and_kclass_caps(monkeypatch):
    from recovar.em.dense_single_volume import batch_planning, firstiter_cc
    from scripts.run_k_class_parity import _safe_k_class_replay_batch_plan

    captured = {}

    def fake_estimator(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(image_batch_size=250, rotation_block_size=5000)

    monkeypatch.setattr(batch_planning, "_estimate_relion_em_batch_sizes", fake_estimator)
    monkeypatch.setattr(firstiter_cc, "_safe_firstiter_cc_image_batch_size", lambda *_args: 17)
    monkeypatch.setattr(firstiter_cc, "_safe_dense_k_class_rotation_block_size", lambda *_args: 31)

    plan = _safe_k_class_replay_batch_plan(
        requested_image_batch_size=250,
        requested_rotation_block_size=5000,
        n_rot=24576,
        n_trans=116,
        n_classes=4,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        current_size=128,
    )

    assert captured["requested_image_batch_size"] == 250
    assert captured["requested_rotation_block_size"] == 5000
    assert captured["n_rot"] == 24576
    assert captured["n_trans"] == 116
    assert captured["n_classes"] == 4
    assert captured["current_size"] == 128
    assert plan.image_batch_size == 17
    assert plan.rotation_block_size == 31
    assert plan.requested_image_batch_size == 250
    assert plan.requested_rotation_block_size == 5000


def test_k_class_replay_recovers_exact_restart_perturbation():
    from scripts.run_k_class_parity import _resolve_target_random_perturbation

    value, source = _resolve_target_random_perturbation(
        star_value=-0.12306,
        perturbation_factor=0.5,
        random_seed=1778628798,
        target_iteration=10,
        restart_state_iteration=9,
        precision_mode="seed_exact",
    )

    assert value == -0.12305957078933716
    assert source == "seed-exact-restart@9"


def test_k_class_replay_rejects_missing_restart_boundary():
    from scripts.run_k_class_parity import _resolve_target_random_perturbation

    with pytest.raises(
        ValueError,
        match="pass --perturb-restart-state-iteration",
    ):
        _resolve_target_random_perturbation(
            star_value=-0.12306,
            perturbation_factor=0.5,
            random_seed=1778628798,
            target_iteration=10,
            restart_state_iteration=None,
            precision_mode="seed_exact",
        )


def test_k_class_replay_star_precision_is_explicitly_rounded():
    from scripts.run_k_class_parity import _resolve_target_random_perturbation

    value, source = _resolve_target_random_perturbation(
        star_value=-0.12306,
        perturbation_factor=0.5,
        random_seed=None,
        target_iteration=10,
        restart_state_iteration=None,
        precision_mode="star",
    )

    assert value == -0.12306
    assert source == "star-rounded"


def test_k_class_replay_firstiter_best_coarse_shortcut_is_diagnostic_only():
    import scripts.run_k_class_parity as run_k_class_parity

    source = inspect.getsource(run_k_class_parity.main)
    assert "--no-firstiter-cc-pass2-only-best-coarse" in source
    assert "--firstiter-cc-pass2-only-best-coarse" in source
    assert "args.no_firstiter_cc_pass2_only_best_coarse" in source
    assert "args.firstiter_cc_pass2_only_best_coarse" in source
    assert "Patched RELION storeWavg dumps" in source


def test_k_class_replay_reads_relion_firstiter_cc_cli_flag(tmp_path):
    from scripts.run_k_class_parity import _read_relion_optimiser_cli_flags

    (tmp_path / "run_it000_optimiser.star").write_text(
        "# --i particles.star --K 4 --firstiter_cc --ini_high 30 --random_seed 2802\n"
        "data_optimiser_general\n"
    )

    flags = _read_relion_optimiser_cli_flags(tmp_path, 0)

    assert flags["do_firstiter_cc"] is True
    assert flags["ini_high_angstrom"] == 30.0
    assert "--random_seed 2802" in flags["cli_line"]


def test_k_class_replay_reads_missing_ini_high_as_none(tmp_path):
    from scripts.run_k_class_parity import _read_relion_optimiser_cli_flags

    (tmp_path / "run_it000_optimiser.star").write_text(
        "# --i particles.star --K 4 --firstiter_cc --random_seed 2805\n"
        "data_optimiser_general\n"
    )

    flags = _read_relion_optimiser_cli_flags(tmp_path, 0)

    assert flags["do_firstiter_cc"] is True
    assert flags["ini_high_angstrom"] is None


def test_k_class_replay_splits_joint_direction_prior_like_production():
    from scripts.run_k_class_parity import _split_class_direction_prior_for_replay

    raw = np.asarray(
        [
            [0.20, 0.10, 0.10],
            [0.06, 0.18, 0.36],
        ],
        dtype=np.float32,
    )

    conditional, class_log_priors, class_weights = _split_class_direction_prior_for_replay(raw, n_classes=2)

    np.testing.assert_allclose(conditional.sum(axis=1), np.ones(2), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(conditional[0], [0.5, 0.25, 0.25], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(conditional[1], [0.1, 0.3, 0.6], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(class_weights, [0.4, 0.6], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(class_log_priors, np.log([0.4, 0.6]), rtol=1e-6, atol=1e-7)


def test_k_class_replay_auto_firstiter_cc_tracks_relion_cli():
    from scripts.run_k_class_parity import _resolve_firstiter_cc_mode

    args = SimpleNamespace(
        prev_iter=0,
        target_iter=1,
        firstiter_cc_mode="auto",
        winner_take_all_mstep=False,
    )

    off = _resolve_firstiter_cc_mode(args, {"do_firstiter_cc": False})
    on = _resolve_firstiter_cc_mode(args, {"do_firstiter_cc": True})

    assert off["emulate"] is False
    assert off["score_mode"] == "gaussian"
    assert on["emulate"] is True
    assert on["score_mode"] == "normalized_cc"


def test_k_class_replay_legacy_winner_take_all_forces_diagnostic_mode():
    from scripts.run_k_class_parity import _resolve_firstiter_cc_mode

    args = SimpleNamespace(
        prev_iter=0,
        target_iter=1,
        firstiter_cc_mode="auto",
        winner_take_all_mstep=True,
    )

    mode = _resolve_firstiter_cc_mode(args, {"do_firstiter_cc": False})

    assert mode["effective_mode"] == "force"
    assert mode["forced_by_winner_take_all_mstep"] is True
    assert mode["relion_requested"] is False
    assert mode["emulate"] is True


def test_k_class_replay_firstiter_lowpass_follows_relion_ini_high():
    from scripts.run_k_class_parity import (
        _resolve_firstiter_cc_mode,
        _resolve_firstiter_lowpass_ini_high_angstrom,
    )

    args = SimpleNamespace(
        prev_iter=0,
        target_iter=1,
        firstiter_cc_mode="auto",
        winner_take_all_mstep=False,
        firstiter_cc_ini_high_angstrom=None,
    )
    mode = _resolve_firstiter_cc_mode(args, {"do_firstiter_cc": True})

    assert _resolve_firstiter_lowpass_ini_high_angstrom(
        args,
        {"do_firstiter_cc": True, "ini_high_angstrom": None},
        mode,
    ) is None
    assert _resolve_firstiter_lowpass_ini_high_angstrom(
        args,
        {"do_firstiter_cc": True, "ini_high_angstrom": 30.0},
        mode,
    ) == 30.0

    args.firstiter_cc_ini_high_angstrom = 0.0
    assert _resolve_firstiter_lowpass_ini_high_angstrom(
        args,
        {"do_firstiter_cc": True, "ini_high_angstrom": 30.0},
        mode,
    ) is None

    args.firstiter_cc_ini_high_angstrom = 25.0
    assert _resolve_firstiter_lowpass_ini_high_angstrom(
        args,
        {"do_firstiter_cc": True, "ini_high_angstrom": None},
        mode,
    ) == 25.0


def test_k_class_replay_firstiter_lowpass_uses_exact_relion_helper():
    import scripts.run_k_class_parity as run_k_class_parity

    source = inspect.getsource(run_k_class_parity.main)

    assert "_apply_relion_initial_lowpass_filter" in source
    assert "filter_edgewidth=2.0" in source
    assert "locres.low_pass_filter_map" not in source


def test_k_class_replay_batch_plan_preserves_smaller_estimator_plan(monkeypatch):
    from recovar.em.dense_single_volume import batch_planning, firstiter_cc
    from scripts.run_k_class_parity import _safe_k_class_replay_batch_plan

    monkeypatch.setattr(
        batch_planning,
        "_estimate_relion_em_batch_sizes",
        lambda **_kwargs: SimpleNamespace(image_batch_size=9, rotation_block_size=11),
    )
    monkeypatch.setattr(firstiter_cc, "_safe_firstiter_cc_image_batch_size", lambda *_args: 17)
    monkeypatch.setattr(firstiter_cc, "_safe_dense_k_class_rotation_block_size", lambda *_args: 31)

    plan = _safe_k_class_replay_batch_plan(
        requested_image_batch_size=250,
        requested_rotation_block_size=5000,
        n_rot=1,
        n_trans=1,
        n_classes=4,
        image_shape=(64, 64),
        volume_shape=(64, 64, 64),
        padding_factor=2,
        current_size=32,
    )

    assert plan.image_batch_size == 9
    assert plan.rotation_block_size == 11


def test_relion_bpref_adaptive_diagnostic_recomputes_same_window_logz():
    import scripts.run_k_class_parity as run_k_class_parity

    source = inspect.getsource(run_k_class_parity.main)

    assert "bpref_significant_sample_indices" in source
    assert "RELION BPref diagnostic: recomputing same-window support" in source
    assert "if args.relion_bpref_mstep and args.adaptive_2pass:" in source
    assert "current_size=current_size" in source
    assert 'bpref_full_stats["normalization_log_z"]' in source
    assert 'significant_full_stats["normalization_log_z"]' not in source.split("if args.relion_bpref_mstep:", 1)[1]
    assert 'normalization_score_mode="gaussian"' in source


def test_relion_bpref_diagnostic_is_bounded_and_nonfatal():
    import scripts.run_k_class_parity as run_k_class_parity

    main_source = inspect.getsource(run_k_class_parity.main)
    module_source = inspect.getsource(run_k_class_parity)

    assert "--relion-bpref-max-images-per-microbatch" in main_source
    assert "max_images_per_microbatch=args.relion_bpref_max_images_per_microbatch" in main_source
    assert "max_images_per_microbatch=max_images_per_microbatch" in module_source
    assert "RELION BPref diagnostic failed" in main_source
    assert '"error": f"{type(exc).__name__}: {exc}"' in main_source


def test_k_class_replay_adaptive_sparse_pass2_enables_fine_mstep_pruning():
    import scripts.run_k_class_parity as run_k_class_parity

    source = inspect.getsource(run_k_class_parity.main)
    adaptive_block = source.split("if args.adaptive_2pass:", 1)[1].split("else:", 1)[0]

    assert 'adaptive_em_kwargs["relion_fine_mstep_prune"] = bool(args.sparse_pass2)' in adaptive_block


def test_k_class_replay_sets_numbered_half_capture_context():
    import scripts.run_k_class_parity as run_k_class_parity

    source = inspect.getsource(run_k_class_parity.main)

    assert "set_bpref_contribution_dump_context(" in source
    assert "iteration=args.target_iter" in source
    assert "half=1" in source
    assert "clear_bpref_contribution_dump_context()" in source


def test_k_class_replay_exposes_relion_x_half_capture_path():
    import scripts.run_k_class_parity as run_k_class_parity

    source = inspect.getsource(run_k_class_parity.main)

    assert "--relion-x-half-mstep" in source
    assert "mstep_relion_x_half=bool(args.relion_x_half_mstep)" in source
    assert "bpref_device_signature_active=bool(" in source
    assert 'os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR")' in source


def test_k_class_replay_records_responsibility_class_diagnostics():
    import scripts.run_k_class_parity as run_k_class_parity

    source = inspect.getsource(run_k_class_parity.main)

    assert "recovar_class_responsibilities" in source
    assert "mapped_recovar_class_by_responsibility" in source
    assert "class_assignment_by_responsibility_accuracy_after_permutation" in source
    assert "class_assignment_best_vs_responsibility_disagreement_count" in source


def test_k_class_replay_reports_mstep_class_weights_against_relion():
    import scripts.run_k_class_parity as run_k_class_parity

    source = inspect.getsource(run_k_class_parity.main)

    assert 'getattr(result, "class_mstep_posterior_sums", None)' in source
    assert '"recovar_class_weights": recovar_weights' in source
    assert '"recovar_full_posterior_class_weights": recovar_full_posterior_weights' in source


def test_k_class_replay_supports_stop_after_pass2_dump_diagnostic():
    import scripts.run_k_class_parity as run_k_class_parity

    main_source = inspect.getsource(run_k_class_parity.main)
    module_source = inspect.getsource(run_k_class_parity)

    assert "--stop-after-pass2-dump" in main_source
    assert "RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET" in main_source
    assert "RECOVAR_K_CLASS_PARITY_STOP_AFTER_PASS2_DUMP" in module_source
    assert "Pass2DumpComplete" in module_source


def test_k_class_replay_uses_exact_relion_projector_for_scoring():
    import scripts.run_k_class_parity as run_k_class_parity

    source = inspect.getsource(run_k_class_parity.main)

    assert "reference_to_relion_projector_half_maps" in source
    assert "prev_reference_real" in source
    assert "relion_projector_half_by_class, relion_projector_r_max" in source
    assert "current_size=projector_current_size" in source
    assert "projection_padding_factor=args.projection_padding_factor" in source
    assert "relion_projector_half=relion_projector_half_by_class" in source
    assert "relion_projector_r_max=relion_projector_r_max" in source
    assert "relion_projector_half=relion_projector_half_by_class[class_index]" in source


def test_relion_adaptive_coarse_image_size_matches_replay_case8():
    from scripts.run_k_class_parity import _relion_adaptive_coarse_image_size

    assert (
        _relion_adaptive_coarse_image_size(
            healpix_order=1,
            pixel_size=4.25,
            grid_size=128,
            particle_diameter=380.0,
            current_size=44,
        )
        == 14
    )


def test_relion_adaptive_coarse_image_size_clamps_to_current_size():
    from scripts.run_k_class_parity import _relion_adaptive_coarse_image_size

    assert (
        _relion_adaptive_coarse_image_size(
            healpix_order=3,
            pixel_size=4.25,
            grid_size=128,
            particle_diameter=380.0,
            current_size=44,
        )
        == 44
    )


def test_relion_adaptive_fine_translation_perturbation_uses_coarse_step():
    from scripts.run_k_class_parity import _relion_adaptive_fine_translation_grid

    base_translations = np.asarray([[0.0, 0.0]], dtype=np.float32)
    fine_translations, parent_map = _relion_adaptive_fine_translation_grid(
        base_translations,
        offset_step_px=2.0,
        adaptive_oversampling=1,
        random_perturbation=-0.35825,
    )

    np.testing.assert_array_equal(parent_map, np.zeros(4, dtype=np.int64))
    np.testing.assert_allclose(
        fine_translations,
        [
            [-1.2165, -1.2165],
            [-1.2165, -0.2165],
            [-0.2165, -1.2165],
            [-0.2165, -0.2165],
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.isclose(float(fine_translations[-1, 0]), 0.14175)


def test_runtime_scale_dump_override_uses_relion_one_based_stack_index(tmp_path):
    from scripts.run_k_class_parity import _apply_runtime_scale_dump_override

    dump_dir = tmp_path / "relion_dump"
    dump_dir.mkdir()
    np.asarray([2347.0], dtype=np.float64).tofile(dump_dir / "pass0_acc_stack_index.bin")
    np.asarray([0.7572658658], dtype=np.float64).tofile(dump_dir / "pass0_img0_scale_correction.bin")

    image_corrections = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    scale_corrections = np.asarray([1.0, 0.998293996, 1.5], dtype=np.float32)
    relion_df_ordered = {
        "rlnImageName": [
            "000001@particles.mrcs",
            "002347@particles.mrcs",
            "002348@particles.mrcs",
        ]
    }

    out_image, out_scale = _apply_runtime_scale_dump_override(
        image_corrections,
        scale_corrections,
        relion_df_ordered,
        dump_dir,
    )

    expected_image = image_corrections[1] * (0.7572658658 / float(scale_corrections[1]))
    np.testing.assert_allclose(out_scale, [1.0, 0.7572658658, 1.5], rtol=1e-7)
    np.testing.assert_allclose(out_image, [10.0, expected_image, 30.0], rtol=1e-6)
