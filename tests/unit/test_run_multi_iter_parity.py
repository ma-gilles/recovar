import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import diff_relion_recovar_per_iter as parity_diff
from scripts.postprocess_multi_iter_gt import resolve_intermediates_dir
from scripts.run_multi_iter_parity import (
    _normalized_fsc_auc,
    _read_relion_scheduling_average_pmax,
    add_significant_count_artifacts,
    apply_iteration_normalization_factor_overrides,
    build_gt_postprocess_command,
    filter_fresh_initial_reference,
    final_only_replay_override,
    final_output_fourier_volumes,
    initial_scoring_noise_pair,
    load_initial_direction_prior,
    load_initial_fourier_volume,
    load_initial_noise_variance,
    map_pose_arrays_to_particle_order,
    map_relion_half_orders_to_dataset_rows,
    map_relion_scale_groups_to_half_order,
    parse_iteration_normalization_factor_overrides,
    parse_relion_optimiser_cli_flags,
    particle_half_indices,
    relion_final_gt_series,
    replay_control_relion_iteration,
    replay_override_iteration_pairs,
    replay_previous_relion_iteration,
    resolve_firstiter_cc_mode,
    resolve_relion_final_oracle_paths,
    retain_group_scale_update_state,
    select_final_replay_override,
    stack_index_from_image_name,
    validate_final_only_replay_args,
    validate_fresh_initial_reference_args,
    validate_fresh_particle_order_args,
)


def test_iteration_normalization_override_applies_only_at_requested_boundary():
    overrides = parse_iteration_normalization_factor_overrides(
        ["2:79452:0.9788520932197571"]
    )
    corrections = [
        np.asarray([1.0, 1.1], dtype=np.float32),
        np.asarray([1.2, 1.3], dtype=np.float32),
    ]
    scales = [
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([0.5, 2.0], dtype=np.float32),
    ]

    unchanged, applied_before = apply_iteration_normalization_factor_overrides(
        corrections,
        scales,
        half_stack_indices=[[10, 11], [79452, 13]],
        scoring_iteration=1,
        overrides=overrides,
    )
    corrected, applied = apply_iteration_normalization_factor_overrides(
        corrections,
        scales,
        half_stack_indices=[[10, 11], [79452, 13]],
        scoring_iteration=2,
        overrides=overrides,
    )

    np.testing.assert_array_equal(unchanged[0], corrections[0])
    np.testing.assert_array_equal(unchanged[1], corrections[1])
    assert applied_before == []
    np.testing.assert_array_equal(corrected[0], corrections[0])
    assert corrected[1][0] == np.float32(0.9788520932197571 * 0.5)
    assert corrected[1][1] == corrections[1][1]
    assert applied == [
        {
            "scoring_iteration": 2,
            "stack_index": 79452,
            "half": 2,
            "half_position": 0,
            "factor": float(np.float32(0.9788520932197571)),
        }
    ]


@pytest.mark.parametrize(
    "spec,match",
    [
        ("2:79452", "SCORING_ITER"),
        ("0:79452:1", "positive"),
        ("2:-1:1", "nonnegative"),
        ("2:79452:nan", "finite and positive"),
    ],
)
def test_iteration_normalization_override_rejects_invalid_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_iteration_normalization_factor_overrides([spec])


def test_iteration_normalization_override_rejects_duplicate_boundary():
    with pytest.raises(ValueError, match="duplicate"):
        parse_iteration_normalization_factor_overrides(["2:4:1", "2:4:1.1"])


def test_multi_iter_parity_supports_stop_after_coarse_significance_dump():
    import scripts.run_multi_iter_parity as runner

    source = inspect.getsource(runner)
    assert "RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET" in source
    assert "SignificanceDumpComplete" in source
    assert "coarse-significance dump completed" in source
    assert "pass-2/M-step work" in source


def test_particle_half_indices_preserve_source_order_and_int64_dtype():
    half1, half2 = particle_half_indices(np.asarray([2, 1, 2, 1, 1, 2]))

    np.testing.assert_array_equal(half1, np.asarray([1, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(half2, np.asarray([0, 2, 5], dtype=np.int64))
    assert half1.dtype == np.int64
    assert half2.dtype == np.int64


def test_particle_half_indices_can_reconstruct_fresh_relion_order(monkeypatch):
    from recovar.em.dense_single_volume.helpers import expected_accuracy

    observed = {}

    def fake_orders(subsets, seed, first_iteration, *, optics_group_ids=None):
        observed.update(
            subsets=np.asarray(subsets),
            seed=seed,
            first_iteration=first_iteration,
            optics=np.asarray(optics_group_ids),
        )
        return (
            np.asarray([4, 1, 3], dtype=np.int64),
            np.asarray([5, 2, 0], dtype=np.int64),
        )

    monkeypatch.setattr(expected_accuracy, "relion_auto_refine_half_orders", fake_orders)
    subsets = np.asarray([2, 1, 2, 1, 1, 2])
    optics = np.asarray([1, 2, 1, 2, 1, 1])

    half1, half2 = particle_half_indices(
        subsets,
        fresh_order_seed=1707,
        optics_group_ids=optics,
    )

    np.testing.assert_array_equal(half1, np.asarray([4, 1, 3]))
    np.testing.assert_array_equal(half2, np.asarray([5, 2, 0]))
    np.testing.assert_array_equal(observed["subsets"], subsets)
    np.testing.assert_array_equal(observed["optics"], optics)
    assert observed["seed"] == 1707
    assert observed["first_iteration"] == 1


def test_map_relion_half_orders_to_dataset_rows_uses_image_identity():
    dataset_names = ["3@stack.mrcs", "1@stack.mrcs", "4@stack.mrcs", "2@stack.mrcs"]
    relion_names = ["1@stack.mrcs", "2@stack.mrcs", "3@stack.mrcs", "4@stack.mrcs"]

    half1, half2 = map_relion_half_orders_to_dataset_rows(
        dataset_names,
        relion_names,
        (np.asarray([2, 0]), np.asarray([3, 1])),
    )

    np.testing.assert_array_equal(half1, np.asarray([0, 1]))
    np.testing.assert_array_equal(half2, np.asarray([2, 3]))


def test_significant_count_artifacts_expose_source_image_order():
    save_dict = {}
    half1 = np.asarray([3, 0], dtype=np.int64)
    half2 = np.asarray([1, 2, 4], dtype=np.int64)
    counts_half_order = np.asarray([30, 10, 20, 40, 50], dtype=np.int32)

    add_significant_count_artifacts(
        save_dict,
        [counts_half_order, None],
        [half1, half2],
        n_images=5,
    )

    np.testing.assert_array_equal(save_dict["sig_counts_iter_000"], counts_half_order)
    np.testing.assert_array_equal(
        save_dict["sig_counts_half_order_iter_000"], counts_half_order
    )
    np.testing.assert_array_equal(
        save_dict["sig_counts_by_image_iter_000"],
        np.asarray([10, 20, 40, 30, 50], dtype=np.int32),
    )
    assert "sig_counts_iter_001" not in save_dict


def test_gt_postprocess_command_uses_module_with_pythonpath_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    command = build_gt_postprocess_command(
        recovar_dir=tmp_path / "recovar",
        relion_dir=tmp_path / "relion",
        relion_start_iter=3,
        relion_run_prefix="custom",
        gt_volume=tmp_path / "gt.mrc",
        max_iter=7,
        intermediates_dir=tmp_path / "external" / "intermediates",
    )

    assert command[:3] == [sys.executable, "-m", "scripts.postprocess_multi_iter_gt"]
    assert "scripts/postprocess_multi_iter_gt.py" not in command
    assert command[command.index("--relion_start_iter") + 1] == "3"
    assert command[command.index("--max_iter") + 1] == "7"
    assert command[command.index("--intermediates_dir") + 1] == str(
        tmp_path / "external" / "intermediates"
    )


def test_postprocess_intermediates_dir_defaults_below_output(tmp_path):
    assert resolve_intermediates_dir(tmp_path) == tmp_path / "intermediates"


def test_postprocess_intermediates_dir_honors_explicit_path(tmp_path):
    explicit = tmp_path / "external" / "intermediates"

    assert resolve_intermediates_dir(tmp_path / "output", explicit) == explicit


def test_relion_scheduling_pmax_uses_authoritative_model_scalar():
    model = {"model_general": {"rlnAveragePmax": 0.922993}}

    assert _read_relion_scheduling_average_pmax(model, relion_iteration=8) == pytest.approx(0.922993)


def test_relion_scheduling_pmax_allows_zero_for_missing_iteration_zero_scalar():
    assert _read_relion_scheduling_average_pmax({"model_general": {}}, relion_iteration=0) == 0.0


def test_relion_scheduling_pmax_fails_closed_after_iteration_zero():
    with pytest.raises(ValueError, match="rlnAveragePmax"):
        _read_relion_scheduling_average_pmax({"model_general": {}}, relion_iteration=8)


def test_diff_reports_optimizer_and_particle_pmax_as_distinct_metrics():
    model = {"model_general": {"rlnAveragePmax": 0.922993}}
    scalars = parity_diff.extract_relion_scalars(
        {
            "optimiser": None,
            "model_h1": model,
            "model_h2": model,
            "data": {
                "particles": {
                    "rlnMaxValueProbDistribution": np.asarray([0.9, 0.94572950956]),
                }
            },
        }
    )

    assert scalars["ave_Pmax"] == pytest.approx(0.922993)
    assert scalars["ave_Pmax_particles"] == pytest.approx(0.92286475478)
    assert scalars["ave_Pmax_mstep"] == scalars["ave_Pmax"]


def test_initial_scoring_noise_pair_defaults_to_relion_mpi_restart_broadcast():
    half1 = np.asarray([1.0, 2.0], dtype=np.float32)
    half2 = np.asarray([3.0, 4.0], dtype=np.float32)

    got = initial_scoring_noise_pair(half1, half2, continuous_relion_noise_state=False)

    np.testing.assert_array_equal(got[0], half1)
    np.testing.assert_array_equal(got[1], half1)


def test_initial_scoring_noise_pair_can_preserve_uninterrupted_half_state():
    half1 = np.asarray([1.0, 2.0], dtype=np.float32)
    half2 = np.asarray([3.0, 4.0], dtype=np.float32)

    got = initial_scoring_noise_pair(half1, half2, continuous_relion_noise_state=True)

    np.testing.assert_array_equal(got[0], half1)
    np.testing.assert_array_equal(got[1], half2)


def test_initial_scoring_noise_pair_preserves_binary64_state():
    half1 = np.asarray([0.3, 0.7], dtype=np.float64)
    half2 = np.asarray([1.1, 2.3], dtype=np.float64)

    got = initial_scoring_noise_pair(
        half1,
        half2,
        continuous_relion_noise_state=True,
    )

    assert got[0].dtype == np.float64
    assert got[1].dtype == np.float64
    np.testing.assert_array_equal(got[0], half1)
    np.testing.assert_array_equal(got[1], half2)


def test_final_output_uses_joined_reconstruction_not_average_of_regularized_halves():
    result = {
        "mean": np.asarray([7.0 + 2.0j, 8.0 + 3.0j], dtype=np.complex64),
        "means": [
            np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64),
            np.asarray([3.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex64),
        ],
    }

    half1, half2, merged = final_output_fourier_volumes(result)

    np.testing.assert_array_equal(half1, result["means"][0])
    np.testing.assert_array_equal(half2, result["means"][1])
    np.testing.assert_array_equal(merged, result["mean"])
    assert not np.array_equal(merged, (half1 + half2) / 2.0)


def test_normalized_fsc_auc_excludes_dc_and_uses_canonical_input_range():
    fsc = np.asarray([1.0, 0.2, 0.6, 0.6], dtype=np.float64)

    assert _normalized_fsc_auc(fsc) == pytest.approx(0.5)


def test_final_only_replay_requires_zero_numbered_iterations():
    with pytest.raises(ValueError, match="requires --max_iter 0"):
        validate_final_only_replay_args(
            max_iter=1,
            force_final_after_zero_iterations=True,
            initial_half1_mrc="half1.mrc",
            initial_half2_mrc="half2.mrc",
        )


def test_final_only_replay_promotes_slot_zero_to_explicit_final_override():
    boundary = {"noise_variance": "per-half"}

    assert final_only_replay_override([boundary], enabled=True) is boundary
    assert final_only_replay_override([boundary], enabled=False) is None

    with pytest.raises(ValueError, match="missing its RELION boundary state"):
        final_only_replay_override([None], enabled=True)


def test_final_only_replay_requires_paired_initial_half_maps():
    with pytest.raises(ValueError, match="must be provided together"):
        validate_final_only_replay_args(
            max_iter=0,
            force_final_after_zero_iterations=True,
            initial_half1_mrc="half1.mrc",
            initial_half2_mrc=None,
        )


def test_final_only_replay_requires_paired_initial_fourier_references():
    with pytest.raises(ValueError, match="must be provided together"):
        validate_final_only_replay_args(
            max_iter=0,
            force_final_after_zero_iterations=True,
            initial_half1_mrc=None,
            initial_half2_mrc=None,
            initial_half1_ft_npz="half1.npz",
            initial_half2_ft_npz=None,
        )


def test_initial_noise_override_requires_paired_halves():
    with pytest.raises(ValueError, match="must be provided together"):
        validate_final_only_replay_args(
            max_iter=1,
            force_final_after_zero_iterations=False,
            initial_half1_mrc=None,
            initial_half2_mrc=None,
            initial_noise_half1_npy="half1.npy",
            initial_noise_half2_npy=None,
        )


def test_initial_direction_prior_override_requires_paired_halves():
    with pytest.raises(ValueError, match="must be provided together"):
        validate_final_only_replay_args(
            max_iter=1,
            force_final_after_zero_iterations=False,
            initial_half1_mrc=None,
            initial_half2_mrc=None,
            initial_direction_prior_half1_npy="half1.npy",
            initial_direction_prior_half2_npy=None,
        )


def test_final_only_replay_rejects_mixed_mrc_and_fourier_references():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_final_only_replay_args(
            max_iter=0,
            force_final_after_zero_iterations=True,
            initial_half1_mrc="half1.mrc",
            initial_half2_mrc="half2.mrc",
            initial_half1_ft_npz="half1.npz",
            initial_half2_ft_npz="half2.npz",
        )


def test_fresh_initial_reference_is_confined_to_fresh_runs():
    with pytest.raises(ValueError, match="requires --iter 0"):
        validate_fresh_initial_reference_args(
            fresh_initial_reference_mrc="reference.mrc",
            start_iteration=1,
            initial_half1_mrc=None,
            initial_half1_ft_npz=None,
        )


def test_fresh_initial_reference_rejects_serialized_half_override():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_fresh_initial_reference_args(
            fresh_initial_reference_mrc="reference.mrc",
            start_iteration=0,
            initial_half1_mrc="half1.mrc",
            initial_half1_ft_npz=None,
        )


def test_fresh_particle_order_requires_seed_when_bpref_order_is_preserved():
    with pytest.raises(ValueError, match="requires.*fresh-particle-order-seed"):
        validate_fresh_particle_order_args(
            fresh_particle_order_seed=None,
            preserve_bpref_particle_order=True,
            start_iteration=0,
            initial_half1_mrc=None,
            initial_half2_mrc=None,
            initial_half1_ft_npz=None,
            initial_half2_ft_npz=None,
            final_replay_fields=None,
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"start_iteration": 1}, "requires --iter 0"),
        ({"initial_half1_mrc": "half1.mrc"}, "initial half boundary"),
        ({"initial_half2_ft_npz": "half2.npz"}, "initial half boundary"),
        ({"final_replay_fields": "all"}, "cannot be combined.*final-replay-fields"),
    ],
)
def test_fresh_particle_order_rejects_continuation_and_replay_boundaries(overrides, match):
    kwargs = {
        "fresh_particle_order_seed": 1707,
        "preserve_bpref_particle_order": True,
        "start_iteration": 0,
        "initial_half1_mrc": None,
        "initial_half2_mrc": None,
        "initial_half1_ft_npz": None,
        "initial_half2_ft_npz": None,
        "final_replay_fields": None,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        validate_fresh_particle_order_args(**kwargs)


def test_fresh_particle_order_accepts_unsealed_fresh_k1_boundary():
    validate_fresh_particle_order_args(
        fresh_particle_order_seed=1707,
        preserve_bpref_particle_order=True,
        start_iteration=0,
        initial_half1_mrc=None,
        initial_half2_mrc=None,
        initial_half1_ft_npz=None,
        initial_half2_ft_npz=None,
        final_replay_fields=None,
    )


def test_filter_fresh_initial_reference_preserves_binary64_real_handoff():
    volume = np.ones((8, 8, 8), dtype=np.float32)

    filtered = filter_fresh_initial_reference(
        volume,
        pixel_size=2.0,
        ini_high_angstrom=8.0,
    )

    assert filtered.dtype == np.float64
    np.testing.assert_allclose(filtered, 1.0, rtol=0.0, atol=1e-12)


def test_load_initial_fourier_volume_preserves_complex_dtype_and_values(tmp_path):
    expected = np.arange(8, dtype=np.float64).astype(np.complex128) * (1.0 + 2.0j)
    source = tmp_path / "half.npz"
    np.savez(source, mean_vol_ft=expected)

    actual = load_initial_fourier_volume(source, (2, 2, 2))

    assert actual.dtype == np.complex128
    np.testing.assert_array_equal(actual, expected)


def test_load_initial_fourier_volume_rejects_wrong_size(tmp_path):
    source = tmp_path / "half.npz"
    np.savez(source, mean_vol_ft=np.ones(7, dtype=np.complex64))

    with pytest.raises(ValueError, match="7 elements, expected 8"):
        load_initial_fourier_volume(source, (2, 2, 2))


def test_load_initial_noise_variance_preserves_values(tmp_path):
    source = tmp_path / "noise.npy"
    expected = np.arange(1, 17, dtype=np.float64)
    np.save(source, expected)

    actual = load_initial_noise_variance(source, (4, 4))

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_load_initial_noise_variance_rejects_nonpositive_values(tmp_path):
    source = tmp_path / "noise.npy"
    np.save(source, np.asarray([1.0, 2.0, 0.0, 4.0]))

    with pytest.raises(ValueError, match="finite, real, and positive"):
        load_initial_noise_variance(source, (2, 2))


def test_load_initial_direction_prior_preserves_zeros_and_values(tmp_path):
    source = tmp_path / "direction.npy"
    expected = np.asarray([0.0, 0.25, 0.0, 0.75], dtype=np.float32)
    np.save(source, expected)

    actual = load_initial_direction_prior(source, expected.size)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad", [[0.0, 0.0], [0.5, -0.5], [0.5, np.nan]])
def test_load_initial_direction_prior_rejects_invalid_values(tmp_path, bad):
    source = tmp_path / "direction.npy"
    np.save(source, np.asarray(bad, dtype=np.float32))

    with pytest.raises(ValueError, match="nonnegative|positive total mass"):
        load_initial_direction_prior(source, 2)


def test_stack_index_from_image_name_is_zero_based():
    assert stack_index_from_image_name("1@particles.mrcs") == 0
    assert stack_index_from_image_name("27@particles.mrcs") == 26
    assert stack_index_from_image_name("not_a_relion_name") == -1


def test_map_pose_arrays_to_particle_order_uses_exact_stack_row():
    our_names = [
        "3@particles.mrcs",
        "1@particles.mrcs",
        "2@particles.mrcs",
        "missing_name",
    ]
    gt_rot_all = np.arange(3 * 3 * 3, dtype=np.float64).reshape(3, 3, 3)
    gt_trans_all = np.arange(3 * 2, dtype=np.float64).reshape(3, 2)

    mapped_rot, mapped_trans = map_pose_arrays_to_particle_order(
        our_names,
        gt_rot_all,
        gt_trans_all,
    )

    np.testing.assert_array_equal(mapped_rot[0], gt_rot_all[2])
    np.testing.assert_array_equal(mapped_rot[1], gt_rot_all[0])
    np.testing.assert_array_equal(mapped_rot[2], gt_rot_all[1])
    assert np.isnan(mapped_rot[3]).all()

    np.testing.assert_array_equal(mapped_trans[0], gt_trans_all[2])
    np.testing.assert_array_equal(mapped_trans[1], gt_trans_all[0])
    np.testing.assert_array_equal(mapped_trans[2], gt_trans_all[1])
    assert np.isnan(mapped_trans[3]).all()


def test_map_relion_scale_groups_to_half_order_preserves_full_group_axis():
    group_ids, group_count = map_relion_scale_groups_to_half_order(
        [4, 1, 7, 2],
        {30: 0, 10: 1, 40: 2, 20: 3},
        [20, 30, 10],
    )

    np.testing.assert_array_equal(group_ids, [1, 3, 0])
    assert group_count == 7


def test_map_relion_scale_groups_to_half_order_rejects_invalid_or_missing_rows():
    with pytest.raises(ValueError, match="1-based positive"):
        map_relion_scale_groups_to_half_order([0], {10: 0}, [10])
    with pytest.raises(ValueError, match="missing particle identity"):
        map_relion_scale_groups_to_half_order([1], {10: 0}, [20])


def test_retain_group_scale_update_state_only_omits_terminal_one_step():
    assert not retain_group_scale_update_state(
        max_iter=1,
        skip_final_iteration=True,
    )
    assert retain_group_scale_update_state(
        max_iter=1,
        skip_final_iteration=False,
    )
    assert retain_group_scale_update_state(
        max_iter=2,
        skip_final_iteration=True,
    )


def test_replay_iteration_helpers_split_previous_vs_control_state():
    assert replay_previous_relion_iteration(0, 0) == 0
    assert replay_control_relion_iteration(0, 0) == 1
    assert replay_previous_relion_iteration(1, 0) == 1
    assert replay_control_relion_iteration(1, 0) == 2
    assert replay_previous_relion_iteration(13, 1) == 14
    assert replay_control_relion_iteration(13, 1) == 15


def test_replay_override_pairs_include_last_numbered_state_for_final_all_data():
    pairs = replay_override_iteration_pairs(0, 10)

    assert len(pairs) == 10
    assert pairs[0] == (1, 1, 2)
    assert pairs[-1] == (10, 10, 11)


def test_select_final_replay_override_uses_last_slot_and_requested_groups():
    replay = [
        {"image_corrections": "wrong-slot"},
        {
            "image_corrections": "images",
            "scale_corrections": "scales",
            "noise_variance": "noise",
            "direction_prior": "directions",
            "previous_best_translations": "translations",
            "previous_best_rotations": "rotations",
            "previous_best_rotation_eulers": "eulers",
            "translation_sigma_angstrom": "sigma",
            "translation_sigma_angstrom_per_half": "sigma-pair",
        },
    ]

    selected = select_final_replay_override(replay, "normalization,noise")

    assert selected == {
        "image_corrections": "images",
        "noise_variance": "noise",
        "scale_corrections": "scales",
    }


def test_select_final_replay_override_all_is_complete_union():
    source = {
        "image_corrections": object(),
        "scale_corrections": object(),
        "noise_variance": object(),
        "direction_prior": object(),
        "previous_best_translations": object(),
        "previous_best_rotations": object(),
        "previous_best_rotation_eulers": object(),
        "translation_sigma_angstrom": object(),
        "translation_sigma_angstrom_per_half": object(),
    }

    assert set(select_final_replay_override([source], "all")) == set(source)
    assert set(select_final_replay_override([source], "corrections")) == {
        "image_corrections",
        "scale_corrections",
        "noise_variance",
        "direction_prior",
    }


def test_select_final_replay_override_empty_is_disabled():
    assert select_final_replay_override([], None) is None
    assert select_final_replay_override([], "") is None


def test_select_final_replay_override_rejects_unknown_or_missing_state():
    with pytest.raises(ValueError, match="unknown final replay field"):
        select_final_replay_override([{}], "references")
    with pytest.raises(ValueError, match="missing the last-numbered"):
        select_final_replay_override([None], "poses")
    with pytest.raises(ValueError, match="missing selected fields"):
        select_final_replay_override([{}], "noise")


def test_parse_relion_optimiser_cli_flags_reads_ini_high_and_firstiter_cc():
    parsed = parse_relion_optimiser_cli_flags(
        "# --auto_refine --firstiter_cc --ini_high 30 --ctf --iter 8\n_rlnParticleDiameter 544\n"
    )
    assert parsed["do_firstiter_cc"] is True
    assert parsed["ini_high_angstrom"] == 30.0


def test_parse_relion_optimiser_cli_flags_defaults_when_flag_is_absent():
    parsed = parse_relion_optimiser_cli_flags(
        "# --auto_refine --ini_high 30 --ctf --iter 8\n_rlnParticleDiameter 544\n"
    )
    assert parsed["do_firstiter_cc"] is False
    assert parsed["ini_high_angstrom"] == 30.0


@pytest.mark.parametrize(
    ("mode", "oracle_enabled", "expected"),
    [
        ("auto", True, True),
        ("auto", False, False),
        ("on", False, True),
        ("off", True, False),
    ],
)
def test_resolve_firstiter_cc_mode(mode, oracle_enabled, expected):
    assert resolve_firstiter_cc_mode(mode, oracle_enabled=oracle_enabled, start_iteration=0) is expected


def test_resolve_firstiter_cc_mode_rejects_force_on_after_iter_zero():
    with pytest.raises(ValueError, match="requires --iter 0"):
        resolve_firstiter_cc_mode("on", oracle_enabled=True, start_iteration=1)


def test_resolve_firstiter_cc_mode_disables_auto_after_iter_zero():
    assert resolve_firstiter_cc_mode("auto", oracle_enabled=True, start_iteration=1) is False


def test_resolve_relion_final_oracle_uses_unnumbered_map_after_all_data(tmp_path):
    mode, paths = resolve_relion_final_oracle_paths(
        tmp_path,
        run_prefix="custom",
        start_iteration=0,
        completed_iterations=10,
        final_all_data_ran=True,
    )

    assert mode == "all_data"
    assert paths == {"merged": tmp_path / "custom_class001.mrc"}


def test_resolve_relion_final_oracle_uses_completed_numbered_halves_without_all_data(tmp_path):
    mode, paths = resolve_relion_final_oracle_paths(
        tmp_path,
        run_prefix="custom",
        start_iteration=3,
        completed_iterations=4,
        final_all_data_ran=False,
    )

    assert mode == "split_half"
    assert paths == {
        "half1": tmp_path / "custom_it007_half1_class001.mrc",
        "half2": tmp_path / "custom_it007_half2_class001.mrc",
    }


def test_diff_loader_uses_custom_relion_run_prefix(tmp_path, monkeypatch):
    seen = []

    def fake_parse(path):
        seen.append(path.name)
        return {"model_general": {}}

    monkeypatch.setattr(parity_diff, "parse_relion_optimiser", fake_parse)
    monkeypatch.setattr(parity_diff, "parse_relion_model", fake_parse)
    data_path = tmp_path / "custom_it005_data.star"
    data_path.touch()
    monkeypatch.setattr(parity_diff.starfile, "read", lambda path: {"source": path})

    loaded = parity_diff.load_relion_iter(tmp_path, 5, run_prefix="custom")

    assert seen == [
        "custom_it005_optimiser.star",
        "custom_it005_half1_model.star",
        "custom_it005_half2_model.star",
    ]
    assert loaded["data"] == {"source": str(data_path)}


def test_relion_final_gt_series_accepts_unnumbered_all_data_without_half_maps():
    merged = np.ones(4, dtype=np.complex64)

    series = relion_final_gt_series({"merged": merged}, merged)

    assert set(series) == {"relion_merged"}
    np.testing.assert_array_equal(series["relion_merged"], merged)


def test_case07_native_texture_trajectory_launcher_accepts_pinned_build_overrides():
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_k1_case07_native_texture_trajectory3.sbatch"
    ).read_text()

    assert "CUDA_LIB=${K1_CUDA_LIB:-" in launcher
    assert "RELION_BIND=${K1_RELION_BIND_BUILD_DIR:-" in launcher
    assert "RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE" in launcher
    assert 'provenance/environment_${SLURM_JOB_ID}.txt' in launcher
    assert 'provenance/repo_diff_${SLURM_JOB_ID}.patch' in launcher
    assert "EXPECTED_REPO_DIFF_SHA256" in launcher
    assert 'sha256sum "${CUDA_LIB}"' in launcher
    assert 'sha256sum "${RELION_BIND}"/_relion_bind_core*.so' in launcher
    assert "--numbered-only --allow-incomplete" in launcher
