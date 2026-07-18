import sys

import numpy as np
import pytest

from scripts import diff_relion_recovar_per_iter as parity_diff
from scripts.run_multi_iter_parity import (
    _normalized_fsc_auc,
    _read_relion_scheduling_average_pmax,
    build_gt_postprocess_command,
    final_output_fourier_volumes,
    initial_scoring_noise_pair,
    map_pose_arrays_to_particle_order,
    parse_relion_optimiser_cli_flags,
    relion_final_gt_series,
    replay_control_relion_iteration,
    replay_override_iteration_pairs,
    replay_previous_relion_iteration,
    resolve_firstiter_cc_mode,
    resolve_relion_final_oracle_paths,
    stack_index_from_image_name,
    validate_final_only_replay_args,
)


def test_gt_postprocess_command_uses_module_with_pythonpath_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    command = build_gt_postprocess_command(
        recovar_dir=tmp_path / "recovar",
        relion_dir=tmp_path / "relion",
        relion_start_iter=3,
        relion_run_prefix="custom",
        gt_volume=tmp_path / "gt.mrc",
        max_iter=7,
    )

    assert command[:3] == [sys.executable, "-m", "scripts.postprocess_multi_iter_gt"]
    assert "scripts/postprocess_multi_iter_gt.py" not in command
    assert command[command.index("--relion_start_iter") + 1] == "3"
    assert command[command.index("--max_iter") + 1] == "7"


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


def test_final_only_replay_requires_paired_initial_half_maps():
    with pytest.raises(ValueError, match="must be provided together"):
        validate_final_only_replay_args(
            max_iter=0,
            force_final_after_zero_iterations=True,
            initial_half1_mrc="half1.mrc",
            initial_half2_mrc=None,
        )


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
