import numpy as np
import pytest

from scripts.run_multi_iter_parity import (
    map_pose_arrays_to_particle_order,
    parse_relion_optimiser_cli_flags,
    relion_final_gt_series,
    replay_control_relion_iteration,
    replay_override_iteration_pairs,
    replay_previous_relion_iteration,
    resolve_firstiter_cc_mode,
    resolve_relion_final_oracle_paths,
    stack_index_from_image_name,
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
        start_iteration=0,
        completed_iterations=10,
        final_all_data_ran=True,
    )

    assert mode == "all_data"
    assert paths == {"merged": tmp_path / "run_class001.mrc"}


def test_resolve_relion_final_oracle_uses_completed_numbered_halves_without_all_data(tmp_path):
    mode, paths = resolve_relion_final_oracle_paths(
        tmp_path,
        start_iteration=3,
        completed_iterations=4,
        final_all_data_ran=False,
    )

    assert mode == "split_half"
    assert paths == {
        "half1": tmp_path / "run_it007_half1_class001.mrc",
        "half2": tmp_path / "run_it007_half2_class001.mrc",
    }


def test_relion_final_gt_series_accepts_unnumbered_all_data_without_half_maps():
    merged = np.ones(4, dtype=np.complex64)

    series = relion_final_gt_series({"merged": merged}, merged)

    assert set(series) == {"relion_merged"}
    np.testing.assert_array_equal(series["relion_merged"], merged)
