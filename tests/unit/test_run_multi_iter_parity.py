import numpy as np

from scripts.run_multi_iter_parity import (
    concatenate_half_trajectory_entry,
    map_pose_arrays_to_particle_order,
    parse_relion_optimiser_cli_flags,
    replay_control_relion_iteration,
    replay_previous_relion_iteration,
    save_float32_trajectory_entry,
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


def test_save_float32_trajectory_entry_preserves_uneven_half_arrays():
    save_dict = {}
    half1 = np.ones((2, 3), dtype=np.float64)
    half2 = np.zeros((3, 3), dtype=np.float64)

    save_float32_trajectory_entry(save_dict, "best_rotation_eulers_iter_000", [half1, half2])

    assert save_dict["best_rotation_eulers_iter_000"].dtype == np.float32
    assert save_dict["best_rotation_eulers_iter_000"].shape == (5, 3)
    np.testing.assert_array_equal(save_dict["best_rotation_eulers_iter_000_half1"], half1.astype(np.float32))
    np.testing.assert_array_equal(save_dict["best_rotation_eulers_iter_000_half2"], half2.astype(np.float32))


def test_concatenate_half_trajectory_entry_handles_per_half_history():
    half1 = np.ones((2, 2), dtype=np.float64)
    half2 = np.zeros((3, 2), dtype=np.float64)

    combined = concatenate_half_trajectory_entry([half1, half2], dtype=np.float64)

    assert combined.shape == (5, 2)
    np.testing.assert_array_equal(combined[:2], half1)
    np.testing.assert_array_equal(combined[2:], half2)
