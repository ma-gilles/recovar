import numpy as np

from scripts.run_multi_iter_parity import (
    map_pose_arrays_to_particle_order,
    parse_relion_optimiser_cli_flags,
    replay_control_relion_iteration,
    replay_previous_relion_iteration,
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
