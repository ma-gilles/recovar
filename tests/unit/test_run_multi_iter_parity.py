import numpy as np

from scripts.run_multi_iter_parity import (
    map_pose_arrays_to_particle_order,
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
