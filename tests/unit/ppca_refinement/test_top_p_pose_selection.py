import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.ppca_refinement.config import PoseSelectionConfig
from recovar.em.ppca_refinement.pose_selection import (
    merge_top_p_pose_scores,
    pack_pose_ids,
    select_distinct_top_poses,
    top_p_from_score_block,
)

pytestmark = pytest.mark.unit


def test_top_p_from_score_block_matches_numpy_sort_and_p1_best():
    score = jnp.asarray(
        [
            [[0.0, 2.0, 1.0], [4.0, 3.0, -1.0]],
            [[5.0, 0.0, 2.0], [1.0, 6.0, 4.0]],
        ],
        dtype=jnp.float32,
    )
    top_scores, top_rot, top_trans = top_p_from_score_block(score, rotation_offset=10, candidate_count=3)
    flat = np.asarray(score).reshape(2, -1)
    expected_order = np.argsort(flat, axis=1)[:, ::-1][:, :3]
    np.testing.assert_allclose(np.asarray(top_scores), np.take_along_axis(flat, expected_order, axis=1))
    np.testing.assert_array_equal(np.asarray(top_rot), expected_order % 3 + 10)
    np.testing.assert_array_equal(np.asarray(top_trans), expected_order // 3)

    p1_scores, p1_rot, p1_trans = top_p_from_score_block(score, rotation_offset=0, candidate_count=1)
    np.testing.assert_allclose(np.asarray(p1_scores[:, 0]), np.max(flat, axis=1))
    np.testing.assert_array_equal(np.asarray(p1_rot[:, 0]), np.argmax(flat, axis=1) % 3)
    np.testing.assert_array_equal(np.asarray(p1_trans[:, 0]), np.argmax(flat, axis=1) // 3)


def test_distinct_top_p_filters_near_duplicates_and_pads():
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy()
    rotations[2] = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    translations = np.asarray([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0]], dtype=np.float32)
    selection = select_distinct_top_poses(
        np.asarray([[10.0, 9.5, 8.0]], dtype=np.float32),
        np.asarray([[0, 1, 2]], dtype=np.int32),
        np.asarray([[0, 1, 2]], dtype=np.int32),
        logZ=np.asarray([11.0]),
        rotations=rotations,
        translations=translations,
        config=PoseSelectionConfig(
            top_p_poses=3,
            top_pose_max_log_score_gap=3.0,
            top_pose_min_angle_deg=1.0,
            top_pose_min_translation_px=0.5,
        ),
    )
    np.testing.assert_array_equal(selection.rotation_idx, np.asarray([[0, 2, -1]], dtype=np.int32))
    np.testing.assert_array_equal(selection.translation_idx, np.asarray([[0, 2, -1]], dtype=np.int32))
    assert np.isneginf(selection.log_score[0, 2])
    assert selection.posterior[0, 2] == 0.0


def test_pack_pose_ids_uses_kclass_convention():
    packed = pack_pose_ids(np.asarray([[2, 4]]), np.asarray([[1, 3]]), n_translations=5)
    np.testing.assert_array_equal(packed, np.asarray([[11, 23]]))


def test_merge_top_p_pose_scores_matches_full_grid_sort():
    score = np.asarray(
        [
            [[0.0, 1.0, 3.0, 2.0], [5.0, 4.0, -1.0, 7.0]],
            [[9.0, 8.0, 1.0, 0.0], [3.0, 4.0, 6.0, 5.0]],
        ],
        dtype=np.float32,
    )
    block0 = top_p_from_score_block(jnp.asarray(score[:, :, :2]), rotation_offset=0, candidate_count=4)
    block1 = top_p_from_score_block(jnp.asarray(score[:, :, 2:]), rotation_offset=2, candidate_count=4)
    logZ = np.log(np.sum(np.exp(score.reshape(2, -1)), axis=1))
    merged = merge_top_p_pose_scores(
        [block0[0], block1[0]],
        [block0[1], block1[1]],
        [block0[2], block1[2]],
        logZ,
        config=PoseSelectionConfig(top_p_poses=3, top_pose_max_log_score_gap=np.inf),
    )
    flat = score.reshape(2, -1)
    order = np.argsort(flat, axis=1)[:, ::-1][:, :3]
    np.testing.assert_allclose(merged.log_score, np.take_along_axis(flat, order, axis=1))
    np.testing.assert_array_equal(merged.rotation_idx, order % 4)
    np.testing.assert_array_equal(merged.translation_idx, order // 4)
    np.testing.assert_allclose(merged.posterior, np.exp(merged.log_score - logZ[:, None]), rtol=1e-6)
