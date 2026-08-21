from __future__ import annotations

import numpy as np
import pytest

from scripts import audit_em_k4_joint_direction_prior as auditor


def test_reports_exact_uniform_joint_prior() -> None:
    joint = np.full((4, 8), 1.0 / 32.0, dtype=np.float32)

    report = auditor.audit_joint_direction_prior(joint)

    assert report["classification"] == (
        "exact_float32_joint_direction_log_split"
    )
    assert report["split_vs_relion_direct_joint_log"]["bitwise_exact"] is True
    assert report["scorecard_change_admissible"] is False
    assert report["causal_claim_admissible"] is False


def test_detects_rounding_from_class_conditional_split() -> None:
    joint = np.asarray(
        [
            [0.11, 0.13, 0.07],
            [0.05, 0.17, 0.08],
            [0.09, 0.06, 0.04],
            [0.03, 0.10, 0.07],
        ],
        dtype=np.float32,
    )
    joint /= np.sum(joint, dtype=np.float64)

    report = auditor.audit_joint_direction_prior(joint)

    assert report["classification"] == (
        "float32_joint_direction_log_split_mismatch"
    )
    assert (
        report["split_vs_relion_direct_joint_log"]["mismatch_count"] > 0
    )
    mismatch_indices = report["split_vs_relion_direct_joint_log"][
        "mismatch_flat_indices"
    ]
    assert len(mismatch_indices) == report[
        "split_vs_relion_direct_joint_log"
    ]["mismatch_count"]
    assert report["split_vs_relion_direct_joint_log"][
        "first_mismatch_flat_index"
    ] == mismatch_indices[0]
    assert (
        report["split_vs_relion_direct_joint_log"][
            "maximum_absolute_delta"
        ]
        > 0.0
    )
    assert report["causal_claim_admissible"] is False


def test_rejects_invalid_joint_prior() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        auditor.audit_joint_direction_prior(
            np.asarray([[0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
        )


def test_audits_live_sparse_capture_exposure() -> None:
    joint = np.asarray(
        [
            [0.11, 0.13, 0.07],
            [0.05, 0.17, 0.08],
            [0.09, 0.06, 0.04],
            [0.03, 0.10, 0.07],
        ],
        dtype=np.float32,
    )
    joint /= np.sum(joint, dtype=np.float64)
    base_report = auditor.audit_joint_direction_prior(joint)
    mismatch_flat = base_report["per_class"][0]["split_vs_direct"][
        "mismatch_flat_indices"
    ]
    assert mismatch_flat
    direction_id = int(mismatch_flat[0])

    row_sums = joint.sum(axis=1, dtype=np.float64)
    conditional = (
        joint / row_sums[:, None].astype(np.float32)
    ).astype(np.float32)
    class_log = np.log(
        (row_sums / float(row_sums.sum())).astype(np.float32)
    ).astype(np.float32)
    split_log = (
        np.log(conditional).astype(np.float32) + class_log[:, None]
    ).astype(np.float32)
    global_parents = np.asarray(
        [direction_id, direction_id + joint.shape[1]], dtype=np.int64
    )
    capture = {
        "class_index": np.asarray(0, dtype=np.int64),
        "oversampled_rot_indices": global_parents * 8,
        "parent_map": np.asarray([0, 1], dtype=np.int32),
        "rotation_log_prior": split_log[0, [direction_id, direction_id]],
        "candidate_mask": np.asarray(
            [[True, False, True], [True, True, False]], dtype=bool
        ),
        "reconstruction_mask": np.asarray(
            [[True, False, False], [False, True, False]], dtype=bool
        ),
        "probs": np.asarray(
            [[0.1, 0.0, 0.2], [0.3, 0.4, 0.0]], dtype=np.float64
        ),
        "reconstruction_probs": np.asarray(
            [[0.1, 0.0, 0.0], [0.0, 0.4, 0.0]], dtype=np.float64
        ),
    }
    raw_diff2 = np.asarray(
        [[500.0, 501.0, 502.0], [503.0, 504.0, 505.0]],
        dtype=np.float32,
    )
    translation_prior = np.asarray([-2.0, -2.5, -3.0], dtype=np.float32)
    rotation_prior = np.asarray(capture["rotation_log_prior"], dtype=np.float32)
    current_score = auditor._relion_score_replay(
        raw_diff2,
        rotation_prior[:, None],
        translation_prior[None, :],
        np.float32(500.0),
    )
    capture.update(
        {
            "relion_raw_diff2": raw_diff2,
            "relion_min_diff2": np.asarray(500.0, dtype=np.float32),
            "translation_log_prior": translation_prior,
            "scores_with_prior": current_score,
        }
    )

    report = auditor.audit_prior_capture_exposure(
        joint,
        capture,
        fine_children_per_parent=8,
    )

    assert report["classification"] == "live_split_prior_mismatch"
    assert report["local_parent_mapping_exact"] is True
    assert report["captured_prior_matches_split_all_rows"] is True
    assert report["captured_prior_direct_mismatch_rows"] == 2
    assert report["candidate_active_mismatch_pair_count"] == 4
    assert report["reconstruction_significant_mismatch_pair_count"] == 2
    assert report["posterior_mass_on_mismatch_rows"] == pytest.approx(1.0)
    assert report[
        "reconstruction_posterior_mass_on_mismatch_rows"
    ] == pytest.approx(0.5)
    assert report["score_replay"]["saved_score_matches_current_replay"][
        "bitwise_exact"
    ] is True
    assert report["score_replay"]["classification"] == (
        "direct_joint_prior_inert_at_float32_combined_score"
    )
    assert report["score_replay"]["maximum_tie_sets_exact"] is True
    assert report["causal_claim_admissible"] is False


def test_capture_exposure_rejects_inverted_local_parent_map() -> None:
    joint = np.full((2, 3), np.float32(1.0 / 6.0), dtype=np.float32)
    capture = {
        "class_index": np.asarray(0, dtype=np.int64),
        "oversampled_rot_indices": np.asarray([0, 8], dtype=np.int64),
        "parent_map": np.asarray([1, 0], dtype=np.int32),
        "rotation_log_prior": np.log(joint[0, :2]).astype(np.float32),
        "candidate_mask": np.ones((2, 1), dtype=bool),
    }

    with pytest.raises(ValueError, match="parent_map"):
        auditor.audit_prior_capture_exposure(
            joint,
            capture,
            fine_children_per_parent=8,
        )
