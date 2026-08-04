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
