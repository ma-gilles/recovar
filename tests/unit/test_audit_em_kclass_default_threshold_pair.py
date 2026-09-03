"""Focused guards for the reusable K-class threshold-pair audit."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import aggregate_em_kclass_default_threshold_pairs as aggregate
from scripts import audit_em_kclass_default_threshold_pair as audit

pytestmark = pytest.mark.unit


def _saved_state(*, n_iterations: int = 2) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {
        "current_sizes": np.asarray([64, 72], dtype=np.int32),
        "pixel_resolutions": np.asarray([3.0, 2.5], dtype=np.float32),
        "healpix_order_trajectory": np.asarray([1, 1], dtype=np.int32),
        "ave_Pmax_trajectory": np.asarray([0.2, 0.3], dtype=np.float32),
    }
    for iteration in range(n_iterations):
        suffix = f"{iteration:03d}"
        state[f"class_assignments_by_image_iter_{suffix}"] = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        state[f"sig_counts_by_image_iter_{suffix}"] = np.arange(8, dtype=np.int32)
        state[f"pmax_per_image_by_image_iter_{suffix}"] = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        state[f"best_rotation_eulers_by_image_iter_{suffix}"] = np.zeros((8, 3), dtype=np.float32)
        state[f"best_translations_by_image_iter_{suffix}"] = np.zeros((8, 2), dtype=np.float32)
    return state


def test_default_fft_workers_honors_slurm_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert audit.default_fft_workers() == 6

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "invalid")
    assert audit.default_fft_workers() == 1

    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert audit.default_fft_workers() == 1


def test_saved_state_and_occupancy_comparison_detects_pose_and_collapse():
    baseline = _saved_state()
    candidate = {key: value.copy() for key, value in baseline.items()}

    exact = audit.compare_saved_state_arrays(baseline, candidate, n_iterations=2)
    occupancy = audit.class_occupancy_rows(baseline, candidate, n_iterations=2, n_classes=4)
    exact_summary = audit.summarize_pair(
        exact,
        occupancy,
        [{"signed_non_dc_fsc_auc": 1.0, "relative_l2": 0.0}],
    )
    assert exact_summary["controller_exact"]
    assert exact_summary["assignment_mismatch_count"] == 0
    assert exact_summary["poses_exact"]
    assert exact_summary["no_class_collapse"]

    candidate["best_translations_by_image_iter_001"][0, 0] = 1.0
    candidate["class_assignments_by_image_iter_001"][:] = 0
    changed = audit.compare_saved_state_arrays(baseline, candidate, n_iterations=2)
    changed_occupancy = audit.class_occupancy_rows(baseline, candidate, n_iterations=2, n_classes=4)
    changed_summary = audit.summarize_pair(
        changed,
        changed_occupancy,
        [{"signed_non_dc_fsc_auc": 1.0, "relative_l2": 0.0}],
    )
    assert changed_summary["assignment_mismatch_count"] == 6
    assert not changed_summary["poses_exact"]
    assert not changed_summary["no_class_collapse"]


def test_euler_audit_distinguishes_representation_changes_from_physical_pose_changes():
    # At zero tilt, redistributing angle between the two ZXZ rotations changes
    # the RELION Euler row but not the physical rotation.
    baseline = np.asarray([[10.0, 0.0, 20.0]], dtype=np.float32)
    equivalent = np.asarray([[15.0, 0.0, 15.0]], dtype=np.float32)
    distinct = np.asarray([[13.690024, 161.02734, 157.18436]], dtype=np.float32)
    observed_seed42002 = np.asarray([[-142.99463, 150.59065, 2.5717983]], dtype=np.float32)

    assert not np.array_equal(baseline, equivalent)
    assert audit.maximum_physical_rotation_delta_deg(baseline, equivalent) < 1e-6
    assert audit.maximum_physical_rotation_delta_deg(observed_seed42002, distinct) == pytest.approx(
        47.374945194620494,
        abs=1e-6,
    )


def test_scientific_acceptance_cannot_rewrite_formal_rejection():
    summary = {
        "controller_exact": True,
        "assignment_mismatch_count": 0,
        "poses_exact": True,
        "no_class_collapse": True,
        "min_final_map_signed_non_dc_fsc_auc": 0.9999999,
        "max_final_map_relative_l2": 3.2e-6,
    }
    performance = {
        "sampled_hbm_ratio": 1.0,
        "sparse_group_wall_ratio": 0.85,
    }
    formal = {
        "final_map_min_signed_non_dc_fsc_auc": 0.99999999,
        "final_map_max_relative_l2": 1.02e-5,
        "max_sampled_hbm_ratio": 1.05,
        "max_sparse_group_wall_ratio": 0.95,
    }
    science = {
        "final_map_min_signed_non_dc_fsc_auc": 0.999999,
        "final_map_max_relative_l2": 1.0e-4,
        "max_sampled_hbm_ratio": 1.05,
        "max_sparse_group_wall_ratio": 0.95,
    }

    tiers = audit.evaluate_tiers(summary, performance, formal, science)

    assert tiers["formal"]["decision"] == "reject"
    assert tiers["formal"]["gates"]["map_fsc"] is False
    assert tiers["scientific_equivalence"]["decision"] == "accept"
    assert all(tiers["scientific_equivalence"]["gates"].values())


def test_parse_log_requires_complete_trajectory_and_sparse_measurement(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "RELION Iteration 1: current_size=64, time=10.0s",
                "Sparse fused K-class pass-2 bucket group done: wall=3.5s",
                "RELION Iteration 2: current_size=72, time=12.0s",
                "Sparse fused K-class pass-2 bucket group done: wall=4.5s",
            ]
        )
    )
    parsed = audit.parse_log(log_path, n_iterations=2)
    assert parsed["iteration_time_sum_s"] == 22.0
    assert parsed["sparse_group_count"] == 2
    assert parsed["sparse_group_wall_sum_s"] == 8.0

    with pytest.raises(RuntimeError, match="expected 3 completed iterations"):
        audit.parse_log(log_path, n_iterations=3)


def test_pair_aggregate_is_a_conjunction_without_cross_seed_averaging():
    pairs = [
        {
            "label": "seed42001",
            "tiers": {
                "formal": {"decision": "reject"},
                "scientific_equivalence": {"decision": "accept"},
            },
        },
        {
            "label": "seed42002",
            "tiers": {
                "formal": {"decision": "accept"},
                "scientific_equivalence": {"decision": "accept"},
            },
        },
        {
            "label": "seed42003",
            "tiers": {
                "formal": {"decision": "accept"},
                "scientific_equivalence": {"decision": "reject"},
            },
        },
    ]

    result = aggregate.aggregate_pairs(pairs)

    assert result == {
        "rule": "all-pair conjunction; no averaging",
        "pair_count": 3,
        "formal_accept_pair_count": 2,
        "formal_decision": "reject",
        "scientific_equivalence_accept_pair_count": 2,
        "scientific_equivalence_decision": "reject",
    }


def test_pair_aggregate_rejects_duplicate_labels():
    pair = {
        "label": "seed42001",
        "tiers": {
            "formal": {"decision": "accept"},
            "scientific_equivalence": {"decision": "accept"},
        },
    }
    with pytest.raises(RuntimeError, match="duplicate pair labels"):
        aggregate.aggregate_pairs([pair, pair])
