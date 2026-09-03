from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/run_vdam_hybrid_same_state_transition.py"
RUNNER = ROOT / "scripts/run_vdam_hybrid_same_state_transition.sbatch"
SPEC = importlib.util.spec_from_file_location("run_vdam_hybrid_same_state_transition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@dataclasses.dataclass
class TinyState:
    count: int
    values: np.ndarray
    optional: np.ndarray | None = None


def test_same_state_array_comparison_reports_exact_and_particle_mismatches() -> None:
    left = {
        "selected_particle_ids": np.asarray([91, 143, 181], dtype=np.int64),
        "best_pose_rotation_ids": np.asarray([4, 7, 11], dtype=np.int32),
        "pose_assignments": np.asarray([40, 70, 110], dtype=np.int32),
        "significant_counts": np.asarray([2, 3, 4], dtype=np.int32),
    }
    right = {
        **left,
        "best_pose_rotation_ids": np.asarray([4, 8, 12], dtype=np.int32),
        "pose_assignments": np.asarray([40, 80, 120], dtype=np.int32),
    }

    report = runner._meta_comparison(left, right)

    assert report["best_pose_rotation_ids"]["mismatch_count"] == 2
    assert report["best_pose_rotation_ids"]["mismatching_selected_particle_ids"] == [143, 181]
    assert report["pose_assignments"]["mismatching_selected_particle_ids"] == [143, 181]
    assert report["significant_counts"]["exact_equal"] is True


def test_same_state_support_audit_comparison_finds_nested_profile_audits() -> None:
    audit = {
        "schema": "recovar.coarse_significance_support_audit.v2",
        "aggregate_support_sha256": "abc123",
        "n_images": 2,
        "per_class_image_selected_counts": [[2, 1]],
    }
    left = {"halfset_0_profile_summary": {"coarse_significance_support_audit": audit}}
    right = {"halfset_0_profile_summary": {"coarse_significance_support_audit": dict(audit)}}

    comparison = runner._support_audit_comparison(left, right)

    key = "halfset_0_profile_summary.coarse_significance_support_audit"
    assert comparison["exact_equal"] is True
    assert comparison["left_count"] == comparison["right_count"] == 1
    assert comparison["entries"][key]["left_aggregate_support_sha256"] == "abc123"
    assert comparison["entries"][key]["left_sha256"] == comparison["entries"][key]["right_sha256"]


def test_same_state_support_audit_comparison_rejects_missing_or_changed_audit() -> None:
    left = {
        "halfset_0_profile_summary": {
            "coarse_significance_support_audit": {"aggregate_support_sha256": "left"}
        }
    }
    changed = {
        "halfset_0_profile_summary": {
            "coarse_significance_support_audit": {"aggregate_support_sha256": "right"}
        }
    }

    assert runner._support_audit_comparison(left, changed)["exact_equal"] is False
    missing = runner._support_audit_comparison(left, {})
    assert missing["exact_equal"] is False
    assert missing["left_count"] == 1
    assert missing["right_count"] == 0


def test_same_state_dataclass_manifest_is_value_stable_and_sensitive() -> None:
    left = TinyState(3, np.asarray([1.0, 2.0], dtype=np.float32))
    same = TinyState(3, np.asarray([1.0, 2.0], dtype=np.float32))
    changed = TinyState(3, np.asarray([1.0, 2.5], dtype=np.float32))

    assert runner._dataclass_manifest(left) == runner._dataclass_manifest(same)
    assert (
        runner._dataclass_manifest(left)["manifest_sha256"]
        != runner._dataclass_manifest(changed)["manifest_sha256"]
    )
    comparison = runner._dataclass_comparison(left, changed)
    assert comparison["count"]["exact_equal"] is True
    assert comparison["values"]["mismatch_count"] == 1
    assert comparison["values"]["max_abs_delta"] == 0.5


def test_same_state_environment_is_restored_after_each_arm(monkeypatch) -> None:
    monkeypatch.setenv("RECOVAR_TEST_SAME_STATE_EXISTING", "before")
    monkeypatch.delenv("RECOVAR_TEST_SAME_STATE_MISSING", raising=False)

    with runner._temporary_environment(
        {
            "RECOVAR_TEST_SAME_STATE_EXISTING": "during",
            "RECOVAR_TEST_SAME_STATE_MISSING": "new",
        }
    ):
        assert runner.os.environ["RECOVAR_TEST_SAME_STATE_EXISTING"] == "during"
        assert runner.os.environ["RECOVAR_TEST_SAME_STATE_MISSING"] == "new"

    assert runner.os.environ["RECOVAR_TEST_SAME_STATE_EXISTING"] == "before"
    assert "RECOVAR_TEST_SAME_STATE_MISSING" not in runner.os.environ


def test_same_state_candidate_modes_keep_control_and_candidate_scoped() -> None:
    assert runner._arm_order("hybrid") == runner.ARM_ORDER
    assert runner._arm_order("flat_rows") == runner.FLAT_ROW_ARM_ORDER
    assert runner._arm_order("packed_projection") == runner.PACKED_PROJECTION_ARM_ORDER
    assert runner._arm_order("packed_deferred") == runner.PACKED_DEFERRED_ARM_ORDER
    assert (
        runner._arm_order("hybrid_packed_deferred")
        == runner.HYBRID_PACKED_DEFERRED_ARM_ORDER
    )

    control = runner._candidate_environment("flat_rows", enabled=False)
    flat_rows = runner._candidate_environment("flat_rows", enabled=True)
    packed_projection = runner._candidate_environment(
        "packed_projection", enabled=True
    )
    packed_deferred = runner._candidate_environment(
        "packed_deferred", enabled=True
    )
    hybrid_packed_deferred = runner._candidate_environment(
        "hybrid_packed_deferred", enabled=True
    )
    hybrid = runner._candidate_environment("hybrid", enabled=True)

    assert all(control[name] == "0" for name in runner.HYBRID_ENVIRONMENT)
    assert control[runner.FLAT_ROW_ENVIRONMENT] == "0"
    assert control[runner.PACKED_PROJECTION_ENVIRONMENT] == "0"
    assert control[runner.PACKED_DEFERRED_ENVIRONMENT] == "0"
    assert all(flat_rows[name] == "0" for name in runner.HYBRID_ENVIRONMENT)
    assert flat_rows[runner.FLAT_ROW_ENVIRONMENT] == "1"
    assert flat_rows[runner.PACKED_PROJECTION_ENVIRONMENT] == "0"
    assert flat_rows[runner.PACKED_DEFERRED_ENVIRONMENT] == "0"
    assert all(
        packed_projection[name] == "0" for name in runner.HYBRID_ENVIRONMENT
    )
    assert packed_projection[runner.FLAT_ROW_ENVIRONMENT] == "1"
    assert packed_projection[runner.PACKED_PROJECTION_ENVIRONMENT] == "1"
    assert packed_projection[runner.PACKED_DEFERRED_ENVIRONMENT] == "0"
    assert all(
        packed_deferred[name] == "0" for name in runner.HYBRID_ENVIRONMENT
    )
    assert packed_deferred[runner.FLAT_ROW_ENVIRONMENT] == "1"
    assert packed_deferred[runner.PACKED_PROJECTION_ENVIRONMENT] == "1"
    assert packed_deferred[runner.PACKED_DEFERRED_ENVIRONMENT] == "1"
    assert all(
        hybrid_packed_deferred[name] == "1"
        for name in runner.HYBRID_ENVIRONMENT
    )
    assert hybrid_packed_deferred[runner.FLAT_ROW_ENVIRONMENT] == "1"
    assert hybrid_packed_deferred[runner.PACKED_PROJECTION_ENVIRONMENT] == "1"
    assert hybrid_packed_deferred[runner.PACKED_DEFERRED_ENVIRONMENT] == "1"
    assert all(hybrid[name] == "1" for name in runner.HYBRID_ENVIRONMENT)
    assert hybrid[runner.FLAT_ROW_ENVIRONMENT] == "0"
    assert hybrid[runner.PACKED_PROJECTION_ENVIRONMENT] == "0"
    assert hybrid[runner.PACKED_DEFERRED_ENVIRONMENT] == "0"


def test_same_state_runner_seals_abba_and_exact_snapshot_contract() -> None:
    source = SCRIPT.read_text()
    sbatch = RUNNER.read_text()

    assert runner.ARM_ORDER == ("direct_1", "hybrid_1", "hybrid_2", "direct_2")
    assert "copy.deepcopy(checkpoint[\"result\"].state)" in source
    assert "copy.deepcopy(checkpoint[\"particle_state\"])" in source
    assert "copy.deepcopy(checkpoint[\"sampling_state\"])" in source
    assert "did not start from the exact shared" in source
    assert "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS" in source
    assert "--candidate-mode" in source
    assert "--constraint=h100" in sbatch
    assert "--checkpoint-iteration \"${CHECKPOINT_ITERATION}\"" in sbatch
    assert "direct_1,hybrid_1,hybrid_2,direct_2" in sbatch
    assert "direct_1,flat_rows_1,flat_rows_2,direct_2" in sbatch
    assert (
        "direct_1,packed_projection_1,packed_projection_2,direct_2" in sbatch
    )
    assert "direct_1,packed_deferred_1,packed_deferred_2,direct_2" in sbatch
    assert (
        "direct_1,hybrid_packed_deferred_1,hybrid_packed_deferred_2,direct_2"
        in sbatch
    )
    assert "make -B -C \"${REPO_ROOT}/recovar/cuda\"" in sbatch
    assert "status --porcelain=v1 --untracked-files=all" in sbatch
    assert "VDAM_SAME_STATE_NOISE_SPLIT_DIAGNOSTICS" in sbatch
    assert "RECOVAR_NOISE_DEBUG_DUMP_DIR=${RUNTIME}/noise_split_enabled" in sbatch
