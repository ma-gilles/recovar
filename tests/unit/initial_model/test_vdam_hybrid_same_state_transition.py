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


def test_same_state_runner_seals_abba_and_exact_snapshot_contract() -> None:
    source = SCRIPT.read_text()
    sbatch = RUNNER.read_text()

    assert runner.ARM_ORDER == ("direct_1", "hybrid_1", "hybrid_2", "direct_2")
    assert "copy.deepcopy(checkpoint[\"result\"].state)" in source
    assert "copy.deepcopy(checkpoint[\"particle_state\"])" in source
    assert "copy.deepcopy(checkpoint[\"sampling_state\"])" in source
    assert "did not start from the exact shared" in source
    assert "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS" in source
    assert "--constraint=h100" in sbatch
    assert "--checkpoint-iteration \"${CHECKPOINT_ITERATION}\"" in sbatch
    assert "direct_1,hybrid_1,hybrid_2,direct_2" in sbatch
    assert "status --porcelain=v1 --untracked-files=all" in sbatch
