from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

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
    assert runner._arm_order("stable_shapes") == runner.STABLE_SHAPES_ARM_ORDER
    assert (
        runner._arm_order("stable_flat_capacity")
        == runner.STABLE_FLAT_CAPACITY_ARM_ORDER
    )
    assert (
        runner._arm_order("compact_posterior")
        == runner.COMPACT_POSTERIOR_ARM_ORDER
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
    stable_off = runner._candidate_environment("stable_shapes", enabled=False)
    stable_on = runner._candidate_environment("stable_shapes", enabled=True)
    stable_flat_off = runner._candidate_environment(
        "stable_flat_capacity", enabled=False
    )
    stable_flat_on = runner._candidate_environment(
        "stable_flat_capacity", enabled=True
    )
    compact_posterior = runner._candidate_environment(
        "compact_posterior",
        enabled=True,
    )

    assert all(control[name] == "0" for name in runner.HYBRID_ENVIRONMENT)
    assert control[runner.COMPACT_POSTERIOR_ENVIRONMENT] == "0"
    assert control[runner.FLAT_ROW_ENVIRONMENT] == "0"
    assert control[runner.STABLE_FLAT_CAPACITY_ENVIRONMENT] == "0"
    assert control[runner.PACKED_PROJECTION_ENVIRONMENT] == "0"
    assert control[runner.PACKED_DEFERRED_ENVIRONMENT] == "0"
    assert all(flat_rows[name] == "0" for name in runner.HYBRID_ENVIRONMENT)
    assert flat_rows[runner.COMPACT_POSTERIOR_ENVIRONMENT] == "0"
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
    assert hybrid[runner.COMPACT_POSTERIOR_ENVIRONMENT] == "0"
    assert hybrid[runner.FLAT_ROW_ENVIRONMENT] == "0"
    assert hybrid[runner.PACKED_PROJECTION_ENVIRONMENT] == "0"
    assert hybrid[runner.PACKED_DEFERRED_ENVIRONMENT] == "0"
    assert stable_off == stable_on
    assert all(stable_off[name] == "1" for name in runner.HYBRID_ENVIRONMENT)
    assert stable_off[runner.FLAT_ROW_ENVIRONMENT] == "1"
    assert stable_off[runner.PACKED_PROJECTION_ENVIRONMENT] == "1"
    assert stable_off[runner.PACKED_DEFERRED_ENVIRONMENT] == "1"
    assert stable_off[runner.STABLE_FLAT_CAPACITY_ENVIRONMENT] == "0"
    assert stable_flat_off[runner.STABLE_FLAT_CAPACITY_ENVIRONMENT] == "0"
    assert stable_flat_on[runner.STABLE_FLAT_CAPACITY_ENVIRONMENT] == "1"
    assert all(
        stable_flat_off[name] == stable_flat_on[name] == "1"
        for name in runner.HYBRID_ENVIRONMENT
    )
    assert stable_flat_off[runner.FLAT_ROW_ENVIRONMENT] == "1"
    assert stable_flat_on[runner.PACKED_PROJECTION_ENVIRONMENT] == "1"
    assert stable_flat_on[runner.PACKED_DEFERRED_ENVIRONMENT] == "1"
    assert runner._candidate_uses_hybrid("hybrid") is True
    assert runner._candidate_uses_hybrid("hybrid_packed_deferred") is True
    assert runner._candidate_uses_hybrid("stable_shapes") is True
    assert runner._candidate_uses_hybrid("stable_flat_capacity") is True
    assert all(
        compact_posterior[name] == "1" for name in runner.HYBRID_ENVIRONMENT
    )
    assert compact_posterior[runner.COMPACT_POSTERIOR_ENVIRONMENT] == "1"
    assert compact_posterior[runner.FLAT_ROW_ENVIRONMENT] == "0"
    assert compact_posterior[runner.PACKED_PROJECTION_ENVIRONMENT] == "0"
    assert compact_posterior[runner.PACKED_DEFERRED_ENVIRONMENT] == "0"
    assert runner._candidate_uses_hybrid("compact_posterior") is True
    assert runner._candidate_uses_hybrid("packed_deferred") is False
    assert runner._arm_candidate_enabled("stable_off_1", "stable_shapes") is False
    assert runner._arm_candidate_enabled("stable_on_1", "stable_shapes") is True
    assert (
        runner._arm_candidate_enabled(
            "stable_flat_off_1", "stable_flat_capacity"
        )
        is False
    )
    assert (
        runner._arm_candidate_enabled(
            "stable_flat_on_1", "stable_flat_capacity"
        )
        is True
    )


def test_same_state_stable_flat_capacity_profiles_prove_engine_execution() -> None:
    packed = {
        "halfset_0_profile_summary": {
            "flat_local_rows_enabled": True,
            "stable_flat_row_capacity_enabled": False,
            "chunk_flat_score_rows": [4752, 4752],
            "chunk_padded_rotations": [13824, 13824],
        }
    }
    stable = {
        "halfset_0_profile_summary": {
            "flat_local_rows_enabled": True,
            "stable_flat_row_capacity_enabled": True,
            "chunk_flat_score_rows": [13824, 13824],
            "chunk_padded_rotations": [13824, 13824],
        }
    }

    packed_contract = runner._validate_stable_flat_capacity_profiles(
        packed, enabled=False, label="off"
    )
    stable_contract = runner._validate_stable_flat_capacity_profiles(
        stable, enabled=True, label="on"
    )

    assert packed_contract["strict_reduction_count"] == 2
    assert stable_contract["strict_reduction_count"] == 0


def test_same_state_stable_flat_capacity_profiles_fail_closed() -> None:
    ignored = {
        "halfset_0_profile_summary": {
            "flat_local_rows_enabled": True,
            "stable_flat_row_capacity_enabled": False,
            "chunk_flat_score_rows": [4752],
            "chunk_padded_rotations": [13824],
        }
    }

    with np.testing.assert_raises_regex(RuntimeError, "reported"):
        runner._validate_stable_flat_capacity_profiles(
            ignored, enabled=True, label="on"
        )


def _compact_profile(**updates):
    profile = {
        "enabled": True,
        "default_enabled": False,
        "compact_posterior_enabled": True,
        "compact_posterior_default_enabled": False,
        "selected_score_layout": "fixed_capacity_source16",
        "positive_only_scan_role": "correctness_oracle_not_runtime",
        "published_score_source": "exact_relion_source16_or_full_rectangular",
        "expanded_gemm_scores_published": False,
        "whole_batch_fail_closed_fallback": True,
        "batch_count": 2,
        "selected_rescore_batch_count": 2,
        "fallback_batch_count": 0,
        "selected_rescore_image_count": 300,
        "fallback_image_count": 0,
        "selected_source16_block_count": 500,
        "selected_exact_candidate_count": 232_000,
        "selected_score_table_capacity_candidates": 5_939_200,
        "dense_global_score_table_capacity_candidates": 213_811_200,
        "selected_score_table_capacity_bytes_f32": 23_756_800,
        "dense_global_score_table_capacity_bytes_f32": 855_244_800,
        "selected_to_dense_score_table_capacity_fraction": 1.0 / 36.0,
    }
    profile.update(updates)
    return profile


def test_compact_posterior_execution_contract_requires_effective_profile() -> None:
    requested = runner._candidate_environment("compact_posterior", enabled=True)
    contract = runner._validate_arm_execution_contract(
        candidate_mode="compact_posterior",
        candidate_enabled=True,
        requested_environment=requested,
        effective_environment=dict(requested),
        estep_meta={
            "halfset_0_profile_summary": {
                "coarse_gaussian_gemm_hybrid": _compact_profile(),
            },
        },
    )

    assert contract["environment_exact"] is True
    assert contract["compact_requested"] is True
    assert contract["compact_effective"] is True
    assert contract["profile_exact"] is True
    assert list(contract["hybrid_profiles"]) == ["halfset_0_profile_summary"]


@pytest.mark.parametrize(
    ("profile", "match"),
    [
        (None, "did not publish a hybrid profile"),
        (_compact_profile(fallback_batch_count=1), "fallback_batch_count"),
        (
            _compact_profile(compact_posterior_enabled=False),
            "compact_posterior_enabled",
        ),
        (
            _compact_profile(selected_to_dense_score_table_capacity_fraction=1.0),
            "table fraction",
        ),
    ],
)
def test_compact_posterior_execution_contract_fails_closed(profile, match) -> None:
    requested = runner._candidate_environment("compact_posterior", enabled=True)
    estep_meta = (
        {}
        if profile is None
        else {
            "halfset_0_profile_summary": {
                "coarse_gaussian_gemm_hybrid": profile,
            },
        }
    )

    with pytest.raises(RuntimeError, match=match):
        runner._validate_arm_execution_contract(
            candidate_mode="compact_posterior",
            candidate_enabled=True,
            requested_environment=requested,
            effective_environment=dict(requested),
            estep_meta=estep_meta,
        )


def test_compact_posterior_control_rejects_environment_or_profile_leak() -> None:
    requested = runner._candidate_environment("compact_posterior", enabled=False)
    leaked_environment = dict(requested)
    leaked_environment[runner.COMPACT_POSTERIOR_ENVIRONMENT] = "1"
    with pytest.raises(RuntimeError, match="environment differs"):
        runner._validate_arm_execution_contract(
            candidate_mode="compact_posterior",
            candidate_enabled=False,
            requested_environment=requested,
            effective_environment=leaked_environment,
            estep_meta={},
        )

    with pytest.raises(RuntimeError, match="unexpectedly published"):
        runner._validate_arm_execution_contract(
            candidate_mode="compact_posterior",
            candidate_enabled=False,
            requested_environment=requested,
            effective_environment=dict(requested),
            estep_meta={
                "halfset_0_profile_summary": {
                    "coarse_gaussian_gemm_hybrid": _compact_profile(),
                },
            },
        )


def _science_pair(normalized_l2: float) -> dict:
    exact = {"exact_equal": True}
    return {
        "estep_meta": {
            key: dict(exact)
            for key in (
                "selected_particle_ids",
                "best_pose_rotation_ids",
                "pose_assignments",
                "class_assignments",
                "best_pose_translations",
                "significant_counts",
            )
        },
        "support_audits": {"exact_equal": True},
        "accumulators": {
            "entries": [
                {
                    "A2": {
                        "exact_equal": normalized_l2 == 0.0,
                        "normalized_l2_delta": normalized_l2,
                    },
                },
            ],
        },
        "particle_state": {"rot": dict(exact)},
        "sampling_state": {"order": dict(exact)},
        "final_state": {
            "data": {
                "exact_equal": normalized_l2 == 0.0,
                "normalized_l2_delta": normalized_l2,
            },
        },
    }


def test_compact_science_contract_requires_exact_decisions_and_reports_envelope() -> None:
    order = runner.COMPACT_POSTERIOR_ARM_ORDER
    pair_values = {
        "direct_1__vs__direct_2": 2.0e-7,
        "compact_posterior_1__vs__compact_posterior_2": 3.0e-7,
        "direct_1__vs__compact_posterior_1": 1.0e-7,
        "direct_1__vs__compact_posterior_2": 2.5e-7,
        "direct_2__vs__compact_posterior_1": 2.0e-7,
        "direct_2__vs__compact_posterior_2": 1.5e-7,
    }
    comparisons = {
        name: _science_pair(value) for name, value in pair_values.items()
    }

    contract = runner._compact_science_contract(comparisons, order)

    assert contract["hard_exact_contract_passed"] is True
    accumulator = contract["accumulator_repeat_envelope"]
    assert accumulator["all_cross_within_observed_repeat_envelope"] is True
    assert accumulator["rows"]["0.A2"]["repeat_envelope_normalized_l2"] == 3.0e-7
    assert contract["final_state_repeat_envelope"][
        "all_cross_within_observed_repeat_envelope"
    ] is True


def test_compact_science_contract_localizes_exact_and_repeat_envelope_failure() -> None:
    order = runner.COMPACT_POSTERIOR_ARM_ORDER
    pair_values = {
        "direct_1__vs__direct_2": 1.0e-7,
        "compact_posterior_1__vs__compact_posterior_2": 1.0e-7,
        "direct_1__vs__compact_posterior_1": 4.0e-7,
        "direct_1__vs__compact_posterior_2": 2.0e-7,
        "direct_2__vs__compact_posterior_1": 2.0e-7,
        "direct_2__vs__compact_posterior_2": 2.0e-7,
    }
    comparisons = {
        name: _science_pair(value) for name, value in pair_values.items()
    }
    comparisons["direct_1__vs__compact_posterior_1"]["estep_meta"][
        "pose_assignments"
    ]["exact_equal"] = False

    contract = runner._compact_science_contract(comparisons, order)

    assert contract["hard_exact_contract_passed"] is False
    failed_pair = contract["exact_cross_pair_checks"][
        "direct_1__vs__compact_posterior_1"
    ]
    assert failed_pair["unequal_meta"] == ["pose_assignments"]
    accumulator = contract["accumulator_repeat_envelope"]
    assert accumulator["all_cross_within_observed_repeat_envelope"] is False
    assert accumulator["outside_paths"] == ["0.A2"]


def test_arm_performance_summary_exposes_compact_table_geometry() -> None:
    profile = _compact_profile()
    summary = runner._arm_performance_summary(
        wall_s=1.25,
        estep_meta={
            "sparse_pass2_profile_summary": {
                "pass1_time_s": 0.4,
                "pass2_time_s": 0.6,
            },
            "halfset_0_profile_summary": {
                "coarse_gaussian_gemm_hybrid": profile,
            },
        },
    )

    assert summary["wall_s"] == 1.25
    assert summary["pass1_time_s"] == 0.4
    table = summary["coarse_hybrid_tables"]["halfset_0_profile_summary"]
    assert table["selected_score_table_capacity_candidates"] == 5_939_200
    assert table["dense_global_score_table_capacity_bytes_f32"] == 855_244_800


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
    assert "stable_off_1,stable_on_1,stable_on_2,stable_off_2" in sbatch
    assert (
        "stable_flat_off_1,stable_flat_on_1,stable_flat_on_2,stable_flat_off_2"
        in sbatch
    )
    assert '"RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY=0"' in sbatch
    assert (
        "direct_1,compact_posterior_1,compact_posterior_2,direct_2" in sbatch
    )
    assert "RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR=0" in sbatch
    assert ".compact_effective == true" in sbatch
    assert "make -B -C \"${REPO_ROOT}/recovar/cuda\"" in sbatch
    assert "status --porcelain=v1 --untracked-files=all" in sbatch
    assert "VDAM_SAME_STATE_NOISE_SPLIT_DIAGNOSTICS" in sbatch
    assert "RECOVAR_NOISE_DEBUG_DUMP_DIR=${RUNTIME}/noise_split_enabled" in sbatch
