from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from recovar.em.dense_single_volume import local_debug

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
        "noise_sumw": np.asarray(2.75, dtype=np.float32),
        "wsum_sigma2_offset": np.asarray(8.5, dtype=np.float32),
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
    assert report["noise_sumw"]["exact_equal"] is True
    assert report["wsum_sigma2_offset"]["exact_equal"] is True


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


def test_fused_posterior_dump_keeps_arm_label_when_score_label_is_also_set(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_LABEL", "single class")
    monkeypatch.setenv(
        "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL",
        "stable_all_on_1 single class",
    )

    assert local_debug._local_debug_dump_label_suffix() == "_single_class"
    assert (
        local_debug._local_fused_posterior_dump_label_suffix()
        == "_stable_all_on_1_single_class"
    )


def test_same_state_candidate_modes_keep_control_and_candidate_scoped() -> None:
    assert runner._arm_order("hybrid") == runner.ARM_ORDER
    assert runner._arm_order("flat_rows") == runner.FLAT_ROW_ARM_ORDER
    assert runner._arm_order("packed_projection") == runner.PACKED_PROJECTION_ARM_ORDER
    assert runner._arm_order("packed_deferred") == runner.PACKED_DEFERRED_ARM_ORDER
    assert (
        runner._arm_order("packed_final_noise")
        == runner.PACKED_FINAL_NOISE_ARM_ORDER
    )
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
    assert (
        runner._arm_order("compact_packed_deferred")
        == runner.COMPACT_PACKED_DEFERRED_ARM_ORDER
    )
    assert runner._arm_order("all_optimized") == runner.ALL_OPTIMIZED_ARM_ORDER
    assert (
        runner._arm_order("exact_coarse_single_translate")
        == runner.EXACT_COARSE_SINGLE_TRANSLATE_ARM_ORDER
    )
    assert (
        runner._arm_order("exact_compact_preprocess")
        == runner.EXACT_COMPACT_PREPROCESS_ARM_ORDER
    )
    assert (
        runner._arm_order("fused_pair_fine_score")
        == runner.FUSED_PAIR_FINE_SCORE_ARM_ORDER
    )

    control = runner._candidate_environment("flat_rows", enabled=False)
    flat_rows = runner._candidate_environment("flat_rows", enabled=True)
    packed_projection = runner._candidate_environment(
        "packed_projection", enabled=True
    )
    packed_deferred = runner._candidate_environment(
        "packed_deferred", enabled=True
    )
    packed_final_noise = runner._candidate_environment(
        "packed_final_noise", enabled=True
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
    compact_packed_deferred = runner._candidate_environment(
        "compact_packed_deferred",
        enabled=True,
    )
    all_optimized_control = runner._candidate_environment(
        "all_optimized",
        enabled=False,
    )
    all_optimized = runner._candidate_environment(
        "all_optimized",
        enabled=True,
    )
    single_translate_off = runner._candidate_environment(
        "exact_coarse_single_translate",
        enabled=False,
    )
    single_translate_on = runner._candidate_environment(
        "exact_coarse_single_translate",
        enabled=True,
    )
    compact_preprocess_off = runner._candidate_environment(
        "exact_compact_preprocess",
        enabled=False,
    )
    compact_preprocess_on = runner._candidate_environment(
        "exact_compact_preprocess",
        enabled=True,
    )
    pair_fine_off = runner._candidate_environment(
        "fused_pair_fine_score",
        enabled=False,
    )
    pair_fine_on = runner._candidate_environment(
        "fused_pair_fine_score",
        enabled=True,
    )

    assert all(control[name] == "0" for name in runner.HYBRID_ENVIRONMENT)
    assert control[runner.COMPACT_POSTERIOR_ENVIRONMENT] == "0"
    assert control[runner.FLAT_ROW_ENVIRONMENT] == "0"
    assert control[runner.STABLE_FLAT_CAPACITY_ENVIRONMENT] == "0"
    assert control[runner.PACKED_PROJECTION_ENVIRONMENT] == "0"
    assert control[runner.PACKED_DEFERRED_ENVIRONMENT] == "0"
    assert control[runner.PACKED_FINAL_NOISE_ENVIRONMENT] == "0"
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
    assert packed_deferred[runner.PACKED_FINAL_NOISE_ENVIRONMENT] == "0"
    assert all(
        packed_final_noise[name] == "0" for name in runner.HYBRID_ENVIRONMENT
    )
    assert packed_final_noise[runner.FLAT_ROW_ENVIRONMENT] == "1"
    assert packed_final_noise[runner.PACKED_PROJECTION_ENVIRONMENT] == "1"
    assert packed_final_noise[runner.PACKED_DEFERRED_ENVIRONMENT] == "1"
    assert packed_final_noise[runner.PACKED_FINAL_NOISE_ENVIRONMENT] == "1"
    assert all(
        hybrid_packed_deferred[name] == "1"
        for name in runner.HYBRID_ENVIRONMENT
    )
    assert hybrid_packed_deferred[runner.FLAT_ROW_ENVIRONMENT] == "1"
    assert hybrid_packed_deferred[runner.PACKED_PROJECTION_ENVIRONMENT] == "1"
    assert hybrid_packed_deferred[runner.PACKED_DEFERRED_ENVIRONMENT] == "1"
    assert hybrid_packed_deferred[runner.PACKED_FINAL_NOISE_ENVIRONMENT] == "0"
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
    assert hybrid[runner.PACKED_FINAL_NOISE_ENVIRONMENT] == "0"
    assert runner._candidate_uses_hybrid("hybrid") is True
    assert runner._candidate_uses_hybrid("hybrid_packed_deferred") is True
    assert runner._candidate_uses_hybrid("stable_shapes") is True
    assert runner._candidate_uses_hybrid("stable_flat_capacity") is True
    assert runner._candidate_uses_hybrid("packed_final_noise") is False
    assert all(
        compact_posterior[name] == "1" for name in runner.HYBRID_ENVIRONMENT
    )
    assert compact_posterior[runner.COMPACT_POSTERIOR_ENVIRONMENT] == "1"
    assert compact_posterior[runner.FLAT_ROW_ENVIRONMENT] == "0"
    assert compact_posterior[runner.PACKED_PROJECTION_ENVIRONMENT] == "0"
    assert compact_posterior[runner.PACKED_DEFERRED_ENVIRONMENT] == "0"
    assert all(
        compact_packed_deferred[name] == "1"
        for name in runner.HYBRID_ENVIRONMENT
    )
    assert compact_packed_deferred[runner.COMPACT_POSTERIOR_ENVIRONMENT] == "1"
    assert compact_packed_deferred[runner.FLAT_ROW_ENVIRONMENT] == "1"
    assert compact_packed_deferred[runner.PACKED_PROJECTION_ENVIRONMENT] == "1"
    assert compact_packed_deferred[runner.PACKED_DEFERRED_ENVIRONMENT] == "1"
    assert all(value == "0" for value in all_optimized_control.values())
    assert all(all_optimized[name] == "1" for name in runner.HYBRID_ENVIRONMENT)
    for name in (
        runner.COMPACT_POSTERIOR_ENVIRONMENT,
        runner.FLAT_ROW_ENVIRONMENT,
        runner.STABLE_FLAT_CAPACITY_ENVIRONMENT,
        runner.PACKED_PROJECTION_ENVIRONMENT,
        runner.PACKED_DEFERRED_ENVIRONMENT,
        runner.PACKED_FINAL_NOISE_ENVIRONMENT,
    ):
        assert all_optimized[name] == "1"
    assert all_optimized[runner.EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT] == "0"
    assert all_optimized[runner.EXACT_COMPACT_PREPROCESS_ENVIRONMENT] == "0"
    assert all_optimized[runner.FUSED_PAIR_FINE_SCORE_ENVIRONMENT] == "0"
    assert runner.HYBRID_IMAGE_BATCH_ENVIRONMENT not in all_optimized_control
    assert runner.HYBRID_IMAGE_BATCH_ENVIRONMENT not in all_optimized
    assert runner._candidate_uses_hybrid("compact_posterior") is True
    assert runner._candidate_uses_hybrid("compact_packed_deferred") is True
    assert runner._candidate_uses_hybrid("all_optimized") is True
    assert runner._candidate_uses_hybrid("exact_coarse_single_translate") is True
    assert runner._candidate_uses_hybrid("exact_compact_preprocess") is True
    assert runner._candidate_uses_hybrid("packed_deferred") is False
    assert runner._candidate_uses_compact_posterior("compact_posterior") is True
    assert (
        runner._candidate_uses_compact_posterior("compact_packed_deferred")
        is True
    )
    assert runner._candidate_uses_compact_posterior("hybrid") is False
    assert runner._candidate_uses_compact_posterior("all_optimized") is True
    assert (
        runner._candidate_uses_compact_posterior(
            "exact_coarse_single_translate"
        )
        is True
    )
    assert (
        runner._candidate_uses_compact_posterior("exact_compact_preprocess")
        is True
    )
    assert runner._candidate_uses_packed_deferred("all_optimized") is True
    assert (
        runner._candidate_uses_packed_deferred(
            "exact_coarse_single_translate"
        )
        is True
    )
    assert (
        runner._candidate_uses_packed_deferred("exact_compact_preprocess")
        is True
    )
    assert runner._candidate_uses_packed_deferred("compact_posterior") is False
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
    isolated_environment = runner.EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT
    assert single_translate_off[isolated_environment] == "0"
    assert single_translate_on[isolated_environment] == "1"
    assert {
        key: value
        for key, value in single_translate_off.items()
        if key != isolated_environment
    } == {
        key: value
        for key, value in single_translate_on.items()
        if key != isolated_environment
    }
    for key in (
        *runner.HYBRID_ENVIRONMENT,
        runner.COMPACT_POSTERIOR_ENVIRONMENT,
        runner.FLAT_ROW_ENVIRONMENT,
        runner.STABLE_FLAT_CAPACITY_ENVIRONMENT,
        runner.PACKED_PROJECTION_ENVIRONMENT,
        runner.PACKED_DEFERRED_ENVIRONMENT,
        runner.PACKED_FINAL_NOISE_ENVIRONMENT,
    ):
        assert single_translate_on[key] == "1"
    assert single_translate_on[runner.EXACT_COMPACT_PREPROCESS_ENVIRONMENT] == "0"
    assert single_translate_on[runner.FUSED_PAIR_FINE_SCORE_ENVIRONMENT] == "0"
    assert not runner._arm_candidate_enabled(
        "single_translate_off_1",
        "exact_coarse_single_translate",
    )
    assert runner._arm_candidate_enabled(
        "single_translate_on_1",
        "exact_coarse_single_translate",
    )
    compact_preprocess_environment = runner.EXACT_COMPACT_PREPROCESS_ENVIRONMENT
    assert compact_preprocess_off[runner.EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT] == "1"
    assert compact_preprocess_on[runner.EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT] == "1"
    assert compact_preprocess_off[compact_preprocess_environment] == "0"
    assert compact_preprocess_on[compact_preprocess_environment] == "1"
    assert {
        key: value
        for key, value in compact_preprocess_off.items()
        if key != compact_preprocess_environment
    } == {
        key: value
        for key, value in compact_preprocess_on.items()
        if key != compact_preprocess_environment
    }
    assert not runner._arm_candidate_enabled(
        "compact_preprocess_off_1",
        "exact_compact_preprocess",
    )
    assert runner._arm_candidate_enabled(
        "compact_preprocess_on_1",
        "exact_compact_preprocess",
    )
    pair_environment = runner.FUSED_PAIR_FINE_SCORE_ENVIRONMENT
    assert pair_fine_off[runner.EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT] == "1"
    assert pair_fine_off[runner.EXACT_COMPACT_PREPROCESS_ENVIRONMENT] == "1"
    assert pair_fine_off[pair_environment] == "0"
    assert pair_fine_on[pair_environment] == "1"
    assert {
        key: value for key, value in pair_fine_off.items() if key != pair_environment
    } == {
        key: value for key, value in pair_fine_on.items() if key != pair_environment
    }
    assert runner._candidate_uses_hybrid("fused_pair_fine_score") is True
    assert runner._candidate_uses_compact_posterior("fused_pair_fine_score") is True
    assert runner._candidate_uses_packed_deferred("fused_pair_fine_score") is True
    assert not runner._arm_candidate_enabled(
        "pair_fine_off_1",
        "fused_pair_fine_score",
    )
    assert runner._arm_candidate_enabled(
        "pair_fine_on_1",
        "fused_pair_fine_score",
    )


def test_all_optimized_stable_pair_changes_only_the_two_stable_abis() -> None:
    mode = "all_optimized_stable_shapes"
    assert runner._arm_order(mode) == runner.ALL_OPTIMIZED_STABLE_SHAPES_ARM_ORDER
    assert runner._arm_candidate_enabled("stable_all_off_1", mode) is False
    assert runner._arm_candidate_enabled("stable_all_on_1", mode) is True

    stable_off = runner._candidate_environment(mode, enabled=False)
    stable_on = runner._candidate_environment(mode, enabled=True)
    assert {
        key for key in stable_off if stable_off[key] != stable_on[key]
    } == {runner.STABLE_FLAT_CAPACITY_ENVIRONMENT}
    assert stable_off[runner.STABLE_FLAT_CAPACITY_ENVIRONMENT] == "0"
    assert stable_on[runner.STABLE_FLAT_CAPACITY_ENVIRONMENT] == "1"
    assert stable_on[runner.STABLE_FOURIER_QUANTUM_ENVIRONMENT] == "32"
    assert stable_on[runner.HYBRID_IMAGE_BATCH_ENVIRONMENT] == "200"
    assert stable_on[runner.BATCHED_POSTERIOR_ENVIRONMENT] == "1"
    assert stable_on[runner.EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT] == "1"
    assert stable_on[runner.EXACT_COMPACT_PREPROCESS_ENVIRONMENT] == "1"
    assert stable_on[runner.FUSED_PAIR_FINE_SCORE_ENVIRONMENT] == "0"
    assert runner._candidate_uses_hybrid(mode) is True
    assert runner._candidate_uses_compact_posterior(mode) is True
    assert runner._candidate_uses_packed_deferred(mode) is True


def test_same_state_accepts_a_focused_posterior_dump_target(tmp_path) -> None:
    args = runner._parse_args(
        [
            "--fixture-dir",
            str(tmp_path),
            "--acceptance-config",
            str(tmp_path / "acceptance.json"),
            "--output-root",
            str(tmp_path / "out"),
            "--fused-posterior-dump-original-index",
            "2798",
        ]
    )

    assert args.fused_posterior_dump_original_index == 2798


def test_hybrid_image_batch_gate_uses_one_oracle_and_mirrored_four_repeats() -> None:
    specs = runner._hybrid_image_batch_arm_specs()
    assert tuple(label for label, _request in specs) == (
        runner.HYBRID_IMAGE_BATCH_GATE_ARM_ORDER
    )
    assert specs[0] == ("direct_oracle", None)
    assert [request for _label, request in specs].count(110) == 4
    assert [request for _label, request in specs].count(500) == 4

    pairs = runner._hybrid_image_batch_pair_labels()
    assert len(pairs) == 36
    assert len(set(pairs)) == len(pairs)
    assert ("direct_oracle", "abba_batch110_1") in pairs
    assert ("abba_batch110_1", "abba_batch200_1") in pairs
    assert ("abba_batch110_1", "baab_batch110_1") in pairs
    assert ("abba_batch200_1", "baab_batch200_1") in pairs


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
        "score_representation_policy": (
            "compact_only_when_fixed_physical_capacity_is_smaller_than_dense"
        ),
        "static_preferred_score_representation": "compact_selected_exact",
        "score_representation_batch_counts": {"compact_selected_exact": 2},
        "batch_count": 2,
        "certificate_chunk_count_per_batch": 8,
        "certificate_chunk_rows": 4608,
        "selected_rescore_batch_count": 2,
        "static_dense_batch_count": 0,
        "fallback_batch_count": 0,
        "selected_rescore_image_count": 300,
        "static_dense_image_count": 0,
        "fallback_image_count": 0,
        "overflow_latch_scope": (
            "current_significance_call_exact_geometry_and_capacity"
        ),
        "overflow_latch_active_at_return": False,
        "overflow_latch_activation_count": 0,
        "overflow_latch_static_dense_batch_count": 0,
        "overflow_latch_static_dense_image_count": 0,
        "selected_source16_block_count": 500,
        "selected_exact_candidate_count": 232_000,
        "selected_score_table_capacity_candidates": 5_939_200,
        "dense_global_score_table_capacity_candidates": 213_811_200,
        "selected_score_table_capacity_bytes_f32": 23_756_800,
        "dense_global_score_table_capacity_bytes_f32": 855_244_800,
        "selected_to_dense_score_table_capacity_fraction": 1.0 / 36.0,
        "static_compact_to_dense_capacity_fraction": 1.0 / 4.5,
        "topology_full_to_compact_sha256": "a" * 64,
    }
    profile.update(updates)
    return profile


def _hybrid_image_batch_meta(requested_batch_size: int) -> dict:
    if requested_batch_size == 110:
        effective_batch_size = 110
        batch_count = 2
    elif requested_batch_size == 500:
        effective_batch_size = 200
        batch_count = 1
    else:
        raise ValueError(requested_batch_size)
    profile = _compact_profile(
        input_image_batch_size=110,
        requested_hybrid_image_batch_size=requested_batch_size,
        effective_image_batch_size=effective_batch_size,
        streamed_certificate_candidate_count_at_effective_batch=(
            effective_batch_size * 4_608 * 49
        ),
        batch_count=batch_count,
        selected_rescore_batch_count=batch_count,
        selected_rescore_image_count=200,
        actual_image_batch_sizes=(
            [110, 90] if requested_batch_size == 110 else [200]
        ),
        physical_image_batch_sizes=(
            [110, 110] if requested_batch_size == 110 else [200]
        ),
    )
    return {
        "halfset_0_profile_summary": {
            "coarse_gaussian_gemm_hybrid": profile,
        },
    }


@pytest.mark.parametrize(
    ("requested_batch_size", "effective_batch_size", "batch_count"),
    [(110, 110, 2), (500, 200, 1)],
)
def test_hybrid_image_batch_profile_contract_is_exact(
    requested_batch_size: int,
    effective_batch_size: int,
    batch_count: int,
) -> None:
    contract = runner._validate_hybrid_image_batch_profiles(
        _hybrid_image_batch_meta(requested_batch_size),
        requested_batch_size=requested_batch_size,
        label="arm",
    )

    assert contract["requested_batch_size"] == requested_batch_size
    assert contract["effective_batch_size"] == effective_batch_size
    assert contract["profile_exact"] is True
    profile = contract["profiles"]["halfset_0_profile_summary"]
    assert profile["batch_count"] == batch_count
    assert profile["fallback_batch_count"] == 0


@pytest.mark.parametrize(
    ("requested_batch_size", "broken_field", "broken_value"),
    [
        (110, "batch_count", 1),
        (500, "effective_image_batch_size", 110),
        (500, "streamed_certificate_candidate_count_at_effective_batch", 1),
        (500, "fallback_batch_count", 1),
    ],
)
def test_hybrid_image_batch_profile_contract_fails_closed(
    requested_batch_size: int,
    broken_field: str,
    broken_value: int,
) -> None:
    meta = _hybrid_image_batch_meta(requested_batch_size)
    meta["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"][
        broken_field
    ] = broken_value

    with pytest.raises(RuntimeError, match=broken_field):
        runner._validate_hybrid_image_batch_profiles(
            meta,
            requested_batch_size=requested_batch_size,
            label="arm",
        )


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


def test_compact_posterior_execution_contract_accepts_static_dense_schedule() -> None:
    requested = runner._candidate_environment("compact_posterior", enabled=True)
    profile = _compact_profile(
        static_preferred_score_representation="dense_full_direct_static_capacity",
        score_representation_batch_counts={
            "dense_full_direct_static_capacity": 2,
        },
        selected_rescore_batch_count=0,
        static_dense_batch_count=2,
        selected_rescore_image_count=0,
        static_dense_image_count=300,
        selected_source16_block_count=0,
        selected_exact_candidate_count=0,
        selected_score_table_capacity_candidates=0,
        dense_global_score_table_capacity_candidates=0,
        selected_score_table_capacity_bytes_f32=0,
        dense_global_score_table_capacity_bytes_f32=0,
        selected_to_dense_score_table_capacity_fraction=None,
        static_compact_to_dense_capacity_fraction=16.0 / 9.0,
    )
    contract = runner._validate_arm_execution_contract(
        candidate_mode="compact_posterior",
        candidate_enabled=True,
        requested_environment=requested,
        effective_environment=dict(requested),
        estep_meta={
            "halfset_0_profile_summary": {
                "coarse_gaussian_gemm_hybrid": profile,
            },
        },
    )

    assert contract["profile_exact"] is True
    observed = contract["hybrid_profiles"]["halfset_0_profile_summary"]
    assert observed["static_dense_batch_count"] == 2


@pytest.mark.parametrize(
    "broken_field",
    [
        "flat_local_rows_enabled",
        "packed_local_projection_enabled",
        "defer_packed_vdam_enabled",
        "packed_vdam_reuses_flat_score_projection",
    ],
)
def test_compact_packed_execution_contract_requires_both_optimized_profiles(
    broken_field: str,
) -> None:
    requested = runner._candidate_environment(
        "compact_packed_deferred",
        enabled=True,
    )
    profile = {
        "coarse_gaussian_gemm_hybrid": _compact_profile(),
        "flat_local_rows_enabled": True,
        "packed_local_projection_enabled": True,
        "defer_packed_vdam_enabled": True,
        "packed_vdam_reuses_flat_score_projection": True,
    }
    contract = runner._validate_arm_execution_contract(
        candidate_mode="compact_packed_deferred",
        candidate_enabled=True,
        requested_environment=requested,
        effective_environment=dict(requested),
        estep_meta={"halfset_0_profile_summary": profile},
    )

    assert contract["compact_effective"] is True
    assert contract["packed_deferred_effective"] is True
    assert contract["profile_exact"] is True
    assert contract["packed_profiles"] == {
        "halfset_0_profile_summary": {
            "flat_local_rows_enabled": True,
            "packed_local_projection_enabled": True,
            "defer_packed_vdam_enabled": True,
            "packed_vdam_reuses_flat_score_projection": True,
        },
    }

    profile[broken_field] = False
    with pytest.raises(RuntimeError, match=broken_field):
        runner._validate_arm_execution_contract(
            candidate_mode="compact_packed_deferred",
            candidate_enabled=True,
            requested_environment=requested,
            effective_environment=dict(requested),
            estep_meta={"halfset_0_profile_summary": profile},
        )


def _all_optimized_estep_meta(*, enabled: bool) -> dict:
    physical_size = 88 if enabled else 84
    physical_score_pixels = 3104 if enabled else 2834
    physical_projection_pixels = 3105 if enabled else 2835
    profile = {
        "flat_local_rows_enabled": enabled,
        "stable_flat_row_capacity_enabled": enabled,
        "packed_local_projection_enabled": enabled,
        "defer_packed_vdam_enabled": enabled,
        "packed_vdam_reuses_flat_score_projection": enabled,
        "packed_final_noise_enabled": enabled,
        "packed_vdam_avoids_dense_noise_rows": enabled,
        "packed_final_noise_preserves_dense_scalar_order": enabled,
        "sum_packed_final_noise_rows": 800 if enabled else 0,
        "stable_fourier_window_shapes": enabled,
        "stable_fourier_window_quantum": 8,
        "logical_current_size": 84,
        "physical_current_size": physical_size,
        "logical_reconstruction_pixels": 2835,
        "physical_reconstruction_pixels": physical_projection_pixels,
        "n_windowed": physical_score_pixels,
        "n_projection_windowed": physical_projection_pixels,
        "big_jit_projection_pixels": physical_projection_pixels,
        "chunk_flat_score_rows": [13824],
        "chunk_padded_rotations": [13824],
    }
    if enabled:
        profile["coarse_gaussian_gemm_hybrid"] = _compact_profile(
            coarse_square_layout={
                "stable_fourier_window_shapes_requested": True,
                "stable_fourier_window_shapes_effective": True,
                "logical_current_size": 84,
                "physical_current_size": 88,
                "logical_square_pixels": 84 * 43,
                "physical_square_pixels": 88 * 45,
                "logical_issue_stream_is_prefix": True,
                "physical_tail_zero_weighted": True,
            },
        )
    return {
        "requested_stable_fourier_window_shapes": enabled,
        "effective_stable_fourier_window_shapes": enabled,
        "requested_stable_flat_row_capacity": enabled,
        "effective_stable_flat_row_capacity": enabled,
        "halfset_0_profile_summary": profile,
    }


def test_all_optimized_execution_contract_proves_every_candidate_seam() -> None:
    requested = runner._candidate_environment("all_optimized", enabled=True)
    meta = _all_optimized_estep_meta(enabled=True)

    compact = runner._validate_arm_execution_contract(
        candidate_mode="all_optimized",
        candidate_enabled=True,
        requested_environment=requested,
        effective_environment=dict(requested),
        estep_meta=meta,
    )
    composed = runner._validate_all_optimized_profiles(
        meta,
        enabled=True,
        label="all_optimized_1",
        image_shape=(128, 128),
    )

    assert compact["compact_effective"] is True
    assert compact["packed_deferred_effective"] is True
    assert composed["profile_exact"] is True
    assert composed["enabled_seams"] == list(runner.ALL_OPTIMIZED_SEAMS)
    assert composed["disabled_seams"] == []
    assert composed["stable_fourier"]["enabled"] is True
    assert composed["stable_fourier"]["stable_fourier_window_quantum"] == 8
    assert composed["stable_flat_capacity"]["strict_reduction_count"] == 0
    assert composed["packed_final_noise"]["enabled"] is True


def test_all_optimized_execution_contract_validates_q32_capacity() -> None:
    meta = _all_optimized_estep_meta(enabled=True)
    profile = meta["halfset_0_profile_summary"]
    profile.update(
        stable_fourier_window_quantum=32,
        physical_current_size=96,
        physical_reconstruction_pixels=3691,
        n_windowed=3690,
        n_projection_windowed=3691,
        big_jit_projection_pixels=3691,
    )
    coarse_layout = profile["coarse_gaussian_gemm_hybrid"][
        "coarse_square_layout"
    ]
    coarse_layout.update(
        physical_current_size=96,
        physical_square_pixels=96 * 49,
    )

    contract = runner._validate_all_optimized_profiles(
        meta,
        enabled=True,
        label="all_optimized_q32",
        image_shape=(128, 128),
        stable_fourier_window_quantum=32,
    )

    stable = contract["stable_fourier"]
    assert stable["stable_fourier_window_quantum"] == 32
    assert stable["profiles"]["halfset_0_profile_summary"][
        "physical_current_size"
    ] == 96


@pytest.mark.parametrize("enabled", [False, True])
def test_all_optimized_stable_pair_profile_keeps_other_seams_on(
    enabled: bool,
) -> None:
    meta = _all_optimized_estep_meta(enabled=True)
    profile = meta["halfset_0_profile_summary"]
    meta["requested_stable_fourier_window_shapes"] = enabled
    meta["effective_stable_fourier_window_shapes"] = enabled
    meta["requested_stable_flat_row_capacity"] = enabled
    meta["effective_stable_flat_row_capacity"] = enabled
    profile.update(
        stable_fourier_window_shapes=enabled,
        stable_fourier_window_quantum=32 if enabled else 8,
        physical_current_size=96 if enabled else 84,
        physical_reconstruction_pixels=3691 if enabled else 2835,
        n_windowed=3690 if enabled else 2834,
        n_projection_windowed=3691 if enabled else 2835,
        big_jit_projection_pixels=3691 if enabled else 2835,
        stable_flat_row_capacity_enabled=enabled,
        chunk_flat_score_rows=[13824 if enabled else 4752],
    )
    profile["coarse_gaussian_gemm_hybrid"]["coarse_square_layout"] = {
        "stable_fourier_window_shapes_requested": enabled,
        "stable_fourier_window_shapes_effective": enabled,
        "logical_current_size": 84,
        "physical_current_size": 96 if enabled else 84,
        "logical_square_pixels": 84 * 43,
        "physical_square_pixels": (96 * 49) if enabled else (84 * 43),
        "logical_issue_stream_is_prefix": True,
        "physical_tail_zero_weighted": True,
    }

    contract = runner._validate_all_optimized_stable_pair_profiles(
        meta,
        enabled=enabled,
        label=f"stable_all_{'on' if enabled else 'off'}",
        image_shape=(128, 128),
    )

    assert contract["profile_exact"] is True
    assert contract["stable_representation_enabled"] is enabled
    assert contract["all_other_optimized_seams_enabled"] is True
    assert contract["batched_posterior_primitives_enabled"] is True
    assert contract["stable_coarse_significance"]["profile_exact"] is True
    assert contract["packed_final_noise"]["enabled"] is True


def test_all_optimized_execution_contract_proves_direct_control_is_off() -> None:
    requested = runner._candidate_environment("all_optimized", enabled=False)
    meta = _all_optimized_estep_meta(enabled=False)

    compact = runner._validate_arm_execution_contract(
        candidate_mode="all_optimized",
        candidate_enabled=False,
        requested_environment=requested,
        effective_environment=dict(requested),
        estep_meta=meta,
    )
    composed = runner._validate_all_optimized_profiles(
        meta,
        enabled=False,
        label="direct_1",
        image_shape=(128, 128),
    )

    assert compact["compact_effective"] is False
    assert compact["packed_deferred_effective"] is False
    assert composed["enabled_seams"] == []
    assert composed["disabled_seams"] == list(runner.ALL_OPTIMIZED_SEAMS)
    assert composed["stable_fourier"]["enabled"] is False
    assert composed["stable_flat_capacity"] is None
    assert composed["packed_final_noise"]["enabled"] is False


def _fused_pair_fine_meta(*, enabled: bool) -> dict:
    meta = _all_optimized_estep_meta(enabled=True)
    meta["requested_fused_pair_fine_score"] = enabled
    meta["effective_fused_pair_fine_score"] = enabled
    profile = meta["halfset_0_profile_summary"]
    profile.update(
        fused_pair_fine_score_enabled=enabled,
        fused_pair_fine_score_default_enabled=False,
        fused_pair_fine_uses_shared_compact_order=enabled,
        fused_pair_fine_avoids_pair_pixel_gathers=enabled,
        fused_pair_fine_restores_dense_posterior_order=enabled,
        chunk_fused_pair_capacities=[10_000, 10_000] if enabled else [],
        chunk_fused_pair_counts=[7000, 8000] if enabled else [],
        chunk_fused_pair_dense_capacities=[5_000_000, 5_000_000]
        if enabled
        else [],
        sum_fused_pair_candidates=15_000 if enabled else 0,
        sum_fused_pair_capacity=20_000 if enabled else 0,
        sum_fused_pair_dense_capacity=10_000_000 if enabled else 0,
        fused_pair_valid_fraction_of_dense=0.0015 if enabled else 0.0,
        fused_pair_padded_fraction_of_dense=0.002 if enabled else 0.0,
    )
    return meta


@pytest.mark.parametrize("enabled", [False, True])
def test_fused_pair_fine_profile_proves_shared_compact_execution(
    enabled: bool,
) -> None:
    contract = runner._validate_fused_pair_fine_profiles(
        _fused_pair_fine_meta(enabled=enabled),
        enabled=enabled,
        label="arm",
    )

    assert contract["profile_exact"] is True
    assert contract["enabled"] is enabled
    assert contract["pair_pixel_gathers_materialized"] is False
    assert contract["compile_shape_capacity_family_count"] == (1 if enabled else 0)
    assert contract["sum_candidates"] == (15_000 if enabled else 0)


def test_fused_pair_fine_profile_fails_closed_on_candidate_geometry() -> None:
    meta = _fused_pair_fine_meta(enabled=True)
    meta["halfset_0_profile_summary"]["chunk_fused_pair_capacities"][0] = 6_000

    with pytest.raises(RuntimeError, match="invalid pair counts"):
        runner._validate_fused_pair_fine_profiles(
            meta,
            enabled=True,
            label="arm",
        )


def test_fused_pair_fine_profile_fails_closed_on_stale_chunk_totals() -> None:
    meta = _fused_pair_fine_meta(enabled=True)
    meta["halfset_0_profile_summary"]["sum_fused_pair_capacity"] = 19_999

    with pytest.raises(RuntimeError, match="chunk sums"):
        runner._validate_fused_pair_fine_profiles(
            meta,
            enabled=True,
            label="arm",
        )


def _exact_coarse_single_translate_meta(*, enabled: bool) -> dict:
    meta = _all_optimized_estep_meta(enabled=True)
    batch_count = meta["halfset_0_profile_summary"][
        "coarse_gaussian_gemm_hybrid"
    ]["batch_count"]
    meta["halfset_0_profile_summary"]["exact_coarse_operand_assembly"] = {
        "skip_generic_default_enabled": False,
        "skip_generic_requested": enabled,
        "skip_generic_effective": enabled,
        "exact_coarse_operands_effective": True,
        "generic_assembly_count": 0 if enabled else batch_count,
        "exact_assembly_count": batch_count,
        "translate_score_call_site_count": (
            batch_count if enabled else 2 * batch_count
        ),
        "translate_score_call_count": (
            batch_count if enabled else 2 * batch_count
        ),
        "downstream_operand_source": "exact_source_star",
        "diagnostic_operand_source": "exact_source_star",
        "raw_score_capture_changed": False,
        "skipped_generic_outputs": (
            [
                "coarse_gaussian_shifted_corrected",
                "coarse_gaussian_pixel_weight",
                "coarse_gaussian_unshifted_corrected",
            ]
            if enabled
            else []
        ),
    }
    return meta


@pytest.mark.parametrize(("enabled", "expected_calls"), [(False, 4), (True, 2)])
def test_exact_coarse_single_translate_profile_proves_actual_calls(
    enabled: bool,
    expected_calls: int,
) -> None:
    contract = runner._validate_exact_coarse_single_translate_profiles(
        _exact_coarse_single_translate_meta(enabled=enabled),
        enabled=enabled,
        label="arm",
    )

    assert contract["profile_exact"] is True
    assert contract["total_batch_count"] == 2
    assert contract["total_translate_score_call_count"] == expected_calls
    assert contract["expected_calls_per_batch"] == (1 if enabled else 2)
    assert contract["profile_counter_device_synchronization"] is False


def test_exact_coarse_single_translate_profile_fails_closed() -> None:
    meta = _exact_coarse_single_translate_meta(enabled=True)
    meta["halfset_0_profile_summary"]["exact_coarse_operand_assembly"][
        "translate_score_call_count"
    ] = 4

    with pytest.raises(RuntimeError, match="assembly mismatch"):
        runner._validate_exact_coarse_single_translate_profiles(
            meta,
            enabled=True,
            label="arm",
        )


def _exact_compact_preprocess_meta(*, enabled: bool) -> dict:
    meta = _all_optimized_estep_meta(enabled=True)
    batch_count = meta["halfset_0_profile_summary"][
        "coarse_gaussian_gemm_hybrid"
    ]["batch_count"]
    generic_count = 0 if enabled else batch_count
    meta["halfset_0_profile_summary"]["exact_coarse_operand_assembly"] = {
        "skip_generic_default_enabled": False,
        "skip_generic_requested": True,
        "skip_generic_effective": True,
        "exact_coarse_operands_effective": True,
        "exact_compact_preprocess_default_enabled": False,
        "exact_compact_preprocess_requested": enabled,
        "exact_compact_preprocess_effective": enabled,
        "generic_score_preprocess_count": generic_count,
        "exact_source_preprocess_count": batch_count,
        "generic_ctf_evaluation_count": generic_count,
        "generic_full_translation_count": generic_count,
        "generic_assembly_count": 0,
        "exact_assembly_count": batch_count,
        "translate_score_call_site_count": batch_count,
        "translate_score_call_count": batch_count,
        "downstream_operand_source": "exact_source_star",
        "diagnostic_operand_source": "exact_source_star",
        "raw_score_capture_changed": False,
        "generic_fallback_policy": (
            "exact_full_direct_scores_available" if enabled else "available"
        ),
        "skipped_generic_outputs": [
            "coarse_gaussian_shifted_corrected",
            "coarse_gaussian_pixel_weight",
            "coarse_gaussian_unshifted_corrected",
        ],
    }
    return meta


@pytest.mark.parametrize(
    ("enabled", "expected_generic", "expected_calls"),
    [(False, 2, 2), (True, 0, 1)],
)
def test_exact_compact_preprocess_profile_proves_removed_duplicate_work(
    enabled: bool,
    expected_generic: int,
    expected_calls: int,
) -> None:
    contract = runner._validate_exact_compact_preprocess_profiles(
        _exact_compact_preprocess_meta(enabled=enabled),
        enabled=enabled,
        label="arm",
    )

    assert contract["profile_exact"] is True
    assert contract["total_batch_count"] == 2
    assert contract["total_generic_score_preprocess_count"] == expected_generic
    assert contract["total_exact_source_preprocess_count"] == 2
    assert contract["expected_process_half_image_calls_per_batch"] == expected_calls
    assert contract["profile_counter_device_synchronization"] is False


def test_exact_compact_preprocess_profile_fails_closed() -> None:
    meta = _exact_compact_preprocess_meta(enabled=True)
    meta["halfset_0_profile_summary"]["exact_coarse_operand_assembly"][
        "generic_full_translation_count"
    ] = 1

    with pytest.raises(RuntimeError, match="preprocessing mismatch"):
        runner._validate_exact_compact_preprocess_profiles(
            meta,
            enabled=True,
            label="arm",
        )


@pytest.mark.parametrize(
    "broken_field",
    [
        "flat_local_rows_enabled",
        "stable_flat_row_capacity_enabled",
        "packed_local_projection_enabled",
        "defer_packed_vdam_enabled",
        "packed_vdam_reuses_flat_score_projection",
        "packed_final_noise_enabled",
        "packed_vdam_avoids_dense_noise_rows",
        "packed_final_noise_preserves_dense_scalar_order",
        "stable_fourier_window_shapes",
    ],
)
def test_all_optimized_execution_contract_fails_closed_on_local_seam(
    broken_field: str,
) -> None:
    meta = _all_optimized_estep_meta(enabled=True)
    meta["halfset_0_profile_summary"][broken_field] = False

    with pytest.raises(RuntimeError, match=broken_field):
        runner._validate_all_optimized_profiles(
            meta,
            enabled=True,
            label="all_optimized_1",
            image_shape=(128, 128),
        )


@pytest.mark.parametrize(
    "broken_field",
    [
        "requested_stable_fourier_window_shapes",
        "effective_stable_fourier_window_shapes",
        "requested_stable_flat_row_capacity",
        "effective_stable_flat_row_capacity",
    ],
)
def test_all_optimized_execution_contract_fails_closed_on_adapter_seam(
    broken_field: str,
) -> None:
    meta = _all_optimized_estep_meta(enabled=True)
    meta[broken_field] = False

    with pytest.raises(RuntimeError, match=broken_field):
        runner._validate_all_optimized_profiles(
            meta,
            enabled=True,
            label="all_optimized_1",
            image_shape=(128, 128),
        )


def test_all_optimized_execution_contract_fails_closed_on_shape_or_row_abi() -> None:
    wrong_shape = _all_optimized_estep_meta(enabled=True)
    wrong_shape["halfset_0_profile_summary"]["physical_current_size"] = 84
    with pytest.raises(RuntimeError, match="physical_current_size"):
        runner._validate_all_optimized_profiles(
            wrong_shape,
            enabled=True,
            label="all_optimized_1",
            image_shape=(128, 128),
        )


    wrong_rows = _all_optimized_estep_meta(enabled=True)
    wrong_rows["halfset_0_profile_summary"]["chunk_flat_score_rows"] = [4752]
    with pytest.raises(RuntimeError, match="mature B.R ABI"):
        runner._validate_all_optimized_profiles(
            wrong_rows,
            enabled=True,
            label="all_optimized_1",
            image_shape=(128, 128),
        )


def test_all_optimized_execution_contract_fails_closed_on_coarse_shape_abi() -> None:
    meta = _all_optimized_estep_meta(enabled=True)
    coarse_layout = meta["halfset_0_profile_summary"][
        "coarse_gaussian_gemm_hybrid"
    ]["coarse_square_layout"]
    coarse_layout["physical_current_size"] = 84

    with pytest.raises(RuntimeError, match="coarse stable-square mismatch"):
        runner._validate_all_optimized_profiles(
            meta,
            enabled=True,
            label="all_optimized_1",
            image_shape=(128, 128),
        )


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
        (
            _compact_profile(
                score_representation_batch_counts={"compact_selected_exact": 1},
            ),
            "representation counts",
        ),
        (
            _compact_profile(
                static_preferred_score_representation=(
                    "dense_full_direct_static_capacity"
                ),
            ),
            "static preference",
        ),
        (
            _compact_profile(certificate_chunk_count_per_batch=0),
            "certificate_chunk_count_per_batch",
        ),
        (
            _compact_profile(topology_full_to_compact_sha256="not-a-digest"),
            "topology_full_to_compact_sha256",
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
                "cutoff_counts",
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


def _hybrid_image_batch_science_pair(normalized_l2: float) -> dict:
    exact = {"exact_equal": True}
    array_delta = {
        "exact_equal": normalized_l2 == 0.0,
        "left_shape": [2],
        "normalized_l2_delta": normalized_l2,
    }
    return {
        "estep_meta": {
            key: dict(exact) for key in runner.HYBRID_IMAGE_BATCH_REQUIRED_META
        },
        "support_audits": {"exact_equal": True},
        "accumulators": {
            "comparable": True,
            "entries": [
                {
                    "class_idx": dict(exact),
                    "halfset_idx": dict(exact),
                    "data": dict(array_delta),
                },
            ],
        },
        "particle_state": {"rot": dict(exact)},
        "sampling_state": {"order": dict(exact)},
        "final_state": {
            **{
                field: dict(exact)
                for field in runner.HYBRID_IMAGE_BATCH_PUBLIC_STATE
            },
            "iter": dict(exact),
            "Igrad1": dict(array_delta),
        },
    }


def _hybrid_image_batch_science_comparisons() -> dict:
    comparisons = {}
    for left, right in runner._hybrid_image_batch_pair_labels():
        if left == "direct_oracle":
            normalized_l2 = 8.0e-7
        elif "batch110" in left and "batch110" in right:
            normalized_l2 = 3.0e-7
        elif "batch200" in left and "batch200" in right:
            normalized_l2 = 2.0e-7
        else:
            normalized_l2 = 1.0e-7
        comparisons[f"{left}__vs__{right}"] = _hybrid_image_batch_science_pair(
            normalized_l2,
        )
    return comparisons


def test_hybrid_image_batch_science_contract_uses_all_four_warm_repeats() -> None:
    contract = runner._hybrid_image_batch_science_contract(
        _hybrid_image_batch_science_comparisons(),
    )

    assert contract["hard_exact_contract_passed"] is True
    assert len(contract["exact_pair_checks"]) == 36
    accumulator = contract["accumulator_repeat_envelope"]
    assert accumulator["control_repeat_pair_count"] == 6
    assert accumulator["candidate_repeat_pair_count"] == 6
    assert accumulator["cross_pair_count"] == 16
    assert accumulator["rows"]["0.data"]["repeat_envelope_normalized_l2"] == 3.0e-7
    assert contract["atomic_repeat_envelope_passed"] is True


def test_hybrid_image_batch_science_contract_fails_closed() -> None:
    comparisons = _hybrid_image_batch_science_comparisons()
    failed_pair = "abba_batch110_1__vs__abba_batch200_1"
    comparisons[failed_pair]["accumulators"]["entries"][0]["data"][
        "normalized_l2_delta"
    ] = 4.0e-7
    comparisons[failed_pair]["final_state"]["Mavg"]["exact_equal"] = False

    contract = runner._hybrid_image_batch_science_contract(comparisons)

    assert contract["hard_exact_contract_passed"] is False
    assert contract["exact_pair_checks"][failed_pair]["unequal_public_state"] == [
        "Mavg"
    ]
    assert contract["atomic_repeat_envelope_passed"] is False
    assert contract["accumulator_repeat_envelope"]["outside_paths"] == ["0.data"]


def test_hybrid_image_batch_runtime_contract_compares_four_warm_arms_each() -> None:
    arms = {}
    for index, label in enumerate(runner.HYBRID_IMAGE_BATCH_ARM_ORDER):
        candidate = "batch200" in label
        arms[label] = {
            "wall_s": (0.8 if candidate else 1.0) + index * 0.001,
            "performance_summary": {
                "pass1_time_s": (0.4 if candidate else 0.5) + index * 0.001,
                "pass2_time_s": 0.3 + index * 0.001,
            },
        }

    contract = runner._hybrid_image_batch_runtime_contract(arms)

    assert contract["batch110"]["repeat_count"] == 4
    assert contract["batch200"]["repeat_count"] == 4
    assert len(contract["batch110"]["measurements"]["pass1_time_s"]) == 4
    assert contract["batch200_vs_batch110"]["pass1_time_s"]["speedup"] > 1.0


def test_exact_compact_preprocess_runtime_contract_is_balanced_and_isolated() -> None:
    arms = {}
    for index, label in enumerate(runner.EXACT_COMPACT_PREPROCESS_ARM_ORDER):
        enabled = "_on_" in label
        requested = runner._candidate_environment(
            "exact_compact_preprocess",
            enabled=enabled,
        )
        arms[label] = {
            "wall_s": (0.8 if enabled else 1.0) + index * 0.001,
            "performance_summary": {
                "pass1_time_s": (0.4 if enabled else 0.5) + index * 0.001,
                "pass2_time_s": 0.3 + index * 0.001,
            },
            "execution_contract": {"requested_environment": requested},
        }

    contract = runner._exact_compact_preprocess_runtime_contract(arms)

    assert contract["preprocess_off"]["repeat_count"] == 2
    assert contract["preprocess_on"]["repeat_count"] == 2
    assert contract["preprocess_on_vs_preprocess_off"]["wall_s"]["speedup"] > 1.0
    assert contract["isolated_environment_contract"] == {
        "only_difference": runner.EXACT_COMPACT_PREPROCESS_ENVIRONMENT,
        "all_other_environment_exact": True,
        "profile_free_wall_timing": True,
        "profile_counter_device_synchronization": False,
    }


def _cache_state(count: int) -> dict:
    counts = {
        family: (count if family == "jit_run_local_bucket_big_jit" else 0)
        for family in runner.PERSISTENT_CACHE_TARGET_FAMILIES
    }
    return {
        "available": True,
        "root": "/tmp/cache",
        "file_count": count,
        "bytes": count * 10,
        "target_family_counts": counts,
        "target_family_bytes": {family: value * 10 for family, value in counts.items()},
    }


def test_fused_pair_fine_runtime_contract_is_warmed_balanced_and_isolated() -> None:
    arms = {}
    for index, label in enumerate(runner.FUSED_PAIR_FINE_SCORE_ARM_ORDER):
        enabled = "_on_" in label
        requested = runner._candidate_environment(
            "fused_pair_fine_score",
            enabled=enabled,
        )
        cache_state = _cache_state(7)
        cache = {
            "before": cache_state,
            "after": cache_state,
            "delta": runner._persistent_cache_delta(cache_state, cache_state),
        }
        performance = {
            "pass1_time_s": 0.4 + index * 0.001,
            "pass2_time_s": (0.5 if enabled else 0.8) + index * 0.001,
            "local_em_time_s": (0.4 if enabled else 0.7) + index * 0.001,
            "local_big_jit_bucket_s": (0.2 if enabled else 0.5) + index * 0.001,
            "local_pack_s": 0.01,
            "local_noise_s": 0.05,
            "local_postprocess_s": 0.01,
            "local_final_accumulator_s": 0.02,
            "local_unattributed_em_time_s": 0.01,
        }
        arms[label] = {
            "wall_s": (0.75 if enabled else 1.0) + index * 0.001,
            "performance_summary": performance,
            "persistent_cache": cache,
            "execution_contract": {
                "requested_environment": requested,
                "fused_pair_fine_score": {
                    "enabled": enabled,
                    "profile_exact": True,
                },
            },
        }

    contract = runner._fused_pair_fine_runtime_contract(arms)

    assert contract["pair_off"]["repeat_count"] == 2
    assert contract["pair_on"]["repeat_count"] == 2
    assert contract["pair_on_vs_pair_off"]["wall_s"]["speedup"] > 1.05
    assert contract["material_speedup_passed"] is True
    assert contract["persistent_cache_contract"][
        "timed_arms_add_no_target_family_programs_after_prewarm"
    ] is True
    assert contract["isolated_environment_contract"]["only_difference"] == (
        runner.FUSED_PAIR_FINE_SCORE_ENVIRONMENT
    )


def test_persistent_cache_snapshot_counts_selected_local_families(
    tmp_path: Path,
    monkeypatch,
) -> None:
    family = "jit_run_local_bucket_big_jit"
    (tmp_path / f"{family}-{'a' * 64}-cache").write_bytes(b"1234")
    (tmp_path / f"{family}-{'b' * 64}-cache").write_bytes(b"12")
    (tmp_path / f"jit_unrelated-{'c' * 64}-cache").write_bytes(b"1")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path))

    snapshot = runner._persistent_cache_snapshot()

    assert snapshot["available"] is True
    assert snapshot["file_count"] == 3
    assert snapshot["target_family_counts"][family] == 2
    assert snapshot["target_family_bytes"][family] == 6


def test_gpu_memory_report_attributes_samples_to_arm_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monitor = tmp_path / "gpu.csv"
    monitor.write_text(
        "timestamp, index, name, uuid, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n"
        "2026/09/03 12:00:00.000, 0, H100, GPU-x, 100 MiB, 81559 MiB, 20 %\n"
        "2026/09/03 12:00:01.000, 0, H100, GPU-x, 120 MiB, 81559 MiB, 80 %\n"
    )
    monkeypatch.setenv(runner.GPU_MONITOR_ENVIRONMENT, str(monitor))
    start = runner.dt.datetime(2026, 9, 3, 12, 0, 0).timestamp()
    report = runner._gpu_memory_report(
        {
            "arm": {
                "wall_clock_started_epoch_s": start,
                "wall_clock_ended_epoch_s": start + 1.1,
            },
        },
    )

    assert report["available"] is True
    assert report["all_arms_sampled"] is True
    assert report["arms"]["arm"]["sample_count"] == 2
    assert report["arms"]["arm"]["peak_memory_used_mib"] == 120
    assert report["arms"]["arm"]["peak_utilization_percent"] == 80


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
    assert table["overflow_latch_activation_count"] == 0
    assert table["overflow_latch_static_dense_batch_count"] == 0


def test_arm_performance_summary_preserves_canonical_local_timing_names() -> None:
    timing = {field: 0.0 for field in runner.LOCAL_TIMING_FIELDS}
    timing.update(
        {
            "em_time_s": 1.5,
            "big_jit_bucket_s": 0.75,
            "local_pack_s": 0.125,
            "local_noise_s": 0.0,
        }
    )

    summary = runner._arm_performance_summary(
        wall_s=2.0,
        estep_meta={"halfset_0_profile_summary": timing},
    )

    assert summary["local_em_time_s"] == 1.5
    assert summary["local_big_jit_bucket_s"] == 0.75
    assert summary["local_pack_s"] == 0.125
    assert summary["local_noise_s"] == 0.0
    assert not any(key.startswith("local_local_") for key in summary)


def test_local_timing_summary_fails_closed_on_missing_canonical_timer() -> None:
    timing = {field: 0.0 for field in runner.LOCAL_TIMING_FIELDS}
    del timing["local_pack_s"]

    with pytest.raises(RuntimeError, match="omitted canonical fields.*local_pack_s"):
        runner._local_timing_summary({"halfset_0_profile_summary": timing})


def test_same_state_incremental_gate_is_mirrored_repeated_and_backend_scoped() -> None:
    assert runner.PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER == (
        "abba_packed_deferred_1",
        "abba_packed_final_noise_1",
        "abba_packed_final_noise_2",
        "abba_packed_deferred_2",
    )
    assert runner.PACKED_FINAL_NOISE_INCREMENTAL_BAAB_ARM_ORDER == (
        "baab_packed_final_noise_1",
        "baab_packed_deferred_1",
        "baab_packed_deferred_2",
        "baab_packed_final_noise_2",
    )
    specs = runner._incremental_arm_specs()
    assert tuple(label for label, _backend in specs) == (
        "direct_oracle",
        *runner.PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER,
        *runner.PACKED_FINAL_NOISE_INCREMENTAL_BAAB_ARM_ORDER,
    )
    assert [backend for _label, backend in specs].count("direct") == 1
    assert [backend for _label, backend in specs].count("packed_deferred") == 4
    assert [backend for _label, backend in specs].count("packed_final_noise") == 4

    pairs = runner._incremental_pair_labels()
    assert len(pairs) == 16
    assert len(set(pairs)) == len(pairs)
    assert (
        "abba_packed_deferred_1",
        "abba_packed_final_noise_1",
    ) in pairs
    assert (
        "abba_packed_deferred_1",
        "baab_packed_deferred_1",
    ) in pairs
    assert (
        "abba_packed_final_noise_1",
        "baab_packed_final_noise_1",
    ) in pairs
    assert ("direct_oracle", "abba_packed_deferred_1") in pairs
    assert ("direct_oracle", "abba_packed_final_noise_1") in pairs


def _packed_final_noise_estep_meta(backend_mode: str) -> dict:
    enabled = backend_mode == "packed_final_noise"
    packed_deferred = backend_mode in {"packed_deferred", "packed_final_noise"}
    return {
        "halfset_0_profile_summary": {
            "chunk_padded_rotations": [13_824],
            "flat_local_rows_enabled": packed_deferred,
            "packed_local_projection_enabled": packed_deferred,
            "defer_packed_vdam_enabled": packed_deferred,
            "packed_vdam_reuses_flat_score_projection": packed_deferred,
            "packed_final_noise_enabled": enabled,
            "packed_vdam_avoids_dense_noise_rows": enabled,
            "packed_final_noise_preserves_dense_scalar_order": enabled,
            "sum_packed_final_noise_rows": 800 if enabled else 0,
        },
    }


@pytest.mark.parametrize(
    "backend_mode",
    ["direct", "packed_deferred", "packed_final_noise"],
)
def test_packed_final_noise_gate_proves_each_named_backend_profile(
    backend_mode: str,
) -> None:
    requested = runner._candidate_environment(
        backend_mode if backend_mode != "direct" else "packed_final_noise",
        enabled=backend_mode != "direct",
    )
    contract = runner._validate_arm_execution_contract(
        candidate_mode="packed_final_noise",
        candidate_enabled=backend_mode == "packed_final_noise",
        requested_environment=requested,
        effective_environment=dict(requested),
        estep_meta=_packed_final_noise_estep_meta(backend_mode),
        backend_mode=backend_mode,
    )

    assert contract["profile_checked"] is True
    assert contract["packed_final_noise"]["backend_mode"] == backend_mode
    assert contract["packed_final_noise"]["enabled"] is (
        backend_mode == "packed_final_noise"
    )
    assert contract["packed_final_noise"]["profile_exact"] is True


@pytest.mark.parametrize(
    ("declared_backend", "actual_backend", "match"),
    [
        ("packed_final_noise", "packed_deferred", "packed_final_noise_enabled"),
        ("packed_deferred", "direct", "flat_local_rows_enabled"),
        ("direct", "packed_deferred", "flat_local_rows_enabled"),
    ],
)
def test_packed_final_noise_gate_rejects_mislabeled_backend_profile(
    declared_backend: str,
    actual_backend: str,
    match: str,
) -> None:
    requested = runner._candidate_environment(
        declared_backend if declared_backend != "direct" else "packed_final_noise",
        enabled=declared_backend != "direct",
    )
    with pytest.raises(RuntimeError, match=match):
        runner._validate_arm_execution_contract(
            candidate_mode="packed_final_noise",
            candidate_enabled=declared_backend == "packed_final_noise",
            requested_environment=requested,
            effective_environment=dict(requested),
            estep_meta=_packed_final_noise_estep_meta(actual_backend),
            backend_mode=declared_backend,
        )


def test_packed_final_noise_gate_requires_observed_final_rows() -> None:
    requested = runner._candidate_environment("packed_final_noise", enabled=True)
    meta = _packed_final_noise_estep_meta("packed_final_noise")
    meta["halfset_0_profile_summary"]["sum_packed_final_noise_rows"] = 0

    with pytest.raises(RuntimeError, match="no packed final-noise rows"):
        runner._validate_arm_execution_contract(
            candidate_mode="packed_final_noise",
            candidate_enabled=True,
            requested_environment=requested,
            effective_environment=dict(requested),
            estep_meta=meta,
            backend_mode="packed_final_noise",
        )


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
        "direct_1,packed_final_noise_1,packed_final_noise_2,direct_2"
        in sbatch
    )
    assert (
        "direct_oracle,abba_packed_deferred_1,abba_packed_final_noise_1,"
        "abba_packed_final_noise_2,abba_packed_deferred_2,"
        "baab_packed_final_noise_1,baab_packed_deferred_1,"
        "baab_packed_deferred_2,baab_packed_final_noise_2"
        in sbatch
    )
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
    assert (
        "direct_1,compact_packed_deferred_1,compact_packed_deferred_2,direct_2"
        in sbatch
    )
    assert "direct_1,all_optimized_1,all_optimized_2,direct_2" in sbatch
    assert (
        "stable_all_off_1,stable_all_on_1,stable_all_on_2,stable_all_off_2"
        in sbatch
    )
    assert "all_optimized_stable_shapes_on_q32_batched" in source
    assert 'values[BATCHED_POSTERIOR_ENVIRONMENT] = "1"' in source
    assert "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_SCORES" in source
    assert "VDAM_SAME_STATE_FUSED_POSTERIOR_DUMP_ORIGINAL_INDEX" in sbatch
    assert (
        "single_translate_off_1,single_translate_on_1,"
        "single_translate_on_2,single_translate_off_2"
        in sbatch
    )
    assert (
        "compact_preprocess_off_1,compact_preprocess_on_1,"
        "compact_preprocess_on_2,compact_preprocess_off_2"
        in sbatch
    )
    assert (
        "pair_fine_off_1,pair_fine_on_1,pair_fine_on_2,pair_fine_off_2"
        in sbatch
    )
    assert "RECOVAR_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE" in source
    assert "RECOVAR_K1_RELION_EXACT_COMPACT_PREPROCESS" in source
    assert "profile_free_wall_timing" in sbatch
    assert "translate_score_call_count" in sbatch
    assert "generic_score_preprocess_count" in sbatch
    assert "exact_source_preprocess_count" in sbatch
    assert "generic_full_translation_count" in sbatch
    assert "raise_before_generic_score_fallback" in sbatch
    assert "RECOVAR_EXACT_LOCAL_FUSED_PAIR_FINE_SCORE" in source
    assert "FUSED_PAIR_FINE_SCORE_ENVIRONMENT in os.environ" in source
    assert "timed_arms_add_no_target_family_programs_after_prewarm" in sbatch
    assert "pair_pixel_gathers_materialized == false" in sbatch
    assert "stdbuf -oL nvidia-smi" in sbatch
    assert "RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR=0" in sbatch
    assert ".compact_effective == true" in sbatch
    assert ".packed_deferred_effective == true" in sbatch
    assert ".execution_contract.all_optimized_profile_exact == true" in sbatch
    assert "helpers/fourier_window.py" in sbatch
    assert "make -B -C \"${REPO_ROOT}/recovar/cuda\"" in sbatch
    assert '"${REPO_ROOT}/recovar/em/dense_single_volume/local_backprojection.py"' in sbatch
    assert "status --porcelain=v1 --untracked-files=all" in sbatch
    assert "VDAM_SAME_STATE_NOISE_SPLIT_DIAGNOSTICS" in sbatch
    assert "RECOVAR_NOISE_DEBUG_DUMP_DIR=${RUNTIME}/noise_split_enabled" in sbatch
    assert "VDAM_SAME_STATE_MIRRORED_INCREMENTAL_PANELS" in sbatch
    assert "--mirrored-incremental-panels" in source
    assert "for backend_mode in (\"packed_deferred\", \"packed_final_noise\")" in source
    assert "del warm\n            gc.collect()" in source
    assert "direct_oracle__vs__abba_packed_deferred_1" in sbatch
    assert "direct_oracle__vs__abba_packed_final_noise_1" in sbatch
    assert '.packed_final_noise.backend_mode == "direct"' in sbatch
    assert '.packed_final_noise.backend_mode == "packed_deferred"' in sbatch
    assert '.packed_final_noise.backend_mode == "packed_final_noise"' in sbatch
    assert "sum_packed_final_noise_rows] | all(. > 0)" in sbatch
    assert "VDAM_SAME_STATE_MIRRORED_HYBRID_IMAGE_BATCH_PANELS" in sbatch
    assert "--mirrored-hybrid-image-batch-panels" in source
    assert (
        "direct_oracle,abba_batch110_1,abba_batch200_1,abba_batch200_2,"
        "abba_batch110_2,baab_batch200_1,baab_batch110_1,"
        "baab_batch110_2,baab_batch200_2"
        in sbatch
    )
    assert "streamed_certificate_candidate_count_at_effective_batch == 24837120" in sbatch
    assert "streamed_certificate_candidate_count_at_effective_batch == 45158400" in sbatch
    assert "pass1_jax_cache_name_counts.txt" in sbatch
