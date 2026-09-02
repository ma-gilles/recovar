from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_vdam_coarse_combined_true200 as analyzer
from scripts import resolve_vdam_coarse_combined_true200_launch as resolver

LABELS = (
    "control_serial_1",
    "combined_candidate_1",
    "combined_candidate_2",
    "control_serial_2",
    "combined_candidate_3",
    "control_serial_3",
    "control_serial_4",
    "combined_candidate_4",
    "control_serial_5",
    "combined_candidate_5",
    "combined_candidate_6",
    "control_serial_6",
)
BLOCKS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
ITERATIONS = tuple(range(201))
THRESHOLDS = {
    "phase_segments": ((1, 59), (61, 159), (161, 200)),
    "phase_boundary_windows": ((59, 60, 61), (159, 160, 161)),
    "late_window": (180, 200),
    "centroid_max": 1.0,
    "cross_within_max": 1.25,
    "variance_inflation_max": 1.25,
    "permutation_p_min": 0.05,
    "no_growth_primary_family_endpoints": {
        "centroid_growth_across_declared_phases_boundaries": "centroid_separation:max_declared_excursion",
        "between_group_variance_growth_across_declared_phases_boundaries": "between_group_variance_rms_gap:max_declared_excursion",
    },
    "no_growth_familywise_alpha": 0.05,
    "no_growth_materiality_loo_multiplier": 1.0,
    "no_growth_normalized_numerical_noise_floor": 4.0 * np.finfo(np.float32).eps,
    "witness_map_normalized_distance_at_most": 4.0 * np.finfo(np.float32).eps,
}
HARNESS = Path(analyzer.__file__).with_name("run_vdam_coarse_combined_true200.sbatch")
ACCEPTANCE = Path(analyzer.__file__).with_name("vdam_coarse_combined_true200_acceptance.json")
SERIAL_INDICES = np.asarray([index for index, label in enumerate(LABELS) if "serial" in label])
LANE_INDICES = np.asarray([index for index, label in enumerate(LABELS) if "combined" in label])
CENTROID_GROWTH_FAMILY = "centroid_growth_across_declared_phases_boundaries"
VARIANCE_GROWTH_FAMILY = "between_group_variance_growth_across_declared_phases_boundaries"


def _base_maps() -> np.ndarray:
    grid = np.arange(8, dtype=np.float64).reshape(2, 2, 2) - 3.5
    return np.stack([grid * (1.0 + iteration / 500.0) for iteration in ITERATIONS])


def _map_panel(noise: np.ndarray) -> np.ndarray:
    base = _base_maps()
    return base[None, ...] + noise[..., None, None, None]


def _mismatch_history(values: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(values)
    assert values.shape[0] == len(LABELS)
    return [
        analyzer._exact_mismatch_squared_distances(values[:, checkpoint])
        for checkpoint in range(values.shape[1])
    ]


def _paired_joint_paths(checkpoints: int = 3) -> dict[str, list[np.ndarray]]:
    serial, lanes = analyzer._panel_indices(LABELS)
    key_a = np.empty((len(LABELS), checkpoints), dtype=np.int64)
    key_b = np.empty_like(key_a)
    star_image = np.empty((len(LABELS), checkpoints), dtype=object)
    star_half = np.empty_like(key_a)
    for mode, pair in enumerate(((serial[0], serial[1]), (serial[2], serial[3]), (serial[4], serial[5]))):
        for arm in pair:
            key_a[arm] = mode
            key_b[arm] = 10 + mode
            star_image[arm] = [f"mode-{mode}-cp-{value}" for value in range(checkpoints)]
            star_half[arm] = mode % 2 + 1
    for lane, witness in zip(lanes, serial, strict=True):
        key_a[lane] = key_a[witness]
        key_b[lane] = key_b[witness]
        star_image[lane] = star_image[witness]
        star_half[lane] = star_half[witness]
    return {
        "metadata:key_a": _mismatch_history(key_a),
        "metadata:key_b": _mismatch_history(key_b),
        "particle_star_identity:rlnImageName": _mismatch_history(star_image),
        "particle_star_identity:rlnRandomSubset": _mismatch_history(star_half),
    }


def _squared_path_distances(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    delta = positions[:, :, None] - positions[:, None, :]
    return np.square(delta)


def test_exact_map_equality_passes_without_requiring_bitwise_policy() -> None:
    maps = np.repeat(_base_maps()[None, ...], len(LABELS), axis=0)
    result = analyzer.analyze_map_panel(maps, LABELS, BLOCKS, ITERATIONS, thresholds=THRESHOLDS)
    assert result["pass"]
    assert result["trajectory_permutation"]["joint_max_one_sided_permutation_p"] == 1.0
    assert result["no_growth"]["joint_max_one_sided_permutation_p"] == 1.0
    assert result["no_growth"]["diagnostic_endpoint_count"] == 14
    assert result["no_growth"]["primary_familywise_gate"]["pass"]
    assert result["trajectory_permutation"]["n_blocked_whole_trajectory_permutations"] == 216


def test_unseen_011_path_fails_against_six_serial_complete_paths() -> None:
    serial, lanes = analyzer._panel_indices(LABELS)
    serial_paths = ("000", "001", "010", "100", "111", "111")
    paths = np.empty((len(LABELS), 3), dtype=np.int64)
    for arm, path in zip(serial, serial_paths, strict=True):
        paths[arm] = np.asarray(list(path), dtype=np.int64)
    for arm in lanes:
        paths[arm] = np.asarray([0, 1, 1])
    result = analyzer.classify_joint_exact_serial_witnesses(
        {"metadata:path": _mismatch_history(paths)}, LABELS
    )
    assert not result["pass"]
    assert not any(result["lane_has_complete_path_serial_witness"].values())


def test_cross_key_chimera_has_no_single_joint_serial_witness() -> None:
    history = _paired_joint_paths(checkpoints=1)
    serial, lanes = analyzer._panel_indices(LABELS)
    key_a = np.asarray([0, 0, 1, 1, 2, 2])
    key_b = np.asarray([0, 0, 1, 1, 2, 2])
    values_a = np.zeros((len(LABELS), 1), dtype=np.int64)
    values_b = np.zeros_like(values_a)
    for value, arm in zip(key_a, serial, strict=True):
        values_a[arm, 0] = value
    for value, arm in zip(key_b, serial, strict=True):
        values_b[arm, 0] = value
    for lane, witness in zip(lanes, serial, strict=True):
        values_a[lane] = values_a[witness]
        values_b[lane] = values_b[witness]
    values_a[lanes[0], 0] = 0
    values_b[lanes[0], 0] = 1
    history["metadata:key_a"] = _mismatch_history(values_a)
    history["metadata:key_b"] = _mismatch_history(values_b)
    result = analyzer.classify_joint_exact_serial_witnesses(history, LABELS)
    assert not result["pass"]
    assert not result["lane_has_complete_path_serial_witness"][LABELS[lanes[0]]]


def test_one_checkpoint_serial_switch_has_no_complete_path_witness() -> None:
    history = _paired_joint_paths(checkpoints=3)
    serial, lanes = analyzer._panel_indices(LABELS)
    values = np.zeros((len(LABELS), 3), dtype=np.int64)
    for mode, pair in enumerate(((serial[0], serial[1]), (serial[2], serial[3]), (serial[4], serial[5]))):
        values[list(pair)] = mode
    for lane, witness in zip(lanes, serial, strict=True):
        values[lane] = values[witness]
    values[lanes[0], 1] = values[serial[2], 1]
    history["metadata:switch"] = _mismatch_history(values)
    result = analyzer.classify_joint_exact_serial_witnesses(history, LABELS)
    assert not result["pass"]
    assert not result["lane_has_complete_path_serial_witness"][LABELS[lanes[0]]]


def test_lane_matching_one_serial_jointly_across_all_features_passes() -> None:
    result = analyzer.classify_joint_exact_serial_witnesses(
        _paired_joint_paths(), LABELS
    )
    assert result["pass"]
    assert all(result["serial_has_complete_path_peer"].values())
    assert all(result["lane_has_complete_path_serial_witness"].values())
    assert set(result["feature_checkpoint_counts"]) == {
        "metadata:key_a",
        "metadata:key_b",
        "particle_star_identity:rlnImageName",
        "particle_star_identity:rlnRandomSubset",
    }


def test_serial_control_without_complete_path_peer_fails() -> None:
    serial, lanes = analyzer._panel_indices(LABELS)
    values = np.zeros((len(LABELS), 1), dtype=np.int64)
    for value, arm in enumerate(serial):
        values[arm, 0] = value
    for lane, witness in zip(lanes, serial, strict=True):
        values[lane] = values[witness]
    result = analyzer.classify_joint_exact_serial_witnesses(
        {"metadata:key": _mismatch_history(values)}, LABELS
    )
    assert all(result["lane_has_complete_path_serial_witness"].values())
    assert not any(result["serial_has_complete_path_peer"].values())
    assert not result["pass"]


def test_nonfinite_exact_state_never_creates_a_serial_witness() -> None:
    assert not analyzer._values_equal(np.asarray([np.nan]), np.asarray([np.nan]))
    assert not analyzer._values_equal(np.asarray([np.inf]), np.asarray([np.inf]))


@pytest.mark.parametrize(
    "value",
    (
        {"outer": {"inner": [1.0, np.nan]}},
        {"outer": (1.0, complex(np.inf, 0.0))},
        np.asarray([{"inner": np.asarray([1.0, -np.inf])}], dtype=object),
    ),
)
def test_nested_nonfinite_exact_state_fails_closed(value: object) -> None:
    assert analyzer._has_nonfinite_numeric(value)
    assert not analyzer._values_equal(value, value)


def test_recursive_exact_state_retains_ordinary_string_semantics() -> None:
    left = {"outer": ["particle@stack.mrcs", {"mode": "C1"}]}
    assert analyzer._values_equal(left, {"outer": ["particle@stack.mrcs", {"mode": "C1"}]})
    assert not analyzer._values_equal(left, {"outer": ["particle@stack.mrcs", {"mode": "c1"}]})


def _minimal_particle_state_contract() -> dict[str, object]:
    return {
        "star_identity_columns": ["rlnImageName", "rlnRandomSubset"],
        "star_numeric_groups": {},
    }


def test_particle_star_identity_is_canonical_and_strict() -> None:
    table = pd.DataFrame(
        {
            "rlnImageName": ["000002@particles.mrcs", "000001@particles.mrcs"],
            "rlnRandomSubset": [2.0, 1],
        }
    )
    result = analyzer._particle_table_values(
        table,
        _minimal_particle_state_contract(),
        "valid",
    )
    assert result["identity_columns"]["rlnImageName"].tolist() == [
        "000001@particles.mrcs",
        "000002@particles.mrcs",
    ]
    assert result["identity_columns"]["rlnRandomSubset"].tolist() == [1, 2]


@pytest.mark.parametrize(
    ("image_names", "random_subsets", "message"),
    (
        (["", "000002@particles.mrcs"], [1, 2], "empty or non-finite"),
        ([None, "000002@particles.mrcs"], [1, 2], "empty or non-finite"),
        (["same", "same"], [1, 2], "duplicate image identities"),
        (["one", "two"], [1, np.nan], "non-finite"),
        (["one", "two"], [1, 1.5], "not integral"),
        (["one", "two"], [0, 2], "outside the RELION halfset domain"),
        (["one", "two"], [1, 3], "outside the RELION halfset domain"),
    ),
)
def test_particle_star_identity_corruption_fails_closed(
    image_names: list[object],
    random_subsets: list[object],
    message: str,
) -> None:
    table = pd.DataFrame(
        {"rlnImageName": image_names, "rlnRandomSubset": random_subsets}
    )
    with pytest.raises(analyzer.GateSetupError, match=message):
        analyzer._particle_table_values(
            table,
            _minimal_particle_state_contract(),
            "corrupt",
        )


def _selector_audit(*, workers: int, atomic: bool) -> dict[str, object]:
    multistream = workers > 0
    fused_calls = 3
    return {
        "score_mode": "gaussian",
        "translation_count": 29,
        "requested_fused": True,
        "effective_fused": True,
        "requested_workers": workers,
        "effective_workers": workers,
        "requested_atomic": atomic,
        "effective_atomic": atomic,
        "wrapper": (
            "relion_coarse_diff2_projector_multistream_f32"
            if multistream
            else "relion_coarse_diff2_projector_f32"
        ),
        "target": (
            "cuda_relion_coarse_diff2_projector_multistream_f32"
            if multistream
            else "cuda_relion_coarse_diff2_projector_f32"
        ),
        "counts": {
            "fused_calls": fused_calls,
            "actual_rows": 17,
            "multistream_calls": fused_calls if multistream else 0,
            "native_atomic_selected_calls": fused_calls if atomic else 0,
        },
    }


def _joint_stream_metadata(audit: dict[str, object]) -> dict[str, object]:
    return {
        "n_translations": 116,
        "oversampling": 1,
        "halfset_ids": [0, 1],
        "joint_halfset_particle_stream": True,
        "halfset_0_profile_summary": {"coarse_selector_audit": audit},
    }


def _inactive_selector_audit() -> dict[str, object]:
    audit = _selector_audit(workers=0, atomic=False)
    audit.update(
        requested_fused=False,
        effective_fused=False,
        wrapper=None,
        target=None,
        counts={
            "fused_calls": 0,
            "actual_rows": 0,
            "multistream_calls": 0,
            "native_atomic_selected_calls": 0,
        },
    )
    return audit


def _hybrid_stats(
    *,
    selected_images: int = 1_000,
    fallback_images: int = 0,
    rotation_count: int = 36_864,
) -> dict[str, object]:
    selected_batches = 6 if selected_images else 0
    fallback_batches = 6 - selected_batches
    selected_candidates = selected_images * 16 * 29
    full_candidates = selected_images * rotation_count * 29
    return {
        "enabled": True,
        "default_enabled": False,
        "published_score_source": "exact_relion_source16_or_full_rectangular",
        "expanded_gemm_scores_published": False,
        "whole_batch_fail_closed_fallback": True,
        "batch_count": 6,
        "selected_rescore_batch_count": selected_batches,
        "fallback_batch_count": fallback_batches,
        "selected_rescore_image_count": selected_images,
        "fallback_image_count": fallback_images,
        "selected_source16_block_count": selected_images,
        "selected_exact_candidate_count": selected_candidates,
        "full_candidate_count_for_selected_images": full_candidates,
        "selected_exact_candidate_fraction": (
            selected_candidates / full_candidates if selected_images else None
        ),
        "max_selected_blocks_per_image": 1 if selected_images else 0,
        "selected_block_capacity": 64,
        "certificate_chunk_rows": 4_608,
        "certificate_chunk_count_per_batch": (rotation_count + 4_607) // 4_608,
        "topology_full_to_compact_sha256": "a" * 64,
        "fallback_reasons": ({"capacity_overflow": fallback_batches} if fallback_batches else {}),
    }


def _direct_or_hybrid_metadata(*, hybrid: bool) -> dict[str, object]:
    metadata = _joint_stream_metadata(_inactive_selector_audit())
    metadata["selected_particle_ids"] = list(range(1_000))
    if hybrid:
        metadata["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"] = _hybrid_stats()
    return metadata


def test_direct_hybrid_profile_proof_accepts_exact_score_accounting() -> None:
    direct_rows = analyzer._validate_coarse_selector_profile_audits(
        _direct_or_hybrid_metadata(hybrid=False),
        label="control_serial_1",
        iteration=181,
        multistream_workers=0,
        native_atomic_reduction=0,
        coarse_gemm_hybrid=0,
    )
    hybrid_rows = analyzer._validate_coarse_selector_profile_audits(
        _direct_or_hybrid_metadata(hybrid=True),
        label="combined_candidate_1",
        iteration=181,
        multistream_workers=0,
        native_atomic_reduction=0,
        coarse_gemm_hybrid=1,
    )
    assert direct_rows[0]["hybrid"] is None
    assert hybrid_rows[0]["hybrid"]["published_score_source"] == (
        "exact_relion_source16_or_full_rectangular"
    )
    assert hybrid_rows[0]["hybrid"]["selected_rescore_image_count"] == 1_000
    assert hybrid_rows[0]["hybrid"]["inferred_rotation_count"] == 36_864


def test_hybrid_profile_proof_accepts_accounted_full_direct_fallback() -> None:
    metadata = _direct_or_hybrid_metadata(hybrid=True)
    metadata["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"] = (
        _hybrid_stats(selected_images=0, fallback_images=1_000)
    )
    rows = analyzer._validate_coarse_selector_profile_audits(
        metadata,
        label="combined_candidate_1",
        iteration=181,
        multistream_workers=0,
        native_atomic_reduction=0,
        coarse_gemm_hybrid=1,
    )
    assert rows[0]["hybrid"]["selected_rescore_image_count"] == 0
    assert rows[0]["hybrid"]["fallback_image_count"] == 1_000
    assert rows[0]["hybrid"]["fallback_reasons"] == {"capacity_overflow": 6}
    assert rows[0]["hybrid"]["inferred_rotation_count"] is None


def test_completed_hybrid_arm_proves_every_checkpoint_and_maximum_cache_shape(
    tmp_path: Path,
) -> None:
    label = "combined_candidate_1"
    output = tmp_path / "runs" / label / "output"
    output.mkdir(parents=True)
    for iteration, rotation_count in ((1, 36_864), (2, 294_912)):
        metadata = _direct_or_hybrid_metadata(hybrid=True)
        metadata["current_size"] = 100 if iteration == 1 else 128
        metadata["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"] = (
            _hybrid_stats(rotation_count=rotation_count)
        )
        (output / f"run_it{iteration:03d}_recovar_meta.json").write_text(
            json.dumps(metadata)
        )
    result = analyzer._validate_arm_coarse_hybrid_execution(
        tmp_path,
        label,
        multistream_workers=0,
        native_atomic_reduction=0,
        coarse_gemm_hybrid=1,
        iterations=(1, 2),
        cache_contract={
            "maximum_declared_rotation_count": 294_912,
            "maximum_declared_compact_pixel_count": 8_320,
        },
    )
    assert result["checkpoint_count"] == 2
    assert result["maximum_inferred_rotation_count"] == 294_912
    assert result["maximum_compact_pixel_count"] == 8_320
    assert result["total_fallback_image_count"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("expanded_gemm_scores_published", True, "hybrid contract differs"),
        ("selected_block_capacity", 63, "hybrid contract differs"),
        ("fallback_image_count", 1, "hybrid image accounting differs"),
        ("selected_exact_candidate_fraction", 0.5, "selected-rescore accounting differs"),
        ("certificate_chunk_count_per_batch", 7, "selected-rescore accounting differs"),
        ("topology_full_to_compact_sha256", "bad", "topology digest is invalid"),
    ),
)
def test_direct_hybrid_profile_proof_rejects_unsafe_telemetry(
    field: str,
    value: object,
    message: str,
) -> None:
    metadata = _direct_or_hybrid_metadata(hybrid=True)
    metadata["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"][field] = value
    with pytest.raises(analyzer.GateSetupError, match=message):
        analyzer._validate_coarse_selector_profile_audits(
            metadata,
            label="combined_candidate_1",
            iteration=181,
            multistream_workers=0,
            native_atomic_reduction=0,
            coarse_gemm_hybrid=1,
        )


def test_direct_profile_rejects_hybrid_telemetry() -> None:
    with pytest.raises(analyzer.GateSetupError, match="direct control published hybrid telemetry"):
        analyzer._validate_coarse_selector_profile_audits(
            _direct_or_hybrid_metadata(hybrid=True),
            label="control_serial_1",
            iteration=181,
            multistream_workers=0,
            native_atomic_reduction=0,
            coarse_gemm_hybrid=0,
        )


def test_selector_proof_accepts_serial_control_and_combined_candidate() -> None:
    control_audit = _selector_audit(workers=0, atomic=False)
    candidate_audit = _selector_audit(workers=8, atomic=True)
    control = _joint_stream_metadata(control_audit)
    candidate = _joint_stream_metadata(candidate_audit)
    control_rows = analyzer._validate_coarse_selector_profile_audits(
        control,
        label="control_serial_1",
        iteration=1,
        multistream_workers=0,
        native_atomic_reduction=0,
    )
    candidate_rows = analyzer._validate_coarse_selector_profile_audits(
        candidate,
        label="combined_candidate_1",
        iteration=1,
        multistream_workers=8,
        native_atomic_reduction=1,
    )
    assert len(control_rows) == len(candidate_rows) == 1
    assert control_rows[0]["profile_halfset"] == 0
    assert control_rows[0]["joint_halfset_ids"] == [0, 1]
    assert control_rows[0]["joint_halfset_particle_stream"] is True
    assert control_rows[0]["audit"]["counts"]["fused_calls"] > 0
    assert control_rows[0]["audit"]["counts"]["multistream_calls"] == 0
    assert candidate_rows[0]["audit"]["counts"]["native_atomic_selected_calls"] > 0


def test_selector_proof_rejects_coarse_translation_count_mismatch() -> None:
    audit = _selector_audit(workers=8, atomic=True)
    metadata = _joint_stream_metadata(audit)
    metadata["n_translations"] = 120
    with pytest.raises(analyzer.GateSetupError, match="selector translation count differs"):
        analyzer._validate_coarse_selector_profile_audits(
            metadata,
            label="combined_candidate_1",
            iteration=181,
            multistream_workers=8,
            native_atomic_reduction=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("n_translations", 116.0, "n_translations must be an integer"),
        ("n_translations", 0, "n_translations must be positive"),
        ("n_translations", -116, "n_translations must be positive"),
        ("n_translations", 117, "not divisible by its oversampling factor"),
        ("oversampling", -1, "oversampling must be non-negative"),
        ("oversampling", 9, "oversampling is implausibly large"),
    ),
)
def test_selector_proof_rejects_invalid_sampling_plan(
    field: str,
    value: object,
    message: str,
) -> None:
    metadata = _joint_stream_metadata(_selector_audit(workers=0, atomic=False))
    metadata[field] = value
    with pytest.raises(analyzer.GateSetupError, match=message):
        analyzer._validate_coarse_selector_profile_audits(
            metadata,
            label="control_serial_1",
            iteration=181,
            multistream_workers=0,
            native_atomic_reduction=0,
        )


@pytest.mark.parametrize("field", ("n_translations", "oversampling"))
def test_selector_proof_rejects_missing_sampling_plan_field(field: str) -> None:
    metadata = _joint_stream_metadata(_selector_audit(workers=0, atomic=False))
    metadata.pop(field)
    with pytest.raises(analyzer.GateSetupError, match=rf"{field} must be an integer"):
        analyzer._validate_coarse_selector_profile_audits(
            metadata,
            label="control_serial_1",
            iteration=181,
            multistream_workers=0,
            native_atomic_reduction=0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("requested_fused", False),
        ("effective_workers", 0),
        ("wrapper", "relion_coarse_diff2_projector_f32"),
        ("target", "cuda_relion_coarse_diff2_projector_f32"),
    ),
)
def test_selector_proof_rejects_requested_effective_or_target_mutation(
    field: str,
    value: object,
) -> None:
    audit = _selector_audit(workers=8, atomic=True)
    audit[field] = value
    metadata = _joint_stream_metadata(audit)
    with pytest.raises(analyzer.GateSetupError, match="coarse selector"):
        analyzer._validate_coarse_selector_profile_audits(
            metadata,
            label="combined_candidate_1",
            iteration=42,
            multistream_workers=8,
            native_atomic_reduction=1,
        )


def test_selector_proof_validates_joint_stream_profile_counts() -> None:
    bad = _selector_audit(workers=8, atomic=True)
    bad["counts"] = dict(bad["counts"], actual_rows=0)
    metadata = _joint_stream_metadata(bad)
    with pytest.raises(analyzer.GateSetupError, match="halfset_0_profile_summary"):
        analyzer._validate_coarse_selector_profile_audits(
            metadata,
            label="combined_candidate_1",
            iteration=200,
            multistream_workers=8,
            native_atomic_reduction=1,
        )


@pytest.mark.parametrize("joint_flag", ("missing", False))
def test_selector_proof_rejects_missing_or_false_joint_stream_flag(joint_flag: object) -> None:
    metadata = _joint_stream_metadata(_selector_audit(workers=0, atomic=False))
    if joint_flag == "missing":
        metadata.pop("joint_halfset_particle_stream")
    else:
        metadata["joint_halfset_particle_stream"] = joint_flag
    with pytest.raises(analyzer.GateSetupError, match="not a joint-halfset particle stream"):
        analyzer._validate_coarse_selector_profile_audits(
            metadata,
            label="control_serial_1",
            iteration=1,
            multistream_workers=0,
            native_atomic_reduction=0,
        )


@pytest.mark.parametrize("halfset_ids", ([], [0], [1, 0], [0, 2]))
def test_selector_proof_rejects_wrong_joint_halfset_ids(halfset_ids: list[int]) -> None:
    metadata = _joint_stream_metadata(_selector_audit(workers=0, atomic=False))
    metadata["halfset_ids"] = halfset_ids
    with pytest.raises(analyzer.GateSetupError, match="joint-halfset IDs differ"):
        analyzer._validate_coarse_selector_profile_audits(
            metadata,
            label="control_serial_1",
            iteration=1,
            multistream_workers=0,
            native_atomic_reduction=0,
        )


@pytest.mark.parametrize("profile_mutation", ("missing", "extra"))
def test_selector_proof_rejects_missing_or_extra_halfset_profile(profile_mutation: str) -> None:
    metadata = _joint_stream_metadata(_selector_audit(workers=0, atomic=False))
    if profile_mutation == "missing":
        metadata.pop("halfset_0_profile_summary")
    else:
        metadata["halfset_1_profile_summary"] = {
            "coarse_selector_audit": _selector_audit(workers=0, atomic=False)
        }
    with pytest.raises(analyzer.GateSetupError, match="halfset profile topology differs"):
        analyzer._validate_coarse_selector_profile_audits(
            metadata,
            label="control_serial_1",
            iteration=1,
            multistream_workers=0,
            native_atomic_reduction=0,
        )


def test_stable_tiny_map_noise_passes_its_joint_exact_witnesses() -> None:
    exact = analyzer.classify_joint_exact_serial_witnesses(
        _paired_joint_paths(checkpoints=3), LABELS
    )
    rng = np.random.default_rng(20)
    positions = rng.normal(scale=1e-8, size=(201, len(LABELS)))
    result = analyzer.classify_witness_coupled_map_trajectory(
        _squared_path_distances(positions),
        LABELS,
        exact,
        maximum_normalized_distance=THRESHOLDS["witness_map_normalized_distance_at_most"],
    )
    assert np.any(_squared_path_distances(positions) > 0.0)
    assert result["pass"]


def test_map_matching_different_serial_than_joint_exact_state_fails() -> None:
    exact = analyzer.classify_joint_exact_serial_witnesses(
        _paired_joint_paths(checkpoints=3), LABELS
    )
    serial, lanes = analyzer._panel_indices(LABELS)
    positions = np.empty((201, len(LABELS)), dtype=np.float64)
    for mode, pair in enumerate(((serial[0], serial[1]), (serial[2], serial[3]), (serial[4], serial[5]))):
        positions[:, list(pair)] = float(mode) * 1e-3
    for lane, witness in zip(lanes, serial, strict=True):
        positions[:, lane] = positions[:, witness]
    positions[:, lanes[0]] = positions[:, serial[2]]
    result = analyzer.classify_witness_coupled_map_trajectory(
        _squared_path_distances(positions),
        LABELS,
        exact,
        maximum_normalized_distance=THRESHOLDS["witness_map_normalized_distance_at_most"],
    )
    assert not result["combined_candidates"][LABELS[lanes[0]]]["pass"]
    assert not result["pass"]


def test_single_checkpoint_map_spike_cannot_hide_inside_small_trajectory_rms() -> None:
    exact = analyzer.classify_joint_exact_serial_witnesses(
        _paired_joint_paths(checkpoints=3), LABELS
    )
    _, lanes = analyzer._panel_indices(LABELS)
    tolerance = THRESHOLDS["witness_map_normalized_distance_at_most"]
    positions = np.zeros((201, len(LABELS)), dtype=np.float64)
    positions[100, lanes[0]] = 2.0 * tolerance
    result = analyzer.classify_witness_coupled_map_trajectory(
        _squared_path_distances(positions),
        LABELS,
        exact,
        maximum_normalized_distance=tolerance,
    )
    row = result["combined_candidates"][LABELS[lanes[0]]]["exact_state_witnesses"][0]
    assert row["whole_trajectory_rms_normalized_distance"] < tolerance
    assert row["maximum_checkpoint_normalized_distance"] > tolerance
    assert not result["pass"]


def test_exchangeable_null_is_not_rejected_by_pointwise_conjunction() -> None:
    rng = np.random.default_rng(20)
    noise = rng.normal(scale=1e-7, size=(len(LABELS), len(ITERATIONS)))
    result = analyzer.analyze_map_panel(_map_panel(noise), LABELS, BLOCKS, ITERATIONS, thresholds=THRESHOLDS)
    assert result["pass"]
    assert not result["pointwise_serial_leave_one_out_diagnostic_only"]["used_for_candidate_acceptance"]


def _growth_panel(alternative: str) -> dict[str, object]:
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=1e-7, size=(len(LABELS), len(ITERATIONS)))
    lane_indices = np.asarray([index for index, label in enumerate(LABELS) if "combined" in label])
    if alternative == "final_shift":
        noise[lane_indices, -1] += 2e-6
    elif alternative == "directional_ramp":
        ramp = np.maximum(np.asarray(ITERATIONS) - 160, 0) * 5e-8
        noise[lane_indices] += ramp
    elif alternative == "variance_growth":
        coefficients = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        ramp = np.maximum(np.asarray(ITERATIONS) - 160, 0) * 5e-8
        noise[lane_indices] += coefficients[:, None] * ramp[None, :]
    else:
        raise AssertionError(alternative)
    result = analyzer.analyze_map_panel(_map_panel(noise), LABELS, BLOCKS, ITERATIONS, thresholds=THRESHOLDS)
    return result


def test_final_only_lane_shift_above_numerical_floor_fails_no_growth() -> None:
    result = _growth_panel("final_shift")
    assert not result["no_growth"]["pass"]
    family = result["no_growth"]["primary_familywise_gate"]["families"]
    assert not family[CENTROID_GROWTH_FAMILY]["pass"]


def test_directional_centroid_ramp_fails_no_growth() -> None:
    result = _growth_panel("directional_ramp")
    assert not result["no_growth"]["pass"]
    family = result["no_growth"]["primary_familywise_gate"]["families"]
    assert not family[CENTROID_GROWTH_FAMILY]["pass"]


def test_nondirectional_lane_variance_growth_fails_no_growth() -> None:
    result = _growth_panel("variance_growth")
    assert not result["no_growth"]["pass"]
    family = result["no_growth"]["primary_familywise_gate"]["families"]
    assert not family[VARIANCE_GROWTH_FAMILY]["pass"]
    endpoint = result["no_growth"]["endpoints"]["lane_variance_excess_rms:late_growth_180_200"]
    assert endpoint["observed"] > 0.0


def _trajectory_grams(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=1, keepdims=True)
    return np.einsum("tid,tjd->tij", centered, centered)


def _classify_no_growth_grams(grams: np.ndarray, serial: np.ndarray = SERIAL_INDICES) -> dict[str, object]:
    return analyzer.classify_no_growth(
        grams,
        ITERATIONS,
        blocks=BLOCKS,
        observed_serial_indices=tuple(int(value) for value in serial),
        phase_segments=THRESHOLDS["phase_segments"],
        phase_boundary_windows=THRESHOLDS["phase_boundary_windows"],
        late_window=THRESHOLDS["late_window"],
        primary_family_endpoints=THRESHOLDS["no_growth_primary_family_endpoints"],
        familywise_alpha=THRESHOLDS["no_growth_familywise_alpha"],
        materiality_loo_multiplier=THRESHOLDS["no_growth_materiality_loo_multiplier"],
        normalized_numerical_noise_floor=THRESHOLDS[
            "no_growth_normalized_numerical_noise_floor"
        ],
    )


def _higher_dimensional_growth_grams(alternative: str, *, mask_aggregate: bool) -> np.ndarray:
    rng = np.random.default_rng(7319)
    points = rng.normal(scale=1e-7, size=(len(ITERATIONS), len(LABELS), 3))
    final_distances = np.linalg.norm(points[-1, :, None] - points[-1, None, :], axis=-1)
    serial_loo_radius = max(
        min(final_distances[index, peer] for peer in SERIAL_INDICES if peer != index) for index in SERIAL_INDICES
    )
    if mask_aggregate:
        # This balanced nuisance split is orthogonal to the sealed labels. It
        # dominates the aggregate Gram while leaving both primary contrasts
        # unchanged, so the terminal family is independently necessary.
        nuisance_group = np.asarray((0, 1, 4, 5, 8, 9))
        points[:200, nuisance_group, 2] += 20.0 * serial_loo_radius
    terminal_amplitude = 10.0 * serial_loo_radius
    ramp = np.maximum(np.asarray(ITERATIONS) - 160, 0) / 40.0 * terminal_amplitude
    if alternative == "final_shift":
        points[-1, LANE_INDICES, 0] += terminal_amplitude
    elif alternative == "directional_ramp":
        points[:, LANE_INDICES, 0] += ramp[:, None]
    elif alternative == "variance_growth":
        coefficients = np.asarray((-3.0, -2.0, -1.0, 1.0, 2.0, 3.0))
        coefficients /= np.sqrt(np.mean(np.square(coefficients)))
        points[:, LANE_INDICES, 0] += ramp[:, None] * coefficients[None, :]
    else:
        raise AssertionError(alternative)
    return _trajectory_grams(points)


@pytest.mark.parametrize(
    ("alternative", "family_name", "old_joint_p"),
    (
        ("final_shift", CENTROID_GROWTH_FAMILY, 29.0 / 216.0),
        ("directional_ramp", CENTROID_GROWTH_FAMILY, 29.0 / 216.0),
        ("variance_growth", VARIANCE_GROWTH_FAMILY, 14.0 / 216.0),
    ),
)
def test_primary_families_close_higher_dimensional_joint_max_escape(
    alternative: str,
    family_name: str,
    old_joint_p: float,
) -> None:
    result = _classify_no_growth_grams(_higher_dimensional_growth_grams(alternative, mask_aggregate=False))
    family = result["primary_familywise_gate"]["families"][family_name]
    assert result["joint_max_used_for_acceptance"] is False
    assert result["joint_max_one_sided_permutation_p"] == pytest.approx(old_joint_p)
    assert family["exact_blocked_one_sided_permutation_p"] == pytest.approx(2.0 / 216.0)
    assert not family["pass"]
    assert not result["pass"]


def test_sub_repeat_radius_terminal_difference_is_bounded_numerical_noise() -> None:
    rng = np.random.default_rng(7319)
    points = rng.normal(scale=1e-7, size=(len(ITERATIONS), len(LABELS), 3))
    baseline_grams = _trajectory_grams(points)
    baseline_radius = analyzer._leave_one_out_radius(baseline_grams[180], SERIAL_INDICES)
    threshold = max(baseline_radius, THRESHOLDS["no_growth_normalized_numerical_noise_floor"])
    points[-1, LANE_INDICES, 0] += 0.5 * threshold

    result = _classify_no_growth_grams(_trajectory_grams(points))
    family = result["primary_familywise_gate"]["families"][CENTROID_GROWTH_FAMILY]
    assert family["exact_blocked_one_sided_permutation_p"] == pytest.approx(2.0 / 216.0)
    assert family["statistically_directional"]
    assert not family["materially_larger_than_repeat_noise"]
    assert family["pass"]
    materiality = family["materiality_endpoints"]["late_excursion_180_200"]
    assert materiality["serial_control_leave_one_out_radius"] == pytest.approx(baseline_radius)
    assert materiality["threshold"] == pytest.approx(threshold)
    assert not materiality["materially_larger_than_repeat_noise"]


@pytest.mark.parametrize("amplitude", (1e-15, 1e-9))
def test_zero_repeat_radius_microscopic_boundary_shift_is_numerical_noise(
    amplitude: float,
) -> None:
    points = np.zeros((len(ITERATIONS), len(LABELS), 1), dtype=np.float64)
    points[60, LANE_INDICES, 0] = amplitude

    result = _classify_no_growth_grams(_trajectory_grams(points))
    family = result["primary_familywise_gate"]["families"][CENTROID_GROWTH_FAMILY]
    boundary = family["materiality_endpoints"]["boundary_excursion_59_60_61"]
    assert family["exact_blocked_one_sided_permutation_p"] == pytest.approx(2.0 / 216.0)
    assert family["statistically_directional"]
    assert boundary["serial_control_leave_one_out_radius"] == 0.0
    assert boundary["observed"] == pytest.approx(amplitude)
    assert not boundary["materially_larger_than_repeat_noise"]
    assert family["pass"]
    assert result["pass"]


def test_material_boundary_spike_cannot_hide_outside_late_window() -> None:
    points = np.zeros((len(ITERATIONS), len(LABELS), 1), dtype=np.float64)
    points[60, LANE_INDICES, 0] = 1e-5

    result = _classify_no_growth_grams(_trajectory_grams(points))
    family = result["primary_familywise_gate"]["families"][CENTROID_GROWTH_FAMILY]
    boundary = family["materiality_endpoints"]["boundary_excursion_59_60_61"]
    assert family["exact_blocked_one_sided_permutation_p"] == pytest.approx(2.0 / 216.0)
    assert family["statistically_directional"]
    assert boundary["materially_larger_than_repeat_noise"]
    assert not family["pass"]
    assert not result["pass"]


def test_candidate_contamination_and_orthogonal_shift_cannot_mask_growth() -> None:
    points = np.zeros((len(ITERATIONS), len(LABELS), 3), dtype=np.float64)
    nuisance_group = np.asarray((0, 1, 4, 5, 8, 9))
    points[:, nuisance_group, 2] += 20.0
    points[180, LANE_INDICES[0], 1] += 1.0
    points[200, LANE_INDICES[0], 1] += 1.0
    points[200, LANE_INDICES, 0] += 1.1
    grams = _trajectory_grams(points)

    aggregate = analyzer._blocked_permutation_statistics(np.mean(grams, axis=0), BLOCKS, SERIAL_INDICES)
    observed = aggregate["observed"]
    assert observed["centroid_over_pooled_run_rms"] <= THRESHOLDS["centroid_max"]
    assert observed["cross_within_median_distance_ratio"] <= THRESHOLDS["cross_within_max"]
    assert observed["lane_variance_inflation"] <= THRESHOLDS["variance_inflation_max"]
    assert aggregate["joint_max_one_sided_permutation_p"] >= THRESHOLDS["permutation_p_min"]

    result = _classify_no_growth_grams(grams)
    family = result["primary_familywise_gate"]["families"][CENTROID_GROWTH_FAMILY]
    late = family["materiality_endpoints"]["late_excursion_180_200"]
    assert family["exact_blocked_one_sided_permutation_p"] == pytest.approx(2.0 / 216.0)
    assert late["serial_control_leave_one_out_radius"] == 0.0
    assert late["observed"] == pytest.approx(1.1)
    assert late["materially_larger_than_repeat_noise"]
    assert not family["pass"]
    assert not result["pass"]


def test_excursion_energy_is_stable_at_large_finite_scale() -> None:
    assert analyzer._positive_energy_increment(1e300, 0.0) == 1e300
    close_reference = np.nextafter(1e300, 0.0)
    increment = analyzer._positive_energy_increment(1e300, close_reference)
    assert np.isfinite(increment)
    assert 0.0 < increment < 1e300
    assert analyzer._positive_energy_increment(1e300, 1e300) == 0.0


@pytest.mark.parametrize(
    ("alternative", "family_name"),
    (
        ("final_shift", CENTROID_GROWTH_FAMILY),
        ("directional_ramp", CENTROID_GROWTH_FAMILY),
        ("variance_growth", VARIANCE_GROWTH_FAMILY),
    ),
)
def test_primary_families_reject_when_balanced_nuisance_masks_aggregate_gate(
    alternative: str,
    family_name: str,
) -> None:
    grams = _higher_dimensional_growth_grams(alternative, mask_aggregate=True)
    aggregate = analyzer._blocked_permutation_statistics(np.mean(grams, axis=0), BLOCKS, SERIAL_INDICES)
    observed = aggregate["observed"]
    assert observed["centroid_over_pooled_run_rms"] <= THRESHOLDS["centroid_max"]
    assert observed["cross_within_median_distance_ratio"] <= THRESHOLDS["cross_within_max"]
    assert observed["lane_variance_inflation"] <= THRESHOLDS["variance_inflation_max"]
    assert aggregate["joint_max_one_sided_permutation_p"] >= THRESHOLDS["permutation_p_min"]

    result = _classify_no_growth_grams(grams)
    family = result["primary_familywise_gate"]["families"][family_name]
    assert family["exact_blocked_one_sided_permutation_p"] == pytest.approx(2.0 / 216.0)
    assert not family["pass"]
    assert not result["pass"]


def test_primary_no_growth_fwer_is_calibrated_across_48_exchangeable_panels() -> None:
    rejected = 0
    for seed in range(10_000, 10_048):
        rng = np.random.default_rng(seed)
        points = rng.normal(scale=1e-7, size=(len(ITERATIONS), len(LABELS), 4))
        result = _classify_no_growth_grams(_trajectory_grams(points))
        gate = result["primary_familywise_gate"]
        assert gate["method"] == (
            "bonferroni_two_exact_blocked_max_excursion_families_with_control_materiality"
        )
        assert gate["familywise_alpha"] == 0.05
        assert gate["per_family_alpha"] == 0.025
        assert gate["materiality"]["failure_requires_significance_and_materiality"]
        rejected += int(not gate["pass"])
    # Five is the 95% binomial upper-tail cutoff for 48 draws at p=0.05.
    assert rejected <= 5


def test_primary_no_growth_statistics_are_exactly_complement_symmetric() -> None:
    grams = _higher_dimensional_growth_grams("variance_growth", mask_aggregate=True)
    serial = _classify_no_growth_grams(grams, SERIAL_INDICES)
    complement = _classify_no_growth_grams(grams, LANE_INDICES)
    serial_families = serial["primary_familywise_gate"]["families"]
    complement_families = complement["primary_familywise_gate"]["families"]
    for family_name in (CENTROID_GROWTH_FAMILY, VARIANCE_GROWTH_FAMILY):
        assert serial_families[family_name]["observed"] == complement_families[family_name]["observed"]
        assert (
            serial_families[family_name]["exact_blocked_one_sided_permutation_p"]
            == complement_families[family_name]["exact_blocked_one_sided_permutation_p"]
        )
    assert serial["primary_familywise_gate"]["permutation_statistic_label_and_complement_symmetric"]


def test_checkpoint_map_scale_is_all_arm_pooled_and_permutation_invariant() -> None:
    rng = np.random.default_rng(31)
    noise = rng.normal(scale=1e-7, size=(len(LABELS), len(ITERATIONS)))
    maps = _map_panel(noise)
    original = analyzer.analyze_map_panel(maps, LABELS, BLOCKS, ITERATIONS, thresholds=THRESHOLDS)
    permutation = np.asarray((3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8))
    permuted_labels = tuple(LABELS[index] for index in permutation)
    permuted = analyzer.analyze_map_panel(
        maps[permutation],
        permuted_labels,
        BLOCKS,
        ITERATIONS,
        thresholds=THRESHOLDS,
    )
    original_scales = [row["pooled_all_arm_map_norm_scale"] for row in original["checkpoints"]]
    permuted_scales = [row["pooled_all_arm_map_norm_scale"] for row in permuted["checkpoints"]]
    assert original_scales == permuted_scales


def test_discrete_escape_fails_but_continuous_pointwise_loo_is_diagnostic() -> None:
    contract = {
        "joint_exact_complete_path_metadata_keys": ["selected_particle_ids"],
        "serial_leave_one_out_numeric_keys": ["continuous"],
        "serial_leave_one_out_multiplier": 1.0,
    }
    metadata = [
        {"selected_particle_ids": [1, 2], "continuous": [1.0 + index * 1e-9]}
        for index in range(len(LABELS))
    ]
    metadata[1]["continuous"] = [1.001]
    result = analyzer.classify_metadata_iteration(metadata, LABELS, contract)
    assert result["pass"]
    assert not result["continuous_serial_loo_diagnostic_only_pass"]
    metadata[1]["selected_particle_ids"] = [1, 3]
    assert not analyzer.classify_metadata_iteration(metadata, LABELS, contract)["pass"]


def _continuous_state_rows(*, growing: bool) -> list[dict[str, object]]:
    offsets = np.asarray([-5.0, -5.0, -3.0, -3.0, -1.0, -1.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0]) * 1e-7
    lane = np.asarray(["combined" in label for label in LABELS])
    rows = []
    for iteration in range(1, 201):
        values = offsets.copy()
        if growing:
            values[lane] += max(iteration - 180, 0) * 1e-5
        distances = np.abs(values[:, None] - values[None, :])
        result = {
            "distance_matrix": distances.tolist(),
            "trajectory_squared_euclidean_distance_matrix": np.square(distances).tolist(),
            "pass": True,
        }
        rows.append(
            {
                "iteration": iteration,
                "metadata": {"numeric": {"continuous": result}},
                "particle_star": {"numeric_groups": {}},
            }
        )
    return rows


def test_continuous_state_exchangeable_null_passes_and_growing_bias_fails() -> None:
    null = analyzer.classify_continuous_state_trajectory(
        _continuous_state_rows(growing=False), LABELS, BLOCKS, thresholds=THRESHOLDS
    )
    growing = analyzer.classify_continuous_state_trajectory(
        _continuous_state_rows(growing=True), LABELS, BLOCKS, thresholds=THRESHOLDS
    )
    assert null["pass"]
    assert not growing["pass"]


def test_two_feature_direct_sum_preserves_square_geometry_without_psd_clipping() -> None:
    x = np.asarray([-1.0, 1.0, 1.0, -1.0])[:, None]
    y = np.asarray([-1.0, -1.0, 1.0, 1.0])[:, None]

    def squared(values: np.ndarray) -> np.ndarray:
        delta = values[:, None, :] - values[None, :, :]
        return np.sum(np.square(delta), axis=-1)

    gram, diagnostics = analyzer._direct_sum_gram(
        {"x": squared(x), "y": squared(y)},
        {"x": 1.0, "y": 1.0},
    )
    direct_sum = np.concatenate((x, y), axis=1) / np.sqrt(2.0)
    centered = direct_sum - np.mean(direct_sum, axis=0, keepdims=True)
    np.testing.assert_allclose(gram, centered @ centered.T, rtol=0.0, atol=2e-15)
    assert set(diagnostics) == {"x", "y"}
    assert max(row["negative_energy_fraction"] for row in diagnostics.values()) <= 1e-15
    assert max(row["distance_reconstruction_relative_l2"] for row in diagnostics.values()) <= 1e-15


def test_non_euclidean_continuous_distance_fails_closed_without_spectral_clipping() -> None:
    invalid = np.asarray(
        [
            [0.0, 1.0, 9.0],
            [1.0, 0.0, 1.0],
            [9.0, 1.0, 0.0],
        ]
    )
    with pytest.raises(analyzer.GateSetupError, match="not squared-Euclidean"):
        analyzer._squared_euclidean_distances_to_gram(invalid)


def _quality_rows(*, degraded: bool) -> list[dict[str, object]]:
    rows = []
    for index, label in enumerate(LABELS):
        lane = label.startswith("combined_candidate_")
        auc = 0.72 + (index - 3.5) * 1e-5
        shell = 30
        scale = 1.0 + (index - 3.5) * 1e-6
        model_resolution = 8.0 + (index - 3.5) * 1e-4
        if degraded and lane:
            auc -= 0.01
            shell -= 3
            scale *= 1.02
            model_resolution *= 1.05
        rows.append(
            {
                "label": label,
                "model_resolution_angstrom": model_resolution,
                "references": {
                    "gt": {
                        "fsc_auc": auc,
                        "resolution_shell_0p143": shell,
                        "resolution_angstrom_0p143": 4.0,
                        "global_scale_candidate_to_reference": scale,
                    }
                },
            }
        )
    return rows


def _classify_quality(rows: list[dict[str, object]]) -> dict[str, object]:
    return analyzer.classify_final_quality(
        rows,
        LABELS,
        blocks=BLOCKS,
        fsc_auc_outer_degradation=0.002,
        resolution_shell_outer_degradation=1,
        model_resolution_relative_degradation=0.01,
        scale_relative_deviation=0.005,
        permutation_p_min=0.05,
    )


def test_final_quality_exchangeable_null_passes_and_material_regression_fails() -> None:
    assert _classify_quality(_quality_rows(degraded=False))["pass"]
    assert not _classify_quality(_quality_rows(degraded=True))["pass"]


def test_final_quality_exchangeable_null_is_nominally_calibrated() -> None:
    passed = 0
    for seed in range(24):
        rng = np.random.default_rng(seed)
        rows = _quality_rows(degraded=False)
        for row in rows:
            reference = row["references"]["gt"]
            reference["fsc_auc"] += float(rng.normal(scale=2e-5))
            reference["global_scale_candidate_to_reference"] *= float(1.0 + rng.normal(scale=2e-5))
            row["model_resolution_angstrom"] *= float(1.0 + rng.normal(scale=2e-5))
        passed += int(_classify_quality(rows)["pass"])
    assert passed >= 22


def test_runtime_requires_material_wall_and_expectation_gain() -> None:
    serial = {"end_to_end_wall_s": 100.0, "expectation_stage_s": 80.0, "peak_gpu_memory_mib": 1000.0}
    passing_lane = {"end_to_end_wall_s": 93.0, "expectation_stage_s": 74.0, "peak_gpu_memory_mib": 1020.0}
    failing_lane = {"end_to_end_wall_s": 98.0, "expectation_stage_s": 79.0, "peak_gpu_memory_mib": 1020.0}

    def panel(lane: dict[str, float]) -> list[dict[str, float]]:
        return [dict(lane if "combined" in label else serial) for label in LABELS]

    kwargs = dict(
        relion_wall_s=[50.0] * 4,
        wall_percent_max=-5.0,
        expectation_percent_max=-5.0,
        peak_memory_percent_max=5.0,
        relion_ratio_target=1.1,
    )
    assert analyzer.classify_runtime(panel(passing_lane), LABELS, **kwargs)["candidate_pass"]
    assert not analyzer.classify_runtime(panel(failing_lane), LABELS, **kwargs)["candidate_pass"]


@pytest.mark.parametrize(
    "mutation",
    ("gpu_uuids", "gpu_uuid_count", "peak_device_uuid"),
)
def test_arm_runtime_rejects_monitor_uuid_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    expected_uuid = "GPU-expected"
    monitor = {
        "sample_count": 2,
        "gpu_count": 1,
        "peak_memory_mib": 1024.0,
        "gpu_uuids": [expected_uuid],
        "gpu_uuid_count": 1,
        "peak_device_uuid": expected_uuid,
    }
    if mutation == "gpu_uuids":
        monitor[mutation] = ["GPU-other"]
    elif mutation == "gpu_uuid_count":
        monitor[mutation] = 2
    else:
        monitor[mutation] = "GPU-other"

    def fake_load(path: Path, _label: str) -> dict[str, object]:
        if path.name == "timing.json":
            return {"external_wall_s": 10.0}
        return {"vdam_iteration_profile_summary": {"expectation_time_s": 1.0}}

    monkeypatch.setattr(analyzer, "_validate_runtime_artifact_manifest", lambda _root: "sealed")
    monkeypatch.setattr(analyzer, "_load_json", fake_load)
    monkeypatch.setattr(analyzer, "_read_gpu_monitor", lambda _path: monitor)
    with pytest.raises(analyzer.GateSetupError, match="GPU monitor UUID evidence differs"):
        analyzer._arm_runtime(
            tmp_path,
            "control_serial_1",
            (0, 1),
            expected_gpu_uuid=expected_uuid,
        )


def test_incomplete_hash_evidence_is_setup_failure(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"sealed")
    with pytest.raises(analyzer.GateSetupError, match="missing artifact manifest"):
        analyzer._validate_saved_manifest(tmp_path / "missing.sha256", tmp_path, [artifact])
    stale = tmp_path / "stale.sha256"
    stale.write_text(f"{'0' * 64}  artifact.bin\n")
    with pytest.raises(analyzer.GateSetupError, match="stale or incomplete"):
        analyzer._validate_saved_manifest(stale, tmp_path, [artifact])


@pytest.mark.parametrize(
    "relative_path",
    (
        "command.json",
        "timing.json",
        "gpu_monitor.csv",
        "process.time",
        "runner.stdout",
        "runner.stderr",
        "arm.json",
        "science_environment.json",
        "output/run_native_options.json",
    ),
)
def test_runtime_artifact_manifest_fails_after_any_member_mutation(
    tmp_path: Path,
    relative_path: str,
) -> None:
    run_root = tmp_path / "arm"
    expected = analyzer._expected_runtime_artifacts(run_root)
    for path in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sealed:{path.relative_to(run_root)}\n")
    manifest = run_root / "runtime_artifact_manifest.sha256"
    manifest.write_text(analyzer._manifest_text(run_root, expected))
    assert analyzer._validate_runtime_artifact_manifest(run_root) == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    (run_root / relative_path).write_text("mutated\n")
    with pytest.raises(analyzer.GateSetupError, match="stale or incomplete"):
        analyzer._validate_runtime_artifact_manifest(run_root)


def test_launch_manifest_rehashes_every_input_and_fails_after_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    fixture = tmp_path / "fixture"
    native = tmp_path / "native"
    repo.mkdir()
    fixture.mkdir()
    native.mkdir()
    source = repo / "source.py"
    source.write_text("sealed\n")
    native_repeat = native / "repeat-01" / "vdam-gf46"
    (native_repeat / "relion").mkdir(parents=True)
    native_repeat_paths = {
        "paired_gpu_uuid_sha256": native_repeat / "paired_gpu_uuid.json",
        "run_provenance_sha256": native_repeat / "run_provenance.json",
        "relion_timing_sha256": native_repeat / "relion" / "relion.timing.json",
        "relion_command_sha256": native_repeat / "relion" / "relion_command.json",
    }
    for name, path in native_repeat_paths.items():
        path.write_text(json.dumps({"sealed": name}) + "\n")
    runtime_contract = {
        "override_prefixes_rejected_unless_allowlisted": [
            "RECOVAR_",
            "VDAM_",
            "JAX_",
            "XLA_",
            "CUDA_",
            "NVIDIA_",
            "CUBLAS_",
            "TF_",
        ],
        "override_exact_names_rejected_unless_allowlisted": [
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
        ],
        "launch_environment_allowlist": [],
        "science_environment_allowlist": [],
        "science_environment_capture_names": ["PATH"],
    }
    acceptance = repo / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "schema": "recovar.vdam_coarse_combined_true200_acceptance.v3",
                "source_contract": {"files": ["source.py"]},
                "native_reference": {
                    "physical_gpu_uuid": "GPU-sealed",
                    "repeat_count": 1,
                    **{
                        name: [hashlib.sha256(path.read_bytes()).hexdigest()]
                        for name, path in native_repeat_paths.items()
                    },
                },
                "runtime_contract": runtime_contract,
            }
        )
    )
    fixture_manifest = fixture / "fixture_materialization.json"
    fixture_manifest.write_text("fixture\n")
    native_manifest = native / "science_manifest.json"
    native_manifest.write_text("native\n")
    other = tmp_path / "other.bin"
    other.write_bytes(b"other")

    def entry(path: Path) -> dict[str, str]:
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    files = {
        "acceptance": entry(acceptance),
        "fixture_manifest": entry(fixture_manifest),
        "native_science_manifest": entry(native_manifest),
        "other": entry(other),
    }
    files.update(resolver._native_repeat_file_entries(native, json.loads(acceptance.read_text())))
    source_sha, source_entries = resolver._source_manifest(repo, ["source.py"])
    payload = {
        "schema": resolver.SCHEMA,
        "repo_root": str(repo.resolve()),
        "fixture_dir": str(fixture.resolve()),
        "native_reference_root": str(native.resolve()),
        "expected_overlay_head": "a" * 40,
        "production_candidate_head": "b" * 40,
        "expected_node_name": "node",
        "target_gpu_uuid": "GPU-sealed",
        "files": files,
        "source_manifest": {"sha256": source_sha, "entries": source_entries},
        "runtime_contract": runtime_contract,
        "sealed_override_environment": {},
    }
    manifest = tmp_path / "launch.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(resolver, "_validate_repo", lambda *_args: None)
    exact_names = set(runtime_contract["override_exact_names_rejected_unless_allowlisted"])
    prefixes = tuple(runtime_contract["override_prefixes_rejected_unless_allowlisted"])
    for name in tuple(resolver.os.environ):
        if name.startswith(prefixes) or name in exact_names:
            monkeypatch.delenv(name, raising=False)
    assert resolver.validate_launch_manifest(manifest, digest) == payload
    source.write_text("mutated\n")
    with pytest.raises(resolver.LaunchResolutionError, match="source-manifest"):
        resolver.validate_launch_manifest(manifest, digest)


def test_source_manifest_is_canonical_for_unsorted_input_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("a\n")
    (repo / "z.py").write_text("z\n")

    digest, entries = resolver._source_manifest(repo, ["z.py", "a.py"])
    expected_rows = "".join(
        f"{hashlib.sha256((repo / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("a.py", "z.py")
    )
    assert [entry["path"] for entry in entries] == ["a.py", "z.py"]
    assert digest == hashlib.sha256(expected_rows.encode()).hexdigest()

    with pytest.raises(resolver.LaunchResolutionError, match="repeats a canonical path"):
        resolver._source_manifest(repo, ["a.py", "a.py"])
    with pytest.raises(resolver.LaunchResolutionError, match="repeats a canonical path"):
        resolver._source_manifest(repo, ["a.py", "./a.py"])


@pytest.mark.parametrize(
    ("contract_name", "relative_path"),
    (
        ("paired_gpu_uuid_sha256", "paired_gpu_uuid.json"),
        ("run_provenance_sha256", "run_provenance.json"),
        ("relion_timing_sha256", "relion/relion.timing.json"),
        ("relion_command_sha256", "relion/relion_command.json"),
    ),
)
def test_native_repeat_launch_pins_fail_after_mutation(
    tmp_path: Path,
    contract_name: str,
    relative_path: str,
) -> None:
    native = tmp_path / "native"
    repeat = native / "repeat-01" / "vdam-gf46"
    relative_paths = (
        "paired_gpu_uuid.json",
        "run_provenance.json",
        "relion/relion.timing.json",
        "relion/relion_command.json",
    )
    paths = {}
    for relative in relative_paths:
        path = repeat / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sealed:{relative}\n")
        paths[relative] = path
    acceptance = {
        "native_reference": {
            "repeat_count": 1,
            "paired_gpu_uuid_sha256": [hashlib.sha256(paths[relative_paths[0]].read_bytes()).hexdigest()],
            "run_provenance_sha256": [hashlib.sha256(paths[relative_paths[1]].read_bytes()).hexdigest()],
            "relion_timing_sha256": [hashlib.sha256(paths[relative_paths[2]].read_bytes()).hexdigest()],
            "relion_command_sha256": [hashlib.sha256(paths[relative_paths[3]].read_bytes()).hexdigest()],
        }
    }
    assert len(resolver._native_repeat_file_entries(native, acceptance)) == 4
    paths[relative_path].write_text("mutated\n")
    with pytest.raises(resolver.LaunchResolutionError, match="hash differs"):
        resolver._native_repeat_file_entries(native, acceptance)
    assert contract_name in acceptance["native_reference"]


def test_override_environment_rejects_all_declared_prefixes_and_exact_names() -> None:
    contract = json.loads(ACCEPTANCE.read_text())["runtime_contract"]
    observed = resolver._validate_override_environment(
        {
            "PATH": "/bin",
            "CUDA_VISIBLE_DEVICES": "GPU-sealed",
            "VDAM_TRUE200_ROOT": "/sealed/output",
            "VDAM_TRUE200_RESUME": "0",
        },
        contract,
    )
    assert observed == {
        "CUDA_VISIBLE_DEVICES": "GPU-sealed",
        "VDAM_TRUE200_RESUME": "0",
        "VDAM_TRUE200_ROOT": "/sealed/output",
    }
    for name in (
        "RECOVAR_K1_UNDECLARED_OVERRIDE",
        "VDAM_UNDECLARED_OVERRIDE",
        "JAX_UNDECLARED_OVERRIDE",
        "XLA_UNDECLARED_OVERRIDE",
        "CUDA_UNDECLARED_OVERRIDE",
        "NVIDIA_UNDECLARED_OVERRIDE",
        "CUBLAS_UNDECLARED_OVERRIDE",
        "TF_UNDECLARED_OVERRIDE",
        "CUDA_TOOLKIT",
        "RELION_CUDA_TOOLKIT",
        "RELION_RUNTIME",
        "MPI_ROOT",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
    ):
        with pytest.raises(resolver.LaunchResolutionError, match=name):
            resolver._validate_override_environment({name: "1"}, contract)


def test_science_override_allowlist_is_explicit_and_does_not_admit_unknowns() -> None:
    contract = json.loads(ACCEPTANCE.read_text())["runtime_contract"]
    observed = resolver._validate_override_environment(
        {
            "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID": "1",
            "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1",
            "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE": "1",
            "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "8",
            "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "1",
            "JAX_COMPILATION_CACHE_DIR": "/sealed/cache",
        },
        contract,
        include_science=True,
    )
    assert set(observed) == {
        "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE",
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION",
        "JAX_COMPILATION_CACHE_DIR",
    }
    with pytest.raises(resolver.LaunchResolutionError, match="RECOVAR_UNSEALED"):
        resolver._validate_override_environment(
            {"RECOVAR_UNSEALED": "1"}, contract, include_science=True
        )


def test_effective_science_environment_snapshot_is_complete_and_canonical() -> None:
    contract = json.loads(ACCEPTANCE.read_text())["runtime_contract"]
    environment = {
        name: f"sealed:{name}"
        for name in contract["science_environment_capture_names"]
    }
    environment.update(contract["fixed_science_environment"])
    environment["VDAM_TRUE200_ROOT"] = "/sealed/output"
    snapshot = resolver._science_environment_snapshot(environment, contract)
    assert list(snapshot) == sorted(snapshot)
    assert set(contract["science_environment_capture_names"]).issubset(snapshot)
    assert snapshot["PATH"] == contract["fixed_science_environment"]["PATH"]
    assert snapshot["VDAM_TRUE200_ROOT"] == "/sealed/output"

    missing = dict(environment)
    del missing["PATH"]
    with pytest.raises(resolver.LaunchResolutionError, match="missing: PATH"):
        resolver._science_environment_snapshot(missing, contract)
    with pytest.raises(resolver.LaunchResolutionError, match="XLA_UNSEALED"):
        resolver._science_environment_snapshot(
            {**environment, "XLA_UNSEALED": "1"},
            contract,
        )


def test_launch_resolver_reports_every_unresolved_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment_names = (
        "REPO_ROOT",
        "VDAM_GF46_FIXTURE_DIR",
        "VDAM_NATIVE_REFERENCE_ROOT",
        "QUALIFIED_CUDA_OVERRIDE",
        "RELION_BIND_BINARY_OVERRIDE",
        "PIXI_PY_OVERRIDE",
        "GPU_SELECTION_HELPER",
        "CUSPARSE_LIBRARY",
        "EXPECTED_OVERLAY_HEAD",
        "EXPECTED_NODE_NAME",
        "TARGET_GPU_UUID",
    )
    for name in environment_names:
        monkeypatch.delenv(name, raising=False)
    args = Namespace(
        repo_root=None,
        fixture_dir=None,
        native_reference_root=None,
        qualified_cuda=None,
        relion_bind=None,
        interpreter=None,
        gpu_selection_helper=None,
        cusparse_library=None,
        expected_overlay_head=None,
        expected_node_name=None,
        target_gpu_uuid=None,
        output=tmp_path / "launch.json",
    )
    with pytest.raises(resolver.LaunchResolutionError, match="unresolved launch inputs") as error:
        resolver._create(args)
    assert "fixture_dir" in str(error.value)
    assert "gpu_selection_helper" in str(error.value)
    assert "target_gpu_uuid" in str(error.value)


def test_true200_wrapper_is_fail_closed_resumable_and_seals_terminal_state_last() -> None:
    text = HARNESS.read_text()
    for required in (
        'test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_OVERLAY_HEAD}"',
        'git -C "${REPO_ROOT}" merge-base --is-ancestor',
        'test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"',
        "VDAM_TRUE200_LAUNCH_MANIFEST",
        "EXPECTED_LAUNCH_MANIFEST_SHA256",
        '"${PIXI_PY}" "${RESOLVER}" validate',
        'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0',
        'vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}" "${initial_allocation_spec}"',
        'test "${selected_gpu_uuid}" = "$(jq -r \'.native_reference.physical_gpu_uuid\'',
        'test "$(sha256sum "${CUDA_BINARY}"',
        'test "$(sha256sum "${RELION_BIND_BINARY}"',
        'test "$(sha256sum "${PIXI_PY}"',
        'test "$(sha256sum "${CUSPARSE_LIBRARY}"',
        'ATTEMPT_RUNTIME=${RUNTIME}/attempts/${SLURM_JOB_ID}',
        'export TMPDIR=${ATTEMPT_RUNTIME}/tmp',
        'export PIXI_HOME=${ATTEMPT_RUNTIME}/pixi_home',
        'export RATTLER_CACHE_DIR=${ATTEMPT_RUNTIME}/rattler_cache',
    ):
        assert required in text

    # Resume may reuse only a verified completed prefix. The shared analyzer
    # verifies provenance, exact 804 topology/full hashes, options, timing,
    # profiles, and GPU-monitor evidence before the wrapper skips an arm.
    assert "VDAM_TRUE200_RESUME" in text
    assert "initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}" in text
    assert "allocated_gpu_uuids.csv" in text
    assert "visible_gpu_uuids.csv" in text
    assert 'nvidia-smi -i "${selected_gpu_uuid}" -q' in text
    assert "--query-gpu=timestamp,index,name,uuid,memory.used" in text
    assert "LC_ALL=C sort" in text
    assert 'ATTEMPT_PROVENANCE=${PROVENANCE}/attempts/${ATTEMPT_ID}' in text
    assert '"${ATTEMPT_PROVENANCE}/harness_failure.txt"' in text
    assert '"${PROVENANCE}/harness_failure.txt"' not in text
    assert "source_manifest.resume.sha256" not in text
    assert "science_contract.resume.json" not in text
    assert "completed_prefix=1" in text
    assert 'test "${completed_prefix}" -eq 1' in text
    assert "_validate_arm_artifacts" in text
    assert "_validate_arm_coarse_hybrid_execution" in text
    assert "_validate_native_options" in text
    assert "_arm_runtime" in text
    assert 'if [[ -f "${run_root}/SCIENCE_COMPLETED" ]]' in text
    assert 'mv "${run_root}" "${quarantine}"' in text
    assert "rm -rf" not in text
    assert "rm -f" not in text
    assert 'control_serial_5 combined_candidate_5 combined_candidate_6 control_serial_6' in text
    assert "RUN_WORKERS=(0 0 0 0 0 0 0 0 0 0 0 0)" in text
    assert "RUN_ATOMIC=(0 0 0 0 0 0 0 0 0 0 0 0)" in text
    assert "RUN_HYBRID=(0 1 1 0 1 0 0 1 0 1 1 0)" in text
    assert '"RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=${workers}"' in text
    assert '"RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=${native_atomic}"' in text
    assert '"RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID=${hybrid}"' in text
    assert '"RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=${hybrid}"' in text
    assert '"RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE=${hybrid}"' in text
    assert "FIXED_SCIENCE_ENVIRONMENT_ARGS" in text
    assert "VDAM_TRUE200_SENTINEL" in text
    assert 'write_marker_once "${ROOT}/SENTINEL_PASSED"' in text
    assert 'validate_completed_arm "${label}"' in text
    assert "runtime_libraries.json" in text
    assert 'cmp "${PROVENANCE}/runtime_libraries.json"' in text
    assert "sealed_override_environment.json" in text
    assert "science_environment.json" in text
    assert "_science_environment_snapshot" in text
    assert "runtime_artifact_manifest.sha256" in text
    assert "output/run_native_options.json" in text
    assert ".runtime_contract.fixed_science_environment.PATH" in text
    assert ".runtime_contract.fixed_science_environment.LD_LIBRARY_PATH" in text
    assert "${LD_LIBRARY_PATH:-}" not in text
    assert "${CUDA_TOOLKIT:-" not in text
    assert 'mv "${ROOT}/${marker}" "${terminal_quarantine}/${marker}"' in text
    assert 'touch "${ROOT}/RUNS_COMPLETED"' not in text

    arm_complete = text.index('write_marker_once "${run_root}/SCIENCE_COMPLETED"')
    runtime_artifact_seal = text.index('> "${run_root}/runtime_artifact_manifest.sha256"')
    allocation_audit = text.index('vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}"')
    gpu_select = text.index('vdam_select_target_gpu "${TARGET_GPU_UUID}" 0')
    runs_complete = text.index('write_marker_once "${ROOT}/RUNS_COMPLETED"')
    analysis = text.index("analysis_status=$?")
    setup_failed = text.index('write_marker_once "${ATTEMPT_PROVENANCE}/ANALYSIS_SETUP_FAILED"')
    sealed_hashes = text.index('> "${ATTEMPT_PROVENANCE}/sealed_analysis.sha256"')
    science_failed = text.index('write_marker_once "${ROOT}/SCIENCE_FAILED"')
    completed = text.rindex('write_marker_once "${ROOT}/COMPLETED"')
    terminal_exit = text.index('exit "${analysis_status}"')
    assert allocation_audit < gpu_select
    assert runtime_artifact_seal < arm_complete < runs_complete < analysis < setup_failed < sealed_hashes
    assert sealed_hashes < science_failed < completed < terminal_exit


def test_acceptance_seals_twelve_arm_power_and_truthful_resource_estimate() -> None:
    contract = json.loads(ACCEPTANCE.read_text())
    assert contract["schema"] == "recovar.vdam_coarse_combined_true200_acceptance.v3"
    assert resolver.SCHEMA == "recovar.vdam_coarse_combined_true200_launch.v3"
    assert analyzer.SCHEMA == "recovar.vdam_coarse_combined_true200_analysis.v3"
    panel = contract["panel"]
    assert len(panel["execution_order"]) == 12
    assert panel["coarse_multistream_workers"] == [0] * 12
    assert panel["coarse_native_atomic_reduction"] == [0] * 12
    assert panel["coarse_gemm_hybrid"] == [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    assert panel["sentinel_prefix_arm_count"] == 2
    assert panel["blocked_assignment_count"] == 216
    assert analyzer._balanced_assignments(BLOCKS).__len__() == 216
    assert panel["complement_pair_minimum_one_sided_p"] < 0.05
    assert contract["required_gates"]["no_growth_familywise_alpha"] == 0.05
    assert contract["required_gates"]["no_growth_materiality_serial_control_loo_multiplier"] == 1.0
    assert contract["required_gates"]["map_matches_same_joint_exact_serial_witness"] is True
    assert contract["required_gates"]["witness_map_normalized_distance_at_most"] == pytest.approx(
        4.0 * np.finfo(np.float32).eps
    )
    assert contract["required_gates"]["no_growth_normalized_numerical_noise_floor"] == pytest.approx(
        4.0 * np.finfo(np.float32).eps
    )
    source_files = contract["source_contract"]["files"]
    assert source_files == sorted(source_files)
    assert len(source_files) == len(set(source_files))
    repo = ACCEPTANCE.parents[1]
    assert all((repo / relative).is_file() for relative in source_files)
    scorecard = repo / "docs/math/vdam_k1_full_trajectory_expansion_v3.json"
    assert hashlib.sha256(scorecard.read_bytes()).hexdigest() == contract["case"]["scorecard_sha256"]
    assert (
        contract["qualified_candidate"]["production_head"]
        == "e0c1d1746570e64bad3618b09a50be31b79ebe60"
    )
    assert (
        contract["qualified_candidate"]["relion_bind_sha256"]
        == "9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980"
    )
    assert (
        contract["qualified_candidate"]["interpreter_sha256"]
        == "1e43e23601e6369d52fd56b0405882463297c9ba6ad5238b460da90db6771b9c"
    )
    assert (
        contract["trajectory"]["no_growth_primary_family_endpoints"] == THRESHOLDS["no_growth_primary_family_endpoints"]
    )
    assert contract["numerical_acceptance_policy"]["joint_exact_complete_path_serial_witness_required"] is True
    assert contract["numerical_acceptance_policy"]["map_must_match_same_exact_state_witness"] is True
    exact_state_keys = contract["state_contract"]["joint_exact_complete_path_metadata_keys"]
    assert "halfset_ids" in exact_state_keys
    assert "joint_halfset_particle_stream" in exact_state_keys
    assert (
        contract["required_gates"]["coarse_hybrid_execution_proven_every_postinit_joint_stream"]
        is True
    )
    assert contract["trajectory"]["exact_state_first_iteration"] == 1
    assert contract["trajectory"]["exact_state_last_iteration"] == 200
    native = contract["native_reference"]
    for name in (
        "paired_gpu_uuid_sha256",
        "run_provenance_sha256",
        "relion_timing_sha256",
        "relion_command_sha256",
    ):
        assert len(native[name]) == native["repeat_count"] == 4
        assert all(len(value) == 64 for value in native[name])
    runtime_contract = contract["runtime_contract"]
    assert set(("XLA_", "CUDA_", "NVIDIA_", "CUBLAS_", "TF_")).issubset(
        runtime_contract["override_prefixes_rejected_unless_allowlisted"]
    )
    assert set(("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH")).issubset(
        runtime_contract["override_exact_names_rejected_unless_allowlisted"]
    )
    assert "PATH" not in runtime_contract["override_exact_names_rejected_unless_allowlisted"]
    assert set(runtime_contract["fixed_science_environment"]).issubset(
        runtime_contract["science_environment_capture_names"]
    )
    for name in (
        "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE",
    ):
        assert name in runtime_contract["science_environment_capture_names"]
    cache = contract["science_contract"]["hybrid_projection_cache_contract"]
    assert cache["budget_gib"] == 40.0
    assert cache["maximum_retained_cache_gib"] == pytest.approx(18.28125)
    assert cache["maximum_conservative_peak_gib"] == pytest.approx(37.1337890625)
    assert cache["maximum_conservative_peak_gib"] < cache["budget_gib"]
    assert cache["destination_aliasing_assumed_for_admission"] is False
    assert contract["runtime_contract"]["transitive_runtime_library_closure_claimed"] is False
    resources = contract["resource_estimate"]
    assert 0.0 < resources["estimated_science_gpu_hours"] < 12.0
    assert resources["estimated_total_allocation_hours"] > resources["estimated_science_gpu_hours"]
    assert resources["estimated_single_allocation_fit"]
    assert resources["minimum_expected_allocations"] == 1
    assert resources["expected_allocation_count"].startswith("one")


def test_direct_hybrid_panel_contract_matches_labels_and_blocks() -> None:
    panel = json.loads(ACCEPTANCE.read_text())["panel"]
    labels, workers, atomic, hybrid, blocks = analyzer._validate_direct_hybrid_panel(
        panel
    )
    assert labels == LABELS
    assert workers == atomic == (0,) * 12
    assert hybrid == (0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0)
    assert blocks == BLOCKS


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("label_mode", "modes do not match their arm labels"),
        ("legacy_worker", "must disable legacy multistream and atomic modes"),
        ("sentinel", "sentinel must contain one direct arm"),
        ("duplicate_block_arm", "permutation blocks do not partition the arms"),
        ("unbalanced_block", "permutation block 0 is not balanced"),
    ),
)
def test_direct_hybrid_panel_contract_rejects_assignment_corruption(
    mutation: str,
    message: str,
) -> None:
    panel = copy.deepcopy(json.loads(ACCEPTANCE.read_text())["panel"])
    if mutation == "label_mode":
        panel["coarse_gemm_hybrid"][0] = 1
    elif mutation == "legacy_worker":
        panel["coarse_multistream_workers"][1] = 8
    elif mutation == "sentinel":
        panel["sentinel_prefix_arm_count"] = 3
    elif mutation == "duplicate_block_arm":
        panel["whole_trajectory_permutation_blocks"][2][3] = "control_serial_5"
    elif mutation == "unbalanced_block":
        panel["whole_trajectory_permutation_blocks"][0] = [
            "control_serial_1",
            "control_serial_2",
            "control_serial_3",
            "combined_candidate_1",
        ]
    else:  # pragma: no cover - the parameter table is closed above
        raise AssertionError(mutation)
    with pytest.raises(analyzer.GateSetupError, match=message):
        analyzer._validate_direct_hybrid_panel(panel)


def test_maximum_declared_projection_cache_plan_fits_without_alias_assumption() -> None:
    from recovar.em.dense_single_volume.helpers.significance import (
        _plan_coarse_gaussian_gemm_projection_cache,
    )

    cache = json.loads(ACCEPTANCE.read_text())["science_contract"][
        "hybrid_projection_cache_contract"
    ]
    gib = 2**30
    plan = _plan_coarse_gaussian_gemm_projection_cache(
        n_rotations=cache["maximum_declared_rotation_count"],
        compact_pixel_count=cache["maximum_declared_compact_pixel_count"],
        image_shape=(128, 128),
        budget_bytes=int(cache["budget_gib"] * gib),
    )
    assert plan.cache_shape == (1, 294_912, 8_320)
    assert plan.chunk_rows == 4_608
    assert plan.chunk_count_per_table == 64
    assert plan.retained_bytes / gib == pytest.approx(
        cache["maximum_retained_cache_gib"]
    )
    assert plan.predicted_peak_bytes / gib == pytest.approx(
        cache["maximum_conservative_peak_gib"]
    )
    assert plan.destination_alias_proven is False
    assert plan.admitted
