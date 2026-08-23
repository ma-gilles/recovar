from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts import validate_bpref_device_signature as validator

pytestmark = pytest.mark.unit


def _records(values=(1.0, 2.0, 3.0)):
    size = len(values)
    return validator.ContributionRecords(
        target_indices=np.zeros(size, dtype=np.int32),
        coefficients=np.ones(size, dtype=np.float32),
        source_data=np.asarray(values, dtype=np.complex64),
        source_weight=np.asarray(values, dtype=np.float32),
        row_conjugated=np.zeros(size, dtype=bool),
        neighbor_conjugated=np.zeros(size, dtype=bool),
        launch_ordinal=np.arange(size, dtype=np.int64),
        particle_local_row=np.arange(size, dtype=np.int32),
        original_index=np.zeros(size, dtype=np.int64),
        canonical_rotation_key=np.arange(size, dtype=np.int32),
        dense_pixel=np.zeros(size, dtype=np.int32),
        neighbor=np.zeros(size, dtype=np.int32),
    )


def _permute(records, permutation):
    values = {}
    for key in records.__dataclass_fields__:
        value = getattr(records, key)
        if key == "recomputation_provenance":
            values[key] = value
        else:
            values[key] = None if value is None else value[permutation]
    return validator.ContributionRecords(**values)


def _verified_records(
    tmp_path: Path,
    name: str,
    records,
    *,
    coefficients,
    source_data,
    source_weight,
):
    parent_path = tmp_path / f"{name}.parent.npz"
    companion_path = tmp_path / f"{name}.companion.npz"
    np.savez(parent_path, identity=np.asarray(name))
    np.savez(companion_path, identity=np.asarray(name))
    parent_sha = (validator._sha256_file(parent_path),)
    companion_sha = (validator._sha256_file(companion_path),)
    path = tmp_path / f"{name}.npz"
    np.savez(
        path,
        magic=np.asarray(validator.RECOMPUTATION_MAGIC),
        schema=np.asarray(validator.RECOMPUTATION_SCHEMA),
        schema_version=np.int32(1),
        parent_signature_sha256=np.asarray(parent_sha),
        companion_contribution_sha256=np.asarray(companion_sha),
        semantic_identity_sha256=np.asarray(
            validator._semantic_identity_digest(records)
        ),
        formula_name=np.asarray(validator.RECOMPUTATION_FORMULA_NAME),
        formula_version=np.asarray(validator.RECOMPUTATION_FORMULA_VERSION),
        numeric_policy=np.asarray(validator.RECOMPUTATION_NUMERIC_POLICY),
        source_dtype=np.asarray(validator.RECOMPUTATION_SOURCE_POLICY),
        recomputed_coefficients=np.asarray(coefficients, dtype=np.float64),
        recomputed_source_data=np.asarray(source_data, dtype=np.complex128),
        recomputed_source_weight=np.asarray(source_weight, dtype=np.float64),
    )
    return validator.load_verified_recomputation(
        path,
        records,
        parent_signature_paths=(parent_path,),
        companion_contribution_paths=(companion_path,),
    )


def _minimal_signature():
    source = np.zeros((1, 1, 6), dtype=np.float32)
    source[0, 0, :3] = (2.0, 3.0, 5.0)
    return {
        "magic": np.asarray(validator.SIGNATURE_MAGIC),
        "schema": np.asarray(validator.SIGNATURE_SCHEMA),
        "schema_version": np.int32(1),
        "row_flags": np.asarray([[64 | 16]], dtype=np.int32),
        "source_values": source,
        "neighbor_indices": np.arange(8, dtype=np.int32).reshape(1, 1, 8),
        "neighbor_coefficients": np.arange(1, 9, dtype=np.float32).reshape(1, 1, 8),
        "neighbor_flags": np.asarray([[[3, 1, 1, 1, 1, 1, 1, 1]]], dtype=np.int32),
        "launch_ordinal": np.asarray([4], dtype=np.int64),
        "particle_local_row": np.asarray([2], dtype=np.int32),
        "original_indices": np.asarray([91], dtype=np.int64),
        "contributor_canonical_rotation_keys": np.asarray([123], dtype=np.int32),
    }


def test_canonical_replay_is_invariant_to_record_permutation():
    records = _records((1.25, -3.5, 7.0, 0.125))
    permuted = _permute(records, np.asarray([2, 0, 3, 1]))

    expected = validator.replay_contribution_records(
        records, 1, order="canonical", precision="float32"
    )
    actual = validator.replay_contribution_records(
        permuted, 1, order="canonical", precision="float32"
    )

    assert np.array_equal(actual.data, expected.data)
    assert np.array_equal(actual.weight, expected.weight)


def test_logical_host_order_is_float32_sensitive_and_collapses_in_float64():
    records = _records((1e8, 1.0, -1e8))
    reordered = replace(records, launch_ordinal=np.asarray([0, 2, 1], dtype=np.int64))

    left32 = validator.replay_contribution_records(
        records, 1, order="logical_host_order", precision="float32"
    )
    right32 = validator.replay_contribution_records(
        reordered, 1, order="logical_host_order", precision="float32"
    )
    left64 = validator.replay_contribution_records(
        records, 1, order="logical_host_order", precision="float64"
    )
    right64 = validator.replay_contribution_records(
        reordered, 1, order="logical_host_order", precision="float64"
    )

    assert not np.array_equal(left32.data, right32.data)
    assert np.array_equal(left64.data, right64.data)
    assert validator.compare_contribution_engines(records, reordered, 1)[
        "classification"
    ] == "logical_schedule_difference"


def test_coefficient_precision_closure_requires_verified_provenance(tmp_path):
    base = _records((1.0,))
    changed = replace(
        base,
        coefficients=np.nextafter(base.coefficients, np.float32(2.0)),
    )
    left = _verified_records(
        tmp_path,
        "left0",
        base,
        coefficients=[1.0],
        source_data=[1.0 + 0j],
        source_weight=[1.0],
    )
    right = _verified_records(
        tmp_path,
        "right1",
        changed,
        coefficients=[1.0],
        source_data=[1.0 + 0j],
        source_weight=[1.0],
    )

    report = validator.compare_contribution_engines(left, right, 1)

    assert report["classification"] == (
        "precision_consistent_with_verified_recomputation"
    )
    assert report["recomputed_high_precision_equal"] is True
    assert report["recomputed_high_precision_cross_engine"]["data"]["array_equal"]


def test_arbitrary_float64_casts_are_unverified_and_cannot_justify_precision():
    base = _records((1.0,))
    unverified = {
        "recomputed_coefficients": base.coefficients.astype(np.float64),
        "recomputed_source_data": base.source_data.astype(np.complex128),
        "recomputed_source_weight": base.source_weight.astype(np.float64),
    }
    left = replace(base, **unverified)
    right = replace(
        base,
        coefficients=np.nextafter(base.coefficients, np.float32(2.0)),
        **unverified,
    )

    report = validator.compare_contribution_engines(left, right, 1)

    assert report["classification"] == "unresolved"
    assert report["caller_supplied_high_precision_arrays_unverified"] is True
    with pytest.raises(ValueError, match="lack validated provenance"):
        validator.replay_contribution_records(
            left,
            1,
            order="canonical",
            precision="float64",
            operand_provenance=validator.RECOMPUTED_HIGH_PRECISION,
        )


def test_verified_loader_rejects_captured_float32_promotion(tmp_path):
    records = _records((1.0,))
    parent_path = tmp_path / "parent.npz"
    companion_path = tmp_path / "companion.npz"
    np.savez(parent_path, identity=np.asarray("parent"))
    np.savez(companion_path, identity=np.asarray("companion"))
    artifact_path = tmp_path / "recompute.npz"
    np.savez(
        artifact_path,
        magic=np.asarray(validator.RECOMPUTATION_MAGIC),
        schema=np.asarray(validator.RECOMPUTATION_SCHEMA),
        schema_version=np.int32(1),
        parent_signature_sha256=np.asarray([validator._sha256_file(parent_path)]),
        companion_contribution_sha256=np.asarray(
            [validator._sha256_file(companion_path)]
        ),
        semantic_identity_sha256=np.asarray(
            validator._semantic_identity_digest(records)
        ),
        formula_name=np.asarray(validator.RECOMPUTATION_FORMULA_NAME),
        formula_version=np.asarray(validator.RECOMPUTATION_FORMULA_VERSION),
        numeric_policy=np.asarray(validator.RECOMPUTATION_NUMERIC_POLICY),
        source_dtype=np.asarray("captured-float32-promoted-before-formula"),
        recomputed_coefficients=np.ones(1, dtype=np.float64),
        recomputed_source_data=np.ones(1, dtype=np.complex128),
        recomputed_source_weight=np.ones(1, dtype=np.float64),
    )

    with pytest.raises(ValueError, match="captured-float32 promotion"):
        validator.load_verified_recomputation(
            artifact_path,
            records,
            parent_signature_paths=(parent_path,),
            companion_contribution_paths=(companion_path,),
        )


def test_v2_verified_recomputation_manifest_binds_frozen_inputs(tmp_path):
    records = _records((1.0,))
    parent_path = tmp_path / "parent.npz"
    companion_path = tmp_path / "companion.npz"
    np.savez(parent_path, identity=np.asarray("parent"))
    companion = {
        "image_identities": np.asarray(["1@/frozen/particles.mrcs"]),
        "raw_real_images": np.ones((1, 2, 2), dtype=np.float32),
        "integer_pre_shifts": np.zeros((1, 2), dtype=np.int32),
        "relion_preprocess_normalization_factors": np.ones(1, dtype=np.float32),
        "ctf_params": np.ones((1, 9), dtype=np.float32),
        "noise_variance_half": np.ones(4, dtype=np.float32),
        "scale_corrections": np.ones(1, dtype=np.float32),
        "voxel_size": np.float64(1.0),
        "ctf_mode": np.asarray("legacy"),
        "ctf_dose_per_tilt": np.float64(0.0),
        "ctf_angle_per_tilt": np.float64(0.0),
        "reconstruction_probs": np.ones((1, 1, 1), dtype=np.float64),
        "reconstruction_mask": np.ones((1, 1, 1), dtype=bool),
        "reconstruction_sum_weight": np.ones(1, dtype=np.float64),
        "reconstruction_threshold": np.ones(1, dtype=np.float64),
        "active_particle_rows": np.zeros(1, dtype=np.int32),
        "active_rotation_rows": np.zeros(1, dtype=np.int32),
        "active_rotations": np.eye(3, dtype=np.float32)[None],
        "oversampled_rotation_indices": np.zeros((1, 1), dtype=np.int64),
        "fine_translations": np.zeros((1, 2), dtype=np.float32),
        "window_indices": np.arange(4, dtype=np.int32),
    }
    np.savez(companion_path, **companion)

    def digest(fields):
        return validator._sha256_named_arrays(
            (f"shard0:{field}", companion[field]) for field in fields
        )

    artifact_path = tmp_path / "recompute-v2.npz"
    np.savez(
        artifact_path,
        magic=np.asarray(validator.RECOMPUTATION_MAGIC),
        schema=np.asarray(validator.RECOMPUTATION_SCHEMA),
        schema_version=np.int32(2),
        parent_signature_sha256=np.asarray([validator._sha256_file(parent_path)]),
        companion_contribution_sha256=np.asarray([validator._sha256_file(companion_path)]),
        semantic_identity_sha256=np.asarray(validator._semantic_identity_digest(records)),
        formula_name=np.asarray(validator.RECOMPUTATION_FORMULA_NAME),
        formula_version=np.asarray(validator.RECOMPUTATION_FORMULA_VERSION),
        numeric_policy=np.asarray(validator.RECOMPUTATION_NUMERIC_POLICY),
        source_dtype=np.asarray(validator.RECOMPUTATION_SOURCE_POLICY),
        source_boundary=np.asarray(
            "native float32 stack pixels; downstream operands recomputed without captured-complex64 promotion"
        ),
        fft_layout=np.asarray("centered-y packed-x rfft; flattened C order"),
        fft_normalization=np.asarray("unnormalized forward numpy.fft.rfft2"),
        posterior_weight_policy=np.asarray(
            "captured reconstruction_probs frozen at M-step boundary"
        ),
        canonical_sort_key_legend=np.asarray(
            "original_index,canonical_rotation_key,dense_pixel,neighbor"
        ),
        raw_image_identity_sha256=np.asarray(digest(("image_identities",))),
        raw_image_input_sha256=np.asarray(digest((
            "raw_real_images", "integer_pre_shifts",
            "relion_preprocess_normalization_factors",
        ))),
        ctf_noise_input_sha256=np.asarray(digest((
            "ctf_params", "noise_variance_half", "scale_corrections", "voxel_size",
            "ctf_mode", "ctf_dose_per_tilt", "ctf_angle_per_tilt",
        ))),
        posterior_weight_sha256=np.asarray(digest((
            "reconstruction_probs", "reconstruction_mask",
            "reconstruction_sum_weight", "reconstruction_threshold",
        ))),
        hypothesis_geometry_input_sha256=np.asarray(digest((
            "active_particle_rows", "active_rotation_rows", "active_rotations",
            "oversampled_rotation_indices", "fine_translations", "window_indices",
        ))),
        canonical_original_index=records.original_index,
        canonical_rotation_key=records.canonical_rotation_key,
        canonical_dense_pixel=records.dense_pixel,
        canonical_neighbor=records.neighbor,
        captured_target_indices=records.target_indices,
        captured_row_conjugated=records.row_conjugated,
        captured_neighbor_conjugated=records.neighbor_conjugated,
        recomputed_coefficients=np.ones(1, dtype=np.float64),
        recomputed_source_data=np.ones(1, dtype=np.complex128),
        recomputed_source_weight=np.ones(1, dtype=np.float64),
    )

    verified = validator.load_verified_recomputation(
        artifact_path,
        records,
        parent_signature_paths=(parent_path,),
        companion_contribution_paths=(companion_path,),
    )

    assert verified.has_verified_recomputed_high_precision


def test_high_precision_recompute_refuses_stale_output_and_bounds_mismatch_sample(tmp_path):
    from scripts import recompute_bpref_high_precision

    stale = tmp_path / "stale.npz"
    stale.write_bytes(b"not-a-current-artifact")
    with pytest.raises(FileExistsError, match="output already exists"):
        recompute_bpref_high_precision._require_new_output_path(stale)

    mask = np.ones((100, 2), dtype=bool)
    diagnostic = recompute_bpref_high_precision._mismatch_diagnostic(
        mask,
        label="test_mismatch",
    )
    assert diagnostic["test_mismatch_count"] == 200
    assert diagnostic["test_mismatch_sample_limit"] == 32
    assert len(diagnostic["test_mismatch_sample"]) == 32
    assert len(diagnostic["test_mismatch_mask_sha256"]) == 64


@pytest.mark.parametrize(
    ("active_delta", "repeat_delta"),
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
)
def test_high_precision_source_control_requires_exact_capture_and_repeat(
    monkeypatch, active_delta, repeat_delta
):
    from scripts import recompute_bpref_high_precision

    records = _records((2.0,))
    result = {
        "contribution": {
            "image_shape": np.asarray([2, 2], dtype=np.int32),
            "window_indices": np.asarray([0], dtype=np.int32),
        },
        "signature": {
            "max_r": np.float32(1.0),
            "particle_launch_ordinals": np.asarray([0], dtype=np.int64),
            "launch_ordinal": np.asarray([0], dtype=np.int64),
            "particle_local_row": np.asarray([0], dtype=np.int32),
        },
        "contribution_records": records,
    }
    call_count = 0

    def fake_source_rows(
        _contribution,
        _signature,
        *,
        legacy_reconstruction_prob_dtype=None,
    ):
        del legacy_reconstruction_prob_dtype
        nonlocal call_count
        call_count += 1
        value = np.complex64(2.0 + (repeat_delta if call_count == 2 else 0.0))
        metrics = {
            "data_vs_captured_active": validator.exact_array_metrics(
                np.asarray([2.0 + active_delta], dtype=np.complex64),
                np.asarray([2.0], dtype=np.complex64),
            ),
            "weight_vs_captured_active": validator.exact_array_metrics(
                np.asarray([2.0], dtype=np.float32),
                np.asarray([2.0], dtype=np.float32),
            ),
        }
        return (
            np.asarray([[value]], dtype=np.complex64),
            np.asarray([[2.0]], dtype=np.float32),
            metrics,
        )

    monkeypatch.setattr(
        recompute_bpref_high_precision,
        "_source_rows_production_f32",
        fake_source_rows,
    )

    control = recompute_bpref_high_precision._validate_production_source_control(result)

    assert control["validated"] is (active_delta == 0.0 and repeat_delta == 0.0)
    assert control["data_vs_captured_signature"]["array_equal"]
    assert control["weight_vs_captured_signature"]["array_equal"]
    assert control["control_repeat_data"]["array_equal"] is (repeat_delta == 0.0)


def test_ordinary_capture_policy_cannot_silently_select_sequential_reduction():
    from scripts import recompute_bpref_high_precision

    contribution = {
        "operand_source": np.asarray(
            "authoritative-ordinary-translation-reduction"
        ),
        "production_adjoint_topology": np.asarray(
            "ordinary-flattened-production-adjoint"
        ),
    }
    signature = {
        "topology_claim": np.asarray("ordinary-flattened-production-adjoint"),
    }

    policy = recompute_bpref_high_precision._captured_source_reduction_policy(
        contribution, signature
    )

    assert policy["name"] == "authoritative-ordinary-translation-reduction"
    assert policy["sequential_translation_reduction"] is False
    assert policy["order_control_name"] == (
        "relion-f32-sequential-translation-reduction"
    )
    ordinary = (
        np.asarray([[1.0 + 0.0j]], dtype=np.complex64),
        np.asarray([[1.0]], dtype=np.float32),
    )
    sequential = (
        np.asarray([[2.0 + 0.0j]], dtype=np.complex64),
        np.asarray([[1.0]], dtype=np.float32),
    )
    selected, order_control = (
        recompute_bpref_high_precision._select_captured_and_order_reductions(
            policy, ordinary, sequential
        )
    )
    assert np.array_equal(selected[0], ordinary[0])
    assert np.array_equal(order_control[0], sequential[0])
    assert not validator.exact_array_metrics(selected[0], sequential[0])["array_equal"]


def test_source_reduction_policy_rejects_conflicting_topology_metadata():
    from scripts import recompute_bpref_high_precision

    contribution = {
        "operand_source": np.asarray(
            "authoritative-ordinary-translation-reduction"
        ),
    }
    signature = {
        "topology_claim": np.asarray(
            "causal-arm-not-relion-hypothesis-arithmetic-closure"
        ),
        "causal_arm": np.asarray("soft-posterior-per-particle-fused-xhalf"),
    }

    with pytest.raises(ValueError, match="mixes ordinary-production and causal-arm"):
        recompute_bpref_high_precision._captured_source_reduction_policy(
            contribution, signature
        )


def test_causal_arm_policy_keeps_sequential_as_explicit_captured_reduction():
    from scripts import recompute_bpref_high_precision

    policy = recompute_bpref_high_precision._captured_source_reduction_policy(
        {},
        {
            "topology_claim": np.asarray(
                "causal-arm-not-relion-hypothesis-arithmetic-closure"
            ),
            "causal_arm": np.asarray("soft-posterior-per-particle-fused-xhalf"),
        },
    )

    assert policy["name"] == "relion-f32-sequential-translation-reduction"
    assert policy["sequential_translation_reduction"] is True
    assert policy["order_control_name"] == "ordinary-gemm-translation-reduction"


def test_serialized_probability_dtype_cannot_silently_determine_live_dtype():
    from scripts import recompute_bpref_high_precision

    contribution = {
        "reconstruction_probs": np.asarray([0.0, 0.25, 1.0], dtype=np.float64),
    }

    with pytest.raises(ValueError, match="lacks reconstruction_probs_native_dtype"):
        recompute_bpref_high_precision._production_reconstruction_probabilities(
            contribution,
            legacy_dtype=None,
        )

    restored, policy = (
        recompute_bpref_high_precision._production_reconstruction_probabilities(
            contribution,
            legacy_dtype="float32",
        )
    )
    assert restored.dtype == np.float32
    assert policy["source"] == "explicit-legacy-command-line-override"
    assert policy["storage_roundtrip_exact"] is True


def test_probability_dtype_metadata_rejects_conflicting_legacy_override():
    from scripts import recompute_bpref_high_precision

    contribution = {
        "reconstruction_probs": np.asarray([0.0, 1.0], dtype=np.float32),
        "reconstruction_probs_native_dtype": np.asarray("float32"),
        "reconstruction_probs_native_itemsize": np.int32(4),
        "reconstruction_probs_native_nbytes": np.int64(8),
        "reconstruction_probs_storage_policy": np.asarray(
            "native-dtype-preserved;dtype-itemsize-nbytes-bound"
        ),
    }

    with pytest.raises(ValueError, match="override conflicts"):
        recompute_bpref_high_precision._production_reconstruction_probabilities(
            contribution,
            legacy_dtype="float64",
        )


def test_probability_dtype_metadata_restores_and_validates_native_bytes():
    from scripts import recompute_bpref_high_precision

    values = np.asarray([[0.0, 0.25, 1.0]], dtype=np.float32)
    restored, policy = (
        recompute_bpref_high_precision._production_reconstruction_probabilities(
            {
                "reconstruction_probs": values,
                "reconstruction_probs_native_dtype": np.asarray("float32"),
                "reconstruction_probs_native_itemsize": np.int32(4),
                "reconstruction_probs_native_nbytes": np.int64(values.nbytes),
                "reconstruction_probs_storage_policy": np.asarray(
                    "native-dtype-preserved;dtype-itemsize-nbytes-bound"
                ),
            },
            legacy_dtype=None,
        )
    )

    assert np.array_equal(restored, values)
    assert restored.dtype == np.float32
    assert policy["source"] == "capture-native-dtype-metadata"
    assert policy["production_itemsize"] == "4"
    assert policy["stored_nbytes"] == str(values.nbytes)


def test_probability_dtype_restore_requires_exact_storage_roundtrip():
    from scripts import recompute_bpref_high_precision

    contribution = {
        "reconstruction_probs": np.asarray([0.1], dtype=np.float64),
    }

    with pytest.raises(ValueError, match="do not exactly round-trip"):
        recompute_bpref_high_precision._production_reconstruction_probabilities(
            contribution,
            legacy_dtype="float32",
        )


@pytest.mark.parametrize(
    ("changed", "classification"),
    [
        (
            {"source_weight": np.asarray([2.0], dtype=np.float32)},
            "operand_generation_difference",
        ),
        (
            {"target_indices": np.asarray([1], dtype=np.int32)},
            "discrete_geometry_difference",
        ),
    ],
)
def test_cross_engine_classifies_operand_and_geometry_boundaries(
    changed, classification
):
    base = _records((1.0,))

    report = validator.compare_contribution_engines(base, replace(base, **changed), 2)

    assert report["classification"] == classification


def test_extracts_all_eight_neighbors_and_applies_both_conjugation_flags():
    records = validator.extract_contribution_records(_minimal_signature())

    replay = validator.replay_contribution_records(
        records, 8, order="canonical", precision="float32"
    )

    assert records.size == 8
    assert np.array_equal(records.neighbor, np.arange(8, dtype=np.int32))
    assert np.array_equal(records.target_indices, np.arange(8, dtype=np.int32))
    expected = np.asarray(
        [2 + 3j] + [factor * (2 - 3j) for factor in range(2, 9)],
        dtype=np.complex64,
    )
    assert np.array_equal(replay.data, expected)
    assert np.array_equal(replay.weight, 5 * np.arange(1, 9, dtype=np.float32))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values: values.pop("source_values"), "missing required field"),
        (
            lambda values: values.update(
                neighbor_indices=np.zeros((1, 1, 7), dtype=np.int32)
            ),
            "neighbor_indices.*shape",
        ),
        (
            lambda values: values["neighbor_flags"].__setitem__((0, 0, 0), 16),
            "unknown neighbor flag bit",
        ),
    ],
)
def test_malformed_signature_schema_fails_closed(mutation, message):
    signature = _minimal_signature()
    mutation(signature)
    with pytest.raises(ValueError, match=message):
        validator.extract_contribution_records(signature)


def test_float64_capture_replay_is_labeled_as_cast_not_recomputation():
    report = validator.canonical_replay_diagnostics(_records(), 1)

    assert report["captured_operand_provenance"] == validator.CAPTURED_F32_CAST
    assert report["verified_recomputed_high_precision_available"] is False
    assert "cannot recover precision lost" in report["captured_f32_cast_limitation"]
    assert "correlation" not in str(report).lower()


def test_recomputed_high_precision_operands_require_true_high_precision_dtypes():
    records = replace(
        _records((1.0,)),
        recomputed_coefficients=np.ones(1, dtype=np.float32),
        recomputed_source_data=np.ones(1, dtype=np.complex64),
        recomputed_source_weight=np.ones(1, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="float64/complex128/float64"):
        validator.replay_contribution_records(
            records,
            1,
            order="canonical",
            precision="float64",
            operand_provenance=validator.RECOMPUTED_HIGH_PRECISION,
        )


def test_canonical_replay_rejects_incomplete_semantic_identity():
    records = replace(
        _records((1.0, 2.0)),
        original_index=np.asarray([7, 7], dtype=np.int64),
        canonical_rotation_key=np.asarray([11, 11], dtype=np.int32),
        dense_pixel=np.asarray([3, 3], dtype=np.int32),
        neighbor=np.asarray([2, 2], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="identity is not unique"):
        validator.replay_contribution_records(
            records, 1, order="canonical", precision="float32"
        )


def test_particle_local_row_difference_is_classified_as_logical_schedule():
    base = _records((1.0, 2.0))
    reordered = replace(
        base, particle_local_row=np.asarray([4, 3], dtype=np.int32)
    )

    report = validator.compare_contribution_engines(base, reordered, 1)

    assert report["identity_equal"] is True
    assert report["discrete_geometry_equal"] is True
    assert report["classification"] == "logical_schedule_difference"
