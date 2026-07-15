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
