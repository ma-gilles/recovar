#!/usr/bin/env python3
"""Validate and replay RECOVAR BPref device-scatter signature artifacts.

The signature captures exact device-computed scatter geometry for positive
contributor rows while the companion contribution dump proves that every
omitted row is exactly zero. Replay uses a deterministic lexicographic
``(launch, particle-local row, dense pixel, neighbor)`` logical host order. It
does not observe or reconstruct the CUDA atomic schedule. This script reports
(and optionally gates) relative L1 and accumulator FSC/FSC-AUC differences
rather than requiring bitwise equality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

SIGNATURE_MAGIC = "RECOVAR_DEVICE_SCATTER_SIGNATURE"
SIGNATURE_SCHEMA = "recovar-device-scatter-signature-v1"
PANEL_MAGIC = "RECOVAR_DEVICE_PANEL_NATIVE"
PANEL_SCHEMA = "recovar-device-panel-native-v1"
CONTRIBUTION_MAGIC = "RECOVAR_BPREF_CONTRIBUTION_ROWS"
CONTRIBUTION_SCHEMA = "recovar-bpref-contribution-rows-v3"
ROW_FLAG_MASK = 1 | 2 | 4 | 8 | 16 | 32 | 64
NEIGHBOR_FLAG_MASK = 1 | 2 | 4 | 8
CAPTURED_F32_CAST = "captured-f32-cast"
RECOMPUTED_HIGH_PRECISION = "recomputed-high-precision"
RECOMPUTATION_MAGIC = "RECOVAR_BPREF_F64_RECOMPUTATION"
RECOMPUTATION_SCHEMA = "recovar-bpref-f64-recomputation-v1"
RECOMPUTATION_NUMERIC_POLICY = "operands-recomputed-float64-complex128"
RECOMPUTATION_FORMULA_NAME = "recovar-bpref-trilinear-operands"
RECOMPUTATION_FORMULA_VERSION = "1"
RECOMPUTATION_SOURCE_POLICY = "upstream-sources-recomputed-float64-complex128"
_VERIFIED_RECOMPUTATION_TOKEN = object()


@dataclass(frozen=True)
class VerifiedRecomputationProvenance:
    """Validated provenance for operands recomputed above production precision.

    Instances that justify a ``precision`` classification are created only by
    :func:`load_verified_recomputation`.  The private token prevents arbitrary
    dataclass construction from being treated as evidence.
    """

    artifact_path: str
    artifact_sha256: str
    parent_signature_sha256: tuple[str, ...]
    companion_contribution_sha256: tuple[str, ...]
    semantic_identity_sha256: str
    formula_name: str
    formula_version: str
    numeric_policy: str
    source_dtype: str
    _validation_token: object | None = None


@dataclass(frozen=True)
class ContributionRecords:
    """One record per valid BPref neighbor contribution.

    ``source_data`` is the captured complex value before either recorded
    Hermitian fold.  The optional recomputed arrays must come from an actual
    high-precision operand computation; casting the captured arrays does not
    populate them.
    """

    target_indices: np.ndarray
    coefficients: np.ndarray
    source_data: np.ndarray
    source_weight: np.ndarray
    row_conjugated: np.ndarray
    neighbor_conjugated: np.ndarray
    launch_ordinal: np.ndarray
    particle_local_row: np.ndarray
    original_index: np.ndarray
    canonical_rotation_key: np.ndarray
    dense_pixel: np.ndarray
    neighbor: np.ndarray
    recomputed_coefficients: np.ndarray | None = None
    recomputed_source_data: np.ndarray | None = None
    recomputed_source_weight: np.ndarray | None = None
    recomputation_provenance: VerifiedRecomputationProvenance | None = None

    @property
    def size(self) -> int:
        return int(self.target_indices.size)

    @property
    def has_recomputed_high_precision(self) -> bool:
        optional = (
            self.recomputed_coefficients,
            self.recomputed_source_data,
            self.recomputed_source_weight,
        )
        return all(value is not None for value in optional)

    @property
    def has_verified_recomputed_high_precision(self) -> bool:
        provenance = self.recomputation_provenance
        return (
            self.has_recomputed_high_precision
            and provenance is not None
            and provenance._validation_token is _VERIFIED_RECOMPUTATION_TOKEN
        )


@dataclass(frozen=True)
class ReplayResult:
    data: np.ndarray
    weight: np.ndarray
    order: str
    precision: str
    operand_provenance: str


def _scalar(values: dict[str, np.ndarray], key: str):
    if key not in values:
        raise ValueError(f"missing required field {key!r}")
    value = np.asarray(values[key])
    if value.shape != ():
        raise ValueError(f"field {key!r} must be scalar, got {value.shape}")
    return value.item()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _require_header(values, *, magic, schema, version=1):
    if _scalar(values, "magic") != magic:
        raise ValueError(f"magic mismatch: {_scalar(values, 'magic')!r} != {magic!r}")
    if _scalar(values, "schema") != schema:
        raise ValueError(f"schema mismatch: {_scalar(values, 'schema')!r} != {schema!r}")
    if int(_scalar(values, "schema_version")) != version:
        raise ValueError(f"schema_version must be {version}")


def _noncontributor_digest(
    rows: np.ndarray,
    rotation_keys: np.ndarray,
    summed: np.ndarray,
    weights: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(rows, dtype=np.int32).tobytes(order="C"))
    digest.update(np.ascontiguousarray(rotation_keys, dtype=np.int32).tobytes(order="C"))
    for values in (np.ascontiguousarray(summed), np.ascontiguousarray(weights)):
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_companion_identity(signature, contribution, contribution_path: Path):
    _require_header(
        contribution,
        magic=CONTRIBUTION_MAGIC,
        schema=CONTRIBUTION_SCHEMA,
        version=3,
    )
    expected_path = Path(str(_scalar(signature, "companion_contribution_path")))
    if not expected_path.is_absolute() or expected_path.resolve() != contribution_path.resolve():
        raise ValueError("companion contribution path is not the resolved input path")
    expected_sha = str(_scalar(signature, "companion_contribution_sha256"))
    actual_sha = _sha256_file(contribution_path)
    if expected_sha != actual_sha:
        raise ValueError(f"companion SHA256 mismatch: {expected_sha} != {actual_sha}")
    for key in ("run_id", "iteration", "half", "rank", "pass_index", "class_index", "call_index", "dump_index"):
        if _scalar(signature, key) != _scalar(contribution, key):
            raise ValueError(f"signature/contribution identity mismatch for {key}")
    if _scalar(signature, "source_stack_sha256") != _scalar(
        contribution, "source_stack_sha256"
    ):
        raise ValueError("source-stack SHA256 mismatch")


def _require_array(values, key, *, shape=None, dtype=None):
    if key not in values:
        raise ValueError(f"missing required field {key!r}")
    array = np.asarray(values[key])
    if shape is not None and array.shape != tuple(shape):
        raise ValueError(f"field {key!r} shape {array.shape} != {tuple(shape)}")
    if dtype is not None:
        expected = (dtype,) if isinstance(dtype, np.dtype) else tuple(dtype)
        expected = tuple(np.dtype(value) for value in expected)
        if array.dtype not in expected:
            names = ", ".join(str(value) for value in expected)
            raise ValueError(f"field {key!r} dtype {array.dtype} not in {{{names}}}")
    return array


def _require_finite(values, key, **kwargs):
    array = _require_array(values, key, **kwargs)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"field {key!r} contains nonfinite values")
    return array


def _semantic_identity_digest(records: ContributionRecords) -> str:
    """Hash semantic identity and discrete geometry in canonical order."""

    order = _semantic_order(records)
    digest = hashlib.sha256()
    for key in (
        "original_index",
        "canonical_rotation_key",
        "dense_pixel",
        "neighbor",
        "target_indices",
        "row_conjugated",
        "neighbor_conjugated",
    ):
        values = np.ascontiguousarray(getattr(records, key)[order])
        digest.update(key.encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validated_sha256_vector(values, key: str) -> tuple[str, ...]:
    array = _require_array(values, key)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"field {key!r} must be a nonempty vector")
    result = tuple(str(value) for value in array)
    if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value) for value in result):
        raise ValueError(f"field {key!r} contains an invalid lowercase SHA256")
    return result


def load_verified_recomputation(
    path: Path,
    records: ContributionRecords,
    *,
    parent_signature_paths: tuple[Path, ...],
    companion_contribution_paths: tuple[Path, ...],
) -> ContributionRecords:
    """Load and validate a versioned float64/complex128 operand artifact.

    The artifact must freeze the parent signature and companion hashes and the
    exact semantic contribution identity. Merely attaching float64 arrays to a
    :class:`ContributionRecords` instance is intentionally unverified.
    """

    _validate_contribution_records(records)
    artifact_path = Path(path).resolve()
    artifact = _load(artifact_path)
    _require_header(
        artifact,
        magic=RECOMPUTATION_MAGIC,
        schema=RECOMPUTATION_SCHEMA,
    )
    if not parent_signature_paths or not companion_contribution_paths:
        raise ValueError("recomputation provenance requires parent and companion files")
    expected_parent = tuple(_sha256_file(Path(item)) for item in parent_signature_paths)
    expected_companion = tuple(
        _sha256_file(Path(item)) for item in companion_contribution_paths
    )
    recorded_parent = _validated_sha256_vector(artifact, "parent_signature_sha256")
    recorded_companion = _validated_sha256_vector(
        artifact, "companion_contribution_sha256"
    )
    if recorded_parent != expected_parent:
        raise ValueError("recomputation parent-signature hashes do not match frozen inputs")
    if recorded_companion != expected_companion:
        raise ValueError("recomputation companion hashes do not match frozen inputs")
    semantic_digest = _semantic_identity_digest(records)
    if str(_scalar(artifact, "semantic_identity_sha256")) != semantic_digest:
        raise ValueError("recomputation semantic identity digest mismatch")

    formula_name = str(_scalar(artifact, "formula_name"))
    formula_version = str(_scalar(artifact, "formula_version"))
    numeric_policy = str(_scalar(artifact, "numeric_policy"))
    source_dtype = str(_scalar(artifact, "source_dtype"))
    if formula_name != RECOMPUTATION_FORMULA_NAME:
        raise ValueError("recomputation formula is not recognized")
    if formula_version != RECOMPUTATION_FORMULA_VERSION:
        raise ValueError("recomputation formula version is not recognized")
    if numeric_policy != RECOMPUTATION_NUMERIC_POLICY:
        raise ValueError("recomputation numeric policy is not recognized")
    if source_dtype != RECOMPUTATION_SOURCE_POLICY:
        raise ValueError(
            "recomputation source policy must certify upstream float64/complex128 "
            "operand generation; captured-float32 promotion is not sufficient"
        )

    coefficients = _require_finite(
        artifact,
        "recomputed_coefficients",
        shape=(records.size,),
        dtype=np.dtype(np.float64),
    )
    source_data = _require_finite(
        artifact,
        "recomputed_source_data",
        shape=(records.size,),
        dtype=np.dtype(np.complex128),
    )
    source_weight = _require_finite(
        artifact,
        "recomputed_source_weight",
        shape=(records.size,),
        dtype=np.dtype(np.float64),
    )
    provenance = VerifiedRecomputationProvenance(
        artifact_path=str(artifact_path),
        artifact_sha256=_sha256_file(artifact_path),
        parent_signature_sha256=recorded_parent,
        companion_contribution_sha256=recorded_companion,
        semantic_identity_sha256=semantic_digest,
        formula_name=formula_name,
        formula_version=formula_version,
        numeric_policy=numeric_policy,
        source_dtype=source_dtype,
        _validation_token=_VERIFIED_RECOMPUTATION_TOKEN,
    )
    verified = replace(
        records,
        recomputed_coefficients=coefficients,
        recomputed_source_data=source_data,
        recomputed_source_weight=source_weight,
        recomputation_provenance=provenance,
    )
    _validate_contribution_records(verified)
    return verified


def _validate_v3_replay_bundle(contribution):
    """Validate the v3 candidate replay and optional source-operand bundle."""

    image_shape = tuple(int(value) for value in _require_array(
        contribution, "image_shape", shape=(2,), dtype=np.dtype(np.int32)
    ))
    volume_shape = tuple(int(value) for value in _require_array(
        contribution, "volume_shape", shape=(3,), dtype=np.dtype(np.int32)
    ))
    if min(image_shape) <= 0 or min(volume_shape) <= 0:
        raise ValueError("contribution image/volume shapes must be positive")

    original_indices = _require_array(
        contribution, "original_indices", dtype=np.dtype(np.int64)
    )
    if original_indices.ndim != 1 or original_indices.size == 0:
        raise ValueError("contribution original_indices must be a nonempty vector")
    particle_count = original_indices.size
    vector_identity_fields = {
        "local_indices": np.dtype(np.int64),
        "star_rows": np.dtype(np.int64),
        "stack_indices_1based": np.dtype(np.int64),
    }
    for key, dtype in vector_identity_fields.items():
        _require_array(contribution, key, shape=(particle_count,), dtype=dtype)
    for key in ("image_identities", "resolved_stack_paths"):
        _require_array(contribution, key, shape=(particle_count,))
    if not np.array_equal(contribution["star_rows"], original_indices):
        raise ValueError("contribution star_rows/original_indices identity mismatch")
    expected_stack_indices = np.asarray(
        [int(str(value).split("@", 1)[0]) for value in contribution["image_identities"]],
        dtype=np.int64,
    )
    expected_stack_paths = np.asarray(
        [str(value).split("@", 1)[1] for value in contribution["image_identities"]]
    )
    if not np.array_equal(contribution["stack_indices_1based"], expected_stack_indices):
        raise ValueError("contribution stack-index/image identity mismatch")
    if not np.array_equal(contribution["resolved_stack_paths"], expected_stack_paths):
        raise ValueError("contribution stack-path/image identity mismatch")

    actual_counts = _require_array(
        contribution, "actual_counts", shape=(particle_count,), dtype=np.dtype(np.int64)
    )
    rotations = _require_array(
        contribution, "oversampled_rotation_indices", dtype=np.dtype(np.int64)
    )
    if rotations.ndim != 2 or rotations.shape[0] != particle_count:
        raise ValueError("oversampled_rotation_indices must have shape (particles, rotations)")
    rotation_count = rotations.shape[1]
    if np.any(actual_counts < 0) or np.any(actual_counts > rotation_count):
        raise ValueError("actual_counts lies outside the captured rotation axis")

    combined = _require_array(
        contribution, "candidate_combined_scores", dtype=np.dtype(np.float64)
    )
    if combined.ndim != 3 or combined.shape[:2] != (particle_count, rotation_count):
        raise ValueError("candidate_combined_scores must have shape (particles, rotations, translations)")
    candidate_shape = combined.shape
    translation_count = candidate_shape[2]
    for key in ("candidate_preprior_scores", "posterior_probs", "reconstruction_probs"):
        _require_array(contribution, key, shape=candidate_shape, dtype=np.dtype(np.float64))
    for key in ("candidate_mask", "reconstruction_mask"):
        _require_array(contribution, key, shape=candidate_shape, dtype=np.dtype(np.bool_))
    _require_array(
        contribution, "candidate_raw_exp_weights_f32", shape=candidate_shape, dtype=np.dtype(np.float32)
    )
    _require_array(
        contribution, "candidate_rotation_log_prior",
        shape=(particle_count, rotation_count), dtype=np.dtype(np.float64),
    )
    _require_array(
        contribution, "candidate_translation_log_prior",
        shape=(particle_count, translation_count), dtype=np.dtype(np.float64),
    )
    for key in (
        "candidate_best_log_score",
        "candidate_log_z",
        "candidate_normalized_sum_exp",
        "reconstruction_sum_weight",
        "reconstruction_threshold",
    ):
        _require_array(contribution, key, shape=(particle_count,), dtype=np.dtype(np.float64))
    _require_array(
        contribution, "candidate_exponent_shift_f32",
        shape=(particle_count,), dtype=np.dtype(np.float32),
    )
    _require_finite(
        contribution, "fine_translations", shape=(translation_count, 2), dtype=np.dtype(np.float32)
    )
    candidate_mask = np.asarray(contribution["candidate_mask"], dtype=bool)
    reconstruction_mask = np.asarray(contribution["reconstruction_mask"], dtype=bool)
    if np.any(reconstruction_mask & ~candidate_mask):
        raise ValueError("reconstruction_mask is not a subset of candidate_mask")
    for key in ("posterior_probs", "reconstruction_probs"):
        probabilities = np.asarray(contribution[key])
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError(f"field {key!r} must contain finite nonnegative probabilities")
    if np.any(np.asarray(contribution["posterior_probs"])[~candidate_mask] != 0):
        raise ValueError("posterior_probs must be zero outside candidate_mask")
    if np.any(np.asarray(contribution["reconstruction_probs"])[~reconstruction_mask] != 0):
        raise ValueError("reconstruction_probs must be zero outside reconstruction_mask")
    finite_candidate = np.isfinite(combined)
    if np.any(candidate_mask & ~finite_candidate) or np.any(np.isnan(combined)):
        raise ValueError("candidate_combined_scores must be finite on candidate_mask and never NaN")

    shadow_only = bool(_require_array(
        contribution, "shadow_only_mode", shape=(), dtype=np.dtype(np.bool_)
    ).item())
    shadow_score_equal = bool(_require_array(
        contribution, "shadow_score_bitwise_equal", shape=(), dtype=np.dtype(np.bool_)
    ).item())
    shadow_metric_keys = (
        "shadow_reduction_data_rel_l1",
        "shadow_reduction_data_normalized_max",
        "shadow_reduction_weight_rel_l1",
        "shadow_reduction_weight_normalized_max",
        "shadow_reduction_rel_l1_bound",
        "shadow_reduction_normalized_max_bound",
    )
    shadow_metrics = {key: float(_scalar(contribution, key)) for key in shadow_metric_keys}
    if shadow_only:
        if not shadow_score_equal:
            raise ValueError("shadow-only contribution did not certify exact score agreement")
        if not all(np.isfinite(value) and value >= 0 for value in shadow_metrics.values()):
            raise ValueError("shadow-only contribution reduction metrics must be finite and nonnegative")
        if (
            shadow_metrics["shadow_reduction_data_rel_l1"]
            > shadow_metrics["shadow_reduction_rel_l1_bound"]
            or shadow_metrics["shadow_reduction_weight_rel_l1"]
            > shadow_metrics["shadow_reduction_rel_l1_bound"]
            or shadow_metrics["shadow_reduction_data_normalized_max"]
            > shadow_metrics["shadow_reduction_normalized_max_bound"]
            or shadow_metrics["shadow_reduction_weight_normalized_max"]
            > shadow_metrics["shadow_reduction_normalized_max_bound"]
        ):
            raise ValueError("shadow-only contribution reduction metrics exceed their recorded bounds")

    high_precision_field = _require_array(
        contribution, "high_precision_operand_bundle", shape=(), dtype=np.dtype(np.bool_)
    )
    high_precision = bool(high_precision_field.item())
    raw_images = _require_array(contribution, "raw_real_images", dtype=np.dtype(np.float32))
    raw_shape = _require_array(
        contribution, "raw_source_shape", dtype=np.dtype(np.int64)
    )
    if not np.array_equal(raw_shape, np.asarray(raw_images.shape, dtype=np.int64)):
        raise ValueError("raw_source_shape does not match raw_real_images")
    raw_source_dtype = str(_scalar(contribution, "raw_source_dtype"))
    high_precision_fields = (
        "ctf_params",
        "noise_variance_half",
        "integer_pre_shifts",
        "image_corrections",
        "scale_corrections",
        "relion_preprocess_normalization_factors",
        "image_mask",
    )
    if high_precision:
        _require_finite(
            contribution, "raw_real_images",
            shape=(particle_count, *image_shape), dtype=np.dtype(np.float32),
        )
        if not raw_source_dtype:
            raise ValueError("raw_source_dtype must record the pre-cast source dtype")
        ctf = _require_finite(
            contribution, "ctf_params", dtype=(np.dtype(np.float32), np.dtype(np.float64))
        )
        if ctf.ndim != 2 or ctf.shape[0] != particle_count or ctf.shape[1] == 0:
            raise ValueError("ctf_params must have shape (particles, parameters)")
        noise = _require_finite(
            contribution, "noise_variance_half", dtype=(np.dtype(np.float32), np.dtype(np.float64))
        )
        if noise.ndim != 1 or noise.size == 0:
            raise ValueError("noise_variance_half must be a nonempty vector")
        _require_array(
            contribution, "integer_pre_shifts",
            shape=(particle_count, 2), dtype=np.dtype(np.int32),
        )
        for key in (
            "image_corrections",
            "scale_corrections",
            "relion_preprocess_normalization_factors",
        ):
            _require_finite(
                contribution, key, shape=(particle_count,), dtype=np.dtype(np.float32)
            )
        _require_finite(
            contribution, "image_mask", shape=image_shape, dtype=np.dtype(np.float32)
        )
    else:
        if raw_source_dtype:
            raise ValueError("raw_source_dtype must be empty when the high-precision bundle is disabled")
        expected_empty = {
            "raw_real_images": ((0,), np.dtype(np.float32)),
            "ctf_params": ((0,), np.dtype(np.float32)),
            "noise_variance_half": ((0,), np.dtype(np.float32)),
            "integer_pre_shifts": ((0, 2), np.dtype(np.int32)),
            "image_corrections": ((0,), np.dtype(np.float32)),
            "scale_corrections": ((0,), np.dtype(np.float32)),
            "relion_preprocess_normalization_factors": ((0,), np.dtype(np.float32)),
            "image_mask": ((0,), np.dtype(np.float32)),
        }
        for key, (shape, dtype) in expected_empty.items():
            _require_array(contribution, key, shape=shape, dtype=dtype)

    for key in ("relion_cuda_preprocess", "score_with_masked_images"):
        _require_array(contribution, key, shape=(), dtype=np.dtype(np.bool_))
    for key in (
        "preprocess_backend",
        "preprocess_convention",
        "image_mask_mode",
        "ctf_mode",
        "disc_type",
        "ctf_parameter_convention",
    ):
        if not str(_scalar(contribution, key)):
            raise ValueError(f"field {key!r} must be a nonempty string")
    for key in ("voxel_size", "ctf_dose_per_tilt", "ctf_angle_per_tilt"):
        if not np.isfinite(float(_scalar(contribution, key))):
            raise ValueError(f"field {key!r} must be finite")
    for key in ("projection_padding_factor", "reconstruction_padding_factor"):
        if int(_scalar(contribution, key)) <= 0:
            raise ValueError(f"field {key!r} must be positive")

    validated_fields = sorted(
        set(vector_identity_fields)
        | set(high_precision_fields)
        | {
            "raw_real_images", "raw_source_dtype", "raw_source_shape",
            "candidate_preprior_scores", "candidate_combined_scores",
            "posterior_probs", "reconstruction_probs", "candidate_mask",
            "reconstruction_mask", "oversampled_rotation_indices", "fine_translations",
        }
    )
    return {
        "schema_replay_ready": True,
        "shadow_only_mode": shadow_only,
        "shadow_score_bitwise_equal": shadow_score_equal,
        "shadow_reduction_metrics": shadow_metrics if shadow_only else None,
        "high_precision_operand_bundle": high_precision,
        "particle_count": int(particle_count),
        "rotation_count": int(rotation_count),
        "translation_count": int(translation_count),
        "raw_source_dtype": raw_source_dtype or None,
        "validated_fields": validated_fields,
    }


def reconstruct_atomic_tuples(signature: dict[str, np.ndarray], volume_size: int):
    """Return valid-neighbor triples in canonical row/pixel/neighbor order."""

    row_flags = np.asarray(signature["row_flags"], dtype=np.int32)
    source = np.asarray(signature["source_values"], dtype=np.float32)
    neighbor_indices = np.asarray(signature["neighbor_indices"], dtype=np.int32)
    coefficients = np.asarray(signature["neighbor_coefficients"], dtype=np.float32)
    neighbor_flags = np.asarray(signature["neighbor_flags"], dtype=np.int32)
    valid = ((row_flags & 64) != 0)[..., None] & ((neighbor_flags & 1) != 0)
    row_index, pixel_index, neighbor_index = np.nonzero(valid)
    targets = neighbor_indices[row_index, pixel_index, neighbor_index]
    if np.any(targets < 0) or np.any(targets >= volume_size):
        raise ValueError("valid neighbor index lies outside the native accumulator")
    coeff = coefficients[row_index, pixel_index, neighbor_index]
    if not np.all(np.isfinite(coeff)):
        raise ValueError("valid neighbor coefficient is nonfinite")
    source_values = source[row_index, pixel_index]
    if not np.all(np.isfinite(source_values)):
        raise ValueError("reached-scatter source tuple is nonfinite")
    data_re = source_values[:, 0]
    data_im = source_values[:, 1]
    data_im = np.where((row_flags[row_index, pixel_index] & 16) != 0, -data_im, data_im)
    data_im = np.where((neighbor_flags[row_index, pixel_index, neighbor_index] & 2) != 0, -data_im, data_im)
    atomic_values = np.stack(
        (
            (coeff * data_re).astype(np.float32),
            (coeff * data_im).astype(np.float32),
            (coeff * source_values[:, 2]).astype(np.float32),
        ),
        axis=1,
    )
    program = np.stack((row_index, pixel_index, neighbor_index), axis=1).astype(np.int32)
    return targets.astype(np.int32, copy=False), atomic_values, program


def extract_contribution_records(signature: dict[str, np.ndarray]) -> ContributionRecords:
    """Purely extract valid-neighbor records from a validated schema-v1 signature."""

    _require_header(signature, magic=SIGNATURE_MAGIC, schema=SIGNATURE_SCHEMA)
    row_flags = _require_array(signature, "row_flags", dtype=np.dtype(np.int32))
    if row_flags.ndim != 2:
        raise ValueError("row_flags must have shape (contributor rows, dense pixels)")
    row_count, pixel_count = row_flags.shape
    source = _require_array(
        signature,
        "source_values",
        shape=(row_count, pixel_count, 6),
        dtype=np.dtype(np.float32),
    )
    neighbor_shape = (row_count, pixel_count, 8)
    targets = _require_array(
        signature, "neighbor_indices", shape=neighbor_shape, dtype=np.dtype(np.int32)
    )
    coefficients = _require_array(
        signature,
        "neighbor_coefficients",
        shape=neighbor_shape,
        dtype=np.dtype(np.float32),
    )
    neighbor_flags = _require_array(
        signature, "neighbor_flags", shape=neighbor_shape, dtype=np.dtype(np.int32)
    )
    if np.any(row_flags & ~ROW_FLAG_MASK):
        raise ValueError("unknown row flag bit")
    if np.any(neighbor_flags & ~NEIGHBOR_FLAG_MASK):
        raise ValueError("unknown neighbor flag bit")
    primary_bits = np.stack(
        [(row_flags & bit) != 0 for bit in (1, 2, 4, 8, 32, 64)], axis=-1
    )
    if np.any(np.sum(primary_bits, axis=-1) != 1):
        raise ValueError("each row-pixel must have exactly one primary gate/scatter state")
    reached = (row_flags & 64) != 0
    valid = (neighbor_flags & 1) != 0
    if np.any(~np.isin(neighbor_flags, np.asarray([1, 3, 5, 8], dtype=np.int32))):
        raise ValueError("neighbor flag value is outside {1,3,5,8}")
    if np.any(valid & ~reached[..., None]):
        raise ValueError("valid neighbor belongs to a row that did not reach scatter")
    invalid = ~valid
    if np.any(targets[invalid] != -1) or np.any(coefficients[invalid] != 0):
        raise ValueError("invalid neighbor does not carry index/weight sentinels")
    if np.any(~np.isfinite(coefficients[valid])):
        raise ValueError("valid neighbor coefficient is nonfinite")
    if np.any(targets[valid] < 0):
        raise ValueError("valid neighbor index is negative")

    row_fields = {
        "launch_ordinal": np.int64,
        "particle_local_row": np.int32,
        "original_indices": np.int64,
        "contributor_canonical_rotation_keys": np.int32,
    }
    rows = {}
    for key, dtype in row_fields.items():
        rows[key] = _require_array(
            signature, key, shape=(row_count,), dtype=np.dtype(dtype)
        )
    row_index, pixel_index, neighbor_index = np.nonzero(reached[..., None] & valid)
    source_values = source[row_index, pixel_index]
    if not np.all(np.isfinite(source_values)):
        raise ValueError("reached-scatter source tuple is nonfinite")
    records = ContributionRecords(
        target_indices=targets[row_index, pixel_index, neighbor_index],
        coefficients=coefficients[row_index, pixel_index, neighbor_index],
        source_data=(source_values[:, 0] + 1j * source_values[:, 1]).astype(
            np.complex64
        ),
        source_weight=source_values[:, 2],
        row_conjugated=(row_flags[row_index, pixel_index] & 16) != 0,
        neighbor_conjugated=(
            neighbor_flags[row_index, pixel_index, neighbor_index] & 2
        )
        != 0,
        launch_ordinal=rows["launch_ordinal"][row_index],
        particle_local_row=rows["particle_local_row"][row_index],
        original_index=rows["original_indices"][row_index],
        canonical_rotation_key=rows["contributor_canonical_rotation_keys"][row_index],
        dense_pixel=pixel_index.astype(np.int32),
        neighbor=neighbor_index.astype(np.int32),
    )
    _validate_contribution_records(records)
    return records


def concatenate_contribution_records(parts: list[ContributionRecords]) -> ContributionRecords:
    """Concatenate signature shards without changing either replay ordering key."""

    if not parts:
        raise ValueError("at least one contribution-record shard is required")
    fields = (
        "target_indices",
        "coefficients",
        "source_data",
        "source_weight",
        "row_conjugated",
        "neighbor_conjugated",
        "launch_ordinal",
        "particle_local_row",
        "original_index",
        "canonical_rotation_key",
        "dense_pixel",
        "neighbor",
    )
    kwargs = {key: np.concatenate([getattr(part, key) for part in parts]) for key in fields}
    if all(part.has_recomputed_high_precision for part in parts):
        for key in (
            "recomputed_coefficients",
            "recomputed_source_data",
            "recomputed_source_weight",
        ):
            kwargs[key] = np.concatenate([getattr(part, key) for part in parts])
    elif any(part.has_recomputed_high_precision for part in parts):
        raise ValueError("recomputed high-precision operands are missing from some shards")
    # A concatenated operand set requires its own artifact and semantic digest.
    # Do not propagate per-shard verification to a new combined identity.
    records = ContributionRecords(**kwargs)
    _validate_contribution_records(records)
    return records


def _validate_contribution_records(records: ContributionRecords) -> None:
    fields = (
        "target_indices",
        "coefficients",
        "source_data",
        "source_weight",
        "row_conjugated",
        "neighbor_conjugated",
        "launch_ordinal",
        "particle_local_row",
        "original_index",
        "canonical_rotation_key",
        "dense_pixel",
        "neighbor",
    )
    if records.size == 0:
        raise ValueError("contribution records must not be empty")
    if any(np.asarray(getattr(records, key)).shape != (records.size,) for key in fields):
        raise ValueError("every contribution-record field must be a same-length vector")
    expected_dtypes = {
        "target_indices": np.dtype(np.int32),
        "coefficients": np.dtype(np.float32),
        "source_data": np.dtype(np.complex64),
        "source_weight": np.dtype(np.float32),
        "row_conjugated": np.dtype(np.bool_),
        "neighbor_conjugated": np.dtype(np.bool_),
        "launch_ordinal": np.dtype(np.int64),
        "particle_local_row": np.dtype(np.int32),
        "original_index": np.dtype(np.int64),
        "canonical_rotation_key": np.dtype(np.int32),
        "dense_pixel": np.dtype(np.int32),
        "neighbor": np.dtype(np.int32),
    }
    for key, expected_dtype in expected_dtypes.items():
        actual_dtype = np.asarray(getattr(records, key)).dtype
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"{key} must have dtype {expected_dtype}, got {actual_dtype}"
            )
    if np.any(records.target_indices < 0):
        raise ValueError("target_indices contains a negative index")
    for key in ("coefficients", "source_data", "source_weight"):
        if not np.all(np.isfinite(getattr(records, key))):
            raise ValueError(f"{key} contains nonfinite values")
    if np.any((records.neighbor < 0) | (records.neighbor >= 8)):
        raise ValueError("neighbor lies outside [0, 8)")
    optional = (
        records.recomputed_coefficients,
        records.recomputed_source_data,
        records.recomputed_source_weight,
    )
    if any(value is not None for value in optional) and not all(
        value is not None for value in optional
    ):
        raise ValueError("recomputed high-precision operand fields are incomplete")
    provenance = records.recomputation_provenance
    if provenance is not None and not records.has_recomputed_high_precision:
        raise ValueError("recomputation provenance has no complete operand arrays")
    if provenance is not None and provenance._validation_token is not _VERIFIED_RECOMPUTATION_TOKEN:
        raise ValueError("recomputation provenance was not produced by the validated loader")
    if records.has_recomputed_high_precision:
        expected_recomputed_dtypes = (
            np.dtype(np.float64),
            np.dtype(np.complex128),
            np.dtype(np.float64),
        )
        for value, expected_dtype in zip(optional, expected_recomputed_dtypes):
            if np.asarray(value).shape != (records.size,) or not np.all(np.isfinite(value)):
                raise ValueError("recomputed high-precision operands are malformed")
            if np.asarray(value).dtype != expected_dtype:
                raise ValueError(
                    "recomputed high-precision operands must have dtypes "
                    "float64/complex128/float64"
                )
    if records.has_verified_recomputed_high_precision:
        if provenance.semantic_identity_sha256 != _semantic_identity_digest(records):
            raise ValueError("verified recomputation semantic identity no longer matches records")


def _semantic_order(records: ContributionRecords) -> np.ndarray:
    """Return the common deterministic order, rejecting ambiguous identities."""

    identity_fields = (
        records.original_index,
        records.canonical_rotation_key,
        records.dense_pixel,
        records.neighbor,
    )
    order = np.lexsort(tuple(reversed(identity_fields)))
    if records.size > 1:
        duplicate = np.ones(records.size - 1, dtype=bool)
        for field in identity_fields:
            sorted_field = field[order]
            duplicate &= sorted_field[1:] == sorted_field[:-1]
        if np.any(duplicate):
            raise ValueError(
                "canonical contribution identity is not unique; capture a complete "
                "semantic identity before canonical replay"
            )
    return order


def _record_order(records: ContributionRecords, order: str) -> np.ndarray:
    if order == "logical_host_order":
        keys = (
            records.neighbor,
            records.dense_pixel,
            records.particle_local_row,
            records.launch_ordinal,
        )
    elif order == "canonical":
        # This excludes launch and particle-local row because they describe
        # program scheduling rather than a cross-engine semantic identity.
        return _semantic_order(records)
    else:
        raise ValueError(f"unknown replay order {order!r}")
    return np.lexsort(keys)


def replay_contribution_records(
    records: ContributionRecords,
    volume_size: int,
    *,
    order: str,
    precision: str,
    operand_provenance: str = CAPTURED_F32_CAST,
) -> ReplayResult:
    """Replay contribution records with explicit order, precision, and provenance."""

    _validate_contribution_records(records)
    if volume_size <= 0 or np.any(records.target_indices >= volume_size):
        raise ValueError("target index lies outside the replay accumulator")
    if precision not in {"float32", "float64"}:
        raise ValueError(f"unknown replay precision {precision!r}")
    if operand_provenance == RECOMPUTED_HIGH_PRECISION:
        if precision != "float64":
            raise ValueError("recomputed high-precision operands require float64 replay")
        if not records.has_verified_recomputed_high_precision:
            raise ValueError(
                "recomputed high-precision operands lack validated provenance"
            )
        coefficients = records.recomputed_coefficients
        source_data = records.recomputed_source_data
        source_weight = records.recomputed_source_weight
    elif operand_provenance == CAPTURED_F32_CAST:
        coefficients = records.coefficients
        source_data = records.source_data
        source_weight = records.source_weight
    else:
        raise ValueError(f"unknown operand provenance {operand_provenance!r}")

    dtype = np.float32 if precision == "float32" else np.float64
    complex_dtype = np.complex64 if precision == "float32" else np.complex128
    indices = _record_order(records, order)
    coefficients = np.asarray(coefficients, dtype=dtype)[indices]
    source_data = np.asarray(source_data, dtype=complex_dtype)[indices]
    source_weight = np.asarray(source_weight, dtype=dtype)[indices]
    conjugated = np.logical_xor(
        records.row_conjugated[indices], records.neighbor_conjugated[indices]
    )
    effective_data = np.where(conjugated, np.conj(source_data), source_data)
    data_values = (coefficients * effective_data).astype(complex_dtype)
    weight_values = (coefficients * source_weight).astype(dtype)
    targets = records.target_indices[indices]
    data = np.zeros(volume_size, dtype=complex_dtype)
    weight = np.zeros(volume_size, dtype=dtype)
    np.add.at(data, targets, data_values)
    np.add.at(weight, targets, weight_values)
    return ReplayResult(data, weight, order, precision, operand_provenance)


def exact_array_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    """Return exact/absolute array diagnostics; deliberately no correlation."""

    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    promoted = np.result_type(left.dtype, right.dtype)
    if np.issubdtype(promoted, np.bool_):
        promoted = np.dtype(np.int8)
    delta = np.asarray(left, dtype=promoted) - np.asarray(right, dtype=promoted)
    abs_delta = np.abs(delta).astype(np.float64)
    denom_l1 = max(float(np.sum(np.abs(right), dtype=np.float64)), np.finfo(np.float64).tiny)
    denom_l2 = max(float(np.linalg.norm(np.asarray(right).ravel())), np.finfo(np.float64).tiny)
    return {
        "shape_equal": True,
        "array_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "max_abs": float(np.max(abs_delta, initial=0.0)),
        "relative_l1": float(np.sum(abs_delta, dtype=np.float64) / denom_l1),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / denom_l2),
    }


def _replay_metrics(left: ReplayResult, right: ReplayResult) -> dict:
    return {
        "left": {
            "order": left.order,
            "precision": left.precision,
            "operand_provenance": left.operand_provenance,
        },
        "right": {
            "order": right.order,
            "precision": right.precision,
            "operand_provenance": right.operand_provenance,
        },
        "data": exact_array_metrics(left.data, right.data),
        "weight": exact_array_metrics(left.weight, right.weight),
    }


def _provenance_report(provenance):
    if provenance is None:
        return None
    return {
        "artifact_path": provenance.artifact_path,
        "artifact_sha256": provenance.artifact_sha256,
        "parent_signature_sha256": list(provenance.parent_signature_sha256),
        "companion_contribution_sha256": list(
            provenance.companion_contribution_sha256
        ),
        "semantic_identity_sha256": provenance.semantic_identity_sha256,
        "formula_name": provenance.formula_name,
        "formula_version": provenance.formula_version,
        "numeric_policy": provenance.numeric_policy,
        "source_dtype": provenance.source_dtype,
        "validated": (
            provenance._validation_token is _VERIFIED_RECOMPUTATION_TOKEN
        ),
    }


def canonical_replay_diagnostics(
    records: ContributionRecords,
    volume_size: int,
) -> dict:
    """Compare logical-host/canonical order and captured-cast precision."""

    replays = {
        (order, precision): replay_contribution_records(
            records, volume_size, order=order, precision=precision
        )
        for order in ("logical_host_order", "canonical")
        for precision in ("float32", "float64")
    }
    return {
        "captured_operand_provenance": CAPTURED_F32_CAST,
        "captured_f32_cast_limitation": (
            "float64/complex128 replay casts captured float32 operands and cannot "
            "recover precision lost during upstream operand generation"
        ),
        "caller_supplied_high_precision_arrays_present": records.has_recomputed_high_precision,
        "verified_recomputed_high_precision_available": (
            records.has_verified_recomputed_high_precision
        ),
        "logical_host_vs_canonical_float32": _replay_metrics(
            replays[("logical_host_order", "float32")],
            replays[("canonical", "float32")],
        ),
        "logical_host_vs_canonical_float64": _replay_metrics(
            replays[("logical_host_order", "float64")],
            replays[("canonical", "float64")],
        ),
        "precision_in_canonical_order": _replay_metrics(
            replays[("canonical", "float32")], replays[("canonical", "float64")]
        ),
    }


def compare_contribution_engines(
    left: ContributionRecords,
    right: ContributionRecords,
    volume_size: int,
) -> dict:
    """Compare two captured engines and classify the earliest supported boundary."""

    _validate_contribution_records(left)
    _validate_contribution_records(right)
    left_order = _semantic_order(left)
    right_order = _semantic_order(right)
    identity_fields = (
        "original_index",
        "canonical_rotation_key",
        "dense_pixel",
        "neighbor",
    )
    discrete_geometry_fields = (
        "target_indices",
        "row_conjugated",
        "neighbor_conjugated",
    )
    captured_operand_fields = ("coefficients", "source_data", "source_weight")
    logical_schedule_fields = ("launch_ordinal", "particle_local_row")
    compared_fields = (
        identity_fields
        + discrete_geometry_fields
        + captured_operand_fields
        + logical_schedule_fields
    )
    field_metrics = {
        key: exact_array_metrics(
            getattr(left, key)[left_order], getattr(right, key)[right_order]
        )
        for key in compared_fields
    }

    def fields_equal(fields):
        return left.size == right.size and all(
            field_metrics[key].get("array_equal", False) for key in fields
        )

    same_identity = fields_equal(identity_fields)
    same_geometry = same_identity and fields_equal(discrete_geometry_fields)
    same_operands = same_identity and fields_equal(captured_operand_fields)
    same_logical_schedule = same_identity and fields_equal(logical_schedule_fields)

    cross = {}
    for order in ("logical_host_order", "canonical"):
        for precision in ("float32", "float64"):
            left_replay = replay_contribution_records(
                left, volume_size, order=order, precision=precision
            )
            right_replay = replay_contribution_records(
                right, volume_size, order=order, precision=precision
            )
            cross[f"{order}_{precision}"] = _replay_metrics(left_replay, right_replay)

    recomputed_equal = None
    recomputed_cross = None
    recomputed_field_metrics = None
    provenance_compatible = False
    if (
        left.has_verified_recomputed_high_precision
        and right.has_verified_recomputed_high_precision
    ):
        left_provenance = left.recomputation_provenance
        right_provenance = right.recomputation_provenance
        provenance_compatible = all(
            getattr(left_provenance, key) == getattr(right_provenance, key)
            for key in (
                "semantic_identity_sha256",
                "formula_name",
                "formula_version",
                "numeric_policy",
                "source_dtype",
            )
        )
        recomputed_fields = (
            "recomputed_coefficients",
            "recomputed_source_data",
            "recomputed_source_weight",
        )
        recomputed_field_metrics = {
            key: exact_array_metrics(
                getattr(left, key)[left_order], getattr(right, key)[right_order]
            )
            for key in recomputed_fields
        }
        recomputed_equal = (
            same_identity
            and provenance_compatible
            and all(
                metrics.get("array_equal", False)
                for metrics in recomputed_field_metrics.values()
            )
        )
        recomputed_cross = _replay_metrics(
            replay_contribution_records(
                left,
                volume_size,
                order="canonical",
                precision="float64",
                operand_provenance=RECOMPUTED_HIGH_PRECISION,
            ),
            replay_contribution_records(
                right,
                volume_size,
                order="canonical",
                precision="float64",
                operand_provenance=RECOMPUTED_HIGH_PRECISION,
            ),
        )

    unverified_arrays_present = (
        left.has_recomputed_high_precision
        or right.has_recomputed_high_precision
    ) and not (
        left.has_verified_recomputed_high_precision
        and right.has_verified_recomputed_high_precision
    )
    if not same_identity or not same_geometry:
        classification = "discrete_geometry_difference"
    elif not same_operands:
        if recomputed_equal:
            classification = "precision_consistent_with_verified_recomputation"
        elif unverified_arrays_present or recomputed_equal is False:
            classification = "unresolved"
        else:
            classification = "operand_generation_difference"
    elif not same_logical_schedule:
        classification = "logical_schedule_difference"
    elif all(
        metrics[field]["array_equal"]
        for metrics in cross.values()
        for field in ("data", "weight")
    ):
        classification = "exact"
    else:
        classification = "unresolved"
    return {
        "classification": classification,
        "identity_equal": same_identity,
        "discrete_geometry_equal": same_geometry,
        "captured_operands_equal": same_operands,
        "logical_schedule_keys_equal": same_logical_schedule,
        "field_metrics": field_metrics,
        "captured_operand_provenance": CAPTURED_F32_CAST,
        "caller_supplied_high_precision_arrays_unverified": unverified_arrays_present,
        "verified_recomputation_provenance_compatible": provenance_compatible,
        "left_verified_recomputation_provenance": _provenance_report(
            left.recomputation_provenance
        ),
        "right_verified_recomputation_provenance": _provenance_report(
            right.recomputation_provenance
        ),
        "recomputed_high_precision_equal": recomputed_equal,
        "recomputed_high_precision_field_metrics": recomputed_field_metrics,
        "cross_engine": cross,
        "recomputed_high_precision_cross_engine": recomputed_cross,
    }


def _validate_signature(path: Path):
    signature = _load(path)
    _require_header(signature, magic=SIGNATURE_MAGIC, schema=SIGNATURE_SCHEMA)
    if str(_scalar(signature, "signature_inertness_gate")) != (
        "bitwise-post-accum-shadow-and-operand-exact"
    ):
        raise ValueError("signature deterministic inertness gate is missing or unknown")
    if not bool(_scalar(signature, "signature_inertness_gate_passed")):
        raise ValueError("signature deterministic inertness gate did not pass")
    if not bool(_scalar(signature, "signature_accumulator_shadow_bitwise_equal")):
        raise ValueError("signature accumulator shadow is not bitwise equal")
    if not bool(_scalar(signature, "signature_prepared_operands_bitwise_equal")):
        raise ValueError("signature prepared operands are not bitwise equal")
    if bool(_scalar(signature, "signature_kernel_accumulate")):
        raise ValueError("signature-only kernel must have ACCUMULATE=false")
    contribution_path = Path(str(_scalar(signature, "companion_contribution_path")))
    if not contribution_path.is_file():
        raise ValueError(f"companion contribution does not exist: {contribution_path}")
    contribution = _load(contribution_path)
    _validate_companion_identity(signature, contribution, contribution_path)
    replay_bundle = _validate_v3_replay_bundle(contribution)
    if not replay_bundle["shadow_only_mode"]:
        raise ValueError("device signature companion must be a checked shadow-only contribution")

    q = int(np.asarray(signature["contributor_canonical_rotation_keys"]).size)
    image_shape = tuple(int(value) for value in np.asarray(signature["image_shape"]))
    volume_shape = tuple(int(value) for value in np.asarray(signature["volume_shape"]))
    if len(image_shape) != 2 or len(volume_shape) != 3 or len(set(volume_shape)) != 1:
        raise ValueError("invalid image/volume shape metadata")
    max_r = float(_scalar(signature, "max_r"))
    dense_height = 2 * int(round(max_r))
    dense_pixels = dense_height * (dense_height // 2 + 1)
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    if not 0 < dense_height <= image_shape[0]:
        raise ValueError("invalid dense current-size topology")
    if not np.array_equal(
        np.asarray(signature["program_axis_sizes"], dtype=np.int64),
        np.asarray([q, dense_pixels, 8], dtype=np.int64),
    ):
        raise ValueError("program_axis_sizes does not match signature tensors")

    shape2 = (q, dense_pixels)
    shape3 = (q, dense_pixels, 8)
    expected_shapes = {
        "canonical_rotation_keys": shape2,
        "canonical_pixel_indices": shape2,
        "row_flags": shape2,
        "source_values": (q, dense_pixels, 6),
        "neighbor_indices": shape3,
        "neighbor_coefficients": shape3,
        "neighbor_flags": shape3,
        "launch_ordinal": (q,),
        "particle_local_row": (q,),
        "program_row": (q,),
        "image_identity": (q,),
        "original_indices": (q,),
        "program_lane": (dense_pixels,),
        "program_serial_pass": (dense_pixels,),
        "program_neighbor": (8,),
    }
    for key, expected in expected_shapes.items():
        if np.asarray(signature[key]).shape != expected:
            raise ValueError(f"{key} shape {np.asarray(signature[key]).shape} != {expected}")
    expected_dtypes = {
        "canonical_rotation_keys": np.dtype(np.int32),
        "canonical_pixel_indices": np.dtype(np.int32),
        "row_flags": np.dtype(np.int32),
        "source_values": np.dtype(np.float32),
        "neighbor_indices": np.dtype(np.int32),
        "neighbor_coefficients": np.dtype(np.float32),
        "neighbor_flags": np.dtype(np.int32),
        "launch_ordinal": np.dtype(np.int64),
        "particle_local_row": np.dtype(np.int32),
        "program_row": np.dtype(np.int32),
        "original_indices": np.dtype(np.int64),
        "program_lane": np.dtype(np.int32),
        "program_serial_pass": np.dtype(np.int32),
        "program_neighbor": np.dtype(np.int32),
        "contributor_canonical_rotation_keys": np.dtype(np.int32),
    }
    for key, expected in expected_dtypes.items():
        if np.asarray(signature[key]).dtype != expected:
            raise ValueError(f"{key} dtype {np.asarray(signature[key]).dtype} != {expected}")
    if not np.array_equal(signature["program_row"], signature["particle_local_row"]):
        raise ValueError("program_row must equal the particle-local source row")
    if not np.array_equal(signature["program_lane"], np.arange(dense_pixels) % 128):
        raise ValueError("program_lane does not encode dense_pixel % 128")
    if not np.array_equal(signature["program_serial_pass"], np.arange(dense_pixels) // 128):
        raise ValueError("program_serial_pass does not encode dense_pixel // 128")
    if not np.array_equal(signature["program_neighbor"], np.arange(8)):
        raise ValueError("program_neighbor order is not d0*4+d1*2+d2")

    canonical_keys = np.asarray(signature["contributor_canonical_rotation_keys"], dtype=np.int32)
    if not np.array_equal(
        signature["canonical_rotation_keys"],
        np.broadcast_to(canonical_keys[:, None], shape2),
    ):
        raise ValueError("canonical rotation keys do not broadcast from contributor keys")
    expected_pixels = np.arange(dense_pixels, dtype=np.int32)
    if not np.array_equal(
        signature["canonical_pixel_indices"],
        np.broadcast_to(expected_pixels[None, :], shape2),
    ):
        raise ValueError("canonical dense pixels are not in native increasing order")
    if "row-major [contributor_row,dense_pixel,neighbor]" not in str(
        _scalar(signature, "signature_tensor_axis_legend")
    ):
        raise ValueError("signature tensor axis legend is missing or inconsistent")
    if "atomicAdd(data_real)" not in str(
        _scalar(signature, "atomic_component_program_order_legend")
    ):
        raise ValueError("atomic component program-order legend is missing or inconsistent")

    particle_fields = (
        "particle_launch_ordinals",
        "particle_total_row_counts",
        "particle_contributor_row_counts",
        "particle_noncontributor_row_counts",
        "particle_noncontributor_exact_zero",
        "particle_noncontributor_zero_sha256",
        "particle_image_identities",
        "particle_original_indices",
    )
    particle_count = np.asarray(signature[particle_fields[0]]).size
    if particle_count <= 0 or any(np.asarray(signature[key]).shape != (particle_count,) for key in particle_fields):
        raise ValueError("particle manifest fields have inconsistent shapes")
    launches = np.asarray(signature["particle_launch_ordinals"], dtype=np.int64)
    if launches.size > 1 and np.any(np.diff(launches) != 1):
        raise ValueError("particle launch ordinals must be strictly consecutive")
    total_counts = np.asarray(signature["particle_total_row_counts"], dtype=np.int64)
    contributor_counts = np.asarray(signature["particle_contributor_row_counts"], dtype=np.int64)
    omitted_counts = np.asarray(signature["particle_noncontributor_row_counts"], dtype=np.int64)
    if np.any(total_counts != contributor_counts + omitted_counts) or int(contributor_counts.sum()) != q:
        raise ValueError("particle total/contributor/noncontributor counts do not close")
    if not np.all(signature["particle_noncontributor_exact_zero"]):
        raise ValueError("particle manifest does not certify omitted rows as exact zero")
    if not np.array_equal(total_counts, np.asarray(contribution["actual_counts"], dtype=np.int64)):
        raise ValueError("particle total row counts differ from contribution actual_counts")
    if not np.array_equal(signature["particle_image_identities"], contribution["image_identities"]):
        raise ValueError("particle image identities differ from companion contribution")
    if not np.array_equal(signature["particle_original_indices"], contribution["original_indices"]):
        raise ValueError("particle original indices differ from companion contribution")

    contributor_launches = np.asarray(signature["launch_ordinal"], dtype=np.int64)
    contributor_rows = np.asarray(signature["particle_local_row"], dtype=np.int32)
    if not np.array_equal(contributor_launches, np.repeat(launches, contributor_counts)):
        raise ValueError("contributor rows are not contiguous in particle launch order")
    rotation_grid = np.asarray(contribution["oversampled_rotation_indices"], dtype=np.int64)
    active_particle = np.asarray(contribution["active_particle_rows"], dtype=np.int32)
    active_rows = np.asarray(contribution["active_rotation_rows"], dtype=np.int32)
    active_summed = np.asarray(contribution["active_summed"])
    active_weights = np.asarray(contribution["active_ctf_probs"])
    expected_contributor_keys = []
    expected_dense_summed = []
    expected_dense_weights = []
    cursor = 0
    for particle in range(particle_count):
        launch = launches[particle]
        mask = contributor_launches == launch
        rows = contributor_rows[mask]
        if rows.size != contributor_counts[particle]:
            raise ValueError("per-particle contributor count does not match signature rows")
        if rows.size > 1 and np.any(np.diff(rows) <= 0):
            raise ValueError("particle-local contributor rows are not strictly increasing")
        if np.any(rows < 0) or np.any(rows >= total_counts[particle]):
            raise ValueError("particle-local contributor row lies outside actual_count")
        if not np.all(signature["image_identity"][mask] == signature["particle_image_identities"][particle]):
            raise ValueError("contributor image identity differs from particle manifest")
        if not np.all(signature["original_indices"][mask] == signature["particle_original_indices"][particle]):
            raise ValueError("contributor original index differs from particle manifest")
        expected_contributor_keys.extend(rotation_grid[particle, rows].tolist())

        all_rows = np.arange(total_counts[particle], dtype=np.int32)
        omitted = np.setdiff1d(all_rows, rows, assume_unique=True).astype(np.int32, copy=False)
        active_mask = active_particle == particle
        if not np.array_equal(active_rows[active_mask], all_rows):
            raise ValueError("companion contribution does not retain every valid row in order")
        particle_summed = active_summed[active_mask]
        particle_weights = active_weights[active_mask]
        particle_summed = np.asarray(particle_summed, dtype=np.complex64)
        particle_weights = np.asarray(particle_weights, dtype=np.float32)
        expected_rows = np.flatnonzero(np.any(particle_weights > 0, axis=1)).astype(
            np.int32, copy=False
        )
        if not np.array_equal(rows, expected_rows):
            raise ValueError(
                "signature contributor rows do not equal exact positive-weight companion rows"
            )
        omitted_summed = np.ascontiguousarray(particle_summed[omitted], dtype=np.complex64)
        omitted_weights = np.ascontiguousarray(particle_weights[omitted], dtype=np.float32)
        if np.any(omitted_summed != 0) or np.any(omitted_weights != 0):
            raise ValueError("omitted signature row is not exactly zero in the companion")
        digest = _noncontributor_digest(
            omitted,
            rotation_grid[particle, omitted].astype(np.int32),
            omitted_summed,
            omitted_weights,
        )
        if digest != str(signature["particle_noncontributor_zero_sha256"][particle]):
            raise ValueError("omitted-row exact-zero digest mismatch")
        compact_indices = np.asarray(contribution["window_indices"], dtype=np.int32)
        full_half_width = image_shape[1] // 2 + 1
        full_rows = compact_indices // full_half_width
        columns = compact_indices % full_half_width
        signed_rows = np.where(
            full_rows <= image_shape[0] // 2,
            full_rows,
            full_rows - image_shape[0],
        )
        current_indices = np.mod(signed_rows, dense_height) * (dense_height // 2 + 1) + columns
        if np.any(columns >= dense_height // 2 + 1):
            raise ValueError("companion compact pixel lies outside current half-width")
        if np.unique(current_indices).size != current_indices.size:
            raise ValueError("companion compact window does not map uniquely into the dense square")
        dense_summed = np.zeros((rows.size, dense_pixels), dtype=np.complex64)
        dense_weights = np.zeros((rows.size, dense_pixels), dtype=np.float32)
        dense_summed[:, current_indices] = particle_summed[rows]
        dense_weights[:, current_indices] = particle_weights[rows]
        expected_dense_summed.append(dense_summed)
        expected_dense_weights.append(dense_weights)
        cursor += rows.size
    if not np.array_equal(canonical_keys, np.asarray(expected_contributor_keys, dtype=np.int32)):
        raise ValueError("signature contributor keys do not close against the companion grid")
    expected_dense_summed = np.concatenate(expected_dense_summed, axis=0)
    expected_dense_weights = np.concatenate(expected_dense_weights, axis=0)

    row_flags = np.asarray(signature["row_flags"], dtype=np.int32)
    neighbor_flags = np.asarray(signature["neighbor_flags"], dtype=np.int32)
    if np.any(row_flags & ~ROW_FLAG_MASK):
        raise ValueError("unknown row flag bit")
    if np.any(neighbor_flags & ~NEIGHBOR_FLAG_MASK):
        raise ValueError("unknown neighbor flag bit")
    primary_bits = np.stack([(row_flags & bit) != 0 for bit in (1, 2, 4, 8, 32, 64)], axis=-1)
    if np.any(np.sum(primary_bits, axis=-1) != 1):
        raise ValueError("each row-pixel must have exactly one primary gate/scatter state")
    orientation_fold = (row_flags & 16) != 0
    if np.any(orientation_fold & ((row_flags & (32 | 64)) == 0)):
        raise ValueError("orientation-fold flag may appear only with compact-OOB or reached-scatter state")
    reached = (row_flags & 64) != 0
    valid_neighbors = (neighbor_flags & 1) != 0
    if np.any(~np.isin(neighbor_flags, np.asarray([1, 3, 5, 8], dtype=np.int32))):
        raise ValueError("neighbor flag value is outside {1,3,5,8}")
    if np.any(valid_neighbors & ~reached[..., None]):
        raise ValueError("valid neighbor belongs to a row that did not reach scatter")
    if np.any(valid_neighbors & ((neighbor_flags & 8) != 0)):
        raise ValueError("valid neighbor also carries out-of-bounds flag")
    invalid_neighbors = ~valid_neighbors
    if np.any(np.asarray(signature["neighbor_indices"])[invalid_neighbors] != -1):
        raise ValueError("invalid neighbor index is not sentinel -1")
    if np.any(np.asarray(signature["neighbor_coefficients"])[invalid_neighbors] != 0):
        raise ValueError("invalid neighbor coefficient is not zero")
    source = np.asarray(signature["source_values"])
    # NaNs are intentional before source load for redundant-x0 and 2-D-radius
    # gates, and after source load for coordinates behind the Fweight gate.
    if np.any(~np.isfinite(source[reached])):
        raise ValueError("reached-scatter source values must be finite")
    flag1_or_2 = ((row_flags & 1) != 0) | ((row_flags & 2) != 0)
    flag4 = (row_flags & 4) != 0
    loaded = ((row_flags & 8) != 0) | ((row_flags & 32) != 0) | reached
    if np.any(~np.isnan(source[flag1_or_2])):
        raise ValueError("redundant-x0/2-D-radius rows must retain all-NaN source sentinels")
    flag4_source = source[flag4]
    if np.any(~np.isfinite(flag4_source[:, :3])) or np.any(~np.isnan(flag4_source[:, 3:])):
        raise ValueError("nonpositive-weight rows must have finite values/weight and NaN coordinates")
    if np.any(~np.isfinite(source[loaded])):
        raise ValueError("3-D/OOB/reached rows must have fully finite source tuples")
    source_loaded = ~flag1_or_2
    if not np.array_equal(
        source[..., 0][source_loaded], expected_dense_summed.real[source_loaded]
    ) or not np.array_equal(
        source[..., 1][source_loaded], expected_dense_summed.imag[source_loaded]
    ) or not np.array_equal(
        source[..., 2][source_loaded], expected_dense_weights[source_loaded]
    ):
        raise ValueError("device source values do not close against companion dense expansion")

    targets, atomic_values, program = reconstruct_atomic_tuples(signature, volume_size)
    contribution_records = extract_contribution_records(signature)
    if not np.array_equal(contribution_records.target_indices, targets):
        raise ValueError("canonical contribution targets do not close against atomic tuples")
    atomic_launches = contributor_launches[program[:, 0]]
    atomic_local_rows = contributor_rows[program[:, 0]]
    atomic_rotation_keys = canonical_keys[program[:, 0]]
    tuple_digest = hashlib.sha256()
    tuple_digest.update(atomic_launches.tobytes(order="C"))
    tuple_digest.update(atomic_local_rows.tobytes(order="C"))
    tuple_digest.update(atomic_rotation_keys.tobytes(order="C"))
    tuple_digest.update(targets.tobytes(order="C"))
    tuple_digest.update(atomic_values.tobytes(order="C"))
    tuple_digest.update(program.tobytes(order="C"))
    return {
        "path": str(path.resolve()),
        "signature": signature,
        "contribution": contribution,
        "contribution_replay_bundle": replay_bundle,
        "launch_min": int(launches.min()),
        "launch_max": int(launches.max()),
        "particle_count": int(particle_count),
        "contributor_rows": q,
        "omitted_rows": int(omitted_counts.sum()),
        "dense_pixels": dense_pixels,
        "volume_shape": volume_shape,
        "source_stack_sha256": str(_scalar(signature, "source_stack_sha256")),
        "atomic_targets": targets,
        "atomic_values": atomic_values,
        "atomic_program": program,
        "atomic_launches": atomic_launches,
        "atomic_local_rows": atomic_local_rows,
        "atomic_rotation_keys": atomic_rotation_keys,
        "atomic_tuple_sha256": tuple_digest.hexdigest(),
        "contribution_records": contribution_records,
    }


def _replay(results, panel):
    volume_shape = results[0]["volume_shape"]
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    real = np.zeros(volume_size, dtype=np.float32)
    imag = np.zeros(volume_size, dtype=np.float32)
    weight = np.zeros(volume_size, dtype=np.float32)
    for result in results:
        targets = result["atomic_targets"]
        values = result["atomic_values"]
        np.add.at(real, targets, values[:, 0])
        np.add.at(imag, targets, values[:, 1])
        np.add.at(weight, targets, values[:, 2])
    data = (real + 1j * imag).astype(np.complex64)
    native_data = np.asarray(panel["data_accumulator"], dtype=np.complex64)
    native_weight = np.asarray(panel["weight_accumulator"], dtype=np.float32)
    if data.shape != native_data.shape or weight.shape != native_weight.shape:
        raise ValueError("replay/native accumulator shape mismatch")
    data_rel_l1 = float(
        np.sum(np.abs(data - native_data))
        / max(np.sum(np.abs(native_data)), np.finfo(np.float64).tiny)
    )
    weight_rel_l1 = float(
        np.sum(np.abs(weight - native_weight))
        / max(np.sum(np.abs(native_weight)), np.finfo(np.float64).tiny)
    )
    data_fsc_auc, data_min_fsc, data_fsc = _accumulator_fsc(data, native_data, volume_shape)
    weight_fsc_auc, weight_min_fsc, weight_fsc = _accumulator_fsc(
        weight, native_weight, volume_shape
    )
    return (
        data,
        weight,
        data_rel_l1,
        weight_rel_l1,
        data_fsc_auc,
        data_min_fsc,
        weight_fsc_auc,
        weight_min_fsc,
        data_fsc,
        weight_fsc,
    )


def _normalized_fsc_auc(fsc):
    """Integrate finite non-DC FSC shells on a normalized shell axis."""

    values = np.asarray(fsc, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if finite.size:
        finite[0] = False
    if np.count_nonzero(finite) < 2:
        return float("nan")
    shell_axis = np.flatnonzero(finite).astype(np.float64)
    shell_axis = (shell_axis - shell_axis[0]) / (shell_axis[-1] - shell_axis[0])
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(values[finite], shell_axis))


def _accumulator_fsc_curve(a_flat, b_flat, volume_shape):
    n0, n1, n2 = volume_shape
    shape = (n0, n1, n2 // 2 + 1)
    # This is a reporting-only metric.  Promote before shell reductions so
    # complex64/float32 accumulation roundoff cannot produce an impossible
    # FSC above one (or make a bitwise-identical replay appear imperfect).
    metric_dtype = np.complex128 if (
        np.iscomplexobj(a_flat) or np.iscomplexobj(b_flat)
    ) else np.float64
    a = np.asarray(a_flat, dtype=metric_dtype).reshape(shape)
    b = np.asarray(b_flat, dtype=metric_dtype).reshape(shape)
    k0 = np.arange(n0) - n0 // 2
    k1 = np.arange(n1) - n1 // 2
    k2 = np.arange(n2 // 2 + 1)
    # Packed x-half storage represents both +/-kz planes except at the
    # self-conjugate kz=0 plane and, for even N, the Nyquist plane.  Weight
    # shell reductions by that Hermitian multiplicity so this reports the FSC
    # of the represented full Fourier volume rather than the packed array.
    kz_multiplicity = np.full(k2.size, 2.0, dtype=np.float64)
    kz_multiplicity[0] = 1.0
    if n2 % 2 == 0:
        kz_multiplicity[-1] = 1.0
    shells = np.rint(
        np.sqrt(k0[:, None, None] ** 2 + k1[None, :, None] ** 2 + k2[None, None, :] ** 2)
    ).astype(np.int32)
    fsc = np.full(n2 // 2 + 1, np.nan, dtype=np.float64)
    for shell in range(1, n2 // 2 + 1):
        mask = shells == shell
        aa = a[mask]
        bb = b[mask]
        multiplicity = np.broadcast_to(kz_multiplicity, shape)[mask]
        a_power = float(np.real(np.sum(multiplicity * np.conj(aa) * aa)))
        b_power = float(np.real(np.sum(multiplicity * np.conj(bb) * bb)))
        denom = np.sqrt(a_power * b_power)
        if denom > 0:
            numerator = np.sum(multiplicity * np.conj(bb) * aa)
            value = float(np.real(numerator) / denom)
            if np.isfinite(value):
                value = float(np.clip(value, -1.0, 1.0))
            fsc[shell] = value
    if np.count_nonzero(np.isfinite(fsc[1:])) == 0:
        raise ValueError("native/replay accumulators have no nonzero FSC shells")
    return fsc


def _accumulator_fsc(a_flat, b_flat, volume_shape):
    fsc = _accumulator_fsc_curve(a_flat, b_flat, volume_shape)
    finite_non_dc = np.isfinite(fsc)
    finite_non_dc[0] = False
    values = fsc[finite_non_dc]
    return _normalized_fsc_auc(fsc), float(np.min(values)), fsc


def _validate_shard_set(results, panel, *, label: str, scalar_fields, array_fields):
    """Validate one engine's complete native-panel launch coverage."""

    if not results:
        raise ValueError(f"{label} signature shard set is empty")
    reference = results[0]["signature"]
    for result in results:
        for key in scalar_fields:
            if _scalar(result["signature"], key) != _scalar(reference, key):
                raise ValueError(f"{label} signature identity/topology mismatch for {key}")
        for key in array_fields:
            if not np.array_equal(result["signature"][key], reference[key]):
                raise ValueError(f"{label} signature shape topology mismatch for {key}")
        expected_count = result["launch_max"] - result["launch_min"] + 1
        if result["particle_count"] != expected_count:
            raise ValueError(f"{label} shard launch range/count mismatch")
    for key in scalar_fields:
        if _scalar(panel, key) != _scalar(reference, key):
            raise ValueError(f"{label} panel/signature identity/topology mismatch for {key}")
    for key in array_fields:
        if not np.array_equal(panel[key], reference[key]):
            raise ValueError(f"{label} panel/signature shape topology mismatch for {key}")
    for previous, current in zip(results, results[1:]):
        if current["launch_min"] <= previous["launch_max"]:
            raise ValueError(f"{label} signature launch ranges overlap")
        if current["launch_min"] != previous["launch_max"] + 1:
            raise ValueError(f"{label} signature files do not form a consecutive launch sequence")
    if results[0]["launch_min"] != 0:
        raise ValueError(f"{label} validated panel launch sequence must begin at zero")
    launch_count = int(_scalar(panel, "launch_count"))
    total_particles = sum(result["particle_count"] for result in results)
    if launch_count != total_particles:
        raise ValueError(
            f"{label} panel launch_count does not equal the validated particle manifest"
        )
    if results[-1]["launch_max"] != launch_count - 1:
        raise ValueError(
            f"{label} validated launch sequence does not end at panel.launch_count-1"
        )
    return total_particles


def _validate_native_panel_arrays(panel, volume_size: int, *, label: str):
    data = np.asarray(panel["data_accumulator"])
    weight = np.asarray(panel["weight_accumulator"])
    if data.dtype != np.dtype(np.complex64):
        raise ValueError(f"{label} panel data accumulator must be complex64")
    if weight.dtype != np.dtype(np.float32):
        raise ValueError(f"{label} panel weight accumulator must be float32")
    if data.shape != (volume_size,) or weight.shape != (volume_size,):
        raise ValueError(f"{label} panel accumulator shape does not match volume_shape")
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(weight)):
        raise ValueError(f"{label} panel accumulator contains nonfinite values")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signatures", nargs="+", type=Path)
    parser.add_argument("--panel-native", required=True, type=Path)
    parser.add_argument("--atomic-tuples-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--max-replay-rel-l1", type=float, default=1e-4)
    parser.add_argument(
        "--compare-signatures",
        nargs="+",
        type=Path,
        help="optional second engine's schema-v1 signature shards",
    )
    parser.add_argument(
        "--compare-panel-native",
        type=Path,
        help="required native panel closing the second engine's complete shard set",
    )
    parser.add_argument(
        "--allow-nonexact-cross-diagnostic",
        action="store_true",
        help=(
            "exit zero for a nonexact cross-engine result while labeling the overall "
            "result DIAGNOSTIC_NONEXACT"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if bool(args.compare_signatures) != bool(args.compare_panel_native):
        raise ValueError(
            "--compare-signatures and --compare-panel-native must be supplied together"
        )
    if args.allow_nonexact_cross_diagnostic and not args.compare_signatures:
        raise ValueError(
            "--allow-nonexact-cross-diagnostic requires a cross-engine comparison"
        )
    results = sorted(
        (_validate_signature(path) for path in args.signatures),
        key=lambda result: (result["launch_min"], result["path"]),
    )
    panel = _load(args.panel_native)
    _require_header(panel, magic=PANEL_MAGIC, schema=PANEL_SCHEMA)
    scalar_identity_fields = (
        "run_id",
        "iteration",
        "half",
        "rank",
        "current_size",
        "max_r",
        "reconstruction_padding_factor",
        "source_stack_sha256",
        "causal_arm",
        "winner_take_all",
        "topology_claim",
    )
    array_identity_fields = ("image_shape", "volume_shape")
    reference = results[0]["signature"]
    total_particles = _validate_shard_set(
        results,
        panel,
        label="primary",
        scalar_fields=scalar_identity_fields,
        array_fields=array_identity_fields,
    )
    if any(result["volume_shape"] != results[0]["volume_shape"] for result in results):
        raise ValueError("signature volume shapes differ")

    volume_shape = results[0]["volume_shape"]
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    contribution_records = concatenate_contribution_records(
        [result["contribution_records"] for result in results]
    )
    canonical_diagnostics = canonical_replay_diagnostics(
        contribution_records, volume_size
    )
    cross_engine_diagnostics = None
    compare_native_replay = None
    if args.compare_signatures:
        compare_results = sorted(
            (_validate_signature(path) for path in args.compare_signatures),
            key=lambda result: (result["launch_min"], result["path"]),
        )
        signature_boundary_fields = (
            "iteration",
            "half",
            "rank",
            "pass_index",
            "class_index",
            "call_index",
            "dump_index",
            "current_size",
            "max_r",
            "reconstruction_padding_factor",
            "source_stack_sha256",
            "causal_arm",
            "winner_take_all",
            "topology_claim",
        )
        for result in compare_results:
            for key in signature_boundary_fields:
                if _scalar(result["signature"], key) != _scalar(reference, key):
                    raise ValueError(
                        f"cross-engine signature boundary mismatch for {key}"
                    )
            for key in array_identity_fields:
                if not np.array_equal(result["signature"][key], reference[key]):
                    raise ValueError(
                        f"cross-engine signature boundary mismatch for {key}"
                    )
        companion_boundary_fields = (
            "window_indices",
            "local_indices",
            "image_identities",
            "original_indices",
            "star_rows",
            "stack_indices_1based",
            "resolved_stack_paths",
        )
        primary_companion = results[0]["contribution"]
        for result in results[1:]:
            for key in companion_boundary_fields:
                if not np.array_equal(result["contribution"][key], primary_companion[key]):
                    raise ValueError(
                        f"primary companion boundary mismatch for {key}"
                    )
        compare_companion = compare_results[0]["contribution"]
        for result in compare_results[1:]:
            for key in companion_boundary_fields:
                if not np.array_equal(result["contribution"][key], compare_companion[key]):
                    raise ValueError(
                        f"compare companion boundary mismatch for {key}"
                    )
        for key in companion_boundary_fields:
            if not np.array_equal(primary_companion[key], compare_companion[key]):
                raise ValueError(f"cross-engine companion boundary mismatch for {key}")
        compare_panel = _load(args.compare_panel_native)
        _require_header(compare_panel, magic=PANEL_MAGIC, schema=PANEL_SCHEMA)
        compare_total_particles = _validate_shard_set(
            compare_results,
            compare_panel,
            label="compare",
            scalar_fields=scalar_identity_fields,
            array_fields=array_identity_fields,
        )
        if compare_total_particles != total_particles:
            raise ValueError("cross-engine particle manifest counts differ")
        _validate_native_panel_arrays(compare_panel, volume_size, label="compare")
        compare_records = concatenate_contribution_records(
            [result["contribution_records"] for result in compare_results]
        )
        cross_engine_diagnostics = compare_contribution_engines(
            contribution_records, compare_records, volume_size
        )
        compare_replay_values = _replay(compare_results, compare_panel)
        compare_native_replay = {
            "panel_native": str(args.compare_panel_native.resolve()),
            "host_replay_data_rel_l1": compare_replay_values[2],
            "host_replay_weight_rel_l1": compare_replay_values[3],
        }
        if (
            compare_replay_values[2] > args.max_replay_rel_l1
            or compare_replay_values[3] > args.max_replay_rel_l1
        ):
            raise ValueError(
                "compare host replay exceeds relative-L1 bound: "
                f"data={compare_replay_values[2]:.6g}, "
                f"weight={compare_replay_values[3]:.6g}, "
                f"bound={args.max_replay_rel_l1:.6g}"
            )
    _validate_native_panel_arrays(panel, volume_size, label="primary")

    (
        replay_data,
        replay_weight,
        data_rel_l1,
        weight_rel_l1,
        data_fsc_auc,
        data_min_fsc,
        weight_fsc_auc,
        weight_min_fsc,
        data_fsc,
        weight_fsc,
    ) = _replay(results, panel)
    if data_rel_l1 > args.max_replay_rel_l1 or weight_rel_l1 > args.max_replay_rel_l1:
        raise ValueError(
            f"host replay exceeds relative-L1 bound: data={data_rel_l1:.6g}, "
            f"weight={weight_rel_l1:.6g}, bound={args.max_replay_rel_l1:.6g}"
        )
    if args.atomic_tuples_out:
        args.atomic_tuples_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.atomic_tuples_out,
            target_indices=np.concatenate([result["atomic_targets"] for result in results]),
            launch_ordinal=np.concatenate([result["atomic_launches"] for result in results]),
            particle_local_row=np.concatenate([result["atomic_local_rows"] for result in results]),
            canonical_rotation_key=np.concatenate(
                [result["atomic_rotation_keys"] for result in results]
            ),
            atomic_values_data_real_data_imag_weight=np.concatenate(
                [result["atomic_values"] for result in results]
            ),
            program_contributor_row_dense_pixel_neighbor=np.concatenate(
                [result["atomic_program"] for result in results]
            ),
            replay_data_accumulator=replay_data,
            replay_weight_accumulator=replay_weight,
        )
    cross_classification = (
        None if cross_engine_diagnostics is None else cross_engine_diagnostics["classification"]
    )
    cross_exact = cross_classification in (None, "exact")
    if cross_classification is None:
        overall_status = "PASS"
        cross_status = "NOT_REQUESTED"
    elif cross_exact:
        overall_status = "PASS"
        cross_status = "PASS"
    elif args.allow_nonexact_cross_diagnostic:
        overall_status = "DIAGNOSTIC_NONEXACT"
        cross_status = "DIAGNOSTIC_NONEXACT"
    else:
        overall_status = "FAIL"
        cross_status = "FAIL"
    summary = {
        "status": overall_status,
        "artifact_validation_status": "PASS",
        "cross_comparison_status": cross_status,
        "signature_files": [result["path"] for result in results],
        "panel_native": str(args.panel_native.resolve()),
        "particle_count": total_particles,
        "contributor_rows": sum(result["contributor_rows"] for result in results),
        "omitted_exact_zero_rows": sum(result["omitted_rows"] for result in results),
        "atomic_tuple_count": sum(result["atomic_targets"].size for result in results),
        "atomic_tuple_sha256_by_file": [result["atomic_tuple_sha256"] for result in results],
        "contribution_replay_bundles": [
            result["contribution_replay_bundle"] for result in results
        ],
        "host_replay_data_rel_l1": data_rel_l1,
        "host_replay_weight_rel_l1": weight_rel_l1,
        "host_replay_data_accumulator_fsc_auc": data_fsc_auc,
        "host_replay_data_accumulator_min_non_dc_shell_fsc": data_min_fsc,
        "host_replay_data_accumulator_fsc": [
            float(value) if np.isfinite(value) else None for value in data_fsc
        ],
        "host_replay_weight_accumulator_fsc_auc": weight_fsc_auc,
        "host_replay_weight_accumulator_min_non_dc_shell_fsc": weight_min_fsc,
        "host_replay_weight_accumulator_fsc": [
            float(value) if np.isfinite(value) else None for value in weight_fsc
        ],
        "replay_bound": args.max_replay_rel_l1,
        "canonical_contribution_replay": canonical_diagnostics,
        "cross_engine_contribution_replay": cross_engine_diagnostics,
        "compare_native_replay": compare_native_replay,
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text)
    print(text, end="")
    if not cross_exact and not args.allow_nonexact_cross_diagnostic:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
