"""Deterministic CPU replay of RELION's coarse Gaussian reduction.

This diagnostic is not wired into production scoring and has no JAX
dependency.  It requires one immutable production capture containing all
1,624 packed contributions, the captured float32 ``highres_Xi2_img/2`` term,
and the observed float32 raw diff2.  RELION's four nonzero lane accumulators
and all 24 possible atomic arrival orders are replayed exactly.  Matching one
or more orders proves only numerical compatibility; the atomic order itself is
never claimed to have been observed.

Promoted float64 replays cast the captured float32 operands.  Genuine float64
replays instead form contributions inside this module from complete
complex128 reference/shifted-image operands and float64 real weights.  Because
the high-resolution initial term is not recomputed from genuine float64 source
terms, genuine results are explicitly centered/contribution-only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import permutations
from typing import Literal

import numpy as np

CAPTURE_SCHEMA = "recovar.em.gaussian_score_operand_capture"
CAPTURE_SCHEMA_VERSION = 1
REPLAY_SCHEMA = "recovar-relion-coarse-gaussian-reduction-replay-v1"
REPLAY_SCHEMA_VERSION = 1
RELION_COARSE_PACKED_PIXELS = 1624
RELION_COARSE_CHUNK_PIXELS = 32
RELION_COARSE_NONZERO_LANES = 4
RELION_COARSE_ATOMIC_LANES = 5
RELION_V1_ENGINE = "RELION"
RELION_V1_SOURCE_COMMIT = "f2c1a384400aec37dc6805856a5ba645650a44f1"
RELION_V1_KERNEL_ID = "cuda_kernel_diff2_coarse:block128:prefetch4:translations29"
RELION_V1_CURRENT_SIZE = 56
RELION_V1_BLOCK_SIZE = 128
RELION_V1_PREFETCH = 4
RELION_V1_TRANSLATION_COUNT = 29
RELION_V1_ORIGINAL_PARTICLE_ID = 7881
RELION_V1_ITERATION = 2
RELION_V1_PASS_INDEX = 0
RELION_V1_CLASS_INDEX = 0
RELION_V1_ROLE_TRANSLATION_INDEX = {
    "global_winner": 8,
    "selected_cutoff_neighbor_rank47": 10,
    "relion_only_boundary_parent_rank48": 2,
    "first_excluded_control_rank49": 0,
}
RELION_V1_ROLE_ROTATION_INDEX = {
    "global_winner": 8246,
    "selected_cutoff_neighbor_rank47": 4504,
    "relion_only_boundary_parent_rank48": 4215,
    "first_excluded_control_rank49": 8246,
}

SourceKind = Literal[
    "production_capture",
    "promoted_capture",
    "recomputed_high_precision",
]


def _dtype_name(dtype) -> str:
    return np.dtype(dtype).name


def _require_sha256(value: str, *, name: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex string")
    return value


def _require_nonempty(value, *, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value


def _require_int(value, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_float32_scalar(value, *, name: str) -> np.float32:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype != np.dtype(np.float32):
        raise ValueError(
            f"{name} must be an explicitly captured float32 scalar; got dtype={array.dtype}, shape={array.shape}"
        )
    scalar = array[()]
    if not np.isfinite(scalar) or scalar < np.float32(0.0):
        raise ValueError(f"{name} must be finite and nonnegative, got {scalar}")
    return scalar


def _validated_packed_identities(identities, *, name: str) -> np.ndarray:
    identities = np.asarray(identities)
    if identities.shape != (RELION_COARSE_PACKED_PIXELS,):
        raise ValueError(
            f"{name} must retain all {RELION_COARSE_PACKED_PIXELS} packed identities; got shape {identities.shape}"
        )
    if identities.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    identities = identities.astype(np.int64, copy=True)
    expected = np.arange(RELION_COARSE_PACKED_PIXELS, dtype=np.int64)
    if not np.array_equal(np.sort(identities), expected):
        raise ValueError(
            f"{name} must be a unique, complete permutation of 0..1623; "
            "missing or duplicate identities invalidate replay"
        )
    identities.flags.writeable = False
    return identities


def _in_packed_order(values: np.ndarray, identities: np.ndarray) -> np.ndarray:
    ordered = np.empty_like(values)
    ordered[identities] = values
    return ordered


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _has_beyond_float32_information(array: np.ndarray) -> bool:
    low_dtype = np.complex64 if np.iscomplexobj(array) else np.float32
    return not np.array_equal(array, array.astype(low_dtype).astype(array.dtype))


@dataclass(frozen=True)
class GaussianCaptureIdentity:
    """Complete immutable identity shared by production and diagnostic operands."""

    engine: str
    source_commit: str
    source_diff_sha256: str
    executable_sha256: str
    gpu_uuid: str
    dataset_id: str
    original_particle_id: int
    iteration: int
    pass_index: int
    current_size: int
    image_sha256: str
    map_sha256: str
    ctf_sha256: str
    noise_sha256: str
    support_sha256: str
    class_index: int
    candidate_role: str
    rotation_id: str
    local_rotation_index: int
    translation_id: str
    local_translation_index: int
    candidate_geometry_sha256: str
    packed_count: int
    block_size: int
    prefetch: int
    translation_count: int
    kernel_id: str
    schema: str = CAPTURE_SCHEMA
    version: int = CAPTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        engine = _require_nonempty(self.engine, name="engine")
        source_commit = _require_nonempty(self.source_commit, name="source_commit").lower()
        kernel_id = _require_nonempty(self.kernel_id, name="kernel_id")
        if engine != RELION_V1_ENGINE:
            raise ValueError(f"v1 coarse replay engine must be {RELION_V1_ENGINE!r}")
        if source_commit != RELION_V1_SOURCE_COMMIT:
            raise ValueError(f"v1 coarse replay source_commit must be {RELION_V1_SOURCE_COMMIT}")
        if kernel_id != RELION_V1_KERNEL_ID:
            raise ValueError(f"v1 coarse replay kernel_id must be {RELION_V1_KERNEL_ID!r}")
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "source_commit", source_commit)
        object.__setattr__(
            self,
            "source_diff_sha256",
            _require_sha256(self.source_diff_sha256, name="source_diff_sha256"),
        )
        object.__setattr__(
            self,
            "executable_sha256",
            _require_sha256(self.executable_sha256, name="executable_sha256"),
        )
        object.__setattr__(self, "gpu_uuid", _require_nonempty(self.gpu_uuid, name="gpu_uuid"))
        object.__setattr__(self, "dataset_id", _require_nonempty(self.dataset_id, name="dataset_id"))
        frozen_scalars = {
            "original_particle_id": (self.original_particle_id, RELION_V1_ORIGINAL_PARTICLE_ID),
            "iteration": (self.iteration, RELION_V1_ITERATION),
            "pass_index": (self.pass_index, RELION_V1_PASS_INDEX),
            "class_index": (self.class_index, RELION_V1_CLASS_INDEX),
        }
        for field_name, (actual, expected) in frozen_scalars.items():
            actual = _require_int(actual, name=field_name)
            if actual != expected:
                raise ValueError(f"v1 coarse replay {field_name} must be {expected}, got {actual}")
            object.__setattr__(self, field_name, actual)
        current_size = _require_int(self.current_size, name="current_size", minimum=1)
        if current_size != RELION_V1_CURRENT_SIZE:
            raise ValueError(f"v1 coarse replay current_size must be {RELION_V1_CURRENT_SIZE}")
        object.__setattr__(self, "current_size", current_size)
        for field_name in ("image_sha256", "map_sha256", "ctf_sha256", "noise_sha256", "support_sha256"):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), name=field_name),
            )
        candidate_role = _require_nonempty(self.candidate_role, name="candidate_role")
        if candidate_role not in RELION_V1_ROLE_TRANSLATION_INDEX:
            raise ValueError(f"unknown v1 candidate_role {candidate_role!r}")
        object.__setattr__(self, "candidate_role", candidate_role)
        rotation_id = _require_nonempty(self.rotation_id, name="rotation_id")
        local_rotation_index = _require_int(self.local_rotation_index, name="local_rotation_index")
        expected_rotation_index = RELION_V1_ROLE_ROTATION_INDEX[candidate_role]
        if local_rotation_index != expected_rotation_index:
            raise ValueError(
                f"v1 candidate_role {candidate_role!r} requires local_rotation_index "
                f"{expected_rotation_index}, got {local_rotation_index}"
            )
        expected_rotation_id = f"rotation-{local_rotation_index}"
        if rotation_id != expected_rotation_id:
            raise ValueError(f"rotation_id must be the numeric-derived {expected_rotation_id!r}, got {rotation_id!r}")
        object.__setattr__(self, "rotation_id", rotation_id)
        object.__setattr__(self, "local_rotation_index", local_rotation_index)
        translation_id = _require_nonempty(self.translation_id, name="translation_id")
        local_translation_index = _require_int(
            self.local_translation_index,
            name="local_translation_index",
        )
        expected_translation_index = RELION_V1_ROLE_TRANSLATION_INDEX[candidate_role]
        if local_translation_index != expected_translation_index:
            raise ValueError(
                f"v1 candidate_role {candidate_role!r} requires local_translation_index "
                f"{expected_translation_index}, got {local_translation_index}"
            )
        expected_translation_id = f"translation-{local_translation_index}"
        if translation_id != expected_translation_id:
            raise ValueError(
                f"translation_id must be the numeric-derived {expected_translation_id!r}, got {translation_id!r}"
            )
        object.__setattr__(self, "translation_id", translation_id)
        object.__setattr__(self, "local_translation_index", local_translation_index)
        object.__setattr__(
            self,
            "candidate_geometry_sha256",
            _require_sha256(self.candidate_geometry_sha256, name="candidate_geometry_sha256"),
        )
        layout = {
            "packed_count": (self.packed_count, RELION_COARSE_PACKED_PIXELS),
            "block_size": (self.block_size, RELION_V1_BLOCK_SIZE),
            "prefetch": (self.prefetch, RELION_V1_PREFETCH),
            "translation_count": (self.translation_count, RELION_V1_TRANSLATION_COUNT),
        }
        for field_name, (actual, expected) in layout.items():
            actual = _require_int(actual, name=field_name, minimum=1)
            if actual != expected:
                raise ValueError(f"v1 coarse replay {field_name} must be {expected}, got {actual}")
            object.__setattr__(self, field_name, actual)
        object.__setattr__(self, "kernel_id", kernel_id)
        if self.schema != CAPTURE_SCHEMA or self.version != CAPTURE_SCHEMA_VERSION:
            raise ValueError(f"capture identity must use {CAPTURE_SCHEMA!r} version {CAPTURE_SCHEMA_VERSION}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "engine": self.engine,
            "source_commit": self.source_commit,
            "source_diff_sha256": self.source_diff_sha256,
            "executable_sha256": self.executable_sha256,
            "gpu_uuid": self.gpu_uuid,
            "dataset_id": self.dataset_id,
            "original_particle_id": self.original_particle_id,
            "iteration": self.iteration,
            "pass": self.pass_index,
            "current_size": self.current_size,
            "image_sha256": self.image_sha256,
            "map_sha256": self.map_sha256,
            "ctf_sha256": self.ctf_sha256,
            "noise_sha256": self.noise_sha256,
            "support_sha256": self.support_sha256,
            "class_index": self.class_index,
            "candidate_role": self.candidate_role,
            "rotation_id": self.rotation_id,
            "local_rotation_index": self.local_rotation_index,
            "translation_id": self.translation_id,
            "local_translation_index": self.local_translation_index,
            "candidate_geometry_sha256": self.candidate_geometry_sha256,
            "packed_count": self.packed_count,
            "block_size": self.block_size,
            "prefetch": self.prefetch,
            "translation_count": self.translation_count,
            "kernel_id": self.kernel_id,
        }


@dataclass(frozen=True)
class GaussianContributionCapture:
    """Complete float32 production reduction boundary for one candidate."""

    capture_identity: GaussianCaptureIdentity
    contributions: np.ndarray
    packed_identities: np.ndarray
    source_dtypes: tuple[str, ...]
    captured_lane_partials_float32: np.ndarray
    initial_highres_xi2_over_2: np.float32
    observed_raw_diff2: np.float32

    def __post_init__(self) -> None:
        if not isinstance(self.capture_identity, GaussianCaptureIdentity):
            raise TypeError("capture_identity must be a GaussianCaptureIdentity")
        contributions = np.asarray(self.contributions)
        if contributions.shape != (RELION_COARSE_PACKED_PIXELS,):
            raise ValueError(
                "exact RELION coarse replay requires all "
                f"{RELION_COARSE_PACKED_PIXELS} contributions; got {contributions.shape}"
            )
        if contributions.dtype != np.dtype(np.float32):
            raise ValueError("production contributions must be captured float32 values")
        if not np.all(np.isfinite(contributions)) or np.any(contributions < 0):
            raise ValueError("Gaussian contributions must be finite and nonnegative")
        identities = _validated_packed_identities(self.packed_identities, name="packed_identities")
        source_dtypes = tuple(_dtype_name(dtype) for dtype in self.source_dtypes)
        if not source_dtypes:
            raise ValueError("source_dtypes must record production operand dtypes")
        captured_lanes = np.asarray(self.captured_lane_partials_float32)
        if captured_lanes.shape != (RELION_COARSE_ATOMIC_LANES,):
            raise ValueError("captured_lane_partials_float32 must have shape (5,)")
        if captured_lanes.dtype != np.dtype(np.float32):
            raise ValueError("captured lane partials must be device-captured float32 values")
        if not np.all(np.isfinite(captured_lanes)) or np.any(captured_lanes < 0):
            raise ValueError("captured lane partials must be finite and nonnegative")
        if captured_lanes[-1].view(np.uint32) != np.float32(0.0).view(np.uint32):
            raise ValueError("the fifth captured lane partial must be positive float32 zero")
        initial = _require_float32_scalar(
            self.initial_highres_xi2_over_2,
            name="initial_highres_xi2_over_2",
        )
        observed = _require_float32_scalar(self.observed_raw_diff2, name="observed_raw_diff2")

        contributions = contributions.copy()
        captured_lanes = captured_lanes.copy()
        contributions.flags.writeable = False
        captured_lanes.flags.writeable = False
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "packed_identities", identities)
        object.__setattr__(self, "source_dtypes", source_dtypes)
        object.__setattr__(self, "captured_lane_partials_float32", captured_lanes)
        object.__setattr__(self, "initial_highres_xi2_over_2", initial)
        object.__setattr__(self, "observed_raw_diff2", observed)

    def in_packed_order(self) -> np.ndarray:
        return _in_packed_order(self.contributions, self.packed_identities)


@dataclass(frozen=True)
class HighPrecisionGaussianOperands:
    """Complete complex128/float64 operands used to recompute contributions."""

    capture_identity: GaussianCaptureIdentity
    reference_complex: np.ndarray
    shifted_image_complex: np.ndarray
    corr_over_2: np.ndarray
    packed_identities: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.capture_identity, GaussianCaptureIdentity):
            raise TypeError("capture_identity must be a GaussianCaptureIdentity")
        reference = np.asarray(self.reference_complex)
        shifted = np.asarray(self.shifted_image_complex)
        weight = np.asarray(self.corr_over_2)
        required_shape = (RELION_COARSE_PACKED_PIXELS,)
        if reference.shape != required_shape or shifted.shape != required_shape or weight.shape != required_shape:
            raise ValueError("high-precision reference, shifted image, and weight must each retain all 1624 pixels")
        if reference.dtype != np.dtype(np.complex128) or shifted.dtype != np.dtype(np.complex128):
            raise ValueError("high-precision reference and shifted image must be complex128")
        if weight.dtype != np.dtype(np.float64):
            raise ValueError("high-precision corr_over_2 must be float64")
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(shifted)):
            raise ValueError("high-precision complex operands must be finite")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("high-precision corr_over_2 must be finite and nonnegative")
        beyond_production = tuple(_has_beyond_float32_information(array) for array in (reference, shifted, weight))
        if not any(beyond_production):
            raise ValueError(
                "all high-precision operands round-trip exactly through production precision; "
                "cannot accept an all-cast-spoofed capture as genuine float64"
            )
        identities = _validated_packed_identities(self.packed_identities, name="packed_identities")

        reference = reference.copy()
        shifted = shifted.copy()
        weight = weight.copy()
        reference.flags.writeable = False
        shifted.flags.writeable = False
        weight.flags.writeable = False
        object.__setattr__(self, "reference_complex", reference)
        object.__setattr__(self, "shifted_image_complex", shifted)
        object.__setattr__(self, "corr_over_2", weight)
        object.__setattr__(self, "packed_identities", identities)

    @property
    def source_array_sha256(self) -> tuple[str, str, str]:
        return (
            _array_sha256(self.reference_complex),
            _array_sha256(self.shifted_image_complex),
            _array_sha256(self.corr_over_2),
        )

    @property
    def source_beyond_production_precision(self) -> tuple[bool, bool, bool]:
        return tuple(
            _has_beyond_float32_information(array)
            for array in (self.reference_complex, self.shifted_image_complex, self.corr_over_2)
        )

    def in_packed_order(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            _in_packed_order(self.reference_complex, self.packed_identities),
            _in_packed_order(self.shifted_image_complex, self.packed_identities),
            _in_packed_order(self.corr_over_2, self.packed_identities),
        )


@dataclass(frozen=True)
class InitialReductionTerm:
    value: float
    source_dtype: str
    replay_dtype: str
    source_kind: SourceKind

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic": "highres_xi2_over_2",
            "value": self.value,
            "source_dtype": self.source_dtype,
            "replay_dtype": self.replay_dtype,
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True)
class ObservedRawDiff2:
    value: float
    dtype: str = "float32"
    source_kind: SourceKind = "production_capture"
    atomic_order_observed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "dtype": self.dtype,
            "source_kind": self.source_kind,
            "atomic_order_observed": self.atomic_order_observed,
        }


@dataclass(frozen=True)
class ReductionProvenance:
    source_dtypes: tuple[str, ...]
    source_array_sha256: tuple[str, ...]
    source_beyond_production_precision: tuple[bool, ...]
    contribution_dtype: str
    accumulation_dtype: str
    source_kind: SourceKind
    order_kind: str
    scope: str
    initial_term: InitialReductionTerm | None = None

    @property
    def genuine_source_high_precision(self) -> bool:
        return self.source_kind == "recomputed_high_precision"

    def to_dict(self) -> dict[str, object]:
        return {
            "source_dtypes": list(self.source_dtypes),
            "source_array_sha256": list(self.source_array_sha256),
            "source_beyond_production_precision": list(self.source_beyond_production_precision),
            "contribution_dtype": self.contribution_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "source_kind": self.source_kind,
            "order_kind": self.order_kind,
            "scope": self.scope,
            "initial_term": None if self.initial_term is None else self.initial_term.to_dict(),
            "genuine_source_high_precision": self.genuine_source_high_precision,
        }


@dataclass(frozen=True)
class LanePartialReplay:
    values: tuple[float, float, float, float, float]
    pixel_counts: tuple[int, int, int, int, int]
    provenance: ReductionProvenance

    def to_dict(self) -> dict[str, object]:
        return {
            "values": list(self.values),
            "pixel_counts": list(self.pixel_counts),
            "dtype_provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class GaussianReduction:
    value: float
    provenance: ReductionProvenance
    atomic_order: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "value": self.value,
            "dtype_provenance": self.provenance.to_dict(),
        }
        if self.atomic_order is not None:
            result["atomic_order"] = list(self.atomic_order)
        return result


@dataclass(frozen=True)
class GaussianReductionReplayReport:
    capture_identity: GaussianCaptureIdentity
    lane_partials_float32: LanePartialReplay
    initial_highres_xi2_over_2_float32: InitialReductionTerm
    observed_raw_diff2_float32: ObservedRawDiff2
    possible_atomic_float32: tuple[GaussianReduction, ...]
    compatible_atomic_orders_float32: tuple[tuple[int, int, int, int], ...]
    canonical_float32: GaussianReduction
    promoted_float64_initial_highres_xi2_over_2: InitialReductionTerm
    promoted_float64_relion_lane_orders: tuple[GaussianReduction, ...]
    promoted_float64_canonical: GaussianReduction
    genuine_float64_centered_relion_lane_orders: tuple[GaussianReduction, ...] | None = None
    genuine_float64_centered_canonical: GaussianReduction | None = None
    schema: str = REPLAY_SCHEMA
    schema_version: int = REPLAY_SCHEMA_VERSION

    @property
    def has_genuine_centered_float64(self) -> bool:
        return self.genuine_float64_centered_canonical is not None

    def require_genuine_centered_float64(self) -> GaussianReduction:
        if self.genuine_float64_centered_canonical is None:
            raise ValueError("no genuine complex128/float64 operand capture was supplied")
        return self.genuine_float64_centered_canonical

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "capture_identity": self.capture_identity.to_dict(),
            "genuine_float64_scope": "centered_contribution_only_no_highres_xi2",
            "has_genuine_centered_float64": self.has_genuine_centered_float64,
            "lane_partials_float32": self.lane_partials_float32.to_dict(),
            "initial_highres_xi2_over_2_float32": self.initial_highres_xi2_over_2_float32.to_dict(),
            "observed_raw_diff2_float32": self.observed_raw_diff2_float32.to_dict(),
            "possible_atomic_float32": [reduction.to_dict() for reduction in self.possible_atomic_float32],
            "compatible_atomic_orders_float32": [list(order) for order in self.compatible_atomic_orders_float32],
            "production_atomic_order_observed": False,
            "canonical_float32": self.canonical_float32.to_dict(),
            "promoted_float64_initial_highres_xi2_over_2": (self.promoted_float64_initial_highres_xi2_over_2.to_dict()),
            "promoted_float64_relion_lane_orders": [
                reduction.to_dict() for reduction in self.promoted_float64_relion_lane_orders
            ],
            "promoted_float64_canonical": self.promoted_float64_canonical.to_dict(),
            "genuine_float64_centered_relion_lane_orders": (
                None
                if self.genuine_float64_centered_relion_lane_orders is None
                else [reduction.to_dict() for reduction in self.genuine_float64_centered_relion_lane_orders]
            ),
            "genuine_float64_centered_canonical": (
                None
                if self.genuine_float64_centered_canonical is None
                else self.genuine_float64_centered_canonical.to_dict()
            ),
        }


def _production_initial(capture: GaussianContributionCapture) -> InitialReductionTerm:
    return InitialReductionTerm(
        float(capture.initial_highres_xi2_over_2),
        "float32",
        "float32",
        "production_capture",
    )


def _promoted_initial(capture: GaussianContributionCapture) -> InitialReductionTerm:
    return InitialReductionTerm(
        float(np.float64(capture.initial_highres_xi2_over_2)),
        "float32",
        "float64",
        "promoted_capture",
    )


def _left_fold(values, *, dtype, initial_value):
    total = dtype(initial_value)
    for value in values:
        total = dtype(total + dtype(value))
    return total


def _lane_values(values: np.ndarray, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    values = values.astype(dtype, copy=False)
    lanes = np.zeros(RELION_COARSE_ATOMIC_LANES, dtype=dtype)
    counts = np.zeros(RELION_COARSE_ATOMIC_LANES, dtype=np.int64)
    for chunk_start in range(0, RELION_COARSE_PACKED_PIXELS, RELION_COARSE_CHUNK_PIXELS):
        chunk_stop = min(chunk_start + RELION_COARSE_CHUNK_PIXELS, RELION_COARSE_PACKED_PIXELS)
        for lane in range(RELION_COARSE_NONZERO_LANES):
            for pixel in range(chunk_start + lane, chunk_stop, RELION_COARSE_NONZERO_LANES):
                lanes[lane] = dtype(lanes[lane] + values[pixel])
                counts[lane] += 1
    return lanes, counts


def _provenance(
    *,
    source_dtypes: tuple[str, ...],
    source_hashes: tuple[str, ...],
    source_precision_flags: tuple[bool, ...],
    contribution_dtype,
    accumulation_dtype,
    source_kind: SourceKind,
    order_kind: str,
    scope: str,
    initial_term: InitialReductionTerm | None = None,
) -> ReductionProvenance:
    return ReductionProvenance(
        source_dtypes=source_dtypes,
        source_array_sha256=source_hashes,
        source_beyond_production_precision=source_precision_flags,
        contribution_dtype=_dtype_name(contribution_dtype),
        accumulation_dtype=_dtype_name(accumulation_dtype),
        source_kind=source_kind,
        order_kind=order_kind,
        scope=scope,
        initial_term=initial_term,
    )


def relion_coarse_lane_partials_float32(capture: GaussianContributionCapture) -> LanePartialReplay:
    """Validate and return five device-captured production lane partials."""

    values = capture.in_packed_order()
    replayed_lanes, counts = _lane_values(values, dtype=np.float32)
    captured_lanes = capture.captured_lane_partials_float32
    mismatch = np.flatnonzero(replayed_lanes.view(np.uint32) != captured_lanes.view(np.uint32))
    if mismatch.size:
        raise ValueError(
            "captured float32 lane partials do not bitwise match the contribution-derived RELION schedule; "
            f"mismatched_lanes={mismatch.tolist()}"
        )
    provenance = _provenance(
        source_dtypes=capture.source_dtypes,
        source_hashes=(
            _array_sha256(capture.contributions),
            _array_sha256(capture.captured_lane_partials_float32),
        ),
        source_precision_flags=(False, False),
        contribution_dtype=np.float32,
        accumulation_dtype=np.float32,
        source_kind="production_capture",
        order_kind="device_captured_relion_lanes_validated_against_replay",
        scope="centered_contribution_only",
    )
    return LanePartialReplay(
        values=tuple(float(value) for value in captured_lanes),
        pixel_counts=tuple(int(value) for value in counts),
        provenance=provenance,
    )


def _enumerate_lane_orders(
    values: np.ndarray,
    *,
    accumulation_dtype,
    source_dtypes: tuple[str, ...],
    source_hashes: tuple[str, ...],
    source_precision_flags: tuple[bool, ...],
    source_kind: SourceKind,
    scope: str,
    initial_term: InitialReductionTerm | None,
) -> tuple[GaussianReduction, ...]:
    lanes, _ = _lane_values(values, dtype=accumulation_dtype)
    initial_value = 0.0 if initial_term is None else initial_term.value
    provenance = _provenance(
        source_dtypes=source_dtypes,
        source_hashes=source_hashes,
        source_precision_flags=source_precision_flags,
        contribution_dtype=values.dtype,
        accumulation_dtype=accumulation_dtype,
        source_kind=source_kind,
        order_kind="relion_four_nonzero_atomic_arrival_order",
        scope=scope,
        initial_term=initial_term,
    )
    reductions = []
    for order in permutations(range(RELION_COARSE_NONZERO_LANES)):
        total = _left_fold(
            lanes[list(order)],
            dtype=accumulation_dtype,
            initial_value=initial_value,
        )
        reductions.append(GaussianReduction(float(total), provenance, order))
    return tuple(reductions)


def _enumerate_captured_float32_lane_orders(
    capture: GaussianContributionCapture,
) -> tuple[GaussianReduction, ...]:
    lane_replay = relion_coarse_lane_partials_float32(capture)
    lanes = np.asarray(lane_replay.values[:RELION_COARSE_NONZERO_LANES], dtype=np.float32)
    initial = _production_initial(capture)
    provenance = _provenance(
        source_dtypes=capture.source_dtypes,
        source_hashes=(_array_sha256(capture.contributions), _array_sha256(capture.captured_lane_partials_float32)),
        source_precision_flags=(False, False),
        contribution_dtype=np.float32,
        accumulation_dtype=np.float32,
        source_kind="production_capture",
        order_kind="device_captured_four_nonzero_atomic_arrival_order",
        scope="full_raw_diff2",
        initial_term=initial,
    )
    reductions = []
    for order in permutations(range(RELION_COARSE_NONZERO_LANES)):
        total = _left_fold(lanes[list(order)], dtype=np.float32, initial_value=initial.value)
        reductions.append(GaussianReduction(float(total), provenance, order))
    return tuple(reductions)


def enumerate_relion_coarse_atomic_float32(
    capture: GaussianContributionCapture,
) -> tuple[GaussianReduction, ...]:
    """Enumerate all 24 possible production atomic arrival orders."""

    return _enumerate_captured_float32_lane_orders(capture)


def _canonical_reduce(
    values: np.ndarray,
    *,
    accumulation_dtype,
    source_dtypes: tuple[str, ...],
    source_hashes: tuple[str, ...],
    source_precision_flags: tuple[bool, ...],
    source_kind: SourceKind,
    scope: str,
    initial_term: InitialReductionTerm | None,
) -> GaussianReduction:
    initial_value = 0.0 if initial_term is None else initial_term.value
    total = _left_fold(values, dtype=accumulation_dtype, initial_value=initial_value)
    return GaussianReduction(
        float(total),
        _provenance(
            source_dtypes=source_dtypes,
            source_hashes=source_hashes,
            source_precision_flags=source_precision_flags,
            contribution_dtype=values.dtype,
            accumulation_dtype=accumulation_dtype,
            source_kind=source_kind,
            order_kind="canonical_packed_identity",
            scope=scope,
            initial_term=initial_term,
        ),
    )


def recompute_high_precision_gaussian_contributions(
    operands: HighPrecisionGaussianOperands,
) -> np.ndarray:
    """Form genuine float64 Gaussian contributions from captured operands."""

    reference, shifted, weight = operands.in_packed_order()
    diff_real = np.subtract(reference.real, shifted.real, dtype=np.float64)
    diff_imag = np.subtract(reference.imag, shifted.imag, dtype=np.float64)
    squared = np.add(
        np.multiply(diff_real, diff_real, dtype=np.float64),
        np.multiply(diff_imag, diff_imag, dtype=np.float64),
        dtype=np.float64,
    )
    return np.multiply(squared, weight, dtype=np.float64)


def replay_relion_coarse_gaussian(
    capture: GaussianContributionCapture,
    *,
    high_precision_operands: HighPrecisionGaussianOperands | None = None,
) -> GaussianReductionReplayReport:
    """Replay one complete production capture and optional genuine operands."""

    production_values = capture.in_packed_order()
    initial_f32 = _production_initial(capture)
    possible_atomic = enumerate_relion_coarse_atomic_float32(capture)
    observed_bits = capture.observed_raw_diff2.view(np.uint32)
    compatible = tuple(
        reduction.atomic_order
        for reduction in possible_atomic
        if np.float32(reduction.value).view(np.uint32) == observed_bits
    )
    if not compatible:
        possible_bits = sorted({int(np.float32(item.value).view(np.uint32)) for item in possible_atomic})
        raise ValueError(
            "observed production raw diff2 is incompatible with all 24 RELION atomic orders; "
            f"observed_bits={int(observed_bits)}, possible_bits={possible_bits}"
        )

    production_hashes = (_array_sha256(capture.contributions),)
    canonical_f32 = _canonical_reduce(
        production_values,
        accumulation_dtype=np.float32,
        source_dtypes=capture.source_dtypes,
        source_hashes=production_hashes,
        source_precision_flags=(False,),
        source_kind="production_capture",
        scope="full_raw_diff2",
        initial_term=initial_f32,
    )
    promoted_initial = _promoted_initial(capture)
    promoted_orders = _enumerate_lane_orders(
        production_values,
        accumulation_dtype=np.float64,
        source_dtypes=capture.source_dtypes,
        source_hashes=production_hashes,
        source_precision_flags=(False,),
        source_kind="promoted_capture",
        scope="full_raw_diff2",
        initial_term=promoted_initial,
    )
    promoted_canonical = _canonical_reduce(
        production_values,
        accumulation_dtype=np.float64,
        source_dtypes=capture.source_dtypes,
        source_hashes=production_hashes,
        source_precision_flags=(False,),
        source_kind="promoted_capture",
        scope="full_raw_diff2",
        initial_term=promoted_initial,
    )

    genuine_orders = None
    genuine_canonical = None
    if high_precision_operands is not None:
        if not isinstance(high_precision_operands, HighPrecisionGaussianOperands):
            raise TypeError("high_precision_operands must be HighPrecisionGaussianOperands")
        if capture.capture_identity != high_precision_operands.capture_identity:
            raise ValueError(
                "production and high-precision capture identities differ; "
                "dataset/particle/iteration/pass/size/class/candidate geometry must match exactly"
            )
        if not np.array_equal(
            np.sort(capture.packed_identities),
            np.sort(high_precision_operands.packed_identities),
        ):
            raise ValueError("production and high-precision packed identity mappings differ")
        genuine_values = recompute_high_precision_gaussian_contributions(high_precision_operands)
        source_dtypes = ("complex128", "complex128", "float64")
        source_hashes = high_precision_operands.source_array_sha256
        genuine_orders = _enumerate_lane_orders(
            genuine_values,
            accumulation_dtype=np.float64,
            source_dtypes=source_dtypes,
            source_hashes=source_hashes,
            source_precision_flags=high_precision_operands.source_beyond_production_precision,
            source_kind="recomputed_high_precision",
            scope="centered_contribution_only_no_highres_xi2",
            initial_term=None,
        )
        genuine_canonical = _canonical_reduce(
            genuine_values,
            accumulation_dtype=np.float64,
            source_dtypes=source_dtypes,
            source_hashes=source_hashes,
            source_precision_flags=high_precision_operands.source_beyond_production_precision,
            source_kind="recomputed_high_precision",
            scope="centered_contribution_only_no_highres_xi2",
            initial_term=None,
        )

    return GaussianReductionReplayReport(
        capture_identity=capture.capture_identity,
        lane_partials_float32=relion_coarse_lane_partials_float32(capture),
        initial_highres_xi2_over_2_float32=initial_f32,
        observed_raw_diff2_float32=ObservedRawDiff2(float(capture.observed_raw_diff2)),
        possible_atomic_float32=possible_atomic,
        compatible_atomic_orders_float32=compatible,
        canonical_float32=canonical_f32,
        promoted_float64_initial_highres_xi2_over_2=promoted_initial,
        promoted_float64_relion_lane_orders=promoted_orders,
        promoted_float64_canonical=promoted_canonical,
        genuine_float64_centered_relion_lane_orders=genuine_orders,
        genuine_float64_centered_canonical=genuine_canonical,
    )
