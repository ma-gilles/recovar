"""Deterministic normalized-CC reduction replay diagnostics.

This module operates on frozen per-pixel operands.  It does not replace the
production scorer and deliberately has no JAX dependency.  In particular, a
float64 replay of captured float32 contributions is labelled as *promoted*:
it cannot recover precision lost while producing those contributions.
``normalized_cc_pixel_contributions`` evaluates the logical expression; an
exact device replay must supply contributions captured after device operand
formation so compiler contraction and texture/interpolation effects are not
silently attributed to reduction order.

The RELION reducers mirror ``cuda_kernel_diff2_CC_coarse`` and
``cuda_kernel_diff2_CC_fine`` in
``src/acc/cuda/cuda_kernels/diff2.cuh`` for the SPA ``REF3D`` path: 256 lanes
for fine scoring and 128 lanes for coarse scoring.  Lanes first accumulate
pixels ``lane + lane_count * pass`` and are then reduced by the shared-memory
power-of-two tree.  The input must therefore retain the full packed-grid pixel
order, including zero-contribution pixels outside support.

Map-level effects of a diagnosed score difference must be assessed with
shellwise FSC/FSC-AUC, not map correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

REPLAY_SCHEMA = "recovar-normalized-cc-replay-v1"
REPLAY_SCHEMA_VERSION = 1
RELION_FINE_REDUCTION_LANES = 256
RELION_COARSE_REDUCTION_LANES = 128

PrecisionOrigin = Literal[
    "captured_production",
    "promoted_captured",
    "recomputed_high_precision",
]


def _dtype_name(dtype) -> str:
    return np.dtype(dtype).name


def _require_float32_or_float64(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"expected float32 or float64, got {dtype}")
    return dtype


def _is_high_precision_source_dtype(dtype) -> bool:
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        return dtype.itemsize >= np.dtype(np.complex128).itemsize
    if dtype.kind == "f":
        return dtype.itemsize >= np.dtype(np.float64).itemsize
    return dtype.kind in "iub"


def _as_1d(values, *, dtype=None, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


@dataclass(frozen=True)
class DTypeProvenance:
    """Dtype lineage for one contribution or reduction result."""

    source_dtypes: tuple[str, ...]
    contribution_dtype: str
    accumulation_dtype: str
    precision_origin: PrecisionOrigin

    def __post_init__(self) -> None:
        if self.precision_origin not in {
            "captured_production",
            "promoted_captured",
            "recomputed_high_precision",
        }:
            raise ValueError(f"unknown precision_origin {self.precision_origin!r}")
        for source_dtype in self.source_dtypes:
            np.dtype(source_dtype)
        _require_float32_or_float64(self.contribution_dtype)
        _require_float32_or_float64(self.accumulation_dtype)
        if self.precision_origin == "recomputed_high_precision" and not all(
            _is_high_precision_source_dtype(dtype) for dtype in self.source_dtypes
        ):
            raise ValueError("recomputed_high_precision requires genuinely high-precision source dtypes")

    @property
    def genuine_source_high_precision(self) -> bool:
        """Whether high precision was present before contribution formation."""

        return self.precision_origin == "recomputed_high_precision"

    def require_genuine_source_high_precision(self) -> None:
        """Reject promoted float32 evidence when genuine precision is required."""

        if not self.genuine_source_high_precision:
            raise ValueError(
                "this replay only promotes captured low-precision operands; "
                "recompute the operands from high-precision sources"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_dtypes": list(self.source_dtypes),
            "contribution_dtype": self.contribution_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "precision_origin": self.precision_origin,
            "genuine_source_high_precision": self.genuine_source_high_precision,
        }


@dataclass(frozen=True)
class NormalizedCCContributions:
    """Per-pixel numerator and projection-norm contributions."""

    numerator: np.ndarray
    norm: np.ndarray
    provenance: DTypeProvenance

    def __post_init__(self) -> None:
        numerator = _as_1d(self.numerator, name="numerator")
        norm = _as_1d(self.norm, name="norm")
        if numerator.shape != norm.shape:
            raise ValueError(f"numerator and norm shapes differ: {numerator.shape} != {norm.shape}")
        if numerator.dtype != norm.dtype:
            raise TypeError(f"contribution dtypes differ: {numerator.dtype} != {norm.dtype}")
        if _dtype_name(numerator.dtype) != self.provenance.contribution_dtype:
            raise ValueError("contribution dtype does not match its provenance")


@dataclass(frozen=True)
class NormalizedCCReduction:
    """Reduced numerator, norm, and positive normalized-CC score."""

    numerator: float
    norm: float
    score: float
    provenance: DTypeProvenance

    def to_dict(self) -> dict[str, object]:
        return {
            "numerator": self.numerator,
            "norm": self.norm,
            "score": self.score,
            "dtype_provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class NormalizedCCReplayReport:
    """Schema-v1 comparison of production and canonical reductions."""

    classification: str
    recovar_logical_float32: NormalizedCCReduction
    relion_256lane_float32: NormalizedCCReduction
    canonical_float32: NormalizedCCReduction
    canonical_float64: NormalizedCCReduction
    schema: str = REPLAY_SCHEMA
    schema_version: int = REPLAY_SCHEMA_VERSION

    @property
    def has_genuine_source_float64(self) -> bool:
        return self.canonical_float64.provenance.genuine_source_high_precision

    def require_genuine_source_float64(self) -> None:
        self.canonical_float64.provenance.require_genuine_source_high_precision()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "classification": self.classification,
            "has_genuine_source_float64": self.has_genuine_source_float64,
            "reductions": {
                "recovar_logical_float32": self.recovar_logical_float32.to_dict(),
                "relion_256lane_float32": self.relion_256lane_float32.to_dict(),
                "canonical_float32": self.canonical_float32.to_dict(),
                "canonical_float64": self.canonical_float64.to_dict(),
            },
        }


@dataclass(frozen=True)
class NormalizedCCCandidateReplay:
    """Winner-preserving replay of several candidates from one engine.

    Ties are retained as a tuple instead of being resolved by array order.
    This is important at parity boundaries where an arbitrary discrete
    tie-break is not evidence of an algorithmic mismatch.
    """

    candidate_ids: tuple[int, ...]
    production_scores: tuple[float, ...]
    canonical_float64_scores: tuple[float, ...]
    production_winners: tuple[int, ...]
    canonical_float64_winners: tuple[int, ...]
    production_reducer: str
    precision_origin: PrecisionOrigin

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "production_scores": list(self.production_scores),
            "canonical_float64_scores": list(self.canonical_float64_scores),
            "production_winners": list(self.production_winners),
            "canonical_float64_winners": list(self.canonical_float64_winners),
            "production_reducer": self.production_reducer,
            "precision_origin": self.precision_origin,
        }


@dataclass(frozen=True)
class NormalizedCCCrossEngineClassification:
    """Earliest classified source of a cross-engine winner difference."""

    classification: str
    recovar: NormalizedCCCandidateReplay
    relion: NormalizedCCCandidateReplay
    geometry_equal: bool
    genuine_float64_recovar: NormalizedCCCandidateReplay | None = None
    genuine_float64_relion: NormalizedCCCandidateReplay | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPLAY_SCHEMA,
            "schema_version": REPLAY_SCHEMA_VERSION,
            "classification": self.classification,
            "geometry_equal": self.geometry_equal,
            "recovar": self.recovar.to_dict(),
            "relion": self.relion.to_dict(),
            "genuine_float64_recovar": (
                None if self.genuine_float64_recovar is None else self.genuine_float64_recovar.to_dict()
            ),
            "genuine_float64_relion": (
                None if self.genuine_float64_relion is None else self.genuine_float64_relion.to_dict()
            ),
        }


def normalized_cc_pixel_contributions(
    projection,
    shifted_image,
    score_weight,
    half_weights,
    *,
    arithmetic_dtype=np.float32,
    precision_origin: PrecisionOrigin = "captured_production",
) -> NormalizedCCContributions:
    """Form numerator and norm operands using RECOVAR's logical expression.

    ``projection`` and ``shifted_image`` are complex Fourier samples;
    ``score_weight`` and ``half_weights`` are real.  The returned score has the
    positive log-weight convention ``sum(numerator) / sqrt(sum(norm))``.
    RELION stores its corresponding fine ``diff2`` contribution with a minus
    sign after reduction.

    ``precision_origin`` is an evidence label, not a request to invent source
    precision.  Use ``promoted_captured`` when casting captured float32 or
    complex64 inputs to float64.  ``recomputed_high_precision`` is accepted
    only with float64 arithmetic.
    """

    arithmetic_dtype = _require_float32_or_float64(arithmetic_dtype)
    if precision_origin == "captured_production" and arithmetic_dtype != np.dtype(np.float32):
        raise ValueError("captured_production contributions must use float32 arithmetic")
    if precision_origin == "promoted_captured" and arithmetic_dtype != np.dtype(np.float64):
        raise ValueError("promoted_captured contributions must use float64 arithmetic")
    if precision_origin == "recomputed_high_precision" and arithmetic_dtype != np.dtype(np.float64):
        raise ValueError("recomputed_high_precision contributions must use float64 arithmetic")
    if precision_origin not in {
        "captured_production",
        "promoted_captured",
        "recomputed_high_precision",
    }:
        raise ValueError(f"unknown precision_origin {precision_origin!r}")

    projection = _as_1d(projection, name="projection")
    shifted_image = _as_1d(shifted_image, name="shifted_image")
    score_weight = _as_1d(score_weight, name="score_weight")
    half_weights = _as_1d(half_weights, name="half_weights")
    shapes = {projection.shape, shifted_image.shape, score_weight.shape, half_weights.shape}
    if len(shapes) != 1:
        raise ValueError("projection, shifted_image, score_weight, and half_weights must have one common shape")
    if not np.iscomplexobj(projection) or not np.iscomplexobj(shifted_image):
        raise TypeError("projection and shifted_image must be complex")
    if precision_origin == "recomputed_high_precision" and not all(
        _is_high_precision_source_dtype(value.dtype)
        for value in (projection, shifted_image, score_weight, half_weights)
    ):
        raise ValueError(
            "recomputed_high_precision requires complex128/float64 source operands; "
            "casting captured complex64/float32 operands is only promoted_captured"
        )

    source_dtypes = tuple(_dtype_name(value.dtype) for value in (projection, shifted_image, score_weight, half_weights))
    projection_real = projection.real.astype(arithmetic_dtype, copy=False)
    projection_imag = projection.imag.astype(arithmetic_dtype, copy=False)
    shifted_real = shifted_image.real.astype(arithmetic_dtype, copy=False)
    shifted_imag = shifted_image.imag.astype(arithmetic_dtype, copy=False)
    score_weight = score_weight.astype(arithmetic_dtype, copy=False)
    half_weights = half_weights.astype(arithmetic_dtype, copy=False)

    numerator = np.add(
        np.multiply(projection_real, shifted_real, dtype=arithmetic_dtype),
        np.multiply(projection_imag, shifted_imag, dtype=arithmetic_dtype),
        dtype=arithmetic_dtype,
    )
    numerator = np.multiply(numerator, half_weights, dtype=arithmetic_dtype)
    projection_abs2 = np.add(
        np.multiply(projection_real, projection_real, dtype=arithmetic_dtype),
        np.multiply(projection_imag, projection_imag, dtype=arithmetic_dtype),
        dtype=arithmetic_dtype,
    )
    projection_abs2 = np.multiply(projection_abs2, half_weights, dtype=arithmetic_dtype)
    norm = np.multiply(score_weight, projection_abs2, dtype=arithmetic_dtype)

    provenance = DTypeProvenance(
        source_dtypes=source_dtypes,
        contribution_dtype=_dtype_name(arithmetic_dtype),
        accumulation_dtype=_dtype_name(arithmetic_dtype),
        precision_origin=precision_origin,
    )
    return NormalizedCCContributions(numerator, norm, provenance)


def recovar_logical_float32_reduce(values) -> np.float32:
    """Replay the logical flat float32 reduction in pixel storage order."""

    values = _as_1d(values, dtype=np.float32, name="values")
    total = np.float32(0.0)
    for value in values:
        total = np.float32(total + value)
    return total


def relion_256lane_float32_reduce(values) -> np.float32:
    """Replay RELION SPA fine-score CUDA's exact 256-lane float32 tree.

    ``values`` must include every full packed-grid position.  Compacting away
    zero positions changes lane ownership and therefore changes this result.
    """

    values = _as_1d(values, dtype=np.float32, name="values")
    lanes = np.zeros(RELION_FINE_REDUCTION_LANES, dtype=np.float32)
    for pixel, value in enumerate(values):
        lane = pixel % RELION_FINE_REDUCTION_LANES
        lanes[lane] = np.float32(lanes[lane] + value)
    width = RELION_FINE_REDUCTION_LANES // 2
    while width:
        lanes[:width] = np.add(lanes[:width], lanes[width : 2 * width], dtype=np.float32)
        width //= 2
    return lanes[0]


def relion_128lane_float32_reduce(values) -> np.float32:
    """Replay RELION SPA coarse-CC CUDA's exact 128-lane float32 tree.

    This is the reducer used by ``cuda_kernel_diff2_CC_coarse`` during the
    first-iteration CC pass.  As with the fine replay, ``values`` must retain
    every packed-grid pixel so lane ownership is unchanged.
    """

    values = _as_1d(values, dtype=np.float32, name="values")
    lanes = np.zeros(RELION_COARSE_REDUCTION_LANES, dtype=np.float32)
    for pixel, value in enumerate(values):
        lane = pixel % RELION_COARSE_REDUCTION_LANES
        lanes[lane] = np.float32(lanes[lane] + value)
    width = RELION_COARSE_REDUCTION_LANES // 2
    while width:
        lanes[:width] = np.add(lanes[:width], lanes[width : 2 * width], dtype=np.float32)
        width //= 2
    return lanes[0]


def _canonical_values(values, identities, *, dtype) -> np.ndarray:
    values = _as_1d(values, dtype=dtype, name="values")
    if identities is None:
        return values
    identities = _as_1d(identities, name="identities")
    if identities.shape != values.shape:
        raise ValueError(f"identity shape {identities.shape} != value shape {values.shape}")
    if identities.dtype.kind not in "iu":
        raise TypeError("canonical identities must be integers")
    if np.unique(identities).size != identities.size:
        raise ValueError("canonical identities must be unique")
    return values[np.argsort(identities, kind="stable")]


def _left_fold(values, *, dtype):
    dtype = _require_float32_or_float64(dtype)
    total = dtype.type(0.0)
    for value in values:
        total = dtype.type(total + value)
    return total


def canonical_float32_reduce(values, identities=None) -> np.float32:
    """Deterministically sum float32 contributions by canonical identity."""

    ordered = _canonical_values(values, identities, dtype=np.float32)
    return _left_fold(ordered, dtype=np.float32)


def canonical_float64_reduce(values, identities=None) -> np.float64:
    """Deterministically sum contributions in float64 by canonical identity."""

    ordered = _canonical_values(values, identities, dtype=np.float64)
    return _left_fold(ordered, dtype=np.float64)


def _normalized_score(numerator, norm, *, dtype):
    dtype = _require_float32_or_float64(dtype)
    numerator = dtype.type(numerator)
    norm = dtype.type(norm)
    if not np.isfinite(numerator) or not np.isfinite(norm) or norm <= dtype.type(0.0):
        raise ValueError(
            f"normalized-CC reduction requires finite numerator and positive norm, got {numerator}, {norm}"
        )
    denominator = np.sqrt(norm, dtype=dtype)
    return dtype.type(numerator / denominator)


def _reduction_result(
    contributions: NormalizedCCContributions,
    reducer,
    *,
    accumulation_dtype,
    identities=None,
) -> NormalizedCCReduction:
    if identities is None:
        numerator = reducer(contributions.numerator)
        norm = reducer(contributions.norm)
    else:
        numerator = reducer(contributions.numerator, identities)
        norm = reducer(contributions.norm, identities)
    accumulation_dtype = _require_float32_or_float64(accumulation_dtype)
    precision_origin: PrecisionOrigin = contributions.provenance.precision_origin
    if accumulation_dtype == np.dtype(np.float64) and precision_origin != "recomputed_high_precision":
        precision_origin = "promoted_captured"
    provenance = DTypeProvenance(
        source_dtypes=contributions.provenance.source_dtypes,
        contribution_dtype=contributions.provenance.contribution_dtype,
        accumulation_dtype=_dtype_name(accumulation_dtype),
        precision_origin=precision_origin,
    )
    return NormalizedCCReduction(
        numerator=float(numerator),
        norm=float(norm),
        score=float(_normalized_score(numerator, norm, dtype=accumulation_dtype)),
        provenance=provenance,
    )


def replay_normalized_cc(
    contributions: NormalizedCCContributions,
    *,
    canonical_identities=None,
) -> NormalizedCCReplayReport:
    """Replay one candidate through RECOVAR, RELION, and canonical reductions."""

    recovar = _reduction_result(
        contributions,
        recovar_logical_float32_reduce,
        accumulation_dtype=np.float32,
    )
    relion = _reduction_result(
        contributions,
        relion_256lane_float32_reduce,
        accumulation_dtype=np.float32,
    )
    canonical_f32 = _reduction_result(
        contributions,
        canonical_float32_reduce,
        accumulation_dtype=np.float32,
        identities=canonical_identities,
    )
    canonical_f64 = _reduction_result(
        contributions,
        canonical_float64_reduce,
        accumulation_dtype=np.float64,
        identities=canonical_identities,
    )
    same_float32_result = (
        recovar.numerator == relion.numerator and recovar.norm == relion.norm and recovar.score == relion.score
    )
    classification = "float32_reductions_agree" if same_float32_result else "float32_reduction_order_difference"
    return NormalizedCCReplayReport(
        classification=classification,
        recovar_logical_float32=recovar,
        relion_256lane_float32=relion,
        canonical_float32=canonical_f32,
        canonical_float64=canonical_f64,
    )


def _winner_ids(candidate_ids: tuple[int, ...], scores: Sequence[float]) -> tuple[int, ...]:
    values = np.asarray(scores)
    if values.shape != (len(candidate_ids),):
        raise ValueError("candidate score count does not match candidate ids")
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate scores must be finite")
    best = np.max(values)
    return tuple(candidate_ids[index] for index in np.flatnonzero(values == best))


def replay_normalized_cc_candidates(
    candidate_ids: Sequence[int],
    contributions: Sequence[NormalizedCCContributions],
    *,
    production_reducer: Literal["recovar_flat", "relion_coarse_128", "relion_fine_256"],
    canonical_identities: Sequence[np.ndarray | None] | None = None,
) -> NormalizedCCCandidateReplay:
    """Replay a candidate set without silently resolving exact score ties."""

    candidate_ids = tuple(int(value) for value in candidate_ids)
    contributions = tuple(contributions)
    if not candidate_ids or len(candidate_ids) != len(contributions):
        raise ValueError("candidate_ids and contributions must have one common non-zero length")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    if canonical_identities is None:
        canonical_identities = (None,) * len(contributions)
    else:
        canonical_identities = tuple(canonical_identities)
        if len(canonical_identities) != len(contributions):
            raise ValueError("canonical_identities must have one entry per candidate")

    reducer = {
        "recovar_flat": recovar_logical_float32_reduce,
        "relion_coarse_128": relion_128lane_float32_reduce,
        "relion_fine_256": relion_256lane_float32_reduce,
    }[production_reducer]
    production_scores = []
    canonical_float64_scores = []
    for operand_set, identities in zip(contributions, canonical_identities):
        production_scores.append(
            _reduction_result(operand_set, reducer, accumulation_dtype=np.float32).score
        )
        canonical_float64_scores.append(
            _reduction_result(
                operand_set,
                canonical_float64_reduce,
                accumulation_dtype=np.float64,
                identities=identities,
            ).score
        )
    origins = {operand_set.provenance.precision_origin for operand_set in contributions}
    if len(origins) != 1:
        raise ValueError("all candidates in one replay must have one precision origin")
    origin = origins.pop()
    return NormalizedCCCandidateReplay(
        candidate_ids=candidate_ids,
        production_scores=tuple(production_scores),
        canonical_float64_scores=tuple(canonical_float64_scores),
        production_winners=_winner_ids(candidate_ids, production_scores),
        canonical_float64_winners=_winner_ids(candidate_ids, canonical_float64_scores),
        production_reducer=production_reducer,
        precision_origin=origin,
    )


def classify_normalized_cc_candidate_replays(
    recovar: NormalizedCCCandidateReplay,
    relion: NormalizedCCCandidateReplay,
    *,
    geometry_equal: bool,
    genuine_float64_recovar: NormalizedCCCandidateReplay | None = None,
    genuine_float64_relion: NormalizedCCCandidateReplay | None = None,
) -> NormalizedCCCrossEngineClassification:
    """Classify the earliest supported cause of a winner disagreement.

    Captured float32 operands can distinguish reduction ordering from operand
    differences, but promoted float64 cannot diagnose precision already lost
    upstream.  Therefore the stronger ``precision`` classification is emitted
    only when both optional replays were genuinely recomputed from high-
    precision operands.
    """

    if recovar.candidate_ids != relion.candidate_ids or not geometry_equal:
        classification = "geometry"
    elif recovar.production_winners == relion.production_winners:
        classification = "production_agreement"
    elif recovar.canonical_float64_winners == relion.canonical_float64_winners:
        classification = "reduction_order_or_accumulation_precision"
    elif genuine_float64_recovar is None or genuine_float64_relion is None:
        classification = "operand_generation_or_upstream_precision_unresolved"
    else:
        if (
            genuine_float64_recovar.precision_origin != "recomputed_high_precision"
            or genuine_float64_relion.precision_origin != "recomputed_high_precision"
        ):
            raise ValueError("genuine float64 classifications require recomputed_high_precision operands")
        if genuine_float64_recovar.candidate_ids != recovar.candidate_ids:
            raise ValueError("RECOVAR genuine-float64 candidate identities differ")
        if genuine_float64_relion.candidate_ids != relion.candidate_ids:
            raise ValueError("RELION genuine-float64 candidate identities differ")
        if genuine_float64_recovar.canonical_float64_winners == genuine_float64_relion.canonical_float64_winners:
            classification = "precision"
        else:
            classification = "operand_generation"
    return NormalizedCCCrossEngineClassification(
        classification=classification,
        recovar=recovar,
        relion=relion,
        geometry_equal=bool(geometry_equal),
        genuine_float64_recovar=genuine_float64_recovar,
        genuine_float64_relion=genuine_float64_relion,
    )
