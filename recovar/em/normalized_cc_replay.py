"""Deterministic normalized-CC reduction replay diagnostics.

This module operates on frozen per-pixel operands.  It does not replace the
production scorer and deliberately has no JAX dependency.  In particular, a
float64 replay of captured float32 contributions is labelled as *promoted*:
it cannot recover precision lost while producing those contributions.
``normalized_cc_pixel_contributions`` evaluates the logical expression; an
exact device replay must supply contributions captured after device operand
formation so compiler contraction and texture/interpolation effects are not
silently attributed to reduction order.

The RELION reducer mirrors ``cuda_kernel_diff2_CC_fine`` in
``src/acc/cuda/cuda_kernels/diff2.cuh`` for the SPA ``REF3D`` path: 256 lanes
first accumulate pixels ``lane + 256 * pass`` and are then reduced by the
shared-memory 128, 64, ..., 1 tree.  The input must therefore retain the full
packed-grid pixel order, including zero-contribution pixels outside support.

Map-level effects of a diagnosed score difference must be assessed with
shellwise FSC/FSC-AUC, not map correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


def _relion_tree_float32_reduce(values, *, lane_count: int) -> np.float32:
    """Replay one power-of-two RELION CUDA lane tree."""

    values = _as_1d(values, dtype=np.float32, name="values")
    if lane_count <= 0 or lane_count & (lane_count - 1):
        raise ValueError(f"lane_count must be a positive power of two, got {lane_count}")
    lanes = np.zeros(lane_count, dtype=np.float32)
    for pixel, value in enumerate(values):
        lane = pixel % lane_count
        lanes[lane] = np.float32(lanes[lane] + value)
    width = lane_count // 2
    while width:
        lanes[:width] = np.add(lanes[:width], lanes[width : 2 * width], dtype=np.float32)
        width //= 2
    return lanes[0]


def relion_coarse_128lane_float32_reduce(values) -> np.float32:
    """Replay RELION SPA coarse normalized-CC CUDA's 128-lane tree.

    The coarse first-iteration kernel uses ``D2C_BLOCK_SIZE_REF3D=128``.
    Input must retain its complete packed current-image order so pixel-to-lane
    ownership is not changed by compaction or sorting.
    """

    return _relion_tree_float32_reduce(values, lane_count=RELION_COARSE_REDUCTION_LANES)


def relion_256lane_float32_reduce(values) -> np.float32:
    """Replay RELION SPA fine-score CUDA's exact 256-lane float32 tree.

    ``values`` must include every full packed-grid position.  Compacting away
    zero positions changes lane ownership and therefore changes this result.
    """

    return _relion_tree_float32_reduce(values, lane_count=RELION_FINE_REDUCTION_LANES)


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
