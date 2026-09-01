"""GPU-aware batch sizing + raw-image host cache for the EM iteration loop.

``_estimate_relion_em_batch_sizes`` chooses microbatch sizes from pose-grid,
image, class, and GPU size so the dense RELION loop's transient memory
drivers (score tensor + projection tile + translation-expanded half-images)
stay within available memory. ``maybe_cache_raw_image_loaders`` keeps
file-backed raw particles in host memory across passes.

Extracted from ``iteration_loop.py`` so the master loop stays focused on
EM dispatch.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Integral

import numpy as np

logger = logging.getLogger(__name__)


# Constants mirror those in iteration_loop.py so monkeypatches at either
# module level continue to bind correctly.
RELION_SCORE_TENSOR_FLOAT_BUDGET = 200_000_000
_RELION_EM_BATCH_DEFAULT_GPU_GB = 80.0
_RELION_EM_BATCH_USABLE_FRACTION = 0.65
_RELION_EM_BATCH_PROJECTION_FRACTION = 0.20
_RELION_EM_BATCH_SCORE_FRACTION = 0.20
_RELION_EM_BATCH_MAX_PROJECTION_GB = 10.0
_RELION_EM_BATCH_MIN_PROJECTION_GB = 0.5
_RELION_EM_BATCH_PROJECTION_LIVE_FACTOR = 1.5
_RELION_EM_BATCH_SCORE_MATMUL_LIVE_FACTOR = 7.0
_RELION_EM_BATCH_POSE_PIXEL_LIVE_FACTOR = 1.25
_RELION_EM_BATCH_POSE_PIXEL_FRACTION = 0.035
_RELION_EM_BATCH_POSE_PIXEL_WINDOW_FRACTION = 0.30
_RELION_EM_BATCH_ACTIVE_SCORE_TILE_LIVE_FACTOR = 4.0
_RELION_EM_BATCH_ACTIVE_SCORE_TILE_FRACTION = 0.10
_RELION_EM_BATCH_TRANSLATION_TILE_FRACTION = 0.35
_RELION_EM_BATCH_RUNTIME_TRANSLATION_TILE_FRACTION = 0.17
_RELION_EM_BATCH_MAX_TRANSLATION_TILE_GB = 14.0
_RELION_EM_BATCH_MIN_TRANSLATION_TILE_GB = 0.5
_RELION_EM_BATCH_RUNTIME_FREE_FRACTION = 0.80
_RELION_EM_BATCH_PROJECTION_FRACTION_ENV = "RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION"

_EM_RAW_IMAGE_CACHE_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE"
_EM_RAW_IMAGE_CACHE_MAX_GB_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB"
_EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB = 16.0


@dataclass(frozen=True)
class _RelionEMBatchPlan:
    image_batch_size: int
    rotation_block_size: int
    score_float_budget: int
    projection_budget_gb: float
    translation_tile_budget_gb: float
    persistent_estimate_gb: float
    usable_estimate_gb: float
    gpu_used_estimate_gb: float
    runtime_free_estimate_gb: float
    projection_block_gb: float
    active_score_tile_budget_gb: float
    active_score_tile_gb: float
    pose_pixel_tile_gb: float
    translation_tile_gb: float
    score_pixel_count: int


@dataclass(frozen=True)
class _ConsecutivePaddedBatch:
    """One physical-order batch and its static padded shape."""

    item_indices: np.ndarray
    padded_size: int
    padded_item_capacity: int


@dataclass(frozen=True)
class _FixedCapacityLocalCall:
    """One already-planned local call in its authoritative chronology.

    ``row_counts`` contains the number of real candidate rows for each image;
    ``radix_bucket`` is the existing rectangular rotation capacity.  This
    descriptor deliberately does not regroup images or choose a new radix.
    ``image_capacity`` preserves an authoritative call's existing padded image
    shape when provided; synthetic planner probes may leave it unset and use
    the smallest fitting palette entry.
    """

    image_indices: np.ndarray
    row_counts: np.ndarray
    radix_bucket: int
    image_capacity: int | None = None


@dataclass(frozen=True)
class _FixedCapacityPhysicalOrder:
    """Independent immutable seal for one authoritative physical image order."""

    image_indices: np.ndarray

    def __post_init__(self):
        raw_indices = np.asarray(self.image_indices)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("fixed-capacity physical order must be a one-dimensional integer array")
        if raw_indices.size == 0:
            raise ValueError("fixed-capacity physical order cannot be empty")
        sealed_indices = raw_indices.astype(np.int64, copy=True)
        if np.any(sealed_indices < 0):
            raise ValueError("fixed-capacity physical order must contain non-negative image IDs")
        if np.unique(sealed_indices).size != sealed_indices.size:
            raise ValueError("fixed-capacity physical order must not contain duplicate image IDs")
        sealed_indices.setflags(write=False)
        object.__setattr__(self, "image_indices", sealed_indices)


@dataclass(frozen=True, eq=False)
class _FixedCapacityLocalGenerationToken:
    """Opaque identity shared only by components from one planned generation."""


@dataclass(frozen=True)
class _FixedCapacityWholeLocalPlan:
    """Fixed-shape host descriptors for a future whole-local executor.

    The fixed array capacities are static executor shapes.  The valid counts,
    row offsets, call chronology, radix buckets, and logical Fourier cutoff are
    runtime values.  Candidate payloads are packed without padded radix rows;
    an executor can materialize its existing radix-specific scratch tile for
    each call without changing the authoritative call or particle order.
    """

    physical_image_capacity: int
    physical_row_capacity: int
    physical_call_capacity: int
    logical_cutoff_capacity: int
    valid_image_count: int
    valid_row_count: int
    valid_call_count: int
    image_indices: np.ndarray
    row_offsets: np.ndarray
    call_valid_mask: np.ndarray
    call_image_offsets: np.ndarray
    call_row_offsets: np.ndarray
    call_valid_images: np.ndarray
    call_valid_rows: np.ndarray
    call_image_capacities: np.ndarray
    call_radix_buckets: np.ndarray
    logical_cutoff: np.ndarray
    generation_token: _FixedCapacityLocalGenerationToken = field(init=False, repr=False, compare=False)
    descriptor_fingerprint: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "generation_token", _FixedCapacityLocalGenerationToken())
        scalar_fields = (
            "physical_image_capacity",
            "physical_row_capacity",
            "physical_call_capacity",
            "logical_cutoff_capacity",
            "valid_image_count",
            "valid_row_count",
            "valid_call_count",
        )
        for field_name in scalar_fields:
            raw_value = getattr(self, field_name)
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, Integral):
                raise ValueError(f"fixed-capacity plan {field_name} must be an integer")
            object.__setattr__(self, field_name, int(raw_value))

        if (
            min(
                self.physical_image_capacity,
                self.physical_row_capacity,
                self.physical_call_capacity,
                self.logical_cutoff_capacity,
            )
            <= 0
        ):
            raise ValueError("fixed-capacity plan physical capacities must be positive")
        if min(self.valid_image_count, self.valid_row_count, self.valid_call_count) <= 0:
            raise ValueError("fixed-capacity plan valid counts must be positive")
        if self.valid_image_count > self.physical_image_capacity:
            raise ValueError("fixed-capacity plan image capacity overflow")
        if self.valid_row_count > self.physical_row_capacity:
            raise ValueError("fixed-capacity plan candidate-row capacity overflow")
        if self.valid_call_count > self.physical_call_capacity:
            raise ValueError("fixed-capacity plan call capacity overflow")

        array_specs = {
            "image_indices": (np.dtype(np.int64), (self.physical_image_capacity,)),
            "row_offsets": (np.dtype(np.int64), (self.physical_image_capacity + 1,)),
            "call_valid_mask": (np.dtype(np.bool_), (self.physical_call_capacity,)),
            "call_image_offsets": (np.dtype(np.int32), (self.physical_call_capacity,)),
            "call_row_offsets": (np.dtype(np.int64), (self.physical_call_capacity,)),
            "call_valid_images": (np.dtype(np.int32), (self.physical_call_capacity,)),
            "call_valid_rows": (np.dtype(np.int64), (self.physical_call_capacity,)),
            "call_image_capacities": (np.dtype(np.int32), (self.physical_call_capacity,)),
            "call_radix_buckets": (np.dtype(np.int32), (self.physical_call_capacity,)),
            "logical_cutoff": (np.dtype(np.int32), ()),
        }
        for field_name, (expected_dtype, expected_shape) in array_specs.items():
            raw_value = np.asarray(getattr(self, field_name))
            if raw_value.dtype != expected_dtype:
                raise ValueError(
                    f"fixed-capacity plan {field_name} must have canonical dtype {expected_dtype}; "
                    f"got {raw_value.dtype}",
                )
            if raw_value.shape != expected_shape:
                raise ValueError(
                    f"fixed-capacity plan {field_name} must have canonical shape {expected_shape}; "
                    f"got {raw_value.shape}",
                )
            sealed_value = np.array(raw_value, copy=True, order="C")
            sealed_value.setflags(write=False)
            object.__setattr__(self, field_name, sealed_value)

        logical_cutoff = int(self.logical_cutoff)
        if logical_cutoff <= 0 or logical_cutoff > self.logical_cutoff_capacity:
            raise ValueError("fixed-capacity plan logical cutoff must fit its capacity")
        object.__setattr__(
            self,
            "descriptor_fingerprint",
            _fixed_capacity_plan_descriptor_fingerprint(self),
        )


def _fixed_capacity_plan_descriptor_fingerprint(plan: _FixedCapacityWholeLocalPlan) -> str:
    """Return a deterministic digest of every fixed-capacity plan descriptor."""

    if not isinstance(plan, _FixedCapacityWholeLocalPlan):
        raise ValueError("fixed-capacity descriptor fingerprint requires a fixed-capacity local plan")
    digest = sha256(b"recovar.fixed-capacity-local-plan.v1\0")
    for field_name in (
        "physical_image_capacity",
        "physical_row_capacity",
        "physical_call_capacity",
        "logical_cutoff_capacity",
        "valid_image_count",
        "valid_row_count",
        "valid_call_count",
    ):
        digest.update(field_name.encode("ascii") + b"\0")
        digest.update(np.asarray(int(getattr(plan, field_name)), dtype="<i8").tobytes())
    for field_name in (
        "image_indices",
        "row_offsets",
        "call_valid_mask",
        "call_image_offsets",
        "call_row_offsets",
        "call_valid_images",
        "call_valid_rows",
        "call_image_capacities",
        "call_radix_buckets",
        "logical_cutoff",
    ):
        value = np.asarray(getattr(plan, field_name))
        digest.update(field_name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _seal_fixed_capacity_physical_order(image_indices) -> _FixedCapacityPhysicalOrder:
    """Snapshot and validate physical image IDs before bucket conversion."""

    return _FixedCapacityPhysicalOrder(image_indices=image_indices)


def _normalized_fixed_capacity_palette(
    image_capacity_palette: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    palette: dict[int, tuple[int, ...]] = {}
    for raw_radix, raw_capacities in image_capacity_palette.items():
        radix = int(raw_radix)
        if radix <= 0:
            raise ValueError("fixed-capacity radix buckets must be positive")
        capacities = tuple(sorted({int(value) for value in raw_capacities}))
        if not capacities or capacities[0] <= 0:
            raise ValueError(
                "each fixed-capacity radix bucket needs at least one positive image capacity",
            )
        palette[radix] = capacities
    return palette


def _plan_fixed_capacity_whole_local(
    calls: Sequence[_FixedCapacityLocalCall],
    *,
    expected_image_order,
    physical_image_capacity: int,
    physical_row_capacity: int,
    physical_call_capacity: int,
    image_capacity_palette: Mapping[int, Sequence[int]],
    logical_cutoff: int,
    logical_cutoff_capacity: int,
    enabled: bool = False,
) -> _FixedCapacityWholeLocalPlan | None:
    """Pack existing local calls into one fixed-capacity execution program.

    This is a host-only seam for a default-off whole-local executor.  It never
    creates, splits, coalesces, or reorders calls.  The caller must provide the
    sealed physical image order and a stable image-capacity palette for each
    existing radix bucket.  Calls converted from authoritative local buckets
    preserve their existing image capacity exactly; synthetic probes without
    one map to the smallest fitting palette capacity.  Real image/candidate
    counts remain runtime descriptors in either case.

    Returning ``None`` while disabled makes the seam inert for both mature EM
    and InitialModel.  Once enabled, every unsupported capacity or chronology
    fails closed instead of falling back to a different grouping policy.
    """

    if not enabled:
        return None

    physical_image_capacity = int(physical_image_capacity)
    physical_row_capacity = int(physical_row_capacity)
    physical_call_capacity = int(physical_call_capacity)
    logical_cutoff = int(logical_cutoff)
    logical_cutoff_capacity = int(logical_cutoff_capacity)
    if (
        min(
            physical_image_capacity,
            physical_row_capacity,
            physical_call_capacity,
            logical_cutoff_capacity,
        )
        <= 0
    ):
        raise ValueError("fixed-capacity executor capacities must be positive")
    if logical_cutoff <= 0 or logical_cutoff > logical_cutoff_capacity:
        raise ValueError(
            "logical_cutoff must be positive and no larger than logical_cutoff_capacity",
        )

    calls = tuple(calls)
    if not calls:
        raise ValueError("fixed-capacity call program cannot be empty")
    authoritative_calls = any(call.image_capacity is not None for call in calls)
    if isinstance(expected_image_order, _FixedCapacityPhysicalOrder):
        expected_image_order_array = expected_image_order.image_indices
    else:
        if authoritative_calls:
            raise ValueError(
                "fixed-capacity authoritative calls require an independently sealed expected physical order",
            )
        expected_image_order_array = expected_image_order
    if len(calls) > physical_call_capacity:
        raise ValueError(
            f"fixed-capacity call program overflow: valid={len(calls)}, capacity={physical_call_capacity}",
        )
    palette = _normalized_fixed_capacity_palette(image_capacity_palette)

    image_parts: list[np.ndarray] = []
    row_count_parts: list[np.ndarray] = []
    call_image_offsets = np.full(
        physical_call_capacity,
        -1,
        dtype=np.int32,
    )
    call_row_offsets = np.full(
        physical_call_capacity,
        -1,
        dtype=np.int64,
    )
    call_valid_images = np.zeros(physical_call_capacity, dtype=np.int32)
    call_valid_rows = np.zeros(physical_call_capacity, dtype=np.int64)
    call_image_capacities = np.zeros(physical_call_capacity, dtype=np.int32)
    call_radix_buckets = np.zeros(physical_call_capacity, dtype=np.int32)
    call_valid_mask = np.zeros(physical_call_capacity, dtype=bool)

    running_images = 0
    running_rows = 0
    for call_index, call in enumerate(calls):
        image_indices = np.asarray(call.image_indices, dtype=np.int64).reshape(-1)
        row_counts = np.asarray(call.row_counts, dtype=np.int64).reshape(-1)
        radix_bucket = int(call.radix_bucket)
        if image_indices.size == 0:
            raise ValueError("fixed-capacity local calls cannot be empty")
        if row_counts.shape != image_indices.shape:
            raise ValueError(
                f"fixed-capacity call row_counts must match image_indices: {row_counts.shape} vs {image_indices.shape}",
            )
        if np.any(image_indices < 0):
            raise ValueError("fixed-capacity image indices must be non-negative")
        if radix_bucket not in palette:
            raise ValueError(
                f"fixed-capacity palette has no entry for radix bucket {radix_bucket}",
            )
        if np.any(row_counts <= 0) or np.any(row_counts > radix_bucket):
            raise ValueError(
                f"fixed-capacity candidate row counts must be positive and no larger than radix bucket {radix_bucket}",
            )

        valid_images = int(image_indices.size)
        if call.image_capacity is None:
            fitting_capacities = [capacity for capacity in palette[radix_bucket] if capacity >= valid_images]
            if not fitting_capacities:
                raise ValueError(
                    "fixed-capacity image palette overflow for radix bucket "
                    f"{radix_bucket}: valid={valid_images}, "
                    f"capacities={palette[radix_bucket]}",
                )
            image_capacity = fitting_capacities[0]
        else:
            image_capacity = int(call.image_capacity)
            if image_capacity < valid_images:
                raise ValueError(
                    "fixed-capacity preserved image capacity is smaller than its valid image count: "
                    f"capacity={image_capacity}, valid={valid_images}",
                )
            if image_capacity not in palette[radix_bucket]:
                raise ValueError(
                    "fixed-capacity palette does not contain the preserved image capacity for radix bucket "
                    f"{radix_bucket}: preserved={image_capacity}, capacities={palette[radix_bucket]}",
                )
        valid_rows = int(np.sum(row_counts, dtype=np.int64))
        call_valid_mask[call_index] = True
        call_image_offsets[call_index] = running_images
        call_row_offsets[call_index] = running_rows
        call_valid_images[call_index] = valid_images
        call_valid_rows[call_index] = valid_rows
        call_image_capacities[call_index] = image_capacity
        call_radix_buckets[call_index] = radix_bucket
        image_parts.append(image_indices)
        row_count_parts.append(row_counts)
        running_images += valid_images
        running_rows += valid_rows

    chronological_image_indices = np.concatenate(image_parts) if image_parts else np.zeros(0, dtype=np.int64)
    chronological_row_counts = np.concatenate(row_count_parts) if row_count_parts else np.zeros(0, dtype=np.int64)
    expected_image_order_array = np.asarray(expected_image_order_array, dtype=np.int64).reshape(-1)
    if not np.array_equal(chronological_image_indices, expected_image_order_array):
        raise ValueError(
            "fixed-capacity call chronology does not match expected physical image order",
        )
    if np.unique(chronological_image_indices).size != chronological_image_indices.size:
        raise ValueError("fixed-capacity physical image order must not contain duplicates")
    if running_images > physical_image_capacity:
        raise ValueError(
            f"fixed-capacity image storage overflow: valid={running_images}, capacity={physical_image_capacity}",
        )
    if running_rows > physical_row_capacity:
        raise ValueError(
            f"fixed-capacity candidate-row storage overflow: valid={running_rows}, capacity={physical_row_capacity}",
        )

    image_indices = np.full(physical_image_capacity, -1, dtype=np.int64)
    image_indices[:running_images] = chronological_image_indices
    row_offsets = np.full(
        physical_image_capacity + 1,
        running_rows,
        dtype=np.int64,
    )
    row_offsets[0] = 0
    if chronological_row_counts.size:
        row_offsets[1 : running_images + 1] = np.cumsum(
            chronological_row_counts,
            dtype=np.int64,
        )
    if len(calls) < physical_call_capacity:
        call_image_offsets[len(calls) :] = running_images
        call_row_offsets[len(calls) :] = running_rows

    return _FixedCapacityWholeLocalPlan(
        physical_image_capacity=physical_image_capacity,
        physical_row_capacity=physical_row_capacity,
        physical_call_capacity=physical_call_capacity,
        logical_cutoff_capacity=logical_cutoff_capacity,
        valid_image_count=running_images,
        valid_row_count=running_rows,
        valid_call_count=len(calls),
        image_indices=image_indices,
        row_offsets=row_offsets,
        call_valid_mask=call_valid_mask,
        call_image_offsets=call_image_offsets,
        call_row_offsets=call_row_offsets,
        call_valid_images=call_valid_images,
        call_valid_rows=call_valid_rows,
        call_image_capacities=call_image_capacities,
        call_radix_buckets=call_radix_buckets,
        logical_cutoff=np.asarray(logical_cutoff, dtype=np.int32),
    )


def _validate_fixed_capacity_call_values(
    plan: _FixedCapacityWholeLocalPlan,
    values_by_call,
    *,
    candidate_rows: bool,
) -> tuple[list[np.ndarray], tuple[int, ...], np.dtype]:
    values = [np.asarray(value) for value in values_by_call]
    if len(values) != plan.valid_call_count:
        raise ValueError(
            "fixed-capacity packed values must contain one array per valid call: "
            f"got {len(values)}, expected {plan.valid_call_count}",
        )
    if not values:
        raise ValueError("fixed-capacity packing requires at least one valid call")

    first_trailing_shape = values[0].shape[2 if candidate_rows else 1 :]
    first_dtype = values[0].dtype
    for call_index, value in enumerate(values):
        valid_images = int(plan.call_valid_images[call_index])
        expected_prefix = (
            (valid_images, int(plan.call_radix_buckets[call_index])) if candidate_rows else (valid_images,)
        )
        if value.shape[: len(expected_prefix)] != expected_prefix:
            raise ValueError(
                "fixed-capacity packed call has an unexpected leading shape: "
                f"got {value.shape}, expected prefix {expected_prefix}",
            )
        trailing_shape = value.shape[len(expected_prefix) :]
        if trailing_shape != first_trailing_shape or value.dtype != first_dtype:
            raise ValueError(
                "fixed-capacity packed call values must share dtype and trailing shape",
            )
    return values, first_trailing_shape, first_dtype


def _pack_fixed_capacity_local_images(
    plan: _FixedCapacityWholeLocalPlan,
    values_by_call,
    *,
    fill_value=0,
) -> np.ndarray:
    """Pack per-image call operands into the plan's stable image axis."""

    values, trailing_shape, dtype = _validate_fixed_capacity_call_values(
        plan,
        values_by_call,
        candidate_rows=False,
    )
    packed = np.full(
        (plan.physical_image_capacity,) + trailing_shape,
        fill_value,
        dtype=dtype,
    )
    for call_index, value in enumerate(values):
        start = int(plan.call_image_offsets[call_index])
        stop = start + int(plan.call_valid_images[call_index])
        packed[start:stop] = value
    return packed


def _pack_fixed_capacity_local_candidate_rows(
    plan: _FixedCapacityWholeLocalPlan,
    values_by_call,
    *,
    fill_value=0,
) -> np.ndarray:
    """Pack real candidate rows without copying rectangular radix padding."""

    values, trailing_shape, dtype = _validate_fixed_capacity_call_values(
        plan,
        values_by_call,
        candidate_rows=True,
    )
    packed = np.full(
        (plan.physical_row_capacity,) + trailing_shape,
        fill_value,
        dtype=dtype,
    )
    for call_index, value in enumerate(values):
        image_start = int(plan.call_image_offsets[call_index])
        valid_images = int(plan.call_valid_images[call_index])
        for local_image_index in range(valid_images):
            global_image_index = image_start + local_image_index
            row_start = int(plan.row_offsets[global_image_index])
            row_stop = int(plan.row_offsets[global_image_index + 1])
            packed[row_start:row_stop] = value[
                local_image_index,
                : row_stop - row_start,
            ]
    return packed


def _plan_consecutive_padded_batches(
    padded_sizes,
    *,
    processing_order=None,
    target_items_per_batch: int,
    max_items_per_batch: int,
    max_padded_values_per_batch: int,
    values_per_padded_size: int = 1,
    item_alignment: int = 1,
) -> list[_ConsecutivePaddedBatch]:
    """Plan bounded mixed-size batches without changing physical item order.

    Each batch uses the largest padded size among its consecutive items.  The
    planner greedily admits aligned item groups until either the target item
    count or padded-work cap would be exceeded.  ``padded_item_capacity`` is
    the stable image-axis shape for a tail batch with the same padded size.

    This is the shared shape policy used around the mature sparse EM pass and
    InitialModel's exact-local pass.  It changes padding and executable reuse,
    never candidate membership or processing order.
    """

    padded_sizes = np.asarray(padded_sizes, dtype=np.int64).reshape(-1)
    n_items = int(padded_sizes.size)
    if np.any(padded_sizes <= 0):
        raise ValueError("padded_sizes must contain only positive values")
    if processing_order is None:
        processing_order = np.arange(n_items, dtype=np.int64)
    else:
        processing_order = np.asarray(processing_order, dtype=np.int64).reshape(-1)
    if processing_order.shape != (n_items,):
        raise ValueError(
            f"processing_order must have shape ({n_items},), got {processing_order.shape}",
        )
    if not np.array_equal(np.sort(processing_order), np.arange(n_items, dtype=np.int64)):
        raise ValueError("processing_order must be a permutation of item indices")

    target_items_per_batch = int(target_items_per_batch)
    max_items_per_batch = int(max_items_per_batch)
    max_padded_values_per_batch = int(max_padded_values_per_batch)
    values_per_padded_size = int(values_per_padded_size)
    item_alignment = int(item_alignment)
    if target_items_per_batch <= 0 or max_items_per_batch <= 0:
        raise ValueError("batch item limits must be positive")
    if max_padded_values_per_batch <= 0 or values_per_padded_size <= 0:
        raise ValueError("padded-work limits must be positive")
    if item_alignment <= 0:
        raise ValueError("item_alignment must be positive")
    if n_items == 0:
        return []

    item_limit = min(target_items_per_batch, max_items_per_batch)
    if item_alignment > 1 and n_items > item_limit and item_limit < item_alignment:
        raise ValueError(
            "batch item limits are too small to preserve the requested item alignment: "
            f"limit={item_limit}, alignment={item_alignment}",
        )
    effective_alignment = item_alignment
    plans: list[_ConsecutivePaddedBatch] = []
    start = 0
    while start < n_items:
        stop = start
        batch_padded_size = 0
        while stop < n_items:
            next_stop = min(n_items, stop + effective_alignment)
            next_count = next_stop - start
            next_padded_size = max(
                batch_padded_size,
                int(np.max(padded_sizes[processing_order[stop:next_stop]], initial=1)),
            )
            next_work = next_count * next_padded_size * values_per_padded_size
            if stop == start and next_work > max_padded_values_per_batch and effective_alignment > 1:
                raise ValueError(
                    "padded-work cap is too small for one aligned item group: "
                    f"required={next_work}, cap={max_padded_values_per_batch}, "
                    f"alignment={effective_alignment}",
                )
            if stop > start and (
                next_count > item_limit
                or next_work > max_padded_values_per_batch
            ):
                break
            stop = next_stop
            batch_padded_size = next_padded_size
            if next_count >= item_limit:
                break

        item_indices = np.asarray(processing_order[start:stop], dtype=np.int64)
        capacity_by_work = max(
            1,
            max_padded_values_per_batch
            // max(1, batch_padded_size * values_per_padded_size),
        )
        padded_item_capacity = min(item_limit, capacity_by_work)
        if padded_item_capacity >= effective_alignment:
            padded_item_capacity = (
                padded_item_capacity // effective_alignment
            ) * effective_alignment
        padded_item_capacity = max(int(item_indices.size), padded_item_capacity)
        plans.append(
            _ConsecutivePaddedBatch(
                item_indices=item_indices,
                padded_size=int(batch_padded_size),
                padded_item_capacity=int(padded_item_capacity),
            )
        )
        start = stop

    return plans


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite float, got {raw!r}") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite float, got {raw!r}")
    return float(value)


def _active_half_spectrum_pixels(image_shape, current_size: int | None) -> int:
    full_half_pixels = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    if current_size is None:
        return full_half_pixels
    radius = max(0, int(current_size) // 2)
    if radius <= 0:
        return 1
    row_freq = np.minimum(np.arange(int(image_shape[0])), int(image_shape[0]) - np.arange(int(image_shape[0])))
    col_freq = np.arange(int(image_shape[1]) // 2 + 1)
    keep = row_freq[:, None] * row_freq[:, None] + col_freq[None, :] * col_freq[None, :] <= radius * radius
    return max(1, min(full_half_pixels, int(np.count_nonzero(keep))))


def _estimate_relion_em_batch_sizes(
    *,
    requested_image_batch_size: int,
    requested_rotation_block_size: int,
    n_rot: int,
    n_trans: int,
    image_shape,
    volume_shape,
    padding_factor: int,
    n_classes: int = 1,
    gpu_memory_gb: float | None = None,
    current_size: int | None = None,
) -> _RelionEMBatchPlan:
    """Choose EM microbatch sizes from pose-grid, image, class, and GPU size."""
    # Indirection through iteration_loop module so test monkeypatches on
    # ``iteration_loop.utils.get_gpu_memory_total`` / ``get_gpu_memory_used``
    # win at this call site too.
    from recovar.em.dense_single_volume import iteration_loop as _il

    requested_image_batch_size = max(1, _safe_int(requested_image_batch_size, 1))
    requested_rotation_block_size = max(1, _safe_int(requested_rotation_block_size, 1))
    n_rot = max(1, _safe_int(n_rot, 1))
    n_trans = max(1, _safe_int(n_trans, 1))
    n_classes = max(1, _safe_int(n_classes, 1))
    padding_factor = max(1, _safe_int(padding_factor, 1))
    image_shape = tuple(int(s) for s in image_shape)
    volume_shape = tuple(int(s) for s in volume_shape)

    gpu_used_gb = 0.0
    if gpu_memory_gb is None:
        try:
            gpu_memory_gb = float(_il.utils.get_gpu_memory_total())
        except Exception:
            gpu_memory_gb = _RELION_EM_BATCH_DEFAULT_GPU_GB
        try:
            gpu_used_gb = float(_il.utils.get_gpu_memory_used())
        except Exception:
            gpu_used_gb = 0.0
    if not np.isfinite(gpu_memory_gb) or gpu_memory_gb <= 0:
        gpu_memory_gb = _RELION_EM_BATCH_DEFAULT_GPU_GB
    if not np.isfinite(gpu_used_gb) or gpu_used_gb < 0:
        gpu_used_gb = 0.0
    gpu_used_gb = min(gpu_used_gb, max(0.0, gpu_memory_gb - 1.0))

    padded_volume_voxels = float(np.prod([d * padding_factor for d in volume_shape]))
    native_volume_voxels = float(np.prod(volume_shape))
    persistent_bytes = (
        2.0 * padded_volume_voxels * np.dtype(np.complex64).itemsize * n_classes
        + 4.0 * native_volume_voxels * np.dtype(np.complex64).itemsize * n_classes
    )
    persistent_gb = persistent_bytes / 1e9
    runtime_free_gb = max(1.0, gpu_memory_gb - gpu_used_gb)
    usable_from_total_gb = max(1.0, gpu_memory_gb * _RELION_EM_BATCH_USABLE_FRACTION - persistent_gb)
    usable_from_runtime_gb = max(1.0, runtime_free_gb * _RELION_EM_BATCH_RUNTIME_FREE_FRACTION)
    usable_gb = min(usable_from_total_gb, usable_from_runtime_gb)

    score_float_budget = int(
        max(
            1_000_000,
            min(
                RELION_SCORE_TENSOR_FLOAT_BUDGET,
                usable_gb * _RELION_EM_BATCH_SCORE_FRACTION * 1e9 / np.dtype(np.float32).itemsize,
            ),
        )
    )
    projection_fraction = _positive_env_float(
        _RELION_EM_BATCH_PROJECTION_FRACTION_ENV,
        _RELION_EM_BATCH_PROJECTION_FRACTION,
    )
    projection_budget_gb = max(
        _RELION_EM_BATCH_MIN_PROJECTION_GB,
        min(_RELION_EM_BATCH_MAX_PROJECTION_GB, usable_gb * projection_fraction),
    )
    translation_tile_budget_gb = max(
        _RELION_EM_BATCH_MIN_TRANSLATION_TILE_GB,
        min(
            _RELION_EM_BATCH_MAX_TRANSLATION_TILE_GB,
            usable_gb * _RELION_EM_BATCH_TRANSLATION_TILE_FRACTION,
            usable_from_runtime_gb * _RELION_EM_BATCH_RUNTIME_TRANSLATION_TILE_FRACTION,
        ),
    )

    score_image_cap = max(1, score_float_budget // max(n_rot * n_trans * n_classes, 1))
    full_half_pixels = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    score_half_pixels = _active_half_spectrum_pixels(image_shape, current_size)
    active_score_tile_budget_gb = max(
        _RELION_EM_BATCH_MIN_PROJECTION_GB,
        min(
            projection_budget_gb,
            gpu_memory_gb * _RELION_EM_BATCH_ACTIVE_SCORE_TILE_FRACTION,
            usable_from_runtime_gb * _RELION_EM_BATCH_ACTIVE_SCORE_TILE_FRACTION,
        ),
    )
    active_score_bytes_per_image = max(
        1,
        int(
            np.ceil(
                n_trans
                * score_half_pixels
                * np.dtype(np.complex128).itemsize
                * n_classes
                * _RELION_EM_BATCH_ACTIVE_SCORE_TILE_LIVE_FACTOR,
            )
        ),
    )
    active_score_image_cap = max(1, int(active_score_tile_budget_gb * 1e9 // active_score_bytes_per_image))
    translation_bytes_per_image = max(
        1,
        2 * n_trans * full_half_pixels * np.dtype(np.complex64).itemsize * n_classes,
    )
    translation_image_cap = max(1, int(translation_tile_budget_gb * 1e9 // translation_bytes_per_image))
    image_batch = min(requested_image_batch_size, score_image_cap, translation_image_cap, active_score_image_cap)

    score_rotation_cap = max(1, score_float_budget // max(image_batch * n_trans * n_classes, 1))
    projection_half_pixels = full_half_pixels if current_size is None else score_half_pixels
    projection_bytes_per_rotation = max(
        1,
        int(
            np.ceil(
                projection_half_pixels
                * np.dtype(np.complex64).itemsize
                * n_classes
                * _RELION_EM_BATCH_PROJECTION_LIVE_FACTOR,
            ),
        ),
    )
    projection_rotation_cap = max(1, int(projection_budget_gb * 1e9 // projection_bytes_per_rotation))
    score_matmul_bytes_per_rotation = max(
        1,
        int(
            np.ceil(
                score_half_pixels
                * np.dtype(np.complex64).itemsize
                * n_classes
                * _RELION_EM_BATCH_SCORE_MATMUL_LIVE_FACTOR,
            )
        ),
    )
    score_matmul_rotation_cap = max(1, int(projection_budget_gb * 1e9 // score_matmul_bytes_per_rotation))
    active_window_fraction = score_half_pixels / max(1, full_half_pixels)
    if active_window_fraction > _RELION_EM_BATCH_POSE_PIXEL_WINDOW_FRACTION:
        pose_pixel_budget_gb = max(
            _RELION_EM_BATCH_MIN_PROJECTION_GB,
            min(
                projection_budget_gb,
                gpu_memory_gb * _RELION_EM_BATCH_POSE_PIXEL_FRACTION,
                usable_from_runtime_gb * _RELION_EM_BATCH_RUNTIME_FREE_FRACTION,
            ),
        )
    else:
        pose_pixel_budget_gb = projection_budget_gb
    # Dense scoring can materialize a rotation x translation x active-pixel
    # complex tile. JAX runs with x64 enabled, so budget this as complex128.
    # This is load-bearing for 100k/256 K=1 global searches: otherwise the
    # planner allows the full 36,864-rotation block and XLA tries to allocate
    # a 22 GiB pose-pixel tile in one shot.
    pose_pixel_bytes_per_rotation = max(
        1,
        int(
            np.ceil(
                n_trans
                * score_half_pixels
                * np.dtype(np.complex128).itemsize
                * n_classes
                * _RELION_EM_BATCH_POSE_PIXEL_LIVE_FACTOR,
            )
        ),
    )
    pose_pixel_rotation_cap = max(1, int(pose_pixel_budget_gb * 1e9 // pose_pixel_bytes_per_rotation))
    rotation_cap = min(
        score_rotation_cap,
        projection_rotation_cap,
        score_matmul_rotation_cap,
        pose_pixel_rotation_cap,
    )

    if requested_rotation_block_size >= 64:
        rotation_block = min(requested_rotation_block_size, rotation_cap)
        if rotation_cap >= 64:
            rotation_block = max(64, rotation_block)
    else:
        rotation_block = min(requested_rotation_block_size, rotation_cap)
    rotation_block = max(1, min(rotation_block, n_rot))

    projection_block_gb = rotation_block * projection_bytes_per_rotation / 1e9
    active_score_tile_gb = image_batch * active_score_bytes_per_image / 1e9
    pose_pixel_tile_gb = rotation_block * pose_pixel_bytes_per_rotation / 1e9
    translation_tile_gb = image_batch * translation_bytes_per_image / 1e9
    return _RelionEMBatchPlan(
        image_batch_size=int(image_batch),
        rotation_block_size=int(rotation_block),
        score_float_budget=int(score_float_budget),
        projection_budget_gb=float(projection_budget_gb),
        translation_tile_budget_gb=float(translation_tile_budget_gb),
        persistent_estimate_gb=float(persistent_gb),
        usable_estimate_gb=float(usable_gb),
        gpu_used_estimate_gb=float(gpu_used_gb),
        runtime_free_estimate_gb=float(runtime_free_gb),
        projection_block_gb=float(projection_block_gb),
        active_score_tile_budget_gb=float(active_score_tile_budget_gb),
        active_score_tile_gb=float(active_score_tile_gb),
        pose_pixel_tile_gb=float(pose_pixel_tile_gb),
        translation_tile_gb=float(translation_tile_gb),
        score_pixel_count=int(score_half_pixels),
    )


def _image_backend(ds):
    return getattr(getattr(ds, "image_source", None), "backend", None)


def _dataset_raw_image_loader(ds):
    backend = _image_backend(ds)
    loader = getattr(backend, "source", None)
    if loader is None or not hasattr(loader, "load_all"):
        return None
    return loader


def _estimate_raw_image_cache_bytes(loader) -> int:
    n_images = int(getattr(loader, "num_images", getattr(loader, "n", 0)))
    image_size = int(getattr(loader, "image_size", getattr(loader, "D", 0)))
    dtype = np.dtype(getattr(loader, "_dtype", np.float32))
    return int(n_images * image_size * image_size * dtype.itemsize)


def _em_raw_image_cache_mode() -> str:
    return os.environ.get(_EM_RAW_IMAGE_CACHE_ENV, "auto").strip().lower()


def maybe_cache_raw_image_loaders(experiment_datasets) -> None:
    """Keep file-backed raw particles in host memory across RELION EM passes."""
    mode = _em_raw_image_cache_mode()
    if mode in {"0", "false", "no", "off", "disable", "disabled"}:
        logger.info("RELION mode raw image cache disabled by %s=%s", _EM_RAW_IMAGE_CACHE_ENV, mode)
        return
    force = mode in {"1", "true", "yes", "on", "force", "always"}

    planned = []
    seen = set()
    total_bytes = 0
    for ds in experiment_datasets:
        loader = _dataset_raw_image_loader(ds)
        if loader is None:
            continue
        loader_id = id(loader)
        if loader_id in seen:
            continue
        seen.add(loader_id)
        if getattr(loader, "_cached", None) is not None:
            continue
        estimated_bytes = _estimate_raw_image_cache_bytes(loader)
        if estimated_bytes <= 0:
            continue
        planned.append((loader, estimated_bytes))
        total_bytes += estimated_bytes

    if not planned:
        return

    max_gb = float(os.environ.get(_EM_RAW_IMAGE_CACHE_MAX_GB_ENV, _EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB))
    max_bytes = int(max_gb * (1024**3))
    if not force and total_bytes > max_bytes:
        logger.info(
            "RELION mode raw image cache skipped: estimated %.2f GiB exceeds %.2f GiB; "
            "set %s=force or increase %s to override",
            total_bytes / (1024**3),
            max_gb,
            _EM_RAW_IMAGE_CACHE_ENV,
            _EM_RAW_IMAGE_CACHE_MAX_GB_ENV,
        )
        return

    cache_t0 = time.time()
    for loader, estimated_bytes in planned:
        loader_t0 = time.time()
        loader.load_all()
        logger.info(
            "RELION mode raw image cache loaded %.2f GiB for %s in %.1fs",
            estimated_bytes / (1024**3),
            type(loader).__name__,
            time.time() - loader_t0,
        )
    logger.info(
        "RELION mode raw image cache ready: %.2f GiB across %d loader(s) in %.1fs",
        total_bytes / (1024**3),
        len(planned),
        time.time() - cache_t0,
    )
