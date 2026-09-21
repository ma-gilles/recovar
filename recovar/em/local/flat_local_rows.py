"""Source-ordered flat-row plans for exact local scoring."""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.deterministic_reduce import deterministic_reductions_enabled
from recovar.em.local.local_layout import (
    _exact_bucket_rotation_size,
    _exact_local_large_bucket_quantum,
)

# Consecutive images share a padded rotation width. Larger pools produce fewer
# compiled shapes and more padding; ``valid_mask`` keeps the result unchanged.
EXACT_LOCAL_FLAT_POOL_SIZE = 3
EXACT_LOCAL_FLAT_POOL_SIZE_ENV = "RECOVAR_EXACT_LOCAL_FLAT_POOL_SIZE"

# Per-image widths retain the earlier exact-local bucket rounding. Packed rows
# no longer require it for correctness, but it can make neighboring pools share
# capacity, so keep the measured default configurable.
EXACT_LOCAL_FLAT_ROW_ROUNDING = True
EXACT_LOCAL_FLAT_ROW_ROUNDING_ENV = "RECOVAR_EXACT_LOCAL_FLAT_ROW_ROUNDING"

# Quantizing the running packed-row capacity lets neighboring iterations reuse
# compiled shapes. The value is ladder steps per power of two; zero preserves
# the exact running maximum. Padding stays score-inert through ``valid_mask``.
EXACT_LOCAL_FLAT_ROW_CAPACITY_STEPS = 0
EXACT_LOCAL_FLAT_ROW_CAPACITY_STEPS_ENV = "RECOVAR_EXACT_LOCAL_FLAT_ROW_CAPACITY_STEPS"


def resolve_flat_local_row_capacity_steps(explicit: int | None = None) -> int:
    """Resolve the packed-row capacity ladder granularity, in steps per octave."""

    source = "flat_local_row_capacity_steps"
    raw_value = explicit
    if raw_value is None:
        source = EXACT_LOCAL_FLAT_ROW_CAPACITY_STEPS_ENV
        raw_value = os.environ.get(
            EXACT_LOCAL_FLAT_ROW_CAPACITY_STEPS_ENV,
            str(EXACT_LOCAL_FLAT_ROW_CAPACITY_STEPS),
        ).strip()
        if not raw_value:
            raw_value = EXACT_LOCAL_FLAT_ROW_CAPACITY_STEPS
    try:
        steps = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a non-negative integer") from exc
    if steps < 0:
        raise ValueError(f"{source} must be a non-negative integer")
    return steps


def quantize_packed_row_capacity(capacity: int, steps: int) -> int:
    """Round a packed-row capacity up onto a ladder of ``steps`` values per octave.

    Returns ``capacity`` unchanged when the ladder is disabled or the capacity is
    not positive. The result is never smaller than ``capacity``, so a quantized
    plan always has room for every row the unquantized plan required.
    """

    capacity = int(capacity)
    if steps <= 0 or capacity <= 0:
        return capacity
    octave = 1 << (capacity.bit_length() - 1)
    stride = max(1, octave // steps)
    return -(-capacity // stride) * stride


def resolve_flat_local_pool_size(explicit: int | None = None) -> int:
    """Resolve the packed-row pool size from an explicit value or the environment."""

    source = "flat_local_pool_size"
    raw_value = explicit
    if raw_value is None:
        source = EXACT_LOCAL_FLAT_POOL_SIZE_ENV
        raw_value = os.environ.get(
            EXACT_LOCAL_FLAT_POOL_SIZE_ENV, str(EXACT_LOCAL_FLAT_POOL_SIZE)
        ).strip()
        if not raw_value:
            raw_value = EXACT_LOCAL_FLAT_POOL_SIZE
    try:
        pool_size = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a positive integer") from exc
    if pool_size < 1:
        raise ValueError(f"{source} must be a positive integer")
    return pool_size


def resolve_flat_local_row_rounding(explicit: bool | None = None) -> bool:
    """Resolve whether packed row blocks round up to ordinary rotation buckets."""

    if explicit is not None:
        return bool(explicit)
    raw_value = os.environ.get(EXACT_LOCAL_FLAT_ROW_ROUNDING_ENV, "").strip()
    if not raw_value:
        return bool(EXACT_LOCAL_FLAT_ROW_ROUNDING)
    if raw_value in ("0", "false", "False"):
        return False
    if raw_value in ("1", "true", "True"):
        return True
    raise ValueError(f"{EXACT_LOCAL_FLAT_ROW_ROUNDING_ENV} must be 0 or 1")


@dataclass(frozen=True)
class FlatLocalRowPlan:
    """Map one rectangular local bucket to source-ordered packed rows."""

    image_indices: np.ndarray
    rotation_rows: np.ndarray
    present_mask: np.ndarray
    valid_mask: np.ndarray
    packed_row_count: int
    # Class-segmented plans only. Rows are emitted class-major, so class k owns the
    # contiguous block [sum(class_row_counts[:k]), +class_row_counts[k]). Projection
    # slices those blocks to use each class's own volume; None for single-class plans.
    class_row_counts: tuple[int, ...] | None = None


def encode_flat_local_row_plan(plan: FlatLocalRowPlan) -> np.ndarray:
    """Encode row/image/validity metadata as one JAX-friendly int32 array.

    Pool and static-capacity padding rows remain in the physical array so JIT
    shapes stay unchanged, but downstream kernels can reject them before doing
    any pixel work.  Scatters likewise only publish logical rotation rows.
    """

    if not isinstance(plan, FlatLocalRowPlan):
        raise TypeError("flat local row encoding requires a FlatLocalRowPlan")
    return np.stack(
        (
            plan.image_indices,
            plan.rotation_rows,
            plan.valid_mask.astype(np.int32),
        ),
        axis=1,
    ).astype(np.int32, copy=False)


def build_dense_to_flat_local_row_lookup(
    encoded_plan,
    *,
    batch_size: int,
    dense_rotation_count: int,
) -> np.ndarray:
    """Invert valid encoded rows to ``(image, dense rotation) -> flat row``.

    Invalid pool-padding and static-tail rows are deliberately absent from the
    lookup.  The returned ``-1`` sentinel lets callers fail closed before a
    device gather when a requested reconstruction row was not scored.
    """

    encoded_plan = np.asarray(encoded_plan)
    batch_size = int(batch_size)
    dense_rotation_count = int(dense_rotation_count)
    if (
        encoded_plan.dtype != np.int32
        or encoded_plan.ndim != 2
        or encoded_plan.shape[0] <= 0
        or encoded_plan.shape[1] != 3
    ):
        raise ValueError(
            "encoded flat local row plan must be a nonempty int32 array "
            "with shape (rows, 3)",
        )
    if batch_size <= 0 or dense_rotation_count <= 0:
        raise ValueError("dense flat-row lookup dimensions must be positive")
    if np.any((encoded_plan[:, 2] != 0) & (encoded_plan[:, 2] != 1)):
        raise ValueError("encoded flat local row validity must contain only 0 or 1")

    valid = encoded_plan[:, 2] != 0
    image_indices = encoded_plan[valid, 0]
    rotation_rows = encoded_plan[valid, 1]
    if np.any((image_indices < 0) | (image_indices >= batch_size)):
        raise ValueError("a valid flat local row has an out-of-range image index")
    if np.any((rotation_rows < 0) | (rotation_rows >= dense_rotation_count)):
        raise ValueError("a valid flat local row has an out-of-range rotation row")

    dense_indices = (
        image_indices.astype(np.int64) * dense_rotation_count + rotation_rows
    )
    if np.unique(dense_indices).size != dense_indices.size:
        raise ValueError("valid flat local rows contain duplicate dense coordinates")
    lookup = np.full((batch_size, dense_rotation_count), -1, dtype=np.int32)
    lookup[image_indices, rotation_rows] = np.flatnonzero(valid).astype(
        np.int32,
        copy=False,
    )
    return lookup


def map_dense_local_rows_to_flat_rows(
    dense_to_flat_lookup,
    dense_rotation_rows,
    row_mask,
) -> np.ndarray:
    """Map a packed source-order reconstruction grid onto scored flat rows.

    Masked padding rows map to row zero so the device gather is in bounds; the
    caller must mask their gathered values back to zero.  Every live row must
    exist in the canonical lookup or this helper raises before GPU execution.
    """

    dense_to_flat_lookup = np.asarray(dense_to_flat_lookup)
    dense_rotation_rows = np.asarray(dense_rotation_rows)
    row_mask = np.asarray(row_mask, dtype=bool)
    if dense_to_flat_lookup.dtype != np.int32 or dense_to_flat_lookup.ndim != 2:
        raise ValueError("dense-to-flat lookup must be a rank-two int32 array")
    if dense_rotation_rows.dtype != np.int32 or dense_rotation_rows.ndim != 2:
        raise ValueError("dense rotation rows must be a rank-two int32 array")
    if row_mask.shape != dense_rotation_rows.shape:
        raise ValueError("dense rotation rows and row mask must have matching shapes")
    if dense_rotation_rows.shape[0] > dense_to_flat_lookup.shape[0]:
        raise ValueError("dense rotation rows exceed the lookup image axis")
    if np.any(
        row_mask
        & (
            (dense_rotation_rows < 0)
            | (dense_rotation_rows >= dense_to_flat_lookup.shape[1])
        )
    ):
        raise ValueError("a live reconstruction rotation row is out of range")

    safe_dense_rows = np.where(row_mask, dense_rotation_rows, 0)
    flat_rows = np.take_along_axis(
        dense_to_flat_lookup[: dense_rotation_rows.shape[0]],
        safe_dense_rows,
        axis=1,
    )
    if np.any(row_mask & (flat_rows < 0)):
        raise ValueError("a live reconstruction row is absent from the flat score plan")
    return np.where(row_mask, flat_rows, 0).astype(np.int32, copy=False)


def build_pool_flat_local_row_plan(
    rotation_counts,
    dense_rotation_count: int,
    *,
    pool_size: int = 3,
    rotation_block_size: int = 5000,
    exact_local_bucket_radix: int | None = None,
    packed_row_count: int | None = None,
    dense_batch_size: int | None = None,
    large_bucket_quantum: int | None = None,
    round_row_widths: bool | None = None,
) -> FlatLocalRowPlan:
    """Pack consecutive physical pools without changing image/rotation order.

    Each pool uses the largest ordinary exact-local rotation bucket needed by
    one member. ``packed_row_count`` may pad the final flat axis to one static
    shape shared by several outer calls; padded rows are marked absent.
    ``large_bucket_quantum`` must be the bucket planner's quantum for supports
    above the engine cap, otherwise a pool bucket can outgrow the dense axis.
    """

    rotation_counts = np.asarray(rotation_counts, dtype=np.int32).reshape(-1)
    physical_image_count = int(rotation_counts.size)
    if dense_batch_size is None:
        dense_batch_size = physical_image_count
    dense_batch_size = int(dense_batch_size)
    dense_rotation_count = int(dense_rotation_count)
    pool_size = int(pool_size)
    rotation_block_size = int(rotation_block_size)
    if physical_image_count < 1:
        raise ValueError("rotation_counts must contain at least one image")
    if dense_batch_size < physical_image_count:
        raise ValueError("dense_batch_size cannot be smaller than the physical image count")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if dense_rotation_count < 1:
        raise ValueError("dense_rotation_count must be positive")
    if np.any(rotation_counts < 1) or np.any(rotation_counts > dense_rotation_count):
        raise ValueError("rotation counts must lie in [1, dense_rotation_count]")

    if resolve_flat_local_row_rounding(round_row_widths):
        ordinary_buckets = np.asarray(
            [
                _exact_bucket_rotation_size(
                    int(count),
                    rotation_block_size,
                    large_bucket_quantum=large_bucket_quantum,
                    exact_local_bucket_radix=exact_local_bucket_radix,
                )
                for count in rotation_counts
            ],
            dtype=np.int32,
        )
    else:
        ordinary_buckets = rotation_counts.astype(np.int32, copy=True)
    if np.any(ordinary_buckets > dense_rotation_count):
        raise ValueError("a pool bucket exceeds the enclosing dense rotation axis")

    image_parts: list[np.ndarray] = []
    rotation_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    for pool_start in range(0, physical_image_count, pool_size):
        pool_stop = min(physical_image_count, pool_start + pool_size)
        pool_bucket = int(np.max(ordinary_buckets[pool_start:pool_stop]))
        rows = np.arange(pool_bucket, dtype=np.int32)
        for image_index in range(pool_start, pool_stop):
            image_parts.append(np.full(pool_bucket, image_index, dtype=np.int32))
            rotation_parts.append(rows)
            valid_parts.append(rows < int(rotation_counts[image_index]))

    image_indices = np.concatenate(image_parts)
    rotation_rows = np.concatenate(rotation_parts)
    valid_mask = np.concatenate(valid_parts)
    required_rows = int(image_indices.size)
    if packed_row_count is None:
        packed_row_count = required_rows
    packed_row_count = int(packed_row_count)
    if packed_row_count < required_rows:
        raise ValueError(
            f"packed_row_count {packed_row_count} is smaller than required rows {required_rows}",
        )
    present_mask = np.arange(packed_row_count, dtype=np.int32) < required_rows
    if packed_row_count > required_rows:
        pad = packed_row_count - required_rows
        image_indices = np.pad(image_indices, (0, pad), constant_values=0)
        rotation_rows = np.pad(rotation_rows, (0, pad), constant_values=0)
        valid_mask = np.pad(valid_mask, (0, pad), constant_values=False)

    return FlatLocalRowPlan(
        image_indices=image_indices,
        rotation_rows=rotation_rows,
        present_mask=present_mask,
        valid_mask=valid_mask,
        packed_row_count=packed_row_count,
    )


def build_pool_flat_local_row_plan_for_classes(
    class_rotation_counts,
    segment_rotation_count: int,
    *,
    pool_size: int = 3,
    rotation_block_size: int = 5000,
    exact_local_bucket_radix: int | None = None,
    large_bucket_quantum: int | None = None,
    packed_row_count: int | None = None,
    dense_batch_size: int | None = None,
    round_row_widths: bool | None = None,
) -> FlatLocalRowPlan:
    """Pack class-segmented local rows without changing image/class/rotation order.

    A class-segmented bucket lays each image's rows out class-major,
    ``[class 0 rows | pad | class 1 rows | pad | ...]``, so a row's index on the
    dense axis is ``class * segment_rotation_count + rotation``. That is the only
    thing the flat plan needs to know about classes, which is why the returned
    plan is an ordinary :class:`FlatLocalRowPlan`: downstream kernels index rows
    through ``rotation_rows`` and never learn that classes exist.

    Unlike the rectangular layout, each class is sized independently from the
    largest count that class needs inside the pool, instead of every class and
    image sharing one segment width taken over the whole halfset.

    ``class_rotation_counts`` is ``[n_images, n_classes]``.
    """

    class_rotation_counts = np.asarray(class_rotation_counts, dtype=np.int32)
    if class_rotation_counts.ndim != 2:
        raise ValueError("class_rotation_counts must be [n_images, n_classes]")
    physical_image_count, n_classes = (int(x) for x in class_rotation_counts.shape)
    segment_rotation_count = int(segment_rotation_count)
    pool_size = int(pool_size)
    rotation_block_size = int(rotation_block_size)
    if dense_batch_size is None:
        dense_batch_size = physical_image_count
    dense_batch_size = int(dense_batch_size)
    if physical_image_count < 1:
        raise ValueError("class_rotation_counts must contain at least one image")
    if n_classes < 1:
        raise ValueError("class_rotation_counts must contain at least one class")
    if dense_batch_size < physical_image_count:
        raise ValueError("dense_batch_size cannot be smaller than the physical image count")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if segment_rotation_count < 1:
        raise ValueError("segment_rotation_count must be positive")
    if np.any(class_rotation_counts < 0) or np.any(class_rotation_counts > segment_rotation_count):
        raise ValueError("class rotation counts must lie in [0, segment_rotation_count]")

    dense_rotation_count = segment_rotation_count * n_classes
    # The segment width came from the bucketer's resolved large-bucket quantum, so a
    # class must be quantized with the same one; a different quantum can round a class
    # above its own segment.
    resolved_large_bucket_quantum = _exact_local_large_bucket_quantum(
        rotation_block_size, large_bucket_quantum,
    )
    if resolve_flat_local_row_rounding(round_row_widths):
        ordinary_buckets = np.asarray(
            [
                [
                    _exact_bucket_rotation_size(
                        int(count),
                        rotation_block_size,
                        large_bucket_quantum=resolved_large_bucket_quantum,
                        exact_local_bucket_radix=exact_local_bucket_radix,
                    )
                    if int(count) > 0
                    else 0
                    for count in image_counts
                ]
                for image_counts in class_rotation_counts
            ],
            dtype=np.int32,
        )
    else:
        ordinary_buckets = class_rotation_counts.astype(np.int32, copy=True)
    if np.any(ordinary_buckets > segment_rotation_count):
        raise ValueError("a pool bucket exceeds the enclosing class segment")

    image_parts: list[np.ndarray] = []
    rotation_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    class_row_counts: list[int] = []
    # Class-major: every row of class k is emitted before any row of class k+1, so a
    # projection can slice class k's block and use class k's volume. Within a class the
    # original pool/image order is preserved.
    for class_index in range(n_classes):
        rows_before = int(sum(part.size for part in image_parts))
        for pool_start in range(0, physical_image_count, pool_size):
            pool_stop = min(physical_image_count, pool_start + pool_size)
            # Each class gets its own width inside the pool; a class that no image in
            # the pool uses contributes no rows at all.
            width = int(ordinary_buckets[pool_start:pool_stop, class_index].max())
            if width == 0:
                continue
            rows = np.arange(width, dtype=np.int32)
            for image_index in range(pool_start, pool_stop):
                image_parts.append(np.full(width, image_index, dtype=np.int32))
                rotation_parts.append(rows + class_index * segment_rotation_count)
                valid_parts.append(rows < int(class_rotation_counts[image_index, class_index]))
        class_row_counts.append(int(sum(part.size for part in image_parts)) - rows_before)

    if not image_parts:
        raise ValueError("class-segmented flat rows need at least one populated class")
    image_indices = np.concatenate(image_parts)
    rotation_rows = np.concatenate(rotation_parts)
    valid_mask = np.concatenate(valid_parts)
    required_rows = int(image_indices.size)
    if packed_row_count is None:
        packed_row_count = required_rows
    packed_row_count = int(packed_row_count)
    if packed_row_count < required_rows:
        raise ValueError(
            f"packed_row_count {packed_row_count} is smaller than required rows {required_rows}",
        )
    present_mask = np.arange(packed_row_count, dtype=np.int32) < required_rows
    if packed_row_count > required_rows:
        pad = packed_row_count - required_rows
        image_indices = np.pad(image_indices, (0, pad), constant_values=0)
        rotation_rows = np.pad(rotation_rows, (0, pad), constant_values=0)
        valid_mask = np.pad(valid_mask, (0, pad), constant_values=False)

    return FlatLocalRowPlan(
        image_indices=image_indices,
        rotation_rows=rotation_rows,
        present_mask=present_mask,
        valid_mask=valid_mask,
        packed_row_count=packed_row_count,
        class_row_counts=tuple(class_row_counts),
    )


def scatter_flat_local_rows(
    flat_values,
    image_indices,
    rotation_rows,
    present_mask,
    *,
    batch_size: int,
    dense_rotation_count: int,
    fill_value,
):
    """Restore packed rows to a rectangular image/rotation tensor.

    Present packed rows are unique. Static tail padding is routed to one
    out-of-bounds sentinel and dropped by the scatter.
    """

    flat_values = jnp.asarray(flat_values)
    image_indices = jnp.asarray(image_indices, dtype=jnp.int32)
    rotation_rows = jnp.asarray(rotation_rows, dtype=jnp.int32)
    present_mask = jnp.asarray(present_mask, dtype=bool)
    dense_row_count = int(batch_size) * int(dense_rotation_count)
    dense_indices = image_indices * int(dense_rotation_count) + rotation_rows
    scatter_indices = jnp.where(present_mask, dense_indices, dense_row_count)
    if deterministic_reductions_enabled():
        # ``.at[].set`` lowers to an XLA scatter whose writer order is not
        # fixed when two packed rows address one dense row.  Pick the winner
        # with an order-independent integer scatter-max on the packed row id
        # and gather it; identical to the set whenever present rows are unique.
        flat_ids = jnp.arange(int(flat_values.shape[0]), dtype=jnp.int32)
        winner = jnp.full((dense_row_count,), -1, dtype=jnp.int32).at[scatter_indices].max(
            jnp.where(present_mask, flat_ids, jnp.int32(-1)), mode="drop"
        )
        has_row = winner >= 0
        gathered = flat_values[jnp.maximum(winner, 0)]
        fill = jnp.asarray(fill_value, dtype=flat_values.dtype)
        dense = jnp.where(
            has_row.reshape((dense_row_count,) + (1,) * (flat_values.ndim - 1)),
            gathered,
            fill,
        )
    else:
        dense = jnp.full(
            (dense_row_count,) + flat_values.shape[1:],
            jnp.asarray(fill_value, dtype=flat_values.dtype),
            dtype=flat_values.dtype,
        )
        dense = dense.at[scatter_indices].set(flat_values, mode="drop")
    return dense.reshape(
        (int(batch_size), int(dense_rotation_count)) + flat_values.shape[1:],
    )
