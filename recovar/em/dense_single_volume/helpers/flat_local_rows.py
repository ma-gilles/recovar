"""Source-ordered flat-row plans for exact local scoring."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.deterministic_reduce import (
    deterministic_reductions_enabled,
)
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size


@dataclass(frozen=True)
class FlatLocalRowPlan:
    """Map one rectangular local bucket to source-ordered packed rows."""

    image_indices: np.ndarray
    rotation_rows: np.ndarray
    present_mask: np.ndarray
    valid_mask: np.ndarray
    batch_size: int
    physical_image_count: int
    dense_rotation_count: int
    packed_row_count: int


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
) -> FlatLocalRowPlan:
    """Pack consecutive physical pools without changing image/rotation order.

    Each pool uses the largest ordinary exact-local rotation bucket needed by
    one member. ``packed_row_count`` may pad the final flat axis to one static
    shape shared by several outer calls; padded rows are marked absent.
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

    ordinary_buckets = np.asarray(
        [
            _exact_bucket_rotation_size(
                int(count),
                rotation_block_size,
                exact_local_bucket_radix=exact_local_bucket_radix,
            )
            for count in rotation_counts
        ],
        dtype=np.int32,
    )
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
        batch_size=dense_batch_size,
        physical_image_count=physical_image_count,
        dense_rotation_count=dense_rotation_count,
        packed_row_count=packed_row_count,
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
