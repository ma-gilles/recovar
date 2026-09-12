"""Exact memory planning and blockwise assembly for projection caches.

This module deliberately does not project volumes or score images.  Callers
provide the existing numerical projection function as a callback; the shared
code here only plans bytes, allocates one exact destination, and fills it in
bounded row blocks.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ProjectionCacheTransientSpec:
    """One additional per-row buffer live while a projection block is built.

    The compact projection block returned by the caller is accounted for
    automatically and must not be repeated here.  Specs describe other live
    projector buffers, for example a full centered half-image.
    """

    name: str
    elements_per_row: int
    dtype: object
    count: int = 1

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        elements_per_row = operator.index(self.elements_per_row)
        count = operator.index(self.count)
        dtype = _normalize_dtype(self.dtype)
        if not name:
            raise ValueError("projection-cache transient name must be non-empty")
        if elements_per_row <= 0:
            raise ValueError("projection-cache transient elements_per_row must be positive")
        if count <= 0:
            raise ValueError("projection-cache transient count must be positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "elements_per_row", elements_per_row)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "dtype", dtype)

    def bytes_for_rows(self, row_count: int) -> int:
        """Return exact storage for this transient at ``row_count`` rows."""

        rows = operator.index(row_count)
        if rows < 0:
            raise ValueError("projection-cache transient row_count must be nonnegative")
        return rows * self.elements_per_row * self.dtype.itemsize * self.count


@dataclass(frozen=True)
class ProjectionCachePlan:
    """Exact retained/build bytes and admission for one cache assembly."""

    table_count: int
    row_count: int
    pixel_count: int
    cache_dtype: np.dtype
    requested_max_chunk_rows: int
    chunk_rows: int
    row_alignment: int
    chunk_count_per_table: int
    transient_specs: tuple[ProjectionCacheTransientSpec, ...]
    table_bytes: int
    retained_bytes: int
    projection_block_bytes: int
    additional_transient_bytes: int
    destination_copy_bytes: int
    predicted_peak_bytes: int
    budget_bytes: int
    destination_alias_proven: bool
    admitted: bool
    admission_reason: str | None

    @property
    def cache_shape(self) -> tuple[int, int, int]:
        """Return the exact ``(table, row, pixel)`` destination shape."""

        return self.table_count, self.row_count, self.pixel_count


def _normalize_dtype(dtype) -> np.dtype:
    normalized = np.dtype(dtype)
    if normalized.hasobject or normalized.itemsize <= 0:
        raise TypeError(f"projection-cache dtype must have fixed-width storage, got {normalized}")
    return normalized


def array_nbytes(shape: Iterable[int], dtype) -> int:
    """Return exact dense-array storage without allocating the array."""

    element_count = 1
    for raw_dimension in shape:
        dimension = operator.index(raw_dimension)
        if dimension < 0:
            raise ValueError("array dimensions must be nonnegative")
        element_count *= dimension
    return element_count * _normalize_dtype(dtype).itemsize


def aligned_projection_cache_chunk_rows(
    row_count: int,
    requested_max_chunk_rows: int,
    *,
    row_alignment: int = 1,
) -> int:
    """Choose aligned non-final chunks without padding the physical row set."""

    rows = operator.index(row_count)
    requested = operator.index(requested_max_chunk_rows)
    alignment = operator.index(row_alignment)
    if rows <= 0:
        raise ValueError("projection-cache row_count must be positive")
    if requested <= 0:
        raise ValueError("projection-cache requested_max_chunk_rows must be positive")
    if alignment <= 0:
        raise ValueError("projection-cache row_alignment must be positive")
    if rows <= requested:
        return rows
    aligned = requested - requested % alignment
    if aligned <= 0:
        raise ValueError(
            "projection-cache requested_max_chunk_rows is smaller than the required row_alignment",
        )
    return aligned


def plan_projection_cache(
    *,
    table_count: int,
    row_count: int,
    pixel_count: int,
    cache_dtype,
    requested_max_chunk_rows: int,
    budget_bytes: int,
    row_alignment: int = 1,
    transient_specs: Iterable[ProjectionCacheTransientSpec] = (),
    destination_alias_proven: bool = False,
) -> ProjectionCachePlan:
    """Plan exact retained and peak bytes before any device allocation.

    ``budget_bytes`` is explicit so callers can retain their existing device
    memory and environment policy.  The peak contains all retained tables,
    one compact callback result, every declared projector transient, and a
    second complete destination unless input/output aliasing has been proven
    for the target lowering.
    """

    tables = operator.index(table_count)
    rows = operator.index(row_count)
    pixels = operator.index(pixel_count)
    requested = operator.index(requested_max_chunk_rows)
    alignment = operator.index(row_alignment)
    budget = operator.index(budget_bytes)
    if tables <= 0:
        raise ValueError("projection-cache table_count must be positive")
    if rows <= 0:
        raise ValueError("projection-cache row_count must be positive")
    if pixels <= 0:
        raise ValueError("projection-cache pixel_count must be positive")
    if budget < 0:
        raise ValueError("projection-cache budget_bytes must be nonnegative")

    dtype = _normalize_dtype(cache_dtype)
    specs = tuple(transient_specs)
    if not all(isinstance(spec, ProjectionCacheTransientSpec) for spec in specs):
        raise TypeError("projection-cache transient_specs must contain ProjectionCacheTransientSpec values")
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("projection-cache transient names must be unique")

    if not isinstance(destination_alias_proven, (bool, np.bool_)):
        raise TypeError("projection-cache destination_alias_proven must be boolean")

    chunk_rows = aligned_projection_cache_chunk_rows(rows, requested, row_alignment=alignment)
    table_bytes = array_nbytes((rows, pixels), dtype)
    retained_bytes = tables * table_bytes
    projection_block_bytes = array_nbytes((chunk_rows, pixels), dtype)
    additional_transient_bytes = sum(spec.bytes_for_rows(chunk_rows) for spec in specs)
    alias_proven = bool(destination_alias_proven)
    destination_copy_bytes = 0 if alias_proven else retained_bytes
    predicted_peak_bytes = retained_bytes + projection_block_bytes + additional_transient_bytes + destination_copy_bytes
    admitted = predicted_peak_bytes <= budget
    admission_reason = None
    if not admitted:
        admission_reason = f"predicted projection-cache peak {predicted_peak_bytes} bytes exceeds budget {budget} bytes"

    return ProjectionCachePlan(
        table_count=tables,
        row_count=rows,
        pixel_count=pixels,
        cache_dtype=dtype,
        requested_max_chunk_rows=requested,
        chunk_rows=chunk_rows,
        row_alignment=alignment,
        chunk_count_per_table=(rows + chunk_rows - 1) // chunk_rows,
        transient_specs=specs,
        table_bytes=table_bytes,
        retained_bytes=retained_bytes,
        projection_block_bytes=projection_block_bytes,
        additional_transient_bytes=additional_transient_bytes,
        destination_copy_bytes=destination_copy_bytes,
        predicted_peak_bytes=predicted_peak_bytes,
        budget_bytes=budget,
        destination_alias_proven=alias_proven,
        admitted=admitted,
        admission_reason=admission_reason,
    )


@partial(jax.jit, donate_argnums=(0,))
def _write_projection_cache_rows(cache, projection_block, table_index, row_start):
    """Insert one row block while donating the private destination buffer."""

    pixel_start = jnp.asarray(0, dtype=table_index.dtype)
    return jax.lax.dynamic_update_slice(
        cache,
        projection_block[jnp.newaxis, :, :],
        (table_index, row_start, pixel_start),
    )


def _allocate_projection_cache(plan: ProjectionCachePlan):
    return jnp.zeros(plan.cache_shape, dtype=plan.cache_dtype)


def build_projection_cache(
    plan: ProjectionCachePlan,
    project_block: Callable[[int, int, int], object],
):
    """Build an admitted cache from sequential class/table projection blocks.

    ``project_block(table_index, start, stop)`` must return exactly
    ``(stop - start, pixel_count)`` in ``cache_dtype``.  Each donated insert is
    synchronized before the next projection so asynchronous projector and
    scatter temporaries cannot accumulate.  The returned cache is published;
    callers must not donate it to later scoring operations.
    """

    if not isinstance(plan, ProjectionCachePlan):
        raise TypeError("plan must be a ProjectionCachePlan")
    if not callable(project_block):
        raise TypeError("project_block must be callable")
    if not plan.admitted:
        raise MemoryError(plan.admission_reason or "projection-cache plan was not admitted")
    canonical_dtype = np.dtype(jax.dtypes.canonicalize_dtype(plan.cache_dtype))
    if canonical_dtype != plan.cache_dtype:
        raise ValueError(
            f"JAX canonicalizes requested projection-cache dtype {plan.cache_dtype} "
            f"to {canonical_dtype}; enable the required precision before building",
        )

    cache = _allocate_projection_cache(plan)
    cache.block_until_ready()
    for table_index in range(plan.table_count):
        for start in range(0, plan.row_count, plan.chunk_rows):
            stop = min(start + plan.chunk_rows, plan.row_count)
            projection_block = jnp.asarray(project_block(table_index, start, stop))
            expected_shape = (stop - start, plan.pixel_count)
            actual_shape = tuple(int(dimension) for dimension in projection_block.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"projection block {table_index}:{start}:{stop} has shape "
                    f"{actual_shape}, expected {expected_shape}",
                )
            actual_dtype = np.dtype(projection_block.dtype)
            if actual_dtype != plan.cache_dtype:
                raise TypeError(
                    f"projection block {table_index}:{start}:{stop} has dtype "
                    f"{actual_dtype}, expected {plan.cache_dtype}",
                )
            cache = _write_projection_cache_rows(
                cache,
                projection_block,
                jnp.asarray(table_index, dtype=jnp.int32),
                jnp.asarray(start, dtype=jnp.int32),
            )
            cache.block_until_ready()
            del projection_block
    return cache
