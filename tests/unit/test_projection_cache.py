"""Focused CPU contracts for the shared projection-cache seam."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.helpers import projection_cache

pytestmark = pytest.mark.unit


def _gf46_plan(
    *,
    destination_alias_proven: bool,
    budget_bytes: int = 2 * 1024**3,
    requested_max_chunk_rows: int = 5_000,
):
    return projection_cache.plan_projection_cache(
        table_count=1,
        row_count=36_864,
        pixel_count=5_100,
        cache_dtype=np.complex64,
        requested_max_chunk_rows=requested_max_chunk_rows,
        row_alignment=16,
        transient_specs=(
            projection_cache.ProjectionCacheTransientSpec(
                name="full_centered_projection",
                elements_per_row=128 * 65,
                dtype=np.complex64,
            ),
        ),
        budget_bytes=budget_bytes,
        destination_alias_proven=destination_alias_proven,
    )


def test_projection_cache_plan_records_exact_gf46_bytes_and_alignment():
    plan = _gf46_plan(destination_alias_proven=True)

    assert plan.cache_shape == (1, 36_864, 5_100)
    assert plan.chunk_rows == 4_992
    assert plan.chunk_count_per_table == 8
    assert plan.table_bytes == 1_504_051_200
    assert plan.retained_bytes == 1_504_051_200
    assert plan.projection_block_bytes == 203_673_600
    assert plan.additional_transient_bytes == 332_267_520
    assert plan.destination_copy_bytes == 0
    assert plan.predicted_peak_bytes == 2_039_992_320
    assert plan.admitted
    assert plan.admission_reason is None


def test_projection_cache_plan_records_h100_qualified_uniform_gf46_chunks():
    plan = _gf46_plan(
        destination_alias_proven=True,
        requested_max_chunk_rows=4_608,
    )

    assert 36_864 == 8 * plan.chunk_rows
    assert plan.chunk_rows == 4_608
    assert plan.chunk_count_per_table == 8
    assert plan.retained_bytes == 1_504_051_200
    assert plan.projection_block_bytes == 188_006_400
    assert plan.additional_transient_bytes == 306_708_480
    assert plan.predicted_peak_bytes == 1_998_766_080
    assert plan.admitted


def test_projection_cache_plan_is_conservative_until_destination_alias_is_proven():
    plan = _gf46_plan(destination_alias_proven=False)

    assert plan.destination_copy_bytes == plan.retained_bytes
    assert plan.predicted_peak_bytes == 3_544_043_520
    assert not plan.admitted
    assert plan.admission_reason == ("predicted projection-cache peak 3544043520 bytes exceeds budget 2147483648 bytes")


@pytest.mark.parametrize(
    "field",
    [
        "table_count",
        "row_count",
        "pixel_count",
        "requested_max_chunk_rows",
        "row_alignment",
        "budget_bytes",
    ],
)
def test_projection_cache_plan_rejects_fractional_host_counts(field):
    arguments = {
        "table_count": 1,
        "row_count": 4,
        "pixel_count": 3,
        "cache_dtype": np.complex64,
        "requested_max_chunk_rows": 2,
        "row_alignment": 1,
        "budget_bytes": 1_000,
    }
    arguments[field] = 1.5

    with pytest.raises(TypeError, match="integer"):
        projection_cache.plan_projection_cache(**arguments)


@pytest.mark.parametrize("field", ["elements_per_row", "count"])
def test_projection_cache_transient_rejects_fractional_host_counts(field):
    arguments = {
        "name": "workspace",
        "elements_per_row": 4,
        "dtype": np.complex64,
        "count": 1,
    }
    arguments[field] = 1.5

    with pytest.raises(TypeError, match="integer"):
        projection_cache.ProjectionCacheTransientSpec(**arguments)


def test_projection_cache_array_bytes_rejects_fractional_dimensions():
    with pytest.raises(TypeError, match="integer"):
        projection_cache.array_nbytes((2, 1.5), np.complex64)


@pytest.mark.parametrize("alias_evidence", ["false", 0, 1, None])
def test_projection_cache_plan_requires_boolean_alias_evidence(alias_evidence):
    with pytest.raises(TypeError, match="must be boolean"):
        projection_cache.plan_projection_cache(
            table_count=1,
            row_count=4,
            pixel_count=3,
            cache_dtype=np.complex64,
            requested_max_chunk_rows=2,
            budget_bytes=1_000,
            destination_alias_proven=alias_evidence,
        )

    numpy_bool_plan = projection_cache.plan_projection_cache(
        table_count=1,
        row_count=4,
        pixel_count=3,
        cache_dtype=np.complex64,
        requested_max_chunk_rows=2,
        budget_bytes=1_000,
        destination_alias_proven=np.bool_(True),
    )
    assert numpy_bool_plan.destination_alias_proven is True


def test_projection_cache_builder_fills_exact_physical_rows_in_bounded_blocks():
    plan = projection_cache.plan_projection_cache(
        table_count=2,
        row_count=7,
        pixel_count=3,
        cache_dtype=np.complex64,
        requested_max_chunk_rows=5,
        row_alignment=2,
        budget_bytes=10_000,
        destination_alias_proven=True,
    )
    calls = []
    expected = np.empty(plan.cache_shape, dtype=np.complex64)

    def project_block(table_index: int, start: int, stop: int):
        calls.append((table_index, start, stop))
        rows = np.arange(start, stop, dtype=np.float32)[:, None]
        columns = np.arange(plan.pixel_count, dtype=np.float32)[None, :]
        block = (100 * table_index + 10 * rows + columns).astype(np.complex64)
        expected[table_index, start:stop] = block
        return block

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        actual = projection_cache.build_projection_cache(plan, project_block)

    assert calls == [
        (0, 0, 4),
        (0, 4, 7),
        (1, 0, 4),
        (1, 4, 7),
    ]
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_projection_cache_builder_rejects_unadmitted_plan_before_allocation(monkeypatch):
    plan = projection_cache.plan_projection_cache(
        table_count=1,
        row_count=4,
        pixel_count=3,
        cache_dtype=np.complex64,
        requested_max_chunk_rows=2,
        budget_bytes=1,
        destination_alias_proven=True,
    )

    def fail_allocation(_plan):
        raise AssertionError("an unadmitted plan must not allocate")

    monkeypatch.setattr(projection_cache, "_allocate_projection_cache", fail_allocation)
    with pytest.raises(MemoryError, match="predicted projection-cache peak"):
        projection_cache.build_projection_cache(
            plan,
            lambda _table_index, start, stop: np.zeros((stop - start, 3), dtype=np.complex64),
        )


def test_projection_cache_builder_rejects_noncallable_before_allocation(monkeypatch):
    plan = projection_cache.plan_projection_cache(
        table_count=1,
        row_count=4,
        pixel_count=3,
        cache_dtype=np.complex64,
        requested_max_chunk_rows=2,
        budget_bytes=1_000,
        destination_alias_proven=True,
    )

    def fail_allocation(_plan):
        raise AssertionError("a noncallable project_block must not allocate")

    monkeypatch.setattr(projection_cache, "_allocate_projection_cache", fail_allocation)
    with pytest.raises(TypeError, match="project_block must be callable"):
        projection_cache.build_projection_cache(plan, None)


@pytest.mark.parametrize(
    ("block", "error_type", "message"),
    [
        (np.zeros((2, 2), dtype=np.complex64), ValueError, "has shape"),
        (np.zeros((2, 3), dtype=np.complex128), TypeError, "has dtype"),
    ],
)
def test_projection_cache_builder_fails_closed_on_block_contract(block, error_type, message):
    plan = projection_cache.plan_projection_cache(
        table_count=1,
        row_count=2,
        pixel_count=3,
        cache_dtype=np.complex64,
        requested_max_chunk_rows=2,
        budget_bytes=1_000,
        destination_alias_proven=True,
    )

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device), pytest.raises(error_type, match=message):
        projection_cache.build_projection_cache(plan, lambda _table_index, _start, _stop: block)


def test_projection_cache_row_insert_has_compiled_cpu_destination_alias():
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        cache = jnp.zeros((1, 7, 3), dtype=jnp.complex64)
        block = jnp.ones((2, 3), dtype=jnp.complex64)
        lowered = projection_cache._write_projection_cache_rows.lower(
            cache,
            block,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        )
        hlo = lowered.compiler_ir("hlo").as_hlo_text()
        memory = lowered.compile().memory_analysis()

    expected_bytes = projection_cache.array_nbytes(cache.shape, cache.dtype)
    assert "input_output_alias={ {}: (0, {}, may-alias) }" in hlo
    assert memory.output_size_in_bytes == expected_bytes
    assert memory.alias_size_in_bytes == expected_bytes
