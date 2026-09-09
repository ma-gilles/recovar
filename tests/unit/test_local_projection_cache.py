"""Projection-cache budgets, grouping and initialized-row ownership."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_projection_cache as cache
from recovar.em.dense_single_volume.local_layout import LocalBucketSpec

pytestmark = pytest.mark.unit


def bucket(ids, *, image=0, mask=None):
    ids = np.asarray([ids], dtype=np.int32)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (*ids.shape, 3, 3)).copy()
    rotations[0, :, 0, 0] = np.arange(ids.size) + image * 10
    return LocalBucketSpec(
        image_indices=np.array([image], dtype=np.int32),
        bucket_image_count=1,
        bucket_rotation_count=ids.size,
        actual_rotation_counts=np.array([ids.size], dtype=np.int32),
        local_rotation_ids=ids,
        local_rotations=rotations,
        local_rotation_log_prior=np.zeros(ids.shape, dtype=np.float32),
        local_rotation_mask=np.ones(ids.shape, dtype=bool) if mask is None else np.asarray([mask]),
        translation_log_prior=np.zeros((1, 1), dtype=np.float32),
    )


@pytest.mark.parametrize("value", [None, "", "  ", "0", "0.000000024"])
def test_budget_uses_complex64_row_bytes(monkeypatch, value):
    name = cache.EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV
    monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(name, value)
    rows, gb = cache.cache_capacity_rows(3)
    assert rows == (1 if value == "0.000000024" else 0)
    assert gb == (2.4e-8 if rows else 0.0)


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "invalid"])
def test_invalid_budget_fails_closed(monkeypatch, value):
    monkeypatch.setenv(cache.EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV, value)
    with pytest.raises(ValueError, match="must be a non-negative"):
        cache.cache_capacity_rows(3)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", ""])
def test_invalid_group_limit_fails_closed(monkeypatch, value):
    monkeypatch.setenv(cache.EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS_ENV, value)
    with pytest.raises(ValueError, match="must be positive"):
        cache.max_cache_groups()


def test_grouping_uses_valid_distinct_ids_and_consecutive_ranges():
    buckets = [bucket([2, 2, -1, 99], mask=[True, True, True, False]), bucket([2, 5]), bucket([5, 8]), bucket([-1])]
    assert cache.plan_cache_groups(buckets, cache_row_capacity=2) == [(0, 2, 2), (2, 4, 2)]
    assert cache.plan_cache_groups(buckets, cache_row_capacity=1) == []
    assert cache.plan_cache_groups(buckets, cache_row_capacity=0) == []
    assert cache.plan_cache_groups([], cache_row_capacity=2) == []
    assert cache.plan_cache_groups([bucket([-1])], cache_row_capacity=2) == [(0, 1, 0)]


def test_sort_preserves_ties_and_input_list():
    first, second = bucket([3, 5], image=1), bucket([5, 3], image=1)
    empty, small = bucket([-1, -1], image=4), bucket([9], image=2)
    original = [first, second, empty, small]
    ordered = cache.sort_buckets(original)
    assert [id(b) for b in ordered] == [id(b) for b in [small, empty, first, second]]
    assert [id(b) for b in original] == [id(b) for b in [first, second, empty, small]]


@pytest.mark.parametrize("mask_disk", [True, False])
def test_builder_preserves_first_rotation_for_id_and_chunk_arguments(monkeypatch, mask_disk):
    monkeypatch.setenv(cache.EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS_ENV, "3")
    buckets = [bucket([5, 2, -1], image=1), bucket([2, 8], image=2)]
    calls, barriers = [], []

    def project(projector, rotations, image_shape, **kwargs):
        calls.append((np.asarray(rotations).copy(), kwargs))
        return jnp.broadcast_to(rotations[:, 0, 0, None], (len(rotations), 3)).astype(jnp.complex64), None

    monkeypatch.setattr(cache, "_compute_relion_projector_projections_block", project)
    monkeypatch.setattr(cache, "_block_until_ready", lambda *values: barriers.append(len(values)))
    result = cache.build_cache(
        buckets,
        object(),
        image_shape=(4, 4),
        n_projection_pixels=3,
        relion_projector_r_max=2,
        projection_padding_factor=2,
        projection_relion_texture_interp=True,
        projection_pixel_indices=np.array([7, 0, 7]),
        projection_relion_acc_double_floorf_quirk=True,
        projector_output_size=4,
        cache_row_capacity=4,
        max_global_rotation_id=10,
        group_index=0,
        n_groups=1,
        projection_mask_current_image_disk=mask_disk,
    )
    assert result.enabled and result.row_count == 3 and result.id_map_row_count == 11
    assert result.projections.shape == (4, 3) and result.projections.dtype == np.complex64
    np.testing.assert_array_equal(np.asarray(result.id_map)[[2, 5, 8]], [0, 1, 2])
    # Only defined rows are read; padding remains uninitialized by contract.
    np.testing.assert_array_equal(np.asarray(result.projections)[:3], np.repeat([[11], [10], [21]], 3, axis=1))
    assert barriers == [1, 1, 1, 2]
    for rotations, kwargs in calls:
        assert rotations.dtype == np.float32 and rotations.shape == (1, 3, 3)
        assert kwargs["relion_texture_interp"] is True and kwargs["relion_acc_double_floorf_quirk"] is True
        assert kwargs["r_max"] == 2 and kwargs["padding_factor"] == 2
        assert kwargs["return_abs2"] is False and kwargs["centered_rows"] and kwargs["dense_scale"]
        assert kwargs["projector_output_size"] == 4
        assert kwargs.get("mask_current_image_disk", True) is mask_disk
        np.testing.assert_array_equal(kwargs["pixel_indices"], [7, 0, 7])


def test_builder_disabled_and_oversized_groups_do_not_project(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Projection must not run for disabled/invalid groups")

    monkeypatch.setattr(cache, "_compute_relion_projector_projections_block", forbidden)
    kwargs = dict(
        image_shape=(4, 4),
        n_projection_pixels=3,
        relion_projector_r_max=2,
        projection_padding_factor=1,
        projection_relion_texture_interp=None,
        projection_pixel_indices=None,
        projector_output_size=0,
        max_global_rotation_id=2,
        group_index=0,
        n_groups=1,
    )
    for buckets, capacity in [([], 1), ([bucket([0])], 0), ([bucket([-1])], 1)]:
        result = cache.build_cache(buckets, object(), cache_row_capacity=capacity, **kwargs)
        assert not result.enabled and result.projections.shape == (1, 1)
        np.testing.assert_array_equal(result.id_map, [0])
    with pytest.raises(RuntimeError, match="group has 2 rows but capacity is 1"):
        cache.build_cache([bucket([0, 1])], object(), cache_row_capacity=1, **kwargs)
