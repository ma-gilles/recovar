from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_engine
from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed
from recovar.em.dense_single_volume.helpers.flat_local_rows import (
    build_dense_to_flat_local_row_lookup,
    build_pool_flat_local_row_plan,
    encode_flat_local_row_plan,
    gather_flat_local_rows,
    map_dense_local_rows_to_flat_rows,
    scatter_flat_local_rows,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_noise_block,
    compute_norm_residual_per_image,
    compute_scale_correction_terms_per_image,
)
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_weighted_sums,
)
from recovar.em.dense_single_volume.local_layout import LocalBucketSpec


@pytest.mark.unit
def test_pool_flat_rows_preserve_source_chronology_and_static_tail():
    counts = np.asarray([3, 17, 5, 65, 9], dtype=np.int32)
    plan = build_pool_flat_local_row_plan(
        counts,
        256,
        pool_size=3,
        exact_local_bucket_radix=4,
        packed_row_count=3 * 64 + 2 * 256 + 7,
        dense_batch_size=8,
    )

    expected = []
    for image_index, bucket_size in ((0, 64), (1, 64), (2, 64), (3, 256), (4, 256)):
        expected.extend((image_index, rotation) for rotation in range(bucket_size))
    present = plan.present_mask
    assert list(zip(plan.image_indices[present], plan.rotation_rows[present], strict=True)) == expected
    assert np.array_equal(
        plan.valid_mask[present],
        np.asarray([rotation < counts[image] for image, rotation in expected]),
    )
    assert not np.any(plan.present_mask[-7:])
    assert not np.any(plan.valid_mask[-7:])
    assert plan.physical_image_count == 5
    assert plan.batch_size == 8
    encoded = encode_flat_local_row_plan(plan)
    assert encoded.dtype == np.int32
    assert encoded.shape == (plan.packed_row_count, 3)
    assert np.array_equal(encoded[:, 0], plan.image_indices)
    assert np.array_equal(encoded[:, 1], plan.rotation_rows)
    assert np.array_equal(encoded[:, 2] != 0, plan.valid_mask)
    assert np.count_nonzero(encoded[:, 2]) == int(np.sum(counts))


@pytest.mark.unit
def test_flat_row_gather_and_scatter_restore_present_dense_rows_exactly():
    counts = np.asarray([3, 17, 5, 65, 9], dtype=np.int32)
    dense_rotation_count = 256
    plan = build_pool_flat_local_row_plan(
        counts,
        dense_rotation_count,
        pool_size=3,
        exact_local_bucket_radix=4,
        packed_row_count=712,
    )
    dense = np.arange(5 * dense_rotation_count * 2, dtype=np.float32).reshape(5, dense_rotation_count, 2)
    gathered = gather_flat_local_rows(
        jnp.asarray(dense),
        jnp.asarray(plan.image_indices),
        jnp.asarray(plan.rotation_rows),
    )
    gathered = gathered.at[jnp.logical_not(jnp.asarray(plan.present_mask))].set(
        jnp.nan,
    )
    restored = np.asarray(
        scatter_flat_local_rows(
            gathered,
            plan.image_indices,
            plan.rotation_rows,
            plan.present_mask,
            batch_size=plan.batch_size,
            dense_rotation_count=plan.dense_rotation_count,
            fill_value=-1.0,
        ),
    )

    expected_present = np.zeros(dense.shape[:2], dtype=bool)
    expected_present[
        plan.image_indices[plan.present_mask],
        plan.rotation_rows[plan.present_mask],
    ] = True
    assert np.array_equal(restored[expected_present], dense[expected_present])
    assert np.all(restored[~expected_present] == -1.0)
    assert not np.any(np.isnan(restored))


@pytest.mark.unit
def test_dense_to_flat_lookup_gathers_final_rows_in_requested_source_order():
    counts = np.asarray([3, 5], dtype=np.int32)
    plan = build_pool_flat_local_row_plan(
        counts,
        16,
        pool_size=2,
        exact_local_bucket_radix=4,
        packed_row_count=34,
        dense_batch_size=3,
    )
    encoded = encode_flat_local_row_plan(plan)
    lookup = build_dense_to_flat_local_row_lookup(
        encoded,
        batch_size=plan.batch_size,
        dense_rotation_count=plan.dense_rotation_count,
    )

    valid_flat_rows = np.flatnonzero(plan.valid_mask)
    np.testing.assert_array_equal(
        lookup[
            plan.image_indices[valid_flat_rows],
            plan.rotation_rows[valid_flat_rows],
        ],
        valid_flat_rows,
    )
    assert np.all(lookup[2] == -1)
    assert np.all(lookup[0, counts[0] :] == -1)
    assert np.all(lookup[1, counts[1] :] == -1)

    final_dense_rows = np.asarray([[0, 2, 0], [1, 4, 0]], dtype=np.int32)
    final_mask = np.asarray([[True, True, False], [True, True, False]])
    flat_take = map_dense_local_rows_to_flat_rows(
        lookup,
        final_dense_rows,
        final_mask,
    )
    dense_values = np.arange(3 * 16 * 2, dtype=np.float32).reshape(3, 16, 2)
    flat_values = dense_values[plan.image_indices, plan.rotation_rows]
    gathered = flat_values[flat_take]
    gathered = np.where(final_mask[..., None], gathered, 0.0)
    expected = np.take_along_axis(
        dense_values[:2],
        final_dense_rows[..., None],
        axis=1,
    )
    expected = np.where(final_mask[..., None], expected, 0.0)
    np.testing.assert_array_equal(gathered, expected)


@pytest.mark.unit
def test_local_fused_pairs_reuse_compact_source_order_and_map_flat_projection_rows(
    monkeypatch,
):
    rotation_counts = np.asarray([3, 2], dtype=np.int32)
    dense_rotation_count = 16
    plan = build_pool_flat_local_row_plan(
        rotation_counts,
        dense_rotation_count,
        pool_size=2,
        packed_row_count=34,
        dense_batch_size=3,
    )
    encoded = encode_flat_local_row_plan(plan)
    rotation_mask = np.zeros((3, dense_rotation_count), dtype=bool)
    rotation_mask[0, :3] = True
    rotation_mask[1, :2] = True
    sample_mask = np.zeros((3, dense_rotation_count, 3), dtype=bool)
    sample_mask[0, 0, [0, 2]] = True
    sample_mask[0, 2, [1, 2]] = True
    sample_mask[1, 1, [0, 1, 2]] = True
    bucket = LocalBucketSpec(
        image_indices=np.asarray([5, 7, 5], dtype=np.int32),
        bucket_image_count=3,
        bucket_rotation_count=dense_rotation_count,
        actual_rotation_counts=np.asarray([3, 2, 0], dtype=np.int32),
        local_rotation_ids=np.zeros((3, dense_rotation_count), dtype=np.int32),
        local_rotations=np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (3, dense_rotation_count, 3, 3),
        ).copy(),
        local_rotation_log_prior=np.zeros(
            (3, dense_rotation_count),
            dtype=np.float32,
        ),
        local_rotation_mask=rotation_mask,
        translation_log_prior=np.zeros((3, 3), dtype=np.float32),
        local_sample_mask=sample_mask,
    )

    actual = local_em_engine._build_local_fused_pair_fine_arguments(
        bucket,
        encoded,
        np.asarray([True, True, False]),
    )
    shared = sparse_pass2_bucketed.build_compact_pair_index_arrays(
        sample_mask & rotation_mask[:, :, None],
    )

    assert actual["pair_bucket_size"] == shared["pair_bucket_size"] == 16
    for field in (
        "pair_counts",
        "local_rotation_row",
        "translation_idx",
        "pair_mask",
    ):
        np.testing.assert_array_equal(actual[field], shared[field])
    assert actual["valid_pair_count"] == 7
    assert actual["dense_candidate_capacity"] == 144
    assert actual["valid_job_count"] == 7
    assert actual["job_bucket_size"] == 16

    dense_to_flat = build_dense_to_flat_local_row_lookup(
        encoded,
        batch_size=3,
        dense_rotation_count=dense_rotation_count,
    )
    for image_row in range(3):
        count = int(actual["pair_counts"][image_row])
        expected_rotation, expected_translation = np.nonzero(
            sample_mask[image_row] & rotation_mask[image_row, :, None]
        )
        np.testing.assert_array_equal(
            actual["local_rotation_row"][image_row, :count],
            expected_rotation,
        )
        np.testing.assert_array_equal(
            actual["translation_idx"][image_row, :count],
            expected_translation,
        )
        np.testing.assert_array_equal(
            actual["reference_row"][image_row, :count],
            dense_to_flat[image_row, expected_rotation],
        )
        assert np.all(actual["reference_row"][image_row, count:] == -1)

    expected_jobs = np.asarray(
        [
            [0, dense_to_flat[0, 0], 0, 0],
            [0, dense_to_flat[0, 0], 0, 2],
            [0, dense_to_flat[0, 2], 2, 1],
            [0, dense_to_flat[0, 2], 2, 2],
            [1, dense_to_flat[1, 1], 1, 0],
            [1, dense_to_flat[1, 1], 1, 1],
            [1, dense_to_flat[1, 1], 1, 2],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(actual["job_plan"][:7], expected_jobs)
    assert np.all(actual["job_plan"][7:] == -1)

    monkeypatch.setenv("RECOVAR_EXACT_FINE_JOB_BUCKET_QUANTUM", "8192")
    capacities = local_em_engine._plan_local_fine_job_capacities([bucket])
    assert capacities == {(3, dense_rotation_count): 4096}
    stable = local_em_engine._build_local_fused_pair_fine_arguments(
        bucket,
        encoded,
        np.asarray([True, True, False]),
        fine_job_bucket_size=capacities[(3, dense_rotation_count)],
    )
    assert stable["job_plan"].shape == (4096, 4)
    np.testing.assert_array_equal(stable["job_plan"][:7], expected_jobs)
    assert np.all(stable["job_plan"][7:] == -1)


@pytest.mark.unit
def test_dense_to_flat_lookup_fails_closed_on_missing_or_duplicate_live_rows():
    encoded = np.asarray([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
    lookup = build_dense_to_flat_local_row_lookup(
        encoded,
        batch_size=1,
        dense_rotation_count=4,
    )
    with pytest.raises(ValueError, match="absent from the flat score plan"):
        map_dense_local_rows_to_flat_rows(
            lookup,
            np.asarray([[2]], dtype=np.int32),
            np.asarray([[True]]),
        )

    duplicate = np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="duplicate dense coordinates"):
        build_dense_to_flat_local_row_lookup(
            duplicate,
            batch_size=1,
            dense_rotation_count=4,
        )


@pytest.mark.unit
def test_final_row_noise_helpers_equal_dense_zero_row_contract_exactly():
    probs = np.zeros((2, 4, 2), dtype=np.float32)
    probs[0, 0] = [0.5, 0.25]
    probs[0, 2] = [0.25, 0.0]
    probs[1, 1] = [0.5, 0.5]
    shifted = np.asarray(
        [
            [[1 + 1j, 2 + 0j, 0 + 1j], [3 + 1j, 0 + 2j, 2 + 0j]],
            [[2 + 0j, 1 + 1j, 4 + 0j], [0 + 2j, 3 + 1j, 2 + 2j]],
        ],
        dtype=np.complex64,
    )
    ctf2_over_nv = np.asarray([[1.0, 2.0, 0.5], [2.0, 1.0, 0.25]], dtype=np.float32)
    projection = np.asarray(
        [
            [[1 + 0j, 2 + 1j, 1 + 1j], [0, 0, 0], [2 + 0j, 1 + 0j, 0 + 1j], [0, 0, 0]],
            [[0, 0, 0], [1 + 1j, 2 + 0j, 1 + 0j], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.complex64,
    )
    dense_summed = compute_local_weighted_sums(jnp.asarray(probs), jnp.asarray(shifted))
    dense_ctf = compute_local_ctf_sums(jnp.asarray(probs), jnp.asarray(ctf2_over_nv))

    take = np.asarray([[0, 2], [1, 0]], dtype=np.int32)
    mask = np.asarray([[True, True], [True, False]])
    packed_probs = np.take_along_axis(probs, take[..., None], axis=1)
    packed_probs = np.where(mask[..., None], packed_probs, 0.0)
    packed_projection = np.take_along_axis(projection, take[..., None], axis=1)
    packed_projection = np.where(mask[..., None], packed_projection, 0.0)
    packed_summed = compute_local_weighted_sums(
        jnp.asarray(packed_probs),
        jnp.asarray(shifted),
    )
    packed_ctf = compute_local_ctf_sums(
        jnp.asarray(packed_probs),
        jnp.asarray(ctf2_over_nv),
    )
    np.testing.assert_array_equal(
        np.asarray(packed_summed),
        np.where(
            mask[..., None],
            np.take_along_axis(np.asarray(dense_summed), take[..., None], axis=1),
            0.0,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(packed_ctf),
        np.where(
            mask[..., None],
            np.take_along_axis(np.asarray(dense_ctf), take[..., None], axis=1),
            0.0,
        ),
    )

    noise_variance = jnp.asarray([1.0, 2.0, 0.5], dtype=jnp.float32)
    shell_indices = jnp.asarray([0, 1, 1], dtype=jnp.int32)
    dense_noise = compute_noise_block(
        jnp.asarray(projection).reshape(-1, 3),
        (jnp.abs(jnp.asarray(projection)) ** 2).reshape(-1, 3),
        dense_summed.reshape(-1, 3),
        dense_ctf.reshape(-1, 3),
        noise_variance,
        shell_indices,
        2,
    )
    packed_noise = compute_noise_block(
        jnp.asarray(packed_projection).reshape(-1, 3),
        (jnp.abs(jnp.asarray(packed_projection)) ** 2).reshape(-1, 3),
        packed_summed.reshape(-1, 3),
        packed_ctf.reshape(-1, 3),
        noise_variance,
        shell_indices,
        2,
    )
    for dense_value, packed_value in zip(dense_noise, packed_noise, strict=True):
        np.testing.assert_array_equal(np.asarray(packed_value), np.asarray(dense_value))

    dense_norm = compute_norm_residual_per_image(
        jnp.asarray(projection),
        jnp.abs(jnp.asarray(projection)) ** 2,
        dense_summed,
        dense_ctf,
        noise_variance,
    )
    packed_norm = compute_norm_residual_per_image(
        jnp.asarray(packed_projection),
        jnp.abs(jnp.asarray(packed_projection)) ** 2,
        packed_summed,
        packed_ctf,
        noise_variance,
    )
    np.testing.assert_array_equal(np.asarray(packed_norm), np.asarray(dense_norm))

    dense_scale = compute_scale_correction_terms_per_image(
        jnp.asarray(projection),
        jnp.abs(jnp.asarray(projection)) ** 2,
        dense_summed,
        dense_ctf,
        noise_variance,
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
    )
    packed_scale = compute_scale_correction_terms_per_image(
        jnp.asarray(packed_projection),
        jnp.abs(jnp.asarray(packed_projection)) ** 2,
        packed_summed,
        packed_ctf,
        noise_variance,
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
    )
    for dense_value, packed_value in zip(dense_scale, packed_scale, strict=True):
        np.testing.assert_array_equal(np.asarray(packed_value), np.asarray(dense_value))


@pytest.mark.unit
def test_flat_row_capacity_reuses_one_shape_per_dense_bucket_abi():
    first = SimpleNamespace(
        image_indices=np.arange(3, dtype=np.int32),
        bucket_image_count=4,
        bucket_rotation_count=256,
        actual_rotation_counts=np.asarray([3, 17, 5], dtype=np.int32),
    )
    second = SimpleNamespace(
        image_indices=np.arange(2, dtype=np.int32),
        bucket_image_count=4,
        bucket_rotation_count=256,
        actual_rotation_counts=np.asarray([65, 9], dtype=np.int32),
    )

    capacities = local_em_engine._plan_flat_local_row_capacities(
        (first, second),
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    encoded = local_em_engine._build_flat_local_row_argument(
        first,
        capacities,
        dense_batch_size=4,
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )

    assert capacities == {(4, 256): 256}
    assert encoded.shape == (256, 3)
    assert np.count_nonzero(encoded[:, 2]) == int(
        np.sum(first.actual_rotation_counts),
    )


@pytest.mark.unit
def test_stable_flat_row_capacity_reuses_mature_rectangular_bucket_abi():
    first = SimpleNamespace(
        image_indices=np.arange(3, dtype=np.int32),
        bucket_image_count=4,
        bucket_rotation_count=256,
        actual_rotation_counts=np.asarray([3, 17, 5], dtype=np.int32),
    )
    second = SimpleNamespace(
        image_indices=np.arange(2, dtype=np.int32),
        bucket_image_count=4,
        bucket_rotation_count=256,
        actual_rotation_counts=np.asarray([65, 9], dtype=np.int32),
    )

    capacities = local_em_engine._plan_flat_local_row_capacities(
        (first, second),
        rotation_block_size=128,
        exact_local_bucket_radix=4,
        stable_rectangular_capacity=True,
    )
    first_encoded = local_em_engine._build_flat_local_row_argument(
        first,
        capacities,
        dense_batch_size=4,
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    second_encoded = local_em_engine._build_flat_local_row_argument(
        second,
        capacities,
        dense_batch_size=4,
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    ordinary_capacities = local_em_engine._plan_flat_local_row_capacities(
        (first, second),
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    ordinary_first = local_em_engine._build_flat_local_row_argument(
        first,
        ordinary_capacities,
        dense_batch_size=4,
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    first_required_rows = build_pool_flat_local_row_plan(
        first.actual_rotation_counts,
        first.bucket_rotation_count,
        pool_size=3,
        rotation_block_size=128,
        exact_local_bucket_radix=4,
        dense_batch_size=4,
    ).packed_row_count

    assert capacities == {(4, 256): 4 * 256}
    assert first_encoded.shape == second_encoded.shape == (4 * 256, 3)
    np.testing.assert_array_equal(
        first_encoded[: ordinary_first.shape[0]],
        ordinary_first,
    )
    assert np.count_nonzero(first_encoded[:, 2]) == 25
    assert np.count_nonzero(second_encoded[:, 2]) == 74
    np.testing.assert_array_equal(
        first_encoded[first_required_rows:],
        np.zeros_like(first_encoded[first_required_rows:]),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("counts", "dense_rotation_count", "kwargs", "message"),
    [
        ([], 16, {}, "at least one image"),
        ([1], 0, {}, "dense_rotation_count"),
        ([0], 16, {}, "rotation counts"),
        ([17], 16, {}, "rotation counts"),
        ([1], 16, {"pool_size": 0}, "pool_size"),
        ([1, 1], 16, {"dense_batch_size": 1}, "dense_batch_size"),
        ([17], 64, {"packed_row_count": 63}, "smaller than required"),
    ],
)
def test_pool_flat_row_plan_rejects_invalid_shapes(counts, dense_rotation_count, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_pool_flat_local_row_plan(
            counts,
            dense_rotation_count,
            exact_local_bucket_radix=4,
            **kwargs,
        )
