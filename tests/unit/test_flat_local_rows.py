from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.helpers.projection import (
    compute_noise_block,
    compute_norm_residual_per_image,
    compute_scale_correction_terms_per_image,
)
from recovar.em.local import local_bucket_stages
from recovar.em.local.flat_local_rows import (
    build_dense_to_flat_local_row_lookup,
    build_pool_flat_local_row_plan,
    encode_flat_local_row_plan,
    map_dense_local_rows_to_flat_rows,
    scatter_flat_local_rows,
)
from recovar.em.local.local_backprojection import compute_local_ctf_sums, compute_local_weighted_sums
from recovar.em.local.local_layout import LocalBucketSpec
from recovar.em.scoring import compact_candidates


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
    gathered = jnp.asarray(dense)[jnp.asarray(plan.image_indices), jnp.asarray(plan.rotation_rows)]
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

    actual = local_bucket_stages._build_local_fused_pair_fine_arguments(
        bucket,
        encoded,
        np.asarray([True, True, False]),
    )
    shared = compact_candidates.build_compact_pair_index_arrays(
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
    capacities = local_bucket_stages._plan_local_fine_job_capacities([bucket])
    assert capacities == {(3, dense_rotation_count): 144}
    stable = local_bucket_stages._build_local_fused_pair_fine_arguments(
        bucket,
        encoded,
        np.asarray([True, True, False]),
        fine_job_bucket_size=capacities[(3, dense_rotation_count)],
    )
    assert stable["job_plan"].shape == (144, 4)
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

    capacities = local_bucket_stages._plan_flat_local_row_capacities(
        (first, second),
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    encoded = local_bucket_stages._build_flat_local_row_argument(
        first,
        capacities,
        dense_batch_size=4,
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )

    assert capacities.capacities == {(4, 256): 256}
    assert capacities.pool_size == 3
    # 3 images pooled at 64 rows, then 2 pooled at 128: the shared shape is the larger
    assert capacities.required_rows == 192 + 256
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

    capacities = local_bucket_stages._plan_flat_local_row_capacities(
        (first, second),
        rotation_block_size=128,
        exact_local_bucket_radix=4,
        stable_rectangular_capacity=True,
    )
    first_encoded = local_bucket_stages._build_flat_local_row_argument(
        first,
        capacities,
        dense_batch_size=4,
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    second_encoded = local_bucket_stages._build_flat_local_row_argument(
        second,
        capacities,
        dense_batch_size=4,
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    ordinary_capacities = local_bucket_stages._plan_flat_local_row_capacities(
        (first, second),
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    ordinary_first = local_bucket_stages._build_flat_local_row_argument(
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

    assert capacities.capacities == {(4, 256): 4 * 256}
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


@pytest.mark.unit
def test_class_flat_rows_cover_every_candidate_exactly_once():
    """Class-segmented rows pack without losing or duplicating a candidate.

    The dense axis of a class-segmented bucket is class-major, so a packed row's
    dense index must decode as ``class * segment + rotation`` and every real
    (image, class, rotation) must appear exactly once.
    """
    from recovar.em.local.flat_local_rows import build_pool_flat_local_row_plan_for_classes

    counts = np.asarray(
        [[3, 200, 5, 1], [17, 180, 9, 2], [5, 220, 4, 1],
         [400, 6, 3, 2], [9, 5, 2, 1], [7, 190, 6, 2]], dtype=np.int32,
    )
    segment = 512
    plan = build_pool_flat_local_row_plan_for_classes(
        counts, segment, pool_size=3, exact_local_bucket_radix=2,
    )

    seen = set()
    for row in range(plan.packed_row_count):
        if not plan.present_mask[row] or not plan.valid_mask[row]:
            continue
        image = int(plan.image_indices[row])
        dense = int(plan.rotation_rows[row])
        class_index, rotation = divmod(dense, segment)
        assert rotation < counts[image, class_index]
        assert (image, class_index, rotation) not in seen
        seen.add((image, class_index, rotation))
    assert len(seen) == int(counts.sum())


@pytest.mark.unit
def test_class_flat_rows_pack_far_tighter_than_the_rectangular_layout():
    from recovar.em.local.flat_local_rows import build_pool_flat_local_row_plan_for_classes

    counts = np.asarray([[3, 200, 5, 1], [17, 180, 9, 2], [5, 220, 4, 1]], dtype=np.int32)
    segment = 512
    plan = build_pool_flat_local_row_plan_for_classes(
        counts, segment, pool_size=3, exact_local_bucket_radix=2,
    )

    rectangular = counts.shape[0] * counts.shape[1] * segment
    assert plan.packed_row_count < rectangular / 3
    # and the dense axis it maps into is still the full class-major axis
    assert plan.dense_rotation_count == segment * counts.shape[1]
    assert int(plan.rotation_rows[plan.present_mask].max()) < plan.dense_rotation_count


@pytest.mark.unit
def test_class_flat_rows_skip_classes_no_image_in_the_pool_uses():
    """A class with no candidates anywhere in a pool must contribute no rows."""
    from recovar.em.local.flat_local_rows import build_pool_flat_local_row_plan_for_classes

    counts = np.asarray([[4, 0, 6], [5, 0, 7]], dtype=np.int32)
    segment = 64
    plan = build_pool_flat_local_row_plan_for_classes(
        counts, segment, pool_size=3, exact_local_bucket_radix=2,
    )

    classes = {int(d) // segment for d in plan.rotation_rows[plan.present_mask]}
    assert classes == {0, 2}


@pytest.mark.unit
def test_class_flat_rows_static_capacity_marks_padding_absent():
    from recovar.em.local.flat_local_rows import build_pool_flat_local_row_plan_for_classes

    counts = np.asarray([[4, 6], [5, 7]], dtype=np.int32)
    natural = build_pool_flat_local_row_plan_for_classes(counts, 64, pool_size=3)
    padded = build_pool_flat_local_row_plan_for_classes(
        counts, 64, pool_size=3, packed_row_count=natural.packed_row_count + 32,
    )

    assert padded.packed_row_count == natural.packed_row_count + 32
    assert int(padded.present_mask.sum()) == natural.packed_row_count
    assert not padded.valid_mask[natural.packed_row_count:].any()


@pytest.mark.unit
def test_bucket_stage_flat_plan_dispatches_on_class_segmented_buckets():
    """The bucket-stage helpers must use the class-aware builder at K>1.

    A previous change in this area was applied only to the single-class path and
    was dead code at K=4, so this drives the real class-segmented bucket through
    the stage helpers rather than the plan builder directly.
    """
    from recovar.em import sampling
    from recovar.em.local import local_layout
    from recovar.em.local.local_bucket_stages import (
        _build_flat_local_row_argument,
        _build_pool_flat_plan_for_bucket,
        _plan_flat_local_row_capacities,
    )

    n = sampling.rotation_grid_size(0)
    n_classes = 3
    layouts = [
        local_layout.build_pass2_hypothesis_layout(
            [np.arange(2), np.arange(9), np.arange(n)],
            n_coarse_rotations=n, n_coarse_translations=1, nside_level=0,
            translations=np.zeros((1, 2), dtype=np.float32),
            translation_step=1.0, oversampling_order=1,
        )
        for _ in range(n_classes)
    ]
    # the plan must be built with the same rotation_block_size the bucket used:
    # it sets the quantizer's engine cap, and a mismatch can size a class wider
    # than its own segment.
    rotation_block_size = 5000
    buckets = local_layout.bucket_class_local_hypothesis_layouts(
        layouts, np.zeros(n_classes), 3, rotation_block_size,
    )
    assert buckets and all(int(b.n_classes) == n_classes for b in buckets)

    capacities = _plan_flat_local_row_capacities(
        buckets, rotation_block_size=rotation_block_size, exact_local_bucket_radix=2,
    )
    assert capacities.capacities

    for bucket in buckets:
        physical = int(np.asarray(bucket.image_indices).shape[0])
        dense_batch = max(physical, int(bucket.bucket_image_count))
        encoded = _build_flat_local_row_argument(
            bucket, capacities,
            dense_batch_size=dense_batch,
            rotation_block_size=rotation_block_size,
            exact_local_bucket_radix=2,
        )
        # packed rows must stay inside the bucket's own dense class-major axis
        assert encoded.shape[1] == 3
        assert int(encoded[:, 1].max()) < int(bucket.bucket_rotation_count)
        # and must be a genuine reduction against the rectangular layout
        assert encoded.shape[0] <= dense_batch * int(bucket.bucket_rotation_count)

    # the whole point: packed capacity is smaller than the rectangular ABI
    rectangular = sum(
        max(int(np.asarray(b.image_indices).shape[0]), int(b.bucket_image_count))
        * int(b.bucket_rotation_count)
        for b in buckets
    )
    packed = sum(capacities.capacities.values())
    assert packed < rectangular

    # Prove the class-aware builder actually ran rather than silently falling back.
    # The single-class builder treats actual_rotation_counts as counts on one
    # rotation axis; on a class bucket those are summed across classes, so
    # quantizing them overflows the dense axis and it refuses outright.
    from recovar.em.local.flat_local_rows import build_pool_flat_local_row_plan

    bucket = buckets[0]
    physical = int(np.asarray(bucket.image_indices).shape[0])
    dense_batch = max(physical, int(bucket.bucket_image_count))
    with pytest.raises(ValueError, match="dense rotation axis"):
        build_pool_flat_local_row_plan(
            np.asarray(bucket.actual_rotation_counts, dtype=np.int32),
            int(bucket.bucket_rotation_count),
            pool_size=3,
            rotation_block_size=rotation_block_size,
            exact_local_bucket_radix=2,
            dense_batch_size=dense_batch,
        )
    # the class-aware builder handles the same bucket
    klass = _build_pool_flat_plan_for_bucket(
        bucket,
        dense_rotation_count=int(bucket.bucket_rotation_count),
        rotation_block_size=rotation_block_size,
        exact_local_bucket_radix=2,
        dense_batch_size=dense_batch,
        pool_size=3,
        round_row_widths=True,
    )
    assert klass.packed_row_count <= dense_batch * int(bucket.bucket_rotation_count)


@pytest.mark.unit
def test_class_flat_rows_are_class_major_with_block_sizes():
    """Projection slices one class at a time, so a class's rows must be contiguous.

    Flat rows interleave classes by construction, and projecting an interleaved list
    against a single class volume would silently use the wrong volume for most rows.
    Emitting class-major lets each class's block be projected with its own volume.
    """
    from recovar.em.local.flat_local_rows import build_pool_flat_local_row_plan_for_classes

    counts = np.asarray(
        [[3, 200, 5], [17, 180, 9], [5, 220, 4], [400, 6, 3]], dtype=np.int32,
    )
    segment = 512
    plan = build_pool_flat_local_row_plan_for_classes(
        counts, segment, pool_size=3, exact_local_bucket_radix=2,
    )

    assert plan.class_row_counts is not None
    assert len(plan.class_row_counts) == counts.shape[1]
    assert sum(plan.class_row_counts) == int(plan.present_mask.sum())

    # each class's block is contiguous and holds only that class's rows
    start = 0
    for class_index, block in enumerate(plan.class_row_counts):
        block_rows = plan.rotation_rows[start:start + block]
        assert np.all(block_rows // segment == class_index)
        start += block

    # and every real candidate still appears exactly once
    seen = set()
    for row in range(plan.packed_row_count):
        if not plan.present_mask[row] or not plan.valid_mask[row]:
            continue
        image = int(plan.image_indices[row])
        class_index, rotation = divmod(int(plan.rotation_rows[row]), segment)
        assert (image, class_index, rotation) not in seen
        seen.add((image, class_index, rotation))
    assert len(seen) == int(counts.sum())


@pytest.mark.unit
def test_class_flat_rows_quantize_with_the_bucketer_large_quantum():
    """A class must be quantized with the same large-bucket quantum as its segment.

    The segment width is produced by the bucketer's resolved large-bucket quantum. If
    the plan quantizes a class with a different one, a class above the engine cap can
    round above its own segment and the plan is rejected. This reproduced in a real
    K=4 run as "a pool bucket exceeds the enclosing class segment".
    """
    from recovar.em.local.flat_local_rows import build_pool_flat_local_row_plan_for_classes
    from recovar.em.local.local_layout import (
        _exact_bucket_rotation_size,
        _exact_local_large_bucket_quantum,
    )

    rotation_block_size = 5000
    quantum = _exact_local_large_bucket_quantum(rotation_block_size, None)
    # a count above the engine cap, so the large-quantum path decides the width
    count = quantum + 1
    segment = _exact_bucket_rotation_size(
        count, rotation_block_size, large_bucket_quantum=quantum,
    )
    assert segment > quantum  # the case that matters

    counts = np.asarray([[count, 4], [count - 1, 6]], dtype=np.int32)
    plan = build_pool_flat_local_row_plan_for_classes(
        counts, segment, pool_size=3, rotation_block_size=rotation_block_size,
    )

    assert plan.class_row_counts is not None
    # no class block may reach past its own segment on the dense axis
    for class_index in range(counts.shape[1]):
        block = plan.rotation_rows[plan.present_mask] // segment == class_index
        rows_in_class = plan.rotation_rows[plan.present_mask][block] % segment
        if rows_in_class.size:
            assert int(rows_in_class.max()) < segment


@pytest.mark.unit
@pytest.mark.parametrize("rows", [256, 255, 128, 3, 2, 1])
def test_relion_ctf_row_shift_matches_negated_fftshift(rows):
    """The in-place CTF row shift must equal `-fftshift(native, axes=0)` exactly.

    `relion_ctf` builds each particle's cached CTF row by shifting RELION's standard-order
    y axis and flipping sign for RECOVAR's forward-model convention. That ran once per
    particle and cost ~40 s of a 100k run, so it now writes both row blocks into one
    buffer instead of allocating for the roll and again for the negation. The split point
    is `rows - rows // 2`, which differs from `rows // 2` when rows is odd.
    """
    native = np.random.default_rng(rows).standard_normal((rows, 129))

    expected = (-np.fft.fftshift(native, axes=0)).reshape(-1)

    shift = rows // 2
    split = rows - shift
    shifted = np.empty_like(native)
    np.negative(native[split:], out=shifted[:shift])
    np.negative(native[:split], out=shifted[shift:])

    np.testing.assert_array_equal(shifted.reshape(-1), expected)


@pytest.mark.unit
def test_flat_local_pool_size_changes_padding_only(monkeypatch):
    """The pool size is a shape knob: it must not change which rows are valid.

    Pooling consecutive images onto one rotation width is what makes a packed
    bucket rectangular enough to reuse a compiled shape. It costs padded rows and
    buys nothing scientific, so the valid (image, rotation) pairs must be identical
    at every pool size while the row count falls as the pool shrinks.
    """

    from recovar.em.local.flat_local_rows import (
        build_pool_flat_local_row_plan,
        resolve_flat_local_pool_size,
    )

    counts = np.asarray([5, 300, 7, 9, 600, 11, 13, 17], dtype=np.int32)

    def valid_pairs(pool_size):
        plan = build_pool_flat_local_row_plan(
            counts,
            1024,
            pool_size=pool_size,
            rotation_block_size=1024,
        )
        valid = plan.valid_mask
        return (
            set(zip(plan.image_indices[valid].tolist(), plan.rotation_rows[valid].tolist())),
            plan.packed_row_count,
        )

    expected = {
        (image, rotation)
        for image, count in enumerate(counts.tolist())
        for rotation in range(count)
    }
    pairs_one, rows_one = valid_pairs(1)
    pairs_three, rows_three = valid_pairs(3)
    assert pairs_one == expected
    assert pairs_three == expected
    assert rows_one < rows_three

    monkeypatch.setenv("RECOVAR_EXACT_LOCAL_FLAT_POOL_SIZE", "1")
    assert resolve_flat_local_pool_size() == 1
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_FLAT_POOL_SIZE")
    assert resolve_flat_local_pool_size() == 3
    assert resolve_flat_local_pool_size(2) == 2
    monkeypatch.setenv("RECOVAR_EXACT_LOCAL_FLAT_POOL_SIZE", "0")
    with pytest.raises(ValueError, match="positive integer"):
        resolve_flat_local_pool_size()


@pytest.mark.unit
def test_flat_local_row_rounding_off_packs_exactly_the_needed_rows(monkeypatch):
    """With rounding off and no pooling, a packed plan holds no padding at all."""

    from recovar.em.local.flat_local_rows import (
        build_pool_flat_local_row_plan,
        resolve_flat_local_row_rounding,
    )

    counts = np.asarray([5, 300, 7, 9, 600, 11, 13, 17], dtype=np.int32)
    exact = build_pool_flat_local_row_plan(
        counts, 1024, pool_size=1, rotation_block_size=1024, round_row_widths=False,
    )
    rounded = build_pool_flat_local_row_plan(
        counts, 1024, pool_size=1, rotation_block_size=1024, round_row_widths=True,
    )

    assert exact.packed_row_count == int(counts.sum())
    assert bool(exact.valid_mask.all())
    assert rounded.packed_row_count > exact.packed_row_count

    expected = {
        (image, rotation)
        for image, count in enumerate(counts.tolist())
        for rotation in range(count)
    }
    for plan in (exact, rounded):
        valid = plan.valid_mask
        assert set(
            zip(plan.image_indices[valid].tolist(), plan.rotation_rows[valid].tolist())
        ) == expected

    monkeypatch.setenv("RECOVAR_EXACT_LOCAL_FLAT_ROW_ROUNDING", "0")
    assert resolve_flat_local_row_rounding() is False
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_FLAT_ROW_ROUNDING")
    assert resolve_flat_local_row_rounding() is True
    monkeypatch.setenv("RECOVAR_EXACT_LOCAL_FLAT_ROW_ROUNDING", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        resolve_flat_local_row_rounding()


@pytest.mark.unit
def test_xhalf_projection_budget_follows_device_memory(monkeypatch):
    """The x-half projection budget belongs to the device, not to a constant.

    The 40 M default was tuned on a 384-box run and is far too small once the Fourier
    window opens: at K=1 100k/256 it yields 1211 hypotheses per microbatch and 3334
    buckets of three images, and lifting it moved six exclusive 200-mini-batch runs from
    7352 s to 6042 s with quality unchanged. It is now derived from free device memory,
    and these are the properties that keep that safe.
    """

    from recovar.em.local import local_batch_planning as planning

    floor = planning.EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS
    bytes_per = planning.EXACT_LOCAL_XHALF_PROJECTION_BYTES_PER_ROW_PIXEL

    # A device with room uses it, at the conservative per-row-pixel cost.
    free = 64 * 2**30
    got = planning._exact_local_xhalf_projection_target_row_pixels(
        runtime_free_memory_bytes=free,
    )
    expected = int(free * planning.EXACT_LOCAL_XHALF_PROJECTION_FREE_MEMORY_FRACTION
                   // bytes_per)
    assert got == expected
    assert got > floor

    # A device without room is never taken below the historical floor.
    assert planning._exact_local_xhalf_projection_target_row_pixels(
        runtime_free_memory_bytes=1 << 20,
    ) == floor
    # Nor is one where the probe fails.
    assert planning._exact_local_xhalf_projection_target_row_pixels(
        runtime_free_memory_bytes=None,
    ) >= floor

    # An explicit request wins outright, which is how the sweeps were run.
    monkeypatch.setenv(
        planning.EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV, "320000000",
    )
    assert planning._exact_local_xhalf_projection_target_row_pixels(
        runtime_free_memory_bytes=free,
    ) == 320_000_000
    monkeypatch.delenv(planning.EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV)

    # Deterministic reductions decline the probe: a budget that reads live allocator
    # state would change accumulation grouping between otherwise identical runs.
    monkeypatch.setattr(planning, "deterministic_reductions_enabled", lambda: True)
    assert planning._exact_local_xhalf_projection_target_row_pixels(
        runtime_free_memory_bytes=free,
    ) == floor


@pytest.mark.unit
def test_score_tile_free_memory_fraction_is_resolvable(monkeypatch):
    """The cap that actually binds max_hypotheses must be sweepable.

    On the K=1 100k/256 production schedule the score-tile cap, not the projection
    row budget, is what sets `max_hypotheses_per_microbatch`: forcing the budget to
    320 M row-pixels leaves it at 4312 with 667 buckets, and neither the tail nor
    the projection cap logs a reduction. It was a hardcoded constant, so it could
    not be measured without editing the source.
    """

    from recovar.em.local import local_batch_planning as planning

    default = planning.EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION
    env = planning.EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION_ENV

    assert planning._exact_local_score_tile_free_memory_fraction() == default

    monkeypatch.setenv(env, "0.45")
    assert planning._exact_local_score_tile_free_memory_fraction() == 0.45

    # An unparsable value warns and keeps the default rather than failing a long run.
    monkeypatch.setenv(env, "not-a-number")
    assert planning._exact_local_score_tile_free_memory_fraction() == default

    # An out-of-range value is a mistake worth stopping for.
    for bad in ("0", "-0.1", "1.5"):
        monkeypatch.setenv(env, bad)
        with pytest.raises(ValueError, match="must lie in"):
            planning._exact_local_score_tile_free_memory_fraction()
