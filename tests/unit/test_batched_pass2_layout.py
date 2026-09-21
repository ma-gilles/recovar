"""Exact batch invariance for shared pass-2 orientation construction."""

import weakref

import numpy as np
import pytest

from recovar.em import sampling
from recovar.em.local import local_layout

pytestmark = pytest.mark.unit


def test_pass2_buckets_reuse_scoring_rotations_without_mstep_override():
    layout = local_layout.build_pass2_hypothesis_layout(
        [None, np.array([0])],
        n_coarse_rotations=sampling.rotation_grid_size(0),
        n_coarse_translations=1, nside_level=0,
        translations=np.zeros((1, 2), dtype=np.float32),
        translation_step=1.0, oversampling_order=1,
    )
    buckets = local_layout.bucket_local_hypothesis_layout(layout, 2, 32)
    planned = local_layout.LocalBucketSequence(layout, local_layout.plan_local_hypothesis_buckets(layout, 2, 32))
    first = planned[0]
    released = weakref.ref(first)
    del first
    assert released() is None  # The sequence must not cache materialized buckets.
    assert len(planned) == len(buckets)
    np.testing.assert_array_equal(planned[-1].local_rotations, buckets[-1].local_rotations)
    np.testing.assert_array_equal(planned[1:][0].local_rotations, buckets[1].local_rotations)
    for bucket in buckets:
        assert bucket.local_mstep_rotations is None
        assert local_layout._local_mstep_rotations(bucket) is bucket.local_rotations
        for row, image in enumerate(bucket.image_indices):
            start, stop = layout.rotation_offsets[image:image + 2]
            np.testing.assert_array_equal(
                local_layout._local_mstep_rotations(bucket)[row, :stop - start],
                layout.rotations_flat[start:stop],
            )


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("index_order", ["recovar", "relion", "relion_hidden"])
@pytest.mark.parametrize("perturbation", [0.0, -0.43747416138648987])
def test_batched_pass2_matches_independent_image_layouts(order, index_order, perturbation):
    n_rotations = sampling.rotation_grid_size(0)
    samples = [
        np.array([2 * 3, 2 * 3 + 1, 5 * 3 + 2, 2 * 3]),
        np.array([1 * 3 + 2, 5 * 3, (n_rotations - 1) * 3]),
        np.array([], dtype=np.int64),
        None,
    ]
    kwargs = dict(
        n_coarse_rotations=n_rotations,
        n_coarse_translations=3,
        nside_level=0,
        translations=np.array([[-1, 0], [0, 0], [1, 0]], dtype=np.float32),
        translation_step=1.0,
        rotation_log_prior=np.linspace(-2, -1, n_rotations, dtype=np.float32),
        translation_log_prior=np.array([-3, -2, -1], dtype=np.float32),
        oversampling_order=order,
        random_perturbation=perturbation,
        rotation_index_order=index_order,
        allow_empty=True,
    )
    batched = local_layout.build_pass2_hypothesis_layout(samples, **kwargs)
    for image, sample in enumerate(samples):
        single = local_layout.build_pass2_hypothesis_layout([sample], **kwargs)
        start, stop = batched.rotation_offsets[image : image + 2]
        assert batched.rotation_counts[image] == single.rotation_counts[0]
        for field in (
            "rotations_flat", "rotation_ids_flat", "rotation_posterior_ids_flat",
            "rotation_log_priors_flat", "sample_mask_bits",
        ):
            np.testing.assert_array_equal(getattr(batched, field)[start:stop], getattr(single, field))
        np.testing.assert_array_equal(batched.translation_grid, single.translation_grid)
        # Compare matrices directly to the mature per-image sampler as well.
        parents = (np.arange(n_rotations) if sample is None else
                   np.unique(sample // 3) if sample.size else np.array([0]))
        rotations, _, ids = sampling.get_oversampled_rotation_grid_from_samples(
            parents, 0, order, random_perturbation=perturbation,
            return_rotation_indices=True, rotation_index_order=index_order,
        )
        np.testing.assert_array_equal(batched.rotations_flat[start:stop], rotations)
        np.testing.assert_array_equal(batched.rotation_ids_flat[start:stop], ids)


def test_pass2_generates_only_requested_parent_union_once(monkeypatch):
    calls = []
    original = local_layout.get_oversampled_rotation_grid_from_samples

    def record(parents, *args, **kwargs):
        calls.append(np.asarray(parents).copy())
        return original(parents, *args, **kwargs)

    monkeypatch.setattr(local_layout, "get_oversampled_rotation_grid_from_samples", record)
    local_layout.build_pass2_hypothesis_layout(
        [np.array([3, 9]), np.array([9, 15]), np.array([3])],
        n_coarse_rotations=2**40, n_coarse_translations=3, nside_level=0,
        translations=np.array([[-1, 0], [0, 0], [1, 0]], dtype=np.float32),
        translation_step=1.0, oversampling_order=1,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [1, 3, 5])


@pytest.mark.parametrize("n_translations", [0, 1, 7, 8, 9, 29, 116])
def test_packed_mask_preserves_boolean_rows_and_bucket_padding(n_translations):
    mask = np.random.default_rng(91).integers(0, 2, (10, n_translations), dtype=np.uint8).view(bool)
    layout = local_layout.LocalHypothesisLayout(
        n_global_rotations=1, n_pixels=1, n_psi=1,
        rotation_offsets=np.array([0, 3, 10]), rotation_counts=np.array([3, 7]),
        rotation_ids_flat=np.zeros(10, dtype=np.int64),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (10, 3, 3)),
        rotation_log_priors_flat=np.zeros(10, dtype=np.float32),
        translation_grid=np.zeros((n_translations, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, n_translations), dtype=np.float32),
        sample_mask_bits=np.packbits(mask, axis=1, bitorder="little"),
    )
    assert layout.sample_mask_bits.nbytes == 10 * ((n_translations + 7) // 8)
    np.testing.assert_array_equal(layout.sample_mask_rows(), mask)
    np.testing.assert_array_equal(layout.sample_mask_rows(3, 8), mask[3:8])
    for bucket in local_layout.bucket_local_hypothesis_layout(layout, 2, 16):
        assert bucket.local_sample_mask.dtype == np.bool_
        for row, image in enumerate(bucket.image_indices):
            start, stop = layout.rotation_offsets[image:image + 2]
            count = stop - start
            np.testing.assert_array_equal(bucket.local_sample_mask[row, :count], mask[start:stop])
            assert not bucket.local_sample_mask[row, count:].any()


def _unequal_support_layout():
    """A layout whose images have deliberately unequal local supports."""
    n = sampling.rotation_grid_size(0)
    return local_layout.build_pass2_hypothesis_layout(
        [np.arange(1), np.arange(3), np.arange(n)],
        n_coarse_rotations=n,
        n_coarse_translations=1,
        nside_level=0,
        translations=np.zeros((1, 2), dtype=np.float32),
        translation_step=1.0,
        oversampling_order=1,
    )


def test_bucket_unification_applies_while_the_padding_it_adds_is_bounded(monkeypatch):
    monkeypatch.setenv(local_layout.EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS_ENV, str(10**9))
    layout = _unequal_support_layout()

    plans = local_layout.plan_local_hypothesis_buckets(layout, 2, 32, unify_bucket_sizes=True)

    assert len({plan.bucket_rotation_count for plan in plans}) == 1


def test_bucket_unification_is_dropped_when_it_would_add_too_many_rows(monkeypatch):
    """Unification pads every image to the largest neighborhood; past a bound that costs
    more than the extra compiled shapes it avoids. Measured on K=4 100k/256, where it
    turned 54.1 M padded rows into 204.8 M and doubled pass-2 time at the all-data
    iteration."""
    monkeypatch.setenv(local_layout.EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS_ENV, "0")
    layout = _unequal_support_layout()

    plans = local_layout.plan_local_hypothesis_buckets(layout, 2, 32, unify_bucket_sizes=True)

    sizes = {plan.bucket_rotation_count for plan in plans}
    assert len(sizes) > 1
    # Never below the true per-image neighborhood cardinality.
    for plan in plans:
        assert plan.bucket_rotation_count >= int(np.max(plan.actual_rotation_counts))


def test_kclass_bucket_unification_is_dropped_when_it_would_add_too_many_rows(monkeypatch):
    """K>1 uses its own bucketer, so the bound must be enforced there too.

    ``local_em_engine`` routes ``n_classes > 1`` through
    ``bucket_class_local_hypothesis_layouts`` rather than the plan-then-materialize
    path, so a policy applied only to the single-class planner is dead code at K=4 --
    which is exactly what a 100k/256 K=4 arm showed, unchanged at 12500 buckets and
    204.8 M padded rows.
    """
    n_classes = 4
    layouts = [_unequal_support_layout() for _ in range(n_classes)]
    priors = np.zeros(n_classes)

    monkeypatch.setenv(local_layout.EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS_ENV, str(10**12))
    unified = local_layout.bucket_class_local_hypothesis_layouts(
        layouts, priors, 2, 32, unify_bucket_sizes=True,
    )
    assert len({bucket.bucket_rotation_count for bucket in unified}) == 1

    monkeypatch.setenv(local_layout.EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS_ENV, "0")
    split = local_layout.bucket_class_local_hypothesis_layouts(
        layouts, priors, 2, 32, unify_bucket_sizes=True,
    )
    assert len({bucket.bucket_rotation_count for bucket in split}) > 1
    for bucket in split:
        # ``actual_rotation_counts`` on a class-segmented bucket is the total across
        # all class segments, so it is bounded by the whole bucket width.
        assert bucket.bucket_rotation_count % n_classes == 0
        assert bucket.bucket_rotation_count >= int(np.max(bucket.actual_rotation_counts))


def test_bucket_size_class_cap_rounds_up_and_never_truncates(monkeypatch):
    """Capping distinct sizes trades padding back for fewer compiled shapes.

    It must only ever round a bucket UP: sizing below an image's true neighborhood
    would silently drop candidate rotations.
    """
    monkeypatch.setenv(local_layout.EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS_ENV, "0")
    layout = _unequal_support_layout()

    monkeypatch.setenv(local_layout.EXACT_LOCAL_MAX_BUCKET_SIZE_CLASSES_ENV, "0")
    uncapped = local_layout.plan_local_hypothesis_buckets(layout, 2, 32, unify_bucket_sizes=True)
    monkeypatch.setenv(local_layout.EXACT_LOCAL_MAX_BUCKET_SIZE_CLASSES_ENV, "2")
    capped = local_layout.plan_local_hypothesis_buckets(layout, 2, 32, unify_bucket_sizes=True)

    n_uncapped = len({plan.bucket_rotation_count for plan in uncapped})
    n_capped = len({plan.bucket_rotation_count for plan in capped})
    assert n_capped < n_uncapped and n_capped <= 2
    for plan in capped:
        assert plan.bucket_rotation_count >= int(np.max(plan.actual_rotation_counts))


def test_bucket_size_class_cap_is_off_by_default(monkeypatch):
    monkeypatch.delenv(local_layout.EXACT_LOCAL_MAX_BUCKET_SIZE_CLASSES_ENV, raising=False)
    sizes = np.asarray([16, 32, 64, 128], dtype=np.int32)

    np.testing.assert_array_equal(local_layout._cap_bucket_size_classes(sizes), sizes)
