"""Class-segmented local buckets: K=1 reproduces the single-class bucketer; K>1 lays classes out class-major."""
import numpy as np
import pytest

from recovar.em.local.local_layout import (
    LocalHypothesisLayout,
    _exact_bucket_rotation_size,
    bucket_class_local_hypothesis_layouts,
    bucket_local_hypothesis_layout,
)

pytestmark = pytest.mark.unit


def _layout(rng, counts, n_trans=5, n_global=48, with_optional=True, translation_log_priors=None):
    counts = np.asarray(counts, dtype=np.int32)
    total = int(counts.sum())
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    rot = rng.standard_normal((total, 3, 3)).astype(np.float32)
    n_images = counts.shape[0]
    return LocalHypothesisLayout(
        n_global_rotations=n_global,
        n_pixels=12,
        n_psi=4,
        rotation_offsets=offsets,
        rotation_ids_flat=rng.integers(0, n_global, total).astype(np.int64),
        rotations_flat=rot,
        rotation_log_priors_flat=rng.standard_normal(total).astype(np.float32),
        rotation_counts=counts,
        translation_grid=np.arange(n_trans * 2, dtype=np.float32).reshape(n_trans, 2),
        translation_log_priors=(
            rng.standard_normal((n_images, n_trans)).astype(np.float32)
            if translation_log_priors is None else translation_log_priors
        ),
        rotation_posterior_ids_flat=rng.integers(0, n_global, total).astype(np.int32) if with_optional else None,
        sample_mask_bits=(
            np.packbits(rng.random((total, n_trans)) > 0.3, axis=1, bitorder="little")
            if with_optional else None
        ),
        mstep_rotations_flat=rng.standard_normal((total, 3, 3)).astype(np.float32) if with_optional else None,
        source_eulers_flat=rng.standard_normal((total, 3)) if with_optional else None,
    )


def _assert_buckets_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        for name in ("image_indices", "actual_rotation_counts", "local_rotation_ids", "local_rotations",
                     "local_rotation_log_prior", "local_rotation_mask", "translation_log_prior",
                     "local_mstep_rotations", "local_rotation_posterior_ids", "local_sample_mask",
                     "local_source_eulers"):
            xa, ya = getattr(x, name), getattr(y, name)
            if xa is None or ya is None:
                assert xa is None and ya is None, name
            else:
                np.testing.assert_array_equal(np.asarray(xa), np.asarray(ya), err_msg=name)
                assert np.asarray(xa).dtype == np.asarray(ya).dtype, name
        assert x.bucket_image_count == y.bucket_image_count
        assert x.bucket_rotation_count == y.bucket_rotation_count
        assert y.segment_rotation_count == y.bucket_rotation_count


@pytest.mark.parametrize("preserve_image_order", [False, True])
@pytest.mark.parametrize("unify", [False, True])
@pytest.mark.parametrize("with_optional", [True, False])
def test_single_class_reproduces_the_single_class_bucketer_exactly(preserve_image_order, unify, with_optional):
    rng = np.random.default_rng(7)
    layout = _layout(rng, rng.integers(1, 90, 23), with_optional=with_optional)
    kwargs = dict(image_batch_size=6, rotation_block_size=256, max_hypotheses_per_microbatch=2048,
                  unify_bucket_sizes=unify, preserve_image_order=preserve_image_order)
    single = bucket_local_hypothesis_layout(layout, **kwargs)
    merged = bucket_class_local_hypothesis_layouts([layout], [0.0], **kwargs)
    _assert_buckets_equal(single, merged)
    assert all(b.n_classes == 1 for b in merged)


def test_class_segments_are_class_major_with_one_segment_width_per_bucket():
    rng = np.random.default_rng(11)
    n_images, K = 17, 3
    counts = rng.integers(1, 40, (n_images, K))
    tlp = rng.standard_normal((n_images, 5)).astype(np.float32)
    layouts = [_layout(rng, counts[:, k], translation_log_priors=tlp) for k in range(K)]
    priors = np.log(np.asarray([0.2, 0.5, 0.3]))
    buckets = bucket_class_local_hypothesis_layouts(layouts, priors, image_batch_size=4, rotation_block_size=256,
                                                    max_hypotheses_per_microbatch=4096)
    seen = set()
    for b in buckets:
        assert b.n_classes == K
        seg = b.segment_rotation_count
        assert b.bucket_rotation_count == K * seg
        for row, image in enumerate(b.image_indices.tolist()):
            seen.add(image)
            widest = int(counts[image].max())
            assert seg >= widest and seg == _exact_bucket_rotation_size(widest, 256) or b.bucket_rotation_count >= K * widest
            np.testing.assert_array_equal(b.class_actual_rotation_counts[row], counts[image])
            assert b.actual_rotation_counts[row] == counts[image].sum()
            np.testing.assert_array_equal(b.translation_log_prior[row], tlp[image])
            for k, layout in enumerate(layouts):
                lo, hi = k * seg, k * seg + int(counts[image, k])
                s0, s1 = int(layout.rotation_offsets[image]), int(layout.rotation_offsets[image + 1])
                np.testing.assert_array_equal(b.local_rotations[row, lo:hi], layout.rotations_flat[s0:s1])
                np.testing.assert_array_equal(b.local_mstep_rotations[row, lo:hi], layout.mstep_rotations_flat[s0:s1])
                np.testing.assert_array_equal(b.local_rotation_ids[row, lo:hi], layout.rotation_ids_flat[s0:s1])
                np.testing.assert_array_equal(b.local_rotation_posterior_ids[row, lo:hi], layout.rotation_posterior_ids_flat[s0:s1])
                np.testing.assert_array_equal(b.local_sample_mask[row, lo:hi], layout.sample_mask_rows(s0, s1))
                np.testing.assert_allclose(
                    b.local_rotation_log_prior[row, lo:hi],
                    (layout.rotation_log_priors_flat[s0:s1].astype(np.float64) + priors[k]).astype(np.float32),
                    rtol=0, atol=0,
                )
                assert b.local_rotation_mask[row, lo:hi].all()
                assert not b.local_rotation_mask[row, hi:(k + 1) * seg].any()
                assert (b.local_rotation_ids[row, hi:(k + 1) * seg] == -1).all()
    assert seen == set(range(n_images))


def test_class_layouts_must_share_translations_and_priors():
    rng = np.random.default_rng(3)
    a = _layout(rng, [3, 4])
    b = _layout(rng, [2, 5])
    with pytest.raises(ValueError, match="translation log priors"):
        bucket_class_local_hypothesis_layouts([a, b], [0.0, 0.0], image_batch_size=2, rotation_block_size=64)
    with pytest.raises(ValueError, match="entries"):
        bucket_class_local_hypothesis_layouts([a], [0.0, 0.0], image_batch_size=2, rotation_block_size=64)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_class_segments_never_compile_more_shapes_than_per_class_buckets(seed):
    """The segment width comes from each image's widest class, so the shape set can only shrink.

    Every compiled bucket shape is one XLA program per stage. Per-class layouts
    contribute one shape per distinct padded count over all (image, class) pairs;
    class-segmented rows contribute one per distinct padded count of the per-image
    maxima, which is a subset. This is the compile-cost argument for one pass.
    """
    rng = np.random.default_rng(seed)
    n_images, K = 60, 4
    # Heavy-tailed, class-dependent candidate counts, as adaptive pass 2 produces.
    counts = np.maximum(1, (rng.lognormal(3.0, 1.1, (n_images, K)) * (1 + rng.random(K))).astype(int))
    tlp = rng.standard_normal((n_images, 5)).astype(np.float32)
    layouts = [_layout(rng, counts[:, k], translation_log_priors=tlp) for k in range(K)]
    kwargs = dict(image_batch_size=8, rotation_block_size=256, max_hypotheses_per_microbatch=1 << 20)

    per_class = {b.bucket_rotation_count for k in range(K) for b in bucket_local_hypothesis_layout(layouts[k], **kwargs)}
    segmented = {b.bucket_rotation_count for b in bucket_class_local_hypothesis_layouts(layouts, np.zeros(K), **kwargs)}
    assert len(segmented) <= len(per_class), (sorted(segmented), sorted(per_class))

    # And one pass over the data replaces the per-class passes.
    per_class_calls = sum(len(bucket_local_hypothesis_layout(layouts[k], **kwargs)) for k in range(K))
    segmented_calls = len(bucket_class_local_hypothesis_layouts(layouts, np.zeros(K), **kwargs))
    assert segmented_calls <= per_class_calls


def test_bucket_rebuilds_preserve_the_class_segmentation():
    """Image-axis padding and reordering must not erase the row axis's class layout.

    Both helpers rebuild a LocalBucketSpec field by field, so a class-segmented
    bucket silently became a single class of the same total width on the paths that
    pad to a planned image count or reorder to returned indices, and the first real
    run died on it.

    The padding helper returns the bucket untouched when it is already full, so the
    test picks a partially filled bucket explicitly; an earlier version selected the
    first bucket, which was full, and never executed the padding path at all.
    """
    from recovar.em.local.local_bucket_stages import _pad_local_big_jit_image_axis, _reorder_bucket_to_indices

    rng = np.random.default_rng(101)
    n_images, K = 5, 3
    counts = rng.integers(1, 6, (n_images, K))
    tlp = rng.standard_normal((n_images, 5)).astype(np.float32)
    layouts = [_layout(rng, counts[:, k], translation_log_priors=tlp) for k in range(K)]
    buckets = bucket_class_local_hypothesis_layouts(
        layouts, np.zeros(K), image_batch_size=4, rotation_block_size=64, max_hypotheses_per_microbatch=1 << 16,
    )
    assert all(b.n_classes == K and b.class_segment_rotation_count is not None for b in buckets)

    partial = next(b for b in buckets if int(b.image_indices.shape[0]) < int(b.bucket_image_count))
    actual_count = int(partial.image_indices.shape[0])
    batch = np.zeros((actual_count, 4, 4), dtype=np.float32)
    ctf = np.zeros((actual_count, 9), dtype=np.float32)
    padded, _, _, valid_image_mask, padded_batch_size = _pad_local_big_jit_image_axis(partial, batch, ctf)
    # The padding path really ran.
    assert padded is not partial and padded_batch_size > actual_count
    assert padded.n_classes == K
    assert padded.segment_rotation_count == partial.segment_rotation_count
    assert padded.bucket_rotation_count == K * padded.segment_rotation_count
    assert padded.class_actual_rotation_counts.shape == (padded_batch_size, K)
    np.testing.assert_array_equal(padded.class_actual_rotation_counts[:actual_count], partial.class_actual_rotation_counts)
    # The added tail holds no class rows and is not valid.
    np.testing.assert_array_equal(
        padded.class_actual_rotation_counts[actual_count:], np.zeros((padded_batch_size - actual_count, K), dtype=np.int32),
    )
    assert not padded.local_rotation_mask[actual_count:].any()
    assert not np.asarray(valid_image_mask)[actual_count:].any()
    assert np.asarray(valid_image_mask)[:actual_count].all()

    # Reordering needs several images to be a nontrivial permutation.
    multi = max(buckets, key=lambda b: int(b.image_indices.shape[0]))
    assert int(multi.image_indices.shape[0]) > 1
    reversed_indices = np.asarray(multi.image_indices)[::-1]
    reordered = _reorder_bucket_to_indices(multi, reversed_indices)
    assert reordered is not multi
    assert reordered.n_classes == K
    assert reordered.segment_rotation_count == multi.segment_rotation_count
    np.testing.assert_array_equal(
        reordered.class_actual_rotation_counts, np.asarray(multi.class_actual_rotation_counts)[::-1],
    )
    np.testing.assert_array_equal(reordered.local_rotation_ids, np.asarray(multi.local_rotation_ids)[::-1])
