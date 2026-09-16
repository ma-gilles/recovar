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
        sample_mask_flat=rng.random((total, n_trans)) > 0.3 if with_optional else None,
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
                np.testing.assert_array_equal(b.local_sample_mask[row, lo:hi], layout.sample_mask_flat[s0:s1])
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
