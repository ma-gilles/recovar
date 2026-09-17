"""The local score dump must describe a class-segmented bucket's real candidates.

A class-segmented bucket lays class ``k`` at rows ``[k*seg, k*seg + count_k)`` and
pads the rest of each segment, while ``actual_rotation_counts`` holds the sum over
classes. Reading a contiguous prefix of that sum therefore walks off the end of
class 0 into its padding and into the next class, which would make every candidate
identity and score in the dump wrong for K>1. These tests pin the dump to the rows
the layout actually filled.
"""
import numpy as np
import pytest

from recovar.em.diagnostics.local_debug import maybe_write_debug_score_dump
from recovar.em.local.local_layout import (
    LocalHypothesisLayout,
    bucket_class_local_hypothesis_layouts,
)

pytestmark = pytest.mark.unit

N_TRANS = 3
N_GLOBAL = 64


def _layout(rng, counts, translation_log_priors=None, rotation_posterior_ids_flat=None):
    counts = np.asarray(counts, dtype=np.int32)
    total = int(counts.sum())
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n_images = counts.shape[0]
    return LocalHypothesisLayout(
        n_global_rotations=N_GLOBAL,
        n_pixels=8,
        n_psi=8,
        rotation_offsets=offsets,
        rotation_ids_flat=rng.choice(N_GLOBAL, size=total, replace=False).astype(np.int64),
        rotations_flat=rng.standard_normal((total, 3, 3)).astype(np.float32),
        rotation_log_priors_flat=rng.standard_normal(total).astype(np.float32),
        rotation_counts=counts,
        translation_grid=np.arange(N_TRANS * 2, dtype=np.float32).reshape(N_TRANS, 2),
        translation_log_priors=(
            rng.standard_normal((n_images, N_TRANS)).astype(np.float32)
            if translation_log_priors is None else translation_log_priors
        ),
        rotation_posterior_ids_flat=(
            rng.integers(0, N_GLOBAL, total).astype(np.int32)
            if rotation_posterior_ids_flat is None else rotation_posterior_ids_flat
        ),
        sample_mask_bits=np.packbits(
            rng.random((total, N_TRANS)) > 0.3, axis=1, bitorder="little"
        ),
        mstep_rotations_flat=rng.standard_normal((total, 3, 3)).astype(np.float32),
        source_eulers_flat=rng.standard_normal((total, 3)),
    )


class _Dataset:
    """Minimal dataset stub exposing only the original-id mapping the dump uses."""

    def __init__(self, n_images):
        self._n = n_images

    def original_image_indices_from_local(self, image_indices):
        return np.asarray(image_indices, dtype=np.int64) + 1000


def _segmented_bucket(rng, per_class_counts):
    """One class-segmented bucket whose classes have deliberately unequal counts."""
    # Every class shares the translation prior, as the bucketer requires.
    shared_translation_prior = rng.standard_normal(
        (len(per_class_counts[0]), N_TRANS)).astype(np.float32)
    class_layouts = [
        _layout(rng, counts, translation_log_priors=shared_translation_prior)
        for counts in per_class_counts
    ]
    buckets = bucket_class_local_hypothesis_layouts(
        class_layouts,
        np.log(np.full(len(class_layouts), 1.0 / len(class_layouts))),
        len(per_class_counts[0]),
        1,
        preserve_image_order=True,
    )
    assert len(buckets) == 1
    return class_layouts, buckets[0]


def _dump(tmp_path, dataset, layout, bucket, *, rng):
    """Run the dump over synthetic scores and return the single written payload."""
    b = bucket.local_rotation_ids.shape[0]
    r = bucket.local_rotation_ids.shape[1]
    scores = rng.standard_normal((b, r, N_TRANS)).astype(np.float32)
    # Padding rows are inert in the engine because their prior is -1e30; mirror that
    # here so a dump that reads padding produces recognisably impossible scores.
    scores = np.where(np.asarray(bucket.local_rotation_mask)[:, :, None], scores, -np.inf)
    probs = np.zeros_like(scores)
    remaining = maybe_write_debug_score_dump(
        experiment_dataset=dataset,
        local_layout=layout,
        bucket=bucket,
        image_pre_shifts=np.zeros((b, 2), dtype=np.float32),
        scores=scores,
        probs=probs,
        log_Z=np.zeros(b, dtype=np.float32),
        best_log_score=np.zeros(b, dtype=np.float32),
        max_posterior=np.zeros(b, dtype=np.float32),
        reconstruction_sample_mask=np.ones_like(scores, dtype=bool),
        reconstruction_rotation_mask=np.ones((b, r), dtype=bool),
        n_significant_samples=np.zeros(b, dtype=np.int32),
        current_size=None,
        debug_iteration=1,
        dump_dir=tmp_path,
        pending_targets={1000},
    )
    assert remaining == set(), "the requested target should have been consumed"
    written = sorted(tmp_path.glob("*.npz"))
    assert len(written) == 1, written
    return np.load(written[0], allow_pickle=True)


def test_dump_lists_every_class_segment_and_no_padding(tmp_path):
    rng = np.random.default_rng(5)
    per_class_counts = ([7, 4], [2, 9], [5, 3], [1, 6])
    class_layouts, bucket = _segmented_bucket(rng, per_class_counts)
    assert bucket.class_actual_rotation_counts is not None
    seg = bucket.segment_rotation_count
    assert seg > int(np.max(bucket.class_actual_rotation_counts)) - 1, "fixture must pad at least one class"

    payload = _dump(tmp_path, _Dataset(2), class_layouts[0], bucket, rng=rng)

    expected_ids = np.concatenate([
        np.asarray(layout.rotation_ids_flat[
            int(layout.rotation_offsets[0]):int(layout.rotation_offsets[1])
        ], dtype=np.int64)
        for layout in class_layouts
    ])
    got = np.asarray(payload["local_rotation_indices"], dtype=np.int64)
    assert got.size == expected_ids.size, (
        f"dump lists {got.size} candidates for a bucket whose classes hold "
        f"{expected_ids.size}"
    )
    np.testing.assert_array_equal(got, expected_ids)
    assert not np.any(got < 0), "a negative rotation id means padding was reported as a candidate"


def test_dump_scores_are_finite_for_every_reported_candidate(tmp_path):
    """Padding scores -inf, so a finite score on every row proves no padding was read."""
    rng = np.random.default_rng(11)
    class_layouts, bucket = _segmented_bucket(rng, ([6, 3], [2, 8], [4, 2], [1, 5]))
    payload = _dump(tmp_path, _Dataset(2), class_layouts[0], bucket, rng=rng)
    total = np.asarray(payload["pass2_scores_total"])[0]
    assert np.isfinite(total).all(), (
        f"{int((~np.isfinite(total)).sum())} of {total.size} reported candidate scores are "
        "not finite, so padded rows were included"
    )


def test_dump_attributes_each_candidate_to_its_class(tmp_path):
    rng = np.random.default_rng(23)
    per_class_counts = ([7, 4], [2, 9], [5, 3], [1, 6])
    class_layouts, bucket = _segmented_bucket(rng, per_class_counts)
    payload = _dump(tmp_path, _Dataset(2), class_layouts[0], bucket, rng=rng)
    assert "candidate_class_indices" in payload.files, (
        "a K>1 dump must say which class each candidate belongs to"
    )
    counts = np.asarray(bucket.class_actual_rotation_counts)[0]
    expected = np.concatenate([np.full(int(c), k, dtype=np.int32) for k, c in enumerate(counts)])
    np.testing.assert_array_equal(np.asarray(payload["candidate_class_indices"]), expected)


def test_dump_handles_a_class_with_no_candidates_for_this_image(tmp_path):
    """An image can have candidates in some classes and none in others."""
    rng = np.random.default_rng(31)
    # Class 1 contributes nothing to the first image, class 2 nothing to the second.
    per_class_counts = ([5, 3], [0, 4], [6, 0], [2, 2])
    class_layouts, bucket = _segmented_bucket(rng, per_class_counts)
    payload = _dump(tmp_path, _Dataset(2), class_layouts[0], bucket, rng=rng)

    counts = np.asarray(bucket.class_actual_rotation_counts)[0]
    assert int(counts[1]) == 0, "fixture must leave one class empty for the dumped image"
    expected_classes = np.concatenate(
        [np.full(int(c), k, dtype=np.int32) for k, c in enumerate(counts)]
    )
    np.testing.assert_array_equal(
        np.asarray(payload["candidate_class_indices"]), expected_classes,
    )
    assert 1 not in set(np.asarray(payload["candidate_class_indices"]).tolist())
    assert np.isfinite(np.asarray(payload["pass2_scores_total"])[0]).all()


def test_child_ordinals_restart_in_each_class(tmp_path):
    """The same parent appears once per class, so its children must renumber.

    The layout copies each class's parent ids without a class offset. Counted in
    one pass across the selected rows, class 1's first child of a parent would be
    numbered as a later child of class 0's parent with the same id, and the
    oversampling factor read off these ordinals would be inflated by the number of
    classes.
    """
    from recovar.em.diagnostics import local_debug

    # Two classes, each with two children of the same parent 7.
    parents = np.array([7, 7, 7, 7], dtype=np.int32)
    classes = np.array([0, 0, 1, 1], dtype=np.int32)
    np.testing.assert_array_equal(
        local_debug._child_ordinals_from_parent_ids(parents, groups=classes),
        np.array([0, 1, 0, 1], dtype=np.int32),
    )
    # Ungrouped behaviour, which single-class buckets rely on, is unchanged.
    np.testing.assert_array_equal(
        local_debug._child_ordinals_from_parent_ids(parents),
        np.array([0, 1, 2, 3], dtype=np.int32),
    )


def test_dump_oversampling_fields_are_per_class(tmp_path):
    """A four-class bucket whose classes share parent ids must not report 4x children."""
    rng = np.random.default_rng(47)
    per_class_counts = ([4, 4], [4, 4], [4, 4], [4, 4])
    shared_translation_prior = rng.standard_normal((2, N_TRANS)).astype(np.float32)
    class_layouts = []
    # Every class offers the same two parents with two children each, which is what
    # a real adaptive pass-2 layout looks like when the classes agree on the parents.
    parents = np.array([3, 3, 9, 9], dtype=np.int32)
    for _ in range(4):
        class_layouts.append(_layout(
            rng, [4, 4],
            translation_log_priors=shared_translation_prior,
            rotation_posterior_ids_flat=np.concatenate([parents, parents]),
        ))
    buckets = bucket_class_local_hypothesis_layouts(
        class_layouts, np.log(np.full(4, 0.25)), 2, 1, preserve_image_order=True,
    )
    payload = _dump(tmp_path, _Dataset(2), class_layouts[0], buckets[0], rng=rng)
    child = np.asarray(payload["local_rotation_child_indices"])
    assert int(child.max()) == 1, (
        f"child ordinals reach {int(child.max())}; with two children per parent in "
        "each class they must reach 1, not run on across classes"
    )
    assert int(np.asarray(payload["n_rotation_children"])[0]) == 2
