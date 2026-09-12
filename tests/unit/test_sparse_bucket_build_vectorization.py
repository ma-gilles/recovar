"""The compact-pair bucket builder must be byte-identical to its loop form.

The build stage marshals ragged per-image candidate lists into padded arrays. On the
100k/256 K=4 fixture the per-image Python loop form was 17.6% of the pass-2 bucket loop
(1.0% at 5k/128), so it was replaced by flat gathers. That is pure data movement: the
values written and their positions must not change at all.
"""

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import (
    _build_compact_pair_bucket_arrays_from_per_image_inputs,
    build_compact_pair_index_arrays,
)


def _reference_builder(bucket, per_image_inputs):
    """The original per-image loop, kept here as the independent reference."""
    pair_bucket_size = int(bucket["pair_bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    index_arrays = build_compact_pair_index_arrays(
        (per_image_inputs["candidate_mask"][int(i)] for i in image_indices),
        pair_bucket_size=pair_bucket_size,
    )
    batch = int(image_indices.shape[0])
    padded_rotation_index = np.zeros((batch, pair_bucket_size), dtype=np.int64)
    log_prior_dtype = np.result_type(
        *(np.asarray(per_image_inputs["log_prior"][int(i)]).dtype for i in image_indices)
    )
    padded_log_prior = np.full((batch, pair_bucket_size), -1e30, dtype=log_prior_dtype)
    for row, image_idx in enumerate(image_indices.tolist()):
        count = int(index_arrays["pair_counts"][row])
        if count == 0:
            continue
        local_rot_rows = index_arrays["local_rotation_row"][row, :count]
        rotation_indices = np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64)
        rotation_log_prior = np.asarray(per_image_inputs["log_prior"][image_idx], dtype=log_prior_dtype)
        padded_rotation_index[row, :count] = rotation_indices[local_rot_rows]
        padded_log_prior[row, :count] = rotation_log_prior[local_rot_rows]
    return padded_rotation_index, padded_log_prior


def _fixture(seed, n_images, n_rot, n_trans, pair_bucket_size, empty_rows=()):
    rng = np.random.default_rng(seed)
    candidate_mask, rot_indices, log_prior = [], [], []
    for i in range(n_images):
        mask = rng.random((n_rot, n_trans)) < 0.35
        if i in empty_rows:
            mask[:] = False
        # keep the padded width honest
        while int(mask.sum()) > pair_bucket_size:
            hot = np.flatnonzero(mask.reshape(-1))
            mask.reshape(-1)[rng.choice(hot)] = False
        candidate_mask.append(mask)
        rot_indices.append(rng.integers(0, 10_000, size=n_rot, dtype=np.int64))
        log_prior.append(rng.normal(size=n_rot).astype(np.float32))
    bucket = {"pair_bucket_size": pair_bucket_size,
              "image_indices": np.arange(n_images, dtype=np.int64)}
    inputs = {"candidate_mask": candidate_mask,
              "oversampled_rot_indices": rot_indices,
              "log_prior": log_prior}
    return bucket, inputs


@pytest.mark.parametrize(
    ("seed", "n_images", "n_rot", "n_trans", "pair_bucket_size", "empty_rows"),
    [
        (11, 6, 5, 4, 64, ()),
        (12, 1, 3, 2, 16, ()),
        (13, 9, 7, 3, 128, (0, 4)),
        (14, 5, 4, 4, 32, (0, 1, 2, 3, 4)),   # every row empty
        (15, 32, 8, 5, 256, (7,)),
        (16, 3, 12, 1, 64, ()),
    ],
)
def test_vectorized_builder_matches_the_loop_exactly(seed, n_images, n_rot, n_trans, pair_bucket_size, empty_rows):
    bucket, inputs = _fixture(seed, n_images, n_rot, n_trans, pair_bucket_size, empty_rows)
    want_rot, want_prior = _reference_builder(bucket, inputs)
    got = _build_compact_pair_bucket_arrays_from_per_image_inputs(bucket, inputs)
    np.testing.assert_array_equal(got["rotation_index"], want_rot)
    np.testing.assert_array_equal(got["log_prior"], want_prior)
    assert got["rotation_index"].dtype == want_rot.dtype
    assert got["log_prior"].dtype == want_prior.dtype


def test_per_image_rotation_tables_of_differing_length_are_offset_correctly():
    """Each image has its own rotation table; the flat gather must not cross images."""
    bucket = {"pair_bucket_size": 8, "image_indices": np.asarray([0, 1, 2], dtype=np.int64)}
    masks = [np.array([[True, False], [True, True]]),
             np.array([[False, True], [False, False]], dtype=bool),
             np.array([[True, True], [True, False]])]
    inputs = {
        "candidate_mask": masks,
        # deliberately different lengths and disjoint value ranges per image
        "oversampled_rot_indices": [np.array([10, 11], np.int64),
                                    np.array([20, 21], np.int64),
                                    np.array([30, 31], np.int64)],
        "log_prior": [np.array([-1.0, -2.0], np.float32),
                      np.array([-3.0, -4.0], np.float32),
                      np.array([-5.0, -6.0], np.float32)],
    }
    want_rot, want_prior = _reference_builder(bucket, inputs)
    got = _build_compact_pair_bucket_arrays_from_per_image_inputs(bucket, inputs)
    np.testing.assert_array_equal(got["rotation_index"], want_rot)
    np.testing.assert_array_equal(got["log_prior"], want_prior)
    # image 1's rows must only ever contain image 1's rotation ids
    used = got["rotation_index"][1][got["pair_mask"][1]]
    assert set(used.tolist()) <= {20, 21}
