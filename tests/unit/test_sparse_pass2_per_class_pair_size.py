"""``RECOVAR_SPARSE_PASS2_PER_CLASS_PAIR_SIZE``: each K class sizes its own compact pair axis.

Bucket membership groups images by the pair count of the largest class, and today every class is
padded to that shared width. RELION instead carries a per-class candidate count (its accelerated
weight array is the sum over classes of ``orientation_num[iclass] * nr_trans``). These tests check
that the opt-in only narrows the padded width, never the valid prefix, and that the scores of the
valid pairs are unchanged.
"""

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import sparse_bucket_arrays as sba
from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import (
    _build_compact_pair_bucket_arrays_from_per_image_inputs,
)
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size

FUSED_WIDTH = 512
N_ROT, N_TRANS = 8, 8


def _per_image_inputs(seed, n_images, n_true):
    rng = np.random.default_rng(seed)
    masks, rot_indices, log_priors = [], [], []
    for _ in range(n_images):
        flat = np.zeros(N_ROT * N_TRANS, dtype=bool)
        flat[rng.choice(N_ROT * N_TRANS, n_true, replace=False)] = True
        masks.append(flat.reshape(N_ROT, N_TRANS))
        rot_indices.append(rng.integers(0, 1000, N_ROT).astype(np.int64))
        log_priors.append(rng.normal(size=N_ROT).astype(np.float64))
    return {"candidate_mask": masks, "oversampled_rot_indices": rot_indices, "log_prior": log_priors}


def _bucket(n_images):
    return {"pair_bucket_size": FUSED_WIDTH, "image_indices": np.arange(n_images, dtype=np.int64)}


def test_flag_defaults_off_and_rejects_non_binary(monkeypatch):
    monkeypatch.delenv(sba.PER_CLASS_PAIR_SIZE_ENV, raising=False)
    assert sba.per_class_pair_size_enabled() is False
    monkeypatch.setenv(sba.PER_CLASS_PAIR_SIZE_ENV, "1")
    assert sba.per_class_pair_size_enabled() is True
    monkeypatch.setenv(sba.PER_CLASS_PAIR_SIZE_ENV, "on")
    with pytest.raises(ValueError):
        sba.per_class_pair_size_enabled()


@pytest.mark.parametrize("n_true", [1, 20, 60])
def test_width_follows_this_class_and_never_exceeds_the_shared_width(n_true):
    per_image_inputs = _per_image_inputs(0, 6, n_true)
    narrow = _build_compact_pair_bucket_arrays_from_per_image_inputs(
        _bucket(6), per_image_inputs, per_class_pair_size=True
    )
    expected = _exact_bucket_rotation_size(n_true, 5000)
    assert int(narrow["pair_bucket_size"]) == expected <= FUSED_WIDTH
    assert narrow["pair_mask"].shape == (6, expected)
    assert int(narrow["pair_counts"].max()) == n_true


def test_flag_off_is_byte_for_byte_todays_shared_width():
    per_image_inputs = _per_image_inputs(1, 5, 17)
    shared = _build_compact_pair_bucket_arrays_from_per_image_inputs(
        _bucket(5), per_image_inputs, per_class_pair_size=False
    )
    assert int(shared["pair_bucket_size"]) == FUSED_WIDTH
    for key, value in shared.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            assert value.shape[1] == FUSED_WIDTH


def test_narrow_arrays_are_the_shared_width_prefix_and_drop_only_inert_fill():
    per_image_inputs = _per_image_inputs(2, 7, 23)
    shared = _build_compact_pair_bucket_arrays_from_per_image_inputs(
        _bucket(7), per_image_inputs, per_class_pair_size=False
    )
    narrow = _build_compact_pair_bucket_arrays_from_per_image_inputs(
        _bucket(7), per_image_inputs, per_class_pair_size=True
    )
    width = int(narrow["pair_bucket_size"])
    assert width < FUSED_WIDTH
    np.testing.assert_array_equal(narrow["pair_counts"], shared["pair_counts"])
    for key in ("local_rotation_row", "translation_idx", "rotation_index", "log_prior", "pair_mask"):
        np.testing.assert_array_equal(narrow[key], shared[key][:, :width])
    # everything the narrow build drops is padding, not a candidate
    dropped = shared["pair_mask"][:, width:]
    assert not dropped.any()
    np.testing.assert_array_equal(shared["local_rotation_row"][:, width:], -1)
    np.testing.assert_array_equal(shared["translation_idx"][:, width:], -1)


def test_valid_pair_scores_are_unchanged_by_the_narrower_width():
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _score_pass2_pairs_relion_gpu_diff2_raw,
    )

    n_images, n_pixels = 7, 24
    per_image_inputs = _per_image_inputs(3, n_images, 23)
    shared = _build_compact_pair_bucket_arrays_from_per_image_inputs(
        _bucket(n_images), per_image_inputs, per_class_pair_size=False
    )
    narrow = _build_compact_pair_bucket_arrays_from_per_image_inputs(
        _bucket(n_images), per_image_inputs, per_class_pair_size=True
    )
    rng = np.random.default_rng(4)
    shifted = jnp.asarray(
        rng.normal(size=(n_images, N_TRANS, n_pixels)) + 1j * rng.normal(size=(n_images, N_TRANS, n_pixels)),
        jnp.complex64,
    )
    corr = jnp.asarray(rng.random((n_images, n_pixels)), jnp.float32)
    proj = jnp.asarray(
        rng.normal(size=(n_images, N_ROT, n_pixels)) + 1j * rng.normal(size=(n_images, N_ROT, n_pixels)),
        jnp.complex64,
    )
    weights = jnp.asarray(rng.random(n_pixels), jnp.float32)

    def score(arrays):
        return np.asarray(
            _score_pass2_pairs_relion_gpu_diff2_raw(
                shifted,
                corr,
                proj,
                weights,
                jnp.asarray(arrays["local_rotation_row"]),
                jnp.asarray(arrays["translation_idx"]),
                jnp.asarray(arrays["pair_mask"]),
            )
        )

    width = int(narrow["pair_bucket_size"])
    shared_scores, narrow_scores = score(shared), score(narrow)
    mask = np.asarray(narrow["pair_mask"])
    np.testing.assert_array_equal(narrow_scores[mask], shared_scores[:, :width][mask])


def test_width_is_clamped_when_the_shared_width_is_below_the_quantizer_floor():
    # Callers (and diagnostics) may build buckets with an exact shared width; the
    # per-class quantizer's floor of 16 must not widen such a bucket.
    per_image_inputs = _per_image_inputs(5, 4, 2)
    bucket = {"pair_bucket_size": 5, "image_indices": np.arange(4, dtype=np.int64)}
    narrow = _build_compact_pair_bucket_arrays_from_per_image_inputs(
        bucket, per_image_inputs, per_class_pair_size=True
    )
    shared = _build_compact_pair_bucket_arrays_from_per_image_inputs(
        bucket, per_image_inputs, per_class_pair_size=False
    )
    assert int(narrow["pair_bucket_size"]) == 5
    for key in ("local_rotation_row", "translation_idx", "rotation_index", "log_prior", "pair_mask"):
        np.testing.assert_array_equal(narrow[key], shared[key])
