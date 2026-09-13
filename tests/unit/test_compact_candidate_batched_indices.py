"""The batched compact-candidate index path must equal the per-image path exactly.

build_compact_pair_index_arrays used to call compact_candidate_indices_in_source_order once
per image, materializing a dense (R, T) mask and running np.nonzero each time; on the
100k/256 K=4 fixture that loop was 22% of the pass-2 bucket loop. The batched path does one
(B, R, T) gather and one nonzero. It is pure index bookkeeping: every id, its order, its
dtype and the padded layout must be byte-identical, and any bucket the fast path cannot
handle must fall back to the per-image path rather than approximate it.
"""

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import compact_candidates as cc
from recovar.em.dense_single_volume.helpers.compact_candidates import (
    SparseCandidateMask,
    _batched_compact_candidate_indices,
    build_compact_pair_index_arrays,
    compact_candidate_indices_in_source_order,
)


def _per_image_reference(masks, pair_bucket_size):
    idx = [compact_candidate_indices_in_source_order(m) for m in masks]
    b = len(masks)
    lr = np.full((b, pair_bucket_size), -1, np.int32)
    ti = np.full_like(lr, -1)
    pm = np.zeros((b, pair_bucket_size), bool)
    counts = np.zeros(b, np.int32)
    for i, (r, t) in enumerate(idx):
        c = r.shape[0]; counts[i] = c
        if c:
            lr[i, :c] = r; ti[i, :c] = t; pm[i, :c] = True
    return {"local_rotation_row": lr, "translation_idx": ti, "pair_mask": pm, "pair_counts": counts}


def _coarse_mask(rng, n_rows, n_trans, n_coarse_rot, n_coarse_trans, density, ftp):
    coarse_valid = rng.random((n_coarse_rot, n_coarse_trans)) < density
    parent_map = rng.integers(0, n_coarse_rot, size=n_rows)
    return SparseCandidateMask(
        mode="coarse", n_rows=n_rows, n_fine_trans=n_trans,
        parent_map=parent_map, coarse_valid=coarse_valid, fine_translation_parent=ftp,
    )


def _assert_same(masks, pair_bucket_size):
    want = _per_image_reference(masks, pair_bucket_size)
    got = build_compact_pair_index_arrays(masks, pair_bucket_size=pair_bucket_size)
    for k in ("local_rotation_row", "translation_idx", "pair_mask", "pair_counts"):
        np.testing.assert_array_equal(got[k], want[k], err_msg=k)
        assert got[k].dtype == want[k].dtype, k


@pytest.mark.parametrize("seed", range(6))
def test_coarse_bucket_with_differing_coarse_row_counts_matches_per_image(seed):
    """Images in one bucket share R, T and the translation parent but not the coarse table size."""
    rng = np.random.default_rng(seed)
    n_rows, n_trans, n_coarse_trans = 24, 7, 3
    ftp = rng.integers(0, n_coarse_trans, size=n_trans)
    masks = [
        _coarse_mask(rng, n_rows, n_trans, rng.integers(2, 11), n_coarse_trans, rng.uniform(0.1, 0.9), ftp)
        for _ in range(9)
    ]
    assert _batched_compact_candidate_indices(masks) is not None, "fast path must engage here"
    _assert_same(masks, pair_bucket_size=n_rows * n_trans)


def test_full_empty_and_coarse_mixed_in_one_bucket():
    rng = np.random.default_rng(99)
    n_rows, n_trans = 5, 4
    ftp = np.array([0, 0, 1, 1])
    masks = [
        SparseCandidateMask(mode="full", n_rows=n_rows, n_fine_trans=n_trans),
        SparseCandidateMask(mode="empty", n_rows=n_rows, n_fine_trans=n_trans),
        _coarse_mask(rng, n_rows, n_trans, 3, 2, 0.6, ftp),
        SparseCandidateMask(mode="empty", n_rows=n_rows, n_fine_trans=n_trans),
        _coarse_mask(rng, n_rows, n_trans, 6, 2, 0.4, ftp),
    ]
    assert _batched_compact_candidate_indices(masks) is not None
    _assert_same(masks, pair_bucket_size=n_rows * n_trans)


def test_source_order_is_rotation_major_translation_minor():
    rng = np.random.default_rng(7)
    ftp = np.array([0, 1, 1, 0, 1])
    masks = [_coarse_mask(rng, 10, 5, 4, 2, 0.5, ftp) for _ in range(4)]
    out = build_compact_pair_index_arrays(masks, pair_bucket_size=50)
    for i in range(4):
        c = out["pair_counts"][i]
        flat = out["local_rotation_row"][i, :c].astype(np.int64) * 5 + out["translation_idx"][i, :c]
        assert np.all(np.diff(flat) > 0), "ids must be strictly increasing in C order"


@pytest.mark.parametrize(
    "case", ["dense_numpy", "different_translation_parent", "different_shape"]
)
def test_unsupported_buckets_fall_back_to_the_per_image_path(case, monkeypatch):
    rng = np.random.default_rng(3)
    n_rows, n_trans = 6, 4
    ftp = np.array([0, 1, 0, 1])
    masks = [_coarse_mask(rng, n_rows, n_trans, 3, 2, 0.5, ftp) for _ in range(3)]
    if case == "dense_numpy":
        masks[1] = rng.random((n_rows, n_trans)) < 0.5
    elif case == "different_translation_parent":
        masks[2] = _coarse_mask(rng, n_rows, n_trans, 3, 2, 0.5, np.array([1, 0, 1, 0]))
    elif case == "different_shape":
        masks[2] = _coarse_mask(rng, n_rows + 1, n_trans, 3, 2, 0.5, ftp)
        # per-image reference needs a common capacity; compare the fast-path decision only
        assert _batched_compact_candidate_indices(masks) is None
        return
    assert _batched_compact_candidate_indices(masks) is None, "fast path must decline"
    calls = []
    orig = cc.compact_candidate_indices_in_source_order

    def spy(m):
        calls.append(m); return orig(m)

    monkeypatch.setattr(cc, "compact_candidate_indices_in_source_order", spy)
    _assert_same(masks, pair_bucket_size=n_rows * n_trans)
    assert len(calls) == len(masks), "fallback must use the per-image path for every image"


def test_all_empty_bucket_and_zero_size_axes():
    masks = [SparseCandidateMask(mode="empty", n_rows=4, n_fine_trans=3) for _ in range(3)]
    _assert_same(masks, pair_bucket_size=12)
    out = _batched_compact_candidate_indices(
        [SparseCandidateMask(mode="full", n_rows=0, n_fine_trans=3)]
    )
    assert out is not None and out[0][0].size == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_coarse_exclude_specs_take_the_batched_path_and_match_per_image(seed, monkeypatch):
    """Real-size buckets carry complement (coarse_exclude) specs; the batched path
    must accept them (job 13808173 fell back to per-image dense nonzero, 80 s per
    iteration) and stay byte-identical to the per-image encoder."""
    rng = np.random.default_rng(seed)
    n_rows, n_trans, n_coarse_rot, n_coarse_trans = 30, 8, 5, 4
    ftp = np.repeat(np.arange(n_coarse_trans), 2).astype(np.int32)
    masks = []
    for k in range(5):
        n_excl = int(rng.integers(0, n_coarse_rot * n_coarse_trans))
        excluded = np.sort(rng.choice(n_coarse_rot * n_coarse_trans, size=n_excl, replace=False)).astype(np.int32)
        masks.append(SparseCandidateMask(
            mode="coarse_exclude", n_rows=n_rows, n_fine_trans=n_trans,
            parent_map=rng.integers(0, n_coarse_rot, size=n_rows), coarse_excluded=excluded,
            fine_translation_parent=ftp,
        ))
    masks.append(_coarse_mask(rng, n_rows, n_trans, n_coarse_rot, n_coarse_trans, 0.5, ftp))  # mixed bucket
    masks.append(SparseCandidateMask(mode="full", n_rows=n_rows, n_fine_trans=n_trans))
    assert _batched_compact_candidate_indices(masks) is not None, "batched path must accept coarse_exclude"
    calls = []
    orig = cc.compact_candidate_indices_in_source_order
    monkeypatch.setattr(cc, "compact_candidate_indices_in_source_order", lambda m: (calls.append(m) or orig(m)))
    _assert_same(masks, pair_bucket_size=n_rows * n_trans)
    assert not calls, "no per-image fallback expected"
