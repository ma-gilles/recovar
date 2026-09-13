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
        masks[2] = _coarse_mask(rng, n_rows, n_trans + 1, 3, 2, 0.5, np.array([0, 1, 0, 1, 1]))
        # a different translation count cannot share one bucket
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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_images_with_different_rotation_row_counts_share_the_batched_path(seed, monkeypatch):
    """Real-size buckets mix images with different rotation counts (the bucket pads rows
    separately); the batched enumerator must accept them and stay byte-identical."""
    rng = np.random.default_rng(seed)
    n_trans, n_coarse_rot, n_coarse_trans = 8, 6, 4
    ftp = np.repeat(np.arange(n_coarse_trans), 2).astype(np.int32)
    masks = []
    for n_rows in (5, 17, 9, 33, 1):
        n_excl = int(rng.integers(0, n_coarse_rot * n_coarse_trans))
        excluded = np.sort(rng.choice(n_coarse_rot * n_coarse_trans, size=n_excl, replace=False)).astype(np.int32)
        masks.append(SparseCandidateMask(
            mode="coarse_exclude", n_rows=n_rows, n_fine_trans=n_trans,
            parent_map=rng.integers(0, n_coarse_rot, size=n_rows), coarse_excluded=excluded,
            fine_translation_parent=ftp,
        ))
    masks.append(_coarse_mask(rng, 12, n_trans, n_coarse_rot, n_coarse_trans, 0.5, ftp))
    masks.append(SparseCandidateMask(mode="full", n_rows=7, n_fine_trans=n_trans))
    masks.append(SparseCandidateMask(mode="empty", n_rows=3, n_fine_trans=n_trans))
    assert _batched_compact_candidate_indices(masks) is not None
    calls = []
    orig = cc.compact_candidate_indices_in_source_order
    monkeypatch.setattr(cc, "compact_candidate_indices_in_source_order", lambda m: (calls.append(m) or orig(m)))
    _assert_same(masks, pair_bucket_size=33 * n_trans)
    assert not calls


# ---------------------------------------------------------------------------
# Device-built pair index arrays (RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DEVICE_INDEX)
# ---------------------------------------------------------------------------


def _host_reference_at_capacity(masks, pair_bucket_size, n_alloc):
    from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import _rows_at_capacity

    host = build_compact_pair_index_arrays(masks, pair_bucket_size=pair_bucket_size)
    return {
        "pair_counts": _rows_at_capacity(host["pair_counts"], n_alloc, 0),
        "local_rotation_row": _rows_at_capacity(host["local_rotation_row"], n_alloc, 0),
        "translation_idx": _rows_at_capacity(host["translation_idx"], n_alloc, 0),
        "pair_mask": _rows_at_capacity(host["pair_mask"], n_alloc, False),
    }


def _assert_device_matches_host(masks, pair_bucket_size, n_alloc=None, rows_capacity=None):
    jax = pytest.importorskip("jax")
    with jax.default_device(jax.devices("cpu")[0]):
        got = cc.compact_pair_index_arrays_device(
            masks, pair_bucket_size=pair_bucket_size, n_alloc=n_alloc, rows_capacity=rows_capacity
        )
    assert got is not None, "device path must engage for this bucket"
    want = _host_reference_at_capacity(masks, pair_bucket_size, len(masks) if n_alloc is None else n_alloc)
    for k in ("pair_counts", "local_rotation_row", "translation_idx", "pair_mask"):
        np.testing.assert_array_equal(np.asarray(got[k]), want[k], err_msg=k)
        assert np.asarray(got[k]).dtype == want[k].dtype, k
    for k in ("local_rotation_row", "translation_idx", "pair_mask"):
        assert isinstance(got[k], jax.Array), k
    assert isinstance(got["pair_counts"], np.ndarray)


def _coarse_exclude_mask(rng, n_rows, n_trans, n_coarse_rot, n_coarse_trans, n_excluded, ftp):
    parent_map = rng.integers(0, n_coarse_rot, size=n_rows)
    excluded = np.unique(rng.integers(0, n_coarse_rot * n_coarse_trans, size=n_excluded)).astype(np.int32)
    return SparseCandidateMask(
        mode="coarse_exclude", n_rows=n_rows, n_fine_trans=n_trans,
        parent_map=parent_map, coarse_excluded=excluded, fine_translation_parent=ftp,
    )


@pytest.mark.parametrize("seed", range(8))
def test_device_index_arrays_match_the_host_builder_bitwise(seed):
    """Same ids, order, dtypes and padding as build_compact_pair_index_arrays + _rows_at_capacity."""
    rng = np.random.default_rng(seed)
    n_trans, n_coarse_trans = 7, 3
    ftp = rng.integers(0, n_coarse_trans, size=n_trans)
    masks = []
    for _ in range(9):
        n_rows = int(rng.integers(1, 40))
        kind = rng.choice(["coarse", "coarse_exclude", "full", "empty"], p=[0.4, 0.4, 0.1, 0.1])
        if kind == "coarse":
            masks.append(_coarse_mask(rng, n_rows, n_trans, int(rng.integers(2, 12)), n_coarse_trans, rng.uniform(0.1, 0.9), ftp))
        elif kind == "coarse_exclude":
            masks.append(_coarse_exclude_mask(rng, n_rows, n_trans, int(rng.integers(2, 12)), n_coarse_trans, int(rng.integers(0, 20)), ftp))
        else:
            masks.append(SparseCandidateMask(mode=kind, n_rows=n_rows, n_fine_trans=n_trans))
    required = max(m.count for m in masks)
    pair_bucket_size = int(rng.integers(required, required + 37)) if required else 8
    _assert_device_matches_host(masks, pair_bucket_size)
    # image-axis capacity: padded image rows are 0 / 0 / False with count 0
    _assert_device_matches_host(masks, pair_bucket_size, n_alloc=len(masks) + int(rng.integers(1, 6)))
    # class row capacity larger than every image's row count
    _assert_device_matches_host(masks, pair_bucket_size, n_alloc=len(masks) + 2, rows_capacity=64)


def test_device_index_arrays_source_order_and_row_padding_values():
    rng = np.random.default_rng(11)
    ftp = np.array([0, 1, 1, 0, 2])
    masks = [_coarse_mask(rng, 12, 5, 6, 3, 0.5, ftp) for _ in range(3)]
    masks.append(SparseCandidateMask(mode="empty", n_rows=12, n_fine_trans=5))
    jax = pytest.importorskip("jax")
    with jax.default_device(jax.devices("cpu")[0]):
        got = cc.compact_pair_index_arrays_device(masks, pair_bucket_size=64, n_alloc=6)
    lrr = np.asarray(got["local_rotation_row"]); tid = np.asarray(got["translation_idx"]); pm = np.asarray(got["pair_mask"])
    for i, m in enumerate(masks):
        c = int(m.count)
        assert got["pair_counts"][i] == c
        flat = lrr[i, :c].astype(np.int64) * 5 + tid[i, :c]
        assert np.all(np.diff(flat) > 0), "ids must be strictly increasing in C order"
        assert np.all(lrr[i, c:] == -1) and np.all(tid[i, c:] == -1) and not pm[i, c:].any()
        assert pm[i, :c].all()
    assert np.all(lrr[4:] == 0) and np.all(tid[4:] == 0) and not pm[4:].any()
    assert np.all(got["pair_counts"][4:] == 0)


@pytest.mark.parametrize("case", ["dense_numpy", "different_translation_parent", "different_translation_count"])
def test_device_index_arrays_decline_unsupported_buckets(case):
    rng = np.random.default_rng(5)
    ftp = np.array([0, 1, 0, 1])
    masks = [_coarse_mask(rng, 6, 4, 3, 2, 0.5, ftp) for _ in range(3)]
    if case == "dense_numpy":
        masks[1] = rng.random((6, 4)) < 0.5
    elif case == "different_translation_parent":
        masks[2] = _coarse_mask(rng, 6, 4, 3, 2, 0.5, np.array([1, 0, 1, 0]))
    else:
        masks[2] = _coarse_mask(rng, 6, 5, 3, 2, 0.5, np.array([0, 1, 0, 1, 1]))
    jax = pytest.importorskip("jax")
    with jax.default_device(jax.devices("cpu")[0]):
        assert cc.compact_pair_index_arrays_device(masks, pair_bucket_size=32) is None


def test_device_index_arrays_reject_too_small_bucket():
    rng = np.random.default_rng(2)
    masks = [SparseCandidateMask(mode="full", n_rows=4, n_fine_trans=3)]
    jax = pytest.importorskip("jax")
    with jax.default_device(jax.devices("cpu")[0]), pytest.raises(ValueError, match="smaller than the source-ordered"):
        cc.compact_pair_index_arrays_device(masks, pair_bucket_size=11)

