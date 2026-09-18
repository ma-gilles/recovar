"""Compact masks retain their stored identity and source-ordered index ABI."""

import pickle

import numpy as np
import pytest

from recovar.em.scoring import compact_candidates
from recovar.em.scoring.compact_candidates import (
    SparseCandidateMask,
    build_compact_fine_job_plan_from_pair_arrays,
    build_compact_pair_index_arrays,
    compact_candidate_indices_in_source_order,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["coarse", "coarse_exclude"])
@pytest.mark.parametrize("n_rows,n_trans", [(0, 0), (0, 7), (12, 0), (1, 1), (31, 23)])
@pytest.mark.parametrize("seed", range(4))
def test_coarse_row_expansion_matches_independent_dense_mask(monkeypatch, mode, n_rows, n_trans, seed):
    rng = np.random.default_rng(seed)
    mask = SparseCandidateMask(
        mode=mode,
        n_rows=n_rows,
        n_fine_trans=n_trans,
        parent_map=rng.integers(-3, 4, n_rows, dtype=np.int32),
        fine_translation_parent=rng.integers(0, 3, n_trans, dtype=np.int32),
        coarse_valid=rng.random((5, 4)) < 0.3,
        coarse_excluded=rng.integers(-9, 19, 9, dtype=np.int32) if n_trans else np.zeros(0, dtype=np.int32),
    )
    expected = np.nonzero(np.asarray(mask))
    monkeypatch.setattr(
        compact_candidates, "_candidate_mask_to_dense",
        lambda _: pytest.fail("compact enumeration expanded the dense fine-row mask"),
    )
    actual = compact_candidate_indices_in_source_order(mask)
    for old, new in zip(expected, actual, strict=True):
        assert new.dtype == old.dtype
        np.testing.assert_array_equal(new, old)


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("mode", ["full", "empty", "coarse", "coarse_exclude"])
def test_sparse_mask_roundtrip_and_source_order(mode, protocol):
    mask = SparseCandidateMask(
        mode=mode,
        n_rows=2,
        n_fine_trans=3,
        parent_map=np.array([1, 0]),
        fine_translation_parent=np.array([0, 1, 0]),
        coarse_valid=np.array([[True, False], [False, True]]),
        coarse_excluded=np.array([1]),
    )
    expected = {
        "full": [[True, True, True], [True, True, True]],
        "empty": [[False, False, False], [False, False, False]],
        "coarse": [[False, True, False], [True, False, True]],
        "coarse_exclude": [[True, True, True], [True, False, True]],
    }[mode]
    np.testing.assert_array_equal(np.asarray(mask), expected)
    assert mask.count == np.count_nonzero(expected)
    packed = pickle.dumps(mask, protocol=protocol)
    restored = pickle.loads(packed)
    assert type(restored) is SparseCandidateMask
    np.testing.assert_array_equal(np.asarray(restored), expected)
    rows, translations = compact_candidate_indices_in_source_order(restored)
    np.testing.assert_array_equal(rows * 3 + translations, np.flatnonzero(expected))


def test_pair_and_job_prefix_preserve_image_rotation_translation_order():
    masks = np.array([[[False, True], [True, True]], [[True, False], [False, False]]])
    pair = build_compact_pair_index_arrays(masks, pair_bucket_size=4)
    np.testing.assert_array_equal(pair["pair_counts"], [3, 1])
    np.testing.assert_array_equal(pair["local_rotation_row"], [[0, 1, 1, -1], [0, -1, -1, -1]])
    np.testing.assert_array_equal(pair["translation_idx"], [[1, 0, 1, -1], [0, -1, -1, -1]])
    jobs = build_compact_fine_job_plan_from_pair_arrays(
        pair, np.array([[7, 3], [5, 2]], dtype=np.int32), job_bucket_size=6
    )
    np.testing.assert_array_equal(
        jobs["job_plan"],
        [[0, 7, 0, 1], [0, 3, 1, 0], [0, 3, 1, 1], [1, 5, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]],
    )
    assert jobs["valid_job_count"] == 4


def test_job_plan_rejects_nonprefix_pair_mask():
    pair = build_compact_pair_index_arrays(np.ones((1, 1, 1), dtype=bool), pair_bucket_size=2)
    pair["pair_mask"][0] = [False, True]
    with pytest.raises(ValueError, match="source-ordered prefix"):
        build_compact_fine_job_plan_from_pair_arrays(pair, np.zeros((1, 1), dtype=np.int32))
