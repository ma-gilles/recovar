from __future__ import annotations

import numpy as np

from scripts import analyze_em_k1_candidate_parent_groups as analyzer


def test_complete_parent_groups_are_reported_independently() -> None:
    parents = np.asarray([12] * 8 + [4] * 8, dtype=np.uint64)
    children = np.asarray(list(range(8)) * 2, dtype=np.uint64)
    groups = analyzer.summarize_parent_groups(parents, children)
    assert [group["orientation_class_key"] for group in groups] == [4, 12]
    assert all(group["child_count"] == 8 for group in groups)
    assert all(group["children_unique"] for group in groups)
    assert all(group["complete_expected_children"] for group in groups)


def test_partial_or_duplicate_parent_group_is_not_complete() -> None:
    groups = analyzer.summarize_parent_groups(
        np.asarray([3, 3, 3]),
        np.asarray([0, 1, 1]),
    )
    assert len(groups) == 1
    assert not groups[0]["children_unique"]
    assert not groups[0]["complete_expected_children"]


def test_matrix_keys_preserve_exact_float32_bytes() -> None:
    matrices = np.zeros((3, 3, 3), dtype=np.float32)
    matrices[1, 0, 0] = np.float32(1.0)
    matrices[2, 0, 0] = np.nextafter(np.float32(1.0), np.float32(2.0))
    keys = analyzer._matrix_keys(matrices)
    assert keys.shape == (3,)
    assert np.unique(keys).size == 3
