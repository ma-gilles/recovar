from __future__ import annotations

import numpy as np

from scripts.analyze_vdam_mapped_block_chronology import _correlation, _ranks


def test_mapped_block_rank_helpers_use_stable_logical_tie_breaks():
    timestamps = np.array([30, 10, 10, 20], dtype=np.uint64)
    particles = np.array([1, 0, 0, 1], dtype=np.int64)
    rows = np.array([0, 2, 1, 1], dtype=np.int64)
    ranks = _ranks(timestamps, particles, rows)
    assert ranks.tolist() == [3, 1, 0, 2]
    assert _correlation(ranks, ranks) == 1.0
    assert _correlation(ranks, 3 - ranks) == -1.0
