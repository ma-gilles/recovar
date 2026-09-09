from __future__ import annotations

import numpy as np

from scripts.analyze_vdam_mapped_block_chronology import (
    BLOCK_NO_ATOMIC,
    _analyze_candidate_physical_repeat,
    _candidate_physical_schedules,
    _correlation,
    _midranks,
    _ranks,
    _summary,
)


def test_mapped_block_rank_helpers_use_stable_logical_tie_breaks():
    timestamps = np.array([30, 10, 10, 20], dtype=np.uint64)
    particles = np.array([1, 0, 0, 1], dtype=np.int64)
    rows = np.array([0, 2, 1, 1], dtype=np.int64)
    ranks = _ranks(timestamps, particles, rows)
    assert ranks.tolist() == [3, 1, 0, 2]
    assert _correlation(ranks, ranks) == 1.0
    assert _correlation(ranks, 3 - ranks) == -1.0


def _candidate_records(start_orders: list[list[int]]) -> np.ndarray:
    dtype = np.dtype(
        [
            ("launch_sequence", "<u8"),
            ("particle_id", "<i8"),
            ("block_start_globaltimer", "<u8"),
            ("first_atomic_globaltimer", "<u8"),
            ("orientation_row", "<u4"),
            ("sm_id", "<u4"),
            ("image_count", "<u4"),
            ("flags", "<u4"),
        ]
    )
    records = np.zeros(sum(map(len, start_orders)), dtype=dtype)
    offset = 0
    for launch_sequence, order in enumerate(start_orders):
        count = len(order)
        rows = records[offset : offset + count]
        rows["launch_sequence"] = launch_sequence
        rows["particle_id"] = 10 + launch_sequence
        rows["image_count"] = count
        rows["orientation_row"] = np.arange(count)
        rows["sm_id"] = np.arange(count) % 2
        for rank, physical_id in enumerate(order):
            rows["block_start_globaltimer"][physical_id] = 100 + rank
            rows["first_atomic_globaltimer"][physical_id] = 200 + rank
        rows["flags"][-1] = BLOCK_NO_ATOMIC
        rows["first_atomic_globaltimer"][-1] = 0
        offset += count
    return records


def test_candidate_physical_schedules_use_append_order_as_physical_id():
    schedules = _candidate_physical_schedules(_candidate_records([[2, 0, 1]]))
    start_rank, atomic, _, _ = schedules[10]
    assert start_rank.tolist() == [1, 2, 0]
    assert atomic.tolist() == [True, True, False]


def test_physical_scheduler_midranks_do_not_invent_timer_tie_order():
    assert _midranks(np.array([30, 10, 10, 20])).tolist() == [3.0, 0.5, 0.5, 2.0]


def test_candidate_physical_repeat_reports_unstable_start_order():
    arm_a = _candidate_records([[0, 1, 2], [0, 1, 2]])
    arm_b = _candidate_records([[2, 1, 0], [2, 1, 0]])
    report = _analyze_candidate_physical_repeat(arm_a, arm_b)
    assert report["particle_count"] == 2
    assert report["start_rank_repeat"]["median"] == -1.0
    assert report["contributing_physical_index_jaccard"]["median"] == 1.0
    assert _summary([1.0, 3.0]) == {
        "mean": 2.0,
        "median": 2.0,
        "p10": 1.2,
        "p90": 2.8,
    }
