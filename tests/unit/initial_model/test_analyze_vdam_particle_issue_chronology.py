from __future__ import annotations

import numpy as np

from scripts.analyze_vdam_particle_issue_chronology import (
    _correlation,
    _particle_events,
    _ranks,
    _spacing,
    analyze,
)


def _records(start_order=(0, 1, 2), *, particle_offset=10):
    dtype = np.dtype(
        [
            ("launch_sequence", "<u8"),
            ("particle_id", "<i8"),
            ("worker_id", "<i4"),
            ("block_start_globaltimer", "<u8"),
            ("first_atomic_globaltimer", "<u8"),
            ("orientation_row", "<u4"),
        ]
    )
    records = np.zeros(6, dtype=dtype)
    for launch_sequence in range(3):
        rows = records[2 * launch_sequence : 2 * launch_sequence + 2]
        rows["launch_sequence"] = launch_sequence
        rows["particle_id"] = particle_offset + launch_sequence
        rows["worker_id"] = launch_sequence % 2
        start_rank = start_order[launch_sequence]
        rows["block_start_globaltimer"] = 100 + 10 * start_rank + np.arange(2)
        rows["first_atomic_globaltimer"] = 200 + 10 * start_rank + np.arange(2)
        rows["orientation_row"] = np.arange(2)
    return records


def test_particle_events_join_native_internal_ids_to_stack_indices():
    events = _particle_events(
        _records(),
        particle_id_map={10: 90, 11: 17, 12: 42},
        label="native",
    )
    assert list(events) == [90, 17, 42]
    assert events[17] == (1, 1, 110, 210)


def test_particle_issue_ranks_expose_reversed_device_admission():
    particle_ids = np.asarray([10, 11, 12], dtype=np.int64)
    launch_rank = _ranks(np.asarray([0, 1, 2]), particle_ids)
    reversed_start_rank = _ranks(np.asarray([120, 110, 100]), particle_ids)
    assert launch_rank.tolist() == [0, 1, 2]
    assert reversed_start_rank.tolist() == [2, 1, 0]
    assert _correlation(launch_rank, reversed_start_rank) == -1.0


def test_particle_issue_spacing_uses_launch_order_and_positive_gaps():
    spacing = _spacing(
        np.asarray([100, 130, 120], dtype=np.int64),
        np.asarray([0, 2, 1], dtype=np.int64),
    )
    assert spacing == {
        "span_cycles": 30,
        "nondecreasing_gap_fraction": 1.0,
        "median_signed_gap_cycles": 15.0,
        "median_absolute_gap_cycles": 15.0,
        "p10_signed_gap_cycles": 11.0,
        "p90_signed_gap_cycles": 19.0,
    }


def test_particle_issue_analyzer_separates_host_order_from_device_admission(tmp_path):
    native_path = tmp_path / "native.npz"
    candidate_path = tmp_path / "candidate.npz"
    schedule_path = tmp_path / "schedule.npz"
    native = _records()
    candidate = _records(start_order=(2, 1, 0))
    for launch_sequence in range(3):
        indices = np.flatnonzero(candidate["launch_sequence"] == launch_sequence)
        candidate["block_start_globaltimer"][indices] = candidate[
            "block_start_globaltimer"
        ][indices[::-1]]
        candidate["first_atomic_globaltimer"][indices] = candidate[
            "first_atomic_globaltimer"
        ][indices[::-1]]
    candidate_particle_ids = np.asarray([90, 17, 42], dtype=np.int64)
    for launch_sequence, particle_id in enumerate(candidate_particle_ids):
        candidate["particle_id"][candidate["launch_sequence"] == launch_sequence] = (
            particle_id
        )
    np.savez(
        native_path,
        schema_version=np.asarray(1),
        iteration=np.asarray(58),
        records=native,
    )
    np.savez(
        candidate_path,
        schema_version=np.asarray(1),
        iteration=np.asarray(58),
        records=candidate,
    )
    np.savez(
        schedule_path,
        schema_version=np.asarray(2),
        iteration=np.asarray(58),
        internal_particle_id_by_sorted_position=np.asarray([10, 11, 12]),
        stack_index_by_sorted_position=candidate_particle_ids,
    )

    report = analyze(native_path, candidate_path, schedule_path)

    assert report["particle_count"] == 3
    assert report["launch_sequence_exact_fraction"] == 1.0
    assert report["worker_owner_exact_fraction"] == 1.0
    assert report["launch_sequence_rank_correlation"] == 1.0
    assert report["first_block_start_rank_correlation"] == -1.0
    assert report["first_atomic_rank_correlation"] == -1.0
    assert report["schema"] == "recovar.vdam_particle_issue_chronology.v2"
    assert report["block_key_jaccard"] == 1.0
    assert report["atomic_block_key_jaccard"] == 1.0
    assert np.isclose(
        report["within_particle_block_start_rank_correlation"]["median"], -1.0
    )
    assert np.isclose(
        report["within_particle_atomic_rank_correlation"]["median"], -1.0
    )
