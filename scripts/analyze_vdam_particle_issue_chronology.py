#!/usr/bin/env python
"""Compare native and RECOVAR VDAM particle-level device chronologies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def _load_chronology(path: Path, *, label: str) -> tuple[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as sealed:
        _require(int(sealed["schema_version"]) == 1, f"{label} schema mismatch")
        iteration = int(sealed["iteration"])
        records = np.asarray(sealed["records"]).copy()
    required = {
        "launch_sequence",
        "particle_id",
        "worker_id",
        "block_start_globaltimer",
        "first_atomic_globaltimer",
        "orientation_row",
    }
    _require(
        records.ndim == 1
        and records.dtype.names is not None
        and required.issubset(records.dtype.names),
        f"{label} record schema mismatch",
    )
    return iteration, records


def _load_native_particle_map(path: Path, *, iteration: int) -> dict[int, int]:
    with np.load(path, allow_pickle=False) as sealed:
        _require(int(sealed["schema_version"]) == 2, "worker schedule schema mismatch")
        _require(int(sealed["iteration"]) == iteration, "worker schedule iteration mismatch")
        internal_ids = np.asarray(
            sealed["internal_particle_id_by_sorted_position"], dtype=np.int64
        )
        stack_indices = np.asarray(
            sealed["stack_index_by_sorted_position"], dtype=np.int64
        )
    _require(
        internal_ids.ndim == 1 and stack_indices.shape == internal_ids.shape,
        "worker schedule particle IDs differ",
    )
    _require(
        np.unique(internal_ids).size == internal_ids.size
        and np.unique(stack_indices).size == stack_indices.size,
        "worker schedule particle IDs are not unique",
    )
    return dict(zip(internal_ids.tolist(), stack_indices.tolist(), strict=True))


def _particle_events(
    records: np.ndarray,
    *,
    particle_id_map: dict[int, int] | None,
    label: str,
) -> dict[int, tuple[int, int, int, int]]:
    events: dict[int, tuple[int, int, int, int]] = {}
    for launch_sequence in np.unique(records["launch_sequence"]).tolist():
        rows = records[records["launch_sequence"] == launch_sequence]
        particle_ids = np.unique(rows["particle_id"])
        worker_ids = np.unique(rows["worker_id"])
        _require(particle_ids.size == 1, f"{label} launch has multiple particles")
        _require(worker_ids.size == 1, f"{label} launch has multiple workers")
        raw_particle_id = int(particle_ids[0])
        if particle_id_map is None:
            particle_id = raw_particle_id
        else:
            _require(raw_particle_id in particle_id_map, f"{label} particle is not scheduled")
            particle_id = particle_id_map[raw_particle_id]
        _require(particle_id not in events, f"{label} particle has multiple launches")
        positive_atomics = rows["first_atomic_globaltimer"]
        positive_atomics = positive_atomics[positive_atomics > 0]
        _require(positive_atomics.size > 0, f"{label} launch has no contributing block")
        events[particle_id] = (
            int(launch_sequence),
            int(worker_ids[0]),
            int(np.min(rows["block_start_globaltimer"])),
            int(np.min(positive_atomics)),
        )
    return events


def _ranks(values: np.ndarray, particle_ids: np.ndarray) -> np.ndarray:
    order = np.lexsort((particle_ids, values))
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    _require(left.shape == right.shape and left.ndim == 1, "rank vectors must match")
    _require(left.size > 1, "rank correlation requires at least two particles")
    return float(np.corrcoef(left, right)[0, 1])


def _spacing(values: np.ndarray, launch_sequences: np.ndarray) -> dict[str, float | int]:
    ordered = values[np.argsort(launch_sequences, kind="stable")]
    gaps = np.diff(ordered.astype(np.int64))
    _require(gaps.size > 0, "chronology has fewer than two launches")
    return {
        "span_cycles": int(ordered.max() - ordered.min()),
        "nondecreasing_gap_fraction": float(np.mean(gaps >= 0)),
        "median_signed_gap_cycles": float(np.median(gaps)),
        "median_absolute_gap_cycles": float(np.median(np.abs(gaps))),
        "p10_signed_gap_cycles": float(np.quantile(gaps, 0.1)),
        "p90_signed_gap_cycles": float(np.quantile(gaps, 0.9)),
    }


def _block_events(
    records: np.ndarray,
    *,
    particle_id_map: dict[int, int] | None,
    label: str,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Index every physical block by stable particle and logical orientation row."""

    events: dict[tuple[int, int], tuple[int, int]] = {}
    for row in records:
        raw_particle_id = int(row["particle_id"])
        if particle_id_map is None:
            particle_id = raw_particle_id
        else:
            _require(raw_particle_id in particle_id_map, f"{label} particle is not scheduled")
            particle_id = particle_id_map[raw_particle_id]
        key = (particle_id, int(row["orientation_row"]))
        _require(key not in events, f"{label} block key is not unique")
        block_start = int(row["block_start_globaltimer"])
        _require(block_start > 0, f"{label} block has no start timestamp")
        events[key] = (block_start, int(row["first_atomic_globaltimer"]))
    return events


def _matched_event_rank_correlation(
    native_events: dict[tuple[int, int], tuple[int, int]],
    candidate_events: dict[tuple[int, int], tuple[int, int]],
    *,
    event_index: int,
    positive_only: bool,
) -> tuple[float, int, int, int, int]:
    native_keys = {
        key
        for key, event in native_events.items()
        if not positive_only or event[event_index] > 0
    }
    candidate_keys = {
        key
        for key, event in candidate_events.items()
        if not positive_only or event[event_index] > 0
    }
    matched_keys = sorted(native_keys & candidate_keys)
    _require(len(matched_keys) > 1, "fewer than two matched block events")
    tie_breaker = np.arange(len(matched_keys), dtype=np.int64)
    native_values = np.asarray(
        [native_events[key][event_index] for key in matched_keys], dtype=np.int64
    )
    candidate_values = np.asarray(
        [candidate_events[key][event_index] for key in matched_keys], dtype=np.int64
    )
    correlation = _correlation(
        _ranks(native_values, tie_breaker),
        _ranks(candidate_values, tie_breaker),
    )
    return (
        correlation,
        len(native_keys),
        len(candidate_keys),
        len(matched_keys),
        len(native_keys | candidate_keys),
    )


def _within_particle_rank_summary(
    native_events: dict[tuple[int, int], tuple[int, int]],
    candidate_events: dict[tuple[int, int], tuple[int, int]],
    *,
    event_index: int,
    positive_only: bool,
) -> dict[str, float | int]:
    correlations = []
    particle_ids = sorted(
        {key[0] for key in native_events} & {key[0] for key in candidate_events}
    )
    for particle_id in particle_ids:
        keys = sorted(
            key
            for key in native_events.keys() & candidate_events.keys()
            if key[0] == particle_id
            and (
                not positive_only
                or (
                    native_events[key][event_index] > 0
                    and candidate_events[key][event_index] > 0
                )
            )
        )
        if len(keys) < 2:
            continue
        tie_breaker = np.asarray([key[1] for key in keys], dtype=np.int64)
        native_rank = _ranks(
            np.asarray([native_events[key][event_index] for key in keys], dtype=np.int64),
            tie_breaker,
        )
        candidate_rank = _ranks(
            np.asarray(
                [candidate_events[key][event_index] for key in keys], dtype=np.int64
            ),
            tie_breaker,
        )
        correlations.append(_correlation(native_rank, candidate_rank))
    _require(correlations, "no particle has two matched block events")
    values = np.asarray(correlations, dtype=np.float64)
    return {
        "particle_count": int(values.size),
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": float(values.max()),
    }


def analyze(
    native_chronology: Path,
    candidate_chronology: Path,
    worker_schedule: Path,
) -> dict[str, object]:
    native_chronology = native_chronology.resolve(strict=True)
    candidate_chronology = candidate_chronology.resolve(strict=True)
    worker_schedule = worker_schedule.resolve(strict=True)
    iteration, native_records = _load_chronology(
        native_chronology, label="native chronology"
    )
    candidate_iteration, candidate_records = _load_chronology(
        candidate_chronology, label="candidate chronology"
    )
    _require(candidate_iteration == iteration, "chronology iterations differ")
    native_id_map = _load_native_particle_map(worker_schedule, iteration=iteration)
    native_events = _particle_events(
        native_records,
        particle_id_map=native_id_map,
        label="native chronology",
    )
    candidate_events = _particle_events(
        candidate_records,
        particle_id_map=None,
        label="candidate chronology",
    )
    _require(native_events.keys() == candidate_events.keys(), "chronology particles differ")
    native_block_events = _block_events(
        native_records,
        particle_id_map=native_id_map,
        label="native chronology",
    )
    candidate_block_events = _block_events(
        candidate_records,
        particle_id_map=None,
        label="candidate chronology",
    )
    (
        all_block_start_rank_correlation,
        native_block_count,
        candidate_block_count,
        matched_block_count,
        union_block_count,
    ) = _matched_event_rank_correlation(
        native_block_events,
        candidate_block_events,
        event_index=0,
        positive_only=False,
    )
    (
        all_atomic_rank_correlation,
        native_atomic_count,
        candidate_atomic_count,
        matched_atomic_count,
        union_atomic_count,
    ) = _matched_event_rank_correlation(
        native_block_events,
        candidate_block_events,
        event_index=1,
        positive_only=True,
    )

    particle_ids = np.asarray(sorted(native_events), dtype=np.int64)
    native = np.asarray([native_events[int(value)] for value in particle_ids], dtype=np.int64)
    candidate = np.asarray(
        [candidate_events[int(value)] for value in particle_ids], dtype=np.int64
    )
    native_launch_rank = _ranks(native[:, 0], particle_ids)
    candidate_launch_rank = _ranks(candidate[:, 0], particle_ids)
    native_start_rank = _ranks(native[:, 2], particle_ids)
    candidate_start_rank = _ranks(candidate[:, 2], particle_ids)
    native_atomic_rank = _ranks(native[:, 3], particle_ids)
    candidate_atomic_rank = _ranks(candidate[:, 3], particle_ids)
    native_start_spacing = _spacing(native[:, 2], native[:, 0])
    candidate_start_spacing = _spacing(candidate[:, 2], candidate[:, 0])
    native_atomic_spacing = _spacing(native[:, 3], native[:, 0])
    candidate_atomic_spacing = _spacing(candidate[:, 3], candidate[:, 0])
    return {
        "schema": "recovar.vdam_particle_issue_chronology.v2",
        "status": "complete",
        "iteration": iteration,
        "particle_count": int(particle_ids.size),
        "launch_sequence_exact_fraction": float(np.mean(native[:, 0] == candidate[:, 0])),
        "worker_owner_exact_fraction": float(np.mean(native[:, 1] == candidate[:, 1])),
        "launch_sequence_rank_correlation": _correlation(
            native_launch_rank, candidate_launch_rank
        ),
        "first_block_start_rank_correlation": _correlation(
            native_start_rank, candidate_start_rank
        ),
        "first_atomic_rank_correlation": _correlation(
            native_atomic_rank, candidate_atomic_rank
        ),
        "native_launch_to_start_rank_correlation": _correlation(
            native_launch_rank, native_start_rank
        ),
        "candidate_launch_to_start_rank_correlation": _correlation(
            candidate_launch_rank, candidate_start_rank
        ),
        "native_launch_to_first_atomic_rank_correlation": _correlation(
            native_launch_rank, native_atomic_rank
        ),
        "candidate_launch_to_first_atomic_rank_correlation": _correlation(
            candidate_launch_rank, candidate_atomic_rank
        ),
        "all_block_start_rank_correlation": all_block_start_rank_correlation,
        "all_atomic_rank_correlation": all_atomic_rank_correlation,
        "native_block_count": native_block_count,
        "candidate_block_count": candidate_block_count,
        "matched_block_count": matched_block_count,
        "block_key_jaccard": matched_block_count / union_block_count,
        "native_atomic_block_count": native_atomic_count,
        "candidate_atomic_block_count": candidate_atomic_count,
        "matched_atomic_block_count": matched_atomic_count,
        "atomic_block_key_jaccard": matched_atomic_count / union_atomic_count,
        "within_particle_block_start_rank_correlation": _within_particle_rank_summary(
            native_block_events,
            candidate_block_events,
            event_index=0,
            positive_only=False,
        ),
        "within_particle_atomic_rank_correlation": _within_particle_rank_summary(
            native_block_events,
            candidate_block_events,
            event_index=1,
            positive_only=True,
        ),
        "native_first_block_start_spacing": native_start_spacing,
        "candidate_first_block_start_spacing": candidate_start_spacing,
        "candidate_to_native_start_span_ratio": (
            candidate_start_spacing["span_cycles"] / native_start_spacing["span_cycles"]
        ),
        "native_first_atomic_spacing": native_atomic_spacing,
        "candidate_first_atomic_spacing": candidate_atomic_spacing,
        "candidate_to_native_atomic_span_ratio": (
            candidate_atomic_spacing["span_cycles"] / native_atomic_spacing["span_cycles"]
        ),
        "first_native_start_particle_ids": particle_ids[
            np.argsort(native[:, 2], kind="stable")[:20]
        ].tolist(),
        "first_candidate_start_particle_ids": particle_ids[
            np.argsort(candidate[:, 2], kind="stable")[:20]
        ].tolist(),
        "input_sha256": {
            native_chronology.name: _sha256(native_chronology),
            candidate_chronology.name: _sha256(candidate_chronology),
            worker_schedule.name: _sha256(worker_schedule),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-chronology", required=True, type=Path)
    parser.add_argument("--candidate-chronology", required=True, type=Path)
    parser.add_argument("--worker-schedule", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = analyze(
        args.native_chronology,
        args.candidate_chronology,
        args.worker_schedule,
    )
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
