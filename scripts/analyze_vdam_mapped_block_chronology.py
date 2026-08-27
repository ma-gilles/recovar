#!/usr/bin/env python
"""Compare corresponding native and RECOVAR VDAM atomic-block chronologies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BLOCK_NO_ATOMIC = np.uint32(1 << 3)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def _ranks(values: np.ndarray, particle_ids: np.ndarray, orientation_rows: np.ndarray) -> np.ndarray:
    order = np.lexsort((orientation_rows, particle_ids, values))
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    _require(left.shape == right.shape and left.ndim == 1, "rank vectors must match")
    _require(left.size > 1, "rank correlation requires at least two rows")
    return float(np.corrcoef(left, right)[0, 1])


def _load_records(path: Path, *, label: str) -> tuple[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as sealed:
        _require(int(sealed["schema_version"]) == 1, f"{label} schema mismatch")
        return int(sealed["iteration"]), sealed["records"].copy()


def _load_native_id_map(path: Path, *, iteration: int) -> dict[int, int]:
    with np.load(path, allow_pickle=False) as sealed:
        _require(int(sealed["schema_version"]) >= 2, "worker schedule schema is too old")
        _require(int(sealed["iteration"]) == iteration, "worker schedule iteration mismatch")
        internal = np.asarray(sealed["internal_particle_id_by_sorted_position"], dtype=np.int64)
        stack = np.asarray(sealed["stack_index_by_sorted_position"], dtype=np.int64)
    _require(internal.shape == stack.shape and internal.ndim == 1, "worker schedule IDs differ")
    _require(np.unique(internal).size == internal.size, "native internal IDs are not unique")
    _require(np.unique(stack).size == stack.size, "stack IDs are not unique")
    return dict(zip(internal.tolist(), stack.tolist(), strict=True))


def analyze_arm(arm_root: Path) -> tuple[dict[str, object], dict[tuple[int, int], tuple[int, ...]]]:
    analysis = arm_root / "analysis"
    native_path = analysis / "relion_block_chronology_it001.npz"
    candidate_path = analysis / "recovar_block_chronology_it001.npz"
    map_path = analysis / "recovar_candidate_block_map_it001.npz"
    schedule_path = analysis / "relion_worker_schedule_it001.npz"
    iteration, native = _load_records(native_path, label="native chronology")
    candidate_iteration, candidate = _load_records(candidate_path, label="candidate chronology")
    _require(candidate_iteration == iteration, "native and candidate chronology iterations differ")
    with np.load(map_path, allow_pickle=False) as sealed:
        _require(int(sealed["schema_version"]) == 1, "candidate block-map schema mismatch")
        _require(int(sealed["iteration"]) == iteration, "candidate block-map iteration mismatch")
        candidate_to_native = np.asarray(sealed["candidate_to_native"], dtype=np.int64)
    _require(candidate_to_native.shape == (candidate.size,), "candidate-to-native map shape mismatch")
    native_id_map = _load_native_id_map(schedule_path, iteration=iteration)
    candidate_indices = np.flatnonzero(candidate_to_native >= 0)
    native_indices = candidate_to_native[candidate_indices]
    _require(np.unique(native_indices).size == native_indices.size, "mapped native indices are not unique")
    _require(np.all(native_indices < native.size), "mapped native index is out of range")
    _require(
        np.all((candidate["flags"][candidate_indices] & BLOCK_NO_ATOMIC) == 0),
        "mapped candidate row is atomic-free",
    )
    _require(
        np.all((native["flags"][native_indices] & BLOCK_NO_ATOMIC) == 0),
        "mapped native row is atomic-free",
    )
    candidate_particles = candidate["particle_id"][candidate_indices].astype(np.int64)
    native_particles = np.asarray(
        [native_id_map[int(value)] for value in native["particle_id"][native_indices]],
        dtype=np.int64,
    )
    _require(np.array_equal(candidate_particles, native_particles), "mapped particle IDs differ")
    native_rows = native["orientation_row"][native_indices].astype(np.int64)
    logical_keys = list(
        zip(candidate_particles.tolist(), native_rows.tolist(), strict=True)
    )
    _require(len(set(logical_keys)) == len(logical_keys), "mapped logical keys are not unique")

    native_start_rank = _ranks(
        native["block_start_globaltimer"][native_indices],
        candidate_particles,
        native_rows,
    )
    candidate_start_rank = _ranks(
        candidate["block_start_globaltimer"][candidate_indices],
        candidate_particles,
        native_rows,
    )
    native_atomic_rank = _ranks(
        native["first_atomic_globaltimer"][native_indices],
        candidate_particles,
        native_rows,
    )
    candidate_atomic_rank = _ranks(
        candidate["first_atomic_globaltimer"][candidate_indices],
        candidate_particles,
        native_rows,
    )
    per_particle: list[tuple[int, float, float]] = []
    for particle_id in np.unique(candidate_particles):
        selected = np.flatnonzero(candidate_particles == particle_id)
        if selected.size <= 1:
            continue
        native_start_local = _ranks(
            native["block_start_globaltimer"][native_indices[selected]],
            candidate_particles[selected],
            native_rows[selected],
        )
        candidate_start_local = _ranks(
            candidate["block_start_globaltimer"][candidate_indices[selected]],
            candidate_particles[selected],
            native_rows[selected],
        )
        native_atomic_local = _ranks(
            native["first_atomic_globaltimer"][native_indices[selected]],
            candidate_particles[selected],
            native_rows[selected],
        )
        candidate_atomic_local = _ranks(
            candidate["first_atomic_globaltimer"][candidate_indices[selected]],
            candidate_particles[selected],
            native_rows[selected],
        )
        per_particle.append(
            (
                int(selected.size),
                _correlation(native_start_local, candidate_start_local),
                _correlation(native_atomic_local, candidate_atomic_local),
            )
        )
    weights = np.asarray([row[0] for row in per_particle], dtype=np.float64)
    metrics = {
        "mapped_atomic_count": int(candidate_indices.size),
        "logical_key_count": len(logical_keys),
        "global_start_rank_correlation": _correlation(native_start_rank, candidate_start_rank),
        "global_first_atomic_rank_correlation": _correlation(native_atomic_rank, candidate_atomic_rank),
        "native_start_to_first_atomic_rank_correlation": _correlation(
            native_start_rank,
            native_atomic_rank,
        ),
        "candidate_start_to_first_atomic_rank_correlation": _correlation(
            candidate_start_rank,
            candidate_atomic_rank,
        ),
        "per_particle_start_rank_correlation_weighted": float(
            np.average([row[1] for row in per_particle], weights=weights)
        ),
        "per_particle_first_atomic_rank_correlation_weighted": float(
            np.average([row[2] for row in per_particle], weights=weights)
        ),
        "sm_id_exact_fraction": float(
            np.mean(native["sm_id"][native_indices] == candidate["sm_id"][candidate_indices])
        ),
        "native_start_to_first_atomic_latency_median_cycles": float(
            np.median(
                native["first_atomic_globaltimer"][native_indices]
                - native["block_start_globaltimer"][native_indices]
            )
        ),
        "candidate_start_to_first_atomic_latency_median_cycles": float(
            np.median(
                candidate["first_atomic_globaltimer"][candidate_indices]
                - candidate["block_start_globaltimer"][candidate_indices]
            )
        ),
        "input_sha256": {
            native_path.name: _sha256(native_path),
            candidate_path.name: _sha256(candidate_path),
            map_path.name: _sha256(map_path),
            schedule_path.name: _sha256(schedule_path),
        },
    }
    repeat_rows = {
        key: (
            int(native_start_rank[index]),
            int(candidate_start_rank[index]),
            int(native_atomic_rank[index]),
            int(candidate_atomic_rank[index]),
            int(native["sm_id"][native_indices[index]]),
            int(candidate["sm_id"][candidate_indices[index]]),
        )
        for index, key in enumerate(logical_keys)
    }
    return metrics, repeat_rows


def analyze_panel(panel_root: Path) -> dict[str, object]:
    arm_a, repeat_a = analyze_arm(panel_root / "a")
    arm_b, repeat_b = analyze_arm(panel_root / "b")
    _require(repeat_a.keys() == repeat_b.keys(), "mapped logical keys differ across arms")
    keys = sorted(repeat_a)
    values_a = np.asarray([repeat_a[key] for key in keys], dtype=np.int64)
    values_b = np.asarray([repeat_b[key] for key in keys], dtype=np.int64)
    repeat = {
        "logical_key_count": len(keys),
        "native_start_rank_correlation": _correlation(values_a[:, 0], values_b[:, 0]),
        "candidate_start_rank_correlation": _correlation(values_a[:, 1], values_b[:, 1]),
        "native_first_atomic_rank_correlation": _correlation(values_a[:, 2], values_b[:, 2]),
        "candidate_first_atomic_rank_correlation": _correlation(values_a[:, 3], values_b[:, 3]),
        "native_sm_id_exact_fraction": float(np.mean(values_a[:, 4] == values_b[:, 4])),
        "candidate_sm_id_exact_fraction": float(np.mean(values_a[:, 5] == values_b[:, 5])),
    }
    return {
        "schema": "recovar.vdam_mapped_block_chronology_panel.v1",
        "status": "complete",
        "arm_a": arm_a,
        "arm_b": arm_b,
        "repeat": repeat,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    panel_root = args.panel_root.expanduser().resolve(strict=True)
    report = analyze_panel(panel_root)
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
