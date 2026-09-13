#!/usr/bin/env python
"""Seal the RECOVAR compact-block to native RELION orientation-row map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recovar.em.diagnostics import vdam_replay

MAGIC = vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_MAGIC
HEADER_DTYPE = vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_HEADER_DTYPE
RECORD_DTYPE = vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_RECORD_DTYPE
MAP_VALID = vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_VALID
MAP_CONTRIBUTING = vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_CONTRIBUTING
INVALID_ROW = vdam_replay.VDAM_CANDIDATE_BLOCK_MAP_INVALID_ROW
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


def load_map(path: Path) -> tuple[dict[str, int], np.ndarray]:
    payload = path.read_bytes()
    _require(len(payload) >= HEADER_DTYPE.itemsize, "block map is shorter than its header")
    header_row = np.frombuffer(payload, dtype=HEADER_DTYPE, count=1)[0]
    _require(payload[:16] == MAGIC, "block-map magic is not schema v1")
    header = {
        name: int(header_row[name])
        for name in HEADER_DTYPE.names
        if name != "magic"
    }
    _require(header["schema_version"] in {1, 2}, "block-map schema version must be one or two")
    _require(header["header_size"] == HEADER_DTYPE.itemsize, "block-map header size mismatch")
    _require(header["record_size"] == RECORD_DTYPE.itemsize, "block-map record size mismatch")
    _require(header["record_count"] > 0, "block map contains zero records")
    _require(header["record_count"] <= header["capacity"], "block-map capacity was exceeded")
    _require(
        header["reserved0"] == 0 and header["reserved1"] == 0,
        "block-map reserved header fields must be zero",
    )
    expected_size = HEADER_DTYPE.itemsize + header["record_count"] * RECORD_DTYPE.itemsize
    _require(len(payload) == expected_size, "block map has truncated records or trailing bytes")
    records = np.frombuffer(
        payload,
        dtype=RECORD_DTYPE,
        count=header["record_count"],
        offset=HEADER_DTYPE.itemsize,
    ).copy()
    return header, records


def _load_npz_records(path: Path, *, label: str, iteration: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as sealed:
        _require(int(sealed["schema_version"]) == 1, f"{label} chronology schema mismatch")
        _require(int(sealed["iteration"]) == iteration, f"{label} chronology iteration mismatch")
        return sealed["records"].copy()


def _native_internal_to_stack(schedule_path: Path, *, iteration: int) -> dict[int, int]:
    with np.load(schedule_path, allow_pickle=False) as sealed:
        _require(int(sealed["schema_version"]) >= 2, "worker schedule schema is too old")
        _require(int(sealed["iteration"]) == iteration, "worker schedule iteration mismatch")
        internal = np.asarray(sealed["internal_particle_id_by_sorted_position"], dtype=np.int64)
        stack = np.asarray(sealed["stack_index_by_sorted_position"], dtype=np.int64)
    _require(internal.shape == stack.shape and internal.ndim == 1, "worker schedule IDs have invalid shapes")
    _require(np.unique(internal).size == internal.size, "worker schedule internal IDs are not unique")
    _require(np.unique(stack).size == stack.size, "worker schedule stack IDs are not unique")
    return dict(zip(internal.tolist(), stack.tolist(), strict=True))


def validate_map(
    header: dict[str, int],
    map_records: np.ndarray,
    *,
    candidate_records: np.ndarray,
    native_records: np.ndarray,
    native_internal_to_stack: dict[int, int],
    iteration: int,
    n_particles: int,
) -> dict[str, object]:
    _require(iteration > 0 and n_particles > 0, "iteration and particle count must be positive")
    _require(header["iteration"] == iteration, "block-map header iteration mismatch")
    _require(map_records.shape == (header["record_count"],), "block-map record shape mismatch")
    _require(map_records.size == candidate_records.size, "block map and candidate trace cardinalities differ")
    _require(np.all(map_records["iteration"] == iteration), "block-map record iteration mismatch")
    _require(np.all(map_records["particle_id"] >= 0), "block-map particle IDs must be nonnegative")
    _require(np.all(map_records["class_id"] == 0), "K=1 block-map class IDs must be zero")
    _require(np.all(map_records["reserved"] == 0), "block-map record reserved fields must be zero")
    schema_version = int(header.get("schema_version", 1))
    known_flags = MAP_VALID if schema_version == 1 else MAP_VALID | MAP_CONTRIBUTING
    _require(np.all((map_records["flags"] & ~known_flags) == 0), "block-map flags are unknown")
    valid = (map_records["flags"] & MAP_VALID) != 0
    contributing = (
        valid
        if schema_version == 1
        else (map_records["flags"] & MAP_CONTRIBUTING) != 0
    )
    _require(np.all(~contributing | valid), "contributing block-map rows must be valid")
    _require(
        np.all(map_records["native_orientation_row"][~valid] == INVALID_ROW),
        "invalid candidate rows must use the native-row sentinel",
    )
    _require(
        np.all(map_records["global_rotation_id"][~valid] == -1),
        "invalid candidate rows must use the global-rotation sentinel",
    )
    _require(
        np.all(map_records["native_orientation_row"][valid] != INVALID_ROW),
        "valid candidate rows cannot use the native-row sentinel",
    )
    _require(
        np.all(map_records["global_rotation_id"][valid] >= 0),
        "valid candidate rows require global rotation IDs",
    )
    particles = np.unique(map_records["particle_id"])
    _require(particles.size == n_particles, "block-map particle count mismatch")

    map_by_candidate: dict[tuple[int, int], int] = {}
    for index, row in enumerate(map_records):
        key = (int(row["particle_id"]), int(row["candidate_orientation_row"]))
        _require(key not in map_by_candidate, "block map has duplicate candidate rows")
        map_by_candidate[key] = index
    candidate_by_key: dict[tuple[int, int], int] = {}
    for index, row in enumerate(candidate_records):
        key = (int(row["particle_id"]), int(row["orientation_row"]))
        _require(key not in candidate_by_key, "candidate chronology has duplicate physical rows")
        candidate_by_key[key] = index
    _require(map_by_candidate.keys() == candidate_by_key.keys(), "block map does not biject candidate trace rows")

    native_by_key: dict[tuple[int, int], int] = {}
    for index, row in enumerate(native_records):
        internal_id = int(row["particle_id"])
        _require(internal_id in native_internal_to_stack, "native trace particle is absent from worker schedule")
        key = (native_internal_to_stack[internal_id], int(row["orientation_row"]))
        _require(key not in native_by_key, "native chronology has duplicate logical rows")
        native_by_key[key] = index

    candidate_to_native = np.full(candidate_records.size, -1, dtype=np.int64)
    native_to_candidate = np.full(native_records.size, -1, dtype=np.int64)
    mapped_native_keys: set[tuple[int, int]] = set()
    for candidate_key, map_index in map_by_candidate.items():
        candidate_index = candidate_by_key[candidate_key]
        candidate_atomic = (
            int(candidate_records["flags"][candidate_index]) & int(BLOCK_NO_ATOMIC)
        ) == 0
        map_row = map_records[map_index]
        map_valid = bool(int(map_row["flags"]) & int(MAP_VALID))
        map_contributing = (
            map_valid
            if schema_version == 1
            else bool(int(map_row["flags"]) & int(MAP_CONTRIBUTING))
        )
        _require(
            candidate_atomic == map_contributing,
            "candidate atomic flags disagree with the compact-row map",
        )
        if not map_valid:
            continue
        native_key = (candidate_key[0], int(map_row["native_orientation_row"]))
        _require(native_key in native_by_key, "mapped logical row is absent from native chronology")
        _require(native_key not in mapped_native_keys, "multiple candidate rows map to one native row")
        mapped_native_keys.add(native_key)
        native_index = native_by_key[native_key]
        native_atomic = (
            int(native_records["flags"][native_index]) & int(BLOCK_NO_ATOMIC)
        ) == 0
        _require(
            native_atomic == map_contributing,
            "mapped candidate/native contributing flags disagree",
        )
        candidate_to_native[candidate_index] = native_index
        native_to_candidate[native_index] = candidate_index

    native_atomic_indices = np.flatnonzero((native_records["flags"] & BLOCK_NO_ATOMIC) == 0)
    candidate_atomic_indices = np.flatnonzero((candidate_records["flags"] & BLOCK_NO_ATOMIC) == 0)
    _require(
        np.all(native_to_candidate[native_atomic_indices] >= 0),
        "native atomic rows are missing candidate mappings",
    )
    _require(
        np.all(candidate_to_native[candidate_atomic_indices] >= 0),
        "candidate atomic rows are missing native mappings",
    )
    _require(
        native_atomic_indices.size == candidate_atomic_indices.size == int(np.sum(contributing)),
        "mapped atomic cardinalities differ",
    )
    return {
        "particle_count": int(particles.size),
        "map_record_count": int(map_records.size),
        "mapped_logical_count": int(np.sum(valid)),
        "mapped_atomic_count": int(np.sum(contributing)),
        "mapped_atomic_free_count": int(np.sum(valid & ~contributing)),
        "candidate_padding_count": int(np.sum(~valid)),
        "native_atomic_free_count": int(native_records.size - native_atomic_indices.size),
        "candidate_atomic_free_count": int(candidate_records.size - candidate_atomic_indices.size),
        "candidate_to_native": candidate_to_native,
        "native_to_candidate": native_to_candidate,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-capture", required=True, type=Path)
    parser.add_argument("--candidate-chronology", required=True, type=Path)
    parser.add_argument("--native-chronology", required=True, type=Path)
    parser.add_argument("--worker-schedule", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--n-particles", required=True, type=int)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    map_path = args.map_capture.expanduser().resolve(strict=True)
    candidate_path = args.candidate_chronology.expanduser().resolve(strict=True)
    native_path = args.native_chronology.expanduser().resolve(strict=True)
    schedule_path = args.worker_schedule.expanduser().resolve(strict=True)
    header, map_records = load_map(map_path)
    candidate_records = _load_npz_records(
        candidate_path,
        label="candidate",
        iteration=args.iteration,
    )
    native_records = _load_npz_records(native_path, label="native", iteration=args.iteration)
    native_id_map = _native_internal_to_stack(schedule_path, iteration=args.iteration)
    result = validate_map(
        header,
        map_records,
        candidate_records=candidate_records,
        native_records=native_records,
        native_internal_to_stack=native_id_map,
        iteration=args.iteration,
        n_particles=args.n_particles,
    )
    output_npz = args.output_npz.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    hashes = {
        "map_capture_sha256": _sha256(map_path),
        "candidate_chronology_sha256": _sha256(candidate_path),
        "native_chronology_sha256": _sha256(native_path),
        "worker_schedule_sha256": _sha256(schedule_path),
    }
    np.savez_compressed(
        output_npz,
        schema_version=np.int64(header["schema_version"]),
        iteration=np.int64(args.iteration),
        n_particles=np.int64(args.n_particles),
        map_records=map_records,
        candidate_to_native=result["candidate_to_native"],
        native_to_candidate=result["native_to_candidate"],
        **{key: np.asarray(value) for key, value in hashes.items()},
    )
    report = {
        "schema": f"recovar.vdam_candidate_block_map.v{header['schema_version']}",
        "result": "pass",
        "iteration": args.iteration,
        "particle_count": result["particle_count"],
        "map_record_count": result["map_record_count"],
        "mapped_logical_count": result["mapped_logical_count"],
        "mapped_atomic_count": result["mapped_atomic_count"],
        "mapped_atomic_free_count": result["mapped_atomic_free_count"],
        "candidate_padding_count": result["candidate_padding_count"],
        "native_atomic_free_count": result["native_atomic_free_count"],
        "candidate_atomic_free_count": result["candidate_atomic_free_count"],
        "map_capture": str(map_path),
        "candidate_chronology": str(candidate_path),
        "native_chronology": str(native_path),
        "worker_schedule": str(schedule_path),
        "output_npz": str(output_npz),
        **hashes,
    }
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
