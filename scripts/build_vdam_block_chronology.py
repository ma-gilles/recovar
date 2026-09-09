#!/usr/bin/env python
"""Validate and seal a passive RELION VDAM CUDA block chronology capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MAGIC = b"RELION_VDAM_BT1"
MAGIC_BYTES = MAGIC + b"\0"
HEADER_DTYPE = np.dtype(
    [
        ("magic", "S16"),
        ("schema_version", "<u4"),
        ("header_size", "<u4"),
        ("record_size", "<u4"),
        ("iteration", "<u4"),
        ("record_count", "<u8"),
        ("capacity", "<u8"),
        ("reserved0", "<u8"),
        ("reserved1", "<u8"),
    ]
)
RECORD_DTYPE = np.dtype(
    [
        ("launch_sequence", "<u8"),
        ("particle_id", "<i8"),
        ("block_start_globaltimer", "<u8"),
        ("first_atomic_globaltimer", "<u8"),
        ("block_end_globaltimer", "<u8"),
        ("orientation_row", "<u4"),
        ("worker_id", "<i4"),
        ("class_id", "<i4"),
        ("sm_id", "<u4"),
        ("image_count", "<u4"),
        ("iteration", "<u4"),
        ("flags", "<u4"),
        ("reserved", "<u4"),
    ]
)
FLAG_DATA3D = np.uint32(1 << 0)
FLAG_CTF_PREMULTIPLIED = np.uint32(1 << 1)
FLAG_SGD = np.uint32(1 << 2)
FLAG_NO_ATOMIC = np.uint32(1 << 3)
KNOWN_FLAGS = FLAG_DATA3D | FLAG_CTF_PREMULTIPLIED | FLAG_SGD | FLAG_NO_ATOMIC

if HEADER_DTYPE.itemsize != 64 or RECORD_DTYPE.itemsize != 72:
    raise RuntimeError("VDAM block chronology binary schema has an invalid item size")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_capture(path: Path) -> tuple[dict[str, int], np.ndarray]:
    """Load one exact binary capture and reject truncation or trailing bytes."""

    payload = path.read_bytes()
    _require(len(payload) >= HEADER_DTYPE.itemsize, "capture is shorter than its header")
    header_row = np.frombuffer(payload, dtype=HEADER_DTYPE, count=1)[0]
    header = {
        name: int(header_row[name])
        for name in HEADER_DTYPE.names
        if name != "magic"
    }
    _require(payload[:16] == MAGIC_BYTES, "capture magic is not schema v1")
    _require(header["schema_version"] == 1, "capture schema version must be one")
    _require(header["header_size"] == HEADER_DTYPE.itemsize, "capture header size mismatch")
    _require(header["record_size"] == RECORD_DTYPE.itemsize, "capture record size mismatch")
    _require(header["record_count"] > 0, "capture contains zero records")
    _require(
        header["record_count"] <= header["capacity"],
        "capture record count exceeds its declared capacity",
    )
    _require(
        header["reserved0"] == 0 and header["reserved1"] == 0,
        "capture reserved header fields must be zero",
    )
    expected_size = HEADER_DTYPE.itemsize + header["record_count"] * RECORD_DTYPE.itemsize
    _require(len(payload) == expected_size, "capture has truncated records or trailing bytes")
    records = np.frombuffer(
        payload,
        dtype=RECORD_DTYPE,
        count=header["record_count"],
        offset=HEADER_DTYPE.itemsize,
    ).copy()
    return header, records


def validate_capture(
    header: dict[str, int],
    records: np.ndarray,
    *,
    iteration: int,
    n_particles: int,
    n_threads: int,
    n_classes: int,
    sm_count: int,
) -> dict[str, object]:
    """Validate identity, launch topology, timestamps, and H100 SM bounds."""

    _require(iteration > 0, "expected iteration must be positive")
    _require(n_particles > 0, "expected particle count must be positive")
    _require(n_threads > 0, "expected worker count must be positive")
    _require(n_classes > 0, "expected class count must be positive")
    _require(sm_count > 0, "expected SM count must be positive")
    _require(header["iteration"] == iteration, "capture header iteration mismatch")
    _require(records.shape == (header["record_count"],), "capture record shape mismatch")
    _require(np.all(records["iteration"] == iteration), "record iteration mismatch")
    _require(np.all(records["particle_id"] >= 0), "particle IDs must be nonnegative")
    _require(
        np.all((records["worker_id"] >= 0) & (records["worker_id"] < n_threads)),
        "worker IDs are outside the configured range",
    )
    _require(
        np.all((records["class_id"] >= 0) & (records["class_id"] < n_classes)),
        "class IDs are outside the configured range",
    )
    _require(np.all(records["sm_id"] < sm_count), "SM IDs are outside the device range")
    _require(np.all(records["image_count"] > 0), "launch image counts must be positive")
    _require(np.all(records["reserved"] == 0), "record reserved fields must be zero")
    _require(
        np.all(records["orientation_row"] < records["image_count"]),
        "orientation row is outside its launch grid",
    )
    start = records["block_start_globaltimer"]
    first_atomic = records["first_atomic_globaltimer"]
    end = records["block_end_globaltimer"]
    flags = records["flags"]
    atomic_free = (flags & FLAG_NO_ATOMIC) != 0
    _require(np.all((flags & FLAG_SGD) != 0), "records must identify the SGD kernel")
    _require(np.all((flags & ~KNOWN_FLAGS) == 0), "record flags contain unknown bits")
    _require(np.all(start > 0), "block-start timestamps must be nonzero")
    _require(
        np.array_equal(first_atomic == 0, atomic_free),
        "zero first-atomic timestamps must exactly match atomic-free flags",
    )
    _require(
        np.all(first_atomic[~atomic_free] >= start[~atomic_free]),
        "first-atomic timestamp precedes block start",
    )
    _require(
        np.all(end[~atomic_free] >= first_atomic[~atomic_free]),
        "block-end timestamp precedes first atomic",
    )

    launch_ids = np.unique(records["launch_sequence"])
    _require(
        np.array_equal(launch_ids, np.arange(launch_ids.size, dtype=np.uint64)),
        "launch sequences are not an exact zero-based bijection",
    )
    launch_summaries: list[dict[str, int]] = []
    particle_class_keys: set[tuple[int, int]] = set()
    for launch_sequence in launch_ids.tolist():
        rows = records[records["launch_sequence"] == launch_sequence]
        for name in ("particle_id", "worker_id", "class_id", "image_count", "iteration"):
            _require(
                np.unique(rows[name]).size == 1,
                f"launch {launch_sequence} has inconsistent {name}",
            )
        _require(
            np.unique(rows["flags"] & ~FLAG_NO_ATOMIC).size == 1,
            f"launch {launch_sequence} has inconsistent base flags",
        )
        image_count = int(rows["image_count"][0])
        _require(rows.size == image_count, f"launch {launch_sequence} is incomplete")
        _require(
            np.array_equal(
                np.sort(rows["orientation_row"]),
                np.arange(image_count, dtype=np.uint32),
            ),
            f"launch {launch_sequence} orientation rows are not a bijection",
        )
        key = (int(rows["particle_id"][0]), int(rows["class_id"][0]))
        _require(key not in particle_class_keys, "particle/class pair has multiple launches")
        particle_class_keys.add(key)
        launch_summaries.append(
            {
                "launch_sequence": int(launch_sequence),
                "particle_id": key[0],
                "worker_id": int(rows["worker_id"][0]),
                "class_id": key[1],
                "image_count": image_count,
                "atomic_free_count": int(np.sum((rows["flags"] & FLAG_NO_ATOMIC) != 0)),
                "first_start_globaltimer": int(np.min(rows["block_start_globaltimer"])),
                "first_atomic_globaltimer": int(
                    np.min(rows["first_atomic_globaltimer"][rows["first_atomic_globaltimer"] > 0])
                ) if np.any(rows["first_atomic_globaltimer"] > 0) else 0,
                "last_end_globaltimer": int(np.max(rows["block_end_globaltimer"])),
            }
        )

    particle_ids = np.unique(records["particle_id"])
    _require(particle_ids.size == n_particles, "capture particle count mismatch")
    _require(
        len(particle_class_keys) == n_particles * n_classes,
        "capture does not contain exactly one launch per particle and class",
    )
    atomic_indices = np.flatnonzero(~atomic_free)
    first_atomic_order = atomic_indices[
        np.lexsort(
            (
                records["orientation_row"][atomic_indices],
                records["launch_sequence"][atomic_indices],
                first_atomic[atomic_indices],
            )
        )
    ].astype(np.int64)
    block_start_order = np.lexsort(
        (records["orientation_row"], records["launch_sequence"], start)
    ).astype(np.int64)
    return {
        "launch_count": int(launch_ids.size),
        "particle_ids": particle_ids.astype(np.int64),
        "first_atomic_order": first_atomic_order,
        "atomic_free_indices": np.flatnonzero(atomic_free).astype(np.int64),
        "block_start_order": block_start_order,
        "launch_summaries": launch_summaries,
        "sm_ids": np.unique(records["sm_id"]).astype(np.int64),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--n-particles", required=True, type=int)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-classes", type=int, default=1)
    parser.add_argument("--sm-count", type=int, default=132)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    capture = args.capture.expanduser().resolve(strict=True)
    header, records = load_capture(capture)
    result = validate_capture(
        header,
        records,
        iteration=args.iteration,
        n_particles=args.n_particles,
        n_threads=args.n_threads,
        n_classes=args.n_classes,
        sm_count=args.sm_count,
    )
    capture_sha256 = hashlib.sha256(capture.read_bytes()).hexdigest()
    output_npz = args.output_npz.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        schema_version=np.int64(1),
        iteration=np.int64(args.iteration),
        n_particles=np.int64(args.n_particles),
        n_threads=np.int64(args.n_threads),
        n_classes=np.int64(args.n_classes),
        sm_count=np.int64(args.sm_count),
        source_capture_sha256=np.asarray(capture_sha256),
        records=records,
        first_atomic_order=result["first_atomic_order"],
        atomic_free_indices=result["atomic_free_indices"],
        block_start_order=result["block_start_order"],
    )
    report = {
        "schema": "recovar.vdam_block_chronology.v1",
        "result": "pass",
        "source_capture": str(capture),
        "source_capture_sha256": capture_sha256,
        "iteration": args.iteration,
        "record_count": int(records.size),
        "launch_count": result["launch_count"],
        "particle_count": int(np.asarray(result["particle_ids"]).size),
        "n_threads": args.n_threads,
        "n_classes": args.n_classes,
        "sm_count": args.sm_count,
        "observed_sm_count": int(np.asarray(result["sm_ids"]).size),
        "observed_sm_ids": np.asarray(result["sm_ids"]).tolist(),
        "atomic_free_block_count": int(np.asarray(result["atomic_free_indices"]).size),
        "launches": result["launch_summaries"],
        "output_npz": str(output_npz),
    }
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
