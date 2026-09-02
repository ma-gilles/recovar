#!/usr/bin/env python3
"""Fail-closed validation of a class-complete RELION PPref capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "relion-k4-ppref-validation-v1"
MAGIC = b"RLNPPREFV1"
HEADER_WORDS = 16
FILE_NAME = re.compile(
    r"ppref_iter(?P<iteration>\d+)_rank(?P<rank>\d+)_model(?P<model>\d+)\.bin"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _signed(value: np.uint64) -> int:
    return int(np.asarray(value, dtype=np.uint64).view(np.int64).item())


def load_ppref(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and structurally validate one schema-v1 projector input."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected PPref file name: {path.name}")
    payload = path.read_bytes()
    header_bytes = HEADER_WORDS * np.dtype("<u8").itemsize
    _require(len(payload) >= 16 + header_bytes, f"truncated PPref capture: {path}")
    magic = payload[:16].split(b"\0", 1)[0]
    _require(magic == MAGIC, f"invalid PPref magic: {path}")
    header = np.frombuffer(payload, dtype="<u8", count=HEADER_WORDS, offset=16).copy()
    _require(int(header[0]) == 1, f"unsupported PPref schema: {path}")
    _require(int(header[14]) == 4, f"PPref scalar is not float32: {path}")
    _require(int(header[15]) == 0, f"PPref reserved header word is nonzero: {path}")

    iteration = int(header[1])
    rank = _signed(header[2])
    model = int(header[3])
    current_size = int(header[4])
    xdim, ydim, zdim = (int(value) for value in header[5:8])
    origin = [_signed(value) for value in header[8:11]]
    r_max = int(header[11])
    padding_factor = float(
        struct.unpack("<f", struct.pack("<I", int(header[12]) & 0xFFFFFFFF))[0]
    )
    count = int(header[13])
    expected_bytes = 16 + header_bytes + 2 * count * np.dtype("<f4").itemsize
    _require(len(payload) == expected_bytes, f"PPref byte count differs: {path}")
    _require(
        xdim > 0 and ydim > 0 and zdim > 0 and xdim * ydim * zdim == count,
        f"PPref dimensions differ from element count: {path}",
    )
    _require(ydim == zdim, f"PPref full axes differ: {path}")
    _require(xdim == ydim // 2 + 1, f"PPref half-spectrum axis differs: {path}")
    _require(origin == [0, -(ydim // 2), -(zdim // 2)], f"PPref origin differs: {path}")
    _require(current_size > 0 and r_max > 0, f"invalid PPref support: {path}")
    _require(np.isfinite(padding_factor) and padding_factor > 0, f"invalid PPref padding: {path}")
    assert match is not None
    _require(int(match["iteration"]) == iteration, f"PPref iteration identity mismatch: {path}")
    _require(int(match["rank"]) == rank, f"PPref rank identity mismatch: {path}")
    _require(int(match["model"]) == model, f"PPref model identity mismatch: {path}")

    values = np.frombuffer(payload, dtype="<f4", offset=16 + header_bytes).copy()
    ppref = (values[0::2] + np.complex64(1j) * values[1::2]).astype(
        np.complex64
    ).reshape(zdim, ydim, xdim)
    _require(np.isfinite(ppref).all(), f"non-finite PPref payload: {path}")
    _require(bool(np.any(ppref != 0)), f"zero PPref payload: {path}")
    metadata = {
        "version": 1,
        "iteration": iteration,
        "rank": rank,
        "model": model,
        "current_size": current_size,
        "shape_zyx": [zdim, ydim, xdim],
        "origin_xyz": origin,
        "r_max": r_max,
        "padding_factor": padding_factor,
        "complex_count": count,
    }
    return ppref, metadata


def validate_directory(
    capture_dir: Path,
    *,
    expected_iteration: int,
    expected_rank: int = 0,
    n_classes: int = 4,
    expected_current_size: int | None = None,
) -> dict[str, Any]:
    """Require exactly one structurally compatible PPref for every class."""

    capture_dir = Path(capture_dir)
    _require(capture_dir.is_dir(), f"missing PPref capture directory: {capture_dir}")
    _require(n_classes > 0, "class count must be positive")
    paths = sorted(capture_dir.glob("ppref_iter*_rank*_model*.bin"))
    _require(len(paths) == n_classes, "PPref class panel is incomplete or has extras")
    records: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    for path in paths:
        ppref, metadata = load_ppref(path)
        arrays.append(ppref)
        records.append(
            {
                **metadata,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "l2_norm": float(np.linalg.norm(ppref.astype(np.complex128))),
                "max_abs": float(np.max(np.abs(ppref))),
            }
        )
    _require(
        [record["model"] for record in records] == list(range(n_classes)),
        "PPref model identities are not a complete zero-based class panel",
    )
    _require(
        all(record["iteration"] == expected_iteration for record in records),
        "PPref iteration differs from expected iteration",
    )
    _require(
        all(record["rank"] == expected_rank for record in records),
        "PPref rank differs from expected rank",
    )
    shared_fields = ("current_size", "shape_zyx", "origin_xyz", "r_max", "padding_factor")
    for field in shared_fields:
        _require(
            all(record[field] == records[0][field] for record in records[1:]),
            f"PPref classes differ in shared field {field}",
        )
    if expected_current_size is not None:
        _require(
            records[0]["current_size"] == expected_current_size,
            "PPref current size differs from expected current size",
        )
    identical_pairs = [
        [left, right]
        for left in range(n_classes)
        for right in range(left + 1, n_classes)
        if np.array_equal(arrays[left], arrays[right])
    ]
    return {
        "schema": SCHEMA,
        "status": "pass",
        "metric_policy": "exact schema/topology/identity plus SHA-256; no tolerance",
        "capture_dir": str(capture_dir.resolve()),
        "iteration": expected_iteration,
        "rank": expected_rank,
        "class_count": n_classes,
        "shared": {field: records[0][field] for field in shared_fields},
        "bitwise_identical_class_pairs": identical_pairs,
        "classes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--expected-iteration", required=True, type=int)
    parser.add_argument("--expected-rank", type=int, default=0)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--expected-current-size", type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = validate_directory(
        args.capture_dir,
        expected_iteration=args.expected_iteration,
        expected_rank=args.expected_rank,
        n_classes=args.n_classes,
        expected_current_size=args.expected_current_size,
    )
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
