#!/usr/bin/env python3
"""Fail-closed validation for RELION BPref pre-scatter capture artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_MAGIC = b"RLNBPREV1HEADER\0"
FOOTER_MAGIC = b"RLNBPREV1FOOTER\0"
HEADER_STRUCT = struct.Struct("<16s40Q")
FOOTER_STRUCT = struct.Struct("<16sQQ")
ROTATION_DTYPE = np.dtype(
    {
        "names": (
            "orientation_class_key",
            "oversampled_rotation",
            "matrix",
            "orientation_local",
            "reserved",
        ),
        "formats": ("<u8", "<u8", ("<f4", (9,)), "<u4", "<u4"),
        "offsets": (0, 8, 16, 52, 56),
        "itemsize": 64,
    }
)
ROW_DTYPE = np.dtype(
    {
        "names": (
            "state",
            "orientation_local",
            "pixel",
            "flags",
            "x",
            "y",
            "z",
            "source_re",
            "source_im",
            "source_weight",
        ),
        "formats": (
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<i4",
            "<i4",
            "<i4",
            "<f4",
            "<f4",
            "<f4",
        ),
        "offsets": tuple(range(0, 40, 4)),
        "itemsize": 40,
    }
)

ROW_FLAG_FWEIGHT_POSITIVE = 1
ROW_FLAG_RADIUS_SUPPORT = 2
ROW_FLAG_HERMITIAN_FOLD = 4
ROW_FLAG_DATA3D = 8
ROW_FLAG_MASK = 15
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)_img(?P<img>\d+)_class(?P<class_>\d+)\.bpre-v1\.bin"
)


@dataclass(frozen=True)
class CaptureArtifact:
    path: Path
    sha256: str
    header: tuple[int, ...]
    rotations: np.ndarray
    rows: np.ndarray

    @property
    def part_id(self) -> int:
        return self.header[7]

    @property
    def stack_index(self) -> int:
        return self.header[8]

    @property
    def mpi_rank(self) -> int:
        return self.header[10]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_artifact(path: Path) -> CaptureArtifact:
    """Load one sealed artifact and reject any structural ambiguity."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected capture file name: {path.name}")
    payload = path.read_bytes()
    _require(len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size, f"truncated artifact: {path}")

    magic, *values = HEADER_STRUCT.unpack_from(payload, 0)
    header = tuple(int(value) for value in values)
    _require(magic == HEADER_MAGIC, f"header magic mismatch: {path}")
    _require(header[0] == 1, f"schema version must be 1: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"header size mismatch: {path}")
    _require(header[2] == ROW_DTYPE.itemsize, f"row size mismatch: {path}")
    _require(header[3] == ROTATION_DTYPE.itemsize, f"rotation size mismatch: {path}")
    _require(header[4] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")

    rotation_count = header[16]
    row_count = header[17]
    expected_size = (
        HEADER_STRUCT.size
        + rotation_count * ROTATION_DTYPE.itemsize
        + row_count * ROW_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")

    rotation_offset = HEADER_STRUCT.size
    row_offset = rotation_offset + rotation_count * ROTATION_DTYPE.itemsize
    footer_offset = row_offset + row_count * ROW_DTYPE.itemsize
    rotations = np.frombuffer(
        payload, dtype=ROTATION_DTYPE, count=rotation_count, offset=rotation_offset
    ).copy()
    rows = np.frombuffer(payload, dtype=ROW_DTYPE, count=row_count, offset=row_offset).copy()
    footer_magic, footer_rows, footer_rotations = FOOTER_STRUCT.unpack_from(payload, footer_offset)
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(footer_rows == row_count, f"footer row count mismatch: {path}")
    _require(footer_rotations == rotation_count, f"footer rotation count mismatch: {path}")

    assert match is not None
    _require(int(match["part"]) == header[7], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[8], f"stack identity mismatch: {path}")
    _require(int(match["img"]) == header[9], f"image identity mismatch: {path}")
    _require(int(match["class_"]) == header[6], f"class identity mismatch: {path}")
    _validate_header(path, header)
    _validate_rotations(path, rotations)
    _validate_rows(path, header, rows)
    return CaptureArtifact(path, _sha256(path), header, rotations, rows)


def _validate_header(path: Path, header: tuple[int, ...]) -> None:
    _require(header[5] > 0, f"iteration must be positive: {path}")
    _require(header[6] > 0, f"class must be one-based and positive: {path}")
    _require(header[12] > 0 and header[13] > 0 and header[14] > 0, f"invalid image shape: {path}")
    _require(header[15] == header[12] * header[13] * header[14], f"image size mismatch: {path}")
    _require(header[16] > 0, f"orientation count must be positive: {path}")
    _require(header[18] > 0 and header[19] > 0, f"invalid support radius: {path}")
    padding = _float32_from_bits(header[20])
    threshold = _float32_from_bits(header[21])
    weight_norm = _float32_from_bits(header[22])
    _require(np.isfinite(padding) and padding > 0, f"invalid padding factor: {path}")
    _require(np.isfinite(threshold), f"invalid significant-weight threshold: {path}")
    _require(np.isfinite(weight_norm) and weight_norm > 0, f"invalid weight norm: {path}")
    _require(header[23] in (0, 1), f"invalid CTF-premultiplied flag: {path}")
    _require(header[24] == 0, f"schema v1 accepts 2D particle data only: {path}")
    _require(header[25] > 0 and header[26] > 0 and header[27] > 0, f"invalid completeness caps: {path}")
    _require(header[25] <= header[26] * header[27], f"particle cap cannot cover expectation: {path}")
    _require(header[29] <= header[28], f"capture byte cap exceeded: {path}")
    _require(header[30] != 0 and header[31] != 0, f"missing canonical identity hashes: {path}")
    _require(all(value > 0 for value in header[32:35]), f"invalid model shape: {path}")
    _require(header[37] == 1, f"artifact is not marked passive shadow-only: {path}")
    positive_count = header[38]
    excluded_count = header[39]
    _require(positive_count == header[17] + excluded_count, f"candidate/support accounting mismatch: {path}")
    _require(positive_count <= header[15] * header[16], f"positive candidate count exceeds panel: {path}")


def _validate_rotations(path: Path, rotations: np.ndarray) -> None:
    expected_local = np.arange(rotations.size, dtype=np.uint32)
    _require(np.array_equal(rotations["orientation_local"], expected_local), f"rotation order mismatch: {path}")
    _require(np.all(rotations["reserved"] == 0), f"nonzero reserved rotation field: {path}")
    _require(np.all(np.isfinite(rotations["matrix"])), f"non-finite Euler matrix: {path}")
    identities = np.stack(
        (rotations["orientation_class_key"], rotations["oversampled_rotation"]), axis=1
    )
    _require(np.unique(identities, axis=0).shape[0] == rotations.size, f"duplicate rotation identity: {path}")


def _validate_rows(path: Path, header: tuple[int, ...], rows: np.ndarray) -> None:
    if rows.size == 0:
        return
    _require(np.all(rows["state"] == 1), f"emitted row is not active: {path}")
    _require(np.all(rows["orientation_local"] < header[16]), f"row orientation is out of range: {path}")
    _require(np.all(rows["pixel"] < header[15]), f"row pixel is out of range: {path}")
    flags = rows["flags"]
    unknown_mask = np.uint32(~ROW_FLAG_MASK & 0xFFFFFFFF)
    _require(np.all((flags & unknown_mask) == 0), f"unknown row flag: {path}")
    required = ROW_FLAG_FWEIGHT_POSITIVE | ROW_FLAG_RADIUS_SUPPORT
    _require(np.all((flags & required) == required), f"emitted row lacks positive/support flags: {path}")
    _require(np.all((flags & ROW_FLAG_DATA3D) == 0), f"2D artifact contains DATA3D row: {path}")
    key = rows["orientation_local"].astype(np.int64) * header[15] + rows["pixel"]
    _require(np.all(np.diff(key) > 0), f"rows are duplicated or not in canonical device order: {path}")
    expected_x = rows["pixel"] % header[12]
    raw_y = rows["pixel"] // header[12]
    expected_y = np.where(raw_y > header[13] // 2, raw_y - header[13], raw_y)
    _require(np.array_equal(rows["x"], expected_x.astype(np.int32)), f"row x/pixel mismatch: {path}")
    _require(np.array_equal(rows["y"], expected_y.astype(np.int32)), f"row y/pixel mismatch: {path}")
    _require(np.all(rows["z"] == 0), f"2D row has nonzero z: {path}")
    source = np.stack((rows["source_re"], rows["source_im"], rows["source_weight"]), axis=1)
    _require(np.all(np.isfinite(source)), f"non-finite source operand: {path}")
    _require(np.all(rows["source_weight"] > 0), f"non-positive source weight: {path}")


def load_recovar_stack_indices(paths: Iterable[Path]) -> np.ndarray:
    indices = []
    expanded = []
    for path in paths:
        path = Path(path)
        expanded.extend(sorted(path.glob("*.npz")) if path.is_dir() else (path,))
    for path in expanded:
        with np.load(path, allow_pickle=False) as archive:
            _require("stack_indices_1based" in archive, f"missing stack_indices_1based: {path}")
            values = np.asarray(archive["stack_indices_1based"])
            _require(values.ndim == 1, f"stack_indices_1based must be one-dimensional: {path}")
            indices.append(values.astype(np.int64, copy=False))
    _require(bool(indices), "no RECOVAR capture shards were provided")
    combined = np.concatenate(indices)
    _require(np.unique(combined).size == combined.size, "RECOVAR stack identities are duplicated")
    return combined


def validate_directory(
    directory: Path,
    *,
    expected_particles: int | None = None,
    expected_stack_indices: np.ndarray | None = None,
) -> tuple[tuple[CaptureArtifact, ...], dict[str, object]]:
    """Validate a complete capture directory before any scientific comparison."""

    directory = Path(directory)
    _require(directory.is_dir(), f"capture directory does not exist: {directory}")
    incomplete = sorted(directory.glob("*.tmp.*"))
    if incomplete:
        raise ValueError(f"incomplete temporary artifacts remain: {incomplete[0]}")
    paths = sorted(directory.glob("*.bpre-v1.bin"))
    _require(bool(paths), f"no sealed capture artifacts in {directory}")
    artifacts = tuple(load_artifact(path) for path in paths)

    common_fields = (0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 18, 19, 20, 23, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 37)
    reference = artifacts[0].header
    for artifact in artifacts[1:]:
        for field in common_fields:
            _require(artifact.header[field] == reference[field], f"inconsistent header field {field}: {artifact.path}")

    configured_expected = reference[25]
    if expected_particles is not None:
        _require(configured_expected == expected_particles, "CLI and capture expected-particle counts differ")
    _require(len(artifacts) == configured_expected, f"capture completeness mismatch: {len(artifacts)} != {configured_expected}")
    part_ids = np.asarray([artifact.part_id for artifact in artifacts], dtype=np.uint64)
    stack_indices = np.asarray([artifact.stack_index for artifact in artifacts], dtype=np.uint64)
    image_hashes = np.asarray([artifact.header[31] for artifact in artifacts], dtype=np.uint64)
    _require(np.unique(part_ids).size == part_ids.size, "duplicate RELION part identity")
    _require(np.unique(stack_indices).size == stack_indices.size, "duplicate RELION stack identity")
    _require(np.unique(image_hashes).size == image_hashes.size, "duplicate RELION image identity hash")

    ranks, counts = np.unique(
        np.asarray([artifact.mpi_rank for artifact in artifacts], dtype=np.uint64), return_counts=True
    )
    _require(np.all(ranks != np.iinfo(np.uint64).max), "missing MPI rank identity")
    _require(ranks.size == reference[27], f"follower-rank coverage mismatch: {ranks.size} != {reference[27]}")
    _require(np.all(counts <= reference[26]), "per-rank particle cap exceeded")

    if expected_stack_indices is not None:
        expected = np.asarray(expected_stack_indices, dtype=np.uint64)
        _require(np.unique(expected).size == expected.size, "expected stack identities are duplicated")
        _require(np.array_equal(np.sort(stack_indices), np.sort(expected)), "RELION/RECOVAR stack identity sets differ")

    total_rows = sum(artifact.rows.size for artifact in artifacts)
    total_positive = sum(artifact.header[38] for artifact in artifacts)
    total_excluded = sum(artifact.header[39] for artifact in artifacts)
    summary: dict[str, object] = {
        "schema": "relion-bpref-prescatter-v1",
        "capture_directory": str(directory.resolve()),
        "iteration": reference[5],
        "class_one_based": reference[6],
        "particle_count": len(artifacts),
        "stack_index_min": int(stack_indices.min()),
        "stack_index_max": int(stack_indices.max()),
        "mpi_rank_counts": {str(int(rank)): int(count) for rank, count in zip(ranks, counts)},
        "emitted_supported_row_count": total_rows,
        "positive_fweight_candidate_count": total_positive,
        "radius_excluded_positive_fweight_count": total_excluded,
        "artifact_sha256": {artifact.path.name: artifact.sha256 for artifact in artifacts},
        "classification_ready": True,
    }
    return artifacts, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--expected-particles", type=int)
    parser.add_argument("--recovar-shard", action="append", default=[], type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    expected_stack_indices = None
    if args.recovar_shard:
        expected_stack_indices = load_recovar_stack_indices(args.recovar_shard)
    _, summary = validate_directory(
        args.capture_directory,
        expected_particles=args.expected_particles,
        expected_stack_indices=expected_stack_indices,
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
