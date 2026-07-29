#!/usr/bin/env python3
"""Fail-closed validation for passive RELION BPref membership captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_MAGIC = b"RLNBPMV1HEADER\0\0"
FOOTER_MAGIC = b"RLNBPMV1FOOTER\0\0"
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
WEIGHT_DTYPE = np.dtype("<f4")
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)_img(?P<img>\d+)"
    r"_class(?P<class_>\d+)\.bpm-v1\.bin"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


@dataclass(frozen=True)
class MembershipArtifact:
    path: Path
    sha256: str
    header: tuple[int, ...]
    rotations: np.ndarray
    weights: np.ndarray

    @property
    def part_id(self) -> int:
        return self.header[7]

    @property
    def stack_index(self) -> int:
        return self.header[8]

    @property
    def mpi_rank(self) -> int:
        return self.header[10]

    @property
    def significant_weight(self) -> float:
        return _float32_from_bits(self.header[15])

    @property
    def weight_norm(self) -> float:
        return _float32_from_bits(self.header[16])


def load_artifact(path: Path) -> MembershipArtifact:
    """Load one artifact and validate its complete byte-level structure."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected capture file name: {path.name}")
    payload = path.read_bytes()
    _require(
        len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size,
        f"truncated artifact: {path}",
    )
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload, 0)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"header magic mismatch: {path}")
    _require(header[0] == 1, f"schema version must be 1: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"header size mismatch: {path}")
    _require(header[2] == ROTATION_DTYPE.itemsize, f"rotation size mismatch: {path}")
    _require(header[3] == WEIGHT_DTYPE.itemsize, f"weight size mismatch: {path}")
    _require(header[4] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")

    rotation_count = header[12]
    translation_count = header[13]
    weight_count = header[14]
    _require(rotation_count > 0, f"orientation count must be positive: {path}")
    _require(translation_count > 0, f"translation count must be positive: {path}")
    _require(
        weight_count == rotation_count * translation_count,
        f"weight topology mismatch: {path}",
    )
    expected_size = (
        HEADER_STRUCT.size
        + rotation_count * ROTATION_DTYPE.itemsize
        + weight_count * WEIGHT_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")

    rotation_offset = HEADER_STRUCT.size
    weight_offset = rotation_offset + rotation_count * ROTATION_DTYPE.itemsize
    footer_offset = weight_offset + weight_count * WEIGHT_DTYPE.itemsize
    rotations = np.frombuffer(
        payload,
        dtype=ROTATION_DTYPE,
        count=rotation_count,
        offset=rotation_offset,
    ).copy()
    weights = np.frombuffer(
        payload,
        dtype=WEIGHT_DTYPE,
        count=weight_count,
        offset=weight_offset,
    ).copy()
    footer_magic, footer_weights, footer_rotations = FOOTER_STRUCT.unpack_from(
        payload, footer_offset
    )
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(footer_weights == weight_count, f"footer weight count mismatch: {path}")
    _require(
        footer_rotations == rotation_count,
        f"footer rotation count mismatch: {path}",
    )

    assert match is not None
    _require(int(match["part"]) == header[7], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[8], f"stack identity mismatch: {path}")
    _require(int(match["img"]) == header[9], f"image identity mismatch: {path}")
    _require(int(match["class_"]) == header[6], f"class identity mismatch: {path}")
    _require(header[5] > 0, f"iteration must be positive: {path}")
    _require(header[6] > 0, f"class must be one-based and positive: {path}")
    _require(header[17] > 0 and header[18] > 0, f"invalid completeness cap: {path}")
    _require(header[19] > 0, f"expected follower count must be positive: {path}")
    _require(
        header[17] <= header[18] * header[19],
        f"particle cap cannot cover expectation: {path}",
    )
    _require(header[21] <= header[20], f"capture byte cap exceeded: {path}")
    _require(header[22] != 0 and header[23] != 0, f"missing identity hashes: {path}")
    _require(header[24] == 1, f"artifact is not marked passive: {path}")
    _require(header[25] > 0, f"current size must be positive: {path}")
    _require(
        np.array_equal(
            rotations["orientation_local"],
            np.arange(rotation_count, dtype=np.uint32),
        ),
        f"rotation order mismatch: {path}",
    )
    _require(np.all(rotations["reserved"] == 0), f"reserved field is nonzero: {path}")
    _require(np.all(np.isfinite(rotations["matrix"])), f"non-finite rotation: {path}")
    identities = np.stack(
        (rotations["orientation_class_key"], rotations["oversampled_rotation"]),
        axis=1,
    )
    _require(
        np.unique(identities, axis=0).shape[0] == rotation_count,
        f"duplicate rotation identity: {path}",
    )
    _require(np.all(np.isfinite(weights)), f"non-finite posterior weight: {path}")
    _require(np.all(weights >= 0.0), f"negative posterior weight: {path}")
    significant_weight = _float32_from_bits(header[15])
    weight_norm = _float32_from_bits(header[16])
    _require(
        np.isfinite(significant_weight) and significant_weight >= 0.0,
        f"invalid significant weight: {path}",
    )
    _require(
        np.isfinite(weight_norm) and weight_norm > 0.0,
        f"invalid weight norm: {path}",
    )
    return MembershipArtifact(
        path=path,
        sha256=_sha256(path),
        header=header,
        rotations=rotations,
        weights=weights.reshape(rotation_count, translation_count),
    )


def validate_directory(
    directory: Path,
    *,
    expected_particles: int | None = None,
    expected_stack_indices: np.ndarray | None = None,
    expected_stack_mpi_rank: int | None = None,
) -> tuple[tuple[MembershipArtifact, ...], dict[str, object]]:
    """Validate a complete capture directory before scientific comparison."""

    directory = Path(directory)
    _require(directory.is_dir(), f"capture directory does not exist: {directory}")
    incomplete = sorted(directory.glob("*.tmp.*"))
    _require(not incomplete, f"incomplete capture artifact remains: {incomplete[:1]}")
    paths = sorted(directory.glob("*.bpm-v1.bin"))
    _require(bool(paths), f"no membership capture artifacts in {directory}")
    artifacts = tuple(load_artifact(path) for path in paths)
    reference = artifacts[0].header
    common_fields = (0, 1, 2, 3, 4, 5, 6, 13, 17, 18, 19, 20, 24, 25)
    for artifact in artifacts[1:]:
        for field in common_fields:
            _require(
                artifact.header[field] == reference[field],
                f"inconsistent header field {field}: {artifact.path}",
            )
    configured_expected = reference[17]
    if expected_particles is not None:
        _require(
            configured_expected == expected_particles,
            "CLI and capture expected-particle counts differ",
        )
    _require(
        len(artifacts) == configured_expected,
        f"capture completeness mismatch: {len(artifacts)} != {configured_expected}",
    )
    part_ids = np.asarray([artifact.part_id for artifact in artifacts], dtype=np.uint64)
    stack_indices = np.asarray(
        [artifact.stack_index for artifact in artifacts], dtype=np.uint64
    )
    _require(np.unique(part_ids).size == part_ids.size, "duplicate RELION part identity")
    _require(
        np.unique(stack_indices).size == stack_indices.size,
        "duplicate RELION stack identity",
    )
    ranks, rank_counts = np.unique(
        np.asarray([artifact.mpi_rank for artifact in artifacts], dtype=np.uint64),
        return_counts=True,
    )
    _require(ranks.size == reference[19], "follower-rank coverage mismatch")
    _require(np.all(rank_counts <= reference[18]), "per-rank particle cap exceeded")

    if expected_stack_indices is not None:
        expected = np.asarray(expected_stack_indices, dtype=np.uint64)
        _require(
            np.unique(expected).size == expected.size,
            "expected stack identities are duplicated",
        )
        observed = stack_indices
        if expected_stack_mpi_rank is not None:
            _require(expected_stack_mpi_rank >= 0, "MPI rank must be nonnegative")
            observed = np.asarray(
                [
                    artifact.stack_index
                    for artifact in artifacts
                    if artifact.mpi_rank == expected_stack_mpi_rank
                ],
                dtype=np.uint64,
            )
        _require(
            np.array_equal(np.sort(observed), np.sort(expected)),
            "RELION/expected stack identity sets differ",
        )

    positive_samples = [
        int(np.count_nonzero(artifact.weights >= artifact.significant_weight))
        for artifact in artifacts
    ]
    summary: dict[str, object] = {
        "schema": "relion-bpref-membership-v1",
        "capture_directory": str(directory.resolve()),
        "iteration": reference[5],
        "class_one_based": reference[6],
        "current_size": reference[25],
        "particle_count": len(artifacts),
        "stack_index_min": int(stack_indices.min()),
        "stack_index_max": int(stack_indices.max()),
        "mpi_rank_counts": {
            str(int(rank)): int(count) for rank, count in zip(ranks, rank_counts)
        },
        "total_rotation_count": int(
            sum(artifact.rotations.size for artifact in artifacts)
        ),
        "total_weight_count": int(sum(artifact.weights.size for artifact in artifacts)),
        "total_significant_sample_count": int(sum(positive_samples)),
        "artifact_sha256": {
            artifact.path.name: artifact.sha256 for artifact in artifacts
        },
        "classification_ready": True,
    }
    return artifacts, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--expected-particles", type=int)
    parser.add_argument("--expected-stack-indices-npy", type=Path)
    parser.add_argument("--expected-stack-mpi-rank", type=int)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    expected_stacks = None
    if args.expected_stack_indices_npy is not None:
        expected_stacks = np.load(args.expected_stack_indices_npy, allow_pickle=False)
    _, summary = validate_directory(
        args.capture_directory,
        expected_particles=args.expected_particles,
        expected_stack_indices=expected_stacks,
        expected_stack_mpi_rank=args.expected_stack_mpi_rank,
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
