#!/usr/bin/env python3
"""Fail-closed validation for compact RELION BPref rotation-mass captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_MAGIC = b"RLNBPMV2HEADER\0\0"
FOOTER_MAGIC = b"RLNBPMV2FOOTER\0\0"
HEADER_STRUCT = struct.Struct("<16s40Q")
FOOTER_STRUCT = struct.Struct("<16sQQ")
UNKNOWN_MPI_RANK = np.iinfo(np.uint64).max
ROW_DTYPE = np.dtype(
    {
        "names": (
            "orientation_class_key",
            "oversampled_rotation",
            "matrix",
            "orientation_local",
            "reserved",
            "candidate_translation_count",
            "significant_translation_count",
            "posterior_rotation_mass",
            "reconstruction_rotation_mass",
        ),
        "formats": (
            "<u8",
            "<u8",
            ("<f4", (9,)),
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<f4",
            "<f4",
        ),
        "offsets": (0, 8, 16, 52, 56, 64, 68, 72, 76),
        "itemsize": 80,
    }
)
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)_img(?P<img>\d+)"
    r"_class(?P<class_>\d+)\.bpm-v2\.bin"
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


def _float32_from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


@dataclass(frozen=True)
class RotationMassArtifact:
    path: Path
    sha256: str
    header: tuple[int, ...]
    rows: np.ndarray

    @property
    def part_id(self) -> int:
        return self.header[6]

    @property
    def stack_index(self) -> int:
        return self.header[7]

    @property
    def mpi_rank(self) -> int:
        return self.header[9]

    @property
    def significant_weight(self) -> float:
        return _float32_from_bits(self.header[13])

    @property
    def weight_norm(self) -> float:
        return _float32_from_bits(self.header[14])


def load_artifact(path: Path) -> RotationMassArtifact:
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
    _require(header[0] == 2, f"schema version must be 2: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"header size mismatch: {path}")
    _require(header[2] == ROW_DTYPE.itemsize, f"row size mismatch: {path}")
    _require(header[3] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")
    translation_count = header[11]
    row_count = header[12]
    _require(translation_count > 0 and row_count > 0, f"empty topology: {path}")
    expected_size = (
        HEADER_STRUCT.size
        + row_count * ROW_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")
    rows = np.frombuffer(
        payload,
        dtype=ROW_DTYPE,
        count=row_count,
        offset=HEADER_STRUCT.size,
    ).copy()
    footer_offset = HEADER_STRUCT.size + row_count * ROW_DTYPE.itemsize
    footer_magic, footer_significant, footer_rows = FOOTER_STRUCT.unpack_from(
        payload,
        footer_offset,
    )
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(footer_significant == header[23], f"significant count mismatch: {path}")
    _require(footer_rows == row_count, f"footer row count mismatch: {path}")

    assert match is not None
    _require(int(match["part"]) == header[6], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"stack identity mismatch: {path}")
    _require(int(match["img"]) == header[8], f"image identity mismatch: {path}")
    _require(int(match["class_"]) == header[5], f"class identity mismatch: {path}")
    _require(header[4] > 0 and header[5] > 0, f"invalid iteration/class: {path}")
    _require(header[15] <= header[16] * header[17], f"particle cap is too small: {path}")
    _require(header[19] <= header[18], f"capture byte cap exceeded: {path}")
    _require(header[20] != 0 and header[21] != 0, f"identity hash is missing: {path}")
    _require(header[22] > 0, f"current size must be positive: {path}")
    _require(row_count <= header[25], f"filtered rows exceed orientation count: {path}")
    _require(
        header[24] == header[25] * translation_count,
        f"original weight topology mismatch: {path}",
    )
    _require(np.all(rows["reserved"] == 0), f"reserved field is nonzero: {path}")
    _require(np.all(np.isfinite(rows["matrix"])), f"non-finite rotation: {path}")
    identities = np.stack(
        (rows["orientation_class_key"], rows["oversampled_rotation"]),
        axis=1,
    )
    _require(
        np.unique(identities, axis=0).shape[0] == row_count,
        f"duplicate rotation identity: {path}",
    )
    candidate_count = rows["candidate_translation_count"].astype(np.uint64)
    significant_count = rows["significant_translation_count"].astype(np.uint64)
    _require(
        np.all((candidate_count > 0) & (candidate_count <= translation_count)),
        f"candidate translation count is invalid: {path}",
    )
    _require(
        np.all(significant_count <= candidate_count),
        f"significant translations exceed candidates: {path}",
    )
    _require(
        int(np.sum(significant_count, dtype=np.uint64)) == header[23],
        f"significant sample total mismatch: {path}",
    )
    posterior = rows["posterior_rotation_mass"]
    reconstruction = rows["reconstruction_rotation_mass"]
    _require(
        np.all(np.isfinite(posterior)) and np.all(np.isfinite(reconstruction)),
        f"non-finite rotation mass: {path}",
    )
    _require(
        np.all(posterior >= 0) and np.all(reconstruction >= 0),
        f"negative rotation mass: {path}",
    )
    _require(
        np.all(reconstruction <= posterior + np.finfo(np.float32).eps),
        f"reconstruction mass exceeds posterior mass: {path}",
    )
    _require(
        np.isfinite(_float32_from_bits(header[13]))
        and _float32_from_bits(header[13]) >= 0,
        f"invalid significant threshold: {path}",
    )
    _require(
        np.isfinite(_float32_from_bits(header[14]))
        and _float32_from_bits(header[14]) > 0,
        f"invalid weight norm: {path}",
    )
    return RotationMassArtifact(
        path=path,
        sha256=_sha256(path),
        header=header,
        rows=rows,
    )


def validate_directory(
    directory: Path,
    *,
    expected_particles: int | None = None,
) -> tuple[tuple[RotationMassArtifact, ...], dict[str, object]]:
    directory = Path(directory)
    _require(directory.is_dir(), f"capture directory does not exist: {directory}")
    _require(not list(directory.glob("*.tmp.*")), "incomplete capture artifact remains")
    paths = sorted(directory.glob("*.bpm-v2.bin"))
    _require(bool(paths), f"no rotation-mass artifacts in {directory}")
    artifacts = tuple(load_artifact(path) for path in paths)
    reference = artifacts[0].header
    common_fields = (0, 1, 2, 3, 4, 5, 11, 15, 16, 17, 18, 22)
    for artifact in artifacts[1:]:
        for field in common_fields:
            _require(
                artifact.header[field] == reference[field],
                f"inconsistent header field {field}: {artifact.path}",
            )
    configured_expected = reference[15]
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
    stacks = np.asarray([artifact.stack_index for artifact in artifacts], dtype=np.uint64)
    _require(np.unique(part_ids).size == part_ids.size, "duplicate RELION part identity")
    _require(np.unique(stacks).size == stacks.size, "duplicate RELION stack identity")
    ranks, rank_counts = np.unique(
        np.asarray([artifact.mpi_rank for artifact in artifacts], dtype=np.uint64),
        return_counts=True,
    )
    unknown_rank = ranks == UNKNOWN_MPI_RANK
    _require(
        np.all(unknown_rank) or not np.any(unknown_rank),
        "MPI rank tracking is inconsistently available",
    )
    if np.all(unknown_rank):
        mpi_rank_tracking = "unavailable_srun_environment"
        mpi_rank_counts = {"unknown": len(artifacts)}
    else:
        _require(ranks.size == reference[17], "follower-rank coverage mismatch")
        _require(np.all(rank_counts <= reference[16]), "per-rank particle cap exceeded")
        mpi_rank_tracking = "available"
        mpi_rank_counts = {
            str(int(rank)): int(count)
            for rank, count in zip(ranks, rank_counts, strict=True)
        }
    summary: dict[str, object] = {
        "schema": "relion-bpref-rotation-mass-v2",
        "capture_directory": str(directory.resolve()),
        "iteration": reference[4],
        "class_one_based": reference[5],
        "current_size": reference[22],
        "particle_count": len(artifacts),
        "mpi_rank_tracking": mpi_rank_tracking,
        "mpi_rank_counts": mpi_rank_counts,
        "total_candidate_rotation_count": int(
            sum(artifact.rows.size for artifact in artifacts)
        ),
        "total_significant_sample_count": int(
            sum(int(artifact.header[23]) for artifact in artifacts)
        ),
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
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    _, summary = validate_directory(
        args.capture_directory,
        expected_particles=args.expected_particles,
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
