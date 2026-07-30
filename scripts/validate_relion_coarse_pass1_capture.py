#!/usr/bin/env python3
"""Fail-closed validation for bounded RELION coarse pass-1 captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_MAGIC = b"RLNP1V1HEADER\0\0\0"
FOOTER_MAGIC = b"RLNP1V1FOOTER\0\0\0"
HEADER_STRUCT = struct.Struct("<16s40Q")
FOOTER_STRUCT = struct.Struct("<16sQQ")
FLOAT_DTYPE = np.dtype("<f4")
MASK_DTYPE = np.dtype("u1")
FILE_NAME = re.compile(r"part(?P<part>\d+)_stack(?P<stack>\d+)\.p1-v1\.bin")


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
class CoarsePass1Artifact:
    path: Path
    sha256: str
    header: tuple[int, ...]
    raw_diff2: np.ndarray
    weights: np.ndarray
    significant_mask: np.ndarray
    translations: np.ndarray
    inferred_significant_weight: float
    significant_weight_semantics: str

    @property
    def part_id(self) -> int:
        return self.header[6]

    @property
    def stack_index(self) -> int:
        return self.header[7]

    @property
    def mpi_rank(self) -> int:
        return self.header[8]

    @property
    def significant_weight(self) -> float:
        return _float32_from_bits(self.header[16])

    @property
    def sum_weight(self) -> float:
        return _float32_from_bits(self.header[17])

    @property
    def max_weight(self) -> float:
        return _float32_from_bits(self.header[18])

    @property
    def min_diff2(self) -> float:
        return _float32_from_bits(self.header[19])


def load_artifact(path: Path) -> CoarsePass1Artifact:
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
    _require(header[2] == FLOAT_DTYPE.itemsize, f"float size mismatch: {path}")
    _require(header[3] == MASK_DTYPE.itemsize, f"mask size mismatch: {path}")
    _require(header[4] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")

    n_dir, n_psi, n_trans = header[10:13]
    candidate_count = header[13]
    significant_count = header[14]
    translation_value_count = header[30]
    _require(n_dir > 0 and n_psi > 0 and n_trans > 0, f"empty topology: {path}")
    _require(
        candidate_count == n_dir * n_psi * n_trans,
        f"candidate topology mismatch: {path}",
    )
    _require(
        translation_value_count == 2 * n_trans,
        f"translation topology mismatch: {path}",
    )
    expected_size = (
        HEADER_STRUCT.size
        + 2 * candidate_count * FLOAT_DTYPE.itemsize
        + candidate_count * MASK_DTYPE.itemsize
        + translation_value_count * FLOAT_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")

    raw_offset = HEADER_STRUCT.size
    weight_offset = raw_offset + candidate_count * FLOAT_DTYPE.itemsize
    mask_offset = weight_offset + candidate_count * FLOAT_DTYPE.itemsize
    translation_offset = mask_offset + candidate_count * MASK_DTYPE.itemsize
    footer_offset = translation_offset + translation_value_count * FLOAT_DTYPE.itemsize
    raw_diff2 = np.frombuffer(
        payload,
        dtype=FLOAT_DTYPE,
        count=candidate_count,
        offset=raw_offset,
    ).copy()
    weights = np.frombuffer(
        payload,
        dtype=FLOAT_DTYPE,
        count=candidate_count,
        offset=weight_offset,
    ).copy()
    significant_mask_u8 = np.frombuffer(
        payload,
        dtype=MASK_DTYPE,
        count=candidate_count,
        offset=mask_offset,
    ).copy()
    translations = np.frombuffer(
        payload,
        dtype=FLOAT_DTYPE,
        count=translation_value_count,
        offset=translation_offset,
    ).copy()
    footer_magic, footer_candidates, footer_significant = FOOTER_STRUCT.unpack_from(
        payload, footer_offset
    )
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(
        footer_candidates == candidate_count,
        f"footer candidate count mismatch: {path}",
    )
    _require(
        footer_significant == significant_count,
        f"footer significant count mismatch: {path}",
    )

    assert match is not None
    _require(int(match["part"]) == header[6], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"stack identity mismatch: {path}")
    _require(header[5] > 0, f"iteration must be positive: {path}")
    _require(header[15] < candidate_count, f"winner index is out of range: {path}")
    _require(header[20] > 0 and header[21] > 0, f"invalid completeness cap: {path}")
    _require(header[22] > 0, f"expected follower count must be positive: {path}")
    _require(
        header[20] <= header[21] * header[22],
        f"particle cap cannot cover expectation: {path}",
    )
    _require(header[24] <= header[23], f"capture byte cap exceeded: {path}")
    _require(header[25] != 0 and header[26] != 0, f"missing identity hashes: {path}")
    _require(header[27] > 0, f"current size must be positive: {path}")
    _require(header[31:34] == (1, 1, 1), f"passive payload flags missing: {path}")
    _require(np.all(np.isfinite(raw_diff2)), f"non-finite raw diff2: {path}")
    _require(np.all(np.isfinite(weights)), f"non-finite production weight: {path}")
    _require(np.all(weights >= 0), f"negative production weight: {path}")
    _require(
        np.all((significant_mask_u8 == 0) | (significant_mask_u8 == 1)),
        f"non-binary significance mask: {path}",
    )
    significant_mask = significant_mask_u8.astype(bool)
    _require(
        np.count_nonzero(significant_mask) == significant_count,
        f"significant mask count mismatch: {path}",
    )
    significant_weight = _float32_from_bits(header[16])
    sum_weight = _float32_from_bits(header[17])
    max_weight = _float32_from_bits(header[18])
    min_diff2 = _float32_from_bits(header[19])
    _require(
        np.isfinite(significant_weight) and significant_weight >= 0,
        f"invalid significant weight: {path}",
    )
    _require(
        np.isfinite(sum_weight) and sum_weight > 0,
        f"invalid sum weight: {path}",
    )
    _require(
        np.isfinite(max_weight) and max_weight > 0,
        f"invalid maximum weight: {path}",
    )
    _require(np.isfinite(min_diff2), f"invalid minimum diff2: {path}")
    _require(np.any(significant_mask), f"empty production significance mask: {path}")
    inferred_significant_weight = float(np.min(weights[significant_mask]))
    inferred_mask = weights >= np.float32(inferred_significant_weight)
    _require(
        np.array_equal(significant_mask, inferred_mask),
        f"production weight rank/mask mismatch: {path}",
    )
    recorded_mask = weights >= significant_weight
    if np.array_equal(significant_mask, recorded_mask):
        significant_weight_semantics = "recorded_threshold_reproduces_mask"
    else:
        _require(
            significant_weight == 0.0,
            f"recorded threshold/mask mismatch without zero sentinel: {path}",
        )
        significant_weight_semantics = (
            "relion_cuda_coarse_op_sentinel_zero__"
            "threshold_inferred_from_exact_production_mask"
        )
    _require(
        weights[header[15]] == max_weight and np.max(weights) == max_weight,
        f"winner/maximum weight mismatch: {path}",
    )
    translations = translations.reshape(n_trans, 2)
    _require(np.all(np.isfinite(translations)), f"non-finite translation: {path}")
    return CoarsePass1Artifact(
        path=path,
        sha256=_sha256(path),
        header=header,
        raw_diff2=raw_diff2.reshape(n_dir * n_psi, n_trans),
        weights=weights.reshape(n_dir * n_psi, n_trans),
        significant_mask=significant_mask.reshape(n_dir * n_psi, n_trans),
        translations=translations,
        inferred_significant_weight=inferred_significant_weight,
        significant_weight_semantics=significant_weight_semantics,
    )


def validate_directory(
    directory: Path,
    *,
    expected_particles: int | None = None,
    expected_stack_indices: np.ndarray | None = None,
    expected_mpi_rank: int | None = None,
) -> tuple[tuple[CoarsePass1Artifact, ...], dict[str, object]]:
    """Validate a complete capture directory before scientific comparison."""

    directory = Path(directory)
    _require(directory.is_dir(), f"capture directory does not exist: {directory}")
    incomplete = sorted(directory.glob("*.tmp.*"))
    _require(not incomplete, f"incomplete capture artifact remains: {incomplete[:1]}")
    paths = sorted(directory.glob("*.p1-v1.bin"))
    _require(bool(paths), f"no coarse-pass1 capture artifacts in {directory}")
    artifacts = tuple(load_artifact(path) for path in paths)
    reference = artifacts[0].header
    common_fields = (0, 1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32, 33)
    for artifact in artifacts[1:]:
        for field in common_fields:
            _require(
                artifact.header[field] == reference[field],
                f"inconsistent header field {field}: {artifact.path}",
            )
    configured_expected = reference[20]
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
    ranks, rank_counts = np.unique(
        np.asarray([artifact.mpi_rank for artifact in artifacts], dtype=np.uint64),
        return_counts=True,
    )
    _require(np.unique(part_ids).size == part_ids.size, "duplicate RELION part identity")
    _require(
        np.unique(stack_indices).size == stack_indices.size,
        "duplicate RELION stack identity",
    )
    _require(ranks.size == reference[22], "follower-rank coverage mismatch")
    _require(np.all(rank_counts <= reference[21]), "per-rank particle cap exceeded")
    if expected_mpi_rank is not None:
        _require(
            np.array_equal(ranks, np.asarray([expected_mpi_rank], dtype=np.uint64)),
            "unexpected MPI follower rank",
        )
    if expected_stack_indices is not None:
        expected = np.asarray(expected_stack_indices, dtype=np.uint64)
        _require(
            np.unique(expected).size == expected.size,
            "expected stack identities are duplicated",
        )
        _require(
            np.array_equal(np.sort(stack_indices), np.sort(expected)),
            "RELION/expected stack identity sets differ",
        )

    summary: dict[str, object] = {
        "schema": "relion-coarse-pass1-capture-v1",
        "capture_directory": str(directory.resolve()),
        "iteration": reference[5],
        "current_size": reference[27],
        "particle_count": len(artifacts),
        "topology": {
            "n_directions": reference[10],
            "n_psi": reference[11],
            "n_translations": reference[12],
            "candidate_count_per_particle": reference[13],
        },
        "mpi_rank_counts": {
            str(int(rank)): int(count) for rank, count in zip(ranks, rank_counts)
        },
        "total_candidate_count": int(sum(artifact.weights.size for artifact in artifacts)),
        "total_significant_count": int(
            sum(np.count_nonzero(artifact.significant_mask) for artifact in artifacts)
        ),
        "significant_weight_semantics_counts": {
            semantics: sum(
                artifact.significant_weight_semantics == semantics
                for artifact in artifacts
            )
            for semantics in sorted(
                {artifact.significant_weight_semantics for artifact in artifacts}
            )
        },
        "inferred_significant_weight_range": {
            "min": float(
                min(artifact.inferred_significant_weight for artifact in artifacts)
            ),
            "max": float(
                max(artifact.inferred_significant_weight for artifact in artifacts)
            ),
        },
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
    parser.add_argument("--expected-mpi-rank", type=int)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    expected_stacks = None
    if args.expected_stack_indices_npy is not None:
        expected_stacks = np.load(args.expected_stack_indices_npy, allow_pickle=False)
    _, summary = validate_directory(
        args.capture_directory,
        expected_particles=args.expected_particles,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=args.expected_mpi_rank,
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
