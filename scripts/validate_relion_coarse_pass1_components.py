#!/usr/bin/env python3
"""Fail-closed validation of RELION coarse diff2 component captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_MAGIC = b"RLNP1V2HEADER\0\0\0"
FOOTER_MAGIC = b"RLNP1V2FOOTER\0\0\0"
HEADER_STRUCT = struct.Struct("<16s40Q")
FOOTER_STRUCT = struct.Struct("<16sQQ")
FLOAT_DTYPE = np.dtype("<f4")
MASK_DTYPE = np.dtype("u1")
FILE_NAME = re.compile(r"part(?P<part>\d+)_stack(?P<stack>\d+)\.p1-v2\.bin")
DEFAULT_REPLAY_P95_MAX_ABS = 5.0e-5
DEFAULT_REPLAY_MAX_ABS = 5.0e-4
DEFAULT_REFERENCE_TRANSLATION_SPREAD = 1.0e-6


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
class CoarsePass1Components:
    path: Path
    sha256: str
    header: tuple[int, ...]
    raw_diff2: np.ndarray
    weights: np.ndarray
    reference_norms: np.ndarray
    cross_terms: np.ndarray
    significant_mask: np.ndarray
    translations: np.ndarray

    @property
    def part_id(self) -> int:
        return self.header[6]

    @property
    def stack_index(self) -> int:
        return self.header[7]

    @property
    def mpi_rank(self) -> int:
        return self.header[8]


def load_artifact(path: Path) -> CoarsePass1Components:
    """Load and validate one complete schema-v2 artifact."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected capture file name: {path.name}")
    payload = path.read_bytes()
    _require(
        len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size,
        f"truncated artifact: {path}",
    )
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"header magic mismatch: {path}")
    _require(header[0] == 2 and header[36] == 2, f"schema mismatch: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"header size mismatch: {path}")
    _require(header[2] == FLOAT_DTYPE.itemsize, f"float size mismatch: {path}")
    _require(header[3] == MASK_DTYPE.itemsize, f"mask size mismatch: {path}")
    _require(header[4] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")
    _require(header[31:36] == (1, 1, 1, 1, 1), f"payload flags missing: {path}")

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
        + 4 * candidate_count * FLOAT_DTYPE.itemsize
        + candidate_count * MASK_DTYPE.itemsize
        + translation_value_count * FLOAT_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")

    offset = HEADER_STRUCT.size

    def take_float(count: int) -> np.ndarray:
        nonlocal offset
        values = np.frombuffer(
            payload, dtype=FLOAT_DTYPE, count=count, offset=offset
        ).copy()
        offset += count * FLOAT_DTYPE.itemsize
        return values

    raw_diff2 = take_float(candidate_count)
    weights = take_float(candidate_count)
    reference_norms = take_float(candidate_count)
    cross_terms = take_float(candidate_count)
    significant_mask_u8 = np.frombuffer(
        payload,
        dtype=MASK_DTYPE,
        count=candidate_count,
        offset=offset,
    ).copy()
    offset += candidate_count * MASK_DTYPE.itemsize
    translations = take_float(translation_value_count)
    footer_magic, footer_candidates, footer_significant = FOOTER_STRUCT.unpack_from(
        payload, offset
    )
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(footer_candidates == candidate_count, f"footer count mismatch: {path}")
    _require(
        footer_significant == significant_count,
        f"footer significant count mismatch: {path}",
    )

    assert match is not None
    _require(int(match["part"]) == header[6], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"stack identity mismatch: {path}")
    _require(header[5] > 0, f"iteration must be positive: {path}")
    _require(header[15] < candidate_count, f"winner index out of range: {path}")
    _require(header[20] > 0 and header[21] > 0, f"invalid capture cap: {path}")
    _require(header[22] > 0, f"invalid follower count: {path}")
    _require(header[20] <= header[21] * header[22], f"capture cap too small: {path}")
    _require(header[24] <= header[23], f"capture byte cap exceeded: {path}")
    _require(header[25] != 0 and header[26] != 0, f"identity hash missing: {path}")
    _require(header[27] > 0, f"current size must be positive: {path}")
    for name, values in (
        ("raw diff2", raw_diff2),
        ("production weights", weights),
        ("reference norms", reference_norms),
        ("cross terms", cross_terms),
        ("translations", translations),
    ):
        _require(np.all(np.isfinite(values)), f"non-finite {name}: {path}")
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
    _require(np.any(significant_mask), f"empty significance mask: {path}")
    inferred_threshold = np.min(weights[significant_mask])
    _require(
        np.array_equal(significant_mask, weights >= inferred_threshold),
        f"production weight rank/mask mismatch: {path}",
    )
    significant_weight = _float32_from_bits(header[16])
    recorded_mask = weights >= significant_weight
    _require(
        np.array_equal(significant_mask, recorded_mask) or significant_weight == 0.0,
        f"recorded threshold/mask mismatch: {path}",
    )
    max_weight = _float32_from_bits(header[18])
    _require(
        weights[header[15]] == max_weight and np.max(weights) == max_weight,
        f"winner/maximum weight mismatch: {path}",
    )
    shape = (n_dir * n_psi, n_trans)
    return CoarsePass1Components(
        path=path,
        sha256=_sha256(path),
        header=header,
        raw_diff2=raw_diff2.reshape(shape),
        weights=weights.reshape(shape),
        reference_norms=reference_norms.reshape(shape),
        cross_terms=cross_terms.reshape(shape),
        significant_mask=significant_mask.reshape(shape),
        translations=translations.reshape(n_trans, 2),
    )


def _component_metrics(artifact: CoarsePass1Components) -> dict[str, float]:
    difference = (
        artifact.raw_diff2 - artifact.reference_norms - artifact.cross_terms
    ).astype(np.float64)
    constant = float(np.median(difference))
    centered = difference - constant
    translation_spread = np.ptp(
        artifact.reference_norms.astype(np.float64), axis=1
    )
    return {
        "image_constant_median": constant,
        "centered_replay_p95_abs": float(np.percentile(np.abs(centered), 95)),
        "centered_replay_max_abs": float(np.max(np.abs(centered))),
        "reference_norm_translation_spread_max": float(
            np.max(translation_spread)
        ),
    }


def validate_directory(
    directory: Path,
    *,
    expected_particles: int | None = None,
    expected_stack_indices: np.ndarray | None = None,
    expected_mpi_rank: int | None = None,
    replay_p95_max_abs: float = DEFAULT_REPLAY_P95_MAX_ABS,
    replay_max_abs: float = DEFAULT_REPLAY_MAX_ABS,
    reference_translation_spread: float = DEFAULT_REFERENCE_TRANSLATION_SPREAD,
) -> tuple[tuple[CoarsePass1Components, ...], dict[str, object]]:
    """Validate completeness and the predeclared component-replay gates."""

    directory = Path(directory)
    _require(directory.is_dir(), f"capture directory does not exist: {directory}")
    _require(not list(directory.glob("*.tmp.*")), "incomplete capture artifact remains")
    paths = sorted(directory.glob("*.p1-v2.bin"))
    _require(bool(paths), f"no schema-v2 artifacts in {directory}")
    artifacts = tuple(load_artifact(path) for path in paths)
    reference = artifacts[0].header
    common_fields = (
        0, 1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 22, 23, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36,
    )
    for artifact in artifacts[1:]:
        for field in common_fields:
            _require(
                artifact.header[field] == reference[field],
                f"inconsistent header field {field}: {artifact.path}",
            )
    if expected_particles is not None:
        _require(reference[20] == expected_particles, "expected-particle mismatch")
    _require(len(artifacts) == reference[20], "capture completeness mismatch")
    part_ids = np.asarray([item.part_id for item in artifacts], dtype=np.uint64)
    stack_ids = np.asarray([item.stack_index for item in artifacts], dtype=np.uint64)
    ranks, rank_counts = np.unique(
        np.asarray([item.mpi_rank for item in artifacts], dtype=np.uint64),
        return_counts=True,
    )
    _require(np.unique(part_ids).size == part_ids.size, "duplicate part identity")
    _require(np.unique(stack_ids).size == stack_ids.size, "duplicate stack identity")
    _require(ranks.size == reference[22], "follower-rank coverage mismatch")
    _require(np.all(rank_counts <= reference[21]), "per-rank capture cap exceeded")
    if expected_mpi_rank is not None:
        _require(
            np.array_equal(ranks, np.asarray([expected_mpi_rank], dtype=np.uint64)),
            "unexpected MPI follower rank",
        )
    if expected_stack_indices is not None:
        expected = np.asarray(expected_stack_indices, dtype=np.uint64)
        _require(np.unique(expected).size == expected.size, "duplicate expected stack")
        _require(
            np.array_equal(np.sort(stack_ids), np.sort(expected)),
            "RELION/expected stack identity sets differ",
        )
    metrics = {item.path.name: _component_metrics(item) for item in artifacts}
    p95_pass = sum(
        row["centered_replay_p95_abs"] <= replay_p95_max_abs
        for row in metrics.values()
    )
    max_pass = sum(
        row["centered_replay_max_abs"] <= replay_max_abs
        for row in metrics.values()
    )
    spread_pass = sum(
        row["reference_norm_translation_spread_max"]
        <= reference_translation_spread
        for row in metrics.values()
    )
    qualified = p95_pass == max_pass == spread_pass == len(artifacts)
    summary: dict[str, object] = {
        "schema": "relion-coarse-pass1-components-v2",
        "capture_directory": str(directory.resolve()),
        "particle_count": len(artifacts),
        "total_candidate_count": int(sum(item.raw_diff2.size for item in artifacts)),
        "topology": {
            "n_directions": reference[10],
            "n_psi": reference[11],
            "n_translations": reference[12],
            "candidate_count_per_particle": reference[13],
        },
        "fixed_gates": {
            "centered_replay_p95_abs_max": replay_p95_max_abs,
            "centered_replay_max_abs_max": replay_max_abs,
            "reference_norm_translation_spread_max": reference_translation_spread,
        },
        "fixed_metric": {
            "evaluated_particles": len(artifacts),
            "expected_particles": reference[20],
            "replay_p95_passed": p95_pass,
            "replay_max_passed": max_pass,
            "reference_translation_invariance_passed": spread_pass,
        },
        "component_metrics": metrics,
        "artifact_sha256": {item.path.name: item.sha256 for item in artifacts},
        "classification_ready": qualified,
        "status": "pass" if qualified else "rejected",
    }
    return artifacts, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--expected-particles", type=int)
    parser.add_argument("--expected-stack-indices-npy", type=Path)
    parser.add_argument("--expected-mpi-rank", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    expected_stacks = (
        None
        if args.expected_stack_indices_npy is None
        else np.load(args.expected_stack_indices_npy, allow_pickle=False)
    )
    _, summary = validate_directory(
        args.capture_directory,
        expected_particles=args.expected_particles,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=args.expected_mpi_rank,
    )
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(encoded)
    print(encoded, end="")
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
