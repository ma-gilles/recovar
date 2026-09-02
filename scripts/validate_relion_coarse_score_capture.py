#!/usr/bin/env python3
"""Fail-closed validation for passive RELION coarse-score captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_MAGIC = b"RLNCRSC1HEADER".ljust(16, b"\0")
FOOTER_MAGIC = b"RLNCRSC1FOOTER".ljust(16, b"\0")
HEADER_STRUCT = struct.Struct("<16s64Q")
FOOTER_STRUCT = struct.Struct("<16s3Q")
CANDIDATE_DTYPE = np.dtype(
    {
        "names": (
            "flat_index",
            "class_one_based",
            "direction",
            "psi",
            "translation",
            "flags",
            "reserved",
            "raw_diff2",
            "orientation_log_prior",
            "translation_log_prior",
            "combined_preexponent",
            "shifted_log_weight",
            "post_exponent_weight",
        ),
        "formats": ("<u8",) + ("<u4",) * 6 + ("<f4",) * 6,
        "offsets": (0, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52),
        "itemsize": 56,
    }
)
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)\.coarse-score-v1\.bin"
)

ORIENTATION_ZERO = np.uint32(1)
TRANSLATION_ZERO = np.uint32(2)
DIFF2_BELOW_MIN = np.uint32(4)
ACTIVE = np.uint32(8)
SIGNIFICANT = np.uint32(16)
REJECTION_FLAGS = ORIENTATION_ZERO | TRANSLATION_ZERO | DIFF2_BELOW_MIN
KNOWN_FLAGS = REJECTION_FLAGS | ACTIVE | SIGNIFICANT


@dataclass(frozen=True)
class CoarseScoreCapture:
    path: Path
    sha256: str
    header: tuple[int, ...]
    candidates: np.ndarray
    algebra_max_abs: float
    shift_max_abs: float
    exponent_max_rel: float

    @property
    def stack_index(self) -> int:
        return self.header[6]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0])


def fnv1a64(text: str) -> int:
    value = 14695981039346656037
    for byte in text.encode():
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _float32_tolerance(reference: np.ndarray, *, ulps: int = 4) -> np.ndarray:
    spacing = np.abs(np.spacing(reference.astype(np.float32)))
    return np.maximum(spacing * np.float32(ulps), np.float32(1e-7))


def load_coarse_score_capture(path: Path) -> CoarseScoreCapture:
    """Load one capture and verify its layout, score algebra, and support."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected coarse-score file name: {path.name}")
    payload = path.read_bytes()
    _require(
        len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size,
        f"truncated coarse-score capture: {path}",
    )
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"coarse-score header magic mismatch: {path}")
    _require(
        header[:4]
        == (1, HEADER_STRUCT.size, CANDIDATE_DTYPE.itemsize, FOOTER_STRUCT.size),
        f"coarse-score schema/record sizes changed: {path}",
    )
    candidate_count = header[14]
    expected_bytes = (
        HEADER_STRUCT.size
        + candidate_count * CANDIDATE_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(len(payload) == expected_bytes, f"coarse-score byte count mismatch: {path}")
    _require(header[32] == expected_bytes, f"coarse-score byte estimate changed: {path}")
    candidates = np.frombuffer(
        payload,
        dtype=CANDIDATE_DTYPE,
        count=candidate_count,
        offset=HEADER_STRUCT.size,
    ).copy()
    footer = FOOTER_STRUCT.unpack_from(
        payload, HEADER_STRUCT.size + candidate_count * CANDIDATE_DTYPE.itemsize
    )
    _require(footer[0] == FOOTER_MAGIC, f"coarse-score footer magic mismatch: {path}")
    _require(
        tuple(int(value) for value in footer[1:])
        == (
            candidate_count,
            int(np.count_nonzero(candidates["flags"] & ACTIVE)),
            header[21],
        ),
        f"coarse-score footer counts changed: {path}",
    )
    assert match is not None
    _require(int(match["part"]) == header[5], f"coarse-score part identity mismatch: {path}")
    _require(int(match["stack"]) == header[6], f"coarse-score stack identity mismatch: {path}")
    metrics = _validate_arrays(path, header, candidates)
    return CoarseScoreCapture(
        path=path,
        sha256=_sha256(path),
        header=header,
        candidates=candidates,
        algebra_max_abs=metrics[0],
        shift_max_abs=metrics[1],
        exponent_max_rel=metrics[2],
    )


def _validate_arrays(
    path: Path,
    header: tuple[int, ...],
    candidates: np.ndarray,
) -> tuple[float, float, float]:
    class_min, class_count, nr_dir, nr_psi, nr_trans = header[9:14]
    _require(
        header[4] > 0 and class_count > 0 and nr_dir > 0 and nr_psi > 0 and nr_trans > 0,
        f"invalid coarse-score runtime dimensions: {path}",
    )
    _require(
        candidates.size == class_count * nr_dir * nr_psi * nr_trans,
        f"coarse-score topology changed: {path}",
    )
    _require(header[32] * header[31] <= header[33], f"coarse-score byte cap exceeded: {path}")
    _require(header[34:36] == (4, 1), f"coarse-score scalar layout changed: {path}")
    _require(header[38:41] == (1, 1, 1), f"coarse-score capture is not passive/complete: {path}")
    _require(header[28] and header[29] and header[30] and header[37], f"coarse-score identity hash is zero: {path}")
    _require(np.all(candidates["reserved"] == 0), f"coarse-score reserved field changed: {path}")
    _require(
        np.array_equal(candidates["flat_index"], np.arange(candidates.size, dtype=np.uint64)),
        f"coarse-score flattened order changed: {path}",
    )
    flat = np.arange(candidates.size, dtype=np.uint64)
    translation = flat % nr_trans
    orientation = flat // nr_trans
    psi = orientation % nr_psi
    class_direction = orientation // nr_psi
    direction = class_direction % nr_dir
    class_one_based = class_min + class_direction // nr_dir + 1
    _require(
        np.array_equal(candidates["translation"], translation.astype(np.uint32))
        and np.array_equal(candidates["psi"], psi.astype(np.uint32))
        and np.array_equal(candidates["direction"], direction.astype(np.uint32))
        and np.array_equal(candidates["class_one_based"], class_one_based.astype(np.uint32)),
        f"coarse-score class/direction/psi/translation identity changed: {path}",
    )
    _require(
        np.all((candidates["flags"] & ~KNOWN_FLAGS) == 0),
        f"unknown coarse-score candidate flag: {path}",
    )

    min_diff2 = _float32_from_bits(header[15])
    weights_max = _float32_from_bits(header[16])
    exponent_shift = _float32_from_bits(header[17])
    significant_weight = _float32_from_bits(header[18])
    sum_weight = _float32_from_bits(header[19])
    maximum_weight = _float32_from_bits(header[25])
    scalars = np.asarray(
        [min_diff2, weights_max, exponent_shift, significant_weight, sum_weight, maximum_weight]
    )
    _require(np.all(np.isfinite(scalars)), f"non-finite coarse-score scalar: {path}")
    expected_shift = np.float32(50.0) - weights_max
    _require(
        exponent_shift.view(np.uint32) == expected_shift.view(np.uint32),
        f"coarse-score exponent shift changed: {path}",
    )

    expected_rejection = candidates["flags"] & (ORIENTATION_ZERO | TRANSLATION_ZERO)
    expected_rejection |= np.where(
        candidates["raw_diff2"] < min_diff2,
        DIFF2_BELOW_MIN,
        np.uint32(0),
    )
    expected_active = expected_rejection == 0
    _require(
        np.array_equal((candidates["flags"] & ACTIVE) != 0, expected_active),
        f"coarse-score active predicate changed: {path}",
    )
    _require(
        np.array_equal(candidates["flags"] & REJECTION_FLAGS, expected_rejection),
        f"coarse-score rejection flags changed: {path}",
    )
    significant = (candidates["flags"] & SIGNIFICANT) != 0
    _require(np.all(~significant | expected_active), f"rejected coarse score is significant: {path}")
    _require(np.count_nonzero(significant) == header[21], f"coarse-score support count changed: {path}")
    _require(
        np.array_equal(significant, candidates["post_exponent_weight"] >= significant_weight),
        f"coarse-score threshold support changed: {path}",
    )
    _require(header[20] <= header[21], f"reported coarse-score count exceeds threshold support: {path}")

    active = candidates[expected_active]
    _require(active.size > 0, f"coarse-score panel has no active candidates: {path}")
    _require(
        np.all(
            np.isfinite(
                np.stack(
                    [
                        active["raw_diff2"],
                        active["orientation_log_prior"],
                        active["translation_log_prior"],
                        active["combined_preexponent"],
                        active["shifted_log_weight"],
                        active["post_exponent_weight"],
                    ]
                )
            )
        ),
        f"non-finite active coarse-score value: {path}",
    )
    expected_combined = np.subtract(
        np.add(
            np.add(active["orientation_log_prior"], active["translation_log_prior"], dtype=np.float32),
            min_diff2,
            dtype=np.float32,
        ),
        active["raw_diff2"],
        dtype=np.float32,
    )
    algebra_error = np.abs(active["combined_preexponent"] - expected_combined)
    _require(
        np.all(algebra_error <= _float32_tolerance(expected_combined)),
        f"coarse-score prior/diff2 algebra changed: {path}",
    )
    expected_shifted = np.add(active["combined_preexponent"], exponent_shift, dtype=np.float32)
    shift_error = np.abs(active["shifted_log_weight"] - expected_shifted)
    _require(
        np.all(shift_error <= _float32_tolerance(expected_shifted)),
        f"coarse-score shifted log weight changed: {path}",
    )
    underflow = expected_shifted < np.float32(-88.0)
    _require(
        np.all(active["post_exponent_weight"][underflow] == 0),
        f"coarse-score underflow predicate changed: {path}",
    )
    expected_post = np.exp(expected_shifted[~underflow], dtype=np.float32)
    exponent_relative = np.abs(
        active["post_exponent_weight"][~underflow] - expected_post
    ) / np.maximum(np.abs(expected_post), np.finfo(np.float32).tiny)
    _require(
        np.all(exponent_relative <= np.float32(3e-6)),
        f"coarse-score exponentiation changed: {path}",
    )
    inactive = candidates[~expected_active]
    _require(
        not inactive.size or np.all(inactive["post_exponent_weight"] == 0),
        f"rejected coarse-score candidate has nonzero weight: {path}",
    )
    _require(
        header[22] == np.count_nonzero(candidates["post_exponent_weight"] > 0),
        f"coarse-score filtered count changed: {path}",
    )
    _require(header[23] < header[22], f"coarse-score threshold index is invalid: {path}")
    _require(header[24] < candidates.size, f"coarse-score maximum index is invalid: {path}")
    _require(
        candidates["post_exponent_weight"][header[24]].view(np.uint32)
        == maximum_weight.view(np.uint32),
        f"coarse-score maximum identity changed: {path}",
    )
    _require(
        weights_max.view(np.uint32)
        == np.max(active["combined_preexponent"]).view(np.uint32),
        f"coarse-score pre-exponent maximum changed: {path}",
    )
    return (
        float(np.max(algebra_error, initial=np.float32(0))),
        float(np.max(shift_error, initial=np.float32(0))),
        float(np.max(exponent_relative, initial=np.float32(0))),
    )


def _parse_stacks(text: str) -> tuple[int, ...]:
    values = tuple(int(token) for token in text.split(",") if token)
    _require(values and len(values) == len(set(values)) and min(values) > 0, "invalid expected stacks")
    return values


def validate_directory(
    directory: Path,
    expected_stacks: tuple[int, ...],
    *,
    expected_iteration: int | None = None,
) -> dict[str, object]:
    directory = Path(directory)
    paths = sorted(directory.glob("*.coarse-score-v1.bin"))
    _require(not list(directory.glob("*.tmp.*")), "coarse-score directory contains temporary files")
    _require(len(paths) == len(expected_stacks), "coarse-score file count differs from target set")
    captures = tuple(load_coarse_score_capture(path) for path in paths)
    _require(
        {capture.stack_index for capture in captures} == set(expected_stacks),
        "coarse-score stack set is incomplete or duplicated",
    )
    selected_hash = fnv1a64(",".join(str(stack) for stack in expected_stacks))
    _require(
        all(capture.header[30] == selected_hash for capture in captures),
        "coarse-score selected-stack hash changed",
    )
    iterations = {capture.header[4] for capture in captures}
    _require(len(iterations) == 1, "coarse-score iteration changed across panel")
    iteration = next(iter(iterations))
    if expected_iteration is not None:
        _require(iteration == expected_iteration, "coarse-score iteration differs from expectation")
    return {
        "schema": "relion-coarse-score-validation-v1",
        "capture_ready": True,
        "directory": str(directory.resolve()),
        "iteration": iteration,
        "particle_count": len(captures),
        "candidate_count": int(sum(capture.candidates.size for capture in captures)),
        "active_candidate_count": int(
            sum(np.count_nonzero(capture.candidates["flags"] & ACTIVE) for capture in captures)
        ),
        "significant_candidate_count": int(
            sum(np.count_nonzero(capture.candidates["flags"] & SIGNIFICANT) for capture in captures)
        ),
        "algebra_max_abs": max(capture.algebra_max_abs for capture in captures),
        "shift_max_abs": max(capture.shift_max_abs for capture in captures),
        "exponent_max_rel": max(capture.exponent_max_rel for capture in captures),
        "files": [
            {
                "path": str(capture.path.resolve()),
                "sha256": capture.sha256,
                "part_id": capture.header[5],
                "stack_index_one_based": capture.stack_index,
                "mpi_rank": capture.header[7],
                "candidate_count": int(capture.candidates.size),
                "significant_candidate_count": int(capture.header[21]),
            }
            for capture in captures
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--expected-stacks", required=True)
    parser.add_argument("--expected-iteration", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_directory(
        args.capture_dir,
        _parse_stacks(args.expected_stacks),
        expected_iteration=args.expected_iteration,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
