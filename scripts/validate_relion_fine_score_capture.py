#!/usr/bin/env python3
"""Fail-closed validation for selected-stack RELION fine-score captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_MAGIC = b"RLNFNSC1HEADER".ljust(16, b"\0")
FOOTER_MAGIC = b"RLNFNSC1FOOTER".ljust(16, b"\0")
HEADER_STRUCT = struct.Struct("<16s48Q")
FOOTER_STRUCT = struct.Struct("<16s2Q")
CANDIDATE_DTYPE = np.dtype(
    {
        "names": (
            "sparse_index",
            "rotation_id",
            "rotation_local",
            "translation_id",
            "coarse_translation",
            "flags",
            "raw_diff2",
            "orientation_log_prior",
            "translation_log_prior",
            "combined_preexponent",
            "shifted_log_weight",
            "post_exponent_weight",
        ),
        "formats": ("<u8",) * 4 + ("<u4",) * 2 + ("<f4",) * 6,
        "offsets": (0, 8, 16, 24, 32, 36, 40, 44, 48, 52, 56, 60),
        "itemsize": 64,
    }
)
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)_class(?P<class_>\d+)\.fine-score-v1\.bin"
)

ORIENTATION_ZERO = np.uint32(1)
TRANSLATION_ZERO = np.uint32(2)
DIFF2_BELOW_MIN = np.uint32(4)
ACTIVE = np.uint32(8)
KNOWN_FLAGS = ORIENTATION_ZERO | TRANSLATION_ZERO | DIFF2_BELOW_MIN | ACTIVE


@dataclass(frozen=True)
class FineScoreCapture:
    path: Path
    sha256: str
    header: tuple[int, ...]
    candidates: np.ndarray
    algebra_max_abs: float
    shift_max_abs: float
    exponent_max_rel: float

    @property
    def stack_index(self) -> int:
        return self.header[7]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
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


def _finite_float32_tolerance(reference: np.ndarray, *, ulps: int = 4) -> np.ndarray:
    spacing = np.abs(np.spacing(reference.astype(np.float32)))
    return np.maximum(spacing * np.float32(ulps), np.float32(1e-7))


def load_fine_score_capture(path: Path) -> FineScoreCapture:
    """Load one sidecar and reject layout, identity, or score-algebra ambiguity."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected fine-score file name: {path.name}")
    payload = path.read_bytes()
    _require(len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size, f"truncated fine-score capture: {path}")
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload, 0)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"fine-score header magic mismatch: {path}")
    _require(
        header[:4] == (1, HEADER_STRUCT.size, CANDIDATE_DTYPE.itemsize, FOOTER_STRUCT.size),
        f"fine-score schema/record sizes changed: {path}",
    )
    candidate_count = header[16]
    expected_bytes = HEADER_STRUCT.size + candidate_count * CANDIDATE_DTYPE.itemsize + FOOTER_STRUCT.size
    _require(len(payload) == expected_bytes, f"fine-score byte count mismatch: {path}")
    candidates = np.frombuffer(
        payload,
        dtype=CANDIDATE_DTYPE,
        count=candidate_count,
        offset=HEADER_STRUCT.size,
    ).copy()
    footer_magic, footer_count, footer_active = FOOTER_STRUCT.unpack_from(
        payload, HEADER_STRUCT.size + candidate_count * CANDIDATE_DTYPE.itemsize
    )
    _require(footer_magic == FOOTER_MAGIC, f"fine-score footer magic mismatch: {path}")
    _require(
        (int(footer_count), int(footer_active)) == (header[16], header[17]),
        f"fine-score footer counts changed: {path}",
    )
    assert match is not None
    _require(int(match["part"]) == header[6], f"fine-score part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"fine-score stack identity mismatch: {path}")
    _require(int(match["class_"]) == header[5], f"fine-score class identity mismatch: {path}")
    algebra_max_abs, shift_max_abs, exponent_max_rel = _validate_arrays(path, header, candidates)
    return FineScoreCapture(
        path=path,
        sha256=_sha256(path),
        header=header,
        candidates=candidates,
        algebra_max_abs=algebra_max_abs,
        shift_max_abs=shift_max_abs,
        exponent_max_rel=exponent_max_rel,
    )


def _validate_arrays(path: Path, header: tuple[int, ...], candidates: np.ndarray) -> tuple[float, float, float]:
    _require(header[4] > 0 and header[5] > 0, f"invalid fine-score iteration/class: {path}")
    _require(
        header[10] > 0
        and header[11] > 0
        and header[12] > 0
        and header[13] > 0
        and header[14] > 0
        and header[15] > 0,
        f"invalid fine-score runtime dimensions: {path}",
    )
    _require(header[16] == candidates.size and candidates.size > 0, f"empty fine-score panel: {path}")
    _require(header[17] <= header[16], f"invalid active fine-score count: {path}")
    _require(header[21] > 0 and header[21] <= header[22] * header[23], f"invalid fine-score capture caps: {path}")
    _require(
        header[25] == HEADER_STRUCT.size + candidates.size * CANDIDATE_DTYPE.itemsize + FOOTER_STRUCT.size,
        f"fine-score particle byte estimate changed: {path}",
    )
    _require(header[25] * header[21] <= header[24], f"fine-score capture byte cap exceeded: {path}")
    _require(header[26] and header[27] and header[28], f"fine-score identity hash is zero: {path}")
    _require(header[29:32] == (1, 1, 1), f"fine-score capture is not passive/canonical/complete: {path}")

    expected_sparse_index = np.arange(candidates.size, dtype=np.uint64)
    _require(
        np.array_equal(candidates["sparse_index"], expected_sparse_index),
        f"fine-score sparse order changed: {path}",
    )
    _require(
        np.all((candidates["flags"] & ~KNOWN_FLAGS) == 0),
        f"unknown fine-score candidate flags: {path}",
    )
    coarse_translation = (
        candidates["translation_id"]
        - candidates["translation_id"] % np.uint64(header[11])
    ) // np.uint64(header[11])
    _require(
        np.array_equal(candidates["coarse_translation"], coarse_translation.astype(np.uint32)),
        f"fine-score coarse translation identity changed: {path}",
    )
    _require(
        np.all(candidates["rotation_id"] < header[12] * header[13])
        and np.all(candidates["coarse_translation"] < header[14]),
        f"fine-score prior index is out of range: {path}",
    )

    min_diff2 = _float32_from_bits(header[18])
    weights_max = _float32_from_bits(header[19])
    exponent_shift = _float32_from_bits(header[20])
    _require(
        np.isfinite(min_diff2) and np.isfinite(weights_max) and np.isfinite(exponent_shift),
        f"non-finite fine-score scalar: {path}",
    )
    expected_exponent_shift = np.float32(np.float32(50.0) - weights_max)
    _require(
        exponent_shift.view(np.uint32) == expected_exponent_shift.view(np.uint32),
        f"fine-score global exponent shift changed: {path}",
    )
    expected_rejection = np.zeros(candidates.size, dtype=np.uint32)
    expected_rejection |= np.where(
        candidates["raw_diff2"] < min_diff2, DIFF2_BELOW_MIN, np.uint32(0)
    )
    expected_rejection |= candidates["flags"] & (ORIENTATION_ZERO | TRANSLATION_ZERO)
    expected_active = expected_rejection == 0
    _require(
        np.array_equal((candidates["flags"] & ACTIVE) != 0, expected_active),
        f"fine-score active predicate changed: {path}",
    )
    _require(np.count_nonzero(expected_active) == header[17], f"fine-score active count changed: {path}")
    _require(
        np.array_equal(candidates["flags"] & ~ACTIVE, expected_rejection),
        f"fine-score rejection flags changed: {path}",
    )

    active = candidates[expected_active]
    _require(active.size > 0, f"fine-score panel has no active candidates: {path}")
    active_fields = np.stack(
        tuple(
            active[name]
            for name in (
                "raw_diff2",
                "orientation_log_prior",
                "translation_log_prior",
                "combined_preexponent",
                "shifted_log_weight",
                "post_exponent_weight",
            )
        )
    )
    _require(np.all(np.isfinite(active_fields)), f"non-finite active fine-score value: {path}")
    _require(np.all(active["post_exponent_weight"] > 0), f"non-positive active exponent weight: {path}")
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
        np.all(algebra_error <= _finite_float32_tolerance(expected_combined)),
        f"fine-score prior/diff2 algebra changed: {path}",
    )
    expected_shifted = np.add(active["combined_preexponent"], exponent_shift, dtype=np.float32)
    shift_error = np.abs(active["shifted_log_weight"] - expected_shifted)
    _require(
        np.all(shift_error <= _finite_float32_tolerance(expected_shifted)),
        f"fine-score shifted log weight changed: {path}",
    )
    _require(
        float(np.max(active["combined_preexponent"])) <= float(weights_max)
        + float(_finite_float32_tolerance(np.asarray([weights_max]))[0]),
        f"captured class exceeds global fine-score maximum: {path}",
    )
    expected_post = np.exp(expected_shifted.astype(np.float32), dtype=np.float32)
    exponent_relative = np.abs(active["post_exponent_weight"] - expected_post) / np.maximum(
        np.abs(expected_post), np.finfo(np.float32).tiny
    )
    _require(
        np.all(exponent_relative <= np.float32(3e-6)),
        f"fine-score exponentiation round trip changed: {path}",
    )
    inactive = candidates[~expected_active]
    if inactive.size:
        _require(
            np.all(inactive["post_exponent_weight"] == 0),
            f"rejected fine-score candidate has nonzero exponent weight: {path}",
        )
    return (
        float(np.max(algebra_error, initial=np.float32(0))),
        float(np.max(shift_error, initial=np.float32(0))),
        float(np.max(exponent_relative, initial=np.float32(0))),
    )


def validate_directory(
    directory: Path,
    selection_json: Path,
    *,
    expected_rank: int | None = None,
) -> dict[str, object]:
    """Validate a complete selected-stack panel and return hash-bound summary metrics."""

    directory = Path(directory)
    selection_json = Path(selection_json)
    selection = json.loads(selection_json.read_text())
    _require(selection.get("schema") == "bpref-factor-stratification-v1", "unexpected selection schema")
    selected = selection.get("selected")
    _require(isinstance(selected, list) and selected, "selection is empty")
    expected_stacks = [int(record["stack_index_1based"]) for record in selected]
    _require(len(expected_stacks) == len(set(expected_stacks)), "selection contains duplicate stacks")
    canonical_stack_text = ",".join(str(value) for value in expected_stacks)
    paths = sorted(directory.glob("*.fine-score-v1.bin"))
    _require(not list(directory.glob("*.tmp.*")), "fine-score capture contains incomplete temporary files")
    _require(len(paths) == len(expected_stacks), "fine-score capture file count differs from selection")
    captures = tuple(load_fine_score_capture(path) for path in paths)
    stacks = [capture.stack_index for capture in captures]
    _require(
        set(stacks) == set(expected_stacks) and len(set(stacks)) == len(stacks),
        "fine-score capture stack set is incomplete or duplicated",
    )
    expected_rank_by_stack = (
        {stack: expected_rank for stack in expected_stacks}
        if expected_rank is not None
        else {
            int(record["stack_index_1based"]): int(record["expected_mpi_rank"])
            for record in selected
        }
    )
    _require(
        len(expected_rank_by_stack) == len(expected_stacks),
        "selection is missing expected_mpi_rank for mixed-rank validation",
    )
    _require(
        all(capture.header[8] == expected_rank_by_stack[capture.stack_index] for capture in captures),
        "fine-score capture MPI rank changed",
    )
    selected_hash = fnv1a64(canonical_stack_text)
    _require(
        all(capture.header[28] == selected_hash for capture in captures),
        "fine-score selected-stack hash changed",
    )
    iterations = {capture.header[4] for capture in captures}
    classes = {capture.header[5] for capture in captures}
    _require(len(iterations) == 1 and len(classes) == 1, "fine-score iteration/class changed across panel")
    rank_by_stack = {str(capture.stack_index): int(capture.header[8]) for capture in captures}
    rank_counts = {
        str(rank): sum(value == rank for value in rank_by_stack.values())
        for rank in sorted(set(rank_by_stack.values()))
    }
    return {
        "schema": "relion-fine-score-validation-v1",
        "capture_ready": True,
        "directory": str(directory.resolve()),
        "selection_json": str(selection_json.resolve()),
        "selection_sha256": _sha256(selection_json),
        "iteration": next(iter(iterations)),
        "class_one_based": next(iter(classes)),
        "particle_count": len(captures),
        "candidate_count": int(sum(candidates.candidates.size for candidates in captures)),
        "active_candidate_count": int(
            sum(np.count_nonzero(capture.candidates["flags"] & ACTIVE) for capture in captures)
        ),
        "mpi_rank": expected_rank,
        "mpi_rank_counts": rank_counts,
        "mpi_rank_by_stack": rank_by_stack,
        "algebra_max_abs": max(capture.algebra_max_abs for capture in captures),
        "shift_max_abs": max(capture.shift_max_abs for capture in captures),
        "exponent_max_rel": max(capture.exponent_max_rel for capture in captures),
        "files": [
            {
                "path": str(capture.path.resolve()),
                "sha256": capture.sha256,
                "stack_index_1based": capture.stack_index,
                "part_id": capture.header[6],
                "mpi_rank": capture.header[8],
                "candidate_count": int(capture.candidates.size),
                "active_candidate_count": int(np.count_nonzero(capture.candidates["flags"] & ACTIVE)),
            }
            for capture in captures
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--expected-rank", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_directory(
        args.capture_dir,
        args.selection_json,
        expected_rank=args.expected_rank,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
