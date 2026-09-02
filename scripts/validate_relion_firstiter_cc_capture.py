#!/usr/bin/env python3
"""Fail-closed reader for bounded RELION first-iteration CC score captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_MAGIC = b"RLNFICC1HEADER".ljust(16, b"\0")
FOOTER_MAGIC = b"RLNFICC1FOOTER".ljust(16, b"\0")
HEADER_STRUCT = struct.Struct("<16s64Q")
FOOTER_STRUCT = struct.Struct("<16s3Q")
CANDIDATE_DTYPE = np.dtype(
    {
        "names": (
            "candidate_index",
            "ihidden_overs",
            "rotation_id",
            "rotation_local",
            "translation_id",
            "class_one_based",
            "flags",
            "raw_score",
            "reserved",
        ),
        "formats": ("<u8",) * 5 + ("<u4",) * 2 + ("<f4", "<u4"),
        "offsets": (0, 8, 16, 24, 32, 40, 44, 48, 52),
        "itemsize": 56,
    }
)
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)_pass(?P<pass_>[01])\.firstiter-cc-v1\.bin"
)

WINNER = np.uint32(1)
DENSE_COARSE = np.uint32(2)
SPARSE_FINE = np.uint32(4)
KNOWN_FLAGS = WINNER | DENSE_COARSE | SPARSE_FINE


@dataclass(frozen=True)
class FirstiterCcCapture:
    path: Path
    sha256: str
    header: tuple[int, ...]
    candidates: np.ndarray

    @property
    def pass_index(self) -> int:
        return self.header[5]

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
    def winner_index(self) -> int:
        return self.header[19]

    @property
    def winner(self) -> np.void:
        return self.candidates[self.winner_index]


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


def load_firstiter_cc_capture(path: Path) -> FirstiterCcCapture:
    """Load one capture and reject incomplete identity or score metadata."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected firstiter-CC file name: {path.name}")
    payload = path.read_bytes()
    _require(
        len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size,
        f"truncated firstiter-CC capture: {path}",
    )
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload, 0)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"firstiter-CC header magic mismatch: {path}")
    _require(
        header[:4] == (1, HEADER_STRUCT.size, CANDIDATE_DTYPE.itemsize, FOOTER_STRUCT.size),
        f"firstiter-CC schema/record sizes changed: {path}",
    )
    candidate_count = header[18]
    expected_bytes = HEADER_STRUCT.size + candidate_count * CANDIDATE_DTYPE.itemsize + FOOTER_STRUCT.size
    _require(len(payload) == expected_bytes, f"firstiter-CC byte count mismatch: {path}")
    candidates = np.frombuffer(
        payload,
        dtype=CANDIDATE_DTYPE,
        count=candidate_count,
        offset=HEADER_STRUCT.size,
    ).copy()
    footer_magic, footer_count, footer_winner, footer_score_bits = FOOTER_STRUCT.unpack_from(
        payload, HEADER_STRUCT.size + candidate_count * CANDIDATE_DTYPE.itemsize
    )
    _require(footer_magic == FOOTER_MAGIC, f"firstiter-CC footer magic mismatch: {path}")
    _require(
        (int(footer_count), int(footer_winner), int(footer_score_bits))
        == (header[18], header[19], header[20]),
        f"firstiter-CC footer changed: {path}",
    )
    assert match is not None
    _require(int(match["part"]) == header[6], f"firstiter-CC part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"firstiter-CC stack identity mismatch: {path}")
    _require(int(match["pass_"]) == header[5], f"firstiter-CC pass identity mismatch: {path}")
    _validate_arrays(path, header, candidates)
    return FirstiterCcCapture(path=path, sha256=_sha256(path), header=header, candidates=candidates)


def _validate_arrays(path: Path, header: tuple[int, ...], candidates: np.ndarray) -> None:
    _require(header[4] > 0 and header[5] in (0, 1), f"invalid firstiter-CC iteration/pass: {path}")
    _require(header[10] > 0, f"invalid firstiter-CC class count: {path}")
    _require(0 <= header[11] <= header[12] < header[10], f"invalid active class range: {path}")
    _require(
        all(header[index] > 0 for index in (13, 14, 15, 16, 17, 18)),
        f"invalid firstiter-CC runtime dimensions: {path}",
    )
    _require(header[19] < header[18], f"invalid firstiter-CC winner index: {path}")
    _require(header[21] > 0 and header[21] <= header[22] * header[23], f"invalid capture caps: {path}")
    _require(
        header[25] == HEADER_STRUCT.size + candidates.size * CANDIDATE_DTYPE.itemsize + FOOTER_STRUCT.size,
        f"firstiter-CC byte estimate changed: {path}",
    )
    _require(header[25] * header[21] * 2 <= header[24], f"firstiter-CC byte cap exceeded: {path}")
    _require(header[26] and header[27] and header[28], f"firstiter-CC identity hash is zero: {path}")
    _require(header[29:32] == (1, 1, 1), f"capture is not passive/canonical/float32: {path}")
    _require(
        (header[32], header[33]) == ((1, 0) if header[5] == 0 else (0, 1)),
        f"firstiter-CC layout flags changed: {path}",
    )
    _require(
        np.array_equal(candidates["candidate_index"], np.arange(candidates.size, dtype=np.uint64)),
        f"firstiter-CC candidate order changed: {path}",
    )
    _require(np.all(candidates["reserved"] == 0), f"firstiter-CC reserved word is nonzero: {path}")
    _require(np.all(np.isfinite(candidates["raw_score"])), f"non-finite firstiter-CC score: {path}")
    _require(np.all((candidates["flags"] & ~KNOWN_FLAGS) == 0), f"unknown candidate flags: {path}")
    winner_mask = (candidates["flags"] & WINNER) != 0
    _require(np.count_nonzero(winner_mask) == 1, f"firstiter-CC winner count changed: {path}")
    _require(bool(winner_mask[header[19]]), f"firstiter-CC winner flag/index mismatch: {path}")
    winner_score = _float32_from_bits(header[20])
    _require(
        candidates["raw_score"][header[19]].view(np.uint32) == winner_score.view(np.uint32),
        f"firstiter-CC winner score bits changed: {path}",
    )
    _require(
        header[19] == int(np.argmin(candidates["raw_score"])),
        f"firstiter-CC winner is not the stable raw-score argmin: {path}",
    )
    _require(
        np.all((candidates["class_one_based"] >= header[11] + 1) & (candidates["class_one_based"] <= header[12] + 1)),
        f"firstiter-CC class identity is out of range: {path}",
    )

    if header[5] == 0:
        _require(
            np.all((candidates["flags"] & DENSE_COARSE) != 0)
            and np.all((candidates["flags"] & SPARSE_FINE) == 0),
            f"coarse firstiter-CC flags changed: {path}",
        )
        class_count = header[12] - header[11] + 1
        class_stride = header[13] * header[14] * header[15]
        _require(candidates.size == class_count * class_stride, f"coarse candidate count changed: {path}")
        indices = np.arange(candidates.size, dtype=np.uint64)
        class_local = indices // np.uint64(class_stride)
        class_position = indices % np.uint64(class_stride)
        expected_class = class_local + np.uint64(header[11] + 1)
        expected_rotation = class_position // np.uint64(header[15])
        expected_translation = class_position % np.uint64(header[15])
        _require(np.array_equal(candidates["class_one_based"], expected_class), f"coarse class map changed: {path}")
        _require(np.array_equal(candidates["rotation_id"], expected_rotation), f"coarse rotation map changed: {path}")
        _require(np.array_equal(candidates["rotation_local"], expected_rotation), f"coarse local rotation changed: {path}")
        _require(np.array_equal(candidates["translation_id"], expected_translation), f"coarse translation map changed: {path}")
        _require(np.array_equal(candidates["ihidden_overs"], indices), f"coarse hidden map changed: {path}")
    else:
        _require(
            np.all((candidates["flags"] & SPARSE_FINE) != 0)
            and np.all((candidates["flags"] & DENSE_COARSE) == 0),
            f"fine firstiter-CC flags changed: {path}",
        )
        _require(
            np.all(candidates["translation_id"] < header[15] * header[17]),
            f"fine translation identity is out of range: {path}",
        )


def _selection_stacks(selection_json: Path) -> list[int]:
    selection = json.loads(Path(selection_json).read_text())
    _require(selection.get("schema") == "bpref-factor-stratification-v1", "unexpected selection schema")
    records = selection.get("selected")
    _require(isinstance(records, list) and bool(records), "selection is empty")
    stacks = [int(record["stack_index_1based"]) for record in records]
    _require(len(stacks) == len(set(stacks)), "selection stack identities are not unique")
    return stacks


def validate_directory(directory: Path, selection_json: Path, *, expected_rank: int | None = None) -> dict[str, object]:
    """Validate both CC passes for every selected stack and summarize winners."""

    directory = Path(directory)
    stacks = _selection_stacks(selection_json)
    paths = sorted(directory.glob("*.firstiter-cc-v1.bin"))
    _require(len(paths) == 2 * len(stacks), "firstiter-CC capture count changed")
    captures = [load_firstiter_cc_capture(path) for path in paths]
    by_key = {(capture.stack_index, capture.pass_index): capture for capture in captures}
    _require(len(by_key) == len(captures), "duplicate firstiter-CC stack/pass capture")
    _require(set(by_key) == {(stack, pass_index) for stack in stacks for pass_index in (0, 1)}, "firstiter-CC panel is incomplete")

    rows = []
    for stack in stacks:
        coarse = by_key[(stack, 0)]
        fine = by_key[(stack, 1)]
        _require(coarse.part_id == fine.part_id, f"stack {stack}: pass particle identities differ")
        _require(coarse.mpi_rank == fine.mpi_rank, f"stack {stack}: pass MPI ranks differ")
        if expected_rank is not None:
            _require(coarse.mpi_rank == expected_rank, f"stack {stack}: MPI rank changed")
        coarse_class = int(coarse.winner["class_one_based"])
        fine_class = int(fine.winner["class_one_based"])
        _require(
            np.all(fine.candidates["class_one_based"] == coarse_class),
            f"stack {stack}: fine support escaped the coarse winner class",
        )
        _require(fine_class == coarse_class, f"stack {stack}: coarse/fine winner classes differ")
        coarse_scores = np.sort(coarse.candidates["raw_score"])
        fine_scores = np.sort(fine.candidates["raw_score"])
        rows.append(
            {
                "stack_index_1based": stack,
                "part_id": coarse.part_id,
                "mpi_rank": coarse.mpi_rank,
                "coarse_candidate_count": int(coarse.candidates.size),
                "fine_candidate_count": int(fine.candidates.size),
                "winner_class_one_based": coarse_class,
                "coarse_winner_score": float(coarse_scores[0]),
                "coarse_runner_up_margin": float(np.float32(coarse_scores[1] - coarse_scores[0])),
                "fine_winner_score": float(fine_scores[0]),
                "fine_runner_up_margin": float(np.float32(fine_scores[1] - fine_scores[0])),
                "coarse_sha256": coarse.sha256,
                "fine_sha256": fine.sha256,
            }
        )
    return {
        "schema": "relion-firstiter-cc-capture-validation-v1",
        "capture_ready": True,
        "particle_count": len(stacks),
        "file_count": len(captures),
        "coarse_candidate_count": int(sum(by_key[(stack, 0)].candidates.size for stack in stacks)),
        "fine_candidate_count": int(sum(by_key[(stack, 1)].candidates.size for stack in stacks)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--expected-rank", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = validate_directory(args.capture_dir, args.selection_json, expected_rank=args.expected_rank)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
