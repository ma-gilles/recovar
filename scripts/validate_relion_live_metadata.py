#!/usr/bin/env python3
"""Validate and normalize exact RELION live-metadata capture artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np


HEADER = struct.Struct("<16s16Q")
FOOTER = struct.Struct("<16sQQ")
HEADER_MAGIC = b"RLNLMDV1HEADER\0\0"
FOOTER_MAGIC = b"RLNLMDV1FOOTER\0\0"
SCHEMA_VERSION = 1
HEADER_SIZE = 144
RECORD_HEADER_SIZE = 64
FOOTER_SIZE = 32

METADATA_NAMES = (
    "rot",
    "tilt",
    "psi",
    "xoff",
    "yoff",
    "zoff",
    "class",
    "dll",
    "pmax",
    "nr_sign",
    "norm",
    "ctf_defocus_u",
    "ctf_defocus_v",
    "ctf_defocus_angle",
    "ctf_bfactor",
    "ctf_kfactor",
    "ctf_phase_shift",
    "rot_prior",
    "tilt_prior",
    "psi_prior",
    "xoff_prior",
    "yoff_prior",
    "zoff_prior",
    "psi_prior_flip_ratio",
    "rot_prior_flip_ratio",
)


@dataclass(frozen=True)
class LiveMetadataCapture:
    path: Path
    iteration: int
    stage: int
    float_size: int
    sorted_position: np.ndarray
    particle_id: np.ndarray
    stack_index_one_based: np.ndarray
    follower_rank: np.ndarray
    metadata: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_live_metadata(path: str | Path) -> LiveMetadataCapture:
    path = Path(path)
    file_size = path.stat().st_size
    _require(file_size >= HEADER_SIZE + FOOTER_SIZE, f"{path}: truncated artifact")

    with path.open("rb") as stream:
        header_raw = stream.read(HEADER_SIZE)
        _require(len(header_raw) == HEADER_SIZE, f"{path}: truncated header")
        magic, *values = HEADER.unpack(header_raw)
        _require(magic == HEADER_MAGIC, f"{path}: bad header magic {magic!r}")
        (
            version,
            header_size,
            record_header_size,
            float_size,
            footer_size,
            iteration,
            particle_count,
            metadata_width,
            stage,
            exact_in_memory,
            *reserved,
        ) = values
        _require(version == SCHEMA_VERSION, f"{path}: unsupported schema {version}")
        _require(header_size == HEADER_SIZE, f"{path}: header-size mismatch")
        _require(record_header_size == RECORD_HEADER_SIZE, f"{path}: record-size mismatch")
        _require(footer_size == FOOTER_SIZE, f"{path}: footer-size mismatch")
        _require(float_size in (4, 8), f"{path}: unsupported RFLOAT size {float_size}")
        _require(iteration > 0, f"{path}: invalid iteration {iteration}")
        _require(particle_count > 0, f"{path}: empty particle capture")
        _require(metadata_width >= len(METADATA_NAMES), f"{path}: metadata width is too small")
        _require(stage in (1, 2), f"{path}: invalid stage {stage}")
        _require(exact_in_memory == 1, f"{path}: artifact is not exact in-memory metadata")
        _require(all(value == 0 for value in reserved), f"{path}: nonzero reserved header value")

        record_size = RECORD_HEADER_SIZE + metadata_width * float_size
        expected_size = HEADER_SIZE + particle_count * record_size + FOOTER_SIZE
        _require(file_size == expected_size, f"{path}: size {file_size} != expected {expected_size}")
        stream.seek(file_size - FOOTER_SIZE)
        footer_magic, footer_records, footer_values = FOOTER.unpack(stream.read(FOOTER_SIZE))
        _require(footer_magic == FOOTER_MAGIC, f"{path}: bad footer magic")
        _require(footer_records == particle_count, f"{path}: footer record-count mismatch")
        _require(
            footer_values == particle_count * metadata_width,
            f"{path}: footer metadata-count mismatch",
        )

    float_dtype = "<f4" if float_size == 4 else "<f8"
    record_dtype = np.dtype(
        {
            "names": (
                "sorted_position",
                "particle_id",
                "stack_index_one_based",
                "follower_rank",
                "metadata_count",
                "reserved",
                "metadata",
            ),
            "formats": ("<u8", "<i8", "<i8", "<i8", "<u8", ("<u8", 3), (float_dtype, metadata_width)),
            "offsets": (0, 8, 16, 24, 32, 40, 64),
            "itemsize": record_size,
        }
    )
    records = np.memmap(path, mode="r", dtype=record_dtype, offset=HEADER_SIZE, shape=(particle_count,))
    _require(np.all(records["metadata_count"] == metadata_width), f"{path}: per-record width mismatch")
    _require(np.all(records["reserved"] == 0), f"{path}: nonzero reserved record value")
    _require(np.all(records["particle_id"] >= 0), f"{path}: negative particle identity")
    _require(np.all(records["stack_index_one_based"] > 0), f"{path}: invalid stack index")
    _require(np.all(records["follower_rank"] > 0), f"{path}: invalid follower rank")

    order = np.argsort(records["sorted_position"], kind="stable")
    sorted_position = np.asarray(records["sorted_position"][order], dtype=np.uint64)
    _require(
        np.array_equal(sorted_position, np.arange(particle_count, dtype=np.uint64)),
        f"{path}: sorted positions are not a complete zero-based range",
    )
    particle_id = np.asarray(records["particle_id"][order], dtype=np.int64)
    _require(np.unique(particle_id).size == particle_count, f"{path}: duplicate particle identity")

    return LiveMetadataCapture(
        path=path,
        iteration=int(iteration),
        stage=int(stage),
        float_size=int(float_size),
        sorted_position=np.array(sorted_position, copy=True),
        particle_id=np.array(particle_id, copy=True),
        stack_index_one_based=np.array(records["stack_index_one_based"][order], copy=True),
        follower_rank=np.array(records["follower_rank"][order], copy=True),
        metadata=np.array(records["metadata"][order], copy=True),
    )


def validate_pair(input_path: str | Path, output_path: str | Path) -> tuple[dict[str, object], LiveMetadataCapture, LiveMetadataCapture]:
    incoming = load_live_metadata(input_path)
    outgoing = load_live_metadata(output_path)
    _require(incoming.stage == 1, "incoming artifact has the wrong stage")
    _require(outgoing.stage == 2, "outgoing artifact has the wrong stage")
    _require(incoming.iteration == outgoing.iteration, "iteration mismatch")
    _require(incoming.float_size == outgoing.float_size, "RFLOAT-size mismatch")
    for name in ("sorted_position", "particle_id", "stack_index_one_based", "follower_rank"):
        _require(np.array_equal(getattr(incoming, name), getattr(outgoing, name)), f"{name} mismatch")
    _require(incoming.metadata.shape == outgoing.metadata.shape, "metadata-shape mismatch")

    changed = incoming.metadata.view(np.uint8).reshape(incoming.metadata.shape + (-1,))
    changed_out = outgoing.metadata.view(np.uint8).reshape(outgoing.metadata.shape + (-1,))
    changed_by_column = np.any(changed != changed_out, axis=2).sum(axis=0)
    max_abs = np.max(np.abs(outgoing.metadata - incoming.metadata), axis=0)
    names = list(METADATA_NAMES) + [
        f"body_{(index - len(METADATA_NAMES)) // 6}_{('rot', 'tilt', 'psi', 'xoff', 'yoff', 'zoff')[(index - len(METADATA_NAMES)) % 6]}"
        for index in range(len(METADATA_NAMES), incoming.metadata.shape[1])
    ]
    report = {
        "schema": "relion-live-metadata-validation-v1",
        "iteration": incoming.iteration,
        "particle_count": int(incoming.metadata.shape[0]),
        "metadata_width": int(incoming.metadata.shape[1]),
        "rfloat_bytes": incoming.float_size,
        "input_path": str(incoming.path.resolve()),
        "output_path": str(outgoing.path.resolve()),
        "input_nonfinite_values": int(np.count_nonzero(~np.isfinite(incoming.metadata))),
        "output_nonfinite_values": int(np.count_nonzero(~np.isfinite(outgoing.metadata))),
        "columns": {
            name: {
                "changed_particles_bitwise": int(changed_by_column[index]),
                "max_abs_change": float(max_abs[index]),
            }
            for index, name in enumerate(names)
        },
    }
    return report, incoming, outgoing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--normalized-npz", type=Path)
    parser.add_argument("--expected-particles", type=int)
    args = parser.parse_args()

    report, incoming, outgoing = validate_pair(args.input, args.output)
    if args.expected_particles is not None:
        _require(report["particle_count"] == args.expected_particles, "unexpected particle count")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.normalized_npz is not None:
        args.normalized_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.normalized_npz,
            sorted_position=incoming.sorted_position,
            particle_id=incoming.particle_id,
            stack_index_one_based=incoming.stack_index_one_based,
            follower_rank=incoming.follower_rank,
            metadata_input=incoming.metadata,
            metadata_output=outgoing.metadata,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
