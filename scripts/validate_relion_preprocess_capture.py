#!/usr/bin/env python3
"""Fail-closed validation of bounded passive RELION preprocessing captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_MAGIC = b"RLNPREPV1HEADER\0"
FOOTER_MAGIC = b"RLNPREPV1FOOTER\0"
HEADER_STRUCT = struct.Struct("<16s40Q")
FOOTER_STRUCT = struct.Struct("<16sQQ")
FLOAT_DTYPE = np.dtype("<f4")
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)\.preprocess-v1\.bin"
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
class RelionPreprocessCapture:
    """One structurally qualified schema-v1 preprocessing artifact."""

    path: Path
    sha256: str
    header: tuple[int, ...]
    raw_input_real: np.ndarray
    normalized_shifted_real: np.ndarray
    unmasked_fourier_pre_optics: np.ndarray
    unmasked_fourier_post_optics: np.ndarray
    masked_real: np.ndarray
    masked_fourier_pre_optics: np.ndarray
    masked_fourier_post_optics: np.ndarray

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
    def iteration(self) -> int:
        return self.header[5]

    @property
    def norm_correction(self) -> float:
        return _float32_from_bits(self.header[20])

    @property
    def old_offset(self) -> np.ndarray:
        return np.asarray(
            [_float32_from_bits(self.header[index]) for index in range(21, 24)],
            dtype=np.float32,
        )

    @property
    def mask_parameters(self) -> dict[str, float]:
        return {
            "radius": _float32_from_bits(self.header[24]),
            "radius_p": _float32_from_bits(self.header[25]),
            "cosine_width": _float32_from_bits(self.header[26]),
            "background": _float32_from_bits(self.header[27]),
        }


def load_artifact(path: Path) -> RelionPreprocessCapture:
    """Load one artifact after validating its byte-level contract."""

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
    _require(header[0] == 1, f"schema mismatch: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"header size mismatch: {path}")
    _require(header[2] == FLOAT_DTYPE.itemsize, f"float size mismatch: {path}")
    _require(header[3] == 8, f"integer size mismatch: {path}")
    _require(header[4] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")
    _require(header[5] > 0, f"iteration must be positive: {path}")
    _require(header[10:12] == (0, 0), f"expected image 0/body 0: {path}")

    real_x, real_y, real_z, real_count = header[12:16]
    fourier_x, fourier_y, fourier_z, fourier_count = header[16:20]
    _require(
        real_x > 0 and real_y > 0 and real_z == 1,
        f"invalid real-space topology: {path}",
    )
    _require(
        fourier_x > 0 and fourier_y > 0 and fourier_z == 1,
        f"invalid Fourier topology: {path}",
    )
    _require(
        real_x * real_y * real_z == real_count,
        f"real-space count mismatch: {path}",
    )
    _require(
        fourier_x * fourier_y * fourier_z == fourier_count,
        f"Fourier count mismatch: {path}",
    )
    _require(
        fourier_x == fourier_y // 2 + 1,
        f"Fourier half-spectrum topology mismatch: {path}",
    )
    _require(header[28] == 1, f"capture must use zero masking: {path}")
    _require(header[29] > 0 and header[30] > 0, f"invalid particle cap: {path}")
    _require(header[31] > 0, f"invalid follower count: {path}")
    _require(
        header[29] <= header[30] * header[31],
        f"particle cap cannot cover expectation: {path}",
    )
    _require(header[33] <= header[32], f"capture byte cap exceeded: {path}")
    _require(header[34] != 0 and header[35] != 0, f"identity hash missing: {path}")
    _require(header[36:39] == (127, 1, 1), f"passive stage flags missing: {path}")
    _require(header[39] == 128, f"unexpected soft-mask block size: {path}")

    expected_size = (
        HEADER_STRUCT.size
        + 3 * real_count * FLOAT_DTYPE.itemsize
        + 8 * fourier_count * FLOAT_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")
    _require(
        header[33] == expected_size * header[30] * header[31],
        f"estimated byte count mismatch: {path}",
    )

    offset = HEADER_STRUCT.size

    def take_real() -> np.ndarray:
        nonlocal offset
        values = np.frombuffer(
            payload,
            dtype=FLOAT_DTYPE,
            count=real_count,
            offset=offset,
        ).copy()
        offset += real_count * FLOAT_DTYPE.itemsize
        return values.reshape(real_z, real_y, real_x)

    def take_fourier() -> np.ndarray:
        nonlocal offset
        real_values = np.frombuffer(
            payload,
            dtype=FLOAT_DTYPE,
            count=fourier_count,
            offset=offset,
        ).copy()
        offset += fourier_count * FLOAT_DTYPE.itemsize
        imag_values = np.frombuffer(
            payload,
            dtype=FLOAT_DTYPE,
            count=fourier_count,
            offset=offset,
        ).copy()
        offset += fourier_count * FLOAT_DTYPE.itemsize
        return (real_values + np.complex64(1j) * imag_values).reshape(
            fourier_z, fourier_y, fourier_x
        )

    raw_input_real = take_real()
    normalized_shifted_real = take_real()
    unmasked_fourier_pre_optics = take_fourier()
    unmasked_fourier_post_optics = take_fourier()
    masked_real = take_real()
    masked_fourier_pre_optics = take_fourier()
    masked_fourier_post_optics = take_fourier()
    footer_magic, footer_real_count, footer_fourier_count = FOOTER_STRUCT.unpack_from(
        payload, offset
    )
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(footer_real_count == real_count, f"footer real count mismatch: {path}")
    _require(
        footer_fourier_count == fourier_count,
        f"footer Fourier count mismatch: {path}",
    )

    assert match is not None
    _require(int(match["part"]) == header[6], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"stack identity mismatch: {path}")
    norm_correction = _float32_from_bits(header[20])
    old_offset = np.asarray(
        [_float32_from_bits(header[index]) for index in range(21, 24)],
        dtype=np.float32,
    )
    mask_values = np.asarray(
        [_float32_from_bits(header[index]) for index in range(24, 28)],
        dtype=np.float32,
    )
    _require(
        np.isfinite(norm_correction) and norm_correction > 0,
        f"invalid norm correction: {path}",
    )
    _require(np.all(np.isfinite(old_offset)), f"non-finite old offset: {path}")
    _require(
        np.array_equal(old_offset, np.rint(old_offset).astype(np.float32)),
        f"old offset is not rounded: {path}",
    )
    _require(np.all(np.isfinite(mask_values)), f"non-finite mask metadata: {path}")
    radius, radius_p, cosine_width, _ = mask_values
    _require(radius > 0 and cosine_width > 0, f"invalid mask geometry: {path}")
    _require(
        radius_p == np.float32(radius + cosine_width),
        f"mask radius-p mismatch: {path}",
    )
    for name, values in (
        ("raw input", raw_input_real),
        ("normalized shifted image", normalized_shifted_real),
        ("unmasked pre-optics Fourier image", unmasked_fourier_pre_optics),
        ("unmasked post-optics Fourier image", unmasked_fourier_post_optics),
        ("masked image", masked_real),
        ("masked pre-optics Fourier image", masked_fourier_pre_optics),
        ("masked post-optics Fourier image", masked_fourier_post_optics),
    ):
        _require(np.all(np.isfinite(values)), f"non-finite {name}: {path}")
    return RelionPreprocessCapture(
        path=path,
        sha256=_sha256(path),
        header=header,
        raw_input_real=raw_input_real,
        normalized_shifted_real=normalized_shifted_real,
        unmasked_fourier_pre_optics=unmasked_fourier_pre_optics,
        unmasked_fourier_post_optics=unmasked_fourier_post_optics,
        masked_real=masked_real,
        masked_fourier_pre_optics=masked_fourier_pre_optics,
        masked_fourier_post_optics=masked_fourier_post_optics,
    )


def validate_directory(
    directory: Path,
    *,
    expected_particles: int | None = None,
    expected_part_ids: np.ndarray | None = None,
    expected_stack_indices: np.ndarray | None = None,
    expected_mpi_rank: int | None = None,
    expected_iteration: int | None = None,
) -> tuple[tuple[RelionPreprocessCapture, ...], dict[str, object]]:
    """Validate a complete capture cohort and return a structural report."""

    directory = Path(directory)
    _require(directory.is_dir(), f"capture directory does not exist: {directory}")
    _require(not list(directory.glob("*.tmp.*")), "incomplete capture artifact remains")
    paths = sorted(directory.glob("*.preprocess-v1.bin"))
    _require(bool(paths), f"no preprocessing artifacts in {directory}")
    artifacts = tuple(load_artifact(path) for path in paths)
    reference = artifacts[0].header
    common_fields = tuple(
        index
        for index in range(40)
        if index not in (6, 7, 8, 9, 20, 21, 22, 23, 27, 35)
    )
    for artifact in artifacts[1:]:
        for field in common_fields:
            _require(
                artifact.header[field] == reference[field],
                f"inconsistent header field {field}: {artifact.path}",
            )

    part_ids = np.asarray([artifact.part_id for artifact in artifacts], dtype=np.int64)
    stack_indices = np.asarray(
        [artifact.stack_index for artifact in artifacts], dtype=np.int64
    )
    _require(
        np.unique(part_ids).size == len(artifacts),
        "duplicate particle identity in preprocessing capture",
    )
    _require(
        np.unique(stack_indices).size == len(artifacts),
        "duplicate stack identity in preprocessing capture",
    )
    _require(
        len(artifacts) == reference[29],
        "preprocessing capture completeness mismatch",
    )
    if expected_particles is not None:
        _require(
            len(artifacts) == expected_particles
            and reference[29] == expected_particles,
            "expected-particle mismatch",
        )
    if expected_part_ids is not None:
        expected = np.asarray(expected_part_ids, dtype=np.int64).reshape(-1)
        _require(
            np.array_equal(np.sort(part_ids), np.sort(expected)),
            "expected particle identities mismatch",
        )
    if expected_stack_indices is not None:
        expected = np.asarray(expected_stack_indices, dtype=np.int64).reshape(-1)
        _require(
            np.array_equal(np.sort(stack_indices), np.sort(expected)),
            "expected stack identities mismatch",
        )
    if expected_mpi_rank is not None:
        _require(
            all(artifact.mpi_rank == expected_mpi_rank for artifact in artifacts),
            "expected MPI rank mismatch",
        )
    if expected_iteration is not None:
        _require(
            all(artifact.iteration == expected_iteration for artifact in artifacts),
            "expected iteration mismatch",
        )

    report: dict[str, object] = {
        "schema": "relion-preprocess-capture-v1",
        "status": "pass",
        "capture_directory": str(directory.resolve()),
        "particle_count": len(artifacts),
        "iteration": reference[5],
        "real_shape": list(artifacts[0].raw_input_real.shape),
        "fourier_shape": list(
            artifacts[0].masked_fourier_post_optics.shape
        ),
        "fixed_metric": {
            "evaluated_particles": len(artifacts),
            "expected_particles": reference[29],
            "structurally_qualified": len(artifacts),
            "complete_seven_stage_payload": len(artifacts),
        },
        "artifacts": [
            {
                "path": str(artifact.path.resolve()),
                "sha256": artifact.sha256,
                "part_id": artifact.part_id,
                "stack_index": artifact.stack_index,
                "mpi_rank": artifact.mpi_rank,
                "norm_correction": artifact.norm_correction,
                "old_offset": artifact.old_offset.tolist(),
                "mask_parameters": artifact.mask_parameters,
            }
            for artifact in artifacts
        ],
    }
    return artifacts, report


def _parse_csv(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    values = [int(value) for value in text.split(",") if value]
    if not values or len(values) != len(set(values)):
        raise ValueError("expected identity CSV must be non-empty and unique")
    return np.asarray(values, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-particles", type=int)
    parser.add_argument("--expected-part-ids")
    parser.add_argument("--expected-stack-indices")
    parser.add_argument("--expected-mpi-rank", type=int)
    parser.add_argument("--expected-iteration", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, report = validate_directory(
        args.directory,
        expected_particles=args.expected_particles,
        expected_part_ids=_parse_csv(args.expected_part_ids),
        expected_stack_indices=_parse_csv(args.expected_stack_indices),
        expected_mpi_rank=args.expected_mpi_rank,
        expected_iteration=args.expected_iteration,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
