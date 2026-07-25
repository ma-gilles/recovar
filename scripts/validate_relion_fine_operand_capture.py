#!/usr/bin/env python3
"""Validate bounded RELION CUDA fine-score operand captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_SIZE = 528
CANDIDATE_SIZE = 1120
PIXEL_SIZE = 64
FOOTER_SIZE = 32
LANE_COUNT = 256

CANDIDATE_DTYPE = np.dtype(
    [
        ("sparse_index", "<u8"),
        ("rotation_id", "<u8"),
        ("rotation_local", "<u8"),
        ("translation_id", "<u8"),
        ("matrix", "<f4", (9,)),
        ("translation", "<f4", (3,)),
        ("sum_init", "<f4"),
        ("production_raw_diff2", "<f4"),
        ("replay_raw_diff2", "<f4"),
        ("flags", "<u4"),
        ("lane_partials", "<f4", (LANE_COUNT,)),
    ],
    align=False,
)
PIXEL_DTYPE = np.dtype(
    [
        ("target_index", "<u4"),
        ("pixel", "<u4"),
        ("x", "<i4"),
        ("y", "<i4"),
        ("z", "<i4"),
        ("flags", "<u4"),
        ("image_real", "<f4"),
        ("image_imag", "<f4"),
        ("reference_real", "<f4"),
        ("reference_imag", "<f4"),
        ("shifted_real", "<f4"),
        ("shifted_imag", "<f4"),
        ("corr", "<f4"),
        ("diff_real", "<f4"),
        ("diff_imag", "<f4"),
        ("contribution", "<f4"),
    ],
    align=False,
)


@dataclass(frozen=True)
class FineOperandCapture:
    path: Path
    header: np.ndarray
    candidates: np.ndarray
    pixels: np.ndarray
    footer_candidate_count: int
    footer_pixel_count: int

    @property
    def iteration(self) -> int:
        return int(self.header[5])

    @property
    def class_one_based(self) -> int:
        return int(self.header[6])

    @property
    def particle_id(self) -> int:
        return int(self.header[7])

    @property
    def stack_index(self) -> int:
        return int(self.header[8])

    @property
    def image_size(self) -> int:
        return int(self.header[13])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_magic(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("ascii")


def _float32_from_bits(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0])


def _float32_bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def load_fine_operand_capture(path: Path) -> FineOperandCapture:
    path = Path(path)
    with path.open("rb") as stream:
        header_magic = stream.read(16)
        _require(
            _decode_magic(header_magic) == "RLNFNOP1HEADER",
            f"{path}: invalid fine-operand header magic",
        )
        header = np.frombuffer(stream.read(64 * 8), dtype="<u8").copy()
        _require(header.size == 64, f"{path}: truncated fine-operand header")
        _require(int(header[0]) == 1, f"{path}: unsupported schema version")
        _require(int(header[1]) == HEADER_SIZE, f"{path}: header size changed")
        _require(int(header[2]) == CANDIDATE_SIZE, f"{path}: candidate size changed")
        _require(int(header[3]) == PIXEL_SIZE, f"{path}: pixel size changed")
        _require(int(header[4]) == FOOTER_SIZE, f"{path}: footer size changed")
        candidate_count = int(header[12])
        image_size = int(header[13])
        pixel_count = int(header[14])
        _require(candidate_count > 0 and image_size > 0, f"{path}: empty runtime dimensions")
        _require(
            pixel_count == candidate_count * image_size,
            f"{path}: pixel count does not match candidates times image size",
        )
        candidates = np.frombuffer(
            stream.read(candidate_count * CANDIDATE_SIZE),
            dtype=CANDIDATE_DTYPE,
        ).copy()
        pixels = np.frombuffer(
            stream.read(pixel_count * PIXEL_SIZE),
            dtype=PIXEL_DTYPE,
        ).copy()
        footer_magic = stream.read(16)
        _require(
            _decode_magic(footer_magic) == "RLNFNOP1FOOTER",
            f"{path}: invalid fine-operand footer magic",
        )
        footer = np.frombuffer(stream.read(16), dtype="<u8").copy()
        _require(footer.size == 2, f"{path}: truncated fine-operand footer")
        _require(stream.read(1) == b"", f"{path}: trailing bytes after fine-operand footer")
    expected_size = (
        HEADER_SIZE
        + candidate_count * CANDIDATE_SIZE
        + pixel_count * PIXEL_SIZE
        + FOOTER_SIZE
    )
    _require(path.stat().st_size == expected_size, f"{path}: file size changed")
    return FineOperandCapture(
        path=path,
        header=header,
        candidates=candidates,
        pixels=pixels,
        footer_candidate_count=int(footer[0]),
        footer_pixel_count=int(footer[1]),
    )


def _replay_lanes(contributions: np.ndarray) -> np.ndarray:
    values = np.asarray(contributions, dtype=np.float32).reshape(-1)
    lanes = np.zeros(LANE_COUNT, dtype=np.float32)
    for pixel, value in enumerate(values):
        lane = pixel % LANE_COUNT
        lanes[lane] = np.float32(lanes[lane] + value)
    return lanes


def _reduce_lanes(lanes: np.ndarray) -> np.float32:
    values = np.asarray(lanes, dtype=np.float32).copy()
    _require(values.shape == (LANE_COUNT,), f"lane shape changed: {values.shape}")
    width = LANE_COUNT // 2
    while width:
        values[:width] = np.add(
            values[:width],
            values[width : 2 * width],
            dtype=np.float32,
        )
        width //= 2
    return np.float32(values[0])


def _cuda_fine_contribution(
    diff_real: np.ndarray,
    diff_imag: np.ndarray,
    corr: np.ndarray,
) -> np.ndarray:
    """Replay NVCC's contracted fine-kernel contribution expression.

    The pinned CUDA build contracts ``diff_real * diff_real + diff_imag *
    diff_imag`` as ``fmaf(diff_real, diff_real, roundf(diff_imag *
    diff_imag))``.  The following float64 host expression reproduces that one
    float32 rounding boundary without requiring a platform ``fmaf`` binding.
    The subsequent ``0.5`` and correlation multiplies remain separately
    rounded float32 operations.
    """

    real = np.asarray(diff_real, dtype=np.float32)
    imag = np.asarray(diff_imag, dtype=np.float32)
    weight = np.asarray(corr, dtype=np.float32)
    imag_square = np.multiply(imag, imag, dtype=np.float32)
    contracted_square_sum = np.asarray(
        real.astype(np.float64) * real.astype(np.float64)
        + imag_square.astype(np.float64),
        dtype=np.float32,
    )
    return np.multiply(
        np.multiply(contracted_square_sum, np.float32(0.5), dtype=np.float32),
        weight,
        dtype=np.float32,
    )


def validate_capture(
    capture: FineOperandCapture,
    *,
    expected_stack: int | None = None,
    expected_class: int | None = None,
    expected_rotation_local: int | None = None,
    expected_translations: tuple[int, ...] | None = None,
) -> dict[str, object]:
    candidates = capture.candidates
    pixels = capture.pixels.reshape(candidates.size, capture.image_size)
    _require(
        capture.footer_candidate_count == candidates.size
        and capture.footer_pixel_count == capture.pixels.size,
        "fine-operand footer counts disagree with the header",
    )
    _require(
        int(capture.header[18]) == LANE_COUNT,
        f"fine-operand reduction lane count changed: {int(capture.header[18])}",
    )
    _require(int(capture.header[28]) == 1, "capture is not marked passive")
    _require(int(capture.header[29]) == 1, "per-pixel operands are not marked complete")
    _require(int(capture.header[30]) == 1, "pre-tree lane partials are not marked complete")
    if expected_stack is not None:
        _require(capture.stack_index == expected_stack, "captured stack identity changed")
    if expected_class is not None:
        _require(capture.class_one_based == expected_class, "captured class identity changed")
    if expected_rotation_local is not None:
        _require(
            np.all(candidates["rotation_local"] == expected_rotation_local),
            "captured rotation-local identity changed",
        )
    if expected_translations is not None:
        _require(
            tuple(int(value) for value in candidates["translation_id"]) == expected_translations,
            "captured translation identities changed",
        )

    expected_target = np.arange(candidates.size, dtype=np.uint32)[:, None]
    expected_pixel = np.arange(capture.image_size, dtype=np.uint32)[None, :]
    _require(
        np.array_equal(pixels["target_index"], np.broadcast_to(expected_target, pixels.shape)),
        "pixel target identities are not dense and ordered",
    )
    _require(
        np.array_equal(pixels["pixel"], np.broadcast_to(expected_pixel, pixels.shape)),
        "pixel identities are not dense and ordered",
    )
    _require(np.all((pixels["flags"] & 1) != 0), "one or more pixel records are invalid")
    _require(np.all((candidates["flags"] & 1) != 0), "one or more candidate records are invalid")

    finite_fields = (
        "image_real",
        "image_imag",
        "reference_real",
        "reference_imag",
        "shifted_real",
        "shifted_imag",
        "corr",
        "diff_real",
        "diff_imag",
        "contribution",
    )
    for name in finite_fields:
        _require(np.all(np.isfinite(pixels[name])), f"non-finite pixel field: {name}")
    _require(np.all(np.isfinite(candidates["matrix"])), "non-finite rotation matrix")
    _require(np.all(np.isfinite(candidates["translation"])), "non-finite translation")
    _require(np.all(np.isfinite(candidates["lane_partials"])), "non-finite lane partial")

    expected_diff_real = np.subtract(
        pixels["reference_real"],
        pixels["shifted_real"],
        dtype=np.float32,
    )
    expected_diff_imag = np.subtract(
        pixels["reference_imag"],
        pixels["shifted_imag"],
        dtype=np.float32,
    )
    expected_contribution = _cuda_fine_contribution(
        expected_diff_real,
        expected_diff_imag,
        pixels["corr"],
    )
    _require(
        np.array_equal(_float32_bits(pixels["diff_real"]), _float32_bits(expected_diff_real)),
        "captured real differences do not replay bitwise",
    )
    _require(
        np.array_equal(_float32_bits(pixels["diff_imag"]), _float32_bits(expected_diff_imag)),
        "captured imaginary differences do not replay bitwise",
    )
    _require(
        np.array_equal(
            _float32_bits(pixels["contribution"]),
            _float32_bits(expected_contribution),
        ),
        "captured per-pixel contributions do not replay bitwise",
    )

    rows = []
    exact_production_count = 0
    for target, candidate in enumerate(candidates):
        replay_lanes = _replay_lanes(pixels[target]["contribution"])
        _require(
            np.array_equal(
                _float32_bits(candidate["lane_partials"]),
                _float32_bits(replay_lanes),
            ),
            f"target {target}: pre-tree lanes do not replay bitwise",
        )
        replay_sum = np.float32(
            _reduce_lanes(replay_lanes) + np.float32(candidate["sum_init"])
        )
        _require(
            _float32_bits(np.asarray([replay_sum]))[0]
            == _float32_bits(np.asarray([candidate["replay_raw_diff2"]]))[0],
            f"target {target}: replay raw diff2 does not match captured lanes",
        )
        production_exact = bool(
            _float32_bits(np.asarray([candidate["production_raw_diff2"]]))[0]
            == _float32_bits(np.asarray([candidate["replay_raw_diff2"]]))[0]
        )
        _require(
            production_exact == bool(int(candidate["flags"]) & 2),
            f"target {target}: exact-replay flag disagrees with the values",
        )
        exact_production_count += int(production_exact)
        rows.append(
            {
                "target_index": target,
                "sparse_index": int(candidate["sparse_index"]),
                "rotation_id": int(candidate["rotation_id"]),
                "rotation_local": int(candidate["rotation_local"]),
                "translation_id": int(candidate["translation_id"]),
                "sum_init": float(candidate["sum_init"]),
                "production_raw_diff2": float(candidate["production_raw_diff2"]),
                "replay_raw_diff2": float(candidate["replay_raw_diff2"]),
                "production_replay_exact": production_exact,
            }
        )
    _require(
        exact_production_count == int(capture.header[21]),
        "header exact-replay count disagrees with candidate records",
    )
    return {
        "schema": "relion_fine_operand_capture_validation_v1",
        "status": "accepted",
        "path": str(capture.path.resolve()),
        "sha256": _sha256(capture.path),
        "iteration": capture.iteration,
        "class_one_based": capture.class_one_based,
        "particle_id": capture.particle_id,
        "stack_index_one_based": capture.stack_index,
        "candidate_count": int(candidates.size),
        "image_size": capture.image_size,
        "pixel_count": int(capture.pixels.size),
        "reduction_lane_count": int(capture.header[18]),
        "reduction_translation_chunk": int(capture.header[19]),
        "sum_init_from_header_bits": float(_float32_from_bits(int(capture.header[20]))),
        "exact_production_replay_count": exact_production_count,
        "production_replay_mismatch_count": int(candidates.size - exact_production_count),
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--expected-stack", type=int)
    parser.add_argument("--expected-class", type=int)
    parser.add_argument("--expected-rotation-local", type=int)
    parser.add_argument("--expected-translations")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    translations = (
        tuple(int(value) for value in args.expected_translations.split(","))
        if args.expected_translations
        else None
    )
    report = validate_capture(
        load_fine_operand_capture(args.capture),
        expected_stack=args.expected_stack,
        expected_class=args.expected_class,
        expected_rotation_local=args.expected_rotation_local,
        expected_translations=translations,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
