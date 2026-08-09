#!/usr/bin/env python3
"""Validate native in-kernel RELION coarse pixel/update captures.

The inline artifact records one selected rotation directly inside RELION's
production coarse CUDA kernel.  It is joined to the passive operand capture
and the production lane-partial capture to distinguish projection,
translation, correction, and contraction/accumulation boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts import validate_relion_coarse_lane_capture as lane_validator
from scripts import validate_relion_coarse_operand_capture as operand_validator

HEADER_MAGIC = b"RLNP1PXV1HEADER\0"
FOOTER_MAGIC = b"RLNP1PXV1FOOTER\0"
HEADER_STRUCT = struct.Struct("<16s40Q")
FOOTER_STRUCT = struct.Struct("<16s4Q")
FLOAT_DTYPE = np.dtype("<f4")
FILE_NAME = re.compile(r"part(?P<part>\d+)_stack(?P<stack>\d+)\.p1-inline-v1\.bin")
FIELD_NAMES = (
    "reference_real",
    "reference_imag",
    "shifted_real",
    "shifted_imag",
    "correction_half",
    "difference_real",
    "difference_imag",
    "accumulator_before",
    "accumulator_after",
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


def _bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _exact_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    _require(actual.shape == expected.shape, f"comparison shape mismatch: {actual.shape} vs {expected.shape}")
    equal = _bits(actual) == _bits(expected)
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    mismatch = np.argwhere(~equal)
    return {
        "evaluated": int(equal.size),
        "bitwise_equal": int(np.count_nonzero(equal)),
        "bitwise_equal_fraction": float(np.count_nonzero(equal) / equal.size),
        "max_abs": float(np.max(difference, initial=0.0)),
        "p95_abs": float(np.percentile(difference, 95)),
        "first_mismatch": None if mismatch.size == 0 else mismatch[0].astype(int).tolist(),
        "exact": bool(np.all(equal)),
    }


@dataclass(frozen=True)
class CoarseInlineCapture:
    path: Path
    sha256: str
    header: tuple[int, ...]
    fields: np.ndarray

    @property
    def part_id(self) -> int:
        return self.header[6]

    @property
    def stack_index(self) -> int:
        return self.header[7]

    @property
    def rotation_key(self) -> int:
        return self.header[15]


def load_artifact(path: Path) -> CoarseInlineCapture:
    """Load and structurally validate one sealed inline artifact."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected inline-capture file name: {path.name}")
    payload = path.read_bytes()
    _require(len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size, f"truncated artifact: {path}")
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"header magic mismatch: {path}")
    _require(header[0] == 1, f"schema mismatch: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"header size mismatch: {path}")
    _require(header[2] == FLOAT_DTYPE.itemsize, f"float size mismatch: {path}")
    _require(header[4] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")
    _require(header[5] > 0 and header[10:12] == (1, 0), f"expected K=1 image-0 capture: {path}")
    image_size, translation_count = header[13:15]
    rotation_key, local_rotation, orientation_count = header[15:18]
    block_size, eulers_per_block, prefetch_fraction = header[18:21]
    field_count = header[22]
    _require(image_size > 0 and translation_count > 0, f"empty inline topology: {path}")
    _require(rotation_key < orientation_count and local_rotation < orientation_count, f"rotation outside plan: {path}")
    _require(block_size > 0 and block_size % prefetch_fraction == 0, f"invalid CUDA block topology: {path}")
    _require(block_size // translation_count > 0 and eulers_per_block > 0, f"invalid CUDA work mapping: {path}")
    _require(field_count == len(FIELD_NAMES), f"field-count mismatch: {path}")
    _require(header[27] <= header[26], f"capture byte cap exceeded: {path}")
    _require(header[28] != 0 and header[29] != 0, f"identity hash missing: {path}")
    _require(header[30:34] == (1, 1, 1, 1), f"inline-capture flags missing: {path}")
    value_count = field_count * translation_count * image_size
    expected_size = HEADER_STRUCT.size + value_count * FLOAT_DTYPE.itemsize + FOOTER_STRUCT.size
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")
    fields = np.frombuffer(payload, dtype=FLOAT_DTYPE, count=value_count, offset=HEADER_STRUCT.size).copy()
    fields = fields.reshape(field_count, translation_count, image_size)
    footer_offset = HEADER_STRUCT.size + value_count * FLOAT_DTYPE.itemsize
    footer_magic, footer_rotation, footer_translations, footer_pixels, footer_fields = FOOTER_STRUCT.unpack_from(
        payload, footer_offset
    )
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(footer_rotation == rotation_key, f"footer rotation mismatch: {path}")
    _require(footer_translations == translation_count, f"footer translation mismatch: {path}")
    _require(footer_pixels == image_size and footer_fields == field_count, f"footer shape mismatch: {path}")
    assert match is not None
    _require(int(match["part"]) == header[6], f"particle identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"stack identity mismatch: {path}")
    _require(np.all(np.isfinite(fields)), f"non-finite inline value: {path}")
    return CoarseInlineCapture(path, _sha256(path), header, fields)


def _lane_pixel_order(image_size: int, block_size: int, prefetch_fraction: int, lane_group: int, stride: int):
    pass_pixels = block_size // prefetch_fraction
    padded_size = ((image_size + block_size - 1) // block_size) * block_size
    for init_pixel in range(0, padded_size, pass_pixels):
        for local_pixel in range(lane_group, pass_pixels, stride):
            pixel = init_pixel + local_pixel
            if pixel < image_size:
                yield pixel


def _accumulation_metrics(inline: CoarseInlineCapture, lane) -> dict[str, object]:
    translation_count = inline.header[14]
    image_size = inline.header[13]
    block_size = inline.header[18]
    prefetch_fraction = inline.header[20]
    stride = block_size // translation_count
    before = inline.fields[7]
    after = inline.fields[8]
    diff_real = inline.fields[5]
    diff_imag = inline.fields[6]
    correction = inline.fields[4]
    target = int(np.flatnonzero(lane.rotation_keys == inline.rotation_key)[0])
    continuity_actual = []
    continuity_expected = []
    replay_actual = []
    replay_expected = []
    final_actual = []
    final_expected = []
    for translation in range(translation_count):
        for lane_group in range(stride):
            pixels = list(_lane_pixel_order(image_size, block_size, prefetch_fraction, lane_group, stride))
            total = np.float32(0.0)
            for pixel in pixels:
                continuity_actual.append(before[translation, pixel])
                continuity_expected.append(total)
                square_real = np.float32(diff_real[translation, pixel] * diff_real[translation, pixel])
                square_imag = np.float32(diff_imag[translation, pixel] * diff_imag[translation, pixel])
                term = np.float32(np.float32(square_real + square_imag) * correction[translation, pixel])
                total = np.float32(total + term)
                replay_actual.append(after[translation, pixel])
                replay_expected.append(total)
                total = after[translation, pixel]
            lane_index = translation + lane_group * translation_count
            final_actual.append(total)
            final_expected.append(lane.lane_partials[target, lane_index])
    return {
        "accumulator_continuity": _exact_metrics(continuity_actual, continuity_expected),
        "separate_float32_update_replay": _exact_metrics(replay_actual, replay_expected),
        "final_lane_partial": _exact_metrics(final_actual, final_expected),
    }


def validate_capture(inline: CoarseInlineCapture, operand, lane) -> dict[str, object]:
    """Join one inline capture to passive operands and native lane partials."""

    _require(inline.part_id == operand.part_id == lane.part_id, "particle identities differ")
    _require(inline.stack_index == operand.stack_index == lane.stack_index, "stack identities differ")
    _require(inline.header[5] == operand.header[5] == lane.header[5], "iterations differ")
    _require(inline.header[13] == operand.header[13] == lane.header[13], "image sizes differ")
    _require(inline.header[14] == operand.header[14] == lane.header[14], "translation counts differ")
    _require(inline.rotation_key in operand.rotation_keys, "inline rotation absent from operand capture")
    _require(inline.rotation_key in lane.rotation_keys, "inline rotation absent from lane capture")
    operand_index = int(np.flatnonzero(operand.rotation_keys == inline.rotation_key)[0])
    _require(
        int(operand.local_rotation_indices[operand_index]) == inline.header[16],
        "local rotation indices differ",
    )
    translation_count = inline.header[14]
    reference_real = np.broadcast_to(operand.reference_real[operand_index], inline.fields[0].shape)
    reference_imag = np.broadcast_to(operand.reference_imag[operand_index], inline.fields[1].shape)
    correction_half = np.broadcast_to(
        np.float32(operand.correction / np.float32(2.0)), inline.fields[4].shape
    )
    comparisons = {
        "projection_real_vs_passive": _exact_metrics(inline.fields[0], reference_real),
        "projection_imag_vs_passive": _exact_metrics(inline.fields[1], reference_imag),
        "translation_real_vs_passive": _exact_metrics(inline.fields[2], operand.shifted_real),
        "translation_imag_vs_passive": _exact_metrics(inline.fields[3], operand.shifted_imag),
        "correction_half_vs_passive": _exact_metrics(inline.fields[4], correction_half),
        "difference_real_subtraction": _exact_metrics(
            inline.fields[5], np.float32(inline.fields[0] - inline.fields[2])
        ),
        "difference_imag_subtraction": _exact_metrics(
            inline.fields[6], np.float32(inline.fields[1] - inline.fields[3])
        ),
    }
    accumulation = _accumulation_metrics(inline, lane)
    capture_valid = (
        accumulation["accumulator_continuity"]["exact"]
        and accumulation["final_lane_partial"]["exact"]
    )
    projection_exact = comparisons["projection_real_vs_passive"]["exact"] and comparisons[
        "projection_imag_vs_passive"
    ]["exact"]
    translation_exact = comparisons["translation_real_vs_passive"]["exact"] and comparisons[
        "translation_imag_vs_passive"
    ]["exact"]
    correction_exact = comparisons["correction_half_vs_passive"]["exact"]
    subtraction_exact = comparisons["difference_real_subtraction"]["exact"] and comparisons[
        "difference_imag_subtraction"
    ]["exact"]
    replay_exact = accumulation["separate_float32_update_replay"]["exact"]
    if not capture_valid:
        classification = "inline_capture_does_not_close_native_lane_trajectory"
        status = "rejected"
    elif not projection_exact:
        classification = "native_projection_differs_from_passive_projection_capture"
        status = "pass"
    elif not translation_exact:
        classification = "native_translation_differs_from_passive_translation_capture"
        status = "pass"
    elif not correction_exact:
        classification = "native_correction_differs_from_passive_correction_capture"
        status = "pass"
    elif not subtraction_exact:
        classification = "native_difference_subtraction_differs_from_separate_float32"
        status = "pass"
    elif not replay_exact:
        classification = "native_weighted_square_or_accumulation_differs_from_separate_float32"
        status = "pass"
    else:
        classification = "native_inline_operands_and_separate_float32_replay_are_exact"
        status = "pass"
    return {
        "schema": "relion-coarse-inline-capture-validation-v1",
        "status": status,
        "classification_ready": capture_valid,
        "classification": classification,
        "path": str(inline.path.resolve()),
        "sha256": inline.sha256,
        "part_id": inline.part_id,
        "stack_index_one_based": inline.stack_index,
        "rotation_key": inline.rotation_key,
        "translation_count": translation_count,
        "pixel_count": inline.header[13],
        "field_names": FIELD_NAMES,
        "passive_operand_comparisons": comparisons,
        "native_accumulation": accumulation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inline", required=True, type=Path)
    parser.add_argument("--operand", required=True, type=Path)
    parser.add_argument("--lane", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate_capture(
        load_artifact(args.inline),
        operand_validator.load_artifact(args.operand),
        lane_validator.load_artifact(args.lane),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "classification": report["classification"]}))


if __name__ == "__main__":
    main()
