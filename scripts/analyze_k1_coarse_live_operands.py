#!/usr/bin/env python3
"""Compare live RELION coarse-kernel operands with passive captures bitwise."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.validate_relion_coarse_lane_capture import load_artifact as load_lanes
from scripts.validate_relion_coarse_operand_capture import load_artifact as load_operands
from scripts.validate_relion_coarse_pass1_components import (
    load_artifact as load_components,
)

HEADER_MAGIC = b"RLNP1LIV1HEADER\0"
FOOTER_MAGIC = b"RLNP1LIV1FOOTER\0"
HEADER_STRUCT = struct.Struct("<16s32Q")
FOOTER_STRUCT = struct.Struct("<16sQQ")
FLOAT_DTYPE = np.dtype("<f4")
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)\.p1-live-v1\.bin"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _has_complete_controls(paths: tuple[Path | None, ...]) -> bool:
    """Return whether a complete optional control-capture group was supplied."""

    supplied = tuple(path is not None for path in paths)
    _require(all(supplied) or not any(supplied), "control captures must be supplied together")
    return all(supplied)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LiveCoarseOperands:
    path: Path
    sha256: str
    header: tuple[int, ...]
    reference_real: np.ndarray
    reference_imag: np.ndarray
    shifted_real: np.ndarray
    shifted_imag: np.ndarray
    correction_half: np.ndarray

    @property
    def part_id(self) -> int:
        return self.header[5]

    @property
    def stack_index(self) -> int:
        return self.header[6]


def load_live_artifact(path: Path) -> LiveCoarseOperands:
    """Load and fail-closed validate one production-kernel operand capture."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected live-operand file name: {path.name}")
    payload = path.read_bytes()
    _require(
        len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size,
        f"truncated live-operand artifact: {path}",
    )
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"header magic mismatch: {path}")
    _require(header[0] == 1, f"schema mismatch: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"header size mismatch: {path}")
    _require(header[2] == FLOAT_DTYPE.itemsize, f"float size mismatch: {path}")
    _require(header[3] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")
    _require(header[4] > 0, f"iteration must be positive: {path}")
    _require(header[9:11] == (1, 0), f"expected K=1 image 0: {path}")
    pixel_count = header[12]
    translation_count = header[13]
    local_rotation = header[15]
    orientation_count = header[16]
    block_size, prefetch_fraction, eulers_per_block = header[17:20]
    value_count = header[20]
    _require(pixel_count > 0 and translation_count > 0, f"empty topology: {path}")
    _require(local_rotation < orientation_count, f"local rotation out of range: {path}")
    _require(block_size == 128 and prefetch_fraction == 4, f"unexpected CUDA topology: {path}")
    _require(eulers_per_block > 0, f"invalid Euler block topology: {path}")
    _require(value_count == (3 + 2 * translation_count) * pixel_count, f"value count mismatch: {path}")
    _require(header[21] > 0, f"estimated byte count missing: {path}")
    _require(header[22] != 0 and header[23] != 0, f"identity hash missing: {path}")
    _require(header[24:27] == (1, 1, 1), f"payload flags missing: {path}")
    expected_size = HEADER_STRUCT.size + value_count * FLOAT_DTYPE.itemsize + FOOTER_STRUCT.size
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")

    values = np.frombuffer(
        payload,
        dtype=FLOAT_DTYPE,
        count=value_count,
        offset=HEADER_STRUCT.size,
    ).copy()
    offset = 0

    def take(count: int) -> np.ndarray:
        nonlocal offset
        result = values[offset : offset + count]
        offset += count
        return result

    reference_real = take(pixel_count)
    reference_imag = take(pixel_count)
    shifted_real = take(translation_count * pixel_count).reshape(
        translation_count, pixel_count
    )
    shifted_imag = take(translation_count * pixel_count).reshape(
        translation_count, pixel_count
    )
    correction_half = take(pixel_count)
    _require(offset == value_count, f"payload offset mismatch: {path}")
    footer_offset = HEADER_STRUCT.size + value_count * FLOAT_DTYPE.itemsize
    footer_magic, footer_translations, footer_pixels = FOOTER_STRUCT.unpack_from(
        payload, footer_offset
    )
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(footer_translations == translation_count, f"footer translation mismatch: {path}")
    _require(footer_pixels == pixel_count, f"footer pixel mismatch: {path}")
    assert match is not None
    _require(int(match["part"]) == header[5], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[6], f"stack identity mismatch: {path}")
    _require(np.all(np.isfinite(values)), f"non-finite live operand: {path}")
    _require(np.all(correction_half >= 0), f"negative correction: {path}")
    return LiveCoarseOperands(
        path,
        _sha256(path),
        header,
        reference_real,
        reference_imag,
        shifted_real,
        shifted_imag,
        correction_half,
    )


def _fma_f32(left: np.float32, right: np.float32, addend: np.float32) -> np.float32:
    return np.float32(np.float64(left) * np.float64(right) + np.float64(addend))


def replay_live_lanes(live: LiveCoarseOperands) -> np.ndarray:
    """Replay all 128 production thread partials from captured live operands."""

    pixel_count = live.header[12]
    translation_count = live.header[13]
    block_size = live.header[17]
    prefetch_fraction = live.header[18]
    pixels_per_pass = block_size // prefetch_fraction
    active_rows = block_size // translation_count
    max_pixel = ((pixel_count + block_size - 1) // block_size) * block_size
    partials = np.zeros(block_size, dtype=np.float32)
    for thread in range(block_size):
        translation = thread % translation_count
        first_pixel_in_pass = thread // translation_count
        if first_pixel_in_pass >= active_rows:
            continue
        accumulator = np.float32(0.0)
        for init_pixel in range(0, max_pixel, pixels_per_pass):
            for pixel_in_pass in range(
                first_pixel_in_pass, pixels_per_pass, active_rows
            ):
                pixel = init_pixel + pixel_in_pass
                if pixel >= pixel_count:
                    break
                diff_real = np.float32(
                    live.reference_real[pixel]
                    - live.shifted_real[translation, pixel]
                )
                diff_imag = np.float32(
                    live.reference_imag[pixel]
                    - live.shifted_imag[translation, pixel]
                )
                square = _fma_f32(
                    diff_real,
                    diff_real,
                    np.float32(diff_imag * diff_imag),
                )
                accumulator = _fma_f32(
                    square, live.correction_half[pixel], accumulator
                )
        partials[thread] = accumulator
    return partials


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    _require(left.shape == right.shape, "comparison shapes differ")
    exact = left.view(np.uint32) == right.view(np.uint32)
    mismatch = np.argwhere(~exact)
    residual = left.astype(np.float64) - right.astype(np.float64)
    return {
        "shape": list(left.shape),
        "value_count": int(left.size),
        "bitwise_equal_count": int(np.count_nonzero(exact)),
        "bitwise_equal_fraction": float(np.mean(exact)),
        "first_mismatch_index": mismatch[0].tolist() if mismatch.size else None,
        "first_left": float(left[tuple(mismatch[0])]) if mismatch.size else None,
        "first_right": float(right[tuple(mismatch[0])]) if mismatch.size else None,
        "max_abs_difference": float(np.max(np.abs(residual))),
    }


def _component_inertness(control, capture) -> dict[str, object]:
    _require(control.part_id == capture.part_id, "component particle identities differ")
    _require(control.stack_index == capture.stack_index, "component stack identities differ")
    return {
        "raw_diff2": _comparison(capture.raw_diff2, control.raw_diff2),
        "weights": _comparison(capture.weights, control.weights),
        "reference_norms": _comparison(
            capture.reference_norms, control.reference_norms
        ),
        "cross_terms": _comparison(capture.cross_terms, control.cross_terms),
        "significant_mask_equal": bool(
            np.array_equal(capture.significant_mask, control.significant_mask)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--lanes", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--control-operands", type=Path)
    parser.add_argument("--control-lanes", type=Path)
    parser.add_argument("--control-components", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")

    live = load_live_artifact(args.live)
    operands = load_operands(args.operands)
    lanes = load_lanes(args.lanes)
    components = load_components(args.components)
    control_paths = (
        args.control_operands,
        args.control_lanes,
        args.control_components,
    )
    has_controls = _has_complete_controls(control_paths)
    control_operands = load_operands(args.control_operands) if has_controls else None
    control_lanes = load_lanes(args.control_lanes) if has_controls else None
    control_components = (
        load_components(args.control_components) if has_controls else None
    )
    _require(
        live.part_id
        == operands.part_id
        == lanes.part_id
        == components.part_id,
        "particle identities differ",
    )
    _require(
        live.stack_index
        == operands.stack_index
        == lanes.stack_index
        == components.stack_index,
        "stack identities differ",
    )
    if has_controls:
        assert control_operands is not None
        assert control_lanes is not None
        assert control_components is not None
        _require(
            live.part_id
            == control_operands.part_id
            == control_lanes.part_id
            == control_components.part_id,
            "control particle identities differ",
        )
        _require(
            live.stack_index
            == control_operands.stack_index
            == control_lanes.stack_index
            == control_components.stack_index,
            "control stack identities differ",
        )
    _require(operands.rotation_keys.size == 1, "expected one passive rotation")
    _require(lanes.rotation_keys.size == 1, "expected one lane rotation")
    _require(
        int(operands.rotation_keys[0]) == live.header[14],
        "live/passive rotation keys differ",
    )
    _require(
        int(operands.local_rotation_indices[0]) == live.header[15],
        "live/passive local rotations differ",
    )

    correction_half = np.asarray(operands.correction * np.float32(0.5), dtype=np.float32)
    passive_comparisons = {
        "reference_real": _comparison(live.reference_real, operands.reference_real[0]),
        "reference_imag": _comparison(live.reference_imag, operands.reference_imag[0]),
        "shifted_real": _comparison(live.shifted_real, operands.shifted_real),
        "shifted_imag": _comparison(live.shifted_imag, operands.shifted_imag),
        "correction_half": _comparison(live.correction_half, correction_half),
    }
    replayed_lanes = replay_live_lanes(live)
    lane_comparison = _comparison(replayed_lanes, lanes.lane_partials[0])
    control_comparisons = None
    all_control_exact = None
    if has_controls:
        assert control_operands is not None
        assert control_lanes is not None
        assert control_components is not None
        control_comparisons = {
            "components": _component_inertness(control_components, components),
            "lane_partials": _comparison(
                lanes.lane_partials, control_lanes.lane_partials
            ),
            "passive_reference_real": _comparison(
                operands.reference_real, control_operands.reference_real
            ),
            "passive_reference_imag": _comparison(
                operands.reference_imag, control_operands.reference_imag
            ),
            "passive_shifted_real": _comparison(
                operands.shifted_real, control_operands.shifted_real
            ),
            "passive_shifted_imag": _comparison(
                operands.shifted_imag, control_operands.shifted_imag
            ),
            "passive_correction": _comparison(
                operands.correction, control_operands.correction
            ),
        }
        all_control_exact = all(
            row["bitwise_equal_count"] == row["value_count"]
            for name, row in control_comparisons.items()
            if name != "components"
        ) and all(
            row["bitwise_equal_count"] == row["value_count"]
            for name, row in control_comparisons["components"].items()
            if name != "significant_mask_equal"
        ) and bool(control_comparisons["components"]["significant_mask_equal"])
    differing_operand_names = [
        name
        for name, row in passive_comparisons.items()
        if row["bitwise_equal_count"] != row["value_count"]
    ]
    report = {
        "schema": "k1-coarse-live-operand-analysis-v1",
        "status": (
            "pass"
            if all_control_exact
            else "rejected_instrumentation_not_inert"
            if all_control_exact is False
            else "pass_live_boundary"
        ),
        "instrumentation_bitwise_inert": all_control_exact,
        "classification": (
            "live_operands_identify_first_passive_capture_difference"
            if all_control_exact is not False and differing_operand_names
            else "live_and_passive_operands_bitwise_equal"
            if all_control_exact is not False
            else "instrumentation_changed_production_outputs"
        ),
        "part_id": live.part_id,
        "stack_index_one_based": live.stack_index,
        "rotation_key": live.header[14],
        "local_rotation_index": live.header[15],
        "live_path": str(live.path.resolve()),
        "live_sha256": live.sha256,
        "differing_operand_names": differing_operand_names,
        "live_vs_passive": passive_comparisons,
        "live_replay_vs_production_lanes": lane_comparison,
        "capture_vs_control": control_comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if all_control_exact is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
