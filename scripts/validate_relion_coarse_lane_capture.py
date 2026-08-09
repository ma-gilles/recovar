#!/usr/bin/env python3
"""Validate passive RELION coarse CUDA lane captures against sealed operands."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts import validate_relion_coarse_operand_capture as operand_validator
from scripts import validate_relion_coarse_pass1_components as component_validator

HEADER_MAGIC = b"RLNP1LNV1HEADER\0"
FOOTER_MAGIC = b"RLNP1LNV1FOOTER\0"
HEADER_STRUCT = struct.Struct("<16s32Q")
FOOTER_STRUCT = struct.Struct("<16sQQ")
UINT64_DTYPE = np.dtype("<u8")
FLOAT_DTYPE = np.dtype("<f4")
FILE_NAME = re.compile(r"part(?P<part>\d+)_stack(?P<stack>\d+)\.p1-lane-v1\.bin")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _float32_from_bits(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0])


@dataclass(frozen=True)
class CoarseLaneCapture:
    path: Path
    sha256: str
    header: tuple[int, ...]
    rotation_keys: np.ndarray
    local_rotation_indices: np.ndarray
    lane_partials: np.ndarray

    @property
    def part_id(self) -> int:
        return self.header[6]

    @property
    def stack_index(self) -> int:
        return self.header[7]


def load_artifact(path: Path) -> CoarseLaneCapture:
    """Load one complete coarse-lane artifact and validate its structure."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected lane-capture file name: {path.name}")
    payload = path.read_bytes()
    _require(len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size, f"truncated artifact: {path}")
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"header magic mismatch: {path}")
    _require(header[0] == 1, f"schema mismatch: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"header size mismatch: {path}")
    _require(header[2] == FLOAT_DTYPE.itemsize, f"float size mismatch: {path}")
    _require(header[3] == UINT64_DTYPE.itemsize, f"integer size mismatch: {path}")
    _require(header[4] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")
    _require(header[5] > 0, f"iteration must be positive: {path}")
    _require(header[10] == 1 and header[11] == 0, f"expected K=1 image 0: {path}")
    image_size, translation_count, rotation_count, orientation_count = header[13:17]
    lane_count = header[17]
    prefetch_fraction = header[18]
    eulers_per_block = header[19]
    _require(
        image_size > 0 and translation_count > 0 and rotation_count > 0,
        f"empty lane topology: {path}",
    )
    _require(orientation_count >= rotation_count, f"rotation cap mismatch: {path}")
    _require(lane_count > 0 and lane_count % prefetch_fraction == 0, f"invalid CUDA lane topology: {path}")
    _require(lane_count // translation_count > 0, f"no active lane per translation: {path}")
    _require(eulers_per_block > 0, f"invalid Euler block topology: {path}")
    _require(header[21] > 0 and header[22] > 0 and header[23] > 0, f"invalid capture cap: {path}")
    _require(header[25] <= header[24], f"capture byte cap exceeded: {path}")
    _require(header[26] != 0 and header[27] != 0, f"identity hash missing: {path}")
    _require(header[28:31] == (1, 1, 1), f"passive lane-capture flags missing: {path}")
    expected_size = (
        HEADER_STRUCT.size
        + 2 * rotation_count * UINT64_DTYPE.itemsize
        + rotation_count * lane_count * FLOAT_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")
    offset = HEADER_STRUCT.size

    def take(dtype: np.dtype, count: int) -> np.ndarray:
        nonlocal offset
        values = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).copy()
        offset += count * dtype.itemsize
        return values

    rotation_keys = take(UINT64_DTYPE, rotation_count)
    local_rotation_indices = take(UINT64_DTYPE, rotation_count)
    lane_partials = take(FLOAT_DTYPE, rotation_count * lane_count).reshape(rotation_count, lane_count)
    footer_magic, footer_rotations, footer_lanes = FOOTER_STRUCT.unpack_from(payload, offset)
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(footer_rotations == rotation_count, f"footer rotation mismatch: {path}")
    _require(footer_lanes == lane_count, f"footer lane mismatch: {path}")
    assert match is not None
    _require(int(match["part"]) == header[6], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"stack identity mismatch: {path}")
    _require(np.unique(rotation_keys).size == rotation_count, f"duplicate rotation key: {path}")
    _require(np.unique(local_rotation_indices).size == rotation_count, f"duplicate local rotation index: {path}")
    _require(np.all(local_rotation_indices < orientation_count), f"local rotation index out of range: {path}")
    _require(np.all(np.isfinite(lane_partials)), f"non-finite lane partial: {path}")
    return CoarseLaneCapture(path, _sha256(path), header, rotation_keys, local_rotation_indices, lane_partials)


def possible_atomic_sums(
    values: np.ndarray,
    *,
    initial: np.float32 = np.float32(0.0),
) -> np.ndarray:
    """Return the unique binary32 results of all legal atomic-add orders."""

    operands = np.asarray(values, dtype=np.float32).reshape(-1)
    _require(operands.size <= 8, "refusing factorial enumeration of more than eight active lanes")
    outcomes: dict[int, np.float32] = {}
    for order in itertools.permutations(range(operands.size)):
        total = np.float32(initial)
        for index in order:
            total = np.float32(total + operands[index])
        bits = int(_float32_bits(np.asarray([total]))[0])
        outcomes[bits] = total
    return np.asarray(list(outcomes.values()), dtype=np.float32)


def validate_capture(
    lane: CoarseLaneCapture,
    operand: operand_validator.CoarseOperandCapture,
    component: component_validator.CoarsePass1Components,
) -> dict[str, object]:
    """Join one lane capture to its operand and production-score artifacts."""

    _require(lane.part_id == operand.part_id == component.part_id, "particle identities differ")
    _require(lane.stack_index == operand.stack_index == component.stack_index, "stack identities differ")
    _require(lane.header[5] == operand.header[5] == component.header[5], "iterations differ")
    _require(lane.header[8] == operand.header[8] == component.header[8], "MPI ranks differ")
    _require(lane.header[12] == operand.header[12] == component.header[27], "current sizes differ")
    _require(lane.header[13] == operand.header[13], "image sizes differ")
    _require(lane.header[14] == operand.header[14] == component.header[12], "translation counts differ")
    _require(lane.header[15] == operand.header[15], "rotation counts differ")
    _require(lane.header[16] == operand.header[16], "orientation counts differ")
    _require(lane.header[17] == operand.header[37], "CUDA block sizes differ")
    _require(lane.header[18] == operand.header[38], "CUDA prefetch factors differ")
    _require(lane.header[19] == operand.header[39], "Euler block counts differ")
    _require(np.array_equal(lane.rotation_keys, operand.rotation_keys), "rotation keys differ")
    _require(np.array_equal(lane.local_rotation_indices, operand.local_rotation_indices), "local rotations differ")
    _require(np.all(lane.rotation_keys < component.raw_diff2.shape[0]), "rotation key outside component table")
    initial_term = _float32_from_bits(lane.header[20])
    _require(np.isfinite(initial_term), "invalid recorded initial term")

    translation_count = lane.header[14]
    lane_count = lane.header[17]
    active_rows_per_translation = lane_count // translation_count
    active_mask = np.asarray(
        [thread // translation_count < active_rows_per_translation for thread in range(lane_count)],
        dtype=bool,
    )
    inactive_bits = _float32_bits(lane.lane_partials[:, ~active_mask])
    _require(np.all(inactive_bits == 0), "inactive CUDA lanes are not positive zero")

    modeled_lanes = operand_validator.replay_production_lanes(operand)
    _require(modeled_lanes.shape == lane.lane_partials.shape, "modeled lane shape differs")
    lane_exact = _float32_bits(modeled_lanes) == _float32_bits(lane.lane_partials)
    active_lane_exact = lane_exact[:, active_mask]
    active_lane_difference = np.abs(
        modeled_lanes[:, active_mask].astype(np.float64) - lane.lane_partials[:, active_mask].astype(np.float64)
    )

    target_scores = component.raw_diff2[lane.rotation_keys]
    _require(
        np.all(target_scores != component_validator.RELION_INVALID_DIFF2), "selected target includes inactive score"
    )
    reachable = np.zeros(target_scores.shape, dtype=bool)
    modeled_reachable = np.zeros(target_scores.shape, dtype=bool)
    outcome_counts = np.zeros(target_scores.shape, dtype=np.int64)
    envelope_low = np.empty(target_scores.shape, dtype=np.float32)
    envelope_high = np.empty(target_scores.shape, dtype=np.float32)
    for rotation in range(lane.rotation_keys.size):
        for translation in range(translation_count):
            thread_ids = translation + np.arange(active_rows_per_translation) * translation_count
            outcomes = possible_atomic_sums(
                lane.lane_partials[rotation, thread_ids],
                initial=initial_term,
            )
            outcome_counts[rotation, translation] = outcomes.size
            envelope_low[rotation, translation] = np.min(outcomes)
            envelope_high[rotation, translation] = np.max(outcomes)
            target_bits = _float32_bits(target_scores[rotation, translation : translation + 1])[0]
            reachable[rotation, translation] = bool(np.any(_float32_bits(outcomes) == target_bits))
            modeled_outcomes = possible_atomic_sums(
                modeled_lanes[rotation, thread_ids],
                initial=initial_term,
            )
            modeled_reachable[rotation, translation] = bool(np.any(_float32_bits(modeled_outcomes) == target_bits))

    candidate_count = int(reachable.size)
    reachable_count = int(np.count_nonzero(reachable))
    lane_value_count = int(lane_exact.size)
    lane_exact_count = int(np.count_nonzero(lane_exact))
    active_lane_value_count = int(active_lane_exact.size)
    active_lane_exact_count = int(np.count_nonzero(active_lane_exact))
    modeled_reachable_count = int(np.count_nonzero(modeled_reachable))
    capture_qualified = reachable_count == candidate_count
    operand_replay_qualified = (
        modeled_reachable_count == candidate_count and active_lane_exact_count == active_lane_value_count
    )
    status = "pass" if capture_qualified else "rejected"
    classification = (
        "native_atomic_reduction_exact_and_operand_replay_exact"
        if capture_qualified and operand_replay_qualified
        else "native_atomic_reduction_exact_but_passive_operand_replay_differs"
        if capture_qualified
        else "native_lane_capture_does_not_reproduce_production_scores"
    )
    return {
        "schema": "relion-coarse-lane-capture-validation-v1",
        "status": status,
        "classification_ready": capture_qualified,
        "classification": classification,
        "path": str(lane.path.resolve()),
        "sha256": lane.sha256,
        "part_id": lane.part_id,
        "stack_index_one_based": lane.stack_index,
        "rotation_count": int(lane.rotation_keys.size),
        "translation_count": translation_count,
        "lane_count": lane_count,
        "active_lanes_per_translation": active_rows_per_translation,
        "recorded_initial_term": float(initial_term),
        "capture_qualified": capture_qualified,
        "operand_replay_qualified": operand_replay_qualified,
        "fixed_metric": {
            "atomic_target_evaluated": candidate_count,
            "atomic_target_exactly_reachable": reachable_count,
            "atomic_target_exactly_reachable_fraction": reachable_count / candidate_count,
            "operand_lane_values_evaluated": lane_value_count,
            "operand_lane_values_bitwise_equal": lane_exact_count,
            "operand_lane_values_bitwise_equal_fraction": lane_exact_count / lane_value_count,
            "active_operand_lane_values_evaluated": active_lane_value_count,
            "active_operand_lane_values_bitwise_equal": active_lane_exact_count,
            "active_operand_lane_values_bitwise_equal_fraction": active_lane_exact_count / active_lane_value_count,
            "operand_atomic_target_exactly_reachable": modeled_reachable_count,
            "operand_atomic_target_exactly_reachable_fraction": modeled_reachable_count / candidate_count,
        },
        "active_operand_lane_abs_difference_p50": float(np.percentile(active_lane_difference, 50)),
        "active_operand_lane_abs_difference_p95": float(np.percentile(active_lane_difference, 95)),
        "active_operand_lane_abs_difference_max": float(np.max(active_lane_difference)),
        "atomic_outcome_count_min": int(np.min(outcome_counts)),
        "atomic_outcome_count_max": int(np.max(outcome_counts)),
        "atomic_envelope_width_max": float(np.max(envelope_high.astype(np.float64) - envelope_low.astype(np.float64))),
        "atomic_unreachable_indices": np.argwhere(~reachable).astype(int).tolist(),
        "operand_atomic_unreachable_indices": np.argwhere(~modeled_reachable).astype(int).tolist(),
        "operand_lane_first_mismatch": (
            None if lane_exact_count == lane_value_count else np.argwhere(~lane_exact)[0].astype(int).tolist()
        ),
    }


def validate_directory(directory: Path) -> tuple[tuple[CoarseLaneCapture, ...], dict[str, object]]:
    """Validate every joined lane/operand/component triple in a directory."""

    directory = Path(directory)
    _require(directory.is_dir(), f"capture directory does not exist: {directory}")
    _require(not list(directory.glob("*.tmp.*")), "incomplete capture artifact remains")
    lanes = tuple(load_artifact(path) for path in sorted(directory.glob("*.p1-lane-v1.bin")))
    _require(bool(lanes), f"no coarse-lane artifacts in {directory}")
    operands = {
        item.part_id: item
        for item in (operand_validator.load_artifact(path) for path in directory.glob("*.p1-op-v2.bin"))
    }
    components = {
        item.part_id: item
        for item in (component_validator.load_artifact(path) for path in directory.glob("*.p1-v2.bin"))
    }
    _require(len(operands) == len(lanes) == len(components), "lane/operand/component denominators differ")
    rows = []
    for lane in lanes:
        _require(lane.part_id in operands and lane.part_id in components, "joined particle identity missing")
        rows.append(validate_capture(lane, operands[lane.part_id], components[lane.part_id]))
    passed = sum(row["status"] == "pass" for row in rows)
    return lanes, {
        "schema": "relion-coarse-lane-capture-directory-validation-v1",
        "status": "pass" if passed == len(rows) else "rejected",
        "classification_ready": passed == len(rows),
        "capture_directory": str(directory.resolve()),
        "fixed_metric": {"evaluated_particles": len(rows), "passed_particles": passed},
        "particles": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    _, report = validate_directory(args.capture_directory)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(encoded)
    print(encoded, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
