#!/usr/bin/env python3
"""Fail-closed validation of class-resolved RELION coarse operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scripts import validate_relion_coarse_component_capture as component_validator
    from scripts import validate_relion_coarse_operand_capture as replay
    from scripts import validate_relion_coarse_score_capture as score_validator
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    import validate_relion_coarse_component_capture as component_validator
    import validate_relion_coarse_operand_capture as replay
    import validate_relion_coarse_score_capture as score_validator

HEADER_MAGIC = b"RLNCROP1HEADER".ljust(16, b"\0")
FOOTER_MAGIC = b"RLNCROP1FOOTER".ljust(16, b"\0")
HEADER_STRUCT = struct.Struct("<16s64Q")
FOOTER_STRUCT = struct.Struct("<16s3Q")
FLOAT_DTYPE = np.dtype("<f4")
UINT64_DTYPE = np.dtype("<u8")
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)_class(?P<class>\d+)"
    r"\.coarse-operands-v1\.bin"
)

DEFAULT_REFERENCE_REPLAY_MAX_ABS = 5.0e-5
DEFAULT_CROSS_REPLAY_P95_ABS = 5.0e-5
DEFAULT_CROSS_REPLAY_MAX_ABS = 5.0e-4
DEFAULT_PRODUCTION_REPLAY_P95_ABS = 5.0e-5
DEFAULT_PRODUCTION_REPLAY_MAX_ABS = 5.0e-4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class K4CoarseOperandCapture:
    """One particle/class operand panel captured after production scoring."""

    path: Path
    sha256: str
    header: tuple[int, ...]
    rotation_keys: np.ndarray
    local_rotation_indices: np.ndarray
    euler_matrices: np.ndarray
    reference_real: np.ndarray
    reference_imag: np.ndarray
    image_real: np.ndarray
    image_imag: np.ndarray
    correction: np.ndarray
    translations: np.ndarray
    shifted_real: np.ndarray
    shifted_imag: np.ndarray

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
    def class_one_based(self) -> int:
        return self.header[10]


def load_artifact(path: Path) -> K4CoarseOperandCapture:
    """Load one complete class-resolved coarse-operand artifact."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected coarse-operand file name: {path.name}")
    payload = path.read_bytes()
    _require(
        len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size,
        f"truncated coarse-operand capture: {path}",
    )
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"coarse-operand header magic mismatch: {path}")
    _require(
        header[:5]
        == (
            1,
            HEADER_STRUCT.size,
            FLOAT_DTYPE.itemsize,
            UINT64_DTYPE.itemsize,
            FOOTER_STRUCT.size,
        ),
        f"coarse-operand schema/record sizes changed: {path}",
    )

    image_size = header[13]
    translation_count = header[14]
    rotation_count = header[15]
    orientation_count = header[16]
    image_x, image_y, image_z = header[17:20]
    _require(
        header[5] > 0
        and header[10] > 0
        and header[11] == 0
        and image_size > 0
        and translation_count > 0
        and rotation_count > 0,
        f"invalid coarse-operand runtime dimensions: {path}",
    )
    _require(
        orientation_count == rotation_count,
        f"coarse-operand rotation count changed during capture: {path}",
    )
    _require(image_z == 1, f"expected 2D particle image: {path}")
    _require(image_x * image_y == image_size, f"image topology mismatch: {path}")
    _require(header[20] < image_y, f"invalid Fourier maximum radius: {path}")
    _require(
        header[21] > 0 and header[22] > 0 and header[23] > 0,
        f"empty projector: {path}",
    )
    _require(
        header[25] > 0 and header[26] > 0 and header[27] > 0,
        f"invalid coarse-operand capture cap: {path}",
    )
    _require(
        header[29] and header[30] and header[31],
        f"coarse-operand identity hash is zero: {path}",
    )
    _require(
        header[32:38] == (1, 1, 1, 1, 1, 1),
        f"coarse-operand capture is not passive/complete: {path}",
    )
    block_size, prefetch_fraction, eulers_per_block = header[38:41]
    _require(
        block_size > 0
        and prefetch_fraction > 0
        and eulers_per_block > 0
        and header[41] >= header[10],
        f"invalid coarse CUDA/class topology: {path}",
    )
    _require(
        block_size % prefetch_fraction == 0
        and block_size // translation_count > 0,
        f"coarse CUDA translation topology mismatch: {path}",
    )

    expected_size = (
        HEADER_STRUCT.size
        + 2 * rotation_count * UINT64_DTYPE.itemsize
        + 9 * rotation_count * FLOAT_DTYPE.itemsize
        + 2 * rotation_count * image_size * FLOAT_DTYPE.itemsize
        + 3 * image_size * FLOAT_DTYPE.itemsize
        + 3 * translation_count * FLOAT_DTYPE.itemsize
        + 2 * translation_count * image_size * FLOAT_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(len(payload) == expected_size, f"coarse-operand byte count mismatch: {path}")
    _require(header[28] == expected_size, f"coarse-operand byte estimate changed: {path}")
    _require(
        expected_size <= header[27]
        and header[26] <= header[27] // expected_size,
        f"coarse-operand byte cap exceeded: {path}",
    )
    offset = HEADER_STRUCT.size

    def take(dtype: np.dtype, count: int) -> np.ndarray:
        nonlocal offset
        values = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).copy()
        offset += count * dtype.itemsize
        return values

    rotation_keys = take(UINT64_DTYPE, rotation_count)
    local_rotation_indices = take(UINT64_DTYPE, rotation_count)
    euler_matrices = take(FLOAT_DTYPE, 9 * rotation_count).reshape(
        rotation_count, 3, 3
    )
    reference_real = take(FLOAT_DTYPE, rotation_count * image_size).reshape(
        rotation_count, image_size
    )
    reference_imag = take(FLOAT_DTYPE, rotation_count * image_size).reshape(
        rotation_count, image_size
    )
    image_real = take(FLOAT_DTYPE, image_size)
    image_imag = take(FLOAT_DTYPE, image_size)
    correction = take(FLOAT_DTYPE, image_size)
    translations = take(FLOAT_DTYPE, 3 * translation_count).reshape(
        3, translation_count
    )
    shifted_real = take(FLOAT_DTYPE, translation_count * image_size).reshape(
        translation_count, image_size
    )
    shifted_imag = take(FLOAT_DTYPE, translation_count * image_size).reshape(
        translation_count, image_size
    )
    footer_magic, footer_rotations, footer_translations, footer_pixels = (
        FOOTER_STRUCT.unpack_from(payload, offset)
    )
    _require(footer_magic == FOOTER_MAGIC, f"coarse-operand footer magic mismatch: {path}")
    _require(
        (footer_rotations, footer_translations, footer_pixels)
        == (rotation_count, translation_count, image_size),
        f"coarse-operand footer dimensions changed: {path}",
    )

    assert match is not None
    _require(int(match["part"]) == header[6], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"stack identity mismatch: {path}")
    _require(int(match["class"]) == header[10], f"class identity mismatch: {path}")
    _require(
        np.unique(rotation_keys).size == rotation_count,
        f"duplicate rotation key: {path}",
    )
    _require(
        np.array_equal(
            local_rotation_indices, np.arange(rotation_count, dtype=np.uint64)
        ),
        f"local rotation order changed: {path}",
    )
    for name, values in (
        ("Euler matrix", euler_matrices),
        ("reference real", reference_real),
        ("reference imaginary", reference_imag),
        ("image real", image_real),
        ("image imaginary", image_imag),
        ("correction", correction),
        ("translations", translations),
        ("shifted real", shifted_real),
        ("shifted imaginary", shifted_imag),
    ):
        _require(np.all(np.isfinite(values)), f"non-finite {name}: {path}")
    _require(np.all(correction >= 0), f"negative correction: {path}")
    return K4CoarseOperandCapture(
        path=path,
        sha256=_sha256(path),
        header=header,
        rotation_keys=rotation_keys,
        local_rotation_indices=local_rotation_indices,
        euler_matrices=euler_matrices,
        reference_real=reference_real,
        reference_imag=reference_imag,
        image_real=image_real,
        image_imag=image_imag,
        correction=correction,
        translations=translations,
        shifted_real=shifted_real,
        shifted_imag=shifted_imag,
    )


def _paired_candidates(
    artifact: K4CoarseOperandCapture,
    component: component_validator.CoarseComponentCapture,
    score: score_validator.CoarseScoreCapture,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return component and production-score panels for captured rotations."""

    nr_trans = component.header[13]
    _require(nr_trans == artifact.header[14], "operand/component translation counts differ")
    _require(component.header[4] == artifact.header[5], "operand/component iterations differ")
    _require(component.header[5] == artifact.part_id, "operand/component part IDs differ")
    _require(component.header[6] == artifact.stack_index, "operand/component stack IDs differ")
    _require(component.header[7] == artifact.mpi_rank, "operand/component MPI ranks differ")
    _require(component.header[10] == artifact.header[41], "operand/component class counts differ")
    _require(component.header[15] == artifact.header[29], "operand/component stack hashes differ")
    _require(component.header[16] == artifact.header[30], "operand/component image hashes differ")
    _require(component.header[17] == artifact.header[31], "operand/component panel hashes differ")
    _require(component.header[18] == artifact.header[25], "operand/component particle counts differ")
    _require(score.header[5] == artifact.part_id, "operand/score part IDs differ")
    _require(score.header[6] == artifact.stack_index, "operand/score stack IDs differ")
    orientation_count = component.header[10] * component.header[11] * component.header[12]
    _require(
        np.all(artifact.rotation_keys < orientation_count),
        "coarse-operand rotation key lies outside component topology",
    )
    class_from_key = (
        artifact.rotation_keys // (component.header[11] * component.header[12]) + 1
    )
    _require(
        np.all(class_from_key == artifact.class_one_based),
        "coarse-operand rotation key belongs to another class",
    )
    flat = (
        artifact.rotation_keys[:, np.newaxis] * nr_trans
        + np.arange(nr_trans, dtype=np.uint64)[np.newaxis, :]
    )
    return (
        component.candidates["reference_norm"][flat],
        component.candidates["cross_term"][flat],
        score.candidates["raw_diff2"][flat],
    )


def _as_replay_capture(
    artifact: K4CoarseOperandCapture,
) -> replay.CoarseOperandCapture:
    """Adapt schema-v1 field locations to the shared arithmetic replayer."""

    header = list(artifact.header)
    header[37:40] = artifact.header[38:41]
    return replay.CoarseOperandCapture(
        path=artifact.path,
        sha256=artifact.sha256,
        header=tuple(header),
        rotation_keys=artifact.rotation_keys,
        local_rotation_indices=artifact.local_rotation_indices,
        euler_matrices=artifact.euler_matrices,
        reference_real=artifact.reference_real,
        reference_imag=artifact.reference_imag,
        image_real=artifact.image_real,
        image_imag=artifact.image_imag,
        correction=artifact.correction,
        translations=artifact.translations,
        shifted_real=artifact.shifted_real,
        shifted_imag=artifact.shifted_imag,
    )


def validate_directory(
    directory: Path,
    expected_stacks: tuple[int, ...],
    *,
    expected_iteration: int | None = None,
    reference_replay_max_abs: float = DEFAULT_REFERENCE_REPLAY_MAX_ABS,
    cross_replay_p95_abs: float = DEFAULT_CROSS_REPLAY_P95_ABS,
    cross_replay_max_abs: float = DEFAULT_CROSS_REPLAY_MAX_ABS,
    production_replay_p95_abs: float = DEFAULT_PRODUCTION_REPLAY_P95_ABS,
    production_replay_max_abs: float = DEFAULT_PRODUCTION_REPLAY_MAX_ABS,
) -> dict[str, object]:
    """Validate a complete particle-by-class operand panel and replay it."""

    directory = Path(directory)
    _require(directory.is_dir(), f"capture directory does not exist: {directory}")
    _require(not list(directory.glob("*.tmp.*")), "incomplete capture artifact remains")
    paths = sorted(directory.glob("*.coarse-operands-v1.bin"))
    _require(paths, f"no class-resolved coarse operands in {directory}")
    artifacts = tuple(load_artifact(path) for path in paths)
    component_report = component_validator.validate_directory(
        directory,
        expected_stacks,
        expected_iteration=expected_iteration,
    )
    classes = artifacts[0].header[41]
    expected_keys = {
        (stack, class_one_based)
        for stack in expected_stacks
        for class_one_based in range(1, classes + 1)
    }
    _require(
        len(artifacts) == len(expected_keys)
        and {(item.stack_index, item.class_one_based) for item in artifacts}
        == expected_keys,
        "coarse-operand particle/class panel is incomplete or duplicated",
    )
    _require(
        all(item.header[25] == len(expected_stacks) for item in artifacts),
        "coarse-operand expected-particle count changed",
    )
    if expected_iteration is not None:
        _require(
            all(item.header[5] == expected_iteration for item in artifacts),
            "coarse-operand iteration differs from expectation",
        )

    component_by_part: dict[int, component_validator.CoarseComponentCapture] = {}
    score_by_part: dict[int, score_validator.CoarseScoreCapture] = {}
    for artifact in artifacts:
        if artifact.part_id in component_by_part:
            continue
        stem = f"part{artifact.part_id}_stack{artifact.stack_index}"
        component = component_validator.load_coarse_component_capture(
            directory / f"{stem}.coarse-components-v1.bin"
        )
        score = score_validator.load_coarse_score_capture(
            directory / f"{stem}.coarse-score-v1.bin"
        )
        component_by_part[artifact.part_id] = component
        score_by_part[artifact.part_id] = score

    class_reference: dict[int, K4CoarseOperandCapture] = {}
    metrics: dict[str, dict[str, float | int]] = {}
    passed = 0
    for artifact in artifacts:
        class_reference.setdefault(artifact.class_one_based, artifact)
        reference = class_reference[artifact.class_one_based]
        for label, values, expected in (
            ("rotation keys", artifact.rotation_keys, reference.rotation_keys),
            ("Euler matrices", artifact.euler_matrices, reference.euler_matrices),
            ("reference real", artifact.reference_real, reference.reference_real),
            ("reference imaginary", artifact.reference_imag, reference.reference_imag),
        ):
            _require(
                np.array_equal(values, expected),
                f"class {artifact.class_one_based} {label} changed across particles",
            )
        component = component_by_part[artifact.part_id]
        score = score_by_part[artifact.part_id]
        target_reference, target_cross, target_diff2 = _paired_candidates(
            artifact, component, score
        )
        replay_artifact = _as_replay_capture(artifact)
        replay_reference, replay_cross = replay.replay_components(replay_artifact)
        replay_diff2 = replay.replay_production_diff2(replay_artifact)
        reference_error = np.abs(
            replay_reference[:, np.newaxis] - target_reference.astype(np.float64)
        )
        cross_error = np.abs(replay_cross - target_cross.astype(np.float64))
        production_difference = (
            replay_diff2.astype(np.float64) - target_diff2.astype(np.float64)
        )
        production_constant = float(np.median(production_difference))
        production_error = np.abs(production_difference - production_constant)
        values = {
            "rotation_count": int(artifact.rotation_keys.size),
            "translation_count": artifact.header[14],
            "reference_replay_max_abs": float(np.max(reference_error)),
            "cross_replay_p95_abs": float(np.percentile(cross_error, 95)),
            "cross_replay_max_abs": float(np.max(cross_error)),
            "production_diff2_additive_constant_median": production_constant,
            "production_diff2_centered_replay_p95_abs": float(
                np.percentile(production_error, 95)
            ),
            "production_diff2_centered_replay_max_abs": float(
                np.max(production_error)
            ),
        }
        artifact_passed = (
            values["reference_replay_max_abs"] <= reference_replay_max_abs
            and values["cross_replay_p95_abs"] <= cross_replay_p95_abs
            and values["cross_replay_max_abs"] <= cross_replay_max_abs
            and values["production_diff2_centered_replay_p95_abs"]
            <= production_replay_p95_abs
            and values["production_diff2_centered_replay_max_abs"]
            <= production_replay_max_abs
        )
        passed += artifact_passed
        values["passed"] = int(artifact_passed)
        metrics[artifact.path.name] = values

    qualified = passed == len(artifacts)
    return {
        "schema": "relion-k4-coarse-operand-validation-v1",
        "capture_ready": qualified,
        "directory": str(directory.resolve()),
        "iteration": artifacts[0].header[5],
        "particle_count": len(expected_stacks),
        "class_count": classes,
        "artifact_count": len(artifacts),
        "fixed_gates": {
            "reference_replay_max_abs": reference_replay_max_abs,
            "cross_replay_p95_abs": cross_replay_p95_abs,
            "cross_replay_max_abs": cross_replay_max_abs,
            "production_diff2_centered_replay_p95_abs": production_replay_p95_abs,
            "production_diff2_centered_replay_max_abs": production_replay_max_abs,
        },
        "fixed_metric": {
            "evaluated_artifacts": len(artifacts),
            "passed_artifacts": passed,
        },
        "paired_component_validation": component_report,
        "operand_metrics": metrics,
        "artifact_sha256": {item.path.name: item.sha256 for item in artifacts},
        "status": "pass" if qualified else "rejected",
    }


def _parse_stacks(text: str) -> tuple[int, ...]:
    values = tuple(int(token) for token in text.split(",") if token)
    _require(
        values and len(values) == len(set(values)) and min(values) > 0,
        "invalid expected stacks",
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--expected-stacks", required=True)
    parser.add_argument("--expected-iteration", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_directory(
        args.capture_dir,
        _parse_stacks(args.expected_stacks),
        expected_iteration=args.expected_iteration,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
