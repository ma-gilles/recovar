#!/usr/bin/env python3
"""Fail-closed validation of passive RELION coarse-score operand captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts import validate_relion_coarse_pass1_components as component_validator

HEADER_MAGIC = b"RLNP1OPV2HEADER\0"
FOOTER_MAGIC = b"RLNP1OPV2FOOTER\0"
HEADER_STRUCT = struct.Struct("<16s40Q")
FOOTER_STRUCT = struct.Struct("<16sQQ")
FLOAT_DTYPE = np.dtype("<f4")
UINT64_DTYPE = np.dtype("<u8")
FILE_NAME = re.compile(r"part(?P<part>\d+)_stack(?P<stack>\d+)\.p1-op-v2\.bin")

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
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CoarseOperandCapture:
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


def load_artifact(path: Path) -> CoarseOperandCapture:
    """Load one complete passive coarse-operand artifact."""

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
    _require(header[0] == 2, f"schema mismatch: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"header size mismatch: {path}")
    _require(header[2] == FLOAT_DTYPE.itemsize, f"float size mismatch: {path}")
    _require(header[3] == UINT64_DTYPE.itemsize, f"integer size mismatch: {path}")
    _require(header[4] == FOOTER_STRUCT.size, f"footer size mismatch: {path}")
    _require(header[32:37] == (1, 1, 1, 1, 1), f"payload flags missing: {path}")

    image_size = header[13]
    translation_count = header[14]
    rotation_count = header[15]
    orientation_count = header[16]
    image_x, image_y, image_z = header[17:20]
    _require(header[5] > 0, f"iteration must be positive: {path}")
    _require(header[10] == 1 and header[11] == 0, f"expected K=1 image 0: {path}")
    _require(
        image_size > 0 and translation_count > 0 and rotation_count > 0,
        f"empty operand topology: {path}",
    )
    _require(orientation_count >= rotation_count, f"rotation cap mismatch: {path}")
    _require(image_z == 1, f"expected 2D particle image: {path}")
    _require(image_x * image_y == image_size, f"image topology mismatch: {path}")
    _require(header[20] < image_y, f"invalid Fourier maximum radius: {path}")
    _require(header[21] > 0 and header[22] > 0 and header[23] > 0, f"empty projector: {path}")
    _require(header[25] > 0 and header[26] > 0, f"invalid capture cap: {path}")
    _require(header[27] > 0, f"invalid follower count: {path}")
    _require(header[25] <= header[26] * header[27], f"capture cap too small: {path}")
    _require(header[29] <= header[28], f"capture byte cap exceeded: {path}")
    _require(header[30] != 0 and header[31] != 0, f"identity hash missing: {path}")
    block_size, prefetch_fraction, eulers_per_block = header[37:40]
    _require(
        block_size > 0 and prefetch_fraction > 0 and eulers_per_block > 0,
        f"invalid production CUDA topology: {path}",
    )
    _require(
        block_size % prefetch_fraction == 0,
        f"CUDA block/prefetch topology mismatch: {path}",
    )
    _require(
        block_size // translation_count > 0,
        f"CUDA block/translation topology mismatch: {path}",
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
    _require(len(payload) == expected_size, f"artifact byte count mismatch: {path}")
    offset = HEADER_STRUCT.size

    def take(dtype: np.dtype, count: int) -> np.ndarray:
        nonlocal offset
        values = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).copy()
        offset += count * dtype.itemsize
        return values

    rotation_keys = take(UINT64_DTYPE, rotation_count)
    local_rotation_indices = take(UINT64_DTYPE, rotation_count)
    euler_matrices = take(FLOAT_DTYPE, 9 * rotation_count).reshape(rotation_count, 3, 3)
    reference_real = take(FLOAT_DTYPE, rotation_count * image_size).reshape(rotation_count, image_size)
    reference_imag = take(FLOAT_DTYPE, rotation_count * image_size).reshape(rotation_count, image_size)
    image_real = take(FLOAT_DTYPE, image_size)
    image_imag = take(FLOAT_DTYPE, image_size)
    correction = take(FLOAT_DTYPE, image_size)
    translations = take(FLOAT_DTYPE, 3 * translation_count).reshape(3, translation_count)
    shifted_real = take(FLOAT_DTYPE, translation_count * image_size).reshape(translation_count, image_size)
    shifted_imag = take(FLOAT_DTYPE, translation_count * image_size).reshape(translation_count, image_size)
    footer_magic, footer_rotations, footer_pixels = FOOTER_STRUCT.unpack_from(payload, offset)
    _require(footer_magic == FOOTER_MAGIC, f"footer magic mismatch: {path}")
    _require(footer_rotations == rotation_count, f"footer rotation mismatch: {path}")
    _require(footer_pixels == image_size, f"footer pixel mismatch: {path}")

    assert match is not None
    _require(int(match["part"]) == header[6], f"part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"stack identity mismatch: {path}")
    _require(
        np.unique(rotation_keys).size == rotation_count,
        f"duplicate rotation key: {path}",
    )
    _require(
        np.unique(local_rotation_indices).size == rotation_count,
        f"duplicate local rotation index: {path}",
    )
    _require(
        np.all(local_rotation_indices < orientation_count),
        f"local rotation index out of range: {path}",
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
    return CoarseOperandCapture(
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


def replay_components(artifact: CoarseOperandCapture) -> tuple[np.ndarray, np.ndarray]:
    """Replay FP64 diagnostic components from the captured production operands."""

    image_size = artifact.header[13]
    image_x, image_y = artifact.header[17:19]
    max_radius = artifact.header[20]
    pixels = np.arange(image_size, dtype=np.int64)
    x = (pixels % image_x).astype(np.float32)
    y_int = pixels // image_x
    y_int = np.where(y_int > max_radius, y_int - image_y, y_int)
    y = y_int.astype(np.float32)
    tx, ty = artifact.translations[:2]
    phase = (tx[:, np.newaxis] * x[np.newaxis, :] + ty[:, np.newaxis] * y[np.newaxis, :]).astype(np.float32)
    sine = np.sin(phase).astype(np.float32)
    cosine = np.cos(phase).astype(np.float32)
    shifted_real = (cosine * artifact.image_real[np.newaxis, :] - sine * artifact.image_imag[np.newaxis, :]).astype(
        np.float32
    )
    shifted_imag = (cosine * artifact.image_imag[np.newaxis, :] + sine * artifact.image_real[np.newaxis, :]).astype(
        np.float32
    )

    correction = (artifact.correction / np.float32(2.0)).astype(np.float64)
    reference_real = artifact.reference_real.astype(np.float64)
    reference_imag = artifact.reference_imag.astype(np.float64)
    reference_norm = np.sum(
        (reference_real * reference_real + reference_imag * reference_imag) * correction[np.newaxis, :],
        axis=1,
        dtype=np.float64,
    )
    cross = -2.0 * np.sum(
        (
            reference_real[:, np.newaxis, :] * shifted_real[np.newaxis, :, :].astype(np.float64)
            + reference_imag[:, np.newaxis, :] * shifted_imag[np.newaxis, :, :].astype(np.float64)
        )
        * correction[np.newaxis, np.newaxis, :],
        axis=2,
        dtype=np.float64,
    )
    return reference_norm, cross


def _fma_float32(
    multiplicand: np.ndarray,
    multiplier: np.ndarray | np.float32,
    addend: np.ndarray,
) -> np.ndarray:
    """Emulate a float32 fused multiply-add using exact float32 operands."""

    return (
        multiplicand.astype(np.float64) * np.asarray(multiplier, dtype=np.float64) + addend.astype(np.float64)
    ).astype(np.float32)


def replay_production_lanes(artifact: CoarseOperandCapture) -> np.ndarray:
    """Replay each CUDA coarse thread's pre-atomic float32 partial."""

    rotation_count = artifact.rotation_keys.size
    image_size = artifact.header[13]
    translation_count = artifact.header[14]
    block_size = artifact.header[37]
    prefetch_fraction = artifact.header[38]
    pixels_per_pass = block_size // prefetch_fraction
    thread_stride = block_size // translation_count
    max_pixel = ((image_size + block_size - 1) // block_size) * block_size
    correction = (artifact.correction / np.float32(2.0)).astype(np.float32)
    partials = np.zeros((block_size, rotation_count), dtype=np.float32)

    for thread in range(block_size):
        translation = thread % translation_count
        first_pixel_in_pass = thread // translation_count
        if first_pixel_in_pass >= thread_stride:
            continue
        accumulator = np.zeros(rotation_count, dtype=np.float32)
        for init_pixel in range(0, max_pixel, pixels_per_pass):
            for pixel_in_pass in range(
                first_pixel_in_pass,
                pixels_per_pass,
                thread_stride,
            ):
                pixel = init_pixel + pixel_in_pass
                if pixel >= image_size:
                    break
                diff_real = (artifact.reference_real[:, pixel] - artifact.shifted_real[translation, pixel]).astype(
                    np.float32
                )
                diff_imag = (artifact.reference_imag[:, pixel] - artifact.shifted_imag[translation, pixel]).astype(
                    np.float32
                )
                squared = _fma_float32(
                    diff_real,
                    diff_real,
                    (diff_imag * diff_imag).astype(np.float32),
                )
                accumulator = _fma_float32(
                    squared,
                    correction[pixel],
                    accumulator,
                )
        partials[thread] = accumulator

    return partials.T.copy()


def replay_production_diff2(artifact: CoarseOperandCapture) -> np.ndarray:
    """Replay the CUDA coarse squared-difference path for captured rotations."""

    translation_count = artifact.header[14]
    block_size = artifact.header[37]
    partials = replay_production_lanes(artifact).T
    scores = np.zeros((artifact.rotation_keys.size, translation_count), dtype=np.float32)
    for thread in range(block_size):
        translation = thread % translation_count
        scores[:, translation] = (scores[:, translation] + partials[thread]).astype(np.float32)
    return scores


def validate_directory(
    directory: Path,
    *,
    expected_particles: int | None = None,
    expected_stack_indices: np.ndarray | None = None,
    expected_mpi_rank: int | None = None,
    reference_replay_max_abs: float = DEFAULT_REFERENCE_REPLAY_MAX_ABS,
    cross_replay_p95_abs: float = DEFAULT_CROSS_REPLAY_P95_ABS,
    cross_replay_max_abs: float = DEFAULT_CROSS_REPLAY_MAX_ABS,
    production_replay_p95_abs: float = DEFAULT_PRODUCTION_REPLAY_P95_ABS,
    production_replay_max_abs: float = DEFAULT_PRODUCTION_REPLAY_MAX_ABS,
) -> tuple[tuple[CoarseOperandCapture, ...], dict[str, object]]:
    """Validate operands and replay the original production diff2 arithmetic."""

    directory = Path(directory)
    _require(directory.is_dir(), f"capture directory does not exist: {directory}")
    _require(not list(directory.glob("*.tmp.*")), "incomplete capture artifact remains")
    paths = sorted(directory.glob("*.p1-op-v2.bin"))
    _require(bool(paths), f"no coarse-operand artifacts in {directory}")
    artifacts = tuple(load_artifact(path) for path in paths)
    components, component_report = component_validator.validate_directory(
        directory,
        expected_particles=expected_particles,
        expected_stack_indices=expected_stack_indices,
        expected_mpi_rank=expected_mpi_rank,
    )
    components_by_part = {item.part_id: item for item in components}
    reference = artifacts[0].header
    common_fields = (
        0,
        1,
        2,
        3,
        4,
        5,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
    )
    for artifact in artifacts[1:]:
        for field in common_fields:
            _require(
                artifact.header[field] == reference[field],
                f"inconsistent operand header field {field}: {artifact.path}",
            )
        _require(
            np.array_equal(artifact.rotation_keys, artifacts[0].rotation_keys),
            f"inconsistent operand rotation keys: {artifact.path}",
        )
        _require(
            np.array_equal(
                artifact.local_rotation_indices,
                artifacts[0].local_rotation_indices,
            ),
            f"inconsistent local rotation indices: {artifact.path}",
        )
        for name, values, reference_values in (
            ("Euler matrices", artifact.euler_matrices, artifacts[0].euler_matrices),
            ("reference real", artifact.reference_real, artifacts[0].reference_real),
            ("reference imaginary", artifact.reference_imag, artifacts[0].reference_imag),
        ):
            _require(
                np.array_equal(values, reference_values),
                f"inconsistent {name}: {artifact.path}",
            )
    if expected_particles is not None:
        _require(reference[25] == expected_particles, "expected-particle mismatch")
    _require(len(artifacts) == reference[25], "operand capture completeness mismatch")
    _require(
        {item.part_id for item in artifacts} == set(components_by_part),
        "operand/component particle identity sets differ",
    )

    metrics: dict[str, dict[str, object]] = {}
    reference_passed = 0
    cross_p95_passed = 0
    cross_max_passed = 0
    production_p95_passed = 0
    production_max_passed = 0
    for artifact in artifacts:
        component = components_by_part[artifact.part_id]
        _require(
            artifact.stack_index == component.stack_index,
            f"operand/component stack mismatch: {artifact.path}",
        )
        _require(
            artifact.mpi_rank == component.mpi_rank,
            f"operand/component rank mismatch: {artifact.path}",
        )
        _require(
            artifact.header[12] == component.header[27],
            f"operand/component current-size mismatch: {artifact.path}",
        )
        _require(
            artifact.header[14] == component.header[12],
            f"operand/component translation-count mismatch: {artifact.path}",
        )
        _require(
            np.all(artifact.rotation_keys < component.reference_norms.shape[0]),
            f"rotation key outside component table: {artifact.path}",
        )
        replay_reference, replay_cross = replay_components(artifact)
        replay_diff2 = replay_production_diff2(artifact)
        target_reference = component.reference_norms[artifact.rotation_keys].astype(np.float64)
        target_cross = component.cross_terms[artifact.rotation_keys].astype(np.float64)
        target_diff2 = component.raw_diff2[artifact.rotation_keys]
        _require(
            np.all(target_diff2 != component_validator.RELION_INVALID_DIFF2),
            f"captured operand rotation is inactive: {artifact.path}",
        )
        reference_error = np.abs(replay_reference[:, np.newaxis] - target_reference)
        cross_error = np.abs(replay_cross - target_cross)
        reference_max = float(np.max(reference_error))
        cross_p95 = float(np.percentile(cross_error, 95))
        cross_max = float(np.max(cross_error))
        production_difference = replay_diff2.astype(np.float64) - target_diff2.astype(np.float64)
        production_constant = float(np.median(production_difference))
        production_error = np.abs(production_difference - production_constant)
        production_p95 = float(np.percentile(production_error, 95))
        production_max = float(np.max(production_error))
        reference_passed += reference_max <= reference_replay_max_abs
        cross_p95_passed += cross_p95 <= cross_replay_p95_abs
        cross_max_passed += cross_max <= cross_replay_max_abs
        production_p95_passed += production_p95 <= production_replay_p95_abs
        production_max_passed += production_max <= production_replay_max_abs
        metrics[artifact.path.name] = {
            "rotation_count": int(artifact.rotation_keys.size),
            "translation_count": artifact.header[14],
            "reference_replay_max_abs": reference_max,
            "cross_replay_p95_abs": cross_p95,
            "cross_replay_max_abs": cross_max,
            "production_diff2_additive_constant_median": production_constant,
            "production_diff2_centered_replay_p95_abs": production_p95,
            "production_diff2_centered_replay_max_abs": production_max,
        }

    qualified = (
        reference_passed
        == cross_p95_passed
        == cross_max_passed
        == production_p95_passed
        == production_max_passed
        == len(artifacts)
    )
    summary: dict[str, object] = {
        "schema": "relion-coarse-operand-capture-v2",
        "capture_directory": str(directory.resolve()),
        "particle_count": len(artifacts),
        "rotation_count_per_particle": reference[15],
        "translation_count": reference[14],
        "fixed_gates": {
            "reference_replay_max_abs": reference_replay_max_abs,
            "cross_replay_p95_abs": cross_replay_p95_abs,
            "cross_replay_max_abs": cross_replay_max_abs,
            "production_diff2_centered_replay_p95_abs": production_replay_p95_abs,
            "production_diff2_centered_replay_max_abs": production_replay_max_abs,
        },
        "fixed_metric": {
            "evaluated_particles": len(artifacts),
            "expected_particles": reference[25],
            "reference_replay_passed": reference_passed,
            "cross_replay_p95_passed": cross_p95_passed,
            "cross_replay_max_passed": cross_max_passed,
            "production_diff2_centered_replay_p95_passed": production_p95_passed,
            "production_diff2_centered_replay_max_passed": production_max_passed,
        },
        "paired_component_status": component_report["status"],
        "operand_metrics": metrics,
        "artifact_sha256": {item.path.name: item.sha256 for item in artifacts},
        "classification_ready": qualified,
        "status": "pass" if qualified else "rejected",
    }
    return artifacts, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--expected-particles", type=int)
    parser.add_argument("--expected-stack-indices-npy", type=Path)
    parser.add_argument("--expected-mpi-rank", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    expected_stacks = (
        None
        if args.expected_stack_indices_npy is None
        else np.load(args.expected_stack_indices_npy, allow_pickle=False)
    )
    _, summary = validate_directory(
        args.capture_directory,
        expected_particles=args.expected_particles,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=args.expected_mpi_rank,
    )
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(encoded)
    print(encoded, end="")
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
