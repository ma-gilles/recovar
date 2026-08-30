#!/usr/bin/env python3
"""Prepare sealed matched inputs for EMPIAR-10202 set-6 K=1/I1 refinement.

This script does input preparation only.  It preserves the deposited particle
order, poses, shifts, CTF parameters, and random-half labels; neither engine is
run in fixed-pose, replay, or local-only mode.  A sealed ``relion_convert_star``
converts the audited legacy STAR into the authoritative RELION 3.1+ optics-
group STAR.  The two output reference files encode the same float32 array in
RECOVAR's and RELION's respective volume frames.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import mrcfile
import numpy as np
import pandas as pd

from recovar.data_io.starfile import read_star
from recovar.utils import helpers

SCHEMA = "recovar.empiar10202_set6_i1_preparation.v1"
DATASET_ID = "EMPIAR-10202"
IMAGE_SET = 6
PARTICLE_COUNT = 30_515
BOX_SIZE = 800
VOXEL_SIZE_ANGSTROM = 0.788
HALF_COUNTS = {1: 15_258, 2: 15_257}
SYMMETRY = "I1"
SYMMETRY_OPERATOR_COUNT = 60
LOWPASS_ANGSTROM = 30.0

SOURCE_ROOT = Path("/home/mg6942/mytigress/10202/06_Final_Stack")
SOURCE_STAR = SOURCE_ROOT / "2017-12-27_MagCorrect_Frames05-19_Numbered_adjusted.star"
PARTICLE_STACK = SOURCE_ROOT / "2017-12-27_MagCorrect_Frames05-19.mrcs"
POSES_PKL = SOURCE_ROOT / "poses.pkl"
CTF_PKL = SOURCE_ROOT / "ctf.pkl"
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_20260830"
)
DEFAULT_RELION_IMAGE_HANDLER = Path(
    "/projects/MOLBIO/local/relion-5.0.1-gcc-11.5.0-cuda-12.6-rhel9-arch80/bin/relion_image_handler"
)
DEFAULT_RELION_CONVERT_STAR = Path(
    "/projects/MOLBIO/local/relion-5.0.1-gcc-11.5.0-cuda-12.6-rhel9-arch80/bin/relion_convert_star"
)

SOURCE_SHA256 = {
    "star": "5b89aa3f2f4c66c40820952982f67dd0ba3d46864c65b5eb35e16b1fe4d11cdb",
    "particle_stack": "8eecf0fbf8e645ac51feff278a86e43e7e4be117921333dc6d3e22e52a628453",
    "particle_stack_header_1mib": "9fc0bdfb6369ba423f519d11fb4bfed8f3dce353903fd667ce3fb1a0ff5cbe32",
    "poses_pkl": "f52394a1ea41f442fdcbf37e4808aa878f52ab033f875aa81d623c5bc9a2b238",
    "ctf_pkl": "62ad1941816e33668051230faa30f970630de4e57508a943f3dd962f87e2fcbb",
    "half_assignment": "1a06a8e0fccc553a99ecbdd24b21115b2b20b14cd1ae267310c6d4fe449d054f",
}
NORMALIZED_LEGACY_STAR_SHA256 = "6c5143a5c4f64bd04c51f44f741abd5cfcde7082ad537e256e45c678fe9e2c2f"
RAW_CONVERTED_STAR_SHA256 = "d8693e4fb788181bf0f3a3b1e4dfe98470b5f83db4f65978f6700783db12dd09"
RAW_CONVERTED_STAR_SIZE_BYTES = 12_440_248
PREPARED_STAR_SHA256 = "d66afb3001e6e43463fb699804fe8b50f8cef1f2ccd9730f5275955b6be7b512"
PREPARED_STAR_SIZE_BYTES = 10_313_843
PARTICLE_STACK_SIZE_BYTES = 78_118_401_024
RELION_IMAGE_HANDLER_SHA256 = "fc5061d05f7c7089351e0feb5dbb1192aa91aa34e0b941d0e5e94885c9fa7df3"
RELION_CONVERT_STAR_SHA256 = "dc6e1128c1c793d0cd1f6de5380aed1145fd48f9b0d5de3ec94f611dfe05bb4a"
ORIGIN_ROUNDTRIP_ATOL_PIXELS = 5e-7
RELION_GROUP_COUNT = 176

EMD_HALF_MAPS = (
    {
        "name": "emd_9012_half_map_1.map",
        "url": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-9012/other/emd_9012_half_map_1.map.gz",
        "gzip_size_bytes": 98_540_753,
        "gzip_sha256": "0422a5d254427d949c81443fd30e73481c386141cc345649734959344c0e492f",
        "map_size_bytes": 186_625_024,
        "map_sha256": "d4da1010f0fde019d667beff5beb6d87cc45d4e3e209aed4468af3039a5e5e3e",
    },
    {
        "name": "emd_9012_half_map_2.map",
        "url": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-9012/other/emd_9012_half_map_2.map.gz",
        "gzip_size_bytes": 98_527_065,
        "gzip_sha256": "0432e925189d03c393714436a632adcad66784f78165d99a2e562244424f4e48",
        "map_size_bytes": 186_625_024,
        "map_sha256": "32aaaf2b45a96d14e521c873c772161f7315f8ac5533c18477c0e3eacf3d021a",
    },
)

POSE_SHIFT_COLUMNS = (
    "_rlnAngleRot",
    "_rlnAngleTilt",
    "_rlnAnglePsi",
    "_rlnOriginX",
    "_rlnOriginY",
)
CTF_COLUMNS = (
    "_rlnVoltage",
    "_rlnDefocusAngle",
    "_rlnSphericalAberration",
    "_rlnDetectorPixelSize",
    "_rlnDefocusU",
    "_rlnDefocusV",
    "_rlnMagnification",
    "_rlnPhaseShift",
    "_rlnAmplitudeContrast",
)
REQUIRED_COLUMNS = ("_rlnImageName", "_rlnRandomSubset", *POSE_SHIFT_COLUMNS, *CTF_COLUMNS)
OPTICS_COLUMNS = (
    "_rlnOpticsGroup",
    "_rlnOpticsGroupName",
    "_rlnAmplitudeContrast",
    "_rlnSphericalAberration",
    "_rlnVoltage",
    "_rlnImagePixelSize",
    "_rlnImageSize",
    "_rlnImageDimensionality",
)
CONVERTED_PARTICLE_COLUMNS = (
    "_rlnRandomSubset",
    "_rlnDefocusAngle",
    "_rlnDefocusU",
    "_rlnDefocusV",
    "_rlnPhaseShift",
    "_rlnImageName",
    "_rlnAngleRot",
    "_rlnAngleTilt",
    "_rlnAnglePsi",
    "_rlnClassNumber",
    "_rlnMicrographName",
    "_rlnCoordinateX",
    "_rlnCoordinateY",
    "_rlnGroupNumber",
    "_rlnGroupName",
    "_rlnOpticsGroup",
    "_rlnOriginXAngst",
    "_rlnOriginYAngst",
)
CONVERTED_EXACT_NUMERIC_COLUMNS = (
    "_rlnRandomSubset",
    "_rlnDefocusAngle",
    "_rlnDefocusU",
    "_rlnDefocusV",
    "_rlnPhaseShift",
    "_rlnAngleRot",
    "_rlnAngleTilt",
    "_rlnAnglePsi",
    "_rlnClassNumber",
    "_rlnCoordinateX",
    "_rlnCoordinateY",
)
CONVERTED_EXACT_TEXT_COLUMNS = ("_rlnImageName", "_rlnMicrographName", "_rlnGroupName")


def sha256_file(path: Path, *, limit_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit_bytes
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            size = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
            chunk = stream.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if limit_bytes is not None and remaining:
        raise ValueError(f"{path} is shorter than the requested {limit_bytes} bytes")
    return digest.hexdigest()


def _require_file(path: Path, *, size_bytes: int | None = None, sha256: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if size_bytes is not None and path.stat().st_size != size_bytes:
        raise ValueError(f"unexpected size for {path}: {path.stat().st_size} != {size_bytes}")
    if sha256 is not None:
        observed = sha256_file(path)
        if observed != sha256:
            raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {sha256}")


def half_assignment_bytes(table: pd.DataFrame) -> bytes:
    halves = table["_rlnRandomSubset"].to_numpy(dtype=np.uint8)
    return halves.astype(np.uint8, copy=False).tobytes(order="C")


def token_table_sha256(table: pd.DataFrame, columns: Iterable[str]) -> str:
    """Hash selected STAR tokens without numeric parsing or reformatting."""

    digest = hashlib.sha256()
    for column in columns:
        digest.update(column.encode("utf-8"))
        digest.update(b"\0")
        for value in table[column].astype(str):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _split_image_name(value: str) -> tuple[str, str]:
    index, separator, stack = str(value).partition("@")
    if separator != "@" or not index or not stack:
        raise ValueError(f"invalid _rlnImageName token: {value!r}")
    return index, stack


def validate_particle_table(
    table: pd.DataFrame,
    *,
    expected_count: int,
    expected_half_counts: dict[int, int],
    expected_stack_basename: str,
) -> None:
    if len(table) != expected_count:
        raise ValueError(f"particle count changed: {len(table)} != {expected_count}")
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"source STAR is missing required columns: {missing}")

    image_names = table["_rlnImageName"].astype(str).tolist()
    for row_index, image_name in enumerate(image_names, start=1):
        image_index, stack = _split_image_name(image_name)
        if int(image_index) != row_index:
            raise ValueError(
                f"particle row identity/order changed at row {row_index}: image index {image_index}"
            )
        if Path(stack).name != expected_stack_basename:
            raise ValueError(f"particle row {row_index} references unexpected stack {stack!r}")

    halves = table["_rlnRandomSubset"].to_numpy(dtype=np.int64)
    observed_half_counts = {half: int(np.count_nonzero(halves == half)) for half in (1, 2)}
    if not np.isin(halves, (1, 2)).all() or observed_half_counts != expected_half_counts:
        raise ValueError(
            f"invalid deposited half assignment: {observed_half_counts} != {expected_half_counts}"
        )

    for column in (*POSE_SHIFT_COLUMNS, *CTF_COLUMNS):
        values = table[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {column}")


def normalized_particle_table(source: pd.DataFrame, particle_stack: Path) -> pd.DataFrame:
    """Return a token-preserving copy with only the stack path normalized."""

    stack_path = particle_stack.resolve()
    if any(character.isspace() for character in str(stack_path)):
        raise ValueError(f"RELION STAR paths may not contain whitespace: {stack_path}")
    prepared = source.copy()
    prepared["_rlnImageName"] = [
        f"{_split_image_name(value)[0]}@{stack_path}" for value in source["_rlnImageName"].astype(str)
    ]
    return prepared


def write_deterministic_star(path: Path, table: pd.DataFrame) -> None:
    """Write a timestamp-free RELION 3.1 data_particles block."""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.star")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("# version 30001\n\ndata_particles\n\nloop_\n")
        for index, column in enumerate(table.columns, start=1):
            stream.write(f"{column} #{index}\n")
        for row in table.astype(str).itertuples(index=False, name=None):
            if any(any(character.isspace() for character in value) for value in row):
                raise ValueError("STAR token contains unsupported whitespace")
            stream.write(" ".join(row))
            stream.write("\n")
    os.replace(temporary, path)


def _write_deterministic_star_block(
    stream,
    block_name: str,
    table: pd.DataFrame,
    *,
    version: int,
) -> None:
    stream.write(f"# version {version}\n\n{block_name}\n\nloop_\n")
    for index, column in enumerate(table.columns, start=1):
        stream.write(f"{column} #{index}\n")
    for row in table.astype(str).itertuples(index=False, name=None):
        if any(any(character.isspace() for character in value) for value in row):
            raise ValueError("STAR token contains unsupported whitespace")
        stream.write(" ".join(row))
        stream.write("\n")


def write_deterministic_star31(
    path: Path,
    particles: pd.DataFrame,
    optics: pd.DataFrame,
) -> None:
    """Write deterministic RELION 3.1+ optics and particle blocks."""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.star")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        _write_deterministic_star_block(stream, "data_optics", optics, version=50_001)
        stream.write("\n")
        _write_deterministic_star_block(stream, "data_particles", particles, version=50_001)
    os.replace(temporary, path)


def verify_prepared_table(source: pd.DataFrame, prepared_path: Path, particle_stack: Path) -> pd.DataFrame:
    prepared, optics = read_star(str(prepared_path))
    if optics is not None:
        raise ValueError("prepared STAR unexpectedly contains an optics block")
    if prepared.columns.tolist() != source.columns.tolist() or len(prepared) != len(source):
        raise ValueError("prepared STAR changed table shape or column order")
    for column in source.columns:
        if column == "_rlnImageName":
            continue
        if not np.array_equal(source[column].astype(str), prepared[column].astype(str)):
            raise ValueError(f"prepared STAR changed deposited tokens in {column}")
    expected = normalized_particle_table(source, particle_stack)["_rlnImageName"].astype(str)
    if not np.array_equal(prepared["_rlnImageName"].astype(str), expected):
        raise ValueError("prepared STAR changed image order/identity or resolved the stack incorrectly")
    return prepared


def prepare_legacy_star(source_star: Path, particle_stack: Path, output_star: Path) -> dict[str, Any]:
    """Write and audit the normalized legacy STAR used only as converter input."""

    source, optics = read_star(str(source_star))
    if optics is not None:
        raise ValueError("the sealed set-6 source is expected to be a single-table STAR")
    validate_particle_table(
        source,
        expected_count=PARTICLE_COUNT,
        expected_half_counts=HALF_COUNTS,
        expected_stack_basename=PARTICLE_STACK.name,
    )
    prepared = normalized_particle_table(source, particle_stack)
    write_deterministic_star(output_star, prepared)
    reread = verify_prepared_table(source, output_star, particle_stack)

    half_bytes = half_assignment_bytes(reread)
    half_sha256 = hashlib.sha256(half_bytes).hexdigest()
    if half_sha256 != SOURCE_SHA256["half_assignment"]:
        raise ValueError(f"half assignment changed: {half_sha256}")
    prepared_sha256 = sha256_file(output_star)
    if prepared_sha256 != NORMALIZED_LEGACY_STAR_SHA256:
        raise ValueError(f"normalized legacy STAR digest changed: {prepared_sha256}")
    return {
        "path": str(output_star.resolve()),
        "sha256": prepared_sha256,
        "particle_count": len(reread),
        "preserves_source_particle_order": True,
        "preserves_source_half_assignment": True,
        "preserves_source_pose_shift_metadata": True,
        "preserves_source_ctf_metadata": True,
        "pose_shift_tokens_sha256": token_table_sha256(reread, POSE_SHIFT_COLUMNS),
        "ctf_tokens_sha256": token_table_sha256(reread, CTF_COLUMNS),
        "half_assignment_sha256": half_sha256,
        "half_assignment_encoding": (
            "bytes(int(rlnRandomSubset)) in source STAR row order; one uint8 byte per particle"
        ),
        "half_assignment_bytes": len(half_bytes),
        "half1_count": HALF_COUNTS[1],
        "half2_count": HALF_COUNTS[2],
    }


def validate_relion31_conversion(
    legacy: pd.DataFrame,
    particles: pd.DataFrame,
    optics: pd.DataFrame | None,
    particle_stack: Path,
    *,
    expected_count: int,
    expected_half_counts: dict[int, int],
    box_size: int,
    pixel_size_angstrom: float,
    origin_atol_pixels: float = ORIGIN_ROUNDTRIP_ATOL_PIXELS,
    preserve_legacy_group_numbers: bool = True,
    expected_group_count: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless RELION's optics conversion preserves particle metadata."""

    if optics is None or len(optics) != 1:
        raise ValueError(f"converted STAR must contain exactly one optics row, got {optics}")
    if tuple(optics.columns) != OPTICS_COLUMNS:
        raise ValueError(f"unexpected converted optics columns: {optics.columns.tolist()}")
    if len(particles) != expected_count or len(legacy) != expected_count:
        raise ValueError(
            f"converted particle count changed: legacy={len(legacy)} converted={len(particles)} "
            f"expected={expected_count}"
        )
    if tuple(particles.columns) != CONVERTED_PARTICLE_COLUMNS:
        raise ValueError(f"unexpected converted particle columns: {particles.columns.tolist()}")

    expected_stack = str(particle_stack.resolve())
    expected_image_names = legacy["_rlnImageName"].astype(str).to_numpy()
    converted_image_names = particles["_rlnImageName"].astype(str).to_numpy()
    if not np.array_equal(expected_image_names, converted_image_names):
        raise ValueError("relion_convert_star changed particle identity/order or stack paths")
    if any(_split_image_name(value)[1] != expected_stack for value in converted_image_names):
        raise ValueError("converted STAR does not resolve every image to the sealed source stack")

    for column in CONVERTED_EXACT_TEXT_COLUMNS:
        if not np.array_equal(legacy[column].astype(str), particles[column].astype(str)):
            raise ValueError(f"relion_convert_star changed text metadata in {column}")
    for column in CONVERTED_EXACT_NUMERIC_COLUMNS:
        before = legacy[column].to_numpy(dtype=np.float64)
        after = particles[column].to_numpy(dtype=np.float64)
        if not np.array_equal(before, after):
            raise ValueError(f"relion_convert_star changed numeric metadata in {column}")
    if preserve_legacy_group_numbers:
        if not np.array_equal(
            legacy["_rlnGroupNumber"].to_numpy(dtype=np.float64),
            particles["_rlnGroupNumber"].to_numpy(dtype=np.float64),
        ):
            raise ValueError("relion_convert_star changed numeric metadata in _rlnGroupNumber")

    halves = particles["_rlnRandomSubset"].to_numpy(dtype=np.int64)
    observed_half_counts = {half: int(np.count_nonzero(halves == half)) for half in (1, 2)}
    if not np.isin(halves, (1, 2)).all() or observed_half_counts != expected_half_counts:
        raise ValueError(f"converted half assignments changed: {observed_half_counts}")

    optics_row = optics.iloc[0]
    expected_optics = {
        "_rlnOpticsGroup": 1.0,
        "_rlnAmplitudeContrast": 0.07,
        "_rlnSphericalAberration": 2.7,
        "_rlnVoltage": 300.0,
        "_rlnImagePixelSize": float(pixel_size_angstrom),
        "_rlnImageSize": float(box_size),
        "_rlnImageDimensionality": 2.0,
    }
    if str(optics_row["_rlnOpticsGroupName"]) != "opticsGroup1":
        raise ValueError(f"unexpected optics group name: {optics_row['_rlnOpticsGroupName']!r}")
    for column, expected_value in expected_optics.items():
        observed_value = float(optics_row[column])
        if observed_value != expected_value:
            raise ValueError(f"unexpected optics value {column}={observed_value} != {expected_value}")
    if not np.array_equal(particles["_rlnOpticsGroup"].to_numpy(dtype=np.int64), np.ones(expected_count)):
        raise ValueError("converted particles do not all reference optics group 1")

    legacy_pixel_size = (
        legacy["_rlnDetectorPixelSize"].to_numpy(dtype=np.float64)
        * 10_000.0
        / legacy["_rlnMagnification"].to_numpy(dtype=np.float64)
    )
    if not np.array_equal(legacy_pixel_size, np.full(expected_count, pixel_size_angstrom)):
        raise ValueError("legacy detector pixel size/magnification do not imply the sealed pixel size")
    for legacy_column, optics_column in (
        ("_rlnAmplitudeContrast", "_rlnAmplitudeContrast"),
        ("_rlnSphericalAberration", "_rlnSphericalAberration"),
        ("_rlnVoltage", "_rlnVoltage"),
    ):
        values = legacy[legacy_column].to_numpy(dtype=np.float64)
        if not np.array_equal(values, np.full(expected_count, float(optics_row[optics_column]))):
            raise ValueError(f"legacy {legacy_column} does not match the converted optics row")

    origin_errors: list[np.ndarray] = []
    for pixel_column, angstrom_column in (
        ("_rlnOriginX", "_rlnOriginXAngst"),
        ("_rlnOriginY", "_rlnOriginYAngst"),
    ):
        legacy_pixels = legacy[pixel_column].to_numpy(dtype=np.float64)
        converted_pixels = particles[angstrom_column].to_numpy(dtype=np.float64) / pixel_size_angstrom
        origin_errors.append(np.abs(legacy_pixels - converted_pixels))
    max_origin_roundtrip_error = max(float(np.max(errors)) for errors in origin_errors)
    if max_origin_roundtrip_error > origin_atol_pixels:
        raise ValueError(
            "converted origin round-trip error exceeds the sealed tolerance: "
            f"{max_origin_roundtrip_error} > {origin_atol_pixels} pixels"
        )

    group_audit: dict[str, Any] = {}
    if not preserve_legacy_group_numbers:
        if expected_group_count is None:
            raise ValueError("expected_group_count is required for canonical group-number validation")
        group_audit = validate_relion_group_semantics(
            particles,
            expected_group_count=expected_group_count,
        )

    return {
        "particle_count": expected_count,
        "preserves_legacy_particle_order": True,
        "preserves_legacy_half_assignment": True,
        "preserves_legacy_pose_metadata": True,
        "preserves_legacy_ctf_metadata": True,
        "origin_conversion": "legacy pixels to RELION 3.1 Angstroms",
        "origin_roundtrip_max_error_pixels": max_origin_roundtrip_error,
        "origin_roundtrip_atol_pixels": origin_atol_pixels,
        "optics_group_count": 1,
        "optics_group": 1,
        "optics_group_name": "opticsGroup1",
        "optics_fields": {key.removeprefix("_rln"): value for key, value in expected_optics.items()},
        **group_audit,
    }


def stable_relion_group_numbers(group_names: Iterable[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Match RELION's first-appearance grouping of ``_rlnGroupName``."""

    mapping: dict[str, int] = {}
    numbers: list[int] = []
    for raw_name in group_names:
        name = str(raw_name)
        if not name:
            raise ValueError("empty _rlnGroupName is not valid")
        if name not in mapping:
            mapping[name] = len(mapping) + 1
        numbers.append(mapping[name])
    return np.asarray(numbers, dtype=np.int64), mapping


def validate_relion_group_semantics(
    particles: pd.DataFrame,
    *,
    expected_group_count: int,
) -> dict[str, Any]:
    names = particles["_rlnGroupName"].astype(str).to_numpy()
    numbers = particles["_rlnGroupNumber"].to_numpy(dtype=np.int64)
    expected_numbers, mapping = stable_relion_group_numbers(names)
    if len(mapping) != expected_group_count:
        raise ValueError(f"unexpected RELION group count: {len(mapping)} != {expected_group_count}")
    if not np.array_equal(numbers, expected_numbers):
        raise ValueError("_rlnGroupNumber does not match stable first-appearance _rlnGroupName labels")
    name_to_numbers: dict[str, set[int]] = {}
    number_to_names: dict[int, set[str]] = {}
    for name, number in zip(names, numbers, strict=True):
        name_to_numbers.setdefault(str(name), set()).add(int(number))
        number_to_names.setdefault(int(number), set()).add(str(name))
    if any(len(values) != 1 for values in name_to_numbers.values()):
        raise ValueError("one _rlnGroupName maps to multiple group numbers")
    if any(len(values) != 1 for values in number_to_names.values()):
        raise ValueError("one _rlnGroupNumber maps to multiple group names")
    if sorted(number_to_names) != list(range(1, expected_group_count + 1)):
        raise ValueError("canonical RELION group numbers are not contiguous from one")
    mapping_rows = [{"group_name": name, "group_number": number} for name, number in mapping.items()]
    mapping_encoding = json.dumps(mapping_rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {
        "group_count": expected_group_count,
        "group_number_policy": "stable first appearance of _rlnGroupName, matching RELION read semantics",
        "group_name_number_bijection": True,
        "group_mapping": mapping_rows,
        "group_mapping_sha256": hashlib.sha256(mapping_encoding).hexdigest(),
    }


def canonicalize_relion_group_numbers(
    particles: pd.DataFrame,
    *,
    expected_group_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical = particles.copy()
    numbers, mapping = stable_relion_group_numbers(canonical["_rlnGroupName"].astype(str))
    if len(mapping) != expected_group_count:
        raise ValueError(f"unexpected RELION group count: {len(mapping)} != {expected_group_count}")
    canonical["_rlnGroupNumber"] = numbers.astype(str)
    return canonical, validate_relion_group_semantics(
        canonical,
        expected_group_count=expected_group_count,
    )


def convert_legacy_star(
    converter: Path,
    legacy_path: Path,
    output_path: Path,
    log_path: Path,
    *,
    expected_output_size: int = RAW_CONVERTED_STAR_SIZE_BYTES,
    expected_output_sha256: str = RAW_CONVERTED_STAR_SHA256,
    expected_half_assignment_sha256: str = SOURCE_SHA256["half_assignment"],
) -> dict[str, Any]:
    """Create and audit RELION's raw 3.1+ converter output."""

    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_suffix(".partial.star")
    if temporary.exists():
        temporary.unlink()
    command = [
        str(converter),
        "--i",
        str(legacy_path),
        "--o",
        str(temporary),
        "--box_size",
        str(BOX_SIZE),
    ]
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, text=True)
    _require_file(temporary, size_bytes=expected_output_size, sha256=expected_output_sha256)
    legacy, legacy_optics = read_star(str(legacy_path))
    if legacy_optics is not None:
        raise ValueError("normalized legacy converter input unexpectedly contains optics metadata")
    particles, optics = read_star(str(temporary))
    audit = validate_relion31_conversion(
        legacy,
        particles,
        optics,
        PARTICLE_STACK,
        expected_count=PARTICLE_COUNT,
        expected_half_counts=HALF_COUNTS,
        box_size=BOX_SIZE,
        pixel_size_angstrom=VOXEL_SIZE_ANGSTROM,
    )
    converted_half_sha256 = hashlib.sha256(half_assignment_bytes(particles)).hexdigest()
    if converted_half_sha256 != expected_half_assignment_sha256:
        raise ValueError(f"converted STAR changed half assignments: {converted_half_sha256}")
    os.replace(temporary, output_path)
    return {
        "path": str(output_path.resolve()),
        "sha256": expected_output_sha256,
        "size_bytes": expected_output_size,
        "authoritative_refinement_star": False,
        "role": "sealed raw relion_convert_star output before group-number canonicalization",
        "converter_command": command,
        "converter_log": str(log_path.resolve()),
        "half_assignment_sha256": converted_half_sha256,
        **audit,
    }


def build_authoritative_star(
    legacy_path: Path,
    raw_converted_path: Path,
    output_path: Path,
    *,
    expected_output_size: int = PREPARED_STAR_SIZE_BYTES,
    expected_output_sha256: str = PREPARED_STAR_SHA256,
) -> dict[str, Any]:
    """Canonicalize RELION grouping and seal the STAR consumed by both engines."""

    legacy, legacy_optics = read_star(str(legacy_path))
    if legacy_optics is not None:
        raise ValueError("normalized legacy STAR unexpectedly contains an optics table")
    raw_particles, optics = read_star(str(raw_converted_path))
    if optics is None:
        raise ValueError("raw relion_convert_star output has no optics table")
    particles, group_audit = canonicalize_relion_group_numbers(
        raw_particles,
        expected_group_count=RELION_GROUP_COUNT,
    )
    write_deterministic_star31(output_path, particles, optics)
    _require_file(output_path, size_bytes=expected_output_size, sha256=expected_output_sha256)
    reread_particles, reread_optics = read_star(str(output_path))
    metadata_audit = validate_relion31_conversion(
        legacy,
        reread_particles,
        reread_optics,
        PARTICLE_STACK,
        expected_count=PARTICLE_COUNT,
        expected_half_counts=HALF_COUNTS,
        box_size=BOX_SIZE,
        pixel_size_angstrom=VOXEL_SIZE_ANGSTROM,
        preserve_legacy_group_numbers=False,
        expected_group_count=RELION_GROUP_COUNT,
    )
    half_sha256 = hashlib.sha256(half_assignment_bytes(reread_particles)).hexdigest()
    if half_sha256 != SOURCE_SHA256["half_assignment"]:
        raise ValueError(f"authoritative STAR changed half assignments: {half_sha256}")
    if metadata_audit["group_mapping_sha256"] != group_audit["group_mapping_sha256"]:
        raise ValueError("authoritative STAR changed the sealed group-name/number mapping")
    return {
        "path": str(output_path.resolve()),
        "sha256": expected_output_sha256,
        "size_bytes": expected_output_size,
        "authoritative_refinement_star": True,
        "role": "shared RECOVAR/RELION refinement input",
        "half_assignment_sha256": half_sha256,
        **metadata_audit,
    }


def download_pinned(url: str, destination: Path, *, size_bytes: int, sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _require_file(destination, size_bytes=size_bytes, sha256=sha256)
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "recovar-em-input-prep/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as output:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != size_bytes:
            raise ValueError(f"unexpected Content-Length for {url}: {content_length}")
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    _require_file(temporary, size_bytes=size_bytes, sha256=sha256)
    os.replace(temporary, destination)


def decompress_pinned_gzip(
    source: Path,
    destination: Path,
    *,
    size_bytes: int,
    sha256: str,
) -> None:
    if destination.exists():
        _require_file(destination, size_bytes=size_bytes, sha256=sha256)
        return
    temporary = destination.with_suffix(".partial.map")
    if temporary.exists():
        temporary.unlink()
    with gzip.open(source, "rb") as compressed, temporary.open("xb") as output:
        shutil.copyfileobj(compressed, output, length=8 * 1024 * 1024)
    _require_file(temporary, size_bytes=size_bytes, sha256=sha256)
    os.replace(temporary, destination)


def mrc_header(path: Path) -> dict[str, Any]:
    with mrcfile.open(path, mode="r", header_only=True, permissive=False) as mrc:
        voxel = mrc.voxel_size
        return {
            "shape": [int(mrc.header.nz), int(mrc.header.ny), int(mrc.header.nx)],
            "mode": int(mrc.header.mode),
            "voxel_size_angstrom": [float(voxel.z), float(voxel.y), float(voxel.x)],
        }


def validate_half_maps(paths: tuple[Path, Path]) -> list[dict[str, Any]]:
    headers = [mrc_header(path) for path in paths]
    for path, header in zip(paths, headers, strict=True):
        if header["shape"] != [360, 360, 360] or header["mode"] != 2:
            raise ValueError(f"unexpected deposited half-map header for {path}: {header}")
        if not np.allclose(header["voxel_size_angstrom"], VOXEL_SIZE_ANGSTROM, rtol=0, atol=1e-6):
            raise ValueError(f"unexpected deposited half-map voxel size for {path}: {header}")
    if headers[0] != headers[1]:
        raise ValueError(f"deposited half-map headers differ: {headers}")
    return headers


def average_relion_maps(paths: tuple[Path, Path], output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_suffix(".partial.mrc")
    if temporary.exists():
        temporary.unlink()
    with mrcfile.mmap(paths[0], mode="r", permissive=False) as first, mrcfile.mmap(
        paths[1], mode="r", permissive=False
    ) as second:
        if first.data.shape != second.data.shape or first.data.dtype != np.float32 or second.data.dtype != np.float32:
            raise ValueError("deposited half maps do not have matching float32 arrays")
        with mrcfile.new_mmap(
            temporary,
            shape=first.data.shape,
            mrc_mode=2,
            overwrite=True,
        ) as average:
            for z_start in range(0, first.data.shape[0], 8):
                z_stop = min(z_start + 8, first.data.shape[0])
                average.data[z_start:z_stop] = (
                    (
                        first.data[z_start:z_stop].astype(np.float64)
                        + second.data[z_start:z_stop].astype(np.float64)
                    )
                    * 0.5
                ).astype(np.float32)
            average.voxel_size = VOXEL_SIZE_ANGSTROM
    os.replace(temporary, output_path)


def run_relion_reference_preparation(
    image_handler: Path,
    average_map: Path,
    output_path: Path,
    log_path: Path,
) -> list[str]:
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_suffix(".partial.mrc")
    if temporary.exists():
        temporary.unlink()
    command = [
        str(image_handler),
        "--i",
        str(average_map),
        "--o",
        str(temporary),
        "--sym",
        SYMMETRY,
        "--lowpass",
        str(LOWPASS_ANGSTROM),
        "--angpix",
        str(VOXEL_SIZE_ANGSTROM),
        "--force_header_angpix",
        str(VOXEL_SIZE_ANGSTROM),
        "--new_box",
        str(BOX_SIZE),
    ]
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, text=True)
    if not temporary.is_file():
        raise RuntimeError("relion_image_handler completed without writing the reference")
    header = mrc_header(temporary)
    if header["shape"] != [BOX_SIZE, BOX_SIZE, BOX_SIZE] or header["mode"] != 2:
        raise ValueError(f"RELION reference has unexpected header: {header}")
    if not np.allclose(header["voxel_size_angstrom"], VOXEL_SIZE_ANGSTROM, rtol=0, atol=1e-6):
        raise ValueError(f"RELION reference has unexpected voxel size: {header}")
    os.replace(temporary, output_path)
    return command


def write_recovar_frame_twin(relion_path: Path, recovar_path: Path) -> None:
    if recovar_path.exists():
        raise FileExistsError(recovar_path)
    canonical = helpers.load_relion_volume(relion_path)
    helpers.write_mrc(recovar_path, canonical, voxel_size=VOXEL_SIZE_ANGSTROM)
    if not np.array_equal(helpers.load_mrc(recovar_path), helpers.load_relion_volume(relion_path)):
        raise ValueError("RECOVAR/RELION initial-reference canonical arrays are not exactly equal")


def canonical_array_sha256(relion_path: Path) -> str:
    """Hash load_relion_volume(path) as little-endian float32 C-order bytes."""

    digest = hashlib.sha256()
    with mrcfile.mmap(relion_path, mode="r", permissive=False) as mrc:
        raw = mrc.data
        for canonical_z in range(raw.shape[2]):
            plane = np.ascontiguousarray(-raw[:, :, canonical_z].T, dtype="<f4")
            digest.update(memoryview(plane).cast("B"))
    return digest.hexdigest()


def _validate_source_inputs() -> dict[str, Any]:
    _require_file(SOURCE_STAR, sha256=SOURCE_SHA256["star"])
    _require_file(POSES_PKL, sha256=SOURCE_SHA256["poses_pkl"])
    _require_file(CTF_PKL, sha256=SOURCE_SHA256["ctf_pkl"])
    _require_file(PARTICLE_STACK, size_bytes=PARTICLE_STACK_SIZE_BYTES)
    header_sha = sha256_file(PARTICLE_STACK, limit_bytes=1024 * 1024)
    if header_sha != SOURCE_SHA256["particle_stack_header_1mib"]:
        raise ValueError(f"particle-stack 1-MiB header digest changed: {header_sha}")
    full_sha = sha256_file(PARTICLE_STACK)
    if full_sha != SOURCE_SHA256["particle_stack"]:
        raise ValueError(f"particle-stack digest changed: {full_sha}")
    header = mrc_header(PARTICLE_STACK)
    if header["shape"] != [PARTICLE_COUNT, BOX_SIZE, BOX_SIZE] or header["mode"] != 2:
        raise ValueError(f"particle stack has unexpected header: {header}")
    return {
        "source_star": {"path": str(SOURCE_STAR), "sha256": SOURCE_SHA256["star"]},
        "particle_stack": {
            "path": str(PARTICLE_STACK),
            "size_bytes": PARTICLE_STACK_SIZE_BYTES,
            "header_1mib_sha256": header_sha,
            "sha256": full_sha,
            "header": header,
        },
        "poses_pkl": {"path": str(POSES_PKL), "sha256": SOURCE_SHA256["poses_pkl"]},
        "ctf_pkl": {"path": str(CTF_PKL), "sha256": SOURCE_SHA256["ctf_pkl"]},
    }


def prepare(output_dir: Path, image_handler: Path, convert_star: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SAFE_TO_DELETE").touch(exist_ok=True)
    downloads = output_dir / "downloads"
    prepared_dir = output_dir / "prepared"
    prepared_dir.mkdir(exist_ok=True)

    manifest_path = output_dir / "preparation_manifest.json"
    final_paths = (
        manifest_path,
        prepared_dir / "particles_set06_preserved_legacy.star",
        prepared_dir / "particles_set06_optics_raw_converter.star",
        prepared_dir / "particles_set06_optics.star",
        prepared_dir / "initial_reference_relion_I1_30A_box800.mrc",
        prepared_dir / "initial_reference_recovar_I1_30A_box800.mrc",
    )
    existing = [str(path) for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite prepared outputs: {existing}")

    source_manifest = _validate_source_inputs()
    _require_file(image_handler, sha256=RELION_IMAGE_HANDLER_SHA256)
    _require_file(convert_star, sha256=RELION_CONVERT_STAR_SHA256)
    legacy_star = final_paths[1]
    normalized_legacy_star = prepare_legacy_star(SOURCE_STAR, PARTICLE_STACK, legacy_star)
    raw_converted_star = final_paths[2]
    convert_star_log = output_dir / "relion_convert_star.log"
    raw_converter_output = convert_legacy_star(
        convert_star,
        legacy_star,
        raw_converted_star,
        convert_star_log,
    )
    authoritative_star = final_paths[3]
    prepared_star = build_authoritative_star(
        legacy_star,
        raw_converted_star,
        authoritative_star,
    )

    half_paths: list[Path] = []
    half_sources: list[dict[str, Any]] = []
    for spec in EMD_HALF_MAPS:
        gzip_path = downloads / f"{spec['name']}.gz"
        map_path = downloads / spec["name"]
        download_pinned(
            spec["url"],
            gzip_path,
            size_bytes=spec["gzip_size_bytes"],
            sha256=spec["gzip_sha256"],
        )
        decompress_pinned_gzip(
            gzip_path,
            map_path,
            size_bytes=spec["map_size_bytes"],
            sha256=spec["map_sha256"],
        )
        half_paths.append(map_path)
        half_sources.append(
            {
                **spec,
                "gzip_path": str(gzip_path),
                "map_path": str(map_path),
            }
        )
    half_tuple = (half_paths[0], half_paths[1])
    half_headers = validate_half_maps(half_tuple)

    average_map = prepared_dir / "emd_9012_half_average_relion_frame.mrc"
    average_relion_maps(half_tuple, average_map)
    relion_reference = final_paths[4]
    image_handler_log = output_dir / "relion_image_handler_reference.log"
    command = run_relion_reference_preparation(
        image_handler,
        average_map,
        relion_reference,
        image_handler_log,
    )
    recovar_reference = final_paths[5]
    write_recovar_frame_twin(relion_reference, recovar_reference)

    canonical_sha = canonical_array_sha256(relion_reference)
    manifest = {
        "schema": SCHEMA,
        "dataset": DATASET_ID,
        "image_set": IMAGE_SET,
        "input_contract": {
            **source_manifest,
            "prepared_star": prepared_star,
            "normalized_legacy_star": normalized_legacy_star,
            "raw_relion_convert_star_output": raw_converter_output,
            "authoritative_particle_star": str(authoritative_star.resolve()),
            "authoritative_particle_star_format": "RELION 3.1+ data_optics + data_particles",
            "relion_group_count": prepared_star["group_count"],
            "relion_group_number_policy": prepared_star["group_number_policy"],
            "relion_group_mapping_sha256": prepared_star["group_mapping_sha256"],
            "box_size": BOX_SIZE,
            "voxel_size_angstrom": VOXEL_SIZE_ANGSTROM,
        },
        "initial_reference": {
            "source": "arithmetic mean of deposited EMD-9012 half maps",
            "source_half_maps": half_sources,
            "source_half_map_headers": half_headers,
            "average_map": {
                "path": str(average_map),
                "sha256": sha256_file(average_map),
            },
            "processing": {
                "symmetry": SYMMETRY,
                "lowpass_angstrom": LOWPASS_ANGSTROM,
                "output_box_size": BOX_SIZE,
                "relion_image_handler_command": command,
                "relion_image_handler_log": str(image_handler_log),
            },
            "relion_file": {
                "path": str(relion_reference),
                "sha256": sha256_file(relion_reference),
            },
            "recovar_frame_file": {
                "path": str(recovar_reference),
                "sha256": sha256_file(recovar_reference),
            },
            "canonical_array_sha256": canonical_sha,
            "canonical_array_encoding": "load_relion_volume output, little-endian float32 C-order bytes",
            "canonical_exact": True,
            "exact_gate": "numpy.array_equal(load_mrc(recovar_frame_file), load_relion_volume(relion_file))",
        },
        "symmetry": {
            "requested_label": SYMMETRY,
            "relion_label": SYMMETRY,
            "recovar_label": SYMMETRY,
            "operator_count": SYMMETRY_OPERATOR_COUNT,
            "forbidden_bare_alias": "I (RELION canonicalizes it to I2)",
        },
        "refinement_contract": {
            "k": 1,
            "autonomous_refinement": True,
            "fixed_poses": False,
            "local_only": False,
            "replay": False,
            "oracle": False,
            "same_particles_and_order": True,
            "same_metadata": True,
            "same_half_assignment": True,
            "same_initial_reference_canonical_array": True,
            "shared_authoritative_star_format": "RELION 3.1+ optics groups",
            "shared_relion_group_semantics": prepared_star["group_number_policy"],
        },
        "software": {
            "preparation_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "relion_image_handler": {
                "path": str(image_handler.resolve()),
                "sha256": RELION_IMAGE_HANDLER_SHA256,
            },
            "relion_convert_star": {
                "path": str(convert_star.resolve()),
                "sha256": RELION_CONVERT_STAR_SHA256,
                "command_contract": (
                    "relion_convert_star --i <normalized legacy STAR> "
                    "--o <particles optics STAR> --box_size 800"
                ),
            },
        },
    }
    temporary_manifest = manifest_path.with_suffix(".partial.json")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--relion-image-handler", type=Path, default=DEFAULT_RELION_IMAGE_HANDLER)
    parser.add_argument("--relion-convert-star", type=Path, default=DEFAULT_RELION_CONVERT_STAR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = prepare(args.output_dir, args.relion_image_handler, args.relion_convert_star)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
