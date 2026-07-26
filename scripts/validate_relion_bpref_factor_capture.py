#!/usr/bin/env python3
"""Fail-closed validation for selected-stack RELION BPref factor captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__:
    from .validate_relion_bpref_prescatter import ROTATION_DTYPE, ROW_DTYPE
else:
    from validate_relion_bpref_prescatter import (  # type: ignore[no-redef]
        ROTATION_DTYPE,
        ROW_DTYPE,
    )

HEADER_MAGIC = b"RLNBPRF2HEADER\0\0"
FOOTER_MAGIC = b"RLNBPRF2FOOTER\0\0"
HEADER_STRUCT = struct.Struct("<16s64Q")
FOOTER_STRUCT = struct.Struct("<16s6Q")
TRANSLATION_DTYPE = np.dtype(
    {
        "names": ("translation", "reserved", "x", "y", "z", "reserved_tail"),
        "formats": ("<u4", "<u4", "<f4", "<f4", "<f4", "<u4"),
        "offsets": (0, 4, 8, 12, 16, 20),
        "itemsize": 24,
    }
)
HYPOTHESIS_DTYPE = np.dtype(
    {
        "names": (
            "orientation_local",
            "translation",
            "flags",
            "reserved",
            "posterior",
            "posterior_over_weight_norm",
        ),
        "formats": ("<u4", "<u4", "<u4", "<u4", "<f4", "<f4"),
        "offsets": (0, 4, 8, 12, 16, 20),
        "itemsize": 24,
    }
)
PIXEL_DTYPE = np.dtype(
    {
        "names": ("pixel", "x", "y", "z", "flags", "reserved", "image_re", "image_im", "ctf", "minvsigma2"),
        "formats": ("<u4", "<i4", "<i4", "<i4", "<u4", "<u4", "<f4", "<f4", "<f4", "<f4"),
        "offsets": tuple(range(0, 40, 4)),
        "itemsize": 40,
    }
)
TERM_DTYPE = np.dtype(
    {
        "names": (
            "state",
            "orientation_local",
            "translation",
            "pixel",
            "flags",
            "reserved",
            "translated_re",
            "translated_im",
            "posterior_over_weight_norm",
            "weighted_ctf",
            "term_re",
            "term_im",
            "weight_term",
            "reserved_float",
        ),
        "formats": ("<u4", "<u4", "<u4", "<u4", "<u4", "<u4") + ("<f4",) * 8,
        "offsets": tuple(range(0, 56, 4)),
        "itemsize": 56,
    }
)
FILE_NAME = re.compile(r"part(?P<part>\d+)_stack(?P<stack>\d+)_img(?P<img>\d+)_class(?P<class_>\d+)\.bpre-v2\.bin")


@dataclass(frozen=True)
class FactorCapture:
    path: Path
    sha256: str
    header: tuple[int, ...]
    rotations: np.ndarray
    translations: np.ndarray
    hypotheses: np.ndarray
    pixels: np.ndarray
    summaries: np.ndarray
    terms: np.ndarray

    @property
    def stack_index(self) -> int:
        return self.header[12]

    @property
    def geometry_only(self) -> bool:
        return bool(self.header[53])


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


def fnv1a64(text: str) -> int:
    value = 14695981039346656037
    for byte in text.encode():
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _read_array(payload: bytes, dtype: np.dtype, count: int, offset: int) -> tuple[np.ndarray, int]:
    values = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).copy()
    return values, offset + count * dtype.itemsize


def load_factor_capture(path: Path) -> FactorCapture:
    """Load one factor capture and reject layout, identity, or completeness ambiguity."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected factor-capture file name: {path.name}")
    payload = path.read_bytes()
    _require(len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size, f"truncated factor capture: {path}")
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload, 0)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"factor header magic mismatch: {path}")
    expected_sizes = (
        2,
        HEADER_STRUCT.size,
        ROTATION_DTYPE.itemsize,
        TRANSLATION_DTYPE.itemsize,
        HYPOTHESIS_DTYPE.itemsize,
        PIXEL_DTYPE.itemsize,
        ROW_DTYPE.itemsize,
        TERM_DTYPE.itemsize,
        FOOTER_STRUCT.size,
    )
    _require(header[:9] == expected_sizes, f"factor schema/record sizes changed: {path}")
    counts = header[46:52]
    offset = HEADER_STRUCT.size
    rotations, offset = _read_array(payload, ROTATION_DTYPE, counts[0], offset)
    translations, offset = _read_array(payload, TRANSLATION_DTYPE, counts[1], offset)
    hypotheses, offset = _read_array(payload, HYPOTHESIS_DTYPE, counts[2], offset)
    pixels, offset = _read_array(payload, PIXEL_DTYPE, counts[3], offset)
    summaries, offset = _read_array(payload, ROW_DTYPE, counts[4], offset)
    terms, offset = _read_array(payload, TERM_DTYPE, counts[5], offset)
    _require(offset + FOOTER_STRUCT.size == len(payload), f"factor byte count mismatch: {path}")
    footer_magic, *footer_counts = FOOTER_STRUCT.unpack_from(payload, offset)
    _require(footer_magic == FOOTER_MAGIC, f"factor footer magic mismatch: {path}")
    _require(tuple(int(value) for value in footer_counts) == counts, f"factor footer counts changed: {path}")
    assert match is not None
    _require(int(match["part"]) == header[11], f"factor part identity mismatch: {path}")
    _require(int(match["stack"]) == header[12], f"factor stack identity mismatch: {path}")
    _require(int(match["img"]) == header[13], f"factor image identity mismatch: {path}")
    _require(int(match["class_"]) == header[10], f"factor class identity mismatch: {path}")
    _validate_arrays(path, header, rotations, translations, hypotheses, pixels, summaries, terms)
    return FactorCapture(path, _sha256(path), header, rotations, translations, hypotheses, pixels, summaries, terms)


def _validate_arrays(path, header, rotations, translations, hypotheses, pixels, summaries, terms) -> None:
    _require(header[53] in (0, 1), f"invalid geometry-only capture flag: {path}")
    geometry_only = bool(header[53])
    _require(header[9] > 0 and header[10] > 0, f"invalid iteration/class: {path}")
    _require(header[16] > 0 and header[17] > 0 and header[18] == 1, f"invalid 2D image shape: {path}")
    _require(header[19] == header[16] * header[17], f"factor image size mismatch: {path}")
    _require(header[20] == rotations.size and header[21] == translations.size, f"factor panel counts changed: {path}")
    if geometry_only:
        _require(
            hypotheses.size == pixels.size == summaries.size == terms.size == 0,
            f"geometry-only capture contains factor-value arrays: {path}",
        )
    else:
        _require(hypotheses.size == rotations.size * translations.size, f"hypothesis panel is not dense: {path}")
        _require(pixels.size == header[19], f"pixel panel is incomplete: {path}")
    _require(header[27] == 0 and header[28] == 0, f"factor v2 requires ordinary 2D non-premultiplied CTF: {path}")
    _require(header[29] > 0 and header[29] <= header[30] * header[31], f"invalid factor capture caps: {path}")
    _require(header[33] <= header[32], f"factor capture byte cap exceeded: {path}")
    _require(header[34] and header[35] and header[36], f"factor capture identity hash is zero: {path}")
    _require(header[42] == 1, f"factor capture is not passive: {path}")
    _require(
        header[52] == (0 if geometry_only else 1),
        f"factor capture density/geometry flags disagree: {path}",
    )

    expected_rotations = np.arange(rotations.size, dtype=np.uint32)
    _require(np.array_equal(rotations["orientation_local"], expected_rotations), f"rotation order changed: {path}")
    _require(
        np.all(rotations["reserved"] == 0) and np.all(np.isfinite(rotations["matrix"])), f"invalid rotations: {path}"
    )
    identities = np.stack((rotations["orientation_class_key"], rotations["oversampled_rotation"]), axis=1)
    _require(np.unique(identities, axis=0).shape[0] == rotations.size, f"duplicate rotation identity: {path}")

    expected_translations = np.arange(translations.size, dtype=np.uint32)
    _require(np.array_equal(translations["translation"], expected_translations), f"translation order changed: {path}")
    _require(
        np.all(translations["reserved"] == 0) and np.all(translations["reserved_tail"] == 0),
        f"translation reserved field changed: {path}",
    )
    _require(
        np.all(np.isfinite(np.stack((translations["x"], translations["y"], translations["z"])))),
        f"non-finite translation: {path}",
    )
    if geometry_only:
        _require(header[43] == 0 and header[44] == 0, f"geometry-only summary accounting changed: {path}")
        _require(
            header[45] <= rotations.size * translations.size,
            f"geometry-only accepted-hypothesis count is impossible: {path}",
        )
        return

    expected_hypothesis = np.arange(hypotheses.size)
    _require(
        np.array_equal(hypotheses["orientation_local"], expected_hypothesis // translations.size),
        f"hypothesis orientation order changed: {path}",
    )
    _require(
        np.array_equal(hypotheses["translation"], expected_hypothesis % translations.size),
        f"hypothesis translation order changed: {path}",
    )
    _require(
        np.all((hypotheses["flags"] & ~np.uint32(1)) == 0) and np.all(hypotheses["reserved"] == 0),
        f"invalid hypothesis flags: {path}",
    )
    posterior_values = np.stack((hypotheses["posterior"], hypotheses["posterior_over_weight_norm"]))
    _require(np.all(np.isfinite(posterior_values)), f"non-finite hypothesis values: {path}")
    threshold = _float32_from_bits(header[25])
    expected_accepted = hypotheses["posterior"] >= threshold
    _require(
        np.array_equal((hypotheses["flags"] & 1) != 0, expected_accepted),
        f"hypothesis acceptance differs from production predicate: {path}",
    )
    _require(np.count_nonzero(expected_accepted) == header[45], f"accepted hypothesis count changed: {path}")

    expected_pixels = np.arange(pixels.size, dtype=np.uint32)
    _require(np.array_equal(pixels["pixel"], expected_pixels), f"pixel order changed: {path}")
    _require(np.array_equal(pixels["x"], expected_pixels % header[16]), f"pixel x coordinate changed: {path}")
    raw_y = expected_pixels.astype(np.int64) // header[16]
    expected_y = np.where(raw_y > header[17] // 2, raw_y - header[17], raw_y)
    _require(np.array_equal(pixels["y"], expected_y), f"pixel y coordinate changed: {path}")
    _require(
        np.all(pixels["z"] == 0) and np.all(pixels["flags"] == 0) and np.all(pixels["reserved"] == 0),
        f"invalid pixel metadata: {path}",
    )
    pixel_values = np.stack((pixels["image_re"], pixels["image_im"], pixels["ctf"], pixels["minvsigma2"]))
    _require(np.all(np.isfinite(pixel_values)), f"non-finite pixel factor: {path}")

    _require(np.all(summaries["state"] == 1), f"inactive summary row was serialized: {path}")
    required_summary = np.uint32(1 | 2)
    _require(
        np.all((summaries["flags"] & required_summary) == required_summary),
        f"summary support/weight flags changed: {path}",
    )
    summary_key = summaries["orientation_local"].astype(np.int64) * pixels.size + summaries["pixel"]
    _require(np.all(np.diff(summary_key) > 0), f"summary rows are not canonical and unique: {path}")
    _require(header[43] == summaries.size + header[44], f"summary support accounting changed: {path}")
    summary_values = np.stack((summaries["source_re"], summaries["source_im"], summaries["source_weight"]))
    _require(
        np.all(np.isfinite(summary_values)) and np.all(summaries["source_weight"] > 0),
        f"invalid summary operands: {path}",
    )

    _require(
        np.all(terms["state"] == 1) and np.all(terms["flags"] == 1),
        f"inactive or unexpected term flag was serialized: {path}",
    )
    _require(
        np.all(terms["reserved"] == 0) and np.all(terms["reserved_float"] == 0), f"term reserved field changed: {path}"
    )
    expected_term_count = int(header[45]) * pixels.size
    _require(terms.size == expected_term_count, f"active term panel is incomplete: {path}")
    accepted_flat = np.flatnonzero(expected_accepted)
    expected_orientation = np.repeat(accepted_flat // translations.size, pixels.size)
    expected_translation = np.repeat(accepted_flat % translations.size, pixels.size)
    expected_term_pixel = np.tile(np.arange(pixels.size), accepted_flat.size)
    _require(
        np.array_equal(terms["orientation_local"], expected_orientation), f"term orientation order changed: {path}"
    )
    _require(np.array_equal(terms["translation"], expected_translation), f"term translation order changed: {path}")
    _require(np.array_equal(terms["pixel"], expected_term_pixel), f"term pixel order changed: {path}")
    term_values = np.stack(tuple(terms[name] for name in TERM_DTYPE.names[6:]))
    _require(np.all(np.isfinite(term_values)), f"non-finite factor term: {path}")


def validate_directory(
    directory: Path,
    selection_json: Path,
    *,
    expected_rank: int | None = None,
) -> dict[str, object]:
    selection_json = Path(selection_json)
    selection = json.loads(selection_json.read_text())
    _require(selection.get("schema") == "bpref-factor-stratification-v1", "unexpected selection schema")
    selected = selection.get("selected")
    _require(isinstance(selected, list) and selected, "selection is empty")
    expected_stacks = [int(record["stack_index_1based"]) for record in selected]
    _require(len(expected_stacks) == len(set(expected_stacks)), "selection contains duplicate stacks")
    canonical_stack_text = ",".join(str(value) for value in expected_stacks)
    paths = sorted(Path(directory).glob("*.bpre-v2.bin"))
    _require(not list(Path(directory).glob("*.tmp.*")), "factor capture contains incomplete temporary files")
    _require(len(paths) == len(expected_stacks), "factor capture file count differs from selection")
    captures = tuple(load_factor_capture(path) for path in paths)
    stacks = [capture.stack_index for capture in captures]
    _require(
        set(stacks) == set(expected_stacks) and len(set(stacks)) == len(stacks),
        "factor capture stack set is incomplete or duplicated",
    )
    if expected_rank is None:
        _require(
            all("expected_mpi_rank" in record for record in selected),
            "selection is missing expected_mpi_rank for mixed-rank validation",
        )
        expected_rank_by_stack = {
            int(record["stack_index_1based"]): int(record["expected_mpi_rank"]) for record in selected
        }
    else:
        expected_rank_by_stack = {stack: expected_rank for stack in expected_stacks}
    _require(
        all(capture.header[14] == expected_rank_by_stack[capture.stack_index] for capture in captures),
        "factor capture MPI rank changed",
    )
    rank_by_stack = {capture.stack_index: int(capture.header[14]) for capture in captures}
    rank_counts = {
        str(rank): sum(captured_rank == rank for captured_rank in rank_by_stack.values())
        for rank in sorted(set(rank_by_stack.values()))
    }
    expected_set_hash = fnv1a64(canonical_stack_text)
    _require(
        all(capture.header[36] == expected_set_hash for capture in captures), "factor capture selected-set hash changed"
    )
    # Fine-orientation support and posterior normalization are particle-local
    # in adaptive refinement.  In particular, orientation_num,
    # significant_weight, weight_norm, and the resulting byte estimate may
    # legitimately differ between selected particles.  Each capture validates
    # those fields against its own arrays above; only run-wide geometry and
    # capture-policy fields must agree across the directory.
    reference_fields = (
        9,
        10,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        27,
        28,
        29,
        30,
        31,
        32,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        52,
        53,
    )
    reference = tuple(captures[0].header[index] for index in reference_fields)
    _require(
        all(tuple(capture.header[index] for index in reference_fields) == reference for capture in captures),
        "factor capture runtime dimensions or policy changed between particles",
    )
    return {
        "schema": "relion-bpref-factor-capture-validation-v2",
        "metric_policy": "exact/array metrics for intermediate operands; no correlation",
        "capture_ready": True,
        "selection_json": str(selection_json.resolve()),
        "selection_sha256": _sha256(selection_json),
        "selected_stack_text": canonical_stack_text,
        "selected_stack_fnv1a64": expected_set_hash,
        "particle_count": len(captures),
        "mpi_rank": next(iter(rank_by_stack.values())) if len(rank_counts) == 1 else None,
        "mpi_rank_by_stack": {str(stack): rank_by_stack[stack] for stack in expected_stacks},
        "mpi_rank_counts": rank_counts,
        "orientation_count": (
            int(captures[0].rotations.size)
            if all(capture.rotations.size == captures[0].rotations.size for capture in captures)
            else None
        ),
        "translation_count": (
            int(captures[0].translations.size)
            if all(capture.translations.size == captures[0].translations.size for capture in captures)
            else None
        ),
        "orientation_counts_per_particle": [int(capture.rotations.size) for capture in captures],
        "translation_counts_per_particle": [int(capture.translations.size) for capture in captures],
        "image_pixel_count": int(captures[0].pixels.size),
        "accepted_hypotheses_per_particle": [int(capture.header[45]) for capture in captures],
        "artifact_sha256": {capture.path.name: capture.sha256 for capture in captures},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument(
        "--expected-rank",
        type=int,
        help="Require one MPI rank for all particles; omit to use expected_mpi_rank from the selection",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite validation artifact: {args.output_json}")
    report = validate_directory(args.capture_directory, args.selection_json, expected_rank=args.expected_rank)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
