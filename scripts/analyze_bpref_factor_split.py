#!/usr/bin/env python3
"""Validate and compare passive RELION/RECOVAR BPref factor captures.

The RELION sidecar is paired with the existing pre-scatter summary artifact.
It records only significant orientation/translation pairs and qualified radius
support.  This loader fails closed on schema, identity, ordering, completeness,
and exact device-running-sum closure before exposing any factor arrays.
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

if __package__:
    from .validate_relion_bpref_prescatter import load_artifact
else:
    from validate_relion_bpref_prescatter import load_artifact  # type: ignore[no-redef]


HEADER_MAGIC = b"RLNBPFACTV1HEAD\0"
FOOTER_MAGIC = b"RLNBPFACTV1FOOT\0"
HEADER_STRUCT = struct.Struct("<16s40Q")
FOOTER_STRUCT = struct.Struct("<16sQQ")
FACTOR_ROW_DTYPE = np.dtype(
    {
        "names": (
            "state",
            "orientation_local",
            "translation_local",
            "pixel",
            "flags",
            "x",
            "y",
            "z",
            "image_re",
            "image_im",
            "trans_x",
            "trans_y",
            "trans_z",
            "posterior",
            "posterior_over_weight_norm",
            "minvsigma2",
            "ctf",
            "phase_re",
            "phase_im",
            "translated_re",
            "translated_im",
            "weighted_ctf",
            "term_re",
            "term_im",
            "weight_term",
            "running_re",
            "running_im",
            "running_weight",
        ),
        "formats": ("<u4",) * 5 + ("<i4",) * 3 + ("<f4",) * 20,
        "offsets": tuple(range(0, 112, 4)),
        "itemsize": 112,
    }
)
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)_img(?P<img>\d+)_class(?P<class_>\d+)\.bpf-v1\.bin"
)
ROW_FLAG_FWEIGHT_POSITIVE = 1
ROW_FLAG_RADIUS_SUPPORT = 2
ROW_FLAG_HERMITIAN_FOLD = 4
ROW_FLAG_MASK = 7


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0])


def array_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    """Return exact and scale-aware array metrics without correlation."""

    left = np.asarray(lhs)
    right = np.asarray(rhs)
    _require(left.shape == right.shape, f"shape mismatch: {left.shape} != {right.shape}")
    promoted_left = left.astype(np.complex128, copy=False).reshape(-1)
    promoted_right = right.astype(np.complex128, copy=False).reshape(-1)
    delta = promoted_right - promoted_left
    denominator = max(float(np.linalg.norm(promoted_left)), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "lhs_dtype": str(left.dtype),
        "rhs_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_lhs": float(np.linalg.norm(delta) / denominator),
        "delta_max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


@dataclass(frozen=True)
class FactorArtifact:
    path: Path
    sha256: str
    header: tuple[int, ...]
    rows: np.ndarray

    @property
    def part_id(self) -> int:
        return self.header[6]

    @property
    def stack_index(self) -> int:
        return self.header[7]

    @property
    def image_id(self) -> int:
        return self.header[8]

    @property
    def class_one_based(self) -> int:
        return self.header[5]


def load_factor_artifact(path: Path) -> FactorArtifact:
    """Load one sealed factor sidecar and reject structural ambiguity."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected factor file name: {path.name}")
    payload = path.read_bytes()
    _require(len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size, f"truncated factor artifact: {path}")
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload, 0)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"factor header magic mismatch: {path}")
    _require(header[0] == 1, f"factor schema version must be 1: {path}")
    _require(header[1] == HEADER_STRUCT.size, f"factor header size mismatch: {path}")
    _require(header[2] == FACTOR_ROW_DTYPE.itemsize, f"factor row size mismatch: {path}")
    _require(header[3] == FOOTER_STRUCT.size, f"factor footer size mismatch: {path}")
    row_count = header[18]
    expected_size = HEADER_STRUCT.size + row_count * FACTOR_ROW_DTYPE.itemsize + FOOTER_STRUCT.size
    _require(len(payload) == expected_size, f"factor artifact byte count mismatch: {path}")
    rows = np.frombuffer(
        payload,
        dtype=FACTOR_ROW_DTYPE,
        count=row_count,
        offset=HEADER_STRUCT.size,
    ).copy()
    footer_magic, footer_rows, footer_pairs = FOOTER_STRUCT.unpack_from(
        payload, HEADER_STRUCT.size + row_count * FACTOR_ROW_DTYPE.itemsize
    )
    _require(footer_magic == FOOTER_MAGIC, f"factor footer magic mismatch: {path}")
    _require(footer_rows == row_count, f"factor footer row count mismatch: {path}")
    _require(footer_pairs == header[17], f"factor footer pair count mismatch: {path}")
    assert match is not None
    _require(int(match["part"]) == header[6], f"factor part identity mismatch: {path}")
    _require(int(match["stack"]) == header[7], f"factor stack identity mismatch: {path}")
    _require(int(match["img"]) == header[8], f"factor image identity mismatch: {path}")
    _require(int(match["class_"]) == header[5], f"factor class identity mismatch: {path}")
    _validate_factor_rows(path, header, rows)
    return FactorArtifact(path, _sha256(path), header, rows)


def _validate_factor_rows(path: Path, header: tuple[int, ...], rows: np.ndarray) -> None:
    _require(header[4] > 0 and header[5] > 0, f"invalid iteration/class: {path}")
    _require(header[11] > 0 and header[12] > 0 and header[13] > 0, f"invalid image shape: {path}")
    _require(header[14] == header[11] * header[12] * header[13], f"factor image size mismatch: {path}")
    _require(header[15] > 0 and header[16] > 0 and header[17] > 0, f"invalid factor dimensions: {path}")
    _require(header[18] > 0, f"factor artifact has no supported rows: {path}")
    _require(np.isfinite(_float32_from_bits(header[19])) and _float32_from_bits(header[19]) > 0, f"invalid weight_norm: {path}")
    _require(np.isfinite(_float32_from_bits(header[20])), f"invalid significant threshold: {path}")
    _require(header[21] in (0, 1) and header[22] == 1, f"factor artifact is not passive: {path}")
    _require(header[23] > 0 and header[24] > 0, f"invalid factor capture cap: {path}")
    _require(header[25] <= header[24], f"factor file exceeds configured byte cap: {path}")
    _require(header[26] != 0 and header[27] != 0, f"missing factor identity hashes: {path}")
    _require(header[28] == 1, f"factor artifact is not paired to summary schema v1: {path}")
    _require(rows.size == header[18], f"factor row count changed: {path}")
    _require(np.all(rows["state"] == 1), f"factor row is not emitted support: {path}")
    _require(np.all(rows["orientation_local"] < header[15]), f"factor orientation out of range: {path}")
    _require(np.all(rows["translation_local"] < header[16]), f"factor translation out of range: {path}")
    _require(np.all(rows["pixel"] < header[14]), f"factor pixel out of range: {path}")
    flags = rows["flags"]
    _require(np.all((flags & np.uint32(~ROW_FLAG_MASK & 0xFFFFFFFF)) == 0), f"unknown factor flag: {path}")
    required = ROW_FLAG_FWEIGHT_POSITIVE | ROW_FLAG_RADIUS_SUPPORT
    _require(np.all((flags & required) == required), f"factor row lacks positive/support flags: {path}")
    pair = rows["orientation_local"].astype(np.int64) * header[16] + rows["translation_local"]
    key = pair * header[14] + rows["pixel"]
    _require(np.all(np.diff(key) > 0), f"factor rows are duplicated or not canonical: {path}")
    _require(np.unique(pair).size == header[17], f"factor active pair count mismatch: {path}")
    expected_x = rows["pixel"] % header[11]
    raw_y = rows["pixel"] // header[11]
    expected_y = np.where(raw_y > header[12] // 2, raw_y - header[12], raw_y)
    _require(np.array_equal(rows["x"], expected_x.astype(np.int32)), f"factor x/pixel mismatch: {path}")
    _require(np.array_equal(rows["y"], expected_y.astype(np.int32)), f"factor y/pixel mismatch: {path}")
    _require(np.all(rows["z"] == 0), f"2D factor row has nonzero z: {path}")
    floating = np.column_stack([rows[name] for name in FACTOR_ROW_DTYPE.names[8:]])
    _require(np.all(np.isfinite(floating)), f"non-finite factor value: {path}")
    _require(np.all(rows["posterior_over_weight_norm"] > 0), f"non-positive normalized posterior: {path}")
    _require(np.all(rows["weight_term"] >= 0), f"negative factor weight term: {path}")


def _summary_path(directory: Path, factor: FactorArtifact) -> Path:
    return Path(directory) / (
        f"part{factor.part_id}_stack{factor.stack_index}_img{factor.image_id}"
        f"_class{factor.class_one_based}.bpre-v1.bin"
    )


def validate_factor_directory(
    factor_directory: Path,
    *,
    expected_stack_indices: np.ndarray,
) -> tuple[tuple[FactorArtifact, ...], dict[str, object]]:
    """Validate a complete selected panel and exact summary-running closure."""

    factor_directory = Path(factor_directory)
    _require(factor_directory.is_dir(), f"factor directory does not exist: {factor_directory}")
    incomplete = sorted(factor_directory.glob("*.tmp.*"))
    _require(not incomplete, f"incomplete factor artifact remains: {incomplete[0] if incomplete else ''}")
    paths = sorted(factor_directory.glob("*.bpf-v1.bin"))
    expected = np.asarray(expected_stack_indices, dtype=np.int64)
    _require(expected.ndim == 1 and expected.size > 0, "expected stack selection is empty or not rank-1")
    _require(np.unique(expected).size == expected.size, "expected stack selection has duplicates")
    _require(len(paths) == expected.size, f"factor panel completeness mismatch: {len(paths)} != {expected.size}")
    factors = tuple(load_factor_artifact(path) for path in paths)
    observed = np.asarray([factor.stack_index for factor in factors], dtype=np.int64)
    _require(np.array_equal(np.sort(observed), np.sort(expected)), "factor/selection stack identities differ")

    closures = []
    hashes: dict[str, str] = {}
    active_pair_counts = []
    for factor in factors:
        summary = load_artifact(_summary_path(factor_directory, factor))
        _require(summary.part_id == factor.part_id, f"factor/summary part mismatch: {factor.path}")
        _require(summary.stack_index == factor.stack_index, f"factor/summary stack mismatch: {factor.path}")
        _require(summary.header[9] == factor.image_id, f"factor/summary image mismatch: {factor.path}")
        _require(summary.header[6] == factor.class_one_based, f"factor/summary class mismatch: {factor.path}")
        _require(summary.header[5] == factor.header[4], f"factor/summary iteration mismatch: {factor.path}")
        _require(summary.header[15] == factor.header[14], f"factor/summary image size mismatch: {factor.path}")
        _require(summary.header[16] == factor.header[15], f"factor/summary orientation count mismatch: {factor.path}")
        summary_key = summary.rows["orientation_local"].astype(np.int64) * factor.header[14] + summary.rows["pixel"]
        factor_key = factor.rows["orientation_local"].astype(np.int64) * factor.header[14] + factor.rows["pixel"]
        unique_factor_key, reverse = np.unique(factor_key, return_inverse=True)
        last = np.zeros(unique_factor_key.size, dtype=np.int64)
        np.maximum.at(last, reverse, np.arange(factor.rows.size, dtype=np.int64))
        running_data = factor.rows["running_re"][last] + 1j * factor.rows["running_im"][last]
        running_weight = factor.rows["running_weight"][last]
        positive = running_weight > 0
        _require(
            np.array_equal(unique_factor_key[positive], summary_key),
            f"positive factor/summary support differs: {factor.path}",
        )
        _require(
            np.all(running_weight[~positive] == 0) and np.all(running_data[~positive] == 0),
            f"zero-weight factor support has nonzero running sum: {factor.path}",
        )
        running_data = running_data[positive]
        running_weight = running_weight[positive]
        summary_data = summary.rows["source_re"] + 1j * summary.rows["source_im"]
        data_closure = array_metrics(summary_data.astype(np.complex64), running_data.astype(np.complex64))
        weight_closure = array_metrics(summary.rows["source_weight"], running_weight)
        _require(data_closure["exact_equal"] is True, f"factor running data does not exactly close summary: {factor.path}")
        _require(weight_closure["exact_equal"] is True, f"factor running weight does not exactly close summary: {factor.path}")
        closures.append(
            {
                "stack_index_1based": factor.stack_index,
                "zero_weight_radius_support_count": int(np.count_nonzero(~positive)),
                "data": data_closure,
                "weight": weight_closure,
            }
        )
        active_pair_counts.append(factor.header[17])
        hashes[factor.path.name] = factor.sha256
        hashes[summary.path.name] = summary.sha256

    report = {
        "schema": "relion-bpref-factor-panel-validation-v1",
        "metric_policy": "exact/array metrics only; no correlation",
        "factor_directory": str(factor_directory.resolve()),
        "particle_count": len(factors),
        "stack_identities_exact": True,
        "active_pair_count_min": int(min(active_pair_counts)),
        "active_pair_count_max": int(max(active_pair_counts)),
        "device_running_sum_exact_summary_closure": True,
        "per_particle_closure": closures,
        "artifact_sha256": hashes,
        "classification_ready": True,
    }
    return factors, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factor_directory", type=Path)
    parser.add_argument("--selection-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    with np.load(args.selection_npz, allow_pickle=False) as selection:
        _require(str(selection["schema"]) == "real10076-factor-split-selection-v1", "unknown selection schema")
        stacks = np.asarray(selection["stack_indices_1based"], dtype=np.int64)
    _, report = validate_factor_directory(args.factor_directory, expected_stack_indices=stacks)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
