#!/usr/bin/env python3
"""Compare exact RELION and RECOVAR cumulative K=1 BPref prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA = "recovar-k1-bpref-prefix-comparison-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_native_flat(path: Path, dtype) -> np.ndarray:
    with path.open("rb") as stream:
        count = np.fromfile(stream, dtype=np.uint64, count=1)
        _require(count.size == 1, f"missing native count in {path}")
        values = np.fromfile(stream, dtype=dtype)
    _require(values.size == int(count[0]), f"native count mismatch in {path}")
    return values


def _native_prefix(native_directory: Path, half: int, internal_index: int) -> dict:
    matches = sorted(
        native_directory.glob(
            f"half{half}_part{internal_index}_stack*_bpref_prefix_metadata.bin"
        )
    )
    _require(len(matches) == 1, f"expected one native prefix for half {half}, row {internal_index}")
    metadata_path = matches[0]
    stem = metadata_path.name.removesuffix("metadata.bin")
    metadata = _load_native_flat(metadata_path, np.uint64)
    _require(metadata.size == 15 and int(metadata[0]) == 1, "native prefix schema changed")
    _require(int(metadata[1]) == 1, "native prefix is not physical iteration 1")
    _require(int(metadata[2]) == half, "native half identity changed")
    _require(int(metadata[3]) == internal_index, "native internal-row identity changed")
    _require(int(metadata[5]) == 0, "native prefix was not captured by --j 1 thread 0")
    real = _load_native_flat(native_directory / f"{stem}real.bin", np.float32)
    imag = _load_native_flat(native_directory / f"{stem}imag.bin", np.float32)
    weight = _load_native_flat(native_directory / f"{stem}weight.bin", np.float32)
    _require(real.size == imag.size == weight.size == int(metadata[14]), "native prefix size changed")
    return {
        "data": np.asarray(real + np.complex64(1j) * imag, dtype=np.complex64),
        "weight": weight,
        "stack_index_1based": int(metadata[4]),
        "shape": [int(metadata[9]), int(metadata[8]), int(metadata[7])],
    }


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    _require(candidate.shape == reference.shape, "prefix shapes differ")
    difference = candidate.astype(
        np.complex128 if np.iscomplexobj(candidate) else np.float64
    ) - reference.astype(
        np.complex128 if np.iscomplexobj(reference) else np.float64
    )
    absolute = np.abs(difference)
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    unequal = np.flatnonzero(candidate != reference)
    return {
        "bitwise_equal": bool(unequal.size == 0),
        "unequal_count": int(unequal.size),
        "first_unequal_flat_index": None if unequal.size == 0 else int(unequal[0]),
        "max_abs": float(absolute.max(initial=0.0)),
        "relative_l2": float(np.linalg.norm(difference) / denominator),
    }


def analyze(selection_path: Path, native_directory: Path, recovar_directory: Path) -> dict:
    selection = json.loads(selection_path.read_text())
    _require(selection["schema"] == "recovar-k1-bpref-prefix-selection-v2", "selection schema changed")
    halves = {}
    for half in (1, 2):
        half_selection = selection[f"half{half}"]
        ordinals = [int(value) for value in half_selection["half_local_ordinals"]]
        originals = [int(value) for value in half_selection["original_indices"]]
        native_internal = [int(value) for value in half_selection["native_internal_indices"]]
        stack_indices = [int(value) for value in half_selection["stack_indices_1based"]]
        _require(
            len(ordinals) == len(originals) == len(native_internal) == len(stack_indices)
            and ordinals,
            "selection rows are incomplete",
        )
        _require(ordinals == sorted(ordinals), "selection ordinals are not increasing")
        rows = []
        for ordinal, original_index, internal_index, stack_index in zip(
            ordinals, originals, native_internal, stack_indices, strict=True
        ):
            native = _native_prefix(native_directory, half, internal_index)
            _require(
                native["stack_index_1based"] == stack_index,
                "native immutable stack identity changed",
            )
            recovar_path = recovar_directory / (
                f"bpref_accumulator_delta_it001_h{half}_orig{original_index:06d}.npz"
            )
            _require(recovar_path.is_file(), f"missing RECOVAR prefix {recovar_path}")
            with np.load(recovar_path, allow_pickle=False) as recovar:
                _require(
                    str(np.asarray(recovar["schema"]).reshape(()))
                    == "recovar-bpref-accumulator-delta-v2",
                    "RECOVAR prefix schema changed",
                )
                _require(int(recovar["iteration"]) == 1, "RECOVAR iteration changed")
                _require(int(recovar["half"]) == half, "RECOVAR half changed")
                _require(int(recovar["original_index"]) == original_index, "RECOVAR identity changed")
                _require(
                    int(recovar["particle_launch_ordinal"]) == ordinal,
                    "RECOVAR physical prefix ordinal changed",
                )
                # RELION BPref's native numerator convention is the negative
                # of RECOVAR's native-unit fused accumulator convention.
                recovar_data = -np.asarray(
                    recovar["after_data"], dtype=np.complex64
                ).reshape(-1)
                recovar_weight = np.asarray(recovar["after_weight"], dtype=np.float32).reshape(-1)
                isolated_data = -np.asarray(
                    recovar["isolated_data"], dtype=np.complex64
                ).reshape(-1)
                isolated_weight = np.asarray(recovar["isolated_weight"], dtype=np.float32).reshape(-1)
            row = {
                "half_local_ordinal": ordinal,
                "prefix_particle_count": ordinal + 1,
                "original_index": original_index,
                "native_internal_index": internal_index,
                "native_stack_index_1based": native["stack_index_1based"],
                "data": _metrics(recovar_data, native["data"]),
                "weight": _metrics(recovar_weight, native["weight"]),
            }
            if ordinal == 0:
                row["first_particle_isolated_data"] = _metrics(isolated_data, native["data"])
                row["first_particle_isolated_weight"] = _metrics(isolated_weight, native["weight"])
            row["joint_bitwise_equal"] = bool(
                row["data"]["bitwise_equal"] and row["weight"]["bitwise_equal"]
            )
            rows.append(row)
        first_joint = next((row for row in rows if not row["joint_bitwise_equal"]), None)
        halves[f"half{half}"] = {
            "selected_prefixes": len(rows),
            "bitwise_equal_prefixes": sum(row["joint_bitwise_equal"] for row in rows),
            "first_unequal_prefix": first_joint,
            "rows": rows,
        }
    return {
        "schema": SCHEMA,
        "selection": str(selection_path.resolve()),
        "native_directory": str(native_directory.resolve()),
        "recovar_directory": str(recovar_directory.resolve()),
        "numerator_alignment": "native_relion_data_equals_negative_recovar_data",
        "halves": halves,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# K=1 cumulative BPref prefix comparison",
        "",
        "| Half | Exact prefixes | Selected | First unequal particle count | First unequal source row |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for half in (1, 2):
        item = report["halves"][f"half{half}"]
        first = item["first_unequal_prefix"]
        lines.append(
            f"| {half} | {item['bitwise_equal_prefixes']} | {item['selected_prefixes']} | "
            f"{('none' if first is None else first['prefix_particle_count'])} | "
            f"{('none' if first is None else first['original_index'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--native-directory", required=True, type=Path)
    parser.add_argument("--recovar-directory", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.selection, args.native_directory, args.recovar_directory)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
