#!/usr/bin/env python3
"""Compare RECOVAR cumulative BPref prefixes with two native RELION repeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_bpref_prefixes import _metrics, _native_prefix, _require


SCHEMA = "recovar-k1-bpref-prefix-repeat-comparison-v1"


def _relative_l2_ratio(cross_engine: dict, native_repeat: dict) -> float | None:
    denominator = float(native_repeat["relative_l2"])
    if denominator == 0.0:
        return 0.0 if float(cross_engine["relative_l2"]) == 0.0 else None
    return float(cross_engine["relative_l2"]) / denominator


def analyze(
    selection_path: Path,
    native_primary_directory: Path,
    native_repeat_directory: Path,
    recovar_directory: Path,
) -> dict:
    selection = json.loads(selection_path.read_text())
    _require(
        selection["schema"] == "recovar-k1-bpref-prefix-selection-v2",
        "selection schema changed",
    )
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
            native_primary = _native_prefix(native_primary_directory, half, internal_index)
            native_repeat = _native_prefix(native_repeat_directory, half, internal_index)
            _require(
                native_primary["stack_index_1based"]
                == native_repeat["stack_index_1based"]
                == stack_index,
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
                # Native RELION's numerator convention is the negative of the
                # RECOVAR native-unit fused accumulator convention.
                recovar_data = -np.asarray(recovar["after_data"], dtype=np.complex64).reshape(-1)
                recovar_weight = np.asarray(recovar["after_weight"], dtype=np.float32).reshape(-1)

            cross_data = _metrics(recovar_data, native_primary["data"])
            cross_weight = _metrics(recovar_weight, native_primary["weight"])
            repeat_data = _metrics(native_repeat["data"], native_primary["data"])
            repeat_weight = _metrics(native_repeat["weight"], native_primary["weight"])
            rows.append(
                {
                    "half_local_ordinal": ordinal,
                    "prefix_particle_count": ordinal + 1,
                    "original_index": original_index,
                    "native_internal_index": internal_index,
                    "native_stack_index_1based": stack_index,
                    "recovar_vs_native_primary": {
                        "data": cross_data,
                        "weight": cross_weight,
                    },
                    "native_repeat_vs_primary": {
                        "data": repeat_data,
                        "weight": repeat_weight,
                    },
                    "cross_to_repeat_relative_l2_ratio": {
                        "data": _relative_l2_ratio(cross_data, repeat_data),
                        "weight": _relative_l2_ratio(cross_weight, repeat_weight),
                    },
                }
            )
        halves[f"half{half}"] = {
            "selected_prefixes": len(rows),
            "last_prefix": rows[-1],
            "rows": rows,
        }
    return {
        "schema": SCHEMA,
        "selection": str(selection_path.resolve()),
        "native_primary_directory": str(native_primary_directory.resolve()),
        "native_repeat_directory": str(native_repeat_directory.resolve()),
        "recovar_directory": str(recovar_directory.resolve()),
        "numerator_alignment": "native_relion_data_equals_negative_recovar_data",
        "halves": halves,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# K=1 cumulative BPref prefix repeat comparison",
        "",
        "| Half | Last particle count | REC/REL data relL2 | REL repeat data relL2 | Ratio | REC/REL weight relL2 | REL repeat weight relL2 | Ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for half in (1, 2):
        row = report["halves"][f"half{half}"]["last_prefix"]
        cross = row["recovar_vs_native_primary"]
        repeat = row["native_repeat_vs_primary"]
        ratio = row["cross_to_repeat_relative_l2_ratio"]
        lines.append(
            f"| {half} | {row['prefix_particle_count']} | {cross['data']['relative_l2']:.9g} | "
            f"{repeat['data']['relative_l2']:.9g} | {ratio['data']} | "
            f"{cross['weight']['relative_l2']:.9g} | {repeat['weight']['relative_l2']:.9g} | "
            f"{ratio['weight']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--native-primary-directory", required=True, type=Path)
    parser.add_argument("--native-repeat-directory", required=True, type=Path)
    parser.add_argument("--recovar-directory", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        args.selection,
        args.native_primary_directory,
        args.native_repeat_directory,
        args.recovar_directory,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
