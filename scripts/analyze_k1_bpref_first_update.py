#!/usr/bin/env python3
"""Inspect the exact accumulator update at the first unequal K=1 prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_bpref_prefixes import _metrics, _native_prefix, _require


SCHEMA = "recovar-k1-bpref-first-update-v1"


def _float_bits(value: np.float32) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _value(value) -> dict:
    if np.iscomplexobj(value):
        value = np.complex64(value)
        return {
            "real": float(value.real),
            "imag": float(value.imag),
            "real_bits": _float_bits(value.real),
            "imag_bits": _float_bits(value.imag),
        }
    value = np.float32(value)
    return {"value": float(value), "bits": _float_bits(value)}


def _first_mismatch_panel(candidate: np.ndarray, reference: np.ndarray, *extras) -> dict | None:
    candidate = np.asarray(candidate).reshape(-1)
    reference = np.asarray(reference).reshape(-1)
    _require(candidate.shape == reference.shape, "first-mismatch arrays differ in shape")
    unequal = np.flatnonzero(candidate != reference)
    if unequal.size == 0:
        return None
    index = int(unequal[0])
    return {
        "flat_index": index,
        "candidate": _value(candidate[index]),
        "reference": _value(reference[index]),
        "extras": [_value(np.asarray(extra).reshape(-1)[index]) for extra in extras],
    }


def _recovar_capture(directory: Path, half: int, original_index: int) -> dict:
    path = directory / f"bpref_accumulator_delta_it001_h{half}_orig{original_index:06d}.npz"
    _require(path.is_file(), f"missing RECOVAR capture {path}")
    with np.load(path, allow_pickle=False) as values:
        _require(
            str(np.asarray(values["schema"]).reshape(()))
            == "recovar-bpref-accumulator-delta-v2",
            "RECOVAR capture schema changed",
        )
        _require(int(values["iteration"]) == 1, "RECOVAR capture iteration changed")
        _require(int(values["half"]) == half, "RECOVAR capture half changed")
        _require(int(values["original_index"]) == original_index, "RECOVAR identity changed")
        return {
            "path": str(path.resolve()),
            "ordinal": int(values["particle_launch_ordinal"]),
            # Native RELION and RECOVAR expose opposite numerator signs.
            "before_data": -np.asarray(values["before_data"], dtype=np.complex64).reshape(-1),
            "after_data": -np.asarray(values["after_data"], dtype=np.complex64).reshape(-1),
            "isolated_data": -np.asarray(values["isolated_data"], dtype=np.complex64).reshape(-1),
            "before_weight": np.asarray(values["before_weight"], dtype=np.float32).reshape(-1),
            "after_weight": np.asarray(values["after_weight"], dtype=np.float32).reshape(-1),
            "isolated_weight": np.asarray(values["isolated_weight"], dtype=np.float32).reshape(-1),
        }


def _component_report(
    native_before: np.ndarray,
    native_after: np.ndarray,
    recovar_before: np.ndarray,
    recovar_after: np.ndarray,
    recovar_isolated: np.ndarray,
) -> dict:
    one_shot_after = np.asarray(
        recovar_before + recovar_isolated,
        dtype=recovar_after.dtype,
    )
    native_delta = native_after.astype(
        np.complex128 if np.iscomplexobj(native_after) else np.float64
    ) - native_before.astype(
        np.complex128 if np.iscomplexobj(native_before) else np.float64
    )
    recovar_delta = recovar_after.astype(
        np.complex128 if np.iscomplexobj(recovar_after) else np.float64
    ) - recovar_before.astype(
        np.complex128 if np.iscomplexobj(recovar_before) else np.float64
    )
    return {
        "before": _metrics(recovar_before, native_before),
        "after": _metrics(recovar_after, native_after),
        "one_shot_after_vs_native": _metrics(one_shot_after, native_after),
        "one_shot_after_vs_recovar": _metrics(one_shot_after, recovar_after),
        "effective_delta_float64": _metrics(recovar_delta, native_delta),
        "after_first_mismatch": _first_mismatch_panel(
            recovar_after,
            native_after,
            recovar_before,
            recovar_isolated,
            one_shot_after,
            recovar_delta,
            native_delta,
        ),
    }


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
        selected = selection[f"half{half}"]
        rows = list(
            zip(
                map(int, selected["half_local_ordinals"]),
                map(int, selected["original_indices"]),
                map(int, selected["native_internal_indices"]),
                strict=True,
            )
        )
        first_unequal = None
        for previous, current in zip(rows, rows[1:], strict=False):
            previous_ordinal, _, previous_internal = previous
            current_ordinal, current_original, current_internal = current
            if current_ordinal != previous_ordinal + 1:
                continue
            native_before = _native_prefix(native_primary_directory, half, previous_internal)
            native_after = _native_prefix(native_primary_directory, half, current_internal)
            recovar = _recovar_capture(recovar_directory, half, current_original)
            _require(recovar["ordinal"] == current_ordinal, "RECOVAR update ordinal changed")
            data_after = _metrics(recovar["after_data"], native_after["data"])
            weight_after = _metrics(recovar["after_weight"], native_after["weight"])
            if data_after["bitwise_equal"] and weight_after["bitwise_equal"]:
                continue
            repeat_before = _native_prefix(native_repeat_directory, half, previous_internal)
            repeat_after = _native_prefix(native_repeat_directory, half, current_internal)
            first_unequal = {
                "previous_particle_count": previous_ordinal + 1,
                "current_particle_count": current_ordinal + 1,
                "current_original_index": current_original,
                "current_native_internal_index": current_internal,
                "recovar_capture": recovar["path"],
                "primary_vs_repeat_before": {
                    "data": _metrics(repeat_before["data"], native_before["data"]),
                    "weight": _metrics(repeat_before["weight"], native_before["weight"]),
                },
                "primary_vs_repeat_after": {
                    "data": _metrics(repeat_after["data"], native_after["data"]),
                    "weight": _metrics(repeat_after["weight"], native_after["weight"]),
                },
                "data": _component_report(
                    native_before["data"],
                    native_after["data"],
                    recovar["before_data"],
                    recovar["after_data"],
                    recovar["isolated_data"],
                ),
                "weight": _component_report(
                    native_before["weight"],
                    native_after["weight"],
                    recovar["before_weight"],
                    recovar["after_weight"],
                    recovar["isolated_weight"],
                ),
            }
            break
        halves[f"half{half}"] = {
            "first_consecutive_unequal_update": first_unequal,
            "available_particle_counts": [ordinal + 1 for ordinal, _, _ in rows],
        }
    return {
        "schema": SCHEMA,
        "selection": str(selection_path.resolve()),
        "native_primary_directory": str(native_primary_directory.resolve()),
        "native_repeat_directory": str(native_repeat_directory.resolve()),
        "recovar_directory": str(recovar_directory.resolve()),
        "halves": halves,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--native-primary-directory", required=True, type=Path)
    parser.add_argument("--native-repeat-directory", required=True, type=Path)
    parser.add_argument("--recovar-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        args.selection,
        args.native_primary_directory,
        args.native_repeat_directory,
        args.recovar_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
