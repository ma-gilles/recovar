#!/usr/bin/env python3
"""Compare target-only native RELION and RECOVAR K=1 XA/AA totals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_k1_scale_state_boundary import _metric, _model_groups, _require, _sha256


def _read_native_rows(
    path: Path,
    expected_half: int,
    expected_iteration: int,
) -> list[dict[str, float | int]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split("\t")
        _require(fields[0] == "scale_totals", f"{path}:{line_number}: invalid record type")
        values = {}
        for field in fields[1:]:
            key, separator, value = field.partition("=")
            _require(bool(separator) and key not in values, f"{path}:{line_number}: invalid field {field!r}")
            values[key] = value
        required = {"iter", "part_id", "halfset", "group_id", "old_scale", "xa", "aa", "raw"}
        _require(set(values) == required, f"{path}:{line_number}: unexpected fields {sorted(values)}")
        row = {
            "iteration": int(values["iter"]),
            "part_id": int(values["part_id"]),
            "halfset": int(values["halfset"]),
            "group_id": int(values["group_id"]),
            "old_scale": float(values["old_scale"]),
            "xa": float(values["xa"]),
            "aa": float(values["aa"]),
            "raw": float(values["raw"]),
        }
        _require(row["halfset"] == expected_half, f"{path}:{line_number}: wrong halfset")
        if row["iteration"] == expected_iteration:
            rows.append(row)
    _require(
        bool(rows),
        f"{path}: no native scale-total rows for iteration {expected_iteration}",
    )
    part_ids = [int(row["part_id"]) for row in rows]
    _require(len(part_ids) == len(set(part_ids)), f"{path}: duplicate particle IDs")
    return rows


def analyze(
    parity_dump: Path,
    native_paths: list[Path],
    relion_input_models: list[Path],
    relion_output_models: list[Path],
    *,
    iteration: int = 2,
) -> dict[str, object]:
    _require(len(native_paths) == len(relion_input_models) == len(relion_output_models) == 2, "two halves required")
    with np.load(parity_dump, allow_pickle=False) as payload:
        dump = {key: np.asarray(payload[key]) for key in payload.files}

    half_reports = []
    all_rec_xa = []
    all_rec_aa = []
    all_rel_xa = []
    all_rel_aa = []
    all_rec_scale = []
    all_rel_scale = []
    for half in (1, 2):
        input_counts, input_scales, input_names, original_size = _model_groups(relion_input_models[half - 1])
        output_counts, output_scales, output_names, output_size = _model_groups(relion_output_models[half - 1])
        _require(original_size == output_size, f"half {half}: model size changed")
        _require(np.array_equal(input_names, output_names), f"half {half}: group identity changed")
        _require(np.array_equal(input_counts, output_counts), f"half {half}: group counts changed")
        prefix = f"half{half}"
        rec_xa_all = np.asarray(dump[f"{prefix}_wsum_scale_correction_xa"], dtype=np.float64)
        rec_aa_all = np.asarray(dump[f"{prefix}_wsum_scale_correction_aa"], dtype=np.float64)
        rec_scale_all = np.asarray(dump[f"{prefix}_group_scale_corrections"], dtype=np.float64)
        divisor = float(original_size**4)
        rows = _read_native_rows(native_paths[half - 1], half, iteration)
        compared = []
        for native in rows:
            part_id = int(native["part_id"])
            group_id = int(native["group_id"])
            _require(part_id == group_id, f"half {half}: part/group identity differs for {part_id}/{group_id}")
            _require(0 <= group_id < rec_xa_all.size, f"half {half}: group out of range")
            _require(input_counts[group_id] == 1, f"half {half}: captured inactive group {group_id}")
            _require(np.isclose(native["old_scale"], input_scales[group_id], rtol=0.0, atol=1e-12), f"half {half}: old scale differs")
            rec_xa = float(rec_xa_all[group_id] / divisor)
            rec_aa = float(rec_aa_all[group_id] / divisor)
            rel_xa = float(native["xa"])
            rel_aa = float(native["aa"])
            rec_raw = rec_xa / rec_aa if rec_aa > 0.0 else 1.0
            rel_raw = rel_xa / rel_aa if rel_aa > 0.0 else 1.0
            _require(np.isclose(rel_raw, native["raw"], rtol=2e-15, atol=0.0), f"half {half}: native raw is inconsistent")
            compared.append(
                {
                    "part_id": part_id,
                    "group_name": str(input_names[group_id]),
                    "recovar_xa": rec_xa,
                    "relion_xa": rel_xa,
                    "xa_residual": rec_xa - rel_xa,
                    "recovar_aa": rec_aa,
                    "relion_aa": rel_aa,
                    "aa_residual": rec_aa - rel_aa,
                    "recovar_raw": rec_raw,
                    "relion_raw": rel_raw,
                    "raw_residual": rec_raw - rel_raw,
                    "raw_with_relion_xa": rel_xa / rec_aa if rec_aa > 0.0 else 1.0,
                    "raw_with_relion_aa": rec_xa / rel_aa if rel_aa > 0.0 else 1.0,
                    "recovar_scale": float(rec_scale_all[group_id]),
                    "relion_scale": float(output_scales[group_id]),
                }
            )
            all_rec_xa.append(rec_xa)
            all_rec_aa.append(rec_aa)
            all_rel_xa.append(rel_xa)
            all_rel_aa.append(rel_aa)
            all_rec_scale.append(float(rec_scale_all[group_id]))
            all_rel_scale.append(float(output_scales[group_id]))
        compared.sort(key=lambda row: abs(float(row["raw_residual"])), reverse=True)
        rec_xa_values = np.asarray([row["recovar_xa"] for row in compared])
        rec_aa_values = np.asarray([row["recovar_aa"] for row in compared])
        rel_xa_values = np.asarray([row["relion_xa"] for row in compared])
        rel_aa_values = np.asarray([row["relion_aa"] for row in compared])
        half_reports.append(
            {
                "half": half,
                "captured_count": len(compared),
                "comparisons": {
                    "xa": _metric(rec_xa_values, rel_xa_values),
                    "aa": _metric(rec_aa_values, rel_aa_values),
                    "raw": _metric(rec_xa_values / rec_aa_values, rel_xa_values / rel_aa_values),
                    "raw_with_relion_xa": _metric(rel_xa_values / rec_aa_values, rel_xa_values / rel_aa_values),
                    "raw_with_relion_aa": _metric(rec_xa_values / rel_aa_values, rel_xa_values / rel_aa_values),
                    "scale": _metric(
                        np.asarray([row["recovar_scale"] for row in compared]),
                        np.asarray([row["relion_scale"] for row in compared]),
                    ),
                },
                "rows_by_raw_residual": compared,
            }
        )

    all_rec_xa = np.asarray(all_rec_xa)
    all_rec_aa = np.asarray(all_rec_aa)
    all_rel_xa = np.asarray(all_rel_xa)
    all_rel_aa = np.asarray(all_rel_aa)
    all_rec_scale = np.asarray(all_rec_scale)
    all_rel_scale = np.asarray(all_rel_scale)
    artifacts = {"parity_dump": str(parity_dump.resolve()), "parity_dump_sha256": _sha256(parity_dump)}
    for half, native_path in enumerate(native_paths, start=1):
        artifacts[f"relion_native_half{half}"] = str(native_path.resolve())
        artifacts[f"relion_native_half{half}_sha256"] = _sha256(native_path)
    return {
        "schema": "recovar.em.k1_scale_native_targets.v1",
        "iteration": int(iteration),
        "halves": half_reports,
        "combined_comparisons": {
            "xa": _metric(all_rec_xa, all_rel_xa),
            "aa": _metric(all_rec_aa, all_rel_aa),
            "raw": _metric(all_rec_xa / all_rec_aa, all_rel_xa / all_rel_aa),
            "raw_with_relion_xa": _metric(all_rel_xa / all_rec_aa, all_rel_xa / all_rel_aa),
            "raw_with_relion_aa": _metric(all_rec_xa / all_rel_aa, all_rel_xa / all_rel_aa),
            "scale": _metric(all_rec_scale, all_rel_scale),
        },
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-dump", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=2)
    for half in (1, 2):
        parser.add_argument(f"--relion-native-half{half}", type=Path, required=True)
        parser.add_argument(f"--relion-input-half{half}-model", type=Path, required=True)
        parser.add_argument(f"--relion-output-half{half}-model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.parity_dump,
        [args.relion_native_half1, args.relion_native_half2],
        [args.relion_input_half1_model, args.relion_input_half2_model],
        [args.relion_output_half1_model, args.relion_output_half2_model],
        iteration=args.iteration,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
