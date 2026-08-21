#!/usr/bin/env python3
"""Compare identity-aligned native and RECOVAR K=1 Wavg norm totals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "recovar.em.k1_wavg_norm_comparison.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    _require(candidate.shape == reference.shape and candidate.size > 0, "metric topology mismatch")
    residual = candidate - reference
    return {
        "count": int(candidate.size),
        "relative_l2": float(
            np.linalg.norm(residual) / max(np.linalg.norm(reference), np.finfo(float).tiny)
        ),
        "median_abs": float(np.median(np.abs(residual))),
        "p95_abs": float(np.percentile(np.abs(residual), 95)),
        "max_abs": float(np.max(np.abs(residual))),
        "signed_sum": float(np.sum(residual)),
    }


def analyze(native_path: Path, recovar_paths: list[Path], *, image_size: int) -> dict[str, object]:
    _require(image_size > 0, "image_size must be positive")
    _require(recovar_paths, "at least one RECOVAR dump is required")
    with np.load(native_path, allow_pickle=False) as payload:
        native_rows = np.asarray(payload["input_row"], dtype=np.int64)
        native_halves = np.asarray(payload["half"], dtype=np.int64)
        native_values = {
            "direct_current_size": np.asarray(payload["direct_current_size"], dtype=np.float64),
            "powerclass_high_shell": np.asarray(
                payload["powerclass_high_shell"], dtype=np.float64
            ),
            "total": np.asarray(payload["total"], dtype=np.float64),
        }
    _require(np.unique(native_rows).size == native_rows.size, "native input rows are not unique")
    for name, values in native_values.items():
        _require(values.shape == native_rows.shape, f"native {name} topology mismatch")
    native_position = {int(row): index for index, row in enumerate(native_rows)}
    divisor = float(image_size**4)
    seen_rows: set[int] = set()
    half_reports = []
    combined_candidate = {name: [] for name in native_values}
    combined_reference = {name: [] for name in native_values}

    for recovar_path in recovar_paths:
        with np.load(recovar_path, allow_pickle=False) as payload:
            half = int(np.asarray(payload["half"]).reshape(-1)[0])
            rows = np.asarray(payload["original_row"], dtype=np.int64)
            candidate = {
                name: np.asarray(payload[name], dtype=np.float64) / divisor
                for name in native_values
            }
            recorded_total = candidate["direct_current_size"] + candidate["powerclass_high_shell"]
            _require(
                np.array_equal(recorded_total, candidate["total"]),
                f"RECOVAR half {half} direct/high/total closure failed",
            )
        _require(np.unique(rows).size == rows.size, f"RECOVAR half {half} rows are not unique")
        _require(not seen_rows.intersection(rows.tolist()), "RECOVAR dumps contain duplicate rows")
        seen_rows.update(int(row) for row in rows)
        try:
            positions = np.asarray([native_position[int(row)] for row in rows], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"RECOVAR row is absent from native dump: {exc.args[0]}") from exc
        _require(
            np.all(native_halves[positions] == half),
            f"RECOVAR half {half} contains rows assigned to another native half",
        )
        reference = {name: values[positions] for name, values in native_values.items()}
        native_total = reference["direct_current_size"] + reference["powerclass_high_shell"]
        half_reports.append(
            {
                "half": half,
                "particle_count": int(rows.size),
                "metrics": {
                    name: _metric(candidate[name], reference[name]) for name in native_values
                },
                "closure": {
                    "native": _metric(native_total, reference["total"]),
                    "recovar": _metric(recorded_total, candidate["total"]),
                },
                "recovar_artifact": {
                    "path": str(recovar_path.resolve()),
                    "sha256": _sha256(recovar_path),
                },
            }
        )
        for name in native_values:
            combined_candidate[name].append(candidate[name])
            combined_reference[name].append(reference[name])

    half_reports.sort(key=lambda record: int(record["half"]))
    return {
        "schema": SCHEMA,
        "image_size": int(image_size),
        "relion_unit_divisor": int(divisor),
        "particle_count": int(len(seen_rows)),
        "native_artifact": {
            "path": str(native_path.resolve()),
            "sha256": _sha256(native_path),
        },
        "metrics": {
            name: _metric(
                np.concatenate(combined_candidate[name]),
                np.concatenate(combined_reference[name]),
            )
            for name in native_values
        },
        "halves": half_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-npz", required=True, type=Path)
    parser.add_argument("--recovar-npz", required=True, action="append", type=Path)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.native_npz, args.recovar_npz, image_size=args.image_size)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
