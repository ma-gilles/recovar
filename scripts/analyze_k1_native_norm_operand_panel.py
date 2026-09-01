#!/usr/bin/env python3
"""Join native and RECOVAR iteration-1 normalization operands by image ID."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_recovar_artifacts(roots: list[Path]) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    relion_unit_divisor = float(256**4)
    for root in roots:
        for path in sorted((root / "pass2").glob("norm_residual_*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                source_index = int(np.asarray(archive["original_index"]).item())
                if source_index in records:
                    raise ValueError(f"duplicate RECOVAR source row {source_index}")
                high = float(np.asarray(archive["relion_norm_high_shell"]).item())
                total = float(np.asarray(archive["weighted_img_per_image"]).item())
                total += float(np.asarray(archive["block_norm_residual"]).item())
                current = total - high
                records[source_index] = {
                    "artifact": str(path.resolve()),
                    "artifact_sha256": _sha256(path),
                    "half": int(np.asarray(archive["half"]).item()),
                    "source_index": source_index,
                    "current_size": current / relion_unit_divisor,
                    "high_shell": high / relion_unit_divisor,
                    "total": total / relion_unit_divisor,
                    "new_norm": float(np.sqrt(2.0 * total / relion_unit_divisor)),
                }
    return records


def analyze(native_panel: dict[str, object], recovar: dict[int, dict[str, object]]) -> dict[str, object]:
    records = []
    for native_record in native_panel["records"]:
        stack_index = int(str(native_record["image_name"]).split("@", 1)[0])
        source_index = stack_index - 1
        if source_index not in recovar:
            raise ValueError(f"missing RECOVAR artifact for source row {source_index}")
        recovar_record = recovar[source_index]
        if int(recovar_record["half"]) != int(native_record["half"]):
            raise ValueError(f"half mismatch for source row {source_index}")
        native = native_record["native"]
        delta = {
            field: float(recovar_record[field]) - float(native[field])
            for field in ("current_size", "high_shell", "total", "new_norm")
        }
        records.append(
            {
                "image_name": native_record["image_name"],
                "native_part_id": int(native_record["native_part_id"]),
                "source_index": source_index,
                "half": int(native_record["half"]),
                "native": native,
                "recovar": recovar_record,
                "delta": delta,
                "dominant_absolute_split_delta": (
                    "current_size"
                    if abs(delta["current_size"]) >= abs(delta["high_shell"])
                    else "high_shell"
                ),
            }
        )
    if len(records) != len(recovar):
        extras = sorted(set(recovar) - {item["source_index"] for item in records})
        raise ValueError(f"unmatched RECOVAR artifacts: {extras}")

    summary = {}
    for field in ("current_size", "high_shell", "total", "new_norm"):
        values = np.asarray([item["delta"][field] for item in records], dtype=np.float64)
        summary[field] = {
            "mean_recovar_minus_native": float(np.mean(values)),
            "mean_abs": float(np.mean(np.abs(values))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "positive_count": int(np.count_nonzero(values > 0)),
            "negative_count": int(np.count_nonzero(values < 0)),
            "exact_count": int(np.count_nonzero(values == 0)),
        }
    summary["dominant_absolute_split_count"] = {
        field: sum(item["dominant_absolute_split_delta"] == field for item in records)
        for field in ("current_size", "high_shell")
    }
    return {
        "schema": "recovar.em.k1_native_norm_operand_panel_comparison.v1",
        "status": "complete",
        "iteration": 1,
        "count": len(records),
        "summary": summary,
        "records": records,
        "metric_policy": "exact normalization operand deltas; FSC-AUC remains trajectory acceptance",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-panel", type=Path, required=True)
    parser.add_argument("--recovar-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_panel = json.loads(args.native_panel.read_text())
    recovar = _load_recovar_artifacts(args.recovar_root)
    report = analyze(native_panel, recovar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
