#!/usr/bin/env python3
"""Freeze the largest source-ID-aligned RECOVAR/RELION Pmax residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import starfile

SCHEMA = "recovar.em.k1_pmax_capture_panel.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _column(table, name: str):
    matches = [column for column in table.columns if str(column).lstrip("_") == name.lstrip("_")]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} column, got {matches}")
    return table[matches[0]]


def _optional_column(table, name: str):
    matches = [column for column in table.columns if str(column).lstrip("_") == name.lstrip("_")]
    if len(matches) > 1:
        raise ValueError(f"expected at most one {name} column, got {matches}")
    return None if not matches else table[matches[0]]


def _particle_table(path: Path):
    payload = starfile.read(str(path))
    if not isinstance(payload, dict):
        return payload
    matches = [table for table in payload.values() if _optional_column(table, "rlnImageName") is not None]
    if len(matches) != 1:
        raise ValueError(f"expected one particle table with rlnImageName in {path}, got {len(matches)}")
    return matches[0]


def build_panel(
    *,
    parity_dump: Path,
    input_star: Path,
    relion_star: Path,
    top_n: int,
) -> dict[str, object]:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    input_table = _particle_table(input_star)
    relion_table = _particle_table(relion_star)
    input_names = np.asarray(_column(input_table, "rlnImageName").astype(str), dtype=str)
    relion_names = np.asarray(_column(relion_table, "rlnImageName").astype(str), dtype=str)
    if len(set(input_names.tolist())) != input_names.size:
        raise ValueError("input STAR image identities are not unique")
    if len(set(relion_names.tolist())) != relion_names.size:
        raise ValueError("RELION STAR image identities are not unique")
    if set(input_names.tolist()) != set(relion_names.tolist()):
        raise ValueError("input and RELION STAR identity sets differ")
    relion_pmax_values = np.asarray(
        _column(relion_table, "rlnMaxValueProbDistribution").astype(float), dtype=np.float64
    )
    relion_pmax = dict(zip(relion_names.tolist(), relion_pmax_values.tolist(), strict=True))
    subset_column = _optional_column(input_table, "rlnRandomSubset")
    if subset_column is not None:
        input_subsets = np.asarray(subset_column.astype(int), dtype=np.int64)
    else:
        relion_subset_column = _optional_column(relion_table, "rlnRandomSubset")
        if relion_subset_column is None:
            input_subsets = None
        else:
            relion_subsets = np.asarray(relion_subset_column.astype(int), dtype=np.int64)
            subset_by_name = dict(zip(relion_names.tolist(), relion_subsets.tolist(), strict=True))
            input_subsets = np.asarray([subset_by_name[name] for name in input_names], dtype=np.int64)

    records: list[dict[str, object]] = []
    with np.load(parity_dump, allow_pickle=False) as payload:
        relion_iteration = int(np.asarray(payload["relion_iteration"]).item())
        for half in (1, 2):
            source_rows = np.asarray(payload[f"half{half}_original_image_indices"], dtype=np.int64)
            recovar_pmax = np.asarray(payload[f"half{half}_max_posterior"], dtype=np.float64)
            if source_rows.shape != recovar_pmax.shape:
                raise ValueError(f"half {half} identity and Pmax arrays differ in shape")
            if input_subsets is not None and np.any(input_subsets[source_rows] != half):
                raise ValueError(f"half {half} dump contains a source row from the other subset")
            for physical_position, (source_row, rec_value) in enumerate(
                zip(source_rows.tolist(), recovar_pmax.tolist(), strict=True)
            ):
                image_name = str(input_names[source_row])
                rel_value = float(relion_pmax[image_name])
                delta = float(rec_value - rel_value)
                records.append(
                    {
                        "half": half,
                        "physical_position_zero_based": physical_position,
                        "source_row_zero_based": int(source_row),
                        "image_name": image_name,
                        "recovar_pmax": float(rec_value),
                        "relion_pmax": rel_value,
                        "signed_pmax_delta_recovar_minus_relion": delta,
                        "absolute_pmax_error": abs(delta),
                    }
                )
    source_rows_all = np.asarray([record["source_row_zero_based"] for record in records], dtype=np.int64)
    if source_rows_all.size != input_names.size or not np.array_equal(
        np.sort(source_rows_all), np.arange(input_names.size, dtype=np.int64)
    ):
        raise ValueError("parity dump halves do not cover every input source row exactly once")

    records.sort(key=lambda record: (-float(record["absolute_pmax_error"]), int(record["source_row_zero_based"])))
    selected = records[: min(top_n, len(records))]
    for rank, record in enumerate(selected, start=1):
        record["rank_by_absolute_pmax_error"] = rank
    deltas = np.asarray(
        [record["signed_pmax_delta_recovar_minus_relion"] for record in records], dtype=np.float64
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "relion_iteration": relion_iteration,
        "selection_policy": {
            "metric": "absolute source-ID-aligned Pmax error",
            "ordering": "descending absolute error, stable by source row for exact ties",
            "requested_top_n": top_n,
            "selected_count": len(selected),
        },
        "population": {
            "n_particles": len(records),
            "mean_signed_delta": float(np.mean(deltas)),
            "mean_absolute_error": float(np.mean(np.abs(deltas))),
            "p95_absolute_error": float(np.quantile(np.abs(deltas), 0.95)),
            "p99_absolute_error": float(np.quantile(np.abs(deltas), 0.99)),
            "max_absolute_error": float(np.max(np.abs(deltas))),
        },
        "rows": selected,
        "artifacts": {
            "parity_dump": str(parity_dump.resolve()),
            "parity_dump_sha256": _sha256(parity_dump),
            "input_star": str(input_star.resolve()),
            "input_star_sha256": _sha256(input_star),
            "relion_star": str(relion_star.resolve()),
            "relion_star_sha256": _sha256(relion_star),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-dump", type=Path, required=True)
    parser.add_argument("--input-star", type=Path, required=True)
    parser.add_argument("--relion-star", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = build_panel(
        parity_dump=args.parity_dump,
        input_star=args.input_star,
        relion_star=args.relion_star,
        top_n=args.top_n,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
