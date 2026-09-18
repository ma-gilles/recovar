#!/usr/bin/env python3
"""Summarize CUDA kernel/API activity and GPU busy time from an Nsight SQLite export."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _string_ids(connection: sqlite3.Connection, tables: set[str]) -> dict[int, str]:
    if "StringIds" not in tables:
        return {}
    return {int(row[0]): str(row[1]) for row in connection.execute("SELECT id, value FROM StringIds")}


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _name(
    row: sqlite3.Row,
    *,
    candidates: tuple[str, ...],
    strings: dict[int, str],
) -> str:
    for column in candidates:
        if column not in row.keys() or row[column] is None:
            continue
        value = row[column]
        if isinstance(value, int) and value in strings:
            return strings[value]
        return str(value)
    return "unknown"


def summarize(sqlite_path: Path) -> dict[str, object]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        tables = _tables(connection)
        kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
        if kernel_table not in tables:
            raise ValueError(f"Nsight export has no {kernel_table}: {sqlite_path}")
        strings = _string_ids(connection, tables)
        kernel_columns = _columns(connection, kernel_table)
        required = {"start", "end"}
        if not required.issubset(kernel_columns):
            raise ValueError(f"kernel table lacks required columns: {sorted(required - kernel_columns)}")
        kernels = list(connection.execute(f'SELECT * FROM "{kernel_table}"'))
        if not kernels:
            raise ValueError(f"Nsight export has no CUDA kernels: {sqlite_path}")

        intervals_by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
        kernel_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "total_ns": 0, "max_ns": 0})
        signature_groups: dict[tuple[object, ...], dict[str, object]] = {}
        shape_columns = tuple(
            column for column in ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ") if column in kernel_columns
        )
        for row in kernels:
            start, end = int(row["start"]), int(row["end"])
            if end < start:
                raise ValueError("Nsight kernel interval has negative duration")
            duration = end - start
            device = int(row["deviceId"]) if "deviceId" in row.keys() else 0
            intervals_by_device[device].append((start, end))
            name = _name(
                row,
                candidates=("shortName", "demangledName", "mangledName", "name"),
                strings=strings,
            )
            group = kernel_groups[name]
            group["count"] += 1
            group["total_ns"] += duration
            group["max_ns"] = max(group["max_ns"], duration)
            shape = tuple(row[column] for column in shape_columns)
            signature_key = (name, device, *shape)
            signature = signature_groups.setdefault(
                signature_key,
                {
                    "name": name,
                    "device_id": device,
                    "shape": {column: row[column] for column in shape_columns},
                    "count": 0,
                    "total_ns": 0,
                    "max_ns": 0,
                },
            )
            signature["count"] = int(signature["count"]) + 1
            signature["total_ns"] = int(signature["total_ns"]) + duration
            signature["max_ns"] = max(int(signature["max_ns"]), duration)

        runtime_table = "CUPTI_ACTIVITY_KIND_RUNTIME"
        api_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "total_ns": 0, "max_ns": 0})
        capture_intervals: list[tuple[int, int]] = []
        if runtime_table in tables:
            runtime_columns = _columns(connection, runtime_table)
            if required.issubset(runtime_columns):
                for row in connection.execute(f'SELECT * FROM "{runtime_table}"'):
                    start, end = int(row["start"]), int(row["end"])
                    if end < start:
                        continue
                    capture_intervals.append((start, end))
                    name = _name(
                        row,
                        candidates=("nameId", "name", "cbid"),
                        strings=strings,
                    )
                    group = api_groups[name]
                    duration = end - start
                    group["count"] += 1
                    group["total_ns"] += duration
                    group["max_ns"] = max(group["max_ns"], duration)

        all_kernel_intervals = [interval for values in intervals_by_device.values() for interval in values]
        window_intervals = capture_intervals or all_kernel_intervals
        capture_start = min(start for start, _ in window_intervals)
        capture_end = max(end for _, end in window_intervals)
        capture_span_ns = capture_end - capture_start
        devices = {}
        for device, intervals in sorted(intervals_by_device.items()):
            busy_ns = _union_ns(intervals)
            devices[str(device)] = {
                "kernel_count": len(intervals),
                "gpu_busy_ns": busy_ns,
                "gpu_idle_ns_within_capture": max(0, capture_span_ns - busy_ns),
                "gpu_busy_fraction_within_capture": (float(busy_ns / capture_span_ns) if capture_span_ns else 0.0),
            }

        def _ranked(groups: dict[str, dict[str, int]]) -> list[dict[str, object]]:
            return [
                {
                    "name": name,
                    **values,
                    "mean_ns": float(values["total_ns"] / values["count"]),
                }
                for name, values in sorted(groups.items(), key=lambda item: (-item[1]["total_ns"], item[0]))
            ]

        signatures = sorted(
            signature_groups.values(),
            key=lambda value: (-int(value["total_ns"]), str(value["name"])),
        )
        for signature in signatures:
            signature["mean_ns"] = float(int(signature["total_ns"]) / int(signature["count"]))
        return {
            "schema": "recovar.vdam_nsys_sqlite_summary.v1",
            "sqlite": str(sqlite_path.resolve()),
            "capture_span_source": "cuda_api" if capture_intervals else "cuda_kernel",
            "capture_start_ns": capture_start,
            "capture_end_ns": capture_end,
            "capture_span_ns": capture_span_ns,
            "devices": devices,
            "kernels": _ranked(kernel_groups),
            "kernel_signatures": signatures,
            "cuda_apis": _ranked(api_groups),
        }
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.sqlite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("capture_span_ns", "devices")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
