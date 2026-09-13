#!/usr/bin/env python3
"""Aggregate cold-only JAX compile call-site records from a VDAM profile."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_vdam_late_profile_pair import _compile_log_summary
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from analyze_vdam_late_profile_pair import _compile_log_summary


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON mapping: {path}")
    return value


def _ranked(
    groups: dict[tuple[object, ...], list[float]],
    *,
    fields: tuple[str, ...],
    top: int,
) -> list[dict[str, object]]:
    rows = []
    for key, durations in groups.items():
        total_s = sum(durations)
        row: dict[str, object] = dict(zip(fields, key))
        row.update(
            {
                "count": len(durations),
                "total_s": total_s,
                "mean_s": total_s / len(durations),
                "max_s": max(durations),
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["total_s"]),
            *(str(row[field]) for field in fields),
        )
    )
    return rows[:top]


def analyze(root: Path, *, top: int = 30) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if top <= 0:
        raise ValueError("top must be positive")
    if not (root / "COMPLETED").is_file():
        raise RuntimeError(f"profile root has no COMPLETED marker: {root}")
    run = _load_mapping(root / "provenance" / "run.json")
    if run.get("cold_compile_attribution") is not True:
        raise RuntimeError("run is not marked as cold compile attribution")
    if run.get("timing_truth_allowed") is not False:
        raise RuntimeError("cold compile attribution must be excluded from timing truth")

    records_path = root / "provenance" / "cold_compile_calls.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    records = []
    for line_number, raw_line in enumerate(records_path.read_text().splitlines(), start=1):
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise TypeError(f"compile record {line_number} is not a mapping")
        records.append(value)
    if not records:
        raise RuntimeError("cold compile attribution log is empty")
    sequences = [int(record["sequence"]) for record in records]
    if sequences != list(range(len(records))):
        raise RuntimeError("cold compile attribution sequences are not contiguous")

    by_module: dict[tuple[object, ...], list[float]] = defaultdict(list)
    by_callsite: dict[tuple[object, ...], list[float]] = defaultdict(list)
    by_file: dict[tuple[object, ...], list[float]] = defaultdict(list)
    by_status: dict[str, list[float]] = defaultdict(list)
    missing_callsites = 0
    for record in records:
        elapsed_s = float(record["elapsed_s"])
        if elapsed_s < 0.0:
            raise RuntimeError("cold compile attribution contains negative time")
        status = str(record["cache_status"])
        by_status[status].append(elapsed_s)
        by_module[(str(record["module"]), status)].append(elapsed_s)
        callsite = record.get("callsite")
        if not isinstance(callsite, dict):
            missing_callsites += 1
            key = (None, None, None, None, status)
        else:
            key = (
                callsite.get("file"),
                callsite.get("line"),
                callsite.get("function"),
                callsite.get("source"),
                status,
            )
            by_file[(callsite.get("file"), status)].append(elapsed_s)
        by_callsite[key].append(elapsed_s)

    total_s = sum(float(record["elapsed_s"]) for record in records)
    stderr_compilation = _compile_log_summary(
        root / "recovar_profiled.stderr", top=top
    )
    return {
        "schema": "recovar.vdam_cold_compile_attribution.v1",
        "classification": "diagnostic_performance_only",
        "timing_truth": False,
        "root": str(root),
        "job_id": run.get("job_id"),
        "git_head": run.get("git_head"),
        "profiled_iteration": run.get("profiled_iteration"),
        "records": {
            "path": str(records_path),
            "count": len(records),
            "missing_callsite_count": missing_callsites,
            "total_s": total_s,
            "by_cache_status": [
                {
                    "status": status,
                    "count": len(durations),
                    "total_s": sum(durations),
                    "mean_s": sum(durations) / len(durations),
                    "max_s": max(durations),
                }
                for status, durations in sorted(
                    by_status.items(), key=lambda item: (-sum(item[1]), item[0])
                )
            ],
        },
        "top_callsites": _ranked(
            by_callsite,
            fields=("file", "line", "function", "source", "cache_status"),
            top=top,
        ),
        "top_files": _ranked(
            by_file, fields=("file", "cache_status"), top=top
        ),
        "top_modules": _ranked(
            by_module, fields=("module", "cache_status"), top=top
        ),
        "stderr_xla_compilation": stderr_compilation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)
    report = analyze(args.root, top=args.top)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "records": report["records"],
                "top_callsites": report["top_callsites"][:15],
                "top_modules": report["top_modules"][:15],
                "stderr_xla_compilation": report["stderr_xla_compilation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
