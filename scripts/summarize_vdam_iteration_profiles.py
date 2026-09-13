#!/usr/bin/env python3
"""Summarize opt-in VDAM per-iteration timing metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


_ITERATION_RE = re.compile(r"run_it(\d+)_recovar_meta\.json$")


def _numeric_scalars(value: object, *, prefix: str = "") -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            out[name] = float(item)
    return out


def _iteration_row(path: Path) -> dict[str, object] | None:
    match = _ITERATION_RE.search(path.name)
    if match is None:
        return None
    meta = json.loads(path.read_text())
    iteration_profile = _numeric_scalars(meta.get("vdam_iteration_profile_summary"))
    if not iteration_profile:
        return None
    row: dict[str, object] = {
        "iteration": int(match.group(1)),
        "current_size": int(meta["current_size"]),
        "healpix_order": int(meta["healpix_order"]),
        "n_rotations": int(meta["n_rotations"]),
        "n_translations": int(meta["n_translations"]),
        **iteration_profile,
        **_numeric_scalars(meta.get("sparse_pass2_profile_summary"), prefix="sparse_"),
    }
    for key, value in meta.items():
        if key.startswith("halfset_") and key.endswith("_profile_summary"):
            row.update(_numeric_scalars(value, prefix=f"{key}_"))
    return row


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key != "iteration" and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    timings = {}
    for key in numeric_keys:
        if not (key.endswith("_time_s") or key.endswith("_s")):
            continue
        values = [float(row[key]) for row in rows if key in row]
        timings[key] = {
            "count": len(values),
            "sum_s": float(sum(values)),
            "mean_s": float(mean(values)),
            "max_s": float(max(values)),
        }
    return {
        "iteration_count": len(rows),
        "first_iteration": min(int(row["iteration"]) for row in rows),
        "last_iteration": max(int(row["iteration"]) for row in rows),
        "timings": timings,
    }


def summarize(recovar_dir: Path) -> dict[str, object]:
    rows = [
        row
        for path in sorted(recovar_dir.glob("run_it*_recovar_meta.json"))
        if (row := _iteration_row(path)) is not None
    ]
    if not rows:
        raise ValueError(f"no profiled VDAM iteration metadata found in {recovar_dir}")
    phase_specs = {
        "pre_transition_1_69": (1, 69),
        "accuracy_transition_70_89": (70, 89),
        "adaptive_and_late_90_plus": (90, max(int(row["iteration"]) for row in rows)),
    }
    phases = {}
    for name, (start, stop) in phase_specs.items():
        selected = [row for row in rows if start <= int(row["iteration"]) <= stop]
        if selected:
            phases[name] = _aggregate(selected)
    return {
        "schema": "recovar.vdam_iteration_profile_summary.v1",
        "recovar_dir": str(recovar_dir.resolve()),
        "all_iterations": _aggregate(rows),
        "phases": phases,
        "iterations": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.recovar_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["all_iterations"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
