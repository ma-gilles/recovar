#!/usr/bin/env python3
"""Validate and compare bounded all-particle K=4 winner summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovar.em.global_winner_analysis import (
    analyze_summaries,
    load_recovar_summary,
    load_relion_summary,
)


def _label_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovar", action="append", type=_label_path, default=[])
    parser.add_argument("--relion", action="append", type=_label_path, default=[])
    parser.add_argument("--relion-data-star", action="append", type=_label_path, default=[])
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--relion-executable", type=Path)
    parser.add_argument("--dispatch-log", action="append", type=_label_path, default=[])
    parser.add_argument("--dispatch-schedule", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    relion_labels = [label for label, _path in args.relion]
    data_stars = dict(args.relion_data_star)
    dispatch_logs = dict(args.dispatch_log)
    if (
        len(set(relion_labels)) != len(relion_labels)
        or len(data_stars) != len(args.relion_data_star)
        or len(dispatch_logs) != len(args.dispatch_log)
    ):
        parser.error("RELION summary, data-STAR, and dispatch-log labels must be unique")
    if args.relion and (
        args.input_manifest is None
        or args.relion_executable is None
        or args.dispatch_schedule is None
        or set(data_stars) != set(relion_labels)
        or set(dispatch_logs) != set(relion_labels)
    ):
        parser.error(
            "each --relion LABEL=PATH requires matching --relion-data-star LABEL=PATH and "
            "--dispatch-log LABEL=PATH, plus --input-manifest, --relion-executable, and "
            "--dispatch-schedule"
        )

    summaries = [load_recovar_summary(path, label=label) for label, path in args.recovar]
    summaries.extend(
        load_relion_summary(
            path,
            data_star=data_stars[label],
            input_manifest=args.input_manifest,
            executable=args.relion_executable,
            dispatch_log=dispatch_logs[label],
            dispatch_schedule=args.dispatch_schedule,
            label=label,
        )
        for label, path in args.relion
    )
    report = analyze_summaries(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "identity_count": report["identity_count"]}))


if __name__ == "__main__":
    main()
