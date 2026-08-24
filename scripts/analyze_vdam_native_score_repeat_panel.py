#!/usr/bin/env python3
"""Compare one RECOVAR VDAM score boundary with repeated native RELION captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_vdam_native_translation_boundary import analyze  # noqa: E402


def summarize_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    if len(reports) < 2:
        raise ValueError("score repeat audit requires at least two reports")
    if any(report.get("status") != "complete" for report in reports):
        raise ValueError("every score-boundary report must be complete")
    boundaries = [report["comparisons"]["top_pair_score_boundary"] for report in reports]
    native_keys = [
        (tuple(item["native_best"]["mapped_key"]), tuple(item["native_second"]["mapped_key"]))
        for item in boundaries
    ]
    if any(keys != native_keys[0] for keys in native_keys[1:]):
        raise ValueError("native repeats do not share one top-pair identity")
    native_log_odds = np.asarray(
        [item["native_log_odds_best_over_second"] for item in boundaries], dtype=np.float64
    )
    recovar_log_odds = np.asarray(
        [item["recovar_log_odds_same_order"] for item in boundaries], dtype=np.float64
    )
    if not np.all(recovar_log_odds == recovar_log_odds[0]):
        raise ValueError("RECOVAR replay changed across native repeat reports")
    candidate = float(recovar_log_odds[0])
    return {
        "top_pair_mapped_keys": [list(native_keys[0][0]), list(native_keys[0][1])],
        "native_log_odds": [float(value) for value in native_log_odds],
        "native_log_odds_min": float(native_log_odds.min()),
        "native_log_odds_max": float(native_log_odds.max()),
        "native_log_odds_span": float(native_log_odds.max() - native_log_odds.min()),
        "recovar_log_odds": candidate,
        "recovar_inside_native_range": bool(native_log_odds.min() <= candidate <= native_log_odds.max()),
        "minimum_absolute_distance_to_native": float(np.min(np.abs(native_log_odds - candidate))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=8)
    parser.add_argument("--live-score", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat_count < 2:
        parser.error("repeat-count must be at least 2")
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")

    reports = [
        analyze(
            args.panel_root / f"repeat-{index:02d}" / "capture",
            args.live_score,
            full_size=args.full_image_size,
        )
        for index in range(1, args.repeat_count + 1)
    ]
    payload = {
        "schema": "recovar.vdam_native_score_repeat_panel.v1",
        "status": "complete",
        "panel_root": str(args.panel_root.resolve()),
        "repeat_count": args.repeat_count,
        "summary": summarize_reports(reports),
        "reports": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
