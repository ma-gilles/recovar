#!/usr/bin/env python3
"""
Capture a run_test_all_metrics all_scores.json into a versioned in-repo baseline folder.

Example:
  python scripts/capture_metrics_baseline.py \
    --scores-json /scratch/.../all_scores.json \
    --name clean_20260219 \
    --volumes-prefix /scratch/.../vol \
    --run-args "--grid-size 128 --n-images 50000 --noise-level 1.0 --contrast-std 0.1"
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Store a metrics baseline JSON and metadata in git.")
    p.add_argument("--scores-json", required=True)
    p.add_argument("--name", required=True, help="Baseline name folder under tests/baselines/run_test_all_metrics/")
    p.add_argument("--volumes-prefix", default="")
    p.add_argument("--run-args", default="")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    src = Path(args.scores_json)
    if not src.exists():
        raise FileNotFoundError(f"scores json not found: {src}")

    out_dir = Path("tests/baselines/run_test_all_metrics") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(src, "r") as f:
        scores = json.load(f)
    with open(out_dir / "all_scores.json", "w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)

    meta = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scores_json": str(src),
        "git_commit": git_commit(),
        "volumes_prefix": args.volumes_prefix,
        "run_args": args.run_args,
        "notes": args.notes,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"Saved baseline to {out_dir}")


if __name__ == "__main__":
    main()
