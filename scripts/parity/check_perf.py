#!/usr/bin/env python3
"""Check per-iter parity-dump wall times against a perf baseline JSON.

Reads ``iter_NNN.npz`` files from ``--dump-dir`` (written by
``recovar.em.dense_single_volume.parity_dump``) and compares the
``wall_time_s`` field against ``--baseline``'s ``per_iter_seconds_total``
table. ``stage_seconds_*`` fields are cumulative stage-completion stamps
since iteration start, not per-stage duration deltas. Emits one line per
checked iter with a status:

    iter 4: 251.3s vs baseline 240.0s (+5%) OK
    iter 5: 248.7s vs baseline 240.0s (+4%) OK
    iter 6: 1487.2s vs baseline 240.0s (+520%) REGRESSED

Status thresholds (configured via the baseline file):
    OK         : actual <= baseline * tolerance_multiplier
    WARN       : tolerance_multiplier <= ratio < regression_threshold_multiplier
    REGRESSED  : ratio >= regression_threshold_multiplier

Usage:
    pixi run python scripts/parity/check_perf.py \
        --dump-dir _agent_scratch/parity/recovar_HEAD \
        --baseline tests/baselines/parity/perf_baseline_5k_128_a100.json \
        [--single-iter N] \
        [--exit-code-on-regression]

Single-iter mode: when ``--single-iter N`` is passed, only check iter N.
If ``iter_NNN.npz`` does not yet exist, prints "iter N: not yet dumped"
and exits 0 (so the caller can poll without spurious failures).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_baseline(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _iter_npz_path(dump_dir: Path, iter_num: int) -> Path:
    return dump_dir / f"iter_{iter_num:03d}.npz"


def _read_iter_wall_time(npz_path: Path) -> tuple[float | None, dict[str, float]]:
    """Return (wall_time_s, stage_seconds) from a dump npz, or (None, {}) if absent."""
    data = np.load(npz_path, allow_pickle=False)
    wall = float(data["wall_time_s"]) if "wall_time_s" in data.files else None
    stages = {}
    for name in data.files:
        if name.startswith("stage_seconds_"):
            stages[name[len("stage_seconds_") :]] = float(data[name])
    return wall, stages


def _classify(actual: float, baseline: float, warn_x: float, regress_x: float) -> str:
    """Return one of "OK" / "WARN" / "REGRESSED"."""
    if baseline <= 0:
        return "UNKNOWN"
    ratio = actual / baseline
    if ratio >= regress_x:
        return "REGRESSED"
    if ratio >= warn_x:
        return "WARN"
    return "OK"


def _format_pct(actual: float, baseline: float) -> str:
    if baseline <= 0:
        return "n/a"
    pct = (actual - baseline) / baseline * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def check_iter(
    dump_dir: Path,
    baseline: dict,
    iter_num: int,
) -> tuple[str, str | None]:
    """Check a single iter; returns (status_line, status_class_or_None_if_missing)."""
    per_iter_total = baseline.get("per_iter_seconds_total", {})
    per_iter_stages = baseline.get("per_iter_seconds_stages", {})
    warn_x = float(baseline.get("tolerance_multiplier", 2.0))
    regress_x = float(baseline.get("regression_threshold_multiplier", 3.0))

    npz_path = _iter_npz_path(dump_dir, iter_num)
    if not npz_path.exists():
        return f"iter {iter_num}: not yet dumped", None

    wall, stages = _read_iter_wall_time(npz_path)
    if wall is None:
        return f"iter {iter_num}: dump exists but no wall_time_s field", "UNKNOWN"

    base = per_iter_total.get(str(iter_num))
    if base is None:
        return (
            f"iter {iter_num}: {wall:.1f}s vs baseline n/a (no entry) — not checked",
            "UNKNOWN",
        )

    base = float(base)
    status = _classify(wall, base, warn_x, regress_x)
    pct = _format_pct(wall, base)
    line = f"iter {iter_num}: {wall:.1f}s vs baseline {base:.1f}s ({pct}) {status}"

    # Stage breakdown (compact, only when we have stage entries to compare)
    base_stages = per_iter_stages.get(str(iter_num), {}) if isinstance(per_iter_stages, dict) else {}
    if stages and base_stages:
        stage_bits = []
        for s_name in sorted(stages.keys()):
            s_actual = stages[s_name]
            s_base = base_stages.get(s_name)
            if s_base is None:
                stage_bits.append(f"{s_name}={s_actual:.1f}s(no_base)")
            else:
                s_status = _classify(s_actual, float(s_base), warn_x, regress_x)
                stage_bits.append(f"{s_name}={s_actual:.1f}s/{float(s_base):.1f}s[{s_status}]")
        line += "  stages: " + " ".join(stage_bits)

    return line, status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument(
        "--single-iter",
        type=int,
        default=None,
        help="Check only this iter (RELION-iter index). No-op if its npz isn't there yet.",
    )
    ap.add_argument(
        "--exit-code-on-regression", action="store_true", help="Exit nonzero if any checked iter is REGRESSED."
    )
    args = ap.parse_args()

    baseline = _load_baseline(args.baseline)

    statuses: list[str] = []

    if args.single_iter is not None:
        line, status = check_iter(args.dump_dir, baseline, args.single_iter)
        print(line, flush=True)
        if status is not None:
            statuses.append(status)
    else:
        # Walk all iters in the baseline that have an npz on disk.
        per_iter_total = baseline.get("per_iter_seconds_total", {})
        all_iters = sorted(int(k) for k in per_iter_total.keys())
        # Also pick up any extra iters that exist on disk but aren't in the baseline.
        if args.dump_dir.exists():
            for p in args.dump_dir.glob("iter_*.npz"):
                stem = p.stem  # iter_NNN
                try:
                    n = int(stem.split("_", 1)[1])
                    if n not in all_iters:
                        all_iters.append(n)
                except ValueError:
                    pass
            all_iters = sorted(set(all_iters))
        for iter_num in all_iters:
            line, status = check_iter(args.dump_dir, baseline, iter_num)
            print(line, flush=True)
            if status is not None:
                statuses.append(status)

    if args.exit_code_on_regression and "REGRESSED" in statuses:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
