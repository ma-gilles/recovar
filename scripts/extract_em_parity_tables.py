#!/usr/bin/env python
"""Format the EM-parity ledger files into Markdown tables for PR descriptions.

For EM-scoped PRs, the PR body must include this output rather than the
SPA/ET pipeline tables produced by ``scripts/extract_regression_tables.py``.
The two extractors target different test scopes and are intentionally kept
separate (see recovar/em/CLAUDE.md "Testing" section).

Usage:
  pixi run python scripts/extract_em_parity_tables.py [--tier fast|long|all]

Reads:
  tests/baselines/em_parity_quality_{fast,long}_ledger_*.json
  tests/baselines/em_parity_quality_{fast,long}_baseline.json   (optional)
  tests/baselines/em_parity_perf_{fast,long}_baseline.json      (optional)

Writes:
  Markdown tables to stdout. Paste directly into PR description.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "tests" / "baselines"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _delta_status(
    current: float | None, baseline: float | None, lower_is_better: bool, threshold_pct: float = 1.0
) -> str:
    if current is None or baseline is None:
        return "N/A"
    delta_pct = (current - baseline) / max(abs(baseline), 1e-12) * 100.0
    arrow = "↑" if delta_pct > 0 else "↓"
    abs_pct = abs(delta_pct)
    regressed = (lower_is_better and delta_pct > threshold_pct) or (not lower_is_better and delta_pct < -threshold_pct)
    if regressed:
        return f"{delta_pct:+.2f}% {arrow} **REGRESSED**"
    if abs_pct < 0.05:
        return "OK"
    return f"{delta_pct:+.2f}% {arrow} OK"


def _row(
    metric: str, baseline: float | None, current: float | None, lower_is_better: bool = False, fmt: str = ".4f"
) -> str:
    base_str = f"{baseline:{fmt}}" if baseline is not None else "—"
    cur_str = f"{current:{fmt}}" if current is not None else "—"
    status = _delta_status(current, baseline, lower_is_better=lower_is_better)
    return f"| {metric} | {base_str} | {cur_str} | {status} |"


def emit_fast_tier() -> int:
    """Emit the fast-tier table. Returns number of metrics rendered."""
    k1_ledger = _load_json(BASELINES_DIR / "em_parity_quality_fast_ledger_k1_replay.json")
    kclass_ledger = _load_json(BASELINES_DIR / "em_parity_quality_fast_ledger_kclass_replay.json")
    baseline = _load_json(BASELINES_DIR / "em_parity_quality_fast_baseline.json") or {}

    if not k1_ledger and not kclass_ledger:
        print("# EM-parity fast tier — no ledger files found.", flush=True)
        print("# Run: pixi run pytest tests/integration/test_em_parity_fast.py", flush=True)
        return 0

    print("### EM-parity Quality Comparison — fast tier (5k 128² replay)\n")
    print("| Metric | Baseline | Current | Status |")
    print("|--------|----------|---------|--------|")
    rendered = 0
    if k1_ledger:
        rendered += _emit_metric(
            "k1_replay_half1_corr_vs_relion", k1_ledger, baseline, lower_is_better=False, fmt=".6f"
        )
        rendered += _emit_metric(
            "k1_replay_half2_corr_vs_relion", k1_ledger, baseline, lower_is_better=False, fmt=".6f"
        )
        rendered += _emit_metric("k1_replay_pmax_abs_diff", k1_ledger, baseline, lower_is_better=True, fmt=".6f")
    if kclass_ledger:
        rendered += _emit_metric("kclass_replay_mean_corr", kclass_ledger, baseline, lower_is_better=False, fmt=".6f")
        rendered += _emit_metric(
            "kclass_replay_pmax_abs_mean", kclass_ledger, baseline, lower_is_better=True, fmt=".6f"
        )
        rendered += _emit_metric(
            "kclass_replay_class_assignment_accuracy", kclass_ledger, baseline, lower_is_better=False, fmt=".4f"
        )

    if k1_ledger or kclass_ledger:
        print("\n### EM-parity Performance — fast tier")
        print("| Stage | Walltime (s) |")
        print("|-------|-------------:|")
        if k1_ledger:
            print(f"| k1_replay (5k 128²) | {k1_ledger.get('k1_replay_walltime_s', 0):.1f} |")
        if kclass_ledger:
            print(f"| kclass_replay (5k 128² K=2) | {kclass_ledger.get('kclass_replay_walltime_s', 0):.1f} |")
    return rendered


def emit_long_tier() -> int:
    """Emit the long-tier table. Returns number of metrics rendered."""
    k1_ledger = _load_json(BASELINES_DIR / "em_parity_quality_long_ledger_k1_long.json")
    kclass_ledger = _load_json(BASELINES_DIR / "em_parity_quality_long_ledger_kclass_long.json")
    baseline = _load_json(BASELINES_DIR / "em_parity_quality_long_baseline.json") or {}

    if not k1_ledger and not kclass_ledger:
        print("# EM-parity long tier — no ledger files found.", flush=True)
        print("# Run: ./scripts/run_em_parity_long_slurm.sh", flush=True)
        return 0

    print("### EM-parity Quality Comparison — long tier (50k 256² ab-initio)\n")
    print("| Metric | Baseline | Current | Status |")
    print("|--------|----------|---------|--------|")
    rendered = 0
    if k1_ledger:
        rendered += _emit_metric(
            "k1_long_recovar_fsc05_resolution_A", k1_ledger, baseline, lower_is_better=True, fmt=".2f"
        )
        rendered += _emit_metric(
            "k1_long_relion_fsc05_resolution_A", k1_ledger, baseline, lower_is_better=True, fmt=".2f"
        )
        rendered += _emit_metric(
            "k1_long_fsc05_resolution_diff_A", k1_ledger, baseline, lower_is_better=True, fmt=".2f"
        )
        rendered += _emit_metric(
            "k1_long_pmax_diff_max_iter3plus", k1_ledger, baseline, lower_is_better=True, fmt=".4g"
        )
    if kclass_ledger:
        rendered += _emit_metric("kclass_long_mean_corr", kclass_ledger, baseline, lower_is_better=False, fmt=".6f")
        rendered += _emit_metric(
            "kclass_long_class_assignment_accuracy", kclass_ledger, baseline, lower_is_better=False, fmt=".4f"
        )

    if k1_ledger or kclass_ledger:
        print("\n### EM-parity Performance — long tier")
        print("| Stage | Walltime (s) |")
        print("|-------|-------------:|")
        if k1_ledger:
            print(f"| k1_long (50k 256² 15-iter) | {k1_ledger.get('k1_long_walltime_s', 0):.1f} |")
        if kclass_ledger:
            print(f"| kclass_long (50k 256² K=4 15-iter) | {kclass_ledger.get('kclass_long_walltime_s', 0):.1f} |")
    return rendered


def _emit_metric(key: str, ledger: dict, baseline: dict, lower_is_better: bool, fmt: str) -> int:
    cur = ledger.get(key)
    base = baseline.get(key)
    if not isinstance(cur, (int, float)):
        return 0
    print(_row(key, base, cur, lower_is_better=lower_is_better, fmt=fmt))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=("fast", "long", "all"),
        default="all",
        help="Which tier to emit. Default: all",
    )
    args = parser.parse_args()

    n_fast = 0
    n_long = 0
    if args.tier in ("fast", "all"):
        n_fast = emit_fast_tier()
        if args.tier == "all":
            print()
    if args.tier in ("long", "all"):
        n_long = emit_long_tier()

    return 0 if (n_fast + n_long) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
