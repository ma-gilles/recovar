#!/usr/bin/env python
"""Format the EM-parity ledger files into Markdown tables for PR descriptions.

For EM-scoped PRs, the PR body must include this output rather than the
SPA/ET pipeline tables produced by ``scripts/extract_regression_tables.py``.
The two extractors target different test scopes and are intentionally kept
separate (see recovar/em/CLAUDE.md "Testing" section).

Usage:
  pixi run python scripts/extract_em_parity_tables.py --ledger-root /path/to/fresh/pytest-run [--tier fast|long|all]

Reads:
  <ledger-root>/**/em_parity_quality_{fast,long}_ledger_*.json
  tests/baselines/em_parity_quality_{fast,long}_baseline.json   (optional)
  tests/baselines/em_parity_perf_{fast,long}_baseline.json      (optional)

Without --ledger-root, read historical ledgers directly from tests/baselines.
An explicit root never falls back to historical results. Duplicate ledger names
are rejected; select a single run. Tables report recorded metrics, not scientific
acceptance or proof that every required test executed. Explicit run roots require
complete finite summary metrics and expected per-class arrays for every present
case. Use --require-case to name the cases a launcher must produce; absent cases
otherwise appear explicitly as not measured. Historical partial ledgers remain
readable without --ledger-root, with unavailable values shown as missing.

Writes:
  Markdown tables to stdout. Paste directly into PR description.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _load_ledger(root: Path | None, name: str) -> dict | None:
    if root is None:
        return _load_json(BASELINES_DIR / name)
    matches = sorted(root.rglob(name))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous ledger {name}: {matches}")
    if not matches:
        return None
    # Unlike optional historical baselines, corrupt current results are errors.
    payload = json.loads(matches[0].read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Ledger must contain an object: {matches[0]}")
    return payload


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
    baseline = baseline if _finite_number(baseline) else None
    current = current if _finite_number(current) else None
    base_str = f"{baseline:{fmt}}" if baseline is not None else "—"
    cur_str = f"{current:{fmt}}" if current is not None else "—"
    status = _delta_status(current, baseline, lower_is_better=lower_is_better)
    return f"| {metric} | {base_str} | {cur_str} | {status} |"


def _get_nested(mapping: dict, path: tuple[str, ...]) -> float | None:
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _emit_perf_rows(prefix: str, ledger: dict, baseline: dict) -> int:
    rendered = 0
    metrics: list[tuple[str, float | None, float | None]] = [
        (f"{prefix}_walltime_s", baseline.get(f"{prefix}_walltime_s"), ledger.get(f"{prefix}_walltime_s")),
    ]
    setup_phases = ledger.get(f"{prefix}_recovar_setup_phase_seconds", {}) or ledger.get("setup_phase_seconds", {})
    baseline_setup_phases = baseline.get(f"{prefix}_recovar_setup_phase_seconds", {}) or baseline.get(
        "setup_phase_seconds", {}
    )
    for phase_name in (
        "mask_and_image_cache",
        "state_init",
        "sampling_grid",
        "initial_arrays",
        "direction_prior",
        "noise_radial_init",
        "before_iterations",
    ):
        cur = setup_phases.get(phase_name) if isinstance(setup_phases, dict) else None
        base = baseline_setup_phases.get(phase_name) if isinstance(baseline_setup_phases, dict) else None
        metrics.append((f"{prefix}_setup_{phase_name}_s", base, cur))

    timing_summary = ledger.get(f"{prefix}_recovar_timing_summary", {}) or ledger.get("timing_summary", {})
    baseline_timing = baseline.get(f"{prefix}_recovar_timing_summary", {}) or baseline.get("timing_summary", {})
    for stage_name in ("e_step", "recon", "fsc", "noise_update", "convergence"):
        cur = _get_nested(timing_summary, ("sum_stage_delta_s", stage_name))
        base = _get_nested(baseline_timing, ("sum_stage_delta_s", stage_name))
        metrics.append((f"{prefix}_{stage_name}_s", base, cur))

    for metric, base, cur in metrics:
        if cur is None and metric != f"{prefix}_walltime_s":
            continue
        print(_row(metric, base, cur, lower_is_better=True, fmt=".1f"))
        rendered += 1
    return rendered


# Each row declares the ledger key, whether lower is better, and display format.
# These are reporting requirements, not scientific acceptance thresholds.
CASE_METRICS = {
    "k1_replay": (
        ("k1_replay_half1_corr_vs_relion", False, ".6f"),
        ("k1_replay_half2_corr_vs_relion", False, ".6f"),
        ("k1_replay_pmax_abs_diff", True, ".6f"),
    ),
    "kclass_replay": (
        ("kclass_replay_mean_corr", False, ".6f"),
        ("kclass_replay_pmax_abs_mean", True, ".6f"),
        ("kclass_replay_class_assignment_accuracy", False, ".4f"),
        ("kclass_replay_pmax_abs_max", True, ".6f"),
    ),
    "k1_coldstart": (
        ("k1_coldstart_half1_corr_vs_relion_it003", False, ".6f"),
        ("k1_coldstart_half2_corr_vs_relion_it003", False, ".6f"),
        ("k1_coldstart_pmax_iter3_abs_diff", True, ".6f"),
    ),
    "k1_perturbreplay": (
        ("k1_perturbreplay_half1_corr_vs_relion_it003", False, ".6f"),
        ("k1_perturbreplay_half2_corr_vs_relion_it003", False, ".6f"),
        ("k1_perturbreplay_pmax_iter3_abs_diff", True, ".6f"),
    ),
    "kclass_coldstart": (
        ("kclass_coldstart_mean_corr", False, ".6f"),
        ("kclass_coldstart_worst_class_corr", False, ".6f"),
    ),
    "kclass_strict": (
        ("kclass_strict_mean_corr", False, ".6f"),
        ("kclass_strict_worst_class_corr", False, ".6f"),
        ("kclass_strict_iter3_class_match", False, ".4f"),
    ),
    "kclass_strict_os1": (
        ("kclass_strict_os1_mean_corr", False, ".6f"),
        ("kclass_strict_os1_worst_class_corr", False, ".6f"),
    ),
    "k1_long": (
        ("k1_long_recovar_fsc05_resolution_A", True, ".2f"),
        ("k1_long_relion_fsc05_resolution_A", True, ".2f"),
        ("k1_long_fsc05_resolution_diff_A", True, ".2f"),
        ("k1_long_pmax_diff_max_iter3plus", True, ".4g"),
    ),
    "k1_native_initialmodel": (
        ("k1_native_initialmodel_vdam_it008_corr_vs_gt", False, ".6f"),
        ("k1_native_initialmodel_relion_it008_corr_vs_gt", False, ".6f"),
        ("k1_native_initialmodel_corr_gap_vs_relion_it008", True, ".6f"),
        ("k1_native_initialmodel_vdam_it008_mean_fsc_1_16", False, ".6f"),
        ("k1_native_initialmodel_relion_it008_mean_fsc_1_16", False, ".6f"),
        ("k1_native_initialmodel_fsc_1_16_gap_vs_relion_it008", True, ".6f"),
        ("k1_native_initialmodel_vdam_it001_corr_vs_gt", False, ".6f"),
        ("k1_native_initialmodel_vdam_it001_mean_fsc_1_16", False, ".6f"),
        ("k1_native_initialmodel_vdam_it002_corr_vs_gt", False, ".6f"),
        ("k1_native_initialmodel_vdam_it002_mean_fsc_1_16", False, ".6f"),
        ("k1_native_initialmodel_vdam_vs_relion_it008_corr", False, ".6f"),
        ("k1_native_initialmodel_vdam_vs_relion_it008_mean_fsc_1_8", False, ".6f"),
        ("k1_native_initialmodel_vdam_vs_relion_it008_mean_fsc_1_16", False, ".6f"),
    ),
    "kclass_long": (
        ("kclass_long_mean_corr", False, ".6f"),
        ("kclass_long_class_assignment_accuracy", False, ".4f"),
    ),
}

TIER_CASES = {
    "fast": (
        "k1_replay",
        "kclass_replay",
        "k1_coldstart",
        "k1_perturbreplay",
        "kclass_coldstart",
        "kclass_strict",
        "kclass_strict_os1",
    ),
    "long": ("k1_long", "k1_native_initialmodel", "kclass_long"),
}

# The producer has already applied its class matching. Preserve that order.
PER_CLASS_METRICS = {
    "kclass_replay": ("kclass_replay_per_class_map_corr", 2),
    "kclass_coldstart": ("kclass_coldstart_per_class_corrs_after_hungarian", 4),
    "kclass_strict": ("kclass_strict_per_class_corrs_after_hungarian", 4),
    "kclass_strict_os1": ("kclass_strict_os1_per_class_corrs_after_hungarian", 4),
    "kclass_long": ("kclass_long_per_class_map_corr", 4),
}


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_case(case: str, ledger: dict) -> None:
    required = [key for key, _, _ in CASE_METRICS[case]] + [f"{case}_walltime_s"]
    invalid = [key for key in required if not _finite_number(ledger.get(key))]
    if invalid:
        raise ValueError(f"{case}: missing or non-finite numeric metrics: {invalid}")
    if ledger[f"{case}_walltime_s"] < 0:
        raise ValueError(f"{case}: negative wall time")
    if case in PER_CLASS_METRICS:
        key, count = PER_CLASS_METRICS[case]
        values = ledger.get(key)
        if not isinstance(values, list) or len(values) != count or not all(map(_finite_number, values)):
            raise ValueError(f"{case}: {key} must contain exactly {count} finite class values")


def _read_tier(tier: str, root: Path | None, required_cases=()) -> dict:
    ledgers = {}
    for case in TIER_CASES[tier]:
        ledger = _load_ledger(root, f"em_parity_quality_{tier}_ledger_{case}.json")
        if ledger is not None:
            if root is not None:
                _validate_case(case, ledger)
            ledgers[case] = ledger
        elif case in required_cases:
            raise ValueError(f"Missing required {tier} ledger: {case}")
    return ledgers


def _emit_tier(tier: str, ledgers: dict) -> int:
    if not ledgers:
        print(f"# EM-parity {tier} tier — no ledger files found.", flush=True)
        return 0
    baseline = _load_json(BASELINES_DIR / f"em_parity_quality_{tier}_baseline.json") or {}
    missing = [case for case in TIER_CASES[tier] if case not in ledgers]
    print(f"### EM-parity recorded metrics — {tier} tier\n")
    print("Reporting completeness does not establish scientific acceptance.")
    print("Legacy map correlations are diagnostics; FSC gates and test outcomes need separate review.")
    if missing:
        print("Cases not measured in this report: " + ", ".join(missing))
    print("\n| Metric | Baseline | Current | Status |")
    print("|--------|----------|---------|--------|")
    rendered = 0
    for case, ledger in ledgers.items():
        for key, lower_is_better, fmt in CASE_METRICS[case]:
            rendered += _emit_metric(key, ledger, baseline, lower_is_better, fmt)
        if case in PER_CLASS_METRICS:
            key, _ = PER_CLASS_METRICS[case]
            values = ledger.get(key, [])
            if isinstance(values, list):
                for index, value in enumerate(values):
                    if _finite_number(value):
                        print(_row(f"{key}[{index}]", None, value, fmt=".6f"))
                        rendered += 1
    print(f"\n### EM-parity Performance — {tier} tier")
    print("| Metric | Baseline | Current | Status |")
    print("|--------|----------|---------|--------|")
    for case, ledger in ledgers.items():
        _emit_perf_rows(case, ledger, baseline)
    return rendered


def _emit_metric(key: str, ledger: dict, baseline: dict, lower_is_better: bool, fmt: str) -> int:
    cur = ledger.get(key)
    base = baseline.get(key)
    if not _finite_number(cur):
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
    parser.add_argument("--ledger-root", type=Path, help="One run directory, searched recursively for ledgers")
    parser.add_argument(
        "--require-case",
        nargs="+",
        choices=tuple(CASE_METRICS),
        default=[],
        help="Cases that must be present; requires --ledger-root",
    )
    args = parser.parse_args()
    if args.ledger_root is not None and not args.ledger_root.is_dir():
        parser.error(f"Ledger root is not a directory: {args.ledger_root}")

    tiers = tuple(TIER_CASES) if args.tier == "all" else (args.tier,)
    allowed_cases = {case for tier in tiers for case in TIER_CASES[tier]}
    if args.require_case and args.ledger_root is None:
        parser.error("--require-case requires --ledger-root; historical ledgers cannot satisfy current runs")
    if set(args.require_case) - allowed_cases:
        parser.error("Required cases must belong to the selected tier")
    try:
        reports = {tier: _read_tier(tier, args.ledger_root, args.require_case) for tier in tiers}
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    rendered = sum(_emit_tier(tier, ledgers) for tier, ledgers in reports.items())
    return 0 if rendered > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
