#!/usr/bin/env python3
"""Extract quality and performance regression tables from long-test results.

Usage:
    python scripts/extract_regression_tables.py <test_results_dir>

Compares current scores against tests/baselines/ and prints markdown tables
suitable for pasting into a PR description.
"""
import json
import sys
import os
from pathlib import Path


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _pct_change(baseline, current):
    if baseline == 0:
        return 0.0
    return (current - baseline) / abs(baseline) * 100


def _fmt_change(pct, higher_is_better=True):
    if abs(pct) < 0.01:
        return "0.0%"
    arrow = "↑" if pct > 0 else "↓"
    good = (pct > 0 and higher_is_better) or (pct < 0 and not higher_is_better)
    return f"{pct:+.2f}% {arrow}"


def quality_table(current_scores_path, baseline_scores_path, label):
    """Generate quality comparison table."""
    if not os.path.exists(current_scores_path) or not os.path.exists(baseline_scores_path):
        return f"### {label} Quality — files not found\n"

    cur = _load_json(current_scores_path)
    base = _load_json(baseline_scores_path)

    # Metrics where higher is better
    higher_better = {"svd_relative_variance", "noise_correlation", "mean_fsc",
                     "variance_fsc", "variance_fourier_fsc", "variance_spatial_fsc"}
    # Metrics where lower is better
    lower_better = {"contrast_abs_error", "embedding_squared_error",
                    "noise_max_relative_error", "noise_mean_relative_error",
                    "noise_median_relative_error"}

    lines = [f"### {label} Quality"]
    lines.append("| Metric | Baseline | Current | Change | Status |")
    lines.append("|--------|----------|---------|--------|--------|")

    for k in sorted(base.keys()):
        b = base[k]
        c = cur.get(k)
        if not isinstance(b, (int, float)) or not isinstance(c, (int, float)):
            continue
        pct = _pct_change(b, c)

        hib = any(h in k for h in higher_better)
        lib = any(h in k for h in lower_better)

        # Determine if change is better or worse
        if abs(pct) < 0.01:
            direction = "same"
        elif hib:
            direction = "better" if pct > 0 else "worse"
        elif lib:
            direction = "better" if pct < 0 else "worse"
        else:
            direction = f"{pct:+.2f}%"

        if direction == "same":
            change_str = "0.0% (same)"
        elif direction in ("better", "worse"):
            change_str = f"{pct:+.2f}% ({direction})"
        else:
            change_str = direction

        if direction == "worse" and abs(pct) > 5:
            status = "**REGRESSED**"
        else:
            status = "OK"

        lines.append(f"| {k} | {b:.6f} | {c:.6f} | {change_str} | {status} |")

    return "\n".join(lines)


def perf_table(baseline_perf_path, current_perf_path, label):
    """Generate performance comparison table with actual timings."""
    if not os.path.exists(baseline_perf_path):
        return f"### {label} Performance — baseline not found\n"

    base = _load_json(baseline_perf_path)
    hw = base.get("NVIDIA H100 80GB HBM3", base.get("NVIDIA A100-SXM4-80GB", {}))
    base_stages = hw.get("stages", {})

    # Load current perf record (saved by the test)
    cur_stages = {}
    if current_perf_path and os.path.exists(current_perf_path):
        cur_data = _load_json(current_perf_path)
        # The record may be nested under a GPU key or flat
        if "stages" in cur_data:
            cur_stages = cur_data["stages"]
        else:
            for v in cur_data.values():
                if isinstance(v, dict) and "stages" in v:
                    cur_stages = v["stages"]
                    break

    lines = [f"### {label} Performance"]
    lines.append("| Stage | Metric | Baseline | Current | Change | Status |")
    lines.append("|-------|--------|----------|---------|--------|--------|")

    for stage_name in ["dataset_generation", "pipeline", "compute_state", "metrics"]:
        b_stage = base_stages.get(stage_name, {})
        c_stage = cur_stages.get(stage_name, {})

        for metric, base_key, unit, lower_better in [
            ("wall_time", "wall_seconds", "s", True),
            ("GPU_memory", "peak_gpu_memory_gb", "GB", True),
            ("CPU_memory", "peak_cpu_memory_gb", "GB", True),
        ]:
            base_val = b_stage.get(base_key)
            cur_val = c_stage.get(base_key)
            if base_val is None:
                continue

            if cur_val is not None:
                pct = _pct_change(base_val, cur_val)
                if abs(pct) < 0.01:
                    change_str = "0.0% (same)"
                elif lower_better:
                    direction = "better" if pct < 0 else "worse"
                    change_str = f"{pct:+.1f}% ({direction})"
                else:
                    change_str = f"{pct:+.1f}%"
                status = "**REGRESSED**" if pct > 10 and lower_better else "OK"
                lines.append(f"| {stage_name} | {metric} | {base_val:.1f}{unit} | {cur_val:.1f}{unit} | {change_str} | {status} |")
            else:
                lines.append(f"| {stage_name} | {metric} | {base_val:.1f}{unit} | — | — | — |")

    return "\n".join(lines)


def main():
    repo_root = Path(__file__).parent.parent
    baselines = repo_root / "tests" / "baselines"

    # Find the most recent test tmp dirs
    tmp_base = repo_root / ".tmp"

    # Quality tables from all_scores.json
    # SPA
    spa_scores = list(tmp_base.glob("**/test_run_test_all_metrics_regr*/current/test_dataset/metrics_plot/all_scores.json"))
    spa_baseline = baselines / "run_test_all_metrics" / "long_generated" / "all_scores.json"

    # ET
    et_scores = list(tmp_base.glob("**/test_run_test_all_metrics_cryo*/current_cryo_et/test_dataset/metrics_plot/all_scores.json"))
    et_baseline = baselines / "run_test_all_metrics" / "long_generated" / "all_scores_cryo_et.json"

    print("## Regression Tables\n")

    if spa_scores:
        latest = sorted(spa_scores, key=lambda p: p.stat().st_mtime)[-1]
        print(quality_table(str(latest), str(spa_baseline), "SPA"))
        print()

    if et_scores:
        latest = sorted(et_scores, key=lambda p: p.stat().st_mtime)[-1]
        print(quality_table(str(latest), str(et_baseline), "Cryo-ET"))
        print()

    # Perf tables — read saved current_perf_record_*.json from test output
    spa_perf_base = baselines / "run_test_all_metrics" / "long_generated" / "perf_baseline_spa.json"
    et_perf_base = baselines / "run_test_all_metrics" / "long_generated" / "perf_baseline_cryo_et.json"

    spa_perf_cur = sorted(
        tmp_base.glob("**/current_perf_record_spa.json"),
        key=lambda p: p.stat().st_mtime,
    )
    et_perf_cur = sorted(
        tmp_base.glob("**/current_perf_record_cryo_et.json"),
        key=lambda p: p.stat().st_mtime,
    )

    spa_perf = str(spa_perf_cur[-1]) if spa_perf_cur else None
    et_perf = str(et_perf_cur[-1]) if et_perf_cur else None

    print(perf_table(str(spa_perf_base), spa_perf, "SPA"))
    print()
    print(perf_table(str(et_perf_base), et_perf, "Cryo-ET"))


if __name__ == "__main__":
    main()
