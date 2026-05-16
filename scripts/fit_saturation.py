#!/usr/bin/env python3
"""Fit SATURATION_TABLE from a saturation sweep recording.

Reads the JSON written by ``validate_memory_formulas.py record --cells
saturation`` and computes, per (grid_size, backend):

  SATURATION = max(observed_peak / budget)  over cells of that group

The "max" is deliberate: we want the worst-case headroom utilization
so the inverse (the BUDGET_INFLATION factor we hand to legacy
formulas) never overshoots the real budget.

Outputs a paste-ready Python snippet and a per-cell residual table.
Does NOT auto-patch any source file — the user reviews the table and
commits constants by hand with a provenance comment.

Usage:
    python scripts/fit_saturation.py path/to/saturation_record.json \\
        [--output path/to/report.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_json", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--phase",
        default="after_covariance",
        help="Which observed_peaks_gb phase to use (default: after_covariance, the dominant phase).",
    )
    args = ap.parse_args()

    data = json.loads(args.sweep_json.read_text())
    cells = data.get("cells", [])

    # ------------------------------------------------------------------
    # Phase A1: at full budget (budget_gb == None) — measures
    # "natural" headroom when planner has the whole GPU.
    # ------------------------------------------------------------------
    a1: dict[tuple, list[dict]] = {}
    a2: dict[tuple, list[dict]] = {}  # constrained budget, custom_cuda
    a3: dict[tuple, list[dict]] = {}  # constrained budget, jax_fallback

    for cell in cells:
        if cell.get("status") != "ok":
            continue
        peak = cell.get("observed_peaks_gb", {}).get(args.phase)
        if peak is None:
            continue

        grid = cell["grid_size"]
        backend = cell["backend"]
        budget = cell.get("budget_gb")

        if budget is None:
            # We need to know what the planner actually used. Without
            # explicit --gpu-budget-gb the budget is jax_limit (typ
            # 0.95 * physical). On H100 80 GB that's ~76 GB.
            budget = 76.0
            bucket = a1
        else:
            bucket = a2 if backend == "custom_cuda" else a3

        bucket.setdefault((grid, backend), []).append(
            {"budget": float(budget), "peak": float(peak), "cell_id": cell.get("cell_id", "?")}
        )

    # ------------------------------------------------------------------
    # Compute SATURATION_TABLE: max(peak / budget) per (grid, backend)
    # ------------------------------------------------------------------
    saturation: dict[tuple, dict] = {}
    for bucket in (a1, a2, a3):
        for key, points in bucket.items():
            for p in points:
                ratio = p["peak"] / p["budget"]
                if key not in saturation or ratio > saturation[key]["ratio"]:
                    saturation[key] = {
                        "ratio": ratio,
                        "peak": p["peak"],
                        "budget": p["budget"],
                        "cell_id": p["cell_id"],
                    }

    # ------------------------------------------------------------------
    # Linearity check on A2 (vary budget at fixed grid=128 custom_cuda).
    # If peak ∝ budget, then linear regression should give R²>0.9.
    # If peak is *flat* in budget (over-conservative legacy), the
    # legacy formula is the bottleneck — we'd need to fix the formula,
    # not just the multiplier.
    # ------------------------------------------------------------------
    linearity = None
    a2_key = (128, "custom_cuda")
    if a2_key in a2 and len(a2[a2_key]) >= 3:
        pts = sorted(a2[a2_key], key=lambda p: p["budget"])
        xs = [p["budget"] for p in pts]
        ys = [p["peak"] for p in pts]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        var_x = sum((x - mx) ** 2 for x in xs)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = cov / var_x if var_x > 0 else 0.0
        intercept = my - slope * mx
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        linearity = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r2,
            "points": list(zip(xs, ys)),
        }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines: list[str] = []
    lines.append("# Saturation sweep fit report\n")
    host = data.get("host", {})
    lines.append(f"- GPU: `{host.get('gpu_kind', 'unknown')}`")
    lines.append(f"- recovar HEAD: `{host.get('git_head', 'unknown')}`")
    lines.append(f"- phase analyzed: `{args.phase}`")
    lines.append(f"- cells in sweep: {len(cells)} ({sum(1 for c in cells if c.get('status') == 'ok')} OK)\n")

    lines.append("## Linearity check (A2: grid=128 custom_cuda, budget sweep)\n")
    if linearity is None:
        lines.append("_Not enough A2 cells with status=ok to fit (need ≥3)._\n")
    else:
        lines.append(f"- linear fit: peak = {linearity['slope']:.4f} × budget + {linearity['intercept']:.2f}")
        lines.append(f"- R² = {linearity['r_squared']:.3f}\n")
        if linearity["r_squared"] < 0.9:
            lines.append(
                "**WARNING: low R². The legacy formula does NOT scale peak linearly with "
                "budget in this regime — multiplier alone won't fix it. Likely need a "
                "grid-aware fix in the legacy formula itself.**\n"
            )
        lines.append("| budget (GB) | peak (GB) |")
        lines.append("|------------:|----------:|")
        for budget, peak in linearity["points"]:
            lines.append(f"| {budget:.0f} | {peak:.1f} |")
        lines.append("")

    lines.append("## SATURATION_TABLE (max peak/budget ratio per group)\n")
    lines.append("| grid | backend | peak | budget | ratio | from cell |")
    lines.append("|-----:|---------|-----:|-------:|------:|-----------|")
    for (grid, backend), entry in sorted(saturation.items()):
        lines.append(
            f"| {grid} | {backend} | {entry['peak']:.1f} | {entry['budget']:.0f} | "
            f"{entry['ratio']:.3f} | `{entry['cell_id']}` |"
        )
    lines.append("")

    lines.append("## Proposed constants (paste into ``parser_args.py``)\n")
    lines.append("```python")
    lines.append("# Fitted from saturation sweep on H100 80GB.")
    lines.append(f"# Sweep at: {args.sweep_json}")
    lines.append("# SATURATION[grid][backend] = max(peak / budget) observed.")
    lines.append("# BUDGET_INFLATION = 1 / SATURATION; legacy formulas get the inflated")
    lines.append("# budget so picked batch sizes fill the real physical budget.")
    lines.append("SATURATION_TABLE: dict[tuple[int, str], float] = {")
    for (grid, backend), entry in sorted(saturation.items()):
        # Cap at 1.0 — never inflate ABOVE physical (would cause OOM).
        # Floor at 0.30 — a conservative minimum so we don't multiply
        # batches by 3x+ on the basis of one cell.
        capped = min(1.0, max(0.30, entry["ratio"]))
        lines.append(f"    ({grid}, {backend!r}): {capped:.3f},  # observed ratio={entry['ratio']:.3f}")
    lines.append("}")
    lines.append("```\n")

    lines.append("## Cells that did NOT contribute (OOM / error)\n")
    skipped = [c for c in cells if c.get("status") != "ok"]
    if not skipped:
        lines.append("_All cells succeeded._")
    else:
        for c in skipped:
            tag = f"budget={c.get('budget_gb')}" if c.get("budget_gb") else "budget=full"
            lines.append(f"- `{c['cell_id']}` (status={c['status']}, {tag})")
    lines.append("")

    report = "\n".join(lines)
    if args.output is not None:
        args.output.write_text(report)
        print(f"wrote report to {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
