#!/usr/bin/env python
"""Aggregate qparity matrix per-cell summary.json files into MATRIX_RESULTS.md.

Reads <run_dir>/<cell>/summary.json for every cell directory and emits a
markdown report with a one-row-per-cell table + per-iter mean_corr table
for each cell. Highlights cells that miss the 0.99 mean_corr or 0.01
|ΔPmax| thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x, w=8, prec=4):
    if x is None:
        return f"{'—':>{w}}"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return f"{str(x):>{w}}"
    if v != v:  # NaN
        return f"{'nan':>{w}}"
    return f"{v:>{w}.{prec}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    cells = []
    for cell_dir in sorted(args.run_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        s = cell_dir / "summary.json"
        if not s.exists():
            cells.append(
                {
                    "cell": cell_dir.name,
                    "status": "MISSING",
                    "reason": "no summary.json (job probably never finished)",
                    "K": None,
                    "requested_iters": None,
                    "effective_iters": None,
                    "adaptive_fraction": None,
                    "firstiter_cc": None,
                    "final_mean_corr": None,
                    "per_iter": {},
                }
            )
            continue
        try:
            cells.append(json.loads(s.read_text()))
        except Exception as exc:
            cells.append(
                {
                    "cell": cell_dir.name,
                    "status": "ERROR",
                    "reason": f"failed to parse summary.json: {exc}",
                }
            )

    lines = []
    lines.append("# Q-parity Matrix Results")
    lines.append("")
    lines.append(f"Run dir: `{args.run_dir}`")
    lines.append("")
    lines.append("Pass criteria: per-iter `mean_corr ≥ 0.99` AND `|ΔPmax| < 0.01`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    header = "| Cell | K | iters req/eff | af | firstiter_cc | final mean_corr | final |ΔPmax| | class_acc | status |"
    sep = "|------|---|---------------|----|--------------|------------------|----------------|-----------|--------|"
    lines.append(header)
    lines.append(sep)

    n_pass = n_fail = n_skipped = n_other = 0
    for c in cells:
        st = c.get("status", "?")
        if st == "PASS":
            n_pass += 1
        elif st == "FAIL":
            n_fail += 1
        elif st == "SKIPPED":
            n_skipped += 1
        else:
            n_other += 1
        K = c.get("K")
        req = c.get("requested_iters")
        eff = c.get("effective_iters")
        af = c.get("adaptive_fraction")
        fcc = c.get("firstiter_cc")
        fmc = c.get("final_mean_corr")
        fpd = c.get("final_pmax_abs_diff", c.get("final_pmax_abs_mean"))
        cacc = c.get("class_assignment_accuracy")
        marker = ""
        if st == "PASS":
            marker = "✓"
        elif st == "FAIL":
            marker = "✗"
        elif st == "SKIPPED":
            marker = "⊘"
        elif st == "ERROR":
            marker = "!"
        lines.append(
            f"| {c.get('cell', '?')} | {K} | {req}/{eff} | {af} | {fcc} | "
            f"{_fmt(fmc, 6, 4)} | {_fmt(fpd, 8, 5)} | {_fmt(cacc, 6, 4)} | {marker} {st} |"
        )

    lines.append("")
    lines.append(f"**Tally:** PASS={n_pass}  FAIL={n_fail}  SKIPPED={n_skipped}  OTHER={n_other}")

    # Per-cell per-iter detail
    lines.append("")
    lines.append("## Per-cell per-iter detail")
    for c in cells:
        cell = c.get("cell", "?")
        lines.append("")
        lines.append(f"### {cell}  ({c.get('status', '?')})")
        if c.get("reason"):
            lines.append(f"- reason: {c['reason']}")
        per = c.get("per_iter") or {}
        iters = per.get("iter") or []
        if not iters:
            lines.append("(no per-iter data)")
            continue
        mc = per.get("mean_corr") or [None] * len(iters)
        if c.get("K", 1) == 1:
            pd = per.get("pmax_abs_diff") or [None] * len(iters)
            lines.append("")
            lines.append("| iter | mean_corr | |ΔPmax| |")
            lines.append("|------|-----------|---------|")
            for i, m, d in zip(iters, mc, pd):
                lines.append(f"| {i} | {_fmt(m, 6, 4)} | {_fmt(d, 8, 5)} |")
        else:
            pcm = per.get("per_class_corr") or [None] * len(iters)
            pam = per.get("pmax_abs_mean") or [None] * len(iters)
            lines.append("")
            lines.append("| iter | mean_corr | per-class corrs | |ΔPmax|.mean |")
            lines.append("|------|-----------|------------------|--------------|")
            for i, m, pc, p in zip(iters, mc, pcm, pam):
                pcs = "[" + ", ".join(_fmt(x, 6, 4).strip() for x in (pc or [])) + "]"
                lines.append(f"| {i} | {_fmt(m, 6, 4)} | {pcs} | {_fmt(p, 8, 5)} |")

    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")
    print(f"Summary: PASS={n_pass} FAIL={n_fail} SKIPPED={n_skipped} OTHER={n_other}")


if __name__ == "__main__":
    main()
