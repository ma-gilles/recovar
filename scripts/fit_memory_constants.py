#!/usr/bin/env python3
"""Fit memory-model constants from a recorded sweep, output a markdown report.

Critical design choice: this script does NOT auto-edit
``recovar/utils/memory_model.py``. It produces a report that a human
reads, decides what to commit, and writes provenance comments by hand.

Usage:

    python scripts/fit_memory_constants.py \\
        _diagnostics/sweep_run_<id>.json \\
        --output _diagnostics/fit_report_<id>.md

Inputs: a sweep JSON produced by ``validate_memory_formulas.py --mode
record`` containing per-cell observed peaks plus current predictions.

Outputs: a markdown report with:

  - inferred exponents per term (log-log fit)
  - per-cell residuals (predicted vs observed)
  - worst cells flagged
  - proposed constant values
  - confidence / instability warnings
  - diff suggestion (paste-able into memory_model.py, but ALWAYS
    review by hand)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def loglog_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Fit y = a × x^p via least squares on log-log. Returns (a, p, R²)."""
    if len(xs) < 2 or any(y <= 0 for y in ys):
        return float("nan"), float("nan"), float("nan")
    lxs = [math.log(x) for x in xs]
    lys = [math.log(y) for y in ys]
    n = len(lxs)
    mean_lx = sum(lxs) / n
    mean_ly = sum(lys) / n
    var_lx = sum((lx - mean_lx) ** 2 for lx in lxs)
    cov_xy = sum((lx - mean_lx) * (ly - mean_ly) for lx, ly in zip(lxs, lys))
    if var_lx == 0:
        return float("nan"), float("nan"), float("nan")
    p = cov_xy / var_lx
    log_a = mean_ly - p * mean_lx
    a = math.exp(log_a)

    # R² on log-log space
    ss_res = sum((ly - (log_a + p * lx)) ** 2 for lx, ly in zip(lxs, lys))
    ss_tot = sum((ly - mean_ly) ** 2 for ly in lys)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return a, p, r_squared


def fit_svd_workspace(cells: list[dict]) -> dict:
    """Fit the SVD-workspace coefficient and exponent from covariance peaks.

    Strategy: at fixed (grid_size, backend, pipeline), look at how the
    covariance phase peak varies with n_pcs. Subtract the basis term
    (which we trust scales as n_pcs × grid³ × 8 ÷ 1e9) to isolate the
    n_pcs-dependent workspace term.
    """
    # Bucket by (pipeline, grid_size, backend); each bucket is a candidate
    # for fitting along the n_pcs axis.
    buckets: dict[tuple, list[dict]] = {}
    for c in cells:
        if c["status"] != "ok":
            continue
        observed = c["observed_peaks_gb"].get("after_covariance")
        if observed is None:
            continue
        key = (c["pipeline"], c["grid_size"], c["backend"])
        buckets.setdefault(key, []).append(
            {
                "n_pcs": c["n_pcs"],
                "observed_cov_gb": observed,
            }
        )

    fits = {}
    for key, points in buckets.items():
        if len(points) < 3:
            continue
        pipeline, grid_size, backend = key
        # Subtract trusted basis term
        volume_bytes = grid_size**3 * 8
        residuals = []
        ns = []
        for p in points:
            basis_gb = p["n_pcs"] * volume_bytes / 1e9
            isolated = p["observed_cov_gb"] - basis_gb
            if isolated > 0:
                ns.append(p["n_pcs"])
                residuals.append(isolated)
        if len(ns) < 3:
            continue
        a, exponent, r_squared = loglog_fit(ns, residuals)
        fits[key] = {
            "n_pcs_values": ns,
            "isolated_workspace_gb": residuals,
            "fitted_coef": a,
            "fitted_exponent": exponent,
            "r_squared": r_squared,
        }
    return fits


def per_cell_table(cells: list[dict]) -> str:
    """Markdown table of per-cell prediction error."""
    rows = ["| cell_id | phase | predicted_gb | observed_gb | underpred | overpred | status |"]
    rows.append("|---|---|---:|---:|---:|---:|---|")
    for c in cells:
        for phase, observed in c["observed_peaks_gb"].items():
            predicted = c["predicted_peaks_gb"].get(phase, 0.0)
            if predicted == 0.0 or observed == 0.0:
                under = "n/a"
                over = "n/a"
            else:
                under = f"{observed / predicted:.2f}"
                over = f"{predicted / observed:.2f}"
            rows.append(
                f"| `{c['cell_id']}` | {phase} | {predicted:.2f} | {observed:.2f} | {under} | {over} | {c['status']} |"
            )
    return "\n".join(rows)


def worst_cells(cells: list[dict], n: int = 5) -> list[tuple]:
    """Return n cells with the largest underprediction ratio (most likely to OOM)."""
    scored = []
    for c in cells:
        if c["status"] != "ok":
            continue
        for phase, observed in c["observed_peaks_gb"].items():
            predicted = c["predicted_peaks_gb"].get(phase, 0.0)
            if predicted == 0.0 or observed == 0.0:
                continue
            scored.append((observed / predicted, c["cell_id"], phase, predicted, observed))
    scored.sort(reverse=True)
    return scored[:n]


def write_report(payload: dict, output: Path) -> None:
    cells = payload["cells"]
    host = payload["host"]
    fits = fit_svd_workspace(cells)

    lines = []
    lines.append("# Memory-model fit report")
    lines.append("")
    lines.append("Generated from sweep recorded on:")
    lines.append("")
    lines.append(f"- GPU: `{host.get('gpu_kind')}`")
    lines.append(f"- Driver: `{host.get('driver')}`")
    lines.append(f"- JAX: `{host.get('jax')}`")
    lines.append(f"- recovar HEAD: `{host.get('git_head')}`")
    lines.append(f"- cells: {len(cells)} ({sum(1 for c in cells if c['status'] == 'ok')} OK)")
    lines.append("")

    lines.append("## SVD-workspace fit (covariance phase)")
    lines.append("")
    lines.append(
        "Strategy: subtract trusted basis term `n_pcs × grid³ × 8 / 1e9` from"
        " each covariance peak, then log-log fit the residual against n_pcs."
        " A clean p≈4 fit confirms the legacy assumption; a clean p≈3 or 2"
        " indicates the legacy form is wrong."
    )
    lines.append("")
    if fits:
        for (pipeline, grid_size, backend), f in fits.items():
            lines.append(f"### {pipeline} grid={grid_size} backend={backend}")
            lines.append("")
            lines.append(f"- n_pcs values: {f['n_pcs_values']}")
            lines.append(f"- isolated workspace (GB): {[f'{r:.3f}' for r in f['isolated_workspace_gb']]}")
            lines.append(f"- fitted: `{f['fitted_coef']:.3e} × n_pcs^{f['fitted_exponent']:.2f}`")
            lines.append(f"- R² = {f['r_squared']:.3f}")
            confidence = (
                "high" if f["r_squared"] > 0.95 else "medium" if f["r_squared"] > 0.85 else "LOW — investigate further"
            )
            lines.append(f"- confidence: {confidence}")
            lines.append("")
    else:
        lines.append("_No fittable buckets (need ≥3 n_pcs values per cell)._")
        lines.append("")

    lines.append("## Worst-prediction cells")
    lines.append("")
    lines.append("Top 5 cells by underprediction ratio (most likely to cause OOM):")
    lines.append("")
    lines.append("| ratio | cell | phase | predicted_gb | observed_gb |")
    lines.append("|---:|---|---|---:|---:|")
    for ratio, cell_id, phase, pred, obs in worst_cells(cells, 5):
        lines.append(f"| {ratio:.2f} | `{cell_id}` | {phase} | {pred:.2f} | {obs:.2f} |")
    lines.append("")

    lines.append("## All-cells residual table")
    lines.append("")
    lines.append(per_cell_table(cells))
    lines.append("")

    lines.append("## Proposed constants (paste-ready, but REVIEW BY HAND)")
    lines.append("")
    lines.append("```python")
    if fits:
        # Pick a representative bucket (SPA, grid=128, custom_cuda) if available.
        pref = ("spa", 128, "custom_cuda")
        if pref in fits:
            f = fits[pref]
            lines.append(f"# Fitted from {pref}: {len(f['n_pcs_values'])} points, R²={f['r_squared']:.3f}")
            lines.append(f"SVD_WORKSPACE_COEF_GB = {f['fitted_coef']:.3e}")
            lines.append(f"SVD_WORKSPACE_EXPONENT = {round(f['fitted_exponent'], 1)}")
        else:
            any_key = next(iter(fits))
            f = fits[any_key]
            lines.append(f"# Fitted from {any_key} (preferred bucket missing)")
            lines.append(f"SVD_WORKSPACE_COEF_GB = {f['fitted_coef']:.3e}")
            lines.append(f"SVD_WORKSPACE_EXPONENT = {round(f['fitted_exponent'], 1)}")
    else:
        lines.append("# Insufficient data to propose constants. Need ≥3 n_pcs points per bucket.")
    lines.append("```")
    lines.append("")
    lines.append(
        "**Reminder**: the fitter does not auto-edit `memory_model.py`. "
        "Read the residuals above, decide whether the fit is trustworthy "
        "(R² high, residuals symmetric, no single bucket dominating), "
        "then update the constants in `memory_model.py` by hand with a "
        "provenance doc-comment naming the sweep run-id and bucket."
    )
    lines.append("")

    lines.append("## Confidence warnings")
    lines.append("")
    warnings = []
    for key, f in fits.items():
        if f["r_squared"] < 0.85:
            warnings.append(
                f"- {key}: R² = {f['r_squared']:.3f} (low). "
                f"Fit may be unstable; collect more n_pcs points or "
                f"investigate per-term breakdown via JAX profile."
            )
    overpred_count = sum(1 for ratio, *_ in worst_cells(cells, 999) if ratio < 0.5)
    if overpred_count > 0:
        warnings.append(
            f"- {overpred_count} cell-phases have predicted > 2× observed. "
            f"The model is over-conservative for these configurations and "
            f"will pick batch sizes smaller than necessary."
        )
    if not warnings:
        warnings.append("_None._")
    lines.extend(warnings)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_json", help="path to sweep_run_*.json")
    parser.add_argument("--output", required=True, help="path for fit_report_*.md")
    args = parser.parse_args()

    payload = json.loads(Path(args.sweep_json).read_text())
    write_report(payload, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
