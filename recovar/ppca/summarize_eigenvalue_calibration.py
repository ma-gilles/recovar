#!/usr/bin/env python
"""Eigenvalue calibration diagnostic for the covariance-vs-PPCA sweep.

Each ``comparison_summary.json`` written by ``compare_covariance_vs_ppca_pipeline.py``
already carries, per method, the first ``zdim`` recovered eigenvalues (``s_est``)
and the ground-truth SVD eigenvalues of the true volume ensemble (``s_gt``).
This script aggregates those fields across a whole sweep (or a collection of
sweeps), produces one long-form CSV of (dataset, snr, method, pc) → ratio, and
emits per-(dataset, snr) plots of the eigenvalue curves against GT.

The goal is to answer the concrete question: do we recover the correct
eigenvalues on the cryobench synthetic datasets, per PPCA variant?
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_ORDER = (
    "covariance",
    "ppca",
    "ppca_projected_covariance",
    "ppca_refitb",
    "ppca_iterative_projcov",
)
METHOD_LABELS = {
    "covariance": "Covariance",
    "ppca": "PPCA",
    "ppca_projected_covariance": "PPCA+ProjCov",
    "ppca_refitb": "PPCA+RefitB",
    "ppca_iterative_projcov": "PPCA+IterProjCov",
}
METHOD_COLORS = {
    "covariance": "#d55e00",
    "ppca": "#0072b2",
    "ppca_projected_covariance": "#009e73",
    "ppca_refitb": "#cc79a7",
    "ppca_iterative_projcov": "#e69f00",
}


def _iter_summaries(roots: list[Path]):
    for root in roots:
        for p in sorted(root.glob("*/comparison_summary.json")):
            try:
                yield p, json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue


def _extract_rows(summary_path: Path, summary: dict) -> list[dict]:
    """Return per-PC rows with BOTH ratio conventions:

    - ``ratio_vs_gt_sorted`` = s_est[k] / s_gt[k], where s_gt is the GT ensemble's
      own sorted spectrum. This is the naive convention and is misleading when
      the inferred subspace is misaligned with the GT top-q subspace.
    - ``ratio_vs_projected_gt`` = s_est[k] / ideal[k], where ideal[k] is the
      per-PC GT-projected variance along the inferred direction k, recovered by
      differencing the cumulative ``variance_score_per_pc`` curve (which is
      ``cumsum(u_est^T Σ_gt u_est)`` under the convention s=eigenvalues). This
      is the honest calibration metric when subspaces differ.
    """
    rows = []
    scores = summary.get("scores", {})
    for method, block in scores.items():
        s_est = np.asarray(block.get("s_est") or [], dtype=np.float64)
        s_gt = np.asarray(block.get("s_gt") or [], dtype=np.float64)
        vcurve = np.asarray(block.get("variance_score_per_pc") or [], dtype=np.float64)
        ideal_per_pc = np.diff(np.concatenate([[0.0], vcurve])) if vcurve.size > 0 else np.asarray([])
        n = int(min(s_est.size, s_gt.size))
        for k in range(n):
            gt = float(s_gt[k])
            est = float(s_est[k])
            ideal = float(ideal_per_pc[k]) if k < ideal_per_pc.size else None
            ratio_sorted = est / gt if gt > 0 else None
            ratio_proj = (est / ideal) if (ideal is not None and ideal > 0) else None
            rows.append(
                {
                    "sweep_root": str(summary_path.parent.parent),
                    "run_dir": str(summary_path.parent),
                    "dataset": summary.get("dataset"),
                    "noise_level": float(summary.get("noise_level", float("nan"))),
                    "n_images": int(summary.get("n_images", 0)),
                    "zdim": int(summary.get("zdim", 0)),
                    "method": method,
                    "pc": k,
                    "s_est": est,
                    "s_gt_sorted": gt,
                    "s_gt_projected": ideal,
                    "ratio_vs_gt_sorted": ratio_sorted,
                    "ratio_vs_projected_gt": ratio_proj,
                    "rel_var_mean": block.get("rel_var_mean"),
                }
            )
    return rows


def _plot_eigenvalues(
    rows: list[dict],
    out_dir: Path,
) -> list[Path]:
    """One subplot per (dataset, snr); four colors per method + black GT."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    keyed: dict[tuple, list[dict]] = {}
    for r in rows:
        keyed.setdefault((r["dataset"], r["noise_level"]), []).append(r)

    # per-(dataset, snr) figure with two panels: eigenvalues and ratio
    for (ds, snr), group in sorted(keyed.items()):
        by_method: dict[str, list[dict]] = {}
        for r in group:
            by_method.setdefault(r["method"], []).append(r)
        if not by_method:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

        # canonical sorted-GT from any variant (they should all agree)
        any_rows = sorted(next(iter(by_method.values())), key=lambda r: r["pc"])
        gt = np.asarray([r["s_gt_sorted"] for r in any_rows], dtype=np.float64)
        pcs_gt = np.arange(gt.size)
        axes[0].plot(pcs_gt, gt, "k-o", lw=2.0, ms=5, label="GT (sorted)")

        for method in METHOD_ORDER:
            if method not in by_method:
                continue
            rs = sorted(by_method[method], key=lambda r: r["pc"])
            est = np.asarray([r["s_est"] for r in rs], dtype=np.float64)
            ideal = np.asarray([r["s_gt_projected"] if r["s_gt_projected"] is not None else np.nan for r in rs])
            ratio_sorted = np.asarray(
                [r["ratio_vs_gt_sorted"] if r["ratio_vs_gt_sorted"] is not None else np.nan for r in rs]
            )
            ratio_proj = np.asarray(
                [r["ratio_vs_projected_gt"] if r["ratio_vs_projected_gt"] is not None else np.nan for r in rs]
            )
            pcs = np.arange(est.size)
            color = METHOD_COLORS[method]
            label = METHOD_LABELS[method]
            axes[0].plot(pcs, est, color=color, marker="o", ms=4, label=label, alpha=0.9)
            axes[0].plot(
                pcs,
                ideal,
                color=color,
                marker="x",
                ms=4,
                lw=1,
                alpha=0.6,
                linestyle=":",
            )
            axes[1].plot(pcs, ratio_sorted, color=color, marker="o", ms=4, label=label, alpha=0.9)
            axes[2].plot(pcs, ratio_proj, color=color, marker="o", ms=4, label=label, alpha=0.9)

        axes[0].set_yscale("log")
        axes[0].set_xlabel("PC index")
        axes[0].set_ylabel("eigenvalue (log)")
        axes[0].set_title(f"{ds} | SNR={snr}\neigenvalues (solid=s_est, dotted=u_inf^T Σ_gt u_inf)")
        axes[0].legend(loc="best", fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].axhline(1.0, color="k", lw=1, alpha=0.5)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("PC index")
        axes[1].set_ylabel("s_est / s_gt_sorted (log)")
        axes[1].set_title("naive ratio to sorted GT\n(misleading if subspaces differ)")
        axes[1].grid(True, alpha=0.3)

        axes[2].axhline(1.0, color="k", lw=1, alpha=0.5)
        axes[2].set_yscale("log")
        axes[2].set_xlabel("PC index")
        axes[2].set_ylabel("s_est / (u_inf^T Σ_gt u_inf) (log)")
        axes[2].set_title("honest ratio\n(1.0 = spectrum calibrated on inferred subspace)")
        axes[2].legend(loc="best", fontsize=7)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        safe_snr = str(snr).replace(".", "p")
        out_path = out_dir / f"eigs_{ds}_snr{safe_snr}.png"
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def _write_long_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _mean_abs_log(values: np.ndarray) -> float:
    pos = values[np.isfinite(values) & (values > 0)]
    return float(np.mean(np.abs(np.log(pos)))) if pos.size else float("nan")


def _write_summary_table(rows: list[dict], out_path: Path) -> None:
    """Per (dataset, snr, method): mean |log ratio| under both conventions + first 5 PCs."""
    keyed: dict[tuple, list[dict]] = {}
    for r in rows:
        keyed.setdefault((r["dataset"], r["noise_level"], r["method"]), []).append(r)

    summary_rows = []
    for (ds, snr, method), rs in sorted(keyed.items()):
        rs = sorted(rs, key=lambda r: r["pc"])
        r_sorted = np.asarray(
            [r["ratio_vs_gt_sorted"] if r["ratio_vs_gt_sorted"] is not None else np.nan for r in rs],
            dtype=np.float64,
        )
        r_proj = np.asarray(
            [r["ratio_vs_projected_gt"] if r["ratio_vs_projected_gt"] is not None else np.nan for r in rs],
            dtype=np.float64,
        )
        summary_rows.append(
            {
                "dataset": ds,
                "noise_level": snr,
                "method": method,
                "n_pcs": len(rs),
                "mean_abs_log_ratio_vs_sorted": _mean_abs_log(r_sorted),
                "mean_abs_log_ratio_vs_projected": _mean_abs_log(r_proj),
                "proj_pc0": float(r_proj[0]) if r_proj.size > 0 else None,
                "proj_pc1": float(r_proj[1]) if r_proj.size > 1 else None,
                "proj_pc2": float(r_proj[2]) if r_proj.size > 2 else None,
                "proj_pc3": float(r_proj[3]) if r_proj.size > 3 else None,
                "proj_pc4": float(r_proj[4]) if r_proj.size > 4 else None,
                "rel_var_mean": rs[0].get("rel_var_mean"),
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        if summary_rows:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)


def summarize(
    results_roots: list[Path],
    output_dir: Path,
) -> dict:
    rows: list[dict] = []
    for path, summary in _iter_summaries(results_roots):
        rows.extend(_extract_rows(path, summary))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_long_csv(rows, output_dir / "eigenvalues_long.csv")
    _write_summary_table(rows, output_dir / "eigenvalues_summary.csv")
    plot_paths = _plot_eigenvalues(rows, output_dir / "plots")

    return {
        "n_rows": len(rows),
        "plots": [str(p) for p in plot_paths],
        "long_csv": str(output_dir / "eigenvalues_long.csv"),
        "summary_csv": str(output_dir / "eigenvalues_summary.csv"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate eigenvalue recovery diagnostics from compare sweeps.")
    p.add_argument(
        "roots",
        nargs="+",
        help="One or more sweep roots (each contains <run>/comparison_summary.json) or a glob pattern.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    roots: list[Path] = []
    for entry in args.roots:
        matches = [Path(p) for p in glob.glob(entry)] or [Path(entry)]
        roots.extend(m for m in matches if m.exists())
    result = summarize(roots, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
