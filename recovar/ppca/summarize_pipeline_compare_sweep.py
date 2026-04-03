#!/usr/bin/env python
"""Aggregate covariance-vs-PPCA sweep results into one table and a few summary plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_ORDER = ("covariance", "ppca", "ppca_projected_covariance")
METHOD_LABELS = {
    "covariance": "Covariance",
    "ppca": "PPCA",
    "ppca_projected_covariance": "PPCA+ProjCov",
}
METHOD_COLORS = {
    "covariance": "#d55e00",
    "ppca": "#0072b2",
    "ppca_projected_covariance": "#009e73",
}


def _flatten_scalar_metrics(prefix: str, value) -> dict:
    flat = {}
    if isinstance(value, dict):
        for key, subval in value.items():
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            flat.update(_flatten_scalar_metrics(child_prefix, subval))
        return flat
    if isinstance(value, (list, tuple)):
        return flat
    if value is None:
        flat[prefix] = None
        return flat
    if isinstance(value, (bool, np.bool_)):
        flat[prefix] = bool(value)
        return flat
    if isinstance(value, (int, float, np.integer, np.floating)):
        flat[prefix] = float(value)
        return flat
    return flat


def _load_summaries(results_root: Path) -> list[dict]:
    summaries = []
    for summary_path in sorted(results_root.glob("*/comparison_summary.json")):
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        summary["summary_path"] = str(summary_path)
        summaries.append(summary)
    return summaries


def _row_from_summary(summary: dict) -> dict:
    scores = summary["scores"]
    runtimes = summary.get("runtimes_seconds", {})
    row = {
        "dataset": summary["dataset"],
        "contrast_std": float(summary["contrast_std"]),
        "grid_size": int(summary["grid_size"]),
        "n_images": int(summary["n_images"]),
        "noise_level": float(summary["noise_level"]),
        "zdim": int(summary["zdim"]),
        "summary_path": summary["summary_path"],
    }
    for method in METHOD_ORDER:
        if method not in scores:
            continue
        row[f"{method}_relvar_mean"] = float(scores[method]["rel_var_mean"])
        row[f"{method}_mean_error"] = float(scores[method]["mean_error"])
        row[f"{method}_runtime_seconds"] = runtimes.get(method)
        row[f"{method}_result_dir"] = scores[method]["result_dir"]
        row.update(_flatten_scalar_metrics(f"{method}_metric", scores[method].get("pipeline_metrics", {})))
    if "covariance" in scores and "ppca" in scores:
        row["ppca_minus_covariance_relvar"] = float(scores["ppca"]["rel_var_mean"] - scores["covariance"]["rel_var_mean"])
    if "covariance" in scores and "ppca_projected_covariance" in scores:
        row["ppca_projected_covariance_minus_covariance_relvar"] = float(
            scores["ppca_projected_covariance"]["rel_var_mean"] - scores["covariance"]["rel_var_mean"]
        )
    if "ppca" in scores and "ppca_projected_covariance" in scores:
        row["ppca_projected_covariance_minus_ppca_relvar"] = float(
            scores["ppca_projected_covariance"]["rel_var_mean"] - scores["ppca"]["rel_var_mean"]
        )
    return row


def _write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_relvar(rows: list[dict], out_path: Path) -> None:
    labels = [f"{row['dataset']}\nc={row['contrast_std']:.1f}" for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    methods = [method for method in METHOD_ORDER if f"{method}_relvar_mean" in rows[0]]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 1.6), 8), sharex=True)
    width = 0.8 / max(len(methods), 1)
    center_offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(center_offsets, methods):
        vals = np.array([row[f"{method}_relvar_mean"] for row in rows], dtype=np.float32)
        axes[0].bar(
            x + offset,
            vals,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            alpha=0.9,
        )
    axes[0].set_ylabel("Mean RelVar")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Method Comparison")
    axes[0].legend(loc="best")

    cov = np.array([row["covariance_relvar_mean"] for row in rows], dtype=np.float32)
    best_alt = np.maximum.reduce(
        [
            np.array([row[f"{method}_relvar_mean"] for row in rows], dtype=np.float32)
            for method in methods
            if method != "covariance"
        ]
    )
    delta = best_alt - cov
    colors = np.where(delta >= 0, "#009e73", "#cc79a7")
    axes[1].bar(x, delta, color=colors, alpha=0.9)
    axes[1].axhline(0.0, color="k", linewidth=1)
    axes[1].set_ylabel("Best Alt - Cov RelVar")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_title("RelVar Delta")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_runtime(rows: list[dict], out_path: Path) -> None:
    methods = [method for method in METHOD_ORDER if f"{method}_runtime_seconds" in rows[0]]
    runtime_rows = [
        row
        for row in rows
        if all(row.get(f"{method}_runtime_seconds") is not None for method in methods)
    ]
    if not runtime_rows:
        return

    labels = [f"{row['dataset']}\nc={row['contrast_std']:.1f}" for row in runtime_rows]
    x = np.arange(len(runtime_rows), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(max(10, len(runtime_rows) * 1.6), 4.5))
    width = 0.8 / max(len(methods), 1)
    center_offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(center_offsets, methods):
        vals = np.array([row[f"{method}_runtime_seconds"] for row in runtime_rows], dtype=np.float32) / 60.0
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            alpha=0.9,
        )
    ax.set_ylabel("Wall Time (min)")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_title("Method Runtime")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize(results_root: Path, output_dir: Path) -> dict:
    summaries = _load_summaries(results_root)
    rows = [_row_from_summary(summary) for summary in summaries]
    rows.sort(key=lambda row: (row["dataset"], row["contrast_std"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "comparison_table.csv")
    _plot_relvar(rows, output_dir / "relvar_summary.png")
    _plot_runtime(rows, output_dir / "runtime_summary.png")

    aggregate = {
        "results_root": str(results_root),
        "n_completed_runs": len(rows),
        "rows": rows,
    }
    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a covariance-vs-PPCA sweep.")
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (args.results_root / "aggregate")
    aggregate = summarize(args.results_root, output_dir)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
