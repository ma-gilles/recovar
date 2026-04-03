#!/usr/bin/env python
"""Aggregate covariance-vs-PPCA sweep results into one table and a few summary plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    cov_relvar = float(scores["covariance"]["rel_var_mean"])
    ppca_relvar = float(scores["ppca"]["rel_var_mean"])
    row = {
        "dataset": summary["dataset"],
        "contrast_std": float(summary["contrast_std"]),
        "grid_size": int(summary["grid_size"]),
        "n_images": int(summary["n_images"]),
        "noise_level": float(summary["noise_level"]),
        "zdim": int(summary["zdim"]),
        "covariance_relvar_mean": cov_relvar,
        "ppca_relvar_mean": ppca_relvar,
        "relvar_delta": ppca_relvar - cov_relvar,
        "covariance_mean_error": float(scores["covariance"]["mean_error"]),
        "ppca_mean_error": float(scores["ppca"]["mean_error"]),
        "covariance_runtime_seconds": runtimes.get("covariance"),
        "ppca_runtime_seconds": runtimes.get("ppca"),
        "summary_path": summary["summary_path"],
        "covariance_result_dir": scores["covariance"]["result_dir"],
        "ppca_result_dir": scores["ppca"]["result_dir"],
    }
    row.update(_flatten_scalar_metrics("covariance_metric", scores["covariance"].get("pipeline_metrics", {})))
    row.update(_flatten_scalar_metrics("ppca_metric", scores["ppca"].get("pipeline_metrics", {})))
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
    cov = np.array([row["covariance_relvar_mean"] for row in rows], dtype=np.float32)
    ppca = np.array([row["ppca_relvar_mean"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 1.6), 8), sharex=True)
    width = 0.36
    axes[0].bar(x - width / 2, cov, width=width, label="Covariance", color="#d55e00", alpha=0.9)
    axes[0].bar(x + width / 2, ppca, width=width, label="PPCA", color="#0072b2", alpha=0.9)
    axes[0].set_ylabel("Mean RelVar")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Covariance vs PPCA")
    axes[0].legend(loc="best")

    delta = ppca - cov
    colors = np.where(delta >= 0, "#009e73", "#cc79a7")
    axes[1].bar(x, delta, color=colors, alpha=0.9)
    axes[1].axhline(0.0, color="k", linewidth=1)
    axes[1].set_ylabel("PPCA - Cov RelVar")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_title("RelVar Delta")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_runtime(rows: list[dict], out_path: Path) -> None:
    runtime_rows = [
        row for row in rows if row["covariance_runtime_seconds"] is not None and row["ppca_runtime_seconds"] is not None
    ]
    if not runtime_rows:
        return

    labels = [f"{row['dataset']}\nc={row['contrast_std']:.1f}" for row in runtime_rows]
    x = np.arange(len(runtime_rows), dtype=np.float32)
    cov = np.array([row["covariance_runtime_seconds"] for row in runtime_rows], dtype=np.float32) / 60.0
    ppca = np.array([row["ppca_runtime_seconds"] for row in runtime_rows], dtype=np.float32) / 60.0

    fig, ax = plt.subplots(figsize=(max(10, len(runtime_rows) * 1.6), 4.5))
    width = 0.36
    ax.bar(x - width / 2, cov, width=width, label="Covariance", color="#d55e00", alpha=0.9)
    ax.bar(x + width / 2, ppca, width=width, label="PPCA", color="#0072b2", alpha=0.9)
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
