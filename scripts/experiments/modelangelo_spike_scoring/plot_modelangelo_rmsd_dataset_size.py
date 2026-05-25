#!/usr/bin/env python3
"""Summarize ModelAngelo moving-region RMSD across dataset sizes."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _n_images(label: str) -> int:
    match = re.search(r"n(\d{8})", label)
    if match is None:
        raise ValueError(f"Could not parse image count from {label}")
    return int(match.group(1))


def _finite_float(value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.csv.open() as handle:
        rows = list(csv.DictReader(handle))
    best_rows = []
    labels = sorted({row["label"] for row in rows}, key=_n_images)
    for label in labels:
        candidates = [row for row in rows if row["label"] == label]
        finite = [row for row in candidates if math.isfinite(_finite_float(row["rmsd_A"]))]
        if finite:
            best = min(finite, key=lambda row: _finite_float(row["rmsd_A"]))
        else:
            best = candidates[0]
        out = dict(best)
        out["n_images"] = _n_images(label)
        best_rows.append(out)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "rmsd_summary_best_available.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = ["n_images"] + [field for field in best_rows[0] if field != "n_images"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    finite_rows = [row for row in best_rows if math.isfinite(_finite_float(row["rmsd_A"]))]
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    if finite_rows:
        ax.semilogx(
            [row["n_images"] for row in finite_rows],
            [_finite_float(row["rmsd_A"]) for row in finite_rows],
            marker="o",
            lw=2.0,
            label="RMSD",
        )
        ax.semilogx(
            [row["n_images"] for row in finite_rows],
            [_finite_float(row["p90_A"]) for row in finite_rows],
            marker="s",
            lw=2.0,
            label="p90",
        )
        ax.semilogx(
            [row["n_images"] for row in finite_rows],
            [_finite_float(row["p99_A"]) for row in finite_rows],
            marker="^",
            lw=2.0,
            label="p99",
        )
    missing = [row for row in best_rows if not math.isfinite(_finite_float(row["rmsd_A"]))]
    if missing:
        ymax = max([_finite_float(row["p99_A"]) for row in finite_rows if math.isfinite(_finite_float(row["p99_A"]))] or [1.0])
        ax.scatter([row["n_images"] for row in missing], [ymax * 1.08] * len(missing), marker="x", color="0.3", label="no scoreable atoms")
    ax.set_xlabel("number of images")
    ax.set_ylabel("moving-region same-element distance (A)")
    ax.set_title("ModelAngelo moving-region atom fit vs image count")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(args.out_dir / "modelangelo_moving_region_rmsd_vs_n.png", dpi=180)
    fig.savefig(args.out_dir / "modelangelo_moving_region_rmsd_vs_n.pdf")
    plt.close(fig)
    print(summary_path)


if __name__ == "__main__":
    main()
