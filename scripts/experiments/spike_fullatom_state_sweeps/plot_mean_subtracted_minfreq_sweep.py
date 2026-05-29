#!/usr/bin/env python3
"""Low-frequency-cut views of mean-subtracted FSC curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fsc_resolution(freq: np.ndarray, fsc: np.ndarray, threshold: float = 0.5) -> tuple[float, bool]:
    valid = np.isfinite(fsc) & (freq > 0)
    freq = freq[valid]
    fsc = fsc[valid]
    if freq.size == 0:
        return float("nan"), False
    below = np.flatnonzero(fsc < threshold)
    if below.size == 0:
        return float(1.0 / freq[-1]), False
    idx = int(below[0])
    if idx == 0:
        return float(1.0 / freq[0]), True
    x0, x1 = float(fsc[idx - 1]), float(fsc[idx])
    f0, f1 = float(freq[idx - 1]), float(freq[idx])
    crossing = f1 if x0 == x1 else f0 + (threshold - x0) * (f1 - f0) / (x1 - x0)
    return (float(1.0 / crossing) if crossing > 0 else float("nan")), False


def _image_counts(data: np.lib.npyio.NpzFile) -> list[int]:
    counts = []
    for key in data.files:
        if key.startswith("n") and key[1:].isdigit():
            counts.append(int(key[1:]))
    return sorted(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--plot-min-frequency", type=float, default=0.05)
    parser.add_argument("--sweep", default="0.02,0.03,0.04,0.05,0.06,0.08,0.10")
    args = parser.parse_args()

    data = np.load(args.metrics_dir / "mean_subtracted_fsc_curves.npz")
    freq = np.asarray(data["frequency_1_per_A"], dtype=np.float64)
    counts = _image_counts(data)
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(counts)))

    fig, ax = plt.subplots(figsize=(8.4, 5.1), constrained_layout=True)
    for color, n_images in zip(colors, counts):
        fsc = np.asarray(data[f"n{n_images:08d}"], dtype=np.float64)
        keep = freq >= args.plot_min_frequency
        res, cutoff_crossing = _fsc_resolution(freq[keep], fsc[keep], threshold=0.5)
        suffix = " at cutoff" if cutoff_crossing else ""
        ax.plot(freq[keep], fsc[keep], color=color, lw=2.0, label=f"{n_images:,}: {res:.2f} A{suffix}")
    ax.axhline(0.5, color="0.35", ls="--", lw=1.0)
    ax.set_xlim(args.plot_min_frequency, 0.4)
    ax.set_ylim(-0.08, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("mean-subtracted masked FSC vs GT")
    ax.set_title(f"Mean-subtracted FSC0.5, ignoring f < {args.plot_min_frequency:g} 1/A")
    ax.grid(True, alpha=0.25)
    ax.legend(title="n images: FSC0.5", fontsize=8, title_fontsize=8, loc="lower left")
    stem = f"mean_subtracted_fsc_curves_fsc05_ignore_below_{args.plot_min_frequency:g}".replace(".", "p")
    fig.savefig(args.metrics_dir / f"{stem}.png", dpi=180)
    fig.savefig(args.metrics_dir / f"{stem}.pdf")
    plt.close(fig)

    sweep = [float(item) for item in args.sweep.split(",") if item.strip()]
    rows = []
    for min_freq in sweep:
        keep = freq >= min_freq
        for n_images in counts:
            fsc = np.asarray(data[f"n{n_images:08d}"], dtype=np.float64)
            res, cutoff_crossing = _fsc_resolution(freq[keep], fsc[keep], threshold=0.5)
            rows.append(
                {
                    "min_frequency_1_per_A": min_freq,
                    "n_images": n_images,
                    "fsc05_resolution_A": res,
                    "already_below_at_cutoff": cutoff_crossing,
                }
            )
    with (args.metrics_dir / "mean_subtracted_fsc05_minfreq_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for min_freq in sweep:
        subset = [row for row in rows if row["min_frequency_1_per_A"] == min_freq]
        ax.semilogx(
            [row["n_images"] for row in subset],
            [row["fsc05_resolution_A"] for row in subset],
            marker="o",
            lw=1.6,
            label=f"min f={min_freq:g}",
        )
    ax.invert_yaxis()
    ax.set_xlabel("number of images")
    ax.set_ylabel("FSC0.5 resolution (A), lower is better")
    ax.set_title("Mean-subtracted FSC0.5 sensitivity to low-frequency cutoff")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(args.metrics_dir / "mean_subtracted_fsc05_resolution_vs_n_minfreq_sweep.png", dpi=180)
    fig.savefig(args.metrics_dir / "mean_subtracted_fsc05_resolution_vs_n_minfreq_sweep.pdf")
    plt.close(fig)

    print(args.metrics_dir)


if __name__ == "__main__":
    main()
