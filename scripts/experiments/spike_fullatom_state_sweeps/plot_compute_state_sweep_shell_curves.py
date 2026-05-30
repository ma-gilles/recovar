#!/usr/bin/env python3
"""Combine per-run compute_state shell metrics across a dataset-size sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_true_pipeline_sweep_noise10_b100_dev2_20260529"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--metrics-relpath", default="08_metrics_zdim4_noreg")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--states", default="0,25,50")
    parser.add_argument("--image-counts", default="10000,30000,100000,300000,1000000,3000000")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_shell_metrics(root: Path, relpath: str, n_images: int, state: int) -> tuple[np.ndarray | None, Path]:
    path = (
        root
        / f"n{n_images:08d}"
        / "runs"
        / f"n{n_images:08d}_seed0000"
        / relpath
        / f"state{state:04d}"
        / "shell_metrics.csv"
    )
    if not path.exists():
        return None, path
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return None, path
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data, path


def last_good_resolution(
    frequency: np.ndarray,
    values: np.ndarray,
    threshold: float,
    higher_is_better: bool,
) -> float:
    frequency = np.asarray(frequency, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0)
    frequency = frequency[valid]
    values = values[valid]
    if frequency.size == 0:
        return float("nan")
    good = values >= threshold if higher_is_better else values <= threshold
    good_indices = np.flatnonzero(good)
    if good_indices.size == 0:
        return float("nan")
    return float(1.0 / frequency[int(good_indices[-1])])


def short_n_label(n_images: int) -> str:
    if n_images >= 1_000_000:
        return f"{n_images / 1_000_000:g}M"
    return f"{n_images / 1000:g}k"


def plot_state_curves(
    out_dir: Path,
    root: Path,
    metrics_relpath: str,
    image_counts: list[int],
    state: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(image_counts)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)

    for color, n_images in zip(colors, image_counts):
        data, metrics_csv = load_shell_metrics(root, metrics_relpath, n_images, state)
        if data is None:
            continue
        frequency = data["frequency_1_per_A"]
        fsc = data["fsc_vs_gt"]
        relative_error = data["relative_error_per_shell"]
        fsc_resolution = last_good_resolution(frequency, fsc, threshold=0.5, higher_is_better=True)
        error_resolution = last_good_resolution(
            frequency, relative_error, threshold=0.5, higher_is_better=False
        )
        rows.append(
            {
                "n_images": n_images,
                "state": state,
                "fsc05_resolution_A": fsc_resolution,
                "relative_error_0p5_resolution_A": error_resolution,
                "metrics_csv": str(metrics_csv),
            }
        )
        label = short_n_label(n_images)
        axes[0].plot(
            frequency,
            fsc,
            color=color,
            lw=2.0,
            label=f"{label} ({fsc_resolution:.2f} A)" if np.isfinite(fsc_resolution) else label,
        )
        axes[1].semilogy(
            frequency,
            relative_error,
            color=color,
            lw=2.0,
            label=f"{label} ({error_resolution:.2f} A)" if np.isfinite(error_resolution) else label,
        )

    if not rows:
        plt.close(fig)
        return rows

    axes[0].axhline(0.5, color="0.35", ls="--", lw=1.0)
    axes[0].set_title(f"State {state}: masked FSC vs GT")
    axes[0].set_xlabel("spatial frequency (1/A)")
    axes[0].set_ylabel("FSC vs GT")
    axes[0].set_xlim(0.0, 0.40)
    axes[0].set_ylim(-0.05, 1.03)
    axes[0].legend(title="n, FSC0.5", fontsize=8)

    axes[1].axhline(0.5, color="0.35", ls="--", lw=1.0)
    axes[1].set_title(f"State {state}: masked relative shell error vs GT")
    axes[1].set_xlabel("spatial frequency (1/A)")
    axes[1].set_ylabel("relative Fourier shell error")
    axes[1].set_xlim(0.0, 0.40)
    axes[1].set_ylim(1e-3, 1e3)
    axes[1].legend(title="n, error<=0.5", fontsize=8)

    fig.suptitle(
        "Spike full-atom true pipeline sweep | zdim4 noreg | unfiltered maps | focus mask",
        fontsize=12,
        weight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"state{state:04d}_fsc_error_by_n.{ext}")
    plt.close(fig)
    return rows


def plot_resolution_vs_n(out_dir: Path, summary: list[dict[str, object]], states: list[int]) -> None:
    specs = [
        ("fsc05_resolution_A", "FSC=0.5 resolution (A), estimate vs GT", "fsc05_resolution_vs_n"),
        (
            "relative_error_0p5_resolution_A",
            "Resolution where relative shell error <= 0.5 (A)",
            "relerr05_resolution_vs_n",
        ),
    ]
    for metric, ylabel, stem in specs:
        fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
        for state, marker in zip(states, ["o", "s", "^", "D", "v"]):
            rows = [
                row
                for row in summary
                if int(row["state"]) == state and np.isfinite(float(row[metric]))
            ]
            rows.sort(key=lambda row: int(row["n_images"]))
            if not rows:
                continue
            ax.plot(
                [int(row["n_images"]) for row in rows],
                [float(row[metric]) for row in rows],
                marker=marker,
                lw=2.2,
                label=f"state {state}",
            )
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("number of images")
        ax.set_ylabel(ylabel)
        ax.set_title("Available completed points")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}")
        plt.close(fig)


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = out_dir / "available_resolution_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (int(row["state"]), int(row["n_images"]))))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir or (root / "09_combined_curves_current")
    out_dir.mkdir(parents=True, exist_ok=True)
    states = parse_int_list(args.states)
    image_counts = parse_int_list(args.image_counts)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )

    summary: list[dict[str, object]] = []
    for state in states:
        summary.extend(plot_state_curves(out_dir, root, args.metrics_relpath, image_counts, state))
    write_summary(out_dir, summary)
    plot_resolution_vs_n(out_dir, summary, states)

    print(out_dir)
    for row in sorted(summary, key=lambda item: (int(item["state"]), int(item["n_images"]))):
        print(
            f"state={row['state']:>2} n={row['n_images']:>8} "
            f"FSC0.5={float(row['fsc05_resolution_A']):.3f} A "
            f"err<=0.5={float(row['relative_error_0p5_resolution_A']):.3f} A"
        )


if __name__ == "__main__":
    main()
