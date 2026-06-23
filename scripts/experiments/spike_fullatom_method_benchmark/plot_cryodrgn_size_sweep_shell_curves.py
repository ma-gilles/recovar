#!/usr/bin/env python3
"""Plot cryoDRGN size-sweep FSC and FSC-error shell curves from saved metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EVAL_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_zdim1_correct_sign_noise10_b100_20260529/"
    "evaluation_size_sweep_zdim1_correct_sign"
)
DEFAULT_METRICS_CSV = DEFAULT_EVAL_DIR / "metrics" / "cryodrgn_zdim1_size_sweep_metrics.csv"
DEFAULT_CURVES_NPZ = (
    DEFAULT_EVAL_DIR / "metrics" / "cryodrgn_zdim1_size_sweep_fsc_error_curves.npz"
)
DEFAULT_OUT_DIR = DEFAULT_EVAL_DIR / "09_combined_curves_current"
DEFAULT_IMAGE_COUNTS = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)
DEFAULT_STATES = (0, 25, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--curves-npz", type=Path, default=DEFAULT_CURVES_NPZ)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--states", default=",".join(str(state) for state in DEFAULT_STATES))
    parser.add_argument(
        "--image-counts",
        default=",".join(str(n_images) for n_images in DEFAULT_IMAGE_COUNTS),
    )
    parser.add_argument(
        "--masks",
        default="focus,global",
        help="Comma-separated metric_mask values to plot. focus is first by default.",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=None,
        help="Fourier box size used for frequency labels. Defaults to 2 * (n_shells + 1).",
    )
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip().replace("_", "")) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def row_index(rows: list[dict[str, str]]) -> dict[tuple[str, int, int], dict[str, str]]:
    index: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        key = (row["metric_mask"], int(row["gt_label"]), int(row["n_images"]))
        index[key] = row
    return index


def short_n_label(n_images: int) -> str:
    if n_images >= 1_000_000:
        return f"{n_images / 1_000_000:g}M"
    return f"{n_images / 1000:g}k"


def curve_key(mask: str, state: int, n_label: str, metric: str) -> str:
    return f"{mask}_state{state:04d}_{n_label}_{metric}"


def frequency_axis(n_shells: int, voxel_size: float, box_size: int | None) -> tuple[np.ndarray, int]:
    resolved_box_size = box_size if box_size is not None else 2 * (n_shells + 1)
    freq = np.arange(n_shells, dtype=np.float64) / (resolved_box_size * voxel_size)
    return freq, resolved_box_size


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


def plot_state_curves(
    out_dir: Path,
    curves: np.lib.npyio.NpzFile,
    rows_by_key: dict[tuple[str, int, int], dict[str, str]],
    mask: str,
    state: int,
    image_counts: list[int],
    box_size: int | None,
    title: str,
) -> list[dict[str, object]]:
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(image_counts)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
    summary: list[dict[str, object]] = []
    resolved_box_size: int | None = None

    for color, n_images in zip(colors, image_counts):
        row = rows_by_key.get((mask, state, n_images))
        if row is None:
            continue
        n_label = row["n_label"]
        fsc_key = curve_key(mask, state, n_label, "fsc")
        error_key = curve_key(mask, state, n_label, "error")
        if fsc_key not in curves or error_key not in curves:
            continue

        fsc = np.asarray(curves[fsc_key], dtype=np.float64)
        fsc_error = np.asarray(curves[error_key], dtype=np.float64)
        voxel_size = float(row["voxel_size"])
        frequency, resolved_box_size = frequency_axis(fsc.size, voxel_size, box_size)

        fsc_resolution = last_good_resolution(
            frequency, fsc, threshold=0.5, higher_is_better=True
        )
        fsc_error_resolution = last_good_resolution(
            frequency, fsc_error, threshold=0.5, higher_is_better=False
        )
        summary.append(
            {
                "metric_mask": mask,
                "state": state,
                "n_images": n_images,
                "n_label": n_label,
                "voxel_size_A": voxel_size,
                "box_size": resolved_box_size,
                "fsc05_resolution_A": fsc_resolution,
                "fsc_error_0p5_resolution_A": fsc_error_resolution,
                "csv_fsc05_resolution_A": float(row["fsc_res_0p5_A"]),
                "csv_fsc1over7_resolution_A": float(row["fsc_res_1over7_A"]),
                "fsc_auc": float(row["fsc_auc"]),
                "fsc_error_auc": float(row["fsc_error_auc"]),
                "masked_rel_l2": float(row["masked_rel_l2"]),
                "masked_corr": float(row["masked_corr"]),
                "volume": row["volume"],
                "gt_volume": row["gt_volume"],
            }
        )

        label = short_n_label(n_images)
        axes[0].plot(
            frequency,
            fsc,
            color=color,
            lw=2.0,
            label=f"{label} ({fsc_resolution:.2f} A)",
        )
        axes[1].semilogy(
            frequency,
            np.maximum(fsc_error, 1e-6),
            color=color,
            lw=2.0,
            label=(
                f"{label} ({fsc_error_resolution:.2f} A)"
                if np.isfinite(fsc_error_resolution)
                else label
            ),
        )

    if not summary:
        plt.close(fig)
        return summary

    axes[0].axhline(0.5, color="0.35", ls="--", lw=1.0)
    axes[0].set_title(f"GT state {state}: {mask} mask FSC vs GT")
    axes[0].set_xlabel("spatial frequency (1/A)")
    axes[0].set_ylabel("FSC vs GT")
    axes[0].set_xlim(0.0, 0.40)
    axes[0].set_ylim(-0.05, 1.03)
    axes[0].legend(title="n, FSC0.5", fontsize=8)

    axes[1].axhline(0.5, color="0.35", ls="--", lw=1.0)
    axes[1].set_title(f"GT state {state}: {mask} mask FSC-error vs GT")
    axes[1].set_xlabel("spatial frequency (1/A)")
    axes[1].set_ylabel("1 - FSC")
    axes[1].set_xlim(0.0, 0.40)
    axes[1].set_ylim(1e-4, 1.2)
    axes[1].legend(title="n, error<=0.5", fontsize=8)

    fig.suptitle(
        f"{title} | {mask} mask | available completed points",
        fontsize=12,
        weight="bold",
    )
    stem = f"cryodrgn_zdim1_{mask}_state{state:04d}_fsc_error_by_n"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    return summary


def plot_resolution_vs_n(
    out_dir: Path,
    summary: list[dict[str, object]],
    states: list[int],
    mask: str,
) -> None:
    specs = [
        ("fsc05_resolution_A", "FSC=0.5 resolution (A), estimate vs GT", "fsc05_resolution_vs_n"),
        (
            "fsc_error_0p5_resolution_A",
            "Resolution where FSC-error <= 0.5 (A)",
            "fsc_error05_resolution_vs_n",
        ),
    ]
    for metric, ylabel, stem in specs:
        fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
        for state, marker in zip(states, ["o", "s", "^", "D", "v"]):
            rows = [
                row
                for row in summary
                if row["metric_mask"] == mask
                and int(row["state"]) == state
                and np.isfinite(float(row[metric]))
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
        ax.set_title(f"{mask} mask | available completed points")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"cryodrgn_zdim1_{mask}_{stem}.{ext}")
        plt.close(fig)


def plot_auc_vs_n(
    out_dir: Path,
    summary: list[dict[str, object]],
    states: list[int],
    mask: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    for state, marker in zip(states, ["o", "s", "^", "D", "v"]):
        rows = [
            row
            for row in summary
            if row["metric_mask"] == mask and int(row["state"]) == state
        ]
        rows.sort(key=lambda row: int(row["n_images"]))
        if not rows:
            continue
        xs = [int(row["n_images"]) for row in rows]
        axes[0].plot(xs, [float(row["fsc_auc"]) for row in rows], marker=marker, lw=2.2, label=f"state {state}")
        axes[1].plot(
            xs,
            [float(row["fsc_error_auc"]) for row in rows],
            marker=marker,
            lw=2.2,
            label=f"state {state}",
        )

    axes[0].set_ylabel("FSC AUC")
    axes[0].set_title("Higher is better")
    axes[1].set_ylabel("FSC-error AUC")
    axes[1].set_title("Lower is better")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("number of images")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.suptitle(f"cryoDRGN zdim1 size sweep | {mask} mask AUC", fontsize=12, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"cryodrgn_zdim1_{mask}_auc_vs_n.{ext}")
    plt.close(fig)


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = out_dir / "cryodrgn_zdim1_available_resolution_summary.csv"
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (str(row["metric_mask"]), int(row["state"]), int(row["n_images"])))
        )


def main() -> None:
    args = parse_args()
    rows = load_metric_rows(args.metrics_csv)
    rows_by_key = row_index(rows)
    states = parse_int_list(args.states)
    image_counts = parse_int_list(args.image_counts)
    masks = parse_str_list(args.masks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )

    run_names = sorted({row["run_name"] for row in rows if row.get("run_name")})
    title = args.title or "cryoDRGN " + ("/".join(run_names) if run_names else "size sweep")

    summary: list[dict[str, object]] = []
    with np.load(args.curves_npz, allow_pickle=False) as curves:
        for mask in masks:
            for state in states:
                summary.extend(
                    plot_state_curves(
                        args.out_dir,
                        curves,
                        rows_by_key,
                        mask,
                        state,
                        image_counts,
                        args.box_size,
                        title,
                    )
                )

    write_summary(args.out_dir, summary)
    for mask in masks:
        plot_resolution_vs_n(args.out_dir, summary, states, mask)
        plot_auc_vs_n(args.out_dir, summary, states, mask)

    print(args.out_dir)
    for row in sorted(summary, key=lambda item: (str(item["metric_mask"]), int(item["state"]), int(item["n_images"]))):
        print(
            f"mask={row['metric_mask']:<6} state={int(row['state']):>2} "
            f"n={int(row['n_images']):>8} FSC0.5={float(row['fsc05_resolution_A']):.3f} A "
            f"FSC-error<=0.5={float(row['fsc_error_0p5_resolution_A']):.3f} A"
        )


if __name__ == "__main__":
    main()
