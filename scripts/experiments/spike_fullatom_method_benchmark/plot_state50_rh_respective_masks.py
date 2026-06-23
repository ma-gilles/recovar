#!/usr/bin/env python3
"""Make state-50 Rosenthal-Henderson plots with mask-matched summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


METHOD_ORDER = ("recovar", "cryodrgn", "3dflex")
METHOD_LABELS = {
    "recovar": "RECOVAR",
    "cryodrgn": "cryoDRGN",
    "3dflex": "3DFlex",
}
METHOD_COLORS = {
    "recovar": "#1b9e77",
    "cryodrgn": "#d95f02",
    "3dflex": "#7570b3",
}
METHOD_MARKERS = {
    "recovar": "o",
    "cryodrgn": "s",
    "3dflex": "^",
}
METHOD_ANNOTATION_OFFSETS = {
    "recovar": (0, 10),
    "cryodrgn": (0, -15),
    "3dflex": (0, 10),
}
MASK_LABELS = {
    "broad_mask": "Moving region, broad mask",
    "not_moving_mask": "Non-moving region, soft mask",
}
N_ORDER = (10_000, 30_000, 100_000, 300_000, 1_000_000)


@dataclass(frozen=True)
class NoiseConfig:
    noise: int
    root: Path
    presentation_dir: Path


CONFIGS = {
    1: NoiseConfig(
        noise=1,
        root=Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530"),
        presentation_dir=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530/"
            "state50_presentation_movies_upto1m_20260603"
        ),
    ),
    3: NoiseConfig(
        noise=3,
        root=Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531"),
        presentation_dir=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/"
            "state50_presentation_movies_upto1m_20260603"
        ),
    ),
}


def _summary_path(config: NoiseConfig, mask_key: str) -> Path:
    mode = "moving" if mask_key == "broad_mask" else "notmoving"
    return config.presentation_dir / f"state50_noise{config.noise}_{mode}_fsc_upto1m_summary.csv"


def _read_rows(path: Path, noise: int, mask_key: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in METHOD_ORDER:
                continue
            n_images = int(row["n_images"])
            if n_images not in N_ORDER:
                continue
            resolution = float(row["resolution_A"])
            rows.append(
                {
                    "noise": noise,
                    "mask": mask_key,
                    "mask_label": MASK_LABELS[mask_key],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n_images": n_images,
                    "n_label": row["n_label"],
                    "resolution_A": resolution,
                    "inv_d2_1_per_A2": 1.0 / (resolution * resolution) if math.isfinite(resolution) and resolution > 0 else math.nan,
                    "source_summary": str(path),
                    "shell_metrics_csv": row.get("shell_metrics_csv", ""),
                }
            )
    rows.sort(key=lambda row: (METHOD_ORDER.index(str(row["method"])), int(row["n_images"])))
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_particles(value: float, _pos: object | None = None) -> str:
    closest = min(N_ORDER, key=lambda item: abs(float(item) - float(value)))
    if abs(float(closest) - float(value)) / float(closest) < 0.01:
        return f"{closest:,}"
    return f"{int(value):,}"


def _format_resolution(value: float) -> str:
    return f"{value:.1f} A"


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xticks(N_ORDER)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_particles))
    ax.tick_params(axis="x", labelsize=9, rotation=0, pad=6)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, which="major", color="#d7dde2", lw=0.8)
    ax.grid(True, which="minor", axis="x", color="#edf0f2", lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#8b949e")
    ax.spines["bottom"].set_color("#8b949e")


def _pad_linear_ylim(values: list[float], *, invert: bool = False) -> tuple[float, float]:
    finite = np.array([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return (0.0, 1.0)
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if ymin == ymax:
        pad = max(abs(ymin) * 0.08, 0.05)
    else:
        pad = (ymax - ymin) * 0.16
    lo = ymin - pad
    hi = ymax + pad
    return (hi, lo) if invert else (lo, hi)


def _annotate_points(
    ax: plt.Axes,
    x: list[int],
    y: list[float],
    labels: list[float],
    method: str,
) -> None:
    dx, dy = METHOD_ANNOTATION_OFFSETS[method]
    for xi, yi, label in zip(x, y, labels, strict=True):
        if not math.isfinite(yi):
            continue
        ax.annotate(
            _format_resolution(label),
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="semibold",
            color="#202428",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "none",
                "alpha": 0.78,
            },
            zorder=5,
        )


def _fit_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fits: list[dict[str, object]] = []
    for noise in sorted({int(row["noise"]) for row in rows}):
        for mask_key in ("broad_mask", "not_moving_mask"):
            for method in METHOD_ORDER:
                subset = [
                    row
                    for row in rows
                    if int(row["noise"]) == noise
                    and row["mask"] == mask_key
                    and row["method"] == method
                    and math.isfinite(float(row["inv_d2_1_per_A2"]))
                ]
                if len(subset) < 2:
                    continue
                subset.sort(key=lambda row: int(row["n_images"]))
                n0 = int(subset[0]["n_images"])
                x = np.array([math.log(int(row["n_images"]) / n0) for row in subset], dtype=np.float64)
                y = np.array([float(row["inv_d2_1_per_A2"]) for row in subset], dtype=np.float64)
                slope, intercept = np.polyfit(x, y, deg=1)
                pred = slope * x + intercept
                ss_res = float(np.sum((y - pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = math.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
                fits.append(
                    {
                        "noise": noise,
                        "mask": mask_key,
                        "mask_label": MASK_LABELS[mask_key],
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "n_min": min(int(row["n_images"]) for row in subset),
                        "n_max": max(int(row["n_images"]) for row in subset),
                        "num_points": len(subset),
                        "rh_slope_1_per_A2_per_lnN": slope,
                        "B_eff_if_slope_1_over_B_A2": 1.0 / slope if slope > 0 else math.nan,
                        "B_eff_if_slope_2_over_B_A2": 2.0 / slope if slope > 0 else math.nan,
                        "intercept": intercept,
                        "r2": r2,
                    }
                )
    return fits


def _plot_resolution(config: NoiseConfig, rows: list[dict[str, object]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), sharey=False, constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, mask_key in zip(axes, ("broad_mask", "not_moving_mask"), strict=True):
        panel_values: list[float] = []
        for method in METHOD_ORDER:
            subset = [
                row
                for row in rows
                if int(row["noise"]) == config.noise and row["mask"] == mask_key and row["method"] == method
            ]
            subset.sort(key=lambda row: int(row["n_images"]))
            if not subset:
                continue
            x = [int(row["n_images"]) for row in subset]
            y = [float(row["resolution_A"]) for row in subset]
            panel_values.extend(y)
            ax.plot(
                x,
                y,
                marker=METHOD_MARKERS[method],
                lw=2.6,
                ms=6.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                solid_capstyle="round",
                zorder=3,
            )
            _annotate_points(ax, x, y, y, method)
        _style_axis(ax)
        ax.set_ylim(*_pad_linear_ylim(panel_values, invert=True))
        ax.set_title(MASK_LABELS[mask_key], fontsize=13, weight="bold", pad=10)
        ax.set_xlabel("Number of particles", fontsize=11)
    axes[0].set_ylabel("FSC=0.5 resolution (A), lower is better", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10,
    )
    fig.suptitle(f"Noise {config.noise} | state 50 | FSC=0.5 resolution scaling", fontsize=15, weight="bold", y=1.08)
    out = out_dir / f"noise{config.noise}_state50_respective_masks_resolution_vs_n.png"
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def _plot_rh_transform(config: NoiseConfig, rows: list[dict[str, object]], fits: list[dict[str, object]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), sharey=False, constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, mask_key in zip(axes, ("broad_mask", "not_moving_mask"), strict=True):
        panel_values: list[float] = []
        for method in METHOD_ORDER:
            subset = [
                row
                for row in rows
                if int(row["noise"]) == config.noise and row["mask"] == mask_key and row["method"] == method
            ]
            subset.sort(key=lambda row: int(row["n_images"]))
            if not subset:
                continue
            n0 = int(subset[0]["n_images"])
            x = np.array([int(row["n_images"]) for row in subset], dtype=np.float64)
            y = np.array([float(row["inv_d2_1_per_A2"]) for row in subset], dtype=np.float64)
            panel_values.extend(y.tolist())
            ax.plot(
                x,
                y,
                marker=METHOD_MARKERS[method],
                lw=2.6,
                ms=6.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                solid_capstyle="round",
                zorder=3,
            )
            _annotate_points(
                ax,
                x.astype(int).tolist(),
                y.astype(float).tolist(),
                [float(row["resolution_A"]) for row in subset],
                method,
            )
            fit = next(
                (
                    item
                    for item in fits
                    if int(item["noise"]) == config.noise and item["mask"] == mask_key and item["method"] == method
                ),
                None,
            )
            if fit is not None:
                xx = np.geomspace(float(np.min(x)), float(np.max(x)), 100)
                yy = (
                    float(fit["intercept"])
                    + float(fit["rh_slope_1_per_A2_per_lnN"]) * np.log(xx / float(n0))
                )
                ax.plot(xx, yy, color=METHOD_COLORS[method], lw=1.5, alpha=0.32, zorder=2)
        _style_axis(ax)
        ax.set_ylim(*_pad_linear_ylim(panel_values, invert=False))
        ax.set_title(MASK_LABELS[mask_key], fontsize=13, weight="bold", pad=10)
        ax.set_xlabel("Number of particles", fontsize=11)
    axes[0].set_ylabel("RH coordinate: 1 / resolution^2 (1/A^2)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10,
    )
    fig.suptitle(f"Noise {config.noise} | state 50 | Rosenthal-Henderson transform", fontsize=15, weight="bold", y=1.08)
    out = out_dir / f"noise{config.noise}_state50_respective_masks_RH_transform.png"
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def _run_one(config: NoiseConfig, args: argparse.Namespace) -> dict[str, object]:
    out_dir = config.root / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for mask_key in ("broad_mask", "not_moving_mask"):
        rows.extend(_read_rows(_summary_path(config, mask_key), config.noise, mask_key))
    fits = _fit_rows(rows)
    summary_csv = out_dir / f"noise{config.noise}_state50_respective_masks_rh_summary.csv"
    fits_csv = out_dir / f"noise{config.noise}_state50_respective_masks_RH_effective_B_fits.csv"
    _write_csv(summary_csv, rows)
    _write_csv(fits_csv, fits)
    resolution_plot = _plot_resolution(config, rows, out_dir)
    rh_plot = _plot_rh_transform(config, rows, fits, out_dir)
    audit = {
        "script": str(Path(__file__).resolve()),
        "noise": config.noise,
        "output_dir": str(out_dir),
        "summaries_used": {
            mask_key: str(_summary_path(config, mask_key)) for mask_key in ("broad_mask", "not_moving_mask")
        },
        "mask_labels": MASK_LABELS,
        "n_order": list(N_ORDER),
        "summary_csv": str(summary_csv),
        "fits_csv": str(fits_csv),
        "plots": [str(resolution_plot), str(rh_plot), str(resolution_plot.with_suffix(".pdf")), str(rh_plot.with_suffix(".pdf"))],
    }
    (out_dir / f"noise{config.noise}_state50_respective_masks_RH_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=int, action="append", choices=sorted(CONFIGS), help="Noise level to process; repeatable. Default: 1 and 3.")
    parser.add_argument("--output-name", default="state50_rh_respective_masks_20260603")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    noise_levels = args.noise or [1, 3]
    outputs = {str(noise): _run_one(CONFIGS[noise], args) for noise in noise_levels}
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
