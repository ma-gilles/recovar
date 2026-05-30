#!/usr/bin/env python3
"""Noise30 real-pipeline FSC/error curves with nearest-100 GT controls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.output import output as output_mod

from gt_embedding_controls import (
    recompute_latent_distances_from_compute_state,
    state_weights_from_nearest_distances,
)


DEFAULT_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518"
)
DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise30_b80_20260528/"
    "true_pipeline_movingfocus_nearest100_current"
)
DEFAULT_BROAD_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
DEFAULT_COUNTS = (30_000, 100_000, 300_000, 1_000_000, 3_000_000)
DEFAULT_STATE = 50
DEFAULT_COMPUTE_RELPATH = "07_compute_state_true_recovar_h100_fullmem_movingfocus_zdim4_reg_lazy"
VOXEL_SIZE_A = 1.25


def run_dir(root: Path, n_images: int) -> Path:
    label = f"n{n_images:08d}"
    return root / label / "runs" / f"{label}_seed0000"


def load_mrc(path: Path) -> np.ndarray:
    return np.asarray(utils.load_mrc(path), dtype=np.float32)


def dft3(volume: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume)))


def shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    labels = np.asarray(ftu.get_grid_of_radial_distances(shape, rounded=True), dtype=np.int32)
    n_shells = shape[0] // 2 - 1
    return np.clip(labels, 0, n_shells - 1), n_shells


def metric_context(target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    labels, n_shells = shell_labels(target.shape)
    target_ft = dft3(target * mask)
    target_power = np.bincount(
        labels.ravel(),
        weights=np.abs(target_ft).ravel() ** 2,
        minlength=n_shells,
    ).astype(np.float64)
    frequency = np.arange(n_shells, dtype=np.float64) / (target.shape[0] * VOXEL_SIZE_A)
    return labels, n_shells, target_ft, target_power, frequency


def masked_metrics(
    volume: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    target_ft: np.ndarray,
    target_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    volume_ft = dft3(volume * mask)
    flat = labels.ravel()
    cross = np.bincount(
        flat,
        weights=np.real(np.conj(volume_ft).ravel() * target_ft.ravel()),
        minlength=n_shells,
    )
    volume_power = np.bincount(flat, weights=np.abs(volume_ft).ravel() ** 2, minlength=n_shells)
    diff_power = np.bincount(flat, weights=np.abs(volume_ft - target_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(volume_power * target_power)
        relerr = diff_power / target_power
    fsc[~np.isfinite(fsc)] = 0.0
    relerr[~np.isfinite(relerr)] = np.nan
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return fsc.astype(np.float32), relerr.astype(np.float32)


def fsc05_resolution(frequency: np.ndarray, fsc: np.ndarray) -> float:
    valid = np.isfinite(fsc) & (frequency > 0)
    frequency = frequency[valid]
    fsc = fsc[valid]
    if frequency.size == 0:
        return float("nan")
    below = np.flatnonzero(fsc < 0.5)
    if below.size == 0:
        return float(1.0 / frequency[-1])
    idx = int(below[0])
    if idx == 0:
        return float(1.0 / frequency[0])
    y0, y1 = float(fsc[idx - 1]), float(fsc[idx])
    x0, x1 = float(frequency[idx - 1]), float(frequency[idx])
    crossing = x1 if y0 == y1 else x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return float(1.0 / crossing) if crossing > 0 else float("nan")


def relerr_resolution(frequency: np.ndarray, relerr: np.ndarray, threshold: float = 0.5) -> float:
    valid = np.isfinite(relerr) & (frequency > 0)
    frequency = frequency[valid]
    relerr = relerr[valid]
    if frequency.size == 0:
        return float("nan")
    good = np.flatnonzero(relerr <= threshold)
    if good.size == 0:
        return float("nan")
    return float(1.0 / frequency[int(good[-1])])


def weighted_gt_volume(run: Path, weights: np.ndarray) -> np.ndarray:
    total = None
    for state, weight in enumerate(weights):
        if weight == 0:
            continue
        volume = load_mrc(run / "04_ground_truth" / f"gt_vol{state:04d}.mrc")
        if total is None:
            total = np.zeros_like(volume, dtype=np.float64)
        total += float(weight) * volume
    if total is None:
        raise ValueError("Nearest GT mixture has all-zero weights")
    return total.astype(np.float32)


def interval_gt_weights(start: int, stop: int, n_states: int = 100) -> np.ndarray:
    start = max(0, int(start))
    stop = min(n_states - 1, int(stop))
    if start > stop:
        raise ValueError(f"Invalid interval {start}-{stop}")
    weights = np.zeros(n_states, dtype=np.float64)
    weights[start : stop + 1] = 1.0 / (stop - start + 1)
    return weights


def parse_intervals(spec: str) -> list[tuple[int, int]]:
    intervals = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.replace(":", "-").split("-")
        if len(pieces) != 2:
            raise ValueError(f"Expected intervals like 45-55, got {part!r}")
        intervals.append((int(pieces[0]), int(pieces[1])))
    return intervals


def target_mean_audit(
    run: Path,
    state: int,
    info,
) -> tuple[float, float]:
    assignments = np.asarray(np.load(run / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    z = np.asarray(
        output_mod.PipelineOutput(str(info.pipeline)).get_unsorted_embedding_component(
            info.coords_entry,
            info.zdim,
        ),
        dtype=np.float64,
    )
    z = z[:, : info.zdim]
    if z.shape[0] != assignments.size:
        raise ValueError(f"Embedding/state length mismatch: {z.shape[0]} vs {assignments.size}")
    state_mean = np.mean(z[assignments == state], axis=0, dtype=np.float64)
    diff = float(np.linalg.norm(state_mean - info.target))
    denom = float(np.linalg.norm(state_mean))
    return diff, diff / denom if denom > 0 else float("nan")


def plot_curves(
    rows: list[dict[str, object]],
    out_dir: Path,
    state: int,
    nearest_count: int,
    gt_average_rows: Sequence[dict[str, object]],
    mask_label: str,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), constrained_layout=True)
    gt_styles = [("0.15", ":"), ("0.35", "-."), ("0.55", (0, (5, 2, 1, 2))), ("0.72", (0, (2, 2)))]
    for color, row in zip(colors, rows):
        n_images = int(row["n_images"])
        label = f"{n_images // 1000}k" if n_images < 1_000_000 else f"{n_images / 1_000_000:g}M"
        frequency = row["frequency"]
        axes[0].plot(frequency, row["estimate_fsc"], color=color, lw=2.1, label=f"{label} estimate")
        axes[0].plot(
            frequency,
            row["nearest_fsc"],
            color=color,
            lw=1.8,
            ls="--",
            label=f"{label} nearest-{nearest_count} GT",
        )
        axes[1].semilogy(frequency, np.maximum(row["estimate_error"], 1e-30), color=color, lw=2.1)
        axes[1].semilogy(
            frequency,
            np.maximum(row["nearest_error"], 1e-30),
            color=color,
            lw=1.8,
            ls="--",
        )
    if rows:
        frequency = rows[0]["frequency"]
        for idx, baseline in enumerate(gt_average_rows):
            color, linestyle = gt_styles[idx % len(gt_styles)]
            label = str(baseline["label"])
            axes[0].plot(
                frequency,
                baseline["fsc"],
                color=color,
                lw=2.0,
                ls=linestyle,
                label=label,
            )
            axes[1].semilogy(
                frequency,
                np.maximum(baseline["error"], 1e-30),
                color=color,
                lw=2.0,
                ls=linestyle,
            )
    axes[0].axhline(0.5, color="0.35", ls=":", lw=1.2)
    axes[0].set_title("FSC vs GT")
    axes[0].set_ylabel("masked FSC")
    axes[0].set_ylim(-0.05, 1.03)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].axhline(0.5, color="0.35", ls=":", lw=1.2)
    axes[1].set_title("Relative Fourier shell error")
    axes[1].set_ylabel("masked relative error")
    axes[1].set_ylim(1e-3, 1e3)
    for ax in axes:
        ax.set_xlim(0.0, 0.4)
        ax.set_xlabel("spatial frequency (1/A)")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"State {state} | noise30 true pipeline zdim4 reg | {mask_label} | solid=compute_state, dashed=nearest-{nearest_count} GT",
        fontsize=12,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"state{state:04d}_noise30_true_pipeline_estimate_vs_nearest{nearest_count}_gt_by_n.{ext}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image-counts", default=",".join(str(v) for v in DEFAULT_COUNTS))
    parser.add_argument("--state", type=int, default=DEFAULT_STATE)
    parser.add_argument("--nearest-count", type=int, default=100)
    parser.add_argument("--compute-relpath", default=DEFAULT_COMPUTE_RELPATH)
    parser.add_argument("--mask", type=Path, default=DEFAULT_BROAD_MASK)
    parser.add_argument("--gt-average-intervals", default="45-55,40-60,30-70")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.mask.resolve()
    mask = load_mrc(mask_path)
    mask_label = mask_path.name
    image_counts = [int(part) for part in args.image_counts.split(",") if part]
    gt_average_intervals = parse_intervals(args.gt_average_intervals)
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    curve_npz: dict[str, np.ndarray] = {}
    frequency_for_npz = None
    gt_average_rows: list[dict[str, object]] | None = None
    for n_images in image_counts:
        run = run_dir(root, n_images)
        compute_root = run / args.compute_relpath
        estimate_path = compute_root / "state000_unfil.mrc"
        if not estimate_path.exists():
            print(f"SKIP missing estimate: {estimate_path}", flush=True)
            continue
        target = load_mrc(run / "04_ground_truth" / f"gt_vol{args.state:04d}.mrc")
        if mask.shape != target.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match target shape {target.shape}: {mask_path}")
        labels, n_shells, target_ft, target_power, frequency = metric_context(target, mask)
        estimate_fsc, estimate_error = masked_metrics(
            load_mrc(estimate_path),
            mask,
            labels,
            n_shells,
            target_ft,
            target_power,
        )
        distances, info = recompute_latent_distances_from_compute_state(compute_root)
        target_mean_diff, target_mean_rel_diff = target_mean_audit(run, args.state, info)
        weights, nearest = state_weights_from_nearest_distances(
            run,
            distances,
            args.nearest_count,
            n_states=100,
        )
        nearest_fsc, nearest_error = masked_metrics(
            weighted_gt_volume(run, weights),
            mask,
            labels,
            n_shells,
            target_ft,
            target_power,
        )
        if gt_average_rows is None:
            gt_average_rows = []
            for start, stop in gt_average_intervals:
                interval_weights = interval_gt_weights(start, stop, n_states=100)
                fsc, error = masked_metrics(
                    weighted_gt_volume(run, interval_weights),
                    mask,
                    labels,
                    n_shells,
                    target_ft,
                    target_power,
                )
                label = f"GT avg {start}-{stop}"
                gt_average_rows.append(
                    {
                        "label": label,
                        "start": start,
                        "stop": stop,
                        "fsc": fsc,
                        "error": error,
                    }
                )
                curve_npz[f"gt_avg_{start}_{stop}_fsc"] = fsc
                curve_npz[f"gt_avg_{start}_{stop}_error"] = error
        assignments = np.asarray(np.load(run / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
        counts = np.bincount(assignments[nearest], minlength=100).astype(np.float64)
        top_states = [
            f"{idx}:{int(counts[idx])}"
            for idx in np.argsort(counts)[::-1][:12]
            if counts[idx] > 0
        ]
        rows.append(
            {
                "n_images": n_images,
                "frequency": frequency,
                "estimate_fsc": estimate_fsc,
                "estimate_error": estimate_error,
                "nearest_fsc": nearest_fsc,
                "nearest_error": nearest_error,
            }
        )
        frequency_for_npz = frequency
        prefix = f"n{n_images:08d}_state{args.state:04d}"
        curve_npz[f"{prefix}_estimate_fsc"] = estimate_fsc
        curve_npz[f"{prefix}_estimate_error"] = estimate_error
        curve_npz[f"{prefix}_nearest{args.nearest_count}_gt_fsc"] = nearest_fsc
        curve_npz[f"{prefix}_nearest{args.nearest_count}_gt_error"] = nearest_error
        summary_rows.append(
            {
                "n_images": n_images,
                "state": args.state,
                "estimate_fsc05_resolution_A": fsc05_resolution(frequency, estimate_fsc),
                "estimate_relerr0p5_resolution_A": relerr_resolution(frequency, estimate_error, 0.5),
                f"nearest{args.nearest_count}_gt_fsc05_resolution_A": fsc05_resolution(frequency, nearest_fsc),
                f"nearest{args.nearest_count}_gt_relerr0p5_resolution_A": relerr_resolution(
                    frequency, nearest_error, 0.5
                ),
                f"nearest{args.nearest_count}_target_state_fraction": float(weights[args.state]),
                "target_minus_state_mean_embedding_l2": target_mean_diff,
                "target_minus_state_mean_embedding_relative_l2": target_mean_rel_diff,
                f"nearest{args.nearest_count}_top_states": " ".join(top_states),
                "mask": str(mask_path),
                "compute_root": str(compute_root),
                "pipeline_dir": str(info.pipeline),
                "target_path": str(info.target_path),
                "coords_entry": info.coords_entry,
                "precision_entry": info.precision_entry,
                "embedding_option": info.embedding_option,
            }
        )
    if not rows:
        raise RuntimeError("No completed compute_state outputs found")
    if gt_average_rows is None:
        raise RuntimeError("No GT average baselines were computed")
    rows.sort(key=lambda row: int(row["n_images"]))
    summary_rows.sort(key=lambda row: int(row["n_images"]))
    plot_curves(rows, out_dir, args.state, args.nearest_count, gt_average_rows, mask_label)
    gt_summary_rows = []
    for baseline in gt_average_rows:
        start = int(baseline["start"])
        stop = int(baseline["stop"])
        gt_summary_rows.append(
            {
                "state": args.state,
                "gt_average_interval": f"{start}-{stop}",
                "gt_average_fsc05_resolution_A": fsc05_resolution(rows[0]["frequency"], baseline["fsc"]),
                "gt_average_relerr0p5_resolution_A": relerr_resolution(rows[0]["frequency"], baseline["error"], 0.5),
                "mask": str(mask_path),
            }
        )
    with (out_dir / f"state{args.state:04d}_estimate_vs_nearest{args.nearest_count}_gt_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / f"state{args.state:04d}_gt_average_baselines_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gt_summary_rows[0]))
        writer.writeheader()
        writer.writerows(gt_summary_rows)
    if frequency_for_npz is not None:
        np.savez_compressed(
            out_dir / f"state{args.state:04d}_estimate_vs_nearest{args.nearest_count}_gt_curves.npz",
            frequency_1_per_A=frequency_for_npz,
            **curve_npz,
        )
    print(out_dir)
    for row in summary_rows:
        print(
            f"n={row['n_images']:>8} state={row['state']} "
            f"estimate_FSC0.5={row['estimate_fsc05_resolution_A']:.3f} A "
            f"nearest{args.nearest_count}_FSC0.5={row[f'nearest{args.nearest_count}_gt_fsc05_resolution_A']:.3f} A "
            f"nearest_target_frac={row[f'nearest{args.nearest_count}_target_state_fraction']:.3f} "
            f"target_mean_diff={row['target_minus_state_mean_embedding_l2']:.3g}"
        )
    for row in gt_summary_rows:
        print(
            f"{row['gt_average_interval']} GT avg FSC0.5="
            f"{row['gt_average_fsc05_resolution_A']:.3f} A "
            f"relerr0.5={row['gt_average_relerr0p5_resolution_A']:.3f} A"
        )


if __name__ == "__main__":
    main()
