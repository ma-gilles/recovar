#!/usr/bin/env python3
"""Plot masked FSC and Fourier relative error for a compute_state result."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu


DEFAULT_RUN_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/"
    "n00100000/runs/n00100000_seed0000"
)
DEFAULT_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)


def dft3(volume: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume)))


def shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    labels = np.asarray(ftu.get_grid_of_radial_distances(shape, rounded=True), dtype=np.int32)
    n_shells = shape[0] // 2 - 1
    return np.clip(labels, 0, n_shells - 1), n_shells


def fsc_resolution(freq: np.ndarray, fsc: np.ndarray, threshold: float = 0.5) -> float:
    valid = np.isfinite(fsc) & (freq > 0)
    freq = freq[valid]
    fsc = fsc[valid]
    if freq.size == 0:
        return float("nan")
    below = np.flatnonzero(fsc < threshold)
    if below.size == 0:
        return float(1.0 / freq[-1])
    idx = int(below[0])
    if idx == 0:
        return float(1.0 / freq[0])
    x0, x1 = float(fsc[idx - 1]), float(fsc[idx])
    f0, f1 = float(freq[idx - 1]), float(freq[idx])
    crossing = f1 if x0 == x1 else f0 + (threshold - x0) * (f1 - f0) / (x1 - x0)
    return float(1.0 / crossing) if crossing > 0 else float("nan")


def error_resolution(freq: np.ndarray, rel_error: np.ndarray, threshold: float = 0.5) -> float:
    valid = np.isfinite(rel_error) & (freq > 0)
    freq = freq[valid]
    rel_error = rel_error[valid]
    if freq.size == 0:
        return float("nan")
    above = np.flatnonzero(rel_error > threshold)
    if above.size == 0:
        return float(1.0 / freq[-1])
    idx = int(above[0])
    return float(1.0 / freq[idx]) if freq[idx] > 0 else float("nan")


def shell_metrics(estimate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    labels, n_shells = shell_labels(estimate.shape)
    estimate_ft = dft3(estimate * mask)
    target_ft = dft3(target * mask)
    diff_ft = estimate_ft - target_ft

    flat = labels.ravel()
    cross = np.bincount(
        flat,
        weights=np.real(np.conj(estimate_ft).ravel() * target_ft.ravel()),
        minlength=n_shells,
    )
    estimate_power = np.bincount(flat, weights=np.abs(estimate_ft).ravel() ** 2, minlength=n_shells)
    target_power = np.bincount(flat, weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    diff_power = np.bincount(flat, weights=np.abs(diff_ft).ravel() ** 2, minlength=n_shells)

    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(estimate_power * target_power)
        rel_error = diff_power / target_power
        cumulative_rel_error = np.cumsum(diff_power) / np.cumsum(target_power)

    fsc[~np.isfinite(fsc)] = 0.0
    rel_error[~np.isfinite(rel_error)] = np.nan
    cumulative_rel_error[~np.isfinite(cumulative_rel_error)] = np.nan
    if fsc.size > 1:
        fsc[0] = fsc[1]

    return {
        "fsc": fsc.astype(np.float32),
        "relative_error": rel_error.astype(np.float32),
        "cumulative_relative_error": cumulative_rel_error.astype(np.float32),
        "target_power": target_power.astype(np.float64),
        "diff_power": diff_power.astype(np.float64),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--compute-state-dir", type=Path, default=None)
    parser.add_argument("--estimate", default="state000_unfil.mrc")
    parser.add_argument("--target-state", type=int, default=50)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--voxel-size", type=float, default=1.25)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    compute_state_dir = args.compute_state_dir or (run_dir / "07_compute_state")
    estimate_path = compute_state_dir / args.estimate
    gt_path = args.gt or (run_dir / "04_ground_truth" / f"gt_vol{args.target_state:04d}.mrc")
    out_dir = args.out_dir or (run_dir / "plots" / "compute_state_shell_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    estimate = np.asarray(utils.load_mrc(estimate_path), dtype=np.float32)
    target = np.asarray(utils.load_mrc(gt_path), dtype=np.float32)
    mask = np.clip(np.asarray(utils.load_mrc(args.mask), dtype=np.float32), 0.0, 1.0)
    if estimate.shape != target.shape or estimate.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: estimate={estimate.shape}, gt={target.shape}, mask={mask.shape}"
        )

    metrics = shell_metrics(estimate, target, mask)
    freq = np.arange(metrics["fsc"].size, dtype=np.float64) / (estimate.shape[0] * args.voxel_size)
    resolution = np.divide(1.0, freq, out=np.full_like(freq, np.inf), where=freq > 0)

    summary = {
        "estimate": str(estimate_path),
        "gt": str(gt_path),
        "mask": str(args.mask),
        "fsc_05_resolution_A": fsc_resolution(freq, metrics["fsc"], 0.5),
        "relative_error_0p5_resolution_A": error_resolution(freq, metrics["relative_error"], 0.5),
    }

    with (out_dir / "shell_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shell",
                "frequency_1_per_A",
                "resolution_A",
                "fsc_vs_gt",
                "relative_error_per_shell",
                "cumulative_relative_error",
                "target_power",
                "diff_power",
            ]
        )
        for i in range(freq.size):
            writer.writerow(
                [
                    i,
                    freq[i],
                    resolution[i],
                    metrics["fsc"][i],
                    metrics["relative_error"][i],
                    metrics["cumulative_relative_error"][i],
                    metrics["target_power"][i],
                    metrics["diff_power"][i],
                ]
            )

    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"{run_dir.name}: compute_state vs GT state {args.target_state}")

    axes[0, 0].plot(freq, metrics["fsc"], color="black", lw=2)
    axes[0, 0].axhline(0.5, color="0.45", ls="--", lw=1)
    axes[0, 0].set_title(f"FSC vs GT, FSC0.5 = {summary['fsc_05_resolution_A']:.2f} A")
    axes[0, 0].set_ylabel("FSC")
    axes[0, 0].set_ylim(-0.05, 1.03)

    axes[0, 1].plot(freq, metrics["relative_error"], color="#d95f02", lw=2)
    axes[0, 1].axhline(0.5, color="0.45", ls="--", lw=1)
    axes[0, 1].set_title("Relative Fourier error per shell")
    axes[0, 1].set_ylabel("||F(mask*(est-GT))||^2 / ||F(mask*GT)||^2")
    axes[0, 1].set_ylim(0, 1.5)

    axes[1, 0].semilogy(freq, metrics["relative_error"], color="#d95f02", lw=2)
    axes[1, 0].axhline(0.5, color="0.45", ls="--", lw=1)
    axes[1, 0].set_title("Relative Fourier error, log scale")
    axes[1, 0].set_ylabel("relative error")

    axes[1, 1].plot(freq, metrics["cumulative_relative_error"], color="#1b9e77", lw=2)
    axes[1, 1].axhline(0.5, color="0.45", ls="--", lw=1)
    axes[1, 1].set_title("Cumulative relative error through shell")
    axes[1, 1].set_ylabel("cumulative relative error")
    axes[1, 1].set_ylim(0, 1.5)

    for ax in axes.flat:
        ax.set_xlim(0.0, 0.4)
        ax.set_xlabel("spatial frequency (1/A)")
        ax.grid(True, alpha=0.25)

    fig.savefig(out_dir / "shell_metrics.png", dpi=180)
    fig.savefig(out_dir / "shell_metrics.pdf")
    plt.close(fig)

    print(out_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
