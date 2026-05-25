#!/usr/bin/env python3
"""FSC0.5 vs GT metrics for full-atom spike dataset-size sweeps."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu


def _n_images_from_run(run_dir: Path) -> int:
    match = re.search(r"n(\d{8})_seed", run_dir.name) or re.search(r"n(\d{8})", str(run_dir))
    if match is None:
        raise ValueError(f"Could not parse image count from {run_dir}")
    return int(match.group(1))


def _dft3(volume: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume)))


def _shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    labels = np.asarray(ftu.get_grid_of_radial_distances(shape, rounded=True), dtype=np.int32)
    n_shells = shape[0] // 2 - 1
    return np.clip(labels, 0, n_shells - 1), n_shells


def _fsc_resolution(freq: np.ndarray, fsc: np.ndarray, threshold: float = 0.5) -> float:
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


def _masked_fsc_and_relative_error(
    volume: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels, n_shells = _shell_labels(volume.shape)
    volume_ft = _dft3(volume * mask)
    target_ft = _dft3(target * mask)
    diff_ft = volume_ft - target_ft
    flat = labels.ravel()
    cross = np.bincount(
        flat,
        weights=np.real(np.conj(volume_ft).ravel() * target_ft.ravel()),
        minlength=n_shells,
    )
    vol_power = np.bincount(flat, weights=np.abs(volume_ft).ravel() ** 2, minlength=n_shells)
    target_power = np.bincount(flat, weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    diff_power = np.bincount(flat, weights=np.abs(diff_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(vol_power * target_power)
        rel_error = diff_power / target_power
    fsc[~np.isfinite(fsc)] = 0.0
    rel_error[~np.isfinite(rel_error)] = np.nan
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return fsc.astype(np.float32), rel_error.astype(np.float32)


def _plot(rows: list[dict], curves: dict[int, np.ndarray], freq: np.ndarray, label: str, out_dir: Path) -> None:
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(rows)))
    fig, ax = plt.subplots(figsize=(8.4, 5.1), constrained_layout=True)
    for color, row in zip(colors, rows):
        n_images = int(row["n_images"])
        ax.plot(freq, curves[n_images], color=color, lw=2.0, label=f"{n_images:,}: {row['fsc_05_resolution_A']:.2f} A")
    ax.axhline(0.5, color="0.35", ls="--", lw=1.0)
    ax.set_xlim(0.0, 0.4)
    ax.set_ylim(-0.08, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("masked FSC vs GT")
    ax.set_title(f"{label}: unfiltered estimate vs GT | FSC 0.5")
    ax.grid(True, alpha=0.25)
    ax.legend(title="n images: FSC0.5", fontsize=8, title_fontsize=8, loc="lower left")
    fig.savefig(out_dir / f"{label}_combined_fsc_vs_gt_fsc05.png", dpi=180)
    fig.savefig(out_dir / f"{label}_combined_fsc_vs_gt_fsc05.pdf")
    plt.close(fig)

    n_images = np.asarray([row["n_images"] for row in rows], dtype=np.float64)
    resolution = np.asarray([row["fsc_05_resolution_A"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.7, 4.6), constrained_layout=True)
    ax.semilogx(n_images, resolution, marker="o", lw=2.0)
    ax.invert_yaxis()
    ax.set_xlabel("number of images")
    ax.set_ylabel("FSC0.5 resolution (A), lower is better")
    ax.set_title(f"{label}: resolution vs image count")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(out_dir / f"{label}_fsc05_resolution_vs_n.png", dpi=180)
    fig.savefig(out_dir / f"{label}_fsc05_resolution_vs_n.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Sweep root containing n*/runs/n*_seed0000")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--target-state", type=int, default=50)
    parser.add_argument("--estimate-relpath", default="07_compute_state/state000_unfil.mrc")
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=1.25)
    args = parser.parse_args()

    run_dirs = sorted(args.root.glob("n*/runs/n*_seed0000"), key=_n_images_from_run)
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs under {args.root}")
    out_dir = args.out_dir or (args.root / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or args.root.name

    mask = np.clip(np.asarray(utils.load_mrc(args.mask), dtype=np.float32), 0.0, 1.0)
    rows = []
    curves: dict[int, np.ndarray] = {}
    freq = None
    for run_dir in run_dirs:
        n_images = _n_images_from_run(run_dir)
        estimate_path = run_dir / args.estimate_relpath
        gt_path = run_dir / "04_ground_truth" / f"gt_vol{args.target_state:04d}.mrc"
        estimate = np.asarray(utils.load_mrc(estimate_path), dtype=np.float32)
        gt = np.asarray(utils.load_mrc(gt_path), dtype=np.float32)
        fsc, rel_error = _masked_fsc_and_relative_error(estimate, gt, mask)
        freq = np.arange(fsc.size, dtype=np.float64) / (estimate.shape[0] * args.voxel_size)
        low_freq = (freq >= 0.02) & (freq <= 0.12)
        row = {
            "n_images": n_images,
            "fsc_05_resolution_A": _fsc_resolution(freq, fsc, threshold=0.5),
            "median_relative_error_low_freq": float(np.nanmedian(rel_error[low_freq])),
            "estimate_kind": "unfiltered",
            "estimate_path": str(estimate_path),
            "gt_path": str(gt_path),
            "mask_source": str(args.mask),
        }
        rows.append(row)
        curves[n_images] = fsc

    assert freq is not None
    rows.sort(key=lambda row: int(row["n_images"]))
    with (out_dir / f"{label}_fsc05_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out_dir / f"{label}_fsc_curves.npz",
        frequency_1_per_A=freq,
        **{f"n{n_images:08d}": fsc for n_images, fsc in curves.items()},
    )
    _plot(rows, curves, freq, label, out_dir)

    print(out_dir)
    for row in rows:
        print(f"{row['n_images']:>8,d} FSC0.5={row['fsc_05_resolution_A']:.3f} A")


if __name__ == "__main__":
    main()
