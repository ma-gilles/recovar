#!/usr/bin/env python3
"""Mean-subtracted FSC and sampled local-resolution metrics for spike sweeps.

For each dataset-size run, this compares

    estimate - mean(GT states)

against

    target GT state - mean(GT states)

inside a focus mask. The FSC threshold is fixed at 0.5 because this is an
estimate-vs-GT comparison, not a half-map FSC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu


def _n_images_from_run(run_dir: Path) -> int:
    match = re.search(r"n(\d{8})_seed", run_dir.name)
    if match:
        return int(match.group(1))
    match = re.search(r"n(\d{8})", str(run_dir))
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse image count from {run_dir}")


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


def _masked_fsc(volume: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    labels, n_shells = _shell_labels(volume.shape)
    volume_ft = _dft3(volume * mask)
    target_ft = _dft3(target * mask)
    flat = labels.ravel()
    cross = np.bincount(
        flat,
        weights=np.real(np.conj(volume_ft).ravel() * target_ft.ravel()),
        minlength=n_shells,
    )
    vol_power = np.bincount(flat, weights=np.abs(volume_ft).ravel() ** 2, minlength=n_shells)
    target_power = np.bincount(flat, weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(vol_power * target_power)
    fsc[~np.isfinite(fsc)] = 0.0
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return fsc.astype(np.float32)


def _load_gt_mean(gt_dir: Path) -> np.ndarray:
    paths = sorted(gt_dir.glob("gt_vol*.mrc"))
    if not paths:
        raise FileNotFoundError(f"No gt_vol*.mrc files under {gt_dir}")
    acc = None
    for path in paths:
        vol = np.asarray(utils.load_mrc(path), dtype=np.float64)
        if acc is None:
            acc = np.zeros_like(vol, dtype=np.float64)
        acc += vol
    assert acc is not None
    return (acc / len(paths)).astype(np.float32)


def _sample_locres_points(
    mask: np.ndarray,
    voxel_size: float,
    sampling_A: float,
    radius_A: float,
    max_points: int,
) -> np.ndarray:
    radius = max(3, int(round(radius_A / voxel_size)))
    step = max(1, int(round(sampling_A / voxel_size)))
    active = np.argwhere(mask > 0.5)
    if active.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    lo = np.maximum(active.min(axis=0), radius)
    hi = np.minimum(active.max(axis=0), np.asarray(mask.shape) - radius - 1)
    grids = [np.arange(lo[axis], hi[axis] + 1, step, dtype=np.int32) for axis in range(3)]
    centers = np.asarray(np.meshgrid(*grids, indexing="ij")).reshape(3, -1).T
    centers = np.asarray([c for c in centers if mask[tuple(c)] > 0.5], dtype=np.int32)
    if centers.shape[0] > max_points:
        rng = np.random.default_rng(0)
        centers = centers[rng.choice(centers.shape[0], size=max_points, replace=False)]
    return centers


def _sampled_locres_summary(
    volume: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    voxel_size: float,
    centers: np.ndarray,
    radius_A: float,
) -> tuple[float, float, int]:
    if centers.size == 0:
        return math.nan, math.nan, 0
    radius = max(3, int(round(radius_A / voxel_size)))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    sphere = ((xx * xx + yy * yy + zz * zz) <= radius * radius).astype(np.float32)
    local_res = []
    for center in centers:
        slices = tuple(slice(int(c) - radius, int(c) + radius + 1) for c in center)
        local_mask = sphere * (mask[slices] > 0.5)
        if float(local_mask.sum()) < 32:
            continue
        fsc = _masked_fsc(volume[slices], target[slices], local_mask)
        freq = np.arange(fsc.size, dtype=np.float64) / ((2 * radius + 1) * voxel_size)
        res = _fsc_resolution(freq, fsc, threshold=0.5)
        if np.isfinite(res):
            local_res.append(res)
    values = np.asarray(local_res, dtype=np.float32)
    if values.size == 0:
        return math.nan, math.nan, 0
    return float(np.median(values)), float(np.percentile(values, 90.0)), int(values.size)


def _plot_curves(rows: list[dict], curves: dict[int, np.ndarray], freq: np.ndarray, out_dir: Path) -> None:
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(rows)))
    fig, ax = plt.subplots(figsize=(8.3, 5.2), constrained_layout=True)
    for color, row in zip(colors, rows):
        n_images = int(row["n_images"])
        ax.plot(freq, curves[n_images], color=color, lw=2.0, label=f"{n_images:,}: {row['fsc05_resolution_A']:.2f} A")
    ax.axhline(0.5, color="0.35", ls="--", lw=1.0)
    ax.set_xlim(0.0, 0.4)
    ax.set_ylim(-0.08, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("mean-subtracted masked FSC vs GT")
    ax.set_title("Mean-subtracted state estimate vs GT residual | FSC 0.5")
    ax.grid(True, alpha=0.25)
    ax.legend(title="n images: FSC0.5", fontsize=8, title_fontsize=8, loc="lower left")
    fig.savefig(out_dir / "mean_subtracted_fsc_curves_fsc05.png", dpi=180)
    fig.savefig(out_dir / "mean_subtracted_fsc_curves_fsc05.pdf")
    plt.close(fig)

    n_images = np.asarray([row["n_images"] for row in rows], dtype=np.float64)
    fsc_res = np.asarray([row["fsc05_resolution_A"] for row in rows], dtype=np.float64)
    loc_med = np.asarray([row["locres_median_A"] for row in rows], dtype=np.float64)
    loc_p90 = np.asarray([row["locres_p90_A"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.semilogx(n_images, fsc_res, marker="o", lw=2.0, label="global FSC0.5")
    ax.semilogx(n_images, loc_med, marker="s", lw=2.0, label="local median FSC0.5")
    ax.semilogx(n_images, loc_p90, marker="^", lw=2.0, label="local p90 FSC0.5")
    ax.invert_yaxis()
    ax.set_xlabel("number of images")
    ax.set_ylabel("resolution (A), lower is better")
    ax.set_title("Mean-subtracted focus metrics vs image count")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "mean_subtracted_resolution_vs_n_fsc05.png", dpi=180)
    fig.savefig(out_dir / "mean_subtracted_resolution_vs_n_fsc05.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset-size sweep root containing n*/runs/n*_seed0000")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--target-state", type=int, default=50)
    parser.add_argument("--estimate-relpath", default="07_compute_state/state000_unfil.mrc")
    parser.add_argument("--mask-relpath", default="05_masks/focus_mask_moving.mrc")
    parser.add_argument("--voxel-size", type=float, default=1.25)
    parser.add_argument("--locres-sampling-A", type=float, default=24.0)
    parser.add_argument("--locres-radius-A", type=float, default=24.0)
    parser.add_argument("--locres-max-points", type=int, default=400)
    args = parser.parse_args()

    run_dirs = sorted(args.root.glob("n*/runs/n*_seed0000"), key=_n_images_from_run)
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs under {args.root}")
    out_dir = args.out_dir or (args.root / "mean_subtracted_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = run_dirs[0] / "04_ground_truth"
    gt_mean = _load_gt_mean(gt_dir)
    target = np.asarray(utils.load_mrc(gt_dir / f"gt_vol{args.target_state:04d}.mrc"), dtype=np.float32)
    target_residual = target - gt_mean
    utils.write_mrc(str(out_dir / "gt_mean_all_states.mrc"), gt_mean, voxel_size=args.voxel_size)
    utils.write_mrc(
        str(out_dir / f"gt_vol{args.target_state:04d}_minus_gt_mean_all_states.mrc"),
        target_residual.astype(np.float32),
        voxel_size=args.voxel_size,
    )

    mask = np.clip(np.asarray(utils.load_mrc(run_dirs[0] / args.mask_relpath), dtype=np.float32), 0.0, 1.0)
    centers = _sample_locres_points(mask, args.voxel_size, args.locres_sampling_A, args.locres_radius_A, args.locres_max_points)
    np.save(out_dir / "locres_sample_centers_xyz.npy", centers)

    rows = []
    curves: dict[int, np.ndarray] = {}
    for run_dir in run_dirs:
        n_images = _n_images_from_run(run_dir)
        estimate_path = run_dir / args.estimate_relpath
        estimate = np.asarray(utils.load_mrc(estimate_path), dtype=np.float32)
        estimate_residual = estimate - gt_mean
        fsc = _masked_fsc(estimate_residual, target_residual, mask)
        freq = np.arange(fsc.size, dtype=np.float64) / (estimate.shape[0] * args.voxel_size)
        loc_med, loc_p90, loc_points = _sampled_locres_summary(
            estimate_residual,
            target_residual,
            mask,
            args.voxel_size,
            centers,
            args.locres_radius_A,
        )
        row = {
            "n_images": n_images,
            "fsc05_resolution_A": _fsc_resolution(freq, fsc, threshold=0.5),
            "locres_median_A": loc_med,
            "locres_p90_A": loc_p90,
            "locres_points": loc_points,
            "estimate": str(estimate_path),
            "target_gt": str(gt_dir / f"gt_vol{args.target_state:04d}.mrc"),
            "gt_mean": str(out_dir / "gt_mean_all_states.mrc"),
            "mask": str(run_dirs[0] / args.mask_relpath),
        }
        rows.append(row)
        curves[n_images] = fsc

    rows.sort(key=lambda row: int(row["n_images"]))
    with (out_dir / "mean_subtracted_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out_dir / "mean_subtracted_fsc_curves.npz",
        frequency_1_per_A=freq,
        **{f"n{n_images:08d}": fsc for n_images, fsc in curves.items()},
    )
    (out_dir / "mean_subtracted_metrics.json").write_text(
        json.dumps(
            {
                "root": str(args.root),
                "target_state": args.target_state,
                "threshold": 0.5,
                "definition": "compare (estimate - mean(GT states)) to (target GT - mean(GT states))",
                "locres_sampling_A": args.locres_sampling_A,
                "locres_radius_A": args.locres_radius_A,
                "locres_max_points": args.locres_max_points,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _plot_curves(rows, curves, freq, out_dir)

    print(out_dir)
    for row in rows:
        print(
            f"{row['n_images']:>8,d} "
            f"FSC0.5={row['fsc05_resolution_A']:.3f} A "
            f"locres_med={row['locres_median_A']:.3f} A "
            f"locres_p90={row['locres_p90_A']:.3f} A "
            f"points={row['locres_points']}"
        )


if __name__ == "__main__":
    main()
