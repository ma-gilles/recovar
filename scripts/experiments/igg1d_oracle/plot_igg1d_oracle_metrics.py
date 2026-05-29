#!/usr/bin/env python3
"""Plot IgG-1D oracle compute_state FSC/error curves against GT controls."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu


VOXEL_SIZE_A = 1.25
TARGET_STATE = 25
FREQ_MAX = 0.40


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
    target_ft = dft3(target * mask).astype(np.complex64)
    target_power = np.bincount(
        labels.ravel(),
        weights=np.abs(target_ft).ravel() ** 2,
        minlength=n_shells,
    ).astype(np.float64)
    freq = np.arange(n_shells, dtype=np.float64) / (target.shape[0] * VOXEL_SIZE_A)
    return labels, n_shells, target_ft, target_power, freq


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
    diff_power = np.bincount(
        flat,
        weights=np.abs(volume_ft.ravel() - target_ft.ravel()) ** 2,
        minlength=n_shells,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(volume_power * target_power)
        relerr = diff_power / np.maximum(target_power, 1e-30)
    fsc[~np.isfinite(fsc)] = np.nan
    relerr[~np.isfinite(relerr)] = np.nan
    if fsc.size > 1 and np.isfinite(fsc[1]):
        fsc[0] = fsc[1]
    return fsc.astype(np.float32), relerr.astype(np.float32)


def fsc05_resolution(freq: np.ndarray, fsc: np.ndarray) -> float:
    valid = np.isfinite(fsc) & (freq > 0)
    if not np.any(valid):
        return float("nan")
    x = freq[valid]
    y = fsc[valid]
    below = np.flatnonzero(y < 0.5)
    if below.size == 0:
        return float(1.0 / x[-1])
    idx = int(below[0])
    if idx == 0:
        return float(1.0 / x[0])
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    crossing = x1 if y0 == y1 else x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return float(1.0 / crossing)


def relerr_resolution(freq: np.ndarray, relerr: np.ndarray, threshold: float = 0.1) -> float:
    valid = np.isfinite(relerr) & (freq > 0)
    if not np.any(valid):
        return float("nan")
    x = freq[valid]
    y = relerr[valid]
    above = np.flatnonzero(y > threshold)
    if above.size == 0:
        return float(1.0 / x[-1])
    return float(1.0 / x[int(above[0])])


def state_dir(compute_root: Path) -> Path:
    diagnostics = compute_root / "diagnostics" / "state000"
    return diagnostics if diagnostics.exists() else compute_root


def candidate_grid(compute_root: Path) -> np.ndarray:
    params_path = state_dir(compute_root) / "params.pkl"
    with params_path.open("rb") as handle:
        params = pickle.load(handle)
    key = "lambda_grid" if "lambda_grid" in params else "heterogeneity_bins"
    return np.asarray(params[key], dtype=np.float64)


def candidate_half_paths(compute_root: Path) -> list[tuple[Path, Path]]:
    base = state_dir(compute_root)
    half1 = sorted((base / "estimates_half1_unfil").glob("*.mrc"))
    half2 = sorted((base / "estimates_half2_unfil").glob("*.mrc"))
    if not half1 or len(half1) != len(half2):
        raise FileNotFoundError(f"Missing unfiltered candidate half-maps in {base}")
    return list(zip(half1, half2))


def candidate_metrics(
    compute_root: Path,
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    target_ft: np.ndarray,
    target_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = candidate_grid(compute_root)
    paths = candidate_half_paths(compute_root)
    if grid.size != len(paths):
        raise ValueError(f"candidate grid/path mismatch: {grid.size} vs {len(paths)}")
    fscs = []
    errors = []
    for idx, (half1, half2) in enumerate(paths):
        volume = 0.5 * (load_mrc(half1) + load_mrc(half2))
        fsc, relerr = masked_metrics(volume, mask, labels, n_shells, target_ft, target_power)
        fscs.append(fsc)
        errors.append(relerr)
        if idx % 10 == 9 or idx == len(paths) - 1:
            print(f"candidate metrics {idx + 1}/{len(paths)}", flush=True)
    return grid, np.stack(fscs), np.stack(errors)


def gt_volume_paths(run: Path) -> list[Path]:
    paths = sorted((run / "04_ground_truth").glob("gt_vol[0-9][0-9][0-9][0-9].mrc"))
    if not paths:
        raise FileNotFoundError(run / "04_ground_truth")
    return paths


def state_weighted_gt(run: Path, weights: np.ndarray) -> np.ndarray:
    paths = gt_volume_paths(run)
    if len(paths) != weights.size:
        raise ValueError(f"GT state count mismatch: {len(paths)} files vs {weights.size} weights")
    total = np.zeros_like(load_mrc(paths[0]), dtype=np.float64)
    for state, weight in enumerate(weights):
        if weight:
            total += float(weight) * load_mrc(paths[state])
    return total.astype(np.float32)


def distribution_mean_gt(run: Path) -> np.ndarray:
    paths = gt_volume_paths(run)
    total = np.zeros_like(load_mrc(paths[0]), dtype=np.float64)
    for path in paths:
        total += load_mrc(path)
    return (total / len(paths)).astype(np.float32)


def distance_nearest_gt(run: Path, compute_root: Path, n_nearest: int) -> tuple[np.ndarray, dict[str, object]]:
    distances_path = state_dir(compute_root) / "heterogeneity_distances.txt"
    distances = np.asarray(np.loadtxt(distances_path), dtype=np.float64).reshape(-1)
    states = np.asarray(np.load(run / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    if distances.size != states.size:
        raise ValueError(f"distance/state mismatch: {distances.size} vs {states.size}")
    n = min(n_nearest, distances.size)
    nearest = np.argpartition(distances, n - 1)[:n] if n < distances.size else np.arange(distances.size)
    n_states = len(gt_volume_paths(run))
    counts = np.bincount(states[nearest], minlength=n_states).astype(np.float64)
    weights = counts / counts.sum()
    top_states = [
        f"{int(state)}:{int(counts[state])}"
        for state in np.argsort(counts)[::-1][:12]
        if counts[state] > 0
    ]
    info = {
        "n_nearest": int(n),
        "nearest_radius": float(np.max(distances[nearest])),
        "target_state_fraction": float(weights[TARGET_STATE]),
        "top_states": " ".join(top_states),
        "distances_path": str(distances_path),
    }
    return state_weighted_gt(run, weights), info


def plot_curves(out_dir: Path, freq: np.ndarray, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    colors = {
        "final compute_state": "black",
        "best candidate by FSC0.5": "#1f77b4",
        "best candidate by error<0.1": "#2ca02c",
        "distance-nearest1000 GT mix": "#d62728",
        "uniform GT mean": "0.35",
    }
    styles = {
        "final compute_state": "-",
        "best candidate by FSC0.5": "--",
        "best candidate by error<0.1": "-.",
        "distance-nearest1000 GT mix": ":",
        "uniform GT mean": (0, (5, 2)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.5), constrained_layout=True)
    fig.suptitle("IgG-1D | 1M images | noise=30 | B80 rendered vols | target state 25")
    for name, (fsc, relerr) in curves.items():
        axes[0].plot(freq, fsc, color=colors[name], ls=styles[name], lw=2.1, label=name)
        axes[1].semilogy(freq, np.maximum(relerr, 1e-30), color=colors[name], ls=styles[name], lw=2.1, label=name)
    axes[0].axhline(0.5, color="0.45", ls=":", lw=1.0)
    axes[1].axhline(0.1, color="0.45", ls=":", lw=1.0)
    axes[0].set_title("FSC vs GT state 25")
    axes[1].set_title("Relative Fourier error vs GT state 25")
    axes[0].set_ylabel("masked FSC")
    axes[1].set_ylabel("masked relative error")
    axes[0].set_ylim(-0.08, 1.03)
    axes[1].set_ylim(1e-3, 1e3)
    for ax in axes:
        ax.set_xlim(0.0, FREQ_MAX)
        ax.set_xlabel("spatial frequency (1/A)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    png = out_dir / "igg1d_oracle_compute_state_vs_gt_weighted_fsc_error.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "igg1d_oracle_compute_state_vs_gt_weighted_fsc_error.pdf")
    plt.close(fig)
    return png


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compute-dir", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--nearest-count", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run_dir
    compute_root = args.compute_dir or (run / "07_compute_state")
    mask_path = args.mask or (run / "05_masks" / "focus_mask_moving.mrc")
    out_dir = args.out_dir or (run / "08_oracle_gt_weighted_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = np.clip(load_mrc(mask_path), 0.0, 1.0)
    target = load_mrc(run / "04_ground_truth" / f"gt_vol{TARGET_STATE:04d}.mrc")
    labels, n_shells, target_ft, target_power, freq = metric_context(target, mask)

    grid, candidate_fsc, candidate_err = candidate_metrics(compute_root, mask, labels, n_shells, target_ft, target_power)
    fsc_res = np.asarray([fsc05_resolution(freq, curve) for curve in candidate_fsc])
    err_res = np.asarray([relerr_resolution(freq, curve, 0.1) for curve in candidate_err])
    best_fsc_idx = int(np.nanargmin(fsc_res))
    best_err_idx = int(np.nanargmin(err_res))

    final_fsc, final_err = masked_metrics(
        load_mrc(compute_root / "state000_unfil.mrc"), mask, labels, n_shells, target_ft, target_power
    )
    nearest_vol, nearest_info = distance_nearest_gt(run, compute_root, args.nearest_count)
    nearest_fsc, nearest_err = masked_metrics(nearest_vol, mask, labels, n_shells, target_ft, target_power)
    mean_fsc, mean_err = masked_metrics(distribution_mean_gt(run), mask, labels, n_shells, target_ft, target_power)

    curves = {
        "final compute_state": (final_fsc, final_err),
        "best candidate by FSC0.5": (candidate_fsc[best_fsc_idx], candidate_err[best_fsc_idx]),
        "best candidate by error<0.1": (candidate_fsc[best_err_idx], candidate_err[best_err_idx]),
        "distance-nearest1000 GT mix": (nearest_fsc, nearest_err),
        "uniform GT mean": (mean_fsc, mean_err),
    }
    plot_path = plot_curves(out_dir, freq, curves)

    rows = [
        {
            "curve": name,
            "fsc05_resolution_A": fsc05_resolution(freq, fsc),
            "relerr10_resolution_A": relerr_resolution(freq, relerr, 0.1),
        }
        for name, (fsc, relerr) in curves.items()
    ]
    rows.append(
        {
            "curve": "metadata",
            "fsc05_resolution_A": float("nan"),
            "relerr10_resolution_A": float("nan"),
            "candidate_grid": " ".join(f"{x:.8g}" for x in grid),
            "best_fsc_candidate_1based": best_fsc_idx + 1,
            "best_fsc_candidate_parameter": float(grid[best_fsc_idx]),
            "best_error_candidate_1based": best_err_idx + 1,
            "best_error_candidate_parameter": float(grid[best_err_idx]),
            "nearest_top_states": nearest_info["top_states"],
            "nearest_target_state_fraction": nearest_info["target_state_fraction"],
            "nearest_radius": nearest_info["nearest_radius"],
            "mask": str(mask_path),
            "compute_state_dir": str(compute_root),
            "distances_path": nearest_info["distances_path"],
        }
    )
    all_keys = sorted({key for row in rows for key in row})
    with (out_dir / "igg1d_oracle_compute_state_vs_gt_weighted_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        out_dir / "igg1d_oracle_compute_state_vs_gt_weighted_curves.npz",
        frequency_1_per_A=freq,
        final_fsc=final_fsc,
        final_error=final_err,
        best_fsc_candidate_fsc=candidate_fsc[best_fsc_idx],
        best_fsc_candidate_error=candidate_err[best_fsc_idx],
        best_error_candidate_fsc=candidate_fsc[best_err_idx],
        best_error_candidate_error=candidate_err[best_err_idx],
        nearest1000_gt_fsc=nearest_fsc,
        nearest1000_gt_error=nearest_err,
        uniform_gt_mean_fsc=mean_fsc,
        uniform_gt_mean_error=mean_err,
    )
    print(f"PLOT {plot_path}")
    print(f"SUMMARY {out_dir / 'igg1d_oracle_compute_state_vs_gt_weighted_summary.csv'}")


if __name__ == "__main__":
    main()
