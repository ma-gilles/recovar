#!/usr/bin/env python3
"""Plot noise30/B80 RECOVAR curves under the broad state-50 mask.

For each state and image count, this plots:

- the unfiltered compute_state estimate vs the matching GT volume
- a GT-volume mixture built from the 1000 particles nearest to the compute_state
  target in the same latent-distance metric used by that compute_state run

The GT mixture is recomputed from the current run metadata and current
compute_state command. No cached GT embedding or cached nearest set is read.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils

SCRIPT_DIR = Path(__file__).resolve().parent
STATE_SWEEP_DIR = SCRIPT_DIR.parent / "spike_fullatom_state_sweeps"
sys.path.insert(0, str(STATE_SWEEP_DIR))
from gt_embedding_controls import (  # noqa: E402
    state_weights_from_compute_state_nearest,
    state_weights_from_nearest_distances,
)

from score_recovar_state_sweep_fsc_error import (  # noqa: E402
    curves,
    parse_image_counts,
    relerr_resolution,
    resolution_at,
    run_dir,
    write_curve_csv,
)


DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise30_b80_20260528/recovar_broad_mask_nearest1000"
)
MASK_PATH = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
STATE_ROOTS = {
    0: Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0000_reuse_20260519"),
    25: Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0025_reuse_20260519"),
    50: Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518"),
}
IMAGE_COUNTS = "10000,30000,100000,300000,1000000,3000000"
VOXEL_SIZE = 1.25
N_NEAREST = 1000


def load_mrc(path: Path) -> np.ndarray:
    return np.asarray(utils.load_mrc(path), dtype=np.float32)


def load_state_weighted_gt(run: Path, weights: np.ndarray) -> np.ndarray:
    total: np.ndarray | None = None
    for state, weight in enumerate(weights):
        if weight == 0:
            continue
        vol = load_mrc(run / "04_ground_truth" / f"gt_vol{state:04d}.mrc")
        if total is None:
            total = np.zeros_like(vol, dtype=np.float64)
        total += float(weight) * vol
    if total is None:
        raise RuntimeError("All GT-mixture weights were zero")
    return total.astype(np.float32)


def nearest_weights(run: Path, compute_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        weights, nearest, distances, info = state_weights_from_compute_state_nearest(run, compute_root, N_NEAREST)
        return weights, nearest, distances, str(info.target_path)
    except FileNotFoundError as exc:
        command_path = compute_root / "command.txt"
        if str(command_path) not in str(exc):
            raise
        distance_path = compute_root / "diagnostics" / "state000" / "heterogeneity_distances.txt"
        distances = np.loadtxt(distance_path, dtype=np.float64)
        weights, nearest = state_weights_from_nearest_distances(run, distances, N_NEAREST)
        return weights, nearest, distances, f"fallback saved distances: {distance_path}"


def plot_state_curves(
    out_dir: Path,
    state: int,
    freq: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    colors = plt.cm.viridis(np.linspace(0.10, 0.92, len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), constrained_layout=True)
    for color, row in zip(colors, rows):
        n_images = int(row["n_images"])
        label = f"{n_images:,}"
        axes[0].plot(freq, row["estimate_fsc"], color=color, lw=2.2, label=f"{label} estimate")
        axes[0].plot(freq, row["nearest_fsc"], color=color, lw=2.0, ls=":", label=f"{label} nearest-1k GT mix")
        axes[1].semilogy(freq, np.maximum(row["estimate_err"], 1e-30), color=color, lw=2.2, label=f"{label} estimate")
        axes[1].semilogy(freq, np.maximum(row["nearest_err"], 1e-30), color=color, lw=2.0, ls=":", label=f"{label} nearest-1k GT mix")

    axes[0].axhline(0.5, color="0.35", ls="--", lw=1)
    axes[1].axhline(0.1, color="0.35", ls="--", lw=1)
    axes[0].set_ylabel("broad-mask FSC vs GT")
    axes[1].set_ylabel("broad-mask relative Fourier error")
    for ax in axes:
        ax.set_xlim(0, 0.4)
        ax.set_xlabel("spatial frequency (1/A)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7, ncols=2)
    fig.suptitle(f"RECOVAR noise30/B80 | state {state} | broad mask | nearest-1000 GT-mixture control", fontweight="bold")
    fig.savefig(out_dir / f"state{state:04d}_broadmask_estimate_vs_nearest1000_fsc_error.png", dpi=190)
    fig.savefig(out_dir / f"state{state:04d}_broadmask_estimate_vs_nearest1000_fsc_error.pdf")
    plt.close(fig)


def plot_resolution_summary(out_dir: Path, summary: list[dict[str, object]]) -> None:
    colors = {0: "#1f77b4", 25: "#ff7f0e", 50: "#2ca02c"}
    markers = {0: "o", 25: "s", 50: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.1), constrained_layout=True)
    for state in (0, 25, 50):
        state_rows = [row for row in summary if int(row["state"]) == state]
        state_rows.sort(key=lambda row: int(row["n_images"]))
        xs = np.asarray([row["n_images"] for row in state_rows], dtype=np.float64)
        axes[0].semilogx(
            xs,
            [row["estimate_fsc05_resolution_A"] for row in state_rows],
            color=colors[state],
            marker=markers[state],
            lw=2.3,
            label=f"state {state} estimate",
        )
        axes[0].semilogx(
            xs,
            [row["nearest1000_fsc05_resolution_A"] for row in state_rows],
            color=colors[state],
            marker=markers[state],
            lw=2.0,
            ls=":",
            label=f"state {state} nearest-1k GT mix",
        )
        axes[1].semilogx(
            xs,
            [row["estimate_relerr10_resolution_A"] for row in state_rows],
            color=colors[state],
            marker=markers[state],
            lw=2.3,
            label=f"state {state} estimate",
        )
        axes[1].semilogx(
            xs,
            [row["nearest1000_relerr10_resolution_A"] for row in state_rows],
            color=colors[state],
            marker=markers[state],
            lw=2.0,
            ls=":",
            label=f"state {state} nearest-1k GT mix",
        )
    axes[0].set_ylabel("FSC=0.5 resolution (A), lower is better")
    axes[1].set_ylabel("relative-error<10% resolution (A), lower is better")
    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlabel("number of images")
        ax.grid(alpha=0.28, which="both")
        ax.legend(fontsize=8, ncols=2)
    fig.suptitle("RECOVAR noise30/B80 | broad-mask resolution summary", fontweight="bold")
    fig.savefig(out_dir / "broadmask_resolution_vs_n_with_nearest1000_gtmix.png", dpi=200)
    fig.savefig(out_dir / "broadmask_resolution_vs_n_with_nearest1000_gtmix.pdf")
    plt.close(fig)


def main() -> None:
    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    mix_dir = out_dir / "nearest1000_gt_mixtures"
    mix_dir.mkdir(exist_ok=True)

    mask = np.clip(load_mrc(MASK_PATH), 0.0, 1.0)
    image_counts = parse_image_counts(IMAGE_COUNTS)
    summary: list[dict[str, object]] = []
    all_fsc_rows: dict[str, np.ndarray] = {}
    all_err_rows: dict[str, np.ndarray] = {}
    freq: np.ndarray | None = None

    for state, root in STATE_ROOTS.items():
        state_rows: list[dict[str, object]] = []
        for n_images in image_counts:
            run = run_dir(root, n_images)
            compute_root = run / "07_compute_state"
            estimate_path = compute_root / "state000_unfil.mrc"
            gt_path = run / "04_ground_truth" / f"gt_vol{state:04d}.mrc"
            mix_path = mix_dir / f"state{state:04d}_n{n_images:08d}_nearest1000_gt_mix.mrc"
            weights, nearest, distances, target_source = nearest_weights(run, compute_root)
            if mix_path.exists():
                mixture = load_mrc(mix_path)
            else:
                mixture = load_state_weighted_gt(run, weights)
                utils.write_mrc(str(mix_path), mixture, voxel_size=VOXEL_SIZE)

            estimate = load_mrc(estimate_path)
            gt = load_mrc(gt_path)
            estimate_fsc, estimate_err = curves(estimate, gt, mask)
            nearest_fsc, nearest_err = curves(mixture, gt, mask)
            freq = np.arange(estimate_fsc.size, dtype=np.float64) / (estimate.shape[0] * VOXEL_SIZE)
            top_states = np.argsort(weights)[::-1][:8]
            row = {
                "state": state,
                "n_images": n_images,
                "estimate_fsc": estimate_fsc,
                "estimate_err": estimate_err,
                "nearest_fsc": nearest_fsc,
                "nearest_err": nearest_err,
                "estimate_fsc05_resolution_A": resolution_at(freq, estimate_fsc, 0.5),
                "nearest1000_fsc05_resolution_A": resolution_at(freq, nearest_fsc, 0.5),
                "estimate_relerr10_resolution_A": relerr_resolution(freq, estimate_err, 0.1),
                "nearest1000_relerr10_resolution_A": relerr_resolution(freq, nearest_err, 0.1),
                "nearest1000_target_state_fraction": float(weights[state]),
                "nearest1000_top_state_weights": ";".join(f"{idx}:{weights[idx]:.4f}" for idx in top_states if weights[idx] > 0),
                "nearest1000_distance_min": float(np.min(distances[nearest])),
                "nearest1000_distance_max": float(np.max(distances[nearest])),
                "estimate": str(estimate_path),
                "gt": str(gt_path),
                "nearest1000_gt_mix": str(mix_path),
                "mask": str(MASK_PATH),
                "compute_state_target": target_source,
            }
            state_rows.append(row)
            summary.append(row)
            prefix = f"state{state:04d}_n{n_images:08d}"
            all_fsc_rows[f"{prefix}_estimate"] = estimate_fsc
            all_fsc_rows[f"{prefix}_nearest1000_gt_mix"] = nearest_fsc
            all_err_rows[f"{prefix}_estimate"] = estimate_err
            all_err_rows[f"{prefix}_nearest1000_gt_mix"] = nearest_err
        assert freq is not None
        plot_state_curves(out_dir, state, freq, state_rows)

    assert freq is not None
    write_curve_csv(out_dir / "broadmask_fsc_curves_estimate_and_nearest1000.csv", freq, all_fsc_rows)
    write_curve_csv(out_dir / "broadmask_relative_error_curves_estimate_and_nearest1000.csv", freq, all_err_rows)
    fieldnames = [
        "state",
        "n_images",
        "estimate_fsc05_resolution_A",
        "nearest1000_fsc05_resolution_A",
        "estimate_relerr10_resolution_A",
        "nearest1000_relerr10_resolution_A",
        "nearest1000_target_state_fraction",
        "nearest1000_top_state_weights",
        "nearest1000_distance_min",
        "nearest1000_distance_max",
        "estimate",
        "gt",
        "nearest1000_gt_mix",
        "mask",
        "compute_state_target",
    ]
    with (out_dir / "broadmask_resolution_summary_with_nearest1000.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row[key] for key in fieldnames})
    plot_resolution_summary(out_dir, summary)
    print(out_dir)


if __name__ == "__main__":
    main()
