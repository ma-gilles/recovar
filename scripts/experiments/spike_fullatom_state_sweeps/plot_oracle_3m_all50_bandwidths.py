#!/usr/bin/env python3
"""Plot all 50 source-oracle bandwidth candidates for the 3M noise=30 run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plot_noise1_sweep_with_gt_nearest import (
    DEFAULT_MASK,
    FREQ_MAX,
    TARGET_STATE,
    candidate_metrics,
    fsc05_resolution,
    load_mrc,
    masked_metrics,
    metric_context,
    relerr_resolution,
    state_dir,
    state_weighted_gt_mrc,
)
from gt_embedding_controls import state_weights_from_compute_state_nearest


DEFAULT_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518/"
    "n03000000/runs/n03000000_seed0000"
)
DEFAULT_COMPUTE = "07_compute_state_oracle_regfix_zdim4_reg_lazy"
DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "oracle_debug_embedding_variants_repro_20260528"
)


def nearest_gt_by_recomputed_distance(
    run: Path,
    compute_root: Path,
    n_nearest: int,
) -> tuple[np.ndarray, dict[str, object]]:
    weights, nearest, distances, info_in = state_weights_from_compute_state_nearest(run, compute_root, n_nearest)
    states = np.asarray(np.load(run / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    counts = np.bincount(states[nearest], minlength=100).astype(np.float64)
    top_states = " ".join(
        f"{int(state)}:{int(counts[state])}"
        for state in np.argsort(counts)[::-1][:12]
        if counts[state] > 0
    )
    info = {
        "n_nearest": int(nearest.size),
        "state50_fraction": float(weights[TARGET_STATE]),
        "top_states": top_states,
        "distance_min": float(np.min(distances[nearest])),
        "distance_max": float(np.max(distances[nearest])),
        "distance_source": "recomputed from compute_state command/pipeline/target",
        "pipeline": str(info_in.pipeline),
        "target_point": str(info_in.target_path),
        "embedding_option": info_in.embedding_option,
        "coords_entry": info_in.coords_entry,
    }
    return state_weighted_gt_mrc(run, weights), info


def write_curves(path: Path, freq: np.ndarray, rows: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shell", "frequency_1_per_A", *rows])
        for idx, value in enumerate(freq):
            writer.writerow([idx, float(value), *[float(curve[idx]) for curve in rows.values()]])


def plot_all50(
    out_dir: Path,
    grid: np.ndarray,
    freq: np.ndarray,
    candidate_fsc: np.ndarray,
    candidate_err: np.ndarray,
    final_fsc: np.ndarray,
    final_err: np.ndarray,
    gt_fsc: np.ndarray,
    gt_err: np.ndarray,
    nearest_info: dict[str, object],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8), constrained_layout=True)
    fig.suptitle(
        "Source-oracle zdim4/reg | 3M noise=30 | all 50 unfiltered bandwidth candidates",
        fontweight="bold",
    )
    positive_grid = grid[grid > 0]
    if positive_grid.size == grid.size and np.nanmax(grid) / np.nanmin(positive_grid) > 10:
        norm: mcolors.Normalize = mcolors.LogNorm(vmin=float(np.nanmin(positive_grid)), vmax=float(np.nanmax(grid)))
    else:
        norm = mcolors.Normalize(vmin=float(np.nanmin(grid)), vmax=float(np.nanmax(grid)))
    cmap = plt.get_cmap("viridis")

    for idx, value in enumerate(grid):
        color = cmap(norm(float(value)))
        axes[0].plot(freq, candidate_fsc[idx], color=color, lw=1.0, alpha=0.65)
        axes[1].semilogy(freq, np.maximum(candidate_err[idx], 1e-30), color=color, lw=1.0, alpha=0.65)

    axes[0].plot(freq, final_fsc, color="black", lw=2.7, label="compute_state output")
    axes[1].semilogy(freq, np.maximum(final_err, 1e-30), color="black", lw=2.7, label="compute_state output")
    axes[0].plot(freq, gt_fsc, color="#005a32", ls="-.", lw=2.4, label="GT nearest-100 by saved distance")
    axes[1].semilogy(freq, np.maximum(gt_err, 1e-30), color="#005a32", ls="-.", lw=2.4, label="GT nearest-100 by saved distance")

    axes[0].axhline(0.5, color="0.45", ls=":", lw=1.0)
    axes[1].axhline(0.1, color="0.45", ls=":", lw=1.0)
    axes[0].set_title("FSC vs GT state 50")
    axes[1].set_title("Relative Fourier error vs GT state 50")
    axes[0].set_ylabel("masked FSC")
    axes[1].set_ylabel("masked relative error")
    axes[0].set_ylim(-0.08, 1.03)
    axes[1].set_ylim(1e-3, 1e3)
    for ax in axes:
        ax.set_xlim(0.0, FREQ_MAX)
        ax.set_xlabel("spatial frequency (1/A)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8.0, loc="lower left")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.012, fraction=0.030)
    cbar.set_label("candidate bandwidth h")

    fig.text(
        0.50,
        0.005,
        f"GT nearest-100: state50={nearest_info['state50_fraction']:.3f}, top states {nearest_info['top_states']}",
        ha="center",
        fontsize=8.5,
    )
    png = out_dir / "source_oracle_3m_all50_bandwidths_with_gt_nearest100.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "source_oracle_3m_all50_bandwidths_with_gt_nearest100.pdf")
    plt.close(fig)
    return png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--compute-relpath", default=DEFAULT_COMPUTE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--nearest-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    compute_root = args.run_dir / args.compute_relpath
    target = load_mrc(args.run_dir / "04_ground_truth" / f"gt_vol{TARGET_STATE:04d}.mrc")
    mask = np.clip(load_mrc(args.mask), 0.0, 1.0)
    labels, n_shells, target_ft, target_power, freq = metric_context(target, mask)

    grid, candidate_fsc, candidate_err = candidate_metrics(
        compute_root,
        mask,
        labels,
        n_shells,
        target_ft,
        target_power,
        args.batch_size,
    )
    final_fsc, final_err = masked_metrics(
        load_mrc(compute_root / "state000_unfil.mrc"),
        mask,
        labels,
        n_shells,
        target_ft,
        target_power,
    )
    nearest_vol, nearest_info = nearest_gt_by_recomputed_distance(args.run_dir, compute_root, args.nearest_count)
    gt_fsc, gt_err = masked_metrics(nearest_vol, mask, labels, n_shells, target_ft, target_power)

    candidate_fsc_res = np.asarray([fsc05_resolution(freq, curve) for curve in candidate_fsc], dtype=np.float64)
    best_fsc_idx = int(np.nanargmin(candidate_fsc_res))
    summary_path = args.out_dir / "source_oracle_3m_all50_bandwidths_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["compute_root", str(compute_root)])
        writer.writerow(["mask", str(args.mask)])
        writer.writerow(["best_candidate_index_0based", best_fsc_idx])
        writer.writerow(["best_candidate_bandwidth_h", float(grid[best_fsc_idx])])
        writer.writerow(["best_candidate_fsc05_resolution_A", float(candidate_fsc_res[best_fsc_idx])])
        writer.writerow(["compute_state_output_fsc05_resolution_A", fsc05_resolution(freq, final_fsc)])
        writer.writerow(["gt_nearest100_fsc05_resolution_A", fsc05_resolution(freq, gt_fsc)])
        writer.writerow(["gt_nearest100_relerr10_resolution_A", relerr_resolution(freq, gt_err, 0.1)])
        for key, value in nearest_info.items():
            writer.writerow([f"gt_nearest100_{key}", value])

    curve_rows = {f"candidate_{idx:02d}_h_{grid[idx]:.9g}_fsc": candidate_fsc[idx] for idx in range(candidate_fsc.shape[0])}
    curve_rows["compute_state_output_fsc"] = final_fsc
    curve_rows["gt_nearest100_fsc"] = gt_fsc
    write_curves(args.out_dir / "source_oracle_3m_all50_bandwidths_fsc_curves.csv", freq, curve_rows)

    err_rows = {f"candidate_{idx:02d}_h_{grid[idx]:.9g}_relative_error": candidate_err[idx] for idx in range(candidate_err.shape[0])}
    err_rows["compute_state_output_relative_error"] = final_err
    err_rows["gt_nearest100_relative_error"] = gt_err
    write_curves(args.out_dir / "source_oracle_3m_all50_bandwidths_relative_error_curves.csv", freq, err_rows)

    png = plot_all50(
        args.out_dir,
        grid,
        freq,
        candidate_fsc,
        candidate_err,
        final_fsc,
        final_err,
        gt_fsc,
        gt_err,
        nearest_info,
    )
    print(f"PLOT {png}")
    print(f"SUMMARY {summary_path}")


if __name__ == "__main__":
    main()
