#!/usr/bin/env python3
"""Compare 1M/noise30 real-pipeline and source-oracle compute_state outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_noise1_sweep_with_gt_nearest import (
    DEFAULT_MASK,
    FREQ_MAX,
    TARGET_STATE,
    candidate_grid,
    candidate_metrics,
    fsc05_resolution,
    load_mrc,
    masked_metrics,
    metric_context,
    relerr_resolution,
    distribution_mean_gt_mrc,
    state_weighted_gt_mrc,
)
from gt_embedding_controls import state_weights_from_compute_state_nearest


DEFAULT_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518/"
    "n01000000/runs/n01000000_seed0000"
)
DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "noise30_1m_pipeline_vs_oracle_20260528"
)


@dataclass(frozen=True)
class Method:
    label: str
    short: str
    compute_relpath: str
    embedding_relpath: str
    target_point_name: str
    color: str


METHODS = (
    Method(
        label="real pipeline zdim4/reg",
        short="pipeline",
        compute_relpath="07_compute_state_true_recovar_h100_fullmem_movingfocus_zdim4_reg_lazy",
        embedding_relpath="06_pipeline_true_recovar_h100_fullmem_movingfocus_20260527/model/zdim_4",
        target_point_name="target_latent_point_true_recovar_h100_fullmem_movingfocus_zdim4_reg_state0050.txt",
        color="#1f77b4",
    ),
    Method(
        label="source oracle zdim4/reg",
        short="source oracle",
        compute_relpath="07_compute_state_oracle_regfix_zdim4_reg_lazy",
        embedding_relpath="06_pipeline_oracle_regfix_20260526/model/zdim_4",
        target_point_name="target_latent_point_oracle_regfix_zdim4_reg_state0050.txt",
        color="#d62728",
    ),
)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_direct_comparison(out_dir: Path, freq: np.ndarray, payloads: dict[str, dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6), constrained_layout=True)
    fig.suptitle(
        "1M noise=30 | real pipeline vs source-oracle compute_state | state 50 broad mask",
        fontweight="bold",
    )
    for method in METHODS:
        payload = payloads[method.short]
        color = method.color
        axes[0].plot(freq, payload["final_fsc"], color=color, lw=3.0, label=f"{method.short} final")
        axes[0].plot(freq, payload["best_fsc"], color=color, lw=2.0, ls="--", label=f"{method.short} best cand")
        axes[0].plot(freq, payload["nearest_fsc"], color=color, lw=1.8, ls=":", label=f"{method.short} distance-nearest GT")
        axes[1].semilogy(freq, np.maximum(payload["final_err"], 1e-30), color=color, lw=3.0, label=f"{method.short} final")
        axes[1].semilogy(freq, np.maximum(payload["best_err"], 1e-30), color=color, lw=2.0, ls="--", label=f"{method.short} best cand")
        axes[1].semilogy(freq, np.maximum(payload["nearest_err"], 1e-30), color=color, lw=1.8, ls=":", label=f"{method.short} distance-nearest GT")
    axes[0].plot(freq, payloads["controls"]["mean_fsc"], color="0.25", lw=1.7, ls="-.", label="GT distribution mean")
    axes[1].semilogy(freq, np.maximum(payloads["controls"]["mean_err"], 1e-30), color="0.25", lw=1.7, ls="-.", label="GT distribution mean")
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
        ax.legend(fontsize=8.0)
    png = out_dir / "noise30_1m_pipeline_vs_oracle_final_best_gt_controls.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "noise30_1m_pipeline_vs_oracle_final_best_gt_controls.pdf")
    plt.close(fig)
    return png


def plot_all_candidates(out_dir: Path, freq: np.ndarray, payloads: dict[str, dict[str, object]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.2), sharex=True)
    fig.suptitle(
        "1M noise=30 | all 50 unfiltered bandwidth candidates | state 50 broad mask",
        fontweight="bold",
    )
    for row, method in enumerate(METHODS):
        payload = payloads[method.short]
        grid = payload["grid"]
        cmap = plt.cm.viridis
        norm = plt.Normalize(float(np.nanmin(grid)), float(np.nanmax(grid)))
        for idx, value in enumerate(grid):
            color = cmap(norm(float(value)))
            axes[row, 0].plot(freq, payload["candidate_fsc"][idx], color=color, lw=1.1, alpha=0.65)
            axes[row, 1].semilogy(freq, np.maximum(payload["candidate_err"][idx], 1e-30), color=color, lw=1.1, alpha=0.65)
        axes[row, 0].plot(freq, payload["final_fsc"], color="black", lw=3.0, label="final compute_state")
        axes[row, 0].plot(freq, payload["nearest_fsc"], color=method.color, lw=2.2, ls="--", label="distance-nearest GT")
        axes[row, 1].semilogy(freq, np.maximum(payload["final_err"], 1e-30), color="black", lw=3.0, label="final compute_state")
        axes[row, 1].semilogy(freq, np.maximum(payload["nearest_err"], 1e-30), color=method.color, lw=2.2, ls="--", label="distance-nearest GT")
        axes[row, 0].axhline(0.5, color="0.45", ls=":", lw=1.0)
        axes[row, 1].axhline(0.1, color="0.45", ls=":", lw=1.0)
        axes[row, 0].set_title(f"{method.short}: FSC vs GT")
        axes[row, 1].set_title(f"{method.short}: relative error vs GT")
        axes[row, 0].set_ylim(-0.08, 1.03)
        axes[row, 1].set_ylim(1e-3, 1e3)
        axes[row, 0].legend(fontsize=8.0)
        axes[row, 1].legend(fontsize=8.0)
    for ax in axes.ravel():
        ax.set_xlim(0.0, FREQ_MAX)
        ax.set_xlabel("spatial frequency (1/A)")
        ax.grid(alpha=0.25, which="both")
    axes[0, 0].set_ylabel("masked FSC")
    axes[1, 0].set_ylabel("masked FSC")
    axes[0, 1].set_ylabel("masked relative error")
    axes[1, 1].set_ylabel("masked relative error")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(
        min(float(np.nanmin(payloads[m.short]["grid"])) for m in METHODS),
        max(float(np.nanmax(payloads[m.short]["grid"])) for m in METHODS),
    ), cmap=plt.cm.viridis)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.78, pad=0.012)
    cbar.set_label("candidate bandwidth h")
    png = out_dir / "noise30_1m_pipeline_vs_oracle_all50_candidates.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "noise30_1m_pipeline_vs_oracle_all50_candidates.pdf")
    plt.close(fig)
    return png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--nearest-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def recomputed_distance_gt_mixture(
    run: Path,
    compute_root: Path,
    n_nearest: int,
) -> tuple[np.ndarray, dict[str, object]]:
    weights, nearest, distances, info_in = state_weights_from_compute_state_nearest(run, compute_root, n_nearest)
    states = np.asarray(np.load(run / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    counts = np.bincount(states[nearest], minlength=100).astype(np.float64)
    top_states = [
        f"{int(state)}:{int(counts[state])}"
        for state in np.argsort(counts)[::-1][:12]
        if counts[state] > 0
    ]
    info = {
        "nearest_source": "recomputed from compute_state command/pipeline/target",
        "pipeline": str(info_in.pipeline),
        "target_point": str(info_in.target_path),
        "embedding_option": info_in.embedding_option,
        "coords_entry": info_in.coords_entry,
        "n_nearest": int(nearest.size),
        "nearest_radius": float(np.nanmax(distances[nearest])),
        "state50_fraction": float(weights[TARGET_STATE]),
        "top_states": " ".join(top_states),
    }
    return state_weighted_gt_mrc(run, weights), info


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask = np.clip(load_mrc(args.mask), 0.0, 1.0)
    target = load_mrc(args.run_dir / "04_ground_truth" / f"gt_vol{TARGET_STATE:04d}.mrc")
    labels, n_shells, target_ft, target_power, freq = metric_context(target, mask)

    mean_fsc, mean_err = masked_metrics(
        distribution_mean_gt_mrc(args.run_dir), mask, labels, n_shells, target_ft, target_power
    )
    payloads: dict[str, dict[str, object]] = {
        "controls": {"mean_fsc": mean_fsc, "mean_err": mean_err}
    }
    rows: list[dict[str, object]] = []

    for method in METHODS:
        compute_root = args.run_dir / method.compute_relpath
        print(method.label, flush=True)
        grid, candidate_fsc, candidate_err = candidate_metrics(
            compute_root,
            mask,
            labels,
            n_shells,
            target_ft,
            target_power,
            args.batch_size,
        )
        fsc_res = np.asarray([fsc05_resolution(freq, curve) for curve in candidate_fsc])
        err_res = np.asarray([relerr_resolution(freq, curve, 0.1) for curve in candidate_err])
        best_fsc_idx = int(np.nanargmin(fsc_res))
        best_err_idx = int(np.nanargmin(err_res))
        final_fsc, final_err = masked_metrics(
            load_mrc(compute_root / "state000_unfil.mrc"), mask, labels, n_shells, target_ft, target_power
        )
        nearest_vol, nearest_info = recomputed_distance_gt_mixture(args.run_dir, compute_root, args.nearest_count)
        nearest_fsc, nearest_err = masked_metrics(nearest_vol, mask, labels, n_shells, target_ft, target_power)

        payloads[method.short] = {
            "grid": grid,
            "candidate_fsc": candidate_fsc,
            "candidate_err": candidate_err,
            "final_fsc": final_fsc,
            "final_err": final_err,
            "best_fsc": candidate_fsc[best_fsc_idx],
            "best_err": candidate_err[best_err_idx],
            "nearest_fsc": nearest_fsc,
            "nearest_err": nearest_err,
        }
        rows.append(
            {
                "method": method.label,
                "compute_state_dir": str(compute_root),
                "final_fsc05_resolution_A": fsc05_resolution(freq, final_fsc),
                "final_relerr10_resolution_A": relerr_resolution(freq, final_err, 0.1),
                "best_fsc_candidate_0based": best_fsc_idx,
                "best_fsc_candidate_1based": best_fsc_idx + 1,
                "best_fsc_candidate_h": float(grid[best_fsc_idx]),
                "best_fsc05_resolution_A": float(fsc_res[best_fsc_idx]),
                "best_error_candidate_0based": best_err_idx,
                "best_error_candidate_1based": best_err_idx + 1,
                "best_error_candidate_h": float(grid[best_err_idx]),
                "best_relerr10_resolution_A": float(err_res[best_err_idx]),
                "distance_nearest1000_gt_fsc05_resolution_A": fsc05_resolution(freq, nearest_fsc),
                "distance_nearest1000_gt_relerr10_resolution_A": relerr_resolution(freq, nearest_err, 0.1),
                "distance_nearest1000_gt_state50_fraction": nearest_info["state50_fraction"],
                "distance_nearest1000_gt_top_states": nearest_info["top_states"],
                "distance_nearest_source": nearest_info["nearest_source"],
                "target_point": str(args.run_dir / method.target_point_name),
            }
        )

    rows.append(
        {
            "method": "GT distribution mean",
            "compute_state_dir": "",
            "final_fsc05_resolution_A": fsc05_resolution(freq, mean_fsc),
            "final_relerr10_resolution_A": relerr_resolution(freq, mean_err, 0.1),
            "best_fsc_candidate_0based": "",
            "best_fsc_candidate_1based": "",
            "best_fsc_candidate_h": "",
            "best_fsc05_resolution_A": "",
            "best_error_candidate_0based": "",
            "best_error_candidate_1based": "",
            "best_error_candidate_h": "",
            "best_relerr10_resolution_A": "",
            "distance_nearest1000_gt_fsc05_resolution_A": "",
            "distance_nearest1000_gt_relerr10_resolution_A": "",
            "distance_nearest1000_gt_state50_fraction": "",
            "distance_nearest1000_gt_top_states": "",
            "distance_nearest_source": "",
            "target_point": "",
        }
    )

    write_summary(args.out_dir / "noise30_1m_pipeline_vs_oracle_summary.csv", rows)
    direct = plot_direct_comparison(args.out_dir, freq, payloads)
    all50 = plot_all_candidates(args.out_dir, freq, payloads)
    print(f"PLOT {direct}", flush=True)
    print(f"PLOT {all50}", flush=True)
    print(f"SUMMARY {args.out_dir / 'noise30_1m_pipeline_vs_oracle_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
