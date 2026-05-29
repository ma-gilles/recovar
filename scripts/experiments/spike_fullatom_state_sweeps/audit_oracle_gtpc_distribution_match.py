#!/usr/bin/env python3
"""Audit whether source-oracle and GT-PC cov-noise embeddings match by label."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from gt_embedding_controls import raw_gt_pc_coordinates


DEFAULT_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518/"
    "n03000000/runs/n03000000_seed0000"
)
DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "oracle_noreg_vs_gtpc_covnoise_repro_20260528/distribution_match_audit"
)
SOURCE_PIPELINE = "06_pipeline_oracle_regfix_20260526"
GTPC_PIPELINE = "06_pipeline_gtpc_covnoise_noreg_trueunits_zdim4_seed20260528"
ZDIM = 4
N_STATES = 100


def load_embedding(pipeline: Path) -> np.ndarray:
    return np.asarray(
        np.load(pipeline / "model" / f"zdim_{ZDIM}" / "latent_coords_noreg.npy"),
        dtype=np.float64,
    )[:, :ZDIM]


def empirical_covariance(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=0, keepdims=True)
    return (centered.T @ centered) / max(x.shape[0] - 1, 1)


def state_means_and_covariances(z: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty((N_STATES, z.shape[1]), dtype=np.float64)
    covariances = np.empty((N_STATES, z.shape[1], z.shape[1]), dtype=np.float64)
    for state in range(N_STATES):
        by_state = z[states == state]
        means[state] = np.mean(by_state, axis=0)
        covariances[state] = empirical_covariance(by_state)
    return means, covariances


def ellipse(center: np.ndarray, covariance_2d: np.ndarray, color: str, *, linestyle: str = "-", label: str | None = None) -> Ellipse:
    values, vectors = np.linalg.eigh(0.5 * (covariance_2d + covariance_2d.T))
    values = np.maximum(values, 0.0)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    return Ellipse(
        xy=(float(center[0]), float(center[1])),
        width=float(2.0 * np.sqrt(values[0])),
        height=float(2.0 * np.sqrt(values[1])),
        angle=angle,
        fill=False,
        edgecolor=color,
        linestyle=linestyle,
        linewidth=1.55,
        alpha=0.75,
        label=label,
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(out_dir: Path, rows: list[dict[str, object]]) -> Path:
    state = np.asarray([row["state"] for row in rows], dtype=np.int32)
    source_bias = np.asarray([row["source_mean_minus_gt_norm"] for row in rows], dtype=np.float64)
    gtpc_bias = np.asarray([row["gtpc_mean_minus_gt_norm"] for row in rows], dtype=np.float64)
    mean_delta = np.asarray([row["source_minus_gtpc_mean_norm"] for row in rows], dtype=np.float64)
    mean_delta_norm = np.asarray([row["source_minus_gtpc_mean_norm_over_source_sigma"] for row in rows], dtype=np.float64)
    trace_source = np.asarray([row["source_cov_trace"] for row in rows], dtype=np.float64)
    trace_gtpc = np.asarray([row["gtpc_cov_trace"] for row in rows], dtype=np.float64)
    trace_ratio = np.asarray([row["gtpc_over_source_cov_trace"] for row in rows], dtype=np.float64)
    cov_rel = np.asarray([row["cov_frobenius_rel_to_source"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.1), constrained_layout=True)
    fig.suptitle(
        "State-conditional distribution audit | source oracle vs GT-PC sampled covariance | zdim4 no-reg",
        fontweight="bold",
    )
    axes[0, 0].plot(state, source_bias, color="#1f77b4", marker="o", ms=3, lw=1.35, label="source mean - raw GT")
    axes[0, 0].plot(state, gtpc_bias, color="#d62728", marker="o", ms=3, lw=1.35, label="GT-PC covnoise mean - raw GT")
    axes[0, 0].set_title("Conditional mean bias to raw GT coordinate")
    axes[0, 0].set_ylabel("L2 norm in latent units")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(state, mean_delta, color="#6a3d9a", marker="o", ms=3, lw=1.35)
    axes[0, 1].set_title("Source vs GT-PC conditional mean mismatch")
    axes[0, 1].set_ylabel("||mean_source - mean_gtpc||")

    axes[0, 2].plot(state, mean_delta_norm, color="#6a3d9a", marker="o", ms=3, lw=1.35)
    axes[0, 2].axhline(np.median(mean_delta_norm), color="0.35", ls="--", lw=1.0, label=f"median {np.median(mean_delta_norm):.3g}")
    axes[0, 2].set_title("Mean mismatch normalized by source sigma")
    axes[0, 2].set_ylabel("mismatch / sqrt(trace Cov_source)")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].semilogy(state, trace_source, color="#1f77b4", marker="o", ms=3, lw=1.35, label="source empirical cov trace")
    axes[1, 0].semilogy(state, trace_gtpc, color="#d62728", marker="o", ms=3, lw=1.35, label="GT-PC covnoise empirical cov trace")
    axes[1, 0].set_title("Conditional empirical covariance size")
    axes[1, 0].set_ylabel("trace covariance")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(state, trace_ratio, color="#ff7f00", marker="o", ms=3, lw=1.35)
    axes[1, 1].axhline(1.0, color="0.35", ls="--", lw=1.0)
    axes[1, 1].set_title("Covariance trace ratio")
    axes[1, 1].set_ylabel("trace GT-PC covnoise / trace source")

    axes[1, 2].plot(state, cov_rel, color="#b15928", marker="o", ms=3, lw=1.35)
    axes[1, 2].set_title("Full covariance matrix mismatch")
    axes[1, 2].set_ylabel("||Cov_gtpc - Cov_source||_F / ||Cov_source||_F")

    for ax in axes.ravel():
        ax.set_xlabel("GT state label")
        ax.grid(alpha=0.25)
    png = out_dir / "state_conditional_distribution_match_summary.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "state_conditional_distribution_match_summary.pdf")
    plt.close(fig)
    return png


def plot_embedding_overlay(
    out_dir: Path,
    states: np.ndarray,
    z_source: np.ndarray,
    z_gtpc: np.ndarray,
    gt_z: np.ndarray,
    source_means: np.ndarray,
    gtpc_means: np.ndarray,
) -> Path:
    rng = np.random.default_rng(20260528)
    show = rng.choice(states.size, size=min(120_000, states.size), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0), constrained_layout=True)
    fig.suptitle("Embedding clouds with raw GT-PC path and conditional state means", fontweight="bold")
    for col, (title, z, means, mean_color) in enumerate(
        (
            ("source oracle no-reg", z_source, source_means, "#1f77b4"),
            ("GT-PC + sampled oracle covariance no-reg", z_gtpc, gtpc_means, "#d62728"),
        )
    ):
        for row, (a, b) in enumerate(((0, 1), (2, 3))):
            ax = axes[row, col]
            points = ax.scatter(z[show, a], z[show, b], c=states[show], cmap="viridis", s=1.1, alpha=0.16, rasterized=True)
            ax.plot(gt_z[:, a], gt_z[:, b], color="black", lw=2.2, label="raw GT-PC path")
            ax.scatter(gt_z[:, a], gt_z[:, b], c=np.arange(N_STATES), cmap="viridis", s=23, edgecolor="black", lw=0.25)
            ax.plot(means[:, a], means[:, b], color=mean_color, lw=1.8, label="mean z | state")
            ax.scatter(means[:, a], means[:, b], c=np.arange(N_STATES), cmap="plasma", s=18, alpha=0.85)
            ax.set_title(f"{title}: z{a} vs z{b}")
            ax.set_xlabel(f"z{a}")
            ax.set_ylabel(f"z{b}")
            ax.grid(alpha=0.20)
            ax.legend(fontsize=8)
    fig.colorbar(points, ax=axes, pad=0.012, fraction=0.025, label="particle GT state label")
    png = out_dir / "embedding_clouds_with_raw_gt_and_state_means.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "embedding_clouds_with_raw_gt_and_state_means.pdf")
    plt.close(fig)
    return png


def plot_covariance_ellipses(
    out_dir: Path,
    gt_z: np.ndarray,
    source_means: np.ndarray,
    gtpc_means: np.ndarray,
    source_covs: np.ndarray,
    gtpc_covs: np.ndarray,
) -> Path:
    selected_states = (0, 12, 25, 50, 75, 99)
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.0), constrained_layout=True)
    fig.suptitle("Empirical covariance ellipses by GT label | 1 sigma around state mean", fontweight="bold")
    for row, (a, b) in enumerate(((0, 1), (2, 3))):
        for col, zoom in enumerate((False, True)):
            ax = axes[row, col]
            ax.plot(gt_z[:, a], gt_z[:, b], color="black", lw=2.0, label="raw GT-PC path")
            for state in selected_states:
                ax.scatter(source_means[state, a], source_means[state, b], color="#1f77b4", s=28)
                ax.scatter(gtpc_means[state, a], gtpc_means[state, b], color="#d62728", s=28)
                ax.add_patch(
                    ellipse(
                        source_means[state, [a, b]],
                        source_covs[state][np.ix_([a, b], [a, b])],
                        "#1f77b4",
                        label="source empirical cov" if state == selected_states[0] else None,
                    )
                )
                ax.add_patch(
                    ellipse(
                        gtpc_means[state, [a, b]],
                        gtpc_covs[state][np.ix_([a, b], [a, b])],
                        "#d62728",
                        linestyle="--",
                        label="GT-PC covnoise empirical cov" if state == selected_states[0] else None,
                    )
                )
                ax.text(gtpc_means[state, a], gtpc_means[state, b], str(state), fontsize=7, color="#d62728")
            if zoom:
                points = np.vstack(
                    [
                        source_means[list(selected_states)][:, [a, b]],
                        gtpc_means[list(selected_states)][:, [a, b]],
                        gt_z[list(selected_states)][:, [a, b]],
                    ]
                )
                lo = np.min(points, axis=0)
                hi = np.max(points, axis=0)
                span = np.maximum(hi - lo, 1.0)
                ax.set_xlim(float(lo[0] - 0.35 * span[0]), float(hi[0] + 0.35 * span[0]))
                ax.set_ylim(float(lo[1] - 0.35 * span[1]), float(hi[1] + 0.35 * span[1]))
            ax.set_title(f"z{a} vs z{b}" + (" zoom" if zoom else " full view"))
            ax.set_xlabel(f"z{a}")
            ax.set_ylabel(f"z{b}")
            ax.grid(alpha=0.20)
            ax.legend(fontsize=8, loc="best")
    png = out_dir / "state_empirical_covariance_ellipses_selected.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "state_empirical_covariance_ellipses_selected.pdf")
    plt.close(fig)
    return png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_pipeline = args.run_dir / SOURCE_PIPELINE
    gtpc_pipeline = args.run_dir / GTPC_PIPELINE
    states = np.asarray(np.load(args.run_dir / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    z_source = load_embedding(source_pipeline)
    z_gtpc = load_embedding(gtpc_pipeline)
    gt_source = raw_gt_pc_coordinates(args.run_dir, source_pipeline, ZDIM)
    gt_gtpc = raw_gt_pc_coordinates(args.run_dir, gtpc_pipeline, ZDIM)
    source_means, source_covs = state_means_and_covariances(z_source, states)
    gtpc_means, gtpc_covs = state_means_and_covariances(z_gtpc, states)

    rows: list[dict[str, object]] = []
    for state in range(N_STATES):
        source_cov = source_covs[state]
        gtpc_cov = gtpc_covs[state]
        source_trace = float(np.trace(source_cov))
        mean_delta = float(np.linalg.norm(source_means[state] - gtpc_means[state]))
        rows.append(
            {
                "state": state,
                "n_particles": int(np.sum(states == state)),
                "raw_gt_source_vs_gtpc_norm": float(np.linalg.norm(gt_source[state] - gt_gtpc[state])),
                "source_mean_minus_gt_norm": float(np.linalg.norm(source_means[state] - gt_source[state])),
                "gtpc_mean_minus_gt_norm": float(np.linalg.norm(gtpc_means[state] - gt_source[state])),
                "source_minus_gtpc_mean_norm": mean_delta,
                "source_minus_gtpc_mean_norm_over_source_sigma": mean_delta / np.sqrt(max(source_trace, 1e-30)),
                "source_cov_trace": source_trace,
                "gtpc_cov_trace": float(np.trace(gtpc_cov)),
                "gtpc_over_source_cov_trace": float(np.trace(gtpc_cov) / source_trace),
                "cov_frobenius_rel_to_source": float(
                    np.linalg.norm(gtpc_cov - source_cov, ord="fro") / max(np.linalg.norm(source_cov, ord="fro"), 1e-30)
                ),
                "source_cov_eigvals": " ".join(f"{val:.8g}" for val in np.linalg.eigvalsh(source_cov)[::-1]),
                "gtpc_cov_eigvals": " ".join(f"{val:.8g}" for val in np.linalg.eigvalsh(gtpc_cov)[::-1]),
            }
        )

    csv_path = args.out_dir / "state_conditional_distribution_match.csv"
    write_rows(csv_path, rows)
    summary_plot = plot_summary(args.out_dir, rows)
    embedding_plot = plot_embedding_overlay(args.out_dir, states, z_source, z_gtpc, gt_source, source_means, gtpc_means)
    ellipse_plot = plot_covariance_ellipses(args.out_dir, gt_source, source_means, gtpc_means, source_covs, gtpc_covs)

    mean_delta_norm = np.asarray([row["source_minus_gtpc_mean_norm_over_source_sigma"] for row in rows], dtype=np.float64)
    trace_ratio = np.asarray([row["gtpc_over_source_cov_trace"] for row in rows], dtype=np.float64)
    cov_rel = np.asarray([row["cov_frobenius_rel_to_source"] for row in rows], dtype=np.float64)
    print(f"CSV {csv_path}")
    print(f"SUMMARY_PLOT {summary_plot}")
    print(f"EMBEDDING_PLOT {embedding_plot}")
    print(f"ELLIPSE_PLOT {ellipse_plot}")
    print(f"mean_mismatch_over_source_sigma median={np.median(mean_delta_norm):.6g} max={np.max(mean_delta_norm):.6g}")
    print(f"cov_trace_ratio median={np.median(trace_ratio):.6g} p10={np.percentile(trace_ratio, 10):.6g} p90={np.percentile(trace_ratio, 90):.6g}")
    print(f"cov_fro_rel median={np.median(cov_rel):.6g} p10={np.percentile(cov_rel, 10):.6g} p90={np.percentile(cov_rel, 90):.6g}")


if __name__ == "__main__":
    main()
