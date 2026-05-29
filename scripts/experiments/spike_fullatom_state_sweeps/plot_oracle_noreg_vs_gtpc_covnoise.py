#!/usr/bin/env python3
"""Compare source-oracle no-reg against GT-PC plus sampled covariance noise.

This is the cleaned version of the 3M/noise30 oracle-gap diagnostic.  It uses
the actual compute_state command files to recover the pipeline, target point,
embedding entry, precision entry, and distance convention.  GT mixture controls
are then recomputed from those current inputs, avoiding stale scratch arrays.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from gt_embedding_controls import (
    _load_unsorted_embedding,
    parse_compute_state_inputs,
    raw_gt_pc_coordinates,
    state_weights_from_compute_state_nearest,
)
from plot_noise1_sweep_with_gt_nearest import (
    DEFAULT_MASK,
    FREQ_MAX,
    TARGET_STATE,
    candidate_metrics,
    distribution_mean_gt_mrc,
    fsc05_resolution,
    load_mrc,
    masked_metrics,
    metric_context,
    relerr_resolution,
    state_weighted_gt_mrc,
)


DEFAULT_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518/"
    "n03000000/runs/n03000000_seed0000"
)
DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "oracle_noreg_vs_gtpc_covnoise_repro_20260528"
)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    compute_relpath: str
    color: str


METHODS = (
    Method(
        key="source_oracle_noreg",
        label="source oracle zdim4 no-reg",
        compute_relpath="07_compute_state_oracle_regfix_zdim4_noreg_lazy",
        color="#1f77b4",
    ),
    Method(
        key="gtpc_covnoise_noreg",
        label="GT-PC + sampled oracle covariance zdim4 no-reg",
        compute_relpath="07_compute_state_gtpc_covnoise_noreg_trueunits_zdim4_noreg_lazy",
        color="#d62728",
    ),
)

CONTROL_COLORS = {
    "source_oracle_noreg_nearest100": "#2ca02c",
    "source_oracle_noreg_nearest1000": "#17becf",
    "gtpc_covnoise_noreg_nearest100": "#ff7f0e",
    "gtpc_covnoise_noreg_nearest1000": "#9467bd",
    "mean": "#6b6b6b",
}


def write_curves(path: Path, freq: np.ndarray, rows: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shell", "frequency_1_per_A", *rows])
        for idx, value in enumerate(freq):
            writer.writerow([idx, float(value), *[float(curve[idx]) for curve in rows.values()]])


def recomputed_nearest_gt(
    run_dir: Path,
    compute_root: Path,
    n_nearest: int,
) -> tuple[np.ndarray, dict[str, object]]:
    weights, nearest, distances, info = state_weights_from_compute_state_nearest(
        run_dir,
        compute_root,
        n_nearest,
    )
    states = np.asarray(np.load(run_dir / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    counts = np.bincount(states[nearest], minlength=100).astype(np.float64)
    top_states = " ".join(
        f"{int(state)}:{int(counts[state])}"
        for state in np.argsort(counts)[::-1][:12]
        if counts[state] > 0
    )
    payload = {
        "n_nearest": int(nearest.size),
        "state50_fraction": float(weights[TARGET_STATE]),
        "top_states": top_states,
        "distance_min": float(np.min(distances[nearest])),
        "distance_max": float(np.max(distances[nearest])),
        "compute_root": str(compute_root),
        "pipeline": str(info.pipeline),
        "target_point": str(info.target_path),
        "embedding_option": info.embedding_option,
        "coords_entry": info.coords_entry,
        "precision_entry": info.precision_entry,
    }
    return state_weighted_gt_mrc(run_dir, weights), payload


def load_method(
    run_dir: Path,
    method: Method,
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    target_ft: np.ndarray,
    target_power: np.ndarray,
    freq: np.ndarray,
    batch_size: int,
    nearest_counts: tuple[int, ...],
) -> dict[str, object]:
    compute_root = run_dir / method.compute_relpath
    if not (compute_root / "state000_unfil.mrc").exists():
        raise FileNotFoundError(f"Missing compute_state output: {compute_root / 'state000_unfil.mrc'}")

    grid, cand_fsc, cand_err = candidate_metrics(
        compute_root,
        mask,
        labels,
        n_shells,
        target_ft,
        target_power,
        batch_size,
    )
    final_fsc, final_err = masked_metrics(
        load_mrc(compute_root / "state000_unfil.mrc"),
        mask,
        labels,
        n_shells,
        target_ft,
        target_power,
    )
    fsc_res = np.asarray([fsc05_resolution(freq, curve) for curve in cand_fsc], dtype=np.float64)
    err_res = np.asarray([relerr_resolution(freq, curve, 0.1) for curve in cand_err], dtype=np.float64)
    best_fsc_idx = int(np.nanargmin(fsc_res))
    best_err_idx = int(np.nanargmin(err_res))

    nearest: dict[int, dict[str, object]] = {}
    for n_nearest in nearest_counts:
        volume, info = recomputed_nearest_gt(run_dir, compute_root, n_nearest)
        gt_fsc, gt_err = masked_metrics(volume, mask, labels, n_shells, target_ft, target_power)
        info.update(
            {
                "fsc": gt_fsc,
                "err": gt_err,
                "fsc05_resolution_A": fsc05_resolution(freq, gt_fsc),
                "relerr10_resolution_A": relerr_resolution(freq, gt_err, 0.1),
            }
        )
        nearest[n_nearest] = info

    return {
        "method": method,
        "compute_root": compute_root,
        "grid": grid,
        "candidate_fsc": cand_fsc,
        "candidate_err": cand_err,
        "final_fsc": final_fsc,
        "final_err": final_err,
        "best_fsc_idx": best_fsc_idx,
        "best_err_idx": best_err_idx,
        "best_fsc_resolution_A": fsc_res[best_fsc_idx],
        "best_err_resolution_A": err_res[best_err_idx],
        "nearest": nearest,
    }


def plot_comparison(
    out_dir: Path,
    freq: np.ndarray,
    loaded: list[dict[str, object]],
    mean_fsc: np.ndarray,
    mean_err: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 9.2), sharex=True, constrained_layout=True)
    fig.suptitle(
        "3M noise30 spike | source oracle no-reg vs GT-PC + sampled oracle covariance | unfiltered maps",
        fontweight="bold",
    )
    for item in loaded:
        method: Method = item["method"]  # type: ignore[assignment]
        color = method.color
        best_fsc_idx = int(item["best_fsc_idx"])
        best_err_idx = int(item["best_err_idx"])
        grid = item["grid"]  # type: ignore[assignment]
        cand_fsc = item["candidate_fsc"]  # type: ignore[assignment]
        cand_err = item["candidate_err"]  # type: ignore[assignment]
        final_fsc = item["final_fsc"]  # type: ignore[assignment]
        final_err = item["final_err"]  # type: ignore[assignment]

        axes[0, 0].plot(freq, final_fsc, color=color, lw=2.8, label=f"{method.label}: compute_state")
        axes[1, 0].semilogy(freq, np.maximum(final_err, 1e-30), color=color, lw=2.8)
        axes[0, 0].plot(
            freq,
            cand_fsc[best_fsc_idx],
            color=color,
            ls="--",
            lw=2.2,
            label=f"{method.label}: best cand #{best_fsc_idx + 1}, h={grid[best_fsc_idx]:.3g}",
        )
        axes[1, 0].semilogy(freq, np.maximum(cand_err[best_err_idx], 1e-30), color=color, ls="--", lw=2.2)

        nearest = item["nearest"]  # type: ignore[assignment]
        for n_nearest, ls, marker in ((100, "-.", "o"), (1000, ":", "s")):
            if n_nearest not in nearest:
                continue
            control = nearest[n_nearest]
            control_color = CONTROL_COLORS[f"{method.key}_nearest{n_nearest}"]
            axes[0, 0].plot(
                freq,
                control["fsc"],
                color=control_color,
                ls=ls,
                lw=2.25,
                marker=marker,
                markevery=11,
                ms=3.0,
                label=f"{method.label}: nearest-{n_nearest} GT mix",
            )
            axes[1, 0].semilogy(
                freq,
                np.maximum(control["err"], 1e-30),
                color=control_color,
                ls=ls,
                lw=2.25,
                marker=marker,
                markevery=11,
                ms=3.0,
            )

        axes[0, 1].plot(
            freq,
            cand_fsc[best_fsc_idx],
            color=color,
            lw=2.4,
            label=f"{method.label}: best FSC cand",
        )
        axes[1, 1].semilogy(
            freq,
            np.maximum(cand_err[best_err_idx], 1e-30),
            color=color,
            lw=2.4,
            label=f"{method.label}: best error cand",
        )

    mean_color = CONTROL_COLORS["mean"]
    axes[0, 0].plot(freq, mean_fsc, color=mean_color, ls=(0, (2, 2)), lw=2.0, label="uniform GT mean")
    axes[1, 0].semilogy(freq, np.maximum(mean_err, 1e-30), color=mean_color, ls=(0, (2, 2)), lw=2.0)
    axes[0, 1].plot(freq, mean_fsc, color=mean_color, ls=(0, (2, 2)), lw=2.0, label="uniform GT mean")
    axes[1, 1].semilogy(freq, np.maximum(mean_err, 1e-30), color=mean_color, ls=(0, (2, 2)), lw=2.0)

    axes[0, 0].set_title("Final, best candidate, and recomputed GT-nearest controls")
    axes[1, 0].set_title("Relative error for the same curves")
    axes[0, 1].set_title("Best candidate only")
    axes[1, 1].set_title("Best relative-error candidate only")
    for ax in axes[0]:
        ax.axhline(0.5, color="0.45", ls=":", lw=1.0)
        ax.set_ylim(-0.08, 1.03)
        ax.set_ylabel("masked FSC vs GT state 50")
    for ax in axes[1]:
        ax.axhline(0.1, color="0.45", ls=":", lw=1.0)
        ax.set_ylim(1e-3, 1e3)
        ax.set_ylabel("masked relative Fourier error")
    for ax in axes.ravel():
        ax.set_xlim(0.0, FREQ_MAX)
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel("spatial frequency (1/A)")
    axes[0, 0].legend(fontsize=7.0, ncols=2, loc="lower left")
    axes[0, 1].legend(fontsize=8.0, loc="lower left")
    axes[1, 1].legend(fontsize=8.0, loc="lower left")
    png = out_dir / "oracle_noreg_vs_gtpc_covnoise_fsc_error.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "oracle_noreg_vs_gtpc_covnoise_fsc_error.pdf")
    plt.close(fig)
    return png


def plot_all_candidates(out_dir: Path, freq: np.ndarray, loaded: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16.2, 9.4), sharex=True, constrained_layout=True)
    fig.suptitle(
        "All 50 candidates | source oracle no-reg vs GT-PC sampled-covariance no-reg",
        fontweight="bold",
    )
    for col, item in enumerate(loaded):
        method: Method = item["method"]  # type: ignore[assignment]
        grid = item["grid"]  # type: ignore[assignment]
        cand_fsc = item["candidate_fsc"]  # type: ignore[assignment]
        cand_err = item["candidate_err"]  # type: ignore[assignment]
        final_fsc = item["final_fsc"]  # type: ignore[assignment]
        final_err = item["final_err"]  # type: ignore[assignment]
        positive = grid[grid > 0]
        if positive.size == grid.size and float(np.max(grid) / np.min(positive)) > 10.0:
            norm: mcolors.Normalize = mcolors.LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(grid)))
        else:
            norm = mcolors.Normalize(vmin=float(np.min(grid)), vmax=float(np.max(grid)))
        cmap = plt.get_cmap("viridis")
        for idx, h_val in enumerate(grid):
            color = cmap(norm(float(h_val)))
            axes[0, col].plot(freq, cand_fsc[idx], color=color, lw=0.95, alpha=0.62)
            axes[1, col].semilogy(freq, np.maximum(cand_err[idx], 1e-30), color=color, lw=0.95, alpha=0.62)
        axes[0, col].plot(freq, final_fsc, color="black", lw=2.7, label="compute_state output")
        axes[1, col].semilogy(freq, np.maximum(final_err, 1e-30), color="black", lw=2.7)
        axes[0, col].set_title(f"{method.label}: FSC")
        axes[1, col].set_title(f"{method.label}: relative error")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:, col], pad=0.012, fraction=0.040)
        cbar.set_label("candidate bandwidth h")
    for ax in axes[0]:
        ax.axhline(0.5, color="0.45", ls=":", lw=1.0)
        ax.set_ylim(-0.08, 1.03)
        ax.legend(fontsize=8.0, loc="lower left")
    for ax in axes[1]:
        ax.axhline(0.1, color="0.45", ls=":", lw=1.0)
        ax.set_ylim(1e-3, 1e3)
    for ax in axes.ravel():
        ax.set_xlim(0.0, FREQ_MAX)
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel("spatial frequency (1/A)")
    axes[0, 0].set_ylabel("masked FSC vs GT state 50")
    axes[1, 0].set_ylabel("masked relative Fourier error")
    png = out_dir / "oracle_noreg_vs_gtpc_covnoise_all50_candidates.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "oracle_noreg_vs_gtpc_covnoise_all50_candidates.pdf")
    plt.close(fig)
    return png


def plot_embeddings(out_dir: Path, run_dir: Path, loaded: list[dict[str, object]]) -> Path:
    states = np.asarray(np.load(run_dir / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(20260528)
    n_show = min(80_000, states.size)
    show = rng.choice(states.size, size=n_show, replace=False) if states.size > n_show else np.arange(states.size)

    fig, axes = plt.subplots(len(loaded), 2, figsize=(14.6, 6.4), constrained_layout=True)
    fig.suptitle(
        "Embedding comparison | colored by GT state | nearest-100 particles highlighted",
        fontweight="bold",
    )
    if len(loaded) == 1:
        axes = np.asarray([axes])

    for row, item in enumerate(loaded):
        method: Method = item["method"]  # type: ignore[assignment]
        compute_root: Path = item["compute_root"]  # type: ignore[assignment]
        info = parse_compute_state_inputs(compute_root)
        z = _load_unsorted_embedding(info.pipeline, info.coords_entry, info.zdim)[:, : info.zdim]
        distances, _ = state_weights_from_compute_state_nearest(run_dir, compute_root, 100)[2:]
        nearest = np.argpartition(distances, 99)[:100]
        dims = ((0, 1), (2, 3))
        for col, (a, b) in enumerate(dims):
            ax = axes[row, col]
            if info.zdim <= max(a, b):
                ax.axis("off")
                continue
            scat = ax.scatter(
                z[show, a],
                z[show, b],
                c=states[show],
                s=1.2,
                alpha=0.20,
                cmap="viridis",
                rasterized=True,
            )
            ax.scatter(
                z[nearest, a],
                z[nearest, b],
                s=24,
                facecolors="none",
                edgecolors="#ff2e00",
                linewidths=0.9,
                label="nearest 100",
            )
            ax.scatter(
                [info.target[a]],
                [info.target[b]],
                marker="x",
                s=90,
                color="black",
                linewidths=2.0,
                label="target",
            )
            ax.set_title(f"{method.label}: z{a} vs z{b}")
            ax.set_xlabel(f"z{a}")
            ax.set_ylabel(f"z{b}")
            ax.grid(alpha=0.20)
            ax.legend(fontsize=7.5, loc="best")
    cbar = fig.colorbar(scat, ax=axes, pad=0.01, fraction=0.025)
    cbar.set_label("GT state label")
    png = out_dir / "oracle_noreg_vs_gtpc_covnoise_embeddings.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "oracle_noreg_vs_gtpc_covnoise_embeddings.pdf")
    plt.close(fig)
    return png


def _ellipse_from_covariance(
    center: np.ndarray,
    covariance_2d: np.ndarray,
    *,
    nsigma: float,
    color: str,
    alpha: float,
    lw: float,
) -> Ellipse:
    values, vectors = np.linalg.eigh(0.5 * (covariance_2d + covariance_2d.T))
    values = np.maximum(values, 0.0)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2.0 * nsigma * np.sqrt(values)
    return Ellipse(
        xy=(float(center[0]), float(center[1])),
        width=float(width),
        height=float(height),
        angle=angle,
        fill=False,
        edgecolor=color,
        alpha=alpha,
        lw=lw,
    )


def _state_means(z: np.ndarray, states: np.ndarray, n_states: int = 100) -> np.ndarray:
    means = np.empty((n_states, z.shape[1]), dtype=np.float64)
    for state in range(n_states):
        means[state] = np.mean(z[states == state], axis=0)
    return means


def plot_gt_overlay_embeddings(out_dir: Path, run_dir: Path, loaded: list[dict[str, object]]) -> Path:
    """Show observed embeddings, observed state means, and raw GT-PC states."""
    states = np.asarray(np.load(run_dir / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(20260528)
    n_show = min(120_000, states.size)
    show = rng.choice(states.size, size=n_show, replace=False) if states.size > n_show else np.arange(states.size)

    fig, axes = plt.subplots(2, len(loaded), figsize=(15.4, 10.0), constrained_layout=True)
    fig.suptitle(
        "Observed embeddings with raw GT-PC state coordinates overlaid | zdim4 no-reg",
        fontweight="bold",
    )
    if len(loaded) == 1:
        axes = axes[:, None]

    scat = None
    for col, item in enumerate(loaded):
        method: Method = item["method"]  # type: ignore[assignment]
        compute_root: Path = item["compute_root"]  # type: ignore[assignment]
        info = parse_compute_state_inputs(compute_root)
        z = _load_unsorted_embedding(info.pipeline, info.coords_entry, info.zdim)[:, : info.zdim]
        gt_z = raw_gt_pc_coordinates(run_dir, info.pipeline, info.zdim)
        means = _state_means(z, states)
        _, nearest, _, _ = state_weights_from_compute_state_nearest(run_dir, compute_root, 100)
        for row, (a, b) in enumerate(((0, 1), (2, 3))):
            ax = axes[row, col]
            scat = ax.scatter(
                z[show, a],
                z[show, b],
                c=states[show],
                s=1.1,
                alpha=0.16,
                cmap="viridis",
                rasterized=True,
            )
            ax.plot(gt_z[:, a], gt_z[:, b], color="black", lw=2.3, label="raw GT-PC state path")
            ax.scatter(gt_z[:, a], gt_z[:, b], c=np.arange(gt_z.shape[0]), s=20, cmap="viridis", edgecolor="black", lw=0.25)
            ax.plot(means[:, a], means[:, b], color="#ff7f0e", lw=1.7, alpha=0.95, label="mean observed z | GT state")
            ax.scatter(
                means[:, a],
                means[:, b],
                c=np.arange(means.shape[0]),
                s=18,
                cmap="plasma",
                edgecolor="none",
                alpha=0.85,
            )
            ax.scatter(
                z[nearest, a],
                z[nearest, b],
                s=28,
                facecolors="none",
                edgecolors="#ff2e00",
                linewidths=0.9,
                label="nearest 100 by cov-dist",
            )
            ax.scatter([info.target[a]], [info.target[b]], marker="x", color="white", edgecolor="black", s=110, lw=2.2, label="compute_state target")
            ax.set_title(f"{method.label}: z{a} vs z{b}")
            ax.set_xlabel(f"z{a}")
            ax.set_ylabel(f"z{b}")
            ax.grid(alpha=0.20)
            ax.legend(fontsize=7.2, loc="best")
    if scat is not None:
        cbar = fig.colorbar(scat, ax=axes, pad=0.012, fraction=0.025)
        cbar.set_label("particle GT state label")
    png = out_dir / "oracle_noreg_vs_gtpc_covnoise_gt_overlay_embeddings.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "oracle_noreg_vs_gtpc_covnoise_gt_overlay_embeddings.pdf")
    plt.close(fig)
    return png


def plot_covariance_ellipses(out_dir: Path, run_dir: Path, loaded: list[dict[str, object]]) -> Path:
    """Overlay 1-sigma covariance ellipses for representative nearest particles."""
    states = np.asarray(np.load(run_dir / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    fig, axes = plt.subplots(2, len(loaded), figsize=(15.4, 10.0), constrained_layout=True)
    fig.suptitle(
        "Per-particle latent covariance ellipses | selected nearest particles | 1 sigma",
        fontweight="bold",
    )
    if len(loaded) == 1:
        axes = axes[:, None]

    for col, item in enumerate(loaded):
        method: Method = item["method"]  # type: ignore[assignment]
        compute_root: Path = item["compute_root"]  # type: ignore[assignment]
        info = parse_compute_state_inputs(compute_root)
        z = _load_unsorted_embedding(info.pipeline, info.coords_entry, info.zdim)[:, : info.zdim]
        precision = _load_unsorted_embedding(info.pipeline, info.precision_entry, info.zdim)[:, : info.zdim, : info.zdim]
        gt_z = raw_gt_pc_coordinates(run_dir, info.pipeline, info.zdim)
        _, nearest, distances, _ = state_weights_from_compute_state_nearest(run_dir, compute_root, 100)
        nearest = nearest[np.argsort(distances[nearest])]
        selected = nearest[:12]
        covariances = np.linalg.inv(precision[selected])
        for row, (a, b) in enumerate(((0, 1), (2, 3))):
            ax = axes[row, col]
            ax.scatter(z[nearest, a], z[nearest, b], c=states[nearest], cmap="viridis", s=20, alpha=0.35, label="nearest 100")
            ax.scatter(z[selected, a], z[selected, b], c=states[selected], cmap="viridis", s=45, edgecolor="black", lw=0.4, label="ellipse subset")
            ax.plot(gt_z[:, a], gt_z[:, b], color="black", lw=2.0, label="raw GT-PC state path")
            ax.scatter([info.target[a]], [info.target[b]], marker="x", color="red", s=95, lw=2.0, label="target")
            for idx, covariance in enumerate(covariances):
                cov2 = covariance[np.ix_([a, b], [a, b])]
                ellipse = _ellipse_from_covariance(
                    z[selected[idx], [a, b]],
                    cov2,
                    nsigma=1.0,
                    color="#d62728",
                    alpha=0.62,
                    lw=1.1,
                )
                ax.add_patch(ellipse)
            ax.set_title(f"{method.label}: z{a} vs z{b}")
            ax.set_xlabel(f"z{a}")
            ax.set_ylabel(f"z{b}")
            ax.grid(alpha=0.20)
            ax.legend(fontsize=7.0, loc="best")
            pad = 0.08
            pts = np.vstack([z[nearest, :][:, [a, b]], gt_z[:, [a, b]], info.target[[a, b]][None, :]])
            span = np.ptp(pts, axis=0)
            lo = np.min(pts, axis=0) - pad * np.maximum(span, 1.0)
            hi = np.max(pts, axis=0) + pad * np.maximum(span, 1.0)
            ax.set_xlim(float(lo[0]), float(hi[0]))
            ax.set_ylim(float(lo[1]), float(hi[1]))
    png = out_dir / "oracle_noreg_vs_gtpc_covnoise_covariance_ellipses.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "oracle_noreg_vs_gtpc_covnoise_covariance_ellipses.pdf")
    plt.close(fig)
    return png


def write_summary(out_dir: Path, loaded: list[dict[str, object]], freq: np.ndarray) -> Path:
    path = out_dir / "oracle_noreg_vs_gtpc_covnoise_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "value"])
        for item in loaded:
            method: Method = item["method"]  # type: ignore[assignment]
            grid = item["grid"]  # type: ignore[assignment]
            final_fsc = item["final_fsc"]  # type: ignore[assignment]
            final_err = item["final_err"]  # type: ignore[assignment]
            best_fsc_idx = int(item["best_fsc_idx"])
            best_err_idx = int(item["best_err_idx"])
            writer.writerow([method.label, "compute_root", str(item["compute_root"])])
            writer.writerow([method.label, "final_fsc05_resolution_A", fsc05_resolution(freq, final_fsc)])
            writer.writerow([method.label, "final_relerr10_resolution_A", relerr_resolution(freq, final_err, 0.1)])
            writer.writerow([method.label, "best_fsc_candidate_1based", best_fsc_idx + 1])
            writer.writerow([method.label, "best_fsc_candidate_h", float(grid[best_fsc_idx])])
            writer.writerow([method.label, "best_fsc_candidate_fsc05_resolution_A", item["best_fsc_resolution_A"]])
            writer.writerow([method.label, "best_err_candidate_1based", best_err_idx + 1])
            writer.writerow([method.label, "best_err_candidate_h", float(grid[best_err_idx])])
            writer.writerow([method.label, "best_err_candidate_relerr10_resolution_A", item["best_err_resolution_A"]])
            for n_nearest, control in item["nearest"].items():  # type: ignore[union-attr]
                writer.writerow([method.label, f"nearest{n_nearest}_gt_fsc05_resolution_A", control["fsc05_resolution_A"]])
                writer.writerow([method.label, f"nearest{n_nearest}_gt_relerr10_resolution_A", control["relerr10_resolution_A"]])
                writer.writerow([method.label, f"nearest{n_nearest}_gt_state50_fraction", control["state50_fraction"]])
                writer.writerow([method.label, f"nearest{n_nearest}_gt_top_states", control["top_states"]])
                writer.writerow([method.label, f"nearest{n_nearest}_distance_source", "recomputed_from_compute_state_inputs"])
                writer.writerow([method.label, f"nearest{n_nearest}_coords_entry", control["coords_entry"]])
                writer.writerow([method.label, f"nearest{n_nearest}_precision_entry", control["precision_entry"]])
    return path


def write_curve_csv(out_dir: Path, loaded: list[dict[str, object]], freq: np.ndarray) -> tuple[Path, Path]:
    fsc_rows: dict[str, np.ndarray] = {}
    err_rows: dict[str, np.ndarray] = {}
    for item in loaded:
        method: Method = item["method"]  # type: ignore[assignment]
        fsc_rows[f"{method.key}_final_fsc"] = item["final_fsc"]  # type: ignore[assignment]
        err_rows[f"{method.key}_final_relative_error"] = item["final_err"]  # type: ignore[assignment]
        best_fsc_idx = int(item["best_fsc_idx"])
        best_err_idx = int(item["best_err_idx"])
        fsc_rows[f"{method.key}_best_candidate_fsc"] = item["candidate_fsc"][best_fsc_idx]  # type: ignore[index]
        err_rows[f"{method.key}_best_candidate_relative_error"] = item["candidate_err"][best_err_idx]  # type: ignore[index]
        for n_nearest, control in item["nearest"].items():  # type: ignore[union-attr]
            fsc_rows[f"{method.key}_nearest{n_nearest}_gt_fsc"] = control["fsc"]
            err_rows[f"{method.key}_nearest{n_nearest}_gt_relative_error"] = control["err"]
    fsc_path = out_dir / "oracle_noreg_vs_gtpc_covnoise_fsc_curves.csv"
    err_path = out_dir / "oracle_noreg_vs_gtpc_covnoise_relative_error_curves.csv"
    write_curves(fsc_path, freq, fsc_rows)
    write_curves(err_path, freq, err_rows)
    return fsc_path, err_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--nearest-counts", default="100,1000")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nearest_counts = tuple(int(part) for part in args.nearest_counts.split(",") if part)
    target = load_mrc(args.run_dir / "04_ground_truth" / f"gt_vol{TARGET_STATE:04d}.mrc")
    mask = np.clip(load_mrc(args.mask), 0.0, 1.0)
    labels, n_shells, target_ft, target_power, freq = metric_context(target, mask)
    mean_fsc, mean_err = masked_metrics(
        distribution_mean_gt_mrc(args.run_dir),
        mask,
        labels,
        n_shells,
        target_ft,
        target_power,
    )

    loaded = [
        load_method(
            args.run_dir,
            method,
            mask,
            labels,
            n_shells,
            target_ft,
            target_power,
            freq,
            args.batch_size,
            nearest_counts,
        )
        for method in METHODS
    ]

    comparison_plot = plot_comparison(args.out_dir, freq, loaded, mean_fsc, mean_err)
    all50_plot = plot_all_candidates(args.out_dir, freq, loaded)
    embedding_plot = plot_embeddings(args.out_dir, args.run_dir, loaded)
    gt_overlay_plot = plot_gt_overlay_embeddings(args.out_dir, args.run_dir, loaded)
    covariance_plot = plot_covariance_ellipses(args.out_dir, args.run_dir, loaded)
    summary = write_summary(args.out_dir, loaded, freq)
    fsc_csv, err_csv = write_curve_csv(args.out_dir, loaded, freq)
    print(f"PLOT {comparison_plot}")
    print(f"ALL50 {all50_plot}")
    print(f"EMBEDDINGS {embedding_plot}")
    print(f"GT_OVERLAY {gt_overlay_plot}")
    print(f"COVARIANCE {covariance_plot}")
    print(f"SUMMARY {summary}")
    print(f"FSC_CSV {fsc_csv}")
    print(f"ERR_CSV {err_csv}")


if __name__ == "__main__":
    main()
