#!/usr/bin/env python3
"""Audit latent-distance rankings used by two compute_state outputs."""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gt_embedding_controls import parse_compute_state_inputs, recompute_latent_distances_from_compute_state


DEFAULT_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518/"
    "n03000000/runs/n03000000_seed0000"
)
DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "oracle_noreg_vs_gtpc_covnoise_repro_20260528/distance_rank_audit"
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


def diagnostics_dir(compute_root: Path) -> Path:
    return compute_root / "diagnostics" / "state000"


def load_saved_distances(compute_root: Path) -> np.ndarray:
    path = diagnostics_dir(compute_root) / "heterogeneity_distances.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)


def load_bins(compute_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = diagnostics_dir(compute_root) / "params.pkl"
    with path.open("rb") as handle:
        params = pickle.load(handle)
    return np.asarray(params["heterogeneity_bins"], dtype=np.float64), np.asarray(params["n_images_per_bin"], dtype=np.int64)


def rank_vector(distances: np.ndarray) -> np.ndarray:
    order = np.argsort(distances, kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=np.int32)
    ranks[order] = np.arange(order.shape[0], dtype=np.int32)
    return ranks


def state_counts(states: np.ndarray, selection: np.ndarray, n_states: int = 100) -> np.ndarray:
    return np.bincount(states[selection], minlength=n_states).astype(np.int64)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_saved_vs_recomputed(out_dir: Path, methods: tuple[Method, ...], saved: dict[str, np.ndarray], recomputed: dict[str, np.ndarray]) -> Path:
    fig, axes = plt.subplots(1, len(methods), figsize=(7.0 * len(methods), 5.4), constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]
    rng = np.random.default_rng(20260528)
    for ax, method in zip(axes, methods):
        x = recomputed[method.key]
        y = saved[method.key]
        show = rng.choice(x.size, size=min(120_000, x.size), replace=False)
        ax.scatter(x[show], y[show], s=1.0, alpha=0.12, color=method.color, rasterized=True)
        lo = float(np.nanquantile(np.r_[x[show], y[show]], 0.001))
        hi = float(np.nanquantile(np.r_[x[show], y[show]], 0.999))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(method.label)
        ax.set_xlabel("recomputed distance from command inputs")
        ax.set_ylabel("saved heterogeneity_distances.txt")
        ax.grid(alpha=0.25)
    path = out_dir / "saved_vs_recomputed_distances.png"
    fig.savefig(path, dpi=180)
    fig.savefig(out_dir / "saved_vs_recomputed_distances.pdf")
    plt.close(fig)
    return path


def plot_distance_rank_scatter(
    out_dir: Path,
    source: np.ndarray,
    gtpc: np.ndarray,
    source_rank: np.ndarray,
    gtpc_rank: np.ndarray,
    states: np.ndarray,
) -> Path:
    rng = np.random.default_rng(20260528)
    show = rng.choice(source.size, size=min(180_000, source.size), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.0), constrained_layout=True)
    fig.suptitle("Distance and rank comparison | source oracle no-reg vs GT-PC cov-noise no-reg", fontweight="bold")

    eps = 1e-12
    axes[0, 0].scatter(source[show], gtpc[show], c=states[show], cmap="viridis", s=1.0, alpha=0.16, rasterized=True)
    lo = float(np.nanquantile(np.r_[source[show], gtpc[show]], 0.001))
    hi = float(np.nanquantile(np.r_[source[show], gtpc[show]], 0.999))
    axes[0, 0].plot([lo, hi], [lo, hi], "k--", lw=1.0)
    axes[0, 0].set_xlim(lo, hi)
    axes[0, 0].set_ylim(lo, hi)
    axes[0, 0].set_xlabel("source-oracle distance")
    axes[0, 0].set_ylabel("GT-PC cov-noise distance")
    axes[0, 0].set_title("Raw distances, colored by GT state")

    axes[0, 1].scatter(np.log10(source[show] + eps), np.log10(gtpc[show] + eps), c=states[show], cmap="viridis", s=1.0, alpha=0.16, rasterized=True)
    axes[0, 1].set_xlabel("log10 source-oracle distance")
    axes[0, 1].set_ylabel("log10 GT-PC cov-noise distance")
    axes[0, 1].set_title("Log distance scatter")

    axes[1, 0].scatter(source_rank[show], gtpc_rank[show], c=states[show], cmap="viridis", s=1.0, alpha=0.16, rasterized=True)
    lim = source.size
    axes[1, 0].plot([0, lim], [0, lim], "k--", lw=1.0)
    axes[1, 0].set_xlim(0, lim)
    axes[1, 0].set_ylim(0, lim)
    axes[1, 0].set_xlabel("source-oracle rank, lower is nearer")
    axes[1, 0].set_ylabel("GT-PC cov-noise rank, lower is nearer")
    axes[1, 0].set_title("Global nearest-neighbor ranking")

    max_rank = 200_000
    near = (source_rank < max_rank) | (gtpc_rank < max_rank)
    near_idx = np.flatnonzero(near)
    if near_idx.size > 160_000:
        near_idx = rng.choice(near_idx, size=160_000, replace=False)
    axes[1, 1].scatter(source_rank[near_idx], gtpc_rank[near_idx], c=states[near_idx], cmap="viridis", s=2.0, alpha=0.22, rasterized=True)
    axes[1, 1].plot([0, max_rank], [0, max_rank], "k--", lw=1.0)
    axes[1, 1].set_xlim(0, max_rank)
    axes[1, 1].set_ylim(0, max_rank)
    axes[1, 1].set_xlabel("source-oracle rank")
    axes[1, 1].set_ylabel("GT-PC cov-noise rank")
    axes[1, 1].set_title(f"Zoom to top {max_rank:,} by either ranking")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    path = out_dir / "source_vs_gtpc_distance_and_rank_scatter.png"
    fig.savefig(path, dpi=180)
    fig.savefig(out_dir / "source_vs_gtpc_distance_and_rank_scatter.pdf")
    plt.close(fig)
    return path


def plot_nearest_overlap(out_dir: Path, overlap_rows: list[dict[str, object]]) -> Path:
    k = np.asarray([row["k"] for row in overlap_rows], dtype=np.float64)
    overlap_frac = np.asarray([row["overlap_fraction_of_k"] for row in overlap_rows], dtype=np.float64)
    jaccard = np.asarray([row["jaccard"] for row in overlap_rows], dtype=np.float64)
    state50_source = np.asarray([row["source_state50_fraction"] for row in overlap_rows], dtype=np.float64)
    state50_gtpc = np.asarray([row["gtpc_state50_fraction"] for row in overlap_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
    fig.suptitle("Nearest-particle set agreement by distance ranking", fontweight="bold")
    axes[0].semilogx(k, overlap_frac, marker="o", color="#4c78a8", label="|intersection| / k")
    axes[0].semilogx(k, jaccard, marker="s", color="#f58518", label="Jaccard")
    axes[0].semilogx(k, k / 3_000_000.0, color="0.5", ls="--", label="random expected |intersection|/k")
    axes[0].set_xlabel("k nearest images")
    axes[0].set_ylabel("set agreement")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()

    axes[1].semilogx(k, state50_source, marker="o", color="#1f77b4", label="source nearest-k state50 fraction")
    axes[1].semilogx(k, state50_gtpc, marker="o", color="#d62728", label="GT-PC nearest-k state50 fraction")
    axes[1].axhline(0.01, color="0.45", ls="--", label="uniform 1/100")
    axes[1].set_xlabel("k nearest images")
    axes[1].set_ylabel("fraction with GT state 50")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    path = out_dir / "nearest_set_overlap_vs_k.png"
    fig.savefig(path, dpi=180)
    fig.savefig(out_dir / "nearest_set_overlap_vs_k.pdf")
    plt.close(fig)
    return path


def plot_state_composition(out_dir: Path, states: np.ndarray, nearest_source: dict[int, np.ndarray], nearest_gtpc: dict[int, np.ndarray]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 9.5), constrained_layout=True)
    fig.suptitle("GT state composition of nearest sets", fontweight="bold")
    for ax, k in zip(axes.flat, (100, 1000, 10_000, 100_000)):
        cs = state_counts(states, nearest_source[k])
        cg = state_counts(states, nearest_gtpc[k])
        x = np.arange(cs.size)
        ax.plot(x, cs / cs.sum(), color="#1f77b4", lw=1.8, marker="o", ms=2.4, label="source oracle")
        ax.plot(x, cg / cg.sum(), color="#d62728", lw=1.8, marker="o", ms=2.4, label="GT-PC cov-noise")
        ax.axvline(50, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_title(f"nearest {k:,} images")
        ax.set_xlabel("GT state label")
        ax.set_ylabel("fraction")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    path = out_dir / "nearest_state_composition.png"
    fig.savefig(path, dpi=180)
    fig.savefig(out_dir / "nearest_state_composition.pdf")
    plt.close(fig)
    return path


def plot_candidate_bin_audit(out_dir: Path, rows: list[dict[str, object]]) -> Path:
    idx = np.asarray([row["candidate"] for row in rows], dtype=np.int32)
    overlap = np.asarray([row["overlap_fraction_source"] for row in rows], dtype=np.float64)
    src_state50 = np.asarray([row["source_state50_fraction"] for row in rows], dtype=np.float64)
    gt_state50 = np.asarray([row["gtpc_state50_fraction"] for row in rows], dtype=np.float64)
    src_count = np.asarray([row["source_count"] for row in rows], dtype=np.float64)
    gt_count = np.asarray([row["gtpc_count"] for row in rows], dtype=np.float64)
    src_mean_abs_label = np.asarray([row["source_mean_abs_label_minus_50"] for row in rows], dtype=np.float64)
    gt_mean_abs_label = np.asarray([row["gtpc_mean_abs_label_minus_50"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.0), constrained_layout=True)
    fig.suptitle("Candidate-bin identity and GT-label composition", fontweight="bold")
    axes[0, 0].plot(idx, overlap, color="#4c78a8", lw=1.8, marker="o", ms=3)
    axes[0, 0].set_title("Overlap of source candidate set with GT-PC candidate set")
    axes[0, 0].set_ylabel("|source set ∩ GT-PC set| / |source set|")

    axes[0, 1].plot(idx, src_count, color="#1f77b4", lw=1.8, marker="o", ms=3, label="source")
    axes[0, 1].plot(idx, gt_count, color="#d62728", lw=1.8, marker="o", ms=3, label="GT-PC")
    axes[0, 1].set_title("Number of images in candidate set")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].legend()

    axes[1, 0].plot(idx, src_state50, color="#1f77b4", lw=1.8, marker="o", ms=3, label="source")
    axes[1, 0].plot(idx, gt_state50, color="#d62728", lw=1.8, marker="o", ms=3, label="GT-PC")
    axes[1, 0].axhline(0.01, color="0.45", ls="--", lw=1.0)
    axes[1, 0].set_title("Fraction of candidate images with GT state 50")
    axes[1, 0].set_ylabel("state50 fraction")
    axes[1, 0].legend()

    axes[1, 1].plot(idx, src_mean_abs_label, color="#1f77b4", lw=1.8, marker="o", ms=3, label="source")
    axes[1, 1].plot(idx, gt_mean_abs_label, color="#d62728", lw=1.8, marker="o", ms=3, label="GT-PC")
    axes[1, 1].set_title("Candidate label spread around target state 50")
    axes[1, 1].set_ylabel("mean |state - 50|")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xlabel("candidate index")
        ax.grid(alpha=0.25)
    path = out_dir / "candidate_bin_overlap_and_label_composition.png"
    fig.savefig(path, dpi=180)
    fig.savefig(out_dir / "candidate_bin_overlap_and_label_composition.pdf")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    states = np.asarray(np.load(args.run_dir / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)

    saved: dict[str, np.ndarray] = {}
    recomputed: dict[str, np.ndarray] = {}
    ranks: dict[str, np.ndarray] = {}
    orders: dict[str, np.ndarray] = {}
    bins: dict[str, np.ndarray] = {}
    bin_counts: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []

    for method in METHODS:
        compute_root = args.run_dir / method.compute_relpath
        info = parse_compute_state_inputs(compute_root)
        saved[method.key] = load_saved_distances(compute_root)
        recomputed[method.key], recompute_info = recompute_latent_distances_from_compute_state(compute_root)
        if (
            info.pipeline != recompute_info.pipeline
            or info.target_path != recompute_info.target_path
            or not np.allclose(info.target, recompute_info.target)
            or info.zdim != recompute_info.zdim
            or info.coords_entry != recompute_info.coords_entry
            or info.precision_entry != recompute_info.precision_entry
            or info.embedding_option != recompute_info.embedding_option
        ):
            raise RuntimeError(f"parse mismatch for {method.key}: {info} != {recompute_info}")
        bins[method.key], bin_counts[method.key] = load_bins(compute_root)
        ranks[method.key] = rank_vector(recomputed[method.key])
        orders[method.key] = np.argsort(recomputed[method.key], kind="mergesort")

        diff = saved[method.key] - recomputed[method.key]
        summary_rows.append(
            {
                "method": method.key,
                "label": method.label,
                "pipeline": str(info.pipeline),
                "target": " ".join(f"{x:.8g}" for x in info.target),
                "coords_entry": info.coords_entry,
                "precision_entry": info.precision_entry,
                "embedding_option": info.embedding_option,
                "saved_vs_recomputed_max_abs": float(np.nanmax(np.abs(diff))),
                "saved_vs_recomputed_median_abs": float(np.nanmedian(np.abs(diff))),
                "distance_min": float(np.nanmin(recomputed[method.key])),
                "distance_p001": float(np.nanquantile(recomputed[method.key], 0.001)),
                "distance_p01": float(np.nanquantile(recomputed[method.key], 0.01)),
                "distance_median": float(np.nanmedian(recomputed[method.key])),
                "distance_p99": float(np.nanquantile(recomputed[method.key], 0.99)),
            }
        )

    source_key, gtpc_key = METHODS[0].key, METHODS[1].key
    source = recomputed[source_key]
    gtpc = recomputed[gtpc_key]
    source_rank = ranks[source_key]
    gtpc_rank = ranks[gtpc_key]
    rank_corr = float(np.corrcoef(source_rank.astype(np.float64), gtpc_rank.astype(np.float64))[0, 1])
    dist_corr = float(np.corrcoef(source, gtpc)[0, 1])

    nearest_source: dict[int, np.ndarray] = {}
    nearest_gtpc: dict[int, np.ndarray] = {}
    overlap_rows: list[dict[str, object]] = []
    for k in (100, 1_000, 10_000, 100_000, 300_000, 1_000_000, 2_000_000):
        k = min(k, source.size)
        nearest_source[k] = orders[source_key][:k]
        nearest_gtpc[k] = orders[gtpc_key][:k]
        sset = set(nearest_source[k].tolist())
        gset = set(nearest_gtpc[k].tolist())
        overlap = len(sset & gset)
        cs = state_counts(states, nearest_source[k])
        cg = state_counts(states, nearest_gtpc[k])
        overlap_rows.append(
            {
                "k": int(k),
                "overlap_count": int(overlap),
                "overlap_fraction_of_k": float(overlap / k),
                "jaccard": float(overlap / (2 * k - overlap)),
                "random_expected_overlap_fraction": float(k / source.size),
                "source_state50_fraction": float(cs[50] / cs.sum()),
                "gtpc_state50_fraction": float(cg[50] / cg.sum()),
                "source_mean_abs_label_minus_50": float(np.average(np.abs(np.arange(cs.size) - 50), weights=cs)),
                "gtpc_mean_abs_label_minus_50": float(np.average(np.abs(np.arange(cg.size) - 50), weights=cg)),
            }
        )

    candidate_rows: list[dict[str, object]] = []
    for i in range(len(bins[source_key])):
        src_sel = np.flatnonzero(source <= bins[source_key][i])
        gt_sel = np.flatnonzero(gtpc <= bins[gtpc_key][i])
        src_set = set(src_sel.tolist())
        gt_set = set(gt_sel.tolist())
        overlap = len(src_set & gt_set)
        cs = state_counts(states, src_sel)
        cg = state_counts(states, gt_sel)
        candidate_rows.append(
            {
                "candidate": i + 1,
                "source_h": float(bins[source_key][i]),
                "gtpc_h": float(bins[gtpc_key][i]),
                "source_count": int(src_sel.size),
                "gtpc_count": int(gt_sel.size),
                "source_saved_count": int(bin_counts[source_key][i]),
                "gtpc_saved_count": int(bin_counts[gtpc_key][i]),
                "overlap_count": int(overlap),
                "overlap_fraction_source": float(overlap / max(src_sel.size, 1)),
                "jaccard": float(overlap / max(src_sel.size + gt_sel.size - overlap, 1)),
                "source_state50_fraction": float(cs[50] / max(cs.sum(), 1)),
                "gtpc_state50_fraction": float(cg[50] / max(cg.sum(), 1)),
                "source_mean_abs_label_minus_50": float(np.average(np.abs(np.arange(cs.size) - 50), weights=cs)) if cs.sum() else np.nan,
                "gtpc_mean_abs_label_minus_50": float(np.average(np.abs(np.arange(cg.size) - 50), weights=cg)) if cg.sum() else np.nan,
            }
        )

    write_csv(args.out_dir / "distance_audit_summary.csv", summary_rows)
    write_csv(args.out_dir / "nearest_overlap_vs_k.csv", overlap_rows)
    write_csv(args.out_dir / "candidate_bin_overlap_and_label_composition.csv", candidate_rows)

    paths = [
        plot_saved_vs_recomputed(args.out_dir, METHODS, saved, recomputed),
        plot_distance_rank_scatter(args.out_dir, source, gtpc, source_rank, gtpc_rank, states),
        plot_nearest_overlap(args.out_dir, overlap_rows),
        plot_state_composition(args.out_dir, states, nearest_source, nearest_gtpc),
        plot_candidate_bin_audit(args.out_dir, candidate_rows),
    ]

    print("distance_corr", dist_corr)
    print("rank_corr", rank_corr)
    for row in summary_rows:
        print(row)
    for row in overlap_rows:
        print(row)
    print("PLOTS")
    for path in paths:
        print(path)
    print("CSVS")
    print(args.out_dir / "distance_audit_summary.csv")
    print(args.out_dir / "nearest_overlap_vs_k.csv")
    print(args.out_dir / "candidate_bin_overlap_and_label_composition.csv")


if __name__ == "__main__":
    main()
