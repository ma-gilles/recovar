#!/usr/bin/env python3
"""Compare 1M/noise30 real, source-oracle, and GT-PC cov-noise embeddings/PCs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils


RUN_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518/"
    "n01000000/runs/n01000000_seed0000"
)
OUT_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "noise30_1m_pipeline_vs_oracle_20260528"
)
N_PCS = 4


@dataclass(frozen=True)
class Method:
    label: str
    short: str
    pipeline_relpath: str


METHODS = (
    Method(
        label="real pipeline zdim4/reg",
        short="pipeline",
        pipeline_relpath="06_pipeline_true_recovar_h100_fullmem_movingfocus_20260527",
    ),
    Method(
        label="source oracle zdim4/reg",
        short="source-oracle",
        pipeline_relpath="06_pipeline_oracle_regfix_20260526",
    ),
    Method(
        label="GT-PC + oracle-cov noise zdim4/reg",
        short="gtpc-covnoise",
        pipeline_relpath="06_pipeline_gtpc_covnoise_trueunits_zdim4_seed20260527",
    ),
)


def load_mrc(path: Path) -> np.ndarray:
    return np.asarray(utils.load_mrc(path), dtype=np.float32)


def load_latent_coords(root: Path) -> np.ndarray:
    return np.asarray(np.load(root / "model" / "zdim_4" / "latent_coords.npy"), dtype=np.float32)


def load_pc_volumes_flat(root: Path, mask: np.ndarray | None) -> np.ndarray:
    pcs = []
    for idx in range(N_PCS):
        vol = load_mrc(root / "output" / "volumes" / f"eigen_pos{idx:04d}.mrc")
        if mask is not None:
            vol = vol * mask
        pcs.append(vol.reshape(-1).astype(np.float64))
    x = np.stack(pcs, axis=1)
    x /= np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-30)
    return x


def load_pc_images(root: Path, mask: np.ndarray | None) -> np.ndarray:
    vols = []
    for idx in range(N_PCS):
        vol = load_mrc(root / "output" / "volumes" / f"eigen_pos{idx:04d}.mrc")
        if mask is not None:
            vol = vol * mask
        vols.append(vol)
    return np.stack(vols, axis=0)


def load_assignments() -> np.ndarray:
    return np.asarray(np.load(RUN_DIR / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)


def fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.concatenate([source, np.ones((source.shape[0], 1), dtype=source.dtype)], axis=1)
    coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    return design @ coeff


def r2_score(target: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    centered = target - np.mean(target, axis=0, keepdims=True)
    ss_tot = np.sum(centered**2)
    return float(1.0 - ss_res / max(float(ss_tot), 1e-30))


def state_means(z: np.ndarray, states: np.ndarray) -> np.ndarray:
    out = []
    for state in range(100):
        keep = states == state
        out.append(np.mean(z[keep], axis=0))
    return np.asarray(out, dtype=np.float64)


def pair_distance_corr(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    n = a.shape[0]
    idx_i = rng.integers(0, n, size=50000)
    idx_j = rng.integers(0, n, size=50000)
    keep = idx_i != idx_j
    da = np.linalg.norm(a[idx_i[keep]] - a[idx_j[keep]], axis=1)
    db = np.linalg.norm(b[idx_i[keep]] - b[idx_j[keep]], axis=1)
    if np.std(da) == 0 or np.std(db) == 0:
        return float("nan")
    return float(np.corrcoef(da, db)[0, 1])


def plot_embedding_panel(
    coords: dict[str, np.ndarray],
    states: np.ndarray,
    out_dir: Path,
    dims: tuple[int, int],
    aligned_to_source: bool,
    zscore_each: bool = False,
) -> Path:
    rng = np.random.default_rng(0)
    finite = states >= 0
    idx = np.flatnonzero(finite)
    if idx.size > 60000:
        idx = rng.choice(idx, size=60000, replace=False)

    source = coords["source-oracle"]
    plot_coords = {}
    for method in METHODS:
        z = coords[method.short]
        if aligned_to_source and method.short != "source-oracle":
            z = fit_affine(z, source)
        if zscore_each:
            z = (z - np.mean(z, axis=0, keepdims=True)) / np.maximum(np.std(z, axis=0, keepdims=True), 1e-30)
        plot_coords[method.short] = z

    fig, axes = plt.subplots(1, len(METHODS), figsize=(15.0, 4.8), constrained_layout=True)
    for ax, method in zip(axes, METHODS, strict=True):
        z = plot_coords[method.short]
        sc = ax.scatter(
            z[idx, dims[0]],
            z[idx, dims[1]],
            c=states[idx],
            s=1.1,
            alpha=0.35,
            cmap="viridis",
            linewidths=0,
        )
        ax.set_title(method.label)
        ax.set_xlabel(f"z{dims[0]}")
        ax.set_ylabel(f"z{dims[1]}")
        ax.grid(alpha=0.2)
    if zscore_each:
        mode = "zscored-each-embedding"
    else:
        mode = "aligned-to-source" if aligned_to_source else "native"
    fig.suptitle(f"1M noise=30 embeddings | {mode} | colored by GT state")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01, label="GT state index")
    path = out_dir / f"embedding_threeway_{mode}_z{dims[0]}_z{dims[1]}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_state_means(coords: dict[str, np.ndarray], states: np.ndarray, out_dir: Path) -> Path:
    source = coords["source-oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    colors = {"pipeline": "#1f77b4", "source-oracle": "#d62728", "gtpc-covnoise": "#2ca02c"}
    for method in METHODS:
        z = coords[method.short]
        aligned = fit_affine(z, source) if method.short != "source-oracle" else z
        means = state_means(aligned, states)
        for ax, dims in zip(axes, [(0, 1), (2, 3)], strict=True):
            ax.plot(
                means[:, dims[0]],
                means[:, dims[1]],
                marker="o",
                ms=2.0,
                lw=1.4,
                color=colors[method.short],
                label=method.label,
            )
            ax.scatter(means[50, dims[0]], means[50, dims[1]], s=60, color=colors[method.short], edgecolor="black")
            ax.set_xlabel(f"aligned z{dims[0]}")
            ax.set_ylabel(f"aligned z{dims[1]}")
            ax.grid(alpha=0.25)
    axes[0].set_title("state mean path, z0/z1")
    axes[1].set_title("state mean path, z2/z3")
    axes[0].legend(fontsize=8)
    fig.suptitle("State means after affine alignment to source-oracle coordinates")
    path = out_dir / "embedding_threeway_state_mean_paths_aligned_to_source.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pc_overlaps(pc_vectors: dict[str, np.ndarray], out_dir: Path, suffix: str) -> tuple[Path, Path]:
    pairs = [
        ("pipeline", "source-oracle"),
        ("gtpc-covnoise", "source-oracle"),
        ("pipeline", "gtpc-covnoise"),
    ]
    fig, axes = plt.subplots(1, len(pairs), figsize=(13.4, 4.2), constrained_layout=True)
    rows = []
    for ax, (a_name, b_name) in zip(axes, pairs, strict=True):
        overlap = np.abs(pc_vectors[a_name].T @ pc_vectors[b_name])
        im = ax.imshow(overlap, vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_title(f"{a_name} vs {b_name}")
        ax.set_xlabel(f"{b_name} PC")
        ax.set_ylabel(f"{a_name} PC")
        ax.set_xticks(range(N_PCS))
        ax.set_yticks(range(N_PCS))
        for i in range(N_PCS):
            for j in range(N_PCS):
                rows.append(
                    {
                        "comparison": f"{a_name}_vs_{b_name}",
                        "row_pc": i,
                        "col_pc": j,
                        "abs_overlap": float(overlap[i, j]),
                    }
                )
                ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.012, label="abs normalized dot product")
    fig.suptitle(f"eigen_pos output volume overlap ({suffix.replace('_', ' ')})")
    png = out_dir / f"eigen_pos_overlap_threeway_{suffix}.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    csv_path = out_dir / f"eigen_pos_overlap_threeway_{suffix}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison", "row_pc", "col_pc", "abs_overlap"])
        writer.writeheader()
        writer.writerows(rows)
    return png, csv_path


def plot_pc_images(pc_images: dict[str, np.ndarray], out_dir: Path, suffix: str) -> tuple[Path, Path]:
    z = pc_images[METHODS[0].short].shape[-1] // 2
    fig, axes = plt.subplots(len(METHODS), N_PCS, figsize=(12.5, 8.2), constrained_layout=True)
    for col in range(N_PCS):
        stack = np.concatenate([pc_images[m.short][col].reshape(1, -1) for m in METHODS], axis=1).reshape(-1)
        vmax = float(np.percentile(np.abs(stack), 99.7))
        vmax = max(vmax, 1e-8)
        for row, method in enumerate(METHODS):
            ax = axes[row, col]
            ax.imshow(pc_images[method.short][col, :, :, z], cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"PC {col}")
            if col == 0:
                ax.set_ylabel(method.short)
    fig.suptitle(f"eigen_pos central slices ({suffix.replace('_', ' ')})")
    central = out_dir / f"eigen_pos_threeway_central_slices_{suffix}.png"
    fig.savefig(central, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(METHODS), N_PCS, figsize=(12.5, 8.2), constrained_layout=True)
    for col in range(N_PCS):
        stack = np.concatenate([pc_images[m.short][col].reshape(1, -1) for m in METHODS], axis=1).reshape(-1)
        vmax = float(np.percentile(np.abs(stack), 99.7))
        vmax = max(vmax, 1e-8)
        for row, method in enumerate(METHODS):
            ax = axes[row, col]
            proj = np.max(np.abs(pc_images[method.short][col]), axis=2)
            ax.imshow(proj, cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"PC {col}")
            if col == 0:
                ax.set_ylabel(method.short)
    fig.suptitle(f"eigen_pos max-abs Z projections ({suffix.replace('_', ' ')})")
    proj_path = out_dir / f"eigen_pos_threeway_maxabs_zproj_{suffix}.png"
    fig.savefig(proj_path, dpi=180)
    plt.close(fig)
    return central, proj_path


def write_embedding_summary(coords: dict[str, np.ndarray], states: np.ndarray, out_dir: Path) -> Path:
    rng = np.random.default_rng(1)
    source = coords["source-oracle"]
    rows = []
    source_state = state_means(source, states)
    for method in METHODS:
        z = coords[method.short]
        pred = fit_affine(z, source) if method.short != "source-oracle" else source
        z_state = state_means(z, states)
        state_pred = fit_affine(z_state, source_state) if method.short != "source-oracle" else source_state
        rows.append(
            {
                "method": method.label,
                "coord_std": " ".join(f"{x:.6g}" for x in np.std(z, axis=0)),
                "affine_r2_to_source_oracle_images": r2_score(source, pred),
                "distance_corr_to_source_oracle_images": pair_distance_corr(z, source, rng),
                "affine_r2_to_source_oracle_state_means": r2_score(source_state, state_pred),
                "state50_mean": " ".join(f"{x:.6g}" for x in z_state[50]),
            }
        )
    path = out_dir / "embedding_threeway_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    states = load_assignments()
    roots = {m.short: RUN_DIR / m.pipeline_relpath for m in METHODS}
    coords = {m.short: load_latent_coords(roots[m.short]) for m in METHODS}

    for suffix, mask_path in [
        ("no_mask", None),
        ("moving_mask", RUN_DIR / "05_masks" / "focus_mask_moving.mrc"),
    ]:
        out_dir = OUT_ROOT / f"embedding_pc_three_way_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        mask = None if mask_path is None else np.clip(load_mrc(mask_path), 0.0, 1.0)

        written = [
            plot_embedding_panel(coords, states, out_dir, (0, 1), aligned_to_source=False),
            plot_embedding_panel(coords, states, out_dir, (2, 3), aligned_to_source=False),
            plot_embedding_panel(coords, states, out_dir, (0, 1), aligned_to_source=True),
            plot_embedding_panel(coords, states, out_dir, (2, 3), aligned_to_source=True),
            plot_embedding_panel(coords, states, out_dir, (0, 1), aligned_to_source=False, zscore_each=True),
            plot_embedding_panel(coords, states, out_dir, (2, 3), aligned_to_source=False, zscore_each=True),
            plot_state_means(coords, states, out_dir),
            write_embedding_summary(coords, states, out_dir),
        ]

        pc_vectors = {m.short: load_pc_volumes_flat(roots[m.short], mask) for m in METHODS}
        pc_images = {m.short: load_pc_images(roots[m.short], mask) for m in METHODS}
        overlap_png, overlap_csv = plot_pc_overlaps(pc_vectors, out_dir, suffix)
        central_png, proj_png = plot_pc_images(pc_images, out_dir, suffix)
        written.extend([overlap_png, overlap_csv, central_png, proj_png])

        print(f"OUT {out_dir}")
        for path in written:
            print(path)


if __name__ == "__main__":
    main()
