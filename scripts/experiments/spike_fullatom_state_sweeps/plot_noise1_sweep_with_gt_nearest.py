#!/usr/bin/env python3
"""Plot noise=1 dataset-size sweep metrics with nearest-particle GT controls.

This script is intentionally concrete: it reproduces the current full-atom
state-50, moving-mask comparison without relying on scratch-only helpers.
All estimate metrics use unfiltered compute_state candidate half-map averages.
GT mixtures are built from the per-state ``gt_volNNNN.mrc`` files so they are
in the same map convention as the FSC target.
"""

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

from recovar import utils
from recovar.core import fourier_transform_utils as ftu


DEFAULT_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise1_b80_true_oracle_sweep_20260527"
)
DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "noise1_sweep_gt_nearest_repro_20260528"
)
DEFAULT_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "state50_broad_motion_union_mask_20260524/masks/state50_broad_seed_mask.mrc"
)
DEFAULT_IMAGE_COUNTS = (30_000, 100_000, 300_000, 1_000_000)
TARGET_STATE = 50
VOXEL_SIZE_A = 1.25
FREQ_MAX = 0.40


@dataclass(frozen=True)
class MethodSpec:
    label: str
    short_label: str
    compute_relpath: str
    embedding_relpath: str
    target_point_name: str
    color: str


METHODS = (
    MethodSpec(
        label="pipeline zdim4 reg",
        short_label="pipeline",
        compute_relpath="07_compute_state_true_recovar_h100_fullmem_movingfocus_zdim4_reg_lazy",
        embedding_relpath="06_pipeline_true_recovar_h100_fullmem_movingfocus_20260527/model/zdim_4",
        target_point_name="target_latent_point_true_recovar_h100_fullmem_movingfocus_zdim4_reg_state0050.txt",
        color="#1f77b4",
    ),
    MethodSpec(
        label="source-oracle zdim4 reg",
        short_label="source-oracle",
        compute_relpath="07_compute_state_oracle_regfix_zdim4_reg_lazy",
        embedding_relpath="06_pipeline_oracle_regfix_20260527/model/zdim_4",
        target_point_name="target_latent_point_oracle_regfix_zdim4_reg_state0050.txt",
        color="#d62728",
    ),
)


def run_dir(root: Path, n_images: int) -> Path:
    label = f"n{n_images:08d}"
    return root / label / "runs" / f"{label}_seed0000"


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
    ).astype(np.float32)
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
    return diagnostics if (diagnostics / "params.pkl").exists() else compute_root


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
        raise FileNotFoundError(f"Missing unfiltered half-map candidates in {base}")
    return list(zip(half1, half2))


def load_half_average_batch(paths: list[tuple[Path, Path]]) -> np.ndarray:
    return np.stack([0.5 * (load_mrc(a) + load_mrc(b)) for a, b in paths], axis=0).astype(np.float32)


def candidate_metrics(
    compute_root: Path,
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    target_ft: np.ndarray,
    target_power: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp

    grid = candidate_grid(compute_root)
    paths = candidate_half_paths(compute_root)
    if len(grid) != len(paths):
        raise ValueError(f"Candidate grid/path mismatch for {compute_root}: {len(grid)} vs {len(paths)}")

    labels_j = jnp.asarray(labels.ravel(), dtype=jnp.int32)
    mask_j = jnp.asarray(mask, dtype=jnp.float32)
    target_ft_j = jnp.asarray(target_ft)
    target_power_j = jnp.asarray(target_power)

    def one_metrics(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        volume_ft = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.fftshift(volume * mask_j)))
        cross = jnp.bincount(
            labels_j,
            weights=jnp.real(jnp.conj(volume_ft).ravel() * target_ft_j.ravel()),
            length=n_shells,
        )
        volume_power = jnp.bincount(labels_j, weights=jnp.abs(volume_ft).ravel() ** 2, length=n_shells)
        diff_power = jnp.bincount(
            labels_j,
            weights=jnp.abs(volume_ft.ravel() - target_ft_j.ravel()) ** 2,
            length=n_shells,
        )
        fsc = cross / jnp.sqrt(volume_power * target_power_j)
        relerr = diff_power / jnp.maximum(target_power_j, 1e-30)
        fsc = jnp.where(jnp.isfinite(fsc), fsc, jnp.nan)
        relerr = jnp.where(jnp.isfinite(relerr), relerr, jnp.nan)
        fsc = fsc.at[0].set(fsc[1])
        return fsc, relerr

    batched = jax.jit(jax.vmap(one_metrics))
    fscs: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        stop = min(start + batch_size, len(paths))
        batch = load_half_average_batch(paths[start:stop])
        fsc_batch, err_batch = batched(jnp.asarray(batch))
        fscs.append(np.asarray(fsc_batch, dtype=np.float32))
        errors.append(np.asarray(err_batch, dtype=np.float32))
        print(f"  {compute_root.name}: candidates {stop}/{len(paths)}", flush=True)
    return grid, np.concatenate(fscs, axis=0), np.concatenate(errors, axis=0)


def gt_volume_paths(run: Path) -> list[Path]:
    paths = [run / "04_ground_truth" / f"gt_vol{i:04d}.mrc" for i in range(100)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GT MRC files; first missing: {missing[0]}")
    return paths


def state_weighted_gt_mrc(run: Path, weights: np.ndarray) -> np.ndarray:
    paths = gt_volume_paths(run)
    total = None
    for state, weight in enumerate(weights):
        if weight == 0:
            continue
        volume = load_mrc(paths[state])
        if total is None:
            total = np.zeros_like(volume, dtype=np.float64)
        total += float(weight) * volume
    if total is None:
        raise ValueError("All GT mixture weights are zero")
    return total.astype(np.float32)


def distribution_mean_gt_mrc(run: Path) -> np.ndarray:
    paths = gt_volume_paths(run)
    total = np.zeros_like(load_mrc(paths[0]), dtype=np.float64)
    for path in paths:
        total += load_mrc(path)
    return (total / len(paths)).astype(np.float32)


def nearest_gt_mixture(
    run: Path,
    method: MethodSpec,
    n_nearest: int,
) -> tuple[np.ndarray, dict[str, object]]:
    z_path = run / method.embedding_relpath / "latent_coords.npy"
    target_path = run / method.target_point_name
    states_path = run / "03_dataset" / "state_assignment.npy"
    z = np.asarray(np.load(z_path), dtype=np.float64)
    target = np.asarray(np.loadtxt(target_path), dtype=np.float64).reshape(-1)
    states = np.asarray(np.load(states_path), dtype=np.int64).reshape(-1)
    if z.shape[0] != states.size:
        raise ValueError(f"Embedding/state length mismatch for {run}: {z.shape[0]} vs {states.size}")
    if z.shape[1] < target.size:
        raise ValueError(f"Embedding dimension mismatch for {z_path}: {z.shape} vs target {target.shape}")
    dist2 = np.sum((z[:, : target.size] - target[None, :]) ** 2, axis=1)
    n = min(n_nearest, dist2.size)
    nearest = np.argpartition(dist2, n - 1)[:n] if n < dist2.size else np.arange(dist2.size)
    counts = np.bincount(states[nearest], minlength=100).astype(np.float64)
    weights = counts / counts.sum()
    top_states = [
        f"{int(state)}:{int(counts[state])}"
        for state in np.argsort(counts)[::-1][:12]
        if counts[state] > 0
    ]
    info = {
        "nearest_source": str(z_path),
        "target_point": str(target_path),
        "n_nearest": int(n),
        "nearest_radius": float(np.sqrt(np.max(dist2[nearest]))),
        "state50_fraction": float(weights[TARGET_STATE]),
        "top_states": " ".join(top_states),
    }
    return state_weighted_gt_mrc(run, weights), info


def write_curves_npz(path: Path, freq: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    safe = {name.replace(" ", "_").replace("/", "_"): curve for name, curve in curves.items()}
    np.savez_compressed(path, frequency_1_per_A=freq, **safe)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_resolution_summary(rows: list[dict[str, object]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.4), constrained_layout=True)
    fig.suptitle(
        "Noise=1 full-atom spike sweep | state 50 broad moving mask | unfiltered candidates",
        fontweight="bold",
    )
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method.label]
        if not method_rows:
            continue
        x = np.asarray([row["n_images"] for row in method_rows], dtype=np.float64)
        axes[0].semilogx(
            x,
            [row["best_estimate_fsc05_resolution_A"] for row in method_rows],
            color=method.color,
            marker="o",
            lw=2.4,
            label=f"{method.short_label}: best estimate",
        )
        axes[0].semilogx(
            x,
            [row["nearest1000_gt_fsc05_resolution_A"] for row in method_rows],
            color=method.color,
            marker="^",
            ls="--",
            lw=2.0,
            label=f"{method.short_label}: nearest-1000 GT mix",
        )
        axes[1].semilogx(
            x,
            [row["best_estimate_relerr10_resolution_A"] for row in method_rows],
            color=method.color,
            marker="o",
            lw=2.4,
            label=f"{method.short_label}: best estimate",
        )
        axes[1].semilogx(
            x,
            [row["nearest1000_gt_relerr10_resolution_A"] for row in method_rows],
            color=method.color,
            marker="^",
            ls="--",
            lw=2.0,
            label=f"{method.short_label}: nearest-1000 GT mix",
        )
    axes[0].set_title("Best FSC0.5 resolution across 50 candidates")
    axes[0].set_ylabel("FSC0.5 resolution (A), lower is better")
    axes[1].set_title("Best relative-error<0.1 resolution across 50 candidates")
    axes[1].set_ylabel("relative-error<0.1 resolution (A), lower is better")
    for ax in axes:
        ax.set_xlabel("number of images")
        ax.invert_yaxis()
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8.0)
    png = out_dir / "noise1_sweep_best_resolution_with_gt_nearest.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "noise1_sweep_best_resolution_with_gt_nearest.pdf")
    plt.close(fig)
    return png


def plot_best_curves(payloads: dict[tuple[str, int], dict[str, np.ndarray]], out_dir: Path) -> Path:
    image_counts = sorted({key[1] for key in payloads})
    colors = plt.cm.viridis(np.linspace(0.10, 0.92, len(image_counts)))
    color_by_n = dict(zip(image_counts, colors))

    fig, axes = plt.subplots(2, len(METHODS), figsize=(15.8, 8.8), sharex=True)
    fig.suptitle(
        "Best candidate curves and nearest-1000 GT mixtures | noise=1 | state 50 broad moving mask",
        fontweight="bold",
    )
    for col, method in enumerate(METHODS):
        for n_images in image_counts:
            payload = payloads.get((method.label, n_images))
            if payload is None:
                continue
            freq = payload["freq"]
            color = color_by_n[n_images]
            label = f"{n_images // 1000:g}k"
            axes[0, col].plot(freq, payload["best_fsc"], color=color, lw=2.0, label=f"{label} best")
            axes[0, col].plot(freq, payload["nearest_fsc"], color=color, ls="--", lw=1.7, label=f"{label} GT nearest")
            axes[1, col].semilogy(freq, np.maximum(payload["best_err"], 1e-30), color=color, lw=2.0)
            axes[1, col].semilogy(freq, np.maximum(payload["nearest_err"], 1e-30), color=color, ls="--", lw=1.7)
        axes[0, col].axhline(0.5, color="0.4", ls=":", lw=1.0)
        axes[1, col].axhline(0.1, color="0.4", ls=":", lw=1.0)
        axes[0, col].set_title(f"{method.short_label}: FSC vs GT")
        axes[1, col].set_title(f"{method.short_label}: relative error vs GT")
        axes[0, col].set_ylim(-0.08, 1.03)
        axes[1, col].set_ylim(1e-3, 1e2)
        axes[0, col].legend(fontsize=7.0, ncols=2, loc="lower left")
    for ax in axes.ravel():
        ax.set_xlim(0.0, FREQ_MAX)
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel("spatial frequency (1/A)")
    axes[0, 0].set_ylabel("masked FSC vs GT")
    axes[1, 0].set_ylabel("masked relative Fourier error")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = out_dir / "noise1_sweep_best_curves_with_gt_nearest.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / "noise1_sweep_best_curves_with_gt_nearest.pdf")
    plt.close(fig)
    return png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--image-counts", default=",".join(str(x) for x in DEFAULT_IMAGE_COUNTS))
    parser.add_argument("--nearest-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_counts = [int(part) for part in args.image_counts.split(",") if part]
    mask = np.clip(load_mrc(args.mask), 0.0, 1.0)

    rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    payloads: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    curve_npz: dict[str, np.ndarray] = {}
    freq_for_npz: np.ndarray | None = None

    for n_images in image_counts:
        run = run_dir(args.root, n_images)
        if not run.exists():
            print(f"SKIP missing run: {run}", flush=True)
            continue
        target = load_mrc(run / "04_ground_truth" / f"gt_vol{TARGET_STATE:04d}.mrc")
        labels, n_shells, target_ft, target_power, freq = metric_context(target, mask)
        freq_for_npz = freq
        print(f"n={n_images:,}", flush=True)
        for method in METHODS:
            compute_root = run / method.compute_relpath
            if not (compute_root / "state000_unfil.mrc").exists():
                print(f"  SKIP missing compute_state: {compute_root}", flush=True)
                continue
            print(f"  {method.label}", flush=True)
            grid, fsc, err = candidate_metrics(
                compute_root,
                mask,
                labels,
                n_shells,
                target_ft,
                target_power,
                args.batch_size,
            )
            fsc_res = np.asarray([fsc05_resolution(freq, fsc[idx]) for idx in range(fsc.shape[0])])
            err_res = np.asarray([relerr_resolution(freq, err[idx], 0.1) for idx in range(err.shape[0])])
            best_fsc_idx = int(np.nanargmin(fsc_res))
            best_err_idx = int(np.nanargmin(err_res))

            nearest_vol, nearest_info = nearest_gt_mixture(run, method, args.nearest_count)
            nearest_fsc, nearest_err = masked_metrics(nearest_vol, mask, labels, n_shells, target_ft, target_power)
            nearest_fsc_res = fsc05_resolution(freq, nearest_fsc)
            nearest_err_res = relerr_resolution(freq, nearest_err, 0.1)

            rows.append(
                {
                    "n_images": n_images,
                    "method": method.label,
                    "best_fsc_candidate_index_0based": best_fsc_idx,
                    "best_fsc_candidate_index_1based": best_fsc_idx + 1,
                    "best_fsc_candidate_parameter": float(grid[best_fsc_idx]),
                    "best_estimate_fsc05_resolution_A": float(fsc_res[best_fsc_idx]),
                    "best_error_candidate_index_0based": best_err_idx,
                    "best_error_candidate_index_1based": best_err_idx + 1,
                    "best_error_candidate_parameter": float(grid[best_err_idx]),
                    "best_estimate_relerr10_resolution_A": float(err_res[best_err_idx]),
                    "nearest1000_gt_fsc05_resolution_A": nearest_fsc_res,
                    "nearest1000_gt_relerr10_resolution_A": nearest_err_res,
                    "nearest1000_gt_state50_fraction": nearest_info["state50_fraction"],
                    "nearest1000_gt_top_states": nearest_info["top_states"],
                    "nearest1000_gt_radius": nearest_info["nearest_radius"],
                    "compute_state_dir": str(compute_root),
                    "nearest_source": nearest_info["nearest_source"],
                    "target_point": nearest_info["target_point"],
                    "mask": str(args.mask),
                }
            )
            state_rows.append(
                {
                    "n_images": n_images,
                    "method": method.label,
                    "n_nearest": nearest_info["n_nearest"],
                    "state50_fraction": nearest_info["state50_fraction"],
                    "nearest_radius": nearest_info["nearest_radius"],
                    "top_states": nearest_info["top_states"],
                    "nearest_source": nearest_info["nearest_source"],
                    "target_point": nearest_info["target_point"],
                }
            )
            payloads[(method.label, n_images)] = {
                "freq": freq,
                "best_fsc": fsc[best_fsc_idx],
                "best_err": err[best_err_idx],
                "nearest_fsc": nearest_fsc,
                "nearest_err": nearest_err,
            }
            prefix = f"{method.short_label}_n{n_images:08d}"
            curve_npz[f"{prefix}_best_fsc_curve"] = fsc[best_fsc_idx]
            curve_npz[f"{prefix}_best_error_curve"] = err[best_err_idx]
            curve_npz[f"{prefix}_nearest_gt_fsc_curve"] = nearest_fsc
            curve_npz[f"{prefix}_nearest_gt_error_curve"] = nearest_err

    if not rows:
        raise RuntimeError("No completed compute_state runs found")

    write_summary(args.out_dir / "noise1_sweep_gt_nearest_summary.csv", rows)
    write_summary(args.out_dir / "noise1_sweep_gt_nearest_state_mix.csv", state_rows)
    if freq_for_npz is not None:
        write_curves_npz(args.out_dir / "noise1_sweep_gt_nearest_curves.npz", freq_for_npz, curve_npz)
    resolution_plot = plot_resolution_summary(rows, args.out_dir)
    curve_plot = plot_best_curves(payloads, args.out_dir)

    print(f"SUMMARY {args.out_dir / 'noise1_sweep_gt_nearest_summary.csv'}")
    print(f"STATE_MIX {args.out_dir / 'noise1_sweep_gt_nearest_state_mix.csv'}")
    print(f"RESOLUTION_PLOT {resolution_plot}")
    print(f"CURVE_PLOT {curve_plot}")


if __name__ == "__main__":
    main()
