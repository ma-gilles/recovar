#!/usr/bin/env python3
"""Overlay a 1M PPCA RECOVAR compute_state result on the noise10 normal sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.output import output as output_mod

VOXEL_SIZE_A = 1.25
DEFAULT_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_true_pipeline_sweep_noise10_b100_dev2_20260529")
DEFAULT_SOURCE_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516")
DEFAULT_MASK = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc")


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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
    target_ft = dft3(target * mask)
    flat = labels.ravel()
    target_power = np.bincount(flat, weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells).astype(np.float64)
    frequency = np.arange(n_shells, dtype=np.float64) / (target.shape[0] * VOXEL_SIZE_A)
    return labels, n_shells, target_ft, target_power, frequency


def masked_metrics(volume: np.ndarray, mask: np.ndarray, labels: np.ndarray, n_shells: int, target_ft: np.ndarray, target_power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    volume_ft = dft3(volume * mask)
    flat = labels.ravel()
    cross = np.bincount(flat, weights=np.real(np.conj(volume_ft).ravel() * target_ft.ravel()), minlength=n_shells)
    volume_power = np.bincount(flat, weights=np.abs(volume_ft).ravel() ** 2, minlength=n_shells)
    diff_power = np.bincount(flat, weights=np.abs(volume_ft - target_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(volume_power * target_power)
        relerr = diff_power / target_power
    fsc[~np.isfinite(fsc)] = 0.0
    relerr[~np.isfinite(relerr)] = np.nan
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return fsc.astype(np.float32), relerr.astype(np.float32)


def fsc05_resolution(frequency: np.ndarray, fsc: np.ndarray) -> float:
    valid = np.isfinite(fsc) & (frequency > 0)
    frequency = frequency[valid]
    fsc = fsc[valid]
    if frequency.size == 0:
        return float("nan")
    below = np.flatnonzero(fsc < 0.5)
    if below.size == 0:
        return float(1.0 / frequency[-1])
    idx = int(below[0])
    if idx == 0:
        return float(1.0 / frequency[0])
    y0, y1 = float(fsc[idx - 1]), float(fsc[idx])
    x0, x1 = float(frequency[idx - 1]), float(frequency[idx])
    crossing = x1 if y0 == y1 else x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return float(1.0 / crossing) if crossing > 0 else float("nan")


def relerr_resolution(frequency: np.ndarray, relerr: np.ndarray, threshold: float = 0.5) -> float:
    valid = np.isfinite(relerr) & (frequency > 0)
    frequency = frequency[valid]
    relerr = relerr[valid]
    if frequency.size == 0:
        return float("nan")
    good = np.flatnonzero(relerr <= threshold)
    if good.size == 0:
        return float("nan")
    return float(1.0 / frequency[int(good[-1])])


def unsorted_embedding(pipeline_dir: Path, entry: str, zdim: int) -> np.ndarray:
    po = output_mod.PipelineOutput(str(pipeline_dir))
    values = np.asarray(po.get_unsorted_embedding_component(entry, zdim), dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return values


def nearest_weights(source_run: Path, pipeline_dir: Path, compute_root: Path, state: int, nearest_count: int) -> tuple[np.ndarray, dict[str, object]]:
    target = np.asarray(np.loadtxt(compute_root / "latent_coords.txt"), dtype=np.float64).reshape(-1)
    zdim = target.size
    coords = unsorted_embedding(pipeline_dir, "latent_coords_noreg", zdim)[:, :zdim]
    precision = unsorted_embedding(pipeline_dir, "latent_precision_noreg", zdim)[:, :zdim, :zdim]
    assignments = np.asarray(np.load(source_run / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    diff = coords - target[None, :]
    finite = np.isfinite(coords).all(axis=1) & np.isfinite(precision).all(axis=(1, 2))
    distances = np.einsum("ni,nij,nj->n", diff, precision, diff, optimize=True)
    distances[~finite] = np.inf
    n = min(int(nearest_count), int(np.sum(np.isfinite(distances))))
    nearest = np.argpartition(distances, n - 1)[:n]
    counts = np.bincount(assignments[nearest], minlength=100).astype(np.float64)
    weights = counts / counts.sum()
    top_states = " ".join(f"{int(i)}:{int(counts[i])}" for i in np.argsort(counts)[::-1][:12] if counts[i] > 0)
    return weights, {
        "target_state_fraction": float(weights[state]) if state < weights.size else float("nan"),
        "top_states": top_states,
        "nearest_min_distance": float(np.nanmin(distances[nearest])),
        "nearest_max_distance": float(np.nanmax(distances[nearest])),
    }


def weighted_gt_volume(source_run: Path, weights: np.ndarray) -> np.ndarray:
    total = None
    for state, weight in enumerate(weights):
        if weight == 0:
            continue
        volume = load_mrc(source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc")
        if total is None:
            total = np.zeros_like(volume, dtype=np.float64)
        total += float(weight) * volume
    if total is None:
        raise ValueError("all-zero GT mixture")
    return total.astype(np.float32)


def load_normal_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--ppca-run", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--states", default="0,25,50")
    parser.add_argument("--image-counts", default="10000,30000,100000,300000,1000000,3000000")
    parser.add_argument("--nearest-count", type=int, default=100)
    parser.add_argument("--normal-summary", type=Path, default=None)
    parser.add_argument("--normal-compute-relpath", default="07_compute_state_zdim4_noreg")
    parser.add_argument("--normal-pipeline-relpath", default="06_pipeline_true_zdim4_no_keep_input_mask")
    parser.add_argument("--ppca-compute-relpath", default="07_compute_state_zdim4_ppca_noreg")
    parser.add_argument("--ppca-pipeline-relpath", default="06_pipeline_true_zdim4_ppca_no_keep_input_mask")
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    args = parser.parse_args()

    root = args.root.resolve()
    source_root = args.source_root.resolve()
    ppca_run = args.ppca_run.resolve() if args.ppca_run else run_dir(root, 1_000_000)
    out_dir = args.out_dir or (root / "09_ppca_1m_overlay_broadmask_soft_current")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.mask.resolve()
    mask = load_mrc(mask_path)
    states = parse_ints(args.states)
    image_counts = parse_ints(args.image_counts)
    normal_summary_path = args.normal_summary or (root / "09_curves_with_nearest100_gt_broadmask_soft_current" / "estimate_vs_nearest100_gt_summary.csv")
    normal_rows = load_normal_rows(normal_summary_path)

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 220, "font.size": 10})
    summary_rows: list[dict[str, object]] = []
    curve_npz: dict[str, np.ndarray] = {}
    frequency_for_npz = None

    for state in states:
        source_run = run_dir(source_root, 1_000_000)
        target = load_mrc(source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc")
        labels, n_shells, target_ft, target_power, frequency = metric_context(target, mask)
        frequency_for_npz = frequency

        ppca_compute = ppca_run / args.ppca_compute_relpath / f"state{state:04d}"
        ppca_pipeline = ppca_run / args.ppca_pipeline_relpath
        ppca_estimate = ppca_compute / "state000_unfil.mrc"
        if not ppca_estimate.exists():
            raise FileNotFoundError(ppca_estimate)
        ppca_fsc, ppca_err = masked_metrics(load_mrc(ppca_estimate), mask, labels, n_shells, target_ft, target_power)
        weights, nearest_info = nearest_weights(source_run, ppca_pipeline, ppca_compute, state, args.nearest_count)
        ppca_nearest_fsc, ppca_nearest_err = masked_metrics(
            weighted_gt_volume(source_run, weights), mask, labels, n_shells, target_ft, target_power
        )

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.1), constrained_layout=True)
        rows = [row for row in normal_rows if int(row["state"]) == state and int(row["n_images"]) in image_counts]
        rows.sort(key=lambda row: int(row["n_images"]))
        colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(len(rows), 1)))
        for color, row in zip(colors, rows):
            n_images = int(row["n_images"])
            normal_run = run_dir(root, n_images)
            normal_source_run = run_dir(source_root, n_images)
            normal_compute = Path(row["compute_root"])
            normal_pipeline = Path(row["pipeline_dir"])
            normal_target = load_mrc(normal_source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc")
            n_labels, n_shells2, n_target_ft, n_target_power, n_frequency = metric_context(normal_target, mask)
            normal_fsc, normal_err = masked_metrics(
                load_mrc(normal_compute / "state000_unfil.mrc"), mask, n_labels, n_shells2, n_target_ft, n_target_power
            )
            normal_weights, _ = nearest_weights(normal_source_run, normal_pipeline, normal_compute, state, args.nearest_count)
            normal_nearest_fsc, normal_nearest_err = masked_metrics(
                weighted_gt_volume(normal_source_run, normal_weights), mask, n_labels, n_shells2, n_target_ft, n_target_power
            )
            label = f"{n_images // 1000}k" if n_images < 1_000_000 else f"{n_images / 1_000_000:g}M"
            axes[0].plot(n_frequency, normal_fsc, color=color, lw=1.8, alpha=0.8, label=f"normal {label}")
            axes[0].plot(n_frequency, normal_nearest_fsc, color=color, lw=1.3, alpha=0.75, ls="--")
            axes[1].semilogy(n_frequency, np.maximum(normal_err, 1e-30), color=color, lw=1.8, alpha=0.8)
            axes[1].semilogy(n_frequency, np.maximum(normal_nearest_err, 1e-30), color=color, lw=1.3, alpha=0.75, ls="--")

        axes[0].plot(frequency, ppca_fsc, color="#d62728", lw=3.1, label="PPCA 1M")
        axes[0].plot(frequency, ppca_nearest_fsc, color="#d62728", lw=2.2, ls="--", label=f"PPCA nearest-{args.nearest_count} GT")
        axes[1].semilogy(frequency, np.maximum(ppca_err, 1e-30), color="#d62728", lw=3.1, label="PPCA 1M")
        axes[1].semilogy(frequency, np.maximum(ppca_nearest_err, 1e-30), color="#d62728", lw=2.2, ls="--", label=f"PPCA nearest-{args.nearest_count} GT")
        axes[0].axhline(0.5, color="0.35", ls=":", lw=1.2)
        axes[1].axhline(0.5, color="0.35", ls=":", lw=1.2)
        axes[0].set_title("FSC vs GT")
        axes[1].set_title("Relative Fourier shell error")
        axes[0].set_ylabel("masked FSC")
        axes[1].set_ylabel("masked relative error")
        axes[0].set_ylim(-0.05, 1.03)
        axes[1].set_ylim(1e-3, 1e3)
        for ax in axes:
            ax.set_xlim(0.0, 0.4)
            ax.set_xlabel("spatial frequency (1/A)")
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].legend(fontsize=7, ncol=1)
        fig.suptitle(
            f"State {state} | noise10 normal RECOVAR sweep + PPCA 1M | {mask_path.name} | solid=estimate, dashed=nearest-{args.nearest_count} GT",
            fontsize=12,
            fontweight="bold",
        )
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"state{state:04d}_normal_sweep_plus_ppca1m.{ext}")
        plt.close(fig)

        curve_npz[f"state{state:04d}_ppca1m_fsc"] = ppca_fsc
        curve_npz[f"state{state:04d}_ppca1m_error"] = ppca_err
        curve_npz[f"state{state:04d}_ppca1m_nearest{args.nearest_count}_fsc"] = ppca_nearest_fsc
        curve_npz[f"state{state:04d}_ppca1m_nearest{args.nearest_count}_error"] = ppca_nearest_err
        summary_rows.append(
            {
                "n_images": 1_000_000,
                "state": state,
                "method": "ppca_zdim4_noreg",
                "estimate_fsc05_resolution_A": fsc05_resolution(frequency, ppca_fsc),
                "estimate_relerr0p5_resolution_A": relerr_resolution(frequency, ppca_err),
                f"nearest{args.nearest_count}_gt_fsc05_resolution_A": fsc05_resolution(frequency, ppca_nearest_fsc),
                f"nearest{args.nearest_count}_gt_relerr0p5_resolution_A": relerr_resolution(frequency, ppca_nearest_err),
                f"nearest{args.nearest_count}_target_state_fraction": nearest_info["target_state_fraction"],
                f"nearest{args.nearest_count}_top_states": nearest_info["top_states"],
                f"nearest{args.nearest_count}_distance_min": nearest_info["nearest_min_distance"],
                f"nearest{args.nearest_count}_distance_max": nearest_info["nearest_max_distance"],
                "mask": str(mask_path),
                "compute_root": str(ppca_compute),
                "pipeline_dir": str(ppca_pipeline),
            }
        )

    with (out_dir / "ppca1m_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    if frequency_for_npz is not None:
        np.savez_compressed(out_dir / "ppca1m_curves.npz", frequency_1_per_A=frequency_for_npz, **curve_npz)
    print(out_dir)
    for row in summary_rows:
        print(
            f"state={row['state']:>2} PPCA1M FSC0.5={row['estimate_fsc05_resolution_A']:.3f} A "
            f"nearest{args.nearest_count}_FSC0.5={row[f'nearest{args.nearest_count}_gt_fsc05_resolution_A']:.3f} A "
            f"target_frac={row[f'nearest{args.nearest_count}_target_state_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
