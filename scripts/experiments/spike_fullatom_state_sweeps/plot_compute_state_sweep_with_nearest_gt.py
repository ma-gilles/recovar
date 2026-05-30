#!/usr/bin/env python3
"""Plot compute_state FSC/error curves with nearest-particle GT-mixture controls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.output import output as output_mod


DEFAULT_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_true_pipeline_sweep_noise10_b100_dev2_20260529"
)
DEFAULT_SOURCE_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516"
)
DEFAULT_IMAGE_COUNTS = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)
DEFAULT_STATES = (0, 25, 50)
DEFAULT_BROAD_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
VOXEL_SIZE_A = 1.25


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
    target_power = np.bincount(
        flat,
        weights=np.abs(target_ft).ravel() ** 2,
        minlength=n_shells,
    ).astype(np.float64)
    frequency = np.arange(n_shells, dtype=np.float64) / (target.shape[0] * VOXEL_SIZE_A)
    return labels, n_shells, target_ft, target_power, frequency


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
    pipeline_output = output_mod.PipelineOutput(str(pipeline_dir))
    values = np.asarray(pipeline_output.get_unsorted_embedding_component(entry, zdim), dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return values


def nearest100_weights_from_compute_state(
    run: Path,
    pipeline_dir: Path,
    compute_root: Path,
    state: int,
    nearest_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    target = np.asarray(np.loadtxt(compute_root / "latent_coords.txt"), dtype=np.float64).reshape(-1)
    zdim = target.size
    coords = unsorted_embedding(pipeline_dir, "latent_coords_noreg", zdim)[:, :zdim]
    precision = unsorted_embedding(pipeline_dir, "latent_precision_noreg", zdim)[:, :zdim, :zdim]
    assignments = np.asarray(np.load(run / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    finite = np.isfinite(coords).all(axis=1) & np.isfinite(precision).all(axis=(1, 2))
    if finite.size != assignments.size:
        raise ValueError(f"Embedding/state length mismatch for {run}: {finite.size} vs {assignments.size}")
    diff = coords - target[None, :]
    distances = np.einsum("ni,nij,nj->n", diff, precision, diff, optimize=True)
    distances[~finite] = np.inf
    n = min(int(nearest_count), int(np.sum(np.isfinite(distances))))
    nearest = np.argpartition(distances, n - 1)[:n]
    counts = np.bincount(assignments[nearest], minlength=100).astype(np.float64)
    weights = counts / counts.sum()
    top_states = [
        f"{int(idx)}:{int(counts[idx])}"
        for idx in np.argsort(counts)[::-1][:12]
        if counts[idx] > 0
    ]
    info = {
        "target_state": state,
        "n_nearest": int(n),
        "target_path": str(compute_root / "latent_coords.txt"),
        "pipeline_dir": str(pipeline_dir),
        "distance": "zdim4 no-reg cov_dist recomputed from compute_state target",
        "target_state_fraction": float(weights[state]) if state < weights.size else float("nan"),
        "nearest_min_distance": float(np.nanmin(distances[nearest])),
        "nearest_max_distance": float(np.nanmax(distances[nearest])),
        "top_states": " ".join(top_states),
    }
    return weights, info


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
        raise ValueError("Nearest GT mixture has all-zero weights")
    return total.astype(np.float32)


def interval_weights_around_state(state: int, half_width: int, n_states: int = 100) -> tuple[str, np.ndarray]:
    start = max(0, int(state) - int(half_width))
    stop = min(n_states - 1, int(state) + int(half_width))
    weights = np.zeros(n_states, dtype=np.float64)
    weights[start : stop + 1] = 1.0 / (stop - start + 1)
    return f"GT avg {start}-{stop}", weights


def plot_state(
    state: int,
    rows: list[dict[str, object]],
    out_dir: Path,
    *,
    nearest_count: int,
    gt_average_rows: Sequence[dict[str, object]],
    mask_label: str,
) -> None:
    rows = [row for row in rows if int(row["state"]) == state]
    if not rows:
        return
    rows.sort(key=lambda row: int(row["n_images"]))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), constrained_layout=True)
    gt_styles = [("0.15", ":"), ("0.35", "-."), ("0.55", (0, (5, 2, 1, 2)))]
    for color, row in zip(colors, rows):
        n_images = int(row["n_images"])
        label = f"{n_images // 1000}k" if n_images < 1_000_000 else f"{n_images / 1_000_000:g}M"
        frequency = row["frequency"]
        axes[0].plot(frequency, row["estimate_fsc"], color=color, lw=2.1, label=f"{label} estimate")
        axes[0].plot(
            frequency,
            row["nearest_fsc"],
            color=color,
            lw=1.8,
            ls="--",
            alpha=0.95,
            label=f"{label} nearest-{nearest_count} GT",
        )
        axes[1].semilogy(frequency, np.maximum(row["estimate_error"], 1e-30), color=color, lw=2.1)
        axes[1].semilogy(
            frequency,
            np.maximum(row["nearest_error"], 1e-30),
            color=color,
            lw=1.8,
            ls="--",
            alpha=0.95,
        )
    state_gt_rows = [row for row in gt_average_rows if int(row["state"]) == state]
    if rows:
        frequency = rows[0]["frequency"]
        for idx, baseline in enumerate(state_gt_rows):
            color, linestyle = gt_styles[idx % len(gt_styles)]
            label = str(baseline["label"])
            axes[0].plot(frequency, baseline["fsc"], color=color, lw=2.0, ls=linestyle, label=label)
            axes[1].semilogy(
                frequency,
                np.maximum(baseline["error"], 1e-30),
                color=color,
                lw=2.0,
                ls=linestyle,
            )
    axes[0].axhline(0.5, color="0.35", ls=":", lw=1.2)
    axes[0].set_title("FSC vs GT")
    axes[0].set_ylabel("masked FSC")
    axes[0].set_ylim(-0.05, 1.03)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].axhline(0.5, color="0.35", ls=":", lw=1.2)
    axes[1].set_title("Relative Fourier shell error")
    axes[1].set_ylabel("masked relative error")
    axes[1].set_ylim(1e-3, 1e3)
    for ax in axes:
        ax.set_xlim(0.0, 0.4)
        ax.set_xlabel("spatial frequency (1/A)")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"State {state} | noise10 true pipeline zdim4 no-reg | {mask_label} | solid=compute_state, dashed=nearest-{nearest_count} GT",
        fontsize=12,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"state{state:04d}_estimate_vs_nearest{nearest_count}_gt_by_n.{ext}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--states", default=",".join(str(v) for v in DEFAULT_STATES))
    parser.add_argument("--image-counts", default=",".join(str(v) for v in DEFAULT_IMAGE_COUNTS))
    parser.add_argument("--nearest-count", type=int, default=100)
    parser.add_argument("--pipeline-relpath", default="06_pipeline_true_zdim4_no_keep_input_mask")
    parser.add_argument("--compute-relpath", default="07_compute_state_zdim4_noreg")
    parser.add_argument("--mask", type=Path, default=DEFAULT_BROAD_MASK)
    parser.add_argument("--gt-average-half-widths", default="5,10,20")
    args = parser.parse_args()

    root = args.root.resolve()
    source_root = args.source_root.resolve()
    out_dir = args.out_dir or (root / f"09_curves_with_nearest{args.nearest_count}_gt_current")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.mask.resolve()
    mask = load_mrc(mask_path)
    mask_label = mask_path.name
    states = parse_ints(args.states)
    image_counts = parse_ints(args.image_counts)
    gt_average_half_widths = parse_ints(args.gt_average_half_widths)

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 220, "font.size": 10})
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    gt_summary_rows: list[dict[str, object]] = []
    gt_average_rows: list[dict[str, object]] = []
    curve_npz: dict[str, np.ndarray] = {}
    frequency_for_npz = None
    gt_average_done: set[tuple[int, int]] = set()
    for n_images in image_counts:
        source_run = run_dir(source_root, n_images)
        run = run_dir(root, n_images)
        pipeline_dir = run / args.pipeline_relpath
        if not pipeline_dir.exists():
            continue
        for state in states:
            compute_root = run / args.compute_relpath / f"state{state:04d}"
            estimate_path = compute_root / "state000_unfil.mrc"
            if not estimate_path.exists() or not (compute_root / "latent_coords.txt").exists():
                continue
            target = load_mrc(source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc")
            if mask.shape != target.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match target shape {target.shape}: {mask_path}")
            labels, n_shells, target_ft, target_power, metric_frequency = metric_context(target, mask)
            estimate_fsc, estimate_error = masked_metrics(
                load_mrc(estimate_path),
                mask,
                labels,
                n_shells,
                target_ft,
                target_power,
            )
            frequency = metric_frequency
            weights, nearest_info = nearest100_weights_from_compute_state(
                source_run,
                pipeline_dir,
                compute_root,
                state,
                args.nearest_count,
            )
            nearest_volume = weighted_gt_volume(source_run, weights)
            nearest_fsc, nearest_error = masked_metrics(
                nearest_volume,
                mask,
                labels,
                n_shells,
                target_ft,
                target_power,
            )
            for half_width in gt_average_half_widths:
                key = (state, half_width)
                if key in gt_average_done:
                    continue
                label, gt_weights = interval_weights_around_state(state, half_width, n_states=100)
                gt_fsc, gt_error = masked_metrics(
                    weighted_gt_volume(source_run, gt_weights),
                    mask,
                    labels,
                    n_shells,
                    target_ft,
                    target_power,
                )
                gt_average_done.add(key)
                gt_average_rows.append(
                    {
                        "state": state,
                        "half_width": half_width,
                        "label": label,
                        "fsc": gt_fsc,
                        "error": gt_error,
                    }
                )
                gt_summary_rows.append(
                    {
                        "state": state,
                        "gt_average": label,
                        "gt_average_fsc05_resolution_A": fsc05_resolution(frequency, gt_fsc),
                        "gt_average_relerr0p5_resolution_A": relerr_resolution(frequency, gt_error, 0.5),
                        "mask": str(mask_path),
                    }
                )
                curve_npz[f"state{state:04d}_gt_avg_hw{half_width}_fsc"] = gt_fsc
                curve_npz[f"state{state:04d}_gt_avg_hw{half_width}_error"] = gt_error
            row = {
                "n_images": n_images,
                "state": state,
                "frequency": frequency,
                "estimate_fsc": estimate_fsc,
                "estimate_error": estimate_error,
                "nearest_fsc": nearest_fsc,
                "nearest_error": nearest_error,
            }
            rows.append(row)
            frequency_for_npz = frequency
            prefix = f"n{n_images:08d}_state{state:04d}"
            curve_npz[f"{prefix}_estimate_fsc"] = estimate_fsc
            curve_npz[f"{prefix}_estimate_error"] = estimate_error
            curve_npz[f"{prefix}_nearest{args.nearest_count}_gt_fsc"] = nearest_fsc
            curve_npz[f"{prefix}_nearest{args.nearest_count}_gt_error"] = nearest_error
            summary_rows.append(
                {
                    "n_images": n_images,
                    "state": state,
                    "estimate_fsc05_resolution_A": fsc05_resolution(frequency, estimate_fsc),
                    "estimate_relerr0p5_resolution_A": relerr_resolution(frequency, estimate_error, 0.5),
                    f"nearest{args.nearest_count}_gt_fsc05_resolution_A": fsc05_resolution(
                        frequency, nearest_fsc
                    ),
                    f"nearest{args.nearest_count}_gt_relerr0p5_resolution_A": relerr_resolution(
                        frequency, nearest_error, 0.5
                    ),
                    f"nearest{args.nearest_count}_target_state_fraction": nearest_info[
                        "target_state_fraction"
                    ],
                    f"nearest{args.nearest_count}_top_states": nearest_info["top_states"],
                    f"nearest{args.nearest_count}_distance_min": nearest_info["nearest_min_distance"],
                    f"nearest{args.nearest_count}_distance_max": nearest_info["nearest_max_distance"],
                    "mask": str(mask_path),
                    "compute_root": str(compute_root),
                    "pipeline_dir": str(pipeline_dir),
                }
            )

    if not rows:
        raise RuntimeError("No completed estimate metrics found")
    for state in states:
        plot_state(
            state,
            rows,
            out_dir,
            nearest_count=args.nearest_count,
            gt_average_rows=gt_average_rows,
            mask_label=mask_label,
        )
    with (out_dir / f"estimate_vs_nearest{args.nearest_count}_gt_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    if gt_summary_rows:
        with (out_dir / "gt_average_baselines_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(gt_summary_rows[0]))
            writer.writeheader()
            writer.writerows(gt_summary_rows)
    if frequency_for_npz is not None:
        np.savez_compressed(
            out_dir / f"estimate_vs_nearest{args.nearest_count}_gt_curves.npz",
            frequency_1_per_A=frequency_for_npz,
            **curve_npz,
        )
    print(out_dir)
    for summary in summary_rows:
        print(
            f"n={summary['n_images']:>8} state={summary['state']:>2} "
            f"estimate_FSC0.5={summary['estimate_fsc05_resolution_A']:.3f} A "
            f"nearest{args.nearest_count}_FSC0.5={summary[f'nearest{args.nearest_count}_gt_fsc05_resolution_A']:.3f} A "
            f"nearest_target_frac={summary[f'nearest{args.nearest_count}_target_state_fraction']:.3f}"
        )
    for summary in gt_summary_rows:
        print(
            f"state={summary['state']:>2} {summary['gt_average']} "
            f"FSC0.5={summary['gt_average_fsc05_resolution_A']:.3f} A "
            f"relerr0.5={summary['gt_average_relerr0p5_resolution_A']:.3f} A"
        )


if __name__ == "__main__":
    main()
