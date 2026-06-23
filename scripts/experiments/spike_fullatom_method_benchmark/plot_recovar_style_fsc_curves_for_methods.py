#!/usr/bin/env python3
"""Plot method volumes with the RECOVAR shell-curve style."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu


SOURCE_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516"
)
DRGN_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_zdim1_correct_sign_noise10_b100_20260529/"
    "evaluation_size_sweep_zdim1_correct_sign"
)
FLEX_EVAL_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sanity_100k_noise10_b100_20260529/"
    "evaluation_3dflex_mean_latents"
)
FLEX_PROJECT_DIR = Path("/tigress/CRYOEM/singerlab/mg6942/CS-testres")

STATE_LABELS = (0, 25, 50)


def dft3(volume: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume)))


def shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    labels = np.asarray(ftu.get_grid_of_radial_distances(shape, rounded=True), dtype=np.int32)
    n_shells = shape[0] // 2 - 1
    return np.clip(labels, 0, n_shells - 1), n_shells


def shell_metrics(estimate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    labels, n_shells = shell_labels(estimate.shape)
    estimate_ft = dft3(estimate * mask)
    target_ft = dft3(target * mask)
    diff_ft = estimate_ft - target_ft
    flat = labels.ravel()
    cross = np.bincount(
        flat,
        weights=np.real(np.conj(estimate_ft).ravel() * target_ft.ravel()),
        minlength=n_shells,
    )
    estimate_power = np.bincount(flat, weights=np.abs(estimate_ft).ravel() ** 2, minlength=n_shells)
    target_power = np.bincount(flat, weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    diff_power = np.bincount(flat, weights=np.abs(diff_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(estimate_power * target_power)
        relative_error = diff_power / target_power
        cumulative_relative_error = np.cumsum(diff_power) / np.cumsum(target_power)
    fsc[~np.isfinite(fsc)] = 0.0
    relative_error[~np.isfinite(relative_error)] = np.nan
    cumulative_relative_error[~np.isfinite(cumulative_relative_error)] = np.nan
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return {
        "fsc": fsc.astype(np.float32),
        "relative_error": relative_error.astype(np.float32),
        "cumulative_relative_error": cumulative_relative_error.astype(np.float32),
        "target_power": target_power.astype(np.float64),
        "diff_power": diff_power.astype(np.float64),
    }


def last_good_resolution(
    frequency: np.ndarray,
    values: np.ndarray,
    threshold: float,
    higher_is_better: bool,
) -> float:
    valid = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0)
    frequency = np.asarray(frequency, dtype=np.float64)[valid]
    values = np.asarray(values, dtype=np.float64)[valid]
    if frequency.size == 0:
        return math.nan
    good = values >= threshold if higher_is_better else values <= threshold
    good_indices = np.flatnonzero(good)
    if good_indices.size == 0:
        return math.nan
    return float(1.0 / frequency[int(good_indices[-1])])


def short_n_label(n_images: int) -> str:
    if n_images >= 1_000_000:
        return f"{n_images / 1_000_000:g}M"
    return f"{n_images / 1000:g}k"


def write_shell_csv(path: Path, metrics: dict[str, np.ndarray], voxel_size: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    freq = np.arange(metrics["fsc"].size, dtype=np.float64) / (metrics["fsc"].size + 1) / (
        2 * voxel_size
    )
    # Match the RECOVAR script exactly: shell / (box * voxel).
    box_size = (metrics["fsc"].size + 1) * 2
    freq = np.arange(metrics["fsc"].size, dtype=np.float64) / (box_size * voxel_size)
    resolution = np.divide(1.0, freq, out=np.full_like(freq, np.inf), where=freq > 0)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shell",
                "frequency_1_per_A",
                "resolution_A",
                "fsc_vs_gt",
                "relative_error_per_shell",
                "cumulative_relative_error",
                "target_power",
                "diff_power",
            ]
        )
        for i in range(freq.size):
            writer.writerow(
                [
                    i,
                    freq[i],
                    resolution[i],
                    metrics["fsc"][i],
                    metrics["relative_error"][i],
                    metrics["cumulative_relative_error"][i],
                    metrics["target_power"][i],
                    metrics["diff_power"][i],
                ]
            )


def plot_group(
    out_dir: Path,
    title: str,
    stem: str,
    curves: list[dict[str, object]],
    colors: np.ndarray | list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
    for color, row in zip(colors, curves):
        data = row["metrics"]
        frequency = row["frequency"]
        axes[0].plot(
            frequency,
            data["fsc"],
            color=color,
            lw=2.0,
            label=f"{row['label']} ({float(row['fsc05_resolution_A']):.2f} A)",
        )
        axes[1].semilogy(
            frequency,
            data["relative_error"],
            color=color,
            lw=2.0,
            label=f"{row['label']} ({float(row['relerr05_resolution_A']):.2f} A)",
        )

    axes[0].axhline(0.5, color="0.35", ls="--", lw=1.0)
    axes[0].set_title("masked FSC vs GT")
    axes[0].set_xlabel("spatial frequency (1/A)")
    axes[0].set_ylabel("FSC vs GT")
    axes[0].set_xlim(0.0, 0.40)
    axes[0].set_ylim(-0.05, 1.03)
    axes[0].legend(title="curve, FSC0.5", fontsize=8)

    axes[1].axhline(0.5, color="0.35", ls="--", lw=1.0)
    axes[1].set_title("masked relative shell error vs GT")
    axes[1].set_xlabel("spatial frequency (1/A)")
    axes[1].set_ylabel("relative Fourier shell error")
    axes[1].set_xlim(0.0, 0.40)
    axes[1].set_ylim(1e-3, 1e3)
    axes[1].legend(title="curve, error<=0.5", fontsize=8)
    fig.suptitle(title, fontsize=12, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen and key != "metrics" and key != "frequency":
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_curve(
    estimate_path: Path,
    gt_path: Path,
    mask_path: Path,
    voxel_size: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, float, float]:
    estimate = np.asarray(utils.load_mrc(estimate_path), dtype=np.float32)
    target = np.asarray(utils.load_mrc(gt_path), dtype=np.float32)
    mask = np.clip(np.asarray(utils.load_mrc(mask_path), dtype=np.float32), 0.0, 1.0)
    if estimate.shape != target.shape or estimate.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: estimate={estimate.shape}, gt={target.shape}, mask={mask.shape}"
        )
    metrics = shell_metrics(estimate, target, mask)
    frequency = np.arange(metrics["fsc"].size, dtype=np.float64) / (estimate.shape[0] * voxel_size)
    fsc05 = last_good_resolution(frequency, metrics["fsc"], 0.5, higher_is_better=True)
    relerr05 = last_good_resolution(
        frequency, metrics["relative_error"], 0.5, higher_is_better=False
    )
    return metrics, frequency, fsc05, relerr05


def plot_cryodrgn(args: argparse.Namespace) -> None:
    out_base = args.out_dir / "cryodrgn_zdim1_recovar_style_shell_curves"
    summary: list[dict[str, object]] = []
    for mask_name, mask_relpath in {
        "focus": "05_masks/focus_mask_moving.mrc",
        "global": "05_masks/volume_mask_union.mrc",
    }.items():
        out_dir = out_base / mask_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for state_index, state in enumerate(args.states):
            curves: list[dict[str, object]] = []
            for n_images in args.image_counts:
                source_run = SOURCE_ROOT / f"n{n_images:08d}" / "runs" / f"n{n_images:08d}_seed0000"
                estimate_path = (
                    DRGN_ROOT
                    / f"n{n_images:08d}"
                    / "cryodrgn/zdim1/decoded_volumes/labels_mean_z_epoch019"
                    / f"gt_label_{state_index:03d}.mrc"
                )
                gt_path = source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc"
                mask_path = source_run / mask_relpath
                if not estimate_path.exists():
                    continue
                metrics, frequency, fsc05, relerr05 = score_curve(
                    estimate_path, gt_path, mask_path, args.voxel_size
                )
                shell_csv = (
                    out_dir
                    / "shell_metrics"
                    / f"n{n_images:08d}"
                    / f"state{state:04d}"
                    / "shell_metrics.csv"
                )
                write_shell_csv(shell_csv, metrics, args.voxel_size)
                row = {
                    "method": "cryodrgn_zdim1",
                    "mask": mask_name,
                    "n_images": n_images,
                    "state": state,
                    "label": short_n_label(n_images),
                    "estimate": str(estimate_path),
                    "gt": str(gt_path),
                    "mask_path": str(mask_path),
                    "shell_metrics_csv": str(shell_csv),
                    "fsc05_resolution_A": fsc05,
                    "relerr05_resolution_A": relerr05,
                    "metrics": metrics,
                    "frequency": frequency,
                }
                curves.append(row)
                summary.append(row)
            if curves:
                colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(curves)))
                plot_group(
                    out_dir,
                    f"cryoDRGN zdim1 | state {state} | {mask_name} mask",
                    f"cryodrgn_zdim1_{mask_name}_state{state:04d}_fsc_relerr_by_n",
                    curves,
                    colors,
                )
    write_summary(out_base / "available_resolution_summary.csv", summary)


def flex_frames(job_uid: str) -> list[Path]:
    return sorted(
        (FLEX_PROJECT_DIR / job_uid).glob(f"{job_uid}_series_*/{job_uid}_series_*_frame_*.mrc")
    )


def plot_3dflex(args: argparse.Namespace) -> None:
    manifest = json.loads((FLEX_EVAL_ROOT / "3dflex_generate_mean_latents_manifest.json").read_text())
    source_run = SOURCE_ROOT / "n00100000/runs/n00100000_seed0000"
    out_base = args.out_dir / "3dflex_mean_latent_recovar_style_shell_curves"
    summary: list[dict[str, object]] = []
    for mask_name, mask_relpath in {
        "focus": "05_masks/focus_mask_moving.mrc",
        "global": "05_masks/volume_mask_union.mrc",
    }.items():
        out_dir = out_base / mask_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for state_index, state in enumerate(args.states):
            curves: list[dict[str, object]] = []
            for model in manifest["models"]:
                job_uid = model.get("mean_latents_generate")
                if not job_uid:
                    continue
                frames = flex_frames(str(job_uid))
                if state_index >= len(frames):
                    continue
                estimate_path = frames[state_index]
                gt_path = source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc"
                mask_path = source_run / mask_relpath
                metrics, frequency, fsc05, relerr05 = score_curve(
                    estimate_path, gt_path, mask_path, args.voxel_size
                )
                model_name = str(model["model"])
                shell_csv = (
                    out_dir
                    / "shell_metrics"
                    / model_name
                    / f"state{state:04d}"
                    / "shell_metrics.csv"
                )
                write_shell_csv(shell_csv, metrics, args.voxel_size)
                row = {
                    "method": "3dflex_mean_latent",
                    "model": model_name,
                    "mask": mask_name,
                    "state": state,
                    "label": model_name,
                    "train_job": model["train_job"],
                    "generate_job": job_uid,
                    "estimate": str(estimate_path),
                    "gt": str(gt_path),
                    "mask_path": str(mask_path),
                    "shell_metrics_csv": str(shell_csv),
                    "fsc05_resolution_A": fsc05,
                    "relerr05_resolution_A": relerr05,
                    "metrics": metrics,
                    "frequency": frequency,
                }
                curves.append(row)
                summary.append(row)
            if curves:
                colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"][: len(curves)]
                plot_group(
                    out_dir,
                    f"3DFlex mean latent | state {state} | {mask_name} mask",
                    f"3dflex_mean_latent_{mask_name}_state{state:04d}_fsc_relerr_by_model",
                    curves,
                    colors,
                )
    write_summary(out_base / "available_resolution_summary.csv", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
            "spike_fullatom_method_sanity_100k_noise10_b100_20260529/"
            "recovar_style_shell_curves"
        ),
    )
    parser.add_argument("--methods", default="cryodrgn,3dflex")
    parser.add_argument("--states", default="0,25,50")
    parser.add_argument("--image-counts", default="10000,30000,100000,300000")
    parser.add_argument("--voxel-size", type=float, default=1.25)
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.states = parse_ints(args.states)
    args.image_counts = parse_ints(args.image_counts)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    if "cryodrgn" in methods:
        plot_cryodrgn(args)
    if "3dflex" in methods:
        plot_3dflex(args)
    print(args.out_dir)


if __name__ == "__main__":
    main()
