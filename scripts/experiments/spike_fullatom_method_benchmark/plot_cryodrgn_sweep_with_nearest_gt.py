#!/usr/bin/env python3
"""Plot cryoDRGN FSC/error curves with nearest-particle GT-mixture controls.

This mirrors the RECOVAR compute_state nearest-GT analysis, but uses Euclidean
distance in cryoDRGN latent space because cryoDRGN does not expose per-particle
latent precision matrices.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu


DEFAULT_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_zdim1_correct_sign_noise10_b100_20260529"
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


def parse_epochs(value: str) -> list[int]:
    epochs: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x.strip()) for x in part.split("-", 1)]
            step = 1 if end >= start else -1
            epochs.extend(range(start, end + step, step))
        else:
            epochs.append(int(part))
    return epochs


def run_dir(root: Path, n_images: int) -> Path:
    label = f"n{n_images:08d}"
    return root / label / "runs" / f"{label}_seed0000"


def cryodrgn_dir(root: Path, n_images: int, run_name: str) -> Path:
    return root / f"n{n_images:08d}" / "cryodrgn" / run_name


def evaluation_run_dir(evaluation_root: Path, n_images: int, run_name: str) -> Path:
    by_count = evaluation_root / f"n{n_images:08d}" / "cryodrgn" / run_name
    if by_count.exists():
        return by_count
    return evaluation_root / "cryodrgn" / run_name


def load_mrc(path: Path) -> np.ndarray:
    return np.asarray(utils.load_mrc(path), dtype=np.float32)


def load_z(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        array = np.asarray(pickle.load(handle), dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return array


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


def epoch_from_path(path: Path) -> int | None:
    match = re.search(r"(?:epoch|z\.|weights\.)(\d+)", path.name)
    return int(match.group(1)) if match else None


def decoded_dir_for_epoch(eval_run: Path, epoch: int) -> Path:
    return eval_run / "decoded_volumes" / f"labels_mean_z_epoch{epoch:03d}"


def available_decoded_epochs(train_dir: Path, eval_run: Path) -> list[int]:
    z_epochs = {
        epoch
        for path in train_dir.glob("z.*.pkl")
        for epoch in [epoch_from_path(path)]
        if epoch is not None
    }
    decoded_epochs = {
        epoch
        for path in (eval_run / "decoded_volumes").glob("labels_mean_z_epoch*")
        for epoch in [epoch_from_path(path)]
        if epoch is not None
    }
    return sorted(z_epochs & decoded_epochs)


def resolve_epoch(epoch_arg: str, train_dir: Path, eval_run: Path) -> int | None:
    if epoch_arg != "latest":
        return int(epoch_arg)
    epochs = available_decoded_epochs(train_dir, eval_run)
    return epochs[-1] if epochs else None


def label_order(eval_run: Path, epoch: int, fallback_states: list[int]) -> list[int]:
    manifest = eval_run / "mean_embeddings" / f"labels_mean_z_epoch{epoch:03d}.json"
    if manifest.exists():
        import json

        data = json.loads(manifest.read_text())
        labels = data.get("labels")
        if labels:
            return [int(x) for x in labels]
    csv_path = eval_run / "mean_embeddings" / f"labels_mean_z_epoch{epoch:03d}.csv"
    if csv_path.exists():
        with csv_path.open() as handle:
            reader = csv.DictReader(handle)
            labels = [int(row["gt_label"]) for row in reader if row.get("gt_label")]
        if labels:
            return labels
    return list(fallback_states)


def nearest_weights(
    z: np.ndarray,
    assignments: np.ndarray,
    state: int,
    nearest_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if z.shape[0] != assignments.shape[0]:
        raise ValueError(f"Latent/state length mismatch: {z.shape[0]} vs {assignments.shape[0]}")
    state_mask = assignments == state
    if not np.any(state_mask):
        raise ValueError(f"No particles assigned to state {state}")
    target = z[state_mask].mean(axis=0)
    finite = np.isfinite(z).all(axis=1)
    diff = z.astype(np.float64) - target.astype(np.float64)[None, :]
    distances = np.einsum("ni,ni->n", diff, diff, optimize=True)
    distances[~finite] = np.inf
    n = min(int(nearest_count), int(np.sum(np.isfinite(distances))))
    nearest = np.argpartition(distances, n - 1)[:n]
    n_states = max(100, int(assignments.max()) + 1)
    counts = np.bincount(assignments[nearest], minlength=n_states).astype(np.float64)
    weights = counts / counts.sum()
    top_states = [
        f"{int(idx)}:{int(counts[idx])}"
        for idx in np.argsort(counts)[::-1][:12]
        if counts[idx] > 0
    ]
    info = {
        "target_state": state,
        "n_nearest": int(n),
        "distance": "Euclidean in cryoDRGN latent z",
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
        path = source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc"
        if not path.exists():
            continue
        volume = load_mrc(path)
        if total is None:
            total = np.zeros_like(volume, dtype=np.float64)
        total += float(weight) * volume
    if total is None:
        raise ValueError("Nearest GT mixture has no loadable nonzero states")
    return total.astype(np.float32)


def interval_weights_around_state(state: int, half_width: int, n_states: int = 100) -> tuple[str, np.ndarray]:
    start = max(0, int(state) - int(half_width))
    stop = min(n_states - 1, int(state) + int(half_width))
    weights = np.zeros(n_states, dtype=np.float64)
    weights[start : stop + 1] = 1.0 / (stop - start + 1)
    return f"GT avg {start}-{stop}", weights


def short_count(n_images: int) -> str:
    if n_images >= 1_000_000:
        return f"{n_images / 1_000_000:g}M"
    return f"{n_images // 1000}k"


def plot_state(
    state: int,
    rows: list[dict[str, object]],
    out_dir: Path,
    *,
    nearest_count: int,
    sweep_axis: str,
    gt_average_rows: list[dict[str, object]],
    mask_label: str,
) -> None:
    rows = [row for row in rows if int(row["state"]) == state]
    if not rows:
        return
    sort_key = "epoch" if sweep_axis == "epoch" else "n_images"
    rows.sort(key=lambda row: int(row[sort_key]))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), constrained_layout=True)
    for color, row in zip(colors, rows):
        if sweep_axis == "epoch":
            label = f"epoch {int(row['epoch'])}"
        else:
            label = short_count(int(row["n_images"]))
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
    gt_styles = [("0.15", ":"), ("0.35", "-."), ("0.55", (0, (5, 2, 1, 2)))]
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
        f"cryoDRGN zdim1 | state {state} | {mask_label} | solid=decoded estimate, dashed=nearest-{nearest_count} GT",
        fontsize=12,
        fontweight="bold",
    )
    suffix = "n" if sweep_axis == "image_count" else sweep_axis
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"state{state:04d}_estimate_vs_nearest{nearest_count}_gt_by_{suffix}.{ext}")
    plt.close(fig)


def plot_resolution_summary(
    summary_rows: list[dict[str, object]],
    out_dir: Path,
    *,
    nearest_count: int,
    sweep_axis: str,
) -> None:
    if not summary_rows:
        return
    x_key = "epoch" if sweep_axis == "epoch" else "n_images"
    x_label = "epoch" if sweep_axis == "epoch" else "number of images"
    for metric, ylabel, stem in [
        ("fsc05_resolution_A", "FSC0.5 resolution (A), lower is better", "fsc05_resolution"),
        ("relerr0p5_resolution_A", "relerr<=0.5 resolution (A), lower is better", "relerr0p5_resolution"),
    ]:
        fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
        for state in sorted({int(row["state"]) for row in summary_rows}):
            subset = sorted(
                [row for row in summary_rows if int(row["state"]) == state],
                key=lambda row: int(row[x_key]),
            )
            x = np.asarray([int(row[x_key]) for row in subset], dtype=np.float64)
            estimate = np.asarray([float(row[f"estimate_{metric}"]) for row in subset], dtype=np.float64)
            nearest = np.asarray(
                [float(row[f"nearest{nearest_count}_gt_{metric}"]) for row in subset],
                dtype=np.float64,
            )
            ax.plot(x, estimate, marker="o", lw=2.0, label=f"state {state} estimate")
            ax.plot(x, nearest, marker="s", lw=1.7, ls="--", label=f"state {state} nearest GT")
        if sweep_axis == "image_count":
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / f"{stem}_vs_{sweep_axis}.png", dpi=220)
        fig.savefig(out_dir / f"{stem}_vs_{sweep_axis}.pdf")
        plt.close(fig)


def write_shell_csv(
    path: Path,
    frequency: np.ndarray,
    estimate_fsc: np.ndarray,
    estimate_error: np.ndarray,
    nearest_fsc: np.ndarray,
    nearest_error: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolution = np.divide(1.0, frequency, out=np.full_like(frequency, np.inf), where=frequency > 0)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shell",
                "frequency_1_per_A",
                "resolution_A",
                "estimate_fsc_vs_gt",
                "estimate_relative_error_per_shell",
                "nearest_gt_fsc_vs_gt",
                "nearest_gt_relative_error_per_shell",
            ]
        )
        for idx in range(frequency.size):
            writer.writerow(
                [
                    idx,
                    frequency[idx],
                    resolution[idx],
                    estimate_fsc[idx],
                    estimate_error[idx],
                    nearest_fsc[idx],
                    nearest_error[idx],
                ]
            )


def resolve_mask_path(source_run: Path, mask: Path | None, mask_relpath: str) -> Path:
    if mask is not None:
        return mask
    rel = Path(mask_relpath)
    return rel if rel.is_absolute() else source_run / rel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--states", default=",".join(str(v) for v in DEFAULT_STATES))
    parser.add_argument("--image-counts", default=",".join(str(v) for v in DEFAULT_IMAGE_COUNTS))
    parser.add_argument("--epoch", default="19", help="Epoch for image_count sweeps, or 'latest'.")
    parser.add_argument("--epochs", default=None, help="Epoch list/ranges for epoch sweeps, e.g. 0-11.")
    parser.add_argument("--sweep-axis", choices=["image_count", "epoch"], default="image_count")
    parser.add_argument("--run-name", default="zdim1")
    parser.add_argument("--nearest-count", type=int, default=100)
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help=f"Metric mask path. If omitted, --mask-relpath is resolved under each source run. Broad default example: {DEFAULT_BROAD_MASK}",
    )
    parser.add_argument("--mask-relpath", default="05_masks/focus_mask_moving.mrc")
    parser.add_argument("--gt-average-half-widths", default="5,10,20")
    parser.add_argument(
        "--binary-mask-threshold",
        type=float,
        default=None,
        help="Optional threshold to binarize the metric mask. Default uses the clipped soft mask.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    source_root = args.source_root.resolve()
    evaluation_root = (args.evaluation_root or (root / "evaluation_size_sweep_zdim1_correct_sign")).resolve()
    out_dir = args.out_dir or (evaluation_root / f"cryodrgn_curves_with_nearest{args.nearest_count}_gt_current")
    out_dir.mkdir(parents=True, exist_ok=True)

    states = parse_ints(args.states)
    image_counts = parse_ints(args.image_counts)
    gt_average_half_widths = parse_ints(args.gt_average_half_widths)
    if args.sweep_axis == "epoch":
        if len(image_counts) != 1:
            raise ValueError("--sweep-axis epoch requires exactly one image count")
        epochs = parse_epochs(args.epochs) if args.epochs is not None else None
    else:
        epochs = None

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 220, "font.size": 10})
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    gt_summary_rows: list[dict[str, object]] = []
    gt_average_rows: list[dict[str, object]] = []
    curve_npz: dict[str, np.ndarray] = {}
    frequency_for_npz: np.ndarray | None = None
    gt_average_done: set[tuple[int, int]] = set()

    for n_images in image_counts:
        train_dir = cryodrgn_dir(root, n_images, args.run_name)
        eval_run = evaluation_run_dir(evaluation_root, n_images, args.run_name)
        source_run = run_dir(source_root, n_images)
        if not train_dir.exists() or not eval_run.exists() or not source_run.exists():
            print(f"SKIP n={n_images}: missing train/evaluation/source directory")
            continue
        epoch_list = epochs
        if epoch_list is None:
            epoch = resolve_epoch(args.epoch, train_dir, eval_run)
            epoch_list = [epoch] if epoch is not None else []
        assignments = np.asarray(np.load(source_run / "03_dataset" / "state_assignment.npy"), dtype=np.int64)
        mask_path = resolve_mask_path(source_run, args.mask, args.mask_relpath).resolve()
        mask = np.clip(load_mrc(mask_path), 0.0, 1.0)
        if args.binary_mask_threshold is not None:
            mask = (mask > args.binary_mask_threshold).astype(np.float32)
        for epoch in epoch_list:
            z_path = train_dir / f"z.{epoch}.pkl"
            decoded_dir = decoded_dir_for_epoch(eval_run, epoch)
            if not z_path.exists() or not decoded_dir.exists():
                print(f"SKIP n={n_images} epoch={epoch}: missing {z_path} or {decoded_dir}")
                continue
            z = load_z(z_path)
            labels = label_order(eval_run, epoch, states)
            for state in states:
                if state not in labels:
                    print(f"SKIP n={n_images} epoch={epoch} state={state}: state not in decoded label order")
                    continue
                volume_index = labels.index(state)
                estimate_path = decoded_dir / f"gt_label_{volume_index:03d}.mrc"
                gt_path = source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc"
                if not estimate_path.exists() or not gt_path.exists():
                    print(f"SKIP n={n_images} epoch={epoch} state={state}: missing estimate or GT")
                    continue
                target = load_mrc(gt_path)
                estimate = load_mrc(estimate_path)
                if estimate.shape != target.shape or mask.shape != target.shape:
                    raise ValueError(
                        f"Shape mismatch for n={n_images} epoch={epoch} state={state}: "
                        f"estimate={estimate.shape} target={target.shape} mask={mask.shape}"
                    )
                metric_labels, n_shells, target_ft, target_power, frequency = metric_context(target, mask)
                estimate_fsc, estimate_error = masked_metrics(
                    estimate,
                    mask,
                    metric_labels,
                    n_shells,
                    target_ft,
                    target_power,
                )
                weights, nearest_info = nearest_weights(z, assignments, state, args.nearest_count)
                nearest_volume = weighted_gt_volume(source_run, weights)
                nearest_fsc, nearest_error = masked_metrics(
                    nearest_volume,
                    mask,
                    metric_labels,
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
                        metric_labels,
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
                shell_csv = (
                    out_dir
                    / "shell_metrics"
                    / f"n{n_images:08d}"
                    / f"epoch{epoch:03d}"
                    / f"state{state:04d}"
                    / "shell_metrics.csv"
                )
                write_shell_csv(shell_csv, frequency, estimate_fsc, estimate_error, nearest_fsc, nearest_error)
                row = {
                    "n_images": n_images,
                    "epoch": epoch,
                    "state": state,
                    "frequency": frequency,
                    "estimate_fsc": estimate_fsc,
                    "estimate_error": estimate_error,
                    "nearest_fsc": nearest_fsc,
                    "nearest_error": nearest_error,
                }
                rows.append(row)
                frequency_for_npz = frequency
                prefix = f"n{n_images:08d}_epoch{epoch:03d}_state{state:04d}"
                curve_npz[f"{prefix}_estimate_fsc"] = estimate_fsc
                curve_npz[f"{prefix}_estimate_error"] = estimate_error
                curve_npz[f"{prefix}_nearest{args.nearest_count}_gt_fsc"] = nearest_fsc
                curve_npz[f"{prefix}_nearest{args.nearest_count}_gt_error"] = nearest_error
                summary_rows.append(
                    {
                        "n_images": n_images,
                        "epoch": epoch,
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
                        "z_path": str(z_path),
                        "estimate": str(estimate_path),
                        "gt": str(gt_path),
                        "mask": str(mask_path),
                        "shell_metrics_csv": str(shell_csv),
                    }
                )

    if not rows:
        raise RuntimeError("No completed cryoDRGN decoded volumes found")
    mask_label = Path(str(summary_rows[0]["mask"])).name if summary_rows else "mask"
    for state in states:
        plot_state(
            state,
            rows,
            out_dir,
            nearest_count=args.nearest_count,
            sweep_axis=args.sweep_axis,
            gt_average_rows=gt_average_rows,
            mask_label=mask_label,
        )
    plot_resolution_summary(summary_rows, out_dir, nearest_count=args.nearest_count, sweep_axis=args.sweep_axis)
    summary_path = out_dir / f"cryodrgn_estimate_vs_nearest{args.nearest_count}_gt_summary.csv"
    with summary_path.open("w", newline="") as handle:
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
            out_dir / f"cryodrgn_estimate_vs_nearest{args.nearest_count}_gt_curves.npz",
            frequency_1_per_A=frequency_for_npz,
            **curve_npz,
        )
    print(out_dir)
    for summary in summary_rows:
        print(
            f"n={summary['n_images']:>8} epoch={summary['epoch']:>2} state={summary['state']:>2} "
            f"estimate_FSC0.5={summary['estimate_fsc05_resolution_A']:.3f} A "
            f"nearest{args.nearest_count}_FSC0.5="
            f"{summary[f'nearest{args.nearest_count}_gt_fsc05_resolution_A']:.3f} A "
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
