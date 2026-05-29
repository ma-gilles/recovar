#!/usr/bin/env python3
"""Score RECOVAR state sweeps with FSC and relative Fourier error curves."""

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


N_IMAGES = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)
STATE_ROOTS = {
    0: Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0000_reuse_20260517"),
    25: Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0025_reuse_20260517"),
    50: Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516"),
}
DEFAULT_OUT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528/recovar")


def parse_state_root(values: list[str] | None) -> dict[int, Path]:
    if not values:
        return STATE_ROOTS
    roots: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected STATE=PATH for --state-root, got {value!r}")
        state_text, path_text = value.split("=", 1)
        roots[int(state_text)] = Path(path_text)
    return roots


def parse_image_counts(value: str) -> tuple[int, ...]:
    return tuple(int(part.replace("_", "")) for part in value.split(",") if part.strip())


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


def curves(volume: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, n_shells = shell_labels(target.shape)
    vol_ft = dft3(volume * mask)
    target_ft = dft3(target * mask)
    flat = labels.ravel()
    cross = np.bincount(flat, weights=np.real(np.conj(vol_ft).ravel() * target_ft.ravel()), minlength=n_shells)
    vol_power = np.bincount(flat, weights=np.abs(vol_ft).ravel() ** 2, minlength=n_shells)
    target_power = np.bincount(flat, weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    diff_power = np.bincount(flat, weights=np.abs(vol_ft.ravel() - target_ft.ravel()) ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(vol_power * target_power)
        relerr = diff_power / np.maximum(target_power, 1e-30)
    fsc[~np.isfinite(fsc)] = np.nan
    relerr[~np.isfinite(relerr)] = np.nan
    if fsc.size > 1 and np.isfinite(fsc[1]):
        fsc[0] = fsc[1]
    return fsc.astype(np.float32), relerr.astype(np.float32)


def resolution_at(freq: np.ndarray, fsc: np.ndarray, threshold: float = 0.5) -> float:
    valid = np.isfinite(fsc) & (freq > 0)
    if not np.any(valid):
        return float("nan")
    x = freq[valid]
    y = fsc[valid]
    below = np.flatnonzero(y < threshold)
    if below.size == 0:
        return float(1.0 / x[-1])
    idx = int(below[0])
    if idx == 0:
        return float(1.0 / x[0])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    x0, x1 = float(x[idx - 1]), float(x[idx])
    crossing = x1 if y0 == y1 else x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return float(1.0 / crossing) if crossing > 0 else float("nan")


def relerr_resolution(freq: np.ndarray, relerr: np.ndarray, threshold: float = 0.1) -> float:
    valid = np.isfinite(relerr) & (freq > 0)
    if not np.any(valid):
        return float("nan")
    x = freq[valid]
    y = relerr[valid]
    above = np.flatnonzero(y > threshold)
    if above.size == 0:
        return float(1.0 / x[-1])
    idx = int(above[0])
    return float(1.0 / x[idx]) if x[idx] > 0 else float("nan")


def write_curve_csv(path: Path, freq: np.ndarray, rows: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shell", "frequency_1_per_A", *rows])
        for idx, f in enumerate(freq):
            writer.writerow([idx, float(f), *[float(row[idx]) for row in rows.values()]])


def plot_state(
    out_dir: Path,
    state: int,
    freq: np.ndarray,
    fsc_rows: dict[int, np.ndarray],
    err_rows: dict[int, np.ndarray],
    summary: list[dict[str, object]],
    image_counts: tuple[int, ...],
    title: str,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(image_counts)))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), constrained_layout=True)
    for color, n_images in zip(colors, image_counts):
        row = next(r for r in summary if int(r["state"]) == state and int(r["n_images"]) == n_images)
        axes[0].plot(freq, fsc_rows[n_images], color=color, lw=2, label=f"{n_images:,}: {row['fsc05_resolution_A']:.2f} A")
        axes[1].semilogy(freq, np.maximum(err_rows[n_images], 1e-30), color=color, lw=2, label=f"{n_images:,}")
    axes[0].axhline(0.5, color="0.35", ls="--", lw=1)
    axes[1].axhline(0.1, color="0.35", ls="--", lw=1)
    axes[0].set_ylabel("masked FSC vs GT")
    axes[1].set_ylabel("masked relative Fourier error")
    for ax in axes:
        ax.set_xlim(0, 0.4)
        ax.set_xlabel("spatial frequency (1/A)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"{title} | state {state} | unfiltered compute_state", fontweight="bold")
    fig.savefig(out_dir / f"recovar_state{state:04d}_fsc_error_curves.png", dpi=180)
    fig.savefig(out_dir / f"recovar_state{state:04d}_fsc_error_curves.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.7), constrained_layout=True)
    xs = np.asarray(image_counts, dtype=np.float64)
    ys = [next(r for r in summary if int(r["state"]) == state and int(r["n_images"]) == n)["fsc05_resolution_A"] for n in image_counts]
    ax.semilogx(xs, ys, marker="o", lw=2)
    ax.invert_yaxis()
    ax.set_xlabel("number of images")
    ax.set_ylabel("FSC0.5 resolution (A), lower is better")
    ax.set_title(f"RECOVAR state {state}: resolution vs image count")
    ax.grid(alpha=0.25, which="both")
    fig.savefig(out_dir / f"recovar_state{state:04d}_resolution_vs_n.png", dpi=180)
    fig.savefig(out_dir / f"recovar_state{state:04d}_resolution_vs_n.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--state-root",
        action="append",
        default=None,
        help="State root as STATE=PATH. Repeat for each state. Defaults to noise10/B100 roots.",
    )
    parser.add_argument("--image-counts", default=",".join(str(n) for n in N_IMAGES))
    parser.add_argument("--mask-relpath", default="05_masks/focus_mask_moving.mrc")
    parser.add_argument("--estimate-relpath", default="07_compute_state/state000_unfil.mrc")
    parser.add_argument("--title", default="RECOVAR noise10/B100")
    parser.add_argument("--voxel-size", type=float, default=1.25)
    args = parser.parse_args()
    image_counts = parse_image_counts(args.image_counts)
    state_roots = parse_state_root(args.state_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for state, root in state_roots.items():
        state_fsc: dict[int, np.ndarray] = {}
        state_err: dict[int, np.ndarray] = {}
        freq = None
        for n_images in image_counts:
            rd = run_dir(root, n_images)
            estimate_path = rd / args.estimate_relpath
            gt_path = rd / "04_ground_truth" / f"gt_vol{state:04d}.mrc"
            mask_path = rd / args.mask_relpath
            estimate = load_mrc(estimate_path)
            gt = load_mrc(gt_path)
            mask = np.clip(load_mrc(mask_path), 0.0, 1.0)
            fsc, relerr = curves(estimate, gt, mask)
            freq = np.arange(fsc.size, dtype=np.float64) / (estimate.shape[0] * args.voxel_size)
            state_fsc[n_images] = fsc
            state_err[n_images] = relerr
            summary.append(
                {
                    "state": state,
                    "n_images": n_images,
                    "fsc05_resolution_A": resolution_at(freq, fsc, 0.5),
                    "relerr10_resolution_A": relerr_resolution(freq, relerr, 0.1),
                    "estimate": str(estimate_path),
                    "gt": str(gt_path),
                    "mask": str(mask_path),
                }
            )
        assert freq is not None
        write_curve_csv(
            args.out_dir / f"recovar_state{state:04d}_fsc_curves.csv",
            freq,
            {f"n{n:08d}": state_fsc[n] for n in image_counts},
        )
        write_curve_csv(
            args.out_dir / f"recovar_state{state:04d}_relative_error_curves.csv",
            freq,
            {f"n{n:08d}": state_err[n] for n in image_counts},
        )
        plot_state(args.out_dir, state, freq, state_fsc, state_err, summary, image_counts, args.title)

    with (args.out_dir / "recovar_state_sweep_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(args.out_dir)


if __name__ == "__main__":
    main()
