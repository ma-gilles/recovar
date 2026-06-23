#!/usr/bin/env python3
"""Make cumulative FSC reveal movies for state-50 method sweeps.

Each frame adds one FSC curve and keeps all previously added curves visible.
This is meant to pair with the ChimeraX state-50 method-progression movies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu


DEFAULT_SUMMARY_CSV = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "not_moving_softmask_fsc_resolution_20260601/"
    "not_moving_softmask_noise3_fsc_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "state50_cumulative_fsc_movies_20260601"
)
DEFAULT_BROAD_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
DEFAULT_NOT_MOVING_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/not_moving_spike_mask_soft_20260601.mrc"
)

METHOD_ORDER = ["recovar", "cryodrgn", "3dflex"]
METHOD_LABELS = {"recovar": "RECOVAR", "cryodrgn": "cryoDRGN", "3dflex": "3DFlex"}
METHOD_STYLES = {"recovar": "-", "cryodrgn": "--", "3dflex": "-."}
N_ORDER = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
N_LABELS = {
    10_000: "10k",
    30_000: "30k",
    100_000: "100k",
    300_000: "300k",
    1_000_000: "1M",
    3_000_000: "3M",
}
N_COLORS = {
    10_000: "#4c1d7a",
    30_000: "#2a6f91",
    100_000: "#2fb47c",
    300_000: "#c8e020",
    1_000_000: "#ff9f1c",
    3_000_000: "#d00000",
}


@dataclass(frozen=True)
class CurveInput:
    method: str
    n_images: int
    n_label: str
    state: int
    estimate: Path
    gt: Path


@dataclass
class Curve:
    method: str
    n_images: int
    n_label: str
    state: int
    estimate: Path
    gt: Path
    mask: Path
    frequency: np.ndarray
    fsc: np.ndarray
    fsc05_resolution_a: float
    shell_metrics_csv: Path

    @property
    def label(self) -> str:
        return f"{METHOD_LABELS[self.method]} {self.n_label} ({self.fsc05_resolution_a:.2f} A)"


def _recovar_filtered_path(path: Path) -> Path:
    if path.name == "state000_unfil.mrc":
        filtered = path.with_name("state000.mrc")
        if filtered.exists():
            return filtered
    return path


def _read_curve_inputs(summary_csv: Path, state: int) -> list[CurveInput]:
    out: list[CurveInput] = []
    with summary_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["state"]) != state:
                continue
            method = row["method"]
            if method not in METHOD_ORDER:
                continue
            estimate = Path(row["estimate"])
            if method == "recovar":
                estimate = _recovar_filtered_path(estimate)
            out.append(
                CurveInput(
                    method=method,
                    n_images=int(row["n_images"]),
                    n_label=row["n_label"],
                    state=state,
                    estimate=estimate,
                    gt=Path(row["gt"]),
                )
            )
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    n_order = {n_images: index for index, n_images in enumerate(N_ORDER)}
    out.sort(key=lambda x: (method_order[x.method], n_order.get(x.n_images, 999), x.n_images))
    return out


def _dft3(volume: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume)))


def _shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    labels = np.asarray(ftu.get_grid_of_radial_distances(shape, rounded=True), dtype=np.int32)
    n_shells = shape[0] // 2 - 1
    return np.clip(labels, 0, n_shells - 1), n_shells


def _shell_fsc(estimate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels, n_shells = _shell_labels(estimate.shape)
    estimate_ft = _dft3(estimate * mask)
    target_ft = _dft3(target * mask)
    flat = labels.ravel()
    cross = np.bincount(
        flat,
        weights=np.real(np.conj(estimate_ft).ravel() * target_ft.ravel()),
        minlength=n_shells,
    )
    estimate_power = np.bincount(flat, weights=np.abs(estimate_ft).ravel() ** 2, minlength=n_shells)
    target_power = np.bincount(flat, weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    diff_power = np.bincount(flat, weights=np.abs(estimate_ft - target_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(estimate_power * target_power)
        relative_error = diff_power / target_power
    fsc[~np.isfinite(fsc)] = 0.0
    relative_error[~np.isfinite(relative_error)] = np.nan
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return fsc.astype(np.float32), {
        "fsc": fsc.astype(np.float32),
        "relative_error": relative_error.astype(np.float32),
        "estimate_power": estimate_power.astype(np.float64),
        "target_power": target_power.astype(np.float64),
        "diff_power": diff_power.astype(np.float64),
    }


def _last_good_resolution(frequency: np.ndarray, values: np.ndarray, threshold: float) -> float:
    valid = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0)
    frequency = frequency[valid]
    values = values[valid]
    good = np.flatnonzero(values >= threshold)
    if good.size == 0:
        return math.nan
    return float(1.0 / frequency[int(good[-1])])


def _write_shell_csv(path: Path, frequency: np.ndarray, metrics: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolution = np.divide(1.0, frequency, out=np.full_like(frequency, np.inf), where=frequency > 0)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shell",
                "frequency_1_per_A",
                "resolution_A",
                "fsc_vs_gt",
                "relative_error_per_shell",
                "estimate_power",
                "target_power",
                "diff_power",
            ]
        )
        for index in range(frequency.size):
            writer.writerow(
                [
                    index,
                    frequency[index],
                    resolution[index],
                    metrics["fsc"][index],
                    metrics["relative_error"][index],
                    metrics["estimate_power"][index],
                    metrics["target_power"][index],
                    metrics["diff_power"][index],
                ]
            )


def _compute_curves(args: argparse.Namespace, mask_path: Path, output_dir: Path) -> list[Curve]:
    inputs = _read_curve_inputs(args.summary_csv, args.state)
    if not inputs:
        raise RuntimeError(f"no state {args.state} curves found in {args.summary_csv}")
    mask = np.clip(np.asarray(utils.load_mrc(mask_path), dtype=np.float32), 0.0, 1.0)
    curves: list[Curve] = []
    for item in inputs:
        if not item.estimate.exists():
            raise FileNotFoundError(item.estimate)
        if not item.gt.exists():
            raise FileNotFoundError(item.gt)
        estimate = np.asarray(utils.load_mrc(item.estimate), dtype=np.float32)
        gt = np.asarray(utils.load_mrc(item.gt), dtype=np.float32)
        if estimate.shape != gt.shape or estimate.shape != mask.shape:
            raise ValueError(
                f"shape mismatch for {item.method} {item.n_label}: estimate={estimate.shape}, gt={gt.shape}, mask={mask.shape}"
            )
        fsc, metrics = _shell_fsc(estimate, gt, mask)
        frequency = np.arange(fsc.size, dtype=np.float64) / (estimate.shape[0] * args.voxel_size)
        fsc05 = _last_good_resolution(frequency, fsc, 0.5)
        shell_csv = (
            output_dir
            / "shell_metrics"
            / item.method
            / f"n{item.n_images:08d}"
            / f"state{item.state:04d}"
            / "shell_metrics.csv"
        )
        _write_shell_csv(shell_csv, frequency, metrics)
        curves.append(
            Curve(
                method=item.method,
                n_images=item.n_images,
                n_label=item.n_label,
                state=item.state,
                estimate=item.estimate,
                gt=item.gt,
                mask=mask_path,
                frequency=frequency,
                fsc=fsc,
                fsc05_resolution_a=fsc05,
                shell_metrics_csv=shell_csv,
            )
        )
    return curves


def _write_summary(path: Path, curves: list[Curve]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "n_images",
                "n_label",
                "state",
                "fsc05_resolution_A",
                "estimate",
                "gt",
                "mask",
                "shell_metrics_csv",
            ],
        )
        writer.writeheader()
        for curve in curves:
            writer.writerow(
                {
                    "method": curve.method,
                    "n_images": curve.n_images,
                    "n_label": curve.n_label,
                    "state": curve.state,
                    "fsc05_resolution_A": curve.fsc05_resolution_a,
                    "estimate": str(curve.estimate),
                    "gt": str(curve.gt),
                    "mask": str(curve.mask),
                    "shell_metrics_csv": str(curve.shell_metrics_csv),
                }
            )


def _draw_frame(
    path: Path,
    curves: list[Curve],
    visible_count: int,
    title: str,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
    ax.axhline(0.5, color="0.35", ls=":", lw=1.2)
    for index, curve in enumerate(curves[:visible_count]):
        is_new = index == visible_count - 1
        ax.plot(
            curve.frequency,
            curve.fsc,
            color=N_COLORS.get(curve.n_images, "0.2"),
            ls=METHOD_STYLES[curve.method],
            lw=3.0 if is_new else 1.8,
            alpha=1.0 if is_new else 0.58,
            label=curve.label,
        )
    if visible_count > 0:
        new_curve = curves[visible_count - 1]
        ax.text(
            0.02,
            0.05,
            f"added: {METHOD_LABELS[new_curve.method]} {new_curve.n_label}",
            transform=ax.transAxes,
            fontsize=12,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.75"},
        )
    ax.set_title(title, fontsize=14, weight="bold")
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10)
    ax.set_xlim(0.0, 0.40)
    ax.set_ylim(-0.05, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("FSC vs GT")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.88)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _filter_curves_by_method(curves: list[Curve], method: str) -> list[Curve]:
    return [curve for curve in curves if curve.method == method]


def _make_animation(output_dir: Path, curves: list[Curve], movie_stem: str, title: str, subtitle: str, fps: float) -> dict[str, object]:
    if not curves:
        raise RuntimeError(f"cannot make empty FSC animation {movie_stem}")
    frames_dir = output_dir / "frames" / movie_stem
    frame_paths: list[Path] = []
    for visible in range(1, len(curves) + 1):
        frame = frames_dir / f"frame_{visible:03d}.png"
        _draw_frame(frame, curves, visible, title, subtitle)
        frame_paths.append(frame)
    final_png = output_dir / f"{movie_stem}_final_all_curves.png"
    _draw_frame(final_png, curves, len(curves), title, subtitle)

    gif_path = output_dir / f"{movie_stem}.gif"
    mp4_path = output_dir / f"{movie_stem}.mp4"
    ffmpeg_log = output_dir / f"{movie_stem}_ffmpeg.log"
    gif_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frames_dir / "frame_%03d.png"),
        "-vf",
        "palettegen=stats_mode=diff",
        str(output_dir / f"{movie_stem}_palette.png"),
    ]
    palette_path = output_dir / f"{movie_stem}_palette.png"
    gif_cmd_2 = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frames_dir / "frame_%03d.png"),
        "-i",
        str(palette_path),
        "-lavfi",
        "paletteuse=dither=bayer:bayer_scale=5",
        str(gif_path),
    ]
    mp4_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frames_dir / "frame_%03d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]
    with ffmpeg_log.open("w") as log:
        rc1 = subprocess.run(gif_cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        rc2 = subprocess.run(gif_cmd_2, stdout=log, stderr=subprocess.STDOUT).returncode
        rc3 = subprocess.run(mp4_cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return {
        "frames_dir": str(frames_dir),
        "n_frames": len(frame_paths),
        "final_png": str(final_png),
        "gif": str(gif_path) if gif_path.exists() else None,
        "mp4": str(mp4_path) if mp4_path.exists() else None,
        "ffmpeg_log": str(ffmpeg_log),
        "ffmpeg_returncodes": [rc1, rc2, rc3],
    }


def _run_one(
    args: argparse.Namespace,
    tag: str,
    mask_path: Path,
    title: str,
    subtitle: str,
) -> dict[str, object]:
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = _compute_curves(args, mask_path, out_dir)
    _write_summary(out_dir / f"{tag}_fsc_summary.csv", curves)
    animations: dict[str, object] = {}
    if not args.method_only:
        animations["all_methods"] = _make_animation(out_dir, curves, f"{tag}_cumulative_fsc", title, subtitle, args.fps)
    if args.split_by_method or args.method_only:
        for method in METHOD_ORDER:
            method_curves = _filter_curves_by_method(curves, method)
            if not method_curves:
                continue
            method_dir = out_dir / method
            method_title = f"{METHOD_LABELS[method]} | {title}"
            method_subtitle = f"{subtitle} | one frame per available n; curves persist"
            animations[method] = _make_animation(
                method_dir,
                method_curves,
                f"{tag}_{method}_cumulative_fsc",
                method_title,
                method_subtitle,
                args.fps,
            )
    audit = {
        "tag": tag,
        "mask": str(mask_path),
        "summary_csv": str(args.summary_csv),
        "state": args.state,
        "voxel_size": args.voxel_size,
        "split_by_method": bool(args.split_by_method),
        "method_only": bool(args.method_only),
        "curve_order": [
            {"method": curve.method, "n_label": curve.n_label, "fsc05_resolution_A": curve.fsc05_resolution_a}
            for curve in curves
        ],
        "animations": animations,
    }
    (out_dir / "fsc_reveal_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state", type=int, default=50)
    parser.add_argument("--voxel-size", type=float, default=1.25)
    parser.add_argument("--fps", type=float, default=0.75)
    parser.add_argument("--broad-mask", type=Path, default=DEFAULT_BROAD_MASK)
    parser.add_argument("--not-moving-mask", type=Path, default=DEFAULT_NOT_MOVING_MASK)
    parser.add_argument(
        "--mode",
        choices=("moving", "notmoving", "both"),
        default="both",
        help="Which paired FSC reveal movie(s) to make.",
    )
    parser.add_argument(
        "--split-by-method",
        action="store_true",
        help="Also write separate cumulative FSC movies for RECOVAR, cryoDRGN, and 3DFlex.",
    )
    parser.add_argument(
        "--method-only",
        action="store_true",
        help="Only write separate per-method movies, not the combined all-method movie.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, object] = {}
    if args.mode in {"moving", "both"}:
        outputs["moving_broadmask"] = _run_one(
            args,
            "moving_broadmask",
            args.broad_mask,
            "State 50 cumulative FSC reveal | moving-region movie companion",
            f"mask for FSC: {args.broad_mask}",
        )
    if args.mode in {"notmoving", "both"}:
        outputs["notmoving_softmask"] = _run_one(
            args,
            "notmoving_softmask",
            args.not_moving_mask,
            "State 50 cumulative FSC reveal | not-moving movie companion",
            f"mask for FSC: {args.not_moving_mask}",
        )
    summary = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(args.output_dir),
        "outputs": outputs,
    }
    (args.output_dir / "cumulative_fsc_movies_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
