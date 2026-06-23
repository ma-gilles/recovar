#!/usr/bin/env python3
"""Render state-50 method progression with the moving-piece mask.

This is a reproducible wrapper around ``render_chimerax_manifest.py``.  It
builds masked estimate/GT MRCs from the scored method-sweep CSV, writes a
ChimeraX manifest, renders each map with a GT overlay, then makes labeled PNGs,
contact sheets, and simple animations.

Example:

    python scripts/experiments/spike_fullatom_method_benchmark/render_state50_moving_mask_progression.py \
        --run-chimerax
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY_CSV = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "not_moving_softmask_fsc_resolution_20260601/"
    "not_moving_softmask_noise3_fsc_summary.csv"
)
DEFAULT_VIEW_JSON = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_notmoving_view_20260601/"
    "not_moving_view_extracted.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_moving_mask_20260601"
)
DEFAULT_BROAD_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
RENDERER = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "spike_fullatom_method_benchmark"
    / "render_chimerax_manifest.py"
)

METHOD_ORDER = ["recovar", "cryodrgn", "3dflex"]
METHOD_LABELS = {
    "recovar": "RECOVAR",
    "cryodrgn": "cryoDRGN",
    "3dflex": "3DFlex",
}
METHOD_COLORS = {
    "recovar": "#1b9e77",
    "cryodrgn": "#d95f02",
    "3dflex": "#7570b3",
}
N_ORDER = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
N_LABELS = {
    10_000: "10k",
    30_000: "30k",
    100_000: "100k",
    300_000: "300k",
    1_000_000: "1M",
    3_000_000: "3M",
}


@dataclass(frozen=True)
class RenderEntry:
    method: str
    n_images: int
    n_label: str
    state: int
    estimate: Path
    gt: Path
    mask: Path


def _recovar_filtered_path(path: Path) -> Path:
    if path.name == "state000_unfil.mrc":
        filtered = path.with_name("state000.mrc")
        if filtered.exists():
            return filtered
    return path


def _read_summary_rows(path: Path, state: int) -> list[RenderEntry]:
    entries: list[RenderEntry] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["state"]) != state:
                continue
            method = row["method"]
            if method not in METHOD_ORDER:
                continue
            gt = Path(row["gt"])
            run_dir = gt.parents[1]
            mask = run_dir / "05_masks" / "focus_mask_moving.mrc"
            estimate = Path(row["estimate"])
            if method == "recovar":
                estimate = _recovar_filtered_path(estimate)
            entry = RenderEntry(
                method=method,
                n_images=int(row["n_images"]),
                n_label=row["n_label"],
                state=state,
                estimate=estimate,
                gt=gt,
                mask=mask,
            )
            entries.append(entry)
    order = {method: i for i, method in enumerate(METHOD_ORDER)}
    entries.sort(key=lambda e: (order[e.method], e.n_images))
    return entries


def _copy_masked_mrc(volume_path: Path, mask_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.open(volume_path, permissive=True) as vol_mrc:
        volume = np.asarray(vol_mrc.data, dtype=np.float32)
        voxel_size = vol_mrc.voxel_size
        origin = vol_mrc.header.origin.copy()
    with mrcfile.open(mask_path, permissive=True) as mask_mrc:
        mask = np.asarray(mask_mrc.data, dtype=np.float32)
    if volume.shape != mask.shape:
        raise ValueError(f"shape mismatch for {volume_path}: volume {volume.shape}, mask {mask.shape}")
    masked = np.asarray(volume * mask, dtype=np.float32)
    with mrcfile.new(output_path, overwrite=True) as out_mrc:
        out_mrc.set_data(masked)
        out_mrc.voxel_size = voxel_size
        out_mrc.header.origin = origin


def _masked_paths(out_dir: Path, entry: RenderEntry, noise_level: str) -> tuple[Path, Path]:
    n_tag = f"n{entry.n_images:08d}"
    est = (
        out_dir
        / "masked_mrc"
        / entry.method
        / f"state{entry.state:04d}_noise{noise_level}_{entry.method}_{n_tag}_moving_masked.mrc"
    )
    gt = (
        out_dir
        / "masked_mrc"
        / "ground_truth"
        / n_tag
        / f"gt_state{entry.state:04d}_{n_tag}_moving_masked.mrc"
    )
    return est, gt


def _prepare_masked_volumes(
    entries: list[RenderEntry],
    out_dir: Path,
    noise_level: str,
) -> dict[tuple[str, int], tuple[Path, Path]]:
    paths: dict[tuple[str, int], tuple[Path, Path]] = {}
    for entry in entries:
        for path in (entry.estimate, entry.gt, entry.mask):
            if not path.exists():
                raise FileNotFoundError(path)
        est_out, gt_out = _masked_paths(out_dir, entry, noise_level)
        _copy_masked_mrc(entry.estimate, entry.mask, est_out)
        _copy_masked_mrc(entry.gt, entry.mask, gt_out)
        paths[(entry.method, entry.n_images)] = (est_out, gt_out)
    return paths


def _write_manifest(
    entries: list[RenderEntry],
    masked_paths: dict[tuple[str, int], tuple[Path, Path]],
    out_dir: Path,
    contour_level: float,
    gt_color: str,
    noise_level: str,
    bfactor: str,
) -> Path:
    manifest = out_dir / "manifest_chimerax_state50_moving_mask.csv"
    fieldnames = [
        "noise_level",
        "bfactor",
        "collection",
        "n_images",
        "n_label",
        "state",
        "method",
        "method_label",
        "role",
        "volume_path",
        "source_volume_path",
        "mask_path",
        "render_name",
        "contour_level",
        "color",
    ]
    rows: list[dict[str, str]] = []
    collection = f"noise{noise_level}_b{bfactor}_state50_moving_mask"
    for entry in entries:
        est_out, gt_out = masked_paths[(entry.method, entry.n_images)]
        n_tag = f"n{entry.n_images:08d}"
        rows.append(
            {
                "noise_level": noise_level,
                "bfactor": bfactor,
                "collection": collection,
                "n_images": str(entry.n_images),
                "n_label": entry.n_label,
                "state": str(entry.state),
                "method": entry.method,
                "method_label": METHOD_LABELS[entry.method],
                "role": "estimate",
                "volume_path": str(est_out),
                "source_volume_path": str(entry.estimate),
                "mask_path": str(entry.mask),
                "render_name": f"{entry.method}/state{entry.state:04d}_noise{noise_level}_{entry.method}_{n_tag}.png",
                "contour_level": f"{contour_level:.8g}",
                "color": METHOD_COLORS[entry.method],
            }
        )
        rows.append(
            {
                "noise_level": noise_level,
                "bfactor": bfactor,
                "collection": collection,
                "n_images": str(entry.n_images),
                "n_label": entry.n_label,
                "state": str(entry.state),
                "method": "ground_truth",
                "method_label": "GT",
                "role": "gt",
                "volume_path": str(gt_out),
                "source_volume_path": str(entry.gt),
                "mask_path": str(entry.mask),
                "render_name": f"ground_truth/state{entry.state:04d}_noise{noise_level}_gt_{n_tag}.png",
                "contour_level": f"{contour_level:.8g}",
                "color": gt_color,
            }
        )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _dft3(volume: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume)))


def _shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    grid = np.indices(shape, dtype=np.float32)
    center = np.asarray(shape, dtype=np.float32)[:, None, None, None] // 2
    labels = np.rint(np.sqrt(np.sum((grid - center) ** 2, axis=0))).astype(np.int32)
    n_shells = shape[0] // 2 - 1
    return np.clip(labels, 0, n_shells - 1), n_shells


def _load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as mrc:
        return np.asarray(mrc.data, dtype=np.float32)


def _shell_metrics(estimate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    labels, n_shells = _shell_labels(estimate.shape)
    estimate_ft = _dft3(estimate * mask)
    target_ft = _dft3(target * mask)
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


def _last_good_resolution(frequency: np.ndarray, values: np.ndarray, threshold: float, higher_is_better: bool) -> float:
    valid = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0)
    frequency = np.asarray(frequency, dtype=np.float64)[valid]
    values = np.asarray(values, dtype=np.float64)[valid]
    if frequency.size == 0:
        return float("nan")
    good = values >= threshold if higher_is_better else values <= threshold
    indices = np.flatnonzero(good)
    if indices.size == 0:
        return float("nan")
    return float(1.0 / frequency[int(indices[-1])])


def _write_shell_csv(path: Path, metrics: dict[str, np.ndarray], frequency: np.ndarray) -> None:
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
                "cumulative_relative_error",
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
                    metrics["cumulative_relative_error"][index],
                    metrics["target_power"][index],
                    metrics["diff_power"][index],
                ]
            )


def _score_entry(
    entry: RenderEntry,
    broad_mask: np.ndarray,
    broad_mask_path: Path,
    out_dir: Path,
    voxel_size: float,
) -> dict[str, object]:
    estimate = _load_mrc(entry.estimate)
    target = _load_mrc(entry.gt)
    if estimate.shape != target.shape or estimate.shape != broad_mask.shape:
        raise ValueError(
            f"FSC shape mismatch for {entry.method} {entry.n_label}: "
            f"estimate={estimate.shape}, gt={target.shape}, mask={broad_mask.shape}"
        )
    metrics = _shell_metrics(estimate, target, broad_mask)
    frequency = np.arange(metrics["fsc"].size, dtype=np.float64) / (estimate.shape[0] * voxel_size)
    fsc05 = _last_good_resolution(frequency, metrics["fsc"], 0.5, higher_is_better=True)
    err05 = _last_good_resolution(frequency, metrics["relative_error"], 0.5, higher_is_better=False)
    shell_csv = (
        out_dir
        / "broadmask_fsc"
        / "shell_metrics"
        / entry.method
        / f"n{entry.n_images:08d}"
        / f"state{entry.state:04d}"
        / "shell_metrics.csv"
    )
    _write_shell_csv(shell_csv, metrics, frequency)
    return {
        "method": entry.method,
        "method_label": METHOD_LABELS[entry.method],
        "n_images": entry.n_images,
        "n_label": entry.n_label,
        "state": entry.state,
        "estimate": str(entry.estimate),
        "gt": str(entry.gt),
        "mask": str(broad_mask_path),
        "shell_metrics_csv": str(shell_csv),
        "fsc05_resolution_A": fsc05,
        "relerr05_resolution_A": err05,
        "frequency": frequency,
        "metrics": metrics,
    }


def _plot_fsc_group(out_dir: Path, stem: str, title: str, curves: list[dict[str, object]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(1, len(curves))))
    for color, row in zip(colors, curves):
        metrics = row["metrics"]
        frequency = row["frequency"]
        label = f"{row['n_label']} ({float(row['fsc05_resolution_A']):.2f} A)"
        axes[0].plot(frequency, metrics["fsc"], color=color, lw=2.0, label=label)
        axes[1].semilogy(
            frequency,
            metrics["relative_error"],
            color=color,
            lw=2.0,
            label=f"{row['n_label']} ({float(row['relerr05_resolution_A']):.2f} A)",
        )
    axes[0].axhline(0.5, color="0.35", ls="--", lw=1.0)
    axes[0].set_title("broad-mask FSC vs GT")
    axes[0].set_xlabel("spatial frequency (1/A)")
    axes[0].set_ylabel("FSC")
    axes[0].set_xlim(0.0, 0.40)
    axes[0].set_ylim(-0.05, 1.03)
    axes[0].legend(title="n, FSC0.5", fontsize=8)
    axes[1].axhline(0.5, color="0.35", ls="--", lw=1.0)
    axes[1].set_title("broad-mask relative Fourier shell error")
    axes[1].set_xlabel("spatial frequency (1/A)")
    axes[1].set_ylabel("relative error")
    axes[1].set_xlim(0.0, 0.40)
    axes[1].set_ylim(1e-3, 1e3)
    axes[1].legend(title="n, error<=0.5", fontsize=8)
    fig.suptitle(title, fontsize=12, weight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=220)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    return png


def _plot_fsc_all_methods(out_dir: Path, rows: list[dict[str, object]], state: int, noise_level: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        10_000: "#4c1d7a",
        30_000: "#2a6f91",
        100_000: "#2fb47c",
        300_000: "#c8e020",
        1_000_000: "#ff9f1c",
        3_000_000: "#d00000",
    }
    styles = {"recovar": "-", "cryodrgn": "--", "3dflex": "-."}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0), constrained_layout=True)
    for row in sorted(rows, key=lambda r: (METHOD_ORDER.index(str(r["method"])), int(r["n_images"]))):
        method = str(row["method"])
        n_images = int(row["n_images"])
        label = f"{METHOD_LABELS[method]} {row['n_label']}"
        frequency = row["frequency"]
        metrics = row["metrics"]
        axes[0].plot(
            frequency,
            metrics["fsc"],
            color=colors.get(n_images, "0.2"),
            ls=styles[method],
            lw=2.0,
            label=f"{label} ({float(row['fsc05_resolution_A']):.2f} A)",
        )
        axes[1].semilogy(
            frequency,
            metrics["relative_error"],
            color=colors.get(n_images, "0.2"),
            ls=styles[method],
            lw=2.0,
            label=f"{label} ({float(row['relerr05_resolution_A']):.2f} A)",
        )
    for ax in axes:
        ax.set_xlim(0.0, 0.40)
        ax.axhline(0.5, color="0.35", ls=":", lw=1.0)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_title("broad-mask FSC vs GT")
    axes[0].set_xlabel("spatial frequency (1/A)")
    axes[0].set_ylabel("FSC")
    axes[0].set_ylim(-0.05, 1.03)
    axes[1].set_title("broad-mask relative Fourier shell error")
    axes[1].set_xlabel("spatial frequency (1/A)")
    axes[1].set_ylabel("relative error")
    axes[1].set_ylim(1e-3, 1e3)
    fig.suptitle(
        f"noise={noise_level} state {state} | broad_mask.mrc | RECOVAR filtered, cryoDRGN/3DFlex standard outputs",
        fontsize=12,
        weight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"noise{noise_level}_state{state:04d}_broadmask_fsc_error_all_methods_recovar_filtered.png"
    fig.savefig(png, dpi=220)
    fig.savefig(out_dir / f"noise{noise_level}_state{state:04d}_broadmask_fsc_error_all_methods_recovar_filtered.pdf")
    plt.close(fig)
    return png


def _make_broadmask_fsc_plots(args: argparse.Namespace, entries: list[RenderEntry]) -> dict[str, object]:
    if not args.broad_mask.exists():
        raise FileNotFoundError(args.broad_mask)
    broad_mask = _load_mrc(args.broad_mask)
    rows = [
        _score_entry(entry, broad_mask, args.broad_mask, args.output_dir, args.voxel_size)
        for entry in entries
    ]
    fsc_dir = args.output_dir / "broadmask_fsc"
    summary_csv = fsc_dir / f"noise{args.noise_level}_state{args.state:04d}_broadmask_fsc_summary_recovar_filtered.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "method",
        "method_label",
        "n_images",
        "n_label",
        "state",
        "fsc05_resolution_A",
        "relerr05_resolution_A",
        "estimate",
        "gt",
        "mask",
        "shell_metrics_csv",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    method_plots = {}
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        method_plots[method] = str(
            _plot_fsc_group(
                fsc_dir,
                f"noise{args.noise_level}_state{args.state:04d}_broadmask_fsc_error_{method}_recovar_filtered",
                f"{METHOD_LABELS[method]} | noise={args.noise_level} state {args.state} | broad_mask.mrc",
                method_rows,
            )
        )
    all_methods_plot = _plot_fsc_all_methods(fsc_dir, rows, args.state, args.noise_level)
    return {
        "summary_csv": str(summary_csv),
        "method_plots": method_plots,
        "all_methods_plot": str(all_methods_plot),
        "mask": str(args.broad_mask),
    }


def _run_chimerax(args: argparse.Namespace, manifest: Path) -> Path:
    raw_dir = args.output_dir / "png_raw"
    log_dir = args.output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    script_payload = (
        f"{RENDERER} "
        f"--manifest {manifest} "
        f"--output-dir {raw_dir} "
        f"--view-json {args.view_json} "
        f"--width {args.width} "
        f"--height {args.height} "
        f"--supersample {args.supersample} "
        f"--background {shlex.quote(args.background)} "
        f"--roles estimate "
        f"--states {args.state} "
        f"--overlay-gt "
        f"--gt-color {shlex.quote(args.gt_color)} "
        f"--gt-opacity {args.gt_opacity} "
        f"--fallback-level {args.contour_level}"
    )
    bash_cmd = (
        f"module purge; module load {shlex.quote(args.chimerax_module)}; "
        f"chimerax --nogui --offscreen --script {shlex.quote(script_payload)}"
    )
    log_path = log_dir / "chimerax_render.log"
    with log_path.open("w") as log:
        proc = subprocess.run(["bash", "-lc", bash_cmd], stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"ChimeraX render failed with code {proc.returncode}; see {log_path}")
    return log_path


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_BIG = _font(42, True)
FONT_MED = _font(34, True)
FONT_SMALL = _font(26, False)
FONT_CELL = _font(22, True)
DURATION_MS = 1500


def _read_manifest_estimates(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row["role"] == "estimate"]


def _raw_render_path(raw_dir: Path, row: dict[str, str]) -> Path:
    render_name = Path(row["render_name"])
    # render_chimerax_manifest.py intentionally flattens render_name when GT is
    # overlaid, so keep this in sync with its _render_name implementation.
    return raw_dir / f"{render_name.stem}_gt_overlay{render_name.suffix}"


def _labeled_path(labeled_dir: Path, row: dict[str, str]) -> Path:
    return labeled_dir / row["method"] / _raw_render_path(Path("."), row).name


def _label_one(raw_dir: Path, labeled_dir: Path, row: dict[str, str], contour_level: float, gt_opacity: float) -> Path:
    src = _raw_render_path(raw_dir, row)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = _labeled_path(labeled_dir, row)
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    title = f"{METHOD_LABELS[row['method']]}  {row['n_label']}  state {row['state']}"
    subtitle = f"moving-mask render; estimate and GT contour {contour_level:.4f}; GT gray opacity {gt_opacity:.1f}"
    margin = 28
    pad_x, pad_y = 22, 14
    title_box = draw.textbbox((0, 0), title, font=FONT_BIG)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=FONT_SMALL)
    width = max(title_box[2], subtitle_box[2]) + 2 * pad_x
    height = (title_box[3] - title_box[1]) + (subtitle_box[3] - subtitle_box[1]) + 3 * pad_y
    draw.rounded_rectangle(
        (margin, margin, margin + width, margin + height),
        radius=10,
        fill=(255, 255, 255, 210),
        outline=(0, 0, 0, 90),
        width=2,
    )
    draw.text((margin + pad_x, margin + pad_y), title, fill=(0, 0, 0, 255), font=FONT_BIG)
    draw.text(
        (margin + pad_x, margin + 2 * pad_y + title_box[3] - title_box[1]),
        subtitle,
        fill=(40, 40, 40, 255),
        font=FONT_SMALL,
    )
    image.convert("RGB").save(dst, quality=95)
    return dst


def _make_thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _placeholder(size: tuple[int, int], text: str) -> Image.Image:
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(190, 190, 190), width=2)
    box = draw.textbbox((0, 0), text, font=FONT_MED)
    draw.text(
        ((size[0] - (box[2] - box[0])) // 2, (size[1] - (box[3] - box[1])) // 2),
        text,
        fill=(90, 90, 90),
        font=FONT_MED,
    )
    return image


def _contact_sheet(index: dict[tuple[str, int], Path], panels_dir: Path, noise_level: str, state: int) -> Path:
    panels_dir.mkdir(parents=True, exist_ok=True)
    cell = (420, 315)
    left = 170
    top = 80
    sheet = Image.new("RGB", (left + cell[0] * len(N_ORDER), top + cell[1] * len(METHOD_ORDER)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), f"Spike benchmark noise={noise_level}, state {state}: moving-mask estimate over GT overlay", fill=(0, 0, 0), font=FONT_MED)
    for j, n_images in enumerate(N_ORDER):
        label = N_LABELS[n_images]
        box = draw.textbbox((0, 0), label, font=FONT_CELL)
        draw.text(
            (left + j * cell[0] + (cell[0] - (box[2] - box[0])) // 2, 48),
            label,
            fill=(0, 0, 0),
            font=FONT_CELL,
        )
    for i, method in enumerate(METHOD_ORDER):
        y = top + i * cell[1]
        draw.text((20, y + cell[1] // 2 - 14), METHOD_LABELS[method], fill=(0, 0, 0), font=FONT_CELL)
        for j, n_images in enumerate(N_ORDER):
            path = index.get((method, n_images))
            tile = _make_thumb(path, cell) if path else _placeholder(cell, "not available")
            x = left + j * cell[0]
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y, x + cell[0] - 1, y + cell[1] - 1), outline=(210, 210, 210), width=1)
    output = panels_dir / f"state{state:04d}_noise{noise_level}_moving_mask_method_by_n_contact_sheet.png"
    sheet.save(output, quality=95)
    return output


def _save_gif(paths: list[Path], output: Path, size: tuple[int, int] | None = None) -> None:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if size is not None:
            image.thumbnail(size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", size, "white")
            canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
            image = canvas
        images.append(image)
    if not images:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=DURATION_MS, loop=0, optimize=True)


def _progression_frames(index: dict[tuple[str, int], Path], frames_dir: Path, noise_level: str, state: int) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cell = (640, 480)
    left = 160
    top = 70
    frames = []
    for n_images in N_ORDER:
        frame = Image.new("RGB", (left + cell[0], top + len(METHOD_ORDER) * cell[1]), "white")
        draw = ImageDraw.Draw(frame)
        draw.text((20, 18), f"noise={noise_level}  state {state}  moving mask  n={N_LABELS[n_images]}", fill=(0, 0, 0), font=FONT_MED)
        for i, method in enumerate(METHOD_ORDER):
            y = top + i * cell[1]
            draw.text((20, y + cell[1] // 2 - 14), METHOD_LABELS[method], fill=(0, 0, 0), font=FONT_CELL)
            path = index.get((method, n_images))
            tile = _make_thumb(path, cell) if path else _placeholder(cell, "not available")
            frame.paste(tile, (left, y))
            draw.rectangle((left, y, left + cell[0] - 1, y + cell[1] - 1), outline=(210, 210, 210), width=1)
        output = frames_dir / f"all_methods_n{n_images:08d}.png"
        frame.save(output, quality=95)
        frames.append(output)
    return frames


def _compose_outputs(args: argparse.Namespace, manifest: Path) -> dict[str, object]:
    raw_dir = args.output_dir / "png_raw"
    labeled_dir = args.output_dir / "png_labeled"
    panels_dir = args.output_dir / "panels"
    animations_dir = args.output_dir / "animations"
    frames_dir = args.output_dir / "animation_frames"
    rows = _read_manifest_estimates(manifest)
    labeled_paths = [_label_one(raw_dir, labeled_dir, row, args.contour_level, args.gt_opacity) for row in rows]
    index = {
        (row["method"], int(row["n_images"])): _labeled_path(labeled_dir, row)
        for row in rows
    }
    contact = _contact_sheet(index, panels_dir, args.noise_level, args.state)
    per_method_gifs: dict[str, str] = {}
    for method in METHOD_ORDER:
        paths = [index[(method, n_images)] for n_images in N_ORDER if (method, n_images) in index]
        output = animations_dir / f"state{args.state:04d}_noise{args.noise_level}_moving_mask_{method}_progression.gif"
        _save_gif(paths, output, size=(960, 720))
        if output.exists():
            per_method_gifs[method] = str(output)
    frames = _progression_frames(index, frames_dir, args.noise_level, args.state)
    all_gif = animations_dir / f"state{args.state:04d}_noise{args.noise_level}_moving_mask_all_methods_progression.gif"
    _save_gif(frames, all_gif, size=(800, 1800))
    all_mp4 = animations_dir / f"state{args.state:04d}_noise{args.noise_level}_moving_mask_all_methods_progression.mp4"
    ffmpeg_log = args.output_dir / "logs" / "ffmpeg_all_methods.log"
    ffmpeg_log.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(1 / 1.5),
        "-pattern_type",
        "glob",
        "-i",
        str(frames_dir / "all_methods_n*.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-movflags",
        "+faststart",
        str(all_mp4),
    ]
    with ffmpeg_log.open("w") as log:
        proc = subprocess.run(ffmpeg_cmd, stdout=log, stderr=subprocess.STDOUT)
    return {
        "labeled_png_dir": str(labeled_dir),
        "n_labeled_png": len(labeled_paths),
        "contact_sheet": str(contact),
        "per_method_gifs": per_method_gifs,
        "all_methods_gif": str(all_gif),
        "all_methods_mp4": str(all_mp4) if proc.returncode == 0 else None,
        "all_method_frames": [str(path) for path in frames],
        "duration_ms": DURATION_MS,
        "ffmpeg_returncode": proc.returncode,
        "ffmpeg_command": ffmpeg_cmd,
        "ffmpeg_log": str(ffmpeg_log),
    }


def _write_audit(args: argparse.Namespace, entries: list[RenderEntry], manifest: Path, outputs: dict[str, object] | None) -> Path:
    audit = args.output_dir / "COMMAND_AUDIT.md"
    counts = {}
    for entry in entries:
        counts[entry.method] = counts.get(entry.method, 0) + 1
    missing = {
        method: [N_LABELS[n] for n in N_ORDER if not any(e.method == method and e.n_images == n for e in entries)]
        for method in METHOD_ORDER
    }
    lines = [
        "# State 50 Moving-Mask ChimeraX Render Audit",
        "",
        f"- Repo script: `{Path(__file__).resolve()}`",
        f"- Summary CSV: `{args.summary_csv}`",
        f"- Output dir: `{args.output_dir}`",
        f"- View JSON: `{args.view_json}`",
        f"- ChimeraX renderer: `{RENDERER}`",
        f"- State: `{args.state}`",
        f"- Contour level for all estimates and GT: `{args.contour_level}`",
        f"- GT overlay opacity: `{args.gt_opacity}`",
        f"- Render mask: per-run `05_masks/focus_mask_moving.mrc`",
        f"- FSC mask: `{args.broad_mask}`",
        "- RECOVAR source: filtered `state000.mrc` when the source CSV points at `state000_unfil.mrc`",
        f"- Counts: `{json.dumps(counts, sort_keys=True)}`",
        f"- Missing n by method: `{json.dumps(missing, sort_keys=True)}`",
        f"- Manifest: `{manifest}`",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(shlex.quote(x) for x in sys.argv),
        "```",
    ]
    if outputs is not None:
        lines.extend(["", "## Outputs", "", "```json", json.dumps(outputs, indent=2, sort_keys=True), "```"])
    audit.write_text("\n".join(lines) + "\n")
    return audit


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view-json", type=Path, default=DEFAULT_VIEW_JSON)
    parser.add_argument("--state", type=int, default=50)
    parser.add_argument("--noise-level", default="3")
    parser.add_argument("--bfactor", default="80")
    parser.add_argument("--contour-level", type=float, default=0.013)
    parser.add_argument("--gt-color", default="#777777")
    parser.add_argument("--gt-opacity", type=float, default=0.3)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--background", default="white")
    parser.add_argument("--chimerax-module", default="chimerax/1.9")
    parser.add_argument("--broad-mask", type=Path, default=DEFAULT_BROAD_MASK)
    parser.add_argument("--voxel-size", type=float, default=1.25)
    parser.add_argument("--run-chimerax", action="store_true")
    parser.add_argument("--skip-fsc", action="store_true")
    parser.add_argument("--skip-compose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = _read_summary_rows(args.summary_csv, args.state)
    if not entries:
        raise RuntimeError(f"no render entries found in {args.summary_csv} for state {args.state}")
    args.noise_level = str(args.noise_level)
    args.bfactor = str(args.bfactor)
    masked_paths = _prepare_masked_volumes(entries, args.output_dir, args.noise_level)
    manifest = _write_manifest(
        entries,
        masked_paths,
        args.output_dir,
        args.contour_level,
        args.gt_color,
        args.noise_level,
        args.bfactor,
    )
    fsc_outputs = None
    if not args.skip_fsc:
        fsc_outputs = _make_broadmask_fsc_plots(args, entries)
    render_log = None
    if args.run_chimerax:
        render_log = _run_chimerax(args, manifest)
    outputs = None
    if not args.skip_compose:
        outputs = _compose_outputs(args, manifest)
        if render_log is not None:
            outputs["chimerax_log"] = str(render_log)
    if fsc_outputs is not None:
        if outputs is None:
            outputs = {}
        outputs["broadmask_fsc"] = fsc_outputs
    audit = _write_audit(args, entries, manifest, outputs)
    summary = {
        "output_dir": str(args.output_dir),
        "manifest": str(manifest),
        "audit": str(audit),
        "render_log": str(render_log) if render_log else None,
        "n_entries": len(entries),
        "outputs": outputs,
    }
    (args.output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
