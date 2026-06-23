#!/usr/bin/env python3
"""Refresh uniform noise=3 state-50 volume/FSC movies through available sizes.

This differs from ``refresh_state50_presentation_movies_upto1m.py`` in two
ways:

1. it includes 3M for methods with completed outputs;
2. it does not invent unavailable panels.  cryoDRGN stops at 1M in the current
   tree, while RECOVAR and 3DFlex continue to 3M.

The movie frames are intentionally minimally annotated: the only large text is
the current number of images, plus resolution numbers inside FSC panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
RENDERER = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "spike_fullatom_method_benchmark"
    / "render_chimerax_manifest.py"
)

SWEEP_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531")
SOURCE_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise3_b80_20260531")
BROAD_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
NOT_MOVING_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/not_moving_spike_mask_soft_20260601.mrc"
)
VIEW_JSON = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_moving_mask_zoomed_view_20260601/"
    "zoomed_moving_view_extracted.json"
)
NOT_MOVING_VIEW_JSON = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_notmoving_view_20260601/"
    "not_moving_view_extracted.json"
)

METHOD_ORDER = ("recovar", "cryodrgn", "3dflex")
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
GT_COLOR = "#8f8f8f"
N_ORDER = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)
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
    30_000: "#287796",
    100_000: "#2bb07f",
    300_000: "#c9df1a",
    1_000_000: "#ff9f1c",
    3_000_000: "#c1121f",
}
THREE_DFLEX_STATE50_JOBS = {
    10_000: "J504",
    30_000: "J506",
    100_000: "J518",
    300_000: "J526",
    1_000_000: "J528",
    3_000_000: "J544",
}
CRYODRGN_STATE50_FRAME = {
    10_000: "gt_label_002.mrc",
    30_000: "gt_label_002.mrc",
    100_000: "gt_label_002.mrc",
    300_000: "gt_label_002.mrc",
    1_000_000: "gt_label_050.mrc",
}


@dataclass(frozen=True)
class Entry:
    method: str
    n_images: int
    estimate: Path
    gt: Path

    @property
    def n_label(self) -> str:
        return N_LABELS[self.n_images]


@dataclass(frozen=True)
class MaskConfig:
    key: str
    label: str
    mask: Path
    view_json: Path
    contour_level: float


MASK_CONFIGS = {
    "broad": MaskConfig("broad", "broad_mask", BROAD_MASK, VIEW_JSON, 0.013),
    "notmoving": MaskConfig(
        "notmoving",
        "not_moving_mask",
        NOT_MOVING_MASK,
        NOT_MOVING_VIEW_JSON,
        0.0213,
    ),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_N = _font(70, True)
FONT_RES = _font(28, True)


def _gt_path(n_images: int, state: int = 50) -> Path:
    n_tag = f"n{n_images:08d}"
    return SOURCE_ROOT / n_tag / "runs" / f"{n_tag}_seed0000" / "04_ground_truth" / f"gt_vol{state:04d}.mrc"


def _recovar_path(n_images: int) -> Path:
    return (
        SWEEP_ROOT
        / "recovar"
        / f"n{n_images:08d}"
        / "compute_state_zdim4_noreg_focus"
        / "state0050"
        / "state000.mrc"
    )


def _cryodrgn_path(n_images: int) -> Path:
    return (
        SWEEP_ROOT
        / f"n{n_images:08d}"
        / "evaluation"
        / "cryodrgn"
        / "zdim1"
        / "decoded_volumes"
        / "labels_mean_z_epoch019"
        / CRYODRGN_STATE50_FRAME[n_images]
    )


def _three_dflex_path(n_images: int) -> Path:
    job = THREE_DFLEX_STATE50_JOBS[n_images]
    return Path(
        f"/projects/CRYOEM/singerlab/mg6942/CS-testres/{job}/"
        f"{job}_series_000/{job}_series_000_frame_002.mrc"
    )


def discover_entries() -> list[Entry]:
    entries: list[Entry] = []
    for n_images in N_ORDER:
        for method, path_fn in (
            ("recovar", _recovar_path),
            ("3dflex", _three_dflex_path),
        ):
            estimate = path_fn(n_images)
            gt = _gt_path(n_images)
            if estimate.exists() and gt.exists():
                entries.append(Entry(method, n_images, estimate, gt))
        if n_images in CRYODRGN_STATE50_FRAME:
            estimate = _cryodrgn_path(n_images)
            gt = _gt_path(n_images)
            if estimate.exists() and gt.exists():
                entries.append(Entry("cryodrgn", n_images, estimate, gt))
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    entries.sort(key=lambda e: (e.n_images, order[e.method]))
    return entries


def _load_mrc(path: Path) -> tuple[np.ndarray, object, object]:
    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
        voxel_size = handle.voxel_size.copy()
        origin = handle.header.origin.copy()
    return data, voxel_size, origin


def _write_mrc(path: Path, data: np.ndarray, voxel_size: object, origin: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(data, dtype=np.float32))
        handle.voxel_size = voxel_size
        handle.header.origin = origin


def _prepare_masked_volumes(
    entries: list[Entry],
    output_dir: Path,
    mask_config: MaskConfig,
) -> dict[tuple[str, int], tuple[Path, Path]]:
    mask, _, _ = _load_mrc(mask_config.mask)
    out: dict[tuple[str, int], tuple[Path, Path]] = {}
    gt_cache: dict[int, Path] = {}
    for entry in entries:
        estimate, voxel_size, origin = _load_mrc(entry.estimate)
        if estimate.shape != mask.shape:
            raise ValueError(f"shape mismatch for {entry.estimate}: volume={estimate.shape}, mask={mask.shape}")
        est_out = (
            output_dir
            / "masked_mrc"
            / mask_config.key
            / entry.method
            / f"state0050_noise3_{entry.method}_n{entry.n_images:08d}_{mask_config.key}_masked.mrc"
        )
        if not est_out.exists():
            _write_mrc(est_out, estimate * mask, voxel_size, origin)

        if entry.n_images not in gt_cache:
            gt, gt_voxel_size, gt_origin = _load_mrc(entry.gt)
            if gt.shape != mask.shape:
                raise ValueError(f"shape mismatch for {entry.gt}: volume={gt.shape}, mask={mask.shape}")
            gt_out = (
                output_dir
                / "masked_mrc"
                / mask_config.key
                / "ground_truth"
                / f"gt_state0050_noise3_n{entry.n_images:08d}_{mask_config.key}_masked.mrc"
            )
            if not gt_out.exists():
                _write_mrc(gt_out, gt * mask, gt_voxel_size, gt_origin)
            gt_cache[entry.n_images] = gt_out
        # The masked GT is written for download/provenance, but render overlays
        # use the full GT map so density outside the focus mask remains visible.
        out[(entry.method, entry.n_images)] = (est_out, entry.gt)
    return out


def _write_manifest(
    entries: list[Entry],
    masked_paths: dict[tuple[str, int], tuple[Path, Path]],
    output_dir: Path,
    mask_config: MaskConfig,
) -> Path:
    manifest = output_dir / f"manifest_state50_uniform_noise3_available_{mask_config.key}.csv"
    fields = [
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
    collection = f"uniform_noise3_b80_state50_{mask_config.label}_available"
    gt_seen: set[int] = set()
    for entry in entries:
        est_out, gt_out = masked_paths[(entry.method, entry.n_images)]
        n_tag = f"n{entry.n_images:08d}"
        rows.append(
            {
                "noise_level": "3",
                "bfactor": "80",
                "collection": collection,
                "n_images": str(entry.n_images),
                "n_label": entry.n_label,
                "state": "50",
                "method": entry.method,
                "method_label": METHOD_LABELS[entry.method],
                "role": "estimate",
                "volume_path": str(est_out),
                "source_volume_path": str(entry.estimate),
                "mask_path": str(mask_config.mask),
                "render_name": f"{entry.method}/state0050_noise3_{entry.method}_{n_tag}.png",
                "contour_level": f"{mask_config.contour_level:.8g}",
                "color": METHOD_COLORS[entry.method],
            }
        )
        if entry.n_images not in gt_seen:
            rows.append(
                {
                    "noise_level": "3",
                    "bfactor": "80",
                    "collection": collection,
                    "n_images": str(entry.n_images),
                    "n_label": entry.n_label,
                    "state": "50",
                    "method": "ground_truth",
                    "method_label": "GT",
                    "role": "gt",
                    "volume_path": str(gt_out),
                    "source_volume_path": str(entry.gt),
                    "mask_path": str(mask_config.mask),
                    "render_name": f"ground_truth/gt_state0050_noise3_{n_tag}.png",
                    "contour_level": f"{mask_config.contour_level:.8g}",
                    "color": GT_COLOR,
                }
            )
            gt_seen.add(entry.n_images)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _run_chimerax(manifest: Path, raw_dir: Path, args: argparse.Namespace, mask_config: MaskConfig) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir.parent / "logs" / "chimerax_render.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f"{RENDERER} "
        f"--manifest {manifest} "
        f"--output-dir {raw_dir} "
        f"--view-json {mask_config.view_json} "
        f"--width {args.render_width} "
        f"--height {args.render_height} "
        f"--supersample {args.supersample} "
        f"--background white "
        f"--roles estimate "
        f"--states 50 "
        f"--overlay-gt "
        f"--gt-color {GT_COLOR} "
        f"--gt-opacity 0.3 "
        f"--fallback-level {mask_config.contour_level:.8g}"
    )
    command = (
        f"module purge; module load {shlex.quote(args.chimerax_module)}; "
        f"chimerax --nogui --offscreen --script {shlex.quote(payload)}"
    )
    with log_path.open("w") as log:
        log.write(command + "\n\n")
        proc = subprocess.run(["bash", "-lc", command], stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"ChimeraX render failed with code {proc.returncode}; see {log_path}")
    return log_path


def _rendered_path(raw_dir: Path, entry: Entry) -> Path:
    n_tag = f"n{entry.n_images:08d}"
    return raw_dir / f"state0050_noise3_{entry.method}_{n_tag}_gt_overlay.png"


def _trim_white(image: Image.Image, threshold: int = 248, pad: int = 24) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)
    if not np.any(mask):
        return image
    ys, xs = np.where(mask)
    left = max(int(xs.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, image.width)
    top = max(int(ys.min()) - pad, 0)
    bottom = min(int(ys.max()) + pad + 1, image.height)
    return image.crop((left, top, right, bottom))


def _fit_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    image = _trim_white(Image.open(path).convert("RGB"))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    coords = np.indices(shape, dtype=np.float32)
    center = np.asarray(shape, dtype=np.float32)[:, None, None, None] // 2
    labels = np.rint(np.sqrt(np.sum((coords - center) ** 2, axis=0))).astype(np.int32)
    n_shells = min(shape) // 2 - 1
    return np.clip(labels, 0, n_shells - 1), n_shells


def _fsc_curve(estimate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, n_shells = _shell_labels(estimate.shape)
    estimate_ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(estimate * mask)))
    target_ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(target * mask)))
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
        relerr = diff_power / target_power
    fsc[~np.isfinite(fsc)] = 0.0
    relerr[~np.isfinite(relerr)] = np.nan
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return fsc.astype(np.float32), relerr.astype(np.float32), target_power.astype(np.float64)


def _first_crossing_resolution(frequency: np.ndarray, fsc: np.ndarray, threshold: float = 0.5) -> float:
    valid = np.isfinite(frequency) & np.isfinite(fsc) & (frequency > 0)
    x = np.asarray(frequency[valid], dtype=np.float64)
    y = np.asarray(fsc[valid], dtype=np.float64)
    if x.size == 0:
        return math.nan
    below = np.flatnonzero(y < threshold)
    if below.size == 0:
        return float(1.0 / x[-1])
    idx = int(below[0])
    if idx == 0:
        return math.nan
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    if y1 == y0:
        cross = x1
    else:
        cross = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return float(1.0 / cross) if cross > 0 else math.nan


def _compute_curves(
    entries: list[Entry],
    output_dir: Path,
    voxel_size: float,
    mask_config: MaskConfig,
) -> dict[tuple[str, int], dict[str, object]]:
    mask, _, _ = _load_mrc(mask_config.mask)
    curves: dict[tuple[str, int], dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []
    for entry in entries:
        estimate, _, _ = _load_mrc(entry.estimate)
        gt, _, _ = _load_mrc(entry.gt)
        if estimate.shape != gt.shape or estimate.shape != mask.shape:
            raise ValueError(f"shape mismatch for {entry.method} {entry.n_label}")
        fsc, relerr, target_power = _fsc_curve(estimate, gt, mask)
        frequency = np.arange(fsc.size, dtype=np.float64) / (estimate.shape[0] * voxel_size)
        resolution = _first_crossing_resolution(frequency, fsc)
        shell_csv = (
            output_dir
            / "fsc_shell_metrics"
            / mask_config.key
            / entry.method
            / f"n{entry.n_images:08d}"
            / f"state0050_{mask_config.key}_shell_metrics.csv"
        )
        shell_csv.parent.mkdir(parents=True, exist_ok=True)
        with shell_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "shell",
                    "frequency_1_per_A",
                    "resolution_A",
                    "fsc_vs_gt",
                    "relative_error_per_shell",
                    "target_power",
                ]
            )
            for shell, (freq, fsc_value, err_value, power_value) in enumerate(
                zip(frequency, fsc, relerr, target_power, strict=True)
            ):
                writer.writerow(
                    [
                        shell,
                        freq,
                        math.inf if freq <= 0 else 1.0 / freq,
                        fsc_value,
                        err_value,
                        power_value,
                    ]
                )
        curves[(entry.method, entry.n_images)] = {
            "frequency": frequency,
            "fsc": fsc,
            "relerr": relerr,
            "resolution_A": resolution,
            "shell_metrics_csv": str(shell_csv),
            "estimate": str(entry.estimate),
            "gt": str(entry.gt),
        }
        summary_rows.append(
            {
                "method": entry.method,
                "n_images": entry.n_images,
                "n_label": entry.n_label,
                "state": 50,
                "fsc05_resolution_A": resolution,
                "estimate": str(entry.estimate),
                "gt": str(entry.gt),
                "mask": str(mask_config.mask),
                "shell_metrics_csv": str(shell_csv),
            }
        )
    summary_csv = output_dir / f"state50_uniform_noise3_available_{mask_config.key}_fsc_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return curves


def _plot_all_methods_same_axes(
    entries: list[Entry],
    curves: dict[tuple[str, int], dict[str, object]],
    output_dir: Path,
    mask_config: MaskConfig,
    plot_n_images: tuple[int, ...] | None,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=220)
    plotted_entries = [entry for entry in entries if plot_n_images is None or entry.n_images in plot_n_images]
    for entry in plotted_entries:
        curve = curves[(entry.method, entry.n_images)]
        ax.plot(
            curve["frequency"],
            curve["fsc"],
            color=METHOD_COLORS[entry.method],
            lw=2.9 if entry.n_images in (1_000_000, 3_000_000) else 1.8,
            alpha=0.95 if entry.n_images in (1_000_000, 3_000_000) else 0.50,
            label=f"{METHOD_LABELS[entry.method]} {entry.n_label}",
        )
    ax.axhline(0.5, color="0.35", ls=":", lw=1.2)
    ax.set_xlim(0.0, 0.40)
    ax.set_ylim(-0.04, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("masked FSC")
    ax.grid(True, color="0.86", lw=0.7)
    ax.legend(fontsize=7.2, ncol=3, frameon=True)
    n_suffix = ""
    n_title = ""
    if plot_n_images is not None:
        labels = [N_LABELS[n] for n in plot_n_images]
        n_suffix = "_" + "_".join(f"n{n:08d}" for n in plot_n_images) + "_only"
        n_title = " | " + ", ".join(labels) + " only"
    ax.set_title(f"Uniform noise=3 state 50 | {mask_config.label} | corrected outputs{n_title}", weight="bold")
    fig.tight_layout()
    png = output_dir / f"state50_uniform_noise3_available_{mask_config.key}_fsc_all_methods_same_axes{n_suffix}.png"
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.8, 8.0),
        dpi=240,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
    )
    ax_curve, ax_delta = axes
    deltas: list[np.ndarray] = []
    method_order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    for entry in sorted(plotted_entries, key=lambda e: (e.n_images, method_order[e.method])):
        curve = curves[(entry.method, entry.n_images)]
        ax_curve.plot(
            curve["frequency"],
            curve["fsc"],
            color=METHOD_COLORS[entry.method],
            lw=2.9,
            alpha=0.96,
            label=f"{METHOD_LABELS[entry.method]} {entry.n_label}",
        )
        recovar_curve = curves.get(("recovar", entry.n_images))
        if entry.method != "recovar" and recovar_curve is not None:
            delta = curve["fsc"] - recovar_curve["fsc"]
            deltas.append(delta)
            ax_delta.plot(
                curve["frequency"],
                delta,
                color=METHOD_COLORS[entry.method],
                lw=2.5,
                alpha=0.96,
                label=f"{METHOD_LABELS[entry.method]} - RECOVAR",
            )
    ax_curve.axhline(0.5, color="0.35", ls=":", lw=1.2)
    ax_curve.set_xlim(0.0, 0.40)
    ax_curve.set_ylim(-0.04, 1.03)
    ax_curve.set_ylabel("masked FSC")
    ax_curve.grid(True, color="0.86", lw=0.7)
    ax_curve.legend(fontsize=8.5, ncol=3, frameon=True)
    ax_curve.set_title(
        f"Uniform noise=3 state 50 | {mask_config.label} | corrected outputs{n_title}",
        weight="bold",
    )
    ax_delta.axhline(0.0, color="0.25", ls=":", lw=1.2)
    ax_delta.set_xlim(0.0, 0.40)
    ax_delta.set_xlabel("spatial frequency (1/A)")
    ax_delta.set_ylabel("FSC delta\nvs RECOVAR")
    if deltas:
        freq = np.asarray(curves[(plotted_entries[0].method, plotted_entries[0].n_images)]["frequency"])
        valid = (freq >= 0.025) & (freq <= 0.36)
        max_abs = max(float(np.nanmax(np.abs(delta[valid]))) for delta in deltas)
        ylim = max(0.06, min(0.50, 1.15 * max_abs))
        ax_delta.set_ylim(-ylim, ylim)
    ax_delta.grid(True, color="0.86", lw=0.7)
    ax_delta.legend(fontsize=8.5, ncol=2, frameon=True)
    delta_png = output_dir / f"state50_uniform_noise3_available_{mask_config.key}_fsc_method_delta_vs_recovar{n_suffix}.png"
    fig.tight_layout()
    fig.savefig(delta_png)
    fig.savefig(delta_png.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "png": str(png),
        "pdf": str(png.with_suffix(".pdf")),
        "delta_vs_recovar_png": str(delta_png),
        "delta_vs_recovar_pdf": str(delta_png.with_suffix(".pdf")),
    }


def _draw_centered(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) // 2, y), text, fill=(0, 0, 0), font=font)


def _plot_fsc_tile(
    curves: dict[tuple[str, int], dict[str, object]],
    method: str,
    current_n: int,
    size: tuple[int, int],
) -> Image.Image:
    if (method, current_n) not in curves:
        return Image.new("RGB", size, "white")
    fig, ax = plt.subplots(figsize=(size[0] / 180, size[1] / 180), dpi=180)
    fig.patch.set_facecolor("white")
    ax.axhline(0.5, color="0.50", ls=":", lw=1.4)
    for n_images in N_ORDER:
        if n_images > current_n or (method, n_images) not in curves:
            continue
        curve = curves[(method, n_images)]
        ax.plot(
            curve["frequency"],
            curve["fsc"],
            color=N_COLORS[n_images],
            lw=3.0 if n_images == current_n else 1.8,
            alpha=1.0 if n_images == current_n else 0.55,
        )
    res = curves[(method, current_n)]["resolution_A"]
    if math.isfinite(float(res)):
        ax.text(
            0.96,
            0.10,
            f"{float(res):.2f} A",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=15,
            weight="bold",
        )
    ax.set_xlim(0.0, 0.40)
    ax.set_ylim(-0.04, 1.03)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, color="0.86", lw=0.7)
    ax.tick_params(labelsize=9, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.18")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.14, top=0.98)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    plt.close(fig)
    return image.resize(size, Image.Resampling.LANCZOS)


def _effective_entry_for_frame(entries_by_method_n: dict[tuple[str, int], Entry], method: str, frame_n: int) -> Entry | None:
    exact = entries_by_method_n.get((method, frame_n))
    if exact is not None:
        return exact
    available = [n_images for candidate_method, n_images in entries_by_method_n if candidate_method == method and n_images <= frame_n]
    if not available:
        return None
    return entries_by_method_n[(method, max(available))]


def _ffmpeg_from_frames(frames_dir: Path, stem: str, fps: float) -> dict[str, object]:
    mp4 = frames_dir.parent / f"{stem}.mp4"
    gif = frames_dir.parent / f"{stem}.gif"
    palette = frames_dir.parent / f"{stem}_palette.png"
    log_path = frames_dir.parent / f"{stem}_ffmpeg.log"
    pattern = str(frames_dir / "frame_%03d.png")
    palette_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        pattern,
        "-vf",
        "palettegen=stats_mode=diff",
        str(palette),
    ]
    gif_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        pattern,
        "-i",
        str(palette),
        "-lavfi",
        "paletteuse=dither=bayer:bayer_scale=3",
        str(gif),
    ]
    mp4_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        pattern,
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
        str(mp4),
    ]
    with log_path.open("w") as log:
        rc1 = subprocess.run(palette_cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        rc2 = subprocess.run(gif_cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        rc3 = subprocess.run(mp4_cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return {
        "gif": str(gif),
        "mp4": str(mp4),
        "ffmpeg_log": str(log_path),
        "returncodes": [rc1, rc2, rc3],
    }


def _compose_movie(
    entries: list[Entry],
    curves: dict[tuple[str, int], dict[str, object]],
    raw_dir: Path,
    output_dir: Path,
    fps: float,
    mask_config: MaskConfig,
) -> dict[str, object]:
    entries_by_n = {(entry.method, entry.n_images): entry for entry in entries}
    frames_dir = output_dir / "frames" / "volume_plus_fsc"
    frames_dir.mkdir(parents=True, exist_ok=True)
    vol_tile = (760, 570)
    fsc_tile = (760, 420)
    gap = 28
    top = 96
    width = len(METHOD_ORDER) * vol_tile[0] + (len(METHOD_ORDER) - 1) * gap
    height = top + vol_tile[1] + 16 + fsc_tile[1]
    frames: list[str] = []
    for idx, n_images in enumerate(N_ORDER, start=1):
        frame = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(frame)
        _draw_centered(draw, width // 2, 12, N_LABELS[n_images], FONT_N)
        for col, method in enumerate(METHOD_ORDER):
            x = col * (vol_tile[0] + gap)
            entry = _effective_entry_for_frame(entries_by_n, method, n_images)
            if entry is None:
                continue
            frame.paste(_fit_tile(_rendered_path(raw_dir, entry), vol_tile), (x, top))
            if entry.n_images != n_images:
                draw.text((x + vol_tile[0] - 108, top + 10), f"({entry.n_label})", fill=(0, 0, 0), font=FONT_RES)
            frame.paste(_plot_fsc_tile(curves, method, entry.n_images, fsc_tile), (x, top + vol_tile[1] + 16))
        out = frames_dir / f"frame_{idx:03d}.png"
        frame.save(out, quality=98)
        frames.append(str(out))
    final_png = output_dir / f"state50_uniform_noise3_available_{mask_config.key}_volume_plus_fsc_final.png"
    Image.open(frames[-1]).save(final_png, quality=98)
    movies = _ffmpeg_from_frames(frames_dir, f"state50_uniform_noise3_available_{mask_config.key}_volume_plus_fsc", fps)
    return {"frames_dir": str(frames_dir), "frames": frames, "final_png": str(final_png), **movies}


def _write_audit(
    output_dir: Path,
    entries: list[Entry],
    manifest: Path,
    render_log: Path | None,
    curves: dict[tuple[str, int], dict[str, object]],
    same_axes_plot: dict[str, str],
    movie: dict[str, object],
    mask_config: MaskConfig,
) -> Path:
    audit = {
        "script": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "sweep_root": str(SWEEP_ROOT),
        "source_root": str(SOURCE_ROOT),
        "mask": str(mask_config.mask),
        "mask_mode": mask_config.key,
        "view_json": str(mask_config.view_json),
        "state": 50,
        "noise_level": 3,
        "bfactor": 80,
        "method_colors": METHOD_COLORS,
        "gt_color": GT_COLOR,
        "contour_level": mask_config.contour_level,
        "entries": [
            {
                "method": entry.method,
                "n_images": entry.n_images,
                "n_label": entry.n_label,
                "estimate": str(entry.estimate),
                "gt": str(entry.gt),
                "fsc05_resolution_A": curves[(entry.method, entry.n_images)]["resolution_A"],
            }
            for entry in entries
        ],
        "missing_by_design": {
            "cryodrgn": "No decoded 3M state-50 output exists in this sweep tree, so cryoDRGN stops at 1M."
        },
        "outputs": {
            "manifest": str(manifest),
            "render_log": None if render_log is None else str(render_log),
            "fsc_summary_csv": str(output_dir / f"state50_uniform_noise3_available_{mask_config.key}_fsc_summary.csv"),
            "same_axes_plot": same_axes_plot,
            "movie": movie,
        },
    }
    path = output_dir / f"state50_uniform_noise3_available_{mask_config.key}_movie_audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SWEEP_ROOT / "state50_uniform_noise3_available_movies_20260607",
    )
    parser.add_argument("--mask-mode", choices=sorted(MASK_CONFIGS), default="broad")
    parser.add_argument(
        "--plot-n-images",
        help="Comma-separated image counts to include in standalone FSC plots, e.g. 1000000.",
    )
    parser.add_argument("--plots-only", action="store_true", help="Only compute FSC curves and standalone plots.")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--render-width", type=int, default=1800)
    parser.add_argument("--render-height", type=int, default=1350)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--fps", type=float, default=0.75)
    parser.add_argument("--voxel-size", type=float, default=1.25)
    parser.add_argument("--chimerax-module", default="chimerax/1.9")
    return parser.parse_args(argv)


def _parse_plot_n_images(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == "":
        return None
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    invalid = [n_images for n_images in parsed if n_images not in N_ORDER]
    if invalid:
        raise ValueError(f"unsupported --plot-n-images values: {invalid}")
    return parsed


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_config = MASK_CONFIGS[args.mask_mode]
    entries = discover_entries()
    if not entries:
        raise RuntimeError("no state-50 entries found")
    masked_paths = _prepare_masked_volumes(entries, args.output_dir, mask_config)
    manifest = _write_manifest(entries, masked_paths, args.output_dir, mask_config)
    raw_dir = args.output_dir / "png_raw"
    plot_n_images = _parse_plot_n_images(args.plot_n_images)
    render_log = None if (args.skip_render or args.plots_only) else _run_chimerax(manifest, raw_dir, args, mask_config)
    curves = _compute_curves(entries, args.output_dir, args.voxel_size, mask_config)
    same_axes_plot = _plot_all_methods_same_axes(entries, curves, args.output_dir, mask_config, plot_n_images)
    movie = {} if args.plots_only else _compose_movie(entries, curves, raw_dir, args.output_dir, args.fps, mask_config)
    audit = _write_audit(args.output_dir, entries, manifest, render_log, curves, same_axes_plot, movie, mask_config)
    print(json.dumps({"output_dir": str(args.output_dir), "audit": str(audit), "movie": movie}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
