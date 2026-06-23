#!/usr/bin/env python3
"""Refresh clean state-50 presentation movies up to 1M particles.

This driver makes horizontally arranged, minimally annotated movies for the
noise=1 and noise=3 spike method sweeps.  It intentionally writes to new output
directories so older labeled/debug movies remain untouched.
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
import mrcfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
MOVING_RENDERER = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "spike_fullatom_method_benchmark"
    / "render_state50_moving_mask_progression.py"
)
NOTMOVING_RENDERER = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "spike_fullatom_method_benchmark"
    / "render_state50_notmoving_sharpened_progression.py"
)

METHOD_ORDER = ("recovar", "cryodrgn", "3dflex")
N_ORDER = (10_000, 30_000, 100_000, 300_000, 1_000_000)
N_LABELS = {
    10_000: "10k",
    30_000: "30k",
    100_000: "100k",
    300_000: "300k",
    1_000_000: "1M",
}
N_COLORS = {
    10_000: "#4c1d7a",
    30_000: "#287796",
    100_000: "#2bb07f",
    300_000: "#c9df1a",
    1_000_000: "#ff9f1c",
}

MASK_BROAD = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
MASK_NOTMOVING = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/not_moving_spike_mask_soft_20260601.mrc"
)
VIEW_MOVING = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_moving_mask_zoomed_view_20260601/"
    "zoomed_moving_view_extracted.json"
)
VIEW_NOTMOVING = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_notmoving_view_20260601/"
    "not_moving_view_extracted.json"
)


@dataclass(frozen=True)
class NoiseConfig:
    noise: int
    sweep_root: Path
    source_root: Path
    moving_summary: Path
    notmoving_summary: Path


CONFIGS = {
    1: NoiseConfig(
        noise=1,
        sweep_root=Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530"),
        source_root=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
            "spike_fullatom_consistency_grid256_noise1_b80_true_oracle_sweep_20260527"
        ),
        moving_summary=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530/"
            "state50_cumulative_fsc_movies_by_method_20260601/moving_broadmask/moving_broadmask_fsc_summary.csv"
        ),
        notmoving_summary=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530/"
            "not_moving_softmask_fsc_resolution_20260601/not_moving_softmask_noise1_fsc_summary.csv"
        ),
    ),
    3: NoiseConfig(
        noise=3,
        sweep_root=Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531"),
        source_root=Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise3_b80_20260531"),
        moving_summary=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/"
            "state50_cumulative_fsc_movies_by_method_20260601/moving_broadmask/moving_broadmask_fsc_summary.csv"
        ),
        notmoving_summary=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/"
            "not_moving_softmask_fsc_resolution_20260601/not_moving_softmask_noise3_fsc_summary.csv"
        ),
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


FONT_N = _font(62, True)
FONT_SMALL = _font(26, True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _state50_cryodrgn_1m_row(config: NoiseConfig, mask: Path) -> dict[str, str]:
    metrics = config.sweep_root / "n01000000/evaluation/metrics/decoded_volume_metrics.csv"
    rows = _read_csv(metrics)
    selected = None
    for row in rows:
        if row.get("method") == "cryodrgn" and row.get("gt_label") == "50" and row.get("metric_mask") == "broad":
            selected = row
            break
    if selected is None:
        for row in rows:
            if row.get("method") == "cryodrgn" and row.get("gt_label") == "50":
                selected = row
                break
    if selected is None:
        raise RuntimeError(f"no cryoDRGN state 50 row in {metrics}")
    return {
        "method": "cryodrgn",
        "method_label": "cryoDRGN",
        "n_images": "1000000",
        "n_label": "1M",
        "state": "50",
        "fsc05_resolution_A": selected.get("fsc_res_0p5_A", ""),
        "estimate": selected["volume"],
        "gt": selected["gt_volume"],
        "mask": str(mask),
        "shell_metrics_csv": "",
    }


def _write_augmented_summary(config: NoiseConfig, mode: str, out_dir: Path) -> Path:
    source = config.moving_summary if mode == "moving" else config.notmoving_summary
    mask = MASK_BROAD if mode == "moving" else MASK_NOTMOVING
    rows = []
    seen: set[tuple[str, int]] = set()
    for row in _read_csv(source):
        if int(row.get("state", -1)) != 50:
            continue
        method = row.get("method", "")
        if method not in METHOD_ORDER:
            continue
        n_images = int(row["n_images"])
        if n_images not in N_ORDER:
            continue
        key = (method, n_images)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "method": method,
                "method_label": row.get("method_label", method),
                "n_images": str(n_images),
                "n_label": row.get("n_label", N_LABELS[n_images]),
                "state": "50",
                "fsc05_resolution_A": row.get("fsc05_resolution_A", ""),
                "estimate": row["estimate"],
                "gt": row["gt"],
                "mask": str(mask),
                "shell_metrics_csv": row.get("shell_metrics_csv", ""),
            }
        )
    if ("cryodrgn", 1_000_000) not in seen:
        rows.append(_state50_cryodrgn_1m_row(config, mask))
    order = {method: i for i, method in enumerate(METHOD_ORDER)}
    n_order = {n: i for i, n in enumerate(N_ORDER)}
    rows.sort(key=lambda row: (order[row["method"]], n_order[int(row["n_images"])]))
    missing = sorted({(method, n) for method in METHOD_ORDER for n in N_ORDER} - {(r["method"], int(r["n_images"])) for r in rows})
    if missing:
        raise RuntimeError(f"missing rows for noise={config.noise} mode={mode}: {missing}")
    out = out_dir / f"state50_noise{config.noise}_{mode}_upto1m_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def _run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(" ".join(command) + "\n\n")
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"command failed with code {proc.returncode}; see {log_path}")


def _run_renderers(config: NoiseConfig, out_dir: Path, summaries: dict[str, Path], args: argparse.Namespace) -> dict[str, Path]:
    render_roots = {
        "moving": out_dir / "render_sources_moving_zoomed",
        "notmoving": out_dir / "render_sources_notmoving_sharpened",
    }
    if args.skip_render:
        return render_roots
    common = [
        "--state",
        "50",
        "--noise-level",
        str(config.noise),
        "--bfactor",
        "80",
        "--width",
        str(args.render_width),
        "--height",
        str(args.render_height),
        "--supersample",
        str(args.supersample),
        "--run-chimerax",
        "--skip-compose",
    ]
    moving_cmd = [
        sys.executable,
        str(MOVING_RENDERER),
        "--summary-csv",
        str(summaries["moving"]),
        "--output-dir",
        str(render_roots["moving"]),
        "--view-json",
        str(VIEW_MOVING),
        "--contour-level",
        "0.013",
        "--gt-opacity",
        "0.3",
        "--skip-fsc",
        *common,
    ]
    _run_command(moving_cmd, out_dir / "logs" / "render_moving.log")
    notmoving_cmd = [
        sys.executable,
        str(NOTMOVING_RENDERER),
        "--summary-csv",
        str(summaries["notmoving"]),
        "--output-dir",
        str(render_roots["notmoving"]),
        "--view-json",
        str(VIEW_NOTMOVING),
        "--gt-opacity",
        "0.3",
        *common,
    ]
    _run_command(notmoving_cmd, out_dir / "logs" / "render_notmoving.log")
    return render_roots


def _raw_png_for_row(raw_dir: Path, row: dict[str, str]) -> Path:
    name = Path(row["render_name"])
    return raw_dir / f"{name.stem}_gt_overlay{name.suffix}"


def _manifest_index(manifest: Path, raw_dir: Path) -> dict[tuple[str, int], Path]:
    index: dict[tuple[str, int], Path] = {}
    for row in _read_csv(manifest):
        if row.get("role") != "estimate" or row.get("method") not in METHOD_ORDER:
            continue
        path = _raw_png_for_row(raw_dir, row)
        if not path.exists():
            raise FileNotFoundError(path)
        index[(row["method"], int(row["n_images"]))] = path
    return index


def _trim_white(image: Image.Image, threshold: int = 248, pad: int = 28) -> Image.Image:
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


def _draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) // 2, xy[1]), text, fill=(0, 0, 0), font=font)


def _ffmpeg_from_frames(frames_dir: Path, stem: str, fps: float) -> dict[str, str | int | list[int]]:
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


def _compose_volume_movie(config: NoiseConfig, mode: str, render_root: Path, out_dir: Path, fps: float) -> dict[str, object]:
    manifest_name = "manifest_chimerax_state50_moving_mask.csv" if mode == "moving" else "manifest_chimerax_state50_notmoving_sharpened.csv"
    index = _manifest_index(render_root / manifest_name, render_root / "png_raw")
    frames_dir = out_dir / "frames" / f"volume_{mode}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    tile = (900, 675)
    top = 92
    gap = 24
    width = len(METHOD_ORDER) * tile[0] + (len(METHOD_ORDER) - 1) * gap
    height = top + tile[1]
    frame_paths: list[str] = []
    for idx, n_images in enumerate(N_ORDER, start=1):
        frame = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(frame)
        _draw_centered(draw, (width // 2, 16), N_LABELS[n_images], FONT_N)
        x = 0
        for method in METHOD_ORDER:
            frame.paste(_fit_tile(index[(method, n_images)], tile), (x, top))
            x += tile[0] + gap
        output = frames_dir / f"frame_{idx:03d}.png"
        frame.save(output, quality=98)
        frame_paths.append(str(output))
    movie = _ffmpeg_from_frames(frames_dir, f"state50_noise{config.noise}_{mode}_volume_upto1m_horizontal", fps)
    final = out_dir / f"state50_noise{config.noise}_{mode}_volume_upto1m_final.png"
    Image.open(frame_paths[-1]).save(final, quality=98)
    return {"frames_dir": str(frames_dir), "frames": frame_paths, "final_png": str(final), **movie}


def _load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float32)


def _shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    n = shape[0]
    coords = np.arange(n, dtype=np.float32) - (n // 2)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij", sparse=True)
    labels = np.rint(np.sqrt(x * x + y * y + z * z)).astype(np.int32)
    n_shells = n // 2 - 1
    return np.clip(labels, 0, n_shells - 1), n_shells


def _fsc_curve(estimate: np.ndarray, target: np.ndarray, mask: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, float]:
    if estimate.shape != target.shape or estimate.shape != mask.shape:
        raise ValueError(f"shape mismatch: estimate={estimate.shape}, target={target.shape}, mask={mask.shape}")
    labels, n_shells = _shell_labels(estimate.shape)
    est_ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(estimate * mask)))
    gt_ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(target * mask)))
    flat = labels.ravel()
    cross = np.bincount(flat, weights=np.real(np.conj(est_ft).ravel() * gt_ft.ravel()), minlength=n_shells)
    est_power = np.bincount(flat, weights=np.abs(est_ft).ravel() ** 2, minlength=n_shells)
    gt_power = np.bincount(flat, weights=np.abs(gt_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(est_power * gt_power)
    fsc[~np.isfinite(fsc)] = 0.0
    if fsc.size > 1:
        fsc[0] = fsc[1]
    freq = np.arange(n_shells, dtype=np.float64) / (estimate.shape[0] * voxel_size)
    valid = (freq > 0) & np.isfinite(fsc)
    good = np.flatnonzero(valid & (fsc >= 0.5))
    resolution = math.nan if good.size == 0 else float(1.0 / freq[int(good[-1])])
    return freq, fsc.astype(np.float32), resolution


def _summary_inputs(summary: Path) -> dict[tuple[str, int], tuple[Path, Path]]:
    out: dict[tuple[str, int], tuple[Path, Path]] = {}
    for row in _read_csv(summary):
        if row["method"] not in METHOD_ORDER or int(row["n_images"]) not in N_ORDER:
            continue
        out[(row["method"], int(row["n_images"]))] = (Path(row["estimate"]), Path(row["gt"]))
    return out


def _compute_curves(summary: Path, mask_path: Path, shell_dir: Path, voxel_size: float) -> dict[tuple[str, int], dict[str, object]]:
    mask = _load_mrc(mask_path)
    curves: dict[tuple[str, int], dict[str, object]] = {}
    shell_dir.mkdir(parents=True, exist_ok=True)
    for (method, n_images), (estimate_path, gt_path) in _summary_inputs(summary).items():
        estimate = _load_mrc(estimate_path)
        gt = _load_mrc(gt_path)
        freq, fsc, resolution = _fsc_curve(estimate, gt, mask, voxel_size)
        shell_csv = shell_dir / method / f"n{n_images:08d}" / "state0050_shell_metrics.csv"
        shell_csv.parent.mkdir(parents=True, exist_ok=True)
        with shell_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["shell", "frequency_1_per_A", "resolution_A", "fsc_vs_gt"])
            for shell, (frequency, value) in enumerate(zip(freq, fsc, strict=True)):
                res = math.inf if frequency <= 0 else 1.0 / frequency
                writer.writerow([shell, frequency, res, value])
        curves[(method, n_images)] = {
            "frequency": freq,
            "fsc": fsc,
            "resolution_A": resolution,
            "estimate": str(estimate_path),
            "gt": str(gt_path),
            "shell_metrics_csv": str(shell_csv),
        }
    return curves


def _compose_fsc_movie(
    config: NoiseConfig,
    mode: str,
    summary: Path,
    mask: Path,
    out_dir: Path,
    fps: float,
    voxel_size: float,
) -> dict[str, object]:
    curves = _compute_curves(summary, mask, out_dir / "shell_metrics" / mode, voxel_size)
    frames_dir = out_dir / "frames" / f"fsc_{mode}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[str] = []
    csv_path = out_dir / f"state50_noise{config.noise}_{mode}_fsc_upto1m_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "n_images", "n_label", "resolution_A", "shell_metrics_csv"])
        writer.writeheader()
        for method in METHOD_ORDER:
            for n_images in N_ORDER:
                curve = curves[(method, n_images)]
                writer.writerow(
                    {
                        "method": method,
                        "n_images": n_images,
                        "n_label": N_LABELS[n_images],
                        "resolution_A": curve["resolution_A"],
                        "shell_metrics_csv": curve["shell_metrics_csv"],
                    }
                )
    for idx, n_images in enumerate(N_ORDER, start=1):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.1), dpi=220, sharex=True, sharey=True)
        fig.patch.set_facecolor("white")
        for ax, method in zip(axes, METHOD_ORDER, strict=True):
            ax.axhline(0.5, color="0.45", ls=":", lw=1.4)
            for visible_n in N_ORDER[:idx]:
                curve = curves[(method, visible_n)]
                ax.plot(
                    curve["frequency"],
                    curve["fsc"],
                    color=N_COLORS[visible_n],
                    lw=3.2 if visible_n == n_images else 2.0,
                    alpha=1.0 if visible_n == n_images else 0.55,
                )
            current = curves[(method, n_images)]
            res = current["resolution_A"]
            text = "" if not math.isfinite(float(res)) else f"{float(res):.2f}A"
            ax.text(0.965, 0.08, text, ha="right", va="bottom", transform=ax.transAxes, fontsize=12, weight="bold")
            ax.set_xlim(0.0, 0.40)
            ax.set_ylim(-0.04, 1.03)
            ax.grid(True, color="0.82", lw=0.7, alpha=0.55)
            ax.tick_params(labelsize=9, width=0.8, length=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
                spine.set_color("0.15")
        fig.text(0.5, 0.965, N_LABELS[n_images], ha="center", va="top", fontsize=30, weight="bold")
        fig.subplots_adjust(left=0.045, right=0.995, bottom=0.10, top=0.86, wspace=0.055)
        frame = frames_dir / f"frame_{idx:03d}.png"
        fig.savefig(frame, facecolor="white")
        plt.close(fig)
        frame_paths.append(str(frame))
    final = out_dir / f"state50_noise{config.noise}_{mode}_fsc_upto1m_final.png"
    Image.open(frame_paths[-1]).save(final, quality=98)
    movie = _ffmpeg_from_frames(frames_dir, f"state50_noise{config.noise}_{mode}_fsc_upto1m_horizontal", fps)
    return {
        "mask": str(mask),
        "summary_csv": str(csv_path),
        "frames_dir": str(frames_dir),
        "frames": frame_paths,
        "final_png": str(final),
        **movie,
    }


def _run_noise(config: NoiseConfig, args: argparse.Namespace) -> dict[str, object]:
    out_dir = config.sweep_root / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "moving": _write_augmented_summary(config, "moving", out_dir / "summaries"),
        "notmoving": _write_augmented_summary(config, "notmoving", out_dir / "summaries"),
    }
    render_roots = _run_renderers(config, out_dir, summaries, args)
    outputs: dict[str, object] = {"summaries": {k: str(v) for k, v in summaries.items()}, "render_roots": {k: str(v) for k, v in render_roots.items()}}
    outputs["volume_moving"] = _compose_volume_movie(config, "moving", render_roots["moving"], out_dir, args.fps)
    outputs["volume_notmoving"] = _compose_volume_movie(config, "notmoving", render_roots["notmoving"], out_dir, args.fps)
    outputs["fsc_moving"] = _compose_fsc_movie(config, "moving", summaries["moving"], MASK_BROAD, out_dir, args.fps, args.voxel_size)
    outputs["fsc_notmoving"] = _compose_fsc_movie(config, "notmoving", summaries["notmoving"], MASK_NOTMOVING, out_dir, args.fps, args.voxel_size)
    audit = {
        "script": str(Path(__file__).resolve()),
        "repo": str(REPO_ROOT),
        "noise": config.noise,
        "sweep_root": str(config.sweep_root),
        "n_order": list(N_ORDER),
        "methods": list(METHOD_ORDER),
        "output_dir": str(out_dir),
        "outputs": outputs,
    }
    (out_dir / "presentation_movie_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=int, action="append", choices=sorted(CONFIGS), help="Noise level to process. Repeatable; default is 1 and 3.")
    parser.add_argument("--output-name", default="state50_presentation_movies_upto1m_20260603")
    parser.add_argument("--skip-render", action="store_true", help="Reuse render_sources_* directories already present in the output root.")
    parser.add_argument("--render-width", type=int, default=1800)
    parser.add_argument("--render-height", type=int, default=1350)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--fps", type=float, default=0.8)
    parser.add_argument("--voxel-size", type=float, default=1.25)
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    noise_levels = args.noise or [1, 3]
    outputs = {str(noise): _run_noise(CONFIGS[noise], args) for noise in noise_levels}
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
